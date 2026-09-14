import json
import os

import numpy as np

from ai.speaker_review import SpeakerReview
from core import glossary


def test_list_speakers_groups_by_label(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    names = sorted(s["name"] for s in r.list_speakers())
    assert names == ["Alice", "Unknown (SPEAKER_01)"]
    spk = {s["name"]: s for s in r.list_speakers()}
    assert spk["Unknown (SPEAKER_01)"]["n_segments"] == 4
    assert spk["Alice"]["n_segments"] == 1
    # rep_clip is the longest turn of the group (a valid (start, end) span)
    assert spk["Alice"]["rep_clip"][1] > spk["Alice"]["rep_clip"][0]
    assert spk["Alice"]["source"] == "db"


def test_rename(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Alice")
    r.rename(gid, "Alicia")
    assert "Alicia" in [s["name"] for s in r.list_speakers()]
    assert "Alice" not in [s["name"] for s in r.list_speakers()]


def test_merge(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    ids = [s["id"] for s in r.list_speakers()]
    new = r.merge(ids)
    r.rename(new, "OnePerson")
    lst = r.list_speakers()
    assert len(lst) == 1 and lst[0]["name"] == "OnePerson"
    assert lst[0]["n_segments"] == 5


def test_split_separates_two_clusters(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    new_ids = r.split(gid, k=2)
    assert len(new_ids) == 2
    sizes = sorted(len(r.groups[g]["seg_idxs"]) for g in new_ids)
    assert sizes == [2, 2]  # the two synthetic clusters
    # the original group is gone; Alice is untouched
    assert gid not in r.groups
    assert "Alice" in [s["name"] for s in r.list_speakers()]


def test_reassign_marks_existing(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Alice")
    r.reassign(gid, "ExistingBob")
    assert r.groups[gid]["name"] == "ExistingBob"


def test_reset_voiceprint_flags(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    r.reset_voiceprint("ExistingBob")
    assert "ExistingBob" in r._reset_names


def test_commit_writes_normalized_centroids(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert "Alice" in db
    assert abs(np.linalg.norm(np.array(db["Alice"])) - 1.0) < 1e-6


def test_commit_folds_punctuation_variant_into_existing_key(sidecar):
    # Enrolling "Riley S. Stone" must UPDATE the existing "Riley S Stone" print,
    # not create a second near-duplicate identity (the real cross-meeting bug).
    vec = np.arange(1, 9, dtype=float)
    with open(sidecar["db"], "w") as f:
        json.dump({"Riley S Stone": (vec / np.linalg.norm(vec)).tolist()}, f)
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    r.rename(gid, "Riley S. Stone")
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert "Riley S Stone" in db
    assert "Riley S. Stone" not in db


def test_commit_never_enrolls_the_local_user(sidecar):
    # Even if a group is (mis)named the local user, their voiceprint must not enter
    # the DB — they are channel 0 / "Me (Local)", not a stored speaker.
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], local_names=["Alex Morgan"])
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Alice")
    r.rename(gid, "Alex Morgan")
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert "Alex Morgan" not in db
    assert "Alex Morgan" not in r.suggestion_names()
    assert "Unknown (SPEAKER_01)" not in db  # Unknown groups are never enrolled


def test_commit_split_rewrites_transcript_per_segment(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    a, b = r.split(gid, k=2)
    r.rename(a, "Xavier")
    r.rename(b, "Yara")
    r.commit()
    note = open(sidecar["note"]).read()
    tsection = note.split("## Transcript")[1]
    assert "**Me (Local)**" in note                     # local override preserved
    assert "Xavier" in tsection and "Yara" in tsection  # split names, per segment
    assert "Unknown (SPEAKER_01)" not in tsection


def test_commit_ema_refines_existing(sidecar):
    # Pre-seed the DB with a different Alice vector; commit should EMA-blend, not overwrite.
    with open(sidecar["db"], "w") as f:
        json.dump({"Alice": [0.0, 1.0, 0, 0, 0, 0, 0, 0]}, f)
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], update_rate=0.5)
    # Confirm the row, as the user does in review. An untouched auto-filled name
    # no longer moves a stored print — that path let a wrong match poison the
    # voiceprint. This test is about the blend itself, so it takes the
    # confirmed path where blending is legitimate.
    for gid, g in r.groups.items():
        if g["name"] == "Alice":
            r.rename(gid, "Alice")
    r.commit()
    v = np.array(json.load(open(sidecar["db"]))["Alice"])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    assert v[0] > 0.1 and v[1] > 0.1  # blended between old [0,1] and new [1,0]


def test_commit_summarizes_named_transcript_and_harvests(sidecar, monkeypatch):
    import export.llm_processor as lp

    def fake_gen(self, transcript):
        # commit must summarize the NAMED transcript, not the anonymous one
        assert "Unknown (SPEAKER_01)" not in transcript
        return {
            "executive_summary": [{"text": "discussed the PBV plan", "timestamp": "01:00"}],
            "action_items": [{"text": "Bob to run registration", "timestamp": "02:00"}],
            "entities": ["PBV plan"],
        }

    monkeypatch.setattr(lp.LLMProcessor, "generate_summary", fake_gen)
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], llm_model="fake:model")
    gid = next(s["id"] for s in r.list_speakers() if s["name"].startswith("Unknown"))
    r.rename(gid, "Bob")
    r.commit()

    note = open(sidecar["note"]).read()
    assert "# Meeting Summary" in note
    assert "PBV plan" in note and "(01:00)" in note        # named summary written
    assert "run registration" in note and "(02:00)" in note
    assert "Bob" in note                                   # assignee is the real name

    gpath = os.path.join(os.path.dirname(sidecar["db"]), "glossary_learned.json")
    active = glossary.active_terms(gpath)
    assert "Bob" in active                # confirmed name pinned active immediately
    assert "PBV plan" not in active       # entity needs a second sighting to stick
