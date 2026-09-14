"""Unidentified speakers get a stable 'Speaker N' identity.

The fixture's `Unknown (SPEAKER_01)` group has four turns totalling 14s, so the
default 60s floor leaves it Unknown; passing placeholder_min_duration=10 makes it
substantial. A placeholder is enrolled like any other voiceprint, which is what
lets a later meeting recognize the same person.
"""

import json

import numpy as np

from ai.speaker_review import SpeakerReview, _next_placeholder_name


def test_next_placeholder_name_starts_at_one():
    assert _next_placeholder_name(set()) == "Speaker 1"


def test_next_placeholder_name_continues_from_taken():
    assert _next_placeholder_name({"Speaker 1", "Speaker 2", "Alice"}) == "Speaker 3"


def test_next_placeholder_name_ignores_lookalikes():
    assert _next_placeholder_name({"Speaker One", "Speaker 2x", "Speaker"}) == "Speaker 1"


def test_substantial_unknown_becomes_speaker_1_and_is_enrolled(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=10.0)
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert "Speaker 1" in db
    assert abs(np.linalg.norm(np.array(db["Speaker 1"])) - 1.0) < 1e-6


def test_short_unknown_stays_unknown_and_is_not_enrolled(sidecar):
    # 14s of speech against the default 60s floor.
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert not any(k.startswith("Speaker ") for k in db)
    assert "Unknown (SPEAKER_01)" not in db


def test_two_unnamed_groups_get_distinct_numbers(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=1.0)
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    r.split(gid, k=2)
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert "Speaker 1" in db and "Speaker 2" in db


def test_numbering_continues_from_existing_db(sidecar):
    with open(sidecar["db"], "w") as f:
        json.dump({"Speaker 2": [0.0, 1.0, 0, 0, 0, 0, 0, 0]}, f)
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=10.0)
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert "Speaker 3" in db


def test_remember_true_enrolls_a_short_group(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])  # default 60s floor
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    r.set_remember(gid, True)
    r.commit()
    assert "Speaker 1" in json.load(open(sidecar["db"]))


def test_remember_false_suppresses_a_substantial_group(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=10.0)
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    r.set_remember(gid, False)
    r.commit()
    assert not any(k.startswith("Speaker ") for k in json.load(open(sidecar["db"])))


def test_named_group_is_never_turned_into_a_placeholder(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=1.0)
    r.commit()
    db = json.load(open(sidecar["db"]))
    assert "Alice" in db


def test_placeholder_name_appears_in_the_rebuilt_transcript(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=10.0)
    r.commit()
    note = open(sidecar["note"]).read()
    tsection = note.split("## Transcript")[1]
    assert "Speaker 1" in tsection
    assert "Unknown (SPEAKER_01)" not in tsection


def test_list_speakers_remember_matches_duration_rule_by_default(sidecar):
    """No override: `remember` in list_speakers() must track the same duration
    rule _assign_placeholders will apply at commit -- this is what the UI
    checkbox reads to decide its pre-checked state."""
    # 14s total against the default 60s floor -> not substantial.
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    s = next(s for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    assert s["remember"] is False

    # Same 14s group against a 10s floor -> substantial.
    r2 = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=10.0)
    s2 = next(s for s in r2.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    assert s2["remember"] is True


def test_list_speakers_remember_reflects_override_true_on_a_short_group(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])  # default 60s floor, 14s group
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    r.set_remember(gid, True)
    s = next(s for s in r.list_speakers() if s["id"] == gid)
    assert s["remember"] is True


def test_list_speakers_remember_reflects_override_false_on_a_substantial_group(sidecar):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"], placeholder_min_duration=10.0)
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == "Unknown (SPEAKER_01)")
    r.set_remember(gid, False)
    s = next(s for s in r.list_speakers() if s["id"] == gid)
    assert s["remember"] is False
