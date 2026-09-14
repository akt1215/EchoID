"""Cross-meeting speaker directory built from saved sidecars + the voiceprint DB."""

import json
import os
import pathlib

import numpy as np
import pytest

from ai.speaker_directory import SpeakerDirectory

MEAN = [5.0, 5.0, 5.0, 5.0]
ALICE = [1.0, 0.0, 0.0, 0.0]     # centered-space voiceprints
BOB = [0.0, 1.0, 0.0, 0.0]
LOCAL_VECTOR = [0.0, 0.0, 1.0, 0.0]


def _turn(start, end, cluster, label, base):
    return {"idx": 0, "start": start, "end": end, "label": label, "cluster": cluster,
            "source": "db", "embedding": list(np.array(MEAN) + np.array(base))}


@pytest.fixture
def ws(tmp_path):
    """A workspace with two meetings: Alice in both, Bob only in the second."""
    vault = tmp_path / "Obsidian_Vault"
    vault.mkdir()
    (tmp_path / "embedding_mean.json").write_text(
        json.dumps({"mean": MEAN, "count": 10, "meetings": []}))
    (tmp_path / "speakers.json").write_text(
        json.dumps({"Alice": ALICE, "Speaker 1": BOB}))

    (tmp_path / "m1.segments.json").write_text(json.dumps({
        "wav": str(tmp_path / "m1.wav"), "note": str(vault / "m1.md"),
        "diarization": [_turn(0, 30, "db::Alice", "Alice", ALICE)],
        "transcript": [],
    }))
    (tmp_path / "m2.segments.json").write_text(json.dumps({
        "wav": str(tmp_path / "m2.wav"), "note": str(vault / "m2.md"),
        "diarization": [_turn(0, 10, "db::Alice", "Alice", ALICE),
                        _turn(20, 60, "SPEAKER_02", "Speaker 1", BOB)],
        "transcript": [],
    }))
    return {"workspace": str(tmp_path), "vault": str(vault)}


def test_identities_lists_named_and_placeholder(ws):
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    by = {e["name"]: e for e in d.identities()}
    assert by["Alice"]["is_placeholder"] is False
    assert by["Speaker 1"]["is_placeholder"] is True


def test_identities_counts_meetings_and_duration(ws):
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    by = {e["name"]: e for e in d.identities()}
    assert by["Alice"]["n_meetings"] == 2
    assert by["Alice"]["total_dur"] == pytest.approx(40.0)   # 30 + 10
    assert by["Speaker 1"]["n_meetings"] == 1
    assert by["Speaker 1"]["total_dur"] == pytest.approx(40.0)


def test_best_clip_returns_the_longest_turn(ws):
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    wav, start, end = d.best_clip("Alice")
    assert (start, end) == (0.0, 30.0)      # the 30s turn in m1, not the 10s one
    assert wav.endswith("m1.wav")


def test_best_clip_is_none_for_an_unmatched_name(ws):
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    assert d.best_clip("Nobody") is None


def test_corrupt_sidecar_is_skipped_not_fatal(ws):
    # A truncated/garbage sidecar dropped into the workspace (e.g. a crash
    # mid-write) must not take down the whole directory -- the good meetings
    # should still resolve.
    pathlib.Path(ws["workspace"], "m3.segments.json").write_text("{not valid json")

    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    by = {e["name"]: e for e in d.identities()}
    assert by["Alice"]["n_meetings"] == 2
    assert by["Speaker 1"]["n_meetings"] == 1


def test_two_clusters_matching_one_name_in_one_meeting_count_as_one_meeting(tmp_path):
    # Rare edge case: two distinct pyannote clusters within the SAME sidecar
    # both cosine-match the same enrolled voiceprint (e.g. the voiceprint was
    # enrolled/updated after this meeting was processed, so the clusters were
    # never merged into a single "db::Alice" cluster id at pipeline time).
    # n_meetings must still count this as ONE meeting, not two.
    vault = tmp_path / "Obsidian_Vault"
    vault.mkdir()
    (tmp_path / "embedding_mean.json").write_text(
        json.dumps({"mean": MEAN, "count": 10, "meetings": []}))
    (tmp_path / "speakers.json").write_text(json.dumps({"Alice": ALICE}))
    (tmp_path / "m1.segments.json").write_text(json.dumps({
        "wav": str(tmp_path / "m1.wav"), "note": str(vault / "m1.md"),
        "diarization": [_turn(0, 5, "SPEAKER_00", "Unknown (SPEAKER_00)", ALICE),
                        _turn(10, 25, "SPEAKER_03", "Unknown (SPEAKER_03)", ALICE)],
        "transcript": [],
    }))

    d = SpeakerDirectory(str(tmp_path), str(vault))
    by = {e["name"]: e for e in d.identities()}
    assert by["Alice"]["n_meetings"] == 1
    assert by["Alice"]["total_dur"] == pytest.approx(20.0)   # 5 + 15, merged


def _write_note(vault, name, who):
    p = vault / name
    p.write_text(
        "---\n"
        "date: 2026-07-16T09:00:00\n"
        f'participants: ["[[{who}]]", "[[Me (Local)]]"]\n'
        "---\n\n"
        "## Transcript\n"
        f"**[[{who}]]** (00:03):\nhello there \n"
        "**Me (Local)** (00:10):\nhi \n"
    )
    return p


def test_rename_plan_reports_notes_and_writes_nothing(ws, tmp_path):
    vault = tmp_path / "Obsidian_Vault"
    note = _write_note(vault, "m2.md", "Speaker 1")
    before = note.read_text()
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    plan = d.rename_plan("Speaker 1", "Jane Doe")
    assert plan["in_db"] is True
    assert plan["merges_into_existing"] is False
    assert plan["n_mentions"] == 2                  # participants + speaker header
    assert [os.path.basename(p) for p, _ in plan["notes"]] == ["m2.md"]
    assert note.read_text() == before               # preview must not write
    assert json.load(open(os.path.join(ws["workspace"], "speakers.json"))).get("Speaker 1")


def test_apply_rename_updates_db_note_and_sidecar(ws, tmp_path):
    vault = tmp_path / "Obsidian_Vault"
    note = _write_note(vault, "m2.md", "Speaker 1")
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    d.apply_rename(d.rename_plan("Speaker 1", "Jane Doe"))

    db = json.load(open(os.path.join(ws["workspace"], "speakers.json")))
    assert "Jane Doe" in db and "Speaker 1" not in db

    text = note.read_text()
    assert "[[Jane Doe]]" in text and "Speaker 1" not in text

    sc = json.load(open(os.path.join(ws["workspace"], "m2.segments.json")))
    assert any(r["label"] == "Jane Doe" for r in sc["diarization"])


def test_apply_rename_into_existing_name_merges_voiceprints(ws):
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    plan = d.rename_plan("Speaker 1", "Alice")
    assert plan["merges_into_existing"] is True
    d.apply_rename(plan)
    db = json.load(open(os.path.join(ws["workspace"], "speakers.json")))
    assert "Speaker 1" not in db
    assert abs(np.linalg.norm(np.array(db["Alice"])) - 1.0) < 1e-6


def test_rename_does_not_touch_a_lookalike_name(ws, tmp_path):
    vault = tmp_path / "Obsidian_Vault"
    note = _write_note(vault, "m3.md", "Speaker 10")
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    plan = d.rename_plan("Speaker 1", "Jane Doe")
    assert all(os.path.basename(p) != "m3.md" for p, _ in plan["notes"])
    d.apply_rename(plan)
    assert "[[Speaker 10]]" in note.read_text()


def test_apply_rename_updates_the_glossary(ws):
    gp = os.path.join(ws["workspace"], "glossary_learned.json")
    with open(gp, "w") as f:
        json.dump({"Speaker 1": 2, "PBV": 3}, f)
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    d.apply_rename(d.rename_plan("Speaker 1", "Jane Doe"))
    store = json.load(open(gp))
    assert "Jane Doe" in store and "Speaker 1" not in store


def test_merges_into_existing_false_when_old_not_in_db(ws):
    # "Alice" already has a voiceprint, but "Not In Db" never enrolled one --
    # apply_rename's merge branch is gated on in_db, so no voiceprint action
    # will happen. The flag must agree, or a confirm screen built on top of it
    # will tell the user a merge is about to occur when it isn't.
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    plan = d.rename_plan("Not In Db", "Alice")
    assert plan["in_db"] is False
    assert plan["merges_into_existing"] is False


def test_rename_plan_includes_sidecar_with_old_only_in_transcript(ws):
    # apply_rename rewrites labels in BOTH diarization and transcript rows, but
    # rename_plan used to check only diarization when deciding which sidecars
    # to list. A sidecar carrying `old` solely in a transcript row must still
    # show up in the plan, or the UI preview would silently miss it.
    sc_path = os.path.join(ws["workspace"], "m5.segments.json")
    with open(sc_path, "w") as f:
        json.dump({
            "wav": os.path.join(ws["workspace"], "m5.wav"),
            "note": os.path.join(ws["vault"], "m5.md"),
            "diarization": [{"idx": 0, "start": 0, "end": 5, "label": "SPEAKER_00",
                              "cluster": "SPEAKER_00", "source": "diar"}],
            "transcript": [{"idx": 0, "start": 0, "end": 5, "label": "Speaker 1",
                             "text": "hi"}],
        }, f)

    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    plan = d.rename_plan("Speaker 1", "Jane Doe")
    assert any(p.endswith("m5.segments.json") for p in plan["sidecars"])

    d.apply_rename(plan)
    sc = json.load(open(sc_path))
    assert sc["transcript"][0]["label"] == "Jane Doe"


def test_apply_rename_does_not_truncate_a_target_file_on_mid_write_failure(ws, monkeypatch):
    # Proves atomicity, not just claims it: a write that fails partway through
    # must leave the pre-existing target file intact, never zero-byted. The
    # second json serialize call (the glossary, after the DB write that
    # precedes it in apply_rename) is made to raise. Against the old
    # open(p, "w") + json.dump(...) code this fires AFTER the file has already
    # been truncated by the open call, so the glossary is left empty/corrupt --
    # this test fails (correctly) against that code. Against an atomic
    # write-to-temp-then-replace implementation, the serialize (json.dumps)
    # happens BEFORE any file is opened, so a raise there never touches the
    # real file at all.
    gp = os.path.join(ws["workspace"], "glossary_learned.json")
    glossary_before = {"Speaker 1": 2, "PBV": 3}
    with open(gp, "w") as f:
        json.dump(glossary_before, f)

    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    plan = d.rename_plan("Speaker 1", "Jane Doe")
    assert plan["in_db"] is True          # write #1: speakers.json
    assert plan["glossary"] is True       # write #2: glossary_learned.json

    calls = {"n": 0}
    real_dump, real_dumps = json.dump, json.dumps

    def fake_dump(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real_dump(*a, **kw)

    def fake_dumps(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real_dumps(*a, **kw)

    monkeypatch.setattr(json, "dump", fake_dump)
    monkeypatch.setattr(json, "dumps", fake_dumps)

    with pytest.raises(RuntimeError):
        d.apply_rename(plan)

    with open(gp) as f:
        assert json.load(f) == glossary_before   # untouched, not truncated


def test_apply_rename_updates_db_matched_cluster_id_and_transcript_label(ws):
    # A sidecar whose cluster was already DB-matched at pipeline time (cluster id
    # "db::<name>") and whose transcript rows carry the same label -- both must be
    # renamed too, so a later re-review of this meeting shows the corrected name.
    sc_path = os.path.join(ws["workspace"], "m4.segments.json")
    with open(sc_path, "w") as f:
        json.dump({
            "wav": os.path.join(ws["workspace"], "m4.wav"),
            "note": os.path.join(ws["vault"], "m4.md"),
            "diarization": [{"idx": 0, "start": 0, "end": 5, "label": "Speaker 1",
                              "cluster": "db::Speaker 1", "source": "db"}],
            "transcript": [{"idx": 0, "start": 0, "end": 5, "label": "Speaker 1",
                             "text": "hello"}],
        }, f)

    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    d.apply_rename(d.rename_plan("Speaker 1", "Jane Doe"))

    sc = json.load(open(sc_path))
    assert sc["diarization"][0]["cluster"] == "db::Jane Doe"
    assert sc["diarization"][0]["label"] == "Jane Doe"
    assert sc["transcript"][0]["label"] == "Jane Doe"


def _local_name_ws(tmp_path, mixed):
    """A workspace with one enrolled local-user print and one sidecar whose
    cluster embedding matches it by cosine (i.e. would match if not excluded)."""
    vault = tmp_path / "Obsidian_Vault"
    vault.mkdir()
    (tmp_path / "embedding_mean.json").write_text(
        json.dumps({"mean": MEAN, "count": 10, "meetings": []}))
    (tmp_path / "speakers.json").write_text(
        json.dumps({"Alex Morgan": LOCAL_VECTOR}))
    (tmp_path / "m1.segments.json").write_text(json.dumps({
        "wav": str(tmp_path / "m1.wav"), "note": str(vault / "m1.md"),
        "mixed": mixed,
        "diarization": [_turn(0, 30, "SPEAKER_00", "Unknown (SPEAKER_00)", LOCAL_VECTOR)],
        "transcript": [],
    }))
    return {"workspace": str(tmp_path), "vault": str(vault)}


def test_dual_mode_sidecar_never_matches_the_local_users_print(tmp_path):
    """A dual-channel meeting's cluster must never be credited to the local
    user's enrolled print, and must never be offered as their sample clip --
    the same hazard the pipeline's _db_match `exclude` guards against, but
    here across meetings via the cross-meeting directory (which also drives
    rename-everywhere)."""
    ws = _local_name_ws(tmp_path, mixed=False)
    d = SpeakerDirectory(ws["workspace"], ws["vault"], local_names=["Alex Morgan"])
    by = {e["name"]: e for e in d.identities()}
    assert by["Alex Morgan"]["n_meetings"] == 0
    assert by["Alex Morgan"]["clips"] == []
    assert d.best_clip("Alex Morgan") is None


def test_mixed_mode_sidecar_still_allows_the_local_users_print_to_match(tmp_path):
    """The exclusion must not undo the whole point of enrolling the local user
    from mixed recordings: a mixed meeting's cluster matching their print is
    the legitimate case and must still be credited."""
    ws = _local_name_ws(tmp_path, mixed=True)
    d = SpeakerDirectory(ws["workspace"], ws["vault"], local_names=["Alex Morgan"])
    by = {e["name"]: e for e in d.identities()}
    assert by["Alex Morgan"]["n_meetings"] == 1
    assert by["Alex Morgan"]["total_dur"] == pytest.approx(30.0)
    wav, start, end = d.best_clip("Alex Morgan")
    assert (start, end) == (0.0, 30.0)


def test_default_local_names_behaves_exactly_as_before(ws):
    """Constructing SpeakerDirectory without local_names (every existing
    caller/test) must be unaffected by this exclusion -- no keys excluded,
    identical results to before local_names existed."""
    d = SpeakerDirectory(ws["workspace"], ws["vault"])
    assert d.local_names == []
    assert d._local_db_keys == ()
    by = {e["name"]: e for e in d.identities()}
    assert by["Alice"]["n_meetings"] == 2
    assert by["Alice"]["total_dur"] == pytest.approx(40.0)
    assert by["Speaker 1"]["n_meetings"] == 1
