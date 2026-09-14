"""A correction made in review must survive in the sidecar, not only the note.

`commit()` wrote the voiceprint DB and the note, and left the sidecar carrying
the names the pipeline had guessed. Everything downstream reads the sidecar:
`tools/retranscribe.py` rebuilds a note from `diarization[].label`, the review
picker lists those labels to identify a meeting, and re-opening the namer shows
them again. So a user's correction was one `retranscribe` away from being
silently undone -- which is exactly how three meetings kept a name the visual
reader misread through a re-transcription that was supposed to preserve
"the reviewed speakers".
"""

import json

from ai.speaker_review import SpeakerReview


def _commit_rename(sidecar, old, new):
    r = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    gid = next(s["id"] for s in r.list_speakers() if s["name"] == old)
    r.rename(gid, new)
    r.commit()
    with open(sidecar["sidecar"]) as f:
        return json.load(f)


def test_commit_writes_the_confirmed_name_into_the_sidecar(sidecar):
    data = _commit_rename(sidecar, "Alice", "Alicia")

    labels = {d["label"] for d in data["diarization"]}
    assert "Alicia" in labels and "Alice" not in labels
    assert "Alicia" in {t["label"] for t in data["transcript"]}


def test_committing_leaves_the_turns_themselves_untouched(sidecar):
    with open(sidecar["sidecar"]) as f:
        before = json.load(f)
    after = _commit_rename(sidecar, "Alice", "Alicia")

    assert len(after["diarization"]) == len(before["diarization"])
    for a, b in zip(after["diarization"], before["diarization"]):
        # Cluster ids key the evaluation ground truth and the embeddings are the
        # voiceprint evidence: a rename must move neither.
        # `.get`: this fixture predates cluster ids, which is itself the case
        # worth covering -- a legacy sidecar must not gain or lose the key.
        assert a.get("cluster") == b.get("cluster")
        assert a["embedding"] == b["embedding"]
        assert (a["start"], a["end"]) == (b["start"], b["end"])


def test_a_confirmed_name_is_recorded_as_reviewed_not_as_the_pipeline_guess(sidecar):
    data = _commit_rename(sidecar, "Alice", "Alicia")

    sources = {d["label"]: d.get("source") for d in data["diarization"]}
    assert sources["Alicia"] == "review", (
        "a name the user confirmed must not still claim it came from the DB or "
        "the visual reader")


def test_the_correction_survives_reopening_the_review(sidecar):
    _commit_rename(sidecar, "Alice", "Alicia")

    reopened = SpeakerReview(sidecar["sidecar"], sidecar["db"])
    names = [s["name"] for s in reopened.list_speakers()]
    assert "Alicia" in names and "Alice" not in names


def test_me_local_segments_keep_their_label(sidecar):
    data = _commit_rename(sidecar, "Alice", "Alicia")

    assert any(t["label"] == "Me (Local)" for t in data["transcript"]), (
        "the local user's segments are decided by channel energy, not by the "
        "review, and must not be reattributed by it")
