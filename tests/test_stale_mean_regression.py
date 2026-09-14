"""Reproduces the Manage Speakers crash after the ECAPA -> WeSpeaker switch.

Clicking "Manage Speakers" raised

    ValueError: operands could not be broadcast together with shapes (256,) (192,)

because embedding_mean.json still held the 192-dim ECAPA mean while sidecars and
voiceprints had moved to 256-dim WeSpeaker, and both the speaker directory and
the review screen subtracted it unconditionally. Only the directory was reported;
review carried the same fault and would have crashed next.
"""

import json

import numpy as np
import pytest

from ai.speaker_directory import SpeakerDirectory
from ai.speaker_review import SpeakerReview

NEW_DIM, OLD_DIM = 256, 192


@pytest.fixture
def workspace(tmp_path):
    """A workspace mid-migration: new-width data, stale old-width mean."""
    ws = tmp_path / "workspace"
    vault = ws / "Obsidian_Vault"
    vault.mkdir(parents=True)

    vec = (np.arange(NEW_DIM, dtype=float) / NEW_DIM).tolist()
    (ws / "speakers.json").write_text(json.dumps({"Jordan Lee": vec}))
    # The stale mean left behind by the previous embedding model.
    (ws / "embedding_mean.json").write_text(
        json.dumps({"mean": [0.01] * OLD_DIM, "count": 4787, "meetings": []}))

    wav = ws / "meeting_2026-01-01_00-00-00.wav"
    wav.write_bytes(b"stub")
    note = vault / "2026-01-01_00-00-00_Meeting.md"
    note.write_text("# Meeting\n\n## Transcript\n**[[Jordan Lee]]** (00:00):\nhi\n")
    (ws / "meeting_2026-01-01_00-00-00.segments.json").write_text(json.dumps({
        "wav": str(wav), "note": str(note),
        "transcript": [{"start": 0.0, "end": 5.0, "text": "hi", "label": "Jordan Lee"}],
        "diarization": [
            {"start": float(i), "end": float(i) + 5.0, "label": "Jordan Lee",
             "cluster": "db::Jordan Lee", "source": "db", "embedding": vec}
            for i in range(6)
        ],
    }))
    return ws


def test_manage_speakers_opens_with_a_stale_mean(workspace):
    d = SpeakerDirectory(str(workspace), str(workspace / "Obsidian_Vault"))
    identities = d.identities()          # this raised ValueError before
    assert any(e["name"] == "Jordan Lee" for e in identities), identities


def test_review_opens_with_a_stale_mean(workspace):
    r = SpeakerReview(
        str(workspace / "meeting_2026-01-01_00-00-00.segments.json"),
        str(workspace / "speakers.json"), placeholder_min_duration=1e9)
    assert r.mean is None, "a 192-dim mean must not apply to 256-dim embeddings"
    r.commit()                            # enrollment path also subtracts it


def test_a_matching_mean_is_still_applied(workspace):
    # The guard must not silently disable centering for the model that needs it.
    (workspace / "embedding_mean.json").write_text(
        json.dumps({"mean": [0.01] * NEW_DIM, "count": 10, "meetings": []}))
    r = SpeakerReview(
        str(workspace / "meeting_2026-01-01_00-00-00.segments.json"),
        str(workspace / "speakers.json"), placeholder_min_duration=1e9)
    assert r.mean is not None and r.mean.shape == (NEW_DIM,)
