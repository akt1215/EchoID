"""The directory must apply the same guards as the pipeline.

Manage Speakers played Sam Rivera's voice under "Avery Chen". The
voiceprint was right; the clip came from a cluster the user had marked as
holding more than one person, which still DB-matched Avery Chen at 0.881.

Two causes, both making the directory more permissive than the pipeline it is
meant to reflect: it ignored the user's `multiple` verdicts, and it applied
neither the runner-up margin nor the minimum-speech gate. It therefore showed
matches the pipeline itself would refuse.
"""

import json
import os

import numpy as np
import pytest

from ai.speaker_directory import SpeakerDirectory

DIM = 8


def _unit(*xs):
    v = np.zeros(DIM)
    for i, x in enumerate(xs):
        v[i] = x
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "workspace"
    (w / "Obsidian_Vault").mkdir(parents=True)
    (w / "eval").mkdir()
    (w / "speakers.json").write_text(json.dumps({
        "Avery Chen": _unit(1, 0),
        "Jordan Lee": _unit(0, 1),
    }))

    def sidecar(name, clusters):
        wav = w / (name + ".wav")
        wav.write_bytes(b"stub")
        (w / (name + ".segments.json")).write_text(json.dumps({
            "wav": str(wav), "note": "", "transcript": [],
            "diarization": [
                {"start": s, "end": e, "label": "x", "cluster": cl,
                 "source": "db", "embedding": vec}
                for cl, vec, spans in clusters for s, e in spans
            ],
        }))
    return w, sidecar


def test_a_cluster_marked_multiple_supplies_no_clip(ws):
    w, sidecar = ws
    # A long mixed cluster that matches Avery Chen, plus his real short one.
    sidecar("meeting_2026-01-01_00-00-00",
            [("SPEAKER_00", _unit(1, 0.1), [(0.0, 300.0)])])
    sidecar("meeting_2026-01-02_00-00-00",
            [("SPEAKER_00", _unit(1, 0), [(0.0, 60.0)])])
    (w / "eval" / "ground_truth.json").write_text(json.dumps({
        "meeting_2026-01-01_00-00-00": {
            "SPEAKER_00": {"name": None, "verdict": "multiple"}},
    }))

    d = SpeakerDirectory(str(w), str(w / "Obsidian_Vault"), db_threshold=0.5)
    entry = next(e for e in d.identities() if e["name"] == "Avery Chen")
    wavs = [os.path.basename(c[0]) for c in entry["clips"]]
    assert all("2026-01-01" not in n for n in wavs), \
        f"a clip came from a cluster holding several people: {wavs}"


def test_margin_rejects_an_ambiguous_cluster(ws):
    w, sidecar = ws
    # Sits between the two prints: neither match is trustworthy.
    sidecar("meeting_2026-01-03_00-00-00",
            [("SPEAKER_00", _unit(1, 0.95), [(0.0, 120.0)])])
    d = SpeakerDirectory(str(w), str(w / "Obsidian_Vault"),
                         db_threshold=0.5, db_margin=0.2)
    assert all(e["total_dur"] == 0 for e in d.identities()), \
        "a near-tie should not be credited to either identity"


def test_a_short_cluster_is_not_credited(ws):
    w, sidecar = ws
    sidecar("meeting_2026-01-04_00-00-00",
            [("SPEAKER_00", _unit(1, 0), [(0.0, 5.0)])])
    d = SpeakerDirectory(str(w), str(w / "Obsidian_Vault"),
                         db_threshold=0.5, min_speech=30.0)
    assert all(e["total_dur"] == 0 for e in d.identities())


def test_a_clean_long_cluster_is_still_credited(ws):
    w, sidecar = ws
    sidecar("meeting_2026-01-05_00-00-00",
            [("SPEAKER_00", _unit(1, 0), [(0.0, 120.0)])])
    d = SpeakerDirectory(str(w), str(w / "Obsidian_Vault"),
                         db_threshold=0.5, db_margin=0.2, min_speech=30.0)
    entry = next(e for e in d.identities() if e["name"] == "Avery Chen")
    assert entry["total_dur"] == 120.0 and entry["n_meetings"] == 1
    assert entry["clips"]


def test_guards_default_off_so_existing_callers_are_unchanged(ws):
    w, sidecar = ws
    sidecar("meeting_2026-01-06_00-00-00",
            [("SPEAKER_00", _unit(1, 0), [(0.0, 5.0)])])
    d = SpeakerDirectory(str(w), str(w / "Obsidian_Vault"), db_threshold=0.5)
    entry = next(e for e in d.identities() if e["name"] == "Avery Chen")
    assert entry["total_dur"] == 5.0
