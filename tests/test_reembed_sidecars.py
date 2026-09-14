"""Re-embedding must replace vectors and nothing else.

Sidecars written before the WeSpeaker switch hold 192-dim ECAPA vectors, so the
review screen and speaker directory compare them against a 256-dim DB. Only the
`embedding` field is stale — labels, clusters, sources and the transcript are
the user's reviewed work and must survive untouched.
"""

import json

import numpy as np
import pytest

from tools import reembed_sidecars as re_embed


class _StubManager:
    """Stands in for BiometricsManager: returns a fixed-width vector per turn."""

    def __init__(self, dim=4, fail_on=()):
        self.dim = dim
        self.fail_on = set(fail_on)
        self.calls = 0

    def embed_all(self, wav, segments):
        self.calls += 1
        out = []
        for i, _ in enumerate(segments):
            if i in self.fail_on:
                out.append(np.array([], dtype=float))
            else:
                out.append(np.arange(self.dim, dtype=float) + i)
        return out


def _sidecar(tmp_path, n=3, dim=2):
    wav = tmp_path / "m.wav"
    wav.write_bytes(b"stub")
    path = tmp_path / "meeting_2026-01-01_00-00-00.segments.json"
    path.write_text(json.dumps({
        "wav": str(wav),
        "note": "/vault/note.md",
        "transcript": [{"speaker": "Jordan Lee", "text": "hello"}],
        "diarization": [
            {"start": float(i), "end": float(i) + 1.0, "label": "Jordan Lee",
             "cluster": "db::Jordan Lee", "source": "db",
             "embedding": [0.5] * dim}
            for i in range(n)
        ],
    }))
    return path


def test_embeddings_are_replaced_with_the_new_width(tmp_path):
    p = _sidecar(tmp_path, n=3, dim=2)
    stats = re_embed.reembed(str(p), _StubManager(dim=4))
    d = json.loads(p.read_text())
    assert [len(s["embedding"]) for s in d["diarization"]] == [4, 4, 4]
    assert stats["old_dim"] == 2 and stats["new_dim"] == 4
    assert stats["turns"] == 3


def test_everything_except_embeddings_is_preserved(tmp_path):
    p = _sidecar(tmp_path)
    before = json.loads(p.read_text())
    re_embed.reembed(str(p), _StubManager())
    after = json.loads(p.read_text())
    assert after["note"] == before["note"]
    assert after["transcript"] == before["transcript"]
    assert after["wav"] == before["wav"]
    for a, b in zip(after["diarization"], before["diarization"]):
        for key in ("start", "end", "label", "cluster", "source"):
            assert a[key] == b[key]


def test_a_backup_is_written_before_replacing(tmp_path):
    p = _sidecar(tmp_path, dim=2)
    re_embed.reembed(str(p), _StubManager(dim=4))
    backup = tmp_path / "meeting_2026-01-01_00-00-00.segments.json.pre-reembed.bak"
    assert backup.exists()
    old = json.loads(backup.read_text())
    assert len(old["diarization"][0]["embedding"]) == 2


def test_dry_run_changes_nothing(tmp_path):
    p = _sidecar(tmp_path, dim=2)
    before = p.read_text()
    stats = re_embed.reembed(str(p), _StubManager(dim=4), dry_run=True)
    assert p.read_text() == before
    assert stats["new_dim"] == 4          # still reports what it would write


def test_a_failed_turn_keeps_an_empty_embedding(tmp_path):
    p = _sidecar(tmp_path, n=3)
    re_embed.reembed(str(p), _StubManager(fail_on=[1]))
    d = json.loads(p.read_text())
    assert d["diarization"][1]["embedding"] == []
    assert len(d["diarization"][0]["embedding"]) == 4


def test_a_missing_wav_is_skipped_not_fatal(tmp_path):
    p = _sidecar(tmp_path)
    d = json.loads(p.read_text())
    d["wav"] = str(tmp_path / "gone.wav")
    p.write_text(json.dumps(d))
    mgr = _StubManager()
    stats = re_embed.reembed(str(p), mgr)
    assert stats["skipped"] == "missing wav"
    assert mgr.calls == 0
    assert json.loads(p.read_text())["diarization"][0]["embedding"] == [0.5, 0.5]


def test_a_sidecar_without_turns_is_skipped(tmp_path):
    p = tmp_path / "meeting_2026-01-02_00-00-00.segments.json"
    p.write_text(json.dumps({"wav": str(tmp_path / "m.wav"), "diarization": []}))
    (tmp_path / "m.wav").write_bytes(b"stub")
    assert re_embed.reembed(str(p), _StubManager())["skipped"] == "no turns"


def test_rerunning_does_not_stack_backups(tmp_path):
    # The first backup holds the ONLY pre-switch vectors; a second run must not
    # overwrite it with already-re-embedded ones.
    p = _sidecar(tmp_path, dim=2)
    re_embed.reembed(str(p), _StubManager(dim=4))
    re_embed.reembed(str(p), _StubManager(dim=8))
    backup = tmp_path / "meeting_2026-01-01_00-00-00.segments.json.pre-reembed.bak"
    assert len(json.loads(backup.read_text())["diarization"][0]["embedding"]) == 2
