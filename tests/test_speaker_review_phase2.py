"""Typeahead suggestions and auto-stop playback in SpeakerReview."""

import json

import numpy as np
import soundfile as sf

from ai.speaker_review import SpeakerReview


def _sidecar(tmp_path, diarization, wav=""):
    data = {"wav": wav, "note": "", "diarization": diarization, "transcript": []}
    p = tmp_path / "m.segments.json"
    p.write_text(json.dumps(data))
    return str(p)


# ── suggestion_names (typeahead source) ────────────────────────────────────

def test_suggestion_names_unions_db_and_session_names(tmp_path):
    db = tmp_path / "speakers.json"
    db.write_text(json.dumps({"Riley Stone": [1.0, 0], "Taylor Kim": [0, 1.0]}))
    sidecar = _sidecar(tmp_path, [
        {"idx": 0, "start": 0, "end": 3, "label": "Unknown (SPEAKER_00)",
         "cluster": "SPEAKER_00", "source": "unknown", "embedding": [0, 1.0]},
    ])
    r = SpeakerReview(sidecar, str(db))
    gid = r.list_speakers()[0]["id"]
    r.rename(gid, "Emilia Hermann")  # a name typed this session

    names = r.suggestion_names()
    assert "Riley Stone" in names            # from the DB
    assert "Taylor Kim" in names
    assert "Emilia Hermann" in names           # from this session
    assert all(not n.startswith("Unknown (") for n in names)  # no placeholders
    assert "Me (Local)" not in names
    assert names == sorted(names)              # stable, sorted


# ── auto-stop playback ─────────────────────────────────────────────────────

class _FakeProc:
    instances = []

    def __init__(self, *a, **k):
        self.terminated = False
        _FakeProc.instances.append(self)

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def _review_with_wav(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros((16000, 2), dtype="float32"), 16000)
    sidecar = _sidecar(tmp_path, [
        {"idx": 0, "start": 0.0, "end": 0.5, "label": "A", "cluster": "c0",
         "source": "db", "embedding": [1.0, 0]},
        {"idx": 1, "start": 0.5, "end": 1.0, "label": "A", "cluster": "c0",
         "source": "db", "embedding": [1.0, 0]},
    ], wav=str(wav))
    return SpeakerReview(sidecar, str(tmp_path / "db.json"))


def test_playing_second_clip_stops_the_first(tmp_path, monkeypatch):
    _FakeProc.instances = []
    monkeypatch.setattr("subprocess.Popen", _FakeProc)
    r = _review_with_wav(tmp_path)
    gid = r.list_speakers()[0]["id"]

    r.play_clip(gid)
    r.play_clip(gid)

    assert len(_FakeProc.instances) == 2
    assert _FakeProc.instances[0].terminated is True   # first was stopped
    assert _FakeProc.instances[1].terminated is False  # second still playing


def test_stop_playback_terminates_current(tmp_path, monkeypatch):
    _FakeProc.instances = []
    monkeypatch.setattr("subprocess.Popen", _FakeProc)
    r = _review_with_wav(tmp_path)
    gid = r.list_speakers()[0]["id"]

    r.play_clip(gid)
    r.stop_playback()

    assert _FakeProc.instances[0].terminated is True
