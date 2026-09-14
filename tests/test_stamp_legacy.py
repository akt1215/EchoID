import json
import os

import numpy as np
import soundfile as sf

from tools.stamp_legacy_recordings import provenance, stamp


def _stereo(path):
    sf.write(str(path), np.zeros((4800, 2), dtype="float32"), 48000,
             subtype="PCM_16")


def test_a_wav_named_by_a_sidecar_is_proven(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    (tmp_path / "meeting_2026-01-02_03-04-05.segments.json").write_text(
        json.dumps({"wav": str(wav), "diarization": [], "transcript": []}))
    assert provenance(str(tmp_path)) == {str(wav): True}


def test_a_wav_with_no_sidecar_is_unproven(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    assert provenance(str(tmp_path)) == {str(wav): False}


def test_a_sidecar_naming_a_different_wav_does_not_prove_this_one(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    (tmp_path / "other.segments.json").write_text(
        json.dumps({"wav": str(tmp_path / "elsewhere.wav")}))
    assert provenance(str(tmp_path)) == {str(wav): False}


def test_a_reprocessed_foreign_wav_is_not_proven_by_its_own_sidecar(tmp_path):
    """A sidecar only proves the pipeline *read* the WAV -- main.py writes one
    for any --from-file path, including a foreign capture that happens to sit
    in workspace/. A WAV must also carry the recorder's own meeting_<ts>.wav
    filename form before it counts as proven; a QuickTime capture reprocessed
    with --from-file must not come out proven just because it now has a
    sidecar naming it."""
    wav = tmp_path / "quicktime.wav"
    _stereo(wav)
    (tmp_path / "quicktime.segments.json").write_text(
        json.dumps({"wav": str(wav)}))
    assert provenance(str(tmp_path)) == {str(wav): False}


def test_stamp_writes_a_dual_marker_ingest_accepts(tmp_path):
    from core import ingest
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    stamp(str(wav), reason="segments-sidecar")
    assert ingest.is_native(str(wav)) is True
    data = json.loads((tmp_path / "meeting_2026-01-02_03-04-05.layout.json").read_text())
    assert data["layout"] == "dual"
    assert data["reason"] == "segments-sidecar"
    st = os.stat(wav)
    assert data["size"] == st.st_size
    assert data["mtime"] == st.st_mtime


def test_stamp_never_rewrites_the_audio(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    before = wav.read_bytes()
    stamp(str(wav), reason="confirmed")
    assert wav.read_bytes() == before
