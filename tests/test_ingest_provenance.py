import json
import os

import numpy as np
import soundfile as sf

from core import ingest


def _stereo(path, marker=None):
    with sf.SoundFile(str(path), "w", samplerate=48000, channels=2,
                      subtype="PCM_16") as f:
        if marker is not None:
            f.comment = marker
        f.write(np.zeros((4800, 2), dtype="float32"))


def test_a_stamped_recording_is_native(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav, ingest.MARKER)
    assert ingest.is_native(str(wav)) is True


def test_an_unstamped_stereo_wav_is_not_native(tmp_path):
    """Channel count is not provenance: QuickTime and OBS produce stereo too."""
    wav = tmp_path / "quicktime.wav"
    _stereo(wav)
    assert ingest.is_native(str(wav)) is False


def test_a_foreign_comment_is_not_native(tmp_path):
    wav = tmp_path / "other.wav"
    _stereo(wav, "Recorded with SomeOtherApp")
    assert ingest.is_native(str(wav)) is False


def _stamp_layout(wav, **fields):
    """Write a layout sidecar beside `wav`, defaulting size/mtime to the WAV's
    own so a test opts into a mismatch explicitly rather than by omission."""
    st = os.stat(wav)
    record = {"layout": "dual", "size": st.st_size, "mtime": st.st_mtime}
    record.update(fields)
    with open(ingest.layout_marker_path(str(wav)), "w") as f:
        json.dump(record, f)


def test_a_layout_sidecar_marks_a_legacy_recording_native(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    _stamp_layout(wav, source="test")
    assert ingest.is_native(str(wav)) is True


def test_a_layout_sidecar_saying_mixed_does_not_grant_trust(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    _stamp_layout(wav, layout="mixed")
    assert ingest.is_native(str(wav)) is False


def test_a_corrupt_layout_sidecar_does_not_grant_trust(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    (tmp_path / "meeting_2026-01-02_03-04-05.layout.json").write_text("{ not json")
    assert ingest.is_native(str(wav)) is False


def test_a_stale_layout_sidecar_does_not_grant_trust(tmp_path):
    """A layout sidecar names its WAV by basename only. If a foreign capture
    later replaces the stamped recording at that same path, the stale sidecar
    -- still saying "dual" -- must not grant it trust; the recorded size/mtime
    no longer match the file actually sitting there."""
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav)
    _stamp_layout(wav)
    assert ingest.is_native(str(wav)) is True   # sanity: matches right now

    # A different, longer foreign capture lands at the same path.
    with sf.SoundFile(str(wav), "w", samplerate=48000, channels=2,
                      subtype="PCM_16") as f:
        f.write(np.zeros((9600, 2), dtype="float32"))
    assert ingest.is_native(str(wav)) is False


def test_a_non_wav_is_never_native(tmp_path):
    src = tmp_path / "zoom_0.mp4"
    src.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    assert ingest.is_native(str(src)) is False


def test_an_unstamped_stereo_wav_is_planned_as_mixed(tmp_path):
    wav = tmp_path / "quicktime.wav"
    _stereo(wav)
    p = ingest.plan_for(str(wav), str(tmp_path))
    assert p.action == "convert"
    assert p.mixed is True


def test_a_stamped_recording_passes_through_as_dual(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav, ingest.MARKER)
    p = ingest.plan_for(str(wav), str(tmp_path))
    assert p.action == "passthrough"
    assert p.mixed is False


def test_mixed_flag_overrides_a_stamped_recording(tmp_path):
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav, ingest.MARKER)
    p = ingest.plan_for(str(wav), str(tmp_path), mixed_flag=True)
    assert p.action == "convert"
    assert p.mixed is True


def test_a_recorded_wav_is_recognized_by_ingest(tmp_path):
    """End-to-end: what AudioRecorder writes is what is_native accepts."""
    from core.audio_recorder import AudioRecorder
    out = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    with sf.SoundFile(str(out), "w", samplerate=48000, channels=2,
                      subtype="PCM_16") as f:
        f.comment = AudioRecorder.MARKER
        f.write(np.zeros((4800, 2), dtype="float32"))
    assert ingest.is_native(str(out)) is True


def test_converting_a_workspace_recording_never_targets_the_recording(tmp_path):
    """The destructive case: every recording this tool makes is named
    meeting_<ts>.wav inside the workspace, so a flat target would resolve to the
    source itself and the downmix would overwrite the only copy."""
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _stereo(wav, ingest.MARKER)
    p = ingest.plan_for(str(wav), str(tmp_path), mixed_flag=True)
    assert p.action == "convert"
    assert os.path.abspath(p.target) != os.path.abspath(p.source)
    assert os.path.dirname(p.target).endswith(ingest.IMPORTED_DIR)
