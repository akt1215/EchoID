import pytest

from core import audio_recorder
from core.audio_recorder import AudioRecorder, CaptureStartupError


class _DummyStream:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class _DummyMonitor:
    def start(self):
        pass

    def stop(self):
        pass


def test_sck_backend_has_no_audio_monitor():
    # SCK taps audio that already plays out the speakers; a monitor would echo.
    rec = AudioRecorder(0, None, "/tmp/x.wav", samplerate=48000,
                        monitor_enabled=True, capture_backend="sck",
                        sck_binary_path="/tmp/sck_capture")
    assert rec.audio_monitor is None
    assert rec.capture_backend == "sck"


def test_blackhole_backend_keeps_audio_monitor():
    rec = AudioRecorder(0, 3, "/tmp/x.wav", samplerate=48000,
                        monitor_enabled=True, capture_backend="blackhole")
    assert rec.audio_monitor is not None


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        AudioRecorder(0, 3, "/tmp/x.wav", capture_backend="bogus")


def test_sck_backend_requires_binary_path():
    with pytest.raises(ValueError):
        AudioRecorder(0, None, "/tmp/x.wav", capture_backend="sck")  # no sck_binary_path


def test_sck_start_missing_binary_raises_capture_startup_error(tmp_path):
    # A missing helper (setup.sh not run) must fail the same graceful way a
    # denied permission does — as CaptureStartupError, before starting anything.
    rec = AudioRecorder(0, None, str(tmp_path / "x.wav"), samplerate=48000,
                        capture_backend="sck", sck_binary_path=str(tmp_path / "nope"))
    with pytest.raises(CaptureStartupError):
        rec.start()
    assert not hasattr(rec, "mic_stream")   # failed before opening any device
    assert rec.recording is False


def test_sck_start_rejects_a_helper_with_a_mismatched_wire_format(tmp_path, monkeypatch):
    # A stale native/sck_capture from a branch speaking the tagged-frame
    # protocol must abort the recording. It is alive and streaming, so every
    # liveness check passes — but ch1 would be full-scale noise for the whole
    # meeting, unrecoverably (this cost two meetings on 2026-07-28).
    helper = tmp_path / "fake_framed.py"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        "import struct, sys, signal, time\n"
        "run = [True]\n"
        "signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))\n"
        "frame = struct.pack('<BI', 1, 960) + struct.pack('<f', 0.0) * 960\n"
        "while run[0]:\n"
        "    try:\n"
        "        sys.stdout.buffer.write(frame); sys.stdout.buffer.flush()\n"
        "    except BrokenPipeError:\n"
        "        break\n"
        "    time.sleep(0.001)\n"
    )
    helper.chmod(0o755)

    monkeypatch.setattr(audio_recorder.sd, "InputStream", _DummyStream)
    rec = AudioRecorder(0, None, str(tmp_path / "x.wav"), samplerate=48000,
                        capture_backend="sck", sck_binary_path=str(helper))
    rec.zoom_monitor = _DummyMonitor()

    with pytest.raises(CaptureStartupError) as excinfo:
        rec.start()
    assert "wire format" in str(excinfo.value)
    assert "swiftc" in str(excinfo.value)    # tells the user how to fix it
    assert rec.recording is False
    assert not hasattr(rec, "writer_thread")  # nothing was ever recorded
