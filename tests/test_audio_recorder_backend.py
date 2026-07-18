import numpy as np
import pytest

from core import audio_recorder
from core.audio_recorder import AudioRecorder, CaptureStartupError
from core.sck_capture import sck_mode_provides_mic


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


def _sck_recorder():
    return AudioRecorder(0, None, "/tmp/x.wav", samplerate=48000,
                         monitor_enabled=True, capture_backend="sck",
                         sck_binary_path="/tmp/sck_capture")


def test_sck_mic_chunk_enqueues_when_not_muted():
    rec = _sck_recorder()
    rec._sck_provides_mic = True          # SCK owns the mic (mic+system mode)
    rec.zoom_monitor.is_muted = False
    rec._on_sck_mic_chunk(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(rec.mic_queue.get_nowait(), [1.0, 2.0, 3.0])


def test_sck_mic_chunk_zeroed_when_muted():
    rec = _sck_recorder()
    rec._sck_provides_mic = True
    rec.zoom_monitor.is_muted = True
    rec._on_sck_mic_chunk(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(rec.mic_queue.get_nowait(), [0.0, 0.0, 0.0])


def test_sck_mic_chunk_dropped_when_sck_does_not_own_mic():
    # If SCK is not the mic owner, a stray SCK mic frame must NOT reach ch0, so it
    # can never double up with a sounddevice mic.
    rec = _sck_recorder()
    rec._sck_provides_mic = False
    rec._on_sck_mic_chunk(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert rec.mic_queue.empty()


def test_sck_system_chunk_routes_to_bh_queue():
    rec = _sck_recorder()
    rec._on_sck_system_chunk(np.array([4.0, 5.0], dtype=np.float32))
    assert np.allclose(rec.bh_queue.get_nowait(), [4.0, 5.0])
    assert rec.mic_queue.empty()


def test_maybe_zero_muted_passthrough_and_zero():
    rec = _sck_recorder()
    rec.zoom_monitor.is_muted = False
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert np.allclose(rec._maybe_zero_muted(a), [1.0, 2.0])
    rec.zoom_monitor.is_muted = True
    assert np.allclose(rec._maybe_zero_muted(a), [0.0, 0.0])


def test_new_recorder_exposes_mic_status_fields():
    rec = _sck_recorder()
    assert rec.mic_silent is False
    assert rec.sck_mic_name is None


def test_provides_mic_predicate_drives_sounddevice_decision():
    # The predicate is the single decision point start() branches on.
    assert sck_mode_provides_mic("mic+system") is True     # SCK owns the mic
    assert sck_mode_provides_mic("system-only") is False    # not mic+system -> fail loud


# --- start() integration with a fake SCK helper -----------------------------
# These drive AudioRecorder.start() against a fake helper subprocess whose stderr
# mirrors the REAL Swift helper EXACTLY ([sck]-prefixed lines, RESULT=MIC before
# RESULT=MODE). This is the layer the unit-only tests missed: a handshake-format
# drift between helper and parser aborts every real recording but leaves the
# bare-token fakes green.

def _write_fake_helper(tmp_path, body):
    p = tmp_path / "fake_sck"
    p.write_text("#!/usr/bin/env python3\n" + body)
    p.chmod(0o755)
    return str(p)


_FAKE_SCK_MIC_SYSTEM = (
    "import struct, sys, signal, time\n"
    "run=[True]\n"
    "signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))\n"
    "sys.stderr.write('[sck] RESULT=MIC Fake Default Mic\\n')\n"
    "sys.stderr.write('[sck] RESULT=MODE mic+system\\n'); sys.stderr.flush()\n"
    "def f(t, xs): return struct.pack('<BI', t, len(xs)) + b''.join(struct.pack('<f', x) for x in xs)\n"
    "while run[0]:\n"
    "    try:\n"
    "        sys.stdout.buffer.write(f(0,[0.1,0.1]) + f(1,[0.2,0.2])); sys.stdout.buffer.flush()\n"
    "    except BrokenPipeError:\n"
    "        break\n"
    "    time.sleep(0.005)\n"
    "sys.exit(0)\n"
)

_FAKE_SCK_WRONG_MODE = (
    "import struct, sys, signal, time\n"
    "run=[True]\n"
    "signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))\n"
    "sys.stderr.write('[sck] RESULT=MODE system-only\\n'); sys.stderr.flush()\n"
    "def f(t, xs): return struct.pack('<BI', t, len(xs)) + b''.join(struct.pack('<f', x) for x in xs)\n"
    "while run[0]:\n"
    "    try:\n"
    "        sys.stdout.buffer.write(f(1,[0.2,0.2])); sys.stdout.buffer.flush()\n"
    "    except BrokenPipeError:\n"
    "        break\n"
    "    time.sleep(0.005)\n"
    "sys.exit(0)\n"
)


def test_sck_start_confirms_mic_system_and_records(tmp_path):
    # Regression for the handshake prefix/order bug: with the real helper stderr
    # format, start() must parse mic+system, take both channels from SCK (no
    # sounddevice mic), surface the mic name, and be recording.
    binary = _write_fake_helper(tmp_path, _FAKE_SCK_MIC_SYSTEM)
    rec = AudioRecorder(None, None, str(tmp_path / "out.wav"), samplerate=48000,
                        capture_backend="sck", sck_binary_path=binary)
    rec.start()
    try:
        assert rec._sck_provides_mic is True
        assert rec.sck_mic_name == "Fake Default Mic"
        assert not hasattr(rec, "mic_stream")   # no sounddevice mic on the sck path
        assert rec.recording is True
    finally:
        rec.stop()


def test_sck_start_fails_loud_when_not_mic_system(tmp_path):
    # A live-but-wrong-mode helper (e.g. macOS < 15 -> not mic+system) must fail
    # loud with CaptureStartupError and clean up, not record silently.
    binary = _write_fake_helper(tmp_path, _FAKE_SCK_WRONG_MODE)
    rec = AudioRecorder(None, None, str(tmp_path / "out.wav"), samplerate=48000,
                        capture_backend="sck", sck_binary_path=binary)
    with pytest.raises(CaptureStartupError):
        rec.start()
    assert rec.recording is False
    assert rec._sck_source is None
    assert not hasattr(rec, "mic_stream")
