import struct
import time

import numpy as np
import pytest

from core.sck_capture import (
    FramedDecoder, SCKCaptureSource, sck_mode_provides_mic, TYPE_MIC, TYPE_SYSTEM,
)


# ---------------------------------------------------------------------------
# FramedDecoder
# ---------------------------------------------------------------------------

def _frame(ftype, samples):
    return struct.pack("<BI", ftype, len(samples)) + b"".join(struct.pack("<f", s) for s in samples)


def _collect():
    mic, sysd = [], []
    dec = FramedDecoder(on_mic=lambda a: mic.append(a), on_system=lambda a: sysd.append(a))
    return dec, mic, sysd


def test_framed_routes_mic_and_system_by_type():
    dec, mic, sysd = _collect()
    dec.feed(_frame(TYPE_MIC, [1.0, 2.0]) + _frame(TYPE_SYSTEM, [3.0, 4.0, 5.0]))
    assert np.allclose(np.concatenate(mic), [1.0, 2.0])
    assert np.allclose(np.concatenate(sysd), [3.0, 4.0, 5.0])
    assert mic[0].dtype == np.float32


def test_framed_carries_partial_header_across_feeds():
    dec, mic, sysd = _collect()
    raw = _frame(TYPE_MIC, [7.0, 8.0])
    dec.feed(raw[:3])        # 3 of the 5 header bytes
    assert not mic
    dec.feed(raw[3:])
    assert np.allclose(np.concatenate(mic), [7.0, 8.0])


def test_framed_carries_partial_payload_across_feeds():
    dec, mic, sysd = _collect()
    raw = _frame(TYPE_SYSTEM, [1.5, 2.5, 3.5])
    dec.feed(raw[:9])        # header (5) + 1 of 3 floats — frame incomplete
    assert not sysd          # frame-atomic: nothing emitted until the whole frame arrives
    dec.feed(raw[9:])        # completes the frame
    assert np.allclose(np.concatenate(sysd), [1.5, 2.5, 3.5])


def test_framed_emitted_arrays_are_writable():
    dec, mic, _ = _collect()
    dec.feed(_frame(TYPE_MIC, [1.0]))
    mic[0][0] = 9.0          # must not raise (downstream mute-zeroing writes in place)


def test_framed_unknown_type_raises():
    dec, _, _ = _collect()
    with pytest.raises(ValueError):
        dec.feed(struct.pack("<BI", 7, 0))    # type 7 is not mic/system


def test_framed_empty_feed_is_noop():
    dec, mic, sysd = _collect()
    dec.feed(b"")
    assert not mic and not sysd


# ---------------------------------------------------------------------------
# SCKCaptureSource — fake helper subprocesses stand in for native/sck_capture,
# so the subprocess plumbing is testable with no ScreenCaptureKit, permissions,
# or hardware.
# ---------------------------------------------------------------------------

def _write_fake(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/usr/bin/env python3\n" + body)
    p.chmod(0o755)
    return str(p)


# Announce mode + mic, then stream one mic frame and one system frame.
_FRAMED_BURST = (
    "import struct, sys\n"
    "sys.stderr.write('RESULT=MODE mic+system\\n'); sys.stderr.write('RESULT=MIC Fake Mic\\n'); sys.stderr.flush()\n"
    "def f(t, xs): return struct.pack('<BI', t, len(xs)) + b''.join(struct.pack('<f', x) for x in xs)\n"
    "sys.stdout.buffer.write(f(0, [1.0, 2.0]) + f(1, [3.0, 4.0, 5.0]))\n"
    "sys.stdout.buffer.flush()\n"
)

# Stream a framed mic frame continuously until SIGTERM.
_FRAMED_STREAM = (
    "import struct, sys, signal, time\n"
    "run = [True]\n"
    "signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))\n"
    "frame = struct.pack('<BI', 0, 1) + struct.pack('<f', 1.0)\n"
    "while run[0]:\n"
    "    try:\n"
    "        sys.stdout.buffer.write(frame); sys.stdout.buffer.flush()\n"
    "    except BrokenPipeError:\n"
    "        break\n"
    "    time.sleep(0.001)\n"
    "sys.exit(0)\n"
)


def test_source_missing_binary_raises(tmp_path):
    src = SCKCaptureSource(str(tmp_path / "nope"), 48000,
                           on_mic_chunk=lambda c: None, on_system_chunk=lambda c: None)
    with pytest.raises(FileNotFoundError):
        src.start()


def test_source_routes_framed_streams_and_reads_handshake(tmp_path):
    binary = _write_fake(tmp_path, "fake_framed.py", _FRAMED_BURST)
    mic, sysd = [], []
    src = SCKCaptureSource(binary, 48000,
                           on_mic_chunk=lambda c: mic.extend(c.tolist()),
                           on_system_chunk=lambda c: sysd.extend(c.tolist()),
                           on_stderr=lambda l: None)
    src.start()
    mode = src.wait_for_mode(timeout=5.0)
    deadline = time.time() + 5
    while len(sysd) < 3 and time.time() < deadline:
        time.sleep(0.01)
    src.stop()
    assert mode == "mic+system"
    assert src.mic_name == "Fake Mic"
    assert np.allclose(mic, [1.0, 2.0])
    assert np.allclose(sysd, [3.0, 4.0, 5.0])


def test_source_passes_mic_device_id_argv(tmp_path):
    # Helper echoes its argv to stderr so we can assert the mic id is forwarded.
    body = "import sys\nsys.stderr.write('ARGV=' + '|'.join(sys.argv[1:]) + '\\n')\nsys.stderr.flush()\n"
    binary = _write_fake(tmp_path, "fake_argv.py", body)
    seen = []
    src = SCKCaptureSource(binary, 48000, on_mic_chunk=lambda c: None,
                           on_system_chunk=lambda c: None, mic_device_id="UID-123",
                           on_stderr=lambda l: seen.append(l))
    src.start()
    deadline = time.time() + 5
    while not any("ARGV=" in l for l in seen) and time.time() < deadline:
        time.sleep(0.01)
    src.stop()
    assert any("ARGV=48000|UID-123" in l for l in seen)


def test_wait_for_mode_times_out_to_none(tmp_path):
    body = "import time\ntime.sleep(2)\n"                # never prints a mode
    binary = _write_fake(tmp_path, "fake_silent.py", body)
    src = SCKCaptureSource(binary, 48000, on_mic_chunk=lambda c: None, on_system_chunk=lambda c: None)
    src.start()
    assert src.wait_for_mode(timeout=0.2) is None
    src.stop()


def test_source_is_alive_tracks_process_lifecycle(tmp_path):
    binary = _write_fake(tmp_path, "fake_stream.py", _FRAMED_STREAM)
    mic = []
    src = SCKCaptureSource(binary, 48000, on_mic_chunk=lambda c: mic.extend(c.tolist()),
                           on_system_chunk=lambda c: None)
    assert not src.is_alive()  # not started yet
    src.start()
    deadline = time.time() + 5
    while not mic and time.time() < deadline:
        time.sleep(0.01)
    assert src.is_alive()      # streaming
    src.stop()
    assert not src.is_alive()  # exited after SIGTERM


def test_source_captures_stderr_tail(tmp_path):
    body = (
        "import sys\n"
        "sys.stderr.write('RESULT=ERROR permission denied\\n')\n"
        "sys.stderr.flush()\n"
    )
    binary = _write_fake(tmp_path, "fake_err.py", body)
    src = SCKCaptureSource(binary, 48000, on_mic_chunk=lambda c: None,
                           on_system_chunk=lambda c: None, on_stderr=lambda l: None)
    src.start()
    deadline = time.time() + 5
    while src.is_alive() and time.time() < deadline:
        time.sleep(0.01)
    src.stop()
    assert "RESULT=ERROR" in src.stderr_tail()


def test_source_stop_terminates_cleanly(tmp_path):
    binary = _write_fake(tmp_path, "fake_stream.py", _FRAMED_STREAM)
    mic = []
    src = SCKCaptureSource(binary, 48000, on_mic_chunk=lambda c: mic.extend(c.tolist()),
                           on_system_chunk=lambda c: None)
    src.start()
    # Poll on real output (not a fixed sleep) so the fake subprocess's cold-start
    # never leaves this racing a fixed deadline; also guarantees its SIGTERM
    # handler is installed before stop(), so the returncode assertion is stable.
    deadline = time.time() + 5
    while not mic and time.time() < deadline:
        time.sleep(0.01)
    src.stop()
    assert len(mic) > 0                 # received streaming audio
    assert src._proc.returncode == 0    # SIGTERM handled gracefully


def test_sck_mode_provides_mic_predicate():
    assert sck_mode_provides_mic("mic+system") is True
    assert sck_mode_provides_mic("system-only") is False
    assert sck_mode_provides_mic(None) is False
