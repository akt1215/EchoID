import os
import signal
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from core.sck_capture import SCKDecoder, SCKCaptureSource, wire_format_error


def _stereo_bytes(frames):
    """frames: list of (L, R) float tuples -> interleaved little-endian float32 bytes."""
    return b"".join(struct.pack("<ff", l, r) for l, r in frames)


def _framed_bytes(frames):
    """frames: list of (type, [samples]) -> the tagged-frame wire format
    ([1B type][uint32 LE count][count x float32]) emitted by the single-clock
    mic+system helper build. Feeding this to SCKDecoder is the July 28 bug."""
    out = b""
    for ftype, samples in frames:
        out += struct.pack("<BI", ftype, len(samples))
        out += b"".join(struct.pack("<f", s) for s in samples)
    return out


# --- wire-format guard -------------------------------------------------------
# native/sck_capture is a gitignored build artifact, so a binary built on a
# branch with a different wire format survives a checkout while the decoder
# changes underneath it. The mismatch is silent: bytes flow at full rate, so
# every liveness check passes, but the 5-byte tagged-frame header is not a
# multiple of 4 and permanently destroys float32 alignment — which reaches the
# WAV as full-scale noise. These pin the guard that makes it fail loudly.

def test_raw_stereo_stream_passes():
    assert wire_format_error(_stereo_bytes([(0.1, -0.2), (0.3, 0.05)] * 2000)) is None


def test_digital_silence_passes():
    # A meeting that opens quiet is normal, not a mismatch.
    assert wire_format_error(_stereo_bytes([(0.0, 0.0)] * 4000)) is None


def test_full_scale_audio_passes():
    # Legitimately loud audio sits at the rails; the guard must not mistake it
    # for garbage, or it would refuse to record a perfectly good meeting.
    assert wire_format_error(_stereo_bytes([(1.0, -1.0)] * 4000)) is None


def test_framed_protocol_is_rejected():
    raw = _framed_bytes([(1, [0.1] * 960), (0, [0.2] * 512), (1, [0.1] * 960)])
    err = wire_format_error(raw)
    assert err is not None
    assert "tagged-frame" in err


def test_silent_framed_protocol_is_still_rejected():
    # The realistic July 28 opening: the call starts quiet, so every payload
    # sample is 0.0 and misparsing it as float32 yields values that look
    # perfectly sane. Only the frame headers betray the mismatch — which is why
    # a value-plausibility check alone is not sufficient.
    raw = _framed_bytes([(1, [0.0] * 960)] * 4)
    err = wire_format_error(raw)
    assert err is not None
    assert "tagged-frame" in err


def test_misaligned_garbage_is_rejected():
    # Any other wire-format mismatch: float32 parsed off its alignment yields
    # absurd magnitudes and NaNs, exactly what reached the WAV on July 28.
    raw = np.array([1e30, -3.4e38, float("nan"), 0.0] * 1000, dtype="<f4").tobytes()
    assert wire_format_error(raw) is not None


def test_too_little_data_is_not_condemned():
    # Never reject on a couple of bytes; the caller buffers until it can judge.
    assert wire_format_error(b"") is None
    assert wire_format_error(b"\x00\x01") is None


def test_decoder_full_frames_downmix_to_mono():
    dec = SCKDecoder()
    out = dec.feed(_stereo_bytes([(1.0, 3.0), (2.0, 4.0)]))
    assert np.allclose(out, [2.0, 3.0])  # per-frame mean of L,R
    assert out.dtype == np.float32


def test_decoder_carries_partial_frame_across_feeds():
    dec = SCKDecoder()
    raw = _stereo_bytes([(1.0, 3.0), (2.0, 4.0)])  # 16 bytes, 2 frames
    first = dec.feed(raw[:12])   # 1.5 frames -> 1 mono sample, 4 bytes carried
    assert np.allclose(first, [2.0])
    second = dec.feed(raw[12:])  # completes frame 2
    assert np.allclose(second, [3.0])


def test_decoder_empty_input_returns_empty():
    dec = SCKDecoder()
    out = dec.feed(b"")
    assert out.size == 0
    assert out.dtype == np.float32


# A fake helper standing in for native/sck_capture: it takes a samplerate argv
# (ignored) and streams known PCM, so the subprocess plumbing is testable with
# no ScreenCaptureKit, permissions, or hardware.
_FAKE_BURST = (
    "import struct, sys\n"
    "frames = [(1.0, 3.0), (2.0, 4.0), (5.0, 7.0)]\n"
    "sys.stdout.buffer.write(b''.join(struct.pack('<ff', l, r) for l, r in frames))\n"
    "sys.stdout.buffer.flush()\n"
)

# Writes in buffers rather than single frames, as the real helper does (SCK
# hands over 960-sample buffers) — the source only releases audio once it has
# seen enough opening bytes to judge the wire format.
_FAKE_STREAM = (
    "import struct, sys, signal, time\n"
    "run = [True]\n"
    "signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))\n"
    "block = struct.pack('<ff', 1.0, 1.0) * 1024\n"
    "while run[0]:\n"
    "    try:\n"
    "        sys.stdout.buffer.write(block); sys.stdout.buffer.flush()\n"
    "    except BrokenPipeError:\n"
    "        break\n"
    "    time.sleep(0.001)\n"
    "sys.exit(0)\n"
)


def _write_fake(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/usr/bin/env python3\n" + body)
    p.chmod(0o755)
    return str(p)


def test_source_missing_binary_raises(tmp_path):
    src = SCKCaptureSource(str(tmp_path / "nope"), 48000, on_chunk=lambda c: None)
    with pytest.raises(FileNotFoundError):
        src.start()


def test_source_streams_decoded_mono_chunks(tmp_path):
    binary = _write_fake(tmp_path, "fake_burst.py", _FAKE_BURST)
    got = []
    src = SCKCaptureSource(binary, 48000, on_chunk=lambda c: got.extend(c.tolist()))
    src.start()
    # Poll for the burst's 3 decoded samples instead of a fixed sleep — the
    # Python fake subprocess's cold-start can exceed a fixed deadline under
    # full-suite CPU load, leaving got empty (np.allclose([], [...]) then raises).
    deadline = time.time() + 5
    while len(got) < 3 and time.time() < deadline:
        time.sleep(0.01)
    src.stop()
    assert np.allclose(got, [2.0, 3.0, 6.0])  # per-frame means


def test_source_is_alive_tracks_process_lifecycle(tmp_path):
    binary = _write_fake(tmp_path, "fake_stream.py", _FAKE_STREAM)
    got = []
    src = SCKCaptureSource(binary, 48000, on_chunk=lambda c: got.extend(c.tolist()))
    assert not src.is_alive()  # not started yet
    src.start()
    deadline = time.time() + 5
    while not got and time.time() < deadline:
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
    src = SCKCaptureSource(binary, 48000, on_chunk=lambda c: None, on_stderr=lambda l: None)
    src.start()
    deadline = time.time() + 5
    while src.is_alive() and time.time() < deadline:
        time.sleep(0.01)
    src.stop()
    assert "RESULT=ERROR" in src.stderr_tail()


def test_source_stop_terminates_cleanly(tmp_path):
    binary = _write_fake(tmp_path, "fake_stream.py", _FAKE_STREAM)
    got = []
    src = SCKCaptureSource(binary, 48000, on_chunk=lambda c: got.extend(c.tolist()))
    src.start()
    # Wait until the helper is actually streaming before stopping. Polling on
    # real output (not a fixed sleep) removes the flake where the Python fake
    # subprocess is still cold-starting at a fixed deadline — and guarantees its
    # SIGTERM handler (installed before its write loop) is in place, so the
    # returncode assertion is deterministic too.
    deadline = time.time() + 5
    while not got and time.time() < deadline:
        time.sleep(0.01)
    src.stop()
    assert len(got) > 0                 # received streaming audio
    assert src._proc.returncode == 0    # SIGTERM handled gracefully


# A helper built from the branch that emits the tagged-frame protocol — a stale
# native/sck_capture surviving a checkout. Payloads are silent, the worst case:
# nothing about the sample values looks wrong.
_FAKE_FRAMED = (
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
    "sys.exit(0)\n"
)


def test_source_rejects_a_helper_speaking_the_framed_protocol(tmp_path):
    binary = _write_fake(tmp_path, "fake_framed.py", _FAKE_FRAMED)
    got = []
    src = SCKCaptureSource(binary, 48000, on_chunk=lambda c: got.extend(c.tolist()))
    src.start()
    err = src.wait_for_verdict(timeout=5)
    src.stop()
    assert err is not None
    assert "tagged-frame" in err


def test_source_stops_delivering_once_the_format_is_rejected(tmp_path):
    # Refusing to record is the point: no garbage may reach the WAV writer.
    binary = _write_fake(tmp_path, "fake_framed.py", _FAKE_FRAMED)
    got = []
    src = SCKCaptureSource(binary, 48000, on_chunk=lambda c: got.extend(c.tolist()))
    src.start()
    assert src.wait_for_verdict(timeout=5) is not None
    settled = len(got)
    time.sleep(0.2)
    src.stop()
    assert len(got) == settled          # delivery stopped at the verdict


def test_source_accepts_a_helper_speaking_raw_float32(tmp_path):
    binary = _write_fake(tmp_path, "fake_stream.py", _FAKE_STREAM)
    got = []
    src = SCKCaptureSource(binary, 48000, on_chunk=lambda c: got.extend(c.tolist()))
    src.start()
    err = src.wait_for_verdict(timeout=5)
    src.stop()
    assert err is None
    assert len(got) > 0                 # and audio still flows
