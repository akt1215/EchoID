import collections
import os
import signal
import struct
import subprocess
import threading
import time

import numpy as np

_CHANNELS = 2
_BYTES_PER_SAMPLE = 4                       # float32
_FRAME_BYTES = _CHANNELS * _BYTES_PER_SAMPLE  # 8 bytes per interleaved stereo frame
_READ_BLOCK = 65536

# Wire-format guard. Enough opening bytes to contain two maximum-size tagged
# frames (~170ms of real audio), so the check below can confirm a mismatch
# rather than infer one from a single header.
_SNIFF_BYTES = 65536
_FRAMED_HEADER = struct.Struct("<BI")       # [1B frame type][uint32 LE sample count]
_FRAMED_TYPES = (0, 1)                      # 0 = mic, 1 = system
_MAX_FRAMED_COUNT = 4096                    # plausible per-frame sample count
_MAX_PLAUSIBLE_SAMPLE = 8.0                 # float PCM sits in [-1, 1]; allow headroom
_MAX_IMPLAUSIBLE_FRACTION = 0.01


def _looks_framed(opening):
    """True when `opening` begins with the tagged-frame header
    ([1B type][uint32 LE sample count]) emitted by the single-clock mic+system
    helper build. Confirmed by walking to a second well-formed header, so a
    chance byte pattern in real audio cannot trip it.
    """
    offset = 0
    for _ in range(2):
        if len(opening) < offset + _FRAMED_HEADER.size:
            return False
        ftype, count = _FRAMED_HEADER.unpack_from(opening, offset)
        if ftype not in _FRAMED_TYPES or not 0 < count <= _MAX_FRAMED_COUNT:
            return False
        offset += _FRAMED_HEADER.size + count * _BYTES_PER_SAMPLE
    return True


def wire_format_error(opening):
    """Return a diagnostic if `opening` is not the raw interleaved float32 stream
    SCKDecoder reads, else None.

    native/sck_capture is a gitignored build artifact, so a helper built on a
    branch with a different wire format survives a checkout while this decoder
    changes underneath it. The mismatch is silent — bytes flow at full rate, so
    every liveness check passes — but the tagged frame header is 5 bytes, not a
    multiple of 4, so parsing it as audio permanently destroys float32
    alignment: exponent bytes land in mantissa positions, and the samples reach
    the WAV as full-scale noise. Inspecting the bytes is what catches it.

    Note the two checks are not redundant. A call that opens quiet carries
    all-zero payloads, which survive a misaligned parse looking perfectly sane,
    so only the frame headers give it away.
    """
    if _looks_framed(opening):
        return ("the helper is emitting the tagged-frame protocol "
                "([1B type][uint32 count][float32 payload]), not the raw "
                "interleaved float32 stream this build decodes")

    usable = (len(opening) // _BYTES_PER_SAMPLE) * _BYTES_PER_SAMPLE
    if usable == 0:
        return None
    samples = np.frombuffer(opening[:usable], dtype="<f4")
    implausible = ~np.isfinite(samples) | (np.abs(samples) > _MAX_PLAUSIBLE_SAMPLE)
    fraction = float(implausible.mean())
    if fraction > _MAX_IMPLAUSIBLE_FRACTION:
        return (f"{fraction:.0%} of the opening samples are NaN or beyond "
                f"±{_MAX_PLAUSIBLE_SAMPLE:g}, so the byte stream is not the "
                f"raw interleaved float32 this build decodes")
    return None


class SCKDecoder:
    """Decodes a stream of interleaved little-endian float32 stereo PCM bytes
    (as emitted by native/sck_capture) into mono float32 frames.

    PCM arrives in arbitrary byte-sized chunks off a pipe, so a chunk can end
    mid-frame; the trailing partial frame is carried into the next feed().
    """

    def __init__(self):
        self._carry = b""

    def feed(self, raw):
        buf = self._carry + raw
        n_full = (len(buf) // _FRAME_BYTES) * _FRAME_BYTES
        self._carry = buf[n_full:]
        if n_full == 0:
            return np.empty(0, dtype=np.float32)
        stereo = np.frombuffer(buf[:n_full], dtype="<f4").reshape(-1, _CHANNELS)
        return stereo.mean(axis=1).astype(np.float32)


class SCKCaptureSource:
    """Owns the native/sck_capture subprocess and turns its stdout PCM stream
    into mono float32 chunks delivered to `on_chunk`. Diagnostics from the
    helper's stderr go to `on_stderr` (console by default), so a Screen
    Recording permission problem surfaces within seconds of a live meeting.
    """

    def __init__(self, binary_path, samplerate, on_chunk, on_stderr=None):
        self._binary_path = binary_path
        self._samplerate = samplerate
        self._on_chunk = on_chunk
        self._on_stderr = on_stderr or (lambda line: print(f"[sck] {line}", end=""))
        self._decoder = SCKDecoder()
        self._proc = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._stderr_tail = collections.deque(maxlen=20)
        self._opening = b""
        self._verdict_reached = False
        self._wire_format_error = None

    def wait_for_verdict(self, timeout):
        """Block until the opening bytes have been checked against the wire
        format this build decodes (or the helper exits, or `timeout` elapses).
        Returns the mismatch diagnostic, or None if the stream looks right."""
        deadline = time.monotonic() + timeout
        while (not self._verdict_reached and self.is_alive()
               and time.monotonic() < deadline):
            time.sleep(0.01)
        return self._wire_format_error

    def _settle(self):
        """Reach a verdict on the buffered opening bytes, once."""
        if self._verdict_reached:
            return
        self._wire_format_error = wire_format_error(self._opening)
        self._verdict_reached = True

    def _take_opening(self):
        """Hand back the buffered opening bytes for decoding, once judged."""
        opening, self._opening = self._opening, b""
        return opening

    def _deliver(self, raw):
        if not raw or self._wire_format_error:
            return
        mono = self._decoder.feed(raw)
        if mono.size:
            self._on_chunk(mono)

    def is_alive(self):
        """True while the helper subprocess is running (delivering audio)."""
        return self._proc is not None and self._proc.poll() is None

    def stderr_tail(self):
        """The last few stderr lines from the helper, for diagnosing a failed
        or silent startup (e.g. a Screen Recording permission denial)."""
        return "".join(self._stderr_tail).strip()

    def start(self):
        if not os.path.exists(self._binary_path):
            raise FileNotFoundError(
                f"SCK helper not found at {self._binary_path}. Run ./setup.sh to build it."
            )
        self._proc = subprocess.Popen(
            [self._binary_path, str(self._samplerate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self):
        stdout = self._proc.stdout
        while True:
            raw = stdout.read1(_READ_BLOCK)  # returns as soon as any data is available
            if not raw:
                self._settle()  # EOF: helper exited; judge whatever arrived
                self._deliver(self._take_opening())
                break
            if not self._verdict_reached:
                # Buffer the opening rather than decoding it: a mismatched wire
                # format decodes to NaN, and none of it may reach the caller.
                self._opening += raw
                if len(self._opening) < _SNIFF_BYTES:
                    continue
                self._settle()
                if self._wire_format_error:
                    break  # refuse to feed a mismatched stream downstream
                raw = self._take_opening()
            self._deliver(raw)

    def _read_stderr(self):
        for line in self._proc.stderr:
            decoded = line.decode("utf-8", "replace")
            self._stderr_tail.append(decoded)
            self._on_stderr(decoded)

    def stop(self):
        if self._proc is None:
            return
        self._proc.send_signal(signal.SIGTERM)
        if self._stdout_thread:
            self._stdout_thread.join(timeout=5)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._stderr_thread:
            self._stderr_thread.join(timeout=2)
