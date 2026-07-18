import collections
import os
import signal
import struct
import subprocess
import threading

import numpy as np

TYPE_MIC = 0
TYPE_SYSTEM = 1
HEADER = struct.Struct("<BI")          # [1B frame type][uint32 LE sample count]
_READ_BLOCK = 65536


class FramedDecoder:
    """Decodes the SCK helper's tagged wire protocol into mono float32 frames
    routed by source. Frame = [1B type][uint32 LE N][N x float32 LE]; type 0=mic,
    1=system; every frame is mono at config.sampleRate (the helper guarantees it).

    Bytes arrive in arbitrary pipe-sized chunks, so a frame's header or payload
    can straddle feed() calls; the partial tail is carried into the next feed().
    A frame is emitted only once its whole payload has arrived (frame-atomic).
    """

    def __init__(self, on_mic, on_system):
        self._on_mic = on_mic
        self._on_system = on_system
        self._carry = b""

    def feed(self, raw):
        buf = self._carry + raw
        n = len(buf)
        pos = 0
        while n - pos >= HEADER.size:
            ftype, count = HEADER.unpack_from(buf, pos)
            end = pos + HEADER.size + count * 4
            if n < end:
                break                                   # payload still arriving
            samples = np.frombuffer(buf, dtype="<f4",
                                    count=count, offset=pos + HEADER.size).copy()
            if ftype == TYPE_MIC:
                self._on_mic(samples)
            elif ftype == TYPE_SYSTEM:
                self._on_system(samples)
            else:
                self._carry = b""
                raise ValueError(f"SCK framed protocol: unknown frame type {ftype}")
            pos = end
        self._carry = buf[pos:]


def sck_mode_provides_mic(mode):
    """True when the SCK helper is capturing BOTH mic and system on one clock, so
    AudioRecorder must NOT open a second sounddevice mic."""
    return mode == "mic+system"


class SCKCaptureSource:
    """Owns the native/sck_capture subprocess and decodes its framed stdout into
    per-source mono chunks: mic -> on_mic_chunk, system -> on_system_chunk. The
    helper announces its mode ('mic+system' on macOS 15+) and the opened mic name
    over stderr (RESULT=MODE / RESULT=MIC); wait_for_mode() blocks until the mode
    line arrives so the caller can confirm SCK is delivering the mic.
    """

    def __init__(self, binary_path, samplerate, on_mic_chunk, on_system_chunk,
                 mic_device_id="", on_stderr=None):
        self._binary_path = binary_path
        self._samplerate = samplerate
        self._mic_device_id = mic_device_id
        self._on_stderr = on_stderr or (lambda line: print(f"[sck] {line}", end=""))
        self._decoder = FramedDecoder(on_mic=on_mic_chunk, on_system=on_system_chunk)
        self._proc = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._stderr_tail = collections.deque(maxlen=20)
        self._mode = None
        self._mode_event = threading.Event()
        self.mic_name = None

    def is_alive(self):
        """True while the helper subprocess is running (delivering audio)."""
        return self._proc is not None and self._proc.poll() is None

    def stderr_tail(self):
        """The last few stderr lines from the helper, for diagnosing a failed
        or silent startup (e.g. a Screen Recording permission denial)."""
        return "".join(self._stderr_tail).strip()

    def wait_for_mode(self, timeout):
        """Block until the helper announces its mode, or timeout. Returns the mode
        string ('mic+system' on success) or None if it never arrived."""
        self._mode_event.wait(timeout)
        return self._mode

    def start(self):
        if not os.path.exists(self._binary_path):
            raise FileNotFoundError(
                f"SCK helper not found at {self._binary_path}. Run ./setup.sh to build it."
            )
        self._proc = subprocess.Popen(
            [self._binary_path, str(self._samplerate), self._mic_device_id],
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
                break  # EOF: helper exited
            self._decoder.feed(raw)

    def _read_stderr(self):
        for line in self._proc.stderr:
            decoded = line.decode("utf-8", "replace")
            self._stderr_tail.append(decoded)
            s = decoded.strip()
            if s.startswith("RESULT=MODE "):
                self._mode = s[len("RESULT=MODE "):].strip()
                self._mode_event.set()
            elif s.startswith("RESULT=MIC "):
                self.mic_name = s[len("RESULT=MIC "):].strip()
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
