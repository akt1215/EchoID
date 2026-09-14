import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
import time
import os
from .zoom_monitor import ZoomMonitor
from .audio_monitor import AudioMonitor
from .sck_capture import SCKCaptureSource
from .channel_pairer import ChannelPairer
from .ingest import MARKER as _MARKER


class CaptureStartupError(RuntimeError):
    """Raised when a capture backend cannot start (e.g. the SCK helper exits
    immediately because Screen Recording permission is denied), so the caller
    fails loudly instead of recording an empty file for the whole meeting."""


class AudioRecorder:
    # Stamped into every recording this class writes, and the only thing that
    # entitles a WAV to be read as ch0=mic/ch1=remote. Channel count is not
    # provenance -- QuickTime, OBS and Audio Hijack all produce stereo, and
    # trusting one of those silently attributes remote speech to the local user.
    # Sourced from core.ingest -- the module that consumes it -- rather than
    # restated here, so there is exactly one string literal.
    MARKER = _MARKER

    def __init__(self, mic_device, blackhole_device, output_filename,
                 samplerate=44100, monitor_enabled=True,
                 capture_backend="blackhole", sck_binary_path=None):
        if capture_backend not in ("sck", "blackhole"):
            raise ValueError(
                f"Unknown capture_backend: {capture_backend!r} (expected 'sck' or 'blackhole')"
            )
        if capture_backend == "sck" and not sck_binary_path:
            raise ValueError("capture_backend='sck' requires sck_binary_path")
        self.mic_device = mic_device
        self.blackhole_device = blackhole_device
        self.samplerate = samplerate
        self.output_filename = output_filename
        self.capture_backend = capture_backend
        self.sck_binary_path = sck_binary_path

        self.mic_queue = queue.Queue()
        self.bh_queue = queue.Queue()
        self.recording = False

        # Set during start()/recording so callers can surface capture problems
        # instead of discovering an empty or frozen recording after the meeting.
        self.ch1_silent = False       # remote channel delivered nothing at start
        self.stalled_side = None      # a producer died mid-recording ('mic'/'bh')
        self._writer_error = None     # exception that killed the writer thread
        self._pairer = ChannelPairer()

        self.zoom_monitor = ZoomMonitor()
        # AudioMonitor only makes sense for the BlackHole reroute. SCK taps audio
        # that already plays out the speakers, so a monitor would double it.
        self.audio_monitor = (
            AudioMonitor(samplerate)
            if (monitor_enabled and capture_backend == "blackhole") else None
        )
        self._sck_source = None

    def _mic_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block of the mic."""
        if status:
            print(f"Mic status: {status}")

        # If Zoom is muted, zero out the mic data
        if self.zoom_monitor.is_muted:
            indata = np.zeros_like(indata)

        self.mic_queue.put(indata.copy())

    def _bh_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block of BlackHole."""
        if status:
            print(f"BlackHole status: {status}")
        chunk = indata.copy()
        self.bh_queue.put(chunk)
        if self.audio_monitor:
            self.audio_monitor.feed(chunk)

    @staticmethod
    def _drain_all(q):
        """Pull and flatten everything currently queued, in order."""
        chunks = []
        try:
            while True:
                chunks.append(q.get_nowait().flatten())
        except queue.Empty:
            pass
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)

    def _file_writer_thread(self):
        """Continuously pair the two channels and write them to a stereo WAV.

        Delegates alignment to ChannelPairer, which silence-pads a stalled
        producer so a dead mic or SCK helper can't freeze the recording or grow
        memory without bound. Any failure is recorded on the instance (surfaced
        by stop()) rather than vanishing into the thread's excepthook.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_filename)), exist_ok=True)
            with sf.SoundFile(self.output_filename, mode='w', samplerate=self.samplerate,
                              channels=2, subtype='PCM_16') as file:
                file.comment = self.MARKER
                while self.recording:
                    self._pairer.add(self._drain_all(self.mic_queue),
                                     self._drain_all(self.bh_queue),
                                     now=time.monotonic())
                    out = self._pairer.take(now=time.monotonic())
                    if len(out):
                        file.write(out)
                    if self._pairer.stalled_side and self.stalled_side is None:
                        self.stalled_side = self._pairer.stalled_side
                        print(f"[recorder] WARNING: the {self.stalled_side} channel "
                              f"stopped producing audio — recording the live channel "
                              f"with silence on the dead one.")
                    time.sleep(0.05)

                # Final flush: producers are stopped by now, so drain whatever
                # is left and write it (padding the shorter side) — no lost tail.
                self._pairer.add(self._drain_all(self.mic_queue),
                                 self._drain_all(self.bh_queue),
                                 now=time.monotonic())
                while True:
                    out = self._pairer.take(now=time.monotonic(), final=True)
                    if len(out) == 0:
                        break
                    file.write(out)
        except Exception as e:
            self._writer_error = e
            print(f"[recorder] ERROR: the recording writer failed ({e}); "
                  f"audio after this point was not saved.")

    @staticmethod
    def _drain_queue(q):
        """Discard everything currently buffered in a queue."""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _wait_for_first_chunk(self, q, timeout):
        """Block until q has data (the producer is actually delivering) or timeout."""
        deadline = time.monotonic() + timeout
        while q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

    def start(self):
        # Fail fast on a missing SCK helper BEFORE starting anything (so nothing
        # leaks) and as a CaptureStartupError, so callers handle it the same
        # graceful way as a denied-permission startup failure.
        if self.capture_backend == "sck" and not os.path.exists(self.sck_binary_path or ""):
            raise CaptureStartupError(
                f"SCK helper not found at {self.sck_binary_path!r}. Run ./setup.sh to build it, "
                f"or set audio.capture_backend: blackhole in config.yaml."
            )

        print("Starting Zoom Monitor...")
        self.zoom_monitor.start()

        print(f"Starting Audio Streams (backend={self.capture_backend})...")
        self.recording = True

        # Construct the mic (ch0) stream identically on both backends; started below.
        self.mic_stream = sd.InputStream(samplerate=self.samplerate, device=self.mic_device,
                                         channels=1, callback=self._mic_callback,
                                         blocksize=512)

        if self.capture_backend == "sck":
            # The SCK helper cold-starts (fork/exec + SCShareableContent + startCapture,
            # up to ~1s). Start it first and wait until it actually delivers, THEN start
            # the mic and drop both queues' backlog, so ch0 and ch1 begin from a common
            # instant — the writer pairs by buffered length, so a startup lead on either
            # side would bake in a permanent inter-channel skew.
            self._sck_source = SCKCaptureSource(
                self.sck_binary_path, self.samplerate, on_chunk=self.bh_queue.put,
            )
            self._sck_source.start()
            self._wait_for_first_chunk(self.bh_queue, timeout=5.0)
            # Distinguish "helper died" (hard failure — abort so we don't record
            # an empty file all meeting) from "helper alive but delivered nothing"
            # (probable Screen Recording permission denial — warn loudly but keep
            # the mic, which is still worth capturing).
            if not self._sck_source.is_alive():
                tail = self._sck_source.stderr_tail()
                self._sck_source = None
                self.recording = False
                self.zoom_monitor.stop()
                self.mic_stream.close()  # created above but never started
                raise CaptureStartupError(
                    "The ScreenCaptureKit helper exited on startup, so system "
                    "audio (ch1) cannot be captured. This is usually a denied "
                    "Screen Recording permission — grant it to this terminal in "
                    "System Settings > Privacy & Security > Screen Recording, then "
                    "quit and reopen the terminal.\n"
                    f"Helper said: {tail or '(no output)'}"
                )
            # A helper whose wire format this build cannot decode streams bytes
            # briskly, so every check above passes while what reaches ch1 is
            # full-scale noise. Liveness cannot see that; only the bytes can.
            wire_error = self._sck_source.wait_for_verdict(timeout=2.0)
            if wire_error:
                self._sck_source.stop()
                self._sck_source = None
                self.recording = False
                self.zoom_monitor.stop()
                self.mic_stream.close()  # created above but never started
                raise CaptureStartupError(
                    "The ScreenCaptureKit helper's wire format does not match "
                    f"this build: {wire_error}.\n"
                    "native/sck_capture is a build artifact that git does not "
                    "track, so a helper built on another branch survives a "
                    "checkout and records ch1 as noise. Rebuild it:\n"
                    "  swiftc -parse-as-library -O native/sck_capture.swift "
                    "-o native/sck_capture\n"
                    "(or re-run ./setup.sh)."
                )
            if self.bh_queue.empty():
                self.ch1_silent = True
                print("[recorder] WARNING: no system audio (ch1) after 5s — the "
                      "remote channel may be silent because Screen Recording "
                      "permission is denied. Recording the mic anyway; check the "
                      "permission if the meeting note is missing remote speech.")
            self.mic_stream.start()
            self._drain_queue(self.bh_queue)
            self._drain_queue(self.mic_queue)
        else:
            self.mic_stream.start()
            # Start ch1 back-to-back with ch0 (nothing slow in between): the writer
            # pairs channels by buffered length, so a gap here bakes permanent skew in.
            self.bh_stream = sd.InputStream(samplerate=self.samplerate,
                                            device=self.blackhole_device,
                                            channels=1, callback=self._bh_callback,
                                            blocksize=512)
            self.bh_stream.start()
            # Monitor only relays BlackHole → output; open it after the input streams.
            if self.audio_monitor:
                print("Starting Audio Monitor (BlackHole → default output)...")
                self.audio_monitor.start()

        # Start the writer only after producers are live (and, on sck, synced) so
        # channel pairing begins from a common instant.
        self.writer_thread = threading.Thread(target=self._file_writer_thread, daemon=True)
        self.writer_thread.start()

        print(f"Recording to {self.output_filename}...")

    def stop(self):
        print("Stopping recording...")

        # Stop the producers FIRST so no more audio is queued, THEN flip the flag
        # and let the writer drain and flush its residual buffers — otherwise the
        # last ~50ms queued after the flag flips is dropped.
        if hasattr(self, 'mic_stream'):
            self.mic_stream.stop()
            self.mic_stream.close()

        if self.capture_backend == "sck":
            if self._sck_source:
                self._sck_source.stop()
        elif hasattr(self, 'bh_stream'):
            self.bh_stream.stop()
            self.bh_stream.close()

        self.recording = False
        if hasattr(self, 'writer_thread'):
            self.writer_thread.join()

        if self.audio_monitor:
            self.audio_monitor.stop()
        self.zoom_monitor.stop()

        if self._writer_error is not None:
            print(f"[recorder] WARNING: recording ended on a write error "
                  f"({self._writer_error}); the WAV may be truncated.")
        print("Recording stopped.")

def list_devices():
    """Helper to list audio devices so the user can find the right indices."""
    print(sd.query_devices())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record dual-channel audio for Zoom Notetaker.")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--mic", type=int, help="Device ID for Physical Microphone.")
    parser.add_argument("--bh", type=int, help="Device ID for BlackHole 2ch.")
    parser.add_argument("--out", type=str, default="meeting_recordings/raw_audio.wav", help="Output file path.")

    args = parser.parse_args()

    if args.list:
        list_devices()
        exit(0)

    if args.mic is None or args.bh is None:
        print("Error: Must provide --mic and --bh device IDs. Use --list to see available devices.")
        exit(1)

    recorder = AudioRecorder(args.mic, args.bh, args.out)
    recorder.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
