"""Play a short audio clip from a recording.

Uses macOS `afplay` rather than sounddevice: PortAudio/AUHAL rejects a playback
sample rate the default output device does not natively run (error -10851), which
is common right after a recording session leaves the audio stack in an odd state.
afplay resamples through the system output and runs in its own process, immune to
both issues. Shared by the speaker review screen and the speaker manager, so the
subprocess and temp-file lifecycle live in exactly one place.
"""

import os
import subprocess
import tempfile
import threading

import numpy as np
import soundfile as sf


class ClipPlayer:
    MAX_SECONDS = 10

    def __init__(self):
        self._proc = None
        self._tmp = None
        self._lock = threading.Lock()

    def play(self, wav, start, end):
        """Play up to MAX_SECONDS of `wav` from `start`. Stops whatever was
        playing first, so clips never overlap. A missing file is a no-op."""
        if not wav or not os.path.exists(wav):
            return
        sr = sf.info(wav).samplerate
        begin = int(start * sr)
        stop = min(int(end * sr), begin + self.MAX_SECONDS * sr)
        data, _ = sf.read(wav, start=begin, stop=stop)
        self._render(data, sr)

    def play_spans(self, wav, spans, gap=0.25):
        """Play several spans of `wav` as one clip, separated by short silences.

        A speaker whose turns are all under a second cannot be judged from any
        single turn; stitching the audible ones gives enough voice to recognize
        while skipping the silence between them.
        """
        if not wav or not os.path.exists(wav) or not spans:
            return
        sr = sf.info(wav).samplerate
        budget = self.MAX_SECONDS * sr
        silence, pieces = None, []
        for start, end in spans:
            if budget <= 0:
                break
            chunk, _ = sf.read(wav, start=int(start * sr),
                               stop=int(end * sr), always_2d=True)
            if not len(chunk):
                continue
            chunk = chunk[:budget]
            budget -= len(chunk)
            if pieces:
                if silence is None:
                    silence = np.zeros((int(gap * sr), chunk.shape[1]),
                                       dtype=chunk.dtype)
                pieces.append(silence)
            pieces.append(chunk)
        if pieces:
            self._render(np.concatenate(pieces), sr)

    def _render(self, data, sr):
        """Reduce to one channel, write a temp WAV, and hand it to afplay."""
        # Write the louder channel so the speaker is audible whether they were on
        # the mic (ch0) or the system-audio channel (ch1).
        if data.ndim == 2 and data.shape[1] >= 2:
            ch = 0 if np.mean(data[:, 0] ** 2) >= np.mean(data[:, 1] ** 2) else 1
            data = data[:, ch]
        elif data.ndim == 2:
            data = data[:, 0]

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, data, sr)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)   # don't leak the temp file if encoding fails
            raise

        with self._lock:
            self._stop_locked()
            self._tmp = tmp
            self._proc = subprocess.Popen(
                ["afplay", tmp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def _stop_locked(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None
        if self._tmp and os.path.exists(self._tmp):
            try:
                os.remove(self._tmp)
            except Exception:
                pass
        self._tmp = None

    def wait(self, timeout=None):
        """Block until the clip now playing finishes, or `timeout` elapses.

        Judging whether a cluster holds one voice means hearing several of its
        turns in a row; without this the next `play` terminates the previous clip
        mid-word and every turn sounds like a cut. The process handle is taken
        under the lock but waited on outside it, so a concurrent `stop` can still
        interrupt playback instead of deadlocking behind it.
        """
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def stop(self):
        """Stop any clip and clean up its temp file."""
        with self._lock:
            self._stop_locked()
