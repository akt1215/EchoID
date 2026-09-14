import numpy as np

DEFAULT_STALL_TIMEOUT = 10.0


class ChannelPairer:
    """Turns two independently-arriving mono streams (mic = ch0, remote = ch1)
    into paired stereo frames.

    - Normal: emit min(len(mic), len(bh)) exactly-aligned paired frames.
    - Stall: if one side produces nothing for `stall_timeout` seconds while the
      other keeps flowing, emit the live side paired with silence, so a dead
      producer (crashed SCK helper, unplugged mic, denied Screen Recording
      permission) can neither freeze the whole recording nor grow memory without
      bound. The stalled side is latched in `stalled_side` for the caller to
      surface. The timeout is generous so a momentary lead never trips it.
    - Final flush: on stop, emit every remaining buffered sample, padding the
      shorter side with silence so no captured tail is dropped.

    Pure logic (no I/O, no threads); the wall-clock is passed in as `now` so the
    behavior is deterministic and unit-testable.
    """

    def __init__(self, stall_timeout=DEFAULT_STALL_TIMEOUT):
        self.stall_timeout = stall_timeout
        self._mic = np.empty(0, dtype=np.float64)
        self._bh = np.empty(0, dtype=np.float64)
        self._started_at = None
        self._last_mic = None
        self._last_bh = None
        self.stalled_side = None

    def add(self, mic_samples, bh_samples, now):
        if self._started_at is None:
            self._started_at = now
        if len(mic_samples):
            self._mic = np.concatenate([self._mic, np.asarray(mic_samples, dtype=np.float64)])
            self._last_mic = now
        if len(bh_samples):
            self._bh = np.concatenate([self._bh, np.asarray(bh_samples, dtype=np.float64)])
            self._last_bh = now

    def _silent_for(self, last_seen, now):
        base = last_seen if last_seen is not None else self._started_at
        return 0.0 if base is None else now - base

    def take(self, now, final=False):
        """Return stereo frames (N, 2) ready to write, dropping them from the
        internal buffers. Call repeatedly with final=True at stop until it
        returns an empty array."""
        min_len = min(len(self._mic), len(self._bh))
        if min_len:
            out = np.column_stack((self._mic[:min_len], self._bh[:min_len]))
            self._mic = self._mic[min_len:]
            self._bh = self._bh[min_len:]
            return out

        if final:
            return self._flush_all()

        # One side is empty. Only pad with silence once the empty side has been
        # dark long enough to be a dead producer rather than a momentary lead.
        if len(self._mic) and self._silent_for(self._last_bh, now) >= self.stall_timeout:
            self.stalled_side = "bh"
            out = np.column_stack((self._mic, np.zeros(len(self._mic))))
            self._mic = np.empty(0, dtype=np.float64)
            return out
        if len(self._bh) and self._silent_for(self._last_mic, now) >= self.stall_timeout:
            self.stalled_side = "mic"
            out = np.column_stack((np.zeros(len(self._bh)), self._bh))
            self._bh = np.empty(0, dtype=np.float64)
            return out
        return np.empty((0, 2), dtype=np.float64)

    def _flush_all(self):
        n = max(len(self._mic), len(self._bh))
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)
        mic = np.zeros(n, dtype=np.float64)
        mic[:len(self._mic)] = self._mic
        bh = np.zeros(n, dtype=np.float64)
        bh[:len(self._bh)] = self._bh
        self._mic = np.empty(0, dtype=np.float64)
        self._bh = np.empty(0, dtype=np.float64)
        return np.column_stack((mic, bh))
