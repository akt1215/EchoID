import queue
import sounddevice as sd
import numpy as np

class AudioMonitor:
    """Relays BlackHole audio to the current default output device.

    Uses a thread-safe queue (not a custom ring buffer) since both
    streams now use matching blocksize=512. Minimal buffering keeps
    latency low (~23ms typical, max ~93ms on clock drift).

    Receives chunks via `feed()` (called from the recorder's BlackHole
    callback) rather than opening its own BlackHole InputStream.
    This avoids CoreAudio conflicts from two concurrent streams on the
    same device.
    """
    def __init__(self, samplerate=44100):
        self.samplerate = samplerate
        self.running = False
        self._queue = queue.Queue(maxsize=8)

    def feed(self, chunk):
        if not self.running:
            return
        data = np.asarray(chunk, dtype=np.float32).ravel()
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            # Drop oldest block to keep latency bounded
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(data)
            except queue.Empty:
                pass

    def _output_callback(self, outdata, frames, time, status):
        if status:
            print(f"Monitor output status: {status}")
        try:
            data = self._queue.get_nowait()
            n = min(len(data), frames)
            outdata[:n] = data[:n].reshape(-1, 1)
            if n < frames:
                outdata[n:] = 0
        except queue.Empty:
            outdata.fill(0)

    def start(self):
        if self.running:
            return
        self.running = True
        self._out_stream = sd.OutputStream(
            samplerate=self.samplerate,
            device=None,
            channels=1,
            callback=self._output_callback,
            blocksize=512,
        )
        self._out_stream.start()

    def stop(self):
        self.running = False
        if hasattr(self, '_out_stream'):
            self._out_stream.stop()
            self._out_stream.close()
