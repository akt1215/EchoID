import threading
import time

import numpy as np
import soundfile as sf

from core import ingest
from core.audio_recorder import AudioRecorder
from core.channel_pairer import ChannelPairer


def _make_recorder(out):
    # sck backend so no AudioMonitor is created; streams are never started in
    # these tests — we feed the queues directly and drive the writer thread.
    return AudioRecorder(0, None, out, samplerate=48000, monitor_enabled=False,
                         capture_backend="sck", sck_binary_path="/does/not/matter")


def _run_writer(rec, seconds):
    rec.recording = True
    t = threading.Thread(target=rec._file_writer_thread)
    t.start()
    time.sleep(seconds)
    rec.recording = False
    t.join(timeout=5)
    assert not t.is_alive()


def test_writer_pairs_both_channels(tmp_path):
    out = str(tmp_path / "m.wav")
    rec = _make_recorder(out)
    # PCM_16 WAV: sample values must live in [-1, 1].
    rec.mic_queue.put(np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32))
    rec.bh_queue.put(np.array([[-0.1], [-0.2], [-0.3], [-0.4]], dtype=np.float32))
    _run_writer(rec, 0.25)
    data, sr = sf.read(out)
    assert sr == 48000
    assert np.allclose(data[:, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-3)
    assert np.allclose(data[:, 1], [-0.1, -0.2, -0.3, -0.4], atol=1e-3)


def test_writer_pads_silence_when_one_channel_never_arrives(tmp_path):
    # ch1 (SCK) is dead: only the mic produces. The old writer paired by
    # min length, so it wrote nothing forever; the pairer must pad ch1 silent.
    out = str(tmp_path / "m.wav")
    rec = _make_recorder(out)
    rec._pairer = ChannelPairer(stall_timeout=0.05)
    rec.mic_queue.put(np.full((10, 1), 0.5, dtype=np.float32))
    _run_writer(rec, 0.3)
    data, _ = sf.read(out)
    assert len(data) == 10
    assert np.allclose(data[:, 0], 0.5)   # mic audio preserved
    assert np.allclose(data[:, 1], 0.0)   # dead ch1 written as silence
    assert rec.stalled_side == "bh"


def test_writer_output_is_recognized_as_native(tmp_path):
    """The entire trust chain rests on `file.comment = self.MARKER` actually
    executing before the first write, inside the real writer thread -- not on
    a hand-built lookalike (see tests/test_ingest_provenance.py, which only
    proves is_native reads the marker correctly, not that the recorder writes
    it). Drive the real thread and check ingest accepts its real output."""
    out = str(tmp_path / "meeting_2026-01-02_03-04-05.wav")
    rec = _make_recorder(out)
    rec.mic_queue.put(np.array([[0.1], [0.2]], dtype=np.float32))
    rec.bh_queue.put(np.array([[-0.1], [-0.2]], dtype=np.float32))
    _run_writer(rec, 0.25)
    assert ingest.is_native(out) is True


def test_writer_error_is_captured_not_swallowed(tmp_path):
    # Make the output path unopenable (parent is a regular file) so SoundFile
    # raises inside the thread; the recorder must record the error, not lose it.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    rec = _make_recorder(str(blocker / "nested" / "m.wav"))
    _run_writer(rec, 0.15)
    assert rec._writer_error is not None
