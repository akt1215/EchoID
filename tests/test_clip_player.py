"""Clip playback is shared by the review screen and the speaker manager."""

import numpy as np
import soundfile as sf

from core.clip_player import ClipPlayer


def test_play_is_a_noop_for_a_missing_file(tmp_path):
    p = ClipPlayer()
    p.play(str(tmp_path / "nope.wav"), 0.0, 1.0)   # must not raise
    p.stop()


def test_play_caps_the_clip_and_picks_the_louder_channel(tmp_path, monkeypatch):
    wav = tmp_path / "m.wav"
    sr = 8000
    quiet = np.zeros(sr * 30, dtype="float32")
    loud = np.ones(sr * 30, dtype="float32") * 0.5
    sf.write(str(wav), np.stack([quiet, loud], axis=1), sr)

    written = {}
    real_write = sf.write

    def spy(path, data, samplerate, *a, **k):
        written["frames"] = len(data)
        written["mono"] = data.ndim == 1
        written["rms"] = float(np.sqrt(np.mean(np.square(data))))
        return real_write(path, data, samplerate, *a, **k)

    monkeypatch.setattr(sf, "write", spy)
    monkeypatch.setattr("core.clip_player.subprocess.Popen", lambda *a, **k: None)

    p = ClipPlayer()
    p.play(str(wav), 0.0, 30.0)          # longer than the cap
    assert written["mono"] is True
    assert written["frames"] == ClipPlayer.MAX_SECONDS * sr   # capped
    assert written["rms"] > 0.1                               # took the loud channel


def test_wait_blocks_on_the_running_clip(tmp_path, monkeypatch):
    """Reviewing a cluster plays turns back to back; without waiting, the next
    play would terminate the previous clip mid-word."""
    wav = tmp_path / "m.wav"
    sr = 8000
    sf.write(str(wav), np.zeros(sr, dtype="float32"), sr)

    waited = []

    class FakeProc:
        def poll(self):
            return None

        def wait(self, timeout=None):
            waited.append(timeout)

        def terminate(self):
            pass

    monkeypatch.setattr("core.clip_player.subprocess.Popen",
                        lambda *a, **k: FakeProc())
    p = ClipPlayer()
    p.play(str(wav), 0.0, 0.5)
    p.wait()
    assert waited == [None]
    p.stop()


def test_wait_is_a_noop_when_nothing_is_playing():
    ClipPlayer().wait()          # must not raise
