"""Embeddings must come from the channel the speaker is actually on.

Recordings are stereo (ch0 = local mic, ch1 = remote), and _load_audio averaged
them, so every remote speaker's embedding carried the local channel too.

This is NOT the cause of weak cross-meeting voiceprints — the meeting with the
worst speaker collisions has a bit-silent ch0, so contamination cannot explain
it. It is fixed because it is wrong: 5-9% of turns in recent recordings carry
local-channel energy within 10 dB of the remote speaker.
"""

import numpy as np
import soundfile as sf

from ai.biometrics import BiometricsManager


def _stereo(tmp_path):
    sr = 16000
    t = np.arange(sr, dtype=float) / sr
    left = np.sin(2 * np.pi * 200 * t)      # local mic
    right = np.sin(2 * np.pi * 900 * t)     # remote
    path = tmp_path / "s.wav"
    sf.write(str(path), np.stack([left, right], axis=1), sr)
    return str(path), left, right


def _manager(target_channel):
    m = BiometricsManager.__new__(BiometricsManager)
    m.target_channel = target_channel
    return m


def test_target_channel_one_returns_the_remote_channel(tmp_path):
    path, _, right = _stereo(tmp_path)
    signal, _ = _manager(1)._load_audio(path)
    assert signal.shape[0] == 1
    assert np.corrcoef(signal[0].numpy(), right)[0, 1] > 0.99


def test_target_channel_zero_returns_the_mic_channel(tmp_path):
    path, left, _ = _stereo(tmp_path)
    signal, _ = _manager(0)._load_audio(path)
    assert np.corrcoef(signal[0].numpy(), left)[0, 1] > 0.99


def test_none_still_downmixes(tmp_path):
    path, left, right = _stereo(tmp_path)
    signal, _ = _manager(None)._load_audio(path)
    assert np.corrcoef(signal[0].numpy(), (left + right) / 2)[0, 1] > 0.99


def test_mono_input_is_unaffected(tmp_path):
    sr = 16000
    mono = np.sin(2 * np.pi * 300 * np.arange(sr, dtype=float) / sr)
    path = tmp_path / "m.wav"
    sf.write(str(path), mono, sr)
    # Channel 1 is out of range for a mono file; it must not crash or truncate.
    signal, _ = _manager(1)._load_audio(str(path))
    assert signal.shape[0] == 1
    assert np.corrcoef(signal[0].numpy(), mono)[0, 1] > 0.99
