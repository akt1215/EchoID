"""The WeSpeaker ONNX backend needs features, not audio.

extract_embedding fed a raw waveform straight into the ONNX session, but
WeSpeaker's exported graphs take 80-dim fbank features. That produces garbage
rather than an error, which is why the backend had never actually worked.
"""

import numpy as np
import torch

from ai.biometrics import BiometricsManager


def _mgr():
    m = BiometricsManager.__new__(BiometricsManager)
    m.backend = "wespeaker_onnx"
    m.target_channel = None
    return m


def _tone(hz, n, sr):
    return torch.from_numpy(
        np.sin(2 * np.pi * hz * np.arange(n, dtype=np.float32) / sr)).unsqueeze(0)


def test_fbank_has_80_bins_and_a_batch_axis():
    feats = _mgr()._fbank(_tone(220, 16000, 16000), 16000)
    assert feats.ndim == 3
    assert feats.shape[0] == 1
    assert feats.shape[2] == 80
    assert feats.shape[1] > 10


def test_fbank_is_float32():
    assert _mgr()._fbank(_tone(300, 16000, 16000), 16000).dtype == np.float32


def test_fbank_is_mean_normalized():
    # WeSpeaker applies per-utterance CMN; without it scores drift with gain.
    feats = _mgr()._fbank(_tone(300, 16000, 16000), 16000)
    assert abs(float(feats.mean())) < 1e-4


def test_fbank_resamples_to_16k():
    # One second of audio yields the same frame count regardless of input rate.
    at_48k = _mgr()._fbank(_tone(300, 48000, 48000), 48000)
    at_16k = _mgr()._fbank(_tone(300, 16000, 16000), 16000)
    assert abs(at_48k.shape[1] - at_16k.shape[1]) <= 1
