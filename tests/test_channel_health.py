"""A railed capture channel must be detected, or the downmix buries the mic.

Two real recordings (2026-07-28) captured full-scale noise on ch1. Diarization
found no speakers and transcription, which averages the channels, heard only the
noise -- 35% of those notes' characters were hallucinated. These pin the
detection and the mic-only fallback.
"""

import numpy as np
import soundfile as sf

from core import channel_health

SR = 16000


def _write(path, ch0, ch1=None):
    data = ch0 if ch1 is None else np.stack([ch0, ch1], axis=1)
    sf.write(str(path), np.asarray(data, dtype="float32"), SR)
    return str(path)


def _speech_like(n, amp=0.05, seed=0):
    """Quiet, varying signal — stands in for audio that never rails."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * amp).clip(-0.9, 0.9)


def _railed(n):
    """A failed capture rails rather than going silent: a full-scale square."""
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def test_a_railed_channel_is_unusable(tmp_path):
    n = SR * 10
    p = _write(tmp_path / "m.wav", _speech_like(n), _railed(n))
    assert channel_health.clipped_fraction(p, 1) > 0.9
    assert channel_health.is_unusable(p, 1)
    assert not channel_health.is_unusable(p, 0)


def test_a_healthy_channel_is_never_flagged(tmp_path):
    """Measured over 20 real recordings, healthy ch1 clips 0.000%. Flagging one
    by mistake would discard every remote speaker in the meeting."""
    n = SR * 10
    p = _write(tmp_path / "m.wav", _speech_like(n, seed=1), _speech_like(n, seed=2))
    assert channel_health.clipped_fraction(p, 1) == 0.0
    assert not channel_health.is_unusable(p, 1)


def test_occasional_peaks_are_not_a_railed_channel(tmp_path):
    """Real audio touches full scale now and then; only sustained railing counts."""
    n = SR * 10
    ch1 = _speech_like(n, seed=3)
    ch1[:: 5000] = 1.0                     # 0.02% of samples at full scale
    p = _write(tmp_path / "m.wav", _speech_like(n), ch1)
    assert 0 < channel_health.clipped_fraction(p, 1) < channel_health.CLIP_FRACTION
    assert not channel_health.is_unusable(p, 1)


def test_mono_and_missing_channel_are_not_flagged(tmp_path):
    p = _write(tmp_path / "mono.wav", _railed(SR * 10))
    assert channel_health.clipped_fraction(p, 1) == 0.0
    assert not channel_health.is_unusable(p, 1)


def test_an_unreadable_file_never_fails_the_run(tmp_path):
    bad = tmp_path / "nope.wav"
    bad.write_bytes(b"not a wav")
    assert not channel_health.is_unusable(str(bad), 1)


def test_extract_channel_writes_normalized_mono(tmp_path):
    n = SR * 2
    quiet = _speech_like(n, amp=0.001, seed=4)      # the buried mic
    p = _write(tmp_path / "m.wav", quiet, _railed(n))
    out = channel_health.extract_channel(p, 0, str(tmp_path / "mic.wav"))
    data, sr = sf.read(out)
    assert data.ndim == 1 and sr == SR and len(data) == n
    # brought up to a level Whisper can work with, and it is ch0 not the noise
    assert np.isclose(np.abs(data).max(), 0.9, atol=0.01)
    assert channel_health.clipped_fraction(out, 0) == 0.0


def test_extract_channel_handles_a_silent_channel(tmp_path):
    n = SR * 2
    p = _write(tmp_path / "m.wav", np.zeros(n), _speech_like(n))
    out = channel_health.extract_channel(p, 0, str(tmp_path / "mic.wav"))
    assert np.abs(sf.read(out)[0]).max() == 0.0      # no divide-by-zero blowup
