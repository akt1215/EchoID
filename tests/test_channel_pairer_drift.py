# tests/test_channel_pairer_drift.py
import numpy as np

from core.channel_pairer import ChannelPairer


def _skew_frames(rate_mic, rate_bh, seconds, marker_frac=0.9):
    """Feed the real pairer two mono streams produced at their own rates, each
    carrying a single unit marker at the SAME real time (marker_frac*seconds),
    and return |output-frame(mic marker) - output-frame(bh marker)|.

    The pairer index-pairs min(len) samples, so with equal rates the two markers
    land on the same output frame (skew 0); with unequal rates the faster stream
    places its marker at a higher index, so the skew = marker_t*|rate_mic-rate_bh|
    and grows linearly with meeting length."""
    total_mic = round(seconds * rate_mic)
    total_bh = round(seconds * rate_bh)
    mic = np.zeros(total_mic, dtype=np.float32)
    bh = np.zeros(total_bh, dtype=np.float32)
    mic[round(marker_frac * seconds * rate_mic)] = 1.0
    bh[round(marker_frac * seconds * rate_bh)] = 1.0

    p = ChannelPairer(stall_timeout=1e9)  # never trip the stall path
    p.add(mic, bh, now=0.0)
    chunks = []
    while True:
        out = p.take(now=0.0)
        if len(out) == 0:
            break
        chunks.append(out)
    paired = np.concatenate(chunks)
    return abs(int(np.argmax(paired[:, 0])) - int(np.argmax(paired[:, 1])))


def test_one_clock_same_rate_stays_aligned():
    # Both channels off ONE clock (equal rate) -> the fix's invariant: no skew,
    # even over a 5-minute meeting.
    assert _skew_frames(48000, 48000, seconds=300) == 0


def test_two_clocks_drift_is_linear_in_time():
    # Two independent clocks at ~1000 ppm apart: skew grows with meeting length.
    # ~300 ms at 5 min at 1000 ppm (the reported earbud symptom).
    skew_300s = _skew_frames(48000, 48048, seconds=300)
    assert skew_300s / 48000.0 > 0.25          # > 250 ms of drift at 5 min
    skew_150s = _skew_frames(48000, 48048, seconds=150)
    # Linear: half the meeting -> ~half the skew (±2% for rounding).
    assert abs(skew_300s - 2 * skew_150s) < 0.02 * skew_300s
