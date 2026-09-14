import numpy as np

from core.channel_pairer import ChannelPairer


def test_pairs_equal_length_streams():
    p = ChannelPairer()
    p.add([1, 2, 3], [4, 5, 6], now=0.0)
    out = p.take(now=0.0)
    assert np.array_equal(out, [[1, 4], [2, 5], [3, 6]])
    assert p.stalled_side is None


def test_pairs_only_overlap_and_keeps_remainder():
    p = ChannelPairer()
    p.add([1, 2, 3], [4], now=0.0)
    assert np.array_equal(p.take(now=0.0), [[1, 4]])
    p.add([], [5, 6], now=0.1)
    assert np.array_equal(p.take(now=0.1), [[2, 5], [3, 6]])


def test_no_output_when_one_side_briefly_empty():
    p = ChannelPairer(stall_timeout=10.0)
    p.add([1, 2], [], now=0.0)
    out = p.take(now=1.0)  # within the stall window
    assert out.shape == (0, 2)
    assert p.stalled_side is None


def test_pads_silence_when_bh_stalls_past_timeout():
    p = ChannelPairer(stall_timeout=10.0)
    p.add([1, 2], [9], now=0.0)
    p.take(now=0.0)  # pairs [1,9], leaves mic=[2]
    p.add([3, 4], [], now=1.0)
    out = p.take(now=11.0)  # bh silent since t=0 → 11s ≥ 10s
    assert np.array_equal(out, [[2, 0], [3, 0], [4, 0]])
    assert p.stalled_side == "bh"


def test_pads_silence_when_mic_stalls_past_timeout():
    p = ChannelPairer(stall_timeout=10.0)
    p.add([1], [7], now=0.0)
    p.take(now=0.0)  # pairs [1,7]
    p.add([], [8, 9], now=1.0)
    out = p.take(now=11.0)  # mic silent since t=0
    assert np.array_equal(out, [[0, 8], [0, 9]])
    assert p.stalled_side == "mic"


def test_silent_ch1_from_start_pads_after_timeout():
    p = ChannelPairer(stall_timeout=10.0)
    p.add([1, 2], [], now=0.0)  # bh never produced
    assert p.take(now=9.0).shape == (0, 2)
    out = p.take(now=10.0)
    assert np.array_equal(out, [[1, 0], [2, 0]])
    assert p.stalled_side == "bh"


def test_final_flush_emits_padded_tail():
    p = ChannelPairer()
    p.add([1, 2, 3], [9], now=0.0)
    p.take(now=0.0)  # pairs [1,9], leaves mic=[2,3]
    flushed = _drain_final(p)
    assert np.array_equal(flushed, [[2, 0], [3, 0]])


def test_final_flush_pairs_then_pads():
    p = ChannelPairer()
    p.add([1, 2], [3], now=0.0)  # no running take() first
    flushed = _drain_final(p)
    assert np.array_equal(flushed, [[1, 3], [2, 0]])


def test_final_flush_empty_when_nothing_buffered():
    p = ChannelPairer()
    assert _drain_final(p).shape == (0, 2)


def _drain_final(p):
    """Loop take(final=True) the way the writer does at stop and concatenate."""
    chunks = []
    while True:
        out = p.take(now=0.0, final=True)
        if len(out) == 0:
            break
        chunks.append(out)
    return np.concatenate(chunks) if chunks else np.empty((0, 2))
