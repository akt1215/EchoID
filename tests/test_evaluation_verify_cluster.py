"""Probe selection for re-verifying a cluster labeled `multiple`.

The point of these tests is the failure mode the selection has to serve: the
intruders in a mixed cluster are SHORT turns, so a review that plays only the
long ones can never surface them, and one that stitches turns together (what
`loudest_turns` does) is what put the labels in doubt in the first place.
"""

import numpy as np

from evaluation import verify_cluster as vc


def _unit(*xs):
    v = np.asarray(xs, dtype=float)
    return v / np.linalg.norm(v)


# One voice on +x, a different voice on +y.
VOICE_A = _unit(1.0, 0.0, 0.0)
VOICE_B = _unit(0.0, 1.0, 0.0)


def test_dominant_centroid_follows_the_long_turns():
    """Many short foreign turns must not drag the dominant voice off the long
    ones — that is exactly how a blended voiceprint gets built."""
    turns = [(0.0, 30.0), (30.0, 60.0)] + [(60.0 + i, 60.4 + i) for i in range(20)]
    embs = np.stack([VOICE_A, VOICE_A] + [VOICE_B] * 20)
    c = vc.dominant_centroid(turns, embs)
    assert float(c @ VOICE_A) > 0.99


def test_dominant_centroid_is_none_without_long_turns():
    turns = [(0.0, 1.0), (2.0, 2.5)]
    assert vc.dominant_centroid(turns, np.stack([VOICE_A, VOICE_A])) is None


def test_probes_are_short_turns_worst_first():
    turns = [(0.0, 30.0)] + [(40.0, 41.0), (50.0, 51.0), (60.0, 61.0)]
    embs = np.stack([VOICE_A, VOICE_A, VOICE_B, _unit(1.0, 1.0, 0.0)])
    _, probes = vc.select_probes(turns, embs)
    # every probe is a short turn, and the least-like-the-dominant comes first
    assert [i for i, _ in probes] == [2, 3, 1]
    assert probes[0][1] < probes[-1][1]


def test_baseline_spreads_across_the_cluster_span():
    """A baseline drawn from one stretch of a long meeting can miss a speaker
    change entirely, so the long turns are sampled across the span."""
    turns = [(0.0, 20.0), (10.0, 25.0), (500.0, 520.0), (1000.0, 1030.0)]
    embs = np.stack([VOICE_A] * 4)
    baseline, _ = vc.select_probes(turns, embs, n_baseline=3)
    starts = [turns[i][0] for i in baseline]
    assert starts == sorted(starts)
    assert max(starts) - min(starts) > 500     # not all from one stretch


def test_baseline_is_played_as_separate_turns_not_stitched():
    """Each baseline entry must be a single real turn, so the reviewer hears
    continuous speech instead of the jump-cut montage that caused the doubt."""
    turns = [(0.0, 20.0), (100.0, 130.0), (200.0, 210.0)]
    embs = np.stack([VOICE_A] * 3)
    baseline, _ = vc.select_probes(turns, embs, n_baseline=3)
    assert len(set(baseline)) == len(baseline)
    for i in baseline:
        assert turns[i] in turns


def test_falls_back_to_longest_turns_when_nothing_is_long_enough():
    """A cluster of only short turns still has to be reviewable."""
    turns = [(0.0, 1.0), (2.0, 2.9), (4.0, 4.2)]
    embs = np.stack([VOICE_A, VOICE_B, VOICE_A])
    baseline, probes = vc.select_probes(turns, embs)
    assert baseline == [0]
    assert [i for i, _ in probes] == [1, 2] or [i for i, _ in probes] == [2, 1]


def test_probe_count_is_capped():
    turns = [(0.0, 30.0)] + [(40.0 + i, 40.5 + i) for i in range(50)]
    embs = np.stack([VOICE_A] + [VOICE_B] * 50)
    _, probes = vc.select_probes(turns, embs, n_probe=6)
    assert len(probes) == 6


def test_no_probes_when_every_turn_is_long():
    turns = [(0.0, 20.0), (30.0, 55.0)]
    embs = np.stack([VOICE_A, VOICE_A])
    baseline, probes = vc.select_probes(turns, embs)
    assert probes == []
    assert set(baseline) == {0, 1}
