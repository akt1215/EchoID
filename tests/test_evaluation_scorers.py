import numpy as np

from evaluation import scorers
from evaluation.clusters import Group


def _g(meeting, cluster, vecs, durs):
    return Group(meeting=meeting, cluster=cluster, wav="w.wav", n_turns=len(vecs),
                 total_seconds=float(sum(durs)),
                 turns=[(0.0, d) for d in durs],
                 embeddings=np.asarray(vecs, dtype=float))


def test_baseline_centroid_is_unit_length():
    s = scorers.Baseline(mean=None)
    c = s.centroid(_g("m", "S0", [[3, 4, 0], [3, 4, 0]], [1.0, 1.0]))
    assert abs(np.linalg.norm(c) - 1.0) < 1e-9


def test_baseline_subtracts_the_global_mean():
    s = scorers.Baseline(mean=np.asarray([1.0, 1.0, 0.0]))
    c = s.centroid(_g("m", "S0", [[2, 1, 0]], [1.0]))
    assert np.allclose(c, [1.0, 0.0, 0.0])


def test_duration_weighting_favours_the_long_turn():
    plain = scorers.Baseline(mean=None)
    weighted = scorers.DurationWeighted(mean=None, min_seconds=0.0)
    g = _g("m", "S0", [[1, 0, 0], [0, 1, 0]], [1.0, 9.0])
    assert weighted.centroid(g)[1] > plain.centroid(g)[1]


def test_short_turns_are_dropped_when_longer_ones_exist():
    s = scorers.DurationWeighted(mean=None, min_seconds=2.0)
    g = _g("m", "S0", [[1, 0, 0], [0, 1, 0]], [0.5, 5.0])
    assert np.allclose(s.centroid(g), [0.0, 1.0, 0.0])


def test_all_short_turns_still_produce_a_centroid():
    s = scorers.DurationWeighted(mean=None, min_seconds=10.0)
    g = _g("m", "S0", [[1, 0, 0]], [0.5])
    assert np.allclose(s.centroid(g), [1.0, 0.0, 0.0])


def test_score_is_cosine():
    s = scorers.Baseline(mean=None)
    assert abs(s.score(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) - 1.0) < 1e-9
    assert abs(s.score(np.array([1.0, 0, 0]), np.array([0.0, 1, 0]))) < 1e-9


def test_asnorm_discounts_similarity_that_everyone_shares():
    """The point of cohort normalization, and why it beats a single global mean.

    Two pairs with the SAME raw cosine: one sits inside the cohort's dense
    region (so being similar there is unremarkable), the other sits away from it
    (so the same similarity is real evidence). Raw cosine cannot tell them
    apart; AS-Norm must rank the crowded pair lower. This is the per-session
    direction that global-mean centering leaves behind.
    """
    cohort = [_g(f"m{k}", "S0", [[1.0, 0.05 * k, 0.0]], [4.0])
              for k in range(-4, 5) if k]
    # Both pairs are cosine ~0.9802 apart.
    crowded = (_g("mA", "S0", [[1.0, 0.1, 0.0]], [4.0]),
               _g("mB", "S0", [[1.0, -0.1, 0.0]], [4.0]))
    distant = (_g("mC", "S0", [[0.0, 0.1, 1.0]], [4.0]),
               _g("mD", "S0", [[0.0, -0.1, 1.0]], [4.0]))

    raw = scorers.Baseline(mean=None)
    raw_crowded = raw.score(raw.centroid(crowded[0]), raw.centroid(crowded[1]))
    raw_distant = raw.score(raw.centroid(distant[0]), raw.centroid(distant[1]))
    assert abs(raw_crowded - raw_distant) < 1e-9, "raw cosine cannot separate them"

    norm = scorers.ASNorm(mean=None)
    norm.prepare(cohort)
    n_crowded = norm.score(norm.centroid(crowded[0]), norm.centroid(crowded[1]))
    n_distant = norm.score(norm.centroid(distant[0]), norm.centroid(distant[1]))
    assert n_crowded < n_distant


def test_asnorm_without_prepare_falls_back_to_cosine():
    s = scorers.ASNorm(mean=None)
    v = np.array([1.0, 0.0, 0.0])
    assert abs(s.score(v, v) - 1.0) < 1e-9


def test_registry_exposes_every_strategy():
    assert set(scorers.SCORERS) >= {"baseline", "duration", "asnorm"}
