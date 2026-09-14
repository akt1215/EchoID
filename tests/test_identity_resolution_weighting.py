"""Duration weighting and the minimum-speech gate.

Both come from measurement on 20 hand-labeled clusters. Weighting each turn by
its length carries the separation result rather than refining it: unweighted,
the same trials give margin -0.324 and 84% recall at zero false accepts.

The gate exists because a cluster with very little speech cannot be identified.
One cluster with 7.3s of usable speech scored 0.368-0.503 against the same
person elsewhere, below the 0.649 impostor ceiling — it both missed its true
match and sat in impostor territory. Leaving such a cluster Unknown is the
correct outcome under "prefer Unknown over a wrong name".
"""

import numpy as np

from ai import identity_resolution as ir


def _unit(*xs):
    v = np.asarray(xs, dtype=float)
    return v / np.linalg.norm(v)


def _segments(spec):
    """spec: [(speaker, [(duration, vector)])] -> (segments, embeddings)."""
    segs, embs, t = [], [], 0.0
    for speaker, turns in spec:
        for dur, vec in turns:
            segs.append({"start": t, "end": t + dur, "speaker": speaker})
            embs.append(vec)
            t += dur + 0.1
    return segs, embs


def test_centroid_weights_by_duration():
    a, b = _unit(1, 0, 0), _unit(0, 1, 0)
    plain = ir._centroid([a, b])
    weighted = ir._centroid([a, b], durations=[1.0, 9.0])
    assert weighted[1] > plain[1], "the long turn should dominate"


def test_centroid_drops_turns_below_the_floor():
    a, b = _unit(1, 0, 0), _unit(0, 1, 0)
    c = ir._centroid([a, b], durations=[0.4, 6.0], min_seconds=1.0)
    assert np.allclose(c, [0.0, 1.0, 0.0])


def test_centroid_keeps_short_turns_when_that_is_all_there_is():
    a = _unit(1, 0, 0)
    c = ir._centroid([a], durations=[0.4], min_seconds=5.0)
    assert np.allclose(c, [1.0, 0.0, 0.0])


def test_centroid_without_durations_is_unchanged():
    a, b = _unit(1, 0, 0), _unit(0, 1, 0)
    assert np.allclose(ir._centroid([a, b]), _unit(1, 1, 0))


def test_a_cluster_with_too_little_speech_stays_unknown():
    v = _unit(1, 0, 0)
    segs, embs = _segments([("S0", [(0.9, v), (0.9, v)])])      # 1.8s total
    out = ir.resolve_clusters(segs, embs, {"Jordan Lee": v}, {},
                              db_threshold=0.5, min_speech=30.0)
    assert out[0]["speaker"].startswith("Unknown")


def test_a_cluster_with_enough_speech_still_matches():
    v = _unit(1, 0, 0)
    segs, embs = _segments([("S0", [(20.0, v), (20.0, v)])])    # 40s total
    out = ir.resolve_clusters(segs, embs, {"Jordan Lee": v}, {},
                              db_threshold=0.5, min_speech=30.0)
    assert out[0]["speaker"] == "Jordan Lee"


def test_the_gate_counts_only_turns_above_the_floor():
    # Forty 0.2s turns is 8s of audio but no usable speech for an embedding.
    v = _unit(1, 0, 0)
    segs, embs = _segments([("S0", [(0.2, v)] * 40)])
    out = ir.resolve_clusters(segs, embs, {"Jordan Lee": v}, {},
                              db_threshold=0.5, min_speech=5.0,
                              min_turn_seconds=0.5)
    assert out[0]["speaker"].startswith("Unknown")


def test_the_gate_is_off_by_default():
    v = _unit(1, 0, 0)
    segs, embs = _segments([("S0", [(0.9, v), (0.9, v)])])
    out = ir.resolve_clusters(segs, embs, {"Jordan Lee": v}, {}, db_threshold=0.5)
    assert out[0]["speaker"] == "Jordan Lee"
