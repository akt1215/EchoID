"""Guards against fusing several people into one identity.

Fusing two speakers is the worst outcome this pipeline can produce and it is
unrecoverable, while an over-split costs one merge click in review. The cases
below are drawn from measurements on real meetings:

  - two clusters 0.547 apart both matched "Jordan Lee" (2026-07-22)
  - two known-distinct speakers sat at 0.826, above merge_threshold 0.72
    (2026-07-15)
"""

import numpy as np

from ai import identity_resolution as ir


def _unit(*xs):
    v = np.asarray(xs, dtype=float)
    return v / np.linalg.norm(v)


def _segments(spec):
    """spec: [(speaker, n_turns)] -> flat segment list with 1s turns."""
    segs, t = [], 0.0
    for speaker, n in spec:
        for _ in range(n):
            segs.append({"start": t, "end": t + 1.0, "speaker": speaker})
            t += 1.0
    return segs


def test_dissimilar_clusters_do_not_collapse_into_one_identity():
    # The real 2026-07-22 case: two clusters both match Jordan Lee, but sit only
    # ~0.55 from each other. Collapsing them fuses two people into one.
    a, b = _unit(1, 0, 0), _unit(0.55, 0.835, 0)
    db = {"Jordan Lee": _unit(0.85, 0.42, 0)}
    segs = _segments([("S0", 3), ("S1", 3)])
    embs = [a] * 3 + [b] * 3
    out = ir.resolve_clusters(segs, embs, db, db_threshold=0.6,
                              collapse_similarity=0.72)
    names = {s["speaker"] for s in out}
    assert len(names) == 2, f"two people were fused: {names}"
    assert any(n.startswith("Unknown") for n in names)


def test_similar_clusters_still_collapse_when_allowed():
    a, b = _unit(1, 0, 0), _unit(0.99, 0.141, 0)
    db = {"Jordan Lee": _unit(1, 0, 0)}
    segs = _segments([("S0", 3), ("S1", 3)])
    out = ir.resolve_clusters(segs, [a] * 3 + [b] * 3, db,
                              db_threshold=0.6, collapse_similarity=0.72)
    assert {s["speaker"] for s in out} == {"Jordan Lee"}


def test_margin_rejects_an_ambiguous_match():
    # top1 0.80 vs top2 0.79 — a coin flip, so refuse to name it.
    q = _unit(1, 0, 0)
    db = {"A": _unit(0.80, 0.6, 0), "B": _unit(0.79, 0.613, 0)}
    segs = _segments([("S0", 3)])
    out = ir.resolve_clusters(segs, [q] * 3, db, db_threshold=0.6, db_margin=0.10)
    assert out[0]["speaker"].startswith("Unknown")


def test_margin_allows_a_clear_match():
    q = _unit(1, 0, 0)
    db = {"A": _unit(0.99, 0.141, 0), "B": _unit(0.2, 0.98, 0)}
    segs = _segments([("S0", 3)])
    out = ir.resolve_clusters(segs, [q] * 3, db, db_threshold=0.6, db_margin=0.10)
    assert out[0]["speaker"] == "A"


def test_db_match_reports_score_and_margin():
    q = _unit(1, 0, 0)
    db = {"A": _unit(1, 0, 0), "B": _unit(0, 1, 0)}
    name, score, margin = ir._db_match(q, db, threshold=0.6)
    assert name == "A"
    assert score == 1.0
    assert abs(margin - 1.0) < 1e-9


def test_default_collapse_similarity_never_fuses():
    # Fail-safe: without an explicit opt-in, two clusters matching one name stay
    # separate rather than risking the worst outcome.
    a, b = _unit(1, 0, 0), _unit(0.99, 0.141, 0)
    db = {"Jordan Lee": _unit(1, 0, 0)}
    segs = _segments([("S0", 3), ("S1", 3)])
    out = ir.resolve_clusters(segs, [a] * 3 + [b] * 3, db, db_threshold=0.6)
    assert len({s["speaker"] for s in out}) == 2


def test_best_scoring_cluster_keeps_the_name():
    a, b = _unit(1, 0, 0), _unit(0.55, 0.835, 0)
    db = {"Jordan Lee": _unit(1, 0, 0)}
    segs = _segments([("S0", 3), ("S1", 3)])
    out = ir.resolve_clusters(segs, [a] * 3 + [b] * 3, db, db_threshold=0.5,
                              collapse_similarity=0.72)
    by_cluster = {s["cluster"]: s["speaker"] for s in out}
    assert by_cluster["db::Jordan Lee"] == "Jordan Lee"          # the closer one
    assert by_cluster["S1"].startswith("Unknown")


def test_precomputed_db_names_are_still_guarded():
    # main.py passes db_names= from cluster_db_names, so this is the path that
    # actually runs in the pipeline; the guard must hold on it too.
    a, b = _unit(1, 0, 0), _unit(0.55, 0.835, 0)
    db = {"Jordan Lee": _unit(1, 0, 0)}
    segs = _segments([("S0", 3), ("S1", 3)])
    out = ir.resolve_clusters(segs, [a] * 3 + [b] * 3, db,
                              db_names={"S0": "Jordan Lee", "S1": "Jordan Lee"},
                              collapse_similarity=0.72)
    names = {s["speaker"] for s in out}
    assert len(names) == 2, f"two people were fused on the precomputed path: {names}"


def test_precomputed_path_ranks_rivals_by_real_score():
    # The better-matching cluster keeps the name even when it is not first.
    far, near = _unit(0.55, 0.835, 0), _unit(1, 0, 0)
    db = {"Jordan Lee": _unit(1, 0, 0)}
    segs = _segments([("S0", 3), ("S1", 3)])       # S0 is the WORSE match
    out = ir.resolve_clusters(segs, [far] * 3 + [near] * 3, db,
                              db_names={"S0": "Jordan Lee", "S1": "Jordan Lee"},
                              collapse_similarity=0.72)
    by_cluster = {s["cluster"]: s["speaker"] for s in out}
    assert by_cluster["db::Jordan Lee"] == "Jordan Lee"
    assert by_cluster["S0"].startswith("Unknown")


def test_merge_requires_mutual_nearest_neighbours():
    # C sits nearest to B, but B is nearest to A. A one-sided threshold test
    # chains all three into one speaker; mutual-best keeps C separate.
    a, b, c = _unit(1, 0, 0), _unit(0.97, 0.24, 0), _unit(0.90, 0.44, 0)
    segs = _segments([("A", 2), ("B", 2), ("C", 2)])
    embs = [a] * 2 + [b] * 2 + [c] * 2
    out = ir.merge_clusters(segs, embs, threshold=0.85, min_dur=0.5)
    assert len({s["speaker"] for s in out}) >= 2


def test_mutual_pair_still_merges():
    a, b = _unit(1, 0, 0), _unit(0.999, 0.045, 0)
    segs = _segments([("A", 2), ("B", 2)])
    out = ir.merge_clusters(segs, [a] * 2 + [b] * 2, threshold=0.85, min_dur=0.5)
    assert len({s["speaker"] for s in out}) == 1


def test_distinct_speakers_below_threshold_are_untouched():
    a, b = _unit(1, 0, 0), _unit(0, 1, 0)
    segs = _segments([("A", 2), ("B", 2)])
    out = ir.merge_clusters(segs, [a] * 2 + [b] * 2, threshold=0.85, min_dur=0.5)
    assert len({s["speaker"] for s in out}) == 2
