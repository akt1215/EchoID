"""Mean-centering the embedding space before cosine comparison.

Raw ECAPA embeddings carry a large shared common-mean component, which compresses
all pairwise cosines into a high band so distinct speakers score ~0.85 and get
fused. Subtracting a global mean before comparing restores discrimination. These
tests pin that behavior on `_centroid`, `merge_clusters`, and `resolve_clusters`.
"""

import numpy as np

from ai.identity_resolution import _centroid, merge_clusters, resolve_clusters


def _seg(start, end, spk):
    return {"start": start, "end": end, "speaker": spk}


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# ── _centroid ──────────────────────────────────────────────────────────────

def test_centroid_subtracts_mean_before_normalizing():
    mean = np.array([5.0, 5.0, 5.0, 5.0])
    vecs = [mean + np.array([1.0, 0, 0, 0]), mean + np.array([1.0, 0, 0, 0])]
    out = _centroid(vecs, mean=mean)
    # After removing the shared mean, the centroid points along the residual only.
    np.testing.assert_allclose(out, _unit([1.0, 0, 0, 0]), atol=1e-6)


def test_centroid_without_mean_is_unchanged():
    vecs = [np.array([1.0, 0]), np.array([1.0, 0])]
    np.testing.assert_allclose(_centroid(vecs), _unit([1.0, 0]), atol=1e-6)


# ── merge_clusters with a shared common mean ────────────────────────────────

def _common_mean_speakers():
    # Two DISTINCT speakers whose raw embeddings share a big common component.
    mean = np.array([5.0, 5.0, 5.0, 5.0])
    a = [mean + np.array([1.0, 0, 0, 0]), mean + np.array([1.0, 0.1, 0, 0])]
    b = [mean + np.array([0, 1.0, 0, 0]), mean + np.array([0.1, 1.0, 0, 0])]
    segs = [_seg(0, 5, "S0"), _seg(5, 10, "S0"),
            _seg(10, 15, "S1"), _seg(15, 20, "S1")]
    return mean, segs, a + b


def test_merge_without_mean_overmerges_shared_mean_speakers():
    # Documents the bug: raw cosine cannot separate them, so they collapse to one.
    mean, segs, embs = _common_mean_speakers()
    out = merge_clusters(segs, embs, threshold=0.85)
    assert len({s["speaker"] for s in out}) == 1


def test_merge_with_mean_keeps_shared_mean_speakers_distinct():
    # The fix: centering by the global mean keeps the two speakers separate.
    mean, segs, embs = _common_mean_speakers()
    out = merge_clusters(segs, embs, mean=mean, threshold=0.85)
    assert len({s["speaker"] for s in out}) == 2


def test_merge_with_mean_still_merges_true_duplicates():
    # A genuine over-split (same residual direction) still merges under centering.
    mean = np.array([5.0, 5.0, 5.0, 5.0])
    embs = [mean + np.array([1.0, 0, 0, 0]), mean + np.array([1.0, 0.01, 0, 0])]
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 11, "SPEAKER_01")]
    out = merge_clusters(segs, embs, mean=mean, threshold=0.85)
    assert len({s["speaker"] for s in out}) == 1


# ── resolve_clusters DB match in the centered space ─────────────────────────

def test_resolve_clusters_db_match_uses_centered_space():
    # Voiceprints are enrolled in the centered space. With the mean supplied, the
    # returning speaker's raw cluster matches its centered print; a different
    # speaker does not collapse onto it.
    mean = np.array([5.0, 5.0, 5.0, 5.0])
    alice_print = _unit([1.0, 0, 0, 0])          # centered-space voiceprint
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 10, "SPEAKER_00"),
            _seg(11, 15, "SPEAKER_01"), _seg(16, 20, "SPEAKER_01")]
    alice_raw = [mean + np.array([1.0, 0, 0, 0]), mean + np.array([1.0, 0.05, 0, 0])]
    bob_raw = [mean + np.array([0, 1.0, 0, 0]), mean + np.array([0.05, 1.0, 0, 0])]
    out = resolve_clusters(segs, alice_raw + bob_raw, db={"Alice": alice_print},
                           mean=mean, db_threshold=0.6)
    by_cluster = {s["cluster"]: s["speaker"] for s in out}
    assert "Alice" in by_cluster.values()
    # Bob stays a distinct, unnamed cluster rather than collapsing onto Alice.
    assert any(name.startswith("Unknown (") for name in by_cluster.values())
    assert len(set(by_cluster.values())) == 2
