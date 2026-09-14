"""Model-free tests for per-cluster identity resolution and name validation."""

import numpy as np

from ai.identity_resolution import (
    cluster_db_names, looks_like_name, merge_clusters, resolve_clusters)


# ── looks_like_name ────────────────────────────────────────────────────────

def test_accepts_ordinary_names():
    assert looks_like_name("Emilia Hermann")
    assert looks_like_name("Riley Stone")
    assert looks_like_name("Alice")
    assert looks_like_name("Jordan Lee")


def test_rejects_ocr_garbage():
    # Real garbage observed in a recorded meeting's OCR log.
    for junk in ["0 a: ay", "~ re oi ~", '"eo', "| le :", "is bh) feheshar)  § an",
                 "ane vl", "Sy if ws", "- 3", "pHa 5"]:
        assert not looks_like_name(junk), junk


def test_rejects_empty_and_degenerate():
    assert not looks_like_name("")
    assert not looks_like_name("   ")
    assert not looks_like_name("a")
    assert not looks_like_name("x" * 60)  # implausibly long
    assert not looks_like_name("a b c d e")  # too many lowercase tokens


def test_fuzzy_accepts_slight_misread_of_known_name():
    # A near-miss of a known name is rescued even if the shape is borderline.
    known = ["Riley Stone", "Emilia Hermann"]
    assert looks_like_name("Emilia Hermnn", known)   # dropped a letter
    assert not looks_like_name("0 a: ay", known)     # still garbage


# ── resolve_clusters ───────────────────────────────────────────────────────

def _seg(start, end, spk):
    return {"start": start, "end": end, "speaker": spk}


def test_one_name_per_cluster_no_fragmentation():
    # Two pyannote clusters, several turns each, no visual names → exactly two
    # identities (one row per cluster), never one row per turn.
    segs = [_seg(0, 4, "SPEAKER_00"), _seg(5, 9, "SPEAKER_00"),
            _seg(10, 14, "SPEAKER_01"), _seg(15, 19, "SPEAKER_01")]
    e00 = np.array([1.0, 0, 0, 0]); e01 = np.array([0, 1.0, 0, 0])
    out = resolve_clusters(segs, [e00, e00, e01, e01], db={}, cluster_visual_names={})

    assert len({s["cluster"] for s in out}) == 2
    assert all(s["speaker"].startswith("Unknown (") for s in out)


def test_db_match_names_and_prefills_cluster():
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 10, "SPEAKER_00")]
    graham = np.array([1.0, 0, 0, 0])
    out = resolve_clusters(segs, [graham, graham], db={"Riley Stone": graham},
                           cluster_visual_names={}, db_threshold=0.85)
    assert all(s["speaker"] == "Riley Stone" for s in out)
    assert all(s["source"] == "db" for s in out)


def test_db_match_merges_over_clustered_pyannote_ids():
    # Same person split by pyannote into two ids → one DB identity, one cluster.
    # Collapsing two clusters into one identity now requires an explicit
    # `collapse_similarity`, because two clusters merely both scoring above the
    # DB threshold is not evidence they are the same person. These two are
    # cosine ~0.9997 apart, so a genuine over-split still re-merges; the
    # pipeline passes this value from config (biometrics.collapse_similarity).
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 10, "SPEAKER_03")]
    v = np.array([1.0, 0, 0, 0])
    out = resolve_clusters(segs, [v, v * 0.99 + 0.01], db={"Riley Stone": v},
                           cluster_visual_names={}, db_threshold=0.55,
                           collapse_similarity=0.80)
    assert len({s["cluster"] for s in out}) == 1
    assert all(s["speaker"] == "Riley Stone" for s in out)


def test_over_split_is_not_remerged_without_an_explicit_threshold():
    # The same input as above, at the fail-safe default: the two clusters stay
    # separate. An over-split costs one merge click in review; fusing two people
    # is unrecoverable, so the default refuses to collapse.
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 10, "SPEAKER_03")]
    v = np.array([1.0, 0, 0, 0])
    out = resolve_clusters(segs, [v, v * 0.99 + 0.01], db={"Riley Stone": v},
                           cluster_visual_names={})
    assert len({s["cluster"] for s in out}) == 2


def test_visual_name_used_when_db_misses():
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 10, "SPEAKER_00")]
    v = np.array([0, 0, 1.0, 0])
    out = resolve_clusters(segs, [v, v], db={},
                           cluster_visual_names={"SPEAKER_00": "Emilia Hermann"})
    assert all(s["speaker"] == "Emilia Hermann" for s in out)
    assert all(s["source"] == "visual" for s in out)


def test_garbage_visual_name_is_rejected_to_unknown():
    segs = [_seg(0, 5, "SPEAKER_00")]
    v = np.array([0, 0, 1.0, 0])
    out = resolve_clusters(segs, [v], db={},
                           cluster_visual_names={"SPEAKER_00": "0 a: ay"})
    assert out[0]["speaker"] == "Unknown (SPEAKER_00)"
    assert out[0]["source"] == "unknown"


def test_unknown_when_no_db_and_no_visual():
    segs = [_seg(0, 5, "SPEAKER_02")]
    v = np.array([0, 0, 0, 1.0])
    out = resolve_clusters(segs, [v], db={}, cluster_visual_names={})
    assert out[0]["speaker"] == "Unknown (SPEAKER_02)"
    assert out[0]["source"] == "unknown"


def test_cluster_db_names_reports_match_and_miss():
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 10, "SPEAKER_01")]
    g = np.array([1.0, 0, 0, 0]); other = np.array([0, 1.0, 0, 0])
    names = cluster_db_names(segs, [g, other], db={"Riley Stone": g}, db_threshold=0.85)
    assert names["SPEAKER_00"] == "Riley Stone"
    assert names["SPEAKER_01"] is None


# ── merge_clusters ─────────────────────────────────────────────────────────

def test_merge_clusters_collapses_similar_centroids():
    segs = [_seg(0, 5, "SPEAKER_00"), _seg(6, 11, "SPEAKER_01"), _seg(12, 17, "SPEAKER_02")]
    a = np.array([1.0, 0, 0]); b = np.array([0.98, 0.02, 0]); c = np.array([0, 1.0, 0])
    out = merge_clusters(segs, [a, b, c], threshold=0.85)
    labels = [s["speaker"] for s in out]
    assert labels[0] == labels[1]      # near-identical 00/01 merge
    assert labels[2] != labels[0]      # 02 stays distinct
    assert [s["start"] for s in out] == [0, 6, 12]  # segments preserved, only relabeled


def test_merge_clusters_noop_when_all_distinct():
    segs = [_seg(0, 5, "A"), _seg(6, 11, "B")]
    out = merge_clusters(segs, [np.array([1.0, 0]), np.array([0, 1.0])], threshold=0.85)
    assert [s["speaker"] for s in out] == ["A", "B"]
