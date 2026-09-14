import numpy as np

from evaluation import ground_truth as gt
from evaluation import trials
from evaluation.clusters import Group


def _g(meeting, cluster, n=10):
    return Group(meeting=meeting, cluster=cluster, wav="w.wav", n_turns=n,
                 total_seconds=float(n), turns=[(0.0, 1.0)] * n,
                 embeddings=np.ones((n, 3)))


def _labels(*rows):
    d = {}
    for meeting, cluster, verdict, name in rows:
        gt.set_label(d, meeting, cluster, verdict, name)
    return d


def test_positive_needs_same_name_across_meetings():
    groups = [_g("m1", "S0"), _g("m2", "S0")]
    labels = _labels(("m1", "S0", "confirmed", "Jordan Lee"),
                     ("m2", "S0", "confirmed", "Jordan Lee"))
    assert len(trials.positives(groups, labels)) == 1


def test_same_meeting_same_name_is_not_a_positive():
    # Optimistically biased (shared session), and it is really a merge case.
    groups = [_g("m1", "S0"), _g("m1", "S1")]
    labels = _labels(("m1", "S0", "confirmed", "Jordan Lee"),
                     ("m1", "S1", "confirmed", "Jordan Lee"))
    assert trials.positives(groups, labels) == []


def test_db_clusters_are_excluded_from_positives():
    groups = [_g("m1", "db::Jordan Lee"), _g("m2", "S0")]
    labels = _labels(("m1", "db::Jordan Lee", "confirmed", "Jordan Lee"),
                     ("m2", "S0", "confirmed", "Jordan Lee"))
    assert trials.positives(groups, labels) == []


def test_db_clusters_are_still_allowed_as_negatives():
    groups = [_g("m1", "db::Jordan Lee"), _g("m2", "S0")]
    labels = _labels(("m1", "db::Jordan Lee", "confirmed", "Jordan Lee"),
                     ("m2", "S0", "confirmed", "Avery Chen"))
    assert len(trials.negatives(groups, labels)) == 1


def test_negatives_include_same_meeting_distinct_people():
    groups = [_g("m1", "S0"), _g("m1", "S1")]
    labels = _labels(("m1", "S0", "confirmed", "Jordan Lee"),
                     ("m1", "S1", "confirmed", "Avery Chen"))
    assert len(trials.negatives(groups, labels)) == 1


def test_multiple_and_skip_are_excluded_from_both():
    groups = [_g("m1", "S0"), _g("m2", "S0"), _g("m2", "S1")]
    labels = _labels(("m1", "S0", "confirmed", "Jordan Lee"),
                     ("m2", "S0", "multiple", None),
                     ("m2", "S1", "skip", None))
    assert trials.positives(groups, labels) == []
    assert trials.negatives(groups, labels) == []


def test_unlabeled_groups_are_ignored():
    groups = [_g("m1", "S0"), _g("m2", "S0")]
    assert trials.positives(groups, {}) == []
    assert trials.negatives(groups, {}) == []


def test_labeled_returns_group_name_pairs():
    groups = [_g("m1", "S0"), _g("m2", "S0")]
    labels = _labels(("m1", "S0", "confirmed", "Jordan Lee"))
    assert [n for _, n in trials.labeled(groups, labels)] == ["Jordan Lee"]


def test_name_matching_ignores_spelling_punctuation():
    groups = [_g("m1", "S0"), _g("m2", "S0")]
    labels = _labels(("m1", "S0", "confirmed", "Riley Stone"),
                     ("m2", "S0", "confirmed", "Riley Stone"))
    assert len(trials.positives(groups, labels)) == 1
