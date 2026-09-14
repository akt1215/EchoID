"""Reprocessing a meeting invalidates its labels.

Cluster ids are assigned per diarization run. Re-running a meeting can drop an
id, and can reuse one for entirely different audio: after a reprocess,
`db::Avery Chen` in 2026-07-22_09-56-58 still existed but pointed at a
different cluster, so a "multiple" verdict recorded for the old one would have
been applied to the new one — quietly, and by the directory guard too.

A label belongs to the diarization it was made against, so when a meeting's
cluster set changes, every label for that meeting is suspect and is dropped.
"""

import pytest

from evaluation import ground_truth as gt
from evaluation.clusters import Group

import numpy as np


def _g(meeting, cluster):
    return Group(meeting=meeting, cluster=cluster, wav="w.wav", n_turns=5,
                 total_seconds=50.0, turns=[(0.0, 10.0)] * 5,
                 embeddings=np.ones((5, 3)))


def test_unchanged_meetings_are_untouched():
    labels = gt.set_label({}, "m1", "S0", "confirmed", "Jordan Lee")
    kept, dropped = gt.prune_stale(labels, [_g("m1", "S0")])
    assert dropped == []
    assert gt.confirmed_name(kept, "m1", "S0") == "Jordan Lee"


def test_a_meeting_whose_clusters_changed_loses_all_labels():
    labels = gt.set_label({}, "m1", "S0", "confirmed", "Jordan Lee")
    gt.set_label(labels, "m1", "S1", "multiple")
    # Reprocessed: S1 is gone, so the whole meeting's labels are suspect.
    kept, dropped = gt.prune_stale(labels, [_g("m1", "S0")])
    assert "m1" in dropped
    assert kept.get("m1", {}) == {}


def test_a_reused_cluster_id_does_not_keep_its_old_verdict():
    labels = gt.set_label({}, "m1", "db::Avery Chen", "multiple")
    gt.set_label(labels, "m1", "SPEAKER_02", "confirmed", "Riley Stone")
    # After reprocessing only db::Avery Chen survives by name -- but it is a
    # different cluster now, so its verdict must not carry over.
    kept, _ = gt.prune_stale(labels, [_g("m1", "db::Avery Chen")])
    assert gt.get(kept, "m1", "db::Avery Chen") is None


def test_meetings_with_no_live_clusters_are_left_alone():
    # The recording may simply not be in this workspace; absence is not change.
    labels = gt.set_label({}, "m1", "S0", "confirmed", "Jordan Lee")
    kept, dropped = gt.prune_stale(labels, [_g("m2", "S0")])
    assert dropped == []
    assert gt.confirmed_name(kept, "m1", "S0") == "Jordan Lee"


def test_extra_live_clusters_alone_do_not_invalidate():
    # A newly discovered cluster that was never labeled is just unlabeled work,
    # not evidence the old labels are wrong.
    labels = gt.set_label({}, "m1", "S0", "confirmed", "Jordan Lee")
    kept, dropped = gt.prune_stale(labels, [_g("m1", "S0"), _g("m1", "S1")])
    assert dropped == []
    assert gt.confirmed_name(kept, "m1", "S0") == "Jordan Lee"


def test_pruning_does_not_mutate_the_input():
    labels = gt.set_label({}, "m1", "S0", "confirmed", "Jordan Lee")
    gt.prune_stale(labels, [_g("m1", "SOMETHING_ELSE")])
    assert gt.confirmed_name(labels, "m1", "S0") == "Jordan Lee"
