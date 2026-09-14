"""A stored mean only applies to embeddings of its own width.

`embedding_mean.json` is model-specific. After the ECAPA -> WeSpeaker switch the
stored mean was still 192-dim while embeddings and voiceprints had become
256-dim, and every consumer subtracted it unconditionally. Opening Manage
Speakers crashed with

    ValueError: operands could not be broadcast together with shapes (256,) (192,)

and the review screen carried the same fault. A width mismatch means the mean
belongs to a different model, so it must be ignored rather than subtracted --
and ignored defensively at the point of use, because a config flag only protects
the call sites someone remembered to change.
"""

import json

import numpy as np

from core import embedding_mean


def _store(tmp_path, dim, count=10):
    p = tmp_path / "embedding_mean.json"
    p.write_text(json.dumps({"mean": [0.5] * dim, "count": count, "meetings": []}))
    return str(p)


def test_a_matching_mean_is_returned(tmp_path):
    mean = embedding_mean.load_aligned(_store(tmp_path, 192), 192)
    assert mean is not None and mean.shape == (192,)


def test_a_mismatched_mean_is_ignored(tmp_path):
    assert embedding_mean.load_aligned(_store(tmp_path, 192), 256) is None


def test_a_missing_store_is_ignored(tmp_path):
    assert embedding_mean.load_aligned(str(tmp_path / "nope.json"), 256) is None


def test_an_unknown_dimension_keeps_the_mean(tmp_path):
    # dim=None means "caller cannot tell yet"; the mean is returned unchanged so
    # existing behaviour is untouched when there is nothing to compare against.
    assert embedding_mean.load_aligned(_store(tmp_path, 192), None) is not None


def test_zero_dimension_is_treated_as_unknown(tmp_path):
    # An empty embedding list must not be read as "width 0" and silently drop a
    # perfectly good mean.
    assert embedding_mean.load_aligned(_store(tmp_path, 192), 0) is not None


def test_align_returns_none_for_a_mismatch():
    assert embedding_mean.aligned(np.zeros(192), 256) is None
    assert embedding_mean.aligned(np.zeros(256), 256) is not None
    assert embedding_mean.aligned(None, 256) is None
