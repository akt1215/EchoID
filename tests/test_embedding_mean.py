"""Persistent global embedding-mean store (running mean + count)."""

import json
import os

import numpy as np
import pytest

from core import embedding_mean


def test_load_absent_returns_none(tmp_path):
    mean, count = embedding_mean.load(str(tmp_path / "missing.json"))
    assert mean is None
    assert count == 0


def test_update_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "mean.json")
    embs = [np.array([1.0, 3.0]), np.array([3.0, 1.0])]
    embedding_mean.update(path, embs)
    mean, count = embedding_mean.load(path)
    assert count == 2
    np.testing.assert_allclose(mean, [2.0, 2.0])


def test_running_mean_accumulates_across_updates(tmp_path):
    path = str(tmp_path / "mean.json")
    embedding_mean.update(path, [np.array([0.0, 0.0]), np.array([0.0, 0.0])])
    embedding_mean.update(path, [np.array([4.0, 4.0]), np.array([4.0, 4.0])])
    mean, count = embedding_mean.load(path)
    assert count == 4
    # Running mean over all four vectors: (0+0+4+4)/4 = 2.
    np.testing.assert_allclose(mean, [2.0, 2.0])


def test_update_ignores_empty_embeddings(tmp_path):
    path = str(tmp_path / "mean.json")
    embedding_mean.update(path, [np.array([2.0, 2.0]), np.array([]), np.array([4.0, 4.0])])
    mean, count = embedding_mean.load(path)
    assert count == 2
    np.testing.assert_allclose(mean, [3.0, 3.0])


def test_update_with_key_is_idempotent(tmp_path):
    # Reprocessing the same meeting (same key) must not double-count its turns.
    path = str(tmp_path / "mean.json")
    embs = [np.array([2.0, 2.0]), np.array([4.0, 4.0])]
    embedding_mean.update(path, embs, key="meeting_A")
    embedding_mean.update(path, embs, key="meeting_A")  # reprocess
    mean, count = embedding_mean.load(path)
    assert count == 2
    np.testing.assert_allclose(mean, [3.0, 3.0])


def test_update_distinct_keys_accumulate(tmp_path):
    path = str(tmp_path / "mean.json")
    embedding_mean.update(path, [np.array([0.0, 0.0]), np.array([0.0, 0.0])], key="A")
    embedding_mean.update(path, [np.array([4.0, 4.0]), np.array([4.0, 4.0])], key="B")
    mean, count = embedding_mean.load(path)
    assert count == 4
    np.testing.assert_allclose(mean, [2.0, 2.0])


def test_load_degrades_on_corrupt_store_and_quarantines_bad_file(tmp_path):
    # A crash mid-write (or any other corruption) must not brick every later
    # run -- load() must degrade to the empty-store result rather than raise,
    # and the corrupt file must be preserved aside for inspection, not lost.
    path = str(tmp_path / "mean.json")
    with open(path, "w") as f:
        f.write("{not valid json")  # simulates a truncated mid-write

    mean, count = embedding_mean.load(path)

    assert mean is None
    assert count == 0
    assert not os.path.exists(path)
    assert os.path.exists(path + ".bad")
    with open(path + ".bad") as f:
        assert f.read() == "{not valid json"


def test_update_degrades_on_corrupt_store_and_bootstraps_fresh_mean(tmp_path):
    # update() must also survive a corrupt store: fold the new embeddings into
    # a fresh mean instead of crashing the whole pipeline on a torn file.
    path = str(tmp_path / "mean.json")
    with open(path, "w") as f:
        f.write("{not valid json")

    mean = embedding_mean.update(path, [np.array([2.0, 2.0]), np.array([4.0, 4.0])])

    np.testing.assert_allclose(mean, [3.0, 3.0])
    assert os.path.exists(path + ".bad")
    new_mean, count = embedding_mean.load(path)
    assert count == 2
    np.testing.assert_allclose(new_mean, [3.0, 3.0])


def test_update_does_not_truncate_existing_store_on_mid_write_failure(tmp_path, monkeypatch):
    # Proves the write is atomic, not just claims it: mirrors
    # test_speaker_directory.py::test_apply_rename_does_not_truncate_a_target_file_on_mid_write_failure.
    # json.dumps is made to raise; against a temp-file-then-replace write that
    # happens BEFORE any file is opened, the pre-existing store must survive
    # untouched. Against a plain open(path, "w") + json.dump, the raise fires
    # after the file has already been truncated by the open call.
    path = str(tmp_path / "mean.json")
    embedding_mean.update(path, [np.array([1.0, 3.0]), np.array([3.0, 1.0])])
    with open(path) as f:
        before = f.read()

    def fake_dumps(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(json, "dumps", fake_dumps)

    with pytest.raises(RuntimeError):
        embedding_mean.update(path, [np.array([5.0, 5.0])])

    with open(path) as f:
        assert f.read() == before
