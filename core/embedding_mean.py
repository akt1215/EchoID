"""Persistent global embedding-mean store.

Raw ECAPA speaker embeddings share a large common-mean direction (~75% of each
vector's magnitude), which compresses cosine similarities into a high narrow band
so distinct speakers score ~0.85 and get fused. Subtracting this global mean
before any cosine restores discrimination. This module owns the running mean:
one vector accumulated across every meeting's turns, persisted beside the speaker
DB (`workspace/embedding_mean.json`, gitignored like `speakers.json`).

The mean is a property of the model + audio domain, not of any one meeting, so a
running mean over many turns is a far better nuisance estimator than a
per-meeting mean (which would subtract a lone dominant speaker's own signal).
"""

import json
import os

import numpy as np

from core import atomic_write


def _read(path):
    if not os.path.exists(path):
        return None, 0, []
    try:
        with open(path) as f:
            data = json.load(f)
        return (np.asarray(data["mean"], dtype=float), int(data["count"]),
                list(data.get("meetings", [])))
    except (OSError, ValueError):
        # Corrupt/unreadable store (e.g. truncated by a crash mid-write):
        # preserve it aside for inspection rather than silently clobbering it,
        # then let the caller start fresh instead of crashing the pipeline.
        try:
            os.replace(path, path + ".bad")
        except OSError:
            pass
        return None, 0, []


def load(path):
    """Return (mean_vector, count). (None, 0) when the store does not exist yet."""
    mean, count, _ = _read(path)
    return mean, count


def update(path, embeddings, key=None):
    """Fold `embeddings` into the persisted running mean and return the new mean.

    Empty vectors (failed extractions) are skipped. When nothing new is supplied
    the stored mean is returned unchanged (None if the store is still empty).

    `key` (a stable per-meeting id) makes folding idempotent: reprocessing the
    same meeting via `--from-file` must not re-count its turns, which would bias
    the mean toward that one meeting's speakers. A key already recorded is a
    no-op; distinct keys accumulate. `key=None` always folds (unkeyed callers).
    """
    prev_mean, prev_count, meetings = _read(path)
    if key is not None and key in meetings:
        return prev_mean

    new = [np.asarray(e, dtype=float) for e in embeddings]
    new = [e for e in new if e.size]
    if not new:
        return prev_mean

    stacked = np.stack(new)
    if prev_mean is None:
        mean = stacked.mean(axis=0)
        count = len(new)
    else:
        total = prev_mean * prev_count + stacked.sum(axis=0)
        count = prev_count + len(new)
        mean = total / count

    if key is not None:
        meetings = meetings + [key]
    atomic_write.write_text(
        path, json.dumps({"mean": mean.tolist(), "count": count, "meetings": meetings}))
    return mean


def aligned(mean, dim):
    """`mean` if it applies to embeddings of width `dim`, else None.

    The stored mean belongs to whichever model produced it. Subtracting a
    192-dim ECAPA mean from a 256-dim WeSpeaker embedding raises deep inside
    numpy, far from the backend setting that actually caused it — so a width
    mismatch is treated as "this mean is not for these vectors" and dropped.

    `dim` of None or 0 means the caller could not determine a width (no
    embeddings to inspect); the mean is returned unchanged so behaviour is
    identical wherever there is nothing to check against.
    """
    if mean is None or not dim:
        return mean
    return mean if getattr(mean, "shape", (None,))[0] == dim else None


def load_aligned(path, dim):
    """Load the stored mean only when it matches `dim`. See `aligned`."""
    mean, _ = load(path)
    return aligned(mean, dim)
