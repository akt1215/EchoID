"""Re-extract per-cluster embeddings with an alternative backend, and cache them.

The sidecars store ECAPA vectors, so the harness can only compare scoring
strategies over one embedding model. Judging a different model — the whole point
of the bake-off — means re-extracting from the WAVs, which is slow enough to be
worth caching.

Cache lives beside the ground truth in `workspace/eval/`, keyed by backend, and
is keyed per (meeting, cluster) so a partial run can be resumed.
"""

import os

import numpy as np

from evaluation import clusters


def cache_path(workspace, backend):
    return os.path.join(workspace, "eval", f"embeddings_{backend}.npz")


def _key(group):
    return f"{group.meeting}||{group.cluster}"


def load_cache(path):
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def extract(groups, manager, cache, max_turns=40, min_seconds=0.5, log=print):
    """{(meeting, cluster): matrix of turn embeddings}, filling `cache` in place.

    Only the longest `max_turns` turns of each cluster are embedded: the tail of
    a long meeting adds little and costs the most, and turns under `min_seconds`
    give poor embeddings whatever the model.
    """
    out = {}
    for i, group in enumerate(groups, 1):
        key = _key(group)
        if key in cache:
            out[(group.meeting, group.cluster)] = cache[key]
            continue

        # Loudest, NOT longest. A cluster's longest turn is often a diarization
        # artifact with nothing audible in it, and embedding those produces a
        # voiceprint built from room noise that matches nobody — including the
        # same speaker in another meeting.
        turns = clusters.ranked_turns(group, max_turns=max_turns,
                                      min_seconds=min_seconds)

        signal, fs = manager._load_audio(group.wav)
        vecs = []
        for start, end in turns:
            try:
                emb = manager.extract_embedding(group.wav, start, end,
                                                signal=signal, fs=fs)
            except Exception:
                continue
            vecs.append(np.asarray(emb, dtype=float))
        if not vecs:
            log(f"  [{i}/{len(groups)}] {group.cluster}: no embeddings, skipped")
            continue
        mat = np.stack(vecs)
        cache[key] = mat
        out[(group.meeting, group.cluster)] = mat
        log(f"  [{i}/{len(groups)}] {group.meeting[8:]} {group.cluster}: "
            f"{len(vecs)} turns -> {mat.shape[1]}d")
    return out


def save_cache(path, cache):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **cache)


def regroup(groups, matrices, max_turns=40, min_seconds=0.5):
    """Rebuild Group tuples carrying the re-extracted embeddings.

    Turn spans are recomputed the same way `extract` chose them, so duration
    weighting lines up with the vectors it is weighting.
    """
    out = []
    for g in groups:
        mat = matrices.get((g.meeting, g.cluster))
        if mat is None:
            continue
        turns = clusters.ranked_turns(g, max_turns=max_turns,
                                      min_seconds=min_seconds)[:len(mat)]
        out.append(clusters.Group(
            meeting=g.meeting, cluster=g.cluster, wav=g.wav,
            n_turns=len(mat), total_seconds=sum(e - s for s, e in turns),
            turns=turns, embeddings=mat))
    return out
