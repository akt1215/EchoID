"""Discover the (meeting, diarization cluster) groups recorded in sidecars.

One group is one row a human can listen to and name. Grouping is by the sidecar
`cluster` field, the same stable key the review screen groups by.
"""

import glob
import json
import os
from collections import namedtuple

import numpy as np
import soundfile as sf

Group = namedtuple(
    "Group", "meeting cluster wav n_turns total_seconds turns embeddings")


def is_leaky(cluster_id):
    """True when this cluster was named BY the voiceprint DB, which was then
    EMA-updated FROM it. Such a group must never supply an evaluation positive."""
    return str(cluster_id).startswith("db::")


def discover(workspace, min_turns=5):
    """Every cluster with at least `min_turns` embedded turns, across sidecars."""
    out = []
    pattern = os.path.join(workspace, "*.segments.json")
    for path in sorted(glob.glob(pattern)):
        meeting = os.path.basename(path)[: -len(".segments.json")]
        with open(path) as f:
            data = json.load(f)
        wav = data.get("wav", "")
        grouped = {}
        for d in data.get("diarization", []):
            emb = d.get("embedding")
            if not emb:
                continue   # extraction failed for this turn; nothing to compare
            key = d.get("cluster", d.get("label"))
            turn = (float(d["start"]), float(d["end"]))
            grouped.setdefault(key, ([], []))
            grouped[key][0].append(turn)
            grouped[key][1].append(np.asarray(emb, dtype=float))
        for cluster, (turns, embs) in grouped.items():
            if len(turns) < min_turns:
                continue
            out.append(Group(
                meeting=meeting, cluster=cluster, wav=wav, n_turns=len(turns),
                total_seconds=sum(e - s for s, e in turns),
                turns=turns, embeddings=np.stack(embs),
            ))
    return out


def best_turn(group, min_seconds=3.0):
    """A turn to play as this group's sample: the longest turn at or above
    `min_seconds`, else simply the longest one available.

    Metadata only, so it cannot tell speech from silence — a long turn may be a
    diarization artifact with nothing audible in it. Prefer `loudest_turns`,
    which reads the audio; this remains the fallback when the WAV is gone.
    """
    long = [t for t in group.turns if t[1] - t[0] >= min_seconds]
    return max(long or group.turns, key=lambda t: t[1] - t[0])


def _turn_levels(group):
    """[(rms, start, end)] on the louder channel, or None if the WAV is missing.

    Read lazily and only for the turns of one group, so labeling stays snappy on
    a long meeting.
    """
    if not group.wav or not os.path.exists(group.wav):
        return None
    try:
        sr = sf.info(group.wav).samplerate
    except (OSError, RuntimeError):
        return None
    levels = []
    for start, end in group.turns:
        try:
            seg, _ = sf.read(group.wav, start=int(start * sr), stop=int(end * sr),
                             always_2d=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if not len(seg):
            continue
        # Whichever channel the speaker is on: mic (ch0) or remote (ch1).
        rms = float(np.sqrt((seg ** 2).mean(axis=0)).max())
        levels.append((rms, start, end))
    return levels or None


def peak_rms(group):
    """Loudest turn level in this group, or None when the WAV is unavailable.

    A near-zero value means the cluster has no audible speech at all, so the
    labeler can say so instead of leaving the user hunting for a voice.
    """
    levels = _turn_levels(group)
    return max(r for r, _, _ in levels) if levels else None


def ranked_turns(group, max_turns=40, min_seconds=0.5):
    """Turns worth embedding, loudest first, in time order.

    Selecting by length instead would embed a cluster's long silent stretches —
    a diarization artifact can easily be the longest turn present — and yield a
    voiceprint built from noise. Falls back to the longest turns when the audio
    cannot be read.
    """
    levels = _turn_levels(group)
    if levels is None:
        return sorted(sorted(group.turns, key=lambda t: t[0] - t[1])[:max_turns])

    floor = max(r for r, _, _ in levels) * 0.15
    usable = [(r, s, e) for r, s, e in levels
              if e - s >= min_seconds and r >= floor]
    if not usable:
        usable = sorted(levels, reverse=True)[:max_turns]
    return sorted((s, e) for _, s, e in sorted(usable, reverse=True)[:max_turns])


def loudest_turns(group, max_seconds=8.0, min_seconds=0.3):
    """Spans to play as this group's sample, in chronological order.

    Selecting by loudness rather than length is the point: a cluster's only long
    turn can be silent while its real speech sits in sub-second bursts, and
    picking the longest turn then plays nothing. Several short bursts are
    stitched together up to `max_seconds`, which also gives more voice to judge
    than one clip would.

    Falls back to the metadata pick when the audio cannot be read.
    """
    levels = _turn_levels(group)
    if levels is None:
        return [best_turn(group)]

    # Spare budget must never be filled with silence: a turn far quieter than
    # this cluster's own peak carries no voice to judge, however long it is.
    floor = max(r for r, _, _ in levels) * 0.15
    usable = [(r, s, e) for r, s, e in levels
              if e - s >= min_seconds and r >= floor]
    if not usable:
        _, start, end = max(levels)
        return [(start, end)]

    # Spread the picks across the cluster's span rather than taking the loudest
    # turns outright. A cluster that holds two people often has one dominating a
    # stretch of it, and a clip drawn only from the loudest turns can come
    # entirely from that speaker — so the second voice is never heard and the
    # cluster gets labeled as one person.
    first = min(s for _, s, _ in usable)
    last = max(e for _, _, e in usable)
    span = max(last - first, 1e-6)
    buckets = max(1, int(max_seconds // max(min_seconds * 2, 1.0)))

    by_bucket = {}
    for rms, start, end in usable:
        idx = min(buckets - 1, int((start - first) / span * buckets))
        by_bucket.setdefault(idx, []).append((rms, start, end))

    chosen, total = [], 0.0
    # One pass taking each bucket's loudest turn, then fill any spare budget with
    # the next-loudest turns anywhere.
    ordered = [max(v) for _, v in sorted(by_bucket.items())]
    rest = sorted((t for v in by_bucket.values() for t in v), reverse=True)
    for rms, start, end in ordered + rest:
        if (start, end) in chosen:
            continue
        length = end - start
        if total + length > max_seconds:
            continue
        chosen.append((start, end))
        total += length
        if total >= max_seconds:
            break
    return sorted(chosen)
