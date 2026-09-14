"""Per-cluster speaker identity resolution (model-free, numpy only).

The pipeline extracts one embedding per diarization turn; this module turns the
anonymous pyannote clusters into names WITHOUT loading any ML stack, so it is
unit-testable and cheap. It resolves **one identity per cluster** (never
per-turn), which is what stops a noisy screen reading from fragmenting a single
speaker into many rows:

  1. voiceprint DB match on the cluster centroid  -> source "db"  (pre-filled)
  2. validated visual name vote across the cluster -> source "visual"
  3. otherwise                                     -> "Unknown (<pyannote id>)"

Clusters that DB-match the *same* enrolled person collapse to one identity.
`looks_like_name` is the safety net: a reading is used only if it is name-shaped
or fuzzy-matches a known name, so OCR garbage ("0 a: ay") can never surface.
"""

import re
from difflib import SequenceMatcher

import numpy as np

_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z.'-]*$")


def normalize_name(name):
    """Canonical form for deciding whether two spellings are the same name:
    casefolded, dots dropped, whitespace collapsed. So "Riley Stone",
    "Riley Stone", and "r graham  barr" all normalize alike. Used to match a
    reading against the local user and to fold a re-spelled name into an existing
    DB key — the original display spelling is always preserved elsewhere."""
    return re.sub(r"[.\s]+", " ", (name or "").strip().lower()).strip()


def looks_like_name(text, known=()):
    """True if `text` plausibly is a person's display name.

    Accept when it fuzzy-matches (ratio >= 0.85) a known name, else require it be
    name-shaped: 1-4 tokens, each a letter-led word (letters/./'/-), with at
    least half the tokens capitalized. Rejects digits, symbols, and lowercase
    OCR fragments.
    """
    s = (text or "").strip()
    if not (2 <= len(s) <= 40):
        return False

    for k in known:
        if SequenceMatcher(None, s.lower(), str(k).lower()).ratio() >= 0.85:
            return True

    tokens = s.split()
    if not tokens or len(tokens) > 4:
        return False
    if any(not _TOKEN_RE.match(t) for t in tokens):
        return False
    capitalized = sum(1 for t in tokens if t[0].isupper())
    return capitalized * 2 >= len(tokens)


def _group_indices(segments):
    """{pyannote id: [segment indices]} in first-seen order."""
    groups = {}
    for i, s in enumerate(segments):
        groups.setdefault(s["speaker"], []).append(i)
    return groups


def _centroid(vecs, mean=None, durations=None, min_seconds=0.0):
    """Unit-normalized cluster centroid.

    When `mean` (the persisted global embedding mean) is given, it is subtracted
    before normalizing so the cosine comparison happens in the centered space
    where ECAPA embeddings actually separate speakers. `mean=None` leaves the raw
    behavior untouched, which is correct for models that need no centering.

    When `durations` is given, turns are weighted by length and anything under
    `min_seconds` is dropped. A half-second turn yields a poor embedding but
    counts as much as a ten-second one in a plain mean; weighting measurably
    carries speaker separation rather than merely refining it. A cluster whose
    turns are all short still gets a centroid — dropping it entirely would lose
    the speaker, and the caller's minimum-speech gate is what handles that case.
    """
    pairs = [(np.asarray(v, dtype=float),
              float(durations[i]) if durations is not None else 1.0)
             for i, v in enumerate(vecs)]
    pairs = [(a, d) for a, d in pairs if a.size]
    if not pairs:
        return None

    if durations is not None:
        long = [(a, d) for a, d in pairs if d >= min_seconds]
        weights = [d for _, d in (long or pairs)]
        arrs = [a for a, _ in (long or pairs)]
        if sum(weights) <= 0:
            weights = None
        m = np.average(arrs, axis=0, weights=weights)
    else:
        m = np.mean([a for a, _ in pairs], axis=0)

    if mean is not None:
        m = m - np.asarray(mean, dtype=float)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m


def _usable_speech(segments, idxs, min_turn_seconds):
    """Seconds of speech in a cluster from turns long enough to embed well."""
    return sum(segments[i]["end"] - segments[i]["start"] for i in idxs
               if segments[i]["end"] - segments[i]["start"] >= min_turn_seconds)


def _cos(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _db_match(centroid, db, threshold, margin=0.0, exclude=()):
    """Best voiceprint for `centroid` as (name, score, margin).

    `margin` is the gap to the runner-up. A near-tie means the embedding cannot
    actually tell the two apart, so requiring a gap turns a coin-flip into an
    Unknown the user can resolve, instead of a confident wrong name that also
    poisons the winner's voiceprint through the enrollment EMA.
    Returns (None, score, margin) when either bar is missed.

    `exclude` drops keys from consideration entirely. The local user's print is
    enrolled from mixed recordings, where their voice is in the embedded track;
    in a dual-channel recording their voice is on ch0 and never embedded, so any
    ch1 match against that print is echo or speakerphone bleed naming a remote
    speaker. Excluding rather than thresholding makes that structurally
    impossible instead of merely unlikely.
    """
    skip = {str(e) for e in exclude}
    candidates = {k: v for k, v in (db or {}).items() if k not in skip}
    if centroid is None or not candidates:
        return None, -1.0, 0.0
    scored = sorted(((_cos(centroid, np.asarray(emb, dtype=float)), name)
                     for name, emb in candidates.items()), reverse=True)
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else -1.0
    gap = best_score - runner_up
    if best_score < threshold or gap < margin:
        return None, best_score, gap
    return best, best_score, gap


def merge_clusters(segments, embeddings, mean=None, threshold=1.01, min_dur=3.0):
    """Merge over-clustered pyannote labels by centroid cosine similarity.

    PyAnnote sometimes splits one speaker into several ids; comparing per-label
    centroids (built from the shared per-turn embeddings, preferring turns
    >= `min_dur`) and merging above `threshold` collapses them before naming.
    Centroids are compared in the mean-centered space (`mean`) so distinct
    speakers are not fused by the raw ECAPA common-mean inflation.
    Returns copies of `segments` with only the `speaker` label rewritten.
    """
    groups = _group_indices(segments)
    if len(groups) < 2:
        return [dict(s) for s in segments]

    cents = {}
    for lbl, idxs in groups.items():
        long = [embeddings[i] for i in idxs
                if (segments[i]["end"] - segments[i]["start"]) >= min_dur]
        c = _centroid(long, mean) if long else _centroid([embeddings[i] for i in idxs], mean)
        if c is not None:
            cents[lbl] = c
    if len(cents) < 2:
        return [dict(s) for s in segments]

    labels = list(cents.keys())
    # Mutual-best: merge A and B only when each is the other's nearest neighbour.
    # A one-sided "above threshold" test chains A-B and B-C into one speaker even
    # when A and C are unrelated, which fuses distinct people — the one outcome
    # this pipeline must never produce. Measured on a real meeting, two pairs of
    # known-distinct speakers sat at 0.826 and 0.781, above the 0.72 threshold.
    nearest = {}
    for a in labels:
        best, best_score = None, -2.0
        for b in labels:
            if a == b:
                continue
            score = _cos(cents[a], cents[b])
            if score > best_score:
                best, best_score = b, score
        nearest[a] = (best, best_score)

    merge_map = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if nearest[a][0] != b or nearest[b][0] != a:
                continue
            if nearest[a][1] > threshold:
                merge_map[b] = a

    def root(x):
        while x in merge_map:
            x = merge_map[x]
        return x

    out = []
    for s in segments:
        s2 = dict(s)
        s2["speaker"] = root(s2["speaker"])
        out.append(s2)
    return out


def cluster_db_names(segments, embeddings, db, mean=None, db_threshold=1.01,
                     db_margin=0.0, exclude=()):
    """{pyannote id: matched DB name or None} from each cluster's centroid. The
    caller uses the Nones to decide which clusters still need the VLM name-read.
    Voiceprints are stored in the centered space, so the query centroid is
    centered by `mean` to compare in the same space."""
    out = {}
    for pid, idxs in _group_indices(segments).items():
        centroid = _centroid([embeddings[i] for i in idxs], mean)
        name, _, _ = _db_match(centroid, db, db_threshold, db_margin,
                               exclude=exclude)
        out[pid] = name
    return out


def resolve_clusters(segments, embeddings, db, cluster_visual_names=None,
                     mean=None, db_threshold=1.01, db_names=None,
                     db_margin=0.0, collapse_similarity=1.01,
                     min_speech=0.0, min_turn_seconds=0.5, exclude=()):
    """Resolve one identity per diarization cluster.

    `segments`             : {start, end, speaker(=pyannote id)} (post-merge).
    `embeddings`           : list aligned to `segments`; each a vector or empty.
    `db`                   : {name: vector} persistent voiceprints (read-only,
                             stored in the mean-centered space).
    `cluster_visual_names` : {pyannote id: name} the VLM read for that cluster.
    `mean`                 : persisted global embedding mean; centers the query
                             centroid to compare in the DB's centered space.
    `db_names`             : optional precomputed {pyannote id: db name or None}
                             (from `cluster_db_names`) to avoid recomputing the
                             centroid match; computed here when not supplied.
    `db_margin`            : required gap between the best and second-best DB
                             score; a near-tie resolves to Unknown.
    `collapse_similarity`  : two clusters may share one DB identity only if they
                             are at least this similar TO EACH OTHER. The default
                             1.01 exceeds any cosine, so nothing collapses unless
                             a caller opts in — fusing two people into one
                             identity is the worst outcome and is unrecoverable,
                             while an over-split is one merge click in review.
    `exclude`              : DB keys never offered as a match. Dual-channel
                             callers pass the local user's keys — their voice is
                             on ch0 and never embedded, so a ch1 match is bleed.

    Tiers per cluster: DB voiceprint match (pre-fill) → validated visual name →
    Unknown. Clusters matching the same DB person collapse to one identity.
    Returns {start, end, speaker(=name), source, cluster}, where `cluster` is the
    stable id the review layer groups by.
    """
    cluster_visual_names = cluster_visual_names or {}
    known = set(db.keys())

    groups = _group_indices(segments)
    centroids = {
        pid: _centroid([embeddings[i] for i in idxs], mean,
                       durations=[segments[i]["end"] - segments[i]["start"]
                                  for i in idxs],
                       min_seconds=min_turn_seconds)
        for pid, idxs in groups.items()}

    # A cluster with very little speech cannot be identified: its embedding is
    # unreliable enough to both miss its own speaker and land in impostor range,
    # so naming it is a guess. Leave it Unknown for the user to resolve.
    too_short = {pid for pid, idxs in groups.items()
                 if min_speech > 0
                 and _usable_speech(segments, idxs, min_turn_seconds) < min_speech}

    # Score every cluster against the DB first, so clusters competing for the
    # same identity can be compared before any of them claims it.
    matches = {}
    for pid in groups:
        if pid in too_short:
            matches[pid] = (None, -1.0)
            continue
        if db_names is not None:
            # Precomputed names carry no score, but rival clusters competing for
            # one identity must be ranked by how well they actually match it —
            # otherwise the winner below is whichever happened to come first.
            name = db_names.get(pid)
            score = -1.0
            if name and name in db and centroids[pid] is not None:
                score = _cos(centroids[pid], np.asarray(db[name], dtype=float))
        else:
            name, score, _ = _db_match(centroids[pid], db, db_threshold, db_margin,
                                       exclude=exclude)
        matches[pid] = (name, score)

    # A DB name may be claimed by only one cluster unless the rival clusters are
    # genuinely similar to each other; otherwise the loser falls back to Unknown.
    # Without this, two clusters that merely both score above the threshold get
    # the same identity even when they are nothing like one another.
    claimed = {}
    for pid, (name, score) in matches.items():
        if name and score > claimed.get(name, (None, -2.0))[1]:
            claimed[name] = (pid, score)
    for pid, (name, _) in list(matches.items()):
        if not name:
            continue
        winner = claimed[name][0]
        if pid == winner:
            continue
        a, b = centroids[pid], centroids[winner]
        if a is None or b is None or _cos(a, b) < collapse_similarity:
            matches[pid] = (None, -1.0)

    resolution = {}
    for pid in groups:
        db_name = matches[pid][0]
        vis = cluster_visual_names.get(pid)
        if db_name:
            resolution[pid] = (db_name, "db", f"db::{db_name}")
        elif vis and looks_like_name(vis, known):
            resolution[pid] = (vis, "visual", pid)
        else:
            resolution[pid] = (f"Unknown ({pid})", "unknown", pid)

    out = []
    for s in segments:
        name, source, cluster = resolution[s["speaker"]]
        out.append({"start": s["start"], "end": s["end"],
                    "speaker": name, "source": source, "cluster": cluster})
    return out
