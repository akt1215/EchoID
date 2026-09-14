"""Human-confirmed speaker identity per (meeting, diarization cluster).

The only trustworthy identity labels this project has. Sidecar `label` fields
come from the vision-LLM name reader that is itself under investigation, and
`db::`-prefixed cluster ids are circular (named by the voiceprint DB, which was
then updated from them). Evaluation therefore scores against this store alone.
"""

import json
import os
import re

from core import atomic_write

# "confirmed" = one person, name known. "multiple" = the cluster holds more than
# one person, which is itself a diarization conflation and must be recorded
# rather than skipped. "skip" = could not tell (unclear audio, too short).
VERDICTS = ("confirmed", "multiple", "skip")


def clean(name):
    """Collapse whitespace so one person is not stored under two spellings."""
    return re.sub(r"\s+", " ", (name or "").strip())


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save(path, data):
    atomic_write.write_text(path, json.dumps(data, indent=2, sort_keys=True))


def set_label(data, meeting, cluster, verdict, name=None):
    """Return `data` with one (meeting, cluster) entry set. Mutates and returns
    the same dict so callers can chain; a name is kept only for "confirmed"."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    cleaned = clean(name)
    if verdict == "confirmed" and not cleaned:
        raise ValueError("a confirmed verdict requires a non-empty name")
    data.setdefault(meeting, {})[cluster] = {
        "name": cleaned if verdict == "confirmed" else None,
        "verdict": verdict,
    }
    return data


def get(data, meeting, cluster):
    return data.get(meeting, {}).get(cluster)


def confirmed_name(data, meeting, cluster):
    """The confirmed name, or None if unlabeled/skipped/multi-person."""
    entry = get(data, meeting, cluster)
    if not entry or entry.get("verdict") != "confirmed":
        return None
    return entry.get("name")


def labeled_keys(data):
    return {(m, c) for m, clusters in data.items() for c in clusters}


def prune_stale(data, groups):
    """Drop labels for meetings whose diarization has changed.

    Returns `(pruned_copy, [meeting, ...])`. Cluster ids are assigned per
    diarization run, so re-running a meeting can drop an id and can reuse one
    for entirely different audio — a verdict recorded against the old cluster
    would then be applied to the new one silently. A label belongs to the run it
    was made against, so if any labeled cluster of a meeting has disappeared,
    every label for that meeting is dropped rather than partly trusted.

    A meeting with no live clusters at all is left alone: its recording may
    simply not be in this workspace, and absence is not evidence of change.
    Extra live clusters that were never labeled are unlabeled work, not change.
    """
    live = {}
    for g in groups:
        live.setdefault(g.meeting, set()).add(g.cluster)

    pruned, dropped = {}, []
    for meeting, clusters in data.items():
        present = live.get(meeting)
        if present is None or set(clusters) <= present:
            pruned[meeting] = dict(clusters)
        else:
            dropped.append(meeting)
            pruned[meeting] = {}
    return pruned, dropped
