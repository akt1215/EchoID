"""Scoring for speaker-identity resolution, in the user's priority order.

1. conflation  — several people fused into one identity. HARD CONSTRAINT, target 0.
2. false accept — a cluster given the wrong existing name.
3. coverage     — share of speech time correctly auto-named, maximized only
                  SUBJECT TO 1 and 2, so labeling everything Unknown cannot win.
"""

from collections import defaultdict

from ai.identity_resolution import normalize_name


def _truth(gt_names, key):
    name = gt_names.get(key)
    return normalize_name(name) if name else None


def conflations(assignments, gt_names):
    """[(assigned identity, {true names})] for identities covering >1 person."""
    by_identity = defaultdict(set)
    for key, assigned in assignments.items():
        if not assigned:
            continue                    # Unknown groups nothing together
        truth = _truth(gt_names, key)
        if truth:
            by_identity[assigned].add(truth)
    return [(ident, names) for ident, names in by_identity.items()
            if len(names) > 1]


def false_accepts(assignments, gt_names):
    """[(key, assigned, expected)] where a cluster got the wrong name."""
    out = []
    for key, assigned in assignments.items():
        if not assigned:
            continue                    # Unknown is the safe outcome, not an error
        truth = _truth(gt_names, key)
        if truth and normalize_name(assigned) != truth:
            out.append((key, assigned, gt_names[key]))
    return out


def coverage(assignments, gt_names, seconds):
    """Share of labeled speech time that was auto-named CORRECTLY."""
    total = sum(seconds.get(k, 0.0) for k in gt_names)
    if not total:
        return 0.0
    good = 0.0
    for key, assigned in assignments.items():
        truth = _truth(gt_names, key)
        if assigned and truth and normalize_name(assigned) == truth:
            good += seconds.get(key, 0.0)
    return good / total


def summarize(assignments, gt_names, seconds):
    return {
        "conflations": len(conflations(assignments, gt_names)),
        "false_accepts": len(false_accepts(assignments, gt_names)),
        "coverage": coverage(assignments, gt_names, seconds),
        "assigned": sum(1 for v in assignments.values() if v),
        "total": len(gt_names),
    }
