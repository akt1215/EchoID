"""Build evaluation trials from discovered clusters plus human ground truth.

Positives (same person, different meetings) are the scarce, expensive side:
`db::`-prefixed clusters are barred from them because such a cluster was named
by the voiceprint DB that was then updated from it, so a "match" there measures
leakage. Negatives are abundant and may draw on any labeled cluster.
"""

from itertools import combinations

from ai.identity_resolution import normalize_name
from evaluation.clusters import is_leaky
from evaluation.ground_truth import confirmed_name


def labeled(groups, gt):
    """[(group, confirmed name)] for every group a human confirmed."""
    out = []
    for g in groups:
        name = confirmed_name(gt, g.meeting, g.cluster)
        if name:
            out.append((g, name))
    return out


def positives(groups, gt):
    """Same person, different meetings, neither side leaky."""
    pairs = []
    for (a, na), (b, nb) in combinations(labeled(groups, gt), 2):
        if a.meeting == b.meeting:
            continue        # same session: optimistically biased, and a merge case
        if is_leaky(a.cluster) or is_leaky(b.cluster):
            continue        # circular — see module docstring
        if normalize_name(na) == normalize_name(nb):
            pairs.append((a, b))
    return pairs


def negatives(groups, gt):
    """Different people. Any labeled cluster qualifies, leaky ones included."""
    pairs = []
    for (a, na), (b, nb) in combinations(labeled(groups, gt), 2):
        if normalize_name(na) != normalize_name(nb):
            pairs.append((a, b))
    return pairs
