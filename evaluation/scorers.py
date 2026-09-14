"""Interchangeable ways to turn a cluster into a vector and compare two of them.

Each strategy is one candidate in the bake-off. Keeping them behind one interface
means the harness loops over strategies instead of forking the pipeline.
"""

import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Baseline:
    """Today's behavior: plain mean of the turn embeddings, globally centered."""

    def __init__(self, mean=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)

    def prepare(self, groups):
        """Fit anything the strategy needs from the corpus. No-op here."""

    def _weights(self, group):
        return np.ones(len(group.embeddings))

    def centroid(self, group):
        w = self._weights(group)
        centre = np.average(group.embeddings, axis=0, weights=w)
        if self.mean is not None:
            centre = centre - self.mean
        return _unit(centre)

    def score(self, a, b):
        return float(np.dot(_unit(a), _unit(b)))


class DurationWeighted(Baseline):
    """Weight each turn by its length, and ignore very short turns.

    A half-second turn yields a poor embedding but counts as much as a ten-second
    one in a plain mean; `min_segment_duration` currently admits 0.5s turns.
    """

    def __init__(self, mean=None, min_seconds=1.5):
        super().__init__(mean)
        self.min_seconds = min_seconds

    def _weights(self, group):
        durations = np.asarray([e - s for s, e in group.turns], dtype=float)
        keep = durations >= self.min_seconds
        if not keep.any():
            # Every turn is short: fall back to weighting by length rather than
            # discarding the cluster entirely.
            return durations if durations.sum() > 0 else np.ones(len(durations))
        return np.where(keep, durations, 0.0)


class ASNorm(DurationWeighted):
    """Adaptive score normalization against a cohort of other speakers.

    Global-mean centering removes one direction shared by everyone, but a
    per-session direction survives it (measured: different-speaker cosine +0.341
    within a meeting vs -0.123 across). AS-Norm instead rescales each score by
    how that vector scores against a cohort, so a vector that is close to
    everything stops earning credit for being close to one more thing.
    """

    def __init__(self, mean=None, min_seconds=1.5, top_k=10):
        super().__init__(mean, min_seconds)
        self.top_k = top_k
        self._cohort = None

    def prepare(self, groups):
        cohort = [self.centroid(g) for g in groups if len(g.embeddings)]
        self._cohort = np.stack(cohort) if cohort else None

    def _stats(self, v):
        sims = np.sort(self._cohort @ _unit(v))[::-1][: self.top_k]
        return float(sims.mean()), float(sims.std()) or 1.0

    def score(self, a, b):
        raw = super().score(a, b)
        if self._cohort is None or len(self._cohort) < 2:
            return raw
        mu_a, sd_a = self._stats(a)
        mu_b, sd_b = self._stats(b)
        return 0.5 * ((raw - mu_a) / sd_a + (raw - mu_b) / sd_b)


SCORERS = {
    "baseline": Baseline,
    "duration": DurationWeighted,
    "asnorm": ASNorm,
}
