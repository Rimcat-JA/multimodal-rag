"""Compare two retrieval methods question by question.

Both methods answered the same questions, so the comparison is paired: take
the per-question difference first, then resample questions. Questions that
are hard for every method cancel out, which gives a much tighter interval
than comparing two averages from 40 questions.
"""

import numpy as np


def reciprocal_rank(rank: int | None) -> float:
    """1 / rank, or 0 for a question the method never makes answerable (as in metrics.mean_reciprocal_rank)."""
    return 0.0 if rank is None else 1.0 / rank


def answerable_within(rank: int | None, k: int) -> float:
    """1.0 if the question is answerable within the top k, else 0.0.

    The rank is metrics.answerable_rank (where every gold group is first
    covered), so "rank <= k" is the same test as metrics.answerable_at_k.
    """
    return float(rank is not None and rank <= k)


def paired_bootstrap(base: list[float], other: list[float], n_resamples: int = 10_000,
                     seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of (other - base) over questions, and a percentile bootstrap interval for it.

    Returns (mean difference, lower bound, upper bound). The seed is fixed so the
    interval is the same on every run.
    """
    if len(base) != len(other) or not base:
        raise ValueError("base and other must be non-empty and the same length (one value per question)")
    diffs = np.asarray(other, dtype=float) - np.asarray(base, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = rng.choice(diffs, size=(n_resamples, len(diffs)), replace=True).mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(resampled, [tail, 1 - tail])
    return float(diffs.mean()), float(low), float(high)
