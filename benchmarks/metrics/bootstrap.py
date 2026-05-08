"""Paired bootstrap CI helper [L28].

`paired_bootstrap_ci(scores_a, scores_b)` returns a confidence interval on
the difference of paired means: mean(a[i] - b[i]) over n samples.

Used by Phase 6/7 evaluation gates: F1(cacheblend) > F1(baseline) iff ci_low > 0.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def paired_bootstrap_ci(
    scores_a: Iterable[float],
    scores_b: Iterable[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (ci_low, ci_high) for mean(a[i] - b[i]).

    `scores_a` and `scores_b` must have the same length n (paired by index).
    Reproducible with fixed seed.

    Method: resample with replacement n indices, compute mean(a[idx] - b[idx]),
    repeat n_bootstrap times, take the (alpha/2, 1-alpha/2) percentiles.
    """
    a = np.asarray(list(scores_a), dtype=np.float64)
    b = np.asarray(list(scores_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}")
    if a.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got ndim={a.ndim}")
    n = a.shape[0]
    if n == 0:
        raise ValueError("empty input")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    diffs = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()

    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
