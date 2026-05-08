"""Phase 5 — Paired bootstrap CI helper unit tests [L28]."""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_known_distribution_a_dominates_b():
    """a[i] = b[i] + 0.10 ⇒ ci_low > 0 (a strictly larger by paired diff)."""
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    rng = np.random.default_rng(0)
    b = rng.uniform(0.0, 1.0, size=100)
    a = b + 0.10
    lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=1000, confidence=0.95)
    assert lo > 0.0, f"expected ci_low > 0, got [{lo:.4f}, {hi:.4f}]"
    assert math.isclose(lo, 0.10, abs_tol=1e-9) and math.isclose(hi, 0.10, abs_tol=1e-9), (
        f"paired diff is constant 0.10, ci should be exactly [0.10, 0.10], got [{lo:.6f}, {hi:.6f}]"
    )


def test_known_distribution_identical():
    """a == b ⇒ ci spans 0."""
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 1.0, size=50)
    b = a.copy()
    lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=500, confidence=0.95)
    assert lo == 0.0 and hi == 0.0, f"identical scores should give ci [0,0], got [{lo}, {hi}]"


def test_n1_edge():
    """n=1: only one pair; ci collapses to that single diff."""
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    lo, hi = paired_bootstrap_ci([0.5], [0.3], n_bootstrap=200, confidence=0.95)
    assert math.isclose(lo, 0.2, abs_tol=1e-9) and math.isclose(hi, 0.2, abs_tol=1e-9)


def test_small_n_bootstrap():
    """n_bootstrap=10 still works; just less precise."""
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    rng = np.random.default_rng(0)
    b = rng.uniform(0.0, 1.0, size=20)
    a = b + 0.05
    lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=10, confidence=0.95)
    # Constant-shift paired diff → CI degenerate at 0.05 regardless of n_bootstrap.
    assert math.isclose(lo, 0.05, abs_tol=1e-9) and math.isclose(hi, 0.05, abs_tol=1e-9)


def test_shape_mismatch_raises():
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    with pytest.raises(ValueError, match="shape mismatch"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_invalid_confidence():
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap_ci([1.0], [0.5], confidence=1.5)


def test_seed_reproducibility():
    """Same seed ⇒ same CI."""
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 1.0, size=50)
    b = rng.uniform(0.0, 1.0, size=50)
    lo1, hi1 = paired_bootstrap_ci(a, b, n_bootstrap=500, seed=42)
    lo2, hi2 = paired_bootstrap_ci(a, b, n_bootstrap=500, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
