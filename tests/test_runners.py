"""Phase 5 — Runner CPU stub-mode dispatch tests.

All 4 Runner subclasses must:
  1. Instantiate with model=None (no GPU).
  2. Accept prepare(system, docs, question).
  3. Return a GenerationResult-like (has .text, .ttft_seconds, .total_seconds, .n_generated_tokens).
"""
from __future__ import annotations

import pytest


def test_imports():
    from cacheblend.runners import (
        FullRecomputeRunner, FullReuseRunner, PrefixCacheRunner, CacheBlendRunner,
    )
    assert all(callable(c) for c in [
        FullRecomputeRunner, FullReuseRunner, PrefixCacheRunner, CacheBlendRunner,
    ])


@pytest.mark.parametrize("runner_name,kwargs", [
    ("FullRecomputeRunner", {}),
    ("FullReuseRunner", {}),
    ("PrefixCacheRunner", {}),
    ("CacheBlendRunner", {"recompute_ratio": 0.15, "check_layer": 1}),
])
def test_dispatch_with_stub_model(runner_name, kwargs):
    """Every runner must instantiate (model=None) and dispatch prepare/generate."""
    from cacheblend import runners

    cls = getattr(runners, runner_name)
    r = cls(**kwargs)
    assert r.model is None
    assert r.tokenizer is None

    r.prepare(
        system="You are a helpful assistant.",
        docs=["Document A.", "Document B."],
        question="What is the answer?",
    )
    result = r.generate(max_new_tokens=8)

    # GenerationResult-like (works whether harness is importable or not).
    assert hasattr(result, "text")
    assert hasattr(result, "ttft_seconds")
    assert hasattr(result, "total_seconds")
    assert hasattr(result, "n_generated_tokens")

    # Stub mode returns empty text + zero timings.
    assert result.text == ""
    assert result.ttft_seconds == 0.0
    assert result.total_seconds == 0.0
    assert result.n_generated_tokens == 0


def test_cacheblend_runner_carries_config():
    from cacheblend.runners import CacheBlendRunner
    r = CacheBlendRunner(recompute_ratio=0.20, check_layer=2)
    assert r.recompute_ratio == 0.20
    assert r.check_layer == 2
