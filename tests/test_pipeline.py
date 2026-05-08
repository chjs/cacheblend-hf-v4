"""Phase 4 — Pipelining & Prefix Cache correctness tests.

Tolerance categories (frozen):
  - 4.1 selective_pipelined ≡ selective : RECOMPUTE_PATH (max_diff < 1e-3)
  - 4.2 prefix_cache vs full_recompute  : MIXED_SHAPE (argmax exact + max_diff < 5e-2)
  - 4.3 LoadingController monotone      : CPU-only, no tolerance check
"""
from __future__ import annotations

import pytest
import torch


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

DOCS = [
    "Paris is the capital of France and a major European city.",
    "The Eiffel Tower was completed in 1889 for the World's Fair.",
    "French cuisine is known for cheese, bread, and pastries.",
]


@pytest.fixture(scope="module")
def lw_model():
    from cacheblend import LayerwiseModel
    m = LayerwiseModel(MODEL_NAME, dtype="float16")
    yield m
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture(scope="module")
def chunks_and_store(lw_model):
    from cacheblend.chunker import chunk_texts
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv

    chunks = chunk_texts(lw_model.tokenizer, DOCS)
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw_model, c)
        store.put(c.chunk_id, K, V)
    return chunks, store


@pytest.mark.requires_model
@pytest.mark.gpu
def test_pipelined_eq_unpipelined(lw_model, chunks_and_store):
    """selective_pipelined logits ≡ selective logits within RECOMPUTE_PATH (max_diff < 1e-3)."""
    from cacheblend import Tolerance, assert_logits_close
    from cacheblend.fusor import fuse_selective, fuse_selective_pipelined

    chunks, store = chunks_and_store
    sel_logits = fuse_selective(lw_model, chunks, store, recompute_ratio=0.15)
    pipe_logits = fuse_selective_pipelined(
        lw_model, chunks, store, recompute_ratio=0.15, prefetch=True,
    )

    result = assert_logits_close(
        actual=pipe_logits, expected=sel_logits,
        category=Tolerance.RECOMPUTE_PATH, name="logits",
    )
    print(f"\n[4.1] {result.detail}")


@pytest.mark.requires_model
@pytest.mark.gpu
def test_prefix_cache_eq_full_recompute(lw_model, chunks_and_store):
    """prefix_cache argmax-equivalent to full_recompute (MIXED_SHAPE)."""
    from cacheblend import Tolerance, assert_logits_close
    from cacheblend.fusor import fuse_prefix_cache, fuse_full_recompute

    chunks, store = chunks_and_store
    pc_logits = fuse_prefix_cache(lw_model, chunks, store)
    truth = fuse_full_recompute(lw_model, chunks)

    result = assert_logits_close(
        actual=pc_logits, expected=truth,
        category=Tolerance.MIXED_SHAPE, name="logits",
    )
    print(f"\n[4.2] {result.detail}")


@pytest.mark.parametrize("dummy", [0])
def test_loading_controller_monotone(dummy):
    """RAM ≤ NVMe ≤ SATA_SSD ≤ SLOW_DISK ratios for fixed base_ratio.

    CPU-only, no GPU required.
    """
    from cacheblend.controller import LoadingController, StorageProfile

    lc = LoadingController()
    base = 0.15
    profiles = [
        StorageProfile.RAM,
        StorageProfile.NVME,
        StorageProfile.SATA_SSD,
        StorageProfile.SLOW_DISK,
    ]
    decisions = [lc.decide_recompute_ratio(p, base) for p in profiles]
    ratios = [d.recompute_ratio for d in decisions]

    print("\n[4.3] LoadingController.decide_recompute_ratio at base=0.15:")
    for d in decisions:
        print(f"     {d.detail}")

    # Strict monotone non-decreasing.
    for i in range(1, len(ratios)):
        assert ratios[i] >= ratios[i - 1], (
            f"monotone violated at {profiles[i].name}: "
            f"{profiles[i-1].name}={ratios[i-1]:.3f} > {profiles[i].name}={ratios[i]:.3f}"
        )

    # Strict monotone increasing (sanity — the multipliers are distinct).
    distinct = len(set(ratios))
    assert distinct == 4 or ratios[-1] > ratios[0], (
        "expected strict ratio increase from RAM to SLOW_DISK"
    )
