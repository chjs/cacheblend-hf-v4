"""Phase 3 — Selective KV Recompute correctness tests.

Tolerance categories (frozen at Phase 3 start):
  - 3.1 ratio=0 vs full_reuse              : IDENTICAL_PATH (max_diff = 0)
  - 3.2 ratio=1 vs full_recompute          : IDENTICAL_PATH
  - 3.3 ratio=0.15 multi-chunk reduces L2 : ≥15% reduction vs full_reuse baseline
  - 3.4 mask is standard causal           : Q×Q sub-block lower-triangular [L15]
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# 3-chunk fixture (same docs as Phase 2 §2.4 baseline measurement).
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
    """Reusable: 3 chunks + KVStore populated with their pre-RoPE K + V."""
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
def test_ratio_0_eq_full_reuse(lw_model, chunks_and_store):
    """ratio=0 boundary safe-shortcut [L13] → fuse_full_reuse path → IDENTICAL_PATH."""
    from cacheblend import Tolerance, assert_logits_close
    from cacheblend.fusor import fuse_selective, fuse_full_reuse

    chunks, store = chunks_and_store
    sel_logits = fuse_selective(lw_model, chunks, store, recompute_ratio=0.0)
    ref_logits = fuse_full_reuse(lw_model, chunks, store)

    result = assert_logits_close(
        actual=sel_logits, expected=ref_logits,
        category=Tolerance.IDENTICAL_PATH, name="logits",
    )
    print(f"\n[3.1] {result.detail}")


@pytest.mark.requires_model
@pytest.mark.gpu
def test_ratio_1_eq_full_recompute(lw_model, chunks_and_store):
    """ratio>=1 boundary safe-shortcut [L13] → fuse_full_recompute path → IDENTICAL_PATH."""
    from cacheblend import Tolerance, assert_logits_close
    from cacheblend.fusor import fuse_selective, fuse_full_recompute

    chunks, store = chunks_and_store
    sel_logits = fuse_selective(lw_model, chunks, store, recompute_ratio=1.0)
    ref_logits = fuse_full_recompute(lw_model, chunks)

    result = assert_logits_close(
        actual=sel_logits, expected=ref_logits,
        category=Tolerance.IDENTICAL_PATH, name="logits",
    )
    print(f"\n[3.2] {result.detail}")


@pytest.mark.requires_model
@pytest.mark.gpu
def test_selective_reduces_divergence(lw_model, chunks_and_store):
    """Selective ratio=0.15 must reduce L2 divergence vs full_reuse baseline at HKVD positions by >=15%.

    Sparse-forward fuse_selective only produces valid logits at top_indices (HKVD
    positions); other positions are zero-scattered. So we measure at top_indices:
      full_reuse_L2  = ||reuse[hkvd] - truth[hkvd]||_2
      selective_L2   = ||sel[hkvd]   - truth[hkvd]||_2
    Pass if selective_L2 <= full_reuse_L2 * 0.85.
    """
    from cacheblend.fusor import fuse_selective, fuse_full_reuse, fuse_full_recompute

    chunks, store = chunks_and_store

    truth = fuse_full_recompute(lw_model, chunks)
    reuse_logits = fuse_full_reuse(lw_model, chunks, store)
    sel_logits, hkvd = fuse_selective(
        lw_model, chunks, store, recompute_ratio=0.15, check_layer=1,
        return_hkvd_indices=True,
    )

    truth_at_hkvd = truth[:, hkvd, :].float()
    reuse_at_hkvd = reuse_logits[:, hkvd, :].float()
    sel_at_hkvd = sel_logits[:, hkvd, :].float()
    full_reuse_l2 = (reuse_at_hkvd - truth_at_hkvd).pow(2).mean().sqrt().item()
    selective_l2 = (sel_at_hkvd - truth_at_hkvd).pow(2).mean().sqrt().item()
    reduction = 1.0 - (selective_l2 / max(full_reuse_l2, 1e-12))

    n_hkvd = int(hkvd.numel()) if hkvd is not None else -1
    total_seq = int(truth.shape[1])
    print(
        f"\n[3.3] full_reuse_L2={full_reuse_l2:.4e}, selective_L2={selective_l2:.4e}, "
        f"reduction={reduction*100:.2f}% (target ≥15%); "
        f"hkvd={n_hkvd}/{total_seq}"
    )

    # Persist for downstream checks (long-chunk sweep, Phase 3 baseline tracking).
    out_path = Path("/workspace/cacheblend-hf-v4/reports/phase-3-attachments/selective_15.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ratio": 0.15,
        "check_layer": 1,
        "n_chunks": len(chunks),
        "total_seq": total_seq,
        "hkvd_count": n_hkvd,
        "full_reuse_l2": full_reuse_l2,
        "selective_l2": selective_l2,
        "reduction_ratio": reduction,
    }, indent=2))

    assert reduction >= 0.15, (
        f"selective ratio=0.15 reduced L2 by only {reduction*100:.2f}% "
        f"(target ≥15%); full_reuse_L2={full_reuse_l2:.4e}, selective_L2={selective_l2:.4e}"
    )


@pytest.mark.requires_model
@pytest.mark.gpu
def test_mask_is_standard_causal(lw_model, chunks_and_store):
    """Selective hooks must NOT corrupt the attention mask (Q×Q lower-triangular) [L15].

    We capture the mask passed into layer-0 attention and assert lower-triangular.
    """
    from cacheblend.fusor import fuse_selective

    chunks, store = chunks_and_store
    captured: dict = {}

    def hook(_module, args, _kwargs):
        # MistralAttention.forward(self, hidden_states, attention_mask=..., ...)
        # attention_mask comes in as kwarg "attention_mask" or positional arg index 1.
        am = _kwargs.get("attention_mask") if _kwargs else None
        if am is None and len(args) >= 2:
            am = args[1]
        if am is not None and "mask" not in captured:
            captured["mask"] = am.detach()

    handle = lw_model._inner.layers[0].self_attn.register_forward_pre_hook(
        hook, with_kwargs=True
    )
    try:
        _ = fuse_selective(
            lw_model, chunks, store, recompute_ratio=0.15, check_layer=1,
        )
    finally:
        handle.remove()

    assert "mask" in captured, "did not capture layer-0 attention_mask"
    mask = captured["mask"]
    print(f"\n[3.4] captured mask shape={tuple(mask.shape)}, dtype={mask.dtype}")

    # HF returns a 4D causal mask: (B, 1, Q, K). For prefill, K may be >= Q
    # because HF's _update_causal_mask pads K to a multiple / adds future-slot
    # columns. The Q×Q sub-block (mask[:, :, :Q, :Q]) MUST be lower-triangular
    # for selective attention to remain causally correct [L15].
    assert mask.ndim == 4, f"expected 4D causal mask, got {mask.shape}"
    B, H, Q, K = mask.shape
    assert K >= Q, f"prefill mask should have K >= Q, got Q={Q} K={K}"

    # Q×Q sub-block: True where token can attend (mask value ≈ 0).
    sub = mask[0, 0, :, :Q].float()
    allowed = sub.abs() < 1.0
    expected = torch.tril(torch.ones(Q, Q, dtype=torch.bool, device=mask.device))
    match = (allowed == expected).all().item()
    diff_count = (allowed != expected).sum().item()
    assert match, (
        f"Q×Q sub-block of mask not lower-triangular: differs at {diff_count} positions; "
        f"shape={tuple(mask.shape)}"
    )
    print(f"[3.4] mask is standard causal: Q×Q={Q}x{Q} sub-block lower-triangular ✓ "
          f"(K={K}, extra {K-Q} cols are future-cache slots)")
