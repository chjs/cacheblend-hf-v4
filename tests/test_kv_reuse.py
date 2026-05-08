"""Phase 2 — KV Storage & Full Reuse correctness tests.

Markers: requires_model and gpu (auto-skipped on no-CUDA / no-cache via conftest).
Tolerance categories (frozen at Phase 2 start, retroactive change forbidden [L05/L13/L16]):
  - 2.1 RoPE shift correctness    : IDENTICAL_PATH (max_diff = 0)
  - 2.2 full_recompute sanity     : IDENTICAL_PATH (vs HF baseline)
  - 2.3 full_reuse single-prefix  : MIXED_SHAPE (argmax exact + max_diff < 5e-2)
                                    NOTE: single-prefix path goes through boundary
                                    safe-shortcut → also IDENTICAL_PATH in practice.
  - 2.4 full_reuse multi-chunk    : NO bound (Phase 3 baseline measurement)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The CacheBlend algorithm reduces TTFT by"

# Multi-chunk test fixture: 3 short documents.
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


@pytest.mark.requires_model
@pytest.mark.gpu
def test_rope_shift_correctness(lw_model):
    """apply_rope_shift at positions 0..L-1 must match HF's RoPE applied via apply_rotary_pos_emb.

    Tolerance: IDENTICAL_PATH — max_diff = 0.
    Layer-0 only (head shape is layer-independent for Mistral; one layer is enough).
    """
    from cacheblend.rope import apply_rope_shift, _kproj_to_heads
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.chunker import Chunk
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

    enc = lw_model.tokenizer(DOCS[0], return_tensors="pt", add_special_tokens=False)
    token_ids = enc["input_ids"][0].tolist()
    L = len(token_ids)
    chunk = Chunk(text=DOCS[0], token_ids=token_ids, chunk_id="test_rope_chunk")

    K_per_layer, _V = precompute_chunk_kv(lw_model, chunk)
    K0_pre = K_per_layer[0]  # (1, L, hidden_kv)

    positions = torch.arange(L, device=lw_model.device).unsqueeze(0)  # (1, L)

    # Our path: apply_rope_shift via the helper.
    K0_shifted = apply_rope_shift(K0_pre, positions, lw_model)

    # Reference path: replicate HF's apply_rotary_pos_emb directly on layer-0 heads.
    inner = lw_model._inner
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim

    K0_heads = _kproj_to_heads(K0_pre, num_kv_heads, head_dim)
    cos, sin = inner.rotary_emb(K0_heads, positions)
    dummy_q = torch.zeros_like(K0_heads)
    _q, K0_ref = apply_rotary_pos_emb(dummy_q, K0_heads, cos, sin)
    K0_ref_flat = K0_ref.transpose(1, 2).reshape(1, L, num_kv_heads * head_dim)

    max_diff = (K0_shifted.float() - K0_ref_flat.float()).abs().max().item()
    print(f"\n[2.1] layer-0 apply_rope_shift max_diff = {max_diff:.3e}")
    assert max_diff == 0.0, f"IDENTICAL_PATH violated: max_diff = {max_diff:.3e}"


@pytest.mark.requires_model
@pytest.mark.gpu
def test_full_recompute_sanity(lw_model):
    """fuse_full_recompute(single chunk) must equal HF's standard forward.

    Tolerance: IDENTICAL_PATH (max_diff = 0). This is just LayerwiseModel
    wrapped with chunk plumbing — both paths run the same forward.
    """
    from cacheblend import Tolerance, assert_logits_close
    from cacheblend.chunker import chunk_texts
    from cacheblend.fusor import fuse_full_recompute

    chunks = chunk_texts(lw_model.tokenizer, [PROMPT])

    fused_logits = fuse_full_recompute(lw_model, chunks)

    # Reference: HF model forward over identical input_ids.
    enc = lw_model.tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False).to(lw_model.device)
    with torch.inference_mode():
        std_logits = lw_model.model(input_ids=enc["input_ids"], use_cache=False).logits

    result = assert_logits_close(
        actual=fused_logits, expected=std_logits,
        category=Tolerance.IDENTICAL_PATH, name="logits",
    )
    print(f"\n[2.2] {result.detail}")


@pytest.mark.requires_model
@pytest.mark.gpu
def test_full_reuse_single_prefix(lw_model):
    """fuse_full_reuse(single chunk) goes through boundary safe-shortcut [L13]
    → identical path to fuse_full_recompute → max_diff = 0.

    Tolerance: MIXED_SHAPE bound (argmax exact + max_diff < 5e-2) is the
    declared category, but single-prefix path achieves IDENTICAL_PATH (max=0).
    """
    from cacheblend import Tolerance, assert_logits_close
    from cacheblend.chunker import chunk_texts
    from cacheblend.fusor import fuse_full_reuse, fuse_full_recompute
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.kv_store import KVStore

    chunks = chunk_texts(lw_model.tokenizer, [PROMPT])
    store = KVStore()
    K, V = precompute_chunk_kv(lw_model, chunks[0])
    store.put(chunks[0].chunk_id, K, V)

    reuse_logits = fuse_full_reuse(lw_model, chunks, store)
    recompute_logits = fuse_full_recompute(lw_model, chunks)

    result = assert_logits_close(
        actual=reuse_logits, expected=recompute_logits,
        category=Tolerance.MIXED_SHAPE, name="logits",
    )
    print(f"\n[2.3] {result.detail}")


@pytest.mark.requires_model
@pytest.mark.gpu
def test_full_reuse_multi_chunk_divergence(lw_model):
    """Measure max_diff between full_reuse and full_recompute on a 3-chunk fused input.

    No tolerance assertion — Phase 3 baseline. Records per-layer divergence
    statistics. Phase 3 selective recompute will reduce these by recomputing
    high-deviation tokens.
    """
    from cacheblend.chunker import chunk_texts
    from cacheblend.fusor import fuse_full_reuse, fuse_full_recompute
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.kv_store import KVStore

    chunks = chunk_texts(lw_model.tokenizer, DOCS)
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw_model, c)
        store.put(c.chunk_id, K, V)

    reuse_logits = fuse_full_reuse(lw_model, chunks, store)
    recompute_logits = fuse_full_recompute(lw_model, chunks)

    diff = (reuse_logits.float() - recompute_logits.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    actual_argmax = reuse_logits.argmax(dim=-1)
    expected_argmax = recompute_logits.argmax(dim=-1)
    argmax_match = float((actual_argmax == expected_argmax).float().mean().item())

    # Per-chunk last-token logit divergence (most relevant for downstream generation).
    from cacheblend.chunker import chunk_offsets
    offs = chunk_offsets(chunks)
    per_chunk_last = []
    for (s, e) in offs:
        last = e - 1
        d = (reuse_logits[0, last].float() - recompute_logits[0, last].float()).abs().max().item()
        per_chunk_last.append(d)

    stats = {
        "n_chunks": len(chunks),
        "total_seq": int(reuse_logits.shape[1]),
        "max_diff_overall": max_diff,
        "mean_diff_overall": mean_diff,
        "argmax_match_ratio": argmax_match,
        "per_chunk_last_token_max_diff": per_chunk_last,
    }
    print(f"\n[2.4] multi-chunk divergence: {json.dumps(stats, indent=2)}")

    # Persist for Phase 3 baseline reference.
    out_path = Path("/workspace/cacheblend-hf-v4/reports/phase-2-attachments/multi_chunk_divergence.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))

    # Sanity only: divergence should be > 0 (positions differ between standalone
    # chunk prefill and fused prefill).
    assert max_diff > 0.0, "multi-chunk full_reuse identical to full_recompute — sanity check failed"
