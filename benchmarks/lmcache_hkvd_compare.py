"""Phase 3 acceptance 3.7 — Compare our `kv_deviation` ranking with LMCache's.

LMCache `external/LMCache/lmcache/v1/compute/blend/blender.py:89-91`:
    diff_k = torch.sum((k.to(fp32) - old_k.to(fp32)) ** 2, dim=[1])
where `k` is post-RoPE fresh K (RoPE applied at fused-sequence positions) and
`old_k` is post-RoPE cached K at chunk-local positions (no shift in LMCache).

Our v4 design (paper §4-faithful) stores pre-RoPE K and shifts at retrieval, so
our deviation runs in the pre-RoPE domain (or post-RoPE-shifted domain — see
note below).

KEY OBSERVATION
RoPE is an orthogonal rotation per position. For the same per-token rotation
matrix R_pos applied to both K_new and K_old:
    ‖R_pos · K_new[pos] − R_pos · K_old[pos]‖² = ‖K_new[pos] − K_old[pos]‖²
So the per-token squared-L2 deviation is **invariant** under RoPE shift to a
common position. That is:

    pre-RoPE deviation == post-RoPE-shifted-to-global deviation
                       == post-RoPE-at-chunk-local deviation (LMCache style)
                          IF we project K_new onto the same chunk-local frame.

Within our 3-chunk fused sequence, the per-token deviation **rankings** in our
pre-RoPE space and LMCache's post-RoPE space (when both K_new and K_old are
rotated to the same per-token positions, which is the LMCache code path with
shared `metadata.positions`) should match exactly. We verify this by computing
both rankings and reporting Spearman corr.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb


DOCS = [
    "Paris is the capital of France and a major European city.",
    "The Eiffel Tower was completed in 1889 for the World's Fair.",
    "French cuisine is known for cheese, bread, and pastries.",
]


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Plain Spearman correlation (ranks → Pearson)."""
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.pow(2).sum().sqrt() * ry.pow(2).sum().sqrt()).item()
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum().item() / denom)


def main():
    from cacheblend import LayerwiseModel
    from cacheblend.chunker import chunk_texts, fused_input_ids, chunk_offsets
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.hkvd import kv_deviation
    from cacheblend.rope import _kproj_to_heads, _heads_to_kproj
    from cacheblend.fusor import fuse_full_recompute  # for fresh K capture via Phase 1 hook

    out_dir = Path("/workspace/cacheblend-hf-v4/reports/phase-3-attachments")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Mistral-7B...", flush=True)
    t0 = time.time()
    model = LayerwiseModel("mistralai/Mistral-7B-Instruct-v0.2", dtype="float16")
    print(f"  loaded in {time.time()-t0:.1f}s")

    # 3-chunk fused setup (same as Phase 2 §2.4 / Phase 3 tests).
    chunks = chunk_texts(model.tokenizer, DOCS)
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(model, c)
        store.put(c.chunk_id, K, V)

    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]
    inner = model._inner
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim
    hidden_kv = num_kv_heads * head_dim
    device = model.device

    # Build concat stored pre-RoPE K at check_layer=1.
    check_layer = 1
    K_stored_pre_check = torch.zeros((1, total_seq, hidden_kv), dtype=model.dtype, device=device)
    for c, (s, e) in zip(chunks, offsets):
        K_stored_pre_check[:, s:e, :] = store.get(c.chunk_id)["K"][check_layer]

    # Run a fresh forward to capture fresh pre-RoPE K at check_layer.
    _ = fuse_full_recompute(model, chunks)
    K_fresh_pre_check = model.get_pre_rope_k(check_layer).clone()

    # ── Our deviation (pre-RoPE space) — Phase 3 hkvd.py ───────────────────
    dev_v4 = kv_deviation(K_fresh_pre_check, K_stored_pre_check)

    # ── LMCache-style deviation (post-RoPE at fused/blended positions) ────
    # Replicates the operation at blender.py:86-91 directly:
    # 1) RoPE both K_new and K_old at fused sequence positions (0..total_seq-1).
    # 2) Squared L2 sum over feature axis per token.
    positions = torch.arange(total_seq, device=device).unsqueeze(0)
    K_fresh_heads = _kproj_to_heads(K_fresh_pre_check, num_kv_heads, head_dim)
    K_stored_heads = _kproj_to_heads(K_stored_pre_check, num_kv_heads, head_dim)
    cos, sin = inner.rotary_emb(K_fresh_heads, positions)
    dummy_q = torch.zeros_like(K_fresh_heads)
    _q1, K_fresh_post = apply_rotary_pos_emb(dummy_q, K_fresh_heads, cos, sin)
    _q2, K_stored_post = apply_rotary_pos_emb(dummy_q, K_stored_heads, cos, sin)
    K_fresh_post_flat = _heads_to_kproj(K_fresh_post)
    K_stored_post_flat = _heads_to_kproj(K_stored_post)
    dev_lmcache = kv_deviation(K_fresh_post_flat, K_stored_post_flat)

    # ── Compare ────────────────────────────────────────────────────────────
    spearman = _spearman(dev_v4.cpu(), dev_lmcache.cpu())
    pearson = float(torch.corrcoef(torch.stack([
        dev_v4.cpu().double(), dev_lmcache.cpu().double()
    ]))[0, 1].item())

    abs_diff = (dev_v4 - dev_lmcache).abs()
    rel_max = (abs_diff / (dev_lmcache.abs() + 1e-9)).max().item()

    # Top-K overlap at ratio=0.15
    from cacheblend.hkvd import select_top_k
    top_v4 = set(select_top_k(dev_v4, 0.15).cpu().tolist())
    top_lm = set(select_top_k(dev_lmcache, 0.15).cpu().tolist())
    overlap = len(top_v4 & top_lm) / max(len(top_v4), 1)

    result = {
        "n_chunks": len(chunks),
        "total_seq": int(total_seq),
        "check_layer": check_layer,
        "spearman_v4_vs_lmcache": spearman,
        "pearson_v4_vs_lmcache": pearson,
        "max_relative_diff_pretok": rel_max,
        "top_15pct_index_overlap": overlap,
        "n_v4_top": len(top_v4),
        "n_lm_top": len(top_lm),
        "v4_deviation_first_8_per_token": dev_v4[:8].tolist(),
        "lmcache_deviation_first_8_per_token": dev_lmcache[:8].tolist(),
    }
    (out_dir / "lmcache_hkvd_compare.json").write_text(json.dumps(result, indent=2))

    print("\n=== LMCache HKVD ranking comparison ===")
    print(f"n_chunks={result['n_chunks']} total_seq={result['total_seq']} check_layer={check_layer}")
    print(f"Spearman ρ (v4 pre-RoPE vs LMCache post-RoPE): {spearman:.6f}")
    print(f"Pearson r:                                     {pearson:.6f}")
    print(f"max relative per-token diff:                   {rel_max:.3e}")
    print(f"top-15% index overlap:                         {overlap*100:.2f}% ({len(top_v4 & top_lm)}/{len(top_v4)})")
    print(f"\nFirst 8 tokens v4 dev:      {dev_v4[:8].tolist()}")
    print(f"First 8 tokens LMCache dev: {dev_lmcache[:8].tolist()}")
    print(f"\nWritten: {out_dir / 'lmcache_hkvd_compare.json'}")


if __name__ == "__main__":
    main()
