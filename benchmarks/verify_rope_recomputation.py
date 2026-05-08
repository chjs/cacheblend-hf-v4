"""사용자 지적 검증: fuse_full_reuse 의 RoPE 재계산 여부 직접 측정.

Hypothesis:
  (A) 코드 정상 — hook injection + HF apply_rotary_pos_emb 가 fused positions 으로
      자동 재적용. paper §4 의 Full KV reuse 동작과 일치.
  (B) 사용자 지적 — RoPE 미재계산, stored chunk-local positions RoPE 가 그대로 사용.
      코드 fix + Phase 6 재평가 필요.

검증 방법:
  - 2 chunks fused. chunk_A 가 두 번째 [L_X, L_X+L_A].
  - precompute_chunk_kv 로 chunk_A 의 pre-RoPE K capture (chunk-local prefill).
  - fuse_full_reuse 후 past_key_values.key_cache[layer] 에서 post-RoPE K 추출.
  - 가설 (A) reference: apply_rope_shift(pre_rope, positions=L_X..L_X+L_A-1).
  - 가설 (B) reference: apply_rope_shift(pre_rope, positions=0..L_A-1).
  - K_observed (chunk_A 영역) vs (A), (B) max_diff 비교.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


def main():
    from cacheblend import LayerwiseModel
    from cacheblend.chunker import chunk_texts
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.fusor import fuse_full_reuse, fuse_full_recompute
    from cacheblend.rope import apply_rope_shift

    out_dir = Path("/workspace/cacheblend-hf-v4/reports/verify-rope-attachments")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Mistral-7B...", flush=True)
    t0 = time.time()
    model = LayerwiseModel("mistralai/Mistral-7B-Instruct-v0.2", dtype="float16")
    print(f"  loaded in {time.time()-t0:.1f}s")

    # 2 chunks. chunk_A 는 두 번째 (fused position [L_X, L_X+L_A]).
    docs = [
        "Paris is the capital of France and a major European city.",
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
    ]
    chunks = chunk_texts(model.tokenizer, docs)
    L_X = chunks[0].length
    L_A = chunks[1].length
    print(f"  L_X={L_X}, L_A={L_A}, total_seq={L_X + L_A}")

    # Precompute pre-RoPE K + V (chunk-local prefill, positions 0..L-1)
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(model, c)
        store.put(c.chunk_id, K, V)

    inner = model._inner
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim
    print(f"  num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Run fuse_full_reuse, return_layerwise_output=True 로 past_key_values 추출
    out = fuse_full_reuse(model, chunks, store, return_layerwise_output=True)
    past_kv = out.past_key_values

    # Multiple layers test
    layers_to_test = [0, 1, 5, 15, 31]
    results = []
    for layer_idx in layers_to_test:
        K_A_pre = store.get(chunks[1].chunk_id)["K"][layer_idx]  # (1, L_A, hidden_kv)

        # past_key_values.key_cache[layer_idx] is (1, num_kv_heads, total_seq, head_dim) — post-RoPE
        K_post_all = past_kv.key_cache[layer_idx]
        # Reshape to (1, total_seq, num_kv_heads * head_dim) to match apply_rope_shift output
        K_post_flat = K_post_all.transpose(1, 2).reshape(1, -1, num_kv_heads * head_dim)
        K_A_observed = K_post_flat[:, L_X : L_X + L_A, :]

        # Hypothesis (A): fused positions
        fused_pos = torch.arange(L_X, L_X + L_A, device=model.device).unsqueeze(0)
        K_A_expected_fused = apply_rope_shift(K_A_pre, fused_pos, model)

        # Hypothesis (B): chunk-local positions (사용자 지적)
        local_pos = torch.arange(0, L_A, device=model.device).unsqueeze(0)
        K_A_expected_local = apply_rope_shift(K_A_pre, local_pos, model)

        diff_fused = (K_A_observed.float() - K_A_expected_fused.float()).abs().max().item()
        diff_local = (K_A_observed.float() - K_A_expected_local.float()).abs().max().item()

        # Also compare chunk_X (first chunk, positions 0..L_X-1 — both hypotheses identical for it)
        K_X_pre = store.get(chunks[0].chunk_id)["K"][layer_idx]
        K_X_observed = K_post_flat[:, :L_X, :]
        x_pos = torch.arange(0, L_X, device=model.device).unsqueeze(0)
        K_X_expected = apply_rope_shift(K_X_pre, x_pos, model)
        diff_X = (K_X_observed.float() - K_X_expected.float()).abs().max().item()

        verdict = (
            "A_fused" if diff_fused < 1e-3 else
            "B_local" if diff_local < 1e-3 else
            "neither"
        )
        results.append({
            "layer": layer_idx,
            "diff_vs_fused_RoPE": diff_fused,
            "diff_vs_local_RoPE": diff_local,
            "diff_chunk0_baseline": diff_X,
            "verdict": verdict,
        })
        print(
            f"  layer {layer_idx:2d}: diff_fused={diff_fused:.3e}, "
            f"diff_local={diff_local:.3e}, diff_X={diff_X:.3e} → {verdict}"
        )

    # Final summary
    verdicts = {r["verdict"] for r in results}
    if verdicts == {"A_fused"}:
        print("\n=== CASE A — RoPE 재계산 정상 ===")
        print("  fuse_full_reuse 가 stored pre-RoPE K 를 hook 으로 inject 후")
        print("  HF apply_rotary_pos_emb 가 fused positions 으로 RoPE 재적용.")
        print("  paper §4 의 Full KV reuse 동작과 일치. 사용자 지적은 오해.")
    elif verdicts == {"B_local"}:
        print("\n=== CASE B — 사용자 지적 옳음, RoPE 미재계산 ===")
        print("  chunk-local positions 로 RoPE 적용된 K 가 그대로 사용됨.")
        print("  fuse_full_reuse / fuse_selective / fuse_prefix_cache 모두 fix 필요.")
        print("  Phase 6 결과도 재평가.")
    else:
        print(f"\n=== MIXED / NEITHER — 추가 분석 필요 ===")
        print(f"  verdicts: {verdicts}")

    json_out = {
        "L_X": L_X, "L_A": L_A, "total_seq": L_X + L_A,
        "layers_tested": layers_to_test,
        "results": results,
        "all_verdicts": sorted(verdicts),
    }
    (out_dir / "rope_recomputation_check.json").write_text(json.dumps(json_out, indent=2))
    print(f"\nWrote: {out_dir / 'rope_recomputation_check.json'}")


if __name__ == "__main__":
    main()
