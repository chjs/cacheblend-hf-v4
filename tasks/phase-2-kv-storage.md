# Phase 2 — KV Storage & Full Reuse with RoPE Recovery

> **Tolerance category**: 
> - boundary (single-prefix vs full_recompute): IDENTICAL_PATH (max_diff = 0)
> - multi-chunk full_reuse vs full_recompute: divergence 측정 (Phase 3 baseline)
> **Estimated cost**: ~$0.5 (Pod GPU, ~18 min wall)

## Goal

청크 단위 pre-RoPE KV 저장(`KVStore`), RoPE 재적용(`apply_rope_shift`), Full KV reuse fusion(`fuse_full_reuse`) 구현. Boundary safe-shortcut 패턴 적용 [L13].

## Acceptance

1. **2.1** — `test_rope_shift_correctness`: layer-0 max_diff = 0.000e+00
2. **2.2** — `test_full_recompute_sanity`: `fuse_full_recompute` vs `hf.forward` → IDENTICAL_PATH (max_diff = 0)
3. **2.3** — `test_full_reuse_single_prefix`: single-prefix `fuse_full_reuse` vs `fuse_full_recompute` → MIXED_SHAPE (argmax exact + max_diff < 5e-2)
4. **2.4** — `test_full_reuse_multi_chunk_divergence`: multi-chunk L2 > 0 (Phase 3 baseline 측정)
5. **2.5** — `verify_phase --phase 2` returns 0
6. **2.6** — Cost ≤ $1.5 누적

## Tasks

- `Chunk` dataclass + `chunk_texts` + `fused_input_ids`
- `KVStore` (OrderedDict + LRU eviction): `put/get/has/evict_lru`
- `apply_rope_shift(K_pre_rope, target_positions, model)` — model.rotary_emb 재사용
- `precompute_chunk_kv(model, token_ids)` — Phase 1 hook으로 pre-RoPE K + V 추출
- `fuse_full_recompute` (baseline)
- `fuse_full_reuse` (k_proj/v_proj forward-hook으로 cached chunk 위치만 K(pre-RoPE)와 V를 inject)

## Tolerance enforce

각 test에 `Tolerance` enum 명시. `assert_logits_close(actual, expected, Tolerance.IDENTICAL_PATH)` 같이 호출.

## v5-lessons 섹션 의무
