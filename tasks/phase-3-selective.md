# Phase 3 — Selective KV Recompute (Core CacheBlend §4)

> **Tolerance**:
> - ratio=0 vs full_reuse: IDENTICAL_PATH (boundary safe-shortcut [L13])
> - ratio=1 vs full_recompute: IDENTICAL_PATH
> - 0 < ratio < 1: divergence reduction 측정
> **Estimated cost**: ~$0.7 (Pod GPU)

## Goal

CacheBlend §4 핵심 — HKVD selection + selective recompute. Boundary safe-shortcut 명시 디자인 패턴 [L13].

## Acceptance

1. **3.1** — `test_ratio_0_eq_full_reuse`: ratio=0 vs full_reuse → IDENTICAL_PATH (max_diff = 0)
2. **3.2** — `test_ratio_1_eq_full_recompute`: ratio=1 vs full_recompute → IDENTICAL_PATH
3. **3.3** — `test_selective_reduces_divergence`: ratio=0.15 multi-chunk L2 < full_reuse L2 × 0.85 (≥ 15% reduction)
4. **3.4** — `test_mask_is_standard_causal`: Q×Q sub-block lower-triangular [L15]
5. **3.5** — `verify_phase --phase 3` returns 0
6. **3.6** — Cost ≤ $2.5 누적
7. **3.7** — `kv_deviation` 함수가 LMCache 코드와 비교됨 (`docs/lmcache-analysis.md` 보강, Phase 0의 ≥10 cite에 더해)

## Tasks

### `hkvd.py` 구현 (LMCache 비교 의무)

`docs/lmcache-analysis.md`에 비교 표:
- (a) K only vs K+V?
- (b) squared L2 vs absolute L1?
- (c) per-head sum/mean/concat?
- (d) fp32 cast 시점?

LMCache 코드 file:line 인용 + 우리 구현 file:line. 차이 있으면 일치시킴 또는 차이 정당화 + design-decisions.md 기록.

### `fuse_selective` 구현

```python
def fuse_selective(model, chunks, kv_store, recompute_ratio: float, check_layer: int = 1):
    # Boundary safe-shortcut [L13]
    if recompute_ratio == 0:
        return fuse_full_reuse(model, chunks, kv_store)
    if recompute_ratio >= 1:
        return fuse_full_recompute(model, chunks)
    
    # Otherwise: selective logic
    # Layer 0: full fresh prefill
    # Layer 1 (= check_layer): full fresh + KV deviation 측정 + top-r% HKVD 선정
    # Layer 2+: HKVD 위치만 fresh, 나머지 cached K/V inject
    ...
```

### Long-chunk sanity (`benchmarks/long_chunk_sanity.py`)

chunk_B ∈ {60, 120, 240} × ratio ∈ {0.05, 0.10, 0.15, 0.20, 0.50} sweep. Paper Figure 6과 비교. Elbow shape 보고. 약하면 모델 특성 결론 (v3 L14).

## v5-lessons 섹션 의무
