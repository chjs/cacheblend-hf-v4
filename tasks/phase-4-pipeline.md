# Phase 4 — Pipelining & Prefix Cache Baseline

> **Tolerance**: pipelined vs unpipelined → RECOMPUTE_PATH; prefix_cache vs full_recompute → MIXED_SHAPE
> **Estimated cost**: ~$0.5

## Goal

KV 비동기 prefetch + LoadingController + prefix_cache baseline. **TTFT 측정만, gate 조건 아님 [L27]**.

## Acceptance

1. **4.1** — `test_pipelined_eq_unpipelined`: max_diff < 1e-3 (RECOMPUTE_PATH)
2. **4.2** — `test_prefix_cache_eq_full_recompute`: argmax exact (MIXED_SHAPE)
3. **4.3** — `test_loading_controller_monotone`: ratio increases as storage gets slower
4. **4.4** — `verify_phase --phase 4`
5. **4.5** — Cost ≤ $3 누적

(TTFT speedup 검증은 보고서 참고만, gate 아님)

## Tasks

- `KVStore.prefetch_chunk` (ThreadPoolExecutor)
- `LoadingController` (StorageProfile RAM/NVMe/SATA/SLOW_DISK)
- `fuse_selective_pipelined`
- `fuse_prefix_cache` (첫 chunk만 reuse)
- `benchmarks/ttft.py` (참고용 측정)

## v5-lessons 섹션 의무
