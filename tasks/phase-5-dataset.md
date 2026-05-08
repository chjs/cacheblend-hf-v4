# Phase 5 — Dataset Pipeline (mydata integration, CPU only)

> **Tolerance**: N/A (no GPU eval)
> **Estimated cost**: $0 (CPU)

## Goal

mydata cacheblend_fig12 prompts.jsonl 활용. Runner 인터페이스 wrap. Stub model로 dispatch 검증. **GPU $0 유지 [L19]**.

## Acceptance

1. **5.1** — mydata clone + SHA verified (Phase 0에서 이미 함)
2. **5.2** — `external/mydata/cacheblend_fig12/harness/runner.py` import 가능
3. **5.3** — 5 Runner 서브클래스 instantiation OK (`FullRecomputeRunner`, `FullReuseRunner`, `PrefixCacheRunner`, `CacheBlendV4Runner`, `GradualV4Runner`)
4. **5.4** — `tests/test_runners.py::test_dispatch_with_stub_model` 통과 (CPU)
5. **5.5** — Bootstrap CI helper `paired_bootstrap_ci(...)` 구현 + 단위 테스트 [L28]
6. **5.6** — Dry-run artifact 생성 (`benchmarks/results/figure12_like/musique_dryrun.jsonl`, pred=null) [L21]
7. **5.7** — `verify_phase --phase 5`

## Tasks

### Step 1 — mydata harness import 테스트

```python
import sys
sys.path.insert(0, "external/mydata/cacheblend_fig12")
from harness.runner import CacheBlendRunner, GenerationResult
from harness.metrics import compute_f1, compute_rouge_l
```

### Step 2 — Runner 서브클래스 (`src/cacheblend/runners.py` 강화)

각 Runner의 `prepare()` + `generate()` 구현. 단 CPU stub mode (model이 None이면 dummy GenerationResult 반환).

### Step 3 — Bootstrap CI helper

`benchmarks/metrics/bootstrap.py`:

```python
def paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, confidence=0.95):
    """Returns (ci_low, ci_high) for mean(a) - mean(b)."""
```

### Step 4 — Dry-run artifact

```python
# benchmarks/run_eval.py --dry-run
# 200 samples × 5 runners → JSONL with pred=null, f1=null
```

### Step 5 — Dataset stats

mydata prompts.jsonl 통계 보고서:
- 200 sample
- supporting recall@6 mean 0.748 (mydata README와 일치 검증)
- prompt 길이 분포

## Files

- `benchmarks/run_eval.py` (mydata harness 위에)
- `benchmarks/metrics/bootstrap.py`
- `tests/test_runners.py` (CPU)
- `tests/test_bootstrap.py` (CPU)

## v5-lessons 섹션 의무
