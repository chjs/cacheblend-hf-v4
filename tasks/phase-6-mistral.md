# Phase 6 — Mistral Evaluation (Figure 12-like, Musique 200)

> **Tolerance**: N/A (F1 metric primary, not logit equivalence)
> **Estimated cost**: ~$10 (Pod GPU, sub-phase 6a/6b/6c)

## Goal

Mistral-7B-Instruct-v0.2 위에서 mydata prompts.jsonl 200개로 Figure 12-like 재현. 4 baseline (full_recompute / full_reuse / prefix_cache / cacheblend(0.15)) F1 비교. **TTFT 측정만, gate 조건 아님 [L27]**.

## Sub-phases

### 6a — Smoke (Musique 20)

- 첫 20 sample
- 4 method run
- F1 보고 + 형식 검증

Acceptance:
- **6a.1** — All 4 runners produce output (no exceptions)
- **6a.2** — F1(full_recompute) > 0.10 (sanity, max_new_tokens 충분 검증)
- **6a.3** — F1(cacheblend) - F1(full_recompute) ≥ -0.10 (catastrophic 차단)
- **6a.4** — Output JSONL schema correct
- **6a.5** — Cost ≤ $4 누적

### 6b — Mid (Musique 50)

- 50 sample
- F1 noise 줄이기

Acceptance:
- **6b.1** — F1(cacheblend) - F1(full_recompute) ≥ -0.05
- **6b.2** — F1(cacheblend) > F1(full_reuse) (qualitative)
- **6b.3** — Cost ≤ $7 누적

### 6c — Full (Musique 200)

- 전체 200 sample
- Paired bootstrap CI

Acceptance:
- **6c.1** — F1(cacheblend) > F1(full_reuse) at 95% CI [L28]
- **6c.2** — F1(cacheblend) - F1(full_recompute) ≥ -0.05
- **6c.3** — Cost ≤ $10 누적

## Pipeline

mydata harness eval.py 인터페이스 사용:

```bash
python external/mydata/cacheblend_fig12/harness/eval.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --prompts external/mydata/cacheblend_fig12/prompts.jsonl \
    --runner harness.runner:FullPrefillRunner \
    --runner cacheblend.runners:FullReuseRunner \
    --runner cacheblend.runners:PrefixCacheRunner \
    --runner cacheblend.runners:CacheBlendV4Runner \
    --n 20 \
    --report reports/phase-6a-attachments/results.jsonl
```

(--runner pkg.mod:Class 형식으로 우리 Runner 등록)

## Incremental checkpointing [L07]

Sub-phase 6c는 1+시간. 50 sample마다 jsonl append. Pod reclaim 시 재시작에서 skip.

## v5-lessons 섹션 의무
