# Phase 6c — Mistral-7B Musique 200 (Full) Mini-Report

> Sub-phase 6c. 통합 보고서는 `phase-6-report.md`.

## Outcome

**PASS 3/3**.

| ID | 결과 | 핵심 |
|---|---|---|
| 6c.1 paired bootstrap CI [L28] | ✅ | `ci_low_cb_vs_reuse = +0.0455 > 0` (95% CI [+0.0455, +0.1141], n_paired=200) |
| 6c.2 catastrophic guard | ✅ | `f1_diff_cb_vs_full = -0.0320 ≥ -0.05` |
| 6c.3 cost | ✅ | $1.09 / $10 cap |

## 4 runner × 200 sample

| Runner | n | f1_mean | f1_std | rouge_l_mean | ttft_s_mean | total_s_mean | f1=0 / f1≈1 |
|---|---:|---:|---:|---:|---:|---:|---|
| FullRecomputeRunner | 200 | **0.2542** | 0.274 | 0.203 | 0.246 | 0.872 | 67/10 |
| FullReuseRunner | 200 | 0.1432 | 0.207 | 0.079 | 0.251 | 1.125 | 102/4 |
| PrefixCacheRunner | 200 | 0.2542 | 0.274 | 0.203 | 0.244 | 0.867 | 67/10 |
| **CacheBlendV4Runner** (0.15, CL=1) | 200 | **0.2222** | 0.264 | 0.143 | 0.263 | 0.979 | 74/9 |

| Diff | 값 |
|---|---|
| `f1_diff_cb_vs_full` | **-0.0320** |
| `f1_diff_cb_vs_reuse` | **+0.0790** |
| `ci_low_cb_vs_reuse` (95%) | **+0.0455** |
| `ci_high_cb_vs_reuse` (95%) | **+0.1141** |
| `n_paired` | 200 |

해석:
- **CacheBlend > FullReuse 95% CI 통계적 우위** [+0.0455, +0.1141] — paper §4 의 핵심 claim 검증.
- CacheBlend < FullRecompute 0.032 F1 (catastrophic 차단 -0.05 안에서 안전).
- PrefixCache ≡ FullRecompute (positions 0..L_first-1 일치, 두 path 동일).

## v5-lessons 신규

없음 (6a 의 L40 이 6b/6c 도 cover).

## Pod (vast.ai)

| 항목 | 값 |
|---|---|
| Phase 6c wall | ~16 min (200 samples × 4 runners ≈ 4 ms/token × 32 token × 200 × 4 + overhead) |
| Phase 6c billing (manual) | **$0.40** |
| 누적 (cost-tracker) | **$1.09 / $10 cap** (6c) — 11% only |
