# Phase 6a — Mistral-7B Musique 20 (Smoke) Mini-Report

> Sub-phase 6a (Phase 6 의 첫 단계). 통합 보고서는 6c 후 `phase-6-report.md`.

## Outcome

**PASS 5/5**.

| ID | 결과 | 근거 |
|---|---|---|
| 6a.1 | ✅ | `reports/phase-6a-attachments/results.jsonl` (80 rows = 20 × 4 runners) |
| 6a.2 F1 sanity | ✅ | `FullRecomputeRunner.f1_mean = 0.1036 ≥ 0.10` |
| 6a.3 catastrophic guard | ✅ | `f1_diff_cb_vs_full = +0.0602 ≥ -0.10` (CB > full!) |
| 6a.4 verify_phase | ✅ | results.jsonl + 본 mini-report 존재 |
| 6a.5 cost | ✅ | $0.59 / $4 cap |

## 4 runner × 20 sample 통계

| Runner | n | f1_mean | f1_std | rouge_l_mean | ttft_s_mean | total_s_mean | f1=0 / f1≈1 |
|---|---:|---:|---:|---:|---:|---:|---|
| FullRecomputeRunner | 20 | 0.1036 | 0.146 | 0.110 | 0.242 | 0.863 | 11/0 |
| FullReuseRunner | 20 | 0.0585 | 0.088 | 0.052 | 0.223 | 1.097 | 13/0 |
| PrefixCacheRunner | 20 | 0.1036 | 0.146 | 0.110 | 0.215 | 0.836 | 11/0 |
| **CacheBlendV4Runner** (0.15, CL=1) | 20 | **0.1638** | 0.249 | 0.097 | 0.238 | 0.889 | 10/**1** |

| diff | 값 |
|---|---|
| `f1_diff_cb_vs_full` | **+0.0602** |
| `f1_diff_cb_vs_reuse` | **+0.1053** |
| `ci_low_cb_vs_reuse` | **+0.0086** (95% CI low > 0 already at n=20!) |
| `ci_high_cb_vs_reuse` | +0.2424 |

관찰:
- **CacheBlend 가 FullRecompute 를 능가** (예상 외). CB 의 selective recompute 가 일부 sample 에서 더 좋은 generation 을 유도. 단 n=20 노이즈 존재 — 6b/6c 에서 안정화 검증 필요.
- **PrefixCache ≡ FullRecompute** (f1 동일). 우리 구현에서 첫 chunk 는 system prompt (positions 0..L_sys-1, RoPE shift 불필요), positional 적으로 fresh recompute 와 동등 path.
- **FullReuse 가 최저** — position-mismatch 노이즈 (chunk-local 위치로 RoPE 적용된 K 가 fused 위치에서 그대로 사용됨, paper §4 의 예측대로).

## Generation 설정

- max_new_tokens = 32
- greedy decode (argmax)
- F1 = `compute_f1_against_aliases(pred, sample.answer + answer_aliases, tokenizer)` — token-level max-over-aliases (mydata harness 식)
- Rouge-L = max over aliases (보고만)
- CacheBlend: ratio=0.15, check_layer=1

## Pod (vast.ai)

| 항목 | 값 |
|---|---|
| Instance | `36296967` (resume 후 direct port 32318 유지) |
| GPU | RTX 3090 24GB |
| 단가 | $0.1611 / hr |
| Phase 6a wall | ~10 min (재실행 3회: OOM 디버깅 v1/v2 + 성공 v3 + cleanup) |
| Phase 6a billing (manual) | **$0.10** |

## v5-lessons 신규

- **L40** — Greedy decode loop 에 `torch.inference_mode()` 미적용 시 GPU OOM. `model.eval()` 만으로는 autograd graph 가 누적되어 32 token × 32 layer 의 intermediate activation 이 GB 단위로 leak. 첫 5-10 sample 후 OOM. fix: `with torch.inference_mode():` 로 wrap. v5 에서 모든 generation loop / forward 에 inference_mode 의무화.

## 다음 단계

→ 6b (Musique 50). 동일 driver 로 `--n 50`.
