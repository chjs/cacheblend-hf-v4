# Phase 8 Step 1 — Per-Model Layer Profiling Report

> Phase 8 의 4-step interactive 첫 단계 (자동, ~$3 cap). Metric (a) 우선 측정.
> (b)/(c) 는 사용자 검토 ① 결정 후 추가 진행 여부.
> **STOP at 사용자 검토 ① — 응답 대기**.

## 1. Outcome

| 모델 | n_samples | total_seq median | 자동 결정 | gap[0] | 판단 강도 |
|---|---:|---:|---|---:|---|
| Mistral-7B | 15 | 733 | **3-check, layers [1, 2, 3]** | 0.069 | weak (< 0.10) |
| Llama-3.1-8B | 15 | 645 | **1-check, layer [1]** | 0.109 | borderline (≥ 0.10) |

산출물: `reports/phase-8-step1-attachments/{mistral, llama8b}_{profile_a.json, top15pct_mass.png}`.

## 2. 측정 metric

**(a) Top-15% mass**: 토큰별 |KV deviation| 정렬 시 top 15% 토큰이 전체 |deviation| 의 X%.
- threshold_a = 0.30 (이 이상이면 significant layer)
- gap_threshold = 0.10 (gap[i] ≥ 0.10 이면 i+1 check 결정)

**(b) Spearman rank corr** / **(c) Information gain**: **이번 step 미측정**. 사용자 검토 ① 결과 따라 추가 진행 여부 결정.

## 3. Mistral-7B 결과

n=15 sample 의 per-layer top-15% mass median:

| Layer | Score | Significant (≥ 0.30) |
|---:|---:|:---:|
| 1 | **0.870** | ✓ |
| 2 | **0.801** | ✓ |
| 3 | **0.784** | ✓ |
| 6 | 0.751 | ✓ |
| 5 | 0.743 | ✓ |
| 4 | 0.741 | ✓ |
| 8 | 0.689 | ✓ |
| 7 | 0.672 | ✓ |
| 10 | 0.642 | ✓ |
| 9 | 0.600 | ✓ |
| ... | (전체 32 layer 모두 ≥ 0.30) | ✓ |

**모든 32 layer 가 significant**. Top layer 1 이 0.870 으로 가장 sharp.

**Gap 분석**:
- gap[0] = 0.870 − 0.801 = 0.069 (< 0.10)
- gap[1] = 0.801 − 0.784 = 0.016 (< 0.10)
- gap[2] = 0.784 − 0.751 = 0.033 (< 0.10)
- 모든 gap 이 threshold 미달 → **3-check fallback** (sorted top-3)

**자동 결정**: `selected_layers = [1, 2, 3]` (sorted ascending).

## 4. Llama-3.1-8B 결과

| Layer | Score | Significant |
|---:|---:|:---:|
| 1 | **0.885** | ✓ |
| 2 | **0.776** | ✓ |
| 3 | 0.742 | ✓ |
| 4 | 0.719 | ✓ |
| 5 | 0.683 | ✓ |
| 7 | 0.650 | ✓ |
| 6 | 0.648 | ✓ |
| 31 | 0.589 | ✓ |
| 29 | 0.563 | ✓ |
| 12 | 0.555 | ✓ |
| ... | | |

**Gap 분석**:
- gap[0] = 0.885 − 0.776 = **0.109** (≥ 0.10 ✓)
- → **1-check** decision

**자동 결정**: `selected_layers = [1]`.

흥미로운 점: Llama 의 layer 31 (마지막 layer) score 0.589 — Mistral 보다 높음. 후반부 layer 의 deviation 도 의미 있을 수 있음 (b)/(c) metric 으로 검증.

## 5. 모델 간 비교

| 항목 | Mistral-7B | Llama-3.1-8B |
|---|---|---|
| Layer 1 score | 0.870 | 0.885 |
| Layer 2 score | 0.801 | 0.776 |
| Layer 3 score | 0.784 | 0.742 |
| Layer 31 score (마지막) | 0.434 | 0.589 |
| Significant layer 수 | 32 (전부) | 32 (전부) |
| Gap[0] | 0.069 | 0.109 |
| Recommendation | 3-check [1,2,3] | 1-check [1] |

**Phase 7c FAIL 와의 연관**: Llama 의 1-check=[1] 은 Phase 7 의 default `check_layer=1` 과 동일. 즉 Phase 7c 에서 CacheBlend 가 FullReuse 우위 못 보인 원인은 **check_layer 위치가 아닌 다른 요소** (ratio 0.15 부적절? schedule type? Llama 의 본질적 cross-chunk 의존성 약함?) 일 가능성. (b)/(c) metric 추가 측정 필요할 수도.

## 6. Pod (vast.ai)

- Instance `36296967` 재사용 (RTX 3090 24GB, $0.1611/hr). RunPod 호출 0건.
- Step 1 wall: ~2 min (Mistral ~1 min + Llama ~1 min, model 캐시 보존 hot)
- Step 1 billing (manual): **$0.02**
- 누적: **$1.68 / $25 cap** (Phase 8) / $1.68 / $5 cap (Phase 0~5 누적 한도와는 무관)

## 7. 사용자 검토 ① 응답 형식 (Phase 8 task spec)

다음 형식으로 응답 부탁:

```
Step 2 진행:
- Budget 정의: A (전체 32 layer 평균) / B (zone average) / Hybrid
- Mistral (a): [1, 2, 3]      ← 자동 결정. 조정 가능 (예: [1, 5, 12])
- Llama (a): [1]              ← 자동 결정. 조정 가능 (예: [1, 4])
- (b)/(c) 추가 측정: yes / no
- threshold_a 조정: 0.30 → ?
- gap_threshold 조정: 0.10 → ?
```

또는 자유 형식 응답.

## 8. v5-lessons 신규

이번 step 에서 신규 lesson 없음. (a) 만 측정했고 핵심 발견 (Phase 7c FAIL 원인 추정) 은 보고서에 기록.

## 9. 다음 단계

사용자 검토 ① 결과 따라:
- **Step 2** (CPU, $0): schedule 생성 (linear_decay + uniform_baseline, budgets {0.05, 0.10, 0.15, 0.20})
- **Step 3** (GPU, ~$15): F1 heatmap 측정
- **Step 4** (자동, ~$1): LMCache flat schedule 비교

또는 (b)/(c) metric 추가 측정 후 Step 2 진행.
