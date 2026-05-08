# Phase 8 — Gradual Filtering Discovery Experiment (Interactive)

> **Tolerance**: N/A (F1 metric primary)
> **Estimated cost cap**: $25 [L22 M7]
> **Wall time**: 사용자 검토 의존 (4 step × 검토 시간)

## Goal

논문 §4.3 multi-check-layer gradual filtering scheme의 효과 측정. LMCache의 단순 check_layer=1 flat schedule과 head-to-head 비교.

**v3의 다른 phase들과 다른 점**: Phase 8은 **discovery experiment**. Pure full auto 아니고 사용자 검토 4회 포함.

## 모델

- Mistral-7B-Instruct-v0.2
- Llama-3.1-8B-Instruct
- (Llama-70B 제외 — 비용)

## 데이터

mydata cacheblend_fig12 prompts.jsonl 에서 100 sample/schedule 샘플링 (시드 고정).

## 4-Step Interactive Flow

### Step 1 — Per-model layer profiling (자동, ~$3)

목표: 각 layer의 3 메트릭 측정 → Top-N candidate check layers 발견.

**메트릭** [L22 M1]:
- **(a) Top-15% mass**: 토큰별 KV deviation 정렬 시 top 15% 토큰이 전체 deviation의 X% 차지
- **(b) Spearman rank corr**: KV deviation rank vs forward attention deviation rank
- **(c) Information gain**: layer i HKVD 선정 후 layer i+1, i+2 attention deviation 감소량 (추가 partial-forward, 비용 ↑)

**자동 결정 알고리즘** [L22 M2 옵션 D]:

```python
For each metric m in [(a), (b), (c)]:
    1. score_m(l) for all 32 layers (Mistral) or 32 layers (Llama-8B)
    2. significant_layers = {l : score_m(l) >= threshold_m}
       - threshold_a = 0.30
       - threshold_b = 0.30  
       - threshold_c = TBD (1차 실험 후 normalize 기준 결정)
    3. Sort significant by score_m desc
    4. Gap analysis:
       gaps[i] = score_m(sorted[i]) - score_m(sorted[i+1])
       max_gap_pos = argmax(gaps[:3])
       gap_threshold = 0.10
       - gap[0] >= 0.10: 1-check, layer = sorted[0]
       - gap[1] >= 0.10: 2-check, layers = sorted[:2]
       - else: 3-check, layers = sorted[:3]
```

**시각화 의무** [L22 M2]:
- 6 plots = 3 metrics × 2 models
- 각 plot: x축 layer index (0~31), y축 metric score, threshold 가로선, significant layers 강조, 선정된 check layer 마커, gap 위치 수직선
- 저장: `reports/phase-8-step1-attachments/{model}_{metric}.png`

**산출물**:
- `reports/phase-8-step1-attachments/profile_data.json` (raw measurements)
- `reports/phase-8-step1-attachments/{model}_{metric}.png` × 6
- `reports/phase-8-step1-report.md` (이메일 전송)

**비용**: ~$3 (Mistral $1.5 + Llama $1.5 + (c) information gain partial-forward 약간)

---

### [사용자 검토 + 프롬프트 ①]

사용자가 6 plots 검토 후 다음 결정:

1. **Budget 정의 방식** (옵션 A/B/Hybrid) [L22 M4]:
   - A: 전체 32 layers 평균 = budget (논문 §4.2)
   - B: zone (first_check_layer 이후) 평균 = budget
   - Hybrid: schedule은 B로 정의, 보고서는 두 수치 모두

2. **메트릭별 check_layers** 채택 또는 조정:
   - 자동 결과 확인
   - 사용자가 직접 layer 추가/제거 가능

3. **Threshold 미세조정**:
   - threshold_a, threshold_b, gap_threshold 변경 가능

사용자는 다음 형식으로 응답:

```
Step 2 진행:
- Budget 정의: B (zone average)
- Mistral (a): [3, 1, 5]
- Mistral (b): [4, 2]
- Mistral (c): [4, 8, 12]
- Llama (a): [2, 6, 10]
- Llama (b): [3]
- Llama (c): [3, 7, 14]
```

---

### Step 2 — Schedule 생성 (CPU, $0)

사용자 결정 반영해 schedule 인스턴스 생성:

- Budget ∈ {0.05, 0.10, 0.15, 0.20}
- Schedule type ∈ {linear_decay, uniform_baseline}
- 메트릭 × budget × type × model 조합

**Linear decay**: weights = [n+1-i for i in 1..n], scale 자동 조정으로 평균 = budget.
- 3-check 예: weights = [3, 2, 1] → r1=3s, r2=2s, r3=1s

**Uniform baseline**: 같은 check_layers 위치, 같은 평균 budget, r_i 모두 같음.

**예상 schedule 개수** [L22 M3]:
- 3 metrics × 4 budgets × 2 schedule_types × 2 models = **48 schedule** (max)
- 1-check 메트릭이 섞이면 더 적음

**산출물**:
- `reports/phase-8-step2-attachments/schedules.json` (모든 schedule 인스턴스, r1/r2/r3 정확한 값 + zone/total compute ratio)
- `reports/phase-8-step2-report.md`

---

### [사용자 검토 + 프롬프트 ②]

schedule 표 검토. 의미 없는 schedule (예: budget 작아 single-check가 budget 초과) skip 결정.

```
Step 3 진행:
- Skip schedules: ['mistral_a_uniform_b005', ...]
- Include all others
```

---

### Step 3 — F1 sweep (GPU, ~$15)

각 schedule × 100 sample (mydata에서 샘플링).

```bash
python benchmarks/run_eval.py \
    --runner cacheblend.runners:GradualV4Runner \
    --schedule_file reports/phase-8-step2-attachments/schedules.json \
    --schedule_id <id> \
    --n 100 \
    --report reports/phase-8-step3-attachments/<id>.jsonl
```

**산출물**:
- `reports/phase-8-step3-attachments/<schedule_id>.jsonl` × 48
- `reports/phase-8-step3-attachments/heatmap.png` (모델 × budget × metric × schedule_type)
- `reports/phase-8-step3-report.md`

**비용 cap**: $15 (이 step 단독). 초과 시 자동 STOP, 사용자 재승인.

---

### [사용자 검토 + 프롬프트 ③]

heatmap 검토. 추가 sweep (예: 더 fine-grained budget) 또는 종료 결정.

```
Step 4 진행 (추가 sweep 없음)
```

또는

```
추가 schedule sweep:
- budget 0.075, 0.125 추가 시도
- 다음 budget 후 Step 4
```

---

### Step 4 — LMCache flat baseline 비교 (자동, ~$1)

LMCache 단순 디자인 (check_layer=1, ratio=budget) vs best gradual schedule:

- 같은 100 sample
- Same model
- Apples-to-apples F1 비교 (paired bootstrap CI)
- Compute ratio 정직한 비교 (zone vs total)

**산출물**:
- `reports/phase-8-final-report.md`:
  - Best gradual schedule per model
  - F1 차이 + paired bootstrap CI
  - Total compute ratio 비교
  - Design recommendation: "gradual is better / flat is sufficient / model-specific"

---

## Acceptance Criteria

각 step 종료 후 사용자 검토를 받기 때문에 acceptance는 **각 step의 파일 산출물 존재 + 비용 한도** 위주.

- **8.step1.1** — 6 plots 생성됨
- **8.step1.2** — `profile_data.json` 존재
- **8.step1.3** — `step1-report.md` 작성, v5-lessons 섹션 포함
- **8.step1.4** — Cost ≤ $33 누적 (v3 끝 $30 + step1 $3)

- **8.step2.1** — `schedules.json` 존재
- **8.step2.2** — 각 schedule이 정확한 r 값 + 평균 budget 만족 (수식 검증)
- **8.step2.3** — `step2-report.md` 작성

- **8.step3.1** — 모든 (skip 안 한) schedule 결과 파일 존재
- **8.step3.2** — heatmap 생성됨
- **8.step3.3** — Cost ≤ $48 누적
- **8.step3.4** — `step3-report.md` 작성

- **8.step4.1** — 최종 보고서 작성
- **8.step4.2** — Apples-to-apples 비교 표 포함
- **8.step4.3** — Cost ≤ $55 누적 (cap)

## v5-lessons 섹션 의무

각 step 보고서에 의무. Phase 8 자체가 discovery라 새 lesson 발견 가능성 높음.
