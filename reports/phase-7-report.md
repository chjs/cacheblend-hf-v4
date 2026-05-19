# Phase 7 — Llama-3.1-8B Evaluation (mydata Musique 200) Report

> Sub-phase 7a (n=20) → 7b (n=50) → 7c (n=200). 7d (Llama-70B 8-bit) 미진행.
> **Result: 7a/7b PASS, 7c FAIL** (gate 7c.1 ci_low > 0 미달).

## 1. Headline

| Sub-phase | n | f1_diff_cb_vs_full | f1_diff_cb_vs_reuse | ci_low_cb_vs_reuse | Verdict |
|---|---:|---:|---:|---:|---|
| 7a | 20 | -0.0804 | -0.0192 | -0.1258 | PASS 2/2 (gate file+cost only) |
| 7b | 50 | -0.0885 | +0.0183 | -0.0388 | PASS 2/2 |
| **7c** | **200** | **-0.0366** | **-0.0105** | **-0.0582** | **FAIL 1/2** (ci_low ≤ 0) |

**Mistral (Phase 6c) 의 ci_low = +0.0455** 와 정반대. Llama-3.1-8B 에서 CacheBlend ratio=0.15/CL=1 default 가 FullReuse 를 능가 못함.

## 2. 4 runner × 200 sample (7c)

| Runner | n | F1 mean | F1 std | Rouge-L | f1=0 | f1≈1 |
|---|---:|---:|---:|---:|---:|---:|
| FullRecompute | 200 | **0.1926** | 0.298 | 0.093 | 100 | 17 |
| FullReuse | 200 | 0.1665 | 0.312 | 0.073 | 137 | 18 |
| PrefixCache | 200 | 0.1926 | 0.298 | 0.093 | 100 | 17 |
| **CacheBlendV4** (0.15, CL=1) | 200 | **0.1560** | 0.269 | 0.075 | 116 | 11 |

**관찰**:
- Llama 의 FullReuse / FullRecompute = **0.1665 / 0.1926 = 86%** (Mistral 은 0.143/0.254 = 56%) → Llama 는 KV reuse 만으로도 quality 손실 작음.
- CacheBlend 의 selective recompute (15% 토큰 fresh) 가 marginal gain 을 noise 에 묻히게 함.
- PrefixCache ≡ FullRecompute (positions 0..L_first-1 동일 path, Mistral 과 같은 패턴).

## 3. Pod (vast.ai)

- Instance `36296967` 재사용 (RTX 3090 24GB, $0.1611/hr)
- 7a wall: ~3 min eval + 18 min Llama-3.1-8B 다운로드 (gated, ~16 GB)
- 7b wall: ~10 min
- 7c wall: ~17 min
- Phase 7 합계 billing (manual): **$0.57** ($0.07 + $0.10 + $0.40)
- 누적 (Phase 0~7 cost-tracker): **$1.66 / $5** (Phase 0~5 한도) / **$1.66 / $20** (Phase 7 cap, 8.3%)

## 4. STOP 사유 (gate 7c.1 FAIL)

`ci_low_cb_vs_reuse = -0.0582 < 0` (95% CI [-0.058, +0.033]). Llama-3.1-8B 에서 CacheBlend default 설정이 FullReuse 를 통계적으로 능가 못함. 7d (70B 8-bit) 자동 진행 차단.

## v5-lessons (이번 phase 에서 발견된 사항)

- **L41** — `compute_f1_against_aliases` 가 빈 prediction 에서 IndexError. mydata harness `_parse_generation` 가 empty/whitespace pred 가드 부재. Llama-8B 일부 sample 에서 빈 generation 발생 (Mistral 거의 안 발생). fix: run_phase6.py 에서 try/except (IndexError, Exception) 으로 f1=0.0 fallback. v5 에서 prediction 정규화 권장.
- **L42** — Llama-3.1-8B 에서 ratio=0.15/check_layer=1 default 가 FullReuse 를 능가 못함 (Mistral +0.0455 vs Llama -0.0582). 모델 별 selective recompute hyperparameter tuning 필요. Phase 8 gradual filtering 의 motivation 강화.

상세: `docs/notes/v5-lessons.md` (현재 L31~L42, **12개**).

## 5. Phase 7d (Llama-70B 8-bit) 진행 권장 사항

**자동 진행 보류**. 근거:
- Llama-8B 에서 selective default 가 FullReuse 우위 못 보임 (gate FAIL).
- 70B 8-bit 가 8B FP16 와 같은 패턴이면 결과도 FAIL 가능성 높음.
- 80GB GPU 부팅 + Llama-70B 다운로드 (40+ GB) + 8-bit 로드 비용 약 $5-10. 검증 안 된 채 진행 시 손해.

**대안**:
1. **Phase 8 (gradual filtering) 먼저** — Llama-8B 의 elbow / 메트릭 측정 후 schedule 발견. 그 결과로 ratio/check_layer 재조정.
2. 또는 Phase 7c 결과를 paper §4 의 model-dependency 한계로 받아들이고 70B 8-bit 측정만 진행 (다른 모델 size 의 reference 데이터).

사용자 결정 후 진행.
