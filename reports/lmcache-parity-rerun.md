# LMCache Parity Re-run Report (Phase 6c + 7c, lmc_parity impl)

> Branch: `compare/lmcache-parity-fix` · Commit: `a14660c`
> Re-ran Phase 6c (Mistral-7B) + 7c (Llama-3.1-8B) with `--cb-impl=lmc_parity`
> (paper §4 / LMCache `process_qkv` 1:1 sparse forward) and compared with prior
> legacy `fuse_selective` results.

## 1. Headline — lmc_parity vs legacy

| Phase | Model | Algorithm | CB F1 | ci_low | ci_high | Verdict |
|---|---|---|---:|---:|---:|---|
| 6c (n=200) | Mistral-7B | **legacy** (prior) | 0.222 | **+0.0455** | +0.121 | PASS |
| 6c (n=200) | Mistral-7B | **lmc_parity** (new) | 0.218 | **+0.0398** | +0.109 | PASS |
| 7c (n=200) | Llama-3.1-8B | **legacy** (prior) | 0.156 | **−0.0582** | +0.033 | FAIL |
| 7c (n=200) | Llama-3.1-8B | **lmc_parity** (new) | 0.156 | **−0.0581** | +0.033 | FAIL |

**핵심 발견: 알고리즘 변경 (full → sparse forward) 의 F1 영향 거의 0** (Mistral −0.004, Llama −0.0001). Phase 7c FAIL 의 원인은 알고리즘 차이가 아닌 **모델 의존성** (Llama 의 cross-chunk drift 가 작아 CacheBlend 의 marginal gain 이 noise 에 묻힘).

## 2. Mistral-7B Phase 6c, n=200, mydata Musique 2-hop

| Runner | n | F1 mean | F1 std | Rouge-L | f1=0 | f1≈1 |
|---|---:|---:|---:|---:|---:|---:|
| FullRecompute | 200 | **0.2542** | 0.274 | 0.203 | 67 | 10 |
| FullReuse | 200 | 0.1432 | 0.207 | 0.079 | 102 | 4 |
| PrefixCache | 200 | 0.2542 | 0.274 | 0.203 | 67 | 10 |
| **CacheBlend (lmc_parity)** | 200 | **0.2180** | 0.257 | 0.146 | 73 | 8 |

**파생 지표**:
- f1_diff_cb_vs_full = −0.0362 (FullRecompute 보다 약간 낮음, 정상)
- f1_diff_cb_vs_reuse = +0.0748 (CB > FullReuse, 정상)
- ci_low_cb_vs_reuse = **+0.0398** (95% CI [+0.040, +0.109]) — **gate 6c PASS**
- n_paired = 200

**legacy 와 비교** (Phase 6c 원본 보고서):
- legacy CB F1 = 0.2218 → lmc_parity 0.2180 (Δ = −0.004, noise 범위)
- legacy ci_low = +0.0455 → lmc_parity +0.0398 (Δ = −0.006)
- 두 CI 모두 0 보다 충분히 큼: 알고리즘 변경 무관 으로 Mistral 은 PASS

## 3. Llama-3.1-8B Phase 7c, n=200

| Runner | n | F1 mean | F1 std | Rouge-L | f1=0 | f1≈1 |
|---|---:|---:|---:|---:|---:|---:|
| FullRecompute | 200 | **0.1926** | 0.298 | 0.093 | 100 | 17 |
| FullReuse | 200 | 0.1665 | 0.312 | 0.073 | 137 | 18 |
| PrefixCache | 200 | 0.1926 | 0.298 | 0.093 | 100 | 17 |
| **CacheBlend (lmc_parity)** | 200 | **0.1561** | 0.266 | 0.075 | 114 | 11 |

**파생 지표**:
- f1_diff_cb_vs_full = −0.0365
- f1_diff_cb_vs_reuse = −0.0105 (CB < FullReuse)
- ci_low_cb_vs_reuse = **−0.0581** (95% CI [−0.058, +0.033]) — **gate 7c FAIL**
- n_paired = 200

**legacy 와 비교**:
- legacy CB F1 = 0.1560 → lmc_parity 0.1561 (Δ = +0.0001, 사실상 동일)
- legacy ci_low = −0.0582 → lmc_parity −0.0581 (Δ = +0.0001)
- 두 알고리즘 모두 Llama 에서 FullReuse 우위 못 함: **Llama 의 본질적 특성**, 알고리즘 한계 아님

## 4. 의미 해석

### Mistral vs Llama 의 갭

| 항목 | Mistral | Llama-8B |
|---|---:|---:|
| FullRecompute F1 | 0.254 | 0.193 |
| FullReuse F1 | 0.143 | 0.167 |
| FullReuse/FullRecompute | **56%** | **86%** |
| CB recover ratio | (0.218−0.143)/(0.254−0.143) = **68%** | (0.156−0.167)/(0.193−0.167) = **−43%** (negative) |

→ Mistral 은 chunk-local KV reuse 만 으로는 quality 손실 큼 (56%) → CacheBlend 의 15% recompute 가 의미. Llama 는 KV reuse 만 으로도 86% retain → 추가 recompute 의 marginal gain 이 selection noise 에 묻힘.

### 알고리즘 차이의 quality 영향 ~0

`fuse_selective` (full forward + K/V hook merge) 와 `fuse_selective_lmc_parity` (sparse forward, paper §4 / LMCache 1:1) 는 **F1 측정에서 사실상 동일**. 차이점:
- 비용: lmc_parity 가 sparse forward 라 layer N-1 의 85% non-top 위치 의 attention/FFN skip → 이론적 FLOPS 절감. **하지만 우리 측정은 quality 만 목표 (L27)** — 실제 wall time 비교 미수행.
- Quality: F1 변화 < 0.005 (둘 다 noise 범위 내).
- Selection: 동일 HKVD metric, 동일 top-K → 같은 token 선택. 차이는 선택된 token 의 hidden_state propagation 이 sparse 인지 full 인지.

해석: HKVD 가 선택한 top-K position 들이 cross-chunk attention 의 핵심 정보 운반자 이고, 이들의 K/V 정확성 (cached chunk-local → fused position 으로 의 RoPE shift + fresh re-projection) 이 quality 의 dominant factor. Hidden state propagation 의 정확성 (sparse vs full at non-top) 은 secondary.

## 5. Pod (vast.ai)

- Instance `37037805` 자동 생성 (RTX 3090 24GB, $0.3009/hr) → auto destroy
- Wall: Mistral 6c 약 16 min + Llama 7c 약 25 min + setup/download 약 25 min = 약 66 min
- Billing (manual estimate): **$0.33** ($0.3009/hr × 66/60 hr)
- 누적: $1.68 + $0.33 = **$2.01** / $55 cap (3.7%)

## 6. v5-lessons 신규

- **L43** (이미 추가됨) — fuse_selective 가 LMCache process_qkv 와 다른 알고리즘 (full vs sparse forward). 본 재실행으로 quality 영향 측정: ~0 (Mistral Δ=−0.004, Llama Δ=+0.0001).
- 추가 lesson 후보: **선택된 top-K 의 K/V 정확성 이 hidden_state propagation 정확성 보다 우세** (L44 후보).

## 7. 결론

1. **LMCache 와의 알고리즘 차이 1개 (selective forward 폭) 가 quality 에 영향 ~0** — F1 측정 으로 검증.
2. **Phase 7c FAIL 의 원인 은 알고리즘 차이 가 아닌 모델 의존성** — Llama 의 FullReuse 가 이미 86% retention 이라 CacheBlend 의 marginal gain 이 noise 에 묻힘. paper §4 의 가정 (selective recompute > full reuse) 이 Llama 에서 충족 안 됨.
3. **lmc_parity 가 paper §4 의 의도 (sparse forward 로 FLOPS 절감) 를 충실히 구현**. Quality 가 legacy 와 동등하므로 default 로 사용 권장.
4. **Phase 8 gradual filtering 의 motivation**: Llama 에서 single-CL/single-ratio 가 부적합 — multi-CL gradual schedule 이 더 fine-grained tuning 으로 우위 가능 여부 검증 필요.

## 8. 산출물

- `reports/phase-6c-lmc-parity-attachments/{results.jsonl, summary.json}` (200×4 runners)
- `reports/phase-7c-lmc-parity-attachments/{results.jsonl, summary.json}` (200×4 runners)
- `src/cacheblend/fusor.py` — `fuse_selective_lmc_parity` 함수 (+292 LOC)
- `tests/test_fusor_lmc_parity.py` — CPU sanity 8/8 PASS
- `benchmarks/run_phase6.py` — `--cb-impl {legacy, lmc_parity}` flag
- `docs/notes/lmcache-1to1-comparison.md` — 1:1 비교 정밀 보고서

## v5-lessons (이번 phase 에서 발견된 사항)

- L43 (이미 commit a14660c 에서 추가) — fuse_selective full vs sparse forward 알고리즘 차이.
- 본 재실행 으로 검증: 알고리즘 차이의 quality 영향 ~0 → L43 의 v5-fix 는 정확하지만 quality 우위 보장 아님 (note 추가 필요).

상세: `docs/notes/v5-lessons.md` (현재 L31~L43, **13 개**).
