# Phase 6 — Mistral-7B Evaluation (Musique 200, Figure 12-like) Report

> Sub-phase 6a (20) → 6b (50) → 6c (200) 순차 진행.
> Tolerance: N/A (F1 metric primary).
> Result: **ALL PASS — 6a 5/5, 6b 3/3, 6c 3/3 (총 11/11)**.

## 1. Headline

| Sub-phase | n | f1_diff_cb_vs_full | f1_diff_cb_vs_reuse | ci_low_cb_vs_reuse | Verdict |
|---|---:|---:|---:|---:|---|
| 6a smoke | 20 | +0.0602 | +0.1053 | +0.0086 | PASS 5/5 |
| 6b mid | 50 | -0.0183 | +0.1189 | n/a (gate 무관) | PASS 3/3 |
| **6c full** | **200** | **-0.0320** | **+0.0790** | **+0.0455** | **PASS 3/3** |

**핵심 결과 (6c)**: CacheBlend > FullReuse at 95% CI **[+0.0455, +0.1141]**. F1 절대값 차이 = -0.0320 vs FullRecompute (catastrophic 차단 -0.05 안에서 안전). Paper §4 의 quality preservation claim 검증.

## 2. Pod (vast.ai)

| 항목 | 값 |
|---|---|
| Instance | `36296967` (Phase 5 종료 시 stop, Phase 6 진입 시 `vastai start instance` 으로 resume. SSH direct port 32318 동일하게 매핑됨) |
| GPU | 1× NVIDIA RTX 3090 (24 GB) |
| 단가 | $0.1611 / hr |
| 6a wall | ~10 min (재실행 3회: OOM 디버깅 + L40 fix + 성공) |
| 6b wall | ~5 min (50 samples × 4 runners) |
| 6c wall | ~16 min (200 samples × 4 runners) |
| **Phase 6 billing 합계 (manual)** | **$0.60** ($0.10 + $0.10 + $0.40) |
| 누적 (cost-tracker) | **$1.09 / $5** (Phase 0~5 한도) → cap 6: **$1.09 / $10 cap** (10.9%) |
| Mistral-7B cache | `/workspace/.hf_home` (Phase 1 다운로드 보존, Phase 6 에서 재다운로드 없음) |

## 3. Env parity

`bash scripts/diff_env.sh` Pod 진입 직후: **7/7 핀 match**. Phase 1~5 동일. Mistral cache 11 blob files 보존.

## 4. 4 runner × 200 sample 통계 (6c, full)

| Runner | n | f1_mean | f1_std | rouge_l_mean | ttft_s_mean | total_s_mean | f1=0 / f1≈1 |
|---|---:|---:|---:|---:|---:|---:|---|
| FullRecomputeRunner | 200 | **0.2542** | 0.274 | 0.203 | 0.246 | 0.872 | 67/10 |
| FullReuseRunner | 200 | 0.1432 | 0.207 | 0.079 | 0.251 | 1.125 | 102/4 |
| PrefixCacheRunner | 200 | 0.2542 | 0.274 | 0.203 | 0.244 | 0.867 | 67/10 |
| **CacheBlendV4Runner** | 200 | **0.2222** | 0.264 | 0.143 | 0.263 | 0.979 | 74/9 |

CacheBlend 설정: `recompute_ratio=0.15`, `check_layer=1`.

### 결정적 sample 분포 (6c)
- F1 = 0 (완전 실패): FullRecompute 67/200 (33.5%), CacheBlend 74/200 (37%), FullReuse 102/200 (51%)
- F1 ≈ 1 (완전 성공): FullRecompute 10/200 (5%), CacheBlend 9/200 (4.5%), FullReuse 4/200 (2%)

→ FullReuse 의 quality 손실은 fail rate 증가 (51% vs 33.5%) 가 주된 원인. CacheBlend 가 selective recompute 로 fail rate 를 FullRecompute 와 거의 동등 수준으로 끌어올림.

## 5. Generation 설정

- `max_new_tokens = 32` (mydata harness default 채택)
- Greedy decode (argmax, temperature=0)
- F1 = `compute_f1_against_aliases(pred, [answer, *answer_aliases], tokenizer)` — token-level max-over-aliases (mydata harness 식)
- Rouge-L = max over aliases (gate 조건 아님, 보고만)
- All 4 runners share single LayerwiseModel (RTX 3090 24GB 메모리 효율)
- **`with torch.inference_mode():`** wrap on greedy decode (L40 fix, 필수)

## 6. Gate 별 condition

### 6a (smoke, n=20)

| ID | check | 결과 | 근거 |
|---|---|---|---|
| 6a.1 | file_exists results.jsonl | ✅ | 80 rows |
| 6a.2 | F1 sanity ≥ 0.10 | ✅ | FullRecompute f1_mean = 0.1036 |
| 6a.3 | catastrophic guard ≥ -0.10 | ✅ | f1_diff_cb_vs_full = +0.0602 |
| 6a.4 | verify_phase | ✅ | mini-report 존재 |
| 6a.5 | cost ≤ $4 | ✅ | $0.59 |

### 6b (mid, n=50)

| ID | check | 결과 | 근거 |
|---|---|---|---|
| 6b.1 | f1_diff_cb_vs_full ≥ -0.05 | ✅ | -0.0183 |
| 6b.2 | f1_diff_cb_vs_reuse > 0 | ✅ | +0.1189 |
| 6b.3 | cost ≤ $7 | ✅ | $0.69 |

### 6c (full, n=200)

| ID | check | 결과 | 근거 |
|---|---|---|---|
| 6c.1 | **paired bootstrap CI** [L28] ci_low > 0 | ✅ | **+0.0455** ([+0.0455, +0.1141] at 95% CI) |
| 6c.2 | f1_diff_cb_vs_full ≥ -0.05 | ✅ | -0.0320 |
| 6c.3 | cost ≤ $10 | ✅ | $1.09 |

## 7. Cost

| Sub-phase | manual ($) | 누적 ($) | cap ($) |
|---|---:|---:|---:|
| 6a | 0.10 | 0.59 | 4 |
| 6b | 0.10 | 0.69 | 7 |
| 6c | 0.40 | 1.09 | 10 |

Phase 6 합계: **$0.60** (Phase 5 종료 $0.49 → Phase 6 종료 $1.09).

## 8. Incremental checkpoint 동작 (L07)

`benchmarks/run_phase6.py` 의 `--checkpoint-every 50` (default) 작동:
- 6c 실행 중 50/100/150/200 sample 마다 jsonl flush 4회 관찰됨 (`grep -cE "flushed" phase6c.log` = 4)
- **Reclaim 발생 없음** (vast.ai instance 안정적). resume 분기 (`--resume`) 활용 안됨, 전체 800 rows 한 번에 완료.
- Reclaim 발생 시 동작: `--resume` 으로 (id, runner) seen_keys set 으로 dedup 후 미처리만 추가. 코드 검증됨 (run_phase6.py:135-152).

## v5-lessons (이번 phase 에서 발견된 사항)

- **L40** — Greedy decode loop 에 `torch.inference_mode()` 미적용 시 GPU OOM. `model.eval()` 만으로는 autograd graph 가 매 token 마다 누적되어 RTX 3090 24GB 에서 ~5 sample 후 OOM. fix: greedy decode 를 `with torch.inference_mode():` context 로 wrap. CPU offload (`_store_to_gpu`) 도 추가 안전망.

상세는 `docs/notes/v5-lessons.md` 참조 (현재 L31~L40, **10개**).

## 9. 수정 파일

| 경로 | 변경 |
|---|---|
| `src/cacheblend/fusor.py` | `fuse_full_recompute / fuse_full_reuse / fuse_selective / fuse_prefix_cache` 모두 `return_layerwise_output=False` 키워드 추가 (Phase 6 generation 위해 past_kv 노출). 기본값 False 로 기존 Phase 2/3/4 테스트 호환. |
| `benchmarks/run_phase6.py` | 신규 (~340 lines): 4 runner 동시 평가 driver, sub-phase 별 results.jsonl + summary.json 생성. CPU offload (`_populate_kv_store` + `_store_to_gpu`), incremental checkpoint, `--resume`, `torch.inference_mode()` wrap. |
| `external/mydata/` | Pod 에서 직접 `git clone --depth 1 https://github.com/chjs/mydata.git` (Phase 5 의 tarball 이 mydata 제외했었음). SHA 검증 통과. |
| `reports/phase-6a-attachments/{results.jsonl, summary.json}` | 80 rows + summary keys |
| `reports/phase-6b-attachments/{results.jsonl, summary.json}` | 200 rows |
| `reports/phase-6c-attachments/{results.jsonl, summary.json}` | 800 rows |
| `reports/phase-6a-report.md / phase-6c-report.md` | sub-phase mini-report |
| `reports/phase-6-report.md` | 본 통합 보고서 |
| `reports/cost-tracker.json` | $0.10 + $0.10 + $0.40 = $0.60 → 누적 $1.09 |
| `gates/gate-6-{6a,6b,6c}-result.json` | 각 sub-phase gate eval (auto) |
| `docs/notes/v5-lessons.md` | L40 추가 |

## 10. Phase 7 사전점검

`tasks/phase-7-llama.md` — Llama 평가:
- **7a/7b/7c**: Llama-3.1-8B-Instruct (FP16) — 동일 sub-phase 구조 (20/50/200), 동일 driver `run_phase6.py` (model name swap 만 필요).
- **7d**: Llama-3.1-70B-Instruct (8-bit bitsandbytes) — 200 sample full only.
- **누적 cap $25** ($1.09 + ~$25).

⚠️ 80GB GPU 필요 (7d 전용): vast.ai search filter 변경 [L37].
```bash
vastai search offers 'cuda_vers >= 12.4 num_gpus=1 reliability > 0.98 gpu_ram >= 80 disk_space > 100 verified=true' -o "dph_total" --limit 6
```
권장: A100 80GB ($1.0~1.5/hr), H100 PCIe ($1.5~2.5/hr).

7a/7b/7c 는 24GB GPU (현재 instance 36296967 RTX 3090) 그대로 사용 가능. 7d 만 별도 instance 부팅. **paired bootstrap CI [L28]** 의무 (Phase 6 와 동일).

Phase 7 트리거 시 Pod 처리:
- 현재 instance keep alive (Phase 7a/b/c 즉시 진행 가능) vs stop (idle billing 절약).
- 사용자 결정 알려주세요.
