# Phase 3 — Selective KV Recompute Report (CacheBlend §4 핵심)

> Tolerance categories (frozen):
> - 3.1 / 3.2: **IDENTICAL_PATH** (boundary safe-shortcut [L13], max_diff = 0)
> - 3.3: divergence reduction ≥ 15% vs full_reuse baseline
> - 3.4: Q×Q sub-block lower-triangular [L15]
>
> Result: **PASS (7/7 conditions)**.

## 1. Outcome

| ID | 결과 | 핵심 수치 |
|---|---|---|
| 3.1 ratio=0 vs full_reuse | ✅ PASS | max_diff = 0.000e+00 (IDENTICAL_PATH) |
| 3.2 ratio=1 vs full_recompute | ✅ PASS | max_diff = 0.000e+00 (IDENTICAL_PATH) |
| 3.3 ratio=0.15 multi-chunk reduction | ✅ PASS | full_reuse_L2=0.6149, selective_L2=0.4465, **reduction = 27.39%** (target ≥ 15%); HKVD = 6/46 tokens |
| 3.4 mask Q×Q causal | ✅ PASS | shape (1,1,46,47), Q×Q=46×46 sub-block lower-triangular ✓ (K=47, +1 future-cache slot — L38) |
| **LMCache HKVD ranking 비교** | ✅ Spearman ρ = **0.9991**, top-15% overlap **100%** | |
| Long-chunk sweep (15 cells) | ✅ 모든 cell 측정됨 | (§7 표) |

## 2. Pod (vast.ai)

| 항목 | 값 |
|---|---|
| Instance | `36296967` (Phase 1 부터 keep alive, Phase 3에 재사용) |
| GPU | 1× NVIDIA RTX 3090 (24 GB) |
| 단가 | $0.1611 / hr |
| Phase 3 wall time | ~30 min (실 GPU 작업 ≈ 10 min: pytest 4개 ~30s + LMCache 비교 ~15s + long-chunk sweep ~9 min) |
| Phase 3 billing 합계 (manual) | **$0.08** |
| 누적 비용 | **$0.41** ($0.16 + $0.17 + $0.08) |
| Idle 압축 결과 | Pod 안에서는 GPU 작업만, 보고서/이메일/git 은 모두 로컬에서 처리. Phase 1/2 의 1시간 wall (Phase 1 다운로드 17분 + 대기) 대비 Phase 3 는 30분 — 50% 절감. |

⚠️ vast.ai dashboard 의 정확 billing 확인 권장. 위 $0.08 은 wall × 단가 추정.

## 3. Env parity

`bash scripts/diff_env.sh` Pod 결과 (Phase 3 시작 직전):
```
torch                2.4.1                ✓
transformers         4.49.0               ✓
... (8개 핀 7/7 match, numpy SKIP range)
Summary: 7 match, 0 mismatch
```
Mac venv 변동 없음 (Phase 1/2 동일).

## 4. 구현 상세

### 4.1 `src/cacheblend/hkvd.py` — 92 lines

| 함수 | 위치 | 1줄 설명 |
|---|---|---|
| `kv_deviation(K_new, K_old)` | `hkvd.py:23-50` | per-token squared L2 sum (LMCache `blender.py:89-91` 식). `(K_new − K_old).pow(2).sum(dim=-1).squeeze(0)`, fp32 cast. K only, no normalize. |
| `select_top_k(deviations, ratio)` | `hkvd.py:53-72` | `int(N × ratio)`, `max(1)`, `topk(largest=True)` → `sort` ascending. LMCache `blender.py:94-101` 와 동일. |

### 4.2 `src/cacheblend/fusor.py:fuse_selective` — `fusor.py:108-211` (104 lines 변경)

**Boundary safe-shortcut [L13] 분기 (헤드 부분 유지)**:
- `recompute_ratio == 0` → `fuse_full_reuse` 직접 호출 (`fusor.py:131-133`)
- `recompute_ratio >= 1` → `fuse_full_recompute` 직접 호출 (`fusor.py:134-136`)
- `len(chunks) <= 1` → `fuse_full_recompute` (single-prefix 동일 path) (`fusor.py:137-139`)

**Selective 본 로직 (단일 패스 디자인)**:

```
Layer 0..check_layer-1:  no hooks. fresh forward, hidden_states 가 cross-chunk
                         attention (causal mask) 으로 자연스럽게 교류.

Layer check_layer (=1):  k_proj output capture hook 만 등록 (return None →
                         output unchanged). 캡처한 fresh K vs K_stored[1] 로
                         kv_deviation → select_top_k → state["hkvd_indices"].

Layer check_layer+1..end: k_proj/v_proj 양쪽 모두에 'selective' hook.
                         output[non_HKVD positions] := stored[non_HKVD positions].
                         output[HKVD positions]    := unchanged (fresh).
                         HF apply_rotary_pos_emb 가 fused 의 blended global
                         positions 으로 RoPE 자동 재적용 → paper §4 와 동등.
```

핵심 코드 위치:
- 단일-pass 의 `state` dict (mutable, layer 간 전달): `fusor.py:155`
- `check_layer_hook` (observation only): `fusor.py:157-162`
- `make_selective_hook` (mask-based override): `fusor.py:164-176`
- Hook attach loop (k_proj at check_layer + k_proj/v_proj at check_layer+1..end): `fusor.py:179-194`

### 4.3 `benchmarks/long_chunk_sanity.py` — 132 lines

`build_chunks_for_target_len(tokenizer, target_len, n_chunks=3)` (`long_chunk_sanity.py:32-55`): seed text 반복 → token 수가 target 도달 시 truncate, decode round-trip → Chunk 생성.

`measure_cell(model, chunks, store, ratio, truth)` (`long_chunk_sanity.py:58-90`): full_reuse vs selective vs full_recompute (truth) 의 L2/argmax_match/max_diff 산출.

`main` (`long_chunk_sanity.py:93-130`): 3 × 5 = 15 cell sweep, JSON + MD 출력.

### 4.4 `benchmarks/lmcache_hkvd_compare.py` — 124 lines

3-chunk fused 입력으로:
- v4 deviation: `kv_deviation(K_fresh_pre_check, K_stored_pre_check)` — pre-RoPE 도메인.
- LMCache deviation 등가: 양쪽 K 모두 `apply_rotary_pos_emb(K, cos, sin)` (positions = 0..total_seq-1) 로 post-RoPE → squared L2 sum.
- Spearman ρ + Pearson r + top-15% index overlap 비교.

## 5. LMCache HKVD 비교 (acceptance 3.7)

### Setup
- Input: 3 chunks (Phase 2 §2.4 와 동일 docs), total_seq = 46 tokens, check_layer = 1.
- v4 path: pre-RoPE 도메인 deviation.
- LMCache 등가 path: 양쪽 K 모두 fused-position post-RoPE → squared L2 sum.

### 결과

| Metric | Value | 인용 |
|---|---|---|
| **Spearman ρ** (per-token rank) | **0.999137** | `lmcache_hkvd_compare.json:5` |
| Pearson r (magnitudes) | 1.000000 | `lmcache_hkvd_compare.json:6` |
| Top-15% HKVD index overlap | **100% (6/6)** | `lmcache_hkvd_compare.json:8-10` |
| max relative per-token diff | 18.3% (FP16/FP32 cast 노이즈) | `lmcache_hkvd_compare.json:7` |

### 차이 (있음, 정당화)

| Aspect | v4 (pre-RoPE) | LMCache (post-RoPE) | 정당화 |
|---|---|---|---|
| 도메인 | pre-RoPE K (k_proj output) | post-RoPE K (LMCache `blender.py:86-91`) | **RoPE = orthogonal rotation per position → squared-L2 보존**. 따라서 동일 token 의 양쪽 K 차이는 두 도메인에서 같은 deviation. design-decisions.md §11 cross-ref. |
| 매그니튜드 절대값 | 약 18% relative diff (mostly FP16 noise) | reference | 절대값 약간 다르지만 ranking 이 같으므로 top-K 선택은 동일. |
| Top-K 선택 | 동일 (overlap 100%) | reference | ✅ 일치 |

### `lmcache-analysis.md` 보강 diff 요약

§Q2(c.5) 신규 추가 — Phase 3 정량 검증 (Spearman / Pearson / overlap 표) + RoPE invariance 이론 증명 한 줄 + 데이터 cross-ref.

## 6. Tests 4개 결과

### 3.1 test_ratio_0_eq_full_reuse — IDENTICAL_PATH ✅
- `fuse_selective(ratio=0)` vs `fuse_full_reuse`: **max_diff = 0.000e+00**, argmax 1.0000.
- ToleranceResult: `max_diff=0.000e+00, argmax_match=1.0000, category=identical, bound=max_diff == 0, passed=True`.

### 3.2 test_ratio_1_eq_full_recompute — IDENTICAL_PATH ✅
- `fuse_selective(ratio=1.0)` vs `fuse_full_recompute`: **max_diff = 0.000e+00**, argmax 1.0000.
- 동일 ToleranceResult.

### 3.3 test_selective_reduces_divergence — ≥15% reduction ✅
| | full_reuse | selective (ratio=0.15) | reduction |
|---|---|---|---|
| L2 vs truth | 0.6149 | 0.4465 | **27.39%** |
| HKVD count | 0 | 6 of 46 tokens | |

`hkvd=6/46` ⇒ ratio = 6/46 ≈ 13% (target 15% × 46 = 6.9 → int(6.9) = 6, max(1) = 6). `reduction = 27.39%` 는 target `15%` 대비 거의 2× 달성.

### 3.4 test_mask_is_standard_causal — Q×Q lower-triangular ✅ (after fix)
- 첫 시도 FAIL (Q=46, K=47 → assertion `Q == K` 깨짐).
- L38 발견: HF `_update_causal_mask` 가 K dim 에 +1 future-cache slot 패딩. mask[:,:,:Q,:Q] sub-block 만 검증하는 것이 올바름.
- Fix 후: Q×Q=46×46 sub-block fully lower-triangular. **PASS**.

## 7. Long-chunk sweep — 3 × 5 = 15 cell

`benchmarks/long_chunk_sanity.py` 결과 (full data: `reports/phase-3-attachments/long_chunk_sweep.json`):

| chunk_B | total_seq | ratio | full_reuse_L2 | selective_L2 | **reduction%** | argmax_match | max_diff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 180 | 0.05 | 7.579e-01 | 6.732e-01 | 11.2 | 96.1% | 9.91e+00 |
| 60 | 180 | 0.10 | 7.579e-01 | 6.428e-01 | 15.2 | 96.7% | 9.48e+00 |
| 60 | 180 | 0.15 | 7.579e-01 | 5.708e-01 | **24.7** | 97.8% | 8.69e+00 |
| 60 | 180 | 0.20 | 7.579e-01 | 4.842e-01 | 36.1 | 98.3% | 9.52e+00 |
| 60 | 180 | 0.50 | 7.579e-01 | 2.492e-01 | 67.1 | 98.9% | 8.20e+00 |
| 120 | 360 | 0.05 | 6.771e-01 | 5.779e-01 | 14.7 | 98.3% | 1.09e+01 |
| 120 | 360 | 0.10 | 6.771e-01 | 4.918e-01 | 27.4 | 99.2% | 1.10e+01 |
| 120 | 360 | 0.15 | 6.771e-01 | 4.374e-01 | **35.4** | 99.2% | 9.05e+00 |
| 120 | 360 | 0.20 | 6.771e-01 | 3.839e-01 | 43.3 | 99.4% | 8.33e+00 |
| 120 | 360 | 0.50 | 6.771e-01 | 2.008e-01 | 70.3 | 99.4% | 6.42e+00 |
| 240 | 720 | 0.05 | 6.179e-01 | 4.861e-01 | 21.3 | 99.7% | 1.10e+01 |
| 240 | 720 | 0.10 | 6.179e-01 | 3.980e-01 | 35.6 | 99.9% | 1.00e+01 |
| 240 | 720 | 0.15 | 6.179e-01 | 3.447e-01 | **44.2** | 99.9% | 9.08e+00 |
| 240 | 720 | 0.20 | 6.179e-01 | 2.959e-01 | 52.1 | 99.9% | 8.81e+00 |
| 240 | 720 | 0.50 | 6.179e-01 | 1.269e-01 | 79.5 | 100.0% | 5.66e+00 |

### Elbow shape (Mistral-7B)

reduction% 의 ratio 별 곡선:
- chunk_B=60: 11→15→25→36→67% (ratio 0.05→0.5). 가파른 단조 증가, **명확한 elbow 없음**.
- chunk_B=120: 15→27→35→43→70%. 0.10~0.15 사이 elbow 약간 (15% 기울기 변화).
- chunk_B=240: 21→36→44→52→80%. 0.10 부근 elbow (21→36 jump 후 평탄화).

**Paper Figure 6 비교**: paper 는 ratio=0.10~0.20 elbow 보고 (Mistral). 우리 측정에서는 chunk_B=240 의 0.10 부근에서 약한 elbow 관찰, chunk_B≤120 에서는 거의 monotone. **모델/data 의존적인 약한 elbow** — v3 L14 의 모델 특성 결론과 일관 (v4-lessons.md L14 참조). 강한 elbow 가 없으므로 fixed ratio (e.g. 0.15) 보다 dataset/sequence-length-aware tuning 이 quality 측면에서 의미 있을 수 있음 — Phase 8 gradual filtering 의 motivation.

**argmax_match**: 모든 cell 96~100%. 즉 selective recompute 가 logit magnitude 만 일부 줄이고 top-1 prediction 은 거의 보존. quality (F1) 측면에서 매우 안정적인 patch 로 작동할 것으로 예상 (Phase 6/7 평가에서 확인).

## 8. Gate 7 condition

`scripts/eval_gate.py --phase 3` 결과:

| ID | check_type | 결과 | 근거 |
|---|---|---|---|
| 3.1 | pytest | ✅ PASS | max_diff=0 (IDENTICAL_PATH) |
| 3.2 | pytest | ✅ PASS | max_diff=0 (IDENTICAL_PATH) |
| 3.3 | pytest | ✅ PASS | reduction 27.39% ≥ 15% |
| 3.4 | pytest | ✅ PASS | Q×Q sub-block lower-triangular (mask shape fix 후) |
| 3.5 | verify_phase | ✅ PASS | hkvd.py / test_selective.py / benchmarks/long_chunk_sanity.py / phase-3-report.md 모두 존재 + `## v5-lessons` 섹션 |
| 3.6 | cost_check | ✅ PASS | $0.41 / $2.50 cap (cumulative_usd) |
| 3.7 | file_exists | ✅ PASS | `docs/lmcache-analysis.md` 존재 + Phase 3 정량 비교 §Q2(c.5) 보강 (Spearman 0.9991, top-15% overlap 100%) |

`gates/gate-3-result.json` 자동 기록.

## 9. Cost

- Phase 3: **$0.08** / 누적 cap $2.50 (3.2%)
- 누적 (Phase 0~5 한도 $5): **$0.41 / $5** (8.2%)
- Pod 사용 효율: Phase 3 wall 30 min, GPU 작업 약 10 min. 이전 phase 대비 idle 압축 성공 (Phase 2 의 62 분 wall → 30 분).

## v5-lessons (이번 phase 에서 발견된 사항)

이번 phase 에서 신규 추가:

- **L38** — HF causal mask K dim 이 Q+1 (future-cache slot). `_update_causal_mask` 가 `use_cache=True + DynamicCache()` 사용 시 K 차원에 +1 sentinel 추가. 검증 test 는 `mask[:,:,:Q,:Q]` sub-block 만 검사해야 함.

상세는 `docs/notes/v5-lessons.md` 참조.

## 10. 수정 파일

| 경로 | 변경 사유 |
|---|---|
| `src/cacheblend/hkvd.py` | Phase 0 stub → 실 구현 (92 lines): kv_deviation + select_top_k. LMCache `blender.py:89-101` 식 동일. |
| `src/cacheblend/fusor.py` | `fuse_selective` Phase 0 stub → 단일-pass 실 구현 (`fusor.py:108-211`, 104 lines). check_layer hook + post-check_layer selective hook. Boundary safe-shortcut 헤드 유지. |
| `tests/test_selective.py` | 신규 (4 tests, ~150 lines). 3.4 mask shape 첫 시도 후 fix (mask[:,:,:Q,:Q] sub-block 검증으로 변경). |
| `benchmarks/long_chunk_sanity.py` | 신규 (132 lines). chunk_B {60,120,240} × ratio {0.05,0.10,0.15,0.20,0.50} sweep. |
| `benchmarks/lmcache_hkvd_compare.py` | 신규 (124 lines). v4 pre-RoPE deviation vs LMCache post-RoPE deviation Spearman/Pearson/overlap. |
| `docs/lmcache-analysis.md` | §Q2(c.5) 신규 — Phase 3 정량 검증 (Spearman 0.9991, top-15% 100%) + RoPE orthogonal invariance 이론. |
| `docs/notes/v5-lessons.md` | L38 추가. |
| `reports/phase-3-attachments/pytest.log` | pytest archive |
| `reports/phase-3-attachments/long_chunk_sweep.json` / `.md` | 15-cell sweep 데이터 |
| `reports/phase-3-attachments/lmcache_hkvd_compare.json` | LMCache HKVD ranking 비교 |
| `reports/phase-3-attachments/selective_15.json` | ratio=0.15 reduction 측정 |
| `reports/cost-tracker.json` | $0.08 → 누적 $0.41 |
| `reports/phase-3-report.md` | Phase 3 보고서 (본 문서) |
| `gates/gate-3-result.json` | gate eval 결과 (auto) |

## 11. Phase 4 사전점검

`tasks/phase-4-pipeline.md` — async prefetch + LoadingController + prefix_cache baseline.

핵심 acceptance:
- KVStore 에 async prefetch (Phase 2 sync 의 superset).
- `LoadingController` (StorageProfile 기반 cost-aware loading).
- `fuse_prefix_cache` baseline runner (only first chunk reused).
- TTFT 측정 코드 (`benchmarks/ttft.py`) — **측정만, gate 조건 아님 (TTFT 비목표 L27)**.

준비: Phase 2/3 의 KVStore + apply_rope_shift + fuse_full_reuse/selective 인프라 그대로 사용. async I/O 추가만 필요. Pod 환경 (instance 36296967, conda `cb`, Mistral-7B cache) 그대로 재사용 가능.
