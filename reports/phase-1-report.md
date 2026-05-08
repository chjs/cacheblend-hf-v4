# Phase 1 — Layerwise Forward Report

> Tolerance category: **SAME_SHAPE** (max_diff < 1e-3, frozen).
> Result: **PASS (4/4 conditions)**.

## 1. Outcome

- `test_layerwise_matches_standard`: **max_diff = 0.000e+00**, argmax_match = 1.0000 (bit-exact, well under SAME_SHAPE bound).
- `test_kv_extraction`: 32 layers all max_diff = 0.000e+00 (perfect pre-RoPE K capture via k_proj forward-hook).
- 2 passed in 19.06s.

## 2. Pod 정보 (vast.ai)

| 항목 | 값 |
|---|---|
| Provider | vast.ai (RunPod 가용성 부족 → L34 으로 pivot) |
| Instance ID | `36296967` (사용자 할당, machine_id 78466) |
| GPU | 1× NVIDIA RTX 3090 (24 GB) |
| GPU UUID | `GPU-99dbf357-6999-fa56-b406-9e7641914ce4` |
| 단가 | $0.1611 / hr |
| Wall time (clock) | ~1 hr (uptime ~53 min from API + setup overhead) |
| Billing 합계 | **$0.16** (수동 기록, vast.ai dashboard 정확 billing 확인 필요) |
| SSH (직접) | `ssh -p 32318 root@120.238.149.205` |
| SSH (vast.ai jump) | `ssh -p 16966 root@ssh5.vast.ai` |
| Driver | NVIDIA 535.288.01 |
| OS / Kernel | Ubuntu 24.04.4 LTS / Linux 5.15.0-176-generic |
| Image | `vastai/pytorch:cuda-12.1.1-auto` |

## 3. Env parity (Pod, conda env `cb`)

`/venv/main` 의 기본 Python 3.12 + torch 2.11 은 우리 핀과 불일치 → miniforge로 별도 conda env `cb` (Python 3.11.15) 생성 후 `requirements.txt` 정확 핀 install.

| Package | Mac venv | Pod (cb env) | requirements.txt | Match |
|---|---|---|---|---|
| python | 3.11.14 | 3.11.15 | >=3.11,<3.13 | ✓ |
| torch | 2.4.1 (CPU) | 2.4.1+cu121 (GPU) | 2.4.1 | ✓ (cuda suffix stripped) |
| transformers | 4.49.0 | 4.49.0 | 4.49.0 | ✓ |
| accelerate | 1.13.0 | 1.13.0 | 1.13.0 | ✓ |
| huggingface-hub | 0.36.2 | 0.36.2 | 0.36.2 | ✓ |
| tokenizers | 0.21.4 | 0.21.4 | 0.21.4 | ✓ |
| safetensors | 0.7.0 | 0.7.0 | 0.7.0 | ✓ |
| datasets | 4.8.5 | 4.8.5 | 4.8.5 | ✓ |
| numpy | 2.4.4 | 2.4.4 | range >=2.0,<3.0 | ✓ |
| scipy | 1.17.1 | 1.17.1 | range >=1.11 | ✓ |
| sentence-transformers | 4.1.0 | 4.1.0 | range >=3.0,<5.0 | ✓ |
| matplotlib | 3.10.9 | 3.10.9 | range >=3.8 | ✓ |
| pandas | 3.0.2 | 3.0.2 | range >=2.0 | ✓ |
| python-dotenv | 1.2.2 | 1.2.2 | range >=1.0 | ✓ |
| pytest | 9.0.3 | 9.0.3 | range >=8.0 | ✓ |
| ruff | 0.15.12 | 0.15.12 | range >=0.5 | ✓ |
| rouge-score | 0.1.2 | 0.1.2 | range >=0.1.2 | ✓ |

**`scripts/diff_env.sh` Pod 결과**: 7 match, 0 mismatch (out of 7 pinned, numpy SKIP range).

CUDA suffix 정합: Pod cu121 vs Mac CPU. `diff_env.sh` 가 strip 하므로 OK. **CUDA 12.4 가 아닌 12.1 이 설치되어 있다는 점만 v5-lessons L36 으로 기록**.

## 4. LayerwiseModel 구현

`src/cacheblend/model.py` (220 라인). HF Mistral-7B 의 forward 를 layer-by-layer 로 분리.

### 7 메서드 (acceptance criteria)

| # | 메서드 | 위치 | 1줄 설명 |
|---|---|---|---|
| 1 | `embed_tokens(input_ids)` | `src/cacheblend/model.py:84-86` | wraps `model.model.embed_tokens` (token IDs → embeddings) |
| 2 | `compute_position_embeddings(hidden_states, position_ids)` | `src/cacheblend/model.py:88-92` | shared (cos, sin) via `model.model.rotary_emb` |
| 3 | `build_causal_mask(input_ids, position_ids, past_key_values)` | `src/cacheblend/model.py:94-110` | wraps HF `_update_causal_mask`, 4D mask |
| 4 | `prefill_layer(layer_idx, hidden_states, position_ids, attention_mask, past_key_values, position_embeddings, cache_position)` | `src/cacheblend/model.py:112-141` | 단일 decoder layer forward, DynamicCache update in-place |
| 5 | `final_norm_and_lm_head(hidden_states)` | `src/cacheblend/model.py:143-146` | RMSNorm → LM head → logits |
| 6 | `forward_layerwise(input_ids, attention_mask, position_ids, use_cache)` | `src/cacheblend/model.py:148-198` | orchestrates 1~5 in HF MistralModel.forward 동일 순서 |
| 7 | `get_pre_rope_k(layer_idx)` | `src/cacheblend/model.py:200-209` | hook 이 모은 layer→tensor dict 에서 lookup |

### Hook 디자인 (k_proj forward-hook으로 pre-RoPE K capture)

- **Attach 위치**: 모든 layer 의 `self_attn.k_proj` (`MistralAttention.k_proj`, nn.Linear). 32 layer × 1 hook = 32 hook handles.
- **Capture 시점**: `k_proj` 의 forward output 직후 (RoPE 적용 직전). `apply_rotary_pos_emb` 가 호출되는 곳보다 ANY operation 앞.
- **저장 구조**: `self._pre_rope_k: dict[int, torch.Tensor]` — 키는 layer_idx (0~31), 값은 tensor of shape `(batch, seq_len, num_kv_heads * head_dim)` = `(B, S, 8 * 128)` for Mistral-7B.
- **재설정**: `forward_layerwise` 진입 시 `self._pre_rope_k = {}` 로 매 forward 마다 초기화 → multi-call 누수 방지.
- **검증** (test_kv_extraction): 같은 input 으로 forward 2번 실행하면서 두 번째 실행 시 `input_layernorm` 출력에 hook 을 추가 capture. 그 hidden_states 를 `k_proj` 에 직접 통과시킨 결과와 hook-captured K 비교 → 32 layer 전부 max_diff = 0 (bit-exact). hook 이 정확히 k_proj 의 raw output 을 잡는다는 증거.
- 정리: `__del__` 에서 hook handles `.remove()`.

**파일 라인**: `src/cacheblend/model.py` = 220 lines.

## 5. Tests

- `tests/test_layerwise.py` (95 lines) — markers: `requires_model and gpu` (Mac auto-skip via conftest).
- Tolerance category: **SAME_SHAPE** (`max_diff < 1e-3`), Phase 1 시작 전 freeze, retroactive 변경 금지.

### test_layerwise_matches_standard (1.1)

- Input: prompt `"The CacheBlend algorithm reduces TTFT by"` → tokenizer → `(B=1, S=12)` input_ids.
- Compare: `LayerwiseModel.forward_layerwise(...).logits` vs `model(input_ids=...).logits` (HF default forward).
- Result:
  ```
  logits: max_diff=0.000e+00, argmax_match=1.0000, category=same_shape, bound=max_diff < 1e-3, passed=True
  ```
- **PASS** with bit-exact match (max_diff = 0). FP16 cuBLAS path identical because layerwise replicates HF MistralModel.forward path operation-by-operation (no shape difference, no kernel switch).

### test_kv_extraction (1.2)

- 32 layer pre-RoPE K capture via hook → compared against direct `k_proj(input_layernorm(per_layer_hidden))` recompute.
- Per-layer max_diff statistics (32 values):
  - **min** = 0.000e+00
  - **median** = 0.000e+00
  - **max** = 0.000e+00
- All 32 layers bit-exact. Hook captures exactly the k_proj output before any subsequent operation.

### Tolerance 결과

| Category | Bound | Observed | Verdict |
|---|---|---|---|
| SAME_SHAPE | max_diff < 1e-3 | 0.000e+00 | PASS (3 orders of magnitude under bound) |

Tolerance 카테고리 변경 없음 (frozen).

## 6. Gate (auto-evaluated)

`gates/gate-1-to-2.json` 의 4 condition:

| ID | check_type | description | 결과 | 근거 |
|---|---|---|---|---|
| 1.1 | pytest | `test_layerwise_matches_standard` SAME_SHAPE | **PASS** | Pod pytest 결과 max_diff=0 (위 §5) |
| 1.2 | pytest | `test_kv_extraction` 32 layers | **PASS** | Pod pytest 32 layers all max_diff=0 (위 §5) |
| 1.3 | verify_phase | `verify_phase --phase 1` returns 0 | **PASS** | model.py / test_layerwise.py / phase-1-report.md 모두 존재 + report 에 v5-lessons 섹션 |
| 1.4 | cost_check | cumulative ≤ $1.00 | **PASS** | $0.16 / $1.00 (cost-tracker.json) |

- `scripts/eval_gate.py --phase 1` 로 자동 검증, 결과는 `gates/gate-1-result.json`.

## 7. Cost

- Phase 1 비용: **$0.16** (수동 기록; vast.ai 단가 $0.1611/hr × wall ~60 min, instance setup + download 17 min + tests 21s 포함).
- 누적 (Phase 0~5 한도 $5): **$0.16 / $5**.
- Phase 1 단독 cap $1: 사용량 16% (well under).

⚠️ vast.ai 의 정확 billing 은 dashboard 에서 직접 확인 필요. 수동 기록은 추정값.

## v5-lessons (이번 phase 에서 발견된 사항)

이번 phase 에서 새로 추가된 lesson 5건 (L31 은 Phase 0 carry-over, 이번엔 L32~L36):

- **L32** — `runpodctl` flag deprecation (`--container-disk-size` → `--container-disk-in-gb`). RunPod 자동 부팅 첫 시도 실패 원인.
- **L33** — Network volume DC 미일치 시 silent 미부착 (volumeInGb=0). `--data-center-ids` 명시 필수.
- **L34** — RunPod US-KS-2 GPU 가용성 부족 시 silent SSH 미실행. 사용자 지시 → vast.ai 로 pivot.
- **L35** — 사용자 할당 instance 는 reboot 금지 (사용자 정책). `auto/user instance` 분기 필요.
- **L36** — vast.ai pytorch base image 의 `/venv/main` 은 Python 3.12 + torch 2.11 (우리 pin 과 불일치). miniforge 로 별도 conda env (Python 3.11) 만들어야 함.

상세는 `docs/notes/v5-lessons.md` 참조.

## 9. 수정 파일

| 경로 | 변경 사유 |
|---|---|
| `src/cacheblend/model.py` | Phase 0 stub → 실제 LayerwiseModel 구현 (220 lines). 7 메서드 + k_proj forward-hook + pre-RoPE K capture. |
| `tests/test_layerwise.py` | 신규 (95 lines). 2 테스트 (matches_standard, kv_extraction). SAME_SHAPE freeze. |
| `scripts/runpod.sh` | flag fix `--container-disk-size` → `--container-disk-in-gb` (L32) + 자동 DC 매칭 추가 (L33). |
| `docs/notes/v5-lessons.md` | L32~L36 추가. |
| `reports/phase-1-attachments/instance.md` | vast.ai instance 메타 기록. |
| `reports/phase-1-attachments/pytest.log` | Pod pytest 출력 archive. |
| `reports/cost-tracker.json` | Phase 1 비용 $0.16 기록 (manual). |

## 10. Phase 2 사전점검

`tasks/phase-2-kv-storage.md` acceptance:
- KVStore 구현 + `fuse_full_reuse` (전체 chunk K 재사용 + RoPE shift). RoPE shift correctness 가 핵심.
- Tolerance: `RECOMPUTE_PATH` (max_diff < 1e-3), 다른 카테고리. Phase 2 시작 전 freeze.

준비 사항: Phase 1 의 LayerwiseModel + pre-RoPE K capture 인프라 그대로 사용 (Phase 2 KVStore 가 hook 으로부터 K 받음). Pod 환경 그대로 재사용 가능 (현 instance 36296967 active, 단 사용자 결정 — keep alive vs stop+resume Phase 2 시).
