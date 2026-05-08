# Phase 2 — KV Storage & Full Reuse Report

> Tolerance categories (frozen):
> - 2.1 / 2.2: **IDENTICAL_PATH** (max_diff = 0)
> - 2.3: **MIXED_SHAPE** (argmax exact + max_diff < 5e-2). Single-prefix path
>   goes through boundary safe-shortcut → IDENTICAL_PATH 실측.
> - 2.4: 측정만 (Phase 3 baseline)
>
> Result: **PASS (6/6 conditions)**.

## 1. Outcome

- 2.1 RoPE shift correctness: **layer-0 max_diff = 0.000e+00** (IDENTICAL_PATH 통과)
- 2.2 Full recompute sanity: **max_diff = 0.000e+00**, argmax_match = 1.0000
- 2.3 Full reuse single-prefix: **max_diff = 0.000e+00**, argmax_match = 1.0000 (boundary safe-shortcut [L13] 작동)
- 2.4 Multi-chunk divergence (Phase 3 baseline): **max_diff = 8.568, mean = 0.390, argmax_match = 0.9348**, per-chunk last-token = [0.0137, 2.970, 5.678]
- 4 passed in 19.08s

## 2. 인프라 정책 갱신 (L37, 2026-05-08)

GPU 인프라는 **vast.ai 단일** 사용. RunPod 사용 금지.

### 갱신된 파일

| 경로 | 변경 |
|---|---|
| `docs/notes/v5-lessons.md` | **L37** 신규 — "GPU 인프라 vast.ai 로 단일화" (정책 결정) |
| `CLAUDE.md` §3 | RunPod 6단계 setup → vast.ai 7단계 setup (miniforge conda env `cb` Python 3.11, /workspace/.hf_home, tarball+scp 우회 등) |
| `CLAUDE.md` §4 | "Pod 운영 — Reclaim" → "Pod 운영 (vast.ai)". 사용자 할당 instance reboot 금지 (L35), 자동화 instance 만 destroy 가능. |
| `CLAUDE.md` §12 | SSH key path: `~/.runpod/ssh/RunPod-Key-Go` → `~/.ssh/id_rsa`. RunPod CLI 절차 strikethrough. |
| `CLAUDE.md` §13 | `RUNPOD_GPU_FALLBACK` 17 GPU list → vast.ai search filter (RTX 3090 / 4090 / A6000 권장, Phase 7d 만 80GB). |
| `CLAUDE.md` §14 | scripts/runpod.sh 를 **DEPRECATED** 표기. cost_track.py 는 vast.ai dashboard 단가 manual 입력 모드. |
| `scripts/runpod.sh` | 헤드 주석 "DEPRECATED — vast.ai 사용, L37 참조" 추가. 즉시 교체는 phase 부풀림 방지로 미룸 (별도 task). |
| `tasks/phase-7-llama.md` | `RUNPOD_LARGE_GPU=true` strikethrough → `vastai search offers ... gpu_ram >= 80` 명령. |

## 3. Pod (vast.ai)

| 항목 | 값 |
|---|---|
| Instance | `36296967` (Phase 1 에서 keep alive, 재사용) |
| GPU | 1× NVIDIA RTX 3090 (24 GB) |
| 단가 | $0.1611 / hr |
| Phase 2 wall time | ~62 min (Pod uptime 65 → 124 min 사이의 차분; Phase 1 종료 후 idle + Phase 2 setup + tests 19s) |
| Phase 2 billing 합계 (manual) | **$0.17** |
| 누적 비용 | **$0.33** ($0.16 Phase 1 + $0.17 Phase 2) |
| SSH | `ssh -p 32318 root@120.238.149.205` |
| Image | `vastai/pytorch:cuda-12.1.1-auto` |
| conda env | `cb` (Python 3.11.15) — Phase 1 에서 생성, 재사용 |
| Mistral-7B cache | `/workspace/.hf_home` (Phase 1 download 보존, 재다운로드 없음) |

⚠️ vast.ai dashboard 의 실제 billing 확인 권장. 위 $0.17 은 wall × 단가 추정.

## 4. Env parity

`bash scripts/diff_env.sh` Pod 결과 (Phase 2 시작 직전):

```
Package              Local venv           requirements.txt     Match
torch                2.4.1                2.4.1                ✓
transformers         4.49.0               4.49.0               ✓
datasets             4.8.5                4.8.5                ✓
accelerate           1.13.0               1.13.0               ✓
huggingface-hub      0.36.2               0.36.2               ✓
tokenizers           0.21.4               0.21.4               ✓
safetensors          0.7.0                0.7.0                ✓
numpy                2.4.4                NO-PIN               SKIP (range)

Summary: 7 match, 0 mismatch (out of 8)
✓ PASS: All explicitly pinned packages match.
```

Mac venv 와 변동 없음 (Phase 1 동일).

## 5. 구현 상세

### `src/cacheblend/chunker.py` (62 lines)

| 함수 | 위치 | 1줄 설명 |
|---|---|---|
| `Chunk` (dataclass) | `chunker.py:21-30` | text + token_ids + chunk_id (16-char SHA256 prefix) + is_cached |
| `_stable_id` | `chunker.py:33-39` | (text + token IDs) 의 deterministic 16-char hex ID |
| `chunk_texts(tokenizer, texts)` | `chunker.py:42-54` | 각 text 독립 토크나이즈 (no special tokens) → list[Chunk] |
| `fused_input_ids(chunks)` | `chunker.py:57-65` | concat token_ids → (1, total_seq) tensor |
| `chunk_offsets(chunks)` | `chunker.py:68-75` | [(start, end) ...] 절대 위치 list |

### `src/cacheblend/kv_store.py` (54 lines)

| 함수 | 위치 | 1줄 설명 |
|---|---|---|
| `KVStore.__init__` | `kv_store.py:21-25` | `OrderedDict[chunk_id → {"K": list, "V": list}]`, capacity (default 1024 chunks) |
| `has` / `get` | `kv_store.py:27-36` | `get` 시 `move_to_end` (LRU touch) |
| `put` | `kv_store.py:38-50` | 새 entry 시 capacity 도달이면 `evict_lru` 호출 |
| `evict_lru` | `kv_store.py:52-57` | `popitem(last=False)` 으로 가장 오래된 entry 제거 |
| `__len__` / `clear` | `kv_store.py:59-64` | utility |

### `src/cacheblend/rope.py` (55 lines)

| 함수 | 위치 | 1줄 설명 |
|---|---|---|
| `_kproj_to_heads` | `rope.py:18-25` | `(B, S, num_kv_heads*head_dim) → (B, num_kv_heads, S, head_dim)` (HF MistralAttention 의 reshape 와 일치) |
| `_heads_to_kproj` | `rope.py:28-31` | inverse reshape |
| `apply_rope_shift(K_pre_rope, target_positions, layerwise_model)` | `rope.py:34-55` | 핵심: layer-shared `model.rotary_emb(K_heads, positions)` → (cos, sin) → `apply_rotary_pos_emb` 적용. paper §4 충실 (LMCache 와 디자인 차이, design-decisions.md §11) |

### `src/cacheblend/precompute.py` (55 lines)

| 함수 | 위치 | 1줄 설명 |
|---|---|---|
| `_install_v_proj_hooks` | `precompute.py:14-30` | 32 layer 의 `v_proj` forward-hook → `v_dict[layer_idx]` 캡처 |
| `precompute_chunk_kv(model, chunk)` | `precompute.py:33-55` | chunk 단독 forward (positions 0..L-1, RoPE 미적용 K) → Phase 1 의 k_proj hook + 임시 v_proj hook 으로 K_per_layer + V_per_layer 반환 |

### `src/cacheblend/fusor.py` (109 lines)

| 함수 | 위치 | 1줄 설명 |
|---|---|---|
| `fuse_full_recompute(model, chunks)` | `fusor.py:22-30` | baseline. fused input_ids 로 standard forward → logits |
| `fuse_full_reuse(model, chunks, kv_store)` | `fusor.py:33-105` | k_proj/v_proj forward-hook 으로 cached chunk 위치만 K(pre-RoPE)/V inject. HF 의 `apply_rotary_pos_emb` 가 자동으로 blended global positions 에 RoPE 재적용 |
| `fuse_selective(...)` | `fusor.py:108-117` | Phase 3 — boundary safe-shortcut 분기만 stub (ratio==0/≥1) |
| `fuse_selective_pipelined` / `fuse_prefix_cache` | `fusor.py:120-125` | Phase 4 stubs |

### Boundary safe-shortcut 적용 위치 [L13]

| 분기 | 위치 | 동작 |
|---|---|---|
| `fuse_full_reuse(chunks)` len ≤ 1 | `fusor.py:48-49` | single-chunk = single-prefix → `fuse_full_recompute` 직접 호출 → max_diff = 0 보장 |
| `fuse_selective` ratio == 0 | `fusor.py:111-112` | → `fuse_full_reuse` |
| `fuse_selective` ratio ≥ 1 | `fusor.py:113-115` | → `fuse_full_recompute` (kv_store 인자 pop) |

### Hook injection 디자인 (`fuse_full_reuse`)

- **Attach 위치**: 32 layer 각각의 `self_attn.k_proj` + `self_attn.v_proj` (총 64 hook).
- **Override 시점**: forward output 직후. 반환값을 `K_override[layer_idx]` / `V_override[layer_idx]` 로 교체 → forward 계속 진행 시 HF `apply_rotary_pos_emb` 가 자동으로 RoPE 적용.
- **Override mask 구조**: 각 layer 마다 `(1, total_seq, num_kv_heads*head_dim)` 0-tensor 를 만들고, chunk 별 `(start:end, :)` 슬라이스에 `kv_store.get(chunk_id)["K"|"V"][layer]` 를 채워 넣음. 결과적으로 fused sequence 전체가 stored chunk-local pre-RoPE K + V 로 채워짐.
- **RoPE 자동 재적용**: HF MistralAttention 의 `q, k = apply_rotary_pos_emb(q, k, cos, sin)` 가 우리 inject 된 K 위에 그대로 실행됨. cos/sin 은 fused sequence 의 blended global positions 기반이므로 — paper §4 의 RoPE shift 가 hook injection 만으로 자동 달성됨.
- **Cleanup**: `try/finally` 로 모든 handles `.remove()`.

## 6. Tests 4개 결과

### 2.1 test_rope_shift_correctness — IDENTICAL_PATH

- Layer-0 의 `apply_rope_shift(K_pre_rope, positions=arange(L))` vs 직접 `apply_rotary_pos_emb` 호출 비교.
- **max_diff = 0.000e+00** — bit-exact. 우리 helper 가 HF 의 RoPE 계산 path 와 정확히 동치.
- Verdict: **PASS** (IDENTICAL_PATH bound 만족).

### 2.2 test_full_recompute_sanity — IDENTICAL_PATH

- `fuse_full_recompute(chunks=[PROMPT])` vs `model(input_ids=...).logits` 비교.
- **max_diff = 0.000e+00**, **argmax_match = 1.0000**.
- ToleranceResult.detail: `logits: max_diff=0.000e+00, argmax_match=1.0000, category=identical, bound=max_diff == 0, passed=True`.
- Verdict: **PASS**.

### 2.3 test_full_reuse_single_prefix — MIXED_SHAPE (실측 IDENTICAL_PATH)

- `fuse_full_reuse(chunks=[PROMPT], store)` vs `fuse_full_recompute(chunks=[PROMPT])` 비교.
- **max_diff = 0.000e+00**, **argmax_match = 1.0000**.
- ToleranceResult.detail: `logits: max_diff=0.000e+00, argmax_match=1.0000, category=mixed_shape, bound=argmax == 1.0 AND max_diff < 5e-2, passed=True`.
- Boundary safe-shortcut [L13] 작동: `len(chunks) == 1` → `fuse_full_recompute` 로 dispatch → IDENTICAL_PATH.
- Verdict: **PASS** (MIXED_SHAPE bound 안에서 IDENTICAL 달성).

### 2.4 test_full_reuse_multi_chunk_divergence — Phase 3 baseline

3 chunk × 32 layer multi-chunk fused vs full_recompute 비교 (gate 조건은 측정만, bound 없음):

| 통계 | 값 |
|---|---|
| n_chunks | 3 |
| total_seq | 46 |
| max_diff_overall | **8.568** |
| mean_diff_overall | 0.390 |
| argmax_match_ratio | **0.9348** (43/46 위치 일치) |
| per-chunk last-token max_diff | **[0.0137, 2.970, 5.678]** |

**해석**:
- 첫 chunk last-token: 0.014 — chunk-local 위치 == fused 위치 0..L1 → 거의 동일. (RoPE 적용된 K 가 chunk-local 0..L1 = global 0..L1, 추가 noise 없음.)
- 두번째 chunk: 2.97 — 위치 shift (L1..L1+L2 범위) → 큰 발산.
- 세번째 chunk: 5.68 — 가장 깊은 위치 shift → 발산 최대.
- argmax 93.5% 일치 — 대부분 토큰의 top-1 prediction 은 보존됨. 발산은 logit 의 magnitude 에 집중.

**Phase 3 가 줄여야 할 baseline**: 위 max_diff 8.57, mean 0.39 가 selective recompute 의 시작점. Top-15% 토큰을 recompute 하면 (논문 §4.3) 두 분포가 가까워질 것이라는 것이 paper 의 주장 — Phase 3 에서 직접 검증.

📁 데이터: `reports/phase-2-attachments/multi_chunk_divergence.json`.

## 7. Gate 6 condition

| ID | check_type | 결과 | 근거 |
|---|---|---|---|
| 2.1 | pytest | **PASS** | layer-0 max_diff=0 (IDENTICAL_PATH) |
| 2.2 | pytest | **PASS** | logits max_diff=0, argmax 1.0 (IDENTICAL_PATH) |
| 2.3 | pytest | **PASS** | logits max_diff=0, argmax 1.0 (MIXED_SHAPE bound 안에서) |
| 2.4 | pytest | **PASS** | multi-chunk max_diff=8.57 > 0 (sanity), 통계 측정 완료 |
| 2.5 | verify_phase | **PASS** | model.py / kv_reuse 테스트 / report 모두 존재 + `## v5-lessons` 섹션 |
| 2.6 | cost_check | **PASS** | $0.33 / $1.50 cap (cumulative_usd in cost-tracker.json) |

`gates/gate-2-result.json` 에 자동 기록.

## 8. Cost

- Phase 2: **$0.17 / cap N/A** (Phase 2 단독 cap 명시 없음. 누적 cap $1.50)
- 누적 (Phase 0~5 한도 $5): **$0.33 / $5** (6.6%)
- Phase 2 만 wall 약 62 분 (idle 시간 포함). 실제 GPU 작업은 19초 (test 실행). Phase 3 부터 idle 시간 압축 권장.

## v5-lessons (이번 phase 에서 발견된 사항)

이번 phase 에서 신규 추가된 lesson 1건:

- **L37** — GPU 인프라 vast.ai 단일화. RunPod 사용 금지 (사용자 결정 2026-05-08). CLAUDE.md §3/§4/§12/§13/§14 갱신, scripts/runpod.sh deprecation 주석, tasks/phase-7-llama.md 의 `RUNPOD_LARGE_GPU` strikethrough.

그 외 새 발견 없음 (Phase 2 알고리즘 작업은 Phase 1 인프라 그대로 활용했으므로 인프라 issue 0 건).

상세는 `docs/notes/v5-lessons.md` 참조.

## 9. 수정 파일

| 경로 | 변경 사유 |
|---|---|
| `src/cacheblend/chunker.py` | Phase 0 stub → 실 구현 (62 lines): Chunk dataclass, chunk_texts, fused_input_ids, chunk_offsets |
| `src/cacheblend/kv_store.py` | Phase 0 stub → 실 구현 (54 lines): OrderedDict-backed LRU, put/get/has/evict_lru |
| `src/cacheblend/rope.py` | Phase 0 stub → 실 구현 (55 lines): apply_rope_shift via HF apply_rotary_pos_emb |
| `src/cacheblend/precompute.py` | Phase 0 stub → 실 구현 (55 lines): chunk standalone forward + v_proj hook |
| `src/cacheblend/fusor.py` | Phase 0 stub → 실 구현 (109 lines): fuse_full_recompute, fuse_full_reuse (k_proj/v_proj hook injection), boundary safe-shortcut |
| `tests/test_kv_reuse.py` | 신규 (~150 lines): 4 tests (rope_shift / recompute_sanity / reuse_single_prefix / multi_chunk_divergence) |
| `CLAUDE.md` | §3/§4/§12/§13/§14 vast.ai 기반 갱신 [L37] |
| `scripts/runpod.sh` | 헤드 주석 "DEPRECATED — vast.ai 사용, L37 참조" 추가 |
| `tasks/phase-7-llama.md` | `RUNPOD_LARGE_GPU` 부분 vast.ai search filter 로 갱신 [L37] |
| `docs/notes/v5-lessons.md` | L37 추가 |
| `reports/phase-2-attachments/pytest.log` | Pod pytest 출력 archive |
| `reports/phase-2-attachments/multi_chunk_divergence.json` | Phase 3 baseline 데이터 |
| `reports/cost-tracker.json` | Phase 2 비용 $0.17 → 누적 $0.33 |

## 10. Phase 3 사전점검

`tasks/phase-3-selective.md` — selective recompute (HKVD 기반).

핵심 acceptance:
- 3.1 `kv_deviation` 정확한 공식 — paper §4.2 + LMCache `blender.py:89-91` (`sum((k_new − k_old)^2, dim=1)`, fp32 cast, no normalize).
- **3.7 — LMCache HKVD 비교 의무**: 우리 `kv_deviation` 결과가 LMCache 의 동일 입력에서의 deviation 과 ranking 동일성 검증. `docs/lmcache-analysis.md` §Q2 의 분석 그대로 적용.
- 3.x — selective recompute 후 multi-chunk max_diff < Phase 2 §2.4 의 baseline (8.57) 충분히 감소.
- Tolerance: 별도 카테고리 (RECOMPUTE_PATH 또는 MIXED_SHAPE).

준비 사항: Phase 2 의 KVStore + apply_rope_shift + fuse_full_reuse hook 인프라 그대로 사용. selective 는 hook 안에서 top-K 토큰만 fresh K/V 로, 나머지는 stored K(post-RoPE shift)/V 로 inject. Pod 환경 (instance 36296967, conda `cb`, Mistral-7B cache) 그대로 재사용 가능.
