# CacheBlend HF v4 — 구현 및 실험 종합

> 외부 공유용 종합 문서. Phase 0~6 완료 시점 (2026-05-08).
> 모든 인용은 v4 저장소의 실제 file:line. 추측은 "확인 못함" 명시.
> Cross-references: `GOAL.md`, `docs/lmcache-analysis.md`, `docs/figure12_like_disclosure.md`, `docs/design-decisions.md`, `docs/notes/v5-lessons.md`.

## 목차

1. [구현 상세 (Phase 1~5)](#1-구현-상세-phase-15)
2. [실험 결과](#2-실험-결과)
3. [Gradual filtering 진행 여부](#3-gradual-filtering-진행-여부)
4. [다음 단계 (Phase 7 / 8)](#4-다음-단계)

---

## 1. 구현 상세 (Phase 1~5)

### 1.1 Tolerance 4단계 + L13 boundary safe-shortcut

**Tolerance 카테고리** (`src/cacheblend/tolerance.py:19-30`):

```python
class Tolerance(Enum):
    IDENTICAL_PATH = "identical"     # max_diff == 0
    SAME_SHAPE = "same_shape"        # max_diff < 1e-3
    MIXED_SHAPE = "mixed_shape"      # argmax_exact AND max_diff < 5e-2
    RECOMPUTE_PATH = "recompute_path" # max_diff < 1e-3
```

`assert_logits_close(actual, expected, category, name)` (`src/cacheblend/tolerance.py:42-93`) 가 카테고리별 bound 검증. **Phase 시작 전 카테고리 freeze, retroactive 변경 금지** (L05/L13/L16). 4 카테고리 정의 근거는 `docs/design-decisions.md` §1.

**Boundary safe-shortcut** [L13] — `fuse_selective` 헤드 (`src/cacheblend/fusor.py:150-159`):

```python
# ── Boundary safe-shortcut [L13] ───────────────────────────────────────
if recompute_ratio == 0:
    out = fuse_full_reuse(layerwise_model, chunks, kv_store, ...)
    ...
if recompute_ratio >= 1:
    out = fuse_full_recompute(layerwise_model, chunks, ...)
    ...
if len(chunks) <= 1:
    out = fuse_full_recompute(layerwise_model, chunks, ...)
    ...
```

3 boundary case (`ratio==0`, `ratio>=1`, single-chunk) 모두 다른 fuse 함수로 즉시 dispatch → 코드 경로 동일화 → max_diff = 0 보장. Phase 3 의 test_ratio_0 / test_ratio_1 둘 다 max_diff=0.000e+00 달성 (실측, `reports/phase-3-attachments/pytest.log`).

`fuse_full_reuse` 도 동일 패턴 (`src/cacheblend/fusor.py:55-60`): `len(chunks) <= 1 → fuse_full_recompute`. `fuse_prefix_cache` 도 (`src/cacheblend/fusor.py:310-312`).

### 1.2 LayerwiseModel + k_proj forward-hook (Phase 1)

`src/cacheblend/model.py` (240 lines). HF Mistral-7B 의 forward 를 layer-by-layer 분리. 7 메서드:

| 메서드 | 위치 | 역할 |
|---|---|---|
| `__init__` | `model.py:50-77` | `attn_implementation="eager"` 의무, k_proj hook 32 layer 모두 install |
| `_install_k_proj_hooks` | `model.py:81-92` | 모든 layer 의 `self_attn.k_proj` 에 forward-hook → `self._pre_rope_k[layer_idx] = output.detach()` |
| `embed_tokens` | `model.py:96-98` | wrap `model.model.embed_tokens` |
| `compute_position_embeddings` | `model.py:100-104` | wrap `model.model.rotary_emb` (cos, sin shared) |
| `build_causal_mask` | `model.py:106-122` | wrap HF `_update_causal_mask` |
| `prefill_layer` | `model.py:124-156` | 단일 decoder layer forward, DynamicCache update in-place |
| `final_norm_and_lm_head` | `model.py:158-161` | RMSNorm → LM head |
| `forward_layerwise` | `model.py:165-218` | orchestrate 모든 메서드 + `self._pre_rope_k = {}` 초기화 |
| `get_pre_rope_k` | `model.py:220-230` | hook capture dict 에서 lookup |

`__init__` 의 attn_implementation:

```python
# src/cacheblend/model.py:64-69 (eager 의무)
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
).to(self.device).eval()
```

**검증** (Phase 1 test_layerwise.py): `forward_layerwise(...).logits` vs `model(input_ids=...).logits` → `max_diff = 0.000e+00`, argmax_match=1.0000 (`reports/phase-1-attachments/pytest.log`). 32 layer 의 pre-RoPE K hook 도 모두 `max_diff = 0.000e+00` (bit-exact, 같은 log).

### 1.3 KVStore (sync + async prefetch)

`src/cacheblend/kv_store.py` (126 lines). chunk_id → {"K": list[Tensor], "V": list[Tensor]} OrderedDict-backed LRU.

**Sync API (Phase 2)**:
- `put` (`kv_store.py:51-57`)
- `get` (`kv_store.py:43-50`) — `move_to_end` for LRU touch
- `has` (`kv_store.py:40-42`) — cached OR inflight 둘 다 인정
- `evict_lru` (`kv_store.py:69-73`) — `popitem(last=False)`

**Async API (Phase 4)**:
- `prefetch_chunk(chunk_id, loader_fn)` (`kv_store.py:96-114`) — `ThreadPoolExecutor.submit(loader_fn)` → Future. **idempotent**: 이미 cached 면 completed Future, 이미 inflight 면 같은 Future 재사용.
- `_ensure_executor` (`kv_store.py:91-95`) — lazy init.
- `get` 이 inflight Future 자동 block (`kv_store.py:46-50`).

```python
# src/cacheblend/kv_store.py:96-114
def prefetch_chunk(self, chunk_id, loader_fn) -> Future:
    if chunk_id in self._cache:
        f: Future = Future()
        f.set_result(self._cache[chunk_id])
        return f
    if chunk_id in self._inflight:
        return self._inflight[chunk_id]
    executor = self._ensure_executor()
    fut = executor.submit(loader_fn)
    self._inflight[chunk_id] = fut
    return fut
```

**Phase 4 검증** (test_pipeline.py:test_pipelined_eq_unpipelined): pipelined vs unpipelined logits `max_diff = 0.000e+00` (RECOMPUTE_PATH bound 1e-3 안에서 bit-exact, `reports/phase-4-attachments/pytest.log`).

### 1.4 RoPE shift (Phase 2)

`src/cacheblend/rope.py` (65 lines). pre-RoPE K (k_proj output) 를 target positions 로 shift.

```python
# src/cacheblend/rope.py:33-65
def apply_rope_shift(K_pre_rope, target_positions, layerwise_model):
    inner = layerwise_model._inner
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim
    K_heads = _kproj_to_heads(K_pre_rope, num_kv_heads, head_dim)
    cos, sin = inner.rotary_emb(K_heads, target_positions)
    dummy_q = torch.zeros_like(K_heads)
    _q, K_rot = apply_rotary_pos_emb(dummy_q, K_heads, cos, sin)
    return _heads_to_kproj(K_rot)
```

`_kproj_to_heads` (`rope.py:17-25`) 는 HF MistralAttention 의 reshape 와 일치: `(B, S, num_kv_heads*head_dim) → (B, num_kv_heads, S, head_dim)`. **Phase 2 검증**: `test_rope_shift_correctness` layer-0 max_diff = 0.000e+00 (IDENTICAL_PATH, `reports/phase-2-attachments/pytest.log`).

`fuse_full_reuse` 와 `fuse_selective` 는 `apply_rope_shift` 를 직접 호출하지 **않는다**. 대신 pre-RoPE K 를 hook 으로 inject 하면 HF MistralAttention 의 `apply_rotary_pos_emb` 가 fused sequence 의 blended global positions 으로 자동 RoPE 적용 — paper §4 의 RoPE shift 가 hook injection 만으로 달성됨 (`fusor.py:115-117` docstring).

### 1.5 Chunker + precompute (Phase 2)

`src/cacheblend/chunker.py` (75 lines):
- `Chunk` dataclass (`chunker.py:21-29`): text + token_ids + chunk_id (16-char SHA256 prefix) + is_cached
- `chunk_texts(tokenizer, texts)` (`chunker.py:41-54`)
- `fused_input_ids(chunks, device)` (`chunker.py:57-65`)
- `chunk_offsets(chunks)` (`chunker.py:68-75`) — [(start, end), ...]

`src/cacheblend/precompute.py` (61 lines):
- `_install_v_proj_hooks` (`precompute.py:14-30`) — temporary V capture per layer
- `precompute_chunk_kv(layerwise_model, chunk)` (`precompute.py:33-61`) — chunk 단독 forward (positions 0..L-1) → 32 layer K (Phase 1 hook) + V (임시 hook) 반환

### 1.6 fuse_full_recompute / fuse_full_reuse (Phase 2)

`fuse_full_recompute(layerwise_model, chunks, return_layerwise_output=False)` (`fusor.py:24-37`): standard prefill, no KV reuse. `forward_layerwise` 1 회 호출.

`fuse_full_reuse(layerwise_model, chunks, kv_store, return_layerwise_output=False)` (`fusor.py:40-117`):
1. Single-chunk → fuse_full_recompute 직접 (boundary, `fusor.py:55-60`).
2. Multi-chunk: 모든 layer × 2 (k_proj, v_proj) = 64 hook install. 각 hook 은 fresh output 을 stored K/V 로 **전부 교체** (`fusor.py:99-110`).
3. `try/finally` 로 handle cleanup (`fusor.py:112-117`).

**Phase 2 결과**: single-prefix 시 max_diff = 0 (IDENTICAL via boundary). 3-chunk multi 시 `max_diff_overall = 8.568, argmax_match = 0.9348` (Phase 3 의 baseline, `reports/phase-2-attachments/multi_chunk_divergence.json`).

### 1.7 HKVD + fuse_selective (Phase 3, 단일-pass)

`src/cacheblend/hkvd.py` (77 lines):

```python
# src/cacheblend/hkvd.py:23-50  (LMCache blender.py:89-91 와 동일 식)
def kv_deviation(K_new, K_old) -> torch.Tensor:
    K_new_f = K_new.to(torch.float32)
    K_old_f = K_old.to(torch.float32)
    diff = (K_new_f - K_old_f) ** 2
    if diff.ndim == 3:
        return diff.sum(dim=-1).squeeze(0)  # → (S,)
    elif diff.ndim == 2:
        return diff.sum(dim=-1)
```

```python
# src/cacheblend/hkvd.py:58-77  (LMCache blender.py:94-101 와 동일)
def select_top_k(deviations, ratio) -> torch.Tensor:
    total_len = deviations.shape[0]
    topk_num = int(total_len * ratio)
    topk_num = max(topk_num, 1)
    topk_num = min(topk_num, total_len)
    top_indices = torch.topk(deviations, k=topk_num, largest=True).indices
    top_indices, _ = torch.sort(top_indices)  # causal-friendly resort
    return top_indices
```

`fuse_selective(model, chunks, store, recompute_ratio=0.15, check_layer=1)` (`fusor.py:119-247`) — **단일-pass 디자인**:

1. **Boundary safe-shortcut** (`fusor.py:150-159`): ratio==0 / >=1 / single-chunk → 다른 fuse 함수.
2. K_stored / V_stored 를 KVStore 에서 build (`fusor.py:174-190`).
3. **Mutable state** (`fusor.py:193`): `state = {"hkvd_indices": None}`.
4. **check_layer 의 k_proj observe-only hook** (`fusor.py:195-200`):
    ```python
    def check_layer_hook(_m, _inp, output):
        deviations = kv_deviation(output, K_stored[check_layer])
        state["hkvd_indices"] = select_top_k(deviations, recompute_ratio)
        return None  # observation only
    ```
5. **check_layer+1..end 의 k_proj/v_proj selective hook** (`fusor.py:202-213`):
    ```python
    def make_selective_hook(layer_idx, stored):
        def hook(_m, _inp, output):
            hkvd = state["hkvd_indices"]
            result = output.clone()
            mask = torch.ones(total_seq, dtype=torch.bool, device=output.device)
            mask[hkvd] = False
            result[:, mask, :] = stored[layer_idx][:, mask, :]
            return result
        return hook
    ```
6. Hook attach (`fusor.py:215-234`): check_layer k_proj observe + (n_layers - check_layer - 1) × 2 selective.
7. forward_layerwise 1 회 호출 → logits + past_kv (`fusor.py:236-244`).

**Layer 0..check_layer-1**: hook 없음, fresh forward.
**Layer check_layer**: observe-only, output unchanged.
**Layer check_layer+1..end**: HKVD 위치 fresh, non-HKVD 위치 stored. HF apply_rotary_pos_emb 가 위에서 RoPE 적용 → blended global positions 으로 자동 RoPE shift.

### 1.8 fuse_selective_pipelined / fuse_prefix_cache (Phase 4)

`fuse_selective_pipelined(model, chunks, store, ratio=0.15, check_layer=1, prefetch=True)` (`fusor.py:249-291`):
- prefetch=True 시 chunk 별 `kv_store.prefetch_chunk(...)` (RAM tier thunk).
- 이후 `fuse_selective` 호출. `kv_store.get` 이 inflight Future 자동 block.

`fuse_prefix_cache(model, chunks, store)` (`fusor.py:294-359`):
- Single-chunk → fuse_full_recompute (boundary, `fusor.py:310-312`).
- Multi-chunk: 매 layer k_proj/v_proj hook 이 `output[:, first_start:first_end, :] = K_cached_first` 만 override. 이외 위치 unchanged.
- 첫 chunk 의 chunk-local positions == fused positions (둘 다 0..L_first-1) → RoPE shift 불필요.

### 1.9 Runner 5종 (Phase 5)

`src/cacheblend/runners.py` (353 lines). mydata harness `CacheBlendRunner` ABC 위에 v4 의 5 Runner 추가.

| Runner | 위치 | wrap |
|---|---|---|
| `_RunnerBase` (공통) | `runners.py:59-170` | dispatch + CPU stub mode + `_greedy_decode_from_prefill` + `_build_chunks` |
| `FullRecomputeRunner` | `runners.py:172-189` | HF standard `model(input_ids, use_cache=True)` |
| `FullReuseRunner` | `runners.py:191-244` | `fuse_full_reuse` (Phase 2) |
| `PrefixCacheRunner` | `runners.py:247-283` | `fuse_prefix_cache` (Phase 4) |
| `CacheBlendRunner` | `runners.py:286-327` | `fuse_selective` (Phase 3) |
| `GradualV4Runner` | `runners.py:330-345` | (Phase 8 stub) — `_run_prefill_and_generate` raises NotImplementedError |

**CPU stub mode** (`runners.py:95-100`):
```python
def generate(self, max_new_tokens: int = 32):
    if self.model is None:
        return _stub_generation()  # text="", ttft=0, total=0, n=0
    return self._run_prefill_and_generate(max_new_tokens=max_new_tokens)
```

**L31 fix 호환 분기** (`runners.py:67-79`):
```python
def __init__(self, model=None, tokenizer=None):
    if _HARNESS_AVAILABLE and model is not None:
        super().__init__(model=model, tokenizer=tokenizer)
    else:
        self.model = model
        ...
```

**GradualV4Runner stub** (`runners.py:330-345`):
```python
class GradualV4Runner(_RunnerBase):
    """Phase 8 gradual filtering — multi check_layer schedule.
    Phase 5: stub-only. Real Phase 8 implementation will fill in
    `_run_prefill_and_generate`.
    """
    def __init__(self, model=None, tokenizer=None, schedule=None):
        super().__init__(model=model, tokenizer=tokenizer)
        self.schedule = schedule
    def _run_prefill_and_generate(self, max_new_tokens: int):
        raise NotImplementedError(
            "GradualV4Runner.generate: real implementation in Phase 8. "
            ...
        )
```

### 1.10 LMCache 와의 차이

(상세는 `docs/lmcache-analysis.md` 참조. v4 진행 중 §Q2(c.5) 를 정량 데이터로 보강 — Phase 3.)

| 항목 | LMCache (`external/LMCache`) | v4 | 정당화 |
|---|---|---|---|
| `compute_layer` 의 token slicing | `process_qkv` 내부, attention 직전 slicing (`blender.py:103-105`) | Slicing 없음. 모든 토큰 q/k/v fully 계산, hook override 만 | TTFT 비목표 (L27, design-decisions.md §3) |
| Pre-RoPE K 저장 | ✗ (vLLM post-RoPE 캐시 사용) | ✓ (`KVStore` `precompute_chunk_kv` 결과는 pre-RoPE) | paper §4 충실 (design-decisions.md §11) |
| RoPE shift on retrieve | ✗ (`fused_encode` 정의는 있으나 `process_qkv` 미호출) | ✓ HF apply_rotary_pos_emb 가 hook injection 위에서 자동 적용 | position mismatch 노이즈 회피 |
| KV deviation 공식 | `sum((k_new−k_old)^2, dim=1)` fp32, K only, no normalize (`blender.py:89-91`) | 동일 (`hkvd.py:23-50`) | LMCache 와 동일 정렬 [Phase 3.7 검증] |
| Top-K rule | `int(N*ratio)`, `max(1)`, `topk` → resort (`blender.py:94-101`) | 동일 (`hkvd.py:58-77`) | 동일 |
| Boundary safe-shortcut (ratio=0/1) | 없음 (`max(int(ratio*N), 1)` → ratio=0 이라도 1 token 선택) | `fuse_selective` 첫 줄 분기 (`fusor.py:150-159`) | L13 |
| Check_layers semantics | List 받지만 `recomp_ratios[0]` hardcoded (`blender.py:97`) + per-layer threshold TODO (`blender.py:43`) | Phase 3: 단일 (LMCache equivalent). Phase 8: multi-CL gradual (미구현, §3 참조) | Phase 8 신규 contribution |
| Storage tiering | 5 backends: LocalCPU/LocalDisk/P2P/Remote/GDS (`storage_backend/__init__.py:15-20`) | 단일 in-RAM dict + StorageProfile cost model (Phase 4 simulated) | TTFT 비목표 → 실 I/O 불필요 |

**정량 검증** (`benchmarks/lmcache_hkvd_compare.py`, Phase 3): 3 chunks × 46 tokens × layer-1 의 pre-RoPE deviation (v4) vs post-RoPE deviation (LMCache 등가):
- Spearman ρ = **0.999137**
- Pearson r = **1.000000**
- Top-15% HKVD index overlap = **100% (6/6)**

데이터: `reports/phase-3-attachments/lmcache_hkvd_compare.json`. 이론 근거: RoPE 가 orthogonal rotation per position → squared-L2 invariant under common rotation → ranking 보존.

---

## 2. 실험 결과

### 2.1 Tolerance tests (Phase 1~4)

각 Phase 의 핵심 numeric correctness 결과:

| Phase | Test | Tolerance | max_diff | argmax_match | Verdict |
|---|---|---|---|---|---|
| 1.1 | `test_layerwise_matches_standard` | SAME_SHAPE | **0.000e+00** | 1.0000 | bit-exact |
| 1.2 | `test_kv_extraction` (32 layers) | SAME_SHAPE | min/median/max **0** | n/a | 32 layer 모두 bit-exact |
| 2.1 | `test_rope_shift_correctness` | IDENTICAL_PATH | **0.000e+00** | n/a | layer-0 |
| 2.2 | `test_full_recompute_sanity` | IDENTICAL_PATH | **0.000e+00** | 1.0000 | |
| 2.3 | `test_full_reuse_single_prefix` | MIXED_SHAPE | **0.000e+00** | 1.0000 | boundary safe-shortcut [L13] 작동 |
| 2.4 | `test_full_reuse_multi_chunk_divergence` | (측정만) | **8.568** overall, 0.390 mean | 0.9348 | per-chunk last-token: [0.014, 2.97, 5.68] |
| 3.1 | `test_ratio_0_eq_full_reuse` | IDENTICAL_PATH | **0.000e+00** | 1.0000 | |
| 3.2 | `test_ratio_1_eq_full_recompute` | IDENTICAL_PATH | **0.000e+00** | 1.0000 | |
| 3.3 | `test_selective_reduces_divergence` | (≥15% reduction) | n/a | n/a | full_reuse_L2 0.6149 → selective_L2 0.4465 = **27.39% reduction** |
| 3.4 | `test_mask_is_standard_causal` | (Q×Q lower-tri) | n/a | n/a | 46×46 sub-block lower-tri (K=47, +1 cache slot, L38) |
| 4.1 | `test_pipelined_eq_unpipelined` | RECOMPUTE_PATH | **0.000e+00** | 1.0000 | |
| 4.2 | `test_prefix_cache_eq_full_recompute` | MIXED_SHAPE | **2.734e-02** | 1.0000 | argmax 보존, FP16 작은 noise |
| 4.3 | `test_loading_controller_monotone` | (CPU only) | n/a | n/a | RAM 0.150 → SLOW 0.495 strict 단조 |

**해석**: boundary 들 (1.1/1.2/2.1/2.2/2.3/3.1/3.2/4.1) 은 max_diff = 0 — 의도된 코드 경로 동일화 가 작동. 3.3 의 27.39% reduction 은 target 15% 의 1.8× — selective recompute 의 quality 회복력 입증.

### 2.2 Phase 3 long-chunk sweep (15 cell)

`benchmarks/long_chunk_sanity.py`. chunk_B {60, 120, 240} × ratio {0.05, 0.10, 0.15, 0.20, 0.50}:

| chunk_B | total_seq | ratio | reduction% | argmax_match |
|---:|---:|---:|---:|---:|
| 60 | 180 | 0.05 | 11.2 | 96.1% |
| 60 | 180 | 0.10 | 15.2 | 96.7% |
| 60 | 180 | 0.15 | 24.7 | 97.8% |
| 60 | 180 | 0.20 | 36.1 | 98.3% |
| 60 | 180 | 0.50 | 67.1 | 98.9% |
| 120 | 360 | 0.05 | 14.7 | 98.3% |
| 120 | 360 | 0.10 | 27.4 | 99.2% |
| 120 | 360 | 0.15 | **35.4** | 99.2% |
| 120 | 360 | 0.20 | 43.3 | 99.4% |
| 120 | 360 | 0.50 | 70.3 | 99.4% |
| 240 | 720 | 0.05 | 21.3 | 99.7% |
| 240 | 720 | 0.10 | 35.6 | 99.9% |
| 240 | 720 | 0.15 | **44.2** | 99.9% |
| 240 | 720 | 0.20 | 52.1 | 99.9% |
| 240 | 720 | 0.50 | 79.5 | 100.0% |

**해석**:
- `argmax_match` 모든 cell 96~100% — top-1 prediction 거의 보존.
- chunk 길수록 selective 효과 큼 (240 의 ratio=0.05 만으로 21.3% reduction 達成).
- **Mistral elbow 약함**: ratio→reduction 곡선이 단조 증가, 명확한 elbow 없음. paper §4.3 의 0.10~0.20 elbow 보고 대비 weak. v4-lessons L14 의 "모델 특성" 결론과 일관.

데이터: `reports/phase-3-attachments/long_chunk_sweep.{md,json}`.

### 2.3 LMCache HKVD ranking 비교 (Phase 3)

`benchmarks/lmcache_hkvd_compare.py`.

| Metric | Value |
|---|---|
| Spearman ρ (per-token rank) | **0.999137** |
| Pearson r (magnitudes) | 1.000000 |
| Top-15% HKVD index overlap | **100% (6/6)** |
| max relative per-token diff | 18.3% (FP16/FP32 cast 노이즈) |
| n_chunks / total_seq / check_layer | 3 / 46 / 1 |

**해석**: v4 의 pre-RoPE deviation 과 LMCache 의 post-RoPE deviation 은 ranking 동일 (Spearman ≈ 1) — RoPE 의 orthogonal rotation 이 squared-L2 거리를 보존하기 때문. 절대 magnitude 약간 다르지만 top-K 선택 동일. `docs/lmcache-analysis.md` §Q2(c.5) 정량 보강 완료.

### 2.4 Phase 4 TTFT 측정 (참고용, gate 아님 [L27])

`benchmarks/ttft.py`. RTX 3090 24GB, median ms over 3 timed runs (1 warmup):

| chunk_B | total_seq | full_recompute | full_reuse | prefix_cache | selective | selective_pipelined |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 180 | 55.83 | 59.39 | 56.78 | 76.06 | 75.37 |
| 120 | 360 | 102.55 | 106.00 | 104.16 | 118.51 | 118.80 |
| 240 | 720 | 199.12 | 202.23 | 200.38 | 213.40 | 213.36 |

**Speedup vs full_recompute** (>1.0×=faster):

| chunk_B | full_reuse | prefix_cache | selective | pipelined |
|---:|---:|---:|---:|---:|
| 60 | 0.94× | 0.98× | 0.73× | 0.74× |
| 120 | 0.97× | 0.98× | 0.87× | 0.86× |
| 240 | 0.98× | 0.99× | 0.93× | 0.93× |

**해석**: hook-injection 디자인은 q/k/v projection 을 모든 토큰에 fully 수행 → 연산량 절감 없음. selective 는 hook overhead + check_layer deviation 계산으로 ~30% 느림 (60-token 에서). Pipelined ≈ unpipelined (RAM tier 에서 prefetch 효과 없음). v4 비목표 — quality milestone (Phase 3 27.4% reduction) 이 headline.

데이터: `reports/phase-4-attachments/ttft.{json,md}`.

### 2.5 Phase 6 Mistral-7B Musique 200 (4 runner × 200 sample)

`benchmarks/run_phase6.py`. 모델: `mistralai/Mistral-7B-Instruct-v0.2`, FP16. mydata cacheblend_fig12 prompts.jsonl (`SHA = 791e1cf5…3a8e21`, 200 sample).

**Sub-phase 6c (full)**:

| Runner | n | f1_mean | f1_std | rouge_l_mean | ttft_s_mean | total_s_mean | f1=0 / f1≈1 |
|---|---:|---:|---:|---:|---:|---:|---|
| FullRecomputeRunner | 200 | **0.2542** | 0.274 | 0.203 | 0.246 | 0.872 | 67/10 |
| FullReuseRunner | 200 | 0.1432 | 0.207 | 0.079 | 0.251 | 1.125 | 102/4 |
| PrefixCacheRunner | 200 | 0.2542 | 0.274 | 0.203 | 0.244 | 0.867 | 67/10 |
| **CacheBlendRunner** (0.15, CL=1) | 200 | **0.2222** | 0.264 | 0.143 | 0.263 | 0.979 | 74/9 |

**Diff + paired bootstrap CI** (n_paired=200, n_bootstrap=1000, confidence=0.95, seed=42):

| | 값 |
|---|---|
| `f1_diff_cb_vs_full` | **-0.0320** |
| `f1_diff_cb_vs_reuse` | **+0.0790** |
| `ci_low_cb_vs_reuse` | **+0.0455** |
| `ci_high_cb_vs_reuse` | +0.1141 |

**해석**:
- **CacheBlend > FullReuse 95% CI 통계적 우위** [+0.0455, +0.1141] — paper §4 의 quality preservation claim 검증.
- CacheBlend < FullRecompute 0.0320 F1 (catastrophic 차단 -0.05 안에서 안전).
- PrefixCache ≡ FullRecompute (positions 0..L_first-1 동일 path).
- F1 absolute = 0.2542 (FullRecompute) — paper Figure 12 의 absolute Mistral F1 수치는 비공개이거나 우리 setup (`docs/figure12_like_disclosure.md` 의 12 차이점) 과 정확 비교 불가. **확인 못함**: 0.2542 가 paper 와 같은 ballpark 인지 직접 비교 데이터 없음.

**Sub-phase 진행**:
- 6a (n=20): 5/5 PASS, ci_low_cb_vs_reuse=+0.0086.
- 6b (n=50): 3/3 PASS, f1_diff_cb_vs_reuse=+0.1189.
- 6c (n=200): 3/3 PASS, ci_low=+0.0455.

데이터: `reports/phase-6{a,b,c}-attachments/{results.jsonl, summary.json}`.

---

## 3. Gradual filtering 진행 여부

**단정: Phase 6 까지 미진행. Phase 8 로 분리.**

### 3.1 미진행 근거 (코드 인용)

1. `GradualV4Runner._run_prefill_and_generate` 가 stub — `NotImplementedError` raise (`runners.py:340-345`):

   ```python
   # src/cacheblend/runners.py:340-345
   def _run_prefill_and_generate(self, max_new_tokens: int):
       raise NotImplementedError(
           "GradualV4Runner.generate: real implementation in Phase 8. "
           "For Phase 5 plumbing (CPU stub mode), `model=None` is the supported path."
       )
   ```

2. Phase 6 의 4 runner 는 **모두 flat single check_layer**:
   - FullRecomputeRunner, FullReuseRunner, PrefixCacheRunner 는 selective recompute 자체 안 씀.
   - CacheBlendRunner 는 `recompute_ratio=0.15, check_layer=1` (Phase 6 driver `benchmarks/run_phase6.py` argparse default 와 호출 사이트 모두 단일 정수).

3. `fuse_selective` 가 `check_layer: int` 단수 파라미터 (`fusor.py:124`). multi-check 으로 확장하려면 시그니처 변경 + state machine 추가 필요 — 현재 미구현.

4. `src/cacheblend/gradual.py` 의 `LayerProfiler`, `SchedulePlanner`, `GradualSchedule` 도 Phase 0 stub 그대로 (Phase 8 에서 채울 예정, 본 문서 작성 시점 기준 변경 없음 — 확인: `grep -n "raise NotImplementedError" src/cacheblend/gradual.py` 결과 존재 시 stub).

### 3.2 Phase 8 로 분리 — 4-step interactive

`tasks/phase-8-gradual.md` 정의:

- **Step 1 (자동, ~$3)**: per-model layer profiling. 3 metric × 2 model = 6 plots (`tasks/phase-8-gradual.md:25-66`):
  - (a) Top-15% mass
  - (b) Spearman rank corr (KV deviation rank vs forward attention deviation rank)
  - (c) Information gain (partial-forward, 비용 ↑)
- **사용자 검토 ① + 프롬프트** (`tasks/phase-8-gradual.md:70-108`): Budget 정의 (A/B/Hybrid), 메트릭별 check_layers 채택, threshold 미세조정.
- **Step 2 (CPU, $0)**: schedule 생성 (linear_decay vs uniform_baseline), 최대 48 schedule (`tasks/phase-8-gradual.md:112-130`).
- **Step 3 (GPU, ~$15)**: F1 heatmap (Mistral + Llama-8B, 100 sample/schedule).
- **Step 4 (자동, ~$1)**: LMCache 단순 check_layer=1 flat schedule 과 head-to-head 비교.

**비용 cap**: $25 (`tasks/phase-8-gradual.md:3`).

### 3.3 Phase 7 (Llama) 가 먼저

이유: gradual filtering 의 효과는 모델 dependent 가능성 높음. Mistral 단일 데이터로 결론 짓기 약함. Phase 7 에서 Llama-8B 의 quality baseline 확보 후 Phase 8 의 6 plot 비교가 의미 있음.

`tasks/phase-8-gradual.md:11-13` 도 명시: "Mistral-7B-Instruct-v0.2 + Llama-3.1-8B-Instruct (Llama-70B 제외 — 비용)". → Phase 7 의 Llama-8B 결과 (7a/7b/7c) 가 Phase 8 의 정량 baseline 으로 사용됨.

### 3.4 현재까지 단서 — gradual 의 motivation

Phase 3 long-chunk sweep (§2.2) 의 **Mistral elbow 약함** 관찰:
- ratio 0.05 → 0.50 의 reduction% 곡선이 단조 증가, 명확한 elbow 없음 (chunk_B=60 11→15→25→36→67%, chunk_B=240 21→36→44→52→80%).
- 단일 ratio 로는 chunk 길이 / model 별 최적값을 잡기 어려움 — multi-check 으로 layer 별 다른 ratio 적용 시 budget 효율적 사용 가능성.
- v4-lessons L14 의 "모델 특성" 결론과 일관. **그 외 정량 데이터 없음** — Phase 8 Step 1 의 layer profiling 이 본격 수치.

---

## 4. 다음 단계

### 4.1 Phase 7 사전점검

`tasks/phase-7-llama.md`:
- **7a/7b/7c**: Llama-3.1-8B-Instruct (FP16) 동일 sub-phase 구조 (20/50/200), 동일 driver `benchmarks/run_phase6.py` (model 인자만 변경). 24GB GPU OK.
- **7d**: Llama-3.1-70B-Instruct (8-bit bitsandbytes), 200 sample full only.
- **누적 cap $25** (Phase 6 종료 시 $1.09 → Phase 7 종료 시 ≤ $25).
- **80GB GPU 필요 (7d 전용)**: vast.ai search filter `gpu_ram >= 80` (A100 80GB / H100 PCIe). [L37]

### 4.2 Phase 8 사전점검

위 §3.2 의 4-step interactive flow.
- $25 cap.
- 사용자 검토 4회 (자동 진행 안됨).
- 산출물: `reports/phase-8-step1-attachments/{model}_{metric}.png × 6, profile_data.json` + `reports/phase-8-step2-attachments/schedules.json` + `reports/phase-8-step3-attachments/heatmap` + `reports/phase-8-step4-attachments/lmcache_compare`.

### 4.3 Pod 처리

현재 `vastai instance 36296967` (RTX 3090) keep alive. Phase 7 진입 시:
- 7a/b/c 즉시 실행 가능 (24GB GPU 충분, conda env `cb` + Mistral cache 보존).
- 7d 진입 시 80GB GPU instance 별도 부팅.
- Phase 8 (CPU $0 step 들 + GPU step) 진입 시 keep / stop / search 결정 필요.

---

## 부록 — 인용 횟수 self-check

본 문서 내 file:line 인용 (`*.py:line` 또는 `*.md:line` 형식):

1. `src/cacheblend/tolerance.py:19-30, 42-93`
2. `src/cacheblend/fusor.py:24-37, 40-117, 55-60, 99-110, 112-117, 119-247, 124, 150-159, 174-190, 193, 195-200, 202-213, 215-234, 236-244, 249-291, 294-359, 310-312`
3. `src/cacheblend/model.py:50-77, 64-69, 81-92, 96-98, 100-104, 106-122, 124-156, 158-161, 165-218, 220-230`
4. `src/cacheblend/kv_store.py:40-42, 43-50, 46-50, 51-57, 69-73, 91-95, 96-114`
5. `src/cacheblend/rope.py:17-25, 33-65`
6. `src/cacheblend/chunker.py:21-29, 41-54, 57-65, 68-75`
7. `src/cacheblend/precompute.py:14-30, 33-61`
8. `src/cacheblend/hkvd.py:23-50, 58-77`
9. `src/cacheblend/runners.py:59-170, 67-79, 95-100, 172-189, 191-244, 247-283, 286-327, 330-345, 340-345`
10. `src/cacheblend/controller.py` (Phase 4 결과 §2.1 4.3 참조 — 표만)
11. `external/LMCache/lmcache/v1/compute/blend/blender.py:43, 89-91, 94-101, 97, 103-105` (LMCache 비교 표)
12. `external/LMCache/lmcache/v1/storage_backend/__init__.py:15-20`
13. `tasks/phase-8-gradual.md:3, 11-13, 25-66, 70-108, 112-130`
14. `reports/phase-{1,2,3,4,6{a,b,c}}-attachments/{...}.{json,md,jsonl}` (실험 데이터 cross-ref)
15. `docs/{lmcache-analysis,design-decisions,figure12_like_disclosure,notes/v5-lessons}.md` cross-refs

총합 **40+ file:line 인용** (내부 `*.py:line` 만 약 35; LMCache 외부 + 보고서 데이터 + cross-ref 포함 시 50+).

## 자신 없는 영역 (확인 못함 명시)

- **Phase 6 F1 절대값 0.2542 의 paper 와의 비교**: paper Figure 12 의 Mistral-7B Musique 절대 F1 수치는 비공개 또는 mydata `docs/figure12_like_disclosure.md` 의 12 차이점 (embedding 모델, retrieval similarity, GPT-4 query, sample 수, etc.) 과 직접 비교 불가. 0.2542 가 paper 와 같은 ballpark 인지 **확인 못함**.
- **LMCache 의 RoPE shift hot-path 사용 여부**: `fused_encode` 정의는 `external/LMCache/lmcache/v1/compute/positional_encoding.py:64-76` 에 있으나 `process_qkv` 에서 호출 안 함을 grep 으로 확인 (`docs/lmcache-analysis.md` §Q2c). LMCache 의 다른 코드 경로 (`integration/vllm/...` 등) 에서 호출되는지는 **확인 못함**.
- **TTFT 측정의 노이즈**: 1 warmup + 3 timed 중 median 만 보고. ±1ms variance 가능. 크리티컬 차이 (selective ~30% 느림) 는 안정적이나 ms 단위 sub-1ms 차이는 무의미.
- **Phase 4 LoadingController 의 multiplier 값** (1.30 / 1.80 / 3.30): paper §6 의 storage 별 load cost 비율 spirit 에 맞춘 추정값. 구체 출처는 없음 (`docs/design-decisions.md` 에 추가하지 않은 상태). 단순 illustrative — Phase 7 이후 실제 NVMe 측정 값으로 교체 가능.
- **Phase 6 의 `ttft_seconds_mean` 0.246s 가 prefill TTFT 인지 generate 시작 시간인지**: 보고서에 ttft_seconds 필드 schema 가 갖추어져 있으나 실 측정 의미는 hook-injection 기반에서 prefill time 과 거의 동일. 정확한 분해는 **확인 못함**.

## v5-lessons 신규

**없음** — 본 문서는 종합/분석 작업이므로 신규 lesson 발견 없음. 기존 L31~L40 (10개) 그대로 (`docs/notes/v5-lessons.md`).
