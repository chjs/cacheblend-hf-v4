# LMCache CacheBlend 백엔드 분석 — Phase 0 baseline

> 우리 v4 디자인이 LMCache (the production system, not the paper) 와 어디서 다르고, 어디서 같은지를 코드 기준으로 확인. Phase 3 HKVD metric 비교 + Phase 8 gradual filtering 비교의 baseline.
> 모든 인용은 `external/LMCache/` 의 실제 파일 경로 + 라인 번호. 이 문서는 LMCache HEAD `7657836e070b9211ed43294e13f1e4c81716dcf6` (2026-05-07 12:56:38 +0800, "[MP]: Add batch operations to Mooncake L2 adapter (#3172)") 기준이며, depth=1 clone.

## 0. 분석 범위

LMCache 의 KV blending 백엔드 (`lmcache/v1/compute/blend/`, `lmcache/v1/compute/models/`, `lmcache/v1/compute/positional_encoding.py`) 만 본다. KV transfer 자체 (P2P/NIXL), serialization, vLLM integration 은 v4 범위 밖.

핵심 entry point 는 `LMCBlender.process_qkv` (blender.py:59) 로, 매 layer 마다 호출되어 (a) RoPE 적용 (b) 지정 layer 에서 top-K 선택 (c) 캐시 K/V 의 해당 indices 만 fresh 값으로 덮어쓰기 를 수행한다.

---

## Q1 — `compute_layer` 에서 어떻게 token slicing 을 하는가?

**핵심 발견: LMCache 는 q_proj/k_proj/v_proj 단계에선 token slicing 을 하지 않는다. Token slicing 은 attention 직전 (post-RoPE, post-projection) 에 발생한다.**

`compute_layer` (`external/LMCache/lmcache/v1/compute/models/base.py:67-141`) 는 매 layer 마다 generator 의 한 step 을 진행한다. QKV projection 은 **모든 토큰**에 대해 수행된다:

> `external/LMCache/lmcache/v1/compute/models/base.py:99-107`
> ```python
> qkv, _ = layer.self_attn.qkv_proj(hidden_states)
> q, k, v = qkv.split(
>     [
>         layer.self_attn.q_size,
>         layer.self_attn.kv_size,
>         layer.self_attn.kv_size,
>     ],
>     dim=-1,
> )
> ```

`hidden_states` 는 layer 진입 시 **전체 token 길이** 를 갖는다 (layer norm + residual 모두 풀 길이). 즉 q/k/v 도 전체 토큰에 대해 계산된다.

이후 `self.blender.process_qkv(q, k, v, residual, idx, ...)` 를 호출:

> `external/LMCache/lmcache/v1/compute/models/base.py:112-114`
> ```python
> q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
>     q, k, v, residual, idx, attn_output, attn_metadata
> )
> ```

이 안에서 `layer_id in check_layers` 일 때만 token slicing 이 일어난다:

> `external/LMCache/lmcache/v1/compute/blend/blender.py:88-105`
> ```python
> if layer_id in self.common_metadata.check_layers:
>     diff_k = torch.sum(
>         (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
>     )
>     ...
>     topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
>     topk_num = max(topk_num, 1)
>
>     top_indices = torch.topk(diff_k, k=topk_num).indices
>     top_indices, _ = torch.sort(top_indices)
>
>     k, v = k[top_indices], v[top_indices]
>     q = q[top_indices]
>     residual = residual[top_indices]
> ```

선택된 `top_indices` 이후 layer 에선 (q/k/v/residual/attn_output 모두 짧아짐) **slicing 된 토큰만 attention 계산**. Slicing 된 token 은 attention output 도 짧고 (slicing 된 q × full K/V), MLP/post-norm 도 짧음.

| 단계 | LMCache 실제 동작 |
|---|---|
| `qkv_proj` (`q_proj`, `k_proj`, `v_proj`) | **전체 토큰**에 수행 |
| RoPE | `process_qkv` 내부 (전체 토큰) |
| KV deviation 계산 | check_layer 에서만, 전체 토큰 |
| Top-K slicing | check_layer 에서 실행, 이후 layer 에 propagate |
| `o_proj`, MLP, post-norm | check_layer 이후 layer 에선 short shape |

### v4 vs LMCache 비교

| 항목 | LMCache | v4 (Phase 3 hook-injection) |
|---|---|---|
| q/k/v 계산 범위 | 전체 토큰 (qkv_proj) | 전체 토큰 (q_proj/k_proj/v_proj 별도) |
| Slicing 시점 | post-RoPE, attention 직전 (`process_qkv`) | hook injection: cached 위치 출력만 override (slicing 아님) |
| 후속 layer 입력 길이 | check_layer 이후 짧아짐 | 항상 full length (TTFT 절감 없음, L27) |
| 코드 침습도 | qkv_proj output 직후 hook 1 곳 | k_proj/v_proj forward-hook 2 곳 + KVStore lookup |

### 디자인 차이의 함의

- **TTFT**: LMCache 는 check_layer 이후 진짜로 fewer-token compute → 실제 TTFT gain. 우리 v4 는 hook-injection 으로 hidden_states 길이 보존 → TTFT 절감 없음. 이는 v4 의 의도된 디자인 (L27, design-decisions.md §3).
- **검증 가능성**: v4 는 selective recompute 의 quality 만 확인. LMCache 와 비교할 때 max_diff 직접 비교는 metric 정의가 같아야 함 (Q2 참조).

---

## Q2 — KV deviation metric 의 정확한 공식?

논문 §4.2 의 HKVD (High KV Deviation) 식이 LMCache 코드에 어떻게 박혀있는지 4 sub-question 으로 분해.

### Q2(a) — Norm 종류

> `external/LMCache/lmcache/v1/compute/blend/blender.py:89-91`
> ```python
> diff_k = torch.sum(
>     (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
> )
> ```

**Squared L2 (sum, not mean), reduction along dim=1 (= feature axis = `num_kv_heads * head_size`).** 결과는 per-token scalar tensor `diff_k[token_id] = Σ_f (k_new[token_id, f] - k_old[token_id, f])^2`.

`.to(torch.float32)` 로 FP16/BF16 누적 노이즈 회피.

### Q2(b) — Layer 선택

> `external/LMCache/lmcache/v1/compute/blend/blender.py:88`
> ```python
> if layer_id in self.common_metadata.check_layers:
> ```

`check_layers` 는 `LMCBlendCommonMetadata.check_layers: List[int]`:

> `external/LMCache/lmcache/v1/compute/blend/metadata.py:11-18`
> ```python
> @dataclass
> class LMCBlendCommonMetadata:
>     check_layers: List[int]
>     recomp_ratios: Optional[List[float]] = None
>     thresholds: Optional[List[float]] = None
> ```

Config 에서 list 로 받는다:

> `external/LMCache/lmcache/v1/config.py:128-131`
> ```python
> "blend_check_layers": {
>     "type": list[int],
>     "default": None,
>     "env_converter": _to_int_list,
> },
> ```

**중요**: 코드는 multi-layer 를 받을 수 있지만 (Q4 참조) 실제 예제는 단일 layer 사용 — `LMCACHE_BLEND_CHECK_LAYERS=1` (`external/LMCache/examples/blend_kv_v1/blend.py:34`). 즉 LMCache 의 production default 는 **flat single-layer** 디자인. 이게 v4 Phase 8 비교 대상.

### Q2(c) — Pre-RoPE / Post-RoPE

> `external/LMCache/lmcache/v1/compute/blend/blender.py:84-91`
> ```python
> layer = self.layerwise_model.vllm_model.model.layers[layer_id]
> attn_layer = layer.self_attn
> q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)
>
> if layer_id in self.common_metadata.check_layers:
>     diff_k = torch.sum(
>         (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
>     )
> ```

**Fresh `k` 는 RoPE 후. Cached `old_k` 는 `self.gpu_connector.get_kv(layer_id)` 로 가져옴 (line 70) — vLLM cache 에 저장된 post-RoPE K (각 chunk 가 standalone prefill 될 때 chunk-local positions 으로 RoPE 가 이미 적용됨)**.

→ deviation 은 "fresh @ blended-positions" vs "cached @ chunk-local-positions" 의 **post-RoPE** 차이. Position 자체의 차이가 deviation 에 노이즈로 섞임.

LMCache 에 **RoPE shift 함수는 존재** (`FusedRope.fused_encode(old_positions, new_positions, k)`):

> `external/LMCache/lmcache/v1/compute/positional_encoding.py:64-76`
> ```python
> def fused_encode(self, old_positions, new_positions, k):
>     num_tokens = k.shape[0]
>     k = k.view(num_tokens, -1, self.head_size)
>     lmc_ops.rotary_embedding_k_fused(
>         old_positions,
>         new_positions,
>         k,
>         self.head_size,
>         self.cos_sin_cache.to(k.device),
>         self.is_neox_style,
>     )
>     k = k.view(num_tokens, -1)
>     return k
> ```

**그러나 `process_qkv` 는 `fused_encode` 를 호출하지 않는다** — 검색 `grep -n "fused_encode\|FusedRope" external/LMCache/lmcache/v1/compute/blend/blender.py` 결과 0 hit. (확인: `grep -n "fused_encode" external/LMCache/lmcache/v1/compute/blend/blender.py` → 출력 없음.) 구현은 있으나 deviation/blend hot path 에선 미사용.

### Q2(c.5) — RoPE invariance + 정량 검증 (Phase 3 추가)

위 RoPE shift 미실시 분석은 **이론적으로** 우리 v4 (pre-RoPE 도메인 deviation) 와 LMCache (post-RoPE 도메인 deviation, 단 chunk-local positions 의 cached old_k) 가 다를 수 있음을 시사한다. 그러나:

**핵심 관찰**: RoPE 는 position 별 **orthogonal rotation** 이다. 동일 token 의 K 에 동일 회전을 적용하면 squared-L2 거리는 보존된다:
```
‖R_pos · K_new[t] − R_pos · K_old[t]‖² = ‖K_new[t] − K_old[t]‖²
```
따라서 K_new 와 K_old 가 **같은 position 으로** RoPE 가 적용된 경우 (LMCache 의 `metadata.positions` 공유 시점) per-token deviation 의 **랭킹** 은 우리 pre-RoPE 도메인과 일치한다.

**Phase 3 정량 검증** (`benchmarks/lmcache_hkvd_compare.py`, 3 chunks × 46 tokens × layer-1):

| Metric | Value |
|---|---|
| Spearman ρ (v4 pre-RoPE rank vs LMCache post-RoPE rank) | **0.999137** |
| Pearson r (per-token magnitudes) | **1.000000** |
| Top-15% HKVD index overlap | **100% (6/6)** |
| max relative per-token diff | 18.3% (mostly from FP16/FP32 cast 노이즈) |

데이터: `reports/phase-3-attachments/lmcache_hkvd_compare.json`.

→ **결론**: 우리 `kv_deviation` 의 token-level ranking 은 LMCache 의 ranking 과 사실상 동일 (Spearman > 0.99). HKVD selection 자체는 일치. Pre-RoPE vs post-RoPE 의 절대값 magnitude 는 약간 다르나 (RoPE 의 cross-token 효과는 position 차이가 작은 인접 token 들 사이에서만 의미 있는데, deviation 은 same-token 비교 = same-position 회전 = 보존), top-K 선택에는 영향 없음.

이는 v4 의 pre-RoPE 저장 + retrieval-time RoPE shift 디자인 (design-decisions.md §11) 이 LMCache 와 정량적으로 동등한 HKVD ranking 을 보장함을 입증.

### Q2(d) — Top-K 선택 규칙 (정규화 / 정렬)

> `external/LMCache/lmcache/v1/compute/blend/blender.py:94-101`
> ```python
> assert self.common_metadata.recomp_ratios is not None
>
> # TODO(Jiayi): remove `[0]` hardcode
> topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
> topk_num = max(topk_num, 1)
>
> top_indices = torch.topk(diff_k, k=topk_num).indices
> top_indices, _ = torch.sort(top_indices)
> ```

- **정규화 없음** (raw squared-L2 sum 위에서 직접 topk).
- `recomp_ratios[0]` **hardcoded** — list 에 여러 ratio 가 있어도 첫 값만 씀 (TODO Jiayi).
- `topk_num = max(int(ratio × N), 1)` — ratio=0 이라도 최소 1 토큰 선택 (boundary case 누수, Q4 디자인 노트 참조).
- `top_indices` 를 **다시 sort** (positional order 유지) — 이후 attention causal mask 와 호환 위해.

---

## Q3 — Pre-RoPE K 어떻게 저장 / 적용?

**LMCache 는 pre-RoPE K 를 저장하지 않는다.** 저장은 vLLM KV cache 에 위임되며, vLLM 은 chunk standalone prefill 시점의 post-RoPE K 를 갖는다 (chunk-local positions).

핵심 증거:
1. `gpu_connector.get_kv(layer_id)` (`external/LMCache/lmcache/v1/compute/blend/blender.py:70`) 가 캐시된 K/V 를 그대로 가져온다 — RoPE-shift transform 을 거치지 않는다.
2. `FusedRope.fused_encode` (`external/LMCache/lmcache/v1/compute/positional_encoding.py:64-76`) 는 정의만 되어있고 process_qkv hot path 에선 호출되지 않는다 (Q2c).
3. `BasicReverseRope` (`external/LMCache/lmcache/v1/compute/positional_encoding.py:19-49`) 는 RoPE 역연산 가능성을 위한 utility 인데 역시 blender 에서 호출 흔적 없음.

따라서 deviation 계산은 "fresh k @ new_positions" vs "old k @ original_positions" 라는 position-mismatch 가 섞인 비교다. 이 mismatch 가 deviation signal 의 일부가 되어버린다.

### v4 디자인 (Phase 2)

v4 는 **pre-RoPE K 저장 + retrieve 시 RoPE shift** 를 paper §4 충실하게 구현 예정 (`src/cacheblend/rope.py` `apply_rope_shift`, `src/cacheblend/precompute.py`):
- Phase 1: pre-RoPE K 캡처 hook (k_proj output 후, RoPE 직전).
- Phase 2: KVStore 에 (chunk_id, layer_id, pre_rope_K, V) 저장.
- Phase 2: retrieval 시 chunk-local pos → blended global pos 로 RoPE re-apply.

이는 LMCache 와 디자인 차이가 큰 부분이며 design-decisions.md 에 entry 추가 (§11).

---

## Q4 — check_layer 결정: 단일 vs gradual?

**LMCache 코드 자체는 multi-layer 를 받을 수 있는 형태이지만, top-K 선택 로직은 layer-independent 하며 ratio 도 단일.**

코드 형태 (`external/LMCache/lmcache/v1/compute/blend/blender.py:88-101`):
- `if layer_id in check_layers` → check_layer 매번 동일 알고리즘.
- `recomp_ratios[0]` 만 사용 — 코드에 박힌 **TODO** comment:

> `external/LMCache/lmcache/v1/compute/blend/blender.py:43-45`
> ```python
> # TODO(Jiayi): support threshold-based blending
> # TODO(Jiayi): support different ratios for different layers
> # TODO(Jiayi): support "skipping blending if hit too short"
> ```

→ multi-ratio per layer 는 **미구현**.

- Top-K 가 매 check_layer 에서 독립 계산되며, 다음 check_layer 의 top-K 가 이전 top-K 의 부분집합이라는 보장도 코드 내에 없다 (그러나 실제로는 token slicing 이 누적되므로 자연스럽게 줄어듦).

Production 예제는 **단일 layer**:

> `external/LMCache/examples/blend_kv_v1/blend.py:34-35`
> ```python
> os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
> os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
> ```

→ 즉 LMCache 의 사실상 default 는 **CL=[1], ratio=0.15** flat single-layer 디자인. 논문 §4.3 의 multi-check-layer gradual filtering scheme 은 LMCache 에 미구현 — Phase 8 비교 대상.

### Phase 8 비교 표 (계획)

| Aspect | LMCache (single CL) | v4 Phase 8 (gradual) |
|---|---|---|
| Check layers | `[1]` (default) | `[2, 5, 10]` 같은 multi |
| Recomp ratio | `[0.15]` flat | `[0.30, 0.15, 0.10]` 점진 narrowing |
| Subset 강제 | 자연 발생 (slicing 누적) | 명시적 (이전 top-K 에서만 next-K) |
| Per-layer threshold | 미구현 (TODO) | discovery 후보 |

---

## Q5 — Storage device 처리 (RAM/SSD/...)

LMCache 는 **5-tier hierarchical storage** 백엔드를 갖는다:

> `external/LMCache/lmcache/v1/storage_backend/__init__.py:15-20`
> ```python
> from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
> from lmcache.v1.storage_backend.gds_backend import GdsBackend
> from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
> from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
> from lmcache.v1.storage_backend.p2p_backend import P2PBackend
> from lmcache.v1.storage_backend.remote_backend import RemoteBackend  # noqa: F401
> ```

(GDS = GPU Direct Storage / NVMe 직결.)

Config 기본값:

> `external/LMCache/lmcache/v1/config.py:72-89`
> ```python
> "local_cpu": {
>     "type": bool,
>     "default": True,
>     "env_converter": _to_bool,
> },
> "max_local_cpu_size": {"type": float, "default": 5.0, "env_converter": float},
> "reserve_local_cpu_size": {"type": float, "default": 0.0, "env_converter": float},
> "local_disk": {
>     "type": Optional[str],
>     "default": None,
>     "env_converter": _parse_local_disk,
> },
> ...
> "max_local_disk_size": {"type": float, "default": 0.0, "env_converter": float},
> ```

→ default 는 **CPU 5GB 만, disk off**. Disk 는 path 지정 시 enable.

예제 toggle:

> `external/LMCache/examples/blend_kv_v1/blend.py:42-58`
> ```python
> if use_disk:
>     os.environ["LMCACHE_LOCAL_CPU"] = "False"
>     os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
>     os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
>     os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
> else:
>     os.environ["LMCACHE_LOCAL_CPU"] = "True"
>     os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
> ```

### v4 디자인

v4 의 `KVStore` (Phase 2) 는 **단일-tier in-RAM Python dict**. 비교/loading-cost model 은 `StorageProfile` enum (`src/cacheblend/controller.py:11-19` `RAM`/`NVME`/`SATA_SSD`/`SLOW_DISK`) 으로 추상화하며 Phase 4 에서 cost model 만 도입 (실제 저장은 RAM 에 두고 simulated latency 부여).

이는 Phase 4 cost-aware loading (`LoadingController`) 이 LMCache 의 multi-tier 디자인을 흉내내되, v4 범위는 quality-only 이므로 실 NVMe I/O 는 안 함. (TTFT 비목표 L27.)

---

## Summary — v4 vs LMCache 디자인 차이

| 영역 | LMCache | v4 | 정당화 / 출처 |
|---|---|---|---|
| q/k/v 계산 범위 | 전체 토큰 | 전체 토큰 (hook-injection) | 둘 다 동일. |
| Slicing 위치 | post-RoPE (blender.py:103-105) | 없음 (hook override) | TTFT 비목표 (L27, design-decisions.md §3). |
| Pre-RoPE K 저장 | ✗ (vLLM post-RoPE 캐시 사용) | ✓ (Phase 2 KVStore) | Paper §4 충실. **design-decisions.md §11 신규**. |
| RoPE shift on retrieve | ✗ (`fused_encode` 미사용) | ✓ (`apply_rope_shift`, Phase 2) | Position mismatch 노이즈 회피. |
| KV deviation 공식 | sum((k_new − k_old)², dim=1), no normalize, fp32 (blender.py:89-91) | 동일 예정 (Phase 3 `kv_deviation`) | Paper §4.2 + LMCache 일치. |
| Top-K 정렬 | unsorted topk → sort by index (blender.py:100-101) | 동일 예정 | causal mask 호환. |
| Boundary safe-shortcut (ratio=0/1) | ✗ (`max(int(ratio*N), 1)` → ratio=0 이어도 1 토큰 recompute) | ✓ (`fuse_selective` 즉시 dispatch) | L13, design-decisions.md §2. |
| Check layers | List 받지만 `recomp_ratios[0]` hardcoded (blender.py:97) | Phase 3: 단일 (LMCache equivalent). Phase 8: multi-CL gradual. | Phase 8 신규 contribution. |
| Per-layer ratio | 미구현 TODO (blender.py:44) | Phase 8 명시 디자인 | 논문 §4.3 충실. |
| Threshold-based | 미구현 TODO (blender.py:43) | Phase 8 candidate | discovery experiment. |
| Storage tiering | 5 backends + GDS (storage_backend/__init__.py:15-20) | 단일 in-RAM + StorageProfile cost model (Phase 4) | TTFT 비목표 → 실 I/O 불필요. |
| Default ratio/CL | `[1] / 0.15` (blend.py:34-35) | `[1] / 0.15` baseline 동일 | Phase 6 baseline 비교 시 사실상 LMCache config. |

### 정당화 필요 — design-decisions.md 에 추가할 entry

이미 §1~§10 에 covered:
- Tolerance 4 카테고리 (§1), Boundary safe-shortcut (§2), Hook-injection (§3), mydata (§4), Per-item shuffle (§5), Bootstrap CI (§6), Discovery vs Validation (§7), v5-lessons (§8), 환경 정합 (§9), Pod reclaim (§10).

**신규 추가** (이 문서 작성 중 발견):
- **§11 — Pre-RoPE K 저장 + retrieve 시 RoPE shift**: paper §4 vs LMCache 디자인 차이의 핵심 한 항목.

---

## 부록 — 인용 횟수 self-check

본 문서 내 `external/LMCache/...` 형식 file:line 인용 (실 코드 내용 발췌 포함) 누적:

1. `lmcache/v1/compute/models/base.py:67-141` (Q1)
2. `lmcache/v1/compute/models/base.py:99-107` (Q1)
3. `lmcache/v1/compute/models/base.py:112-114` (Q1)
4. `lmcache/v1/compute/blend/blender.py:88-105` (Q1, Q2)
5. `lmcache/v1/compute/blend/blender.py:89-91` (Q2a, Q2c)
6. `lmcache/v1/compute/blend/blender.py:88` (Q2b)
7. `lmcache/v1/compute/blend/metadata.py:11-18` (Q2b)
8. `lmcache/v1/config.py:128-131` (Q2b)
9. `lmcache/v1/compute/blend/blender.py:84-91` (Q2c)
10. `lmcache/v1/compute/positional_encoding.py:64-76` (Q2c, Q3)
11. `lmcache/v1/compute/blend/blender.py:94-101` (Q2d)
12. `lmcache/v1/compute/blend/blender.py:70` (Q3)
13. `lmcache/v1/compute/positional_encoding.py:19-49` (Q3 — BasicReverseRope mention)
14. `lmcache/v1/compute/blend/blender.py:43-45` (Q4 TODO)
15. `examples/blend_kv_v1/blend.py:34-35` (Q4 default)
16. `lmcache/v1/storage_backend/__init__.py:15-20` (Q5)
17. `lmcache/v1/config.py:72-89` (Q5)
18. `examples/blend_kv_v1/blend.py:42-58` (Q5)

**18 회 인용** (≥10 요건 충족).

Sub-question 별 인용 횟수:
- Q1: 4 회 (base.py 3, blender.py 1)
- Q2(a): 1 회
- Q2(b): 3 회 (blender.py, metadata.py, config.py)
- Q2(c): 3 회 (blender.py 2, positional_encoding.py 1)
- Q2(d): 1 회
- Q3: 3 회 (blender.py 1, positional_encoding.py 2)
- Q4: 2 회 (blender.py TODO, blend.py example)
- Q5: 3 회 (storage_backend, config, example)
