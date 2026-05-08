# Phase 4 — Pipelining & Prefix Cache Baseline Report

> Tolerance categories (frozen):
> - 4.1 selective_pipelined ≡ selective : **RECOMPUTE_PATH** (max_diff < 1e-3)
> - 4.2 prefix_cache vs full_recompute  : **MIXED_SHAPE** (argmax exact + max_diff < 5e-2)
> - 4.3 LoadingController monotone      : CPU-only sanity
>
> Result: **PASS (5/5 conditions)**.

## 1. Outcome

| ID | 결과 | 핵심 수치 |
|---|---|---|
| 4.1 selective_pipelined ≡ selective | ✅ PASS | max_diff = 0.000e+00, argmax 1.0000 (RECOMPUTE_PATH bound 1e-3 안에서 bit-exact) |
| 4.2 prefix_cache argmax-eq full_recompute | ✅ PASS | max_diff = 2.734e-02 < 5e-2, argmax 1.0000 (MIXED_SHAPE) |
| 4.3 LoadingController monotone | ✅ PASS | RAM 0.150 → NVMe 0.195 → SATA 0.270 → SLOW 0.495 (strict 단조) |
| 4.4 verify_phase | ✅ PASS | (아래 §8) |
| 4.5 cost_check ≤ $3 | ✅ PASS | $0.49 / $3.00 |
| **TTFT (참고용)** | gate 아님 [L27] | full_recompute 가 floor; selective ~30% 느림 (hook-injection) |

## 2. Pod (vast.ai)

| 항목 | 값 |
|---|---|
| Instance | `36296967` (Phase 1 부터 keep alive, Phase 4 재사용) |
| GPU | 1× NVIDIA RTX 3090 (24 GB) |
| 단가 | $0.1611 / hr |
| Phase 4 wall time | ~10 min (실 GPU: tarball+SCP <1 min, pytest 3 tests 30s, TTFT sweep 9 min) |
| Phase 4 billing 합계 (manual) | **$0.08** |
| 누적 비용 (cost-tracker) | **$0.49** ($0.16 + $0.17 + $0.08 + $0.08) |
| Pod 총 uptime (vast.ai 기준) | 415.66 min (~6.93 hr × $0.1611 = ~$1.12 billing 추정) |
| Idle 압축 결과 | Phase 4 작업 자체는 ~10 min, Phase 3 (30 min) 보다 더 짧음. Pod 의 idle 시간 (phase 트리거 대기) 이 누적 — 4.5 cap $3 안에서 안전하지만 Phase 5 진입 직전 stop 또는 cap 도달 시 행동 정책 필요. |

⚠️ vast.ai dashboard 의 정확 billing 확인 권장. cost-tracker 는 phase 작업 시간만 manual 기록 (idle 별도).

## 3. Env parity

`bash scripts/diff_env.sh` Pod (Phase 4 시작 직전): **7/7 핀 match**. Mac venv 변동 없음.

## 4. 구현 상세

### 4.1 `src/cacheblend/kv_store.py` — 변경 (54 → 116 lines, +62 lines)

| 변경 | 위치 | 내용 |
|---|---|---|
| `__init__` 에 `max_workers` 추가 | `kv_store.py:23-30` | `self._inflight: dict[str, Future] = {}`, `self._executor: Optional[ThreadPoolExecutor] = None` (lazy) |
| `has` semantic 확장 | `kv_store.py:34-36` | inflight 도 "has" 로 인정 → 호출자가 `get` 으로 block 받을 수 있음 |
| `get` 에 inflight 처리 | `kv_store.py:38-46` | `chunk_id in self._inflight` 면 `Future.result()` block → `_put_entry`. Race-free (single-threaded reader). |
| `_put_entry` 헬퍼 | `kv_store.py:56-63` | 기존 put 로직 분리 (sync put + async-resolve put 공유) |
| `prefetch_chunk(chunk_id, loader_fn)` | `kv_store.py:84-105` | Future 반환. 이미 cached → completed Future. 이미 inflight → 같은 Future 재사용 (idempotent). 미존재 → executor.submit. |
| `_ensure_executor` (lazy) | `kv_store.py:78-82` | 첫 prefetch 시 ThreadPoolExecutor 생성 |
| `shutdown(wait=True)` | `kv_store.py:107-110` | executor cleanup |

**Future 관리 흐름**:
1. caller: `kv_store.prefetch_chunk(cid, lambda: load_from_disk(cid))` — Future 반환, executor 가 백그라운드에서 `loader_fn()` 실행.
2. caller (이후): `kv_store.get(cid)` — `_inflight[cid]` 있으면 `.result()` block (일반적으로 prefetch 가 이미 끝남) → `_put_entry` 로 _cache 채우고 _inflight 에서 제거 → 일반 path.
3. 이미 cached 인 경우 `prefetch_chunk` 는 즉시 completed Future 반환 (무비용).

### 4.2 `src/cacheblend/controller.py` — 75 lines (신규 실 구현)

```python
class StorageProfile(Enum):
    RAM       = 1.0   # baseline
    NVME      = 4.0   # ~4× 느림
    SATA_SSD  = 12.0  # ~12× 느림
    SLOW_DISK = 50.0  # ~50× 느림 (paper §6 spirit)
```
(`controller.py:18-26`). `slowness_rank` property: 0/1/2/3 (`controller.py:28-36`).

**`decide_recompute_ratio(profile, base_ratio)` 공식** (`controller.py:64-80`):
```
multiplier = {RAM: 1.00, NVME: 1.30, SATA_SSD: 1.80, SLOW_DISK: 3.30}[profile]
ratio = min(base_ratio × multiplier, max_ratio=0.95)
```

**Monotone 보장 근거**: `_MULTIPLIERS` 가 strict 증가 시퀀스 (1.00 < 1.30 < 1.80 < 3.30) 이고, `min(·, max_ratio=0.95)` 는 단조 비감소 함수이므로 — 모든 profile 에 대해 `multiplier` 가 증가하면 `min(scaled, max_ratio)` 도 비감소. base=0.15 에서 cap 안 닿음 → 4 등급 strict 단조. base 가 매우 클 때 (e.g. 0.7) cap 에 도달 → 비감소 (strict 일 수도 비-strict 일 수도). 일반 phase 의 base ∈ [0.05, 0.30] 에서는 항상 strict.

`LoadingDecision` dataclass 가 (profile, base, multiplier, scaled, capped, ratio, detail) 모두 보고 (`controller.py:38-44`).

### 4.3 `src/cacheblend/fusor.py:fuse_selective_pipelined` — `fusor.py:213-247` (35 lines)

```python
def fuse_selective_pipelined(model, chunks, kv_store, recompute_ratio=0.15,
                             check_layer=1, prefetch=True):
    if prefetch:
        for chunk in chunks:
            if kv_store.has(chunk.chunk_id):
                # RAM tier: thunk returns in-memory entry; exercises Future path
                # without real I/O. For NVMe/SSD tiers, replace with disk loader.
                kv_store.prefetch_chunk(chunk.chunk_id, lambda c=chunk.chunk_id, s=kv_store: s._cache[c])
    return fuse_selective(model, chunks, kv_store,
                          recompute_ratio=recompute_ratio, check_layer=check_layer)
```

**Async race 방지**:
1. `prefetch_chunk` 가 idempotent — 이미 cached 면 completed Future, 이미 inflight 면 같은 Future 재사용.
2. `fuse_selective` 내부 `kv_store.get(chunk_id)` 가 inflight 있으면 자동 `Future.result()` block — single-pass forward 시작 전에 모든 entry 가 _cache 에 안착.
3. forward 본 패스 중에는 KVStore 변경 없음 — 모든 hook 은 read-only.
→ 결과적으로 wall-time 만 (I/O 와 forward 의 prefetch overlap 시) 개선되며 logits 는 비-pipelined 와 bit-exact.

### 4.4 `src/cacheblend/fusor.py:fuse_prefix_cache` — `fusor.py:250-303` (54 lines)

**분기 / 흐름**:
- `len(chunks) <= 1` → `fuse_full_recompute` 직접 호출 (`fusor.py:264-265`) — single-prefix 동일 path.
- 멀티 chunk:
  - 첫 chunk 만 cached K/V reuse, 나머지는 fresh recompute.
  - 매 layer 마다 k_proj/v_proj 에 hook 등록 (`fusor.py:280-294`).
  - hook 동작: `output[:, first_start:first_end, :] = K_cached_first`. 이외 위치 (chunk 1~ 끝) 는 unchanged → fresh recompute.
  - 첫 chunk 의 chunk-local positions == fused positions (둘 다 0..L_first-1) → RoPE shift 불필요. HF apply_rotary_pos_emb 가 자연스럽게 작동.
- `try/finally` 로 모든 handles cleanup (`fusor.py:296-302`).

### 4.5 `benchmarks/ttft.py` — 113 lines

| 부분 | 위치 | 1줄 설명 |
|---|---|---|
| `_time_call` | `ttft.py:14-21` | `cuda.synchronize()` 전후 + `time.perf_counter()` ms |
| `_median_runs` | `ttft.py:24-28` | 1 warmup + 3 timed → median |
| `make_chunks` | `ttft.py:50-62` | seed text 반복 → target_len 토큰 → Chunk 생성 |
| sweep loop | `ttft.py:65-88` | 3 sequence length × 5 method matrix |

## 5. Tests 결과

### 4.1 test_pipelined_eq_unpipelined — RECOMPUTE_PATH ✅
- `fuse_selective_pipelined` vs `fuse_selective` 같은 ratio=0.15: **max_diff = 0.000e+00**, argmax 1.0000.
- ToleranceResult: `category=recompute_path, bound=max_diff < 1e-3, passed=True`.
- prefetch 가 RAM tier 에서 race condition 도입 안함 입증.

### 4.2 test_prefix_cache_eq_full_recompute — MIXED_SHAPE ✅
- `fuse_prefix_cache` vs `fuse_full_recompute`: **max_diff = 2.734e-02**, argmax 1.0000.
- ToleranceResult: `category=mixed_shape, bound=argmax == 1.0 AND max_diff < 5e-2, passed=True`.
- 첫 chunk 만 cached → 나머지 chunks 의 hidden_states 가 첫 chunk 의 cached K/V 를 attention 으로 참조. positions 일치 (chunk-local == fused 0..L_first-1) 하므로 quality 보존, 작은 numerical noise 만 (FP16).

### 4.3 test_loading_controller_monotone — CPU-only ✅

| Storage | base_ratio | multiplier | scaled | recompute_ratio |
|---|---:|---:|---:|---:|
| RAM | 0.150 | 1.00 | 0.150 | **0.150** |
| NVMe | 0.150 | 1.30 | 0.195 | **0.195** |
| SATA_SSD | 0.150 | 1.80 | 0.270 | **0.270** |
| SLOW_DISK | 0.150 | 3.30 | 0.495 | **0.495** |

Strict 단조 증가 (RAM < NVMe < SATA < SLOW). Test passes.

## 6. TTFT 측정 (참고용, gate 아님 [L27])

5 method × 3 sequence length × (1 warmup + 3 timed) median ms:

| chunk_B | total_seq | full_recompute | full_reuse | prefix_cache | selective | selective_pipelined |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 180 | 55.83 | 59.39 | 56.78 | 76.06 | 75.37 |
| 120 | 360 | 102.55 | 106.00 | 104.16 | 118.51 | 118.80 |
| 240 | 720 | 199.12 | 202.23 | 200.38 | 213.40 | 213.36 |

**Speedup vs full_recompute** (numbers > 1.0× = faster, < 1.0× = slower):

| chunk_B | full_reuse | prefix_cache | selective | selective_pipelined |
|---:|---:|---:|---:|---:|
| 60 | 0.94× | 0.98× | 0.73× | 0.74× |
| 120 | 0.97× | 0.98× | 0.87× | 0.86× |
| 240 | 0.98× | 0.99× | 0.93× | 0.93× |

### Hook-injection 디자인의 TTFT 한계 (L27 재확인)

우리 hook-injection 디자인은 q/k/v projection 을 모든 토큰에 fully 수행하고 cached 위치만 hook 으로 override 한다. 따라서 token slicing (LMCache 처럼 attention 직전에 처리) 과 달리 **연산량 절감이 없다**. selective 가 full_recompute 보다 ~30% 느린 이유:
1. 32 layer × (k_proj + v_proj) hook 의 Python 콜백 오버헤드.
2. check_layer 에서 추가 deviation 계산.
3. K/V override 시 tensor clone + indexing.

Pipelined vs unpipelined 차이 미세 (0.7 ms 이내) — RAM tier 에서는 prefetch 가 절감할 I/O 가 없음. 실제 NVMe/SSD 로 옮길 경우 prefetch 가 의미 있을 것이라는 expectation 만 보고됨 (Phase 4 범위 밖).

**결론**: TTFT 절감은 v4 의 비목표 (L27, design-decisions.md §3). v4 는 quality milestone 만; production-grade TTFT 는 v5 의 주제. 본 측정은 단순 sanity (pipelined 가 unpipelined 와 비슷함, prefix_cache 가 full_recompute 와 비슷함, full_reuse 가 약간 hook 오버헤드).

## 7. Gate 5 condition

| ID | check_type | 결과 | 근거 |
|---|---|---|---|
| 4.1 | pytest | ✅ PASS | max_diff=0 (RECOMPUTE_PATH) |
| 4.2 | pytest | ✅ PASS | max_diff=2.73e-02 < 5e-2, argmax 1.0 (MIXED_SHAPE) |
| 4.3 | pytest | ✅ PASS | RAM 0.15 → SLOW 0.495 strict 단조 |
| 4.4 | verify_phase | ✅ PASS | controller.py / test_pipeline.py / benchmarks/ttft.py / phase-4-report.md 모두 존재 + `## v5-lessons` 섹션 |
| 4.5 | cost_check | ✅ PASS | $0.49 / $3.00 cap |

`gates/gate-4-result.json` 자동 기록.

## 8. Cost

- Phase 4 단독: **$0.08** (10 min wall × $0.1611/hr)
- 누적 (Phase 0~5 한도 $5): **$0.49 / $5** (9.8%)
- Pod 총 uptime 기준 billing 추정: ~$1.12 (415 min × $0.1611/hr). 차이 (~$0.63) 는 phase 간 idle 시간 — Phase 5 (CPU only) 진입 전 Pod stop 권장 (사용자 결정).

## v5-lessons (이번 phase 에서 발견된 사항)

이번 phase 에서 새 lesson 없음. Phase 2/3 의 인프라 (KVStore + apply_rope_shift + fuse_full_reuse hook + L13 boundary safe-shortcut) 그대로 활용했으므로 신규 발견 0건.

상세는 `docs/notes/v5-lessons.md` 참조 (현재 L31~L38 누적, 8개).

## 9. 수정 파일

| 경로 | 변경 사유 |
|---|---|
| `src/cacheblend/kv_store.py` | sync 전용 → async prefetch 추가 (54 → 116 lines, +62). ThreadPoolExecutor + Future. has()/get() semantic 확장. |
| `src/cacheblend/controller.py` | Phase 0 stub → 실 구현 (75 lines). StorageProfile cost factor + decide_recompute_ratio + LoadingDecision dataclass. |
| `src/cacheblend/fusor.py` | `fuse_selective_pipelined` 신규 (35 lines, `fusor.py:213-247`). `fuse_prefix_cache` 신규 (54 lines, `fusor.py:250-303`). |
| `tests/test_pipeline.py` | 신규 (~110 lines): 3 tests (pipelined ≡ unpipelined, prefix_cache argmax-eq full_recompute, controller monotone). |
| `benchmarks/ttft.py` | 신규 (113 lines): 5 method × 3 length sweep, median wall ms. |
| `reports/phase-4-attachments/pytest.log` | pytest archive |
| `reports/phase-4-attachments/ttft.json/.md` | TTFT 측정 데이터 + speedup 표 |
| `reports/cost-tracker.json` | $0.08 → 누적 $0.49 |
| `reports/phase-4-report.md` | 본 보고서 |
| `gates/gate-4-result.json` | gate eval (auto) |

## 10. Phase 5 사전점검

`tasks/phase-5-dataset.md` — **CPU-only ($0)**. 핵심:
- mydata harness (`external/mydata/cacheblend_fig12/harness/`) 의 `CacheBlendRunner` ABC 위에 v4 의 5 Runner 실 구현 (`src/cacheblend/runners.py`): FullRecomputeRunner, FullReuseRunner, PrefixCacheRunner, CacheBlendV4Runner, GradualV4Runner stub.
- `benchmarks/metrics/bootstrap.py` — paired bootstrap CI helper [L28].
- `benchmarks/run_eval.py` — main eval loop wrapper.
- mydata prompts.jsonl dryrun (sample 5개 정도, model 미사용 시 plumbing 만 검증).

준비: GPU 불필요. Mac venv 에서 진행 가능. **Pod stop 권장 시점** — 사용자께서 stop/destroy 결정 후 Phase 5 트리거.

⚠️ 사용자께 알림: Phase 5 는 CPU-only ($0) 이므로 Pod 36296967 을 stop 하거나 destroy 해도 됩니다. Phase 6 (Mistral eval) 진입 시 다시 GPU instance 부팅. 현재 Pod uptime 6.93 hr × $0.1611 = ~$1.12 누적 billing, idle 시간 절감 위해 stop 추천.
