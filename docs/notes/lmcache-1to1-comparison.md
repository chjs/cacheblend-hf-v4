# 1:1 비교 — Our v4 ↔ LMCache (chjs/fix/cacheblend-vllm-v0.17.1-compat)

> Branch: `compare/lmcache-parity-fix`
> Reference: `external/LMCache/` @ `chjs/fix/cacheblend-vllm-v0.17.1-compat` (HEAD `9f8aa4d`)
> Our source: `src/cacheblend/` (1,650 LOC across 11 files)
> 비교 대상 LMCache: `lmcache/v1/compute/blend/` (368 LOC) + `compute/models/base.py` (142 LOC) + `compute/positional_encoding.py` (202 LOC) + `gpu_connector/gpu_connectors.py` (관련 부분)

## 0. 한 줄 결론

**6개 메커니즘 중 5개 100% 동일 (수학적 또는 알고리즘적 등가) · 1개 (selective recompute의 forward 폭) ALGORITHMIC 차이** — 우리 v4 의 `fuse_selective` 가 check_layer 이후 **full forward at all positions** 인 반면 LMCache는 **sparse forward at top-K positions** (paper §4 의도). 수정 필요.

## 1. 모듈 매핑

| 메커니즘 | 우리 v4 | LMCache |
|---|---|---|
| Layer-by-layer orchestration | `src/cacheblend/model.py` `LayerwiseModel.forward_layerwise` | `lmcache/v1/compute/models/base.py` `LMCBaseModel.compute_layer` (generator) |
| Per-layer QKV processing | (HF 내부에서 처리) + hook 으로 K override | `LMCBlender.process_qkv` |
| KV cache storage | `kv_store.py` (in-mem, **pre-RoPE** K + V) | `gpu_connector.batched_to_gpu` (paged GPU, **post-RoPE** K + V) |
| RoPE 적용 | HF `apply_rotary_pos_emb` (k_proj output 후 자동) | `FusedRope.fused_encode(old_pos, new_pos, k)` (RoPE-shift kernel) |
| HKVD 계산 | `hkvd.py` `kv_deviation`, `select_top_k` | `blender.py:89-101` inline |
| Selective recompute | `fusor.py` `fuse_selective` (hook 기반, full forward) | `blender.py:88-120` (Q sparse slicing) |
| Chunking / fused input | `chunker.py` | LMCache 는 chunk concept 외부에서 처리 (vLLM 요청) |
| Pipelined / prefix-cache 베이스라인 | `fusor.py` `fuse_selective_pipelined`, `fuse_prefix_cache` | 없음 (우리 v4 전용 ablation) |
| Gradual filtering | `gradual.py` (stub) + `phase8_step1_profile.py` | 없음 |
| Bootstrap CI | `benchmarks/metrics/bootstrap.py` | 없음 (offline 분석) |

## 2. 메커니즘별 비교

### 2.1 KV cache 저장 형태 (다른 표현, 수학적 등가) ✓

**우리 v4** (`kv_store.py:1-12`, `precompute.py:33-61`):
- `precompute_chunk_kv` 가 chunk 를 standalone forward (positions 0..L−1, no RoPE shift)
- `k_proj` forward-hook 으로 **pre-RoPE K** capture (RoPE 적용 전)
- KVStore.put: 레이어별 `(B=1, L, num_kv_heads*head_dim)` 로 in-mem 저장
- V 는 v_proj forward-hook 으로 캡처 (V 는 RoPE 무관)

**LMCache** (`gpu_connector/gpu_connectors.py:729-862`):
- 표준 vLLM prefill 시 **post-RoPE K at chunk-local positions** 가 paged KV cache 에 저장
- `batched_to_gpu` 에서 로드 시 line 858: `compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(old_positions, new_positions, k)` 로 RoPE shift (chunk-local → fused global)

**등가성 증명**:
- 우리 v4 가 retrieval 시 적용하는 RoPE: `K_fused_postRope = HF.apply_rotary_pos_emb(K_pre, fused_position)`
- LMCache 가 retrieval 시 적용하는 RoPE shift: `K_fused_postRope = RoPE_shift(K_postRope_atChunkLocal, chunk_local_pos → fused_pos)`
- RoPE 는 회전 변환 이므로 `RoPE(x, p2) = RoPE_shift(RoPE(x, p1), p1→p2)` 가 모든 `x, p1, p2` 에 대해 성립.
- 따라서 두 방식의 retrieval 후 `K_fused_postRope` 는 비트 동등 (FP 누적오차 제외).

→ **PASS** (다른 저장 표현, 같은 의미)

### 2.2 HKVD 계산 (mathematically identical) ✓

**우리 v4** (`hkvd.py:23-55`):
```python
diff = (K_new.to(fp32) - K_old.to(fp32)) ** 2
return diff.sum(dim=-1).squeeze(0)  # (S,)
```

**LMCache** (`blender.py:89-91`):
```python
diff_k = torch.sum(
    (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
)
```

**비교**:
- 두 K 모두 같은 표현 공간 (pre-RoPE vs post-RoPE 가 다르지만 § 2.1 에서 보였듯 비트 등가; RoPE 는 회전이라 squared L2 보존: `||RoPE(x,p) − RoPE(y,p)||² = ||x − y||²`).
- 두 코드 모두 fp32 cast, squared L2, hidden_dim 축 sum, per-token 결과.
- LMCache 의 `dim=[1]` 은 shape `(num_tokens, hidden)` 의 hidden 축. 우리는 `dim=-1` 으로 마지막 축 (= hidden) sum + batch squeeze.

→ **PASS** (수식 + 형 일치)

### 2.3 Top-K 선정 (identical) ✓

**우리 v4** (`hkvd.py:58-77`):
```python
topk_num = int(total_len * ratio); topk_num = max(topk_num, 1); topk_num = min(topk_num, total_len)
top_indices = torch.topk(deviations, k=topk_num, largest=True).indices
top_indices, _ = torch.sort(top_indices)  # ascending
```

**LMCache** (`blender.py:94-101`):
```python
topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
topk_num = max(topk_num, 1)
top_indices = torch.topk(diff_k, k=topk_num).indices
top_indices, _ = torch.sort(top_indices)
```

**비교**: 동일. 우리 측의 `min(topk_num, total_len)` 가드는 ratio≥1 edge 보호 (LMCache 는 ratio≥1 에서 IndexError 가능성, 우리 가 더 robust).

→ **PASS** (동일 + 우리 가드 추가)

### 2.4 RoPE 처리 ✓

**우리 v4** (`fusor.py:fuse_full_reuse`, `fuse_selective` 의 hook 메커니즘):
- 캐시 retrieve 시 pre-RoPE K 를 `k_proj` output 자리에 inject
- HF Transformer 내부의 `apply_rotary_pos_emb` 가 fused positions 로 RoPE 적용

**LMCache** (`positional_encoding.py:55-82`, `gpu_connector.py:858`):
- 캐시 retrieve 시 post-RoPE K (chunk-local) 를 GPU 버퍼로 로드
- `FusedRope.fused_encode(old_pos, new_pos, k)` 가 단일 CUDA kernel 로 RoPE shift

**검증**: 우리 commit `de6d818` 의 `benchmarks/verify_rope_recomputation.py` 가 `fuse_full_reuse` 후 K 가 정확히 fused position 의 post-RoPE 값임을 5 layer × 2 chunk 에서 비트 등가 (diff = 0.000e+00) 로 확인 완료.

→ **PASS** (다른 path, 같은 결과)

### 2.5 Causal mask (sparse Q 의 경우) ✓

**LMCache**: `attn_metadata.update_from_top_indices` 가 sparse-Q × full-K 의 causal mask 셋업 (FlashAttn varlen 의 `cu_seqlens_q=[0, topk_num]`, `cu_seqlens_k=[0, total_len]`).

**우리 v4**: full forward 라서 표준 causal mask (Q×Q). check_layer 이후에도 sparse 가 아님 → causal mask 관련 차이는 § 2.6 의 sparse-forward 차이에 종속.

→ **N/A** (sparse forward 자체가 다른 § 2.6 에서 처리)

### 2.6 ★ Selective recompute 의 forward 폭 — **DIFFERENT** ✗

이게 유일한 알고리즘 차이입니다.

#### LMCache 동작 (`base.py:67-142`, `blender.py:59-120`)

```
Layer 0..check_layer-1:    full forward (Q full × K/V full fresh)
Layer check_layer:         process_qkv:
                             - RoPE(Q, K)
                             - diff_k = ||K_fresh - K_cached_shifted||²
                             - top_indices = topk(diff_k)
                             - Q ← Q[top_indices]      ← Q sparse!
                             - residual ← residual[top_indices]
                             - attn_output ← attn_output[:topk_num]
                             - old_k[top_indices] = K_fresh
                             - old_v[top_indices] = V_fresh
                             - attn_metadata.update_from_top_indices(...)
                           Then: sparse Q × full mixed-K/V attention
                                 → sparse hidden_state (length topk_num)

Layer check_layer+1..end:  hidden_state input is SPARSE (length topk_num)
                           process_qkv:
                             - RoPE(Q_sparse, K_sparse)
                             - imp_indices is set:
                                 old_k[imp_indices] = K_sparse_fresh
                                 old_v[imp_indices] = V_sparse_fresh
                                 return Q_sparse, full mixed-K/V
                           sparse Q × full mixed-K/V → sparse hidden_state
```

End state:
- Past K/V cache: **full-length**, 모든 position 에 valid value.
  - top-K positions: 매 layer 마다 fresh K/V (from sparse forward) 로 갱신됨
  - non-top positions: 원래 chunk-local cached K/V 가 RoPE shifted 된 상태로 유지

Decoding 시 prefix attention 은 이 full K/V cache 사용 → 정상.

#### 우리 v4 동작 (`fusor.py:fuse_selective`, 119-246)

```
Layer 0..check_layer-1:    no hook → full fresh forward
Layer check_layer:         k_proj forward-hook (observation only, return None):
                             - 관찰: kv_deviation(fresh_pre_K, K_stored_pre_K)
                             - state["hkvd_indices"] = top-K
                             - K modify 안 함
                           → full fresh forward (FULL Q × FULL FRESH K/V)
                           → full hidden_state 출력

Layer check_layer+1..end:  k_proj/v_proj forward-hooks:
                             - result = output.clone()
                             - result[:, non_hkvd_mask, :] = stored[:, mask, :]
                             - return result (FULL length, fresh-at-hkvd + cached-at-non-hkvd)
                           HF.apply_rotary_pos_emb(result, fused_positions)
                           → 모든 position 의 K post-RoPE 가 valid
                           Full Q × Full mixed-K/V attention (Q 도 FULL!)
                           → full hidden_state 출력
```

End state:
- Past K/V cache: full-length, valid.
  - top-K positions: fresh K/V from **FULL forward** (다른 hidden_state propagation)
  - non-top positions: chunk-local cached K/V (RoPE shifted via HF) ← LMCache 와 동일

#### 의미적 차이

**top-K positions 에서 K/V 값이 다릅니다**:

| 단계 | LMCache 의 top-K K | 우리 v4 의 top-K K |
|---|---|---|
| check_layer 입력 (layer 0 출력 hidden_state) | full forward, 동일 | full forward, 동일 |
| check_layer 의 attention 출력 (top-K 위치) | sparse Q_top × mixed K/V (cached-at-non-top + fresh-at-top) | full Q × full fresh K/V |
| Layer check_layer+1 의 input hidden_state at top-K | 위 sparse 결과 | 위 full 결과 (다름!) |
| Layer check_layer+1 의 K projection at top-K | k_proj(sparse_hidden) | k_proj(full_hidden) ≠ |
| ... 모든 후속 layer ... | drift 누적 | drift 누적, 다름 |

**FLOPS 차이**: paper §4 의 비용 절감 핵심 (85% non-top 의 attention/FFN skip) — LMCache 가 구현, 우리 v4 가 안 함 (full forward).

**Quality 영향 추정**:
- Phase 6c Mistral: 우리 cb F1=0.222 vs FullReuse 0.143 (+0.046). 우리 알고리즘 도 quality recovery 는 함.
- Phase 7c Llama: 우리 cb F1=0.156 vs FullReuse 0.167 (−0.011). 알고리즘 차이가 quality 손상을 가져올 수 있음.
- LMCache-parity 후 수치 변화는 실험 으로 측정 필요.

#### 작은 차이 1 (보조)

LMCache 의 check_layer 에서 `attn_metadata.update_from_top_indices` 가 호출되며 그 layer 의 **attention 자체** 도 sparse Q 로 수행됨. 우리 v4 의 check_layer 는 attention 이 full Q 로 수행되고 observation 만 함.

#### 작은 차이 2 (정책)

- LMCache: `recomp_ratios[0]` hardcode (TODO 주석 으로 multi-layer 다른 ratio 지원 예정)
- 우리 v4: single `recompute_ratio` scalar. 동등.

- LMCache: `check_layers` list (multi check 지원 코드 있음, `if layer_id in check_layers`)
- 우리 v4: single `check_layer` int. Phase 8 의 gradual 이 multi-check 확장 예정 (현재 stub).

→ default (check_layer=1, ratio=0.15) 동작은 동등 가능. 우리는 gradual.py 미구현 이므로 multi-check 차이 무관.

#### Boundary safe-shortcut (우리 v4 추가, LMCache 없음)

`fusor.py:151-159`:
- `ratio == 0` → `fuse_full_reuse` (LMCache 는 ratio=0 에서도 1 token 선택)
- `ratio >= 1` → `fuse_full_recompute`
- `len(chunks) <= 1` → `fuse_full_recompute`

이는 우리 v4 의 boundary 정확성 정책 (L13) 이며 LMCache 와의 의미 차이는 ratio=0 edge 케이스 에서만 발생.

→ **FIX 필요** (algorithmic 차이가 § 2.6 의 main path)

## 3. Fix 계획

### 새 함수 `fuse_selective_lmc_parity`

`src/cacheblend/fusor.py` 에 새 함수 추가:

```python
def fuse_selective_lmc_parity(
    layerwise_model, chunks, kv_store,
    recompute_ratio=0.15, check_layer=1,
    return_layerwise_output=False,
):
    """LMCache-parity selective recompute (paper §4 faithful, sparse forward at top-K).
    
    LMCache `blender.py:process_qkv` 와 1:1 매핑:
      Layer 0..check_layer-1: full forward
      Layer check_layer:      top-K 선정, Q/residual sparse slicing, K/V merge
      Layer check_layer+1+:   sparse Q × full mixed K/V (KV cache 갱신 at top-K)
    
    End state: past_key_values 는 full-length 로 fresh-at-top + cached-elsewhere.
    """
```

구현 핵심:
1. `LayerwiseModel.prefill_layer` 를 layer 별 명시적 호출
2. Layer 0..check_layer-1: 정상 forward
3. Layer check_layer:
   - 우선 정상 forward 수행 (fresh K 캡처)
   - HKVD 선정
   - **여기서 분기**: paper §4 의 정확한 의미는 "check_layer 자체부터 sparse forward" — 따라서 check_layer 의 attention 도 sparse Q 로 재수행 필요
   - 또는 단순화: check_layer 는 full forward 로 K/V 캐시 갱신 후 그 다음부터 sparse
4. Layer check_layer+1..end: sparse hidden_state 로 prefill_layer 호출. 각 layer 마다 fresh K/V 를 KV cache 의 top-K 위치 에 scatter, non-top 은 stored RoPE-shifted K/V 로 채움.
5. End: `past_key_values` (DynamicCache) 를 full-length 로 reconstruct.

기존 `fuse_selective` 는 deprecated 명시 + 유지 (ablation 용).

### 검증 실험

CPU-runnable unit test (Mac, no pod cost):
- 작은 HF model (e.g. `sshleifer/tiny-gpt2` 또는 `hf-internal-testing/tiny-random-MistralForCausalLM`)
- 합성 2-chunk 입력
- `fuse_selective_lmc_parity` 결과:
  - end-state past_key_values shape = full-length ✓
  - top-K positions 의 K: fresh-from-sparse-forward (예측 가능)
  - non-top positions 의 K: cached RoPE-shifted (예측 가능, 비트 등가)
  - logit at last token: 합리적
- vs 기존 `fuse_selective`: 의도된 차이 (top-K K 값 다름) 확인

GPU 검증 (vast.ai, 옵션):
- Phase 6c Mistral n=20 재실행 (`run_phase6.py` 에 `--cb-impl lmc_parity` flag 추가)
- F1 비교: 기존 cb F1 0.222 vs 새 cb F1 (예측: paper §4 더 가까움, FullRecompute 0.254 에 근접)

비용: $0.10 ~ $0.20 (1 모델, 4 runner × 20 sample, 약 5 분).

## 4. 영향 분석

수정 후 변화:
- Phase 6c, 7c, 8-step1 의 CacheBlend 수치가 변경됨 → 보고서 재생성 필요
- v5-lessons L43 추가: "fuse_selective 의 full-forward 가 LMCache/paper §4 의 sparse-forward 와 다름. fix 후 quality 재측정"
- 새 함수가 primary, 기존 fuse_selective 는 ablation 으로 보존

가 능한 user 결정:
1. **Full fix**: 새 함수 구현 + 기존 함수 deprecated 처리 + 모든 phase 재실행 (Phase 6c, 7c, 8-step1 — 약 $1)
2. **Side-by-side**: 새 함수 추가 + 작은 검증 만 (Phase 6 n=20 만 새 함수 로 비교) (약 $0.20)
3. **Document only**: 차이점 문서화 만 (코드 변경 없음) — 사용자 가 "수정 필요시 즉각 수정" 했으므로 이 옵션 은 user intent 와 충돌

권장: **2 (side-by-side)** → 변동 시 1 로 확장.

## 5. 결론 표

| 메커니즘 | 동일 여부 | 비고 |
|---|---|---|
| KV cache 저장 표현 | ✓ 수학적 등가 | pre vs post RoPE (RoPE 가역성) |
| HKVD 계산 (squared L2, fp32) | ✓ 비트 등가 | dim 축 명명만 다름 |
| Top-K 선정 (int(N*r), max 1, sort asc) | ✓ 동일 | 우리 v4 가 min guard 추가 |
| RoPE 적용 방식 | ✓ 결과 등가 | HF apply_rotary vs FusedRope kernel |
| Causal mask (sparse Q) | n/a | sparse forward 가 다름 |
| **Selective forward 폭** | **✗ DIFFERENT** | full vs sparse — fix 필요 |
| Chunking | n/a | LMCache 는 chunk 추상화 없음 (vLLM 요청) |
| Pipelined / prefix-cache | n/a | 우리 v4 전용 ablation |
| Gradual | n/a | 양쪽 다 미구현 |
