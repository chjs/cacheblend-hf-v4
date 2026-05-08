# Design Decisions — CacheBlend HF v4

> 핵심 디자인 결정과 정량 근거. 변경 시 v5-lessons.md에 기록 의무.

## 1. Tolerance 카테고리 (4개, freeze)

v3에서 tolerance가 phase마다 인플레이션되는 패턴 발견 [L05, L13, L16]. v4는 처음부터 4개 카테고리로 freeze.

| 카테고리 | 정의 | 사용 처 | 근거 |
|---|---|---|---|
| `IDENTICAL_PATH` | max_diff = 0 | Boundary safe-shortcut (ratio=0/1) | 코드 경로 동일 |
| `SAME_SHAPE` | max_diff < 1e-3 | Phase 1 layerwise vs standard | bit-exact 가능, eager attention |
| `MIXED_SHAPE` | argmax exact + max_diff < 5e-2 | cuBLAS shape difference | FP16 ULP, S=11 vs S=18 kernels |
| `RECOMPUTE_PATH` | max_diff < 1e-3 | 같은 fused shape (e.g. ratio=1) | 같은 cuBLAS path |

**Phase 시작 전 카테고리 결정. Retroactive 변경 금지.**

새 카테고리 추가 절차:
1. 정량 근거 (측정값 + 분포)
2. 이론적 설명 (왜 이 bound인가)
3. 이 파일에 entry 추가
4. 사용자 리뷰

## 2. Boundary safe-shortcut 디자인 패턴 [L13]

`fuse_selective(recompute_ratio)` 의 boundary cases는 코드 경로 동일화로 max_diff = 0 보장:

```python
def fuse_selective(model, chunks, kv_store, recompute_ratio: float, ...):
    if recompute_ratio == 0:
        return fuse_full_reuse(model, chunks, kv_store)
    if recompute_ratio >= 1:
        return fuse_full_recompute(model, chunks)
    # else: actual selective logic
```

이 패턴은 v3 Phase 3에서 max_diff = 0.000e+00을 가능하게 한 핵심 결정. v4 fusor.py 첫 줄부터 박힘.

## 3. Hook-injection vs hidden_state slicing [L27]

v3/v4는 hook-injection 사용 (k_proj/v_proj forward-hook으로 cached 위치 출력 override).

**장점**: HF 코드 비침습, 약 300 LoC, Phase 1-3 correctness PASS.

**단점**: q_proj / MLP / RMSNorm 모두 fully 계산 → TTFT 절감 없음.

**v4 결정**: TTFT는 비목표 [L27]. Hook-injection 유지. Hidden_state slicing 침습 회피.

## 4. mydata 저장소 의존성

v4는 [chjs/mydata](https://github.com/chjs/mydata) 의 `cacheblend_fig12/` 사용:
- `prompts.jsonl` 사전 빌드 (200 sample, MuSiQue v1.0 dev)
- `harness/runner.py` Runner ABC
- `harness/metrics.py` F1, Rouge-L

SHA256 = `791e1cf50d984f27b314c8abd49f25e3b27a0a1598a6cfcf53e28d13868a3e21`.

**Disclosure** (paper와의 차이, `docs/figure12_like_disclosure.md` 참조):
- Embedding: `all-mpnet-base-v2` (paper 비공개)
- L2 raw (no normalization)
- GPT-4 simulated query 제외
- Random shuffle: `random.Random(42)` 단일 인스턴스
- 2WikiMQA / SAMSum / MultiNews 제외 (v5)

## 5. Per-item seeded shuffle [L20]

mydata가 `random.Random(42)` 단일 인스턴스로 200 sample 순차 shuffle. 같은 sample = 같은 순서. 다른 sample = 다른 순서.

이 디자인의 의도: prefix caching baseline 약화. 만약 모든 sample이 같은 문서 순서면 prefix cache가 우연히 cacheblend만큼 잘 나옴 (false equivalence).

## 6. F1 평가 — Bootstrap CI [L28]

F1 절대값 0.05 차이 비교 대신 paired bootstrap CI 사용:
- n_bootstrap = 1000
- confidence = 0.95
- 통과 조건: ci_low > 0 → "F1(cb) > F1(baseline) at 95% CI"

이는 F1 noise (Musique multi-hop의 답 정렬 문제) 위에서 statistical significance를 확보.

## 7. Phase 8 — Discovery vs Validation 구분

v3 Phase 0~7: Pure full auto, 알고리즘 검증.
v4 Phase 8: **Interactive discovery experiment**. 사용자 검토 4회 포함.

이 구분이 v4의 새 패턴. PHASES.md에 명시.

## 8. v5-lessons 누적 인프라

매 phase 보고서에 "v5-lessons 섹션" 의무. `scripts/add_lesson.py` CLI. `scripts/retract_lesson.py` 로 strike-through 보존.

## 9. 환경 정합 — Mac venv ↔ Pod [L01, L08]

Phase 0의 첫 단계가 8 패키지 일괄 비교 (`scripts/diff_env.sh`):
torch, transformers, datasets, accelerate, huggingface-hub, tokenizers, safetensors, numpy.

requirements.txt에 정확한 == 핀.

## 10. Pod 운영 — Reclaim + auto-recreate [L07, L23]

`runpod.sh up --auto-recreate`: Pod start 실패 시 자동 terminate + new pod. Network volume 데이터 보존.

장시간 phase (6c/7c/7d/8-step3): incremental jsonl checkpoint per 50 sample. Reclaim 시 재시작에서 skip.

## 11. Pre-RoPE K 저장 + retrieve 시 RoPE shift [Phase 0 분석에서 추가]

논문 §4 의 핵심 디자인 충실도 항목. LMCache (`external/LMCache/lmcache/v1/compute/blend/blender.py:70-91`) 는 vLLM KV cache 위에 의존하므로 **post-RoPE K 만** 사용 — chunk 가 standalone prefill 될 때의 chunk-local positions 으로 RoPE 가 박혀있고, blending 시점에 RoPE shift 를 하지 않는다 (`FusedRope.fused_encode` 정의는 있으나 hot path 미사용).

이 결과 LMCache 의 KV deviation 신호는 (a) 진짜 content drift + (b) chunk-local→blended position mismatch 로 인한 RoPE 차이가 합쳐진 양이다.

**v4 결정**: pre-RoPE K 를 별도 저장 (Phase 1 layerwise hook) + retrieve 시 chunk-local→blended global position 으로 RoPE 재적용 (Phase 2 `apply_rope_shift`).

**근거**:
1. Paper §4 의 deviation 정의는 "RoPE 적용 후" 양 K 모두 동일 position 가정.
2. RoPE shift 없으면 chunk 길이가 길거나 chunk 위치가 깊을수록 deviation 이 sample 별로 다른 noise 를 받음 → HKVD ranking 의 reproducibility 저하.
3. v4 는 hook-injection 디자인 (§3) 으로 forward-hook 으로 pre-RoPE K 캡처가 자연스럽다.

**비용**: KVStore 가 pre-RoPE K + post-RoPE K 둘 다 들고 있을 필요 없음 (post-RoPE 는 retrieve 시점에 fused 로 적용). 메모리는 동일.

**Cross-reference**: `docs/lmcache-analysis.md` §Q3.
