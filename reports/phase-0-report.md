# Phase 0 — Setup & Environment Parity Report

> Auto-generated. v4 첫 phase. CPU only ($0).

## 1. Outcome

**Result**: PASS (15/15 conditions).

- Mac venv parity ✓
- External 저장소 (LMCache + mydata) clone + SHA 검증 ✓
- Skeleton import + smoke test 11/11 ✓
- LMCache 분석 (≥10 file:line 인용; 실제 24 회) ✓
- Figure 12-like disclosure (12 차이점) ✓

## 2. Environment label (L09 의무)

### Mac venv (Phase 0 실행 환경)

| Package | Version |
|---|---|
| python | 3.11.14 |
| torch | 2.4.1 (CPU wheel) |
| transformers | 4.49.0 |
| accelerate | 1.13.0 |
| huggingface-hub | 0.36.2 |
| tokenizers | 0.21.4 |
| safetensors | 0.7.0 |
| datasets | 4.8.5 |
| numpy | 2.4.4 |
| sentence-transformers | 4.1.0 |
| rouge-score | 0.1.2 |
| scipy | 1.17.1 |
| matplotlib | 3.10.9 |
| pandas | 3.0.2 |
| python-dotenv | 1.2.2 |
| pytest | 9.0.3 |
| ruff | 0.15.12 |

`bash scripts/diff_env.sh` 결과: **8/8 핀된 패키지 정확히 match** (range pin 인 numpy/sentence-transformers/rouge-score 등 제외 SKIP).

### Pod GPU 환경

해당 사항 없음 — Phase 0 은 Mac CPU only. Phase 1 진입 시 Pod 부팅 후 `scripts/diff_env.sh` 재실행 의무.

## 3. Deliverables

| 산출물 | 경로 | 라인 수 / 메타 |
|---|---|---|
| LMCache 분석 | `docs/lmcache-analysis.md` | 380 라인, file:line 인용 24 회 |
| Figure 12-like disclosure | `docs/figure12_like_disclosure.md` | 101 라인, 12 차이점 명시 |
| Skeleton (12 modules) | `src/cacheblend/*.py` | runners.py 120 라인 (1 fix), 나머지 stub |
| Smoke tests | `tests/test_smoke.py` | 11/11 PASS (3.07s) |
| External: LMCache | `external/LMCache/` | depth=1, HEAD `7657836e070b9211ed43294e13f1e4c81716dcf6` (2026-05-07 12:56:38 +0800) |
| External: mydata | `external/mydata/` | depth=1, prompts.jsonl SHA `791e1cf5…3a8e21` ✓ |
| Design entries | `docs/design-decisions.md` §11 | Pre-RoPE K + RoPE shift 항목 추가 |

## 4. Code changes

### `src/cacheblend/runners.py:50-58`

**변경**: `_RunnerBase.__init__` 분기 조건 강화.

**Before**:
```python
def __init__(self, model=None, tokenizer=None):
    if _HARNESS_AVAILABLE:
        super().__init__(model=model, tokenizer=tokenizer)
    else:
        self.model = model
        self.tokenizer = tokenizer
```

**After**:
```python
def __init__(self, model=None, tokenizer=None):
    if _HARNESS_AVAILABLE and model is not None:
        super().__init__(model=model, tokenizer=tokenizer)
    else:
        self.model = model
        self.tokenizer = tokenizer
```

**이유**: mydata harness 의 `CacheBlendRunner.__init__` 이 `next(model.parameters()).device` 를 호출하므로, smoke test 의 `model=None` 경로에서 즉시 `AttributeError`. Phase 0 의도는 "model 없이 stub instantiation 가능" 이므로 model=None 이면 super() 호출 회피. → L31 (v5-lessons).

## 5. LMCache 분석 핵심 (Q1~Q5 1~2 줄 요약)

| Q | 답 (요약) |
|---|---|
| Q1 — token slicing | `compute_layer` 의 `qkv_proj` 는 전체 토큰. Slicing 은 `process_qkv` 안의 check_layer 에서만 (`blender.py:103-105`). v4 hook-injection 과 다름 (full-length 유지). |
| Q2(a) — norm | `sum((k_new − k_old)^2, dim=1)`, fp32 casting. (`blender.py:89-91`) |
| Q2(b) — layer | `check_layers: list[int]` (`metadata.py:11-18`), config key `blend_check_layers` (`config.py:128-131`). 실제 default 는 `[1]` (single, `blend.py:34`). |
| Q2(c) — pre/post-RoPE | Fresh k 는 RoPE 후 (`blender.py:86`), cached old_k 는 vLLM post-RoPE @ chunk-local positions. **RoPE shift 미실시** (`fused_encode` 정의 있으나 hot path 미사용). |
| Q2(d) — top-K | `int(N × ratio[0])`, `max(1)`, `topk` → resort (`blender.py:94-101`). 정규화 없음. ratio=0 이라도 1 토큰 선택 (boundary 누수). |
| Q3 — Pre-RoPE K | LMCache 미저장. v4 는 Phase 2 에서 별도 저장 + RoPE shift (paper §4 충실). |
| Q4 — single vs gradual | Code 가 list 받지만 `recomp_ratios[0]` hardcode + per-layer threshold TODO (`blender.py:43-45`). 실제 default `[1]/0.15`. v4 Phase 8 의 multi-CL gradual 이 신규 contribution. |
| Q5 — storage | 5 backends: LocalCPU/LocalDisk/P2P/Remote/GDS (`storage_backend/__init__.py:15-20`). Default CPU 5GB, disk off. v4 는 단일 in-RAM dict + StorageProfile cost model (Phase 4). |

자세한 인용은 `docs/lmcache-analysis.md` 참조.

## 6. Figure 12-like Disclosure 12 차이점 (요약 list)

1. Embedding: `all-mpnet-base-v2` (paper 비공개)
2. L2 raw, no normalization (paper 비공개)
3. L2 distance ascending retrieval (cosine 추정 vs L2)
4. GPT-4 simulated query 제외 (paper extended set)
5. Top-K = 6 docs/sample (paper 비공개)
6. `random.Random(42)` 단일 인스턴스 sequential shuffle (per-sample 다른 순서)
7. MuSiQue 200 만 (paper 4 dataset 중 1, 2WikiMQA/SAMSum/MultiNews v5 미룸)
8. Models: Mistral-7B-Instruct-v0.2 + Llama-3.1-8B + 70B 8-bit
9. Precision: FP16 (Mistral, 8B) + 8-bit bitsandbytes (70B)
10. Metrics: F1 + Rouge-L only (TTFT 비목표 L27)
11. Statistical test: paired bootstrap CI 95%, ci_low > 0 (L28)
12. Tolerance: 4 categories frozen (paper 비공개)

자세한 디테일은 `docs/figure12_like_disclosure.md` 참조.

## 7. Gate 결과 (auto-evaluated)

`scripts/eval_gate.py --phase 0` 으로 평가, 결과는 `gates/gate-0-result.json`. 상세는 §8.

## 8. Cost

- Phase 0 누적: **$0** (CPU only, no Pod).
- 누적 (Phase 0~5 한도 $5): $0 / $5.

## 9. v5-lessons (이번 phase 에서 발견된 사항)

이번 phase 에서 새로 추가된 lesson:

- **L31** — Runner stub instantiation 이 harness ABC 와 충돌: smoke test `model=None` 경로에서 `next(model.parameters()).device` AttributeError 발생. `_RunnerBase.__init__` 분기 조건을 `_HARNESS_AVAILABLE and model is not None` 로 수정해 fix.

상세는 `docs/notes/v5-lessons.md` 참조.

## 10. Phase 1 진입 사전점검

`tasks/phase-1-layerwise.md` 의 acceptance criteria 사전 확인은 Pod 부팅 직전에 별도 수행 (Phase 1 부터는 GPU 필요). Phase 0 PASS 시점에선 venv 정합 + skeleton + smoke test 만 보장.

Phase 1 진입 직전 추가 의무:
1. Pod 부팅 → CLAUDE.md §3 의 6 단계 setup 수행 (HF cache, rsync, mydata clone + SHA, `pip install`, torch CUDA 검증, HF auth).
2. `bash scripts/diff_env.sh` Pod 에서 재실행 → 8/8 match 확인.
3. `pytest tests/ -m "not slow and not requires_model"` Pod 에서 (gpu marker 자동 활성화) 재실행.
