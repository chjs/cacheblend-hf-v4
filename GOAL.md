# GOAL — CacheBlend HF v4

> v3 진행 중 누적된 30개 lesson + Phase 8 (Gradual Filtering) discovery experiment + mydata harness 통합을 반영한 두 번째 iteration.
> v4 진행 중 발견되는 새로운 사항은 `docs/notes/v5-lessons.md`에 누적.

## 핵심 변경 (v3 → v4)

1. **TTFT는 비목표**. Quality-only milestone. Hook-injection 구현 그대로 유지 (hidden_state slicing 같은 architectural 침습 회피). [L27]
2. **Phase 8 신규 — Gradual Filtering Discovery Experiment**. 논문 §4.3 multi-check-layer gradual scheme의 효과를 LMCache의 단순 check_layer=1 flat schedule과 head-to-head 비교. Interactive multi-step. [L22]
3. **mydata harness 통합**. 사용자의 [chjs/mydata](https://github.com/chjs/mydata) 저장소 `cacheblend_fig12/` 가 사전 빌드된 prompts.jsonl + Runner 인터페이스 + F1/Rouge-L metrics 제공. v4는 이 위에 CacheBlendRunner 서브클래스로 결합.
4. **환경 정합 first**. Phase 0의 첫 단계가 Mac venv ↔ Pod 패키지 8개 일괄 비교 + 정합 자동 검증. [L01, L08]
5. **Tolerance 4단계 freeze**. Phase 시작 전 카테고리 결정, retroactive 변경 금지. [L05, L13, L16]
6. **Boundary safe-shortcut 명시 디자인 패턴**. ratio=0/1에서 코드 경로 동일 보장 → max_diff=0. [L13]
7. **v5-lessons 자동 누적 인프라**. 매 phase 보고서에 lesson 섹션 의무. `scripts/add_lesson.py` CLI.

## 데이터 — mydata cacheblend_fig12 사용

**입력**: `external/mydata/cacheblend_fig12/prompts.jsonl` (200 sample, MuSiQue v1.0 dev)

각 sample은 사전 빌드됨:
- 한 multi-hop 질문에 대해 MuSiQue paragraphs 20개 중 L2 거리 top-6 선정
- Embedding: `sentence-transformers/all-mpnet-base-v2` (raw L2, no normalization)
- Top-6를 `random.Random(42).shuffle(...)` 로 random order
- Prompt: `system + 6 docs + question + "Answer:"`

검증: SHA256 = `791e1cf50d984f27b314c8abd49f25e3b27a0a1598a6cfcf53e28d13868a3e21`

**Disclosure**: paper Figure 12와 bit-identical은 아님 (GPT-4 simulated query 제외, all-mpnet-base-v2 사용 — paper는 SentenceTransformer 모델명 비공개).

**범위**: Musique 200만 (1/4 of paper). 2WikiMQA / SAMSum / MultiNews는 v5로 미룸.

## Harness — mydata 의존성

`external/mydata/cacheblend_fig12/harness/`:

- `runner.py`: `CacheBlendRunner` ABC + `FullPrefillRunner` (baseline)
- `metrics.py`: F1, Rouge-L (YaoJiayi/CacheBlend `utils.py` 포팅)
- `eval.py`: argparse 기반 메인 실험 루프

v4는 이 위에 우리의 Runner 서브클래스를 추가:

```
src/cacheblend/runners.py (신규):
  class FullRecomputeRunner(CacheBlendRunner)    # baseline 별도 구현
  class FullReuseRunner(CacheBlendRunner)        # full KV reuse
  class PrefixCacheRunner(CacheBlendRunner)      # prefix caching
  class CacheBlendV4Runner(CacheBlendRunner)     # 우리 selective recompute
  class GradualV4Runner(CacheBlendRunner)        # Phase 8 gradual filtering
```

각 Runner는 v4의 fusor 함수들 (`fuse_full_recompute`, `fuse_selective`, ...)을 wrap.

## 평가 메트릭 (Quality-only)

- **F1**: harness/metrics.py (token-based, max-over-aliases)
- **Rouge-L**: harness/metrics.py
- **Paired bootstrap CI**: F1(cacheblend) > F1(full_reuse) at 95% CI [L28]
- **HKVD elbow shape** (Phase 3): paper Figure 6 비교
- **TTFT**: harness 자동 측정. 보고만, gate 조건 아님 [L27]

## Tolerance 카테고리 (Freeze)

| 카테고리 | 정의 | 사용 처 |
|---|---|---|
| `IDENTICAL_PATH` | max_diff = 0 | Boundary cases (ratio=0, ratio=1) [L13] |
| `SAME_SHAPE` | max_diff < 1e-3 | Phase 1: layerwise vs standard |
| `MIXED_SHAPE` | argmax exact + max_diff < 5e-2 | cuBLAS shape difference [L05] |
| `RECOMPUTE_PATH` | max_diff < 1e-3 | 같은 fused shape |

새 카테고리 추가는 `docs/design-decisions.md`에 정량 근거 + 리뷰.

## Phase 구성 (8 phases + 1 discovery)

| Phase | 이름 | 환경 | 비용 추정 | 비고 |
|---|---|---|---|---|
| 0 | Setup & Env Parity | Mac CPU + Pod sanity | $0 | Mac/Pod 8 패키지 일괄, mydata clone + SHA 검증 |
| 1 | Layerwise Forward | Pod GPU | $0.5 | LayerwiseModel + pre-RoPE K hook |
| 2 | KV Storage & Full Reuse | Pod GPU | $0.5 | KVStore + RoPE shift + fuse_full_reuse |
| 3 | Selective Recompute | Pod GPU | $0.7 | HKVD + fuse_selective + boundary shortcut |
| 4 | Pipelining & Prefix Cache | Pod GPU | $0.5 | Async prefetch + prefix_cache baseline |
| 5 | Dataset Pipeline | Mac CPU | $0 | mydata prompts.jsonl 검증 + Runner 인터페이스 wrap |
| 6 | Mistral Evaluation | Pod GPU | ~$10 | Sub-phase 6a (20) / 6b (50) / 6c (200), F1 only |
| 7 | Llama Evaluation | Pod GPU | ~$15 | Llama-3.1-8B + 70B 8-bit |
| 8 | Gradual Filtering Discovery | Pod GPU | ~$25 | Interactive multi-step, Musique 100 sample/schedule [L22] |

**Total estimated cost**: ~$52 (v3 ~$30 + Phase 8 추가 $22).

## 모델

- **Mistral-7B-Instruct-v0.2** (FP16): Phase 1~6, 8.
- **Llama-3.1-8B-Instruct** (FP16): Phase 7, 8.
- **Llama-3.1-70B-Instruct** (8-bit bitsandbytes): Phase 7만 (Phase 8 제외, 비용).

## 데이터셋

- **MuSiQue** (multi-hop QA): mydata cacheblend_fig12 사전 빌드. Phase 6, 7, 8.
- 2WikiMQA, SAMSum, MultiNews: **v5로 미룸**.

## 비목표 (out of scope)

- TTFT 향상 — 측정/보고만, gate 조건 아님 [L27].
- Hidden_state slicing — hook-injection 디자인 유지.
- 2WikiMQA / SAMSum / MultiNews 평가 — v5.
- GPT-4 simulated query (paper의 "Musique extended") — 비공개성 + 비용.
- Production-ready optimization — 연구 목적 정확성 검증만.

## v5-lessons 자동 누적

v4 진행 중 발견되는 새로운 문제 / 개선 사항은 `docs/notes/v5-lessons.md`에 누적:

1. 모든 phase 보고서에 **"v5-lessons 섹션" 의무** (없으면 "없음" 명시).
2. `scripts/add_lesson.py` CLI로 표준 형식 추가.
3. 사용자 검토 시 lesson 추가/수정/철회 가능 (철회는 strike-through로 보존).

자세한 워크플로는 `CLAUDE.md`의 §v5-lessons 누적 참조.

## Cross-references

- v3 누적 lessons (30개): `docs/notes/v4-lessons.md`
- v4 진행 중 누적: `docs/notes/v5-lessons.md`
- Phase 명세 + gate 통합: `tasks/phase-N-*.md`
- Design decisions: `docs/design-decisions.md`
- mydata harness: `external/mydata/cacheblend_fig12/`
- LMCache 분석: `docs/lmcache-analysis.md`
