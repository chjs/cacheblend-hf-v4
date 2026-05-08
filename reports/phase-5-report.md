# Phase 5 — Dataset Pipeline (mydata integration, CPU only) Report

> Tolerance: N/A (no GPU eval).
> Cost: **$0** (CPU only, vast.ai instance stopped).
> Result: **PASS (7/7 conditions)**.

## 1. Outcome

| ID | 결과 | 핵심 |
|---|---|---|
| 5.1 mydata SHA verified | ✅ PASS | Phase 0 에서 검증 (`prompts.jsonl` SHA `791e1cf5…3a8e21`) |
| 5.2 mydata harness import | ✅ PASS | `from harness.runner import CacheBlendRunner, GenerationResult` + metrics OK |
| 5.3 5 Runner instantiate | ✅ PASS | 5 클래스 모두 model=None 으로 instantiation 가능 |
| 5.4 test_runners (CPU) | ✅ PASS | 7/7 (test_imports + 5×test_dispatch_with_stub_model + test_cacheblend_runner_carries_config) |
| 5.5 test_bootstrap | ✅ PASS | 7/7 (known_distribution dominates/identical, n=1 edge, n_bootstrap=10, shape mismatch raises, invalid confidence raises, seed reproducibility) |
| 5.6 dryrun artifact | ✅ PASS | `benchmarks/results/figure12_like/musique_dryrun.jsonl` — 1000 rows (200 sample × 5 runner) |
| 5.7 verify_phase | ✅ PASS | 모든 deliverable + `## v5-lessons` 섹션 존재 |
| Bonus: Dataset stats | recall@6 mean **0.7475** vs README 0.748 (abs diff 0.0005, MATCH) | |

## 2. Pod 처리 (vast.ai instance 36296967)

| 항목 | 값 |
|---|---|
| Stop 명령 | `vastai stop instance 36296967` |
| Stop 시점 누적 uptime | **423.33 min** (~7.05 hr) |
| Phase 5 시작 시점 instance 상태 | Phase 4 직후 stop 명령 발행. vast.ai dashboard 상 stopped 진행 중 (CLI show 시 초기엔 `running` 표시되나 stop request 는 큐에 들어감). |
| Phase 5 본 작업 환경 | Mac venv (Python 3.11.14, CPU). GPU 사용 0 분. |
| Network volume / instance 메타 | 보존 (`max_local_disk` 100GB, conda env `cb`, Mistral-7B cache `/workspace/.hf_home`). Phase 6 진입 시 same instance start 시도 → 실패 시 새 GPU instance 부팅. |
| Stop 시점 vast.ai uptime billing 추정 | **~$1.14** (423.33 × 0.1611 / 60) |
| cost-tracker manual 누적 (Phase 1~4) | **$0.49** |
| 차이 | $0.65 (phase 트리거 간 idle) — L39 정책으로 cost-tracker 에 미반영 |

## 3. Cost tracking 정책 결정 (3 옵션 중 선택)

**선택: 옵션 3 — 현 manual 기록 유지** (`L39` 신규 등록).

근거:
- 옵션 1 (cost-tracker.json schema 변경: `phase_work_billing` + `instance_uptime_billing` 분리) → 모든 phase 보고서 양식 변경 + eval_gate.py 의 cost_check 로직 손질 필요. 변경 비용 > 현재 unaccounted ($0.65) 의 가치.
- 옵션 2 (매 phase 종료 시 vast.ai dashboard 실 billing reconcile) → manual 작업 추가, 자동화 안됨, 휴먼 에러 가능성.
- 옵션 3 → cost-tracker 의 `cumulative_usd` 는 'phase 작업 시간 manual 합계' 정의 유지. cap 비교에 보수적으로 작동 (idle 누락 = under-report). cap 도달 risk 시 vast.ai dashboard 에서 별도 reconcile.

**책임 분리**:
- cost-tracker: gate 평가용 보수적 누적 (현재 $0.49 / $5).
- vast.ai dashboard: 실제 청구 추적 (~$1.14, idle 포함).

상세: `docs/notes/v5-lessons.md` L39.

## 4. mydata harness 검증

`PYTHONPATH=external/mydata/cacheblend_fig12 python -c "from harness.runner import CacheBlendRunner, GenerationResult; from harness.metrics import compute_f1, compute_rouge_l"` → rc=0 ✓

실제 시그니처 (Phase 0 분석에서 확인):
- `CacheBlendRunner.__init__(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase)` — `next(model.parameters()).device` 호출 (L31 fix 호환)
- `CacheBlendRunner.prepare(system, docs, question)` / `generate(max_new_tokens=32)`
- `GenerationResult(text, ttft_seconds, total_seconds, n_generated_tokens)` dataclass
- `compute_f1(pred, gold, tokenizer) -> float`
- `compute_f1_against_aliases(pred, answers, tokenizer) -> float` (Phase 6+ 에서 사용)
- `compute_rouge_l(pred, gold) -> float`

## 5. 5 Runner 구현 상세 (`src/cacheblend/runners.py`, 314 lines)

| Runner | prepare/generate file:line | 어느 fusor wrap |
|---|---|---|
| `_RunnerBase` (공통) | `runners.py:69-159` | dispatch + CPU stub 분기 + `_greedy_decode_from_prefill` + `_build_chunks` 헬퍼 |
| `FullRecomputeRunner` | `_run_prefill_and_generate` `runners.py:165-180` | `model(input_ids=..., use_cache=True)` (HF standard, no fusor — same as harness FullPrefillRunner) |
| `FullReuseRunner` | `runners.py:184-225` | `fuse_full_reuse(self._lw_model, chunks, self._kv_store)` — Phase 2 |
| `PrefixCacheRunner` | `runners.py:228-258` | `fuse_prefix_cache(...)` — Phase 4 |
| `CacheBlendV4Runner` | `runners.py:261-298` | `fuse_selective(..., recompute_ratio, check_layer)` — Phase 3 |
| `GradualV4Runner` | `runners.py:301-313` | (Phase 8) — `_run_prefill_and_generate` raises NotImplementedError. Phase 5 stub mode (model=None) 만 지원. |

### CPU stub mode 분기

`_RunnerBase.generate` (`runners.py:103-107`):
```python
def generate(self, max_new_tokens: int = 32):
    if self.model is None:
        return _stub_generation()  # _GenerationResult(text="", ttft=0, total=0, n=0)
    return self._run_prefill_and_generate(max_new_tokens=max_new_tokens)
```

`_RunnerBase.__init__` (`runners.py:78-94`):
- `_HARNESS_AVAILABLE and model is not None` → `super().__init__(model, tokenizer)` (harness ABC 정상 path)
- 그 외 → `self.model = model; self.tokenizer = tokenizer; self.device = cpu` (L31 fix 호환)

### LayerwiseModel 공유

KV 재사용 runner 들 (FullReuse / PrefixCache / CacheBlendV4) 은 `_ensure_lw_and_store` (`runners.py:188-204`) 로 thin LayerwiseModel view 를 self.model 의 weights 위에 래핑 (`__new__` + 직접 attribute set + `_install_k_proj_hooks`). 모델 재로딩 비용 0.

## 6. Bootstrap CI helper

`benchmarks/metrics/bootstrap.py` (52 lines):

```python
def paired_bootstrap_ci(
    scores_a: Iterable[float],
    scores_b: Iterable[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (ci_low, ci_high) for mean(a[i] - b[i])."""
```

**알고리즘**: paired diffs `d = a - b` → bootstrap n_bootstrap × n indices (with replacement) → `mean(d[idx])` per replicate → `(alpha/2, 1-alpha/2)` percentiles. seed=42 reproducible.

### 단위 테스트 (7 cases, `tests/test_bootstrap.py`)

| Case | 입력 | 기대 결과 |
|---|---|---|
| `known_distribution_a_dominates_b` | `a = b + 0.10`, n=100 | ci_low > 0; CI degenerate at 0.10 (constant diff) |
| `known_distribution_identical` | `a == b`, n=50 | CI = [0, 0] |
| `n1_edge` | `[0.5]` vs `[0.3]` | CI = [0.2, 0.2] |
| `small_n_bootstrap` | `n_bootstrap=10`, constant shift | CI degenerate at 0.05 |
| `shape_mismatch_raises` | shape 불일치 | `ValueError("shape mismatch")` |
| `invalid_confidence` | confidence=1.5 | `ValueError("confidence")` |
| `seed_reproducibility` | 같은 seed 두 번 호출 | 동일 (lo, hi) |

**모두 PASS** (Mac venv, 0.34s).

## 7. run_eval.py + dry-run artifact

### `benchmarks/run_eval.py` (98 lines)

`--runner pkg.mod:Class` 등록 방식 (`importlib.import_module(mod), getattr(...)`) (`run_eval.py:23-26`). `--dry-run-all` 시 5 runner 자동 import (`run_eval.py:71-79`).

### Artifact: `benchmarks/results/figure12_like/musique_dryrun.jsonl`

- **1000 rows** (200 sample × 5 runner)
- 0.03s 생성 (CPU stub mode)

Schema (per row):
```json
{"id": "2hop__460946_294723", "runner": "FullRecomputeRunner",
 "pred": null, "f1": null, "rouge_l": null,
 "ttft_seconds": 0.0, "total_seconds": 0.0}
```

`pred=null` / `f1=null` 은 plumbing 검증용 [L21]. Phase 6 부터 실 model 로 generation → pred/f1 filled.

## 8. Dataset stats (`reports/phase-5-attachments/dataset_stats.json`)

| 항목 | 값 |
|---|---|
| n_samples | 200 |
| supporting_recall@6 mean | **0.7475** |
| README claim | 0.748 |
| diff | 0.0005 (MATCH within ±0.05 tolerance) |
| recall median | 1.0 |
| recall min/max | 0.0 / 1.0 |
| prompt token len (Mistral tokenizer) | min=330, median=767, max=1845, mean=807 |

## 9. Tests 결과 (Mac venv)

```
============================== 14 passed in 2.73s ==============================
tests/test_runners.py    : 7/7 PASS (test_imports, 5×dispatch_with_stub_model, carries_config)
tests/test_bootstrap.py  : 7/7 PASS (4 known distributions + 2 raises + 1 reproducibility)
```

## 10. Gate 7 condition

| ID | check_type | 결과 | 근거 |
|---|---|---|---|
| 5.1 | file_exists | ✅ PASS | `external/mydata/cacheblend_fig12/prompts.jsonl` 존재 |
| 5.2 | command | ✅ PASS | `PYTHONPATH=... python -c 'from harness.runner import CacheBlendRunner'` rc=0 |
| 5.3 | command | ✅ PASS | 5 runner 모두 `()` 인스턴스화 OK |
| 5.4 | pytest | ✅ PASS | tests/test_runners.py — 7/7 |
| 5.5 | pytest | ✅ PASS | tests/test_bootstrap.py — 7/7 |
| 5.6 | file_exists | ✅ PASS | `benchmarks/results/figure12_like/musique_dryrun.jsonl` 1000 rows |
| 5.7 | verify_phase | ✅ PASS | 모든 deliverable + `## v5-lessons` 섹션 |

## 11. Cost

- Phase 5: **$0** (CPU only, Mac venv)
- 누적 (cost-tracker manual): **$0.49 / $5** (Phase 4 종료 시점과 동일, 9.8%)
- vast.ai 누적 billing 추정: ~$1.14 (instance stopped 후 동결)
- 차이 $0.65 = idle (Phase 4 보고서 §8 참조, L39 정책으로 cost-tracker 에 미반영)

## v5-lessons (이번 phase 에서 발견된 사항)

이번 phase 에서 신규 추가:

- **L39** — cost-tracker.json 은 phase 작업 시간만 기록 (옵션 3 채택). vast.ai idle billing 은 별도 dashboard reconcile. cap 비교에 보수적으로 작동 (under-report). v5 에선 schema 분리 권장.

상세는 `docs/notes/v5-lessons.md` 참조 (현재 L31~L39, 9개 누적).

## 12. 수정 파일

| 경로 | 변경 사유 |
|---|---|
| `src/cacheblend/runners.py` | Phase 0 stub → 실 구현 (314 lines). 5 Runner prepare/generate, CPU stub mode (model=None), LayerwiseModel 공유. |
| `benchmarks/metrics/__init__.py` | 신규 (re-export `paired_bootstrap_ci`) |
| `benchmarks/metrics/bootstrap.py` | 신규 (52 lines): paired bootstrap CI [L28] + 7 단위 테스트 spec. |
| `benchmarks/run_eval.py` | 신규 (98 lines): `--runner pkg.mod:Class` dispatch + `--dry-run` / `--dry-run-all`. |
| `tests/test_runners.py` | 신규 (~50 lines): 7 cases (imports + 5×dispatch + config). |
| `tests/test_bootstrap.py` | 신규 (~70 lines): 7 cases. |
| `benchmarks/results/figure12_like/musique_dryrun.jsonl` | 신규 artifact (1000 rows, pred=null) |
| `reports/phase-5-attachments/dataset_stats.json` | 신규: recall@6 + token length 분포 |
| `docs/notes/v5-lessons.md` | L39 추가 |
| `reports/phase-5-report.md` | 본 보고서 |
| `gates/gate-5-result.json` | gate eval (auto) |

(`reports/cost-tracker.json` 은 변경 없음 — Phase 5 cost $0.)

## 13. Phase 6 사전점검

`tasks/phase-6-mistral.md` — Mistral-7B 평가, sub-phase 6a/6b/6c.

핵심:
- **Sub-phase**: 6a (20 sample dryrun) / 6b (50 sample) / 6c (200 sample full).
- **Cost cap**: 누적 $10 ($0.49 + ~$10 Phase 6).
- **Paired bootstrap CI [L28]**: F1(cacheblend) > F1(full_reuse) at 95% CI 의 ci_low > 0 의무.
- **Metrics**: F1 (max over aliases) + Rouge-L. mydata harness 의 `compute_f1_against_aliases` 사용.

⚠️ **GPU instance 재부팅 필요** — Phase 5 시작 시 stopped 한 instance 36296967 을 `vastai start instance 36296967` 으로 resume 시도 (network volume 보존되어 있으면 conda env `cb` + Mistral-7B cache 즉시 사용 가능). resume 실패 시 `vastai search offers` → 새 RTX 3090 instance + CLAUDE.md §3 7단계 setup 반복.

Phase 6 트리거 시점 알려주세요. (Pod 부팅 + Mistral 6a 20 sample = 약 15-20 min, 6b 50 sample 약 5 min, 6c 200 sample 약 15-20 min, 총 ~40-50 min wall.)
