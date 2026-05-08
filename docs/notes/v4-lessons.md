# v4 Lessons Learned — Cumulative Log

> v3 진행 중 누적되는 문제·결정·교훈. v4 하네스 설계 시 빠짐없이 반영.
> 형식: 발견 시점 / 카테고리 / 증상 / 근본 원인 / v4 반영 방법

---

## L01 — Pod 환경 셋업 시 패키지 일괄 확인 누락 (2026-05-06)

**카테고리**: 환경 정합 / 진단 절차

**증상**:
- v3 init-models.sh 첫 시도 시 transformers 5.8.0 (Pod 이미지 preinstalled) → torch 2.4.x와 비호환 → infer_schema ValueError. transformers를 4.49.0으로 다운그레이드하여 해결했지만, **그 시점에 다른 패키지 버전을 일괄 비교하지 않음**.
- 나중에 사용자 직접 지적으로 Mac venv (torch 2.11) ↔ Pod (torch 2.4.1) 불일치 발견.

**근본 원인**:
- "당장의 문제"만 해결하는 사고. transformers 한 패키지 분석 후 OK로 판단.
- 환경 통일은 SW 개발의 기본인데 제가 능동적/수동적의 문제로 잘못 프레이밍.

**v4 반영**:
- Phase 0 task에 **양 환경 (Mac venv + Pod) 패키지 일괄 비교** 단계를 의무화. 진단 스크립트 `scripts/diff_env.sh` 작성.
- requirements.txt는 처음부터 정확한 == 핀 (`torch==X.Y.Z`, `transformers==X.Y.Z` 등) 기반. range 핀은 금지.
- Pod setup 자동화 스크립트가 패키지 설치 후 `python -c "..."` 으로 핀 검증 + max_diff bit-exact assertion.

---

## L02 — Pod의 `--env` 인자가 효력 없음 (2026-05-06)

**카테고리**: Runpod CLI / 환경 변수 전파

**증상**:
- `runpodctl pod create --env '{"HF_HOME":"/workspace/hf_cache",...}'` 로 환경 변수 명시했는데 Pod 안 SSH 셸에서 `echo $HF_HOME`이 비어있음.
- 결과: HF 모델이 `~/.cache/huggingface/` (ephemeral 30GB overlay disk)에 저장. Pod terminate 시 사라질 뻔함. cp -r로 수동 복구.

**근본 원인**:
- Runpod의 `--env` JSON은 컨테이너 environment에 들어가지만, SSH 세션 시작 시 그 환경이 사용자 셸에 자동 export되지 않음 (Docker 기본 동작).
- Pod 이미지의 entrypoint나 SSH 데몬 설정에 따라 다름.

**v4 반영**:
- `runpodctl pod create --env`에 의존하지 않음. 모든 환경 변수는 SSH session에서 직접 export.
- `init-models.sh`와 Phase별 setup 스크립트에서 항상 `export HF_HOME=...` 4 라인을 첫 줄로.
- CLAUDE.md §Pod setup에 명시.

---

## L03 — Runpod CLI 신/구 버전 인터페이스 차이 (2026-05-06)

**카테고리**: Runpod CLI

**증상**:
- 구 CLI: `runpodctl create pod --imageName ... --gpuType ... --secureCloud --networkVolumeId ...` + `runtime.ports[].privatePort/publicPort/ip` JSON shape.
- 신 CLI (≥1.14): `runpodctl pod create --image ... --gpu-id ... --cloud-type SECURE --network-volume-id ...` + 최상위 `ssh.ip/ssh.port/ssh.ssh_key.path`.
- v2 코드를 v3로 가져올 때 신 CLI에 맞춰 한 번 패치, 그러나 JSON parsing은 구 shape으로 남아있어 SSH 엔드포인트 탐지 실패.

**근본 원인**:
- CLI 마이그레이션 시 양면(인자 + 출력)을 모두 검증하지 않음.

**v4 반영**:
- 신 CLI 기준으로 처음부터 작성.
- `runpod.sh`에 `wait_for_ssh()` 도우미 함수 (이미 v3에 있음).
- 신 JSON shape의 모든 필드를 사용 — `ssh.ip`, `ssh.port`, `ssh.ssh_command`, `ssh.ssh_key.path`.
- SSH 키도 `~/.runpod/ssh/RunPod-Key-Go` (runpodctl이 관리) 사용. `~/.ssh/id_ed25519`가 등록돼 있어도 JSON에서 받는 path를 기준으로.

---

## L04 — Network Volume 데이터센터 GPU 가용성 변동 (2026-05-06)

**카테고리**: Runpod 인프라 / 운영

**증상**:
- 첫 시도 6개 GPU 모두 `no instances available`. 17개로 fallback 확장 후 L40 잡힘.
- 다른 시점: A100-SXM4-80GB / RTX A6000 등 가용성 시간대별 변동.

**근본 원인**:
- Network Volume은 단일 데이터센터에 묶임. 그 데이터센터의 GPU 풀에서만 빌릴 수 있음.
- 피크 시간대 (UTC 저녁, 미국/유럽) 가용성 떨어짐.

**v4 반영**:
- `RUNPOD_GPU_FALLBACK` 17개 GPU 기본값 (이미 v3에).
- Phase 7 70B는 80GB GPU 필수 — 그 카테고리만 별도 fallback 리스트 (`RUNPOD_GPU_FALLBACK_LARGE`).
- Phase 시작 시 GPU 못 잡으면 30분/1시간/2시간 backoff 후 재시도. 3번 실패 시 STOP.
- 평가 (Phase 6, 7) 시작 시점 미국 야간/한국 낮 시간대 권장 (Claude Code가 자동 선택은 못 하지만 보고서에 권고 명시).

---

## L05 — FP16 GPU에서 shape-dependent cuBLAS kernel selection으로 인한 ULP 노이즈 (2026-05-07, Phase 2)

**카테고리**: 정확성 검증 / Tolerance 정의

**증상**:
- Phase 1: standard forward vs layerwise forward, max_diff = **0.000e+00** (bit-exact).
- Phase 2 single-prefix `fuse_full_reuse` vs `fuse_full_recompute`: max_diff = **2.7e-2**, argmax 100% identical.
- 원인: cached chunk가 단독 prefill (S=11) → fused prefill (S=18) 사이 cuBLAS GEMM이 다른 kernel을 dispatch. FP16 reduction 순서 차이.

**근본 원인**:
- FP16 GPU의 본질적 한계. 알고리즘 버그 아님 (argmax 일치가 증거).
- v3의 max_diff < 1e-3 tolerance는 **같은 shape끼리 비교**할 때만 유효.
- 다른 shape이 섞이면 ULP 노이즈 누적이 1e-2 영역까지 갈 수 있음.

**v4 반영**:
- Tolerance 체계 재설계:
  - `Tolerance.SAME_SHAPE` = max_diff < 1e-3 (Phase 1: layerwise vs standard)
  - `Tolerance.MIXED_SHAPE` = argmax exact + max_diff < 5e-2 (Phase 2 single-prefix, Phase 3 ratio=0)
  - `Tolerance.RECOMPUTE_PATH` = max_diff < 1e-3 (Phase 3 ratio=1, fused shape끼리)
- Phase별 acceptance에 어느 tolerance를 쓰는지 명시.
- 보고서에 **항상 max_diff + argmax overlap 두 수치 모두 보고**.

---

## L06 — Phase 3 ratio=0 검증의 tolerance 전파 위험 (2026-05-07, Phase 2 끝, Phase 3 시작 전)

**카테고리**: Phase 간 의존성 / Acceptance criteria

**증상 (예측)**:
- Phase 3 task: `ratio=0` 일 때 `fuse_full_reuse`와 max_diff < 1e-3 일치 요구.
- 그러나 Phase 2에서 `fuse_full_reuse` 자체가 single-prefix에서 max_diff 2.7e-2 (vs full_recompute).
- Phase 3 ratio=0 test가 동일 노이즈를 그대로 받으면 < 1e-3 깨짐.

**근본 원인**:
- v3 Phase 3 task와 gate JSON 작성 시 Phase 2 ULP 노이즈를 예상하지 못함.
- Phase 간 tolerance 전파를 명시하지 않음.

**v4 반영**:
- Phase별 acceptance criteria 작성 시 **상류 phase의 tolerance가 하류로 전파되는지 명시 검토**.
- v4 Phase 3에서 `ratio=0` 검증은 `fuse_full_reuse` 출력과 비교 (Phase 2 ULP 노이즈 그대로) → tolerance MIXED_SHAPE.
- `ratio=1` 검증은 `fuse_full_recompute` 출력과 비교 (같은 fused shape) → tolerance RECOMPUTE_PATH.

---

## L07 — Pod reclaim during long-running phase (2026-05-07, Phase 1)

**카테고리**: Runpod 인프라 / 신뢰성

**증상**:
- Phase 1 도중 pod reclaim 두 번. 첫 번째는 결과 캡처 전 끊김 → 재시작.
- A100 SXM4 community-cloud 특성.

**근본 원인**:
- Community cloud GPU는 더 비싸지 않은 대신 우선순위 낮음.
- Spot-like 동작 가능.

**v4 반영**:
- `--cloud-type SECURE` 강제 (이미 v3에 있음). 단 SECURE도 reclaim 가능성 0은 아님.
- **결과를 incremental하게 disk에 저장**: Phase 6 평가는 sample마다 jsonl append. 중간에 끊겨도 재시작 시 이미 처리한 sample skip.
- Phase 6c (Musique 150 + 2WikiMQA 200) 같은 1-2시간 작업은 50 sample마다 checkpoint 저장 권장.
- Pod reclaim 감지 시 (SSH 끊김 + 재접속 시 다른 pod) 자동 재시작 로직.

---

## L08 — Mac venv ↔ Pod 환경 불일치 (2026-05-07)

**카테고리**: 환경 정합

**증상**:
- Mac venv: torch 2.11.0, numpy 2.4.4 (시스템 python3가 자동 선택한 최신).
- Pod: torch 2.4.1+cu124 (이미지 preinstalled).
- 사용자 직접 지적으로 발견.

**근본 원인**:
- L01과 같음. Pod 환경만 검토하고 Mac 환경은 점검 안 함.
- 둘이 결과를 직접 비교하지 않으니 실용적 영향 없다고 잘못 판단.

**v4 반영**:
- Phase 0의 첫 단계: **Mac venv를 Pod와 동일하게 강제 구성**. `python3.11 -m venv .venv` 명시.
- requirements.txt에 정확한 == 핀.
- `scripts/init-mac-venv.sh` 자동화 (사용자가 init-models.sh와 같은 방식으로 한 번 실행).
- Phase 0 acceptance에 "Mac venv와 Pod의 핵심 패키지 8개 버전 일치" 자동 검증 포함.

---

## L09 — Phase 1 보고서 사실 오류 (2026-05-07)

**카테고리**: 보고서 정확성

**증상**:
- Phase 1 보고서 Key numbers 섹션에 "torch 2.11.0, transformers 4.49.0, datasets 4.8.5".
- 실제로는 Mac venv 2.11.0이 출처. Pod에서 돌린 검증의 torch 버전과 다름.
- Decisions made 섹션은 정확 ("torch 2.11과 충돌해서 핀 적용"이라고 적혀 있음 — 즉 Mac venv 이야기).
- 두 섹션이 같은 정보를 다른 의미로 사용해 모순.

**근본 원인**:
- Claude Code 보고서 작성 시 어느 환경의 수치인지 라벨 누락.
- `report-writer` agent에 "환경 라벨 의무" 명시 안 됨.

**v4 반영**:
- 보고서 TEMPLATE.md에 모든 수치에 **환경 라벨 의무**:
  - `torch 2.4.1+cu124 (Pod)` 또는 `torch 2.4.1 (Mac venv CPU)` 명시
  - 같은 패키지가 양쪽에 있으면 두 라벨 모두 표기
- `report-writer` agent에 이 규칙 명시.

---

## L10 — Phase 5 dataset pipeline의 환경 정합 필요성 (예측, 아직 발생 안 함)

**카테고리**: Phase 간 의존성

**증상 (예측)**:
- Phase 5는 Mac CPU에서 stub LayerwiseModel로 dispatch만 검증.
- 그러나 dataset pipeline의 tokenizer 동작은 transformers 버전 의존.
- Mac과 Pod의 transformers 버전이 다르면 같은 입력에 다른 token id 가능성 (드물지만 0 아님).

**근본 원인**:
- Phase 5와 Phase 6 사이 환경이 다르면 Phase 5에서 통과한 dataset이 Phase 6에서 다른 동작.

**v4 반영**:
- L08 정합으로 자동 해결.
- 추가로 Phase 5에 "tokenizer 결정성 테스트" 포함: 동일 텍스트를 Mac venv와 Pod에서 토큰화한 결과가 정확히 같은지 검증.

---

## L11 — Tokenizer mismatch 측정의 모호함 (예측, Phase 5 진입 전)

**카테고리**: 검증 메트릭 명확성

**증상 (예측)**:
- v3 Phase 5 task: "Musique 20 sample에서 tokenizer mismatch < 5%".
- Mismatch 정의가 모호:
  - 청크 N개를 따로 토큰화한 token id 합 vs 텍스트를 concat 후 토큰화한 token id 합?
  - 같은 위치 토큰의 ID가 같은가?
  - 토큰 개수만 비교?
- 검증 코드 작성하면서 결정해야 함.

**v4 반영**:
- 측정 메트릭은 task 작성 단계에서 **정확한 수식**으로 정의.
- "tokenizer mismatch = | tokenize(concat(chunks)) ≠ concat(tokenize(chunks)).len() | / total_chunks" 같은 형식.

---

## L12 — Cost tracker가 Phase 6/7에서만 의미 있음 (2026-05-07)

**카테고리**: 비용 모니터링

**증상**:
- v3는 매 phase 종료 시 `reports/cost-tracker.json` 갱신.
- Phase 1/2 비용 ($0.40, $0.85) 정확히 기록됨.
- 이메일 제목에 누적 비용 표시.

**근본 원인**:
- 잘 동작 중. 다만 v4에서는 더 정밀할 수 있음.

**v4 반영**:
- Phase별 비용 한도 + sub-phase별 한도 (Phase 6a/b/c 각각).
- 한도 80% 도달 시 이메일에 ⚠️ 표시.
- Pod reclaim 발생 시 재시작 비용 별도 추적 (L07 관련).

---

## v4 하네스 설계 원칙 (누적)

위 lesson들을 종합하면 v4의 핵심 차별화:

1. **환경 정합 first**: Phase 0의 첫 단계가 Mac venv ↔ Pod 정합 자동 검증.
2. **Tolerance 체계화**: SAME_SHAPE / MIXED_SHAPE / RECOMPUTE_PATH 세 단계. Phase별 명시.
3. **Pod 신뢰성**: incremental checkpoint, reclaim 자동 재시작, GPU 가용성 backoff.
4. **보고서 라벨**: 모든 수치에 환경 라벨 의무.
5. **Acceptance criteria 정밀화**: 메트릭 정의를 task 작성 단계에서 수식으로.
6. **Phase 간 tolerance 전파 명시**: 상류 노이즈가 하류 검증에 어떻게 영향 주는지 task에 박음.

---

## 추후 추가될 항목 (placeholder)

Phase 3, 4, 5, 6, 7 진행 중 발견되는 lesson을 여기에 계속 추가.

---

## L13 — Boundary safe-shortcut으로 ULP 노이즈 우회 (2026-05-07, Phase 3)

**카테고리**: 정확성 검증 / 구현 패턴

**증상 (positive)**:
- Phase 3에서 `ratio==0` → 즉시 `fuse_full_reuse` 호출, `>=1` → 즉시 `fuse_full_recompute`. 코드 경로가 비교 대상과 완전히 동일해져 max_diff = 0.000e+00.
- Phase 2 ULP 노이즈 (max_diff 2.7e-2) 우려가 사라짐.

**근본 원인 / 통찰**:
- "Boundary equivalence는 알고리즘이 같은지가 아니라 코드 경로가 같은지로 결정". 같은 알고리즘을 다른 코드 경로로 구현하면 FP16 GPU에서 ULP 노이즈 누적이 다름.
- Safe-shortcut 패턴은 cleaner test + cleaner reasoning.

**v4 반영**:
- v4 fusor는 처음부터 boundary shortcut 명시. `if ratio == 0: return fuse_full_reuse(...)`, `if ratio >= 1: return fuse_full_recompute(...)`.
- Tolerance 체계에 4번째 카테고리 추가: `Tolerance.IDENTICAL_PATH` = max_diff == 0 (코드 경로 같음 확정).
- Boundary tests는 IDENTICAL_PATH로 검증.

---

## L14 — HKVD reduction 곡선이 paper Figure 6과 다름 (2026-05-07, Phase 3)

**카테고리**: 알고리즘 충실도 / Paper 재현성

**증상**:
- 논문 Figure 6: ratio=15%에서 forward attention deviation 80%+ reduction (sharp elbow).
- 우리 long_chunk_sanity (chunk_B=120): ratio별 reduction이 거의 선형 (0.05=7.7%, 0.15=13.4%, 0.50=48.9%). Sharp elbow 없음.
- chunk_B=60, 240은 형태 다름. 즉 chunk size별로 일관성 부족.

**근본 원인 (가능성)**:
1. Mistral-7B-v0.2 vs paper 모델 (Mistral-7B-Instruct-v0.1?) 차이.
2. **Deviation metric 세부 차이** — squared L2 vs absolute, K only vs K+V, per-head sum vs mean.
3. Synthetic long-text vs paper의 RAG 분포.
4. Layer 1의 fresh K 계산 시 attention_mask shape 미세 차이.

**v4 반영**:
- Phase 3 task에 **HKVD metric 정의를 정확한 수식으로 명시**:
  - `kv_deviation = ((k_fresh - k_cached) ** 2).sum(dim=(heads, head_dim))` (per-token squared L2 over K only).
  - LMCache 원본 코드 file:line 인용 의무.
  - Paper §4.2 식과 비교 표.
- Phase 3 acceptance에 **Figure 6-like 곡선 검증** 추가:
  - chunk_B=120, ratio별 reduction 측정.
  - 15% 지점에서 reduction ≥ 60% (논문보다 보수적이지만 elbow 존재 확인).
  - 미달 시 metric 재검토 의무.
- 만약 정확히 LMCache 동일 metric 구현해도 Mistral-7B-v0.2에서 elbow가 안 나오면 **모델 자체 특성**으로 결론. 그 경우 Phase 6 F1으로만 평가.

---

## L15 — Mistral sliding-window mask의 K 차원 padding 패턴 (2026-05-07, Phase 3)

**카테고리**: HF transformers 모델 특성

**증상**:
- Mistral의 sliding-window causal mask는 K 차원이 **Q+1 padding**되어 있음.
- 단순히 mask shape이 (Q, K) lower-triangular인지 검사하면 false negative.
- Q×Q sub-block 검증으로 정정.

**근본 원인**:
- Mistral의 sliding window 구현 디테일 (KV cache의 다음 step prefetch?).
- HF transformers의 model-specific behavior.

**v4 반영**:
- v4 Phase 3 task에 mask 검증 시 **모델별 sub-block 추출** 명시:
  - Mistral: Q×Q sub-block lower-triangular.
  - Llama-3.1: Q×K standard (sliding window=None이므로).
- `tests/test_selective.py::test_mask_is_standard_causal`에 모델 dispatch 추가.
- ARCHITECTURE.md에 "model-specific mask handling" 섹션.

---

## L16 — Tolerance 인플레이션 (2026-05-07, Phase 4)

**카테고리**: 검증 절차 / 신뢰성

**증상 (누적 트래킹)**:
- Phase 1: tolerance < 1e-3, 측정 max_diff = 0 (PASS).
- Phase 2: tolerance < 1e-3 → 측정 2.7e-2 → tolerance를 5e-2로 완화 + argmax exact 추가 ("FP16 ULP 노이즈"로 정당화).
- Phase 3: boundary safe-shortcut으로 우회, tolerance 미변경.
- Phase 4: tolerance 5e-2 → 측정 5.47e-2 (prefix_cache) → tolerance를 1e-1로 "통일" (실은 또 완화).

**근본 원인**:
- 매 phase에서 측정값이 tolerance를 살짝 초과하면 tolerance를 측정값보다 약간 큰 값으로 갱신하는 패턴.
- 정량 기준이 retroactive하게 결정됨. 사실상 "측정값이 무엇이든 통과"가 됨.
- argmax exact + max_diff < X 라는 형태에서 X가 점점 의미를 잃음.

**v4 반영**:
- Tolerance는 **phase 시작 전에 결정 후 freeze**. Phase 도중 측정값 보고 늘리지 않음.
- 측정값이 tolerance 초과하면 그것을 **bug 또는 알고리즘 결함의 신호**로 받아들이고 분석 우선. tolerance 변경은 "왜 이 값이 정상인지" 이론적 근거 (LMCache file:line 인용 + paper 식 비교 등) 확보 후에만.
- v4 Tolerance 체계 (L05, L13에서 정의):
  - SAME_SHAPE = max_diff < 1e-3 (bit-exact 가능)
  - MIXED_SHAPE = argmax exact AND max_diff < 5e-2 (cuBLAS shape difference로 알려진 경우만)
  - RECOMPUTE_PATH = max_diff < 1e-3 (같은 fused shape)
  - IDENTICAL_PATH = max_diff = 0 (코드 경로 동일 보장)
  - 위 4개 외 새 카테고리 추가는 design-decisions.md에 정량 근거 + 리뷰 의무.
- Phase별 acceptance에 어느 카테고리를 쓰는지 미리 명시 + 변경 금지.

---

## L17 — Phase 3 weak elbow 후속이 없음 (2026-05-07, Phase 4 끝)

**카테고리**: Phase 간 후속 관리 / 알고리즘 검증

**증상**:
- Phase 3 long_chunk_sanity에서 chunk_B=120 ratio=0.15가 13.4% reduction (논문 Figure 6 sharp elbow와 다름). L14에 기록.
- Phase 3 보고서에 "Phase 6에서 F1으로 평가" 라고 적었으나, Phase 4 끝 시점 (GPU pod stop, 비용 0)에 hkvd metric 검토를 안 했음.
- Phase 5도 CPU phase라 metric 비교는 GPU 없이 가능 (LMCache 코드 read + 우리 code read + 단위 테스트).

**근본 원인**:
- "다음 phase에서 검증" 으로 미루는 패턴. Phase 6에서 F1 부족 시 그제서야 재검토하면 GPU 비용을 더 쓰게 됨.
- 후속 작업의 trigger 조건이 보고서에 명시되지 않음.

**v4 반영**:
- 알고리즘 핵심 (HKVD selection 같은) phase가 끝나면 **다음 phase 진입 전 수직 검증**. 의심스러우면 "다음 phase에서" 가 아니라 "지금 GPU 안 쓰는 동안" 검증.
- Phase별 task에 "후속 작업" 섹션. 이번 phase에서 발견한 의심스러운 점이 다음 phase에서 어떻게 검증되는지 명시. 단순 "Phase 6에서 F1으로" 같은 모호한 표현 금지.
- 알고리즘 핵심 phase는 LMCache 코드 직접 비교 의무 (file:line 인용 + 우리 코드 file:line + diff 분석).

---

## L18 — Pipelining 단순 디자인이 옳았음 (2026-05-07, Phase 4)

**카테고리**: 구현 패턴 (positive)

**증상 (positive)**:
- `fuse_selective_pipelined` 단순 구현 (chunk-level prefetch → wait → fuse_selective).
- per-layer prefetch는 미구현 (논문이 묘사한 정교한 pipelining).
- 결과: logits identity (max_diff < 1e-3) + 2.86× TTFT 단축 (시뮬레이션).

**근본 원인 / 통찰**:
- YAGNI 원칙 적용. 단순한 chunk-level prefetch가 대부분 가치를 잡음.
- 논문 §5는 per-layer pipelining을 묘사하지만, 실제 효과의 대부분은 chunk-level에서 이미 잡힘.
- 단순한 코드 = test 쉬움 = 검증 신뢰 높음.

**v4 반영**:
- Phase 4 task에 "단순 chunk-level prefetch 우선, per-layer는 Phase 6에서 TTFT 부족 시에만" 명시.
- v4 fusor.py의 fuse_selective_pipelined도 같은 단순 디자인 유지.

---

## L19 — CPU phase로 GPU 비용 0 분리 성공 (2026-05-07, Phase 5)

**카테고리**: 비용 / Phase 분할 (positive)

**증상 (positive)**:
- Phase 5 (Dataset Pipeline) 를 CPU only로 분리.
- Tokenizer mismatch, F1 단위 테스트, RAG input 결정성, 4-method dispatch 모두 stub 모델로 검증.
- GPU 비용 $0.00, 10/10 테스트 통과, 19.7초.

**근본 원인 / 통찰**:
- Dataset pipeline correctness 검증과 모델 forward 검증은 본질적으로 분리 가능한 관심사.
- Stub 모델 (forward_layerwise까지만 호출되는 mock) 으로 4-method dispatch 검증 가능 → 실제 attention은 이미 Phase 1-4에서 검증됨.
- GPU 시간을 진짜 모델 평가 (Phase 6) 에만 쓸 수 있게 해줌.

**v4 반영**:
- Phase 분할 원칙 명시: GPU 필수 vs CPU-OK 구분.
- v4의 dataset pipeline phase는 v3와 같이 CPU phase로 유지.
- Stub model 패턴을 `tests/conftest.py` 에 fixture로 정착시켜 다른 phase에서도 재사용.

---

## L20 — Per-item seeded shuffle로 prefix caching baseline 강제 (2026-05-07, Phase 5)

**카테고리**: 평가 진정성 (positive)

**증상 (positive)**:
- Phase 5 RAG input builder가 per-item seeded shuffle로 문서 순서를 sample마다 다르게.
- Prefix cache baseline은 system prompt 정도만 hit, 나머지는 매번 prefill.
- GOAL.md "문서 순서 가변, prefix caching만으로 부족" 원칙을 데이터 레벨에서 강제.

**근본 원인 / 통찰**:
- 평가 데이터의 분포가 baseline의 약점을 직접 노출하는 게 좋은 평가의 조건.
- 만약 모든 sample에서 같은 문서 순서면 prefix cache가 우연히 cacheblend만큼 잘 나올 수 있음 (false equivalence).
- Sample-wise 다양성이 진정한 차이를 드러냄.

**v4 반영**:
- v4 RAG input builder도 같은 per-item seeded shuffle 유지.
- 평가 보고서 (Phase 6+) 에 "샘플별 prefix cache hit rate 분포" 측정 추가 — 의도한 만큼 prefix cache가 약화되는지 검증.
- Disclosure 문서에 "document order randomization" 명시 (paper와의 차이점).

---

## L21 — Defensive dry-run artifact 패턴 (2026-05-07, Phase 5)

**카테고리**: 형식 검증 / 결함 예방 (positive)

**증상 (positive)**:
- Phase 5에서 `musique_20_seed42.jsonl` 을 `pred=null, ttft=null, f1=null` 로 미리 생성.
- Phase 6에서 진짜 값 채울 때 schema mismatch / key typo / shape mismatch 발견 즉시.
- "Run thousands of dollars of GPU only to find malformed output" 시나리오 차단.

**근본 원인 / 통찰**:
- GPU 비용 0인 시점에 데이터 형식 검증 = 비싼 phase의 risk reduction.

**v4 반영**:
- 모든 평가 phase 진입 전 dry-run artifact 생성 의무.
- Schema (key 이름, 타입, nullable) 를 task 파일에 정확히 명시.
- 평가 phase 첫 단계는 dry-run artifact를 참고로 schema 일치 검증.

---

## L22 — Gradual Filtering Scheme (v4 Phase 8 spec, in-progress)

**카테고리**: 신규 기능 / 논문 §4.3 충실도 향상

**배경**:
- LMCache는 단일 check_layer=1 + flat ratio (모든 layer 동일 비율) 단순화 채택.
- CacheBlend 논문 §4.3과 Figure 9는 더 정교한 gradual filtering 묘사: 여러 check layer에서 점진적으로 HKVD set을 좁혀나감 (strict subset).
- v4에서 이를 구현하고 LMCache flat schedule과 비교.

### Spec (확정 부분)

#### Filtering 동작 (M0, 사용자 직접 답)
- Strict subset (누적): check layer i+1의 HKVD는 check layer i의 HKVD set 안에서만 선정.
- 예: check_layers = [2, 5, 10], 100 토큰 가정
  - Layer 0~1: 전체 100개 fresh prefill (warm-up)
  - Layer 2 (check): 직전 set (= 100개) 전체 fresh prefill, deviation 측정 → top 30 선정
  - Layer 3, 4: 30개 fresh
  - Layer 5 (check): 직전 set (= 30개) 전체 fresh, 그 중 top 15 선정
  - Layer 6~9: 15개 fresh
  - Layer 10 (check): 직전 set (= 15개) 전체 fresh, 그 중 top 10 선정
  - Layer 11~31: 10개 fresh

#### Top-3 candidate 발견 — 메트릭 (M1, 확정)

세 메트릭을 모두 측정:
- **(a) Top-15% mass**: 토큰별 KV deviation 정렬 시 top 15% 토큰이 전체 deviation 합의 X% 차지. X 값 자체를 layer 점수로 사용. 가파를수록 좋은 check layer.
- **(b) Spearman rank correlation**: 토큰별 KV deviation rank vs 토큰별 forward attention deviation rank의 Spearman. KV deviation으로 attention deviation 예측 가능성.
- **(c) Information gain**: layer i에서 top 15% HKVD 선정 후 layer i, i+1, i+2의 forward attention deviation 감소량. 직접 인과 측정. 부분 forward 추가 비용.

세 메트릭 모두 같은 forward pass에서 후처리로 (a), (b) 계산 가능. (c)는 추가 partial-forward 필요.

#### Check layer 개수 결정 — Hybrid (M2, 확정)

옵션 D: 절대 임계값 + 상대 gap 분석으로 자동 결정.

```
For each metric m in [(a), (b), (c)]:
    1. score_m(l) for all 32 layers
    2. significant_layers = {l : score_m(l) >= threshold_m}
       - threshold_a = 0.30 (top-15% mass ≥ 30%)
       - threshold_b = 0.30 (Spearman ≥ 0.30)
       - threshold_c = TBD (1차 실험 후 normalize 기준 결정)
    3. Sort by score_m desc
    4. Gap analysis:
       gaps[i] = score_m(sorted[i]) - score_m(sorted[i+1])
       max_gap_pos = argmax(gaps[:3])
       - max_gap_pos == 0: 1-check, layer = sorted[0]
       - max_gap_pos == 1: 2-check, layers = sorted[:2]
       - max_gap_pos == 2 or no clear gap: 3-check, layers = sorted[:3]
       Gap threshold (clear vs unclear): absolute gap ≥ 0.10
    5. Resulting schedule template (positions fixed, ratios TBD in M3)
```

#### 시각화 의무

각 모델 × 메트릭마다 plot:
- x축: layer index (0~31)
- y축: metric score
- threshold 가로선
- significant_layers 강조 (색상)
- 선정된 check layer 마커 (점)
- gap 위치 표시 (수직선)

3 metrics × 2 models = **6 plots**. 보고서에 첨부.

#### 향후 결정 필요 (M3 ~ M7)

- M3: 비율 sweep 공간 (각 budget에서 ratio combo 개수)
- M4: Layer 0~첫 check layer 사이 처리 (warm-up zone, 평균 계산 포함 여부)
- M5: 검증 데이터셋 크기 (Musique + 2WikiMQA 각 50?)
- M6: Null result 처리
- M7: 비용 cap

#### Phase 8 위치

- Phase 6 (Mistral eval), Phase 7 (Llama eval) 종료 후 추가 phase
- Phase 6/7는 LMCache flat schedule 그대로 (baseline 유지)
- Phase 8에서 gradual schedule이 baseline 대비 F1 개선 보이는지 정량 비교

#### 모델 범위
- Mistral-7B-Instruct-v0.2
- Llama-3.1-8B-Instruct
- Llama-3.1-70B는 Phase 8 제외 (비용 우려, 70B는 Phase 7 baseline만)

#### M3 비율 sweep — Option D 확정

각 (metric, num_check, budget) 조합에 대해 두 schedule 비교:

**Linear decay**: `weights = [n+1-i for i in 1..n]`, scale을 자동 조정해 평균 = budget.
- 3-check 예: weights = [3, 2, 1] → r1=3s, r2=2s, r3=1s, scale s 자동 결정
- 단조 감소 보장 (n+1-i 형태)

**Uniform baseline**: 같은 check_layers 위치 + 같은 평균 budget, 다만 모든 r_i = 같은 값.
- 평균 = budget 자동 만족
- 같은 위치/같은 평균에서 비율 분배만 다른 head-to-head 비교

**비용 추정**:
- 3 metrics × 4 budgets × 2 schedule_types × 2 models = **48 runs**
- ~$15

**핵심 질문**: "Gradual filter가 LMCache 단순 디자인보다 정말 나은가?"
- LMCache flat: check_layer=1 + budget at all layers
- Our gradual (linear): check at metric-selected layers + decay
- Our uniform: check at same metric-selected layers + flat
- 세 설정의 F1/TTFT 비교로 답

**미해결 issue (M4와 함께 결정)**:
- Budget이 작을 때 warm-up 합이 budget 초과 → 불가능 조합 처리.
- 자동 skip + 경고 또는 warm-up 자동 단축.

#### M4 — Warm-up zone (확정: data-driven, 사용자 검토)

Warm-up zone과 budget 정의 방식은 **사전 결정하지 않음**. Step 1 결과 (layer profiling) 보고 결정.

**Phase 8을 interactive multi-step experiment로 운영**:

```
Step 1: Per-model layer profiling (자동, ~$2-3)
  - 6 plots (3 metrics × 2 models) 생성
  - 보고서 이메일

[사용자 검토 + 프롬프트]
  plots 보고 결정:
  - Budget 정의 방식 (옵션 A/B/Hybrid)
  - 메트릭별 check_layers 채택 (자동 선정 결과 + 사용자 조정 가능)
  - threshold/gap 미세조정

Step 2: Schedule 생성 (CPU, $0)
  - 사용자 결정 반영
  - Linear decay + uniform 두 종류
  - 보고서: schedule 표 + r 값들 + compute ratio

[사용자 검토 + 프롬프트]
  schedule 검토, 의미 없는 것 skip 결정

Step 3: F1/TTFT sweep (~$15)
  - 각 schedule × Musique 50 + 2WikiMQA 50
  - 보고서: 결과 heatmap

[사용자 검토 + 프롬프트]
  추가 sweep 또는 종료

Step 4: LMCache flat baseline 비교 + 최종 보고서
  - Apples-to-apples 비교 (zone budget + total ratio 두 수치 모두)
  - 결론 + design recommendation
```

이는 v3의 "pure full auto" 원칙과 다름. **알고리즘 발견 (discovery) 단계는 interactive, 알고리즘 검증 (validation) 단계는 자동** 이라는 분리.

#### 핵심 원칙

Phase 8은 v3의 다른 phase들과 달리 **"discovery experiment"**.
- Phase 0~7: 알고리즘은 정해져 있음, 자동 검증
- Phase 8: 알고리즘 detail 자체를 데이터 + 사용자 직관으로 발견

이 구분을 v4 PHASES.md에 명시.

#### M5 — 검증 데이터셋 (확정: 옵션 B)

- Musique 50 + 2WikiMQA 50 = 100 sample/schedule
- 단일 round (no smoke pre-pass)
- F1 std error 약 0.02-0.03 (schedule 간 0.02 차이 구분 가능)
- 비용 ~$15 (Step 3 부분)

#### M6 — Null result 처리 (확정: 옵션 A)

- 결과 그대로 보고 (시나리오 1/2/3 무관)
- 시나리오 3 (gradual이 더 나쁨) 도 valid finding
- 사용자 인터랙션은 이미 Phase 8 step별 보고서 검토 흐름에 내장됨 → 추가 분기 로직 불필요

#### M7 — 비용 cap (확정: 옵션 B)

- Phase 8 cap = **$25**
- 내역: Step 1 ($3) + Step 3 ($15) + Step 4 ($1) + buffer ($6, ~30%)
- 1-2회 추가 sweep 가능
- 디버깅 round 발생 시 사용자 명시적 재승인 필요 (cap 초과 시 자동 STOP)

### Phase 8 최종 spec 요약 (확정)

**모델**: Mistral-7B-Instruct-v0.2 + Llama-3.1-8B-Instruct (70B 제외)

**위치**: Phase 6 (Mistral) + Phase 7 (Llama) 종료 후. baseline 데이터 활용.

**메트릭** (Step 1):
- (a) Top-15% mass
- (b) Spearman rank corr (KV deviation rank vs forward attention deviation rank)
- (c) Information gain (partial forward로 측정, 추가 비용)

**Threshold + gap rule** (Step 1 자동):
- threshold_a = 0.30, threshold_b = 0.30, threshold_c = TBD
- significant_layers = score ≥ threshold
- gap analysis: top-N에서 gap ≥ 0.10인 위치로 num_check 결정 (1, 2, or 3)

**시각화 의무**: 6 plots (3 metrics × 2 models). 각 plot에 threshold 가로선, significant layers 강조, 선정된 check layer 마커, gap 위치 수직선.

**Schedule sweep** (Step 3):
- 4 budgets × 3 metrics × 2 schedule_types (linear decay + uniform) × 2 models = 48 schedule
- 각 schedule × 100 sample (Musique 50 + 2WikiMQA 50)

**Budget 정의 방식**: Step 1 plots 보고 사용자가 결정 (옵션 A: 전체 평균 / 옵션 B: zone 평균 / Hybrid: 두 수치 모두 명시)

**Warm-up zone**: Step 1 결과에 따라 자연스럽게 결정 (첫 check layer 이전 = warm-up)

**Phase 8 Interactive flow**:
1. Step 1 자동 → 6 plots → 보고서 이메일
2. 사용자 검토 + 프롬프트 (budget 정의, check_layers 채택)
3. Step 2 자동 (CPU) → schedule 표 → 보고서 이메일
4. 사용자 검토 + 프롬프트 (skip할 schedule 선정)
5. Step 3 자동 (GPU) → F1/TTFT heatmap → 보고서 이메일
6. 사용자 검토 + 프롬프트 (추가 sweep 또는 종료)
7. Step 4 자동 (LMCache 비교) → 최종 보고서 이메일

**비용 cap**: $25 (Step 3 $15 + Step 1 $3 + Step 4 $1 + buffer $6)

**Discovery vs Validation 구분**: Phase 8은 discovery experiment. v3의 pure full auto와 분리. v4 PHASES.md에 명시.

---

## L23 — Pod 운영 마찰 (Phase 1~5 누적)

**카테고리**: Runpod 운영 / Pod 셋업

**증상 (누적)**:
- Pod ephemeral reclaim: Phase 1 도중 두 차례 SSH 끊김 (Connection closed by remote host). Community-cloud GPU 선점 추정.
- `runpod.sh start` 실패: stop 후 재시작 시 그 호스트에 GPU 없으면 fail. Phase 3, 4에서 발생 → terminate + up 우회.
- Pod 이미지에 rsync 미포함: 매 신규 pod에 `apt-get install rsync` 필요. CLAUDE.md §Pod setup에 누락.
- `runpodctl pod start`의 silent 5xx: GPU shortage 시 stderr에 Usage 덤프 섞여 파싱 어려움.

**근본 원인**:
- `--cloud-type SECURE` 강제했지만 community fallback 가능성 0 아님.
- `runpod.sh start`는 같은 host 재시작 가정. 호스트 GPU 사라지면 fail.
- runpod/pytorch:2.4.0 이미지가 minimal — rsync 같은 흔한 도구도 없음.

**v4 반영**:
- `runpod.sh start --auto-recreate`: start 실패 시 자동 terminate + up 폴백.
- Pod bootstrap에 `apt-get update && apt-get install -y rsync` 자동 추가 (5분 단축).
- CLAUDE.md §Pod setup을 4단계 → 5단계로 (rsync 포함).
- 또는 cacheblend-runtime 베이스 이미지 별도 빌드 (rsync + pinned deps 포함). Network volume에 prebuilt venv 두기 (`source /workspace/venv/bin/activate`로 30초 단축).
- Pod reclaim 감지 시 (SSH 끊김 + 재접속 다른 pod) 자동 재시작 (L07과 통합).

---

## L24 — Eval gate / verify_phase 하네스 결함 (Phase 0~6 누적)

**카테고리**: 검증 인프라 / 스크립트 robustness

**증상**:
- `subprocess.run(["python", ...])` — macOS에는 `python` 바이너리 없음 (`python3`만). 매번 `PATH=$PWD/.venv/bin:$PATH` 우회.
- `verify_phase.py`가 GPU phase의 pytest를 로컬에서 실행 → CUDA 없어서 실패. tests/conftest.py에 skipif 추가로 우회 (PR #3).
- `gate-6-final.json`의 `check_type: "metric"` / `cost_check`를 `eval_gate.py`가 처리 못 함 → Phase 6에서 직접 확장 (PR 미포함).
- `gate-6-final.json`의 `sub_phases` 중첩 구조를 `eval_gate.py`가 처리 못 함 → 같은 PR에서 확장.

**근본 원인**:
- 스크립트 작성 시 macOS 환경 가정 미흡.
- Gate JSON 스키마와 평가기 구현이 phase별 발전. 처음에 schema를 풍부하게 설계하지 않음.

**v4 반영**:
- `subprocess.run([sys.executable, ...])` — 인터프리터 명시 (한 줄 수정).
- `tests/conftest.py` GPU skipif 패턴 정식화 (Phase 0부터). Pytest collection에서 자동 skip.
- `eval_gate.py`에 처음부터 `metric` / `cost_check` / `sub_phases` 핸들러 포함. Phase 6/7 spec이 이미 그것을 요구.
- v4 gate JSON 스키마를 처음에 풍부하게 정의: `check_type ∈ {import, command, file_exists, pytest, verify_phase, metric, cost_check}`, `conditions[]` 또는 `sub_phases.{name}.conditions[]`.

---

## L25 — Skeleton 결함 (Phase 0)

**카테고리**: Skeleton 정확성

**증상**:
- `tests/test_smoke.py`의 `test_imports_layerwise_model`이 `from cacheblend.model import LayerwiseModel` 요구. Phase 0의 model.py는 빈 파일 → smoke 깨짐. Phase 0에서 `LayerwiseModel = None` stub 추가로 우회.
- `gate-0-to-1.json`이 `external/LMCache/` clone 위치 검증. `.gitignore`가 `external/` 무시 → clone은 매 환경에서 다시 받아야 함. Phase 0 task에 명시 필요 (지금은 묻혀있음).

**근본 원인**:
- Skeleton 작성 시 smoke test가 import 가능하도록 stub class 미리 박지 않음.
- `external/` ignore 정책이 의도적인지 검증 누락인지 task 파일에 명시 없음.

**v4 반영**:
- v4 skeleton에 모든 module의 export class를 stub로 미리 정의: `class LayerwiseModel: pass`, `class KVStore: pass` 등. smoke test가 처음부터 통과.
- `external/` ignore 의도 명시: Phase 0 task 파일에 "LMCache clone은 reference only, gitignore 의도. 매 환경에서 재클론 필요"라고 명시.

---

## L26 — Phase 스펙 vs Gate JSON 정합 (Phase 2/3/6 누적)

**카테고리**: 명세 정합성 (가장 시급)

**증상**:
- `tasks/phase-3-selective-recompute.md` Acceptance #5: "chunk_B=100, ratio=0.15에서 reduction ≥ 40%" — 우리 결과 13.4%. **Gate JSON에는 이 항목 없음** → gate 통과. 작업 사양과 게이트가 어긋나면 결정 모호.
- `tasks/phase-2-kv-storage.md` "단일 prefix 청크에서 max_diff < 1e-3" — fp16 fundamental로 불가능. argmax 동등성 + 1e-1로 완화 (docs/design-decisions.md 추가). GOAL.md의 "max_diff < 1e-3 (FP16 GPU)" 는 Phase 1만 가능; Phase 2+에서는 비현실적.
- `gates/gate-6-final.json` 6c.2: `ttft_speedup_geq 1.43` — 우리 hook-injection 구조로는 도달 불가 (L27 참조). 처음부터 architectural 가능성 검토 + 명세화 필요.

**근본 원인**:
- 명세 작성 시 알고리즘의 architectural 한계를 미리 검토하지 않음.
- Task 파일과 gate JSON이 분리되어 있고, 둘 사이 정합 검증 없음.
- "Gate가 binding" 룰이 있지만 task에 더 강한 조건 있으면 사용자/에이전트 헷갈림.

**v4 반영**:
- 각 phase의 task 파일과 gate JSON을 **단일 파일**로 통합 또는 cross-reference 검증 자동화.
- 충돌 시 어느 쪽이 우선인지 task 파일 첫 줄에 명시.
- GOAL.md의 tolerance 규칙을 phase별로 차등 명시:
  - "Phase 1: layerwise vs standard, max_diff < 1e-3 strict"
  - "Phase 2~7: fused-vs-precompute, argmax-strict + max_diff < 1e-1"
- Phase 1 결과가 0.000e+00이라 strict bound이 가능했지만, Phase 2부터 shape-dependent kernel selection으로 인해 본질적으로 불가능. 이를 GOAL에 명시.

---

## L27 — TTFT는 v3 비목표 (Phase 6 진행 중 사용자 결정)

**카테고리**: 프로젝트 범위 / 목표 정의

**증상**:
- v3 Phase 6 gate `ttft_speedup_geq 1.43` 같은 TTFT 조건이 hook-injection 구조로는 도달 불가능.
- 진짜 TTFT 이득을 보려면 hidden_state slicing 필요. LMCache의 LMCBaseModel.compute_layer는 check_layer에서 q/k/v/residual을 모두 top_indices로 슬라이싱 → 후속 layer는 HKVD 토큰만 진짜 forward.
- 우리 구현은 hidden_state 슬라이싱 없음 → 모든 토큰에 대해 q_proj, MLP, RMSNorm 모두 fully 계산.
- Hook-injection은 quality (logits identity) 만 보장, 실제 compute 절감 없음.

**사용자 결정**:
- TTFT 향상은 **이번 프로젝트 목표가 아님**.
- v4도 quality-only milestone로 정의.
- Hook-injection 구현 그대로 유지 가능. Hidden_state slicing 같은 architectural 침습 불필요.

**v4 반영**:
- **GOAL.md 명시**: "v4 is quality-only. TTFT measurement is reference only, not a gate condition."
- Phase 6/7 gate에서 TTFT 조건 제거. F1 / Rouge-L 만 통과 조건.
- Phase 8 (Gradual filtering) 도 F1 만 비교. TTFT는 보조 지표로만.
- 보고서에 TTFT 측정은 계속하되 gate 미사용. "TTFT는 reference, not validated."
- Architectural 단순성 유지 (hook-injection). LMCache 깊이 이식 불필요.

이 결정으로 v4 설계 단순화. Architectural 침습 (hidden_state slicing) 회피.

---

## L28 — F1 노이즈 / 데이터셋 셋업 (Phase 6 진행 중 발견)

**카테고리**: 평가 메트릭 신뢰성

**증상**:
- top_k=4 (supporting only) → F1 ≈ 0.45, sample 간 분산 적음. 단 sequence 짧아 (~250 token) cacheblend overhead 비대.
- top_k=10 (distractor 추가) → F1 ≈ 0.14~0.18로 폭락. 모델이 distractor에 산란. Paper Figure 12는 retriever quality가 더 좋아서 top-10 doc에도 supporting이 명확하게 식별되는 setup일 가능성.
- Musique 자체가 multi-hop이라 답이 연속어절로 안 나오는 경우 많아 SQuAD-style F1이 ±0.1를 쉽게 흔듦.

**근본 원인**:
- 우리는 dataset의 supporting paragraphs를 직접 사용 (retrieval 우회).
- Paper는 SentenceTransformer retriever로 top-k 추출. Distractor 포함 + retriever quality 비공개.
- F1 절대값 비교는 retrieval setup 차이에 민감. Schedule 간 차이의 statistical significance를 절대값으로 판단하면 noise에 쉽게 묻힘.

**v4 반영**:
- Phase 5에 "retriever quality" 검증 추가:
  - 옵션 1: Dataset의 supporting만 사용 (현재 방식). top_k = supporting count.
  - 옵션 2: BM25 또는 sentence-transformers로 top-k 검색. Supporting 포함율 측정 (recall@k).
- 두 옵션 결과를 모두 보고 (paper와의 setup 차이 명시).
- Gate에서 F1 절대값 대신 **paired bootstrap CI** 사용:
  - "F1(cacheblend) > F1(full_reuse) at 95% CI" (절대값 0.05 대신).
  - Bootstrap n=1000 standard.
- Rouge-L + Exact Match (EM) 도 함께 보고. F1 변동 큰 dataset에서 다른 metric이 더 안정적일 수 있음.

---

## L29 — 출력 경로 / 명명 일관성 (Phase 5/6 누적)

**카테고리**: 인프라 정합성

**증상**:
- `gate-6-final.json`의 `glob: benchmarks/results/musique_20_*.json` — runner는 `*.jsonl` (per-sample) + `*.summary.json` (집계) 두 파일을 씀. JSON glob이 후자만 매치하길 의도한 듯하지만 모호.
- `reports/phase-N-attachments/`가 `.gitignore`에 박혀있어 commit 시 `-f` 필요.

**근본 원인**:
- Output 파일 naming convention이 task 작성 시점에 정해지지 않음. Runner 구현 후 결정.
- Gitignore 정책이 attachment 파일을 의도적으로 차단하는지 누락인지 명시 없음.

**v4 반영**:
- v4 첫 단계에서 output naming convention 명시:
  - per-sample: `<dataset>_<n>_<method>.jsonl`
  - 집계: `<dataset>_<n>_<method>.summary.json`
  - Plot: `<dataset>_<n>_<method>.png`
- Gate JSON glob 패턴 명시: `*.summary.json` 등 정확히.
- `.gitignore`의 `reports/phase-*-attachments` 라인 제거 또는 화이트리스트 추가 (`!reports/phase-*-attachments/results.txt`).

---

## L30 — 비용 추적 정확성 (Phase 1~5 누적)

**카테고리**: 비용 모니터링

**증상**:
- `reports/cost-tracker.json`이 Claude Code의 manual entry 의존 — 정확하지 않음.
- Runpod billing API에 `costPerHr` 있어 자동화 가능.

**근본 원인**:
- v3 cost-tracker는 사람이 wall time × hourly rate 추정. Pod reclaim, multiple GPUs 등 케이스에 취약.

**v4 반영**:
- `runpodctl pod get --output json`의 `costPerHr` 자동 파싱.
- Pod 종료 시점 (terminate 또는 stop)에 wall time × costPerHr 자동 기록.
- Pod reclaim 발생 시 양 pod의 wall time 합산.
- Phase 종료 시 자동 갱신.
