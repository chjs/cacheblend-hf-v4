# v5 Lessons Learned — Cumulative Log

> v4 진행 중 누적되는 문제·결정·교훈. v5 하네스 설계 시 빠짐없이 반영.
> 모든 phase 보고서는 "v5-lessons 섹션" 의무 (없으면 "없음" 명시).
> 자동 추가: `python scripts/add_lesson.py --phase N --category "..." --title "..." --symptom "..." --root-cause "..." --v5-fix "..."`
> 철회: `python scripts/retract_lesson.py L## --reason "..."`

## 이전 iteration 요약 (v3 → v4)

`docs/notes/v4-lessons.md`에 30개 lesson 누적되어 v4 설계에 반영됨. 주요 카테고리:

- **환경 정합** (L01, L08, L23): Mac/Pod 정합, package locking, rsync 자동 설치
- **Tolerance 인플레이션** (L05, L13, L16): 4 카테고리 freeze
- **Pod 운영** (L07, L23): reclaim, GPU 가용성, bootstrap, --auto-recreate
- **명세 정합** (L26): task vs gate JSON 단일 파일 통합
- **TTFT 비목표** (L27): quality-only milestone 결정
- **Algorithm correctness** (L14, L17): HKVD metric LMCache 비교, weak elbow 후속
- **Phase 8 신규** (L22): Gradual filtering discovery experiment
- **Skeleton 결함** (L25): stub class 미리 박기
- **Eval gate 결함** (L24): subprocess.run([sys.executable]), metric/cost_check/sub_phases 처음부터
- **F1 노이즈** (L28): paired bootstrap CI
- **출력 명명** (L29): naming convention 사전 정의
- **Cost tracking** (L30): Runpod billing API 자동화

상세는 `docs/notes/v4-lessons.md` (860 라인, 30개 lesson) 참조.

---

## 이번 iteration (v4) 발견 사항

(여기서부터 v4 진행 중 추가됨. 비어있으면 아직 발견 사항 없음.)

<!-- LESSONS_START -->

## L31 — Runner stub instantiation이 harness ABC와 충돌 (2026-05-07, Phase 0)

**카테고리**: Skeleton 결함

**증상**:
tests/test_smoke.py::test_imports_runners 가 AttributeError: 'NoneType' object has no attribute 'parameters' 로 실패. _RunnerBase가 _HARNESS_AVAILABLE 단독 조건으로 super().__init__()을 호출했지만, harness ABC의 __init__이 next(model.parameters()).device 를 부르므로 model=None smoke test에서 즉시 crash.

**근본 원인**:
Phase 0 smoke test의 의도는 'model 없이 stub 인스턴스화 가능'이지만, _RunnerBase가 harness 존재 여부만으로 super() 호출 결정. mydata harness가 v3 이후 추가되며 model 필수 가정이 strengthened.

**v5 반영**:
_RunnerBase.__init__ 분기 조건을 'if _HARNESS_AVAILABLE and model is not None' 로 변경. v5에서는 처음부터 'smoke 경로 / 실 평가 경로' 두 분기를 명시적으로 디자인. 또는 harness ABC 자체에서 device를 lazy 처리.


## L32 — runpodctl flag deprecation: --container-disk-size → --container-disk-in-gb (2026-05-07, Phase 1)

**카테고리**: Pod 운영

**증상**:
scripts/runpod.sh 의 17 GPU fallback 모두 즉시 실패. 에러는 swallowed 되어 'Failed for X' 만 출력. 직접 runpodctl pod create 실행 시 'unknown flag: --container-disk-size' 발견.

**근본 원인**:
runpodctl 2.1.9 (2026-05 시점) 에서 flag 이름이 --container-disk-in-gb 로 변경. v3 시기 (≤1.14) 의 --container-disk-size 가 더이상 유효하지 않음.

**v5 반영**:
scripts/runpod.sh 의 create_pod 함수에서 --container-disk-size 30 → --container-disk-in-gb 30 으로 수정. v5 에선 (a) runpodctl --version 사전 체크, (b) create 실패 시 stderr 표시 (현재 swallowed), (c) flag 호환성 매트릭스 문서화 권장.


## L33 — Network volume DC 미일치 시 silent 미부착 (2026-05-07, Phase 1)

**카테고리**: Pod 운영

**증상**:
runpodctl pod create 시 --network-volume-id 만 주면 Pod 가 임의 DC 에 배치되고 볼륨이 미부착된다 (volumeInGb=0). pod create 자체는 성공이라 swallowed; SSH ready 후 hf_cache 가 비어있어야 발견. 첫 시도 ex92ca3efkd45t 를 즉시 remove (비용 손실 0).

**근본 원인**:
runpodctl 2.1.9 의 pod create 가 --network-volume-id 만으로는 DC 자동 매칭 안 함. --data-center-ids 인자로 볼륨 DC 와 명시 일치 필요.

**v5 반영**:
scripts/runpod.sh 의 create_pod 함수가 'runpodctl network-volume list' 로 볼륨 DC 자동 조회 → --data-center-ids 자동 주입. v5 에선 (a) volume 미부착 explicit assertion (volumeInGb>0), (b) DC 매칭 실패 시 fail-fast.


## L34 — RunPod US-KS-2 GPU 가용성 부족 시 silent SSH 미실행 (2026-05-07, Phase 1)

**카테고리**: Pod 운영

**증상**:
Phase 1 Pod 부팅 시도 3회 모두 실패. 1) --container-disk-size 옛 flag 로 17 GPU 즉시 fail (L32). 2) DC 자동 매칭 누락으로 wrong DC 배치, volumeInGb=0 (L33). 3) DC 정상 핀했지만 A100-SXM4-80GB 'create 성공' 후 10분간 SSH 미실행, uptimeSeconds=0. 직접 시도하면 actual error 'There are no longer any instances available with the requested specifications'. runpod.sh 의 stderr swallow 가 가용성 이슈를 첫 번째 시도에서 silent fail 로 만든다. 사용자 지시로 vast.ai 로 pivot.

**근본 원인**:
RunPod 의 pod create API 가 'create accept + 실제 가용 GPU 없음' 케이스를 silent 로 RUNNING 상태로 마크. SSH endpoint 는 영원히 안 옴. runpod.sh 가 stderr 를 || true 로 swallow 해 진짜 에러 메시지가 안 보임.

**v5 반영**:
v5 에선 (a) 다중 클라우드 fallback 디자인 (RunPod fail → vast.ai 자동), (b) pod create 의 stderr 보존 (가용성 에러 즉시 표시), (c) SSH 미실행 timeout 후 자동 destroy + 다음 GPU/DC/cloud 시도, (d) 'create 성공' 의 의미를 'SSH ready' 로 strengthen (poll until uptimeSeconds>0).


## L35 — 사용자 할당 instance 는 reboot 금지 (사용자 정책) (2026-05-07, Phase 1)

**카테고리**: Pod 운영 / 사용자 협업

**증상**:
vast.ai instance 36296030 부팅 시 호스트 port collision (port 32319 already in use). 자동 reboot/stop+start 시도. 사용자가 '한번 할당받으면 리부팅하지마' 명시적 지시.

**근본 원인**:
사용자가 instance 라이프사이클 (선택, 시작, 재시작, 종료) 을 직접 통제하길 원함. 자동화 측 reboot 은 사용자 워크플로 (콘솔 모니터링, 비용 추적 등) 와 충돌.

**v5 반영**:
v5 에선 instance 라이프사이클 정책을 명시적 분기로: 자동화가 생성한 instance 는 자동 reboot/destroy 가능, 사용자가 할당해준 instance 는 read-only (SSH 명령 실행만, 호스트 상태 변경 금지). 부팅 실패 등 hard problem 발견 시 즉시 사용자에게 보고 + 결정 대기. 보고서에 instance 출처 (auto/user) 기록 의무.


## L36 — vast.ai pytorch base image 의 /venv/main 은 Python 3.12 + torch 2.11 (우리 pin 불일치) (2026-05-08, Phase 1)

**카테고리**: Pod 환경 / Python 정합

**증상**:
vast.ai instance 의 default Python (/venv/main/python) 은 3.12.13, torch 2.11.0+cu126 — 우리 v4 pin (Python 3.11, torch 2.4.1) 과 불일치. diff_env.sh 가 mismatch FAIL 처리.

**근본 원인**:
vast.ai 가 pytorch:cuda-12.1.1-auto 이미지에 최신 stable torch 를 미리 install 해 둠. v4 의 정합 정책 (정확 핀 동일) 와 충돌.

**v5 반영**:
Pod 부팅 후 miniforge3 로 별도 conda env (e.g. cb) 를 Python 3.11 로 생성하고 그 안에서 requirements.txt install. /venv/main 은 use 안 함. v5 에서는 init-models.sh 첫 단계로 conda env create 자동화 권장.


## L37 — GPU 인프라 vast.ai 로 단일화. RunPod 사용 금지 (2026-05-08, Phase 2)

**카테고리**: 정책 / 인프라

**증상**:
Phase 1 에서 RunPod US-KS-2 가용성 문제 (L34) + 생성 후 SSH 미실행 + 17 GPU fallback 모두 silent fail. vast.ai 로 pivot 후 정상 작동.

**근본 원인**:
사용자 결정 (2026-05-08): GPU 인프라는 vast.ai 단일 사용. RunPod 의 silent failure mode + DC 가용성 매핑 + flag deprecation 등 멀티 클라우드 유지 비용 대비 가치 부족.

**v5 반영**:
v5 에선 인프라 추상화 layer 를 vast.ai-first 로 디자인. CLAUDE.md §3/§4/§13 의 RunPod 절차 → vast.ai 등가물 갱신: HF cache 경로, conda env cb 셋업 (L36), 단가/시간 = vast.ai dashboard 기준. scripts/runpod.sh 는 deprecation 주석만 추가 (즉시 교체는 phase 부풀림 방지). tasks/phase-7-llama.md 의 RUNPOD_LARGE_GPU 등 RunPod 전제도 vast.ai 등가로 갱신. Phase 7d (Llama-70B 8-bit) 만 80GB GPU vast.ai search filter 별도.


## L38 — HF causal mask K dim 이 Q+1 (future-cache slot) (2026-05-08, Phase 3)

**카테고리**: Test 디자인 / HF API

**증상**:
test_mask_is_standard_causal (3.4) 첫 시도 FAIL — captured mask shape (1,1,Q=46,K=47), Q != K. 'prefill mask should be Q×Q' assertion 깨짐.

**근본 원인**:
HF transformers 4.49 의 _update_causal_mask 가 use_cache=True + DynamicCache 사용 시 K 차원에 +1 future-token cache slot 을 패딩. 실제 prefill correctness 는 mask[:,:,:Q,:Q] sub-block 만 lower-triangular 면 충분.

**v5 반영**:
v5 의 mask 검증 test 는 처음부터 Q×Q sub-block 만 검증 (mask[:, :, :Q, :Q]). HF API 가 K dim 에 sentinel/cache slot 을 추가할 가능성을 가정. 또는 use_cache=False 로 측정.


## L39 — cost-tracker.json 은 phase 작업 시간만 기록, vast.ai idle billing 별도 — 'manual' 정책 유지 (2026-05-08, Phase 5)

**카테고리**: Cost tracking 정확성

**증상**:
Phase 4 종료 시점: cost-tracker manual $0.49 vs vast.ai uptime billing 추정 $1.12 = $0.63 차이 (phase 트리거 간 idle). Phase 5 진입 시 사용자 결정 요청.

**근본 원인**:
vast.ai 는 RunPod 와 달리 instance stopped/destroyed 외에는 시간당 단가 charge. phase 작업 시간 외 모든 idle 도 billing. cost-tracker.py manual API 는 phase 별 작업 시간만 받음.

**v5 반영**:
옵션 3 채택 (2026-05-08): cost-tracker.json 의 cumulative_usd 는 'phase 작업 시간 manual 합계' 의 정의 유지. cap 비교에 보수적으로 작동 (idle 누락 = under-report). 정확 reconcile 은 vast.ai dashboard 에서 별도. v5 에선 cost-tracker schema 에 'instance_uptime_billing' 필드 + 'phase_work_billing' 분리 권장. 결정 근거: 옵션 1/2 의 schema/workflow 변경 비용 > Phase 4 의 $0.63 unaccounted (cap $5 안에서 안전).


## L40 — Greedy decode loop 에 torch.inference_mode() 누락 시 24GB GPU OOM (2026-05-08, Phase 6)

**카테고리**: Generation 메모리

**증상**:
Phase 6a 실행 중 RTX 3090 24GB 에서 ~5번째 sample 부터 'OutOfMemoryError: Tried to allocate 20.00 MiB' 빈발 (12/20 examples 만 4 runner 모두 성공). 모델 (14GB) + 800-token DynamicCache (~300MB) + pre-RoPE KV (~1.5GB) 합쳐도 24GB 안 인데 23+GB used 표시.

**근본 원인**:
_greedy_decode 루프의 model(input_ids=...) 호출에 torch.inference_mode()/no_grad() 미적용. 모델이 .eval() 이라도 autograd graph 가 생성되어 매 token 마다 computation graph 메모리 누적. 32 token × 32 layer × intermediate activations = GB 단위 누수.

**v5 반영**:
v5: 모든 generation loop 와 forward 호출에 torch.inference_mode() context 명시 의무. tests/test_no_autograd.py 등으로 model.eval() ≠ no autograd 가정 검증. CPU offload 도 추가 도움 (Phase 6 driver 에 _store_to_gpu 헬퍼 박힘).


## L41 — compute_f1_against_aliases 가 빈 prediction 에서 IndexError (2026-05-08, Phase 7)

**카테고리**: mydata harness 호환성

**증상**:
Phase 7a 의 Llama-8B generation 일부 sample 이 빈 string 반환. mydata harness/metrics.py:33 의 _parse_generation 가 'if s and s.split()[0].startswith(...)' 에서 s='' 또는 whitespace-only 일 때 list index out of range. 전체 7a run rc=1 로 abort.

**근본 원인**:
mydata harness 의 _parse_generation 이 empty/whitespace prediction 을 가드하지 않음. Mistral 은 거의 빈 string 안 나왔으나 Llama-8B 는 일부 sample 에서 발생.

**v5 반영**:
run_phase6.py 의 compute_f1_against_aliases 호출을 try/except (IndexError, Exception) 으로 감싸 빈 prediction 시 f1=0.0 fallback. v5 에선 mydata harness 측 또는 우리 wrapper 에서 prediction 정규화 (None/empty → '').


## L42 — Llama-3.1-8B 에서 ratio=0.15/check_layer=1 default 가 FullReuse 를 능가 못함 (2026-05-08, Phase 7)

**카테고리**: 알고리즘 / 모델 의존성

**증상**:
Phase 7c (n=200) F1: FullRecompute=0.193, FullReuse=0.167, CacheBlend=0.156. ci_low_cb_vs_reuse = -0.058 < 0 (95% CI [-0.058, +0.033], gate 7c.1 FAIL). Mistral 의 +0.046 우위와 정반대. Llama 의 FullReuse F1 가 FullRecompute 의 86% 로 Mistral 의 56% 대비 quality 손실 작음 → selective recompute marginal gain 묻힘.

**근본 원인**:
Llama-3.1 의 RoPE scaling (theta=500000 vs Mistral 1000000) + architecture 차이로 cross-chunk attention drift 패턴 다름. Phase 3 long-chunk sweep 도 Mistral elbow 약함 보고 — 모델 의존성 (v4-lessons L14). ratio=0.15/CL=1 single-CL flat schedule 이 Llama 에 부적절. fixed default 가 모델 별 최적 아님.

**v5 반영**:
v5: 모델 별 ratio/check_layer tuning 단계 의무화 (Phase 8 gradual filtering 의 motivation 강화). 또는 (1) 모델 별 elbow 측정 후 ratio default 결정, (2) multi-check_layer schedule 로 lin_decay 시도 (Phase 8 의 핵심 가설), (3) check_layer 위치를 1 외 다른 layer 로 변경 (예: 2, 5) 시도. Phase 7d (70B 8-bit) 진입 전 8B 결과 분석 필요.


## L43 — fuse_selective 가 LMCache process_qkv 와 다른 알고리즘 (full vs sparse forward) (2026-05-18, Phase 7)

**카테고리**: 알고리즘 정확성

**증상**:
Phase 7c FAIL 이후 LMCache 1:1 비교에서 발견: 우리 fuse_selective 는 check_layer 이후 full forward (Q full, hook 으로 K/V 만 merge); LMCache 는 sparse forward (Q[top_indices] 로 slicing, attention 도 sparse Q × full mixed K/V). 결과적으로 top-K 위치의 K/V 값이 다른 hidden_state propagation 을 거침 — paper §4 의 85% FLOPS 절감 핵심 가설 미충족 + 모델별 quality 차이.

**근본 원인**:
v4 구현 초기에 hook-based 메커니즘 (HF 친화적) 채택. 'K/V 만 merge 하면 의미적으로 동등' 이라는 가정이 잘못. Q 와 hidden_state 도 sparse 가 되어야 LMCache 와 동치. 검증 시 docs/notes/lmcache-1to1-comparison.md 의 6 메커니즘 중 §2.6 만 차이.

**v5 반영**:
fuse_selective_lmc_parity 신규 추가 (fusor.py:248+). HF DecoderLayer 의 sub-module 을 manual 호출하여 check_layer 부터 Q[top]/residual[top] sparse slice + sparse Q × full mixed K/V eager_attention. past_key_values 는 full-length 유지 (decode 호환). CPU 단위 테스트 8/8 PASS. v5 에서는 처음부터 sparse-forward 디자인.

<!-- LESSONS_END -->

---

## 누적 통계

- 총 lessons: 13
- 마지막 업데이트: 2026-05-18
