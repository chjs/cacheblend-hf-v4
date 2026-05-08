# CLAUDE.md — Operational Guide for Claude Code (v4)

> Claude Code가 v4 하네스를 자동으로 진행하기 위한 운영 가이드. v3 진행 중 발견된 30개 lesson 반영.

## 0. 프로젝트 개요

CacheBlend (EuroSys '25) 핵심 알고리즘을 HuggingFace transformers 위에 재구현하여 quality 충실도를 검증. **TTFT는 비목표** (L27).

목표:
- 논문 §4 (selective KV recompute) 의 quality 동등성 검증.
- 논문 §4.3 (gradual filtering) 의 효과를 LMCache 단순 디자인 대비 측정 (Phase 8).
- Mistral-7B + Llama-3.1-8B/70B 평가.

## 1. Pure full auto 원칙 (Phase 0~7)

Phase 0~7은 자동 진행. Phase 8은 interactive (사용자 검토 4회).

### 자동 진행 흐름 (Phase 0~7)

1. Phase task 파일 (`tasks/phase-N-*.md`) 읽기 — 명세 + gate 조건 포함 [L26 통합].
2. 구현 + 테스트.
3. `scripts/eval_gate.py --phase N` 호출 → gate JSON 평가.
4. PASS 시 보고서 작성 + 이메일 + 다음 phase 진입.
5. FAIL 시 BLOCKED 보고서 + 이메일 + STOP.

### Phase 8 흐름 (Interactive)

Phase 8 (Gradual Filtering Discovery) 은 4단계:

```
Step 1 (자동, ~$3) → 6 plots → 보고서 → [사용자 검토 + 프롬프트]
                                              ↓
Step 2 (CPU, $0) → schedule 표 → 보고서 → [사용자 검토 + 프롬프트]
                                              ↓
Step 3 (GPU, ~$15) → F1 heatmap → 보고서 → [사용자 검토 + 프롬프트]
                                              ↓
Step 4 (자동, ~$1) → LMCache 비교 → 최종 보고서
```

각 step 완료 시 STOP. 사용자가 다음 step trigger 보낼 때까지 대기.

## 2. 환경 정합 (Phase 0 의무)

### Mac venv ↔ Pod 패키지 일괄 비교

`scripts/diff_env.sh` 가 8 패키지 버전 자동 비교:

```bash
# Phase 0 첫 단계
bash scripts/diff_env.sh
# Mac venv: torch=2.4.1, transformers=4.49.0, ...
# Pod:      torch=2.4.1+cu124, transformers=4.49.0, ...
# 8/8 match, 0 mismatch
```

8 패키지: torch, transformers, datasets, accelerate, huggingface-hub, tokenizers, safetensors, numpy.

mismatch 발생 시 Phase 0 FAIL — `requirements.txt` 패치 후 재실행.

### `requirements.txt` 정확한 == 핀

```
torch==2.4.1
transformers==4.49.0
accelerate==1.13.0
huggingface-hub==0.36.2
tokenizers==0.21.4
safetensors==0.7.0
datasets==4.8.5
numpy>=2.0,<3.0
sentence-transformers>=3.0,<5.0
rouge-score>=0.1.2
matplotlib>=3.8
pandas>=2.0
python-dotenv>=1.0
pytest>=8.0
pytest-cov>=5.0
ruff>=0.5
# bitsandbytes>=0.43  # Phase 7 Llama-70B에서만 추가 설치
```

## 3. Pod setup (vast.ai, Phase 1+ GPU phase 시작 시) [L37]

GPU 인프라는 **vast.ai 단일** 사용. RunPod 사용 금지 (L34/L37).

### Pod 부팅 — vast.ai

```bash
# RTX 3090 24GB, $0.13~0.20/hr 범위 단일 GPU 검색 (Mistral-7B FP16에 충분).
# Llama-70B 8-bit (Phase 7d) 만 80GB GPU 별도 검색.
vastai search offers 'cuda_vers >= 12.4 num_gpus=1 inet_down >= 1500 reliability > 0.98 dph_total < 0.6 gpu_ram >= 24 disk_space > 60 verified=true' -o "dph_total" --limit 6

vastai create instance <ID> --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime --disk 60 --ssh --direct --label "cacheblend-v4-phase<N>"

# SSH 정보 확인 (직접 IP + 점프 SSH 둘 다 가능)
vastai show instances
# 예: ssh -p 32318 root@<host_ip>  또는  ssh -p <jump_port> root@sshN.vast.ai

# Status loading→running 폴링 (2-5분 image pull):
until vastai show instance <iid> --raw | python3 -c "import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get('actual_status')=='running' else 1)"; do sleep 15; done
```

### Setup 7단계 (vast.ai, /venv/main 미사용 + miniforge conda env `cb`) [L36, L37]

매 SSH 접속 직후:

```bash
# 1. SSH 접속 (사용자가 명시한 endpoint, 자동 reboot 금지 — L35)
ssh -i ~/.ssh/id_rsa -p <port> root@<ip>

# 2. miniforge 로 Python 3.11 conda env 생성 (default /venv/main 은 Py 3.12 + torch 2.11 → 우리 pin 과 불일치, L36)
source /opt/miniforge3/etc/profile.d/conda.sh
conda create -n cb python=3.11 -y
conda activate cb

# 3. (Mac → Pod) 코드 전송 — rsync 가 vast.ai 에서 끊기는 경우 tarball+scp 우회
# Mac 측: tar --exclude='external/mydata/musique' --exclude='external/mydata/2wikimultihop' --exclude='external/LMCache' --exclude='.venv' -czf /tmp/cb-v4.tgz .
# scp -P <port> /tmp/cb-v4.tgz root@<ip>:/tmp/
# Pod 측: mkdir -p /workspace/cacheblend-hf-v4 && cd /workspace/cacheblend-hf-v4 && tar -xzf /tmp/cb-v4.tgz

# 4. HF cache 경로 (vast.ai 의 /workspace 는 컨테이너 로컬 disk, 100GB)
export HF_HOME=/workspace/.hf_home
export HF_HUB_CACHE=/workspace/.hf_home/hub
export HF_DATASETS_CACHE=/workspace/.hf_home/datasets

# 5. Package install (cb env 내, torch 도 함께 install)
cd /workspace/cacheblend-hf-v4
pip install -q torch==2.4.1
grep -v -E '^torch(\s|=|$)' requirements.txt > /tmp/reqs-no-torch.txt
pip install -q -r /tmp/reqs-no-torch.txt
pip install -q -e .

# 6. Torch CUDA build 검증 (cu121/cu124/cu126 등 모두 OK, diff_env.sh 가 strip)
python -c "import torch; v = torch.__version__; assert v.startswith('2.4.1') and 'cu' in v, f'torch missing cuda: {v}'; print(f'torch {v} cuda={torch.cuda.is_available()} OK')"

# 7. HF auth + diff_env 검증
set -a; . /workspace/cacheblend-hf-v4/.env; set +a
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False); print('HF auth OK')"
bash scripts/diff_env.sh   # 8/8 match 의무
```

이 7단계는 셸 세션마다 다시 실행 (vast.ai instance restart 시도 마찬가지). `.bashrc` 에 박지 않음 — pod ephemeral. **사용자가 할당한 instance 는 임의 reboot 금지** [L35].

## 4. Pod 운영 (vast.ai) [L37]

### Pod 가용성 / 재부팅 정책

- 사용자 할당 instance: **reboot/destroy 금지** (사용자 정책 L35). 부팅 실패 시 사용자에게 보고하고 결정 대기.
- 자동화가 만든 instance: 시도 실패 시 destroy + 다른 GPU offer 자동 선택.
- vast.ai instance 가 hung 한 경우 (SSH timeout, container "created" but never "running") 30 분 내 자동 회복 안되면 STOP 후 보고.

### Long phase 에서 instance 끊김

Phase 6/7/8 sub-phase 가 1+ 시간 걸리는 경우 incremental checkpoint:

- 50 sample 마다 jsonl append (`benchmarks/results/<dataset>_<n>.partial.jsonl`)
- vast.ai instance recycle 후 재시작 시 이미 처리한 sample skip
- 결과 통합은 `scripts/merge_partial.py`
- vast.ai 는 RunPod 와 달리 network volume 자동 mount 가 없으므로 (instance ephemeral) — 중요한 데이터는 매 phase 마다 Mac 로 scp 회수 의무

## 5. v5-lessons 누적 워크플로

v4 진행 중 발견되는 새로운 문제 / 개선 사항 → `docs/notes/v5-lessons.md`.

### 보고서 의무 섹션

모든 phase 보고서 마지막에 **v5-lessons 섹션 의무**:

```markdown
## v5-lessons (이번 phase에서 발견된 사항)

이번 phase에서 새로 발견된 문제 / 개선 사항:
- L31 — <제목>: <한 줄 요약>
- L32 — <제목>: <한 줄 요약>

(없으면 "없음" 명시)

상세 내용은 `docs/notes/v5-lessons.md` 참조.
```

### `scripts/add_lesson.py` CLI

표준 형식으로 lesson 추가:

```bash
python scripts/add_lesson.py \
    --phase 3 \
    --category "알고리즘 정확성" \
    --title "HKVD elbow shape이 모델별로 다름" \
    --symptom "Mistral은 elbow at ratio=0.10, Llama는 ratio=0.20" \
    --root-cause "Attention sparsity가 architecture에 의존" \
    --v5-fix "Phase 3 task에 모델별 elbow ratio 측정 단계 추가"
```

→ 자동으로 다음 L## 번호 부여, `docs/notes/v5-lessons.md` 끝에 append, 라인 수 갱신.

### Lesson 철회 (B 방식)

발견 후 재평가에서 lesson 아닌 것으로 판명되면 strike-through로 보존:

```markdown
## ~~L42 — <제목> (철회: <이유>)~~

(원래 내용은 보존, 다만 ~~로 감싸 시각적으로 무효 표시)
```

`scripts/retract_lesson.py L42 --reason "..."` 로 자동 처리.

### 사용자 즉시 추가

사용자가 phase 진행 중 발견한 사항을 한 줄 메시지로 보내면 즉시 v5-lessons.md에 박음. 다음 phase 보고서까지 기다리지 않음.

## 6. Tolerance enforcement (L05, L13, L16)

Phase 시작 전 카테고리 freeze. **Retroactive 변경 금지**.

```python
# src/cacheblend/tolerance.py
class Tolerance(Enum):
    IDENTICAL_PATH = "max_diff == 0"
    SAME_SHAPE = "max_diff < 1e-3"
    MIXED_SHAPE = "argmax_exact AND max_diff < 5e-2"
    RECOMPUTE_PATH = "max_diff < 1e-3"

def assert_tolerance(actual_max_diff, actual_argmax_match, category):
    """Phase 시작 시 정해진 카테고리로 검증. 카테고리 변경 금지."""
    ...
```

각 phase task 파일에 `tolerance_category: MIXED_SHAPE` 같이 명시. 측정값이 초과하면 phase FAIL — tolerance 늘리는 게 아니라 알고리즘 디버깅.

## 7. Boundary safe-shortcut (L13)

ratio=0 / ratio>=1 cases는 코드 경로 동일화로 max_diff=0 보장:

```python
def fuse_selective(model, chunks, kv_store, recompute_ratio, ...):
    if recompute_ratio == 0:
        return fuse_full_reuse(model, chunks, kv_store, ...)
    if recompute_ratio >= 1:
        return fuse_full_recompute(model, chunks, ...)
    # else: actual selective logic
    ...
```

이는 디자인 의무. v4 fusor.py 첫 줄부터 박힘.

## 8. 보고서 환경 라벨 의무 (L09)

모든 수치에 환경 라벨 명시:

```markdown
| 항목 | 값 |
|---|---|
| torch | 2.4.1+cu124 (Pod) |
| transformers | 4.49.0 (Pod) |
| max_diff | 0.000e+00 (A100 80GB, FP16) |
| F1 (Mistral, Musique) | 0.34 (RTX A6000) |
```

같은 패키지가 Mac/Pod 둘 다 있으면 두 라벨 모두 표기.

## 9. Cost tracking 자동화 (L30)

`scripts/cost_track.py` 가 `runpodctl pod get --output json` 의 `costPerHr` 자동 파싱:

```bash
# Pod 종료 시 자동 호출
python scripts/cost_track.py \
    --pod-id <pod_id> \
    --phase 3 \
    --append reports/cost-tracker.json
```

Phase 종료 시점에 wall time × costPerHr 자동 기록. Pod reclaim 시 양 pod의 wall time 합산.

## 10. F1 평가 — Bootstrap CI (L28)

Phase 6/7 sub-phase 평가는 절대값 + bootstrap CI 둘 다:

```python
from benchmarks.metrics.qa import paired_bootstrap_ci

ci_low, ci_high = paired_bootstrap_ci(
    f1_cb_per_sample,
    f1_baseline_per_sample,
    n_bootstrap=1000,
    confidence=0.95,
)
# 통과 조건: ci_low > 0 → "F1(cb) > F1(baseline) at 95% CI"
```

Gate 조건에 절대값 0.05 차이 외 bootstrap CI > 0 도 포함.

## 11. 비용 한도

| Phase | 한도 | 누적 한도 |
|---|---|---|
| 0~5 | $5 | $5 |
| 6 | $10 | $15 |
| 7 | $15 | $30 |
| 8 | $25 | $55 |

한도 80% 도달 시 보고서에 ⚠️ 명시. 한도 초과 시 자동 STOP.

## 12. SSH key (vast.ai) [L37]

- SSH key path: `~/.ssh/id_rsa` (Mac local). vast.ai 는 `vastai doctor` 실행 시 자동 sync.
- vast.ai 는 직접 SSH (host IP + port) 와 점프 SSH (`sshN.vast.ai:port`) 둘 다 제공. 둘 다 시도.
- ~~RunPod 의 `~/.runpod/ssh/RunPod-Key-Go` 는 더이상 사용 안 함 (L37)~~

## 13. GPU 선택 (vast.ai) [L37]

vast.ai search filter 로 가용 인스턴스 자동 검색:

```bash
# 일반 phase (Mistral-7B / Llama-8B FP16): 24GB+ GPU
vastai search offers 'cuda_vers >= 12.4 num_gpus=1 inet_down >= 1500 reliability > 0.98 dph_total < 0.6 gpu_ram >= 24 disk_space > 60 verified=true' -o "dph_total" --limit 6

# Phase 7d 만 80GB GPU (Llama-70B 8-bit):
vastai search offers 'cuda_vers >= 12.4 num_gpus=1 reliability > 0.98 gpu_ram >= 80 disk_space > 100 verified=true' -o "dph_total" --limit 6
```

권장 GPU (단가 순):
- Mistral-7B / Llama-8B: RTX 3090 ($0.13~0.20), RTX 4090 ($0.25~0.40), A6000 ($0.30~0.50)
- Llama-70B 8-bit: A100 80GB ($1.0~1.5), H100 PCIe ($1.5~2.5)

instance 못 잡으면 다른 host 로 자동 전환. 3번 실패 시 STOP. ~~RunPod 의 30분/1시간/2시간 backoff 는 더이상 사용 안 함 (L37)~~

## 14. 인프라 스크립트 명세

| 스크립트 | 설명 |
|---|---|
| `scripts/diff_env.sh` | Mac/Pod 패키지 8개 일괄 비교 |
| ~~`scripts/runpod.sh`~~ | **DEPRECATED** [L37] — vast.ai 단일 사용. 헤드 주석 참조. |
| `scripts/init-models.sh` | (RunPod 시절, vast.ai 에선 setup 7단계 inline 사용) |
| `scripts/eval_gate.py` | Gate JSON 평가 (metric/cost_check/sub_phases 모두 핸들) |
| `scripts/verify_phase.py` | Phase 별 파일/테스트 검증 |
| `scripts/cost_track.py` | vast.ai dashboard 단가 manual 기록 (`--manual-usd`); RunPod billing 자동 파싱 deprecated [L37] |
| `scripts/add_lesson.py` | v5-lessons.md에 표준 형식 추가 |
| `scripts/retract_lesson.py` | Lesson strike-through |
| `scripts/send_report.py` | 이메일 보고 |
| `scripts/merge_partial.py` | Phase 6/7/8의 incremental checkpoint 통합 |

## Cross-references

- v3 누적 lessons (30개): `docs/notes/v4-lessons.md`
- v4 진행 중 누적: `docs/notes/v5-lessons.md`
- Phase task: `tasks/phase-N-*.md` (gate JSON 통합)
- Design decisions: `docs/design-decisions.md`
