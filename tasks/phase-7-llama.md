# Phase 7 — Llama Evaluation

> **Estimated cost**: ~$15

## Goal

Phase 6과 동일한 평가, Llama 모델:
- Llama-3.1-8B-Instruct (FP16): full sub-phase (7a/7b/7c)
- Llama-3.1-70B-Instruct (8-bit bitsandbytes): 200 sample full only (7d)

## Acceptance

7a/7b/7c는 Phase 6과 동일 구조 (Llama-8B 대상).
7d (Llama-70B):
- **7d.1** — F1 results 생성됨
- **7d.2** — F1(cb) > F1(reuse) at 95% CI
- **7d.3** — Cost ≤ $25 누적

## Tasks

- bitsandbytes 설치: `pip install bitsandbytes>=0.43`
- 70B 8-bit 로드 (80GB GPU 필수)
- vast.ai instance 검색 — 80GB+ GPU filter [L37]:
  ```bash
  vastai search offers 'cuda_vers >= 12.4 num_gpus=1 reliability > 0.98 gpu_ram >= 80 disk_space > 100 verified=true' -o "dph_total" --limit 6
  ```
  권장: A100 80GB ($1.0~1.5/hr), H100 PCIe ($1.5~2.5/hr).
- ~~Pod RUNPOD_LARGE_GPU=true 환경 변수~~ (L37, RunPod deprecated)

## v5-lessons 섹션 의무
