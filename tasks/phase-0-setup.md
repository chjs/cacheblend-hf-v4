# Phase 0 — Setup & Environment Parity

> **Tolerance category**: N/A (no GPU work in this phase).
> **Estimated cost**: $0 (CPU only).
> **Estimated wall time**: 30 minutes.

## Goal

v4 첫 phase. 환경을 정확히 셋업하고 Mac venv ↔ Pod의 패키지 정합을 확인한다. mydata 저장소 clone + SHA 검증으로 데이터 무결성 보장. LMCache 분석 시작 (Phase 3 HKVD metric 비교의 기반).

## Acceptance Criteria

이 phase 종료 시 다음 모두 통과해야 PASS:

1. **0.1** — `python -c "import cacheblend"` returns 0
2. **0.2** — `python scripts/send_report.py --phase 0 --dry-run` returns 0
3. **0.3** — `bash scripts/diff_env.sh` returns 0 (Mac venv ↔ requirements.txt parity)
4. **0.4** — Skeleton 파일 13개 존재:
   - `src/cacheblend/{__init__, model, kv_store, chunker, rope, precompute, fusor, hkvd, controller, gradual, runners, tolerance}.py` (12)
   - `tests/test_smoke.py` (1)
5. **0.5** — `external/LMCache/` 존재 (depth=1 clone)
6. **0.6** — `external/mydata/` 존재 + `cacheblend_fig12/prompts.jsonl` SHA256 = `791e1cf50d984f27b314c8abd49f25e3b27a0a1598a6cfcf53e28d13868a3e21`
7. **0.7** — `pytest tests/ -m "not gpu and not slow and not requires_model"` returns 0 (smoke test)
8. **0.8** — `docs/lmcache-analysis.md` 작성됨, ≥10 file:line 인용 (LMCache code 직접 참조)
9. **0.9** — `docs/figure12_like_disclosure.md` 작성됨 (paper와의 차이 명시)

## Tasks

### Step 1 — Mac venv 셋업

```bash
# Python 3.11 강제 (3.14 + transformers 5.x + torch 2.11과 충돌 방지) [L01]
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# Pod와 동일한 torch (CPU wheel for Mac)
pip install torch==2.4.1
pip install -r requirements.txt
pip install -e .
```

### Step 2 — Env parity 검증

```bash
bash scripts/diff_env.sh
# Expected: 8/8 match, 0 mismatch
```

### Step 3 — External 저장소 clone

```bash
mkdir -p external

# LMCache (HKVD metric 비교 reference)
if [ ! -d external/LMCache ]; then
    git clone --depth 1 https://github.com/LMCache/LMCache.git external/LMCache
fi

# mydata (cacheblend_fig12 prompts.jsonl)
if [ ! -d external/mydata ]; then
    git clone --depth 1 https://github.com/chjs/mydata.git external/mydata
fi

# SHA256 검증 [L29]
EXPECTED="791e1cf50d984f27b314c8abd49f25e3b27a0a1598a6cfcf53e28d13868a3e21"
ACTUAL=$(shasum -a 256 external/mydata/cacheblend_fig12/prompts.jsonl | awk '{print $1}')
[ "$ACTUAL" = "$EXPECTED" ] && echo "✓ SHA matches" || { echo "✗ SHA mismatch"; exit 1; }
```

### Step 4 — Skeleton 파일 검증

모든 stub 모듈이 import 가능해야 함 [L25]:

```bash
python -c "from cacheblend import (
    Tolerance, LayerwiseModel, KVStore, Chunk,
    fuse_full_recompute, fuse_selective, kv_deviation,
    LoadingController, GradualSchedule,
    FullRecomputeRunner, CacheBlendV4Runner, GradualV4Runner,
)"
```

### Step 5 — Smoke tests

```bash
pytest tests/ -m "not gpu and not slow and not requires_model" -v
# Expected: 9-11 passed
```

### Step 6 — LMCache analysis (≥10 file:line 인용)

`docs/lmcache-analysis.md` 작성, 5 핵심 질문 답변:

1. **Q1**: `compute_layer`에서 어떻게 token slicing하는가? (Phase 3 우리 hook-injection 비교)
2. **Q2**: KV deviation metric 정확한 공식? (`kv_deviation` 정의)
3. **Q3**: Pre-RoPE K 어떻게 저장/적용? (Phase 2 RoPE shift)
4. **Q4**: Check_layer 결정 (단일 vs gradual)?
5. **Q5**: Storage device 처리 (RAM/SSD/...)?

각 답변에 LMCache 코드 file:line 인용 포함 (총 ≥10회).

### Step 7 — Disclosure 문서

`docs/figure12_like_disclosure.md` 작성:

- Paper Figure 12와의 차이점 12개 명시 (ChatGPT 정리 list 참조)
- 우리가 채택한 substitutes:
  - Embedding: `all-mpnet-base-v2` (paper 비공개)
  - Retrieval: L2 raw (no normalization)
  - GPT-4 simulated query 제외
  - Random shuffle: `random.Random(42)` 단일 인스턴스
  - 2WikiMQA / SAMSum / MultiNews 제외 (v5)
  - TTFT 비목표 (L27)

### Step 8 — 보고서 작성 + 이메일

`reports/phase-0-report.md` 작성. **v5-lessons 섹션 의무**:

```markdown
## v5-lessons (이번 phase에서 발견된 사항)

(없으면 "없음" 명시)
```

`scripts/send_report.py --phase 0` 으로 이메일 전송.

## Gate Conditions (auto-evaluated)

`scripts/eval_gate.py --phase 0` 로 자동 검증. Conditions:

```json
{
  "phase": "0",
  "conditions": [
    {"id": "0.1", "check_type": "import", "module": "cacheblend",
     "description": "import cacheblend works"},
    {"id": "0.2", "check_type": "command", "cmd": "python scripts/send_report.py --phase 0 --dry-run",
     "description": "send_report dry-run"},
    {"id": "0.3", "check_type": "command", "cmd": "bash scripts/diff_env.sh",
     "description": "Mac venv ↔ requirements parity"},
    {"id": "0.4a", "check_type": "file_exists", "path": "src/cacheblend/model.py"},
    {"id": "0.4b", "check_type": "file_exists", "path": "src/cacheblend/kv_store.py"},
    {"id": "0.4c", "check_type": "file_exists", "path": "src/cacheblend/fusor.py"},
    {"id": "0.4d", "check_type": "file_exists", "path": "src/cacheblend/runners.py"},
    {"id": "0.4e", "check_type": "file_exists", "path": "src/cacheblend/gradual.py"},
    {"id": "0.4f", "check_type": "file_exists", "path": "src/cacheblend/tolerance.py"},
    {"id": "0.4g", "check_type": "file_exists", "path": "tests/test_smoke.py"},
    {"id": "0.5", "check_type": "file_exists", "path": "external/LMCache/.git"},
    {"id": "0.6", "check_type": "file_exists", "path": "external/mydata/cacheblend_fig12/prompts.jsonl"},
    {"id": "0.7", "check_type": "pytest", "path": "tests/", "markers": "not gpu and not slow and not requires_model"},
    {"id": "0.8", "check_type": "file_exists", "path": "docs/lmcache-analysis.md"},
    {"id": "0.9", "check_type": "file_exists", "path": "docs/figure12_like_disclosure.md"}
  ]
}
```

이 JSON은 `gates/gate-0-to-1.json`에 저장됨 (eval_gate.py가 읽음).

## Deliverables

- `external/LMCache/` (depth=1)
- `external/mydata/` (depth=1, SHA-verified)
- `docs/lmcache-analysis.md` (≥10 file:line 인용)
- `docs/figure12_like_disclosure.md`
- `reports/phase-0-report.md` (with v5-lessons 섹션)
- `gates/gate-0-to-1-result.json` (auto-generated)

## Cost & Time

- GPU: $0 (CPU phase)
- Wall time: ~30분
- 누적 비용: $0
