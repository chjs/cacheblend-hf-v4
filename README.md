# CacheBlend HF v4 — Quality-only Reproduction with Gradual Filtering Discovery

CacheBlend (EuroSys '25) 알고리즘을 HuggingFace transformers 위에 재구현. v3에서 누적된 30개 lesson 반영 + 논문 §4.3 gradual filtering scheme의 효과 측정 (LMCache 미구현).

## Goals

- **Quality-only milestone**: F1 / Rouge-L / logit equivalence 검증. TTFT 비목표.
- **Figure 12-like 재현**: Mistral-7B-Instruct-v0.2 + Llama-3.1-8B/70B-Instruct on MuSiQue 200.
- **Phase 8 신규**: Gradual filtering vs LMCache flat schedule head-to-head.

## Quick Start

```bash
# Mac venv setup (Phase 0)
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.1
pip install -r requirements.txt
pip install -e .

# Verify env parity
bash scripts/diff_env.sh

# Clone external dependencies
git clone --depth 1 https://github.com/LMCache/LMCache.git external/LMCache
git clone --depth 1 https://github.com/chjs/mydata.git external/mydata

# Verify mydata SHA
shasum -a 256 external/mydata/cacheblend_fig12/prompts.jsonl
# Expected: 791e1cf50d984f27b314c8abd49f25e3b27a0a1598a6cfcf53e28d13868a3e21

# Phase 0 smoke tests
pytest tests/ -m "not gpu and not slow and not requires_model"

# Phase 0 gate
python scripts/eval_gate.py --phase 0
```

## Phase Overview

| Phase | Name | Cost | Auto/Interactive |
|---|---|---|---|
| 0 | Setup & Env Parity | $0 | Auto |
| 1 | Layerwise Forward | ~$0.5 | Auto |
| 2 | KV Storage & Full Reuse | ~$0.5 | Auto |
| 3 | Selective Recompute | ~$0.7 | Auto |
| 4 | Pipelining & Prefix Cache | ~$0.5 | Auto |
| 5 | Dataset Pipeline (mydata) | $0 | Auto |
| 6 | Mistral Eval (200) | ~$10 | Auto (sub-phases) |
| 7 | Llama Eval | ~$15 | Auto (sub-phases) |
| 8 | Gradual Filtering Discovery | ~$25 | **Interactive** (4 review checkpoints) |

Total cost cap: **$55**.

## Documentation

- `GOAL.md` — High-level project goals
- `CLAUDE.md` — Operational guide for Claude Code
- `tasks/phase-N-*.md` — Per-phase specifications + acceptance criteria
- `docs/design-decisions.md` — Tolerance categories, key design decisions
- `docs/notes/v4-lessons.md` — 30 lessons from v3 reflected in v4
- `docs/notes/v5-lessons.md` — v4 progress lessons (auto-accumulated)

## v5-lessons Auto-Accumulation

```bash
python scripts/add_lesson.py --phase 3 --category "..." --title "..." --symptom "..." --root-cause "..." --v5-fix "..."
python scripts/retract_lesson.py L42 --reason "..."
```

Each phase report has mandatory `## v5-lessons` section.

## License

Research artifact. See LICENSE.
