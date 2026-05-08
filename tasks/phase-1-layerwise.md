# Phase 1 — Layerwise Forward

> **Tolerance category**: SAME_SHAPE (max_diff < 1e-3, freeze, retroactive 변경 금지)
> **Estimated cost**: ~$0.5 (Pod GPU, ~12 min wall)

## Goal

`LayerwiseModel`을 구현해 HF 모델의 forward를 layer-by-layer로 분리. Standard `model(...).logits`와 bit-exact (또는 < 1e-3) 일치. `k_proj` forward-hook으로 pre-RoPE K capture.

## Acceptance Criteria

1. **1.1** — `test_layerwise_matches_standard`: `forward_layerwise(...).logits` vs `model(**inputs).logits` → SAME_SHAPE (max_diff < 1e-3)
2. **1.2** — `test_kv_extraction`: pre-RoPE K capture, all 32 layers max_diff = 0.000e+00 (or < SAME_SHAPE)
3. **1.3** — `verify_phase --phase 1` returns 0
4. **1.4** — Cost ≤ $1 (Phase 1 cap)

## Tasks

### Step 1 — Pod up + setup

```bash
bash scripts/runpod.sh up --auto-recreate
# SSH into pod
# Run CLAUDE.md §Pod setup 6 steps
```

### Step 2 — `LayerwiseModel` 구현 (`src/cacheblend/model.py`)

- `__init__(model_name, dtype, device)`: HF model + tokenizer load with `attn_implementation="eager"` [v3 핵심 결정]
- `embed_tokens(input_ids)`: token ids → embeddings
- `compute_position_embeddings(position_ids)`: cos, sin tensors
- `build_causal_mask(input_ids, position_ids)`: standard causal mask
- `prefill_layer(layer_idx, hidden_states, position_ids, mask, past_kv)`: 단일 layer forward, DynamicCache 사용
- `final_norm_and_lm_head(hidden_states)`: 마지막 norm + LM head
- `forward_layerwise(input_ids, ...)`: 위 단계들 순서대로
- `k_proj` forward-hook으로 pre-RoPE K capture, `get_pre_rope_k(layer_idx)` 인터페이스

### Step 3 — Tests

`tests/test_layerwise.py` (markers: `requires_model and gpu`):

```python
def test_layerwise_matches_standard():
    """SAME_SHAPE tolerance enforce."""
    from cacheblend import LayerwiseModel, Tolerance, assert_logits_close
    
    lw_model = LayerwiseModel("mistralai/Mistral-7B-Instruct-v0.2", dtype="float16")
    inputs = lw_model.tokenizer("Hello world", return_tensors="pt").to(lw_model.device)
    
    lw_logits = lw_model.forward_layerwise(**inputs).logits
    std_logits = lw_model.model(**inputs).logits
    
    assert_logits_close(lw_logits, std_logits, Tolerance.SAME_SHAPE, name="logits")


def test_kv_extraction():
    """pre-RoPE K capture per layer."""
    # ... per-layer max_diff measurement
```

### Step 4 — Pod stop + 보고서

```bash
python scripts/cost_track.py --pod-id <pod_id> --phase 1 --append
bash scripts/runpod.sh stop
# Write reports/phase-1-report.md with v5-lessons section
```

## Gate (auto-evaluated)

`gates/gate-1-to-2.json` 참조.

## v5-lessons 섹션 의무

보고서에 명시 (없으면 "없음").
