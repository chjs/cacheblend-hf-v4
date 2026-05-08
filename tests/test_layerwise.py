"""Phase 1 — Layerwise Forward correctness tests.

Markers: requires_model and gpu (auto-skipped on no-CUDA / no-cache via conftest).
Tolerance: SAME_SHAPE (max_diff < 1e-3) — frozen at Phase 1 start, retroactive change forbidden [L05/L13/L16].
"""
from __future__ import annotations

import pytest
import torch


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The CacheBlend algorithm reduces TTFT by"


@pytest.fixture(scope="module")
def lw_model():
    from cacheblend import LayerwiseModel
    m = LayerwiseModel(MODEL_NAME, dtype="float16")
    yield m
    # Free GPU between modules (test session is short anyway).
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.mark.requires_model
@pytest.mark.gpu
def test_layerwise_matches_standard(lw_model):
    """forward_layerwise(.).logits ≈ model(**inputs).logits within SAME_SHAPE."""
    from cacheblend import Tolerance, assert_logits_close

    inputs = lw_model.tokenizer(PROMPT, return_tensors="pt").to(lw_model.device)
    input_ids = inputs["input_ids"]

    with torch.inference_mode():
        lw_out = lw_model.forward_layerwise(input_ids=input_ids, use_cache=True)
        std_out = lw_model.model(input_ids=input_ids, use_cache=False)

    result = assert_logits_close(
        actual=lw_out.logits,
        expected=std_out.logits,
        category=Tolerance.SAME_SHAPE,
        name="logits",
    )
    print(f"\n[1.1] {result.detail}")


@pytest.mark.requires_model
@pytest.mark.gpu
def test_kv_extraction(lw_model):
    """Pre-RoPE K hook captures all 32 layers; cross-check against direct k_proj."""
    inputs = lw_model.tokenizer(PROMPT, return_tensors="pt").to(lw_model.device)
    input_ids = inputs["input_ids"]

    with torch.inference_mode():
        # Run layerwise to populate the pre-RoPE K dict.
        _ = lw_model.forward_layerwise(input_ids=input_ids, use_cache=True)

        # Independent ground truth: replay embed → norm → k_proj manually.
        emb = lw_model.embed_tokens(input_ids)

        # We need *each* layer's pre-attn-layernorm hidden_states to project K.
        # Easiest faithful path: run layerwise again, but capture per-layer
        # post-input_layernorm hidden_states with a temporary hook.
        per_layer_hidden: dict[int, torch.Tensor] = {}
        handles = []
        for idx, layer in enumerate(lw_model._inner.layers):
            def make_h(i):
                def h(_m, _inp, out):
                    per_layer_hidden[i] = out.detach()
                return h
            handles.append(layer.input_layernorm.register_forward_hook(make_h(idx)))
        try:
            _ = lw_model.forward_layerwise(input_ids=input_ids, use_cache=True)
        finally:
            for h in handles:
                h.remove()

        # Compare hook-captured pre-RoPE K vs k_proj(input_layernorm(h)) per layer.
        per_layer_diffs = []
        for layer_idx in range(lw_model.num_layers):
            captured = lw_model.get_pre_rope_k(layer_idx).float()
            ref = lw_model._inner.layers[layer_idx].self_attn.k_proj(
                per_layer_hidden[layer_idx]
            ).float()
            md = (captured - ref).abs().max().item()
            per_layer_diffs.append(md)

    n = len(per_layer_diffs)
    mn, mx = min(per_layer_diffs), max(per_layer_diffs)
    sorted_diffs = sorted(per_layer_diffs)
    med = sorted_diffs[n // 2]
    print(f"\n[1.2] pre-RoPE K capture per layer (n={n}): "
          f"min={mn:.3e}, median={med:.3e}, max={mx:.3e}")

    assert mx < 1e-3, (
        f"pre-RoPE K hook differs from direct k_proj recompute by max_diff={mx:.3e} "
        f"(SAME_SHAPE bound 1e-3 violated)"
    )
