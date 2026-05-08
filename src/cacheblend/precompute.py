"""Pre-compute per-chunk pre-RoPE K + V via Phase 1 hooks.

Each chunk is forwarded *standalone* (positions 0..L-1, no RoPE shift). We
hook `k_proj` (Phase 1's pre-RoPE K hook, already installed in LayerwiseModel)
and `v_proj` to capture per-layer outputs.
"""
from __future__ import annotations

import torch

from cacheblend.chunker import Chunk


def _install_v_proj_hooks(layerwise_model):
    """Capture v_proj outputs per layer for the next forward.

    Returns: (v_proj_dict, list_of_handles). Caller must remove handles.
    """
    v_dict: dict[int, torch.Tensor] = {}
    handles = []
    for idx, layer in enumerate(layerwise_model._inner.layers):
        v_proj = layer.self_attn.v_proj

        def make_hook(i):
            def hook(_m, _inp, output):
                v_dict[i] = output.detach()
            return hook

        handles.append(v_proj.register_forward_hook(make_hook(idx)))
    return v_dict, handles


def precompute_chunk_kv(
    layerwise_model,
    chunk: Chunk,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run a standalone forward over one chunk and return (K_pre_rope, V) per layer.

    Positions 0..L-1 (chunk-local). No RoPE shift applied here — that happens
    at retrieval (cacheblend.rope.apply_rope_shift) using the blended global
    positions of the fused sequence.

    Returns:
        K_per_layer: list[Tensor (1, L, num_kv_heads*head_dim)] of length num_layers
        V_per_layer: list[Tensor (1, L, num_kv_heads*head_dim)] of length num_layers
    """
    device = layerwise_model.device
    input_ids = torch.tensor([chunk.token_ids], dtype=torch.long, device=device)

    v_dict, v_handles = _install_v_proj_hooks(layerwise_model)
    try:
        with torch.inference_mode():
            _ = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
    finally:
        for h in v_handles:
            h.remove()

    n = layerwise_model.num_layers
    K_per_layer = [layerwise_model.get_pre_rope_k(i).clone() for i in range(n)]
    V_per_layer = [v_dict[i].clone() for i in range(n)]
    return K_per_layer, V_per_layer
