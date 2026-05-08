"""Fusion strategies — full_recompute / full_reuse / selective / pipelined / prefix_cache.

Phase 2 implements:
- `fuse_full_recompute`: baseline. Runs forward over fused input_ids end-to-end.
- `fuse_full_reuse`: KV reuse for *all* chunk tokens via forward-hook injection.
  k_proj output → replaced with stored pre-RoPE K (then HF's apply_rotary_pos_emb
  re-applies RoPE at the *blended* global positions, exactly what we want).
  v_proj output → replaced with stored V.

Phase 3 will add `fuse_selective` (with HKVD); Phase 4 will add pipelined +
prefix_cache.

Boundary safe-shortcut [L13]: ratio=0 → full_reuse, ratio≥1 → full_recompute,
single-chunk full_reuse → full_recompute (single prefix is identical path).
"""
from __future__ import annotations

import torch

from cacheblend.chunker import Chunk, chunk_offsets, fused_input_ids
from cacheblend.kv_store import KVStore


def fuse_full_recompute(
    layerwise_model,
    chunks: list[Chunk],
    return_layerwise_output: bool = False,
):
    """Standard prefill over the fused sequence — no KV reuse.

    Returns logits of shape (1, total_seq, vocab_size). When `return_layerwise_output=True`,
    returns the full `LayerwiseOutput(logits, past_key_values)` for greedy decode.
    """
    input_ids = fused_input_ids(chunks, device=layerwise_model.device)
    with torch.inference_mode():
        out = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
    return out if return_layerwise_output else out.logits


def fuse_full_reuse(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    return_layerwise_output: bool = False,
):
    """Reuse stored pre-RoPE K + V for every chunk token via forward-hook injection.

    Boundary safe-shortcut [L13]: with a single chunk, the fused sequence is
    just that chunk at positions 0..L-1 — same as `fuse_full_recompute`. We
    dispatch directly to keep max_diff = 0 for the single-prefix case.

    For multi-chunk, hooks override `k_proj` and `v_proj` outputs at the cached
    chunk positions. HF's `apply_rotary_pos_emb` then runs *on top* of our
    pre-RoPE K with the blended global positions — equivalent to RoPE-shifting
    chunk-local pre-RoPE K to global positions (paper §4).
    """
    # Boundary: single chunk = single prefix → identical path to full_recompute.
    if len(chunks) <= 1:
        return fuse_full_recompute(layerwise_model, chunks, return_layerwise_output=return_layerwise_output)

    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]

    # Build per-layer K/V override tensors of shape (1, total_seq, hidden_kv).
    # Slice in stored chunk K/V at each chunk's [start:end) range.
    n_layers = layerwise_model.num_layers
    inner = layerwise_model._inner
    attn0 = inner.layers[0].self_attn
    hidden_kv = attn0.config.num_key_value_heads * attn0.head_dim

    device = layerwise_model.device
    dtype = layerwise_model.dtype

    K_override = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    V_override = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    for chunk, (start, end) in zip(chunks, offsets):
        if not kv_store.has(chunk.chunk_id):
            raise KeyError(
                f"fuse_full_reuse: chunk {chunk.chunk_id!r} not in KVStore. "
                f"Did you precompute all chunks first?"
            )
        entry = kv_store.get(chunk.chunk_id)
        for li in range(n_layers):
            K_override[li][:, start:end, :] = entry["K"][li]
            V_override[li][:, start:end, :] = entry["V"][li]

    # Install hooks: replace k_proj/v_proj output for each layer.
    handles = []
    for li, layer in enumerate(inner.layers):
        def make_k_hook(idx):
            def hook(_m, _inp, _out):
                return K_override[idx]
            return hook

        def make_v_hook(idx):
            def hook(_m, _inp, _out):
                return V_override[idx]
            return hook

        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(li)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(li)))

    try:
        input_ids = fused_input_ids(chunks, device=device)
        with torch.inference_mode():
            out = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
        return out if return_layerwise_output else out.logits
    finally:
        for h in handles:
            h.remove()


def fuse_selective(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    recompute_ratio: float = 0.15,
    check_layer: int = 1,
    return_hkvd_indices: bool = False,
    return_layerwise_output: bool = False,
):
    """CacheBlend §4 selective recompute (single-pass, single check_layer).

    Boundary safe-shortcut [L13] (must be the first lines for max_diff=0 guarantee):
      - ratio == 0 → fuse_full_reuse
      - ratio >= 1 → fuse_full_recompute
      - len(chunks) <= 1 → fuse_full_recompute (single prefix is identical path)

    Otherwise (0 < ratio < 1, multi-chunk):
      Layer 0..check_layer-1: no hooks (full fresh). Hidden_states naturally see
        cross-chunk context via causal attention.
      Layer check_layer (default 1): observe k_proj output (fresh K) + compute
        kv_deviation against stored pre-RoPE K → select top-K HKVD indices.
      Layer check_layer+1..end: k_proj/v_proj hooks replace non-HKVD positions
        with stored K/V. HKVD positions keep fresh values. HF's
        apply_rotary_pos_emb runs RoPE on the merged K with the blended global
        positions (same auto-RoPE-shift as fuse_full_reuse).

    Returns:
        logits: (1, total_seq, vocab_size).
        hkvd_indices (optional, when return_hkvd_indices=True): the selected
            top-K indices used for downstream layers.
    """
    # ── Boundary safe-shortcut [L13] ────────────────────────────────────────
    if recompute_ratio == 0:
        out = fuse_full_reuse(layerwise_model, chunks, kv_store, return_layerwise_output=return_layerwise_output)
        return (out, None) if return_hkvd_indices else out
    if recompute_ratio >= 1:
        out = fuse_full_recompute(layerwise_model, chunks, return_layerwise_output=return_layerwise_output)
        return (out, None) if return_hkvd_indices else out
    if len(chunks) <= 1:
        out = fuse_full_recompute(layerwise_model, chunks, return_layerwise_output=return_layerwise_output)
        return (out, None) if return_hkvd_indices else out

    # ── Build per-layer concat overrides from KVStore ───────────────────────
    from cacheblend.hkvd import kv_deviation, select_top_k

    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]
    n_layers = layerwise_model.num_layers
    inner = layerwise_model._inner
    attn0 = inner.layers[0].self_attn
    hidden_kv = attn0.config.num_key_value_heads * attn0.head_dim

    device = layerwise_model.device
    dtype = layerwise_model.dtype

    K_stored = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    V_stored = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    for chunk, (start, end) in zip(chunks, offsets):
        if not kv_store.has(chunk.chunk_id):
            raise KeyError(
                f"fuse_selective: chunk {chunk.chunk_id!r} not in KVStore"
            )
        entry = kv_store.get(chunk.chunk_id)
        for li in range(n_layers):
            K_stored[li][:, start:end, :] = entry["K"][li]
            V_stored[li][:, start:end, :] = entry["V"][li]

    # ── Mutable state populated at check_layer ──────────────────────────────
    state: dict = {"hkvd_indices": None}

    def check_layer_hook(_m, _inp, output):
        # output: fresh pre-RoPE K at check_layer (1, total_seq, hidden_kv)
        deviations = kv_deviation(output, K_stored[check_layer])
        state["hkvd_indices"] = select_top_k(deviations, recompute_ratio)
        # Observation only — no modification.
        return None

    def make_selective_hook(layer_idx: int, stored: list[torch.Tensor]):
        def hook(_m, _inp, output):
            hkvd = state["hkvd_indices"]
            if hkvd is None:
                # Defensive: if check_layer hook didn't fire, behave as fresh.
                return output
            result = output.clone()
            mask = torch.ones(total_seq, dtype=torch.bool, device=output.device)
            mask[hkvd] = False
            result[:, mask, :] = stored[layer_idx][:, mask, :]
            return result
        return hook

    # ── Install hooks ───────────────────────────────────────────────────────
    handles = []
    # Observe at check_layer
    handles.append(
        inner.layers[check_layer].self_attn.k_proj.register_forward_hook(
            check_layer_hook
        )
    )
    # Selective override for layers strictly after check_layer
    for li in range(check_layer + 1, n_layers):
        handles.append(
            inner.layers[li].self_attn.k_proj.register_forward_hook(
                make_selective_hook(li, K_stored)
            )
        )
        handles.append(
            inner.layers[li].self_attn.v_proj.register_forward_hook(
                make_selective_hook(li, V_stored)
            )
        )

    try:
        input_ids = fused_input_ids(chunks, device=device)
        with torch.inference_mode():
            out = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
        result = out if return_layerwise_output else out.logits
        if return_hkvd_indices:
            return result, state["hkvd_indices"]
        return result
    finally:
        for h in handles:
            h.remove()


def fuse_selective_pipelined(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    recompute_ratio: float = 0.15,
    check_layer: int = 1,
    prefetch: bool = True,
):
    """Identical-result wrapper around `fuse_selective` with optional async prefetch.

    Phase 4 design: KV cache loading from `kv_store` is overlapped with the
    forward pass. Functionally:
      - We trigger `kv_store.prefetch_chunk(...)` for each chunk's already-cached
        entry (so the Future is immediately satisfied — RAM tier).
      - The first `kv_store.get(...)` inside `fuse_selective` blocks on those
        Futures if needed (no race: KVStore.get awaits inflight before reading).
    Logits must match `fuse_selective` to within RECOMPUTE_PATH (max_diff < 1e-3),
    typically max_diff = 0 because all loading is RAM-side and deterministic.

    For real I/O tiers (NVMe/SSD/...), the prefetch overlap reduces wall-time;
    quality is unchanged because KV bytes are bit-identical regardless of
    storage tier.
    """
    if prefetch:
        for chunk in chunks:
            if kv_store.has(chunk.chunk_id):
                # RAM tier: use a thunk that returns the in-memory entry.
                # This exercises the prefetch path without doing real I/O —
                # it's the correctness-of-overlap test we need for 4.1.
                cid = chunk.chunk_id

                def _ram_loader(_cid=cid, _store=kv_store):
                    return _store._cache[_cid] if _cid in _store._cache else None

                kv_store.prefetch_chunk(cid, _ram_loader)

    return fuse_selective(
        layerwise_model,
        chunks,
        kv_store,
        recompute_ratio=recompute_ratio,
        check_layer=check_layer,
    )


def fuse_prefix_cache(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    return_layerwise_output: bool = False,
):
    """Baseline: only the first chunk reuses cached K/V; remaining chunks fresh.

    Boundary safe-shortcut [L13]: with a single chunk this is identical to
    fuse_full_recompute (the "first chunk" is the entire input, and reusing
    it at positions 0..L-1 gives the same result).

    For multi-chunk: hooks fire only for the first chunk's K/V positions
    (range [0, L_first)) at every layer. Position-correct because the first
    chunk's standalone-prefill positions == its fused-prefill positions
    (both 0..L_first-1). HF's apply_rotary_pos_emb runs on top normally.
    """
    if len(chunks) <= 1:
        return fuse_full_recompute(layerwise_model, chunks, return_layerwise_output=return_layerwise_output)

    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]
    first_chunk = chunks[0]
    first_start, first_end = offsets[0]

    if not kv_store.has(first_chunk.chunk_id):
        raise KeyError(
            f"fuse_prefix_cache: first chunk {first_chunk.chunk_id!r} not in KVStore"
        )

    n_layers = layerwise_model.num_layers
    inner = layerwise_model._inner
    entry = kv_store.get(first_chunk.chunk_id)

    # Per-layer: replace ONLY positions [first_start, first_end) of fresh K/V
    # with cached values. Other positions pass through unchanged.
    handles = []
    for li, layer in enumerate(inner.layers):
        K_cached_first = entry["K"][li]   # (1, L_first, hidden_kv)
        V_cached_first = entry["V"][li]

        def make_k_hook(cached, start=first_start, end=first_end):
            def hook(_m, _inp, output):
                result = output.clone()
                result[:, start:end, :] = cached
                return result
            return hook

        def make_v_hook(cached, start=first_start, end=first_end):
            def hook(_m, _inp, output):
                result = output.clone()
                result[:, start:end, :] = cached
                return result
            return hook

        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(K_cached_first)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(V_cached_first)))

    try:
        input_ids = fused_input_ids(chunks, device=layerwise_model.device)
        with torch.inference_mode():
            out = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
        return out if return_layerwise_output else out.logits
    finally:
        for h in handles:
            h.remove()
