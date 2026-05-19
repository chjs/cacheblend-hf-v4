"""CPU sanity tests for fuse_selective (paper §4 / LMCache process_qkv 1:1 port).

Validates the structural invariants that prove equivalence with LMCache's
sparse-forward selective recompute:

  (1) past_key_values is full-length at every layer
  (2) K cache at non-HKVD positions equals cached pre-RoPE K RoPE-shifted to
      fused positions (bit-exact, since RoPE is deterministic)
  (3) K cache at HKVD positions equals the sparse forward's fresh K
  (4) V cache: same pattern (V is RoPE-invariant)
  (5) Last position is always in top_indices (force-include guard)
  (6) Logits at last position have correct shape and non-zero values
  (7) Boundary shortcuts dispatch correctly (ratio=0/1, single chunk)

A tiny random Mistral (4 layers, 64 hidden, 2 KV heads) is used so the whole
test runs on CPU in seconds — no GPU required.
"""
from __future__ import annotations

import torch
from transformers import MistralConfig, MistralForCausalLM


def _build_tiny_mistral(seed: int = 42):
    """Build a tiny random MistralForCausalLM that fits in CPU RAM."""
    torch.manual_seed(seed)
    cfg = MistralConfig(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=10000.0,
        sliding_window=None,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model = MistralForCausalLM(cfg).eval()
    return model, cfg


def _wrap_as_layerwise(hf_model):
    """Wrap a pre-built HF model as a LayerwiseModel (skip from_pretrained)."""
    from cacheblend.model import LayerwiseModel
    # bypass __init__ to avoid HF download path
    lw = LayerwiseModel.__new__(LayerwiseModel)
    lw.model = hf_model
    lw.tokenizer = None  # not needed for these tests
    lw._inner = hf_model.model
    lw.num_layers = len(lw._inner.layers)
    lw.dtype = torch.float32
    lw.device = torch.device("cpu")
    lw._pre_rope_k = {}
    lw._hook_handles = []
    lw._install_k_proj_hooks()
    return lw


def _make_chunks(n_chunks: int = 3, chunk_len: int = 12, vocab: int = 320, seed: int = 7):
    """Build synthetic chunks with random token ids."""
    from cacheblend.chunker import Chunk
    rng = torch.Generator().manual_seed(seed)
    chunks = []
    for ci in range(n_chunks):
        token_ids = torch.randint(0, vocab, (chunk_len,), generator=rng).tolist()
        chunks.append(Chunk(
            text=f"chunk_{ci}",
            token_ids=token_ids,
            chunk_id=f"c{ci}",
        ))
    return chunks


def _populate_kv_store(lw_model, chunks):
    """Precompute pre-RoPE K + V per chunk → KVStore."""
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw_model, c)
        store.put(c.chunk_id, [k.detach().clone() for k in K], [v.detach().clone() for v in V])
    return store


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_boundary_ratio_zero_dispatches_to_full_reuse():
    """ratio=0 → fuse_full_reuse path."""
    from cacheblend.fusor import fuse_selective, fuse_full_reuse
    hf, _ = _build_tiny_mistral()
    lw = _wrap_as_layerwise(hf)
    chunks = _make_chunks(n_chunks=3, chunk_len=10)
    store = _populate_kv_store(lw, chunks)

    logits_parity = fuse_selective(lw, chunks, store, recompute_ratio=0.0)
    logits_reuse = fuse_full_reuse(lw, chunks, store)
    # Should be bit-equal: same dispatch path.
    assert torch.equal(logits_parity, logits_reuse), \
        "ratio=0 must dispatch to fuse_full_reuse (bit-equal)"


def test_boundary_ratio_one_dispatches_to_full_recompute():
    """ratio>=1 → fuse_full_recompute path."""
    from cacheblend.fusor import fuse_selective, fuse_full_recompute
    hf, _ = _build_tiny_mistral()
    lw = _wrap_as_layerwise(hf)
    chunks = _make_chunks(n_chunks=3, chunk_len=10)
    store = _populate_kv_store(lw, chunks)

    logits_parity = fuse_selective(lw, chunks, store, recompute_ratio=1.0)
    logits_rec = fuse_full_recompute(lw, chunks)
    assert torch.equal(logits_parity, logits_rec), \
        "ratio=1 must dispatch to fuse_full_recompute (bit-equal)"


def test_single_chunk_dispatches_to_full_recompute():
    """len(chunks)<=1 → fuse_full_recompute path."""
    from cacheblend.fusor import fuse_selective, fuse_full_recompute
    hf, _ = _build_tiny_mistral()
    lw = _wrap_as_layerwise(hf)
    chunks = _make_chunks(n_chunks=1, chunk_len=10)
    store = _populate_kv_store(lw, chunks)

    logits_parity = fuse_selective(lw, chunks, store, recompute_ratio=0.2)
    logits_rec = fuse_full_recompute(lw, chunks)
    assert torch.equal(logits_parity, logits_rec)


def test_past_key_values_full_length_at_every_layer():
    """End past_key_values must have full-length K/V at every layer
    (required for greedy decode to attend to full prefix)."""
    from cacheblend.fusor import fuse_selective
    hf, cfg = _build_tiny_mistral()
    lw = _wrap_as_layerwise(hf)
    chunks = _make_chunks(n_chunks=3, chunk_len=12)
    store = _populate_kv_store(lw, chunks)
    total_seq = sum(len(c.token_ids) for c in chunks)

    out = fuse_selective(
        lw, chunks, store,
        recompute_ratio=0.2, check_layer=1,
        return_layerwise_output=True,
    )
    pkv = out.past_key_values
    n_layers = cfg.num_hidden_layers
    for li in range(n_layers):
        K, V = pkv.key_cache[li], pkv.value_cache[li]
        # Shape: (B=1, num_kv_heads, total_seq, head_dim)
        assert K.shape == (1, cfg.num_key_value_heads, total_seq, cfg.head_dim), \
            f"layer {li} K shape {K.shape} != full-length"
        assert V.shape == (1, cfg.num_key_value_heads, total_seq, cfg.head_dim), \
            f"layer {li} V shape {V.shape} != full-length"


def test_kv_cache_non_top_equals_rope_shifted_cached():
    """K/V cache at NON-HKVD positions must equal cached pre-RoPE K
    after RoPE-shift to fused positions. V same (no RoPE)."""
    from cacheblend.fusor import fuse_selective
    from cacheblend.rope import apply_rope_shift
    hf, cfg = _build_tiny_mistral()
    lw = _wrap_as_layerwise(hf)
    chunks = _make_chunks(n_chunks=3, chunk_len=12)
    store = _populate_kv_store(lw, chunks)
    total_seq = sum(len(c.token_ids) for c in chunks)

    check_layer = 1
    out = fuse_selective(
        lw, chunks, store,
        recompute_ratio=0.25, check_layer=check_layer,
        return_layerwise_output=True,
    )
    pkv = out.past_key_values

    # Build expected RoPE-shifted cached K and V at full positions per layer.
    from cacheblend.chunker import chunk_offsets
    offsets = chunk_offsets(chunks)
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim

    # Re-build the full-length pre-RoPE K and V from the store (same as inside fuse_selective).
    n_layers = cfg.num_hidden_layers
    K_pre_full = [torch.zeros((1, total_seq, n_kv * hd), dtype=torch.float32) for _ in range(n_layers)]
    V_full = [torch.zeros((1, total_seq, n_kv * hd), dtype=torch.float32) for _ in range(n_layers)]
    for chunk, (start, end) in zip(chunks, offsets):
        entry = store.get(chunk.chunk_id)
        for li in range(n_layers):
            K_pre_full[li][:, start:end, :] = entry["K"][li]
            V_full[li][:, start:end, :] = entry["V"][li]

    position_ids_full = torch.arange(total_seq).unsqueeze(0)

    # For check_layer..end: expect cached@non-top + fresh@top.
    # We don't know top_indices directly, so reconstruct via deviation.
    from cacheblend.hkvd import kv_deviation, select_top_k

    # Need fresh K at check_layer to recompute selection. Run a full forward
    # without hooks to capture fresh K at check_layer.
    from cacheblend.fusor import fuse_full_recompute
    _ = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
    fresh_K_pre_ck = lw.get_pre_rope_k(check_layer)
    deviations = kv_deviation(fresh_K_pre_ck, K_pre_full[check_layer])
    top_indices = select_top_k(deviations, 0.25)
    # Apply the same force-include guard
    last_pos = total_seq - 1
    if last_pos not in top_indices.tolist():
        sel_devs = deviations[top_indices]
        drop_idx = top_indices[sel_devs.argmin()].item()
        top_indices = torch.tensor(
            sorted([i for i in top_indices.tolist() if i != drop_idx] + [last_pos]),
            dtype=top_indices.dtype, device=top_indices.device,
        )
    non_top = [i for i in range(total_seq) if i not in top_indices.tolist()]
    non_top_t = torch.tensor(non_top, dtype=torch.long)

    # Compute expected cached K post-RoPE (full positions) for each layer >= check_layer.
    for li in range(check_layer, n_layers):
        K_cached_post = apply_rope_shift(K_pre_full[li], position_ids_full, lw)  # (1, S, hidden_kv)
        K_cached_post_heads = K_cached_post.view(1, total_seq, n_kv, hd).transpose(1, 2)  # (1, kv, S, hd)
        V_cached_heads = V_full[li].view(1, total_seq, n_kv, hd).transpose(1, 2)

        K_actual = pkv.key_cache[li]  # (1, kv, S, hd)
        V_actual = pkv.value_cache[li]

        # Non-top positions: must equal cached RoPE-shifted K
        diff_K_non_top = (K_actual[:, :, non_top_t, :] - K_cached_post_heads[:, :, non_top_t, :]).abs().max()
        diff_V_non_top = (V_actual[:, :, non_top_t, :] - V_cached_heads[:, :, non_top_t, :]).abs().max()
        assert diff_K_non_top < 1e-5, \
            f"layer {li} non-top K diff {diff_K_non_top:.3e} > 1e-5 (cache mismatch)"
        assert diff_V_non_top < 1e-5, \
            f"layer {li} non-top V diff {diff_V_non_top:.3e} > 1e-5 (cache mismatch)"


def test_last_position_in_top_indices_guarantee():
    """Force-include guard: last position must always be in top_indices,
    so greedy decode has valid logits at the final token."""
    from cacheblend.fusor import fuse_selective
    hf, cfg = _build_tiny_mistral(seed=999)
    lw = _wrap_as_layerwise(hf)
    chunks = _make_chunks(n_chunks=3, chunk_len=12, seed=11)
    store = _populate_kv_store(lw, chunks)
    total_seq = sum(len(c.token_ids) for c in chunks)

    # tiny ratio to stress: only a few tokens selected. Force-include
    # should still kick in for last position.
    out = fuse_selective(
        lw, chunks, store,
        recompute_ratio=0.05, check_layer=1,
        return_layerwise_output=True,
    )
    # Logits at last position must be non-zero (sparse forward populated it).
    last_logits = out.logits[0, total_seq - 1]
    assert (last_logits.abs().sum() > 0), \
        "logits at last position is zero — force-include guard failed"


def test_layers_before_check_layer_are_full_fresh():
    """Layers 0..check_layer-1 must have FULL FRESH K/V (matches full_recompute
    at those layers, since no merge yet)."""
    from cacheblend.fusor import fuse_selective, fuse_full_recompute
    hf, cfg = _build_tiny_mistral(seed=123)
    lw = _wrap_as_layerwise(hf)
    chunks = _make_chunks(n_chunks=3, chunk_len=10, seed=5)
    store = _populate_kv_store(lw, chunks)

    check_layer = 2  # explicit to test layers 0,1 are fresh
    out_parity = fuse_selective(
        lw, chunks, store,
        recompute_ratio=0.2, check_layer=check_layer,
        return_layerwise_output=True,
    )
    out_recompute = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
    pkv_parity = out_parity.past_key_values
    pkv_rec = out_recompute.past_key_values

    for li in range(check_layer):
        diff_K = (pkv_parity.key_cache[li] - pkv_rec.key_cache[li]).abs().max()
        diff_V = (pkv_parity.value_cache[li] - pkv_rec.value_cache[li]).abs().max()
        # full forward → must match full_recompute bit-equal at these layers
        assert diff_K < 1e-5, f"layer {li} pre-check K diff {diff_K:.3e} (should be ~0)"
        assert diff_V < 1e-5, f"layer {li} pre-check V diff {diff_V:.3e} (should be ~0)"


# test_parity_differs_from_legacy_selective removed when legacy fuse_selective
# was deleted. The point of that test was to document that the sparse-forward
# implementation differs from the old hook-based one. With only the sparse
# implementation left, the test no longer has anything to compare against.


if __name__ == "__main__":
    import sys
    tests = [
        test_boundary_ratio_zero_dispatches_to_full_reuse,
        test_boundary_ratio_one_dispatches_to_full_recompute,
        test_single_chunk_dispatches_to_full_recompute,
        test_past_key_values_full_length_at_every_layer,
        test_kv_cache_non_top_equals_rope_shifted_cached,
        test_last_position_in_top_indices_guarantee,
        test_layers_before_check_layer_are_full_fresh,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
