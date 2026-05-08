"""Phase 3 long-chunk sweep — divergence reduction across chunk size × ratio.

Sweeps:
  chunk_B (target chunk token length)  ∈ {60, 120, 240}
  ratio  (selective recompute fraction) ∈ {0.05, 0.10, 0.15, 0.20, 0.50}

For each cell measures (vs full_recompute = ground truth):
  full_reuse_L2  — Phase 2-style baseline divergence
  selective_L2   — Phase 3 selective with ratio
  reduction_pct  — 1 - selective_L2 / full_reuse_L2
  argmax_match   — fraction of position-wise top-1 predictions matching ground truth

Output:
  reports/phase-3-attachments/long_chunk_sweep.json
  reports/phase-3-attachments/long_chunk_sweep.md (table)

Compares the elbow shape to paper Figure 6. v3 L14 noted that elbow may be
weak/absent for some models; we report the observed pattern, not a target.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


# Long synthetic chunks. Build by repeating a short seed; tokenizer length will
# be ~ proportional to repeat count. We pick repeats to land near 60/120/240
# tokens after tokenization.
SEED_TEXTS = [
    "The library at Alexandria contained scrolls covering geometry, astronomy, and medicine. ",
    "Roman aqueducts delivered water across long distances using a slight downhill gradient. ",
    "Bauhaus design favored simple geometry, primary colors, and industrial materials. ",
]


def build_chunks_for_target_len(tokenizer, target_len: int, n_chunks: int = 3):
    """Return n_chunks Chunk objects whose token_ids length is ≈ target_len.

    We start with the seed text and repeat it until tokenized length crosses
    target_len, then truncate to target_len exactly (so all chunks in a sweep
    cell are the same length and total_seq is deterministic).
    """
    from cacheblend.chunker import Chunk, _stable_id

    chunks = []
    for i in range(n_chunks):
        seed = SEED_TEXTS[i % len(SEED_TEXTS)]
        text = seed
        while True:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= target_len:
                break
            text += seed
        ids = ids[:target_len]
        # Decode back so chunk.text matches the truncated tokens (round-trip safe enough).
        text = tokenizer.decode(ids, skip_special_tokens=True)
        chunks.append(Chunk(
            text=text, token_ids=ids,
            chunk_id=_stable_id(text, ids),
        ))
    return chunks


def measure_cell(model, chunks, store, ratio: float, truth_logits) -> dict:
    from cacheblend.fusor import fuse_selective, fuse_full_reuse

    reuse_logits = fuse_full_reuse(model, chunks, store)
    if ratio == 0.0:
        sel_logits = reuse_logits
    elif ratio >= 1.0:
        from cacheblend.fusor import fuse_full_recompute
        sel_logits = fuse_full_recompute(model, chunks)
    else:
        sel_logits = fuse_selective(model, chunks, store, recompute_ratio=ratio, check_layer=1)

    truth_f = truth_logits.float()
    full_reuse_l2 = (reuse_logits.float() - truth_f).pow(2).mean().sqrt().item()
    selective_l2 = (sel_logits.float() - truth_f).pow(2).mean().sqrt().item()
    reduction = 1.0 - (selective_l2 / max(full_reuse_l2, 1e-12))

    am_truth = truth_logits.argmax(dim=-1)
    am_sel = sel_logits.argmax(dim=-1)
    argmax_match = float((am_truth == am_sel).float().mean().item())

    max_diff = (sel_logits.float() - truth_f).abs().max().item()

    return {
        "ratio": ratio,
        "full_reuse_l2": full_reuse_l2,
        "selective_l2": selective_l2,
        "reduction_pct": 100 * reduction,
        "argmax_match": argmax_match,
        "max_diff_vs_truth": max_diff,
    }


def main():
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.fusor import fuse_full_recompute

    out_dir = Path("/workspace/cacheblend-hf-v4/reports/phase-3-attachments")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Mistral-7B...", flush=True)
    t0 = time.time()
    model = LayerwiseModel("mistralai/Mistral-7B-Instruct-v0.2", dtype="float16")
    print(f"  loaded in {time.time()-t0:.1f}s")

    chunk_lens = [60, 120, 240]
    ratios = [0.05, 0.10, 0.15, 0.20, 0.50]

    grid = {"chunk_lens": chunk_lens, "ratios": ratios, "cells": []}

    for B in chunk_lens:
        chunks = build_chunks_for_target_len(model.tokenizer, B, n_chunks=3)
        total_seq = sum(c.length for c in chunks)
        print(f"\nchunk_B={B} (3 chunks, total_seq={total_seq})", flush=True)
        store = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(model, c)
            store.put(c.chunk_id, K, V)
        truth = fuse_full_recompute(model, chunks)
        for r in ratios:
            stats = measure_cell(model, chunks, store, r, truth)
            row = {"chunk_B": B, "total_seq": total_seq, **stats}
            grid["cells"].append(row)
            print(
                f"  ratio={r:.2f}: full_reuse_L2={stats['full_reuse_l2']:.4e} "
                f"selective_L2={stats['selective_l2']:.4e} "
                f"reduction={stats['reduction_pct']:5.1f}% "
                f"argmax={stats['argmax_match']*100:5.1f}% "
                f"max_diff={stats['max_diff_vs_truth']:.3e}",
                flush=True,
            )

    (out_dir / "long_chunk_sweep.json").write_text(json.dumps(grid, indent=2))

    # Markdown table
    lines = ["# Phase 3 long-chunk sweep\n",
             "| chunk_B | total_seq | ratio | full_reuse_L2 | selective_L2 | reduction% | argmax_match | max_diff |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in grid["cells"]:
        lines.append(
            f"| {c['chunk_B']} | {c['total_seq']} | {c['ratio']:.2f} "
            f"| {c['full_reuse_l2']:.3e} | {c['selective_l2']:.3e} "
            f"| {c['reduction_pct']:.1f} | {c['argmax_match']*100:.1f}% "
            f"| {c['max_diff_vs_truth']:.3e} |"
        )
    (out_dir / "long_chunk_sweep.md").write_text("\n".join(lines) + "\n")

    print("\nDone. Output:")
    print(f"  {out_dir / 'long_chunk_sweep.json'}")
    print(f"  {out_dir / 'long_chunk_sweep.md'}")


if __name__ == "__main__":
    main()
