"""Phase 4 TTFT measurement (참고용, gate 조건 아님 [L27]).

Hook-injection design (cacheblend-hf-v4) computes q/k/v over the full token
range at every layer (not just the selected positions), so we do **NOT**
expect TTFT speedup in absolute terms. Numbers below are wall-time of the
*entry-to-logits* span — useful as a sanity check that pipelining doesn't
add overhead and that prefix_cache as a baseline runs.

Methods (5):
  full_recompute, full_reuse, prefix_cache, selective, selective_pipelined.

Sweep:
  3 sequence-length cells (chunk_B ∈ {60, 120, 240}, n_chunks=3).
  Per cell: 1 warmup + 3 timed runs → median wall ms.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch


def _time_call(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0  # ms


def _median_runs(fn, n_warmup: int = 1, n_timed: int = 3) -> float:
    for _ in range(n_warmup):
        fn()
    samples = [_time_call(fn) for _ in range(n_timed)]
    return statistics.median(samples)


def main():
    from cacheblend import LayerwiseModel
    from cacheblend.chunker import Chunk, _stable_id
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.fusor import (
        fuse_full_recompute, fuse_full_reuse, fuse_prefix_cache,
        fuse_selective, fuse_selective_pipelined,
    )

    out_dir = Path("/workspace/cacheblend-hf-v4/reports/phase-4-attachments")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Mistral-7B...", flush=True)
    t0 = time.time()
    model = LayerwiseModel("mistralai/Mistral-7B-Instruct-v0.2", dtype="float16")
    print(f"  loaded in {time.time()-t0:.1f}s")

    SEEDS = [
        "The library at Alexandria contained scrolls covering geometry, astronomy, and medicine. ",
        "Roman aqueducts delivered water across long distances using a slight downhill gradient. ",
        "Bauhaus design favored simple geometry, primary colors, and industrial materials. ",
    ]

    def make_chunks(target_len: int, n: int = 3):
        chunks = []
        for i in range(n):
            text = SEEDS[i % 3]
            while True:
                ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
                if len(ids) >= target_len:
                    break
                text += SEEDS[i % 3]
            ids = ids[:target_len]
            text = model.tokenizer.decode(ids, skip_special_tokens=True)
            chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
        return chunks

    cells = []
    for B in [60, 120, 240]:
        chunks = make_chunks(B, 3)
        total_seq = sum(c.length for c in chunks)
        store = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(model, c)
            store.put(c.chunk_id, K, V)
        print(f"\nchunk_B={B} (3 chunks, total_seq={total_seq})", flush=True)

        methods = {
            "full_recompute": lambda c=chunks: fuse_full_recompute(model, c),
            "full_reuse": lambda c=chunks, s=store: fuse_full_reuse(model, c, s),
            "prefix_cache": lambda c=chunks, s=store: fuse_prefix_cache(model, c, s),
            "selective": lambda c=chunks, s=store: fuse_selective(model, c, s, recompute_ratio=0.15),
            "selective_pipelined": lambda c=chunks, s=store: fuse_selective_pipelined(model, c, s, recompute_ratio=0.15, prefetch=True),
        }
        row = {"chunk_B": B, "total_seq": total_seq}
        for name, fn in methods.items():
            ms = _median_runs(fn, n_warmup=1, n_timed=3)
            row[name] = ms
            print(f"  {name:25s}: {ms:7.2f} ms", flush=True)
        cells.append(row)

    out = {"cells": cells, "note": "TTFT is reported, not gated [L27]. Hook-injection has no algorithmic TTFT savings."}
    (out_dir / "ttft.json").write_text(json.dumps(out, indent=2))

    # Markdown table
    headers = ["chunk_B", "total_seq", "full_recompute", "full_reuse", "prefix_cache", "selective", "selective_pipelined"]
    lines = ["# Phase 4 TTFT (median ms over 3 timed runs, after 1 warmup)\n",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join("---:" for _ in headers) + "|"]
    for row in cells:
        lines.append("| " + " | ".join(
            f"{row[h]:.2f}" if isinstance(row[h], float) else str(row[h])
            for h in headers
        ) + " |")
    (out_dir / "ttft.md").write_text("\n".join(lines) + "\n")

    print("\nDone. Output:")
    print(f"  {out_dir / 'ttft.json'}")
    print(f"  {out_dir / 'ttft.md'}")


if __name__ == "__main__":
    main()
