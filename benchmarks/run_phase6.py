"""Phase 6 driver — Mistral-7B Musique evaluation, sub-phase aware.

Runs 4 methods × N samples on mydata cacheblend_fig12 prompts.jsonl, computes
F1 (max over aliases) + Rouge-L, writes per-example JSONL + summary.json keyed
for `gates/gate-6-final.json`.

Bypasses cacheblend.runners classes for GPU eval — uses fusor functions
directly with `return_layerwise_output=True` so we can extract `past_key_values`
for fast greedy decode (no second forward).

Usage:
  python benchmarks/run_phase6.py --n 20  --out reports/phase-6a-attachments
  python benchmarks/run_phase6.py --n 50  --out reports/phase-6b-attachments
  python benchmarks/run_phase6.py --n 200 --out reports/phase-6c-attachments
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


HARNESS_PATH = Path(__file__).resolve().parent.parent / "external/mydata/cacheblend_fig12"
PROMPTS_PATH = HARNESS_PATH / "prompts.jsonl"
sys.path.insert(0, str(HARNESS_PATH))
# Make `benchmarks.metrics.bootstrap` importable without installing as package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _greedy_decode(model, tokenizer, prefill_logits, past_kv, max_new_tokens, t_start, device):
    """inference_mode-wrapped greedy decode. Critical: without inference_mode the
    autograd graph accumulates per token and OOM hits within ~5 examples on
    24GB even at FP16. [Phase 6 OOM debugging]."""
    eos = getattr(tokenizer, "eos_token_id", None)
    with torch.inference_mode():
        next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter()
        generated = [int(next_id.item())]
        for _ in range(max_new_tokens - 1):
            if eos is not None and generated[-1] == eos:
                break
            out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    # Drop large tensor refs locally before return.
    del past_kv, out, next_id, prefill_logits
    return {
        "text": text,
        "ttft_seconds": t_first - t_start,
        "total_seconds": t_end - t_start,
        "n_generated_tokens": len(generated),
    }


def _build_chunks(tokenizer, ex):
    from cacheblend.chunker import chunk_texts
    parts = []
    parts.append(ex["prompt_parts"]["system"])
    for i, d in enumerate(ex["prompt_parts"]["docs"]):
        parts.append(f"\n\nDocument {i + 1}:\n{d}")
    parts.append(f"\n\nQuestion: {ex['prompt_parts']['question']}\nAnswer:")
    return chunk_texts(tokenizer, parts)


def _run_full_recompute(lw_model, model, tokenizer, ex, max_new_tokens):
    from cacheblend.fusor import fuse_full_recompute
    chunks = _build_chunks(tokenizer, ex)
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    out = fuse_full_recompute(lw_model, chunks, return_layerwise_output=True)
    return _greedy_decode(model, tokenizer, out.logits, out.past_key_values,
                          max_new_tokens, t_start, lw_model.device)


def _run_full_reuse(lw_model, model, tokenizer, ex, max_new_tokens, kv_store):
    from cacheblend.fusor import fuse_full_reuse
    chunks = _build_chunks(tokenizer, ex)
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    out = fuse_full_reuse(lw_model, chunks, kv_store, return_layerwise_output=True)
    return _greedy_decode(model, tokenizer, out.logits, out.past_key_values,
                          max_new_tokens, t_start, lw_model.device)


def _run_prefix_cache(lw_model, model, tokenizer, ex, max_new_tokens, kv_store):
    from cacheblend.fusor import fuse_prefix_cache
    chunks = _build_chunks(tokenizer, ex)
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    out = fuse_prefix_cache(lw_model, chunks, kv_store, return_layerwise_output=True)
    return _greedy_decode(model, tokenizer, out.logits, out.past_key_values,
                          max_new_tokens, t_start, lw_model.device)


def _run_cacheblend(lw_model, model, tokenizer, ex, max_new_tokens, kv_store, ratio, check_layer):
    """Run CacheBlend selective recompute (paper §4 sparse forward)."""
    from cacheblend.fusor import fuse_selective
    chunks = _build_chunks(tokenizer, ex)
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    out = fuse_selective(lw_model, chunks, kv_store,
                         recompute_ratio=ratio, check_layer=check_layer,
                         return_layerwise_output=True)
    return _greedy_decode(model, tokenizer, out.logits, out.past_key_values,
                          max_new_tokens, t_start, lw_model.device)


def _populate_kv_store(lw_model, tokenizer, ex):
    """Precompute pre-RoPE K + V per chunk, offload to CPU to free GPU memory.

    For RTX 3090 24GB tight, holding 8 chunks × 32 layers × ~120 tokens of K + V
    on GPU between fuse calls is ~1.5GB; offloading frees that for the model
    forward + DynamicCache. Trade: per-fuse-call cost to move back to GPU.
    """
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    chunks = _build_chunks(tokenizer, ex)
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw_model, c)
        # Offload to CPU pinned memory; faster transfer back to GPU later.
        K_cpu = [k.detach().cpu() for k in K]
        V_cpu = [v.detach().cpu() for v in V]
        store.put(c.chunk_id, K_cpu, V_cpu)
        del K, V
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return store, chunks


def _store_to_gpu(store, device):
    """Return a new KVStore with all K/V on `device`. Original CPU-resident
    store is unchanged so the caller can free the GPU copy after use."""
    from cacheblend.kv_store import KVStore
    g = KVStore()
    for cid, entry in list(store._cache.items()):
        g.put(
            cid,
            [k.to(device, non_blocking=True) for k in entry["K"]],
            [v.to(device, non_blocking=True) for v in entry["V"]],
        )
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="Number of samples (20/50/200)")
    ap.add_argument("--out", type=str, required=True, help="Output dir for results.jsonl + summary.json")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--cb-ratio", type=float, default=0.15)
    ap.add_argument("--cb-check-layer", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=50, help="append jsonl every N samples [L07]")
    ap.add_argument("--resume", action="store_true", help="skip already-done sample IDs from results.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    # Resume support: read existing results.jsonl, build (id, runner) dedup set.
    seen_keys: set[tuple[str, str]] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen_keys.add((row["id"], row["runner"]))
        print(f"resume: skipping {len(seen_keys)} already-done (id, runner) pairs")
        out_fp = open(results_path, "a")
    else:
        if results_path.exists():
            results_path.unlink()
        out_fp = open(results_path, "w")

    # Load mydata harness metrics (after sys.path insert at top of file).
    from harness.metrics import compute_f1_against_aliases, compute_rouge_l

    # Load examples.
    examples = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples = examples[: args.n]
    print(f"loaded {len(examples)} examples (truncated to {args.n})")

    # Load model + tokenizer once.
    print(f"loading {args.model}", flush=True)
    from cacheblend import LayerwiseModel
    t0 = time.time()
    lw_model = LayerwiseModel(args.model, dtype="float16")
    print(f"  loaded LayerwiseModel in {time.time()-t0:.1f}s")
    model = lw_model.model
    tokenizer = lw_model.tokenizer

    runners = ["FullRecomputeRunner", "FullReuseRunner", "PrefixCacheRunner", "CacheBlendRunner"]

    rows_in_memory = []  # for summary computation; also holds resumed rows

    # Re-load existing rows (resume) into memory for summary computation.
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                rows_in_memory.append(json.loads(line))

    n_done = 0
    t_phase = time.time()
    for i, ex in enumerate(examples):
        ex_id = ex["id"]
        golds = [ex["answer"], *ex.get("answer_aliases", [])]
        question = ex["question"]
        # Drop empty alias strings (mydata sometimes emits empty list).
        golds = [g for g in golds if g] or [ex["answer"]]

        # Reuse KVStore + chunks for the 3 KV-reuse methods.
        kv_store, _chunks = None, None

        # Free per-example tensors aggressively. RTX 3090 24GB is tight: full
        # prompt of ~800 tokens + 8-chunk KV store + 4 methods' past_kv each
        # easily hits OOM without explicit cache clear between samples.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for runner in runners:
            if (ex_id, runner) in seen_keys:
                continue
            try:
                if runner == "FullRecomputeRunner":
                    res = _run_full_recompute(lw_model, model, tokenizer, ex, args.max_new_tokens)
                else:
                    if kv_store is None:
                        kv_store, _chunks = _populate_kv_store(lw_model, tokenizer, ex)
                    # Materialize KV on GPU just for this fuse call.
                    gpu_store = _store_to_gpu(kv_store, lw_model.device)
                    try:
                        if runner == "FullReuseRunner":
                            res = _run_full_reuse(lw_model, model, tokenizer, ex, args.max_new_tokens, gpu_store)
                        elif runner == "PrefixCacheRunner":
                            res = _run_prefix_cache(lw_model, model, tokenizer, ex, args.max_new_tokens, gpu_store)
                        else:  # CacheBlendRunner
                            res = _run_cacheblend(lw_model, model, tokenizer, ex, args.max_new_tokens,
                                                  gpu_store, args.cb_ratio, args.cb_check_layer)
                    finally:
                        gpu_store.clear()
                        del gpu_store
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR ex={ex_id} runner={runner}: {type(e).__name__}: {e}", flush=True)
                continue

            try:
                # mydata _parse_generation has IndexError on empty pred (L41).
                # Fall back to 0.0 — empty prediction is a fail anyway.
                f1 = compute_f1_against_aliases(res["text"], golds, tokenizer)
            except (IndexError, Exception):
                f1 = 0.0
            try:
                rl = max(compute_rouge_l(res["text"], g) for g in golds)
            except Exception:
                rl = 0.0
            row = {
                "id": ex_id,
                "runner": runner,
                "pred": res["text"],
                "golds": golds,
                "f1": f1,
                "rouge_l": rl,
                "ttft_seconds": res["ttft_seconds"],
                "total_seconds": res["total_seconds"],
                "n_generated_tokens": res["n_generated_tokens"],
            }
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_in_memory.append(row)
            n_done += 1
            # Free the result's past_kv (greedy decode kept refs).
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Drop per-example KV store + chunks before next iteration.
        if kv_store is not None:
            kv_store.clear()
        del kv_store, _chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % args.checkpoint_every == 0 or i + 1 == len(examples):
            out_fp.flush()
            print(f"  [{i+1}/{len(examples)}] flushed; n_new_rows={n_done}", flush=True)

    out_fp.close()
    print(f"done eval: {n_done} new rows in {time.time()-t_phase:.1f}s")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = _build_summary(rows_in_memory)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== summary ===\n{json.dumps(summary, indent=2)}")
    print(f"\nWrote {results_path}\nWrote {summary_path}")


def _build_summary(rows: list[dict]) -> dict:
    """Build summary.json keyed for gates/gate-6-final.json conditions."""
    by_runner: dict[str, list[dict]] = {}
    by_id_runner: dict[tuple[str, str], dict] = {}
    for r in rows:
        by_runner.setdefault(r["runner"], []).append(r)
        by_id_runner[(r["id"], r["runner"])] = r

    per_runner_stats = {}
    for name, rs in by_runner.items():
        f1s = [r["f1"] for r in rs]
        rls = [r["rouge_l"] for r in rs]
        ttfts = [r["ttft_seconds"] for r in rs]
        totals = [r["total_seconds"] for r in rs]
        per_runner_stats[name] = {
            "n": len(rs),
            "f1_mean": statistics.mean(f1s) if f1s else 0.0,
            "f1_std": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "rouge_l_mean": statistics.mean(rls) if rls else 0.0,
            "ttft_seconds_mean": statistics.mean(ttfts) if ttfts else 0.0,
            "total_seconds_mean": statistics.mean(totals) if totals else 0.0,
            "f1_zero_count": sum(1 for x in f1s if x == 0.0),
            "f1_one_count": sum(1 for x in f1s if x >= 0.999),
        }

    summary = {"per_runner": per_runner_stats}
    # Gate keys (flattened with dotted notation supported by eval_gate.py).
    summary["FullRecomputeRunner"] = per_runner_stats.get("FullRecomputeRunner", {})

    f_full = per_runner_stats.get("FullRecomputeRunner", {}).get("f1_mean", 0.0)
    f_reuse = per_runner_stats.get("FullReuseRunner", {}).get("f1_mean", 0.0)
    f_cb = per_runner_stats.get("CacheBlendRunner", {}).get("f1_mean", 0.0)
    summary["f1_diff_cb_vs_full"] = f_cb - f_full
    summary["f1_diff_cb_vs_reuse"] = f_cb - f_reuse

    # Paired bootstrap CI: cb vs reuse (per-sample paired).
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    common_ids = sorted({k[0] for k in by_id_runner if k[1] == "CacheBlendRunner"} &
                        {k[0] for k in by_id_runner if k[1] == "FullReuseRunner"})
    if len(common_ids) >= 2:
        a = [by_id_runner[(i, "CacheBlendRunner")]["f1"] for i in common_ids]
        b = [by_id_runner[(i, "FullReuseRunner")]["f1"] for i in common_ids]
        lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=1000, confidence=0.95, seed=42)
        summary["ci_low_cb_vs_reuse"] = lo
        summary["ci_high_cb_vs_reuse"] = hi
        summary["n_paired"] = len(common_ids)
    else:
        summary["ci_low_cb_vs_reuse"] = 0.0
        summary["ci_high_cb_vs_reuse"] = 0.0
        summary["n_paired"] = 0

    return summary


if __name__ == "__main__":
    main()
