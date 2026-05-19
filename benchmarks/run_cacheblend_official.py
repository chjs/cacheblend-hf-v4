"""CacheBlend official-protocol experiments — port of YaoJiayi/CacheBlend examples.

Runs the original CacheBlend benchmarks (Musique, 2WikiMHQA, SamSum) against
our v4 implementation. Faithful prompt format reproduction (including Mistral
[INST]/[/INST] token markers), same metrics (F1 / ROUGE-L), same per-dataset
recompute ratio (0.15 for Musique, 0.18 for 2WikiMHQA/SamSum).

Additional measurements beyond the official scripts (per user request):
  - Full KV reuse baseline (official has only CB vs Full prefill)
  - Paired bootstrap CI (CB r vs Full reuse)
  - HKVD boundary enrichment analysis (chunk boundary concentration)

Source: external/CacheBlend/example/{blend_musique, blend_wikimqa, blend_samsum}.py
Datasets: external/CacheBlend/inputs/{musique_s, wikimqa_s, samsum}.json
Utils (F1, ROUGE-L): external/CacheBlend/example/utils.py (imported verbatim)

Usage:
  python benchmarks/run_cacheblend_official.py --dataset musique --n 150 \\
      --out reports/cb-official-musique --cb-ratio 0.15
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/CacheBlend/example"))

# Official Mistral [INST]/[/INST] token sequences (from official scripts).
INST_OPEN = [733, 16289, 28793]       # = "[INST]" prepended for QA datasets
INST_CLOSE = [733, 28748, 16289, 28793]  # = "[/INST]" appended for QA datasets


DATASET_CONFIGS = {
    "musique": {
        "path": "external/CacheBlend/inputs/musique_s.json",
        "metric": "f1",
        "max_new_tokens": 32,
        "use_inst": True,
        "default_ratio": 0.15,
        "prefix": (
            "You will be asked a question after reading several passages. "
            "Please directly answer the question based on the given passages. "
            "Do NOT repeat the question. The answer should be within 5 words..\n"
            "Passages:\n"
        ),
        "query_prefix": (
            "\n\nAnswer the question directly based on the given passages. "
            "Do NOT repeat the question. The answer should be within 5 words. "
            "\nQuestion:"
        ),
        "answers_extract": lambda ex: ex["answers"],  # list[str]
    },
    "wikimqa": {
        "path": "external/CacheBlend/inputs/wikimqa_s.json",
        "metric": "f1",
        "max_new_tokens": 32,
        "use_inst": True,
        "default_ratio": 0.18,
        "prefix": (
            "Answer the question based on the given passages. Only give me the "
            "answer and do not output any other words.\n\nThe following are "
            "given passages.\n"
        ),
        "query_prefix": (
            "\n\nAnswer the question based on the given passages. Answer the "
            "question within 5 words. Do NOT repeat the question or output any "
            "other words. Question: "
        ),
        # wikimqa answers is list[list[str]] — flatten to list[str]
        "answers_extract": lambda ex: [a[0] if isinstance(a, list) else a for a in ex["answers"]],
    },
    "samsum": {
        "path": "external/CacheBlend/inputs/samsum.json",
        "metric": "rouge_l",
        "max_new_tokens": 128,
        "use_inst": False,  # samsum has no [INST]/[/INST] in official script
        "default_ratio": 0.18,
        "prefix": (
            "Summarize the dialogue into a few short sentences. The following "
            "are some examples.\n\n"
        ),
        "query_prefix": "\n\n",  # samsum just \n\n + question
        "answers_extract": lambda ex: ex["answers"],
        "max_ctx_len": 3400,  # drop few-shot ctxs to fit (official policy)
    },
}


def _build_chunks_for_example(tokenizer, ex, cfg):
    """Build per-chunk token IDs replicating the official script's structure.

    Returns:
        token_id_chunks: list[list[int]] — [s_start_full_chunk, doc_1, doc_2, ..., doc_N, query_chunk]
        text_chunks: list[str] — chunk texts (decoded for chunker.Chunk hashing)
        golds: list[str] — gold answers
        chunk_boundaries_meta: dict {prefix_end, query_start} for downstream HKVD analysis
    """
    from utils import (
        normalize_question, build_qa_prompt, build_fewshot_prompt
    )

    golds = cfg["answers_extract"](ex)

    if cfg["metric"] == "f1":
        doc_prompts, q_prompt = build_qa_prompt(ex, cfg["query_prefix"])
    else:
        # samsum few-shot
        doc_prompts, q_prompt = build_fewshot_prompt(ex)

    # tokenize prompts (offcial uses [1:] to strip BOS; we do same)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    # samsum: drop middle few-shot ctxs if exceeding max_ctx_len
    if cfg["metric"] == "rouge_l":
        from itertools import chain
        max_ctx_len = cfg["max_ctx_len"]
        while sum(len(c) for c in doc_chunk_ids) > max_ctx_len:
            del_idx = int(len(doc_chunk_ids) / 2)
            del doc_chunk_ids[del_idx]
        if not doc_chunk_ids:
            return None, None, None, None  # skip example

    # Build first chunk (prefix), optionally with [INST] prepended
    if cfg["use_inst"]:
        s_start_full = INST_OPEN + tokenizer.encode(cfg["prefix"])[1:]
        s_end = INST_CLOSE
    else:
        s_start_full = tokenizer.encode(cfg["prefix"])[1:]
        s_end = []

    # Doc chunks (no extra prepend — official's `s_start=[]`)
    # Query chunk: q_ids + s_end ([/INST] for QA)
    token_id_chunks = [s_start_full] + doc_chunk_ids + [q_ids + s_end]

    text_chunks = [tokenizer.decode(ids, skip_special_tokens=False) for ids in token_id_chunks]
    return token_id_chunks, text_chunks, golds, None


def _make_chunks_from_ids(token_id_chunks, text_chunks):
    """Wrap (text, token_ids) tuples as cacheblend.Chunk objects with stable hashes."""
    from cacheblend.chunker import Chunk, _stable_id
    out = []
    for text, ids in zip(text_chunks, token_id_chunks):
        out.append(Chunk(text=text, token_ids=list(ids), chunk_id=_stable_id(text, list(ids))))
    return out


def _greedy_decode(model, tokenizer, prefill_logits, past_kv, max_new_tokens, t_start, device):
    """Same greedy decode loop as run_phase6.py — inference_mode wrapped."""
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
    del past_kv, out, next_id, prefill_logits
    return {"text": text, "ttft_seconds": t_first - t_start,
            "total_seconds": t_end - t_start, "n_generated_tokens": len(generated)}


def _run_method(lw_model, model, tokenizer, chunks, kv_store, method, max_new_tokens,
                ratio=0.15, check_layer=1, capture_hkvd=False):
    from cacheblend.fusor import (
        fuse_full_recompute, fuse_full_reuse, fuse_selective_lmc_parity,
    )
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    hkvd = None
    if method == "FullRecompute":
        out = fuse_full_recompute(lw_model, chunks, return_layerwise_output=True)
    elif method == "FullReuse":
        out = fuse_full_reuse(lw_model, chunks, kv_store, return_layerwise_output=True)
    elif method == "CacheBlend":
        if capture_hkvd:
            out, hkvd = fuse_selective_lmc_parity(
                lw_model, chunks, kv_store,
                recompute_ratio=ratio, check_layer=check_layer,
                return_layerwise_output=True, return_hkvd_indices=True,
            )
        else:
            out = fuse_selective_lmc_parity(
                lw_model, chunks, kv_store,
                recompute_ratio=ratio, check_layer=check_layer,
                return_layerwise_output=True,
            )
    else:
        raise ValueError(method)
    res = _greedy_decode(model, tokenizer, out.logits, out.past_key_values,
                          max_new_tokens, t_start, lw_model.device)
    if hkvd is not None:
        res["hkvd_indices"] = hkvd.detach().cpu().tolist()
    return res


def _populate_kv_store(lw_model, chunks):
    """Per-chunk standalone forward → KVStore. Matches official's per-chunk
    collect=True hack (each chunk's K/V captured in isolation, positions 0..L-1
    with chunk-local attention)."""
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw_model, c)
        store.put(c.chunk_id, [k.detach().cpu() for k in K], [v.detach().cpu() for v in V])
    return store


def _store_to_gpu(store, device):
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
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--n", type=int, default=None, help="If set, only first n examples; else all")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--cb-ratio", type=float, default=None,
                    help="CacheBlend ratio; defaults to dataset's default (0.15 musique, 0.18 others)")
    ap.add_argument("--cb-check-layer", type=int, default=1)
    ap.add_argument("--attn-impl", default="eager")
    args = ap.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    ratio = args.cb_ratio if args.cb_ratio is not None else cfg["default_ratio"]
    print(f"dataset={args.dataset}  metric={cfg['metric']}  cb_ratio={ratio}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    # Load dataset (use official load_dataset for fidelity)
    from utils import load_dataset, compute_f1, compute_rl
    examples = load_dataset(cfg["path"])
    if args.n is not None:
        examples = examples[: args.n]
    print(f"loaded {len(examples)} examples", flush=True)

    # Load model
    from cacheblend import LayerwiseModel
    t0 = time.time()
    lw_model = LayerwiseModel(args.model, dtype="float16", attn_implementation=args.attn_impl)
    print(f"loaded model in {time.time() - t0:.1f}s", flush=True)
    model = lw_model.model
    tokenizer = lw_model.tokenizer

    out_fp = open(results_path, "w")
    n_done = 0
    n_skipped = 0
    t_phase = time.time()

    METHODS = ["FullRecompute", "FullReuse", "CacheBlend"]

    for i, ex in enumerate(examples):
        token_id_chunks, text_chunks, golds, _ = _build_chunks_for_example(tokenizer, ex, cfg)
        if token_id_chunks is None:
            n_skipped += 1
            continue
        chunks = _make_chunks_from_ids(token_id_chunks, text_chunks)
        # Compute chunk boundaries in fused-sequence positions (for HKVD enrichment)
        from cacheblend.chunker import chunk_offsets
        offsets = chunk_offsets(chunks)
        total_tokens = offsets[-1][1]
        chunk_boundaries = [end for (_s, end) in offsets[:-1]]  # interior
        prefix_end = offsets[0][1]
        query_start = offsets[-1][0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Populate KV cache (per-chunk standalone, matches official's collect=True hack)
        kv_store_cpu = _populate_kv_store(lw_model, chunks)

        per_method_res = {}
        for method in METHODS:
            try:
                if method == "FullRecompute":
                    res = _run_method(lw_model, model, tokenizer, chunks, None, method,
                                       cfg["max_new_tokens"], ratio, args.cb_check_layer)
                else:
                    gpu_store = _store_to_gpu(kv_store_cpu, lw_model.device)
                    try:
                        capture = (method == "CacheBlend")
                        res = _run_method(lw_model, model, tokenizer, chunks, gpu_store, method,
                                           cfg["max_new_tokens"], ratio, args.cb_check_layer,
                                           capture_hkvd=capture)
                    finally:
                        gpu_store.clear()
                        del gpu_store
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                per_method_res[method] = res
            except Exception as e:
                print(f"  ERROR ex{i} {method}: {type(e).__name__}: {e}", flush=True)
                per_method_res[method] = None

        # Compute metric per method
        for method, res in per_method_res.items():
            if res is None:
                continue
            pred = res["text"]
            # Strip first-line (samsum policy per official); F1 path uses parse_generation
            if cfg["metric"] == "rouge_l":
                pred_clean = pred.lstrip("\n").split("\n")[0]
                try:
                    score = max(compute_rl(pred_clean, g) for g in golds)
                except Exception:
                    score = 0.0
            else:
                try:
                    score = max(compute_f1(pred, g, tokenizer) for g in golds)
                except Exception:
                    score = 0.0
            row = {
                "ex_idx": i,
                "method": method,
                "pred": pred,
                "gold_first": (golds[0] if golds else "")[:200],
                "score": score,
                "ttft_seconds": res["ttft_seconds"],
                "total_seconds": res["total_seconds"],
                "n_generated_tokens": res["n_generated_tokens"],
                "n_chunks": len(chunks),
                "total_tokens": total_tokens,
                "chunk_boundaries": chunk_boundaries,
                "prefix_end": prefix_end,
                "query_start": query_start,
            }
            if "hkvd_indices" in res:
                row["hkvd_indices"] = res["hkvd_indices"]
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1

        kv_store_cpu.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i + 1 == len(examples):
            out_fp.flush()
            elapsed = time.time() - t_phase
            print(f"  [{i+1}/{len(examples)}] elapsed={elapsed:.0f}s n_done={n_done}", flush=True)

    out_fp.close()
    print(f"done: {n_done} rows, {n_skipped} skipped, {time.time() - t_phase:.0f}s", flush=True)

    # Summary
    summary = _build_summary(results_path, ratio, cfg["metric"])
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== summary ===\n{json.dumps(summary, indent=2)}", flush=True)


def _build_summary(results_path: Path, ratio: float, metric: str) -> dict:
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    by_method = {}
    by_id_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
        by_id_method[(r["ex_idx"], r["method"])] = r

    per_method = {}
    for m, rs in by_method.items():
        scores = [r["score"] for r in rs]
        ttfts = [r["ttft_seconds"] for r in rs]
        per_method[m] = {
            "n": len(rs),
            f"{metric}_mean": statistics.mean(scores),
            f"{metric}_median": statistics.median(scores),
            f"{metric}_zero_count": sum(1 for x in scores if x == 0.0),
            f"{metric}_one_count": sum(1 for x in scores if x >= 0.999),
            "ttft_seconds_mean": statistics.mean(ttfts),
            "ttft_seconds_median": statistics.median(ttfts),
        }

    summary = {
        "dataset_metric": metric,
        "cacheblend_ratio": ratio,
        "per_method": per_method,
    }

    # Comparison gaps
    def f(m, k=f"{metric}_mean"):
        return per_method.get(m, {}).get(k, 0.0)
    summary["comparisons"] = {
        "cb_minus_full_reuse": f("CacheBlend") - f("FullReuse"),
        "cb_minus_full_recompute": f("CacheBlend") - f("FullRecompute"),
        "full_reuse_minus_full_recompute": f("FullReuse") - f("FullRecompute"),
    }

    # Paired bootstrap CI (CB vs FullReuse)
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    common = sorted({k[0] for k in by_id_method if k[1] == "CacheBlend"} &
                    {k[0] for k in by_id_method if k[1] == "FullReuse"})
    if len(common) >= 2:
        a = [by_id_method[(i, "CacheBlend")]["score"] for i in common]
        b = [by_id_method[(i, "FullReuse")]["score"] for i in common]
        lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=1000, confidence=0.95, seed=42)
        summary["ci_low_cb_vs_reuse"] = lo
        summary["ci_high_cb_vs_reuse"] = hi
        summary["n_paired"] = len(common)

    # Failure subset: FullReuse < FullRecompute
    fail_ids = [i for i in {r["ex_idx"] for r in rows}
                if (by_id_method.get((i, "FullReuse"), {}).get("score", 0.0)
                    < by_id_method.get((i, "FullRecompute"), {}).get("score", 0.0))]
    fail_summary = {"n_failure": len(fail_ids), "per_method": {}}
    for m in by_method:
        scores = [by_id_method[(i, m)]["score"] for i in fail_ids if (i, m) in by_id_method]
        fail_summary["per_method"][m] = {
            "n": len(scores),
            f"{metric}_mean": statistics.mean(scores) if scores else 0.0,
        }
    summary["failure_subset"] = fail_summary

    # HKVD boundary enrichment (chunks with HKVD captured = CacheBlend rows)
    cb_with_hkvd = [r for r in rows if r["method"] == "CacheBlend" and "hkvd_indices" in r]
    if cb_with_hkvd:
        windows = [1, 3, 8]
        agg = {f"pm{W}": {"hkvd_in_window": 0, "hkvd_total": 0,
                          "all_in_window": 0, "all_total": 0} for W in windows}
        for r in cb_with_hkvd:
            hkvd = set(r["hkvd_indices"])
            boundaries = r["chunk_boundaries"]
            total = r["total_tokens"]
            for W in windows:
                in_w = set()
                for b in boundaries:
                    for off in range(-W, W + 1):
                        p = b + off
                        if 0 <= p < total:
                            in_w.add(p)
                agg[f"pm{W}"]["hkvd_in_window"] += len(hkvd & in_w)
                agg[f"pm{W}"]["hkvd_total"] += len(hkvd)
                agg[f"pm{W}"]["all_in_window"] += len(in_w)
                agg[f"pm{W}"]["all_total"] += total
        summary["hkvd_boundary_enrichment"] = {}
        for W in windows:
            a = agg[f"pm{W}"]
            hkvd_frac = a["hkvd_in_window"] / max(a["hkvd_total"], 1)
            all_frac = a["all_in_window"] / max(a["all_total"], 1)
            summary["hkvd_boundary_enrichment"][f"pm{W}"] = {
                "hkvd_fraction_in_window": hkvd_frac,
                "all_fraction_in_window": all_frac,
                "enrichment_ratio": hkvd_frac / max(all_frac, 1e-9),
            }

    return summary


if __name__ == "__main__":
    main()
