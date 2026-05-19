"""Loong cache-then-reverse experiment driver.

Protocol summary (user-provided 2026-05):
  For each example:
    1. Build cache prompt:  prefix + SEP + chunk_1 + SEP + ... + chunk_11 + SEP + dummy_query
    2. Build eval prompt:   prefix + SEP + chunk_11 + SEP + ... + chunk_1 + SEP + real_question
    3. Cache phase: run forward over cache prompt to populate per-chunk KV cache
       (LMCache-style: chunks get cross-chunk context during caching).
    4. Eval phase: run 3 methods on eval prompt (reversed chunk order):
         - FullRecompute (no cache reuse) — quality reference
         - FullReuse    (reuse cached chunks, no HKVD recompute)
         - CacheBlend   (reuse cached + HKVD selective recompute, ratio sweep)
  Compare F1 + prefill latency.

This driver adapts the protocol to our v4 code:
  - Per-chunk KV stored via `precompute_from_cache_prompt` (full forward on cache
    prompt, slice per chunk's pre-RoPE K + V). Equivalent to LMCache's
    cache-population since RoPE is invertible.
  - SEP "# #" embedded as trailing text in each chunk's text. Our chunker tracks
    chunks explicitly (no token-level SEP matching needed).
  - Real question is a separate "chunk" not in cache (cache miss). For FullReuse
    and CacheBlend, the cached chunks form a prefix; the question chunk runs
    fresh on top via standard HF model continuation forward.
  - Doc truncation: each Loong doc is truncated to `--doc-tokens` tokens so the
    total prompt fits the model's max_model_len.

Usage:
  python benchmarks/run_loong.py \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --n 50 \\
      --doc-tokens 5000 \\
      --max-model-len 131072 \\
      --ratios 0.0,0.15,1.0 \\
      --out reports/loong-llama-50
"""
from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from pathlib import Path

import torch


SEP = "# #"
DEFAULT_PREFIX = (
    "You are a question-answering assistant. Use the provided passages to "
    "answer the final question. Answer with only the final answer. Use the "
    "shortest possible phrase. Do not explain."
)
DUMMY_QUERY = "This is a cache warmup query. Do not answer."


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "external/mydata/cacheblend_fig12"))
sys.path.insert(0, str(ROOT))


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens (right-truncate)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _build_chunk_texts(
    tokenizer, ex, doc_tokens: int, prefix: str = DEFAULT_PREFIX,
):
    """Build per-chunk text strings for the cache and eval prompts.

    Returns (prefix_chunk_text, doc_chunk_texts_forward, dummy_chunk_text,
             question_chunk_text). All include trailing SEP as part of their text
    (except the trailing terminus chunks: dummy_query and question, which are
    at the end).
    """
    prefix_chunk = f"{prefix}\n{SEP}\n"
    doc_chunks = []
    for i, doc_text in enumerate(ex.docs_text):
        truncated = _truncate_to_tokens(tokenizer, doc_text, doc_tokens)
        doc_chunks.append(f"{truncated}\n{SEP}\n")
    dummy_chunk = f"{DUMMY_QUERY}"
    question_chunk = f"Question: {ex.question}\n\nAnswer:"
    return prefix_chunk, doc_chunks, dummy_chunk, question_chunk


def _make_chunks(tokenizer, texts):
    """Tokenize a list of texts and wrap as cacheblend Chunk objects."""
    from cacheblend.chunker import Chunk
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        from cacheblend.chunker import _stable_id
        out.append(Chunk(text=t, token_ids=ids, chunk_id=_stable_id(t, ids)))
    return out


def _fit_check(chunks, max_prompt_tokens: int) -> int:
    total = sum(len(c.token_ids) for c in chunks)
    return total <= max_prompt_tokens, total


def _greedy_decode_continuation(model, tokenizer, last_logits, past_kv,
                                max_new_tokens: int, t_start: float, device):
    """Greedy decode given the last position's logits and a past_key_values."""
    eos = getattr(tokenizer, "eos_token_id", None)
    with torch.inference_mode():
        next_id = last_logits.argmax().unsqueeze(0).unsqueeze(0)
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
    del past_kv, out, next_id, last_logits
    return {
        "text": text,
        "ttft_seconds": t_first - t_start,
        "total_seconds": t_end - t_start,
        "n_generated_tokens": len(generated),
    }


def _question_continuation(lw_model, model, tokenizer, question_chunk,
                            past_kv, max_new_tokens, t_start):
    """Run forward on question_chunk with past_kv (cached prefix+docs), then
    greedy decode. Used by FullReuse and CacheBlend variants."""
    device = lw_model.device
    q_ids = torch.tensor([question_chunk.token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        q_out = model(input_ids=q_ids, past_key_values=past_kv, use_cache=True)
    last_logits = q_out.logits[0, -1]
    return _greedy_decode_continuation(
        model, tokenizer, last_logits, q_out.past_key_values,
        max_new_tokens, t_start, device,
    )


# ── Method runners ────────────────────────────────────────────────────────────

def _run_full_recompute(lw_model, model, tokenizer, eval_chunks_all, max_new_tokens):
    from cacheblend.fusor import fuse_full_recompute
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    out = fuse_full_recompute(lw_model, eval_chunks_all, return_layerwise_output=True)
    return _greedy_decode_continuation(
        model, tokenizer, out.logits[0, -1], out.past_key_values,
        max_new_tokens, t_start, lw_model.device,
    )


def _run_full_reuse(lw_model, model, tokenizer, cached_chunks, question_chunk,
                    kv_store, max_new_tokens):
    from cacheblend.fusor import fuse_full_reuse
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    cached_out = fuse_full_reuse(
        lw_model, cached_chunks, kv_store, return_layerwise_output=True,
    )
    return _question_continuation(
        lw_model, model, tokenizer, question_chunk,
        cached_out.past_key_values, max_new_tokens, t_start,
    )


def _run_cacheblend(lw_model, model, tokenizer, cached_chunks, question_chunk,
                     kv_store, ratio, check_layer, max_new_tokens,
                     capture_hkvd: bool = False):
    """Run CacheBlend selective recompute on cached_chunks + question continuation.

    If capture_hkvd=True, also return the HKVD-selected token indices (for the
    boundary-enrichment analysis required by the protocol).
    """
    from cacheblend.fusor import fuse_selective
    if lw_model.device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    if capture_hkvd:
        cached_out, hkvd_indices = fuse_selective(
            lw_model, cached_chunks, kv_store,
            recompute_ratio=ratio, check_layer=check_layer,
            return_layerwise_output=True, return_hkvd_indices=True,
        )
    else:
        cached_out = fuse_selective(
            lw_model, cached_chunks, kv_store,
            recompute_ratio=ratio, check_layer=check_layer,
            return_layerwise_output=True,
        )
        hkvd_indices = None
    res = _question_continuation(
        lw_model, model, tokenizer, question_chunk,
        cached_out.past_key_values, max_new_tokens, t_start,
    )
    if capture_hkvd:
        res["hkvd_indices"] = hkvd_indices.detach().cpu().tolist()
    return res


# ── F1 against Loong answer ───────────────────────────────────────────────────

def _format_gold(answer_raw):
    """Loong answer format: JSON {"Reference":[...], "Citation":[...]}.
    Concatenate Reference list as primary gold; Citation as alias if present.
    """
    try:
        d = json.loads(answer_raw)
        refs = d.get("Reference", []) if isinstance(d, dict) else []
        cits = d.get("Citation", []) if isinstance(d, dict) else []
        golds = []
        if refs:
            golds.append(" ".join(refs))
        if cits:
            golds.append(" ".join(cits))
        return golds or [answer_raw]
    except Exception:
        return [answer_raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--doc-tokens", type=int, default=5000,
                    help="Truncate each Loong doc to this many tokens")
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--safety-margin", type=int, default=128)
    ap.add_argument("--cb-check-layer", type=int, default=1)
    ap.add_argument("--ratios", type=str, default="0.0,0.15,1.0",
                    help="Comma-separated CacheBlend ratios")
    ap.add_argument("--split", default="paper", choices=["paper", "financial"])
    ap.add_argument("--attn-impl", default="flash_attention_2",
                    help="HF attn_implementation for the LayerwiseModel")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    # Load metrics + dataset
    from harness.metrics import compute_f1_against_aliases, compute_rouge_l
    from benchmarks.data.loong import load_loong

    examples = load_loong(split=args.split, min_docs=11, max_docs=11, n=args.n)
    print(f"loaded {len(examples)} Loong examples (split={args.split}, 11-doc)", flush=True)

    # Load model
    print(f"loading {args.model} (attn={args.attn_impl})", flush=True)
    from cacheblend import LayerwiseModel
    t0 = time.time()
    lw_model = LayerwiseModel(args.model, dtype="float16", attn_implementation=args.attn_impl)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    model = lw_model.model
    tokenizer = lw_model.tokenizer

    prompt_budget = args.max_model_len - args.max_new_tokens - args.safety_margin
    print(f"prompt budget: {prompt_budget} tokens", flush=True)

    ratios = [float(r) for r in args.ratios.split(",")]
    print(f"ratios: {ratios}", flush=True)

    out_fp = open(results_path, "w")
    n_done = 0
    n_skipped = 0
    t_phase = time.time()

    for i, ex in enumerate(examples):
        # Build chunk texts
        prefix_text, doc_texts_fwd, dummy_text, q_text = _build_chunk_texts(
            tokenizer, ex, doc_tokens=args.doc_tokens,
        )

        # Cache prompt chunks: [prefix, doc_1, ..., doc_11, dummy_query]
        cache_chunks = _make_chunks(tokenizer, [prefix_text] + doc_texts_fwd + [dummy_text])

        # Eval prompt chunks: [prefix, doc_11, ..., doc_1, question]
        doc_texts_rev = list(reversed(doc_texts_fwd))
        eval_chunks_full = _make_chunks(tokenizer, [prefix_text] + doc_texts_rev + [q_text])

        # Length budget check
        cache_total = sum(len(c.token_ids) for c in cache_chunks)
        eval_total = sum(len(c.token_ids) for c in eval_chunks_full)
        if cache_total > prompt_budget or eval_total > prompt_budget:
            n_skipped += 1
            print(f"  SKIP ex{i} id={ex.id[:12]} cache_tok={cache_total} eval_tok={eval_total} > budget", flush=True)
            continue

        golds = _format_gold(ex.answer)

        # Cache population
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from cacheblend.precompute import precompute_from_cache_prompt
        dummy_id = cache_chunks[-1].chunk_id
        try:
            kv_store = precompute_from_cache_prompt(
                lw_model, cache_chunks, skip_chunk_ids={dummy_id},
            )
        except Exception as e:
            print(f"  ERROR ex{i} cache phase: {type(e).__name__}: {e}", flush=True)
            n_skipped += 1
            continue

        # Sanity: real question chunk_id MUST NOT be in cache store (cache miss expected)
        question_chunk = eval_chunks_full[-1]
        cached_chunks_for_eval = eval_chunks_full[:-1]  # prefix + reversed docs
        question_cache_hit = kv_store.has(question_chunk.chunk_id)
        if question_cache_hit:
            print(f"  WARN ex{i} REAL QUESTION CACHE HIT — experimental design warning", flush=True)

        # Run 3 methods + ratio sweep (CB at each ratio)
        method_results = {}

        # 1. FullRecompute (no cache)
        try:
            res = _run_full_recompute(lw_model, model, tokenizer, eval_chunks_full, args.max_new_tokens)
            method_results["FullRecompute"] = res
        except Exception as e:
            print(f"  ERROR ex{i} FullRecompute: {type(e).__name__}: {e}", flush=True)
            method_results["FullRecompute"] = None

        # 2. FullReuse
        try:
            res = _run_full_reuse(lw_model, model, tokenizer, cached_chunks_for_eval,
                                   question_chunk, kv_store, args.max_new_tokens)
            method_results["FullReuse"] = res
        except Exception as e:
            print(f"  ERROR ex{i} FullReuse: {type(e).__name__}: {e}", flush=True)
            method_results["FullReuse"] = None

        # 3. CacheBlend at each ratio. Capture HKVD for r=0.15 (main study ratio).
        for r in ratios:
            tag = f"CacheBlend_r{r:.2f}"
            capture = abs(r - 0.15) < 1e-9
            try:
                res = _run_cacheblend(lw_model, model, tokenizer,
                                       cached_chunks_for_eval, question_chunk, kv_store,
                                       ratio=r, check_layer=args.cb_check_layer,
                                       max_new_tokens=args.max_new_tokens,
                                       capture_hkvd=capture)
                method_results[tag] = res
            except Exception as e:
                print(f"  ERROR ex{i} {tag}: {type(e).__name__}: {e}", flush=True)
                method_results[tag] = None

        # Compute F1 + Rouge-L per method, write rows
        for method_name, res in method_results.items():
            if res is None:
                continue
            try:
                f1 = max(compute_f1_against_aliases(res["text"], [g], tokenizer) for g in golds)
            except Exception:
                f1 = 0.0
            try:
                rl = max(compute_rouge_l(res["text"], g) for g in golds)
            except Exception:
                rl = 0.0
            # Compute chunk boundary positions in the eval prompt (relative to the
            # cached_chunks_for_eval = [prefix, doc_11..doc_1]; question is appended
            # separately so HKVD lives in the cached-region indices 0..cached_len-1).
            from cacheblend.chunker import chunk_offsets as _cho
            cached_offsets = _cho(cached_chunks_for_eval)
            chunk_boundaries = [end for (_s, end) in cached_offsets[:-1]]  # interior boundaries

            row = {
                "id": ex.id,
                "method": method_name,
                "pred": res["text"],
                "golds_first": golds[0][:200],
                "f1": f1,
                "rouge_l": rl,
                "ttft_seconds": res["ttft_seconds"],
                "total_seconds": res["total_seconds"],
                "n_generated_tokens": res["n_generated_tokens"],
                "cache_prompt_tokens": cache_total,
                "eval_prompt_tokens": eval_total,
                "cached_region_tokens": cached_offsets[-1][1],
                "chunk_boundaries": chunk_boundaries,
                "question_cache_hit": question_cache_hit,
                "loong_length": ex.length,
            }
            if "hkvd_indices" in res:
                row["hkvd_indices"] = res["hkvd_indices"]
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1

        # Cleanup
        kv_store.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out_fp.flush()
        print(f"  [{i + 1}/{len(examples)}] ex_id={ex.id[:12]} cache_tok={cache_total} eval_tok={eval_total} F1s=" +
              ",".join(f"{m}={method_results.get(m,{}).get('text','ERR')[:25] if method_results.get(m) else 'ERR'}"
                       for m in ["FullRecompute", "FullReuse"] + [f"CacheBlend_r{r:.2f}" for r in ratios]),
              flush=True)

    out_fp.close()
    print(f"done eval: {n_done} rows, {n_skipped} skipped in {time.time() - t_phase:.1f}s", flush=True)

    # Summary
    summary = _build_summary(results_path, ratios)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== summary ===\n{json.dumps(summary, indent=2)}\n", flush=True)


def _build_summary(results_path: Path, ratios: list[float]) -> dict:
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    per_method = {}
    for m, rs in by_method.items():
        f1s = [r["f1"] for r in rs]
        rls = [r["rouge_l"] for r in rs]
        ttfts = [r["ttft_seconds"] for r in rs]
        per_method[m] = {
            "n": len(rs),
            "f1_mean": statistics.mean(f1s),
            "f1_median": statistics.median(f1s),
            "rouge_l_mean": statistics.mean(rls),
            "ttft_seconds_mean": statistics.mean(ttfts),
            "ttft_seconds_median": statistics.median(ttfts),
            "f1_zero_count": sum(1 for x in f1s if x == 0.0),
            "f1_one_count": sum(1 for x in f1s if x >= 0.999),
        }

    summary = {"per_method": per_method}

    # Required comparisons (per protocol)
    def f1_of(m):
        return per_method.get(m, {}).get("f1_mean", 0.0)
    summary["comparisons"] = {
        "cb_r0.15_minus_full_reuse": f1_of("CacheBlend_r0.15") - f1_of("FullReuse"),
        "cb_r0.15_minus_full_recompute": f1_of("CacheBlend_r0.15") - f1_of("FullRecompute"),
        "full_reuse_minus_full_recompute": f1_of("FullReuse") - f1_of("FullRecompute"),
        "cb_r1.00_minus_full_recompute": f1_of("CacheBlend_r1.00") - f1_of("FullRecompute"),
        "cb_r0.00_minus_full_reuse": f1_of("CacheBlend_r0.00") - f1_of("FullReuse"),
    }

    # Paired bootstrap CI: CB r=0.15 vs FullReuse
    from benchmarks.metrics.bootstrap import paired_bootstrap_ci
    by_id_method = {}
    for r in rows:
        by_id_method[(r["id"], r["method"])] = r
    common = sorted({k[0] for k in by_id_method if k[1] == "CacheBlend_r0.15"} &
                    {k[0] for k in by_id_method if k[1] == "FullReuse"})
    if len(common) >= 2:
        a = [by_id_method[(i, "CacheBlend_r0.15")]["f1"] for i in common]
        b = [by_id_method[(i, "FullReuse")]["f1"] for i in common]
        lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=1000, confidence=0.95, seed=42)
        summary["ci_low_cb_vs_reuse"] = lo
        summary["ci_high_cb_vs_reuse"] = hi
        summary["n_paired"] = len(common)

    # Failure-only subset: examples where FullReuse F1 < FullRecompute F1
    fail_ids = [i for i in {r["id"] for r in rows}
                if (by_id_method.get((i, "FullReuse"), {}).get("f1", 0.0)
                    < by_id_method.get((i, "FullRecompute"), {}).get("f1", 0.0))]
    fail_subset = {}
    for m in by_method:
        sub_f1s = [by_id_method[(i, m)]["f1"] for i in fail_ids if (i, m) in by_id_method]
        fail_subset[m] = {"n": len(sub_f1s),
                          "f1_mean": statistics.mean(sub_f1s) if sub_f1s else 0.0}
    summary["failure_subset"] = {
        "n_failure_examples": len(fail_ids),
        "per_method": fail_subset,
    }

    # Prompt length buckets
    buckets = [(0, 8000), (8000, 16000), (16000, 24000), (24000, 32000),
               (32000, 64000), (64000, 128000)]
    bucket_summary = []
    for lo_t, hi_t in buckets:
        ids_in_bucket = sorted({r["id"] for r in rows
                                if lo_t <= r["eval_prompt_tokens"] < hi_t})
        if not ids_in_bucket:
            continue
        per_method_bucket = {}
        for m in by_method:
            f1s = [by_id_method[(i, m)]["f1"] for i in ids_in_bucket if (i, m) in by_id_method]
            ttfts = [by_id_method[(i, m)]["ttft_seconds"] for i in ids_in_bucket if (i, m) in by_id_method]
            per_method_bucket[m] = {
                "f1_mean": statistics.mean(f1s) if f1s else 0.0,
                "ttft_seconds_mean": statistics.mean(ttfts) if ttfts else 0.0,
            }
        bucket_summary.append({
            "range_tokens": f"{lo_t}-{hi_t}",
            "n_examples": len(ids_in_bucket),
            "per_method": per_method_bucket,
        })
    summary["prompt_length_buckets"] = bucket_summary

    # Cache-hit diagnostics summary
    any_q_hit = sum(1 for r in rows if r.get("question_cache_hit") and r["method"] == "FullRecompute")
    summary["cache_diagnostics"] = {
        "examples_with_question_cache_hit": any_q_hit,
        "note": "0 expected. >0 indicates experimental-design warning (real question cached).",
    }

    # HKVD boundary enrichment analysis (user's hypothesis: HKVD concentrates near chunk boundaries)
    cb_rows_with_hkvd = [r for r in rows if r["method"] == "CacheBlend_r0.15" and "hkvd_indices" in r]
    if cb_rows_with_hkvd:
        # For each example: count HKVD tokens within ±W of any chunk boundary,
        # compare with the fraction of ALL tokens within that window (enrichment).
        windows = [1, 3, 8]
        agg = {f"pm{W}": {"hkvd_in_window": 0, "hkvd_total": 0,
                          "all_in_window": 0, "all_total": 0} for W in windows}
        per_example_pct = {f"pm{W}": [] for W in windows}
        for r in cb_rows_with_hkvd:
            hkvd = set(r["hkvd_indices"])
            boundaries = r["chunk_boundaries"]  # interior boundaries (between chunks)
            cached_len = r["cached_region_tokens"]
            all_positions = set(range(cached_len))
            for W in windows:
                in_window = set()
                for b in boundaries:
                    # ±W tokens around boundary position b (b is the start of next chunk
                    # = end of prev chunk; we include the W tokens on each side)
                    for off in range(-W, W + 1):
                        p = b + off
                        if 0 <= p < cached_len:
                            in_window.add(p)
                hkvd_in_w = len(hkvd & in_window)
                all_in_w = len(in_window)
                agg[f"pm{W}"]["hkvd_in_window"] += hkvd_in_w
                agg[f"pm{W}"]["hkvd_total"] += len(hkvd)
                agg[f"pm{W}"]["all_in_window"] += all_in_w
                agg[f"pm{W}"]["all_total"] += cached_len
                if len(hkvd) > 0:
                    per_example_pct[f"pm{W}"].append(hkvd_in_w / len(hkvd))
        summary["hkvd_boundary_enrichment"] = {}
        for W in windows:
            a = agg[f"pm{W}"]
            hkvd_frac = a["hkvd_in_window"] / max(a["hkvd_total"], 1)
            all_frac = a["all_in_window"] / max(a["all_total"], 1)
            enrichment = hkvd_frac / max(all_frac, 1e-9)
            summary["hkvd_boundary_enrichment"][f"pm{W}"] = {
                "hkvd_fraction_in_window": hkvd_frac,
                "all_fraction_in_window": all_frac,
                "enrichment_ratio": enrichment,
                "n_examples": len(per_example_pct[f"pm{W}"]),
                "per_example_pct_median": (
                    statistics.median(per_example_pct[f"pm{W}"])
                    if per_example_pct[f"pm{W}"] else 0.0
                ),
            }

    return summary


if __name__ == "__main__":
    main()
