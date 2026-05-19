"""Phase 8 Step 1 — per-layer KV deviation profiling.

Metric (a) Top-15% mass: 토큰별 |KV deviation| 정렬 시 top 15% 토큰이
전체 |deviation| mass 의 어느 비율을 차지하는가. layer 별로 측정.
(b)/(c) 는 사용자 검토 ① 결과 따라 후속 진행.

Per model:
  - mydata cacheblend_fig12 prompts.jsonl 첫 N sample.
  - 각 sample: build chunks → populate KVStore → fused forward.
  - LayerwiseModel 의 k_proj forward-hook 이 각 layer 의 fresh pre-RoPE K capture.
  - K_stored (cached pre-RoPE) 와 비교 → per-layer per-token deviation.
  - top-15% mass per layer = sum(top-15% |dev|) / sum(all |dev|).
  - sample 별 → per-layer median 집계.

Outputs:
  reports/phase-8-step1-attachments/{model_short}_profile_a.json
  reports/phase-8-step1-attachments/{model_short}_top15pct_mass.png

Usage:
  python benchmarks/phase8_step1_profile.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 --short mistral \
    --n 15 --out-dir reports/phase-8-step1-attachments

Cost-aware: 1 fused forward × N sample. Mistral-7B 24GB GPU 기준 약 2-3 min for n=15.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


HARNESS_PATH = Path(__file__).resolve().parent.parent / "external/mydata/cacheblend_fig12"
PROMPTS_PATH = HARNESS_PATH / "prompts.jsonl"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_chunks(tokenizer, ex):
    from cacheblend.chunker import chunk_texts
    parts = [ex["prompt_parts"]["system"]]
    for i, d in enumerate(ex["prompt_parts"]["docs"]):
        parts.append(f"\n\nDocument {i + 1}:\n{d}")
    parts.append(f"\n\nQuestion: {ex['prompt_parts']['question']}\nAnswer:")
    return chunk_texts(tokenizer, parts)


def _populate_store_cpu(lw_model, chunks):
    """Precompute pre-RoPE K + V per chunk, offload to CPU. Returns CPU-resident KVStore."""
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw_model, c)
        K_cpu = [k.detach().cpu() for k in K]
        V_cpu = [v.detach().cpu() for v in V]
        store.put(c.chunk_id, K_cpu, V_cpu)
        del K, V
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return store


def _build_concat_K_stored(store, chunks, n_layers, hidden_kv, total_seq, device, dtype):
    """Concat per-chunk stored pre-RoPE K into a (1, total_seq, hidden_kv) tensor per layer."""
    from cacheblend.chunker import chunk_offsets
    offsets = chunk_offsets(chunks)
    K_stored = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    for chunk, (start, end) in zip(chunks, offsets):
        entry = store._cache[chunk.chunk_id]
        for li in range(n_layers):
            K_stored[li][:, start:end, :] = entry["K"][li].to(device, non_blocking=True)
    return K_stored


def _profile_one_sample(lw_model, ex, device):
    """One sample → per-layer top-15% mass.

    Returns:
      list[float] of length n_layers, each in [0, 1].
      total_seq (int) for context.
    """
    from cacheblend.fusor import fuse_full_recompute
    from cacheblend.chunker import fused_input_ids

    chunks = _build_chunks(lw_model.tokenizer, ex)
    total_seq = sum(c.length for c in chunks)
    n_layers = lw_model.num_layers
    inner = lw_model._inner
    attn0 = inner.layers[0].self_attn
    hidden_kv = attn0.config.num_key_value_heads * attn0.head_dim

    # Populate CPU store
    cpu_store = _populate_store_cpu(lw_model, chunks)

    # Build K_stored on GPU just for this sample
    K_stored_gpu = _build_concat_K_stored(
        cpu_store, chunks, n_layers, hidden_kv, total_seq,
        device=device, dtype=lw_model.dtype,
    )

    # Fused forward — Phase 1 hooks capture per-layer fresh pre-RoPE K
    # into lw_model._pre_rope_k.
    with torch.inference_mode():
        _ = fuse_full_recompute(lw_model, chunks, return_layerwise_output=True)

    # Per-layer top-15% mass
    out = []
    for li in range(n_layers):
        K_fresh = lw_model.get_pre_rope_k(li)             # (1, total_seq, hidden_kv)
        K_stor = K_stored_gpu[li]                          # (1, total_seq, hidden_kv)
        dev = ((K_fresh.float() - K_stor.float()) ** 2).sum(dim=-1).squeeze(0)  # (total_seq,)
        sorted_dev, _ = dev.sort(descending=True)
        top_n = max(1, int(0.15 * total_seq))
        top_mass = sorted_dev[:top_n].sum().item()
        total_mass = sorted_dev.sum().item()
        out.append(top_mass / max(total_mass, 1e-12))

    # Cleanup
    del K_stored_gpu, cpu_store
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out, total_seq


def _gap_analysis(scores: list[float], threshold: float = 0.30, gap_threshold: float = 0.10):
    """Return candidate check_layers via gap analysis (task spec §Step 1).

    Returns dict with:
      significant_layers (sorted by score desc)
      gaps (top 3 gaps among significant)
      candidates_1check / 2check / 3check
      recommendation (1/2/3-check + selected layers)
    """
    significant = [(li, s) for li, s in enumerate(scores) if s >= threshold]
    significant.sort(key=lambda x: x[1], reverse=True)

    if not significant:
        return {
            "significant_layers": [],
            "gaps": [],
            "recommendation": "no_layer_above_threshold",
            "selected_layers": [],
            "n_check": 0,
        }

    sorted_layers = [li for li, _ in significant]
    sorted_scores = [s for _, s in significant]

    gaps = []
    for i in range(min(3, len(sorted_scores) - 1)):
        gaps.append(sorted_scores[i] - sorted_scores[i + 1])

    if not gaps:
        # Only 1 significant layer
        return {
            "significant_layers": [{"layer": li, "score": s} for li, s in significant],
            "gaps": [],
            "recommendation": "1-check (only one significant layer)",
            "selected_layers": [sorted_layers[0]],
            "n_check": 1,
        }

    # Decide n_check
    if gaps[0] >= gap_threshold:
        n_check, selected = 1, sorted_layers[:1]
    elif len(gaps) >= 2 and gaps[1] >= gap_threshold:
        n_check, selected = 2, sorted_layers[:2]
    else:
        n_check, selected = 3, sorted_layers[: min(3, len(sorted_layers))]

    return {
        "significant_layers": [{"layer": li, "score": s} for li, s in significant],
        "gaps": gaps,
        "recommendation": f"{n_check}-check (gap analysis)",
        "selected_layers": sorted(selected),
        "n_check": n_check,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name")
    ap.add_argument("--short", required=True, help="short label for output filenames (e.g. 'mistral', 'llama8b')")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out-dir", default="reports/phase-8-step1-attachments")
    ap.add_argument("--threshold-a", type=float, default=0.30)
    ap.add_argument("--gap-threshold", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model}", flush=True)
    from cacheblend import LayerwiseModel
    t0 = time.time()
    lw_model = LayerwiseModel(args.model, dtype="float16")
    print(f"  loaded LayerwiseModel in {time.time()-t0:.1f}s", flush=True)
    device = lw_model.device

    # Load N samples
    examples = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples = examples[: args.n]
    print(f"profiling {len(examples)} samples", flush=True)

    n_layers = lw_model.num_layers
    per_sample = []  # list of list[float]; per_sample[i][layer] = top-15% mass
    seqs = []

    t_phase = time.time()
    for i, ex in enumerate(examples):
        try:
            scores, total_seq = _profile_one_sample(lw_model, ex, device)
            per_sample.append(scores)
            seqs.append(total_seq)
            if (i + 1) % 5 == 0 or i + 1 == len(examples):
                print(f"  [{i+1}/{len(examples)}] total_seq={total_seq}", flush=True)
        except Exception as e:
            print(f"  ERROR sample {i} ({ex['id']}): {type(e).__name__}: {e}", flush=True)
    print(f"profiled {len(per_sample)} samples in {time.time()-t_phase:.1f}s", flush=True)

    if not per_sample:
        print("no samples succeeded", flush=True)
        sys.exit(1)

    # Aggregate: per-layer median across samples
    import statistics
    median_scores = []
    for li in range(n_layers):
        col = [row[li] for row in per_sample]
        median_scores.append(statistics.median(col))

    # Gap analysis
    decision = _gap_analysis(median_scores, threshold=args.threshold_a, gap_threshold=args.gap_threshold)

    profile = {
        "model": args.model,
        "short": args.short,
        "n_samples": len(per_sample),
        "n_layers": n_layers,
        "total_seqs": seqs,
        "per_layer_median_top15pct_mass": median_scores,
        "per_sample": per_sample,  # raw for later inspection
        "threshold_a": args.threshold_a,
        "gap_threshold": args.gap_threshold,
        "decision": decision,
    }
    json_path = out_dir / f"{args.short}_profile_a.json"
    json_path.write_text(json.dumps(profile, indent=2))
    print(f"\nWrote: {json_path}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.5))
        layers = list(range(n_layers))
        bar_colors = [
            "#1a7f37" if s >= args.threshold_a else "#9aa0a6"
            for s in median_scores
        ]
        ax.bar(layers, median_scores, color=bar_colors)
        ax.axhline(y=args.threshold_a, color="#cf222e", linestyle="--",
                   linewidth=1, label=f"threshold = {args.threshold_a}")
        for li in decision["selected_layers"]:
            ax.scatter([li], [median_scores[li]], color="#0969da", s=100,
                       zorder=5, marker="*",
                       label="selected" if li == decision["selected_layers"][0] else None)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Top-15% mass (median over samples)")
        ax.set_title(
            f"{args.short.title()} — Top-15% KV Deviation Mass per Layer "
            f"(n={len(per_sample)}, decision: {decision.get('recommendation', '?')})"
        )
        ax.set_ylim(0, max(1.0, max(median_scores) * 1.1))
        ax.set_xticks(range(0, n_layers, 4))
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        png_path = out_dir / f"{args.short}_top15pct_mass.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"Wrote: {png_path}", flush=True)
    except ImportError:
        print("matplotlib unavailable; skipping plot", flush=True)


if __name__ == "__main__":
    main()
