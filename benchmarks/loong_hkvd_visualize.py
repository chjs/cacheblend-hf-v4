"""Visualize HKVD token positions vs chunk boundaries for Loong examples.

Reads results.jsonl produced by run_loong.py, picks examples with hkvd_indices
recorded, plots:
  - x-axis: token index in the eval prompt's cached region
  - vertical lines: chunk boundaries
  - points: HKVD-selected tokens (height = local density)
  - per-example histogram of distance-to-nearest-boundary for HKVD tokens

Outputs PNG files into the same attachments dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _distance_to_nearest_boundary(idx: int, boundaries: list[int]) -> int:
    return min(abs(idx - b) for b in boundaries) if boundaries else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.jsonl path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-examples", type=int, default=4, help="how many examples to plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.results) if l.strip()]
    cb_rows = [r for r in rows if r.get("method") == "CacheBlend_r0.15" and "hkvd_indices" in r]
    if not cb_rows:
        print("no CacheBlend_r0.15 rows with hkvd_indices found")
        return

    # ── 1. Per-example HKVD position plot ──────────────────────────────────
    n_show = min(args.n_examples, len(cb_rows))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2 * n_show), sharex=False)
    if n_show == 1:
        axes = [axes]
    for ax, r in zip(axes, cb_rows[:n_show]):
        hkvd = r["hkvd_indices"]
        boundaries = r["chunk_boundaries"]
        cached_len = r["cached_region_tokens"]
        # Histogram of HKVD positions (bin per ~1% of cached_len)
        n_bins = 100
        hist, edges = np.histogram(hkvd, bins=n_bins, range=(0, cached_len))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, hist, width=cached_len / n_bins, color="#1f77b4", alpha=0.7,
               label=f"HKVD count per bin (n={len(hkvd)})")
        for b in boundaries:
            ax.axvline(b, color="red", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_title(f'ex {r["id"][:8]}: F1={r["f1"]:.3f}, cached_tok={cached_len}, '
                     f'n_HKVD={len(hkvd)}, n_boundaries={len(boundaries)}')
        ax.set_xlabel("token index")
        ax.set_ylabel("HKVD count")
        ax.set_xlim(0, cached_len)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("HKVD token distribution vs chunk boundaries (red dashed = chunk boundary)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out1 = out_dir / "hkvd_per_example.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out1}")

    # ── 2. Aggregate: distance-to-nearest-boundary histogram ──────────────
    all_dists_hkvd = []
    all_dists_random = []
    rng = np.random.default_rng(42)
    for r in cb_rows:
        hkvd = r["hkvd_indices"]
        boundaries = r["chunk_boundaries"]
        cached_len = r["cached_region_tokens"]
        if not boundaries:
            continue
        all_dists_hkvd.extend([_distance_to_nearest_boundary(i, boundaries) for i in hkvd])
        # Random control: pick same number of indices uniformly
        rand_idx = rng.integers(0, cached_len, size=len(hkvd)).tolist()
        all_dists_random.extend([_distance_to_nearest_boundary(i, boundaries) for i in rand_idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    max_d = max(max(all_dists_hkvd[:20000], default=100), 200)
    bins = np.arange(0, min(max_d, 200) + 1, 2)
    ax.hist(all_dists_hkvd, bins=bins, alpha=0.6, color="#1f77b4",
            label=f"HKVD (n={len(all_dists_hkvd)})", density=True)
    ax.hist(all_dists_random, bins=bins, alpha=0.6, color="#ff7f0e",
            label=f"Uniform random (n={len(all_dists_random)})", density=True)
    ax.set_xlabel("|token index − nearest chunk boundary|  (tokens)")
    ax.set_ylabel("density")
    ax.set_title(
        f"HKVD vs random — distance to nearest chunk boundary (n_examples={len(cb_rows)})"
    )
    ax.legend()
    fig.tight_layout()
    out2 = out_dir / "hkvd_distance_to_boundary.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")

    # ── 3. Enrichment vs window-width curve ──────────────────────────────
    windows = list(range(0, 51, 1))
    hkvd_frac = []
    rand_frac = []
    for W in windows:
        hkvd_count = sum(1 for d in all_dists_hkvd if d <= W)
        rand_count = sum(1 for d in all_dists_random if d <= W)
        hkvd_frac.append(hkvd_count / max(len(all_dists_hkvd), 1))
        rand_frac.append(rand_count / max(len(all_dists_random), 1))
    enrichment = [h / max(r, 1e-9) for h, r in zip(hkvd_frac, rand_frac)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(windows, hkvd_frac, "o-", color="#1f77b4", label="HKVD fraction in ±W window")
    ax1.plot(windows, rand_frac, "s--", color="#ff7f0e", label="Random fraction in ±W window")
    ax1.set_xlabel("Window half-width W (tokens)")
    ax1.set_ylabel("Fraction of tokens within ±W of any boundary")
    ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(windows, enrichment, ".-", color="#2ca02c", label="Enrichment ratio HKVD/random")
    ax2.set_ylabel("Enrichment ratio (HKVD / Random)", color="#2ca02c")
    ax2.tick_params(axis='y', labelcolor="#2ca02c")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.7)
    ax2.legend(loc="upper right")
    fig.suptitle("HKVD boundary concentration vs window width")
    fig.tight_layout()
    out3 = out_dir / "hkvd_enrichment_curve.png"
    fig.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out3}")

    # Print key numbers
    print("\n=== Key enrichment summary ===")
    for W in [1, 3, 8, 16, 32]:
        i = W
        print(f"  ±{W:3d} tokens: HKVD={hkvd_frac[i]*100:.2f}%  Random={rand_frac[i]*100:.2f}%  "
              f"enrichment={enrichment[i]:.2f}x")


if __name__ == "__main__":
    main()
