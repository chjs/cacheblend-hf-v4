"""Phase 5 — eval wrapper around the mydata cacheblend_fig12 harness.

Usage (Phase 5 — dry-run, CPU only, $0):
  python benchmarks/run_eval.py --runner cacheblend.runners:FullRecomputeRunner --dry-run
  python benchmarks/run_eval.py --dry-run-all   # runs all 5 runners in stub mode

Phase 6+ (GPU): pass --model mistralai/Mistral-7B-Instruct-v0.2 (no --dry-run)
to actually load the model and generate.

Output (dry-run):
  benchmarks/results/figure12_like/musique_dryrun.jsonl  (200 × 5 = 1000 rows)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path


PROMPTS_PATH = Path(__file__).resolve().parent.parent / "external/mydata/cacheblend_fig12/prompts.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results/figure12_like"


def _import_runner(spec: str):
    """spec = 'pkg.mod:ClassName'"""
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def _runner_label(runner_cls) -> str:
    return runner_cls.__name__


def _iter_prompts(path: Path = PROMPTS_PATH):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_dryrun_one(runner_cls, n_max: int | None = None) -> list[dict]:
    """Instantiate runner with model=None (stub mode) and emit pred=null rows."""
    runner = runner_cls()  # model=None default
    rows = []
    for i, ex in enumerate(_iter_prompts()):
        if n_max is not None and i >= n_max:
            break
        runner.prepare(ex["prompt_parts"]["system"], ex["prompt_parts"]["docs"], ex["prompt_parts"]["question"])
        gen = runner.generate(max_new_tokens=32)
        rows.append({
            "id": ex["id"],
            "runner": _runner_label(runner_cls),
            "pred": None,         # stub
            "f1": None,           # stub
            "rouge_l": None,      # stub
            "ttft_seconds": gen.ttft_seconds,
            "total_seconds": gen.total_seconds,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", default=None, help="pkg.mod:Class (e.g. cacheblend.runners:FullRecomputeRunner)")
    p.add_argument("--dry-run", action="store_true", help="No model; pred=null. CPU-only plumbing.")
    p.add_argument("--dry-run-all", action="store_true", help="All 5 runners in dry-run.")
    p.add_argument("--n-max", type=int, default=None, help="Cap number of samples (default = all 200)")
    p.add_argument("--out", default=str(RESULTS_DIR / "musique_dryrun.jsonl"))
    args = p.parse_args()

    if not args.dry_run and not args.dry_run_all:
        print("ERROR: Phase 5 only supports --dry-run / --dry-run-all (CPU stub).", file=sys.stderr)
        return 2

    if args.dry_run_all:
        runners = [
            _import_runner("cacheblend.runners:FullRecomputeRunner"),
            _import_runner("cacheblend.runners:FullReuseRunner"),
            _import_runner("cacheblend.runners:PrefixCacheRunner"),
            _import_runner("cacheblend.runners:CacheBlendV4Runner"),
        ]
    else:
        runners = [_import_runner(args.runner)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_total = 0
    with out_path.open("w") as f:
        for runner_cls in runners:
            rows = run_dryrun_one(runner_cls, n_max=args.n_max)
            for row in rows:
                f.write(json.dumps(row) + "\n")
            n_total += len(rows)
            print(f"  {_runner_label(runner_cls):24s}: {len(rows)} rows")

    print(f"\nWrote {n_total} rows to {out_path} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
