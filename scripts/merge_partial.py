#!/usr/bin/env python3
"""Merge incremental jsonl checkpoints from long-running phases [L07].

When Pod reclaim happens during Phase 6c/7c/7d/8-step3, results are saved
incrementally as <name>.partial.jsonl. After resumption (potentially on
a new pod), this script merges all partial files into the final results.

Usage:
    python scripts/merge_partial.py reports/phase-6c-attachments/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("attachment_dir", help="Directory with .partial.jsonl files")
    parser.add_argument("--output", default="results.jsonl", help="Output filename")
    args = parser.parse_args()
    
    attach_dir = Path(args.attachment_dir)
    if not attach_dir.exists():
        print(f"ERROR: {attach_dir} not found", file=sys.stderr)
        return 1
    
    partial_files = sorted(attach_dir.glob("*.partial.jsonl"))
    if not partial_files:
        print(f"No .partial.jsonl files in {attach_dir}")
        return 0
    
    seen_ids = set()
    merged = []
    for pf in partial_files:
        for line in pf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("id")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                merged.append(rec)
            except json.JSONDecodeError:
                print(f"WARN: skipping invalid line in {pf}", file=sys.stderr)
    
    merged.sort(key=lambda r: r.get("id", ""))
    
    output_path = attach_dir / args.output
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")
    
    print(f"✓ Merged {len(partial_files)} files → {output_path}")
    print(f"  Unique records: {len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
