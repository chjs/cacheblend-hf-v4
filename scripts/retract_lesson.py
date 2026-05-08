#!/usr/bin/env python3
"""Retract a lesson by marking it with strike-through (Q2=B preserve history).

Usage:
    python scripts/retract_lesson.py L42 --reason "Phase 7에서 재현 안 됨, 측정 오류였음"

The lesson is preserved but wrapped in ~~ marks.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LESSONS_FILE = Path(__file__).parent.parent / "docs" / "notes" / "v5-lessons.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retract a lesson with strike-through")
    parser.add_argument("lesson_id", help="Lesson ID (e.g. 'L42')")
    parser.add_argument("--reason", required=True, help="Why retracted")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    lesson_id = args.lesson_id.upper().strip()
    if not re.match(r"^L\d+$", lesson_id):
        print(f"ERROR: lesson_id must be like 'L42', got '{lesson_id}'", file=sys.stderr)
        return 1

    content = LESSONS_FILE.read_text(encoding="utf-8")

    # Pattern: "## L42 — title (date)" → "## ~~L42 — title (date) (철회: reason)~~"
    pattern = re.compile(
        rf"^## ({lesson_id} — [^\n]+)$",
        flags=re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        # Already retracted?
        if re.search(rf"^## ~~{lesson_id}", content, flags=re.MULTILINE):
            print(f"ERROR: {lesson_id} already retracted", file=sys.stderr)
        else:
            print(f"ERROR: {lesson_id} not found in {LESSONS_FILE}", file=sys.stderr)
        return 1

    original_header = match.group(1)
    new_header = f"~~{original_header} (철회: {args.reason})~~"

    new_content = pattern.sub(f"## {new_header}", content)

    if args.dry_run:
        print(f"=== DRY RUN ===")
        print(f"Would change:")
        print(f"  ## {original_header}")
        print(f"=>")
        print(f"  ## {new_header}")
        return 0

    LESSONS_FILE.write_text(new_content, encoding="utf-8")
    print(f"✓ Retracted {lesson_id}")
    print(f"  Reason: {args.reason}")
    print(f"  Note: lesson content preserved, only header marked retracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
