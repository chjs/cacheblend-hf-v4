#!/usr/bin/env python3
"""Add a lesson to docs/notes/v5-lessons.md in standard format.

Usage:
    python scripts/add_lesson.py \\
        --phase 3 \\
        --category "알고리즘 정확성" \\
        --title "HKVD elbow shape이 모델별로 다름" \\
        --symptom "Mistral은 elbow at ratio=0.10, Llama는 ratio=0.20" \\
        --root-cause "Attention sparsity가 architecture에 의존" \\
        --v5-fix "Phase 3 task에 모델별 elbow ratio 측정 단계 추가"

Auto-assigns next L## number, appends between LESSONS_START and LESSONS_END markers.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path


LESSONS_FILE = Path(__file__).parent.parent / "docs" / "notes" / "v5-lessons.md"
START_MARKER = "<!-- LESSONS_START -->"
END_MARKER = "<!-- LESSONS_END -->"


def get_next_lesson_number(content: str) -> int:
    """Find the highest L## in the file and return next number. Starts at 31 (continuing v4)."""
    matches = re.findall(r"^## (?:~~)?L(\d+)", content, flags=re.MULTILINE)
    if not matches:
        return 31  # Start at 31 (v4-lessons had L01-L30)
    nums = [int(m) for m in matches]
    return max(nums) + 1


def format_lesson(num: int, phase: str, category: str, title: str,
                  symptom: str, root_cause: str, v5_fix: str) -> str:
    """Format a lesson entry."""
    today = datetime.date.today().isoformat()
    phase_label = f"Phase {phase}" if phase else "Cross-phase"
    return f"""
## L{num:02d} — {title} ({today}, {phase_label})

**카테고리**: {category}

**증상**:
{symptom}

**근본 원인**:
{root_cause}

**v5 반영**:
{v5_fix}
"""


def update_stats(content: str, total_lessons: int) -> str:
    """Update the stats section at the bottom (replace the entire block)."""
    today = datetime.date.today().isoformat()
    # Replace from "## 누적 통계" to end of file
    new_block = (
        f"## 누적 통계\n\n"
        f"- 총 lessons: {total_lessons}\n"
        f"- 마지막 업데이트: {today}\n"
    )
    # Find "## 누적 통계" and replace everything from there
    idx = content.rfind("## 누적 통계")
    if idx == -1:
        return content + "\n\n---\n\n" + new_block
    return content[:idx] + new_block


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a lesson to v5-lessons.md")
    parser.add_argument("--phase", default="", help="Phase number (e.g. '3', '6a'). Empty for cross-phase.")
    parser.add_argument("--category", required=True, help="Lesson category")
    parser.add_argument("--title", required=True, help="Short title")
    parser.add_argument("--symptom", required=True, help="What happened")
    parser.add_argument("--root-cause", required=True, help="Why it happened")
    parser.add_argument("--v5-fix", required=True, help="How v5 should fix this")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    if not LESSONS_FILE.exists():
        print(f"ERROR: {LESSONS_FILE} not found", file=sys.stderr)
        return 1

    content = LESSONS_FILE.read_text(encoding="utf-8")

    if START_MARKER not in content or END_MARKER not in content:
        print(f"ERROR: markers {START_MARKER}/{END_MARKER} not found in {LESSONS_FILE}",
              file=sys.stderr)
        return 1

    num = get_next_lesson_number(content)
    lesson = format_lesson(
        num=num,
        phase=args.phase,
        category=args.category,
        title=args.title,
        symptom=args.symptom,
        root_cause=args.root_cause,
        v5_fix=args.v5_fix,
    )

    # Insert before END_MARKER
    new_content = content.replace(
        f"{START_MARKER}\n{END_MARKER}",
        f"{START_MARKER}\n{lesson}\n{END_MARKER}",
    )
    if new_content == content:
        # Markers not adjacent — insert before END_MARKER
        new_content = content.replace(
            END_MARKER,
            f"{lesson}\n{END_MARKER}",
        )

    # Count total active (non-retracted) lessons
    active_count = len(re.findall(r"^## L\d+", new_content, flags=re.MULTILINE))
    new_content = update_stats(new_content, active_count)

    if args.dry_run:
        print("=== DRY RUN (would write) ===")
        print(lesson)
        print(f"\n=> Total active lessons would be: {active_count}")
        return 0

    LESSONS_FILE.write_text(new_content, encoding="utf-8")
    print(f"✓ Added L{num:02d} — {args.title}")
    print(f"  File: {LESSONS_FILE}")
    print(f"  Total active lessons: {active_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
