#!/usr/bin/env python3
"""Verify phase deliverables by checking required files exist + tests pass.

Usage:
    python scripts/verify_phase.py --phase 0
    python scripts/verify_phase.py --phase 6 --sub-phase 6a

Different from eval_gate.py: this is the phase's "self-verify" hook (lightweight).
eval_gate.py is the binding gate evaluator.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Phase-specific required files (in addition to skeleton)
PHASE_FILES = {
    "0": [
        "external/LMCache/.git",
        "external/mydata/cacheblend_fig12/prompts.jsonl",
        "docs/lmcache-analysis.md",
        "docs/figure12_like_disclosure.md",
        "reports/phase-0-report.md",
    ],
    "1": [
        "src/cacheblend/model.py",
        "tests/test_layerwise.py",
        "reports/phase-1-report.md",
    ],
    "2": [
        "tests/test_kv_reuse.py",
        "reports/phase-2-report.md",
    ],
    "3": [
        "src/cacheblend/hkvd.py",
        "tests/test_selective.py",
        "benchmarks/long_chunk_sanity.py",
        "reports/phase-3-report.md",
    ],
    "4": [
        "src/cacheblend/controller.py",
        "tests/test_pipeline.py",
        "benchmarks/ttft.py",
        "reports/phase-4-report.md",
    ],
    "5": [
        "external/mydata/cacheblend_fig12/harness/runner.py",
        "src/cacheblend/runners.py",
        "benchmarks/run_eval.py",
        "benchmarks/metrics/bootstrap.py",
        "tests/test_runners.py",
        "tests/test_bootstrap.py",
        "benchmarks/results/figure12_like/musique_dryrun.jsonl",
        "reports/phase-5-report.md",
    ],
    "6a": ["reports/phase-6a-attachments/results.jsonl"],
    "6b": ["reports/phase-6b-attachments/results.jsonl"],
    "6c": ["reports/phase-6c-attachments/results.jsonl"],
    "7a": ["reports/phase-7a-attachments/results.jsonl"],
    "7b": ["reports/phase-7b-attachments/results.jsonl"],
    "7c": ["reports/phase-7c-attachments/results.jsonl"],
    "7d": ["reports/phase-7d-attachments/results.jsonl"],
    "8-step2": ["reports/phase-8-step2-attachments/schedules.json"],
}

REQUIRED_V5_LESSONS_SECTION = "## v5-lessons"  # In each phase report


def check_files(phase: str) -> tuple[bool, list[str]]:
    files = PHASE_FILES.get(phase, [])
    missing = [f for f in files if not (ROOT / f).exists()]
    return len(missing) == 0, missing


def check_v5_lessons_section(phase: str) -> tuple[bool, str]:
    """Each phase report must have v5-lessons section."""
    report_path = ROOT / f"reports/phase-{phase}-report.md"
    if not report_path.exists():
        return False, f"report not found: {report_path}"
    content = report_path.read_text()
    if REQUIRED_V5_LESSONS_SECTION not in content:
        return False, f"v5-lessons section missing in {report_path}"
    return True, "v5-lessons section present"


def check_smoke_tests(phase: str) -> tuple[bool, str]:
    """Phase 0 only: smoke tests pass."""
    if phase != "0":
        return True, "skip"
    
    cmd = [sys.executable, "-m", "pytest", "tests/test_smoke.py",
           "-m", "not gpu and not slow and not requires_model", "-q"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode == 0:
        return True, "smoke tests pass"
    return False, f"smoke tests failed: {res.stdout[-200:]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, help="Phase number (e.g. '0', '6a')")
    args = parser.parse_args()
    
    print(f"=== verify_phase {args.phase} ===")
    
    all_ok = True
    
    # File existence
    ok, missing = check_files(args.phase)
    if ok:
        n = len(PHASE_FILES.get(args.phase, []))
        print(f"  ✓ all {n} required files exist")
    else:
        print(f"  ✗ missing files: {missing}")
        all_ok = False
    
    # v5-lessons section
    if not args.phase.startswith("8-"):  # Phase 8 step reports may not exist yet
        ok, msg = check_v5_lessons_section(args.phase)
        if ok:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")
            all_ok = False
    
    # Smoke tests (Phase 0)
    ok, msg = check_smoke_tests(args.phase)
    if msg != "skip":
        if ok:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")
            all_ok = False
    
    print()
    if all_ok:
        print(f"verify_phase {args.phase}: PASS")
        return 0
    else:
        print(f"verify_phase {args.phase}: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
