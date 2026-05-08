#!/usr/bin/env python3
"""Evaluate gate JSON for a phase.

Handles all check_types from Phase 0 [L24]:
  - import: `python -c "import X"` returns 0
  - command: arbitrary shell command returns 0
  - file_exists: file/directory exists
  - pytest: pytest run returns 0
  - verify_phase: verify_phase.py for that phase returns 0
  - metric: numeric metric satisfies bound
  - cost_check: cumulative cost under threshold
  - sub_phases: nested conditions (Phase 6/7 sub-phases)

Uses sys.executable instead of "python" (L24 macOS fix).

Usage:
    python scripts/eval_gate.py --phase 3
    python scripts/eval_gate.py --phase 6 --sub-phase 6a

Output: gates/gate-N-result.json with PASS/FAIL per condition.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
GATES_DIR = ROOT / "gates"


def run_cmd(cmd: list[str], **kwargs) -> tuple[int, str, str]:
    """Run a command, return (rc, stdout, stderr)."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, **kwargs)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def check_import(spec: dict) -> tuple[bool, str]:
    module = spec.get("module", "")
    rc, _, err = run_cmd([sys.executable, "-c", f"import {module}"])
    if rc == 0:
        return True, f"import {module} OK"
    return False, f"import {module} failed: {err.strip()[:200]}"


def check_command(spec: dict) -> tuple[bool, str]:
    cmd = spec.get("cmd", "")
    expect_rc = spec.get("expect_rc", 0)
    rc, out, err = run_cmd(["bash", "-c", cmd], cwd=ROOT)
    detail = f"rc={rc}, stdout[:100]={out[:100]!r}"
    return rc == expect_rc, detail


def check_file_exists(spec: dict) -> tuple[bool, str]:
    path = ROOT / spec.get("path", "")
    if path.exists():
        return True, f"path exists: {path}"
    return False, f"missing: {path}"


def check_pytest(spec: dict) -> tuple[bool, str]:
    test_path = spec.get("path", "tests/")
    markers = spec.get("markers", "not gpu and not slow and not requires_model")
    cmd = [sys.executable, "-m", "pytest", test_path, "-m", markers, "-q"]
    rc, out, err = run_cmd(cmd, cwd=ROOT)
    last = out.strip().splitlines()[-1] if out.strip() else err.strip()[:200]
    return rc == 0, last


def check_verify_phase(spec: dict) -> tuple[bool, str]:
    phase = spec.get("phase", "")
    cmd = [sys.executable, str(ROOT / "scripts" / "verify_phase.py"), "--phase", str(phase)]
    rc, out, err = run_cmd(cmd, cwd=ROOT)
    return rc == 0, f"verify_phase rc={rc}"


def check_metric(spec: dict) -> tuple[bool, str]:
    """Numeric metric check.
    spec = {
        "source": "reports/phase-3-attachments/results.json",  # JSON file
        "key": "f1_score",  # JSON key path (dot-separated supported)
        "op": ">=" | ">" | "<=" | "<" | "==" | "!=",
        "value": 0.30,
    }
    """
    source = ROOT / spec.get("source", "")
    if not source.exists():
        return False, f"metric source not found: {source}"
    try:
        data = json.loads(source.read_text())
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    
    key_path = spec.get("key", "").split(".")
    val = data
    for k in key_path:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return False, f"key path not found: {'.'.join(key_path)}"
    
    op = spec.get("op", ">=")
    expected = spec.get("value", 0)
    
    ops = {
        ">=": lambda a, b: a >= b,
        ">":  lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<":  lambda a, b: a < b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    if op not in ops:
        return False, f"unknown op: {op}"
    
    ok = ops[op](val, expected)
    return ok, f"{spec.get('key')} = {val} {op} {expected}"


def check_cost_check(spec: dict) -> tuple[bool, str]:
    """Cumulative cost check.
    spec = {
        "source": "reports/cost-tracker.json",
        "key": "cumulative_usd",
        "max": 30.0,
    }
    """
    source = ROOT / spec.get("source", "reports/cost-tracker.json")
    if not source.exists():
        return False, f"cost tracker not found: {source}"
    try:
        data = json.loads(source.read_text())
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    
    key = spec.get("key", "cumulative_usd")
    val = data.get(key, 0.0)
    cap = spec.get("max", float("inf"))
    
    ok = val <= cap
    return ok, f"{key} = ${val:.2f} (cap: ${cap:.2f})"


CHECKERS = {
    "import": check_import,
    "command": check_command,
    "file_exists": check_file_exists,
    "pytest": check_pytest,
    "verify_phase": check_verify_phase,
    "metric": check_metric,
    "cost_check": check_cost_check,
}


def evaluate_conditions(conditions: list[dict]) -> tuple[list[dict], int, int]:
    """Run all conditions, return (results, n_pass, n_fail)."""
    results = []
    n_pass = 0
    n_fail = 0
    for cond in conditions:
        check_type = cond.get("check_type", "")
        cond_id = cond.get("id", "?")
        desc = cond.get("description", "")
        
        if check_type not in CHECKERS:
            results.append({
                "id": cond_id, "description": desc, "check_type": check_type,
                "result": "FAIL", "detail": f"unknown check_type: {check_type}",
            })
            n_fail += 1
            continue
        
        try:
            ok, detail = CHECKERS[check_type](cond)
        except Exception as e:
            ok = False
            detail = f"exception: {type(e).__name__}: {e}"
        
        results.append({
            "id": cond_id, "description": desc, "check_type": check_type,
            "result": "PASS" if ok else "FAIL", "detail": detail,
        })
        if ok:
            n_pass += 1
        else:
            n_fail += 1
    
    return results, n_pass, n_fail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, help="Phase number (e.g. '0', '6')")
    parser.add_argument("--sub-phase", default=None, help="Sub-phase (e.g. '6a')")
    args = parser.parse_args()
    
    # Find gate JSON
    candidates = [
        GATES_DIR / f"gate-{args.phase}-to-{int(args.phase)+1}.json",
        GATES_DIR / f"gate-{args.phase}-final.json",
    ]
    gate_file = next((c for c in candidates if c.exists()), None)
    if gate_file is None:
        print(f"ERROR: no gate JSON found for phase {args.phase}", file=sys.stderr)
        print(f"  Tried: {[str(c) for c in candidates]}", file=sys.stderr)
        return 2
    
    gate = json.loads(gate_file.read_text())
    
    # Sub-phase or flat conditions
    if args.sub_phase:
        sub = gate.get("sub_phases", {}).get(args.sub_phase)
        if sub is None:
            print(f"ERROR: sub-phase {args.sub_phase} not in {gate_file}", file=sys.stderr)
            return 2
        conditions = sub.get("conditions", [])
        scope = f"{args.phase}/{args.sub_phase}"
    else:
        conditions = gate.get("conditions", [])
        scope = args.phase
    
    print(f"=== Gate evaluation: phase {scope} ===")
    print(f"Source: {gate_file}")
    print(f"Conditions: {len(conditions)}")
    print()
    
    results, n_pass, n_fail = evaluate_conditions(conditions)
    
    for r in results:
        marker = "✓" if r["result"] == "PASS" else "✗"
        print(f"  {marker} [{r['id']}] {r['description']}: {r['detail']}")
    
    print()
    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"Result: {overall} ({n_pass}/{len(conditions)} passed)")
    
    # Write result JSON
    result_file = GATES_DIR / f"gate-{args.phase}{'-' + args.sub_phase if args.sub_phase else ''}-result.json"
    result_file.write_text(json.dumps({
        "phase": args.phase,
        "sub_phase": args.sub_phase,
        "overall": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"  → {result_file}")
    
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
