#!/usr/bin/env python3
"""Track Pod usage cost from Runpod billing API [L30].

Usage:
    python scripts/cost_track.py --pod-id <pod_id> --phase 1 --append
    python scripts/cost_track.py --report  # Show current cumulative

Auto-fetches `costPerHr` from runpodctl pod get and computes wall time × rate.
Appends to reports/cost-tracker.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent.parent
TRACKER_FILE = ROOT / "reports/cost-tracker.json"


def get_pod_info(pod_id: str) -> dict | None:
    try:
        res = subprocess.run(
            ["runpodctl", "pod", "get", pod_id, "--output", "json"],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode != 0:
            print(f"ERROR: runpodctl failed: {res.stderr}", file=sys.stderr)
            return None
        return json.loads(res.stdout)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None


def load_tracker() -> dict:
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text())
    return {"cumulative_usd": 0.0, "phases": {}, "events": []}


def save_tracker(data: dict) -> None:
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_FILE.write_text(json.dumps(data, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", help="Runpod pod ID (for --append)")
    parser.add_argument("--phase", help="Phase ID (for --append)")
    parser.add_argument("--append", action="store_true", help="Append cost for this pod/phase")
    parser.add_argument("--report", action="store_true", help="Show cumulative cost")
    parser.add_argument("--manual-usd", type=float, help="Manually add USD amount (no Runpod query)")
    args = parser.parse_args()
    
    data = load_tracker()
    
    if args.report:
        print(f"=== Cost tracker ===")
        print(f"Cumulative: ${data['cumulative_usd']:.2f}")
        print(f"Phases:")
        for phase, amount in data.get("phases", {}).items():
            print(f"  Phase {phase}: ${amount:.2f}")
        return 0
    
    if not args.append:
        parser.print_help()
        return 1
    
    if not args.phase:
        print("ERROR: --phase required with --append", file=sys.stderr)
        return 1
    
    cost_usd = 0.0
    detail = ""
    
    if args.manual_usd is not None:
        cost_usd = args.manual_usd
        detail = "manual"
    elif args.pod_id:
        info = get_pod_info(args.pod_id)
        if info is None:
            return 1
        
        cost_per_hr = float(info.get("costPerHr", 0))
        # uptime in seconds (Runpod returns various keys; check both)
        uptime_s = info.get("uptimeSeconds") or info.get("runtime", {}).get("uptimeSeconds", 0)
        cost_usd = cost_per_hr * (uptime_s / 3600.0)
        detail = f"{uptime_s}s × ${cost_per_hr:.3f}/hr"
    else:
        print("ERROR: --pod-id or --manual-usd required", file=sys.stderr)
        return 1
    
    data["cumulative_usd"] += cost_usd
    data["phases"].setdefault(args.phase, 0.0)
    data["phases"][args.phase] += cost_usd
    data["events"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "phase": args.phase,
        "cost_usd": round(cost_usd, 4),
        "detail": detail,
    })
    
    save_tracker(data)
    
    print(f"✓ Phase {args.phase}: +${cost_usd:.2f} ({detail})")
    print(f"  Cumulative: ${data['cumulative_usd']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
