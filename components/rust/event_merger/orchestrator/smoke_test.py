"""
smoke_test.py
=============
Lightweight validation runner for testing single-hour or single-day merges.

Role in Pipeline:
    Used during development and CI verification to run isolated merges 
    without modifying production dataset outputs or deleting source files.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """
    Executes a single smoke test pass against the Rust event merger.
    """
    parser = argparse.ArgumentParser(description="Single-hour/date smoke test")
    parser.add_argument("--root", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--hour", type=int, default=0)
    args = parser.parse_args()

    ext = ".exe" if sys.platform.startswith("win") else ""
    binary = Path(__file__).parents[1] / "target" / "release" / f"event_merger{ext}"

    cmd = [
        str(binary),
        "--root", args.root,
        "--symbol", args.symbol,
        "--date", args.date,
        "--hour", str(args.hour),
        "--smoke",
    ]

    print(f"[SMOKE TEST] Running smoke merge for {args.date} hour {args.hour:02d}...")
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()