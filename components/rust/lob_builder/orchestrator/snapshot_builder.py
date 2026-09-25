"""
snapshot_builder.py
===================
Specialized runner for generating 00:00 UTC Order Book seed snapshots.

Role in Pipeline:
    Before merging intraday streams, the system requires a warm-start 00:00 LOB 
    snapshot state. This script triggers the Rust LOB binary in snapshot mode 
    to write initial state snapshots before full-day stream merging begins.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """
    Dispatches the snapshot creation task to the Rust LOB builder.
    """
    parser = argparse.ArgumentParser(description="00:00 Seed Snapshot Builder")
    parser.add_argument("--root", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--engine", choices=["btree", "hash"], default="btree")
    args = parser.parse_args()

    ext = ".exe" if sys.platform.startswith("win") else ""
    binary = Path(__file__).parents[1] / args.engine / "target" / "release" / f"{args.engine}_ver{ext}"

    cmd = [
        str(binary),
        "--task", "snapshot-seed",
        "--root", args.root,
        "--symbol", args.symbol,
        "--date", args.date,
    ]

    print(f"[SNAPSHOT] Building 00:00 LOB seed for {args.date}...")
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()