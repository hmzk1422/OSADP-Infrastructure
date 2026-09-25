"""
pipeline.py
===========
End-to-end master orchestrator uniting LOB reconstruction and Event Merging.

Role in Pipeline:
    Sequences Stage 1 (LOB State / Seed Reconstruction via `lob_builder`) and 
    Stage 2 (Parallel Event Stream Merging via `event_merger`). Acts as the 
    top-level python controller for the entire Rust backend pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """
    Executes the multi-stage ingestion and merging pipeline sequentially.
    """
    parser = argparse.ArgumentParser(description="Full L2 Ingestion Pipeline")
    parser.add_argument("--root", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--engine", choices=["btree", "hash"], default="btree")
    args = parser.parse_args()

    orchestrations_dir = Path(__file__).parent
    lob_script = orchestrations_dir.parents[1] / "lob_builder" / "orchestration" / "run_lob.py"
    merger_script = orchestrations_dir / "run_merger.py"

    # Step 1: Run LOB Builder Seed / Reconstruction
    print("=== STAGE 1: LOB RECONSTRUCTION ===")
    cmd_lob = [
        sys.executable, str(lob_script),
        "--engine", args.engine,
        "--root", args.root,
        "--symbol", args.symbol,
        "--date", args.start,
    ]
    res = subprocess.run(cmd_lob)
    if res.returncode != 0:
        sys.exit(res.returncode)

    # Step 2: Run Parallel Event Merger
    print("=== STAGE 2: EVENT MERGING ===")
    cmd_merger = [
        sys.executable, str(merger_script),
        "--root", args.root,
        "--symbol", args.symbol,
        "--start", args.start,
        "--end", args.end,
        "--workers", str(args.workers),
    ]
    res = subprocess.run(cmd_merger)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()