"""
run_merger.py
=============
Parallel multi-day batch orchestrator with safe raw data deletion.

Role in Pipeline:
    Executes the Rust `event_merger` binary across date ranges using Python's 
    ThreadPoolExecutor. Implements a critical safety gate: raw input source files 
    are deleted ONLY if the Rust binary exits with code 0 (success).
"""

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path


def process_date(date_str: str, root: Path, symbol: str, binary: Path, delete_raw: bool):
    """
    Processes a single date: runs the Rust merger, checks exit status, and cleans up raw files.

    Args:
        date_str (str): Target date (YYYY-MM-DD).
        root (Path): Base path for market data directory.
        symbol (str): Trading pair ticker.
        binary (Path): Path to compiled Rust executable.
        delete_raw (bool): If True, deletes raw input files after successful merge.
    """
    cmd = [
        str(binary),
        "--root", str(root),
        "--symbol", symbol,
        "--date", date_str,
    ]

    print(f"[START MERGE] {date_str}")
    res = subprocess.run(cmd)

    if res.returncode == 0:
        print(f"[SUCCESS MERGE] {date_str}")
        if delete_raw:
            raw_dir = root / "raw" / symbol / date_str
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
                print(f"[CLEANUP] Deleted raw files: {raw_dir}")
    else:
        print(f"[FAILED] {date_str} exited with code {res.returncode}. Raw files preserved.")


def generate_dates(start_date: str, end_date: str):
    """
    Generates an inclusive sequence of YYYY-MM-DD date strings between start and end.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    curr = start
    while curr <= end:
        yield curr.strftime("%Y-%m-%d")
        curr += timedelta(days=1)


def main():
    """
    Main entry point for multi-threaded parallel event merging across date windows.
    """
    parser = argparse.ArgumentParser(description="Parallel Event Merger Orchestrator")
    parser.add_argument("--root", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delete-raw", action="store_true")
    args = parser.parse_args()

    ext = ".exe" if sys.platform.startswith("win") else ""
    binary = Path(__file__).parents[1] / "target" / "release" / f"event_merger{ext}"

    if not binary.exists():
        binary = Path(__file__).parents[2] / "target" / "release" / f"event_merger{ext}"

    if not binary.exists():
        print(f"[ERROR] Compiled binary not found at {binary}. Run 'cargo build --release' first.")
        sys.exit(1)

    dates = list(generate_dates(args.start, args.end))
    root_path = Path(args.root)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for date_str in dates:
            executor.submit(process_date, date_str, root_path, args.symbol, binary, args.delete_raw)


if __name__ == "__main__":
    main()