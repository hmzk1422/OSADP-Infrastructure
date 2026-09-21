#!/usr/bin/env python3
"""
scan_snapshots.py
=================

Scan the CryptoHFTData orderbook parquet tree for snapshot-bearing hours.

CryptoHFTData stopped embedding `snapshot` events after ~2025-06-30, so the
replayed book in matched data is warm-started from diffs for later windows
(see `_build_multilevel_book` in the MoE data layer). This answers "which
hours still carry snapshots?" while reading only the `event_type` column
(projection pushdown), so it stays cheap on 6-7M-row hourly files.

Usage
-----
    python scan_snapshots.py --start 2025-06-28 --end 2025-07-01
    python scan_snapshots.py                       # whole available range
    python main.py scan-snapshots --start 2026-02-01 --end 2026-02-02
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import polars as pl

import components.python.data_pipeline.CryptoL2.datapaths as paths

MISSING = -1
UNREADABLE = -2


def scan_file(task: tuple[str, str, int]) -> tuple[str, int, int]:
    """Count snapshot rows in one hourly orderbook file (worker process)."""
    base, day, hour = task
    f = Path(base) / day / f"{hour:02d}.parquet"
    if not f.exists():
        return (day, hour, MISSING)
    try:
        n = (
            pl.scan_parquet(str(f))
            .select(
                pl.col("event_type")
                .filter(pl.col("event_type") == "snapshot")
                .len()
            )
            .collect()
            .item()
        )
        return (day, hour, int(n))
    except Exception:  # noqa: BLE001 - report unreadable files, keep scanning
        return (day, hour, UNREADABLE)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan orderbook parquet files for snapshot-bearing hours (parallel)."
    )
    parser.add_argument("--start", default=None, help="YYYY-MM-DD inclusive (default: first available day)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD inclusive (default: last available day)")
    parser.add_argument(
        "--root",
        default=str(paths.MARKET_DATA_ROOT),
        help="market_data root (default: %(default)s)",
    )
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument("--workers", type=int, default=12, help="parallel processes (default: %(default)s)")
    args = parser.parse_args(argv)

    base = paths.parquet_dir(args.root, args.symbol, "orderbook")
    days = paths.orderbook_parquet_days(args.root, args.symbol)
    if not days:
        print(f"No orderbook parquet days under {base}")
        return 1

    start = args.start or days[0]
    end = args.end or days[-1]
    days = [d for d in days if start <= d <= end]

    tasks = [
        (str(base), day, hour)
        for day in days
        for hour in range(24)
        if (base / day / f"{hour:02d}.parquet").exists()
    ]
    print(f"scanning {len(tasks):,} hourly file(s) ({start} .. {end}) from {base} ...", flush=True)

    found: list[tuple[str, int, int]] = []
    missing = unreadable = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for day, hour, n in pool.map(scan_file, tasks, chunksize=8):
            if n == MISSING:
                missing += 1
            elif n == UNREADABLE:
                unreadable += 1
                print(f"  UNREADABLE: {day} {hour:02d}", flush=True)
            elif n > 0:
                found.append((day, hour, n))
                print(f"  SNAPSHOT: {day} {hour:02d} -> {n:,} rows", flush=True)

    print(
        f"\ndone. scanned={len(tasks):,} missing={missing:,} unreadable={unreadable:,} "
        f"snapshot files={len(found):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())