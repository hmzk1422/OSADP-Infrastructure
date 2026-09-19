"""Scan orderbook parquet files for snapshot-bearing files (parallel).

Usage: python _scan_snapshots.py <start_day> <end_day>
Reads ONLY the event_type column (projection pushdown) from each hourly
file in the unified tree between the given dates and reports files that
contain snapshot rows.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

BASE = Path(r'c:\Users\Himaska\Documents\Projects\Business_Projects\intraday-momentum-composite\research\data_v2\parquet\BTCUSDT\orderbook')


def scan_file(args):
    day, hour = args
    f = BASE / day / f'{hour:02d}.parquet'
    if not f.exists():
        return (day, hour, -1)
    try:
        n = (
            pl.scan_parquet(str(f))
            .select(pl.col('event_type').filter(pl.col('event_type') == 'snapshot').len())
            .collect()
            .item()
        )
        return (day, hour, n)
    except Exception as exc:  # noqa: BLE001
        return (day, hour, -2)


def main():
    start, end = sys.argv[1], sys.argv[2]
    days = sorted(d.name for d in BASE.iterdir() if d.is_dir() and start <= d.name <= end)
    tasks = [(d, h) for d in days for h in range(24) if (BASE / d / f'{h:02d}.parquet').exists()]
    print(f'scanning {len(tasks)} files ({start} .. {end}) ...', flush=True)
    found = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for day, hour, n in ex.map(scan_file, tasks, chunksize=8):
            if n > 0:
                found.append((day, hour, n))
                print(f'  SNAPSHOT: {day} {hour:02d} -> {n} rows', flush=True)
    print(f'\ndone. snapshot files found: {len(found)}')


if __name__ == '__main__':
    main()
