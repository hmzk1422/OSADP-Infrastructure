#!/usr/bin/env python3
"""
refresh_matched.py
==================

Rebuild the matched aggTrades x top-of-book LOB parquet files in
research/data_v2/matched/<SYMBOL>/ from the *newest* source data.

Thin wrapper around research.MoE.data.live.fetch_source_data so the refresh
can be run as a single (backgroundable) command instead of ad-hoc snippets.

Usage
-----
    python scripts/refresh_matched.py                     # dataconfig defaults
    python scripts/refresh_matched.py --chunk-size 1      # one hour per pass (low memory)
    python scripts/refresh_matched.py --start 2026-02-01 --end 2026-02-02
"""

import argparse
import sys
from pathlib import Path

# Make `research.*` importable when run as a plain script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.MoE.data import dataconfig as cfg  # noqa: E402
from research.MoE.data import live  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild matched aggTrades x LOB parquet files from the newest source data."
    )
    parser.add_argument("--start", default=cfg.START_DATE, help="Start date (default: dataconfig.START_DATE)")
    parser.add_argument("--end", default=cfg.END_DATE, help="End date (default: dataconfig.END_DATE)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Hours processed per pass; 1 keeps peak memory at ~one hourly LOB file (default: 1)",
    )
    parser.add_argument(
        "--lob-levels",
        type=int,
        default=cfg.LOB_LEVELS,
        help="Number of LOB price levels per side to reconstruct and attach to "
        "each trade (default: dataconfig.LOB_LEVELS = 20; 1 = top-of-book only)",
    )
    parser.add_argument(
        "--include-bookdepth",
        action="store_true",
        help="Also asof-join Binance bookDepth cumulative band depth "
        "(+-1..5%% of mid) as bd_* columns (optional, off by default)",
    )
    args = parser.parse_args()

    print(f"Refreshing matched data: {cfg.TICKER} {args.start} -> {args.end} (chunk_size={args.chunk_size})")
    print(f"aggTrades: {cfg.AGG_TRADES_PATH}")
    print(f"LOB:       {cfg.LOB_PATH}")
    print(f"LOB levels per side: {args.lob_levels}")
    print(f"bookDepth: {'INCLUDED' if args.include_bookdepth else 'off (optional)'}")
    print(f"Output:    {cfg.MATCHED_PATH}")

    live.fetch_source_data(
        start=args.start,
        end=args.end,
        chunk_size=args.chunk_size,
        lob_levels=args.lob_levels,
        include_bookdepth=args.include_bookdepth,
    )

    print("Refresh complete.")


if __name__ == "__main__":
    main()