"""Four-stage terminal matched-data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import components.pipelines.ingestion.CryptoL2.datapaths as paths
from components.pipelines.event_merger.matched.merge import run_merge
from components.pipelines.event_merger.matched.parquet import fetch_source_data
from components.pipelines.event_merger.matched.snapshots import (
    build_day_start_snapshots,
    build_per_row_book_state,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical matched event data in four stages.")
    parser.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--source", choices=["ingest_parquet", "matched"], default="ingest_parquet")
    parser.add_argument("--lob-levels", type=int, default=20)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--keep-work-inputs", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    symbol = args.symbol.upper()
    build_day_start_snapshots(
        root=root, symbol=symbol, start=args.start, end=args.end,
        lob_levels=args.lob_levels, tick_size=args.tick_size,
    )
    fetch_source_data(root=root, symbol=symbol, start=args.start, end=args.end)
    build_per_row_book_state(
        root=root, symbol=symbol, start=args.start, end=args.end,
        lob_levels=args.lob_levels, tick_size=args.tick_size,
    )
    for day in sorted(
        set(paths.binance_parquet_days(root, symbol, "aggTrades"))
        & set(paths.binance_parquet_days(root, symbol, "bookDepth"))
        & set(paths.orderbook_parquet_days(root, symbol))
    ):
        if not (args.start <= day <= args.end):
            continue
        run_merge(
            root=root, symbol=symbol, source=args.source, date=day,
            hour=None, lob_levels=args.lob_levels,
            keep_work_inputs=args.keep_work_inputs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
