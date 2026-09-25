"""Copy one aligned date into a disposable root and run the Rust merger.

The command copies aggTrades, bookDepth, and one already-materialized LOB
Parquet. It never reads ``ingest_parquet`` during the merge and never invokes
any LOB builder.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import polars as pl

import components.python.data_pipeline.CryptoL2.datapaths as paths
from components.python.data_pipeline.matched.merge import run_merge


def _copy_date(
    root: Path, symbol: str, date: str, destination: Path, hours: list[int],
) -> list[Path]:
    target = destination / "ingest_parquet" / symbol
    copied: list[Path] = []
    for dataset in ("aggTrades", "bookDepth"):
        src = paths.parquet_dir(root, symbol, dataset) / f"{date}.parquet"
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = target / dataset / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    orderbook_days = paths.parquet_dir(root, symbol, "orderbook") / date
    if not orderbook_days.is_dir():
        raise FileNotFoundError(orderbook_days)
    for hour in hours:
        src = orderbook_days / f"{hour:02d}.parquet"
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = target / "orderbook" / date / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main(argv: Optional[list[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Copy one aligned date and run the event merger.")
    parser.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument("--date", required=True, help="UTC date present in all three datasets")
    parser.add_argument("--work-root", default=None, help="Disposable root; defaults to <root>/_smoke")
    parser.add_argument("--lob-levels", type=int, default=20)
    parser.add_argument(
        "--lob-file",
        required=True,
        help="Existing materialized LOB Parquet copied into the disposable root",
    )
    parser.add_argument("--hour", type=int, default=0, help="hour represented by --lob-file")
    args = parser.parse_args(argv)

    source_root = Path(args.root).resolve()
    symbol = args.symbol.upper()
    work_root = Path(args.work_root).resolve() if args.work_root else source_root / "_smoke" / symbol / args.date
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    lob_source = Path(args.lob_file).resolve()
    if not lob_source.is_file():
        raise FileNotFoundError(lob_source)
    copied = _copy_date(source_root, symbol, args.date, work_root, [])
    lob_target = work_root / "matched" / symbol / "_pipeline" / "lob_state" / args.date / f"{args.hour:02d}.parquet"
    lob_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lob_source, lob_target)
    copied.append(lob_target)
    print(f"copied {len(copied)} source parquet file(s) to {work_root}")

    outputs = run_merge(
        root=work_root, symbol=symbol, source="ingest_parquet", date=args.date,
        hour=args.hour, lob_levels=args.lob_levels, keep_work_inputs=True,
    )
    for output in outputs:
        frame = pl.read_parquet(output)
        print(f"OUTPUT {output} rows={frame.height} columns={len(frame.columns)}")
        counts = frame.group_by("event_type").len().sort("event_type")
        count_map = {
            row["event_type"]: row["len"]
            for row in counts.iter_rows(named=True)
        }
        source_columns = [column for column in frame.columns if column.startswith("source_")]
        print(f"EVENT_COUNTS {count_map}")
        print(f"SOURCE_COLUMNS {len(source_columns)}")
        print(f"SOURCE_COLUMN_SAMPLE {source_columns[:5] + source_columns[-5:]}")
        sample_columns = [
            "timestamp", "event_type", "agg__price", "agg__quantity",
            "book_state__best_bid_price", "book_state__best_ask_price",
            "depth_state__-5", "depth_state__1",
            "source_agg__quantity", "source_depth__percentage",
        ]
        print(frame.select([column for column in sample_columns if column in frame.columns]).head(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
