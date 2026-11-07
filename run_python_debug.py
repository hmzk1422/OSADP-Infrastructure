#!/usr/bin/env python
"""Debug matched database loading."""

import sys
import os

_REPO_ROOT = "c:/Users/Himaska/Documents/Projects/Business_Projects/intraday-momentum-composite"
sys.path.insert(0, _REPO_ROOT)

import polars as pl
from pathlib import Path

if __name__ == "__main__":
    root = Path(os.path.join(_REPO_ROOT, "market_data", "matched", "BTCUSDT"))
    pattern = str(root / "merged_*.parquet")
    import glob
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files")
    for f in files[:3]:
        print(f"  {f}")

    # Only read one file to avoid timeout
    raw = pl.read_parquet(files[0])
    print(f"\nRaw data: {raw.height} rows x {raw.width} cols")

    # Check schema
    print("\nSchema:")
    for col in raw.columns:
        print(f"  {col}: {raw[col].dtype}")

    # Check event types
    print("\nEvent types:", raw["event_type"].unique().to_list())

    # Check timestamp range
    print("\nTimestamp range:", raw["timestamp"].min(), "-", raw["timestamp"].max())

    # Check LOB rows
    ob = raw.filter(pl.col("event_type") == "LOB")
    print(f"\nLOB rows: {ob.height}")
    print(ob.select(["timestamp", "book_state__best_bid_price", "book_state__best_ask_price"]).head(3))

    # Check TRADE rows
    td = raw.filter(pl.col("event_type") == "TRADE")
    print(f"\nTRADE rows: {td.height}")
    print(td.select(["timestamp", "agg__price", "agg__quantity"]).head(3))
