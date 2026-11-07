#!/usr/bin/env python
"""Compute Python OFMR features from matched data only (no strategy)."""

import sys
import os

_REPO_ROOT = "c:/Users/Himaska/Documents/Projects/Business_Projects/intraday-momentum-composite"
sys.path.insert(0, _REPO_ROOT)

from strategy_system.strategy.ofmr.features import compute_features_from_matched

if __name__ == "__main__":
    out_csv = "c:/Users/Himaska/Documents/Projects/Business_Projects/intraday-momentum-composite/build/python_features.csv"
    # Only use first file for speed
    import glob
    matched_root = os.path.normpath(os.path.join(_REPO_ROOT, "market_data", "matched", "BTCUSDT"))
    files = sorted(glob.glob(os.path.join(matched_root, "merged_2025-06-30_*.parquet")))
    print(f"Using {len(files)} files: {files[:3]}...")
    
    feat_result = compute_features_from_matched(
        matched_root=matched_root,
        symbol="BTCUSDT",
        tag="2025-06-30",
    )
    df = feat_result.frame
    print(f"Feature frame: {df.height} rows x {df.width} cols")
    print("Columns:", df.columns)
    if df.height > 0:
        df.select(["datetime", "price", "Column_0", "Column_1", "Column_2", "Column_3", "Column_4", "Column_5", "Column_6", "Column_7", "Column_8", "Column_9", "Column_10", "Column_11", "Column_12", "Column_13", "Column_14", "bar_vol_std"]).write_csv(out_csv)
        print(f"Wrote {df.height} rows to {out_csv}")
    else:
        print("ERROR: Feature frame is empty!")
