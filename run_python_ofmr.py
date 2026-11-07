#!/usr/bin/env python
"""Run OFMR Python production pipeline on matched data and output CSV."""

import sys
import os

_REPO_ROOT = "c:/Users/Himaska/Documents/Projects/Business_Projects/intraday-momentum-composite"
sys.path.insert(0, _REPO_ROOT)

from strategy_system.strategy.ofmr.strategy import run_strategy_on_matched

if __name__ == "__main__":
    out_csv = "c:/Users/Himaska/Documents/Projects/Business_Projects/intraday-momentum-composite/build/python_ofmr_features.csv"
    matches = run_strategy_on_matched(
        matched_root=os.path.join(_REPO_ROOT, "market_data", "matched"),
        symbol="BTCUSDT",
        tag="2025-06-30",  # first 10 hours of the day
        output_csv=out_csv,
    )
    print(f"Wrote {len(matches)} trade records to {out_csv}")
    if matches:
        print("Columns:", list(matches[0].keys()))
