import numpy as np
import pandas as pd

def lob_report(df):
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

    # Order book columns (levels 2-20, both sides)
    lob_cols = [
        c for c in df.columns
        if c.startswith(("bid_price_", "bid_qty_", "ask_price_", "ask_qty_"))
    ]

    # bookDepth columns + their timestamp
    bookdepth_cols = [c for c in df.columns if c.startswith("bd_")] + (
        ["timestamp"] if "timestamp" in df.columns else []
    )

    def report(name, cols):
        print(f"--- {name} ({len(cols)} columns) ---")
        for c in cols:
            n_null = df[c].isna().sum()
            pct = 100 * n_null / len(df) if len(df) else 0
            status = "ALL NULL" if n_null == len(df) else (
                "SOME NULL" if n_null else "FULLY POPULATED"
            )
            print(f"  {c:20s} null={n_null:>6}/{len(df):<6} ({pct:5.1f}%)  -> {status}")
        print()

    report("Order book (bid/ask levels 2-20)", lob_cols)
    report("bookDepth (bd_* + timestamp)", bookdepth_cols)

    # One-line verdict
    lob_all_populated = all(df[c].isna().sum() == 0 for c in lob_cols)
    bd_all_null = all(df[c].isna().sum() == len(df) for c in bookdepth_cols)

    print(f"Order book fully populated across all rows: {lob_all_populated}")
    print(f"bookDepth entirely null across all rows:    {bd_all_null}")