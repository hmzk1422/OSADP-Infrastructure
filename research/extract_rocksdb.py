import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from rocksdict import Rdict

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# WHY THE OLD CSV NUMBERS "BROKE" (and what this script does about it)
# ---------------------------------------------------------------------------
# The old export went through `DataFrame.to_csv()`. Two things went wrong:
#
# 1. FORMATTING (Rust/C++ vs Python numbering)
#    pandas promotes the pnl integer fields (ts_ms, entry_ms, exit_ms,
#    last_exit_ms, n_trades, n_wins, n_losses) to float64 the moment a single
#    NaN sneaks into the wide DataFrame (trade and snapshot records never
#    share all their columns). The CSV then contained things like:
#       1709252400000.0                (was 1709252400000)
#       -9.223372036854776e+18         (was the i64::MIN sentinel
#                                       -9223372036854775808)
#    A Rust/C++ CSV reader deserializing those columns into i64 fails on
#    both texts. Fix below: integer fields are written as plain integer
#    literals (`str(int(v))`), float fields with Python's shortest round-trip
#    repr -- which emits the SAME digit string as the Rust JSON writer (both
#    use the shortest-round-trip algorithm, e.g. Ryu) -- and NaN as an empty
#    cell. No pandas float formatting in the output anymore.
#
# 2. UNITS (the real, value-level bug)
#    The old script also auto-*rescaled* entry_price / exit_price / qty by a
#    power-of-10 guess and recomputed net_usd from the rescaled inputs. When
#    the DB already contains correct values (it does in this repo), that just
#    corrupted them -- e.g. qty 0.0016252... became 0.016252... in the CSV
#    and net_usd changed to match the corrupted qty. That unit bug lives in
#    whatever Rust code writes the "pnl" column family; this export now leaves
#    the stored numbers untouched by default. The old band-aid is still
#    available behind `--rescale` if you ever encounter a genuinely broken DB.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lossless CSV writing
# ---------------------------------------------------------------------------
def detect_int_columns(records: list) -> set:
    """Columns that were integers in *every* JSON record.

    These are the i64-backing fields (ts_ms, entry_ms, ..., last_exit_ms).
    Formatting them as integer literals keeps them lossless through a CSV
    round-trip, exactly like the JSON export did.
    """
    tally: dict = {}
    for rec in records:
        for key, val in rec.items():
            stats = tally.setdefault(key, {"total": 0, "non_int": 0})
            stats["total"] += 1
            if not (isinstance(val, int) and not isinstance(val, bool)):
                stats["non_int"] += 1
    return {k for k, s in tally.items() if s["total"] > 0 and s["non_int"] == 0}


def format_cell(col: str, value, int_columns: set) -> str:
    """One cell -> the exact text that makes the number survive CSV."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        if col in int_columns:
            # column-wide integer: emit a plain integer literal (exact i64
            # semantics; safe because these values are < 2**53, plus the
            # i64::MIN sentinel -2**63, all exactly representable in f64)
            return str(int(value))
        # shortest round-trip decimal == the digit string Rust serde_json
        # wrote into the DB, so the CSV carries byte-identical numbers
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_pnl_csv(df: pd.DataFrame, path: str, int_columns: set) -> None:
    """Write a DataFrame with JSON-faithful numeric serialization."""
    header = [str(c) for c in df.columns]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in df.itertuples(index=False, name=None):
            writer.writerow(
                [format_cell(c, v, int_columns) for c, v in zip(header, row)]
            )


# ---------------------------------------------------------------------------
# Opt-in `--rescale` machinery (old unit-scaling band-aid, default OFF)
# ---------------------------------------------------------------------------
def column_scale_factor(series, target_mid: float) -> float:
    """One power-of-10 factor for the WHOLE column, based on its median
    magnitude -- not a per-row guess."""
    vals = series.dropna().astype(float)
    vals = vals[vals != 0]
    if vals.empty:
        return 1.0
    shift = round(math.log10(vals.abs().median() / target_mid))
    return 10.0 ** shift


def apply_column_scale(df, col, min_t, max_t) -> None:
    if col not in df.columns:
        return
    target_mid = math.sqrt(min_t * max_t)  # geometric mean of the window
    df[col] = df[col].astype(float) / column_scale_factor(df[col], target_mid)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the RocksDB 'pnl' column family to CSV losslessly."
    )
    parser.add_argument("--db", default="./build/live_test_btcusdt.rocksdb")
    parser.add_argument("--cf", default="pnl")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "build"),
        help="Where to write the CSVs (default: <project>/build)",
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        help="DANGER: apply the old power-of-10 unit-scaling heuristic and "
        "recompute net_usd. Only for genuinely broken databases; leave OFF "
        "for a faithful, lossless export.",
    )
    args = parser.parse_args()

    db = Rdict(args.db)
    cf = db.get_column_family(args.cf)

    records = []
    for _k, v in cf.items():
        v_str = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
        try:
            records.append(json.loads(v_str))
        except (json.JSONDecodeError, TypeError):
            continue
    cf.close()
    db.close()

    if not records:
        print("No records found in pnl column family.")
        return

    int_columns = detect_int_columns(records)

    df = pd.DataFrame(records)
    snapshots = df[df["ts_ms"].notna()].dropna(how="all", axis=1).copy()
    trades = df[df["entry_ms"].notna()].dropna(how="all", axis=1).copy()

    # Only used with --rescale (default OFF: numbers pass through untouched).
    INDEPENDENT_TARGET_RANGES = {
        "entry_price": (10_000.0, 200_000.0),  # BTC price
        "exit_price": (10_000.0, 200_000.0),   # BTC price
        "qty": (0.00001, 100.0),               # BTC position size
    }

    if args.rescale and not trades.empty:
        print(
            "WARNING: --rescale active; qty/price/net_usd will be modified. "
            "This is a band-aid for broken writers, not a lossless export."
        )
        for col, (min_t, max_t) in INDEPENDENT_TARGET_RANGES.items():
            apply_column_scale(trades, col, min_t, max_t)
        if {"entry_price", "exit_price", "qty", "side"}.issubset(trades.columns):
            direction = trades["side"].str.lower().map({"long": 1, "short": -1}).fillna(1)
            trades["net_usd"] = direction * (
                trades["exit_price"] - trades["entry_price"]
            ) * trades["qty"]
            print(
                "NOTE: r_multiple/equity_after need true risk & starting "
                "equity, which a broken DB cannot provide; they are exported "
                "as stored -- verify them separately if you rely on them."
            )

    if not trades.empty:
        # Readable UTC columns (derived, always safe; numeric fields untouched).
        for ts_col in ["entry_ms", "exit_ms"]:
            if ts_col in trades.columns:
                trades[f"{ts_col}_utc"] = pd.to_datetime(
                    trades[ts_col], unit="ms", utc=True
                ).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "duration_s" not in trades.columns and {"entry_ms", "exit_ms"}.issubset(
            trades.columns
        ):
            trades["duration_s"] = (trades["exit_ms"] - trades["entry_ms"]) / 1000.0

    if not snapshots.empty and "ts_ms" in snapshots.columns:
        snapshots["ts_ms_utc"] = pd.to_datetime(
            snapshots["ts_ms"], unit="ms", utc=True
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    def order_by(frame, priority):
        remaining = [c for c in frame.columns if c not in priority]
        return frame[[c for c in priority if c in frame.columns] + remaining]

    trades = order_by(
        trades,
        [
            "entry_ms_utc", "exit_ms_utc", "side", "entry_price", "exit_price",
            "reason", "qty", "net_usd", "r_multiple", "r_raw", "equity_after",
            "duration_s", "entry_ms", "exit_ms",
        ],
    )
    snapshots = order_by(
        snapshots,
        ["ts_ms_utc", "ts_ms", "equity", "n_trades", "n_wins", "n_losses",
         "total_r", "mid", "last_exit_ms"],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    trades_path = os.path.join(args.out_dir, "pnl_trades.csv")
    snapshots_path = os.path.join(args.out_dir, "pnl_snapshots.csv")
    write_pnl_csv(trades, trades_path, int_columns)
    write_pnl_csv(snapshots, snapshots_path, int_columns)
    print(f"Wrote {len(trades):,} trades     -> {trades_path}")
    print(f"Wrote {len(snapshots):,} snapshots -> {snapshots_path}")
    print(
        "CSV uses integer literals for i64 fields and shortest-round-trip "
        "decimal for floats; numbers match what the JSON export carried."
    )


if __name__ == "__main__":
    main()