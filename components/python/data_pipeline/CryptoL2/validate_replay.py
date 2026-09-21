#!/usr/bin/env python3
"""
validate_replay.py
==================

Sanity checks over the matched aggTrades x L2 parquet files.

The LOB side of matched data is a REPLAYED order book (see
`_build_multilevel_book` in the MoE data layer), not a naive asof join of the
raw diff stream. This tool re-checks the invariants that replay is supposed to
guarantee, file by file, and prints an aggregate report:

    * no null best bid / best ask on any trade row
    * no crossed book (best_bid_price >= best_ask_price)
    * level ordering holds (bid_1 >= bid_2 >= ... / ask_1 <= ask_2 <= ...)
    * spread min/max and, when present, bookDepth (bd_*) coverage

Usage
-----
    python validate_replay.py                       # whole matched tree
    python validate_replay.py --start 2026-02-01 --end 2026-02-02
    python main.py validate-replay --start 2026-02-01 --end 2026-02-02
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import polars as pl

import components.python.data_pipeline.CryptoL2.datapaths as paths

# Matches BTCUSDT_2026-02-01_2026-02-01_05-05.parquet
RANGE_PATTERN = re.compile(
    r"^.*?_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})"
    r"(?:_(?P<start_hour>\d{2})-(?P<end_hour>\d{2}))?\.parquet$"
)


def matched_files(
    root: Optional[str] = None,
    symbol: str = paths.DEFAULT_SYMBOL,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> list[Path]:
    """Matched parquet files, optionally filtered to a [start, end] day window."""
    base = paths.matched_dir(root, symbol)
    if not base.is_dir():
        return []

    found = []
    for f in sorted(base.glob("*.parquet")):
        m = RANGE_PATTERN.match(f.name)
        if not m:
            continue
        if start and m.group("end") < start:
            continue
        if end and m.group("start") > end:
            continue
        found.append(f)
    return found


def check_file(path: Path, lob_levels: int) -> dict:
    """Run the invariant checks over a single matched parquet file.

    The schema is inspected first: matched files may carry fewer levels per
    side than requested (lob_levels=1 is legacy top-of-book) and the optional
    `bd_*` bookDepth columns may be absent. Missing optional columns report 0
    rather than raising.
    """
    lf = pl.scan_parquet(str(path))
    cols = set(lf.collect_schema().names())

    # Highest contiguous level available on both sides.
    max_level = 1
    for j in range(2, lob_levels + 1):
        if f"bid_price_{j}" in cols and f"ask_price_{j}" in cols:
            max_level = j
        else:
            break

    # Level ordering: bids must be non-increasing, asks non-decreasing.
    bid_viol = pl.lit(0)
    ask_viol = pl.lit(0)
    for j in range(2, max_level + 1):
        prev_bid = "best_bid_price" if j == 2 else f"bid_price_{j - 1}"
        prev_ask = "best_ask_price" if j == 2 else f"ask_price_{j - 1}"
        bid_viol = bid_viol + (
            (pl.col(f"bid_price_{j}") > pl.col(prev_bid)).fill_null(False).sum()
        )
        ask_viol = ask_viol + (
            (pl.col(f"ask_price_{j}") < pl.col(prev_ask)).fill_null(False).sum()
        )

    select = [
        pl.len().alias("rows"),
        pl.col("best_bid_price").is_null().sum().alias("null_best_bid"),
        pl.col("best_ask_price").is_null().sum().alias("null_best_ask"),
        (pl.col("best_bid_price") >= pl.col("best_ask_price"))
        .fill_null(False)
        .sum()
        .alias("crossed"),
        bid_viol.alias("bid_order_viol"),
        ask_viol.alias("ask_order_viol"),
        (pl.col("best_ask_price") - pl.col("best_bid_price")).min().alias("spread_min"),
        (pl.col("best_ask_price") - pl.col("best_bid_price")).max().alias("spread_max"),
    ]

    deep_bid, deep_ask = f"bid_price_{max_level}", f"ask_price_{max_level}"
    if max_level >= 2:
        select.append(pl.col(deep_bid).is_null().sum().alias("deep_null_bid"))
        select.append(pl.col(deep_ask).is_null().sum().alias("deep_null_ask"))
    else:
        select.append(pl.lit(0).alias("deep_null_bid"))
        select.append(pl.lit(0).alias("deep_null_ask"))

    has_bookdepth = "bd_depth_m1" in cols
    if has_bookdepth:
        select.append(pl.col("bd_depth_m1").is_not_null().sum().alias("bd_rows"))
    else:
        select.append(pl.lit(0).alias("bd_rows"))

    stats = lf.select(select).collect().row(0, named=True)
    stats["file"] = path.name
    stats["levels"] = max_level
    stats["has_bookdepth"] = has_bookdepth
    return stats
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate replayed LOB invariants across the matched parquet tree."
    )
    parser.add_argument(
        "--root",
        default=str(paths.MARKET_DATA_ROOT),
        help="market_data root (default: %(default)s)",
    )
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument(
        "--lob-levels",
        type=int,
        default=20,
        help="levels per side present in the matched files (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    files = matched_files(args.root, args.symbol, args.start, args.end)
    if not files:
        print(
            f"No matched parquet files under {paths.matched_dir(args.root, args.symbol)}"
            + (f" in [{args.start}, {args.end}]" if args.start or args.end else "")
        )
        return 1

    print(f"files: {len(files):,} (lob_levels={args.lob_levels})")

    totals = {
        "rows": 0,
        "null_best_bid": 0,
        "null_best_ask": 0,
        "crossed": 0,
        "bid_order_viol": 0,
        "ask_order_viol": 0,
        "deep_null_bid": 0,
        "deep_null_ask": 0,
        "bd_rows": 0,
    }
    spread_min, spread_max = None, None
    bookdepth_files = 0
    failures = 0
    levels_seen = set()

    for f in files:
        s = check_file(f, args.lob_levels)
        levels_seen.add(s["levels"])
        for key in totals:
            totals[key] += s[key]
        if s["has_bookdepth"]:
            bookdepth_files += 1
        if s["spread_min"] is not None:
            spread_min = (
                s["spread_min"] if spread_min is None else min(spread_min, s["spread_min"])
            )
        if s["spread_max"] is not None:
            spread_max = (
                s["spread_max"] if spread_max is None else max(spread_max, s["spread_max"])
            )

        bad = (
            s["null_best_bid"]
            + s["null_best_ask"]
            + s["crossed"]
            + s["bid_order_viol"]
            + s["ask_order_viol"]
        )
        if bad:
            failures += 1
            print(
                f"  VIOLATION {s['file']}: null_bid={s['null_best_bid']:,} "
                f"null_ask={s['null_best_ask']:,} crossed={s['crossed']:,} "
                f"bid_order={s['bid_order_viol']:,} ask_order={s['ask_order_viol']:,}"
            )

    rows = totals["rows"]
    print("\naggregate")
    print(f"  rows                   : {rows:,}")
    print(
        f"  null best bid / ask    : "
        f"{totals['null_best_bid']:,} / {totals['null_best_ask']:,}"
    )
    print(f"  crossed                : {totals['crossed']:,}")
    print(
        f"  level-order violations : bid={totals['bid_order_viol']:,} "
        f"ask={totals['ask_order_viol']:,}"
    )
    print(
        f"  levels per side        : "
        f"{min(levels_seen)}..{max(levels_seen)} (requested {args.lob_levels}); "
        f"deepest-level nulls bid={totals['deep_null_bid']:,} "
        f"ask={totals['deep_null_ask']:,}"
    )
    print(f"  spread min / max       : {spread_min} / {spread_max}")
    if bookdepth_files:
        coverage = totals["bd_rows"] / rows if rows else 0.0
        print(
            f"  bookDepth coverage     : {totals['bd_rows']:,}/{rows:,} = {coverage:.3%} "
            f"({bookdepth_files:,} file(s))"
        )
    else:
        print("  bookDepth coverage     : not included in these files")
    print(f"  files with violations  : {failures:,}/{len(files):,}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())