"""
snapshot_builder.py
===================
Specialized runner for generating 00:00 UTC Order Book seed snapshots.

Role in Pipeline:
    Writes the 00:00 UTC day-start seed with `lob_replay ingest-parquet
    --snapshot-policy day_start`. This is the optional stage 1 of
    components/commands/01_ingest.py.
    Output: matched/<TICKER>/snapshots/<DATE>/day_start_<DAY>.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Script entry: put the repo root on sys.path before importing components.*.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components.rust.lob_builder.orchestrator import run_lob


def day_start_seed(
    *,
    root: str | Path,
    symbol: str,
    date: str,
    tick_size: float = 0.1,
    warm_start: bool = True,
    seed_input: str = "hour00",
) -> Path | None:
    """
    Run the day-start seed for one UTC date and return the output directory.

    Delegates to ``lob_replay ingest-parquet --snapshot-policy day_start``.

    ``seed_input`` selects the replay input:

    ``hour00``
        only ``orderbook/<DATE>/00.parquet`` — cheapest, and the seed is by
        definition taken from the first valid event of hour 00.
    ``day``
        the whole ``orderbook/<DATE>`` directory (every file, one shared
        engine). Equivalent seed when hour 00 carries valid events, but
        tolerant of an empty/missing ``00.parquet``.

    Returns ``None`` (with a log line) when the input does not exist, so an
    optional stage never aborts the pipeline.
    """
    orderbook_day = Path(root) / "ingest_parquet" / symbol.upper() / "orderbook" / date
    if seed_input == "day":
        input_path = orderbook_day
    else:
        input_path = orderbook_day / "00.parquet"

    if input_path.is_dir():
        has_input = any(orderbook_day.glob("*.parquet"))
    else:
        has_input = input_path.is_file()
    if not has_input:
        print(f"[SNAPSHOT] no orderbook input for {date} ({input_path}); seed skipped")
        return None

    output_dir = Path(root) / "matched" / symbol.upper() / "snapshots" / date
    output_dir.mkdir(parents=True, exist_ok=True)

    run_lob.run_ingest_parquet(
        input_path=input_path,
        output_dir=output_dir,
        tick_size=tick_size,
        snapshot_policy="day_start",
        warm_start=warm_start,
        label=f"day_start seed {date}",
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """CLI for the day-start seed stage."""
    parser = argparse.ArgumentParser(
        description="Build the 00:00 UTC LOB day-start seed for one date.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="market_data root folder.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g. BTCUSDT).")
    parser.add_argument("--date", required=True, help="UTC date YYYY-MM-DD.")
    parser.add_argument("--tick-size", type=float, default=0.1, help="Instrument tick size.")
    parser.add_argument(
        "--seed-input",
        choices=["hour00", "day"],
        default="hour00",
        help="Replay only hour 00, or every orderbook file for the date.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable --warm-start (strict sequence validation only).",
    )
    args = parser.parse_args(argv)

    output = day_start_seed(
        root=args.root,
        symbol=args.symbol,
        date=args.date,
        tick_size=args.tick_size,
        warm_start=not args.no_warm_start,
        seed_input=args.seed_input,
    )
    if output is None:
        return 1
    print(f"[SNAPSHOT] seed output -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())