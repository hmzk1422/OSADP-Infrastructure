"""
pipeline.py
===========
End-to-end master orchestrator uniting LOB reconstruction and Event Merging.

Role in Pipeline:
    Sequences the LOB stages (optional day-start seed, then the every-event
    replay) across a date range, then the parallel Event Stream Merging stage.
    This is the standalone driver; ``components/commands/01_ingest.py`` is the
    command-level entry point built on the same orchestrator functions.

Note:
    This orchestrator assumes the three source streams are already present
    under ``matched/<TICKER>/_pipeline/source`` — it does not clone them. Use
    ``components/commands/01_ingest.py`` for the full chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Script entry: put the repo root on sys.path before importing components.*.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components.rust.event_merger.orchestrator import run_merger
from components.rust.lob_builder.orchestrator import run_lob, snapshot_builder


def replay_day(
    *,
    root: str | Path,
    symbol: str,
    date: str,
    tick_size: float = 0.1,
    warm_start: bool = True,
) -> Path:
    """
    Replay every orderbook file of one date in every-event mode.

    The whole ``orderbook/<DATE>`` directory is passed as a single input so the
    Rust engine keeps one shared book across all 24 hourly files, and writes
    ``_pipeline/orderflow/<DATE>/<HH>_every_event.parquet``.
    """
    root_p = Path(root)
    symbol = symbol.upper()
    orderbook_day = root_p / "ingest_parquet" / symbol / "orderbook" / date
    if not orderbook_day.is_dir() or not any(orderbook_day.glob("*.parquet")):
        raise FileNotFoundError(f"no orderbook files for {date} under {orderbook_day}")

    output_dir = run_merger.orderflow_dir(root_p, symbol, date)
    if output_dir.is_dir() and any(output_dir.rglob("*.parquet")):
        print(f"[PIPELINE] {date}: every-event replay already present, skipping")
        return output_dir

    run_lob.run_ingest_parquet(
        input_path=orderbook_day,
        output_dir=output_dir,
        tick_size=tick_size,
        snapshot_policy="every_event",
        warm_start=warm_start,
        label=f"every-event replay {date}",
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """Run LOB reconstruction then event merging across a date range."""
    parser = argparse.ArgumentParser(
        description="Full L2 pipeline: LOB reconstruction then event merging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="market_data root folder.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g. BTCUSDT).")
    parser.add_argument("--start", required=True, help="First UTC date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--end", required=True, help="Last UTC date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--workers", type=int, default=4, help="Dates processed in parallel.")
    parser.add_argument("--lob-levels", type=int, default=20, help="Book levels per side.")
    parser.add_argument("--tick-size", type=float, default=0.1, help="Instrument tick size.")
    parser.add_argument(
        "--seed-input",
        choices=["hour00", "day"],
        default="hour00",
        help="Day-start seed input: hour 00 only, or every orderbook file.",
    )
    parser.add_argument("--skip-day-start", action="store_true", help="Skip the day-start seed stage.")
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable --warm-start on the every-event replay (strict mode).",
    )
    parser.add_argument(
        "--keep-work-inputs",
        action="store_true",
        help="Keep cloned inputs, orderflow, sparse and state after finalize.",
    )
    args = parser.parse_args(argv)

    root_p = Path(args.root)
    dates = list(run_merger.generate_dates(args.start, args.end))
    if not dates:
        print("[ERROR] empty date range")
        return 1
    warm_start = not args.no_warm_start
    failures: list[tuple[str, str, Exception]] = []

    if not args.skip_day_start:
        print("=== STAGE 1: DAY-START SEED ===")
        for date_str in dates:
            try:
                snapshot_builder.day_start_seed(
                    root=root_p,
                    symbol=args.symbol,
                    date=date_str,
                    tick_size=args.tick_size,
                    warm_start=warm_start,
                    seed_input=args.seed_input,
                )
            except Exception as exc:  # noqa: BLE001 - reported per date
                print(f"[FAILED] seed {date_str}: {exc}")
                failures.append(("seed", date_str, exc))

    print("=== STAGE 2: LOB EVERY-EVENT RECONSTRUCTION ===")
    for date_str in dates:
        try:
            replay_day(
                root=root_p,
                symbol=args.symbol,
                date=date_str,
                tick_size=args.tick_size,
                warm_start=warm_start,
            )
        except Exception as exc:  # noqa: BLE001 - reported per date
            print(f"[FAILED] replay {date_str}: {exc}")
            failures.append(("replay", date_str, exc))

    print("=== STAGE 3: EVENT MERGING ===")
    for date_str in dates:
        try:
            run_merger.merge_and_finalize(
                root=root_p,
                symbol=args.symbol,
                date=date_str,
                lob_levels=args.lob_levels,
                source="matched",
                delete_inputs=not args.keep_work_inputs,
            )
        except Exception as exc:  # noqa: BLE001 - reported per date
            print(f"[FAILED] merge {date_str}: {exc}")
            failures.append(("merge", date_str, exc))

    if failures:
        print(f"[ERROR] pipeline finished with {len(failures)} failure(s)")
        for stage, date_str, exc in failures:
            print(f"  {stage} {date_str}: {exc}")
        return 1
    print(f"[SUCCESS] pipeline complete for {len(dates)} date(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())