"""
01_ingest.py
============
Chain command #1 — L2 ingestion.

Runs the four ingestion stages for an inclusive UTC date range, delegating every
Rust-backed step to the orchestrators under ``components/rust/<library>/orchestrator``:

    1. day-start LOB seed          lob_builder/orchestrator/snapshot_builder.py  (optional)
    2. clone ingest -> matched     matched/parquet.fetch_source_data             (Python)
    3. every-event LOB replay      lob_builder/orchestrator/run_lob.py           (Rust)
    4. merge + finalize + cleanup  event_merger/orchestrator/run_merger.py       (Rust + Python)

Stage 4 writes one merged file per hour (``merged_<DATE>_<HH>.parquet``) with the
as-of carry already applied, then removes the per-hour inputs behind it, so a
completed date leaves only those 24 deliverables.

Usage::

    python components/commands/01_ingest.py \
        --start-date 2025-06-30 --end-date 2025-06-30

    # optional day-start seed, keep work inputs for inspection
    python components/commands/01_ingest.py \
        --start-date 2025-06-30 --end-date 2025-06-30 \
        --no-seed --keep-work-inputs --tick-size 0.1 --lob-levels 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Script entry: put the repo root on sys.path before importing components.*.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components.commands import _common as common  # noqa: E402
from components.rust.event_merger.orchestrator import run_merger  # noqa: E402
from components.rust.lob_builder.orchestrator import run_lob  # noqa: E402
from components.rust.lob_builder.orchestrator import snapshot_builder  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest L2 market data into the matched tree (clone -> LOB replay -> merge).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, help="First UTC date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--end-date", required=True, help="Last UTC date, YYYY-MM-DD (inclusive).")
    parser.add_argument(
        "--root",
        default=None,
        help="market_data root. Defaults to $MARKET_DATA_ROOT or <repo>/market_data.",
    )
    parser.add_argument("--symbol", default=common.DEFAULT_SYMBOL, help="Trading symbol.")
    parser.add_argument(
        "--tick-size",
        type=float,
        default=0.1,
        help="Instrument tick size passed to lob_replay.",
    )
    parser.add_argument("--lob-levels", type=int, default=20, help="Book levels per side for event_merger.")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip stage 1 (day-start snapshot seed).",
    )
    parser.add_argument(
        "--keep-work-inputs",
        action="store_true",
        help="Keep cloned inputs / orderflow / sparse after a successful merge.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable --warm-start on the every-event replay (strict sequence validation only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel dates per stage. Hours within a date always stay sequential.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the whole range on the first date failure instead of continuing.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run 'cargo build --release' for both binaries before the chain.",
    )
    return parser


def _cargo_build() -> None:
    """Build both Rust binaries (opt-in; existing binaries are used as-is)."""
    import subprocess

    components_manifest = common.REPO_ROOT / "components" / "Cargo.toml"
    for binary in ("lob_replay", "event_merger"):
        common.logger.info("[BUILD] cargo build --release --bin %s", binary)
        result = subprocess.run(
            [
                "cargo", "build",
                "--manifest-path", str(components_manifest),
                "--release",
                "--bin", binary,
            ],
            cwd=str(common.REPO_ROOT),
        )
        if result.returncode != 0:
            raise RuntimeError(f"cargo build failed for {binary}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common.configure_logging()
    log = common.logger

    root = common.market_root(args.root)
    symbol = args.symbol.upper()
    if not root.is_dir():
        log.error("market_data root not found: %s", root)
        return 1

    if args.build:
        _cargo_build()

    # Fail fast, and with an actionable message, when a binary is missing.
    for label, resolver in (
        ("lob_replay", run_lob.resolve_binary),
        ("event_merger", run_merger.resolve_binary),
    ):
        try:
            log.info("[%s] %s", label, resolver())
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 1

    dates = common.available_dates(root, symbol, args.start_date, args.end_date)
    if not dates:
        log.error(
            "no date in [%s, %s] has aggTrades + bookDepth + orderbook under %s",
            args.start_date,
            args.end_date,
            root,
        )
        return 1
    log.info("root=%s symbol=%s dates=%d (%s .. %s)", root, symbol, len(dates), dates[0], dates[-1])

    warm_start = not args.no_warm_start
    failures: list[tuple[str, str, Exception]] = []

    def record(stage: str, date: str, exc: Exception) -> None:
        log.error("[%s] %s failed: %s", stage, date, exc)
        failures.append((stage, date, exc))
        if args.stop_on_error:
            raise exc

    # STAGE 1 — optional day-start seed
    if not args.no_seed:
        with common.stage("STAGE 1: DAY-START LOB SEED"):
            for date in dates:
                try:
                    snapshot_builder.day_start_seed(
                        root=root,
                        symbol=symbol,
                        date=date,
                        tick_size=args.tick_size,
                        warm_start=warm_start,
                    )
                except Exception as exc:  # noqa: BLE001 - reported per date
                    record("seed", date, exc)

    # STAGE 2 — clone the three canonical streams into the matched work tree
    with common.stage("STAGE 2: CLONE ingest_parquet -> matched/_pipeline/source"):
        for date in dates:
            try:
                cloned = common.clone_sources(root, symbol, date)
                if not cloned:
                    raise FileNotFoundError(f"no canonical inputs to clone for {date}")
            except Exception as exc:  # noqa: BLE001 - reported per date
                record("clone", date, exc)

    # STAGE 3 — full every-event LOB reconstruction
    with common.stage("STAGE 3: LOB EVERY-EVENT RECONSTRUCTION"):
        def replay(date: str) -> None:
            orderbook_day = root / "ingest_parquet" / symbol / "orderbook" / date
            if not orderbook_day.is_dir() or not any(orderbook_day.glob("*.parquet")):
                raise FileNotFoundError(f"no orderbook files for {date} under {orderbook_day}")
            run_lob.run_ingest_parquet(
                input_path=orderbook_day,
                output_dir=common.orderflow_dir(root, symbol, date),
                tick_size=args.tick_size,
                snapshot_policy="every_event",
                warm_start=warm_start,
                label=f"every-event replay {date}",
            )

        for date in dates:
            try:
                replay(date)
            except Exception as exc:  # noqa: BLE001 - reported per date
                record("replay", date, exc)

    # STAGE 4 — merge, finalize (as-of carry) and delete the per-hour inputs
    with common.stage("STAGE 4: MERGE + FINALIZE + CLEANUP"):
        def merge(date: str) -> None:
            run_merger.merge_and_finalize(
                root=root,
                symbol=symbol,
                date=date,
                lob_levels=args.lob_levels,
                source="matched",
                delete_inputs=not args.keep_work_inputs,
            )

        for date in dates:
            try:
                merge(date)
            except Exception as exc:  # noqa: BLE001 - reported per date
                record("merge", date, exc)

    if failures:
        log.error("ingestion finished with %d failure(s):", len(failures))
        for stage, date, exc in failures:
            log.error("  %s %s: %s", stage, date, exc)
        return 1

    merged = sorted((root / "matched" / symbol).glob("merged_*.parquet"))
    log.info("[DONE] %d merged file(s) under %s", len(merged), root / "matched" / symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
