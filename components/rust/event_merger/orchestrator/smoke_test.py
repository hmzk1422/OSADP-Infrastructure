"""
smoke_test.py
=============
Lightweight validation runner for testing single-hour or single-day merges.

Role in Pipeline:
    Used during development and CI verification to run isolated merges without
    touching canonical data. Nothing is finalized or deleted, so the source
    files and the sparse output stay in place for inspection.

Examples::

    python components/rust/event_merger/orchestrator/smoke_test.py \
        --root market_data --symbol BTCUSDT --date 2025-06-30 --hour 0

    # whole day against a precomputed every-event LOB replay
    python components/rust/event_merger/orchestrator/smoke_test.py \
        --root market_data --symbol BTCUSDT --date 2025-06-30 \
        --matched-lob-root market_data/matched/BTCUSDT/_pipeline/orderflow/2025-06-30
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


def main(argv: list[str] | None = None) -> int:
    """Run a single-hour (or whole-day) merge without finalize or cleanup."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the event_merger on one hour or one day.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="market_data root folder.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g. BTCUSDT).")
    parser.add_argument("--date", required=True, help="UTC date YYYY-MM-DD.")
    parser.add_argument("--hour", type=int, default=0, help="0..23; use 'all' for the whole day.")
    parser.add_argument("--lob-levels", type=int, default=20, help="Book levels per side.")
    parser.add_argument(
        "--source",
        choices=["ingest_parquet", "matched"],
        default="matched",
        help="Where the three input streams are resolved from.",
    )
    parser.add_argument(
        "--matched-lob-root",
        default=None,
        help="Every-event LOB replay dir. Defaults to _pipeline/orderflow/<DATE> when present.",
    )
    args = parser.parse_args(argv)

    hour = None if str(args.hour).lower() == "all" else int(args.hour)
    if hour is not None and not 0 <= hour <= 23:
        parser.error("--hour must be 0..23 or 'all'")

    # Same pre-flight check as merge_and_finalize: without a real every-event
    # replay the merger silently falls back to the raw orderbook and dies with
    # the opaque "LOB is_bid is null".
    if args.matched_lob_root is None:
        default_lob_root = run_merger.orderflow_dir(args.root, args.symbol, args.date)
        if default_lob_root.is_dir():
            args.matched_lob_root = str(default_lob_root)
    if args.matched_lob_root is None or not run_merger._has_lob_book_state(
        args.matched_lob_root
    ):
        print(
            f"[ERROR] no every-event LOB book state for {args.date} under "
            f"{run_merger.orderflow_dir(args.root, args.symbol, args.date)}. "
            "Run the LOB replay stage first, or pass --matched-lob-root. "
            "Without it the merger falls back to the raw orderbook and fails "
            "with 'LOB is_bid is null'."
        )
        return 1

    outputs = run_merger.run_merge(
        root=args.root,
        symbol=args.symbol,
        date=args.date,
        hour=hour,
        lob_levels=args.lob_levels,
        source=args.source,
        matched_lob_root=args.matched_lob_root,
    )
    print(f"[SMOKE] {len(outputs)} sparse file(s) written, nothing finalized or deleted:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())