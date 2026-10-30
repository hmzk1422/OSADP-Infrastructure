"""
Snapshot / book-state builder commands (terminal-focused).

Step 1 -- 00-hour seeds in the matched work tree (same engine, day_start):

    python -m components.python.data_pipeline.matched.snapshots \\
        --root market_data --symbol BTCUSDT \\
        --start 2025-06-28 --end 2025-06-29 \\
        [--lob-levels 20] [--tick-size 0.01]

    For every orderbook/<DATE>/<HH>.parquet chunk this shells out to
    ``lob_replay columnar --snapshot-policy day_start`` and writes:

    * top-N book-state rows -> snapshots/book_state/<DATE>/<HH>.parquet
    * 00-UTC checkpoint images -> snapshots/day/<DATE>.parquet (resume seeds)

Step 3 -- per-row book-state inside the matched work tree (same engine, every_event):

    python -m components.python.data_pipeline.matched.snapshots \\
        --root market_data --symbol BTCUSDT --mode per-row ...

    Uses ``--snapshot-policy every_event`` (alias of every_snapshot) so every
    valid event emits a book-state row. Input chunks are the files copied by
    ``parquet.fetch_source_data()``; output keeps the legacy
    ``<SYM>_<DATE>_<DATE>_<HH>-<HH>.parquet`` naming so downstream readers
    keep working.

``lob_builder`` Rust code is treated as read-only: this module only selects
the ``--snapshot-policy`` parameter, it never reimplements replay.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import polars as pl

import components.pipelines.ingestion.CryptoL2.datapaths as paths
from components.pipelines.event_merger.matched import lob as _lob
from components.pipelines.event_merger.matched.parquet import pipeline_source_dir, pipeline_work_dir
from strategy_system.config.ingestion import config as cfg


SNAPSHOT_BOOK_STATE_DIRNAME = "book_state"
SNAPSHOT_DAY_DIRNAME = "day"

LOB_REPLAY_BIN = _lob.LOB_REPLAY_BIN
SNAPSHOT_POLICIES = _lob.SNAPSHOT_POLICIES


# ===========================================================
# internals
# ===========================================================

def _resolve_bin() -> Path:
    candidates = [
        LOB_REPLAY_BIN,
        Path("components/target/release/lob_replay.exe"),
        Path("components/target/debug/lob_replay.exe"),
        Path("components/target/release/lob_replay"),
        Path("components/target/debug/lob_replay"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"lob_replay binary not found; tried: {[str(c) for c in candidates]}. "
        "Build with: cargo build --manifest-path components/Cargo.toml "
        "--bin lob_replay --release"
    )


def _run_snapshot_policy(
    orderbook_file: Path,
    out_book_state: Path,
    snapshot_dir: Path | None,
    *,
    snapshot_policy: str,
    lob_levels: int,
    tick_size: float,
    warm_start: bool,
    state: _lob.LobReplayState | None = None,
) -> dict:
    """Run one orderbook chunk through lob_replay columnar; return stats."""
    fd, tmp_in_name = tempfile.mkstemp(suffix=".parquet")
    import os as _os
    _os.close(fd)
    tmp_in = Path(tmp_in_name)
    tmp_out: Path | None = None
    # NOTE: lob.build_multilevel_lob() already shells out to
    # `lob_replay columnar` with full resume/seed handling — reuse it so this
    # module only selects --snapshot-policy and never reimplements replay.
    # NOTE: pass the LazyFrame (not .collect()) so lob_columns() runs inside
    # the pipeline exactly as refresh_matched does; eager collect() here OOMs
    # on 6M-row hourly files because scan+collect materializes twice.
    try:
        import polars as _pl

        lob_lazy = _pl.scan_parquet(str(orderbook_file))
        book, _state, stats = _lob.build_multilevel_lob(
            lob_lazy,
            lob_levels=lob_levels,
            state=state,
            snapshot_policy=snapshot_policy,
            snapshot_dir=snapshot_dir,
            snapshot_levels=50,
            tick_size=tick_size,
            warm_start=warm_start,
        )
        # Preserve every raw LOB source row. The Rust replay supplies the
        # reconstructed book state; this join only adds source columns.
        source = _pl.scan_parquet(str(orderbook_file)).select([
            "event_time",
            "received_time",
            "transaction_time",
            "symbol",
            "event_type",
            "side",
            "price",
            "quantity",
            "first_update_id",
            "final_update_id",
            "prev_final_update_id",
            "last_update_id",
        ]).with_columns([
            _pl.col("event_type").cast(_pl.String).alias("source_event_type"),
            _pl.col("side").cast(_pl.String).alias("side"),
            _pl.col("price").cast(_pl.String).alias("source_price"),
            _pl.col("quantity").cast(_pl.String).alias("source_quantity"),
            (_pl.col("side") == "bid").alias("is_bid"),
            (_pl.col("event_type") == "snapshot").alias("is_snapshot"),
        ]).drop("event_type", "price", "quantity").collect()
        book = book.join(source, on="event_time", how="left")
        if book.height == 0:
            raise ValueError(f"lob_replay emitted 0 rows for {orderbook_file}")
        out_book_state.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: `src` and `out_book_state` are the SAME path in
        # per-row mode, so a partial write must never clobber the input.
        tmp_write = out_book_state.with_suffix(".tmp.parquet")
        book.write_parquet(tmp_write)
        _os.replace(tmp_write, out_book_state)
        return stats
    finally:
        tmp_in.unlink(missing_ok=True)
        if tmp_out is not None:
            tmp_out.unlink(missing_ok=True)


def _iter_orderbook_hours(
    root: Path, symbol: str, start: str, end: str,
) -> list[tuple[str, str, Path]]:
    base = paths.parquet_dir(root, symbol, "orderbook")
    found: list[tuple[str, str, Path]] = []
    for day in paths.orderbook_parquet_days(root, symbol):
        if not (start <= day <= end):
            continue
        for hour in range(24):
            f = base / day / f"{hour:02d}.parquet"
            if f.exists():
                found.append((day, f"{hour:02d}", f))
    return found


def build_day_start_snapshots(
    *,
    root: Path | str | None = None,
    symbol: str = paths.DEFAULT_SYMBOL,
    start: str,
    end: str,
    lob_levels: int = 20,
    tick_size: float = _lob.LOB_TICK_SIZE,
    warm_start: bool = True,
) -> dict:
    """Step 1: read canonical L2 and write a 00-hour seed under matched work."""
    root_p = Path(root) if root is not None else paths.MARKET_DATA_ROOT
    symbol = symbol.upper()
    work = pipeline_work_dir(root_p, symbol)
    day_seed_dir = work / "day_seed"
    totals = {"days": 0, "emitted_rows": 0}
    for day in paths.orderbook_parquet_days(root_p, symbol):
        if not (start <= day <= end):
            continue
        src = paths.parquet_chd_file(root_p, symbol, day, 0)
        if not src.exists():
            raise FileNotFoundError(f"missing canonical LOB hour 00: {src}")
        out_book = work / "day_seed_state" / day / "00.parquet"
        # Never let a seed from a previous run make a day with no replayable
        # snapshot look healthy.
        (day_seed_dir / f"{day}.parquet").unlink(missing_ok=True)
        out_book.unlink(missing_ok=True)
        stats = _run_snapshot_policy(
            src,
            out_book,
            day_seed_dir,
            snapshot_policy=_lob.SNAPSHOT_POLICY_DAY_START,
            lob_levels=lob_levels,
            tick_size=tick_size,
            warm_start=warm_start,
        )
        seed_file = day_seed_dir / f"{day}.parquet"
        if not seed_file.exists():
            raise RuntimeError(
                f"day-start seed was not created for {day}; hour 00 contains no usable snapshot "
                f"(emitted_rows={stats.get('emitted_rows', 0)}, malformed={stats.get('malformed_events', 0)})"
            )
        totals["days"] += 1
        totals["emitted_rows"] += int(stats.get("emitted_rows", 0))
    if totals["days"] == 0:
        raise FileNotFoundError(f"no canonical LOB days in [{start}, {end}]")
    return totals


def build_per_row_book_state(
    *,
    root: Path | str | None = None,
    symbol: str = paths.DEFAULT_SYMBOL,
    start: str,
    end: str,
    lob_levels: int = 20,
    tick_size: float = _lob.LOB_TICK_SIZE,
    warm_start: bool = True,
) -> dict:
    """Step 3: materialize per-source-event LOB state inside matched work."""
    root_p = Path(root) if root is not None else paths.MARKET_DATA_ROOT
    symbol = symbol.upper()
    work = pipeline_work_dir(root_p, symbol)
    source_root = pipeline_source_dir(root_p, symbol, "orderbook")
    totals = {"hours": 0, "emitted_rows": 0}
    for day in paths.orderbook_parquet_days(root_p, symbol):
        if not (start <= day <= end):
            continue
        state: _lob.LobReplayState | None = None
        for hour in range(24):
            src = source_root / day / f"{hour:02d}.parquet"
            if not src.exists():
                continue
            if hour > 0:
                seed = _lob.resume_from_day_snapshot(
                    day, snapshot_dir=work / "day_seed"
                )
                if seed is None:
                    raise FileNotFoundError(f"missing day seed for {day}")
                seed.skip_until = max(
                    seed.skip_until, state.skip_until if state else 0
                )
                state = seed
            out_book = work / "lob_state" / day / f"{hour:02d}.parquet"
            out_book.unlink(missing_ok=True)
            stats = _run_snapshot_policy(
                src,
                out_book,
                None,
                snapshot_policy=_lob.SNAPSHOT_POLICY_EVERY_EVENT,
                lob_levels=lob_levels,
                tick_size=tick_size,
                warm_start=warm_start,
                state=state,
            )
            book = pl.read_parquet(out_book)
            if book.height:
                state = _lob.LobReplayState(
                    skip_until=int(book["event_time"][-1]),
                    seed_snapshot_path=(
                        (work / "day_seed" / f"{day}.parquet")
                        if hour > 0
                        else None
                    ),
                )
            totals["hours"] += 1
            totals["emitted_rows"] += int(stats.get("emitted_rows", 0))
    if totals["hours"] == 0:
        raise FileNotFoundError(
            f"no work-tree orderbook files in [{start}, {end}] under {source_root}"
        )
    return totals


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build LOB snapshots/book-state.")
    parser.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument("--mode", choices=["day_start", "per-row"], default="day_start")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lob-levels", type=int, default=20)
    parser.add_argument("--tick-size", type=float, default=_lob.LOB_TICK_SIZE)
    parser.add_argument("--no-warm-start", action="store_true")
    args = parser.parse_args(argv)

    days = paths.orderbook_parquet_days(args.root, args.symbol)
    start = args.start or (days[0] if days else None)
    end = args.end or (days[-1] if days else None)
    if not start or not end:
        print("no orderbook days available")
        return 1

    warm_start = not args.no_warm_start
    if args.mode == "day_start":
        totals = build_day_start_snapshots(
            root=args.root, symbol=args.symbol, start=start, end=end,
            lob_levels=args.lob_levels, tick_size=args.tick_size,
            warm_start=warm_start,
        )
    else:
        totals = build_per_row_book_state(
            root=args.root, symbol=args.symbol, start=start, end=end,
            lob_levels=args.lob_levels, tick_size=args.tick_size,
            warm_start=warm_start,
        )
    print(f"done {args.mode}: {totals}")
    _ = cfg
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
