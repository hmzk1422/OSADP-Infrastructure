"""
pipeline.py — Master orchestrator linking Rust lob_replay and event_merger.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def resolve_binary(name: str, components_root: Path) -> Path:
    """Find release or debug binary executable."""
    ext = ".exe" if sys.platform == "win32" else ""
    release_path = components_root / "target" / "release" / f"{name}{ext}"
    if release_path.is_file():
        return release_path
    debug_path = components_root / "target" / "debug" / f"{name}{ext}"
    if debug_path.is_file():
        return debug_path
    raise FileNotFoundError(
        f"Could not find binary '{name}' in {components_root / 'target'}."
    )


def run_command(cmd: list[str | Path], description: str) -> None:
    """Execute a subprocess command, raising an error on failure."""
    cmd_str = [str(c) for c in cmd]
    logger.info("Executing: %s", " ".join(cmd_str))
    res = subprocess.run(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(
            "Command failed [%s]: %s\nSTDOUT: %s\nSTDERR: %s",
            description,
            res.returncode,
            res.stdout,
            res.stderr,
        )
        raise RuntimeError(f"Command '{description}' failed: {res.stderr or res.stdout}")
    if res.stdout:
        for line in res.stdout.strip().splitlines():
            logger.info("  %s", line)

def build_day_start_snapshot(
    lob_replay_bin: Path,
    ingest_dir: Path,
    matched_dir: Path,
    symbol: str,
    date: str,
    tick_size: str,
) -> Path:
    """Run lob_replay ingest-parquet in day_start mode on hour 00 orderbook file."""
    hour00 = ingest_dir / symbol / "orderbook" / date / "00.parquet"
    if not hour00.is_file():
        raise FileNotFoundError(f"Day start orderbook source not found: {hour00}")

    out_dir = matched_dir / symbol / "snapshots" / date
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        lob_replay_bin,
        "ingest-parquet",
        "--input", hour00,
        "--output-dir", out_dir,
        "--tick-size", tick_size,
        "--snapshot-policy", "day_start",
    ]
    run_command(cmd, f"LOB builder day_start snapshot {date}")
    return out_dir


def replay_orderflow_hour(
    lob_replay_bin: Path,
    input_file: Path,
    output_dir: Path,
    tick_size: str,
    warm_start: bool = True,
) -> Path:
    """Run lob_replay ingest-parquet in every_event mode to produce orderflow replay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        lob_replay_bin,
        "ingest-parquet",
        "--input", input_file,
        "--output-dir", output_dir,
        "--tick-size", tick_size,
        "--snapshot-policy", "every_event",
    ]
    if warm_start:
        cmd.append("--warm-start")
    run_command(cmd, f"LOB builder replay orderflow {input_file.name}")
    return output_dir


def clone_matched_sources(
    ingest_dir: Path,
    matched_dir: Path,
    symbol: str,
    date: str,
) -> None:
    """Clone aggTrades and bookDepth into matched source pipeline directory."""
    matched_source_dir = matched_dir / symbol / "_pipeline" / "source"
    matched_source_dir.mkdir(parents=True, exist_ok=True)

    for stream in ["aggTrades", "bookDepth"]:
        src_file = ingest_dir / symbol / stream / f"{date}.parquet"
        if not src_file.is_file():
            logger.warning("Ingest stream file not found: %s", src_file)
            continue
        dest_dir = matched_source_dir / stream
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{date}.parquet"
        if not dest_file.exists():
            logger.info("Cloning %s -> %s", src_file, dest_file)
            shutil.copy2(src_file, dest_file)
        else:
            logger.info("Target stream file already exists: %s", dest_file)


def run_merger(
    event_merger_bin: Path,
    market_data_root: Path,
    symbol: str,
    date: str,
    hour: int | None,
    lob_levels: int,
    orderflow_lob_root: Path,
) -> None:
    """Run event_merger to aggregate LOB levels, match timestamps, and write merged parquet."""
    cmd = [
        event_merger_bin,
        "--root", market_data_root,
        "--symbol", symbol,
        "--source", "matched",
        "--date", date,
        "--lob-levels", str(lob_levels),
        "--matched-lob-root", orderflow_lob_root,
    ]
    if hour is not None:
        cmd.extend(["--hour", str(hour)])
    else:
        cmd.extend(["--hour", "all"])

    run_command(cmd, f"Event merger {symbol} {date} (hour={hour if hour is not None else 'all'})")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    components_root = repo_root / "components"
    default_market_data = repo_root / "market_data"

    parser = argparse.ArgumentParser(
        description="Unified Orderflow LOB Replay & Event Merger Pipeline."
    )
    parser.add_argument("--root", type=Path, default=default_market_data, help="Market data root.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (e.g. BTCUSDT).")
    parser.add_argument("--date", type=str, required=True, help="UTC Date YYYY-MM-DD.")
    parser.add_argument("--hour", type=int, default=None, help="UTC Hour 0..23 (omitted = all).")
    parser.add_argument("--tick-size", type=str, default="0.1", help="Instrument tick size.")
    parser.add_argument("--lob-levels", type=int, default=20, help="Number of book levels.")
    parser.add_argument(
        "--existing-orderflow-lob",
        type=Path,
        default=None,
        help="Path to precomputed LOB parquet or directory.",
    )
    parser.add_argument(
        "--skip-day-start",
        action="store_true",
        help="Skip day-start snapshot generation.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=True,
        help="Use warm-start diff replay.",
    )

    args = parser.parse_args()

    market_data_root = args.root.resolve()
    ingest_dir = market_data_root / "ingest_parquet"
    matched_dir = market_data_root / "matched"

    lob_replay_bin = resolve_binary("lob_replay", components_root)
    event_merger_bin = resolve_binary("event_merger", components_root)

    logger.info("Pipeline: %s on %s (hour=%s)", args.symbol, args.date, args.hour)

    if not args.skip_day_start and args.existing_orderflow_lob is None:
        logger.info("--- Step 1: Day-start snapshot ---")
        build_day_start_snapshot(
            lob_replay_bin=lob_replay_bin,
            ingest_dir=ingest_dir,
            matched_dir=matched_dir,
            symbol=args.symbol,
            date=args.date,
            tick_size=args.tick_size,
        )

    logger.info("--- Step 2: Clone/verify matched sources ---")
    clone_matched_sources(
        ingest_dir=ingest_dir,
        matched_dir=matched_dir,
        symbol=args.symbol,
        date=args.date,
    )

    if args.existing_orderflow_lob is not None:
        logger.info("--- Step 3: Using precomputed Orderflow LOB: %s ---", args.existing_orderflow_lob)
        orderflow_input = args.existing_orderflow_lob
    else:
        logger.info("--- Step 3: Replaying orderbook to produce Orderflow LOB ---")
        hh = f"{args.hour:02}" if args.hour is not None else "00"
        orderbook_in = ingest_dir / args.symbol / "orderbook" / args.date / f"{hh}.parquet"
        orderflow_out = matched_dir / args.symbol / "_pipeline" / "orderflow" / args.date / hh
        replay_orderflow_hour(
            lob_replay_bin=lob_replay_bin,
            input_file=orderbook_in,
            output_dir=orderflow_out,
            tick_size=args.tick_size,
            warm_start=args.warm_start,
        )
        orderflow_input = orderflow_out

    logger.info("--- Step 4: Running Event Merger ---")
    run_merger(
        event_merger_bin=event_merger_bin,
        market_data_root=market_data_root,
        symbol=args.symbol,
        date=args.date,
        hour=args.hour,
        lob_levels=args.lob_levels,
        orderflow_lob_root=orderflow_input,
    )

    out_sparse = matched_dir / args.symbol / "sparse" / args.date
    logger.info("Pipeline complete! Output in %s", out_sparse)


if __name__ == "__main__":
    main()

