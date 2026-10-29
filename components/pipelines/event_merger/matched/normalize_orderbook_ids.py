"""
normalize_orderbook_ids.py
==========================
Make CHD `orderbook` frames satisfy `lob_replay`'s update-ID continuity check.

Why this exists
---------------
`lob_replay` (btree_ver/replay.rs::process_update) flags a *gap* when any of:

    prev_final_update_id != last_final_update_id
    first_update_id       > last_final_update_id + 1
    final_update_id       < last_final_update_id + 1

Under `--warm-start` a flagged gap does `book.clear()`.

Measured on `ingest_parquet/BTCUSDT/orderbook/2025-06-30/00.parquet`:

    prev_final_update_id == previous.final_update_id   -> 35,233 / 35,245 batches  (healthy)
    first_update_id      >  previous.final_update_id+1 -> 34,567 / 35,258 batches  (false positive)

So the ID *chain* is intact, but `first_update_id` jumps ahead of it, which the
engine reads as a gap.  Result: the book was cleared on ~98% of frames, leaving
21% of LOB rows with fewer than 20 levels and spreads up to $1133 (true spread
is ~$0.10), plus 14,347 rows with fewer than 20 ask levels.

What this does
--------------
Rewrites, for rows that carry `prev_final_update_id`:

    first_update_id := prev_final_update_id + 1

The chain the engine validates (`prev_final_update_id`) is left untouched; only
the redundant lower bound is made consistent with it, so all three gap
conditions pass for genuinely chained frames.  Snapshot rows
(`prev_final_update_id` is null) are left exactly as they are.

This normalizes the *input* only.  `components/rust/lob_builder` is read-only
and is never modified; `ingest_parquet` is never written to.

Usage
-----
    python components/python/normalize_orderbook_ids.py \
        --root market_data --symbol BTCUSDT --date 2025-06-30 \
        --out market_data/matched/BTCUSDT/_pipeline/orderbook_norm

    # single hour
    python components/python/normalize_orderbook_ids.py \
        --root market_data --symbol BTCUSDT --date 2025-06-30 --hours 0 \
        --out <dir>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

logger = logging.getLogger("normalize_orderbook_ids")

_ID_COLUMNS = (
    "first_update_id",
    "final_update_id",
    "prev_final_update_id",
)


def parse_hours(raw: Optional[str]) -> list[int]:
    """Parse `0`, `0-7`, `0,3,5`, or `all` into a sorted hour list."""
    if raw is None or str(raw).strip().lower() in {"", "all"}:
        return list(range(24))
    hours: set[int] = set()
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if not (0 <= start <= 23 and 0 <= end <= 23 and start <= end):
                raise ValueError(f"bad hour range: {chunk!r}")
            hours.update(range(start, end + 1))
        else:
            hour = int(chunk)
            if not 0 <= hour <= 23:
                raise ValueError(f"bad hour: {chunk!r}")
            hours.add(hour)
    if not hours:
        raise ValueError("no hours selected")
    return sorted(hours)


def _gap_expressions() -> list[pl.Expr]:
    """The three forward-jump counters we care about, as lazy expressions."""
    expected = pl.col("prev_final_update_id") + 1
    return [
        pl.len().alias("rows"),
        pl.col("prev_final_update_id").is_not_null().sum().alias("rows_with_chain"),
        (pl.col("first_update_id") > expected).fill_null(False).sum().alias("jump_forward"),
        (pl.col("first_update_id") < expected).fill_null(False).sum().alias("jump_backward"),
        (pl.col("first_update_id") == expected).fill_null(False).sum().alias("chain_exact"),
    ]


def _stats(path: Path) -> dict[str, int]:
    """Compute the jump counters for one orderbook parquet."""
    row = (
        pl.scan_parquet(str(path))
        .select(_gap_expressions())
        .collect()
        .row(0, named=True)
    )
    return {key: int(value) for key, value in row.items()}


def normalize_hour(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Rewrite `first_update_id` on one orderbook hour file.

    Returns a stats dict with before/after jump counters.
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"orderbook source not found: {src}")
    if src == dst:
        raise ValueError("refusing to normalize in place (would overwrite the source)")
    if dst.exists() and not overwrite:
        after = _stats(dst)
        return {
            "src": str(src),
            "dst": str(dst),
            "status": "exists",
            "before": None,
            "after": after,
        }

    before = _stats(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(f".{dst.name}.tmp.parquet")
    if temp.exists():
        temp.unlink()

    try:
        # Keep Arrow's ordinary UTF-8 strings. Polars writes String columns as
        # LargeUtf8; lob_replay's Arrow reader expects Utf8 and rejects those
        # normalized files. A direct Arrow rewrite preserves the input schema.
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        table = pq.read_table(src)
        first_idx = table.schema.get_field_index("first_update_id")
        prev = table.column("prev_final_update_id")
        first = table.column("first_update_id")
        expected = pc.add(prev, pa.scalar(1, type=prev.type))
        normalized = pc.if_else(pc.is_valid(prev), expected, first)
        table = table.set_column(first_idx, table.schema.field(first_idx), normalized)
        # CryptoHFTData orderbook files are grouped by price bands rather than
        # exchange event time. Replay requires every event's price levels to
        # be contiguous and chronological; order update batches by final ID.
        table = table.sort_by([("final_update_id", "ascending"), ("event_time", "ascending")])
        pq.write_table(table, temp, compression="zstd", use_dictionary=False)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    temp.replace(dst)

    after = _stats(dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "status": "written",
        "before": before,
        "after": after,
    }


def orderbook_hour_path(root: Path, symbol: str, date: str, hour: int) -> Path:
    """`<root>/ingest_parquet/<SYM>/orderbook/<date>/<hh>.parquet` (read-only)."""
    return (
        Path(root)
        / "ingest_parquet"
        / symbol.upper()
        / "orderbook"
        / date
        / f"{hour:02d}.parquet"
    )


def normalize_hours(
    *,
    root: Path,
    symbol: str,
    date: str,
    hours: Iterable[int],
    out_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Normalize the requested hours of one UTC date."""
    results: list[dict[str, object]] = []
    for hour in sorted(hours):
        src = orderbook_hour_path(root, symbol, date, hour)
        if not src.is_file():
            logger.warning("skipping missing hour %02d: %s", hour, src)
            continue
        dst = Path(out_dir) / date / f"{hour:02d}.parquet"
        results.append(normalize_hour(src, dst, overwrite=overwrite))
    return results


def _print_result(result: dict[str, object]) -> None:
    before = result.get("before")
    after = result.get("after")
    logger.info("%s -> %s [%s]", result["src"], result["dst"], result["status"])
    if before:
        logger.info(
            "  before: jump_forward=%s jump_backward=%s chain_exact=%s rows=%s",
            before["jump_forward"], before["jump_backward"], before["chain_exact"], before["rows"],
        )
    if after:
        logger.info(
            "  after : jump_forward=%s jump_backward=%s chain_exact=%s rows=%s",
            after["jump_forward"], after["jump_backward"], after["chain_exact"], after["rows"],
        )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Normalize orderbook update IDs so lob_replay stops reporting false gaps."
    )
    parser.add_argument("--root", default="market_data", help="market_data root (read-only)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", required=True, help="UTC date YYYY-MM-DD")
    parser.add_argument("--hours", default="all", help="e.g. 0, 0-7, or all")
    parser.add_argument("--out", required=True, help="output directory (work tree)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    hours = parse_hours(args.hours)

    ingest_root = (root / "ingest_parquet" / args.symbol.upper()).resolve()
    try:
        out_dir.relative_to(ingest_root)
    except ValueError:
        pass
    else:
        raise SystemExit(f"refusing to write inside the read-only ingest tree: {out_dir}")

    results = normalize_hours(
        root=root,
        symbol=args.symbol,
        date=args.date,
        hours=hours,
        out_dir=out_dir,
        overwrite=bool(args.overwrite),
    )
    if not results:
        raise SystemExit(f"no orderbook hours found for {args.date} under {ingest_root}")
    for result in results:
        _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

