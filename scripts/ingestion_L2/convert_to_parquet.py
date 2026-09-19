#!/usr/bin/env python3
"""
convert_to_parquet.py
======================

Post-process step after running ofp_data_downloader.py.

Takes the *native* raw tree it produces:

    <data_root>/
      binance/<SYMBOL>/aggTrades/<DATE>/<SYMBOL>-aggTrades-<DATE>.zip
      binance/<SYMBOL>/bookDepth/<DATE>/<SYMBOL>-bookDepth-<DATE>.zip
      cryptohftdata/<SYMBOL>/orderbook/<DATE>/<HH>/<SYMBOL>_orderbook.parquet

...and normalizes everything into a single, uniform Parquet tree so a
backtester can pull all three datasets with one code path:

    <data_root>/parquet/<SYMBOL>/aggTrades/<DATE>.parquet
    <data_root>/parquet/<SYMBOL>/bookDepth/<DATE>.parquet
    <data_root>/parquet/<SYMBOL>/orderbook/<DATE>/<HH>.parquet

Design choices
--------------
- By default this is a MOVE, not a copy: once a raw file (Binance ZIP or a
  CryptoHFTData hourly Parquet) is successfully converted/verified into the
  unified <data_root>/parquet/ tree, the raw source file is deleted and its
  now-empty date/hour directory is cleaned up. There's no reason to keep the
  ZIP around once the Parquet it contains has been extracted and verified.
  Pass --keep-raw if you want the old copy-only behavior (e.g. for an
  archival/audit copy) instead of deleting sources.
- No schema is invented: Binance's own published column names are used
  as-is (see BINANCE_SCHEMAS below). CryptoHFTData files are already
  Parquet and are copied through unchanged (just re-homed into the unified
  tree) unless --recompress is passed, in which case they're rewritten with
  the same compression/codec as the Binance output for consistent file
  sizes and read performance.
- Idempotent / resumable: existing output files are skipped unless
  --overwrite is passed.
- Dtypes are set explicitly and conservatively (int64 for ids/timestamps,
  float64 for price/qty/notional, bool for is_buyer_maker) so the backtest
  gets a stable schema across every daily file instead of pandas guessing
  per-file.

Usage
-----
    pip install pandas pyarrow

    python convert_to_parquet.py --root ./research/data_v2 --symbol BTCUSDT

    # re-run safely any time after new files land, only new work is done:
    python convert_to_parquet.py --root ./research/data_v2 --symbol BTCUSDT

    # force everything to be rebuilt:
    python convert_to_parquet.py --root ./research/data_v2 --symbol BTCUSDT --overwrite

    # keep the raw ZIPs / raw CHD parquet around instead of deleting them:
    python convert_to_parquet.py --root ./research/data_v2 --symbol BTCUSDT --keep-raw
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install pandas first: pip install pandas") from exc

try:
    import pyarrow  # noqa: F401  (import check only; pandas uses it under the hood)
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install pyarrow first: pip install pyarrow") from exc

# FIX (2026-09-18): pyarrow.parquet is used for footer-metadata-only reads
# (see _parquet_num_rows/_verify_parquet) so conversion verifies row counts
# without materializing 6-7M-row DataFrames per file.
import pyarrow.parquet as pq


# Binance's published native column schemas for these two datasets.
# aggTrades: https://github.com/binance/binance-public-data  (futures aggTrades)
# bookDepth: same repo, futures bookDepth dataset (depth/notional by % from mid)
BINANCE_SCHEMAS = {
    "aggTrades": {
        "columns": [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ],
        "dtypes": {
            "agg_trade_id": "int64",
            "price": "float64",
            "quantity": "float64",
            "first_trade_id": "int64",
            "last_trade_id": "int64",
            "transact_time": "int64",
            "is_buyer_maker": "bool",
        },
    },
    "bookDepth": {
        # FIX (2026-09-18): Binance ships bookDepth `timestamp` as an ISO
        # datetime string (e.g. "2025-06-28 00:00:07"), NOT an epoch int.
        # Casting it straight to int64 raised "invalid literal for int() with
        # base 10" on EVERY file, so no bookDepth parquet was ever produced.
        # read_binance_zip_csv now normalizes it to UTC epoch milliseconds
        # (int64) so the column matches aggTrades.transact_time and
        # CryptoHFTData event_time conventions.
        "columns": ["timestamp", "percentage", "depth", "notional"],
        "dtypes": {
            "timestamp": "int64",
            "percentage": "int64",
            "depth": "float64",
            "notional": "float64",
        },
    },
}


def read_binance_zip_csv(zip_path: Path, dataset: str) -> pd.DataFrame:
    """Read the single CSV inside a Binance daily ZIP into a typed DataFrame."""
    schema = BINANCE_SCHEMAS[dataset]
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected exactly one member in {zip_path}, found {names}")
        with zf.open(names[0]) as fh:
            raw = fh.read()

    # Binance files sometimes ship with a header row and sometimes without,
    # depending on date range. Detect by checking whether the first token of
    # the first line parses as a number (real data) or not (a header).
    first_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    first_token = first_line.split(",", 1)[0].strip()
    has_header = not _looks_numeric(first_token)

    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=schema["columns"],
    )

    # FIX (2026-09-18, bookDepth): the CSV `timestamp` column is an ISO
    # datetime string, so it must be parsed before the int64 cast below.
    # Normalized to UTC epoch MILLISECONDS (matching aggTrades.transact_time
    # and CryptoHFTData event_time). The intermediate datetime64[ms] cast is
    # required because pandas infers the datetime resolution from the string
    # (second-resolution strings yield datetime64[s]/[us], whose raw int64
    # values are seconds/microseconds, not nanoseconds). If a future file
    # ever ships numeric epoch values instead, they are kept as-is.
    if dataset == "bookDepth" and not pd.api.types.is_integer_dtype(df["timestamp"]):
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df["timestamp"] = ts.astype("datetime64[ms]").astype("int64")

    df = df.astype(schema["dtypes"])
    return df


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def convert_binance_dataset(
    *, root: Path, symbol: str, dataset: str, overwrite: bool, delete_raw: bool, verbose: bool
) -> tuple[int, int, int]:
    """Convert every daily ZIP for one Binance dataset. Returns (done, skipped, failed)."""
    src_dir = root / "binance" / symbol / dataset
    dst_dir = root / "parquet" / symbol / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    if not src_dir.exists():
        return done, skipped, failed

    for date_dir in sorted(src_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        day = date_dir.name
        zips = list(date_dir.glob("*.zip"))
        if not zips:
            continue
        zip_path = zips[0]
        out_path = dst_dir / f"{day}.parquet"

        if out_path.exists() and not overwrite:
            # Already converted on a prior run. If the raw ZIP is still here
            # (e.g. this run added --delete-raw after a --keep-raw run),
            # clean it up too so state stays consistent.
            if delete_raw and zip_path.exists():
                _delete_raw_file(zip_path)
            skipped += 1
            continue

        try:
            df = read_binance_zip_csv(zip_path, dataset)
            tmp = out_path.with_suffix(".parquet.part")
            df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)

            # Verify the Parquet we just wrote is readable and has the row
            # count we expect BEFORE touching the raw source. Only delete
            # the ZIP once we've proven the Parquet is good.
            _verify_parquet(tmp, expected_rows=len(df))
            tmp.replace(out_path)

            done += 1
            if verbose:
                print(f"[{dataset}] {day} -> {out_path} ({len(df):,} rows)")

            if delete_raw:
                _delete_raw_file(zip_path)

        except Exception as exc:
            failed += 1
            print(f"[{dataset}] {day} -> FAILED: {exc!r}", file=sys.stderr)

    return done, skipped, failed


def _parquet_num_rows(path: Path) -> int:
    """Row count from Parquet footer metadata (footer-only read, near-instant)."""
    return pq.ParquetFile(path).metadata.num_rows


def _verify_parquet(path: Path, *, expected_rows: int) -> None:
    """Confirm a Parquet file is valid and has the expected row count.

    FIX (2026-09-18): reads footer metadata only (pyarrow) instead of fully
    materializing the frame with pandas. The old implementation re-read every
    row of every file, which made converting the ~10k CryptoHFTData hourly
    files (6-7M rows each) take on the order of days.
    """
    rows = _parquet_num_rows(path)
    if rows != expected_rows:
        raise ValueError(f"Row count mismatch after write: expected {expected_rows}, got {rows}")


def _delete_raw_file(path: Path) -> None:
    """Delete a raw source file and prune its now-empty parent date/hour dirs."""
    path.unlink(missing_ok=True)
    # Walk up removing empty directories (date dir, and hour dir for CHD),
    # but never touch anything above the provider root.
    parent = path.parent
    while parent.exists() and not any(parent.iterdir()):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def convert_chd_dataset(
    *, root: Path, symbol: str, overwrite: bool, recompress: bool, delete_raw: bool, verbose: bool
) -> tuple[int, int, int]:
    """Re-home (or recompress) CryptoHFTData hourly Parquet files."""
    src_dir = root / "cryptohftdata" / symbol / "orderbook"
    dst_dir = root / "parquet" / symbol / "orderbook"
    dst_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    if not src_dir.exists():
        return done, skipped, failed

    for date_dir in sorted(src_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        day = date_dir.name
        out_day_dir = dst_dir / day
        out_day_dir.mkdir(parents=True, exist_ok=True)

        for hour_dir in sorted(date_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            hour = hour_dir.name

            # FIX (2026-09-18): prefer the canonical per-hour file and ignore
            # stray artifacts (e.g. "*.test.parquet" or leftover ".part"
            # partial downloads) that could previously be picked blindly as
            # src_files[0] and fail the conversion for that hour.
            src_path = hour_dir / f"{symbol}_orderbook.parquet"
            if not src_path.exists():
                src_files = list(hour_dir.glob("*.parquet"))
                src_path = src_files[0] if src_files else None
            if src_path is None:
                continue

            out_path = out_day_dir / f"{hour}.parquet"

            if out_path.exists() and not overwrite:
                if delete_raw and src_path.exists():
                    _delete_raw_file(src_path)
                skipped += 1
                continue

            try:
                # FIX (2026-09-18): verify via Parquet footer metadata instead
                # of two full pandas reads (see _verify_parquet).
                src_rows = _parquet_num_rows(src_path)

                if recompress:
                    df = pd.read_parquet(src_path, engine="pyarrow")
                    tmp = out_path.with_suffix(".parquet.part")
                    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
                    _verify_parquet(tmp, expected_rows=len(df))
                    tmp.replace(out_path)
                elif delete_raw:
                    # FIX (2026-09-18): when the raw file is going to be
                    # deleted anyway, use an atomic same-volume rename instead
                    # of a full read+write copy. This turns converting the
                    # ~250GB CryptoHFTData tree into metadata-only renames.
                    tmp = out_path.with_suffix(".parquet.part")
                    try:
                        src_path.replace(tmp)
                    except OSError:
                        # Cross-volume fallback: copy then remove the source.
                        tmp.write_bytes(src_path.read_bytes())
                        src_path.unlink(missing_ok=True)
                    _verify_parquet(tmp, expected_rows=src_rows)
                    tmp.replace(out_path)
                else:
                    # --keep-raw: verify+copy instead of round-tripping
                    # through pandas for nothing.
                    tmp = out_path.with_suffix(".parquet.part")
                    tmp.write_bytes(src_path.read_bytes())
                    _verify_parquet(tmp, expected_rows=src_rows)
                    tmp.replace(out_path)

                done += 1
                if verbose:
                    print(f"[orderbook] {day} {hour} -> {out_path}")

                if delete_raw:
                    _delete_raw_file(src_path)

            except Exception as exc:
                failed += 1
                print(f"[orderbook] {day} {hour} -> FAILED: {exc!r}", file=sys.stderr)

    return done, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize native OFMR v2 raw data into one Parquet tree.")
    parser.add_argument("--root", required=True, help="Same --root used by orderflow_downloader.py")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild Parquet files that already exist")
    parser.add_argument(
        "--recompress",
        action="store_true",
        help="Also rewrite CryptoHFTData files with zstd (default: move through unchanged, faster)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw ZIP / raw CHD Parquet sources after a verified conversion (default: delete them)",
    )
    parser.add_argument(
        "--datasets",
        default="aggTrades,bookDepth,orderbook",
        help="Comma-separated subset of datasets to convert "
             "(default: all of aggTrades,bookDepth,orderbook)",
    )
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    symbol = args.symbol.upper()
    delete_raw = not args.keep_raw

    # FIX (2026-09-18): allow running a single dataset so a targeted re-run
    # (e.g. just bookDepth) does not touch the others.
    datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}
    unknown = datasets - {"aggTrades", "bookDepth", "orderbook"}
    if unknown:
        parser.error(f"Unknown dataset(s): {sorted(unknown)}")

    totals = {"done": 0, "skipped": 0, "failed": 0}
    for dataset in ("aggTrades", "bookDepth"):
        if dataset not in datasets:
            continue
        d, s, f = convert_binance_dataset(
            root=root,
            symbol=symbol,
            dataset=dataset,
            overwrite=args.overwrite,
            delete_raw=delete_raw,
            verbose=args.verbose,
        )
        totals["done"] += d
        totals["skipped"] += s
        totals["failed"] += f

    if "orderbook" in datasets:
        d, s, f = convert_chd_dataset(
            root=root,
            symbol=symbol,
            overwrite=args.overwrite,
            recompress=args.recompress,
            delete_raw=delete_raw,
            verbose=args.verbose,
        )
        totals["done"] += d
        totals["skipped"] += s
        totals["failed"] += f

    print(
        f"\nDone. converted={totals['done']:,} skipped_existing={totals['skipped']:,} "
        f"failed={totals['failed']:,}"
    )
    print(f"Unified Parquet tree: {root / 'parquet' / symbol}")
    if totals["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()