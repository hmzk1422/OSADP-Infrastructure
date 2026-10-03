#!/usr/bin/env python3
"""
orderflow price data downloader L2 native market-data downloader.

Purpose
-------
Download the three raw datasets used by OFP v2 without converting one source
into another schema or inventing bid/ask quotes:

1) Binance USD-M Futures aggTrades        -> native Binance ZIP/CSV
2) Binance USD-M Futures bookDepth         -> native Binance ZIP/CSV
3) CryptoHFTData Binance Futures L2        -> native hourly Parquet files

layout used (see datapaths.py; --root names the market_data directory)
---------------------------------------------------------------------
<root>/
  ingest_source/
    manifests/BTCUSDT_manifest.json
    BTCUSDT/
      aggTrades/YYYY-MM-DD/BTCUSDT-aggTrades-YYYY-MM-DD.zip
      bookDepth/YYYY-MM-DD/BTCUSDT-bookDepth-YYYY-MM-DD.zip
      orderbook/YYYY-MM-DD/HH/BTCUSDT_orderbook.parquet

Files are grouped by DATA TYPE under the ticker, never by provider.

The Binance public archive is daily. CryptoHFTData is hourly.

Security
--------
Pass the CryptoHFTData API key via --cryptohft-api-key or the
CRYPTOHFTDATA_API_KEY environment variable. Do not hard-code credentials.

Notes
-----
- Binance 404s are skipped (e.g. missing data/symbol-date).
- CryptoHFTData currently exposes Binance orderbook history from 2025-06-28.
- CryptoHFTData anonymous/free REST access is rate-limited; this script uses a
  modest worker count and retries 429 responses with Retry-After/backoff.
- Files are written atomically via .part files.
- A manifest records status, paths, byte sizes and errors so interrupted runs
  can be resumed safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
import zipfile
import polars as pl
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zstandard as zstd

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install requests first: pip install requests") from exc

import components.pipelines.ingestion.CryptoL2.datapaths as paths


BINANCE_BASE = "https://data.binance.vision"
CHD_BASE = "https://api.cryptohftdata.com/v1"
CHD_MIN_DATE = date(2025, 6, 28)


@dataclass
class Result:
    provider: str
    dataset: str
    date: str
    hour: Optional[int]
    url: str
    path: str
    status: str
    bytes: int = 0
    error: Optional[str] = None


def parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {s}") from exc


def daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# FIX (2026-09-18): allow targeting a subset of datasets so a re-run does not
# needlessly re-download datasets whose parquet outputs already exist.
VALID_DATASETS = ("aggTrades", "bookDepth", "orderbook")


def parse_datasets(raw: Optional[str]) -> set:
    """Parse a comma-separated --datasets value (None/empty = all)."""
    if not raw:
        return set(VALID_DATASETS)
    selected = {d.strip() for d in raw.split(",") if d.strip()}
    unknown = selected - set(VALID_DATASETS)
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {sorted(unknown)}. Valid: {list(VALID_DATASETS)}"
        )
    return selected or set(VALID_DATASETS)


def atomic_stream_download(
    url: str,
    dest: Path,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 90,
    retries: int = 5,
) -> tuple[str, int]:
    """Download URL atomically. Returns (status, bytes)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "cached", dest.stat().st_size

    part = dest.with_suffix(dest.suffix + ".part")
    last: Optional[BaseException] = None

    for attempt in range(retries + 1):
        try:
            req = Request(
                url,
                headers=headers or {"User-Agent": "OFMR-v2/1.0"},
                method="GET",
            )
            with urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)

            if part.stat().st_size == 0:
                raise IOError(f"Empty download: {url}")

            os.replace(part, dest)
            return "downloaded", dest.stat().st_size

        except HTTPError:
            # 404 and other HTTP errors are intentionally handled by the caller.
            raise
        except (URLError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
            last = exc
            if attempt >= retries:
                break
            sleep_s = min(30.0, (2 ** attempt) + random.random())
            time.sleep(sleep_s)

    raise RuntimeError(f"Download failed after {retries + 1} attempts: {url}; {last}")


def binance_url(symbol: str, dataset: str, day: date, market: str = "um") -> str:
    symbol = symbol.upper()
    return (
        f"{BINANCE_BASE}/data/futures/{market}/daily/{dataset}/{symbol}/"
        f"{symbol}-{dataset}-{day.isoformat()}.zip"
    )


def download_binance_file(
    *,
    symbol: str,
    dataset: str,
    day: date,
    root: Path,
    market: str,
    verbose: bool,
) -> Result:
    url = binance_url(symbol, dataset, day, market)
    # ingest_source/<TICKER>/<dataset>/<DATE>/<TICKER>-<dataset>-<DATE>.zip
    dest = paths.source_binance_file(root, symbol, dataset, day)
    try:
        status, n = atomic_stream_download(url, dest)
        # Validate cached/downloaded ZIPs so a stale/corrupt file is never accepted.
        if not zipfile.is_zipfile(dest):
            dest.unlink(missing_ok=True)
            raise ValueError(f"File is not a valid ZIP: {dest}")
        if verbose:
            print(f"[BINANCE {dataset}] {day} -> {status} ({n:,} bytes)")
        return Result("binance", dataset, day.isoformat(), None, url, str(dest), status, n)
    except HTTPError as exc:
        if exc.code == 404:
            if verbose:
                print(f"[BINANCE {dataset}] {day} -> 404, skipped")
            return Result("binance", dataset, day.isoformat(), None, url, str(dest), "missing", 0)
        return Result("binance", dataset, day.isoformat(), None, url, str(dest), "error", 0, f"HTTP {exc.code}")
    except Exception as exc:
        return Result("binance", dataset, day.isoformat(), None, url, str(dest), "error", 0, repr(exc))


def chd_url(day: date, hour: int, symbol: str) -> str:
    file_path = f"binance_futures/{day.isoformat()}/{hour:02d}/{symbol.upper()}_orderbook.parquet"
    return f"{CHD_BASE}/download?file={file_path}"

def download_chd_file(
    *,
    symbol: str,
    day: date,
    hour: int,
    root: Path,
    api_key: Optional[str],
    verbose: bool,
) -> Result:
    url = chd_url(day, hour, symbol)

    dest = paths.source_chd_file(root, symbol, day, hour)

    if dest.exists() and dest.stat().st_size > 0:
        return Result(
            "cryptohftdata",
            "orderbook",
            day.isoformat(),
            hour,
            url,
            str(dest),
            "cached",
            dest.stat().st_size,
        )

    params = {
        "file": (
            f"binance_futures/{day.isoformat()}/"
            f"{hour:02d}/{symbol.upper()}_orderbook.parquet"
        )
    }

    if api_key:
        params["api_key"] = api_key

    headers = {"User-Agent": "OFMR-v2/1.0"}

    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part = dest.with_suffix(dest.suffix + ".part")

    session = requests.Session()
    last_error: Optional[str] = None

    for attempt in range(6):
        try:
            resp = session.get(
                f"{CHD_BASE}/download",
                params=params,
                headers=headers,
                timeout=180,
                stream=True,
            )

            if resp.status_code == 404:
                return Result(
                    "cryptohftdata",
                    "orderbook",
                    day.isoformat(),
                    hour,
                    resp.url,
                    str(dest),
                    "missing",
                    0,
                )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after
                    else min(60.0, 2.0 ** attempt)
                )
                time.sleep(wait + random.random())
                continue

            resp.raise_for_status()

            # CryptoHFT returns Zstandard-compressed Parquet.
            dctx = zstd.ZstdDecompressor()

            with open(part, "wb") as fh:
                with dctx.stream_reader(resp.raw) as reader:
                    while True:
                        chunk = reader.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)

            n = part.stat().st_size

            if n == 0:
                raise IOError("Empty Parquet download")

            # Validate the decompressed file before replacing the final file.
            pl.read_parquet_schema(part)

            os.replace(part, dest)

            if verbose:
                print(
                    f"[CHD L2] {day} {hour:02d}:00 "
                    f"-> downloaded ({n:,} bytes)"
                )

            return Result(
                "cryptohftdata",
                "orderbook",
                day.isoformat(),
                hour,
                resp.url,
                str(dest),
                "downloaded",
                n,
            )

        except requests.HTTPError as exc:
            last_error = (
                f"HTTP "
                f"{exc.response.status_code if exc.response is not None else '?'}"
            )

            if (
                exc.response is not None
                and 400 <= exc.response.status_code < 500
                and exc.response.status_code != 429
            ):
                break

        except (
            requests.RequestException,
            OSError,
            ValueError,
            zstd.ZstdError,
            pl.exceptions.PolarsError,
        ) as exc:
            last_error = repr(exc)

        part.unlink(missing_ok=True)

        time.sleep(
            min(
                30.0,
                (2 ** attempt) + random.random(),
            )
        )

    return Result(
        "cryptohftdata",
        "orderbook",
        day.isoformat(),
        hour,
        url,
        str(dest),
        "error",
        0,
        last_error,
    )

def write_manifest(root: Path, symbol: str, results: list[Result], args: argparse.Namespace) -> None:
    paths.ensure(paths.manifest_dir(root))
    out = paths.manifest_file(root, symbol)

    # FIX (2026-09-18): merge with the existing manifest instead of clobbering
    # it, so a partial re-run (e.g. --datasets bookDepth) does not erase the
    # recorded status of datasets that were not part of this run. New results
    # for the same (provider, dataset, date, hour) key overwrite old entries.
    merged: dict = {}
    if out.exists():
        try:
            previous = json.loads(out.read_text(encoding="utf-8"))
            for r in previous.get("results", []):
                key = (r.get("provider"), r.get("dataset"), r.get("date"), r.get("hour"))
                merged[key] = r
        except (OSError, json.JSONDecodeError):
            merged = {}

    for r in results:
        merged[(r.provider, r.dataset, r.date, r.hour)] = asdict(r)

    entries = [
        merged[k]
        for k in sorted(merged, key=lambda k: (k[0], k[1], k[2], k[3] if k[3] is not None else -1))
    ]

    dates = [e["date"] for e in entries] or [args.start_date.isoformat()]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "start_date": min(dates),
        "end_date": max(dates),
        "providers": {
            "binance": ["aggTrades", "bookDepth"],
            "cryptohftdata": ["orderbook"],
        },
        "results": entries,
    }
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"Manifest -> {out}")


def build_tasks(args: argparse.Namespace):
    # FIX (2026-09-18): honor --datasets so partial re-runs only build the
    # requested tasks.
    datasets = parse_datasets(getattr(args, "datasets", None))

    # Binance: one daily file per dataset.
    for day in daterange(args.start_date, args.end_date):
        for dataset in ("aggTrades", "bookDepth"):
            if dataset in datasets:
                yield ("binance", dataset, day, None)

    # CryptoHFTData: hourly L2 files. Availability begins 2025-06-28.
    if "orderbook" in datasets:
        chd_start = max(args.start_date, CHD_MIN_DATE)
        if chd_start <= args.end_date:
            for day in daterange(chd_start, args.end_date):
                for hour in range(24):
                    yield ("cryptohftdata", "orderbook", day, hour)


def run(args: argparse.Namespace) -> list[Result]:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    tasks = list(build_tasks(args))
    results: list[Result] = []
    print(f"Tasks: {len(tasks):,}")
    print(f"Root:  {root}")

    def worker(task):
        provider, dataset, day, hour = task
        if provider == "binance":
            return download_binance_file(
                symbol=args.symbol,
                dataset=dataset,
                day=day,
                root=root,
                market=args.market,
                verbose=args.verbose,
            )
        return download_chd_file(
            symbol=args.symbol,
            day=day,
            hour=hour,
            root=root,
            api_key=args.cryptohft_api_key,
            verbose=args.verbose,
        )

    # Keep the worker count modest because each CHD hour can be huge and the
    # free tier is rate-limited. A separate process can run Binance files while
    # CHD is running if desired; within this script we simply mix the task pool.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: (r.provider, r.dataset, r.date, r.hour if r.hour is not None else -1))
    write_manifest(root, args.symbol, results, args)

    from collections import Counter
    counts = Counter((r.provider, r.dataset, r.status) for r in results)
    print("\nSummary")
    for key, value in sorted(counts.items()):
        print(f"  {key[0]:15s} {key[1]:12s} {key[2]:10s}: {value:,}")

    errors = [r for r in results if r.status == "error"]
    if errors:
        print(f"\nERRORS: {len(errors):,}")
        for r in errors[:20]:
            print(f"  {r.provider} {r.dataset} {r.date} {r.hour}: {r.error}")

    return results


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download native Binance + CryptoHFTData market data for OFMR v2.")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--start-date", required=True, type=parse_date, help="YYYY-MM-DD UTC inclusive")
    parser.add_argument("--end-date", required=True, type=parse_date, help="YYYY-MM-DD UTC inclusive")
    parser.add_argument(
        "--root",
        default=str(paths.MARKET_DATA_ROOT),
        help="market_data root holding ingest_source/ + ingest_parquet/ "
             "(default: %(default)s)",
    )
    parser.add_argument("--market", choices=["um", "cm"], default="um", help="Binance futures market type")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download workers (default: 4)")
    parser.add_argument(
        "--datasets",
        default=",".join(VALID_DATASETS),
        help="Comma-separated subset of datasets to download "
             "(default: all of aggTrades,bookDepth,orderbook)",
    )
    parser.add_argument(
        "--cryptohft-api-key",
        default=os.getenv("CRYPTOHFTDATA_API_KEY"),
        help="CryptoHFTData API key; preferably set CRYPTOHFTDATA_API_KEY",
    )
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)
    args = parser.parse_args(argv)

    if args.end_date < args.start_date:
        parser.error("--end-date must be >= --start-date")
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")

    if args.start_date < CHD_MIN_DATE <= args.end_date:
        print(f"CryptoHFTData L2 will begin at {CHD_MIN_DATE.isoformat()} (its current coverage start).")

    run(args)


if __name__ == "__main__":
    main()
