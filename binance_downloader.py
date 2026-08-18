"""
binance_downloader.py
======================
Downloads Binance USD-M / COIN-M Futures `aggTrades` and `bookDepth` DAILY
archives directly from data.binance.vision, extracts the CSVs, converts them
to CCXT-style format (via binance_to_ccxt.py), and writes ONE combined
Parquet (or CSV) file per data type covering the whole requested date range.

Pipeline:  DOWNLOAD  ->  EXTRACT  ->  PARSE  ->  CONVERT  ->  COMBINE  ->  WRITE

    ticker_trades.parquet     (timestamp, price, amount, side, id)
    ticker_orderbook.parquet  (timestamp, bid_price, bid_amount, ask_price, ask_amount)

--------------------------------------------------------------------------
URL layout
--------------------------------------------------------------------------
Matches Binance's own downloader repo (github.com/binance/binance-public-data,
python/utility.py::get_path + python/download-aggTrade.py) — verified
against that repo's source directly, since data.binance.vision's file
listing is a JS-rendered page and can't be scraped for exact filenames:

    https://data.binance.vision/data/futures/{um|cm}/daily/{aggTrades|bookDepth}/{SYMBOL}/{SYMBOL}-{type}-{YYYY-MM-DD}.zip

Checksums (.CHECKSUM sidecar files) are intentionally NOT downloaded or
verified, per request. A missing day (404 — e.g. before a symbol's listing
date) is skipped with a warning rather than failing the whole run.

--------------------------------------------------------------------------
Header handling
--------------------------------------------------------------------------
Binance-vision CSVs normally ship WITHOUT a header row (first line is
already data). This script auto-detects the rare case a header IS present
(by checking whether the first row's cells match known column-name tokens)
and drops it, so it works either way.

--------------------------------------------------------------------------
Caveats inherited from binance_to_ccxt.py (see that file's docstring)
--------------------------------------------------------------------------
bookDepth is a coarse, percentage-banded snapshot, not a real top-of-book
feed — the resulting bid/ask columns are an approximation, not literal
tick-level NBBO. This script doesn't change that; it just automates getting
the raw files.

Usage (CLI):
    python binance_downloader.py \
        --symbol BTCUSDT \
        --start-date 2026-08-01 \
        --end-date 2026-08-03 \
        --outdir ./research/csv

    -> writes ./research/csv/BTCUSDT_trades.parquet
             ./research/csv/BTCUSDT_orderbook.parquet

Usage (import):
    from binance_downloader import download_and_convert
    download_and_convert("BTCUSDT", "2026-08-01", "2026-08-03", outdir="./research/csv")
"""

import argparse
import http.client
import os
import time
import zipfile
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from binance_to_ccxt import convert_orderbook_df, convert_trades_df

BASE_URL_DEFAULT = "https://data.binance.vision/"

AGGTRADES_COLUMNS = [
    'agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id',
    'transact_time', 'is_buyer_maker',
]
BOOKDEPTH_COLUMNS = ['timestamp', 'percentage', 'depth', 'notional']

# Literal header tokens per dataset. If the raw CSV's first row contains any
# of these (case-insensitive), that row is a header and gets dropped.
# Binance-vision files normally do NOT have a header at all, so most of the
# time this check simply confirms "no header" and moves on.
_KNOWN_HEADER_TOKENS = {
    'aggTrades': {c.lower() for c in AGGTRADES_COLUMNS} | {'aggregate tradeid', 'was the buyer the maker'},
    'bookDepth': {c.lower() for c in BOOKDEPTH_COLUMNS},
}


# ============================================================================
# DOWNLOADER
# ============================================================================
def _date_range(start_date: str, end_date: str):
    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date)
    if d1 < d0:
        raise ValueError("end_date must be >= start_date")
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def _build_url(base_url, trading_type, market_data_type, symbol, day):
    path = f"data/futures/{trading_type}/daily/{market_data_type}/{symbol.upper()}/"
    file_name = f"{symbol.upper()}-{market_data_type}-{day}.zip"
    return base_url.rstrip('/') + '/' + path + file_name, file_name

def _download_zip(url, dest_path, timeout=30, retries=4):
    """Download `url` to `dest_path`. Returns True if a new file was
    downloaded, False if it was already cached on disk."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        if zipfile.is_zipfile(dest_path):
            return False
        # Cached file exists but isn't a valid zip (partial/corrupt
        # download from a previous run) — discard it and re-download.
        os.remove(dest_path)

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            tmp_path = dest_path + '.part'

            with urlopen(req, timeout=timeout) as resp, open(tmp_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            if not zipfile.is_zipfile(tmp_path):
                # Downloaded something, but it's not a real zip (e.g. an
                # error page) — treat like a transient failure and retry.
                os.remove(tmp_path)
                raise ValueError(f"Downloaded file is not a valid zip: {url}")

            os.replace(tmp_path, dest_path)
            return True
        except HTTPError:
            raise  # 404 etc. — not retryable, let the caller decide
        except (URLError, http.client.IncompleteRead, ConnectionError,
                TimeoutError, ValueError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_err  # pragma: no cover

# ============================================================================
# EXTRACTOR
# ============================================================================
def _extract_csv(zip_path, extract_dir):
    """Extract the single CSV inside a Binance zip. Returns the CSV path.
    No-ops if already extracted."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if not csv_names:
            raise ValueError(f"No CSV found inside {zip_path}")
        name = csv_names[0]
        target = os.path.join(extract_dir, os.path.basename(name))
        if not os.path.exists(target):
            with zf.open(name) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    return target


def _load_raw_csv(csv_path, columns, dataset_key):
    """Read an extracted Binance CSV and return it with canonical column
    names, auto-dropping a header row if one is present (see module
    docstring)."""
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    first_cells = [c.strip().lower() for c in first_line.split(',')]
    has_header = any(cell in _KNOWN_HEADER_TOKENS[dataset_key] for cell in first_cells)

    if has_header:
        df = pd.read_csv(csv_path, header=0)
        df.columns = columns
    else:
        df = pd.read_csv(csv_path, header=None, names=columns)
    return df


def _fetch_day(base_url, trading_type, market_data_type, symbol, day,
               cache_dir, columns, dataset_key, verbose=True):
    """Download + extract + parse a single day's file. Returns a DataFrame,
    or None if that day doesn't exist upstream (404)."""
    url, file_name = _build_url(base_url, trading_type, market_data_type, symbol, day)
    zip_path = os.path.join(cache_dir, 'zips', market_data_type, file_name)
    csv_dir = os.path.join(cache_dir, 'csv', market_data_type)

    try:
        was_new = _download_zip(url, zip_path)
        if verbose:
            print(f"  [{market_data_type}] {day}: {'downloaded' if was_new else 'cached'}")
    except HTTPError as e:
        if e.code == 404:
            if verbose:
                print(f"  [{market_data_type}] {day}: not available (404) — skipping")
            return None
        raise
    except URLError as e:
        print(f"  [{market_data_type}] {day}: network error ({e}) — skipping")
        return None

    csv_path = _extract_csv(zip_path, csv_dir)
    return _load_raw_csv(csv_path, columns, dataset_key)


# ============================================================================
# ORCHESTRATION: download -> extract -> parse -> convert -> combine -> write
# ============================================================================
def download_and_convert(symbol, start_date, end_date, outdir="./research/csv",
                          trading_type="um", cache_dir="./binance_cache",
                          out_format="parquet", base_url=BASE_URL_DEFAULT,
                          ticker=None, verbose=True):
    """Download aggTrades + bookDepth for [start_date, end_date] (inclusive,
    UTC daily files), convert to CCXT format, and write ONE combined file
    per data type to `outdir`.

    Processes one day at a time and streams each day's converted rows
    straight to disk (Parquet via ParquetWriter, CSV via append mode)
    instead of accumulating every day's raw + converted DataFrame in
    memory before a single final concat/write. For symbols like BTCUSDT
    over a multi-day/week range, holding the *entire* range in RAM twice
    (once as a list of per-day frames, once as the concatenated result)
    is what causes MemoryError/ArrayMemoryError on long ranges — this
    keeps peak memory to roughly one day's worth of data at a time.

    Returns (trades_path, orderbook_path).
    """
    ticker = ticker or symbol.upper()
    days = _date_range(start_date, end_date)

    os.makedirs(outdir, exist_ok=True)
    ext = 'parquet' if out_format == 'parquet' else 'csv'
    trades_out = os.path.join(outdir, f"{ticker}_trades.{ext}")
    ob_out = os.path.join(outdir, f"{ticker}_orderbook.{ext}")

    # Start clean — CSV mode uses append, and a stale file from a previous
    # (e.g. failed) run must not get mixed in with this run's rows.
    for path in (trades_out, ob_out):
        if os.path.exists(path):
            os.remove(path)

    trades_writer = None  # pyarrow ParquetWriter, opened lazily on first day
    ob_writer = None
    trades_header_written = False
    ob_header_written = False
    total_trades = 0
    total_ob = 0
    days_with_trades = 0
    days_with_ob = 0

    if verbose:
        print(f"Streaming aggTrades + bookDepth for {symbol} {start_date}..{end_date} "
              f"({len(days)} day(s)) -> {outdir}")

    try:
        for day in days:
            raw_trades_day = _fetch_day(base_url, trading_type, 'aggTrades', symbol, day,
                                         cache_dir, AGGTRADES_COLUMNS, 'aggTrades', verbose)
            if raw_trades_day is None:
                continue
            trades_day = convert_trades_df(raw_trades_day)
            del raw_trades_day

            raw_depth_day = _fetch_day(base_url, trading_type, 'bookDepth', symbol, day,
                                        cache_dir, BOOKDEPTH_COLUMNS, 'bookDepth', verbose)
            ob_day = None
            if raw_depth_day is not None:
                try:
                    ob_day = convert_orderbook_df(raw_depth_day, trades_day)
                except ValueError as e:
                    if verbose:
                        print(f"  [bookDepth] {day}: {e} — skipping this day's orderbook")
                del raw_depth_day

            if out_format == 'parquet':
                table = pa.Table.from_pandas(trades_day, preserve_index=False)
                if trades_writer is None:
                    trades_writer = pq.ParquetWriter(trades_out, table.schema)
                trades_writer.write_table(table)
            else:
                trades_day.to_csv(trades_out, mode='a', index=False,
                                   header=not trades_header_written)
                trades_header_written = True
            total_trades += len(trades_day)
            days_with_trades += 1
            del trades_day

            if ob_day is not None and len(ob_day) > 0:
                if out_format == 'parquet':
                    table = pa.Table.from_pandas(ob_day, preserve_index=False)
                    if ob_writer is None:
                        ob_writer = pq.ParquetWriter(ob_out, table.schema)
                    ob_writer.write_table(table)
                else:
                    ob_day.to_csv(ob_out, mode='a', index=False,
                                   header=not ob_header_written)
                    ob_header_written = True
                total_ob += len(ob_day)
                days_with_ob += 1
            del ob_day
    finally:
        if trades_writer is not None:
            trades_writer.close()
        if ob_writer is not None:
            ob_writer.close()

    if total_trades == 0:
        raise RuntimeError(f"No aggTrades data found for {symbol} in {start_date}..{end_date}")
    if total_ob == 0:
        raise RuntimeError(f"No bookDepth data found for {symbol} in {start_date}..{end_date}")

    if verbose:
        print(f"Wrote {total_trades:,} trades      ({days_with_trades} day(s)) -> {trades_out}")
        print(f"Wrote {total_ob:,} orderbook rows ({days_with_ob} day(s)) -> {ob_out}")
    return trades_out, ob_out


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance aggTrades/bookDepth from data.binance.vision and "
                    "convert them into one combined CCXT-style Parquet (or CSV) "
                    "trades/orderbook file."
    )
    parser.add_argument('--symbol', required=True, help="e.g. BTCUSDT")
    parser.add_argument('--start-date', required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument('--end-date', required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument('--ticker', default=None,
                         help="Output filename prefix (default: same as --symbol)")
    parser.add_argument('--trading-type', default='um', choices=['um', 'cm'],
                         help="USD-M ('um') or COIN-M ('cm') futures (default: um). "
                              "bookDepth is only published for futures, not spot.")
    parser.add_argument('--outdir', default='./research/csv',
                         help="Output directory (default: ./research/csv, matching "
                              "load_and_build_state_space()'s default paths)")
    parser.add_argument('--cache-dir', default='./binance_cache',
                         help="Where raw zips/CSVs are cached so re-runs don't re-download")
    parser.add_argument('--format', dest='out_format', default='parquet', choices=['parquet', 'csv'],
                         help="Output format for the combined file (default: parquet)")
    parser.add_argument('--base-url', default=BASE_URL_DEFAULT,
                         help=argparse.SUPPRESS)  # advanced: point at a mirror/proxy for testing
    args = parser.parse_args()

    download_and_convert(
        symbol=args.symbol, start_date=args.start_date, end_date=args.end_date,
        outdir=args.outdir, trading_type=args.trading_type, cache_dir=args.cache_dir,
        out_format=args.out_format, ticker=args.ticker, base_url=args.base_url,
    )


if __name__ == '__main__':
    main()