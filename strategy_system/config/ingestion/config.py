"""
dataconfig.py

Central configuration for the data_v2 pipeline, consumed by live.py
(strategy_system.data.dataconfig as cfg).

Reconstructed from usage in live.py:
    cfg.AGG_TRADES_PATH  -> _filter_parquet_files(agg_path=cfg.AGG_TRADES_PATH, ...)
    cfg.LOB_PATH         -> _filter_parquet_files(lob_path=cfg.LOB_PATH, ...)
    cfg.START_DATE       -> fetch_source_data(start=cfg.START_DATE, ...)
    cfg.END_DATE         -> fetch_source_data(end=cfg.END_DATE, ...)
    cfg.MATCHED_PATH     -> output_dir = Path(cfg.MATCHED_PATH); read/write_source_data default path
    cfg.BOOKDEPTH_PATH   -> _load_bookdepth_wide: Path(cfg.BOOKDEPTH_PATH) / f"{day}.parquet"
    cfg.TICKER           -> output filename: f"{cfg.TICKER}_{start_day}_..."

*** EDIT THE VALUES BELOW TO MATCH YOUR ACTUAL data_v2 LAYOUT AND SYMBOL. ***
The folder layout comment in live.py's docstring implies data_v2 sits under
some project root with this structure:

    data_v2
    |_ $exchange\\$ticker\\$datatype   # original source file (non-parquet)
    |_ parquet\\$ticker\\$datatype     # original source file (parquet)
    |_ manifests                       # metadata for atomic writing (ingestion)
    |_ matched\\$ticker                # research read/write copy

The paths below assume that layout with exchange="binance" -- adjust the
exchange folder name and DATA_ROOT if yours differs.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root of the data_v2 tree. Adjust to your actual project root.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("market_data")

# ---------------------------------------------------------------------------
# Symbol this config is scoped to. Used directly in output filenames
# (f"{cfg.TICKER}_{start_day}_{end_day}_{start_hour}-{end_hour}.parquet").
# ---------------------------------------------------------------------------
TICKER = "BTCUSDT"

# Exchange subfolder name under parquet/, per the layout in live.py's docstring.

# ---------------------------------------------------------------------------
# Source data paths (read-only for research; see live.py's module docstring).
# ---------------------------------------------------------------------------
AGG_TRADES_PATH = DATA_ROOT / "ingest_parquet" / TICKER / "aggTrades"
LOB_PATH = DATA_ROOT / "ingest_parquet"  / TICKER / "orderbook"

# Optional Binance bookDepth daily files, keyed by day: BOOKDEPTH_PATH/{day}.parquet
BOOKDEPTH_PATH = DATA_ROOT / "ingest_parquet" / TICKER / "bookDepth"

# ---------------------------------------------------------------------------
# Matched output path (read/write for research; see live.py's module docstring).
# ---------------------------------------------------------------------------
MATCHED_PATH = DATA_ROOT / "matched" / TICKER

# ---------------------------------------------------------------------------
# Default date window used by fetch_source_data() / read_source_data() /
# write_source_data() when no start/end is passed explicitly.
# Format must match MATCH_PATTERN in live.py: "YYYY-MM-DD".
# ---------------------------------------------------------------------------
START_DATE = "2026-02-01"
END_DATE = "2026-02-28"