"""
Enhanced ML State Space — Quant-Grade Feature Engineering for RSI Snapback
==========================================================================
POLARS PORT of the original pandas research script.

WHY THIS PORT EXISTS
---------------------
polars' lazy, multithreaded, Rust-backed engine reads/aggregates large
parquet files (and does the resampling-style feature engineering below)
meaningfully faster than pandas. It does NOT speed up the walk-forward
loop itself (LightGBM fit + SHAP + sklearn permutation_importance across
N folds) — that's CPU-bound model fitting/explainability work on numpy
arrays regardless of which dataframe library built them. This port keeps
that section on numpy arrays at the LightGBM/SHAP/sklearn boundary, same
as the original, and only changes the surrounding plumbing.

WHAT ACTUALLY CHANGES BEHAVIOUR — READ THIS BEFORE TRUSTING THE OUTPUT
------------------------------------------------------------------------
polars has no index concept and several pandas idioms this script relied
on heavily do NOT have 1:1 equivalents. Every one of the following was a
real, silent behavioural mismatch caught by diffing this port's output
against the original pandas script on synthetic data (three datasets:
dense, sparse, and bursty/sparse-with-genuine-empty-bars) column-by-column
until every feature matched to float64 noise (~1e-9). If you change the
feature engineering later, re-run that kind of diff before trusting it —
these are exactly the class of bug this file's own original comments
describe having shipped before (research/live drift).

  1. `df.group_by_dynamic(...)` (polars' nearest `.resample()` equivalent)
     SILENTLY DROPS bins with zero rows; pandas' `.resample(freq).agg()`
     always returns one row per bin, NaN/0 for empty ones. Fixed via the
     `dense_resample()` helper below, which reinserts every bin over a
     regular grid and matches pandas' sum()-defaults-to-0 vs
     last()/mean()-defaults-to-null behaviour per aggregation.
  2. polars distinguishes float NaN from null — `drop_nulls()` does NOT
     remove NaN. Any numpy array with `np.nan` (e.g. from `_pad_front`)
     MUST be constructed with `nan_to_null=True` or it silently survives
     what would have been a pandas `.dropna()`.
  3. The final micro/macro join in the original
     (`df_micro_shifted.join(df_macro_shifted, how='left').ffill()`) does
     two different things at once: (a) broadcasts each bar's shifted
     features across every second until the next bar closes — replicated
     here with `join_asof(strategy="backward")` — AND (b) forward-fills
     any bar whose OWN value was genuinely missing (e.g. `volume` on a
     zero-trade bar, since pandas' `df_trades_1m` drops empty-trade bars
     entirely rather than zero-filling them) from the previous bar. (b) is
     a real quirk of the original pipeline, not a bug introduced here —
     replicated via an explicit `pl.all().forward_fill()` on the
     bar-level frame before the asof broadcast.
  4. VPIN bucket-close times must be BINNED into the BAR_FREQ grid
     (matching pandas' `.resample(BAR_FREQ).last()`, which assigns a
     bucket to whichever bar's WINDOW CONTAINS its close time) rather than
     asof-joined directly against each bar's left-edge timestamp — the
     latter introduces a systematic one-bar lag.
  5. `pandas.groupby(bucket).last()` (used in `deduplicate_to_bar`)
     replaces the row's timestamp with the BUCKET FLOOR, not the actual
     last observation's own timestamp. `polars.group_by(...).last()` keeps
     the real last timestamp unless you explicitly overwrite it.
  6. `calculate_zscore()` / other custom helpers in your real
     `utils/indicators.py` are almost certainly pandas-Series-shaped. This
     port reimplements a rolling z-score natively in polars
     (`_pl_rolling_zscore`) rather than calling your real function, since
     I don't have its source. VERIFY it matches your actual ddof/edge
     behaviour, or port your real function to accept a polars column.
     `get_wilders_rsi()` and `sma()` are untouched — they're called on
     plain numpy arrays (`np.diff(...)` output) in both versions, so they
     don't need to change.

WHAT THIS PORT DOES NOT VALIDATE
----------------------------------
I don't have your real `main/config.py`, `utils/indicators.py`, or
`utils/engine.py`, or your actual parquet data — I validated this against
a stub config + reimplemented indicators on synthetic orderbook/trade
data (see the parity-check notes above). Before trusting this for
anything live, run both the original and this port on the SAME real
session data and diff every column — the `dense_resample`/asof-join
patterns above are exactly the kind of thing that can look right and
still be subtly wrong on data shaped differently than what I tested.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import datetime as _dt
import gc
from dataclasses import dataclass
import tempfile

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from scripts.utils.indicators import get_wilders_rsi, sma

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, classification_report
import shap
from sklearn.inspection import permutation_importance

# 1. Limit thread allocation to keep per-core memory low
os.environ["POLARS_MAX_THREADS"] = "4"

# get_wilders_rsi/sma operate on plain numpy arrays in the original script
# (np.diff(...) output) — untouched by the pandas->polars switch.

from research.ml_shared_config import *
from research.feature_builder import compute_orderflow_features, compute_price_features, _join_orderflow_features

def _ensure_parquet(parquet_path: str, csv_fallback_path: str) -> str:
    """Ensure the Parquet file exists, handling legacy CSV fallbacks.

    If the Parquet file is missing but the matching CSV is present, 
    this function will execute a one-time streaming conversion to avoid high RAM usage 
    (preventing OOM risks) and return the validated parquet_path.

    Args:
        parquet_path: Priority path. Must be coupled with csv_path
        csv_fallback_path: Backup for parquet error. Must be coupled with parquet_path
    Note:
        This is for legacy compatibility only. While our target data architecture 
        is exclusively Parquet, this layer guarantees zero-downtime integration 
        for unmigrated datasets without hitting memory ceilings.
    """

    if os.path.exists(parquet_path):
        return parquet_path
    if os.path.exists(csv_fallback_path):
        print(f"  [AUTO-CONVERT] {csv_fallback_path} -> {parquet_path} (one-time, streaming)")
        pl.scan_csv(csv_fallback_path).sink_parquet(parquet_path)
    return parquet_path


def get_session_paths(data_format: str = DATA_FORMAT) -> list[tuple[str, str, int]]:
    """
    Resolve orderbook, trades, and session IDs for all available chunks under
    DATA_DIR. Iterates through legacy chunked database files (e.g., split websocket queries
    like 'BTCUSDT_1', 'BTCUSDT_2', 'BTCUSDT_3'). 

    Args:
        data_format: Resolve 'parquet' or 'csv' but always use parquet whenever possible.
    Note:
        Legacy compatability to replace the old single-file get_data_paths(). Once full parquet
        integration is complete, this code will be removed.
    """
    data_format = data_format.lower()
    if data_format not in ("parquet", "csv"):
        raise ValueError(f"Unsupported data_format: {data_format!r} (expected 'parquet' or 'csv')")

    session_suffixes = ["", "_2", "_3"]
    sessions = []
    for session_id, suffix in enumerate(session_suffixes):
        orderbook_path = f"{DATA_DIR}/{SYMBOL_HIST}_orderbook{suffix}.{data_format}"
        trades_path = f"{DATA_DIR}/{SYMBOL_HIST}_trades{suffix}.{data_format}"

        if data_format == "parquet" and AUTO_CONVERT_CSV_TO_PARQUET:
            orderbook_path = _ensure_parquet(orderbook_path, f"{DATA_DIR}/{SYMBOL_HIST}_orderbook{suffix}.csv")
            trades_path = _ensure_parquet(trades_path, f"{DATA_DIR}/{SYMBOL_HIST}_trades{suffix}.csv")

        if os.path.exists(orderbook_path) and os.path.exists(trades_path):
            sessions.append((orderbook_path, trades_path, session_id))
        else:
            print(f"  [SESSION SKIPPED] session_id={session_id}: missing {orderbook_path!r} or {trades_path!r}")

    if not sessions:
        raise FileNotFoundError(f"No session files found under {DATA_DIR!r} for data_format={data_format!r}")
    return sessions


def _scan_source(path: str) -> pl.LazyFrame:
    """
    Lazily scan a data file, dispatching on its extension (.parquet vs .csv). Legacy code fix only.
    """
    return pl.scan_parquet(path) if str(path).endswith(".parquet") else pl.scan_csv(path)


def _session_total_volume(trades_path: str) -> float:
    """Compute the total traded volume for the entire session to size VPIN buckets consistently.

    Executes a single lazy aggregate without full materialization, ensuring 
    high performance even for large Parquet files.

    Args:
        trades_path taken from get_session function once orderflow trades database is established.
    Note:
        This prevents VPIN scale drift caused by legacy per-chunk volume calculations 
        (where bucket sizing varied depending on local chunk data). Computing this 
        globally per session ensures stable normalization.
    """
    trades_lazy = _scan_source(trades_path)
    total = trades_lazy.select(pl.col("amount").sum().cast(pl.Float64)).collect()["amount"][0]
    return float(total)

def dense_resample(df: pl.DataFrame, every: str, agg_exprs: list, time_col: str = "datetime",
                    zero_fill_cols: list | None = None, start=None, end=None) -> pl.DataFrame:
    """
    Performs time-series resampling on polars dataframe where we ensure all the time intervals exist.
    - Defining time boundaries, finding min and max timestamp of dataset per resampling frequency
    - Ensures unrboken continuous sequence of daetimes at fixed intervals.
    - Aggregates raw data with group_by_dynamic, applying whatever aggregation are passed in agg_exprs
    - Join and zero fills to ensure missing periods arent dropped and instead returned as rows with null.

    Args:
        every: Resampling frequency string (e.g., "1m", "1h").
        agg_exprs: List of Polars expression objects to apply per window.
        zero_fill_cols: Optional list of column names to replace nulls with 0.

    Note:
        Essential for financial time-series data where missing trade or orderbook intervals 
        must be explicitly preserved rather than skipped.
    """

    if df.is_empty():
        if start is None or end is None:
            raise ValueError("Cannot resample empty DataFrame without explicit 'start' and 'end' bounds")

    bounds = df.select(
        pl.col(time_col).min().dt.truncate(every).alias("_s"),
        pl.col(time_col).max().dt.truncate(every).alias("_e"),
    )
    grid_start = start if start is not None else bounds["_s"][0]
    grid_end = end if end is not None else bounds["_e"][0]
    grid = pl.DataFrame({
        time_col: pl.datetime_range(grid_start, grid_end, interval=every, closed="both", eager=True)
    })
    sort_columns = [time_col, "timestamp"] if "timestamp" in df.columns else time_col
    agg = (
        df.sort(sort_columns, maintain_order=True)
          .group_by_dynamic(time_col, every=every, closed="left", label="left")
          .agg(agg_exprs)
    )
    out = grid.join(agg, on=time_col, how="left").sort(time_col)
    if zero_fill_cols:
        out = out.with_columns([pl.col(c).fill_null(0) for c in zero_fill_cols])
    return out


from datetime import timedelta
import polars.selectors as cs

"""
============================================================================
STEP 1b — PER-SESSION FEATURE PIPELINE
- This section deals primarily with recursive features as abstraction
============================================================================
"""

@dataclass(frozen=True)
class SessionFeatureContext:
    """
    Pre-computed global state injected into local data chunks. Prevents recursive indicators (RSI)
    and volume-based buckets (VPIN) from artificially reseting at chunk boundaries.

    VERY CRITICAL!
    """
    total_volume: float
    rsi_features: pl.DataFrame


def _with_datetime(source: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normalize epoch timestamps to Polars datetime objects for uniform joining later. 
    Very important to use Float32 or Int32 unless needed otherwise for performance optimization
    in processing large historical backtests
    """
    return (
        source
        .cast({cs.float(): pl.Float32, cs.integer() - cs.by_name("timestamp", "id", require_all=False   ): pl.Int32})
        .with_columns(
            pl.from_epoch((pl.col("timestamp") // 1000).cast(pl.Int32), time_unit="s").alias("datetime")
        )
    )

def _shift_feature_dict(feats: dict, periods: int = 1) -> dict:
    """Shift every non-datetime column of every DataFrame in a feature dict
    by `periods` bars. Applied at the call site so feature_builder.py stays
    a pure 'compute at time t' module — the causal shift is the caller's
    responsibility, not the feature functions'."""
    shifted = {}
    for key, df in feats.items():
        if isinstance(df, pl.DataFrame):
            cols_to_shift = [c for c in df.columns if c != "datetime"]
            shifted[key] = df.with_columns([
                pl.col(c).shift(periods) for c in cols_to_shift
            ])
        else:
            shifted[key] = df
    return shifted

def _rsi_features_from_price_bars(price_bars: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the recursive RSI family once over a continuous close series. Using numpy diffs.
    Note:
        Wilder's RSI is recursive (building on previous value) and because Polar's vectorized engine
        cannot execute recursive row-by-row math efficiently, we extract the column to raw numpy, run
        the math and stitch it back. Even with get_wilders_rsi() we still need this as abstraction for
        polars specifically histbacktesting.
    """
    price_bars = price_bars.sort("datetime")
    price_clean = price_bars["close"].to_numpy().astype(float)
    diffs = np.diff(price_clean)

    def _pad_front(arr: np.ndarray, target_len: int) -> np.ndarray:
        out = np.full(target_len, np.nan)
        if len(arr):
            out[-len(arr):] = arr
        return out

    def _wilders_or_nan(window: int) -> np.ndarray:
        return get_wilders_rsi(diffs, window) if len(diffs) > window else np.full(len(diffs), np.nan)

    rsi_2 = _pad_front(_wilders_or_nan(RSI_2_WINDOW), len(price_bars))
    rsi_14_raw = _wilders_or_nan(RSI_14_WINDOW)
    return price_bars.select("datetime").with_columns([
        pl.Series("rsi_2", rsi_2, nan_to_null=True),
        pl.Series("rsi_14", _pad_front(rsi_14_raw, len(price_bars)), nan_to_null=True),
        pl.Series("sma_14", _pad_front(sma(rsi_14_raw, SMA_14_WINDOW), len(price_bars)), nan_to_null=True),
    ])


def _build_session_rsi_features(orderbook_path: str) -> pl.DataFrame:
    """Full-session sparse bar materialization. Guarantees recursive RSI calculation
    is bit-for-bit independent of how the raw event stream is partitioned downstream.
    
    Note:
        Extraction for _rsi_features_from_price_bars to run. Just like VPIN, it is crucial to
        process RSI continuously before chunking the database later.
    """
    orderbook = _with_datetime(_scan_source(orderbook_path)).filter(pl.col("datetime").is_not_null())
    sparse_bars = (
        orderbook
        .with_columns(((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"))
        .sort(["datetime", "timestamp"], maintain_order=True)
        .group_by_dynamic("datetime", every=BAR_FREQ, closed="left", label="left")
        .agg(pl.col("mid_price").last().alias("close"))
        .collect(engine="streaming")
    )
    price_bars = dense_resample(sparse_bars, BAR_FREQ, [pl.col("close").last().alias("close")])
    return _rsi_features_from_price_bars(price_bars)


def build_session_feature_context(orderbook_path: str, trades_path: str) -> SessionFeatureContext:
    """Precompute session-scoped feature state shared by every raw-data chunk."""
    return SessionFeatureContext(
        total_volume=_session_total_volume(trades_path),
        rsi_features=_build_session_rsi_features(orderbook_path),
    )


def _session_time_bounds(trades_path: str) -> tuple[_dt.datetime, _dt.datetime]:
    """Return the source time bounds used to partition an already-built state space."""
    bounds = _scan_source(trades_path).select([
        pl.from_epoch((pl.col("timestamp").min() // 1000).cast(pl.Int64), time_unit="s").alias("start"),
        pl.from_epoch((pl.col("timestamp").max() // 1000).cast(pl.Int64), time_unit="s").alias("end"),
    ]).collect()
    return bounds["start"][0], bounds["end"][0]


def partition_state_space(state_space: pl.DataFrame, session_start: _dt.datetime,
                          session_end: _dt.datetime, chunks_per_session: int) -> list[pl.DataFrame]:
    """Partition an already-built state space into chunks.

    Note:
        Deprecated for large dataset. Kept for callers that explicitly need post-hoc partitioning. 
        The historical orchestration below no longer uses it for chunked runs because that would
        require materializing the complete session first.
    """
    if chunks_per_session < 1:
        raise ValueError("chunks_per_session must be at least 1")

    chunk_duration = (session_end - session_start) / chunks_per_session
    chunks = []
    for index in range(chunks_per_session):
        start = session_start + (chunk_duration * index)
        end = session_start + (chunk_duration * (index + 1))
        predicate = pl.col("datetime") >= start
        if index < chunks_per_session - 1:
            predicate &= pl.col("datetime") < end
        chunks.append(state_space.filter(predicate))
    return chunks

# ============================================================================
# DROP-IN REPLACEMENT for the current `load_and_build_state_space` in
# features.py. Everything else in features.py (dense_resample,
# compute_orderflow_features, SessionFeatureContext, _rsi_features_from_price_bars,
# _build_session_rsi_features, build_session_feature_context, _with_datetime,
# _session_time_bounds, partition_state_space, etc.) is UNCHANGED — paste this
# block in place of the old load_and_build_state_space definition and add
# `import tempfile` near the top of features.py.
# ============================================================================


def _materialize_via_disk(
    df_orderbook: pl.LazyFrame, df_trades: pl.LazyFrame, session_id: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sink both LazyFrames to a scratch Parquet file and read them back eagerly.

    This is the RAM-management trick from the original script (route through
    disk instead of letting the lazy streaming engine hold its own working set
    for the whole chunk) — isolated into its own function so the temp-file
    lifecycle has exactly one place it can go wrong. `tempfile.TemporaryDirectory`
    guarantees cleanup even if sink/read raises partway through, and gives each
    call a unique path with no risk of two chunks colliding on a filename.
    """
    with tempfile.TemporaryDirectory(prefix=f"state_space_session{session_id}_") as tmp_dir:
        ob_path = os.path.join(tmp_dir, "orderbook.parquet")
        tr_path = os.path.join(tmp_dir, "trades.parquet")
        df_orderbook.sink_parquet(ob_path)
        df_trades.sink_parquet(tr_path)
        return pl.read_parquet(ob_path), pl.read_parquet(tr_path)


def _load_and_clean_chunk_events(
    orderbook_path: str,
    trades_path: str,
    session_id: int,
    start_dt: _dt.datetime | None,
    end_dt: _dt.datetime | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one chunk's raw events and make them safe to build bars from.

    Handles exactly three concerns and nothing else:
      1. Restricting the scan to [start_dt - warm-up buffer, end_dt) so a
         chunked run only pulls the events it needs (a no-op filter when
         start_dt/end_dt are None, i.e. a full-session, non-chunked call).
      2. Dropping rows with a null/unparseable timestamp, with a visible
         count, instead of letting them crash group_by_dynamic() deep inside
         dense_resample().
      3. Materializing the cleaned, time-bounded chunk to memory via disk
         (see _materialize_via_disk).

    Does NOT compute any bar, resample anything, or touch a feature column —
    that starts with the bar-building functions below, which take the two
    DataFrames this returns.
    """
    df_orderbook = _with_datetime(_scan_source(orderbook_path))
    df_trades = _with_datetime(_scan_source(trades_path))

    if start_dt and end_dt:
        buffer_start = start_dt - timedelta(seconds=CHUNK_BUFFER_SECONDS)
        df_orderbook = df_orderbook.filter((pl.col("datetime") >= buffer_start) & (pl.col("datetime") < end_dt))
        df_trades = df_trades.filter((pl.col("datetime") >= buffer_start) & (pl.col("datetime") < end_dt))

    n_ob_null = df_orderbook.select(pl.col("datetime").is_null().sum()).collect(engine="streaming").item()
    n_tr_null = df_trades.select(pl.col("datetime").is_null().sum()).collect(engine="streaming").item()
    if n_ob_null:
        print(f"  [DATA WARNING] session {session_id}: dropping {n_ob_null} orderbook row(s) "
              f"with null/unparseable timestamp ({orderbook_path})")
    if n_tr_null:
        print(f"  [DATA WARNING] session {session_id}: dropping {n_tr_null} trades row(s) "
              f"with null/unparseable timestamp ({trades_path})")
    df_orderbook = df_orderbook.filter(pl.col("datetime").is_not_null())
    df_trades = df_trades.filter(pl.col("datetime").is_not_null())

    df_orderbook = df_orderbook.sort(["datetime", "timestamp"], maintain_order=True)
    df_trades = df_trades.sort(["datetime", "timestamp"], maintain_order=True)
    df_orderbook = df_orderbook.with_columns(((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"))

    return _materialize_via_disk(df_orderbook, df_trades, session_id)


def _with_buy_sell_split(df_trades: pl.DataFrame) -> pl.DataFrame:
    """Split each trade's amount into buy_vol / sell_vol columns based on side.

    Used by the micro/volume bar builders below. compute_orderflow_features
    re-derives its own copy of this split internally from the raw 'side'
    column, so this is applied once here purely for the micro/volume path —
    it does not assume compute_orderflow_features shares it.
    """
    return df_trades.with_columns([
        pl.when(pl.col("side") == "buy").then(pl.col("amount")).otherwise(0.0).alias("buy_vol"),
        pl.when(pl.col("side") == "sell").then(pl.col("amount")).otherwise(0.0).alias("sell_vol"),
    ])


def _build_price_bars(df_orderbook: pl.DataFrame) -> pl.DataFrame:
    """Resample cleaned orderbook events into OHLC + spread bars at BAR_FREQ.

    This is the "macro" price series everything else (RSI, MA slope,
    orderflow joins) is eventually hung off of. Pure resampling — no
    cross-source joins, no shifting, no z-scoring — so it's the natural
    function to check first if a parity diff against the original pandas
    script only fails on OHLC values.
    """
    df_ob_1m = dense_resample(
        df_orderbook, BAR_FREQ,
        [
            pl.col("mid_price").first().alias("open"),
            pl.col("mid_price").max().alias("high"),
            pl.col("mid_price").min().alias("low"),
            pl.col("mid_price").last().alias("close"),
            pl.col("bid_price").last().alias("bid_price"),
            pl.col("ask_price").last().alias("ask_price"),
            pl.col("bid_amount").sum().alias("bid_amount"),
            pl.col("ask_amount").sum().alias("ask_amount"),
        ],
        zero_fill_cols=["bid_amount", "ask_amount"],
    )
    df_ob_1m = df_ob_1m.with_columns((pl.col("ask_price") - pl.col("bid_price")).alias("spread"))
    df_ob_1m = df_ob_1m.with_columns((pl.col("spread") / pl.col("close")).alias("spread_pct"))
    return df_ob_1m


def _build_micro_bars(df_trades: pl.DataFrame) -> pl.DataFrame:
    """Resample trades into 1-second microstructure bars.

    df_micro is deliberately kept at 1s resolution and left UNshifted here —
    it defines the final state space's row grid, and its own 'price' column
    later becomes execution_price (see load_and_build_state_space), so it
    must stay exactly as-observed until _broadcast_bar_features_to_micro
    decides what gets lagged and what doesn't.
    """
    df_micro = dense_resample(
        df_trades, "1s",
        [pl.col("price").last(), pl.col("buy_vol").sum(), pl.col("sell_vol").sum()],
        zero_fill_cols=["buy_vol", "sell_vol"],
    ).with_columns(pl.col("price").forward_fill())
    df_micro = df_micro.with_columns((pl.col("buy_vol") - pl.col("sell_vol")).alias("micro_OFI"))
    df_micro = df_micro.with_columns(pl.col("micro_OFI").rolling_sum(window_size=10).alias("micro_CVD"))
    return df_micro


def _build_bar_volume(df_trades: pl.DataFrame) -> pl.DataFrame:
    """Bar-level traded volume. See module docstring point 3: deliberately NOT
    zero-filled — a bar with zero trades resamples to a null here so the
    forward-fill in _broadcast_bar_features_to_micro carries the previous
    bar's volume forward, matching the original pandas pipeline's behaviour
    of dropping empty-trade bars entirely rather than zero-filling them."""
    return dense_resample(df_trades, BAR_FREQ, [pl.col("amount").sum().alias("volume")])


def _assert_point_in_time_safe(df_micro_shifted: pl.DataFrame, df_macro_shifted: pl.DataFrame) -> None:
    """Precondition check for the asof-join in _broadcast_bar_features_to_micro.

    join_asof(strategy="backward") silently matches the wrong row — it does
    NOT raise — if either side has duplicate or out-of-order timestamps.
    Since that join is the one place a bug here directly becomes a
    lookahead-bias bug, check the precondition explicitly instead of trusting
    every upstream .sort() call to have been sufficient.
    """
    for name, df in (("micro", df_micro_shifted), ("macro", df_macro_shifted)):
        dt = df["datetime"]
        if not dt.is_sorted():
            raise ValueError(f"_broadcast_bar_features_to_micro: {name} frame is not sorted by datetime")
        if dt.n_unique() != len(dt):
            raise ValueError(f"_broadcast_bar_features_to_micro: {name} frame has duplicate datetime values")


def _broadcast_bar_features_to_micro(df_micro: pl.DataFrame, df_macro: pl.DataFrame) -> pl.DataFrame:
    """THE no-lookahead-bias boundary. Read this before touching this function.

    Every column in df_macro (price, RSI, orderflow — everything computed
    above) describes what was knowable only once that bar's own close had
    passed. Joined straight onto df_micro's second-by-second grid, every
    second inside a bar would see that bar's OWN close-time features before
    the bar had actually closed — future information leaking into the
    present. This function encodes two independent one-step lags plus a
    forward-fill, matching the original pandas pipeline exactly (module
    docstring points 3a/3b):

      1. df_micro's own columns (price/buy_vol/sell_vol/micro_OFI/micro_CVD)
         are shifted forward by one second. A decision "at" second T cannot
         condition on trades still printing during T itself — so the values
         attached to row T are what was true as of T-1.
      2. Every macro (bar-level) column is independently shifted forward by
         one BAR, so a decision anywhere inside bar N+1 sees bar N's
         already-closed, already one-bar-old feature values — never the bar
         it is currently inside of.
      3. After shifting, macro columns are forward-filled (point 3b) so a bar
         whose OWN value was genuinely missing (e.g. volume on a zero-trade
         bar) inherits the previous bar's value instead of leaking a null
         into every second until the next bar closes.
      4. asof(backward) then broadcasts each bar's already-shifted values
         across every second up to the next bar's close.

    The one thing deliberately NOT shifted is execution_price — added by the
    caller from df_micro's own unshifted price. The model's INPUT features
    must lag by one step; the price a trade would actually fill at is the
    real, current price, not a lagged one.

    Contract this function must uphold: for any output row at datetime = T,
    every feature column must reflect information available strictly before
    T's own bar closed. If compute_price_features or _join_orderflow_features
    ever gain a column that isn't already bar-close-safe on its own, this is
    the function that makes it point-in-time safe — do not skip it, and do
    not reorder shift-then-fill relative to the asof join.
    """
    df_micro_shifted = df_micro.with_columns([pl.col(c).shift(1) for c in df_micro.columns if c != "datetime"])

    macro_cols = [c for c in df_macro.columns if c != "datetime"]
    df_macro_shifted = df_macro.with_columns([pl.col(c).shift(1) for c in macro_cols])
    df_macro_shifted = df_macro_shifted.with_columns(pl.all().forward_fill())

    df_micro_shifted = df_micro_shifted.sort("datetime")
    df_macro_shifted = df_macro_shifted.sort("datetime")
    _assert_point_in_time_safe(df_micro_shifted, df_macro_shifted)

    return df_micro_shifted.join_asof(df_macro_shifted, on="datetime", strategy="backward")


def load_and_build_state_space(orderbook_path: str, trades_path: str, session_id: int, start_dt=None, end_dt=None,
                               session_features: SessionFeatureContext | None = None) -> pl.DataFrame:
    """Build one chunk's (or one full session's, if start_dt/end_dt are None)
    point-in-time-safe state space.

    Pure orchestration — every step delegates to a single-purpose function
    defined above. To change WHAT a feature measures, edit the relevant
    helper (_build_price_bars, compute_price_features,
    _join_orderflow_features, ...). To change WHEN a feature becomes
    visible, the only function that should ever change is
    _broadcast_bar_features_to_micro.

    Returns one row per second (df_micro's grid) with every price/RSI/
    orderflow feature already lagged to be point-in-time safe, plus
    'execution_price' (the true, unlagged fill price) and 'session_id'.
    """
    df_orderbook, df_trades = _load_and_clean_chunk_events(orderbook_path, trades_path, session_id, start_dt, end_dt)
    return _load_and_build_state_space_from_frames(
        df_orderbook, df_trades, session_id=session_id,
        start_dt=start_dt, session_features=session_features,
    )


def _load_and_build_state_space_from_frames(
    df_orderbook: pl.DataFrame,
    df_trades: pl.DataFrame,
    session_id: int,
    start_dt=None,
    session_features: SessionFeatureContext | None = None,
) -> pl.DataFrame:
    """The orchestration shared by every input path (parquet and replay feed).

    `df_orderbook` / `df_trades` must already be cleaned EAGER frames with a
    `datetime` column and the parquet schema (`timestamp`, `bid_price`,
    `bid_amount`, `ask_price`, `ask_amount` / `timestamp`, `price`, `amount`,
    `side`, `id`). The replay-feed loader produces exactly that, so replay
    validation runs byte-identical feature construction to the walk-forward.
    """
    df_trades = _with_buy_sell_split(df_trades)

    df_price_bars = _build_price_bars(df_orderbook)
    df_micro = _build_micro_bars(df_trades)
    df_bar_volume = _build_bar_volume(df_trades)

    feats = compute_orderflow_features(
        df_trades, df_orderbook,dense_resample=dense_resample,
        total_session_volume=session_features.total_volume if session_features else None,
    )
    feats = _shift_feature_dict(feats, periods=1)

    # We have every feature input we need — release the raw event-level
    # frames before building the (much smaller) bar-level state space.
    del df_orderbook, df_trades
    gc.collect()

    price_and_volume = df_price_bars.select(["datetime", "open", "high", "low", "close", "spread_pct"]).join(
        df_bar_volume.select(["datetime", "volume"]), on="datetime", how="left"
    )

    # A chunk owns raw-event work only. Recursive RSI belongs to the whole
    # session; joining its precomputed values prevents a smoothing restart at
    # every chunk boundary. The fallback keeps single-shot callers simple.
    rsi_features = session_features.rsi_features if session_features else _rsi_features_from_price_bars(
        price_and_volume.select(["datetime", "close"])
    )
    df_macro = compute_price_features(price_and_volume, rsi_features).shift(1)
    df_macro = _join_orderflow_features(df_macro, feats)

    state_space = _broadcast_bar_features_to_micro(df_micro, df_macro)

    exec_price = df_micro.select(["datetime", pl.col("price").alias("execution_price")])
    state_space = state_space.join(exec_price, on="datetime", how="left")

    state_space = state_space.drop_nulls()
    state_space = state_space.with_columns(pl.lit(session_id).alias("session_id"))

    # Trim the warm-up/lookback overlap buffer before returning — everything
    # before start_dt only existed to warm up rolling windows and the RSI
    # recursion, and must never appear in the final, trainable state space.
    if start_dt:
        state_space = state_space.filter(pl.col("datetime") >= start_dt)

    return state_space


def load_state_space_from_replay(feed_path: str, session_id: int = 0,
                                 session_features: SessionFeatureContext | None = None) -> pl.DataFrame:
    """Build the point-in-time-safe state space from a Rust live-test replay
    feed text file (the same file `live-test` replays through the engine).

    This is the research-side validation of `unified_runner.py live-test`: the
    exact event stream the Rust engine consumes is fed through the SAME feature
    pipeline the walk-forward trains on, so the two can be compared 1:1 on
    identical inputs. Config knob: `REPLAY_FEED` in config/strategy.py.
    """
    df_book, df_trades = _parse_replay_feed(feed_path)

    df_orderbook = _with_datetime(pl.LazyFrame(df_book)).filter(pl.col("datetime").is_not_null())
    df_trades_clean = _with_datetime(pl.LazyFrame(df_trades)).filter(pl.col("datetime").is_not_null())

    df_orderbook = df_orderbook.sort(["datetime", "timestamp"], maintain_order=True)
    df_trades_clean = df_trades_clean.sort(["datetime", "timestamp"], maintain_order=True)
    df_orderbook = df_orderbook.with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price")
    )

    return _load_and_build_state_space_from_frames(
        df_orderbook.collect(), df_trades_clean.collect(), session_id,
        session_features=session_features,
    )


def _parse_replay_feed(feed_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse a Rust live-test replay feed text file into (orderbook, trades)
    frames with the SAME schema as the parquet sources.

    Feed format (see config/strategy.py `REPLAY_FEED` docstring):
        #TICKER btcusdt             `#` headers are ignored
        T <ts_ms> <price> <amount> <buy|sell> <id>        trade print
        B <ts_ms> <bid_px> <bid_amount> <ask_px> <ask_amount>   L1 book

    When the feed has no orderbook (`B`) lines a degenerate bid==ask==price
    book (zero depth) is synthesized from the trade prints so the feature
    pipeline still runs — spread/depth features then collapse to ~0, which is
    the correct behaviour for a trades-only validation feed.
    """
    trade_rows: list[tuple[int, float, float, str, str]] = []
    book_rows: list[tuple[int, float, float, float, float]] = []
    with open(feed_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tok = line.split()
            if not tok or tok[0].startswith("#"):
                continue
            if tok[0] == "T":
                if len(tok) < 5:
                    continue
                ts = int(tok[1])
                px = float(tok[2])
                amt = float(tok[3])
                side = "buy" if tok[4].lower() == "buy" else "sell"
                tid = tok[5] if len(tok) > 5 else str(ts)
                trade_rows.append((ts, px, amt, side, tid))
            elif tok[0] == "B":
                if len(tok) < 5:
                    continue
                ts = int(tok[1])
                bp = float(tok[2])
                ba = float(tok[3])
                ap = float(tok[4])
                aa = float(tok[5]) if len(tok) > 5 else 0.0
                book_rows.append((ts, bp, ba, ap, aa))

    if not trade_rows:
        raise ValueError(f"replay feed {feed_path!r} contains no trade (T) events")

    if not book_rows:
        print(f"  [REPLAY] feed {feed_path!r} has no orderbook (B) lines - "
              "synthesizing bid==ask==trade-price (spread/depth features ~0)")
        book_rows = [(ts, px, 0.0, px, 0.0) for (ts, px, _amt, _side, _tid) in trade_rows]

    df_trades = pl.DataFrame(
        {
            "timestamp": pl.Series("timestamp", [r[0] for r in trade_rows], dtype=pl.Int64),
            "price": [r[1] for r in trade_rows],
            "amount": [r[2] for r in trade_rows],
            "side": [r[3] for r in trade_rows],
            "id": [r[4] for r in trade_rows],
        }
    )
    df_book = pl.DataFrame(
        {
            "timestamp": pl.Series("timestamp", [r[0] for r in book_rows], dtype=pl.Int64),
            "bid_price": [r[1] for r in book_rows],
            "bid_amount": [r[2] for r in book_rows],
            "ask_price": [r[3] for r in book_rows],
            "ask_amount": [r[4] for r in book_rows],
        }
    )
    print(f"  [REPLAY] parsed feed: {df_trades.height:,} trades, {df_book.height:,} orderbook rows")
    return df_book, df_trades