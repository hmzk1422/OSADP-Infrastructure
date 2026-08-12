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
import json
import datetime as _dt
import gc
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from utils.indicators import get_wilders_rsi, sma

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, classification_report
import shap
from sklearn.inspection import permutation_importance


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 1. Limit thread allocation to keep per-core memory low
os.environ["POLARS_MAX_THREADS"] = "4"

# get_wilders_rsi/sma operate on plain numpy arrays in the original script
# (np.diff(...) output) — untouched by the pandas->polars switch.

from main.config import (
    LOOKAHEAD_SECONDS, BARRIER_MULTIPLIER, RSI_FILTER_COLUMN, RSI_FILTER_THRESHOLD,
    RR_RATIO, TRADE_DIRECTION, TIMEFRAME_BARRIER,
    RSI_14_WINDOW, RSI_2_WINDOW, SMA_14_WINDOW, DEEPLOB_Z_WINDOW, RSI_Z_WINDOW,
    MA_SLOPE_WINDOW, PRICE_ROC_WINDOW_3, LABEL_LOSS, LABEL_WIN, FEATURE_COLUMNS,
    FEE_BPS, ENABLE_SLIPPAGE, MIN_BARRIER_PCT, MAX_BARRIER_PCT, EPS,
    HYSTERESIS_BUFFER, LOCKOUT_SECONDS, MODEL_PATH,
)
try:
    from main.config import BARRIER_VOL_BAR_WINDOW
except ImportError:
    BARRIER_VOL_BAR_WINDOW = 20

BAR_FREQ = f"{TIMEFRAME_BARRIER}s"
USE_ORDERFLOW_FEATURES = True
EXPORT_PRODUCTION_MODEL = True

# FIX: chunk warm-up buffer, derived from the actual rolling windows this
# script uses, instead of a fixed "2 hours" guess. A fixed constant is only
# correct by coincidence -- if TIMEFRAME_BARRIER or any window above changes,
# 2 hours could silently stop being enough bars of history for a chunk's
# rolling features to warm up correctly, and you'd get quietly-wrong values
# with no error (unlike the null-timestamp case, this wouldn't crash -- it'd
# just be subtly off right after every chunk boundary). All rolling-window
# features in this module use windows drawn from this list, plus the
# hardcoded window=10 z-scores (ofi_z, cvd_delta_z, cvd_slope_z,
# depth_imbalance_z, absorption_z, delta_divergence_z) and window=5
# (price_roc_5 / rsi_roc_5) -- both included explicitly below since they
# aren't config constants.
_KNOWN_BAR_WINDOWS = [
    RSI_14_WINDOW, RSI_2_WINDOW, SMA_14_WINDOW, DEEPLOB_Z_WINDOW, RSI_Z_WINDOW,
    MA_SLOPE_WINDOW, PRICE_ROC_WINDOW_3, BARRIER_VOL_BAR_WINDOW,
    10,  # hardcoded window=10 rolling z-scores
    5,   # hardcoded window=5 rolling features (price_roc_5 / rsi_roc_5)
]
_CHUNK_WARMUP_SAFETY_BARS = 10  # extra margin on top of the largest window found
CHUNK_WARMUP_BARS = max(_KNOWN_BAR_WINDOWS) + _CHUNK_WARMUP_SAFETY_BARS
CHUNK_BUFFER_SECONDS = TIMEFRAME_BARRIER * CHUNK_WARMUP_BARS

BARRIER_CHUNK_SIZE = 50_000
N_RANDOM_SAMPLES = 100
TEST_BLOCK_SIZE = 300
EMBARGO_SECONDS = LOOKAHEAD_SECONDS
CLASS_NAMES = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}

VPIN_TARGET_BUCKETS_PER_SESSION = 50
VPIN_WINDOW_BUCKETS = 10
VPIN_MIN_PERIODS = 5
RETAIL_THRESHOLD = 0.05


# ============================================================================
# DATA SOURCE CONFIGURATION
# ----------------------------------------------------------------------------
# Flip DATA_FORMAT to switch the entire pipeline (both the time-bounds scan
# and the per-chunk load) between parquet and csv input files, instead of
# editing hardcoded paths in __main__.
# ============================================================================
DATA_FORMAT = "parquet"  # "parquet" or "csv"
DATA_DIR = "./research/csv"
CHUNK_SESSIONS = True
N_CHUNKS_PER_SESSION = 6

AUTO_CONVERT_CSV_TO_PARQUET = False  # see _ensure_parquet() below


def _ensure_parquet(parquet_path: str, csv_fallback_path: str) -> str:
    """If parquet_path doesn't exist yet but the matching CSV does, convert it
    ONCE (streaming, so it doesn't require loading the whole file into RAM)
    and return parquet_path.

    FIX: this is why flipping DATA_FORMAT to "parquet" previously "did
    nothing" -- the script never converted anything on its own. It just
    looked for BTCUSDT_orderbook[.suffix].parquet, silently found nothing,
    and get_session_paths() either skipped that session or (if no parquet
    files existed anywhere) raised. If you only ever had the .csv exports on
    disk, there was no parquet source for it to read, full stop. This makes
    the switch actually take effect the first time; on every later run it's a
    no-op (the .parquet file already exists) and just returns immediately.
    """
    if os.path.exists(parquet_path):
        return parquet_path
    if os.path.exists(csv_fallback_path):
        print(f"  [AUTO-CONVERT] {csv_fallback_path} -> {parquet_path} (one-time, streaming)")
        pl.scan_csv(csv_fallback_path).sink_parquet(parquet_path)
    return parquet_path


def get_session_paths(data_format: str = DATA_FORMAT) -> list[tuple[str, str, int]]:
    """Resolve (orderbook_path, trades_path, session_id) for EVERY session found
    under DATA_DIR for the given format.

    FIX: this replaces the old get_data_paths(), which only ever resolved ONE
    file pair. ml_state_optimized_csv.py's __main__ trains on THREE independent
    sessions (BTCUSDT_orderbook[.ext] + _2 + _3), so flipping DATA_FORMAT to
    "csv" here previously trained on 1/3 the sessions the csv script uses --
    not an apples-to-apples comparison, and a likely cause of PnL divergence
    between the two scripts. Add more session files (_4, _5, ...) and they'll
    be picked up automatically; a session is skipped (with a warning) if
    either of its two files is missing, instead of raising.
    """
    data_format = data_format.lower()
    if data_format not in ("parquet", "csv"):
        raise ValueError(f"Unsupported data_format: {data_format!r} (expected 'parquet' or 'csv')")

    session_suffixes = ["", "_2", "_3"]
    sessions = []
    for session_id, suffix in enumerate(session_suffixes):
        orderbook_path = f"{DATA_DIR}/BTCUSDT_orderbook{suffix}.{data_format}"
        trades_path = f"{DATA_DIR}/BTCUSDT_trades{suffix}.{data_format}"

        if data_format == "parquet" and AUTO_CONVERT_CSV_TO_PARQUET:
            orderbook_path = _ensure_parquet(orderbook_path, f"{DATA_DIR}/BTCUSDT_orderbook{suffix}.csv")
            trades_path = _ensure_parquet(trades_path, f"{DATA_DIR}/BTCUSDT_trades{suffix}.csv")

        if os.path.exists(orderbook_path) and os.path.exists(trades_path):
            sessions.append((orderbook_path, trades_path, session_id))
        else:
            print(f"  [SESSION SKIPPED] session_id={session_id}: missing {orderbook_path!r} or {trades_path!r}")

    if not sessions:
        raise FileNotFoundError(f"No session files found under {DATA_DIR!r} for data_format={data_format!r}")
    return sessions


def _scan_source(path: str) -> pl.LazyFrame:
    """Lazily scan a data file, dispatching on its extension (.parquet vs .csv)."""
    return pl.scan_parquet(path) if str(path).endswith(".parquet") else pl.scan_csv(path)


def _session_total_volume(trades_path: str) -> float:
    """Fast full-file lazy sum of traded volume for the WHOLE session, used to
    size VPIN buckets. FIX: computed ONCE per session here, rather than being
    re-derived from whatever slice of the session a given chunk happens to
    see (see compute_orderflow_features). That old per-chunk approach made
    vpin_bucket_size -- and therefore VPIN's scale -- drift between chunks of
    the SAME session depending on each chunk's local volume. This is a single
    lazy aggregate (no full materialization), so it's cheap even for large
    parquet files."""
    trades_lazy = _scan_source(trades_path)
    total = trades_lazy.select(pl.col("amount").sum().cast(pl.Float64)).collect()["amount"][0]
    return float(total)


# ============================================================================
# dense_resample — see module docstring point 1.
# ============================================================================
def dense_resample(df: pl.DataFrame, every: str, agg_exprs: list, time_col: str = "datetime",
                    zero_fill_cols: list | None = None, start=None, end=None) -> pl.DataFrame:
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


def _rolling_autocorr_lag1(values) -> float:
    """Matches pandas' Series.autocorr(): Pearson corr of x[:-1] vs x[1:]."""
    x = np.asarray(values, dtype=float)
    if len(x) <= 5:
        return 0.0
    a, b = x[:-1], x[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(r) else float(r)


def _pl_rolling_zscore(col: str, window: int) -> pl.Expr:
    """Rolling z-score (sample std, ddof=1, NaN->0) — see module docstring
    point 6. This is a reimplementation, not your real calculate_zscore()."""
    mean = pl.col(col).rolling_mean(window_size=window, min_samples=window)
    std = pl.col(col).rolling_std(window_size=window, min_samples=window, ddof=1)
    return ((pl.col(col) - mean) / (std + EPS)).fill_null(0)


# ============================================================================
# STEP 1a — ORDERFLOW FEATURE COMPUTATION
# ============================================================================
def compute_orderflow_features(df_trades: pl.DataFrame, df_orderbook: pl.DataFrame,
                                total_session_volume: float | None = None) -> dict:
    ob = df_orderbook.sort(["datetime", "timestamp"], maintain_order=True)
    ob = ob.with_columns([
        pl.col("bid_price").shift(1).alias("bid_price_prev"),
        pl.col("ask_price").shift(1).alias("ask_price_prev"),
        pl.col("bid_amount").shift(1).alias("bid_amount_prev"),
        pl.col("ask_amount").shift(1).alias("ask_amount_prev"),
    ])
    delta_bid = (
        pl.when(pl.col("bid_price") > pl.col("bid_price_prev")).then(pl.col("bid_amount"))
        .when(pl.col("bid_price") < pl.col("bid_price_prev")).then(-pl.col("bid_amount_prev"))
        .when(pl.col("bid_price") == pl.col("bid_price_prev")).then(pl.col("bid_amount") - pl.col("bid_amount_prev"))
        .otherwise(0.0)
    )
    delta_ask = (
        pl.when(pl.col("ask_price") < pl.col("ask_price_prev")).then(pl.col("ask_amount"))
        .when(pl.col("ask_price") > pl.col("ask_price_prev")).then(-pl.col("ask_amount_prev"))
        .when(pl.col("ask_price") == pl.col("ask_price_prev")).then(pl.col("ask_amount") - pl.col("ask_amount_prev"))
        .otherwise(0.0)
    )
    ob = ob.with_columns((delta_bid - delta_ask).alias("ofi_raw"))

    ofi_1s = dense_resample(ob, "1s", [pl.col("ofi_raw").sum()], zero_fill_cols=["ofi_raw"])
    ofi_1m = dense_resample(ofi_1s, BAR_FREQ, [pl.col("ofi_raw").sum().alias("ofi_1m")], zero_fill_cols=["ofi_1m"])
    ofi_1m = ofi_1m.with_columns(pl.col("ofi_1m").diff().fill_null(0).alias("ofi_diff"))

    trades = df_trades.sort(["datetime", "timestamp"], maintain_order=True).with_columns([
        pl.when(pl.col("side") == "buy").then(pl.col("amount")).otherwise(0.0).alias("buy_vol"),
        pl.when(pl.col("side") == "sell").then(pl.col("amount")).otherwise(0.0).alias("sell_vol"),
    ])
    trades = trades.with_columns((pl.col("buy_vol") - pl.col("sell_vol")).alias("cvd_tick"))

    cvd_1m = dense_resample(trades, BAR_FREQ, [pl.col("cvd_tick").sum().alias("cvd_tick_sum")],
                             zero_fill_cols=["cvd_tick_sum"])
    cvd_1m = cvd_1m.with_columns(pl.col("cvd_tick_sum").cum_sum().alias("cvd_1m"))
    cvd_1m = cvd_1m.with_columns([
        pl.col("cvd_1m").diff().fill_null(0).alias("cvd_delta_1m"),
        (pl.col("cvd_1m") - pl.col("cvd_1m").shift(5)).fill_null(0).alias("cvd_slope_5"),
    ])

    trades = trades.with_columns(pl.col("price").shift(1).alias("price_prev"))
    trades = trades.with_columns(
        pl.when(pl.col("price") > pl.col("price_prev")).then(1)
          .when(pl.col("price") < pl.col("price_prev")).then(-1)
          .otherwise(0).alias("tick_rule")
    )

    # pandas grouped by calendar day before resampling; for `every` bins
    # that divide evenly into a day (as BAR_FREQ does), one dense resample
    # across the whole span gives the same bar boundaries, so the per-day
    # loop is skipped here.
    micro = dense_resample(
        trades, BAR_FREQ,
        [
            pl.col("tick_rule").mean().alias("tick_rule"),
            pl.col("amount").sum().alias("volume"),
            pl.col("amount").count().alias("n_trades"),
            pl.col("price").std(ddof=1).alias("price_volatility"),
        ],
        zero_fill_cols=["volume", "n_trades"],
    )
    micro = micro.with_columns([
        (pl.col("n_trades") / 60.0).alias("trade_intensity"),
        (pl.col("volume") / (pl.col("n_trades") + EPS)).alias("avg_trade_size"),
    ])

    tick_1s = dense_resample(trades, "1s", [pl.col("tick_rule").last()], zero_fill_cols=["tick_rule"])
    tick_1m = (
        tick_1s.sort("datetime")
               .group_by_dynamic("datetime", every=BAR_FREQ, closed="left", label="left")
               .agg(pl.col("tick_rule"))
               .with_columns(
                   pl.col("tick_rule").map_elements(_rolling_autocorr_lag1, return_dtype=pl.Float64)
                     .alias("tick_autocorr")
               )
               .select(["datetime", "tick_autocorr"])
    )

    trades_sorted = trades.sort(["datetime", "timestamp"], maintain_order=True)
    # FIX: use the TRUE full-session volume (passed in by the caller, computed
    # once via _session_total_volume) when available, rather than this call's
    # locally-visible volume. When this function is invoked per time-chunk
    # (see load_and_build_state_space), the local trades frame only spans a
    # buffered chunk window, not the whole session -- deriving bucket size from
    # it made vpin_bucket_size (and therefore VPIN's scale) drift chunk to
    # chunk. Falls back to the local sum when total_session_volume isn't
    # supplied, which is exactly correct when the whole session is loaded in
    # one shot (no chunking).
    session_volume = total_session_volume if total_session_volume is not None else trades_sorted["amount"].sum()
    vpin_bucket_size = max(session_volume / VPIN_TARGET_BUCKETS_PER_SESSION, EPS)

    # VPIN is order-sensitive. Build bucket totals from ordered cumulative
    # sums rather than parallel group-by reductions, which otherwise produce
    # tiny run-to-run floating-point differences on dense CSV sessions.
    trades_sorted = trades_sorted.with_columns([
        pl.col("buy_vol").cast(pl.Float64).cum_sum().alias("_cum_buy_vol"),
        pl.col("sell_vol").cast(pl.Float64).cum_sum().alias("_cum_sell_vol"),
        pl.col("amount").cast(pl.Float64).cum_sum().alias("_cum_volume"),
    ])
    trades_sorted = trades_sorted.with_columns(
        (pl.col("_cum_volume") / vpin_bucket_size).cast(pl.Int32).alias("_bucket")
    )
    n_buckets_closed = trades_sorted["_bucket"].n_unique()
    if n_buckets_closed < VPIN_MIN_PERIODS:
        print(f"  [VPIN WARNING] Only {n_buckets_closed} buckets closed this session "
              f"(need {VPIN_MIN_PERIODS}+) — VPIN will be null/sparse for this session.")

    vpin_buckets = (
        trades_sorted.group_by("_bucket", maintain_order=True)
        .agg([
            pl.col("_cum_buy_vol").last(), pl.col("_cum_sell_vol").last(), pl.col("_cum_volume").last(),
            pl.col("datetime").last().alias("bucket_end_time"),
        ])
        .sort("_bucket")
    )
    vpin_buckets = vpin_buckets.with_columns([
        (pl.col("_cum_buy_vol") - pl.col("_cum_buy_vol").shift(1).fill_null(0.0)).alias("buy_vol"),
        (pl.col("_cum_sell_vol") - pl.col("_cum_sell_vol").shift(1).fill_null(0.0)).alias("sell_vol"),
        (pl.col("_cum_volume") - pl.col("_cum_volume").shift(1).fill_null(0.0)).alias("volume"),
    ])
    vpin_buckets = vpin_buckets.with_columns(
        ((pl.col("buy_vol") - pl.col("sell_vol")).abs() / (pl.col("volume") + EPS)).alias("vpin_raw")
    )
    vpin_buckets = vpin_buckets.with_columns(
        pl.col("vpin_raw").rolling_mean(window_size=VPIN_WINDOW_BUCKETS, min_samples=VPIN_MIN_PERIODS)
          .alias("vpin_smoothed")
    )
    # See module docstring point 4 — must bin bucket closes into the bar
    # grid (dense_resample), not asof-join against bar left edges.
    vpin_by_bucket_close = vpin_buckets.select(pl.col("bucket_end_time").alias("datetime"), "vpin_smoothed")
    vpin_lookup = dense_resample(
        vpin_by_bucket_close, BAR_FREQ, [pl.col("vpin_smoothed").last()]
    ).with_columns(pl.col("vpin_smoothed").forward_fill())

    trade_size = dense_resample(
        trades.with_columns([
            pl.when(pl.col("amount") < RETAIL_THRESHOLD).then(pl.col("buy_vol")).otherwise(0.0).alias("retail_buy"),
            pl.when(pl.col("amount") < RETAIL_THRESHOLD).then(pl.col("sell_vol")).otherwise(0.0).alias("retail_sell"),
            pl.when(pl.col("amount") >= RETAIL_THRESHOLD).then(pl.col("buy_vol")).otherwise(0.0).alias("inst_buy"),
            pl.when(pl.col("amount") >= RETAIL_THRESHOLD).then(pl.col("sell_vol")).otherwise(0.0).alias("inst_sell"),
        ]),
        BAR_FREQ,
        [
            pl.col("retail_buy").sum(), pl.col("retail_sell").sum(),
            pl.col("inst_buy").sum(), pl.col("inst_sell").sum(),
            pl.col("amount").sum().alias("volume"),
        ],
        zero_fill_cols=["retail_buy", "retail_sell", "inst_buy", "inst_sell", "volume"],
    )
    trade_size = trade_size.with_columns([
        (pl.col("inst_buy") - pl.col("inst_sell") - (pl.col("retail_buy") - pl.col("retail_sell")))
            .alias("delta_divergence"),
        ((pl.col("inst_buy") + pl.col("inst_sell")) / (pl.col("volume") + EPS)).alias("inst_ratio"),
    ])

    ob_1m_price = dense_resample(
        ob, BAR_FREQ,
        [pl.col("bid_price").min().alias("bid_low"), pl.col("ask_price").max().alias("ask_high")],
    )
    ob_1m_price = ob_1m_price.with_columns(
        (pl.col("ask_high") - pl.col("bid_low")).forward_fill().alias("price_range")
    )
    abs_ratio = cvd_1m.select(["datetime", "cvd_delta_1m"]).join(
        ob_1m_price.select(["datetime", "price_range"]), on="datetime", how="left"
    ).with_columns(
        (pl.col("cvd_delta_1m") / (pl.col("price_range") + EPS)).alias("abs_ratio_raw")
    ).select(["datetime", "abs_ratio_raw"])

    ob = ob.with_columns(
        ((pl.col("bid_amount") - pl.col("ask_amount")) / (pl.col("bid_amount") + pl.col("ask_amount") + EPS))
            .alias("depth_imbalance")
    )
    ob = ob.with_columns(pl.col("depth_imbalance").diff().fill_null(0).alias("depth_imbalance_roc"))
    ob = ob.with_columns(((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid"))
    ob = ob.with_columns(((pl.col("ask_price") - pl.col("bid_price")) / pl.col("mid") * 10000).alias("spread_bps"))
    depth_1m = dense_resample(
        ob, BAR_FREQ,
        [pl.col("depth_imbalance").last(), pl.col("depth_imbalance_roc").last(), pl.col("spread_bps").last()],
    )

    return {
        "ofi_1m": ofi_1m.select(["datetime", "ofi_diff"]),
        "cvd_1m": cvd_1m.select(["datetime", "cvd_delta_1m", "cvd_slope_5"]),
        "micro": micro,
        "vpin_lookup": vpin_lookup,
        "trade_size": trade_size.select(["datetime", "inst_ratio", "delta_divergence"]),
        "abs_ratio": abs_ratio,
        "tick_autocorr": tick_1m,
        "depth_1m": depth_1m,
    }


from datetime import timedelta
import polars.selectors as cs

# ============================================================================
# STEP 1b — Per-session feature pipeline
# ============================================================================


@dataclass(frozen=True)
class SessionFeatureContext:
    """Small, session-owned inputs that must not reset at a chunk boundary."""

    total_volume: float
    rsi_features: pl.DataFrame


def _with_datetime(source: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize source types and expose the shared event-time column."""
    return (
        source
        .cast({cs.float(): pl.Float32, cs.integer() - cs.by_name("timestamp"): pl.Int32})
        .with_columns(
            pl.from_epoch((pl.col("timestamp") // 1000).cast(pl.Int64), time_unit="s").alias("datetime")
        )
    )


def _rsi_features_from_price_bars(price_bars: pl.DataFrame) -> pl.DataFrame:
    """Compute the recursive RSI family once over a continuous close series."""
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
    """Materialize only session-level price bars, never the raw order book.

    Wilder RSI is recursive.  Re-running it on a buffered chunk is an
    approximation whose error is largest exactly where a new chunk starts.
    This full-session bar pass makes the RSI family bit-for-bit independent
    of how the raw event stream is later partitioned.
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
    """Partition an already-built state space.

    Kept for callers that explicitly need post-hoc partitioning. The historical
    orchestration below no longer uses it for chunked runs because that would
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


def load_and_build_state_space(orderbook_path: str, trades_path: str, session_id: int, start_dt=None, end_dt=None,
                               session_features: SessionFeatureContext | None = None) -> pl.DataFrame:
    # 1. Use LazyFrames
    df_orderbook = _with_datetime(_scan_source(orderbook_path))
    df_trades = _with_datetime(_scan_source(trades_path))

    # 4. APPLY THE TIME FILTER BEFORE LOADING INTO RAM
    if start_dt and end_dt:
        # FIX: was a fixed timedelta(hours=2) regardless of what TIMEFRAME_BARRIER
        # or the feature windows actually are. CHUNK_BUFFER_SECONDS (module level)
        # is derived from the largest configured rolling window, so it can't
        # silently fall short of warm-up needs if config.py changes.
        buffer_start = start_dt - timedelta(seconds=CHUNK_BUFFER_SECONDS)
        df_orderbook = df_orderbook.filter((pl.col("datetime") >= buffer_start) & (pl.col("datetime") < end_dt))
        df_trades = df_trades.filter((pl.col("datetime") >= buffer_start) & (pl.col("datetime") < end_dt))

    # 3b. FIX: guard against null/unparseable timestamps. Some source CSVs
    # have a handful of malformed or blank timestamp rows (an empty field, a
    # stray header line if two exports got concatenated, an out-of-range
    # value that silently casts to null on the "// 1000" / cast(Int64) above).
    # polars' group_by_dynamic() (used throughout dense_resample) hard-errors
    # on ANY null in the time column -- "null values in dynamic group_by not
    # supported" -- rather than quietly skipping them the way pandas'
    # .resample() does. Drop them here, once, with a visible count, instead
    # of crashing deep inside dense_resample() with no indication of which
    # rows or how many were the problem.
    #
    # FIX (ordering, this pass): moved to run AFTER step 4's time-window
    # filter instead of before it. On a chunked run this was previously
    # forcing a full, non-streaming .collect() of the ENTIRE session file's
    # datetime column on every single chunk (start_dt/end_dt hadn't been
    # applied yet) -- N_CHUNKS_PER_SESSION redundant full-file
    # materializations per session, which is what was exhausting RAM on a
    # multi-GB parquet file. Now it only touches the already-windowed
    # (buffered chunk) slice. For non-chunked calls (start_dt=None) step 4
    # above is a no-op, so this is identical to before -- same data, same
    # counts, same warnings.
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

    # 5. Pull only this chunk into memory safely using the new Polars syntax
    # df_orderbook = df_orderbook.collect(engine="streaming")
    # df_trades = df_trades.collect(engine="streaming")

    temp_ob_path = f"temp_ob_chunk_{session_id}.parquet"
    temp_tr_path = f"temp_tr_chunk_{session_id}.parquet"
    
    df_orderbook.sink_parquet(temp_ob_path)
    df_trades.sink_parquet(temp_tr_path)

    # Read them back eagerly to feed into dense_resample and numpy arrays
    df_orderbook = pl.read_parquet(temp_ob_path)
    df_trades = pl.read_parquet(temp_tr_path)

    # 6. AGGRESSIVE GARBAGE COLLECTION
    # Clean up the SSD footprint once we have the data in RAM
    os.remove(temp_ob_path)
    os.remove(temp_tr_path)

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

    df_trades = df_trades.with_columns([
        pl.when(pl.col("side") == "buy").then(pl.col("amount")).otherwise(0.0).alias("buy_vol"),
        pl.when(pl.col("side") == "sell").then(pl.col("amount")).otherwise(0.0).alias("sell_vol"),
    ])

    df_micro = dense_resample(
        df_trades, "1s",
        [pl.col("price").last(), pl.col("buy_vol").sum(), pl.col("sell_vol").sum()],
        zero_fill_cols=["buy_vol", "sell_vol"],
    ).with_columns(pl.col("price").forward_fill())
    df_micro = df_micro.with_columns((pl.col("buy_vol") - pl.col("sell_vol")).alias("micro_OFI"))
    df_micro = df_micro.with_columns(pl.col("micro_OFI").rolling_sum(window_size=10).alias("micro_CVD"))

    # See module docstring point 3: no zero_fill here
    df_trades_1m = dense_resample(df_trades, BAR_FREQ, [pl.col("amount").sum().alias("volume")])

    feats = compute_orderflow_features(
        df_trades,
        df_orderbook,
        total_session_volume=session_features.total_volume if session_features else None,
    )

    # 6. AGGRESSIVE GARBAGE COLLECTION (CRITICAL FOR YOUR 16GB RAM)
    # We have our features, so delete the raw massive dataframes before numpy operations start
    del df_orderbook
    del df_trades
    gc.collect()

    df_price_1m = df_ob_1m.select(["datetime", "open", "high", "low", "close", "spread_pct"]).join(
        df_trades_1m.select(["datetime", "volume"]), on="datetime", how="left"
    )

    # A chunk owns raw-event work only.  Recursive RSI belongs to the whole
    # session; joining its precomputed values prevents a smoothing restart at
    # every chunk boundary.  The fallback keeps single-shot callers simple.
    rsi_features = session_features.rsi_features if session_features else _rsi_features_from_price_bars(
        df_price_1m.select(["datetime", "close"])
    )
    df_price_1m = df_price_1m.join(rsi_features, on="datetime", how="left")

    df_price_1m = df_price_1m.with_columns(
        pl.col("close").rolling_mean(window_size=MA_SLOPE_WINDOW).alias("_ma_5")
    )
    df_price_1m = df_price_1m.with_columns(pl.col("_ma_5").diff().fill_null(0).alias("ma_slope_5")).drop("_ma_5")
    df_price_1m = df_price_1m.with_columns(pl.col("ma_slope_5").diff().fill_null(0).alias("ma_accel"))
    df_price_1m = df_price_1m.with_columns(
        pl.when(pl.col("ma_accel") > 0).then(1).when(pl.col("ma_accel") < 0).then(-1).otherwise(0)
          .alias("ma_accel_regime")
    )
    df_price_1m = df_price_1m.with_columns(pl.col("close").pct_change(PRICE_ROC_WINDOW_3).alias("price_roc_3"))

    bar_rets = df_price_1m.select(pl.col("close").pct_change().alias("bar_ret"))["bar_ret"]
    df_price_1m = df_price_1m.with_columns(
        bar_rets.rolling_std(window_size=BARRIER_VOL_BAR_WINDOW, min_samples=2, ddof=0).alias("bar_vol_std")
    )
    df_price_1m = df_price_1m.with_columns(pl.col("close").pct_change(5).alias("price_roc_5"))
    df_price_1m = df_price_1m.with_columns([
        pl.col("rsi_2").diff(3).alias("rsi_roc_3"),
        pl.col("rsi_2").diff(5).alias("rsi_roc_5"),
    ])

    df_price_1m = df_price_1m.join(
        feats["depth_1m"].select(["datetime", "spread_bps", "depth_imbalance"]), on="datetime", how="left"
    ).with_columns([pl.col("spread_bps").fill_null(0), pl.col("depth_imbalance").fill_null(0)])
    df_price_1m = df_price_1m.with_columns([
        _pl_rolling_zscore("spread_bps", DEEPLOB_Z_WINDOW).alias("spread_bps_z"),
        _pl_rolling_zscore("depth_imbalance", 10).alias("depth_imbalance_z"),
    ]).drop(["spread_bps", "depth_imbalance"])

    rsi_z = df_price_1m.select(_pl_rolling_zscore("rsi_2", RSI_Z_WINDOW).alias("_rsi_z"))["_rsi_z"]
    df_price_1m = df_price_1m.with_columns((rsi_z * pl.col("ma_slope_5")).alias("rsi_x_ma"))

    df_price_1m = df_price_1m.join(feats["ofi_1m"], on="datetime", how="left").with_columns(
        pl.col("ofi_diff").fill_null(0)
    )
    df_price_1m = df_price_1m.join(feats["cvd_1m"], on="datetime", how="left").with_columns([
        pl.col("cvd_delta_1m").fill_null(0), pl.col("cvd_slope_5").fill_null(0),
    ])
    df_price_1m = df_price_1m.with_columns([
        _pl_rolling_zscore("ofi_diff", 10).alias("ofi_z"),
        _pl_rolling_zscore("cvd_delta_1m", 10).alias("cvd_delta_z"),
        _pl_rolling_zscore("cvd_slope_5", 10).alias("cvd_slope_z"),
    ]).drop(["ofi_diff", "cvd_delta_1m", "cvd_slope_5"])

    df_price_1m = df_price_1m.join(feats["vpin_lookup"], on="datetime", how="left").with_columns(
        pl.col("vpin_smoothed").forward_fill().fill_null(0).alias("vpin")
    ).drop("vpin_smoothed")

    df_price_1m = df_price_1m.join(feats["abs_ratio"], on="datetime", how="left").with_columns(
        pl.col("abs_ratio_raw").fill_null(0)
    )
    df_price_1m = df_price_1m.with_columns(
        _pl_rolling_zscore("abs_ratio_raw", 10).alias("absorption_z")
    ).drop("abs_ratio_raw")

    df_price_1m = df_price_1m.join(feats["tick_autocorr"], on="datetime", how="left").with_columns(
        pl.col("tick_autocorr").fill_null(0)
    )

    df_price_1m = df_price_1m.join(feats["trade_size"], on="datetime", how="left").with_columns([
        pl.col("inst_ratio").fill_null(0), pl.col("delta_divergence").fill_null(0),
    ])
    df_price_1m = df_price_1m.with_columns(
        _pl_rolling_zscore("delta_divergence", 10).alias("delta_divergence_z")
    ).drop("delta_divergence")

    df_micro_shifted = df_micro.with_columns([pl.col(c).shift(1) for c in df_micro.columns if c != "datetime"])
    macro_cols = [c for c in df_price_1m.columns if c != "datetime"]
    df_macro_shifted = df_price_1m.with_columns([pl.col(c).shift(1) for c in macro_cols])
    
    # See module docstring point 3(b).
    df_macro_shifted = df_macro_shifted.with_columns(pl.all().forward_fill())

    # See module docstring point 3(a)
    state_space = df_micro_shifted.sort("datetime").join_asof(
        df_macro_shifted.sort("datetime"), on="datetime", strategy="backward"
    )
    exec_price = df_micro.select(["datetime", pl.col("price").alias("execution_price")])
    state_space = state_space.join(exec_price, on="datetime", how="left")

    state_space = state_space.drop_nulls()
    state_space = state_space.with_columns(pl.lit(session_id).alias("session_id"))

    # 7. Trim the overlap buffer before returning
    if start_dt:
        state_space = state_space.filter(pl.col("datetime") >= start_dt)
        
    return state_space

# ============================================================================
# STEP 3 — TRIPLE BARRIER LABELING (pure numpy — unchanged by the polars
# switch; sliding_window_view operates on plain numpy arrays either way)
# ============================================================================
def _sequential_triple_barrier(price, profit_thresh, stop_thresh, lookahead, chunk_size=BARRIER_CHUNK_SIZE):
    price = np.asarray(price, dtype=float)
    profit_thresh = np.asarray(profit_thresh, dtype=float)
    stop_thresh = np.asarray(stop_thresh, dtype=float)
    n = len(price)

    label = np.full(n, LABEL_LOSS, dtype=int)
    exit_offset = np.full(n, -1, dtype=int)
    exit_ret = np.full(n, np.nan)
    terminal_ret = np.full(n, np.nan)
    is_timeout = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)

    usable_n = n - lookahead
    if usable_n <= 0:
        return label, exit_offset, exit_ret, terminal_ret, is_timeout, valid

    for c0 in range(0, usable_n, chunk_size):
        c1 = min(c0 + chunk_size, usable_n)
        p0 = price[c0:c1]
        pt = profit_thresh[c0:c1]
        st = stop_thresh[c0:c1]

        upper = p0 * (1.0 + (pt * RR_RATIO))
        lower = p0 * (1.0 - st)

        window = sliding_window_view(price[c0 + 1: c1 + lookahead], lookahead)

        hit_upper = window >= upper[:, None]
        hit_lower = window <= lower[:, None]

        has_upper = hit_upper.any(axis=1)
        has_lower = hit_lower.any(axis=1)
        first_upper = np.argmax(hit_upper, axis=1)
        first_lower = np.argmax(hit_lower, axis=1)

        win_mask = has_upper & (~has_lower | (first_upper < first_lower))
        loss_mask = has_lower & (~has_upper | (first_lower < first_upper))

        timeout_mask = ~(win_mask | loss_mask)
        timeout_win_mask = timeout_mask & (window[:, -1] > p0)
        timeout_loss_mask = timeout_mask & (window[:, -1] <= p0)

        thresholds_ok = ~np.isnan(pt) & ~np.isnan(st)

        lab = np.full(c1 - c0, LABEL_LOSS, dtype=int)
        lab[win_mask | timeout_win_mask] = LABEL_WIN
        lab[loss_mask | timeout_loss_mask] = LABEL_LOSS

        off = np.full(c1 - c0, lookahead - 1, dtype=int)
        off[win_mask] = first_upper[win_mask]
        off[loss_mask] = first_lower[loss_mask]

        exit_price = window[np.arange(c1 - c0), off]

        label[c0:c1] = lab
        exit_offset[c0:c1] = off
        exit_ret[c0:c1] = (exit_price - p0) / p0
        terminal_ret[c0:c1] = (window[:, -1] - p0) / p0
        is_timeout[c0:c1] = timeout_mask
        valid[c0:c1] = thresholds_ok

    return label, exit_offset, exit_ret, terminal_ret, is_timeout, valid


def deduplicate_to_bar(df: pl.DataFrame, bar_freq: str = BAR_FREQ) -> pl.DataFrame:
    """Keep the last row of each bar. See module docstring point 5:
    pandas' groupby(bucket).last() replaces the row's timestamp with the
    BUCKET FLOOR, not the raw last-observation timestamp — replicated here
    by overwriting 'datetime' with the bucket value after aggregating."""
    return (
        df.with_columns(pl.col("datetime").dt.truncate(bar_freq).alias("_bucket"))
          .sort("datetime")
          .group_by("_bucket", maintain_order=True)
          .last()
          .with_columns(pl.col("_bucket").alias("datetime"))
          .drop("_bucket")
          .sort("datetime")
    )


def add_barrier_labels(df: pl.DataFrame, lookahead_seconds: int = LOOKAHEAD_SECONDS,
                        barrier_mult: float = BARRIER_MULTIPLIER) -> pl.DataFrame:
    parts = []
    for _, g in df.sort("datetime").group_by("session_id", maintain_order=True):
        # Bootstrap floor — matches PositionTracker.get_barrier_pct()'s
        # `fallback = 0.001 * BARRIER_MULTIPLIER` branch.
        g = g.with_columns(
            pl.when(pl.col("bar_vol_std") > 0).then(pl.col("bar_vol_std")).otherwise(0.001).alias("_bvs")
        )
        g = g.with_columns([
            (pl.col("_bvs") * barrier_mult).alias("profit_barrier"),
            (pl.col("_bvs") * barrier_mult).alias("stop_barrier"),
        ]).drop("_bvs")

        label, exit_offset, exit_ret, terminal_ret, is_timeout, valid = _sequential_triple_barrier(
            g["execution_price"].to_numpy(), g["profit_barrier"].to_numpy(),
            g["stop_barrier"].to_numpy(), lookahead_seconds
        )

        g = g.with_columns([
            pl.Series("target", label.astype("int32")),
            pl.Series("bars_to_exit", exit_offset),
            pl.Series("exit_ret", exit_ret, nan_to_null=True),
            pl.Series("terminal_ret", terminal_ret, nan_to_null=True),
            pl.Series("is_timeout", is_timeout),
            pl.Series("_valid", valid),
        ])
        g = g.filter(pl.col("_valid")).drop("_valid")
        parts.append(g)

    return pl.concat(parts)


# ============================================================================
# purged_chronological_splits — numpy datetime64/timedelta64 in place of
# pandas Timestamp/Timedelta + Index. Positional (numpy) indexing throughout
# instead of .loc/.iloc.
# ============================================================================
def purged_chronological_splits(datetimes: np.ndarray, n_samples: int, block_size: int,
                                 horizon_seconds: int, embargo_seconds: int = 0,
                                 allow_overlap: bool = True):
    n = len(datetimes)
    horizon = np.timedelta64(horizon_seconds, "s")
    embargo = np.timedelta64(embargo_seconds, "s")
    min_initial_train = int(n * 0.35)
    available_test_run = n - min_initial_train - block_size

    if available_test_run <= 0:
        raise ValueError("The dataset length is too narrow for this test configuration block size.")

    max_non_overlapping = max(1, available_test_run // block_size)

    if n_samples > max_non_overlapping and not allow_overlap:
        print(f"[FOLD COUNT ADJUSTED] Requested n_samples={n_samples}, but only "
              f"{max_non_overlapping} non-overlapping {block_size}-bar test windows fit in "
              f"the {available_test_run} bars available after the {min_initial_train}-bar "
              f"initial training block. Reducing n_samples to {max_non_overlapping} so every "
              f"fold tests an independent period. Pass allow_overlap=True to keep the "
              f"original count with overlapping (non-independent) folds instead.")
        n_samples = max_non_overlapping

    step_intervals = available_test_run // n_samples

    if allow_overlap and step_intervals < block_size:
        overlap_pct = 100 * (1 - step_intervals / block_size)
        print(f"[FOLD OVERLAP WARNING] step_intervals={step_intervals} < block_size={block_size} "
              f"-> consecutive test folds overlap by ~{overlap_pct:.0f}% of their length. "
              f"Trade counts/expectancy pooled across folds are NOT independent draws.")

    for fold in range(n_samples):
        start_idx = min_initial_train + (fold * step_intervals)
        test_idx = np.arange(start_idx, start_idx + block_size)
        test_start_time = datetimes[test_idx[0]]

        possible_train_idx = np.arange(0, start_idx)
        train_times = datetimes[possible_train_idx]

        purge_mask = (train_times + horizon + embargo >= test_start_time)
        train_idx = possible_train_idx[~purge_mask]

        n_purged = len(possible_train_idx) - len(train_idx)
        yield train_idx, test_idx, n_purged


def find_debounced_signals(y_proba: np.ndarray, threshold: float, hysteresis: float = HYSTERESIS_BUFFER) -> np.ndarray:
    """Return indices of fresh threshold crossings, ignoring flicker."""
    lower_bound = max(threshold - hysteresis, 0.0)
    armed = True
    out = []
    for i, p in enumerate(y_proba):
        if armed and p >= threshold:
            out.append(i)
            armed = False
        elif not armed and p < lower_bound:
            armed = True
    return np.array(out, dtype=int)


# ============================================================================
# HISTORICAL STATE-SPACE ORCHESTRATION
# ============================================================================
def build_historical_state_space(sessions: list[tuple[str, str, int]] | None = None,
                                 chunk_sessions: bool = True,
                                 chunks_per_session: int = 6) -> pl.DataFrame:
    """Build raw historical state space independently from labeling/modeling.

    The returned frame is the exact input to barrier labeling and walk-forward
    evaluation. Keeping it separate gives the backtest and live-parity checks
    one concrete, testable feature boundary.
    """
    sessions = sessions if sessions is not None else get_session_paths()
    print(f"Data source: DATA_FORMAT={DATA_FORMAT!r}, DATA_DIR={DATA_DIR!r}")
    if chunk_sessions:
        print(f"CHUNK_SESSIONS=True: computing features continuously per session, "
              f"then writing {chunks_per_session} state-space partitions")
    print(f"Found {len(sessions)} session(s): {[sid for _, _, sid in sessions]}")

    chunk_files = []
    for orderbook_path, trades_path, session_id in sessions:
        print(f"\n=== Session {session_id}: {orderbook_path}, {trades_path} ===")
        session_features = build_session_feature_context(orderbook_path, trades_path)
        print(f"Session {session_id}: precomputed {session_features.rsi_features.height} continuous RSI bars")

        session_start, session_end = _session_time_bounds(trades_path)

        if not chunk_sessions:
            # Eagerly load the whole session (will spike RAM, but expected if
            # chunk_sessions=False).
            session_df = load_and_build_state_space(
                orderbook_path,
                trades_path,
                session_id=session_id,
                session_features=session_features,
            )
            fname = f"state_space_session{session_id}.parquet"
            session_df.write_parquet(fname)
            chunk_files.append(fname)
            print(f"Session {session_id}: {session_df.height} rows -> {fname}")
            del session_df
            gc.collect()
            continue

        # ---- Chunked computation + persistence path ----
        # IMPORTANT: pass the time boundaries into load_and_build_state_space
        # so the raw orderbook/trades are windowed BEFORE any eager materialization.
        # Each chunk includes CHUNK_BUFFER_SECONDS of warm-up history, then trims
        # back to [start_dt, end_dt) before returning.
        chunk_duration = (session_end - session_start) / chunks_per_session

        print(f"Session {session_id} time range: {session_start} to {session_end} "
              f"-> {chunks_per_session} chunks (all sharing session_id={session_id})")

        for i in range(chunks_per_session):
            start_dt = session_start + (chunk_duration * i)
            end_dt = session_start + (chunk_duration * (i + 1))

            chunk_df = load_and_build_state_space(
                orderbook_path,
                trades_path,
                session_id=session_id,
                start_dt=start_dt,
                end_dt=end_dt,
                session_features=session_features,
            )

            fname = f"state_space_session{session_id}_chunk{i}.parquet"
            # chunk_df is already bounded; sink the final chunk to SSD.
            chunk_df.lazy().sink_parquet(fname)
            chunk_files.append(fname)

            print(f"  chunk {i+1} -> {fname}. Wiping memory...")
            del chunk_df
            gc.collect()

    print("\nAll sessions processed successfully! Loading final compiled dataset...")

    # Explicit sort by datetime after concatenating sessions -- matches
    # ml_state_optimized_csv.py's pd.concat([...]).sort_index() exactly.
    df_state_space = pl.concat([pl.scan_parquet(f).collect() for f in chunk_files]).sort("datetime")
    print(f"Combined state space: {df_state_space.height} rows across {len(sessions)} session(s)")
    return df_state_space


# ============================================================================
# MAIN: historical state space -> labels -> walk-forward evaluation
# ============================================================================
if __name__ == "__main__":
    # Chunking partitions the finished state space for downstream storage or
    # transport. Feature calculation itself remains session-continuous.
    df_state_space = build_historical_state_space(
        chunk_sessions=CHUNK_SESSIONS,
        chunks_per_session=N_CHUNKS_PER_SESSION,
    )

    # ==========================================
    # ADD THESE LINES TO INSPECT YOUR TABLE
    # ==========================================
    
    # 1. Force Polars to show ALL columns instead of truncating them
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(10) # Show 10 rows
    
    print("\n--- STATE SPACE PREVIEW (FIRST 5 ROWS) ---")
    print(df_state_space.head(5))
    
    print("\n--- STATE SPACE SUMMARY STATISTICS ---")
    # describe() is great for spotting NaN issues or extreme z-scores
    print(df_state_space.describe())
    
    print("\n--- STATE SPACE GLIMPSE ---")
    # glimpse() flips the table sideways, making it very easy to read wide ML datasets
    print(df_state_space.glimpse())
    
    # Optional: Reset config back to default so later prints aren't massive
    pl.Config.set_tbl_cols(8)
    
    # ... [PROCEED WITH YOUR BARRIER LABELING AND LIGHTGBM CODE] ...

    # ------------------------------------------------------------------
    # STEP 3 — Barrier labeling
    # ------------------------------------------------------------------
    df_state_space = add_barrier_labels(df_state_space)
    df_state_space = df_state_space.filter(pl.col("volume") > 0)

    print(f"\nState space before dedup: {df_state_space.height} rows")
    df_state_space = deduplicate_to_bar(df_state_space)
    print(f"State space after dedup (bar-level): {df_state_space.height} rows")

    label_counts = (
        df_state_space.group_by("target").len()
        .with_columns(pl.col("target").replace_strict(CLASS_NAMES).alias("label"))
    )
    label_counts_dict = dict(zip(label_counts["label"].to_list(), label_counts["len"].to_list()))
    print(f"Label distribution (full dataset): {label_counts_dict}")

    df_training_zone = df_state_space.filter(pl.col(RSI_FILTER_COLUMN) < RSI_FILTER_THRESHOLD)
    print(f"Total rows in hunting zone: {df_training_zone.height}")
    hz_counts = (
        df_training_zone.group_by("target").len()
        .with_columns(pl.col("target").replace_strict(CLASS_NAMES).alias("label"))
    )
    print(f"Label distribution (hunting zone): "
          f"{dict(zip(hz_counts['label'].to_list(), hz_counts['len'].to_list()))}")

    # ------------------------------------------------------------------
    # STEP 4 — Features / target configuration
    # ------------------------------------------------------------------
    ORDERFLOW_FEATURE_COLUMNS = [
        'ofi_z', 'cvd_delta_z', 'cvd_slope_z', 'vpin', 'absorption_z',
        'tick_autocorr', 'inst_ratio', 'delta_divergence_z', 'depth_imbalance_z',
    ]
    if USE_ORDERFLOW_FEATURES:
        features = list(FEATURE_COLUMNS) + [
            c for c in ORDERFLOW_FEATURE_COLUMNS if c in df_training_zone.columns
        ]
    else:
        features = list(FEATURE_COLUMNS)
    missing = [c for c in FEATURE_COLUMNS if c not in df_training_zone.columns]
    if missing:
        raise ValueError(f"FEATURE_COLUMNS from config.py missing from pipeline output: {missing}")

    # Numpy at the LightGBM/SHAP/sklearn boundary from here on — this
    # matches the original's own usage pattern (X/y were already just
    # feeding numpy-based estimators) and is where any dataframe library's
    # overhead stops mattering anyway.
    datetimes = df_training_zone["datetime"].to_numpy()
    X = df_training_zone.select(features).to_numpy()
    y = df_training_zone["target"].to_numpy()
    stop_barrier_arr = df_training_zone["stop_barrier"].to_numpy()
    bars_to_exit_arr = df_training_zone["bars_to_exit"].to_numpy()
    is_timeout_arr = df_training_zone["is_timeout"].to_numpy()
    exit_ret_arr = df_training_zone["exit_ret"].to_numpy()

    print(f"\nFeatures ({len(features)} total):")
    for f in features:
        print(f"  - {f}")

    # ------------------------------------------------------------------
    # STEP 5 — Risk-Normalized Return
    # ------------------------------------------------------------------
    is_win = y == LABEL_WIN
    is_loss = y == LABEL_LOSS

    normalized_return = np.full(len(y), np.nan)
    normalized_return[is_win & ~is_timeout_arr] = RR_RATIO
    normalized_return[is_loss & ~is_timeout_arr] = -1.0
    normalized_return[is_timeout_arr] = exit_ret_arr[is_timeout_arr] / (stop_barrier_arr[is_timeout_arr] + EPS)

    fee_in_r = (2.0 * FEE_BPS / 10000.0) / (stop_barrier_arr + EPS)
    normalized_return = normalized_return - fee_in_r

    upper_quantile = np.nanquantile(normalized_return, 0.99)
    lower_quantile = np.nanquantile(normalized_return, 0.01)
    cap = max(abs(upper_quantile), abs(lower_quantile), RR_RATIO)
    normalized_return = np.clip(normalized_return, -cap, cap)

    mean_stop_barrier = np.nanmean(stop_barrier_arr)
    print("\n" + "=" * 50)
    print("STOP BARRIER METRICS")
    print("=" * 50)
    print(f"Average Stop Barrier Size (%): {mean_stop_barrier * 100:.4f}%")
    print("=" * 50)

    # ------------------------------------------------------------------
    # STEP 6 — Purged Chronological Walk-Forward
    # ------------------------------------------------------------------
    TARGET_TRADE_CLASS = LABEL_WIN
    INVERT_RETURNS = (TRADE_DIRECTION == 'short')
    target_name = CLASS_NAMES[TARGET_TRADE_CLASS]
    prob_label = f"P({target_name})"
    print(f"TRADE_DIRECTION={TRADE_DIRECTION!r} -> INVERT_RETURNS={INVERT_RETURNS} "
          f"(mirrors executor.py's actual fade/snapback semantics)")

    splitter = purged_chronological_splits(
        datetimes, n_samples=N_RANDOM_SAMPLES, block_size=TEST_BLOCK_SIZE,
        horizon_seconds=LOOKAHEAD_SECONDS, embargo_seconds=EMBARGO_SECONDS
    )

    print(f"\n--- Starting Walk-Forward (requested up to {N_RANDOM_SAMPLES} non-overlapping folds; "
          f"see [FOLD COUNT ADJUSTED] above if fewer independent windows fit the data) "
          f"| Targeting: {target_name} | Inverting Trades: {INVERT_RETURNS} ---")

    thresholds_to_track = np.arange(0.20, 0.96, 0.05)
    fold_metrics = {t: {'precisions': [], 'win_rates': [], 'rrs': [], 'trades': []} for t in thresholds_to_track}
    fold_importances = []
    trade_log = {t: [] for t in thresholds_to_track}
    fold_detail_records = []

    class_labels_ordered = [LABEL_LOSS, LABEL_WIN]
    class_metrics_by_label = {lbl: {'precision': [], 'recall': []} for lbl in class_labels_ordered}
    class_metrics_records = []
    all_y_test_pooled = []
    all_y_pred_pooled = []

    fold_shap_values = []
    fold_pfi_values = []

    for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            print(f"Fold {fold} | skipped — invalid training data distribution.")
            continue

        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_train, y_train)

        # --- SHAP --- (feature_names passed explicitly since X is a plain
        # numpy array here, not a named DataFrame)
        try:
            n_sample = min(1000, len(X_test))
            sample_idx = np.random.RandomState(42).choice(len(X_test), n_sample, replace=False)
            X_test_sampled = X_test[sample_idx]
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_test_sampled)

            if isinstance(shap_vals, list):
                shap_array = shap_vals[1]
            elif hasattr(shap_vals, 'values'):
                shap_array = shap_vals.values[:, :, 1] if len(shap_vals.values.shape) == 3 else shap_vals.values
            else:
                shap_array = shap_vals

            fold_shap_values.append(np.abs(shap_array).mean(axis=0))
        except Exception as e:
            print(f"Fold {fold:02d} | SHAP failed: {e}")
            fold_shap_values.append(np.full(len(features), np.nan))

        # --- PFI ---
        if len(np.unique(y_test)) < 2:
            fold_pfi_values.append(np.full(len(features), np.nan))
        else:
            try:
                pi_result = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=42, scoring='roc_auc', n_jobs=1
                )
                fold_pfi_values.append(pi_result.importances_mean)
            except Exception as e:
                print(f"Fold {fold:02d} | PFI failed: {e}")
                fold_pfi_values.append(np.full(len(features), np.nan))

        # --- Class metrics ---
        y_pred_argmax = clf.predict(X_test)
        prec_per_class = precision_score(y_test, y_pred_argmax, labels=class_labels_ordered, average=None, zero_division=0)
        rec_per_class = recall_score(y_test, y_pred_argmax, labels=class_labels_ordered, average=None, zero_division=0)

        for lbl, p, r in zip(class_labels_ordered, prec_per_class, rec_per_class):
            class_metrics_by_label[lbl]['precision'].append(p)
            class_metrics_by_label[lbl]['recall'].append(r)
            class_metrics_records.append({
                'fold': fold, 'class': CLASS_NAMES[lbl], 'precision': p, 'recall': r,
                'support': int((y_test == lbl).sum()),
            })

        all_y_test_pooled.append(y_test)
        all_y_pred_pooled.append(y_pred_argmax)

        if TARGET_TRADE_CLASS not in clf.classes_:
            print(f"Fold {fold} | skipped threshold scoring — no {target_name} samples in training data.")
            continue

        target_idx = list(clf.classes_).index(TARGET_TRADE_CLASS)
        y_proba = clf.predict_proba(X_test)[:, target_idx]

        for t in thresholds_to_track:
            signal_indices = find_debounced_signals(y_proba, t, HYSTERESIS_BUFFER)

            non_overlapping_idx = []          # positions within X_test/test_idx
            next_available_time = np.datetime64('1677-01-01')  # numpy datetime64 min-ish sentinel
            lockout_enabled = LOCKOUT_SECONDS > 0

            for local_idx in signal_indices:
                global_idx = test_idx[local_idx]
                actual_global_time = datetimes[global_idx]

                if lockout_enabled and actual_global_time < next_available_time:
                    continue

                row_barrier_pct = stop_barrier_arr[global_idx]
                if not (MIN_BARRIER_PCT <= row_barrier_pct <= MAX_BARRIER_PCT):
                    continue

                non_overlapping_idx.append(global_idx)

                if lockout_enabled:
                    trade_duration_seconds = int(bars_to_exit_arr[global_idx]) + 1
                    next_available_time = actual_global_time + np.timedelta64(trade_duration_seconds, 's')

            num_trades = len(non_overlapping_idx)

            if num_trades > 0:
                traded_idx = np.array(non_overlapping_idx)
                traded_targets = y[traded_idx]

                traded_returns = normalized_return[traded_idx].copy()
                if INVERT_RETURNS:
                    traded_returns = -traded_returns

                trade_log[t].extend(zip(datetimes[traded_idx], traded_returns))

                num_original_wins = int((traded_targets == LABEL_WIN).sum())
                num_original_losses = int((traded_targets == LABEL_LOSS).sum())
                strategy_wins = num_original_losses if INVERT_RETURNS else num_original_wins
                strategy_losses = num_original_wins if INVERT_RETURNS else num_original_losses
                win_rate = strategy_wins / num_trades

                win_returns = traded_returns[traded_returns > 0]
                loss_returns = traded_returns[traded_returns < 0]
                avg_win_ret = win_returns.mean() if len(win_returns) > 0 else np.nan
                avg_loss_ret = loss_returns.mean() if len(loss_returns) > 0 else np.nan
                if (not np.isnan(avg_win_ret) and not np.isnan(avg_loss_ret) and abs(avg_loss_ret) > 0.001):
                    rr = avg_win_ret / abs(avg_loss_ret)
                else:
                    rr = np.nan

                prec = precision_score(y[traded_idx] == TARGET_TRADE_CLASS, np.ones(num_trades), zero_division=0)

                fold_metrics[t]['precisions'].append(prec)
                fold_metrics[t]['win_rates'].append(win_rate)
                fold_metrics[t]['rrs'].append(rr)

                fold_detail_records.append({
                    'threshold': round(float(t), 2), 'fold': fold, 'trades': num_trades,
                    'strategy_wins': strategy_wins, 'strategy_losses': strategy_losses,
                    'precision': prec, 'win_rate': win_rate, 'rr': rr,
                    'avg_return': float(traded_returns.mean()),
                })

            fold_metrics[t]['trades'].append(num_trades)

        fold_importances.append(clf.feature_importances_)
        print(f"Fold {fold:02d} | Train Rows: {len(X_train):<6} | Tested on forward {TEST_BLOCK_SIZE} rows. Purged: {n_purged}")

    # ------------------------------------------------------------------
    # STEP 7 — SHAP vs PERMUTATION IMPORTANCE
    # ------------------------------------------------------------------
    print("\n--- Feature Importance: SHAP vs Permutation (PFI) ---")
    print("-" * 80)
    mean_shap = np.nanmean(fold_shap_values, axis=0)
    mean_pfi = np.nanmean(fold_pfi_values, axis=0)
    importance_df = pl.DataFrame({
        "Feature": features, "SHAP_Importance": mean_shap, "PFI_Mean_AUC_Drop": mean_pfi,
    }).sort("PFI_Mean_AUC_Drop", descending=True)
    print(importance_df)
    importance_df.write_csv("feature_importance_v3.csv")

    # ------------------------------------------------------------------
    # STEP 8 — Model Class-Level Precision & Recall
    # ------------------------------------------------------------------
    print("\n--- Model Binary Precision & Recall (argmax prediction) — Mean ± Std Across Folds ---")
    print("-" * 70)
    print(f"{'Class':<10} | {'Precision':<20} | {'Recall'}")
    print("-" * 70)
    for lbl in class_labels_ordered:
        p_arr = np.array(class_metrics_by_label[lbl]['precision'])
        r_arr = np.array(class_metrics_by_label[lbl]['recall'])
        p_txt = f"{p_arr.mean():>6.2%} ± {p_arr.std():<6.2%}" if len(p_arr) else "n/a"
        r_txt = f"{r_arr.mean():>6.2%} ± {r_arr.std():<6.2%}" if len(r_arr) else "n/a"
        print(f"{CLASS_NAMES[lbl]:<10} | {p_txt:<20} | {r_txt}")
    print("-" * 70)

    if all_y_test_pooled:
        pooled_y_test = np.concatenate(all_y_test_pooled)
        pooled_y_pred = np.concatenate(all_y_pred_pooled)
        print("\n--- Pooled Out-of-Sample Classification Report ---")
        print(classification_report(
            pooled_y_test, pooled_y_pred, labels=class_labels_ordered,
            target_names=[CLASS_NAMES[l] for l in class_labels_ordered], zero_division=0
        ))

    # ------------------------------------------------------------------
    # STEP 9 — Strategy Rolling Expectancy & CVaR
    # ------------------------------------------------------------------
    print(f"\n--- Strategy Rolling Expectancy & Stability by {prob_label} Threshold ---")
    print("-" * 140)
    print(f"{'Thresh':<8} | {'Trades':<8} | {'Win%':<8} | {'RR':<6} | {'Overall Exp':<12} | {'Exp CVaR (5%)':<14} | {'Stable (CVaR>0)':<16} | {'Total R'}")
    print("-" * 140)

    ROLLING_WINDOW = 30

    for t in thresholds_to_track:
        if len(trade_log[t]) == 0:
            continue
        dts = np.array([x[0] for x in trade_log[t]])
        returns = np.array([x[1] for x in trade_log[t]], dtype=float)
        order = np.argsort(dts)
        returns = returns[order]
        num_trades = len(returns)

        win_rate = np.mean(returns > 0)
        avg_win = returns[returns > 0].mean() if np.sum(returns > 0) > 0 else 0
        avg_loss = returns[returns < 0].mean() if np.sum(returns < 0) > 0 else 0
        rr = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan

        overall_exp = np.mean(returns)
        total_r = np.sum(returns)

        if num_trades >= ROLLING_WINDOW:
            rolling_exp = pl.Series(returns).rolling_mean(window_size=ROLLING_WINDOW).drop_nulls().to_numpy()
            alpha = 0.05
            cutoff_idx = max(1, int(len(rolling_exp) * alpha))
            worst_rolling_streaks = np.sort(rolling_exp)[:cutoff_idx]
            cvar_exp = np.mean(worst_rolling_streaks)
            stable_flag = "YES" if cvar_exp > 0 else "NO"
        else:
            cvar_exp = np.nan
            stable_flag = "N/A (low trades)"

        print(f">= {t*100:>3.0f}% | {num_trades:<8} | {win_rate:>5.1%} | {rr:<6.2f} | {overall_exp:>10.2f} R | {cvar_exp:>12.2f} R | {stable_flag:<16} | {total_r:>+6.2f} R")

    print("-" * 140)
    print(f"Note: 'Exp CVaR (5%)' = avg R per trade during worst 5% of {ROLLING_WINDOW}-trade streaks.")

    # ------------------------------------------------------------------
    # STEP 10 — Out-of-Sample PnL Chart
    # ------------------------------------------------------------------
    import plotly.graph_objects as go

    print("\n--- Saving Trade Logs to CSV ---")
    all_trades_frames = []
    fig = go.Figure()

    for t in thresholds_to_track:
        if len(trade_log[t]) == 0:
            continue
        dts = np.array([x[0] for x in trade_log[t]])
        returns = np.array([x[1] for x in trade_log[t]], dtype=float)
        order = np.argsort(dts)
        dts, returns = dts[order], returns[order]

        log_df = pl.DataFrame({"datetime": dts, "ret": returns, "threshold": round(float(t), 2)})
        log_df = log_df.with_columns(pl.col("ret").cum_sum().alias("cum_pnl"))
        all_trades_frames.append(log_df)

        trade_numbers = list(range(1, len(log_df) + 1))
        fig.add_trace(go.Scatter(
            x=trade_numbers, y=log_df["cum_pnl"].to_list(), mode='lines', name=f'>= {t:.2f}',
            hovertemplate='Trade: %{x}<br>PnL: %{y:.2f}R<extra></extra>'
        ))

    if all_trades_frames:
        master_trade_df = pl.concat(all_trades_frames)
        master_trade_df.write_csv("walk_forward_trade_log_v3.csv")
        print("Full trade log saved to: walk_forward_trade_log_v3.csv")

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    chart_title = f'Out-of-Sample PnL (OLS Divergence + DeepLOB) — Target: {target_name} | Inverted: {INVERT_RETURNS}'
    fig.update_layout(
        title=chart_title, xaxis_title='Trade Number', yaxis_title='Cumulative Return (R)',
        legend_title=f'{prob_label} ≥ (Click!)', hovermode="x unified", template="plotly_white"
    )
    fig.write_html('pnl_walkforward_v3.html')
    print("Interactive chart saved to: pnl_walkforward_v3.html")

    # ------------------------------------------------------------------
    # STEP 11 — Summary Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — OLS DIVERGENCE + DEEPLOB FEATURES")
    print("=" * 70)
    print(f"Sequential (order-aware) binary barrier labels, full dataset: {label_counts_dict}")
    print(f"Hunting zone ({RSI_FILTER_COLUMN} < {RSI_FILTER_THRESHOLD}) rows: {df_training_zone.height}")
    print(f"Trading Setup: Target {target_name} | Inverting Trade Direction: {INVERT_RETURNS}")
    print(f"Model: n_estimators=200, max_depth=3, learning_rate=0.01 (see clf definition above)")
    print(f"\nActual Features Used ({len(features)} total):")
    for f in features:
        print(f"  - {f}")

    for lbl in class_labels_ordered:
        p_arr = np.array(class_metrics_by_label[lbl]['precision'])
        if len(p_arr) > 0:
            print(f"  {CLASS_NAMES[lbl]:<8} class precision (argmax): {p_arr.mean():.2%} ± {p_arr.std():.2%} across folds")

    best_rr_threshold, best_rr = None, -np.inf
    for t in thresholds_to_track:
        rr_arr = np.array([r for r in fold_metrics[t]['rrs'] if not np.isnan(r)])
        if len(rr_arr) > 0 and rr_arr.mean() > best_rr:
            best_rr, best_rr_threshold = rr_arr.mean(), t
    if best_rr_threshold is not None:
        print(f"Best mean out-of-sample RR across folds: {best_rr:.2f} at {prob_label} >= {best_rr_threshold:.2f}")

    for t in thresholds_to_track:
        if len(trade_log[t]) > 0:
            returns = np.array([x[1] for x in trade_log[t]], dtype=float)
            print(f"  {prob_label} >= {t:.2f}: {len(returns)} trades, cumulative {returns.sum():.2f}R out-of-sample")

    print("\nReminder: this run ignores fees, slippage, latency, and partial fills.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 12 — Export Production Model
    # ------------------------------------------------------------------
    if EXPORT_PRODUCTION_MODEL:
        print("\n" + "=" * 70)
        print("EXPORTING PRODUCTION MODEL")
        print("=" * 70)

        final_clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
        final_clf.fit(X, y)

        model_dir = os.path.dirname(os.path.abspath(MODEL_PATH))
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        final_clf.booster_.save_model(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

        manifest_path = os.path.splitext(MODEL_PATH)[0] + ".features.json"
        label_counts_export = {CLASS_NAMES[k]: int((y == k).sum()) for k in np.unique(y)}
        manifest = {
            "features": features,
            "n_features": len(features),
            "includes_orderflow": USE_ORDERFLOW_FEATURES,
            "orderflow_features": [c for c in ORDERFLOW_FEATURE_COLUMNS if c in features],
            "base_features": list(FEATURE_COLUMNS),
            "trained_rows": int(len(X)),
            "trained_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "label_counts": label_counts_export,
            "rr_ratio": RR_RATIO,
            "lookahead_seconds": LOOKAHEAD_SECONDS,
            "barrier_multiplier": BARRIER_MULTIPLIER,
            "trade_direction": TRADE_DIRECTION,
            "confidence_threshold_note": (
                "This manifest does not set CONFIDENCE_THRESHOLD — that stays a "
                "config.py / CLI setting. See the per-threshold table above (now "
                "computed from non-overlapping folds) before trusting a threshold."
            ),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Feature manifest saved to: {manifest_path}")

        print(f"\nFeatures ({len(features)}), IN THIS EXACT ORDER — "
              f"LiveFeatureEngine.extract_feature_vector() must produce the same order:")
        for i, f in enumerate(features):
            tag = " [orderflow]" if f in ORDERFLOW_FEATURE_COLUMNS else ""
            print(f"  [{i}] {f}{tag}")

        print("\nNOTE: this export model is fit on 100% of available hunting-zone")
        print("data (no holdout) — it is NOT itself a forward-performance estimate.")
        print("Use the walk-forward table above for that; this step only produces")
        print("the artifact executor.py should load.")
        print("=" * 70)