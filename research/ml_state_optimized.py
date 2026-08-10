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
<<<<<<< HEAD
import datetime as _dt
import gc

=======
# Make the project root importable regardless of CWD. The old
# os.path.join(os.getcwd(), '..') only worked when run from inside research/.
>>>>>>> origin/main
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

# get_wilders_rsi/sma operate on plain numpy arrays in the original script
# (np.diff(...) output) — untouched by the pandas->polars switch.
from utils.indicators import get_wilders_rsi, sma

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, classification_report
import shap
from sklearn.inspection import permutation_importance
<<<<<<< HEAD
=======
from sklearn.linear_model import LinearRegression
import polars as pl
>>>>>>> origin/main

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

<<<<<<< HEAD
BARRIER_CHUNK_SIZE = 50_000
N_RANDOM_SAMPLES = 20
TEST_BLOCK_SIZE = 2000
EMBARGO_SECONDS = LOOKAHEAD_SECONDS
CLASS_NAMES = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}
=======
# FIX: orderflow/DeepLOB features (OFI, CVD, VPIN, absorption, tick
# autocorrelation, retail/institutional delta, depth imbalance) used to be
# unconditionally appended to `features` whenever the columns existed in
# df_training_zone — with no way to compare against the base-feature-only
# model that's actually what HistoricalDataBuilder trains live. Toggle it
# explicitly here. When True, both the walk-forward chart AND the exported
# production model (STEP 12) include them; the feature manifest written at
# export time records which set was used, so live can verify what it's
# actually loading rather than assume.
USE_ORDERFLOW_FEATURES = True

# FIX: export the final model after walk-forward evaluation. This is what
# gets loaded by executor.py — see STEP 12 at the bottom of this file. Set
# False if you only want to run the evaluation and not touch MODEL_PATH.
EXPORT_PRODUCTION_MODEL = True

BARRIER_CHUNK_SIZE   = 50_000
N_RANDOM_SAMPLES      = 15
# TEST_BLOCK_SIZE was 10000 rows when the state space was 1-second resolution
# (≈10000 consecutive seconds). After deduplicate_to_bar() the dataset is
# bar-level (~4500 bars across 3 sessions), so 10000 rows no longer fits.
# 150 bars = 12.5 hours of 5-minute bars per test fold.
TEST_BLOCK_SIZE       = 150
EMBARGO_SECONDS       = LOOKAHEAD_SECONDS
CLASS_NAMES           = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}
>>>>>>> origin/main

VPIN_TARGET_BUCKETS_PER_SESSION = 50
VPIN_WINDOW_BUCKETS = 10
VPIN_MIN_PERIODS = 5
RETAIL_THRESHOLD = 0.05


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
    agg = (
        df.sort(time_col)
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
def compute_orderflow_features(df_trades: pl.DataFrame, df_orderbook: pl.DataFrame) -> dict:
    ob = df_orderbook.sort("datetime")
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

    trades = df_trades.sort("datetime").with_columns([
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

    trades_sorted = trades.sort("datetime")
    total_session_volume = trades_sorted["amount"].sum()
    vpin_bucket_size = max(total_session_volume / VPIN_TARGET_BUCKETS_PER_SESSION, EPS)

    trades_sorted = trades_sorted.with_columns(
        (pl.col("amount").cum_sum() / vpin_bucket_size).cast(pl.Int32).alias("_bucket")
    )
    n_buckets_closed = trades_sorted["_bucket"].n_unique()
    if n_buckets_closed < VPIN_MIN_PERIODS:
        print(f"  [VPIN WARNING] Only {n_buckets_closed} buckets closed this session "
              f"(need {VPIN_MIN_PERIODS}+) — VPIN will be null/sparse for this session.")

    vpin_buckets = (
        trades_sorted.group_by("_bucket", maintain_order=True)
        .agg([
            pl.col("buy_vol").sum(), pl.col("sell_vol").sum(), pl.col("amount").sum(),
            pl.col("datetime").last().alias("bucket_end_time"),
        ])
        .sort("_bucket")
    )
    vpin_buckets = vpin_buckets.with_columns(
        ((pl.col("buy_vol") - pl.col("sell_vol")).abs() / (pl.col("amount") + EPS)).alias("vpin_raw")
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

def load_and_build_state_space(orderbook_path: str, trades_path: str, session_id: int, start_dt=None, end_dt=None) -> pl.DataFrame:
    # 1. Use LazyFrames
    df_orderbook = pl.scan_parquet(orderbook_path) if str(orderbook_path).endswith(".parquet") else pl.scan_csv(orderbook_path)
    df_trades = pl.scan_parquet(trades_path) if str(trades_path).endswith(".parquet") else pl.scan_csv(trades_path)

    # 2. DOWNCAST TO 32-BIT (EXCLUDING TIMESTAMP)
    # We subtract the timestamp column so it safely remains a 64-bit integer
    df_orderbook = df_orderbook.cast({cs.float(): pl.Float32, cs.integer() - cs.by_name("timestamp"): pl.Int32})
    df_trades = df_trades.cast({cs.float(): pl.Float32, cs.integer() - cs.by_name("timestamp"): pl.Int32})

    # 3. Convert timestamps to datetime (reverted back to Int64)
    df_orderbook = df_orderbook.with_columns(
        pl.from_epoch((pl.col("timestamp") // 1000).cast(pl.Int64), time_unit="s").alias("datetime")
    )
    df_trades = df_trades.with_columns(
        pl.from_epoch((pl.col("timestamp") // 1000).cast(pl.Int64), time_unit="s").alias("datetime")
    )

    # 4. APPLY THE TIME FILTER BEFORE LOADING INTO RAM
    if start_dt and end_dt:
        buffer_start = start_dt - timedelta(hours=2)
        df_orderbook = df_orderbook.filter((pl.col("datetime") >= buffer_start) & (pl.col("datetime") < end_dt))
        df_trades = df_trades.filter((pl.col("datetime") >= buffer_start) & (pl.col("datetime") < end_dt))

    df_orderbook = df_orderbook.sort("datetime")
    df_trades = df_trades.sort("datetime")
    
    df_orderbook = df_orderbook.with_columns(((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"))

    # 5. Pull only this chunk into memory safely using the new Polars syntax
    df_orderbook = df_orderbook.collect(engine="streaming")
    df_trades = df_trades.collect(engine="streaming")

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

    feats = compute_orderflow_features(df_trades, df_orderbook)

    # 6. AGGRESSIVE GARBAGE COLLECTION (CRITICAL FOR YOUR 16GB RAM)
    # We have our features, so delete the raw massive dataframes before numpy operations start
    del df_orderbook
    del df_trades
    gc.collect()

    df_price_1m = df_ob_1m.select(["datetime", "open", "high", "low", "close", "spread_pct"]).join(
        df_trades_1m.select(["datetime", "volume"]), on="datetime", how="left"
    )

    price_clean = df_price_1m["close"].to_numpy().astype(float)
    diffs = np.diff(price_clean)

    def _pad_front(arr, target_len):
        out = np.full(target_len, np.nan)
        out[-len(arr):] = arr
        return out

    rsi_2 = _pad_front(get_wilders_rsi(diffs, RSI_2_WINDOW), len(df_price_1m))
    rsi_14_raw = get_wilders_rsi(diffs, RSI_14_WINDOW)
    rsi_14 = _pad_front(rsi_14_raw, len(df_price_1m))
    sma_14 = _pad_front(sma(rsi_14_raw, SMA_14_WINDOW), len(df_price_1m))

    # nan_to_null=True
    df_price_1m = df_price_1m.with_columns([
        pl.Series("rsi_2", rsi_2, nan_to_null=True),
        pl.Series("rsi_14", rsi_14, nan_to_null=True),
        pl.Series("sma_14", sma_14, nan_to_null=True),
    ])

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

<<<<<<< HEAD
    return pl.concat(parts)
=======
        g = g[g['_valid']].drop(columns=['_valid'])
        g['target'] = g['target'].astype('int32')
        labeled.append(g)

    return pd.concat(labeled)
>>>>>>> origin/main


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

<<<<<<< HEAD
        possible_train_idx = np.arange(0, start_idx)
        train_times = datetimes[possible_train_idx]
=======
    ob['datetime'] = pd.to_datetime(ob['timestamp'].astype('int32') // 1000, unit='s')
    ob = ob.set_index('datetime').sort_index()
>>>>>>> origin/main

        purge_mask = (train_times + horizon + embargo >= test_start_time)
        train_idx = possible_train_idx[~purge_mask]

        n_purged = len(possible_train_idx) - len(train_idx)
        yield train_idx, test_idx, n_purged

<<<<<<< HEAD

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
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Finding time boundaries of the dataset...")
    
    trades_lazy = pl.scan_parquet("./research/csv/BTCUSDT_trades.parquet")
    time_bounds = trades_lazy.select([
        pl.from_epoch((pl.col("timestamp").min() // 1000).cast(pl.Int64), time_unit="s").alias("start"),
        pl.from_epoch((pl.col("timestamp").max() // 1000).cast(pl.Int64), time_unit="s").alias("end")
    ]).collect()
    
    global_start = time_bounds["start"][0]
    global_end = time_bounds["end"][0]
    
    N_CHUNKS = 6
    chunk_duration = (global_end - global_start) / N_CHUNKS
    
    print(f"Total time range: {global_start} to {global_end}")
    print(f"Splitting into {N_CHUNKS} chunks...")
    
    for i in range(N_CHUNKS):
        chunk_start = global_start + (chunk_duration * i)
        chunk_end = global_start + (chunk_duration * (i + 1))
        
        if i == N_CHUNKS - 1:
            chunk_end = chunk_end + timedelta(minutes=1)
            
        print(f"\n--- Processing Chunk {i+1}/{N_CHUNKS} ---")
        print(f"Timeframe: {chunk_start} to {chunk_end}")
        
        chunk_df = load_and_build_state_space(
            "./research/csv/BTCUSDT_orderbook.parquet", 
            "./research/csv/BTCUSDT_trades.parquet", 
            session_id=0,
            start_dt=chunk_start,
            end_dt=chunk_end
        )
        
        # FIX: Save each chunk to its own file instead of breaking a single file
        chunk_filename = f"state_space_chunk_{i}.parquet"
        chunk_df.write_parquet(chunk_filename)
                
        print(f"Chunk {i+1} saved to {chunk_filename}. Wiping memory...")
        
        del chunk_df
        gc.collect()
        
    print("\nAll chunks processed successfully! Loading final compiled dataset...")
    
    # FIX: Polars will scan the folder and stitch all chunks together seamlessly
    df_state_space = pl.scan_parquet("state_space_chunk_*.parquet").collect()
    print(f"Combined state space: {df_state_space.height} rows")
    
    # ... [PROCEED WITH YOUR BARRIER LABELING AND LIGHTGBM CODE] ...

    # ------------------------------------------------------------------
=======
    trades['cvd_tick'] = trades['buy_vol'] - trades['sell_vol']
    trades['datetime'] = pd.to_datetime(trades['timestamp'].astype('int32') // 1000, unit='s')
    trades = trades.set_index('datetime').sort_index()

    cvd_1m = trades['cvd_tick'].resample(BAR_FREQ).sum().cumsum()
    cvd_delta_1m = cvd_1m.diff().fillna(0)
    cvd_slope_5 = cvd_1m.diff(5).fillna(0)

    # ================================================================
    # 3. MICROSTRUCTURE: Tick Rule & Trade Sign Autocorrelation
    # ================================================================
    trades['price_prev'] = trades['price'].shift(1)
    trades['tick_rule'] = np.where(trades['price'] > trades['price_prev'], 1,
                          np.where(trades['price'] < trades['price_prev'], -1, 0))

    resampled_parts = []
    for _, day_df in trades.groupby(trades.index.floor('1D')):
        resampled_parts.append(day_df.resample(BAR_FREQ).agg({
            'tick_rule': 'mean',
            'amount': ['sum', 'count'],
            'price': 'std',
        }))
    micro = pd.concat(resampled_parts).sort_index()
    
    micro.columns = ['tick_rule', 'volume', 'n_trades', 'price_volatility']
    micro['trade_intensity'] = micro['n_trades'] / 60.0
    micro['avg_trade_size'] = micro['volume'] / (micro['n_trades'] + EPS)

    # Trade sign autocorrelation: rolling autocorrelation of tick rule
    # High autocorrelation = hidden institutional order splitting
    tick_1s = trades['tick_rule'].resample('1s').last().fillna(0)
    tick_1m = tick_1s.resample(BAR_FREQ).apply(
        lambda x: x.autocorr() if len(x) > 5 and not np.isnan(x.autocorr()) else 0
    )

    # ================================================================
    # 4. VPIN
    # ================================================================
    # FIX: bucket size is now derived from this session's actual total traded
    # volume / VPIN_TARGET_BUCKETS_PER_SESSION, instead of a hardcoded constant
    # that assumed a specific liquidity regime. This guarantees ~50 buckets close
    # per session regardless of how much volume the CSV actually contains.
    trades_sorted = trades.sort_index()
    total_session_volume = trades_sorted['amount'].sum()
    vpin_bucket_size = max(total_session_volume / VPIN_TARGET_BUCKETS_PER_SESSION, EPS)

    cum_vol = trades_sorted['amount'].cumsum()
    bucket_labels = (cum_vol / vpin_bucket_size).astype("int32")

    vpin_buckets = trades_sorted.groupby(bucket_labels).agg({
        'buy_vol': 'sum',
        'sell_vol': 'sum',
        'amount': 'sum'
    })

    vpin_buckets['vpin_raw'] = np.abs(vpin_buckets['buy_vol'] - vpin_buckets['sell_vol']) / (vpin_buckets['amount'] + EPS)
    vpin_buckets['vpin_smoothed'] = vpin_buckets['vpin_raw'].rolling(VPIN_WINDOW_BUCKETS, min_periods=VPIN_MIN_PERIODS).mean()

    n_buckets_closed = bucket_labels.nunique()
    if n_buckets_closed < VPIN_MIN_PERIODS:
        print(f"  [VPIN WARNING] Only {n_buckets_closed} buckets closed this session "
              f"(need {VPIN_MIN_PERIODS}+) — VPIN will be NaN/sparse for this session.")

    bucket_end_times = trades_sorted.groupby(bucket_labels).apply(lambda g: g.index[-1])
    if isinstance(bucket_end_times, pd.Series):
        bucket_end_times = bucket_end_times.reset_index(drop=True)

    vpin_aligned = pd.Series(vpin_buckets['vpin_smoothed'].values, index=bucket_end_times)
    vpin_1m = vpin_aligned.resample(BAR_FREQ).last().ffill()

    # ================================================================
    # 5. RETAIL vs INSTITUTIONAL DELTA
    # ================================================================
    trades['is_retail'] = trades['amount'] < RETAIL_THRESHOLD
    trades['is_institutional'] = trades['amount'] >= RETAIL_THRESHOLD

    trades['retail_buy'] = trades['buy_vol'].where(trades['is_retail'], 0)
    trades['retail_sell'] = trades['sell_vol'].where(trades['is_retail'], 0)
    trades['inst_buy'] = trades['buy_vol'].where(trades['is_institutional'], 0)
    trades['inst_sell'] = trades['sell_vol'].where(trades['is_institutional'], 0)

    trade_size = trades.resample(BAR_FREQ).agg({
        'retail_buy': 'sum',
        'retail_sell': 'sum',
        'inst_buy': 'sum',
        'inst_sell': 'sum',
        'amount': ['sum', 'count'],
    })
    trade_size.columns = ['retail_buy', 'retail_sell', 'inst_buy', 'inst_sell', 'volume', 'n_trades']

    trade_size['retail_delta'] = trade_size['retail_buy'] - trade_size['retail_sell']
    trade_size['inst_delta'] = trade_size['inst_buy'] - trade_size['inst_sell']
    trade_size['delta_divergence'] = trade_size['inst_delta'] - trade_size['retail_delta']
    trade_size['inst_ratio'] = (trade_size['inst_buy'] + trade_size['inst_sell']) / (trade_size['volume'] + EPS)

    # ================================================================
    # 6. ABSORPTION RATIO
    # ================================================================
    ob_1m_price = ob[['bid_price', 'ask_price']].resample(BAR_FREQ).agg({
        'bid_price': ['min', 'max'],
        'ask_price': ['min', 'max']
    })
    ob_1m_price.columns = ['bid_low', 'bid_high', 'ask_low', 'ask_high']
    ob_1m_price['price_range'] = (ob_1m_price['ask_high'] - ob_1m_price['bid_low']).ffill()

    abs_ratio_raw = cvd_delta_1m / (ob_1m_price['price_range'] + EPS)

    # ================================================================
    # 7. DeepLOB: Order Book Depth Imbalance
    # ================================================================
    # Simulate depth features from top-of-book dynamics
    # Depth imbalance ratio: (bid_amount - ask_amount) / (bid_amount + ask_amount)
    ob['depth_imbalance'] = (ob['bid_amount'] - ob['ask_amount']) / (ob['bid_amount'] + ob['ask_amount'] + EPS)

    # Rate of change of depth imbalance (liquidity absorption speed)
    ob['depth_imbalance_roc'] = ob['depth_imbalance'].diff().fillna(0)

    # Bid-ask spread pressure: how fast the spread is changing
    ob['mid'] = (ob['bid_price'] + ob['ask_price']) / 2
    ob['spread_bps'] = (ob['ask_price'] - ob['bid_price']) / ob['mid'] * 10000

    depth_1m = ob[['depth_imbalance', 'depth_imbalance_roc', 'spread_bps']].resample(BAR_FREQ).last()

    return (ofi_1m, ofi_diff, cvd_delta_1m, cvd_slope_5, micro,
            vpin_1m, trade_size, abs_ratio_raw, tick_1m, depth_1m)


# ============================================================================
# STEP 1b — Per-session feature pipeline
# ============================================================================

def _read_table(path):
    return pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)

def load_and_build_state_space(orderbook_path, trades_path, session_id):
    df_orderbook = _read_table(orderbook_path)
    df_trades = _read_table(trades_path)

    df_orderbook['datetime'] = pd.to_datetime(df_orderbook['timestamp'].astype('int32') // 1000, unit='s')
    df_trades['datetime'] = pd.to_datetime(df_trades['timestamp'].astype('int32') // 1000, unit='s')

    df_orderbook = df_orderbook.set_index('datetime').sort_index()
    df_trades = df_trades.set_index('datetime').sort_index()

    df_orderbook['mid_price'] = (df_orderbook['bid_price'] + df_orderbook['ask_price']) / 2

    df_ob_1m = df_orderbook.resample(BAR_FREQ).agg({
        'mid_price': ['first', 'max', 'min', 'last'],
        'bid_price': 'last',
        'ask_price': 'last',
        'bid_amount': 'sum',
        'ask_amount': 'sum',
    })
    df_ob_1m.columns = ['open', 'high', 'low', 'close', 'bid_price', 'ask_price', 'bid_amount', 'ask_amount']
    df_ob_1m['spread'] = df_ob_1m['ask_price'] - df_ob_1m['bid_price']
    df_ob_1m['spread_pct'] = df_ob_1m['spread'] / df_ob_1m['close']

    df_trades['buy_vol'] = df_trades['amount'].where(df_trades['side'] == 'buy', 0)
    df_trades['sell_vol'] = df_trades['amount'].where(df_trades['side'] == 'sell', 0)

    df_micro = df_trades.resample('1s').agg({
        'price': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum'
    }).ffill()

    df_micro['micro_OFI'] = df_micro['buy_vol'] - df_micro['sell_vol']
    df_micro['micro_CVD'] = df_micro['micro_OFI'].rolling(10).sum()

    df_trades_1m = df_trades.resample(BAR_FREQ).agg({
        'price': ['first', 'max', 'min', 'last', 'count'],
        'amount': 'sum',
        'buy_vol': ['sum', 'max'],
        'sell_vol': ['sum', 'max']
    })
    df_trades_1m.columns = ['open', 'high', 'low', 'close', 'num_trades', 'volume',
                             'agg_buyer', 'largest_buyer', 'agg_seller', 'largest_seller']
    df_trades_1m = df_trades_1m.dropna(subset=['close'])

    # ===== COMPUTE ORDERFLOW FEATURES =====
    (ofi_1m, ofi_diff, cvd_delta_1m, cvd_slope_5, micro,
     vpin_1m, trade_size, abs_ratio_raw, tick_autocorr_1m,
     depth_1m) = compute_orderflow_features(df_trades, df_orderbook)

    # ===== BUILD PRICE DATAFRAME =====
    df_price_1m = pd.DataFrame({
        'open': df_ob_1m['open'],
        'high': df_ob_1m['high'],
        'low': df_ob_1m['low'],
        'close': df_ob_1m['close'],
        'volume': df_trades_1m['volume'],
        'spread_pct': df_ob_1m['spread_pct'],
    })

    # ================================================================
    # PRICE-BASED FEATURES
    # ================================================================
    price_clean = np.asarray(df_price_1m['close'], dtype=float)
    diffs = np.diff(price_clean)

    def _pad_front(arr, target_len):
        out = np.full(target_len, np.nan)
        out[-len(arr):] = arr
        return out

    # RSI features — windows now come from config, not re-hardcoded literals
    df_price_1m['rsi_2'] = _pad_front(get_wilders_rsi(diffs, RSI_2_WINDOW), len(df_price_1m))
    rsi_14_raw = get_wilders_rsi(diffs, RSI_14_WINDOW)
    df_price_1m['rsi_14'] = _pad_front(rsi_14_raw, len(df_price_1m))
    df_price_1m['sma_14'] = _pad_front(sma(rsi_14_raw, SMA_14_WINDOW), len(df_price_1m))

    # MA Slope (velocity) - strictly backward looking
    ma_5 = df_price_1m['close'].rolling(window=MA_SLOPE_WINDOW).mean()
    df_price_1m['ma_slope_5'] = ma_5.diff().fillna(0)

    # MA acceleration (2nd derivative)
    df_price_1m['ma_accel'] = df_price_1m['ma_slope_5'].diff().fillna(0)
    df_price_1m['ma_accel_regime'] = np.where(df_price_1m['ma_accel'] > 0, 1,
                                      np.where(df_price_1m['ma_accel'] < 0, -1, 0))

    # Price rate of change
    df_price_1m['price_roc_3'] = df_price_1m['close'].pct_change(PRICE_ROC_WINDOW_3)

    # Aggregated-bar volatility for barrier sizing — matches
    # LiveFeatureEngine._compute_macro_features()'s _bar_vol_std exactly:
    # std of bar-to-bar % returns on the SAME TIMEFRAME_BARRIER close series
    # RSI(2) is built from, over a trailing BARRIER_VOL_BAR_WINDOW-bar window.
    # This is NOT a model feature (not in FEATURE_COLUMNS) — it only feeds
    # add_barrier_labels(), matching PositionTracker.get_barrier_pct().
    # FIX: ddof=0 to match live EXACTLY.
    # Live features.py _compute_macro_features() computes:
    #     bar_rets = np.diff(bar_closes) / (bar_closes[:-1] + EPS)
    #     bar_vol_std = np.nanstd(bar_rets[-window:])   # np.nanstd defaults ddof=0
    # and live execution.py get_barrier_pct() does:
    #     barrier = bar_vol_std * BARRIER_MULTIPLIER    # NO sqrt(horizon) scaling
    # pandas .rolling().std() defaults to ddof=1 (sample std), which is a
    # ~2.6% constant bias vs np.nanstd's ddof=0 at window=20 — making research
    # barriers slightly wider than live. .std(ddof=0) removes that bias.
    bar_rets = df_price_1m['close'].pct_change()
    df_price_1m['bar_vol_std'] = bar_rets.rolling(
        window=BARRIER_VOL_BAR_WINDOW, min_periods=2
    ).std(ddof=0)
    df_price_1m['price_roc_5'] = df_price_1m['close'].pct_change(5)

    # RSI rate of change — speed of entry into oversold
    df_price_1m['rsi_roc_3'] = df_price_1m['rsi_2'].diff(3)
    df_price_1m['rsi_roc_5'] = df_price_1m['rsi_2'].diff(5)

    # ================================================================
    # MINIMAL FEATURE SET (only proven high-PFI features)
    # ================================================================
    # From v3 analysis, the top features by PFI are:
    # spread_bps_z(0.081), ma_slope_5(0.024), rsi_x_ma(0.006), 
    # ma_accel(0.005), price_roc_3(0.005), depth_imbalance_z(0.002)
    
    # DeepLOB: spread in basis points (PFI=0.081 — dominant feature)
    df_price_1m['spread_bps_z'] = calculate_zscore(
        depth_1m['spread_bps'].reindex(df_ob_1m.index).fillna(0), window=DEEPLOB_Z_WINDOW)
    
    # DeepLOB: depth imbalance (PFI=0.002)
    df_price_1m['depth_imbalance_z'] = calculate_zscore(
        depth_1m['depth_imbalance'].reindex(df_ob_1m.index).fillna(0), window=10)

    # RSI × MA slope interaction (PFI=0.006 — key signal)
    rsi_z = calculate_zscore(df_price_1m['rsi_2'], window=RSI_Z_WINDOW)
    df_price_1m['rsi_x_ma'] = rsi_z * df_price_1m['ma_slope_5']

    # ================================================================
    # FIX: PREVIOUSLY-DEAD ORDERFLOW FEATURES
    # ================================================================
    # ofi_1m, ofi_diff, cvd_delta_1m, cvd_slope_5, vpin_1m, trade_size, and
    # abs_ratio_raw were all computed by compute_orderflow_features() above but
    # were never joined into df_price_1m — they were silently discarded, so the
    # model never actually saw them despite the docstring/feature-importance
    # comments describing them as inputs. Wiring them in here for real, using
    # the same reindex-onto-df_ob_1m.index pattern already used for depth_1m.
    # Their actual predictive value (PFI/SHAP) will show up honestly in Step 7
    # on the next run instead of being assumed away.

    # OFI: structural order flow imbalance (change in imbalance per minute)
    df_price_1m['ofi_z'] = calculate_zscore(
        ofi_diff.reindex(df_ob_1m.index).fillna(0), window=10)

    # CVD: cumulative volume delta, level change and short-window slope
    df_price_1m['cvd_delta_z'] = calculate_zscore(
        cvd_delta_1m.reindex(df_ob_1m.index).fillna(0), window=10)
    df_price_1m['cvd_slope_z'] = calculate_zscore(
        cvd_slope_5.reindex(df_ob_1m.index).fillna(0), window=10)

    # VPIN: informed-trading probability. Forward-filled (not zero-filled) on
    # reindex, since VPIN is a volume-clock signal that should persist its last
    # known estimate between bucket closes rather than snap to 0.
    df_price_1m['vpin'] = vpin_1m.reindex(df_ob_1m.index).ffill().fillna(0)

    # Absorption ratio: CVD delta normalized by price range (institutional absorption)
    df_price_1m['absorption_z'] = calculate_zscore(
        abs_ratio_raw.reindex(df_ob_1m.index).fillna(0), window=10)

    # Trade sign autocorrelation: hidden institutional order-splitting signature
    df_price_1m['tick_autocorr'] = tick_autocorr_1m.reindex(df_ob_1m.index).fillna(0)

    # Retail vs institutional flow: ratio and divergence between size cohorts
    df_price_1m['inst_ratio'] = trade_size['inst_ratio'].reindex(df_ob_1m.index).fillna(0)
    df_price_1m['delta_divergence_z'] = calculate_zscore(
        trade_size['delta_divergence'].reindex(df_ob_1m.index).fillna(0), window=10)

    # ================================================================
    # JOIN MICRO (1s) WITH MACRO (1m)
    # ================================================================
    df_micro_shifted = df_micro.shift(1)
    df_macro_shifted = df_price_1m.shift(1)

    df_state_space = df_micro_shifted.join(df_macro_shifted, how='left').ffill()
    df_state_space['execution_price'] = df_micro['price']

    df_state_space = df_state_space.dropna()
    df_state_space['session_id'] = session_id

    return df_state_space


# ============================================================================
# STEP 2 — Build combined state space
# ============================================================================
if __name__ == "__main__":
    """
    print("Loading sessions...")
    session_0 = load_and_build_state_space(
        "./research/csv/BTCUSDT_orderbook.csv", "./research/csv/BTCUSDT_trades.csv", session_id=0
    )
    print(f"Session 0: {len(session_0)} rows")
    session_1 = load_and_build_state_space(
        "./research/csv/BTCUSDT_orderbook_2.csv", "./research/csv/BTCUSDT_trades_2.csv", session_id=1
    )
    print(f"Session 1: {len(session_1)} rows")
    session_2 = load_and_build_state_space(
        "./research/csv/BTCUSDT_orderbook_3.csv", "./research/csv/BTCUSDT_trades_3.csv", session_id=2
    )
    print(f"Session 2: {len(session_2)} rows")

    #df_state_space = load_and_build_state_space("./research/csv/XAUTUSDT_orderbook.csv", "./research/csv/XAUTUSDT_trades.csv", 0)
    df_state_space = pd.concat([session_0, session_1, session_2]).sort_index()
    print(f"Combined state space: {len(df_state_space)} rows")
    """
    print("Loading sessions...")
    session_0 = load_and_build_state_space(
        "./research/csv/BTCUSDT_orderbook.parquet", "./research/csv/BTCUSDT_trades.parquet", session_id=0
    )
    df_state_space = session_0
    print(f"Combined state space: {len(df_state_space)} rows")

    # ============================================================================
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
    # FIX: gated by USE_ORDERFLOW_FEATURES now instead of being unconditional.
    # This was the source of an undetected research/live mismatch: this block
    # always pulled in all 9 orderflow columns whenever they existed in
    # df_training_zone, so every walk-forward chart included features that
    # HistoricalDataBuilder (executor.py) never computes or serves live. The
    # chart was evaluating a strictly bigger, different model than the one
    # actually deployed. Toggle explicitly; the export step (STEP 12) records
    # which set was used in the feature manifest so live can check it matches.
>>>>>>> origin/main
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

<<<<<<< HEAD
=======

    def purged_chronological_splits(index, n_samples, block_size, horizon_seconds,
                                     embargo_seconds=0, allow_overlap=False):
        n = len(index)
        horizon = pd.Timedelta(seconds=horizon_seconds)
        embargo = pd.Timedelta(seconds=embargo_seconds)
        min_initial_train = int(n * 0.35)
        available_test_run = n - min_initial_train - block_size

        if available_test_run <= 0:
            raise ValueError("The dataset length is too narrow for this test configuration block size.")

        # FIX: this used to just warn and proceed when step_intervals < block_size,
        # which silently let consecutive folds re-test the same historical period
        # with different models — the same real trade opportunity scored (and
        # logged) more than once, inflating apparent trade counts and breaking the
        # independence assumption behind every win-rate/expectancy/CVaR stat below.
        # Now n_samples is capped to the max number of BACK-TO-BACK, non-overlapping
        # block_size windows the data actually supports, so every fold-by-default
        # tests a genuinely independent period. allow_overlap=True restores the old
        # behavior if you deliberately want denser (but non-independent) folds.
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
            test_start_time = index[test_idx[0]]

            possible_train_idx = np.arange(0, start_idx)
            train_times = index[possible_train_idx]

            purge_mask = (train_times + horizon + embargo >= test_start_time)
            train_idx = possible_train_idx[~purge_mask]

            n_purged = len(possible_train_idx) - len(train_idx)
            yield train_idx, test_idx, n_purged


>>>>>>> origin/main
    splitter = purged_chronological_splits(
        datetimes, n_samples=N_RANDOM_SAMPLES, block_size=TEST_BLOCK_SIZE,
        horizon_seconds=LOOKAHEAD_SECONDS, embargo_seconds=EMBARGO_SECONDS
    )

    print(f"\n--- Starting Walk-Forward (requested up to {N_RANDOM_SAMPLES} non-overlapping folds; "
          f"see [FOLD COUNT ADJUSTED] above if fewer independent windows fit the data) "
          f"| Targeting: {target_name} | Inverting Trades: {INVERT_RETURNS} ---")
<<<<<<< HEAD
=======

    # FIX: HYSTERESIS BUFFER for signal detection
    # The original edge-trigger (np.diff(preds, prepend=0) == 1) fires a "fresh"
    # signal every time y_proba crosses from below t to >= t. If y_proba is noisy
    # right around a given threshold (e.g. the model's output clusters near 0.85),
    # it can cross back and forth many times within what is really one continuous
    # elevated-probability period, and each crossing gets counted as a separate
    # trade. This produced the non-monotonic trade-count spike (58 trades at >=85%
    # vs. 27 at >=80% and 6 at >=90%) — a strong signature of flicker, not signal.
    # HYSTERESIS_BUFFER requires proba to drop meaningfully below the threshold
    # before a new signal is allowed to re-arm, collapsing flicker into one trade.
    # NOTE: 0.08 is a starting point, not a proven-correct constant — tested on
    # synthetic noise (std=0.02) where 0.03 only cut flicker by ~79% (66->14
    # spurious signals) while 0.08 fully collapsed it to the true count. Your
    # model's actual y_proba noise level may differ. Before trusting this on your
    # real data, plot a histogram of y_proba for the fold(s) that fed the >=85%
    # threshold and check its local volatility against this buffer.
    # HYSTERESIS_BUFFER is now imported from main.config (see import block above)
    # so it stays in sync with executor.py. Do NOT re-define it here.

    def find_debounced_signals(y_proba, threshold, hysteresis=HYSTERESIS_BUFFER):
        """Return indices of fresh threshold crossings, ignoring flicker.
        Once armed and proba >= threshold, a signal fires and the detector disarms.
        It only re-arms once proba drops below (threshold - hysteresis)."""
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
>>>>>>> origin/main

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

<<<<<<< HEAD
    # ------------------------------------------------------------------
    # STEP 12 — Export Production Model
    # ------------------------------------------------------------------
=======
    # ============================================================================
    # STEP 12 — Export Production Model
    # ============================================================================
    # Everything above (walk-forward, SHAP/PFI, threshold sweep, PnL chart) is
    # EVALUATION ONLY — none of the 15 fold-models get saved anywhere. This step
    # does one FINAL fit on all available hunting-zone data and writes it to
    # MODEL_PATH in the exact format executor.py's ModelManager.load() expects
    # (raw lgb.Booster), plus a <model>.features.json manifest.
    #
    # WHY THIS MATTERS: HistoricalDataBuilder in executor.py trains its own
    # bootstrap model from REST OHLCV bars, and can only APPROXIMATE spread_bps_z
    # (the single most predictive feature — PFI 0.081, ~3x the next feature) from
    # candle high-low range, because OHLCV bars structurally do not contain
    # bid/ask data. No formula tweak fixes that — real spread requires real
    # orderbook ticks, which only this research pipeline (load_and_build_state_space)
    # has access to. This export step is the fix: train on the real tick data used
    # here, ship *that* model, and stop training the deployed model on a proxy.
    #
    # The manifest exists because ModelManager.predict_proba() calls the raw
    # Booster with a plain numpy array (no column names — see the
    # "X does not have valid feature names" warning suppression in executor.py).
    # A feature-order mismatch between what this model was trained on and what
    # LiveFeatureEngine.extract_feature_vector() sends would NOT raise an error —
    # it would just silently feed the model garbage. The manifest lets executor.py
    # check this explicitly instead of assuming it's correct.
>>>>>>> origin/main
    if EXPORT_PRODUCTION_MODEL:
        print("\n" + "=" * 70)
        print("EXPORTING PRODUCTION MODEL")
        print("=" * 70)

        final_clf = lgb.LGBMClassifier(
<<<<<<< HEAD
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
=======
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
>>>>>>> origin/main
        )
        final_clf.fit(X, y)

        model_dir = os.path.dirname(os.path.abspath(MODEL_PATH))
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        final_clf.booster_.save_model(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

        manifest_path = os.path.splitext(MODEL_PATH)[0] + ".features.json"
<<<<<<< HEAD
        label_counts_export = {CLASS_NAMES[k]: int((y == k).sum()) for k in np.unique(y)}
        manifest = {
            "features": features,
=======
        manifest = {
            "features": features,  # EXACT order — must match extract_feature_vector() output order
>>>>>>> origin/main
            "n_features": len(features),
            "includes_orderflow": USE_ORDERFLOW_FEATURES,
            "orderflow_features": [c for c in ORDERFLOW_FEATURE_COLUMNS if c in features],
            "base_features": list(FEATURE_COLUMNS),
            "trained_rows": int(len(X)),
<<<<<<< HEAD
            "trained_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "label_counts": label_counts_export,
=======
            "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
            "label_counts": {CLASS_NAMES[k]: int(v) for k, v in y.value_counts().to_dict().items()},
>>>>>>> origin/main
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