
"""
============================================================================
STEP 1A — ORDERFLOW AND PRICE FEATURE COMPUTATION

Extracts two tiers of orderflow features:
- Micro (1s resampling): Captures granular orderflow dynamics for live execution, 
  allowing the ML model to optimize entries mid-candle rather than waiting for candle close.
- Macro: Serves as a volatility/trend stabilizer to offset the noise of micro features.

Data Processed Into Feature Builder
    df_trades: timestamp,price,amount,side,id
    df_orderbook: timestamp,bid_price,bid_amount,ask_price,ask_amount

Critical Notes:
    - VPIN Bug: VPIN historically suffered from a continuous stream reset when chunked, 
      which degraded trade expectancy. This version preserves global session context.
      It is very important that we should avoid using recursive variables because of
      chunking issue.
    - Architecture: compute_orderflow_features will be refactored into a dedicated module 
      to enforce strict isolation and guarantee zero lookahead bias or backtest leaks.
    - No need to shift(1) because it's already shifted(1) in ml_state_builder. However don't
    use shift(-1) or rolling with center=True.

Prompt:

You are a quant researcher. You can only read from selective sources of the following. Financial reseearch papers,
books, etc. Compute  features without breaking quant practice (all your features must have economic rationale).
If the features fail, logically find out why and make fixes. This is what you can do at root directory.

python -m research.ml_state_optimized --> to run the strategy and output report.
build/backtest_report_v3.md --> You read this to get the output of the result

Objective: Inside of build/backtest_report_v3.md, you must return such that the average trade expectancy > 0.2,
optimize for sortino ratio and edge stability at all cost.

Repeat this until you fufill the objectives.

Rules:
- YOU MAY NOT EDIT ANY OTHER FILE. STRICTLY EDIT ONLY IN FEATURE BUILDER OTHERWISE YOU NEED TO STOP AND ASK PERMISSION.
- DO NOT SHIFT(-1) or ROLL WITH CENTER=TRUE BECAUSE THAT WOULD CAUSE LOOKAHEAD BIAS.
- DO NOT FORCE OVERFIT. Follow quant practices
============================================================================
"""

import sys

import os
import json
import datetime as _dt
import gc
from dataclasses import dataclass
import tempfile

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

os.environ["POLARS_MAX_THREADS"] = "4"

from research.ml_shared_config import *

"""
========================================================================
CUSTOM FUNCTIONS: Place custom functions here for computation
========================================================================
"""


def _pl_rolling_zscore(col: str, window: int) -> pl.Expr:
    """Rolling z-score (sample std, ddof=1, NaN->0) — see module docstring
    point 6. This is a reimplementation, not your real calculate_zscore()."""
    mean = pl.col(col).rolling_mean(window_size=window, min_samples=window)
    std = pl.col(col).rolling_std(window_size=window, min_samples=window, ddof=1)
    
    return ((pl.col(col) - mean) / (std + EPS)).fill_null(0).fill_nan(0)


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


"""
========================================================================
COMPUTE ORDERFLOW FEATURES AND PRICE FEATURES
========================================================================
"""
def compute_orderflow_features(df_trades: pl.DataFrame, df_orderbook: pl.DataFrame, dense_resample,
                                total_session_volume: float | None = None) -> dict:
    ob = df_orderbook.sort(["datetime", "timestamp"], maintain_order=True)
    ob = ob.with_columns([
        pl.col("bid_price").alias("bid_price_prev"),
        pl.col("ask_price").alias("ask_price_prev"),
        pl.col("bid_amount").alias("bid_amount_prev"),
        pl.col("ask_amount").alias("ask_amount_prev"),
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
    del ofi_1s  # 1s intermediate, not returned — free before it compounds

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
    del tick_1s  # 1s intermediate, not returned — free before it compounds

    trades_sorted = trades.sort(["datetime", "timestamp"], maintain_order=True)
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
    del trades_sorted, vpin_buckets, vpin_by_bucket_close  # not returned — free VPIN workspace

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

    # Memory hygiene: the two giant raw event frames and the now-unused bucket
    # workspace are freed right before returning. None of them are part of the
    # returned dict, so this is strictly lossless — outputs are bit-identical.
    del ob, trades
    gc.collect()

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

def compute_price_features(price_and_volume: pl.DataFrame, rsi_features: pl.DataFrame) -> pl.DataFrame:
    """Derive every price-only feature (no orderflow, no cross-source join)
    from a bar-level OHLC + volume frame.

    Args:
        price_and_volume: one row per BAR_FREQ bar with at least 'datetime',
            'close', and 'volume' (see _build_price_bars / _build_bar_volume).
            Extra columns (open, high, low, spread_pct) pass through untouched.
        rsi_features: session-scoped rsi_2 / rsi_14 / sma_14, computed once
            over the whole session's close series (recursive — cannot be
            safely computed per-chunk, see _rsi_features_from_price_bars).
            Left-joined onto price_and_volume by datetime.

    Returns:
        price_and_volume with rsi_2/rsi_14/sma_14 plus every feature below
        added. Every rolling/diff feature here is causal (uses only bars at
        or before the current one) — but the shift that makes the WHOLE
        frame safe to broadcast onto live micro-bars happens later, in
        _broadcast_bar_features_to_micro. This function only guarantees each
        row is internally consistent with its own bar's close time.
    """
    df = price_and_volume.join(rsi_features, on="datetime", how="left")

    df = df.with_columns(pl.col("close").rolling_mean(window_size=MA_SLOPE_WINDOW).alias("_ma_5"))
    df = df.with_columns(pl.col("_ma_5").diff().fill_null(0).alias("ma_slope_5")).drop("_ma_5")
    df = df.with_columns(pl.col("ma_slope_5").diff().fill_null(0).alias("ma_accel"))
    df = df.with_columns(
        pl.when(pl.col("ma_accel") > 0).then(1).when(pl.col("ma_accel") < 0).then(-1).otherwise(0)
          .alias("ma_accel_regime")
    )
    # EWA(3)-smoothed acceleration ("third/momentum derivative" filter signal).
    # Exponential weighted average, span=3 (alpha = 2/(3+1) = 0.5), causal.
    # Point-in-time safe WITHOUT any shift here: ml_state_builder shift(1)-lags
    # the whole macro frame and then asof(backward)-broadcasts it, so any bar
    # only ever sees this value as it stood at the previous bar's close.
    df = df.with_columns(
        pl.col("ma_accel").ewm_mean(span=3, adjust=True).fill_null(0).alias("ma_accel_ewma3")
    )
    df = df.with_columns(pl.col("close").pct_change(PRICE_ROC_WINDOW_3).alias("price_roc_3"))

    bar_rets = df.select(pl.col("close").pct_change().alias("bar_ret"))["bar_ret"]
    df = df.with_columns(
        bar_rets.rolling_std(window_size=BARRIER_VOL_BAR_WINDOW, min_samples=2, ddof=0).alias("bar_vol_std")
    )
    df = df.with_columns(pl.col("close").pct_change(5).alias("price_roc_5"))
    df = df.with_columns([
        pl.col("rsi_2").diff(3).alias("rsi_roc_3"),
        pl.col("rsi_2").diff(5).alias("rsi_roc_5"),
    ])

    rsi_z = df.select(_pl_rolling_zscore("rsi_2", RSI_Z_WINDOW).alias("_rsi_z"))["_rsi_z"]
    df = df.with_columns((rsi_z * pl.col("ma_slope_5")).alias("rsi_x_ma"))

    return df


def _join_orderflow_features(df_price_1m: pl.DataFrame, feats: dict) -> pl.DataFrame:
    """Left-join every precomputed orderflow series (from
    compute_orderflow_features) onto the bar-level price frame, and z-score
    the ones that need it.

    Kept separate from compute_price_features because these columns come
    from a genuinely different data source (trades/orderbook orderflow, not
    price bars), and because most of them share the same
    join -> fill_null(0) -> rolling-zscore pattern, repeated six times — that
    repetition is easier to review as one pass than interleaved with the
    price-only derivations above.
    """
    df = df_price_1m.join(
        feats["depth_1m"].select(["datetime", "spread_bps", "depth_imbalance"]), on="datetime", how="left"
    ).with_columns([pl.col("spread_bps").fill_null(0), pl.col("depth_imbalance").fill_null(0)])
    df = df.with_columns([
        _pl_rolling_zscore("spread_bps", DEEPLOB_Z_WINDOW).alias("spread_bps_z"),
        _pl_rolling_zscore("depth_imbalance", 10).alias("depth_imbalance_z"),
    ]).drop(["spread_bps", "depth_imbalance"])

    df = df.join(feats["ofi_1m"], on="datetime", how="left").with_columns(pl.col("ofi_diff").fill_null(0))
    df = df.join(feats["cvd_1m"], on="datetime", how="left").with_columns([
        pl.col("cvd_delta_1m").fill_null(0), pl.col("cvd_slope_5").fill_null(0),
    ])
    df = df.with_columns([
        _pl_rolling_zscore("ofi_diff", 10).alias("ofi_z"),
        _pl_rolling_zscore("cvd_delta_1m", 10).alias("cvd_delta_z"),
        _pl_rolling_zscore("cvd_slope_5", 10).alias("cvd_slope_z"),
    ]).drop(["ofi_diff", "cvd_delta_1m", "cvd_slope_5"])

    df = df.join(feats["vpin_lookup"], on="datetime", how="left").with_columns(
        pl.col("vpin_smoothed").forward_fill().fill_null(0).alias("vpin")
    ).drop("vpin_smoothed")

    df = df.join(feats["abs_ratio"], on="datetime", how="left").with_columns(pl.col("abs_ratio_raw").fill_null(0))
    df = df.with_columns(_pl_rolling_zscore("abs_ratio_raw", 10).alias("absorption_z")).drop("abs_ratio_raw")

    df = df.join(feats["tick_autocorr"], on="datetime", how="left").with_columns(pl.col("tick_autocorr").fill_null(0))

    df = df.join(feats["trade_size"], on="datetime", how="left").with_columns([
        pl.col("inst_ratio").fill_null(0), pl.col("delta_divergence").fill_null(0),
    ])
    df = df.with_columns(_pl_rolling_zscore("delta_divergence", 10).alias("delta_divergence_z")).drop("delta_divergence")

    return df
