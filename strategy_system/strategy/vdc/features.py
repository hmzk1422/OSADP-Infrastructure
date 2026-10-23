from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from strategy_system.config.research import moe as cfg
from pathlib import Path

# Coexistence trick: make `...vdc.features.<name>` resolve to the
# files inside the `features/` folder (mirrors strategy/moe/features.py).
# Must run before the imports immediately below.

from strategy_system.strategy.vdc.feature.rsidprice import add_rsi_d2_trigger
from strategy_system.strategy.vdc.feature.orderflow import add_orderflow
from components.pipelines.event_merger.matched.reconstruct import (
    reconstruct,
    rule,
    join_timeframes,
)
from components.pipelines.event_merger.matched.lob import (
    LobReplayState,
    build_multilevel_lob,
    resume_from_day_snapshot,
)
from components.pipelines.event_merger.matched.parquet import (
    list_parquet,
    read_parquet,
    read_batched_parquet,
    write_parquet,
    write_parquet_batched,
)


def precision(df: pd.DataFrame) -> None:
    """Calculates and prints the average time interval between consecutive rows."""
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.to_series()
    else:
        raise ValueError("No 'timestamp' column or DatetimeIndex found in DataFrame.")

    timestamps = timestamps.sort_values()
    avg_interval = timestamps.diff().mean()

    print(f"Average time interval: {avg_interval}")

    avg_seconds = avg_interval.total_seconds() if pd.notna(avg_interval) else 0.0
    print(f"Average interval in seconds: {avg_seconds:.6f} s")
    print(f"Average interval in milliseconds: {avg_seconds * 1000:.3f} ms")


def nan_checker(
    df: pd.DataFrame, micro_feat_col: Union[str, List[str]]
) -> None:
    """Logs total NaN counts and NaN counts specific to micro features."""
    print("Total NaNs:", df.isna().sum().sum())
    print("\nColumns with NaNs:")
    print(df.isna().sum().loc[lambda x: x > 0])

    print("\nMicro feature:", micro_feat_col)
    if isinstance(micro_feat_col, list):
        print("Micro feature NaNs:\n", df[micro_feat_col].isna().sum())
    else:
        print("Micro feature NaNs:", df[micro_feat_col].isna().sum())


def macro_features(
    df: pd.DataFrame, timeframe: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Returns higher timeframe (HTF) feature computation DataFrame.

    - Runs reconstruct(df, timeframe) prior to feature computation.
    - Computes RSI D2 trigger features.
    """
    df = df.copy()
    df = reconstruct(df, timeframe)

    df, feat_col = add_rsi_d2_trigger(
        df,
        price_col="price_close",
        rsi_period=2,
        d2_ewm_span=3,
        oversold=30.0,
        overbought=70.0,
    )

    feat_cols = [feat_col] if isinstance(feat_col, str) else list(feat_col)
    return df, feat_cols


def micro_features(
    df: pd.DataFrame, timeframe: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Returns lower timeframe (LTF) feature computation DataFrame.

    - Runs reconstruct(df, timeframe) prior to feature computation.
    - Computes orderflow features.
    """
    df = df.copy()
    df = reconstruct(df, timeframe)
    df, feat_col = add_orderflow(df)

    feat_cols = [feat_col] if isinstance(feat_col, str) else list(feat_col)
    return df, feat_cols


def compute_feature(df: pd.DataFrame, delay: int = 1) -> pd.DataFrame:
    """Joins macro and micro features while applying shifting to avoid lookahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Source tick or bar data.
    delay : int, default=1
        Lookahead prevention delay (in rows/bars).

    Returns
    -------
    pd.DataFrame
        DataFrame containing price_close and calculated feature columns.
    """
    df = df.copy()

    # 1. Compute mid_price on original frame
    df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2.0

    df_1, macro_feat_cols = macro_features(df, "1min")
    df_2, micro_feat_cols = micro_features(df, "100ms")
    df_merged = join_timeframes(df_2, df_1, "1min", causal_col=None)

    # TEMPORARY SHORT FIX:
    # For htf columns and mid price fix.
    macro_feat_cols = [f"htf_{col}" for col in macro_feat_cols]
    if "mid_price" not in df_merged.columns and "mid_price_close" in df_merged.columns:
        df_merged["mid_price"] = df_merged["mid_price_close"]

    # Lookahead bias fix.
    feat_cols = macro_feat_cols + micro_feat_cols + ["mid_price"]
    df_merged[feat_cols] = df_merged[feat_cols].shift(delay)

    # TEMPORARY SHORT FIX:
    # We left the outputted table with lots of NaNs to represent the data hasn't updated yet.
    # Use 'last available information' method via ffill.
    df_merged[feat_cols] = df_merged[feat_cols].ffill()
    df_merged = df_merged.dropna(subset=feat_cols)
    df_merged["price_close"] = df_merged["price_close"].ffill().fillna(df_merged["mid_price"])

    return df_merged[["price_close"] + feat_cols], feat_cols