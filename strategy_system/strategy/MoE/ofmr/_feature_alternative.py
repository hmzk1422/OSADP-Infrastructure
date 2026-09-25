import pandas as pd
import numpy as np

from config.research import moe as cfg
from strategy_system.strategy.MoE.ofmr.transform.rsidprice import add_rsi_d2_trigger
from strategy_system.strategy.MoE.ofmr.transform.orderflow import add_orderflow
from strategy_system.strategy.MoE.ofmr.diagnosis.feature import test_nulls
from components.python.data_pipeline.matched.reconstruct import (
    reconstruct,
    rule,
    join_timeframes,
)

from components.python.data_pipeline.matched.lob import (
    LobReplayState,
    build_multilevel_lob,
    resume_from_day_snapshot,
)

from components.python.data_pipeline.matched.parquet import (
    list_parquet,
    read_parquet,
    read_batched_parquet,
    write_parquet,
    write_parquet_batched,
)

def macro_features(df, timeframe):
    """
    Returns higher timeframe (HTF) feature computation DF.
    - Runs reconstruct(df, timeframe) first before computing the features
    - returns dataframe reconstructed based on the timeframe
    """
    df = reconstruct(
        df,
        timeframe,
    )

    df, feat_col = add_rsi_d2_trigger(
        df,
        price_col="price_close",
        rsi_period=2,
        d2_ewm_span=3,
        oversold=30.0,
        overbought=70.0,
    )

    return df, feat_col

def micro_features(df, timeframe):
    """
    Returns the lower timeframe (LTF) feature computation DF.
    - Runs reconstruct(df, timeframe) first before computing the feature.
    - Run macro_features first before running micro_features as this will be the final output
    """
    df = reconstruct(df, timeframe)
    print("new edit 1", df)
    df, feat_col = add_orderflow(df)

    return df, feat_col

def compute_feature(df, delay=1):
    """
    Joins the timeframe between macro features and micro features.
    - df: Data source from matched parquet files
    - delay: Account for infrastructural limitations. Delay represents time limitation
    to avoid lookahead bias.
    """
    df_1 = macro_features(
        df,
        "5m",
    )
    df_2 = micro_features(
        df,
        "100ms",
    )
    df = join_timeframes(
        df_2,
        df_1,
        causal_col=None
    )
    

    diff = df.index.to_series().diff()

    median_diff = diff.median()

    print("Median interval:", median_diff)
    print("Median milliseconds:", median_diff.total_seconds() * 1_000)
    print("Median microseconds:", median_diff.total_seconds() * 1_000_000)



# ============================================================
# READ MULTIPLE PARQUET FILES
# ============================================================

files = list_parquet()

files = files[:20]

df = pd.concat(
    (
        read_parquet(file)
        for file in files
    ),
    ignore_index=True,
)

# ============================================================
# SORT RAW DATA
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    errors="coerce",
)

df = (
    df
    .dropna(subset=["timestamp"])
    .sort_values(
        ["timestamp", "agg_trade_id"],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

occupied = (
    df["timestamp"]
    .dt.floor("100ms")
    .drop_duplicates()
    .sort_values()
)

gaps = occupied.diff().dropna()

print("Occupied bins:", len(occupied))
print("\nGap statistics:")
print(
    gaps.describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]
    )
)

print("\nFirst 30 occupied bins:")
print(occupied.head(30).to_string(index=False))

print("\nFirst 30 gaps:")
print(gaps.head(30).to_string(index=False))


# ============================================================
# RECONSTRUCT
# ============================================================

compute_feature(df)
print(df)