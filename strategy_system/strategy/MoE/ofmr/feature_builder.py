import pandas as pd
import numpy as np

from config.research import moe as cfg
from strategy_system.strategy.MoE.ofmr.transform.rsidprice import add_rsi_d2_trigger
from strategy_system.strategy.MoE.ofmr.transform.orderflow import add_orderflow
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
    df = df.copy()
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
    df = df.copy()
    df = reconstruct(df, timeframe)
    df, feat_col = add_orderflow(df)

    return df, feat_col

def compute_feature(df):
    """
    Final validation check before state_builder
    - Ensures lookahead bias is avoided clearly here
    - Exports only the columns needed
    """
    df_1, macro_feat_col = macro_features(
        df,
        "5min",
    )

    df_2, micro_feat_col = micro_features(
        df,
        "1min"
    )
    df = join_timeframes(
        df_2,
        df_1,
        "5min",
    )
    # join_timeframes prefixes higher-timeframe columns
    # with "htf_".
    macro_feat_col = [
        f"htf_{col}"
        for col in macro_feat_col
    ]

    feat_col = (
        macro_feat_col
        + micro_feat_col
    )

    # Features use information available at t-1.
    # price_close remains unshifted.

    df[feat_col] = (
        df[feat_col]
        .shift(1)
    )
    df = df.dropna(
        subset=feat_col
    )

    return df[
        ["price_close"] + feat_col
    ]