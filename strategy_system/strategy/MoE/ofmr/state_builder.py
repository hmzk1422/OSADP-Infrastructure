from __future__ import annotations

import numpy as np
import pandas as pd

from components.python.diagnostic.lossless import lob_report
from strategy_system.strategy.MoE.ofmr.feature_builder import compute_feature
from strategy_system.strategy.MoE.ofmr.transform.triple_barrier import add_triple_barrier_sequence
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


def build_state(df):
    """
    Builds the internal strategy mechanism for strategy.py
    to backtest and interact with.
    """

    # -------------------------------------------------------
    # Reconstruct 1-minute timeframe.
    # -------------------------------------------------------

    df = compute_feature(df)

    # -------------------------------------------------------
    # Inspect triggers.
    # -------------------------------------------------------
    
    """
    tb = add_triple_barrier_sequence(
        state,
        signal_col="rsi_d2_signal",
        price_col="price_close",

        # 0.5% profit target
        pt=0.005,

        # 0.3% stop
        sl=0.003,

        # 30 one-minute bars
        horizon=30,

        # Use these when available:
        high_col="price_high" if "price_high" in state.columns else None,
        low_col="price_low" if "price_low" in state.columns else None,
    )
    """

    return df