import pandas as pd
import numpy as np

from strategy_system.strategy.MoE.vdc.plotting import plot_leveraged_pair
from components.python.diagnostic.lossless import lob_report
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
import matplotlib.pyplot as plt

from pathlib import Path

# ===============================================
# CUSTOM FUNCTIONS
# ===============================================

def add_leveraged_pair(
    df: pd.DataFrame,
    price_col: str = "price_close",
    leverage: float = 3.0,
) -> pd.DataFrame:
    """
    Add a synthetic daily-reset leveraged long/short pair.

    The pair is:

        50% × +L leveraged long
        50% × -L leveraged short

    The pair is reset to 1.0 at the beginning of each UTC day.

    Added columns
    -------------
    leveraged_long
        Synthetic +L daily-reset wealth.

    leveraged_short
        Synthetic -L daily-reset wealth.

    leveraged_pair
        Equal-weighted pair value.

    leveraged_pair_cum_return
        Cumulative return of the pair from the daily reset,
        expressed as a decimal.

        -0.01 = -1%
        +0.02 = +2%
    """

    if price_col not in df.columns:
        raise ValueError(
            f"missing price column: {price_col}"
        )

    if leverage <= 0.0:
        raise ValueError(
            "leverage must be > 0"
        )

    out = df.copy()

    if not isinstance(
        out.index,
        pd.DatetimeIndex,
    ):
        out.index = pd.to_datetime(
            out.index,
            errors="raise",
        )

    out = out.sort_index()

    price = pd.to_numeric(
        out[price_col],
        errors="coerce",
    )

    if price.isna().any():
        raise ValueError(
            f"{price_col} contains NaN values"
        )

    if (price <= 0.0).any():
        raise ValueError(
            f"{price_col} must be strictly positive"
        )

    # -------------------------------------------------------
    # UTC day.
    # -------------------------------------------------------

    day = out.index.normalize()

    # -------------------------------------------------------
    # Intraday underlying returns.
    #
    # First observation of each day starts the synthetic
    # leveraged products at 1.0.
    # -------------------------------------------------------

    returns = (
        price
        .groupby(day, sort=False)
        .pct_change()
        .fillna(0.0)
    )

    long_values = np.empty(
        len(out),
        dtype=np.float64,
    )

    short_values = np.empty(
        len(out),
        dtype=np.float64,
    )

    pair_values = np.empty(
        len(out),
        dtype=np.float64,
    )

    current_day = None

    long_value = 1.0
    short_value = 1.0

    for i, (timestamp, ret) in enumerate(
        zip(
            out.index,
            returns.to_numpy(),
        )
    ):

        timestamp_day = timestamp.normalize()

        # Reset every UTC day.
        if (
            current_day is None
            or timestamp_day != current_day
        ):
            current_day = timestamp_day

            long_value = 1.0
            short_value = 1.0

        # +L leveraged leg.
        long_value *= (
            1.0 + leverage * ret
        )

        # -L leveraged leg.
        short_value *= (
            1.0 - leverage * ret
        )

        long_value = max(
            long_value,
            0.0,
        )

        short_value = max(
            short_value,
            0.0,
        )

        long_values[i] = long_value
        short_values[i] = short_value

        # Equal capital.
        pair_values[i] = (
            0.5 * long_value
            + 0.5 * short_value
        )

    # -------------------------------------------------------
    # Store the actual pair value.
    # -------------------------------------------------------

    out["leveraged_long"] = long_values
    out["leveraged_short"] = short_values
    out["leveraged_pair"] = pair_values

    # -------------------------------------------------------
    # Direct cumulative return from the daily reset.
    #
    # 0.00  = 0%
    # -0.01  = -1%
    # +0.01  = +1%
    # -------------------------------------------------------

    out["leveraged_pair_cum_return"] = (
        pair_values - 1.0
    )

    return out

# ===============================================
# STANDARD FUNCTIONS
# ===============================================

def build_state(df):
    """
    Builds the interent strategy mechanism for strategy.py to backtest
    and interact with.
    """
        
    df_1 = reconstruct(
        df,
        "1m",
        rule(),
    )

    df = add_leveraged_pair(
        df_1,
        price_col="price_close",
        leverage=3.0,
    )

    plot_leveraged_pair(df)

    print(df)