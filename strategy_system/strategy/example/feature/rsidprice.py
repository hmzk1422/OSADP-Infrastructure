# ===========================================================
# rsidprice.py
#
# Connor's implementation of original strategy based on RSI(2)
# on S&P500. Instead, test for parabolic movement's relationship
# with RSI(2) snapback.
#
# Research hypothesis
# -------------------
# RSI(2) identifies short-term price exhaustion.
# d² price identifies acceleration / curvature.
#
# The combination is used as the base price-state signal.
#
# Output feature columns
# ----------------------
# rsi_2
# d2_price
# d2_price_norm
# d2_price_ewm3
# rsi_d2_signal
#
# Signal:
#   +1 = long
#    0 = no trigger
#   -1 = short
# ===========================================================

from __future__ import annotations

import numpy as np
import pandas as pd


# ===========================================================
# CUSTOM FUNCTIONS
# ===========================================================

def _rma(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """
    Wilder's RMA.

    Ignores leading NaN values, then uses the first `period`
    valid observations as the initial SMA seed.

    After the seed, Wilder's recursive smoothing is used.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than 0"
        )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64",
    )

    valid = series.dropna()

    if len(valid) < period:
        return result

    # -------------------------------------------------------
    # First Wilder value:
    # SMA of the first `period` valid observations.
    # -------------------------------------------------------

    seed_values = valid.iloc[:period]

    seed_index = seed_values.index[-1]

    result.loc[seed_index] = (
        seed_values.mean()
    )

    alpha = 1.0 / period

    # -------------------------------------------------------
    # Recursive Wilder smoothing.
    # -------------------------------------------------------

    valid_positions = valid.index

    start_position = period

    for position in range(
        start_position,
        len(valid_positions),
    ):
        index = valid_positions[position]

        previous_index = valid_positions[
            position - 1
        ]

        value = valid.loc[index]
        previous = result.loc[previous_index]

        result.loc[index] = (
            alpha * value
            + (1.0 - alpha) * previous
        )

    return result

def _rsi(
    price: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder RSI matching the TradingView/Pine RSI structure.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than 0"
        )

    delta = price.diff()

    gain = delta.clip(
        lower=0.0
    )

    loss = -delta.clip(
        upper=0.0
    )

    avg_gain = _rma(
        gain,
        period,
    )

    avg_loss = _rma(
        loss,
        period,
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0.0,
            np.nan,
        )
    )

    rsi = (
        100.0
        - 100.0 / (1.0 + rs)
    )

    # No average loss = fully upward
    rsi = rsi.mask(
        (avg_loss == 0.0)
        & (avg_gain > 0.0),
        100.0,
    )

    # No average gain = fully downward
    rsi = rsi.mask(
        (avg_gain == 0.0)
        & (avg_loss > 0.0),
        0.0,
    )

    # Completely flat
    rsi = rsi.mask(
        (avg_gain == 0.0)
        & (avg_loss == 0.0),
        50.0,
    )

    return rsi


# ===========================================================
# FEATURE BUILDER
# ===========================================================


def add_rsi_d2_trigger(
    df: pd.DataFrame,
    price_col: str = "price_close",
    rsi_period: int = 2,
    d2_ewm_span: int = 3,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the RSI(2) + smoothed normalized second-price-
    difference trigger.

    Long trigger:
        RSI(2) < oversold
        AND
        smoothed normalized d² price > 0

    Short trigger:
        RSI(2) > overbought
        AND
        smoothed normalized d² price < 0

    Returns
    -------
    df:
        Input dataframe plus derived features.

    feat_col:
        Explicit list of feature columns added by this function.

    Feature columns
    ---------------
    rsi_2
    d2_price
    d2_price_norm
    d2_price_ewm3
    rsi_d2_signal
    """

    if price_col not in df.columns:
        raise ValueError(
            f"missing price column: {price_col}"
        )

    if rsi_period <= 0:
        raise ValueError(
            "rsi_period must be > 0"
        )

    if d2_ewm_span <= 0:
        raise ValueError(
            "d2_ewm_span must be > 0"
        )

    if not (
        0.0 < oversold < 50.0
    ):
        raise ValueError(
            "oversold must be between 0 and 50"
        )

    if not (
        50.0 < overbought < 100.0
    ):
        raise ValueError(
            "overbought must be between 50 and 100"
        )

    out = df.copy()

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
    # RSI(2)
    # -------------------------------------------------------

    out["rsi_2"] = _rsi(
        price,
        period=rsi_period,
    )

    # -------------------------------------------------------
    # Second discrete derivative:
    #
    # d²P[t] =
    #     P[t] - 2P[t-1] + P[t-2]
    # -------------------------------------------------------

    d2_price = (
        price
        - 2.0 * price.shift(1)
        + price.shift(2)
    )

    out["d2_price"] = d2_price

    # -------------------------------------------------------
    # Normalize by price.
    # -------------------------------------------------------

    d2_price_norm = (
        d2_price
        / price.shift(2)
    )

    out["d2_price_norm"] = (
        d2_price_norm
    )

    # -------------------------------------------------------
    # EWM smoothing.
    # -------------------------------------------------------

    d2_price_ewm = (
        d2_price_norm
        .ewm(
            span=d2_ewm_span,
            adjust=False,
            min_periods=d2_ewm_span,
        )
        .mean()
    )

    out["d2_price_ewm3"] = (
        d2_price_ewm
    )

    # -------------------------------------------------------
    # Trigger conditions.
    # -------------------------------------------------------

    long_trigger = (
        (out["rsi_2"] < oversold)
        & (out["d2_price_ewm3"] > 0.0)
    )

    short_trigger = (
        (out["rsi_2"] > overbought)
        & (out["d2_price_ewm3"] < 0.0)
    )

    # -------------------------------------------------------
    # Unified signal.
    #
    # +1 = long
    #  0 = nothing
    # -1 = short
    # -------------------------------------------------------

    out["rsi_d2_signal"] = np.select(
        [
            long_trigger,
            short_trigger,
        ],
        [
            1,
            -1,
        ],
        default=0,
    ).astype(
        np.int8
    )

    feat_col = [
        "rsi_2",
        "d2_price",
        "d2_price_norm",
        "d2_price_ewm3",
        "rsi_d2_signal",
    ]

    return out, feat_col


# ===========================================================
# STANDARD FUNCTIONS
# ===========================================================