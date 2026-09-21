# ===========================================================
# orderflow.py
#
# Microstructure features for short-horizon price prediction.
#
# No reconstruction.
# No resampling.
# No future information.
#
# Research objective
# ------------------
#
# Determine whether order-flow / LOB information tells us
# whether an RSI(2) + d² setup is:
#
#   1. still under strong directional pressure
#   2. being absorbed / exhausted
#
# Feature families
# ----------------
#
#   1. Aggressive trade flow
#   2. BBO liquidity / spread
#   3. Order Flow Imbalance
#   4. Multi-level resting depth
#   5. Depth dynamics
#   6. Flow -> price transmission
#
# ===========================================================

from __future__ import annotations

import numpy as np
import pandas as pd


# ===========================================================
# CONSTANTS
# ===========================================================

_EPS = 1e-12


# ===========================================================
# VALIDATION
# ===========================================================


def _require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"missing required columns: {missing}"
        )


def _validate_windows(
    windows: tuple[int, ...],
) -> None:

    if not windows:
        raise ValueError(
            "windows must not be empty"
        )

    if any(
        int(window) <= 0
        for window in windows
    ):
        raise ValueError(
            "all windows must be > 0"
        )


# ===========================================================
# BASIC HELPERS
# ===========================================================


def _safe_div(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    return (
        numerator
        / denominator.replace(
            0.0,
            np.nan,
        )
    )


def _rolling_sum(
    series: pd.Series,
    window: int,
) -> pd.Series:

    return (
        series
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )


# ===========================================================
# ROBUST HISTORICAL NORMALIZATION
# ===========================================================


def _robust_zscore(
    series: pd.Series,
    baseline_window: int = 1000,
    min_periods: int = 100,
    clip: float | None = 8.0,
) -> pd.Series:
    """
    Historical robust z-score.

    Baseline uses ONLY observations before the current row.

        historical = series.shift(1)

    Robust location:
        rolling median

    Robust scale:
        1.4826 * rolling MAD

    This is deliberately used for magnitude / activity
    variables rather than bounded imbalance variables.

    Example:

        trade_notional_z_5 = unusually large 5-event flow
        ofi_z_20           = unusually large 20-event OFI
        spread_bps_z       = unusually wide spread
    """

    if baseline_window <= 0:
        raise ValueError(
            "baseline_window must be > 0"
        )

    if min_periods <= 0:
        raise ValueError(
            "min_periods must be > 0"
        )

    historical = series.shift(1)

    median = (
        historical
        .rolling(
            window=baseline_window,
            min_periods=min_periods,
        )
        .median()
    )

    deviation = (
        historical
        - median
    ).abs()

    mad = (
        deviation
        .rolling(
            window=baseline_window,
            min_periods=min_periods,
        )
        .median()
    )

    robust_scale = (
        1.4826 * mad
    )

    z = _safe_div(
        series - median,
        robust_scale,
    )

    # -------------------------------------------------------
    # Fallback for extremely discrete / quiet series where
    # MAD becomes zero.
    # -------------------------------------------------------

    historical_std = (
        historical
        .rolling(
            window=baseline_window,
            min_periods=min_periods,
        )
        .std()
    )

    fallback = _safe_div(
        series - median,
        historical_std,
    )

    z = z.where(
        robust_scale > _EPS,
        fallback,
    )

    if clip is not None:
        z = z.clip(
            -clip,
            clip,
        )

    return z


# ===========================================================
# TRADE FLOW
# ===========================================================


def add_trade_flow_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (
        1,
        5,
        20,
    ),
    baseline_window: int = 1000,
    min_periods: int = 100,
) -> pd.DataFrame:
    """
    Aggressive trade flow.

    is_buyer_maker=True:
        seller is aggressor
        => negative flow

    is_buyer_maker=False:
        buyer is aggressor
        => positive flow

    Direction:
        trade_imbalance

    Magnitude:
        trade_notional_z
    """

    _require_columns(
        df,
        [
            "quantity",
            "price_close",
            "is_buyer_maker",
        ],
    )

    _validate_windows(windows)

    out = df.copy()

    quantity = pd.to_numeric(
        out["quantity"],
        errors="coerce",
    )

    price = pd.to_numeric(
        out["price_close"],
        errors="coerce",
    )

    maker = out[
        "is_buyer_maker"
    ]

    if maker.isna().any():
        raise ValueError(
            "is_buyer_maker contains NaN values"
        )

    maker = maker.astype(bool)

    buy_qty = pd.Series(
        np.where(
            maker,
            0.0,
            quantity,
        ),
        index=out.index,
        dtype="float64",
    )

    sell_qty = pd.Series(
        np.where(
            maker,
            quantity,
            0.0,
        ),
        index=out.index,
        dtype="float64",
    )

    buy_notional = (
        buy_qty * price
    )

    sell_notional = (
        sell_qty * price
    )

    # -------------------------------------------------------
    # Preserve primitives internally.
    # -------------------------------------------------------

    out["aggressive_buy_notional"] = (
        buy_notional
    )

    out["aggressive_sell_notional"] = (
        sell_notional
    )

    out["aggressive_signed_notional"] = (
        buy_notional
        - sell_notional
    )

    # -------------------------------------------------------
    # Multi-horizon directional flow.
    # -------------------------------------------------------

    for window in windows:

        buy = _rolling_sum(
            buy_notional,
            window,
        )

        sell = _rolling_sum(
            sell_notional,
            window,
        )

        total = (
            buy + sell
        )

        signed = (
            buy - sell
        )

        # ---------------------------------------------------
        # Direction.
        # Bounded [-1, +1].
        # ---------------------------------------------------

        out[
            f"trade_imbalance_{window}"
        ] = _safe_div(
            signed,
            total,
        )

        # ---------------------------------------------------
        # Magnitude.
        # Normalize relative to historical activity.
        # ---------------------------------------------------

        out[
            f"trade_notional_z_{window}"
        ] = _robust_zscore(
            total,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

        # ---------------------------------------------------
        # Directional magnitudes.
        #
        # Retained because +imbalance with tiny activity is
        # not equivalent to +imbalance with extreme activity.
        # ---------------------------------------------------

        out[
            f"buy_notional_z_{window}"
        ] = _robust_zscore(
            buy,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

        out[
            f"sell_notional_z_{window}"
        ] = _robust_zscore(
            sell,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

    return out


# ===========================================================
# BBO / SPREAD
# ===========================================================


def add_bbo_features(
    df: pd.DataFrame,
    baseline_window: int = 1000,
    min_periods: int = 100,
) -> pd.DataFrame:
    """
    Best bid / ask state.

    Measures:

        spread
        queue imbalance
        microprice displacement
    """

    _require_columns(
        df,
        [
            "best_bid_price_close",
            "best_ask_price_close",
            "best_bid_qty",
            "best_ask_qty",
        ],
    )

    out = df.copy()

    bid = pd.to_numeric(
        out["best_bid_price_close"],
        errors="coerce",
    )

    ask = pd.to_numeric(
        out["best_ask_price_close"],
        errors="coerce",
    )

    bid_qty = pd.to_numeric(
        out["best_bid_qty"],
        errors="coerce",
    )

    ask_qty = pd.to_numeric(
        out["best_ask_qty"],
        errors="coerce",
    )

    # -------------------------------------------------------
    # Mid
    # -------------------------------------------------------

    mid = (
        bid + ask
    ) / 2.0

    out["mid_price"] = mid

    # -------------------------------------------------------
    # Spread
    # -------------------------------------------------------

    spread = (
        ask - bid
    )

    out["spread_bps"] = (
        _safe_div(
            spread,
            mid,
        )
        * 10_000.0
    )

    out["spread_bps_z"] = (
        _robust_zscore(
            out["spread_bps"],
            baseline_window=baseline_window,
            min_periods=min_periods,
        )
    )

    # -------------------------------------------------------
    # Queue imbalance.
    # -------------------------------------------------------

    out["queue_imbalance"] = _safe_div(
        bid_qty - ask_qty,
        bid_qty + ask_qty,
    )

    # -------------------------------------------------------
    # Microprice.
    # -------------------------------------------------------

    microprice = _safe_div(
        ask * bid_qty
        + bid * ask_qty,
        bid_qty + ask_qty,
    )

    out["microprice"] = (
        microprice
    )

    # -------------------------------------------------------
    # Microprice displacement measured in fractions of
    # the current spread.
    # -------------------------------------------------------

    out[
        "microprice_displacement"
    ] = _safe_div(
        microprice - mid,
        spread,
    )

    return out


# ===========================================================
# ORDER FLOW IMBALANCE
# ===========================================================


def add_ofi(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (
        1,
        5,
        20,
    ),
    baseline_window: int = 1000,
    min_periods: int = 100,
) -> pd.DataFrame:
    """
    Best-level Order Flow Imbalance.

    This is intended for chronological book observations.

    Positive:
        demand-side book pressure

    Negative:
        supply-side book pressure

    If rows are actual book events, this is an event-level OFI.

    If rows are already aggregated snapshots, it becomes an
    OFI-like snapshot-difference proxy.
    """

    _require_columns(
        df,
        [
            "best_bid_price_close",
            "best_ask_price_close",
            "best_bid_qty",
            "best_ask_qty",
        ],
    )

    _validate_windows(windows)

    out = df.copy()

    bid_price = pd.to_numeric(
        out["best_bid_price_close"],
        errors="coerce",
    )

    ask_price = pd.to_numeric(
        out["best_ask_price_close"],
        errors="coerce",
    )

    bid_qty = pd.to_numeric(
        out["best_bid_qty"],
        errors="coerce",
    )

    ask_qty = pd.to_numeric(
        out["best_ask_qty"],
        errors="coerce",
    )

    prev_bid_price = (
        bid_price.shift(1)
    )

    prev_ask_price = (
        ask_price.shift(1)
    )

    prev_bid_qty = (
        bid_qty.shift(1)
    )

    prev_ask_qty = (
        ask_qty.shift(1)
    )

    # -------------------------------------------------------
    # Bid-side OFI contribution.
    # -------------------------------------------------------

    bid_event = (
        (bid_price >= prev_bid_price)
        .astype(float)
        * bid_qty
        -
        (bid_price <= prev_bid_price)
        .astype(float)
        * prev_bid_qty
    )

    # -------------------------------------------------------
    # Ask-side OFI contribution.
    # -------------------------------------------------------

    ask_event = (
        (ask_price <= prev_ask_price)
        .astype(float)
        * ask_qty
        -
        (ask_price >= prev_ask_price)
        .astype(float)
        * prev_ask_qty
    )

    ofi_event = (
        bid_event
        - ask_event
    )

    out["ofi_event"] = (
        ofi_event
    )

    out["ofi_bid_event"] = (
        bid_event
    )

    out["ofi_ask_event"] = (
        -ask_event
    )

    # -------------------------------------------------------
    # Multi-horizon OFI.
    # -------------------------------------------------------

    for window in windows:

        ofi = _rolling_sum(
            ofi_event,
            window,
        )

        out[
            f"ofi_z_{window}"
        ] = _robust_zscore(
            ofi,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

    return out


# ===========================================================
# MULTI-LEVEL DEPTH
# ===========================================================
def add_multilevel_depth_features(
    df: pd.DataFrame,
    levels: tuple[int, ...] = (
        1,
        5,
        10,
        20,
    ),
    baseline_window: int = 1000,
    min_periods: int = 100,
) -> pd.DataFrame:
    """
    Multi-level resting liquidity.

    Level 1:
        best_bid_qty
        best_ask_qty

    Levels 2-20:
        bid_qty_N
        ask_qty_N
    """

    _validate_windows(levels)

    out = df.copy()

    def bid_qty_column(level: int) -> str:
        return (
            "best_bid_qty"
            if level == 1
            else f"bid_qty_{level}"
        )

    def ask_qty_column(level: int) -> str:
        return (
            "best_ask_qty"
            if level == 1
            else f"ask_qty_{level}"
        )

    for depth in levels:

        bid_cols = [
            bid_qty_column(level)
            for level in range(1, depth + 1)
        ]

        ask_cols = [
            ask_qty_column(level)
            for level in range(1, depth + 1)
        ]

        _require_columns(
            out,
            bid_cols + ask_cols,
        )

        # ---------------------------------------------------
        # Convert each column individually to numeric.
        # ---------------------------------------------------

        bid_depth = (
            out[bid_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .sum(axis=1)
        )

        ask_depth = (
            out[ask_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .sum(axis=1)
        )

        total_depth = (
            bid_depth
            + ask_depth
        )

        # ---------------------------------------------------
        # Raw directional depth.
        # ---------------------------------------------------

        out[
            f"bid_depth_{depth}"
        ] = bid_depth

        out[
            f"ask_depth_{depth}"
        ] = ask_depth

        # ---------------------------------------------------
        # Historical normalized depth.
        # ---------------------------------------------------

        out[
            f"bid_depth_z_{depth}"
        ] = _robust_zscore(
            bid_depth,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

        out[
            f"ask_depth_z_{depth}"
        ] = _robust_zscore(
            ask_depth,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

        # ---------------------------------------------------
        # Relative directional depth.
        #
        # +1 = bid dominates
        # -1 = ask dominates
        # ---------------------------------------------------

        out[
            f"depth_imbalance_{depth}"
        ] = _safe_div(
            bid_depth - ask_depth,
            total_depth,
        )

    return out

# ===========================================================
# DEPTH DYNAMICS
# ===========================================================


def add_depth_dynamics(
    df: pd.DataFrame,
    depths: tuple[int, ...] = (
        1,
        5,
        20,
    ),
    baseline_window: int = 1000,
    min_periods: int = 100,
) -> pd.DataFrame:
    """
    Measure liquidity entering / leaving the book.

    Positive bid change:
        bid liquidity increased.

    Negative bid change:
        bid liquidity decreased.

    Positive ask change:
        ask liquidity increased.

    Negative ask change:
        ask liquidity decreased.

    The directional information is therefore retained as
    separate bid / ask variables.
    """

    _validate_windows(depths)

    out = df.copy()

    for depth in depths:

        bid_col = (
            f"bid_depth_{depth}"
        )

        ask_col = (
            f"ask_depth_{depth}"
        )

        _require_columns(
            out,
            [
                bid_col,
                ask_col,
            ],
        )

        bid_depth = out[
            bid_col
        ]

        ask_depth = out[
            ask_col
        ]

        # ---------------------------------------------------
        # Relative depth changes.
        #
        # Relative changes are more comparable across
        # market regimes than raw quantity deltas.
        # ---------------------------------------------------

        bid_change = _safe_div(
            bid_depth - bid_depth.shift(1),
            bid_depth.shift(1).abs(),
        )

        ask_change = _safe_div(
            ask_depth - ask_depth.shift(1),
            ask_depth.shift(1).abs(),
        )

        out[
            f"bid_depth_change_{depth}"
        ] = bid_change

        out[
            f"ask_depth_change_{depth}"
        ] = ask_change

        out[
            f"bid_depth_change_z_{depth}"
        ] = _robust_zscore(
            bid_change,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

        out[
            f"ask_depth_change_z_{depth}"
        ] = _robust_zscore(
            ask_change,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )

        # ---------------------------------------------------
        # Net directional liquidity change.
        #
        # Positive:
        # bid strengthened relative to ask.
        #
        # Negative:
        # ask strengthened relative to bid.
        # ---------------------------------------------------

        out[
            f"depth_flow_imbalance_{depth}"
        ] = (
            bid_change
            - ask_change
        )

    return out


# ===========================================================
# FLOW -> PRICE TRANSMISSION
# ===========================================================


def add_flow_price_features(
    df: pd.DataFrame,
    window: int = 5,
    baseline_window: int = 1000,
    min_periods: int = 100,
) -> pd.DataFrame:
    """
    Measure whether aggressive flow is actually transmitting
    into the mid-price.

    Large aggressive flow + large price movement:
        flow is propagating into price.

    Large aggressive flow + weak price movement:
        potential absorption / resistance.

    This feature is strictly backward-looking relative to t.
    The eventual prediction target must begin after t.
    """

    if window <= 0:
        raise ValueError(
            "window must be > 0"
        )

    _require_columns(
        df,
        [
            "mid_price",
            "aggressive_signed_notional",
        ],
    )

    out = df.copy()

    mid = out[
        "mid_price"
    ]

    flow = _rolling_sum(
        out[
            "aggressive_signed_notional"
        ],
        window,
    )

    price_return = np.log(
        mid
        / mid.shift(window)
    )

    # -------------------------------------------------------
    # Historical price movement.
    # -------------------------------------------------------

    out[
        f"mid_return_{window}"
    ] = price_return

    # -------------------------------------------------------
    # Price response per unit of directional aggressive flow.
    # -------------------------------------------------------

    efficiency = (
        price_return
        / (
            flow.abs()
            + _EPS
        )
    )

    out[
        f"flow_price_efficiency_{window}"
    ] = efficiency

    out[
        f"flow_price_efficiency_z_{window}"
    ] = _robust_zscore(
        efficiency,
        baseline_window=baseline_window,
        min_periods=min_periods,
    )

    # -------------------------------------------------------
    # Whether flow and price moved together.
    #
    # +1 = same direction
    # -1 = opposite direction
    #  0 = one side is flat
    # -------------------------------------------------------

    out[
        f"flow_price_alignment_{window}"
    ] = (
        np.sign(flow)
        * np.sign(price_return)
    )

    return out


# ===========================================================
# MASTER ORDERFLOW BUILDER
# ===========================================================


def add_orderflow(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (
        1,
        5,
        20,
    ),
    depth_levels: tuple[int, ...] = (
        1,
        5,
        10,
        20,
    ),
    depth_dynamic_levels: tuple[int, ...] = (
        1,
        5,
        20,
    ),
    baseline_window: int = 1000,
    min_periods: int = 100,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the planned microstructure feature set.

    No reconstruction.
    No resampling.

    Returns
    -------
    df:
        Input dataframe plus derived microstructure columns.

    feat_col:
        Explicit list of the important features added by this
        function.

    The feature list intentionally excludes most intermediate
    diagnostic columns.
    """

    _validate_windows(windows)
    _validate_windows(depth_levels)
    _validate_windows(depth_dynamic_levels)

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must already be in chronological order."
        )

    out = df.copy()

    # -------------------------------------------------------
    # 1. Aggressive trade flow.
    # -------------------------------------------------------

    out = add_trade_flow_features(
        out,
        windows=windows,
        baseline_window=baseline_window,
        min_periods=min_periods,
    )

    # -------------------------------------------------------
    # 2. BBO / spread / queue / microprice.
    # -------------------------------------------------------

    out = add_bbo_features(
        out,
        baseline_window=baseline_window,
        min_periods=min_periods,
    )

    # -------------------------------------------------------
    # 3. OFI.
    # -------------------------------------------------------

    out = add_ofi(
        out,
        windows=windows,
        baseline_window=baseline_window,
        min_periods=min_periods,
    )

    # -------------------------------------------------------
    # 4. Multi-level depth.
    # -------------------------------------------------------

    out = add_multilevel_depth_features(
        out,
        levels=depth_levels,
        baseline_window=baseline_window,
        min_periods=min_periods,
    )

    # -------------------------------------------------------
    # 5. Depth dynamics.
    # -------------------------------------------------------

    out = add_depth_dynamics(
        out,
        depths=depth_dynamic_levels,
        baseline_window=baseline_window,
        min_periods=min_periods,
    )

    # -------------------------------------------------------
    # 6. Flow -> price transmission.
    # -------------------------------------------------------

    out = add_flow_price_features(
        out,
        window=5,
        baseline_window=baseline_window,
        min_periods=min_periods,
    )

    # =======================================================
    # FINAL RESEARCH FEATURE LIST
    # =======================================================
    #
    # This is the important part.
    #
    # Intermediate quantities remain available in `out` for
    # diagnostics, but only these columns are returned in
    # `feat_col`.
    #
    # =======================================================

    feat_col = [

        # ---------------------------------------------------
        # Aggressive flow
        # ---------------------------------------------------

        "trade_imbalance_1",
        "trade_imbalance_5",
        "trade_imbalance_20",

        "trade_notional_z_5",

        "buy_notional_z_5",
        "sell_notional_z_5",

        # ---------------------------------------------------
        # BBO / immediate liquidity
        # ---------------------------------------------------

        "spread_bps_z",

        "queue_imbalance",

        "microprice_displacement",

        # ---------------------------------------------------
        # Order-flow imbalance
        # ---------------------------------------------------

        "ofi_z_1",
        "ofi_z_5",
        "ofi_z_20",

        # ---------------------------------------------------
        # Multi-level resting liquidity
        # ---------------------------------------------------

        "depth_imbalance_1",
        "depth_imbalance_5",
        "depth_imbalance_10",
        "depth_imbalance_20",

        # ---------------------------------------------------
        # Depth dynamics
        # ---------------------------------------------------

        "bid_depth_change_z_5",
        "ask_depth_change_z_5",

        # ---------------------------------------------------
        # Flow -> price transmission
        # ---------------------------------------------------

        "flow_price_efficiency_z_5",
    ]

    return out, feat_col


# ===========================================================
# STANDARD FUNCTIONS
# ===========================================================