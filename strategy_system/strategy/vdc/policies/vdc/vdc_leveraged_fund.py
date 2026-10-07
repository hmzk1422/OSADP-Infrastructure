"""Synthetic intraday +L / -L leveraged-fund bookkeeping state.

For interval returns ``r_i``, ``F=prod(1+L*r_i)`` and
``G=prod(1-L*r_i)``. The odd-order terms cancel in ``(F+G)/2``; to second
order, ``pair-1 ~= (L**2/2) * ((sum(r_i))**2 - sum(r_i**2))``. This depends on
directional displacement versus realized variance, not volatility alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_system.strategy.vdc.policies.vdc.vdc_hedge_intervals import (
    assign_time_intervals,
)


def vdc_leveraged_fund(
    df: pd.DataFrame,
    price_column="mid_price",
    h_interval=60,
    h_lev=3.0,
    min_change_pct=0.0002,
    timestamp_column=None,
    sessionize=False,
) -> pd.DataFrame:
    """Build interval-reset synthetic leveraged funds and hedge state.

    ``h_interval`` is an elapsed-time interval (numeric values mean seconds;
    strings and ``Timedelta`` values are also accepted), not a row count.
    Prices rejected by ``min_change_pct`` are carried forward before returns
    are calculated, so their price moves do not enter research P&L.

    F/G compound recursively only at interval endpoints. Between endpoints,
    each fund is marked from its interval-start value using the cumulative
    underlying move, without compounding raw-bar returns. By default F/G
    compound continuously across UTC days. ``sessionize=True`` selects independent
    UTC-day experiments, restarting both fund paths at 1.0 each day and
    excluding overnight price changes from the strategy path.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if isinstance(price_column, (list, tuple)):
        if len(price_column) != 1:
            raise ValueError("price_column must identify exactly one column")
        price_column = price_column[0]
    if price_column not in df.columns:
        raise ValueError(f"missing price column: {price_column}")
    if not np.isfinite(h_lev) or h_lev <= 0:
        raise ValueError("h_lev must be > 0")
    if not np.isfinite(min_change_pct) or min_change_pct < 0:
        raise ValueError("min_change_pct must be >= 0")

    if timestamp_column is not None:
        if timestamp_column not in df.columns:
            raise ValueError(f"missing timestamp column: {timestamp_column}")
        timestamps = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
        valid_timestamp = timestamps.notna()
        source = df.loc[valid_timestamp].copy()
        timestamps = timestamps.loc[valid_timestamp]
        out = source.drop(columns=[price_column]).copy()
        out.index = pd.DatetimeIndex(timestamps.to_numpy(), name=timestamp_column)
        out[price_column] = pd.to_numeric(source[price_column].to_numpy(), errors="coerce")
    else:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            raise TypeError("df must have a DatetimeIndex or provide timestamp_column")
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")

    out["price"] = pd.to_numeric(out[price_column], errors="coerce")
    out = out.dropna(subset=["price"])
    out = out.sort_index(kind="mergesort")
    if len(out) < 2:
        raise ValueError("Need at least 2 valid price observations")
    if (~np.isfinite(out["price"].to_numpy(dtype=np.float64))).any() or (out["price"] <= 0).any():
        raise ValueError("Prices must be finite and > 0")

    # Filter against the last accepted price. Rejected observations are not
    # used in the tradable price path, fund returns, or hedge P&L.
    raw_price = out["price"].to_numpy(dtype=np.float64)
    tradable_price = np.empty_like(raw_price)
    tradable_price[0] = raw_price[0]
    last_accepted = raw_price[0]
    filter_session = out.index.normalize().asi8 if sessionize else np.zeros(len(out), dtype=np.int64)
    for i in range(1, len(raw_price)):
        if filter_session[i] != filter_session[i - 1]:
            # A fresh session opens at its observed price; do not carry a
            # prior-session close through the overnight gap filter.
            last_accepted = raw_price[i]
        elif abs(raw_price[i] / last_accepted - 1.0) >= min_change_pct:
            last_accepted = raw_price[i]
        tradable_price[i] = last_accepted
    out["raw_price"] = raw_price
    out["price"] = tradable_price
    out["price_filtered"] = raw_price != tradable_price

    intervals = assign_time_intervals(
        out.index,
        h_interval=h_interval,
        sessionize=sessionize,
    )
    # Assign positionally because raw event data can have duplicate timestamps.
    for column in intervals.columns:
        out[column] = intervals[column].to_numpy()
    out["underlying_ret"] = out["price"].pct_change().fillna(0.0)
    n = len(out)
    interval_return = np.zeros(n, dtype=np.float64)
    interval_start_price = np.full(n, np.nan, dtype=np.float64)
    interval_end_price = np.full(n, np.nan, dtype=np.float64)
    interval_observations = np.zeros(n, dtype=np.int64)
    fund_reset = out["session_start"].to_numpy(dtype=bool)
    boundary = out["fund_interval_boundary"].to_numpy(dtype=bool)
    starts = out["interval_start_observation"].to_numpy(dtype=bool)
    price = out["price"].to_numpy(dtype=np.float64)
    reset_group = out["session_id"].to_numpy()
    fund_long = np.empty(n, dtype=np.float64)
    fund_short = np.empty(n, dtype=np.float64)
    fund_long_pre = np.empty(n, dtype=np.float64)
    fund_short_pre = np.empty(n, dtype=np.float64)

    # Within each time bin, one aggregate underlying return drives one +L/-L
    # fund update. This makes the synthetic reset horizon independent of bar
    # frequency while preserving every raw bar for mark-to-market accounting.
    # Since ΔF = L*F_prev*r and ΔG = -L*G_prev*r, the equal-dollar pair change
    # is exactly 0.5*L*(F_prev-G_prev)*r. It is not a reciprocal inverse fund.
    long_state = short_state = 1.0
    previous_session = None
    interval_starts = np.flatnonzero(starts)
    interval_ends = np.flatnonzero(boundary)
    for start, end in zip(interval_starts, interval_ends):
        session = reset_group[start]
        is_new_session = previous_session is None or session != previous_session
        if sessionize and is_new_session:
            long_state = short_state = 1.0
        previous_session = session

        # First observed interval in a session starts at that session's first
        # observed price. Later returns start at the previous bin endpoint.
        start_price = price[start] if is_new_session else price[start - 1]
        interval_ret = price[end] / start_price - 1.0
        next_long = long_state * (1.0 + h_lev * interval_ret)
        next_short = short_state * (1.0 - h_lev * interval_ret)
        if next_long <= 0.0 or next_short <= 0.0:
            raise ValueError(
                "A leveraged fund became <= 0; the simple interval model "
                "requires interval returns above -1/L for +L and below "
                "+1/L for -L. "
                f"interval_return={interval_ret:.6%}, "
                f"start={start_price:.12g}, end={price[end]:.12g}, "
                f"h_interval={h_interval!r}, h_lev={h_lev!r}."
            )

        # Mark each fund through the interval using the cumulative underlying
        # move from its interval start. This is mark-to-market only: F/G are
        # recursively updated once, at the interval endpoint, and are never
        # compounded bar by bar.
        path_returns = price[start:end + 1] / start_price - 1.0
        long_factors = 1.0 + h_lev * path_returns
        short_factors = 1.0 - h_lev * path_returns
        if np.any(long_factors <= 0.0) or np.any(short_factors <= 0.0):
            bad_offset = int(np.flatnonzero((long_factors <= 0.0) | (short_factors <= 0.0))[0])
            bad_row = start + bad_offset
            raise ValueError(
                "A leveraged fund became <= 0 before the interval endpoint; "
                "the fund path is invalid at an observed price. "
                f"interval_return={path_returns[bad_offset]:.6%}, "
                f"timestamp={out.index[bad_row]!r}, "
                f"h_interval={h_interval!r}, h_lev={h_lev!r}."
            )
        fund_long[start:end + 1] = long_state * long_factors
        fund_short[start:end + 1] = short_state * short_factors
        fund_long_pre[start:end + 1] = long_state
        fund_short_pre[start:end + 1] = short_state
        fund_long[end] = next_long
        fund_short[end] = next_short
        interval_return[end] = interval_ret
        interval_start_price[end] = start_price
        interval_end_price[end] = price[end]
        interval_observations[end] = end - start + 1
        long_state, short_state = next_long, next_short

    out["interval_return"] = interval_return
    out["interval_start_price"] = interval_start_price
    out["interval_end_price"] = interval_end_price
    out["interval_observations"] = interval_observations
    out["interval_fund_return"] = interval_return.copy()
    out["fund_reset"] = fund_reset

    out["leveraged_fund_pre"] = fund_long_pre
    out["inverse_leveraged_fund_pre"] = fund_short_pre
    out["leveraged_fund"] = fund_long
    out["inverse_leveraged_fund"] = fund_short
    out["fund_interval_boundary"] = boundary
    out["pair_value"] = 0.5 * (fund_long + fund_short)
    out["fund_divergence"] = fund_long - fund_short
    out["hedge_target_units"] = (
        0.5 * h_lev * (fund_long - fund_short) / price
    )
    out["h_interval"] = h_interval
    out["h_lev"] = h_lev
    out["min_change_pct"] = min_change_pct
    out["sessionize"] = bool(sessionize)
    return out
