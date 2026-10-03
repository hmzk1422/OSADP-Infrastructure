"""Underlying-only delta hedge for the synthetic VDC pair."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_system.strategy.MoE.ofmr.hypothesis.vdc.vdc_hedge_intervals import assign_time_intervals


def minimum_vdc_margin(
    fund_state: pd.DataFrame,
    position: int = -1,
    h_lev: float | None = None,
    hedge_interval=None,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Calculate underlying hedge units from the synthetic fund state.

    ``hedge_interval`` is independent of the fund reset interval. It defaults
    to the fund's ``h_interval`` so the underlying-only path exactly matches
    the discrete fund reset model. Use ``"bar"`` to rebalance each raw
    observation, or supply a time duration for a coarser/staler hedge variant.
    The returned ``hedge_units`` are held over the price move ending on that
    row; a target is generated after the current row's P&L is marked.

    The model carries no TQQQ/SQQQ or synthetic-fund units. Fund values are
    bookkeeping state used to calculate the next underlying hedge target.
    """
    if not isinstance(fund_state, pd.DataFrame):
        raise TypeError("fund_state must be a pandas DataFrame")
    if fund_state.empty:
        raise ValueError("fund_state must contain at least one observation")
    if position not in (-1, 1):
        raise ValueError("position must be either -1 or +1")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")
    required = {
        "price", "leveraged_fund", "inverse_leveraged_fund",
        "fund_interval_boundary",
    }
    missing = sorted(required.difference(fund_state.columns))
    if missing:
        raise ValueError(f"fund_state is missing required columns: {missing}")
    if h_lev is None:
        if "h_lev" not in fund_state:
            raise ValueError("h_lev must be supplied or present in fund_state")
        h_lev = float(fund_state["h_lev"].iloc[0])
    if h_lev <= 0:
        raise ValueError("h_lev must be > 0")

    out = fund_state.copy()
    n = len(out)
    price = out["price"].to_numpy(dtype=np.float64)
    long_fund = out["leveraged_fund"].to_numpy(dtype=np.float64)
    short_fund = out["inverse_leveraged_fund"].to_numpy(dtype=np.float64)
    boundary = out["fund_interval_boundary"].to_numpy(dtype=bool)
    session_end = (
        out["session_end"].to_numpy(dtype=bool)
        if "session_end" in out.columns
        else np.zeros(n, dtype=bool)
    )

    if hedge_interval is None:
        hedge_boundary = boundary.copy()
        hedge_interval_label = (
            out["h_interval"].iloc[0]
            if "h_interval" in out.columns
            else "fund_interval"
        )
    elif isinstance(hedge_interval, str) and hedge_interval.strip().lower() == "bar":
        hedge_boundary = np.ones(n, dtype=bool)
        hedge_interval_label = "bar"
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise TypeError("time-based hedge_interval requires a DatetimeIndex")
        sessionize = bool(
            out["sessionize"].iloc[0]
            if "sessionize" in out.columns
            else True
        )
        mapped = assign_time_intervals(out.index, hedge_interval, sessionize=sessionize)
        hedge_boundary = mapped["fund_interval_boundary"].to_numpy(dtype=bool)
        hedge_interval_label = hedge_interval

    # Mark P&L with the previous row's post-trade target. Rebalance events
    # establish the target for the next observation, then forward-fill it.
    rebalance_event = hedge_boundary | session_end
    desired = (
        initial_capital * position * 0.5 * h_lev
        * (long_fund - short_fund) / price
    )
    desired[session_end] = 0.0
    post_trade_units = np.full(n, np.nan, dtype=np.float64)
    post_trade_units[rebalance_event] = desired[rebalance_event]
    post_trade_units = pd.Series(post_trade_units, index=out.index).ffill().fillna(0.0).to_numpy()
    held_units = np.zeros(n, dtype=np.float64)
    held_units[1:] = post_trade_units[:-1]
    target_units = post_trade_units
    trade_units = post_trade_units - held_units
    trade_units[~rebalance_event] = 0.0
    trade_notional = np.abs(trade_units) * price
    hedge_notional = np.abs(post_trade_units) * price
    rebalance = rebalance_event & (trade_units != 0.0)

    out["hedge_units"] = held_units
    out["hedge_target_units"] = target_units
    out["hedge_trade_units"] = trade_units
    out["hedge_notional"] = hedge_notional
    out["hedge_trade_notional"] = trade_notional
    out["hedge_rebalance"] = rebalance
    out["hedge_interval"] = hedge_interval_label
    out["hedge_exact_fund_replication"] = bool(np.array_equal(hedge_boundary, boundary))
    out["position"] = position
    out["initial_capital"] = initial_capital
    out["hedge_leverage"] = h_lev
    return out
