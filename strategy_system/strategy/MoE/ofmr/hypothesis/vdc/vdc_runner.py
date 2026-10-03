"""Orchestration entry point for the modular intraday VDC research path."""

from __future__ import annotations

import pandas as pd

from strategy_system.strategy.MoE.ofmr.hypothesis.vdc.vdc_leveraged_fund import vdc_leveraged_fund
from strategy_system.strategy.MoE.ofmr.hypothesis.vdc.minimum_vdc_margin import minimum_vdc_margin
from strategy_system.strategy.MoE.ofmr.hypothesis.vdc.vdc_profit import (
    calculate_vdc_profit,
    plot_vdc,
    summarize_vdc,
    _output_report,
)


def run_vdc_strategy(
    df: pd.DataFrame,
    price_column="mid_price",
    h_interval=60,
    h_lev=3.0,
    friction_bps=4.0,
    position=-1,
    execution_mode="minimum_margin",
    min_change_pct=0.0002,
    timestamp_column=None,
    sessionize=False,
    hedge_interval=None,
    initial_capital=1.0,
    report_output=False,
    plot_output=False,
):
    """Run fund construction, hedge sizing, then P&L and reporting.

    ``h_interval`` controls synthetic fund reset time. ``hedge_interval``
    independently controls minimum-margin hedge trading; its default matches
    ``h_interval``. Numeric durations are seconds. By default, fund state and
    P&L continue across UTC days. Set ``sessionize=True`` for independent UTC-
    day experiments: F/G restart each day and overnight P&L is excluded.
    """
    if execution_mode not in ("pair", "minimum_margin"):
        raise ValueError("execution_mode must be 'pair' or 'minimum_margin'")
    funds = vdc_leveraged_fund(
        df=df,
        price_column=price_column,
        h_interval=h_interval,
        h_lev=h_lev,
        min_change_pct=min_change_pct,
        timestamp_column=timestamp_column,
        sessionize=sessionize,
    )
    hedged = minimum_vdc_margin(
        fund_state=funds,
        position=position,
        h_lev=h_lev,
        hedge_interval=hedge_interval,
        initial_capital=initial_capital,
    )
    result, summary = calculate_vdc_profit(
        state=hedged,
        execution_mode=execution_mode,
        position=position,
        initial_capital=initial_capital,
        friction_bps=friction_bps,
    )
    summary.update({
        "h_interval": h_interval,
        "hedge_interval": result["hedge_interval"].iloc[0],
        "h_lev": float(h_lev),
        "min_change_pct": float(min_change_pct),
        "sessionize": bool(sessionize),
    })
    if report_output:
        _output_report(summarize_vdc(result, summary))
    if plot_output:
        plot_vdc(result, initial_capital=initial_capital)
    return result, summary
