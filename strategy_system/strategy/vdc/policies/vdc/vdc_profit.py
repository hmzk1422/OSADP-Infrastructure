"""VDC P&L, transaction costs, summary reporting, and matplotlib plot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_vdc_profit(
    state: pd.DataFrame,
    execution_mode: str = "minimum_margin",
    position: int = -1,
    initial_capital: float = 1.0,
    friction_bps: float = 4.0,
) -> tuple[pd.DataFrame, dict]:
    """Calculate pair or underlying-only P&L with one-way traded-notional fees.

    ``friction_bps`` is a one-way rate charged on every dollar of traded
    notional. It is applied to initial/closing pair legs in pair mode and to
    underlying hedge turnover in minimum-margin mode. Borrow, financing, and
    short availability costs are not modeled.
    """
    if not isinstance(state, pd.DataFrame):
        raise TypeError("state must be a pandas DataFrame")
    if execution_mode not in ("pair", "minimum_margin"):
        raise ValueError("execution_mode must be 'pair' or 'minimum_margin'")
    if position not in (-1, 1):
        raise ValueError("position must be either -1 or +1")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")
    if not np.isfinite(friction_bps) or friction_bps < 0:
        raise ValueError("friction_bps must be >= 0")
    required = {
        "price", "leveraged_fund", "inverse_leveraged_fund",
        "fund_interval_boundary", "session_start", "session_end",
    }
    if execution_mode == "minimum_margin":
        required.update({"hedge_units", "hedge_trade_notional", "hedge_notional"})
    missing = sorted(required.difference(state.columns))
    if missing:
        raise ValueError(f"state is missing required columns: {missing}")
    if state.empty:
        raise ValueError("state must contain at least one observation")

    out = state.copy()
    n = len(out)
    price = out["price"].to_numpy(dtype=np.float64)
    long_fund = out["leveraged_fund"].to_numpy(dtype=np.float64)
    short_fund = out["inverse_leveraged_fund"].to_numpy(dtype=np.float64)
    boundary = out["fund_interval_boundary"].to_numpy(dtype=bool)
    session_start = out["session_start"].to_numpy(dtype=bool)
    session_end = out["session_end"].to_numpy(dtype=bool)

    fund_pnl = np.zeros(n, dtype=np.float64)
    hedge_pnl = np.zeros(n, dtype=np.float64)
    gross_pnl = np.zeros(n, dtype=np.float64)
    trade_notional = np.zeros(n, dtype=np.float64)
    transaction_cost = np.zeros(n, dtype=np.float64)
    one_way_rate = friction_bps / 10_000.0

    if execution_mode == "pair":
        # One normalized pair consists of 0.5 units of each fund. Fund paths
        # are marked within each reset interval, then recursively updated at
        # its endpoint. Re-entry occurs at each independent session start.
        trade_notional[session_start] += initial_capital
        long_change = np.diff(long_fund, prepend=long_fund[0])
        short_change = np.diff(short_fund, prepend=short_fund[0])
        fund_pnl[:] = position * 0.5 * initial_capital * (long_change + short_change)
        fund_pnl[session_start] = 0.0
        trade_notional[session_end] += (
            initial_capital * 0.5
            * (np.abs(long_fund[session_end]) + np.abs(short_fund[session_end]))
        )
        gross_pnl[:] = fund_pnl
    else:
        hedge_units = out["hedge_units"].to_numpy(dtype=np.float64)
        price_change = np.zeros(n, dtype=np.float64)
        if n > 1:
            same_session = ~session_start[1:]
            price_change[1:] = np.diff(price) * same_session
        hedge_pnl[1:] = hedge_units[1:] * price_change[1:]
        gross_pnl[:] = hedge_pnl
        trade_notional = out["hedge_trade_notional"].to_numpy(dtype=np.float64).copy()

    transaction_cost = trade_notional * one_way_rate
    net_pnl = gross_pnl - transaction_cost
    cumulative_pnl = np.cumsum(net_pnl)
    strategy_nav = initial_capital + cumulative_pnl
    cumulative_return = strategy_nav / initial_capital - 1.0

    out["fund_pnl"] = fund_pnl
    out["hedge_pnl"] = hedge_pnl
    out["gross_pnl"] = gross_pnl
    out["trade_notional"] = trade_notional
    out["transaction_cost"] = transaction_cost
    out["net_pnl"] = net_pnl
    out["cumulative_pnl"] = cumulative_pnl
    out["strategy_nav"] = strategy_nav
    out["cumulative_return"] = cumulative_return
    out["execution_mode"] = execution_mode
    out["position"] = position
    out["initial_capital"] = initial_capital
    out["friction_bps_one_way"] = friction_bps

    # Aggregate raw-bar hedge P&L over each actual completed fund interval.
    # This also makes pair and minimum-margin win/risk statistics comparable.
    interval_pnl = (
        pd.Series(net_pnl, index=out.index)
        .groupby(out["interval_id"].to_numpy(), sort=False)
        .sum()
        .to_numpy(dtype=np.float64)
    )
    num_intervals = int(np.count_nonzero(boundary))
    nonzero_interval_pnl = interval_pnl[interval_pnl != 0.0]
    win_rate = (
        float(np.mean(nonzero_interval_pnl > 0.0) * 100.0)
        if len(nonzero_interval_pnl) else np.nan
    )
    downside = np.minimum(interval_pnl, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside ** 2))) if len(interval_pnl) else np.nan
    avg_interval_return = float(np.mean(interval_pnl) / initial_capital) if len(interval_pnl) else np.nan
    downside_deviation_return = downside_deviation / initial_capital
    if np.isfinite(downside_deviation_return) and downside_deviation_return > 0:
        sortino = avg_interval_return / downside_deviation_return
    elif len(interval_pnl) and avg_interval_return > 0:
        sortino = np.inf
    else:
        sortino = np.nan
    running_peak = np.maximum.accumulate(
        np.r_[initial_capital, strategy_nav]
    )[1:]
    drawdown = strategy_nav / running_peak - 1.0
    hedge_trade = (
        out["hedge_trade_notional"].to_numpy(dtype=np.float64)
        if "hedge_trade_notional" in out else np.zeros(n)
    )
    hedge_notional = (
        out["hedge_notional"].to_numpy(dtype=np.float64)
        if "hedge_notional" in out else np.zeros(n)
    )
    hedge_turnover = float(hedge_trade.sum()) if execution_mode == "minimum_margin" else 0.0
    execution_turnover = float(trade_notional.sum())
    summary = {
        "initial_capital": float(initial_capital),
        "final_cumulative_pnl": float(cumulative_pnl[-1]),
        "final_cumulative_return": float(cumulative_return[-1]),
        "final_nav": float(strategy_nav[-1]),
        "underlying_return": float(price[-1] / price[0] - 1.0),
        "gross_pnl": float(gross_pnl.sum()),
        "transaction_cost": float(transaction_cost.sum()),
        "net_pnl": float(net_pnl.sum()),
        "total_hedge_turnover": hedge_turnover,
        "total_execution_turnover": execution_turnover,
        "num_rebalances": int(np.count_nonzero(trade_notional)),
        "num_hedge_rebalances": int(np.count_nonzero(hedge_trade)),
        "num_fund_intervals": num_intervals,
        "final_pair_value": float(out["pair_value"].iloc[-1]),
        "final_fund_divergence": float(out["fund_divergence"].iloc[-1]),
        "final_hedge_notional": float(hedge_notional[-1]),
        "max_hedge_notional": float(np.max(hedge_notional)),
        "hedge_exact_fund_replication": bool(
            out["hedge_exact_fund_replication"].iloc[0]
            if "hedge_exact_fund_replication" in out.columns
            else True
        ),
        "execution_mode": execution_mode,
        "position": position,
        "friction_bps_one_way": float(friction_bps),
        "win_rate_pct": win_rate,
        "mdd_pct": float(np.min(drawdown) * 100.0),
        "sortino": float(sortino),
        "downside_deviation": float(downside_deviation_return),
        "avg_return_pct_per_interval": float(avg_interval_return * 100.0),
        "total_transaction_cost": float(transaction_cost.sum()),
        "avg_hedge_turnover_per_interval": (
            float(execution_turnover / num_intervals) if num_intervals else np.nan
        ),
        "turnover_to_capital": float(execution_turnover / initial_capital),
        "borrow_cost_modeled": False,
    }
    return out, summary


def _output_report(report: dict) -> None:
    """Print the compact tabular report used by the original VDC research."""
    display_report = {
        "Initial Capital": report["initial_capital"],
        "Final NAV": report["final_nav"],
        "Final P&L": report["final_pnl"],
        "Final Return %": report["final_return_pct"],
        "Avg Return / Hedge %": report["avg_return_pct_per_hedge"],
        "Max Drawdown %": report["mdd_pct"],
        "Sortino": report["sortino"],
        "Win Rate %": report["win_rate_pct"],
        "Total Execution Turnover": report["total_execution_turnover"],
        "Avg Turnover / Hedge": report["avg_hedge_turnover_per_period"],
        "Turnover / Capital": report["turnover_to_capital"],
        "Transaction Cost": report["total_transaction_cost"],
        "Fund Intervals": report["num_hedge_periods"],
        "Hedge Trades": report["num_hedge_rebalances"],
        "Execution Rebalances": report["num_rebalances"],
        "Final Pair Value": report["final_pair_value"],
        "Final Fund Divergence": report["final_fund_divergence"],
        "Final Hedge Notional": report["final_hedge_notional"],
        "Max Hedge Notional": report["max_hedge_notional"],
        "Execution Mode": report["execution_mode"],
        "Fund Interval": report.get("h_interval", "n/a"),
        "Hedge Interval": report.get("hedge_interval", "n/a"),
        "Friction (bps, one-way)": report["friction_bps_one_way"],
    }
    table = pd.DataFrame(display_report.items(), columns=["Metric", "Value"])
    print("\n" + "=" * 50)
    print("VOLATILITY DECAY CAPTURE REPORT")
    print("=" * 50)
    print(table.to_string(index=False, formatters={
        "Value": lambda x: f"{x:.6f}" if isinstance(x, (int, float, np.number)) else str(x)
    }))


def summarize_vdc(result: pd.DataFrame, summary: dict) -> dict:
    """Expand the accounting summary with report-style risk/turnover metrics."""
    return {
        **summary,
        "final_pnl": float(result["net_pnl"].sum()),
        "final_nav": float(result["strategy_nav"].iloc[-1]),
        "final_return_pct": float(result["cumulative_return"].iloc[-1] * 100.0),
        "avg_return_pct_per_hedge": summary["avg_return_pct_per_interval"],
        "avg_hedge_turnover_per_period": summary["avg_hedge_turnover_per_interval"],
        "num_hedge_periods": summary["num_fund_intervals"],
        "num_hedge_rebalances": summary["num_hedge_rebalances"],
    }


def plot_vdc(result: pd.DataFrame, initial_capital=1.0, price_col="price") -> None:
    """Plot synthetic pair notionals, underlying price, and cumulative return."""
    required = ["leveraged_fund", "inverse_leveraged_fund", "hedge_notional", "cumulative_return", "strategy_nav", price_col]
    missing = [column for column in required if column not in result]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    long_nav = result["leveraged_fund"].astype(float)
    inverse_nav = result["inverse_leveraged_fund"].astype(float)
    pair_value = result["pair_value"].astype(float)
    execution_mode = result["execution_mode"].iloc[0] if "execution_mode" in result else "pair"
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    ax1.plot(result.index, long_nav, label="Synthetic +L Fund NAV", linewidth=1.5, color="tab:blue")
    ax1.plot(result.index, inverse_nav, label="Synthetic -L Fund NAV", linewidth=1.5, linestyle="--", color="tab:orange")
    ax1.plot(result.index, pair_value, label="Equal-Dollar Pair Value", linewidth=2, color="tab:green")
    ax1.axhline(1.0, linestyle=":", color="gray", linewidth=1.2, label="Initial Fund / Pair Value")
    ax1.set_ylabel("Normalized Fund Value")
    ax1.grid(True, alpha=0.3)

    if execution_mode == "minimum_margin":
        ax1.set_title("Synthetic Fund Paths and Minimum-Margin Hedge")
        hedge_axis = ax1.twinx()
        hedge_axis.plot(
            result.index,
            result["hedge_notional"].astype(float),
            label="Underlying Hedge Notional",
            linewidth=1.2,
            color="tab:purple",
            alpha=0.8,
        )
        if "hedge_rebalance" in result and result["hedge_rebalance"].any():
            rebalance = result["hedge_rebalance"].to_numpy(dtype=bool)
            hedge_axis.scatter(
                result.index[rebalance],
                result["hedge_notional"].to_numpy(dtype=float)[rebalance],
                label="Hedge Rebalance",
                s=10,
                color="tab:red",
                alpha=0.7,
            )
        hedge_axis.set_ylabel("Underlying Hedge Notional ($)")
        lines, labels = ax1.get_legend_handles_labels()
        hedge_lines, hedge_labels = hedge_axis.get_legend_handles_labels()
        ax1.legend(lines + hedge_lines, labels + hedge_labels, loc="upper left")
    else:
        ax1.set_title("Synthetic +L / -L Fund Paths and Pair Value")
        ax1.legend(loc="upper left")
    ax2.plot(result.index, result[price_col].astype(float), color="tab:purple", linewidth=1.5, label="Underlying Asset Price")
    ax2.set_title("Underlying Price Movement")
    ax2.set_ylabel("Price ($)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax3.plot(result.index, result["cumulative_return"].astype(float) * 100.0, color="tab:blue", linewidth=2, label="Cumulative Return")
    ax3.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_title("Cumulative Strategy Return")
    ax3.set_ylabel("Return (%)")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
