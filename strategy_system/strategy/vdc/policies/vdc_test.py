import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_vdc(df, initial_capital=1.0, price_col="price"):
    """
    Plot:
    1. Synthetic leveraged-fund pair notional
    2. Underlying asset price movement
    3. Cumulative strategy return (%)

    IMPORTANT:
    This does NOT represent actual broker margin requirements.
    It shows the synthetic gross notional being carried:
        |leveraged fund| + |inverse leveraged fund|

    Actual margin depends on the broker/exchange and instrument.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = [
        "leveraged_fund",
        "inverse_leveraged_fund",
        "hedge_notional",
        "cumulative_return",
        "strategy_nav",
        price_col,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # -----------------------------
    # Synthetic notionals
    # -----------------------------
    fund_notional = (
        0.5
        * df["leveraged_fund"].abs().astype(float)
        * initial_capital
    )

    inverse_fund_notional = (
        0.5
        * df["inverse_leveraged_fund"].abs().astype(float)
        * initial_capital
    )

    # Total synthetic gross notional being maintained
    gross_notional = (
        fund_notional
        + inverse_fund_notional
    )

    # -----------------------------
    # Cumulative return
    # -----------------------------
    cumulative_return_pct = (
        df["cumulative_return"].astype(float) * 100.0
    )

    # -----------------------------
    # Plot (3 Subplots)
    # -----------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True
    )

    # ===== Plot 1: Leveraged-Fund Pair / Notional =====
    ax1.plot(
        df.index,
        fund_notional,
        label="Leveraged Fund Short Notional",
        linewidth=1.8,
        alpha=0.85
    )

    ax1.plot(
        df.index,
        inverse_fund_notional,
        label="Inverse Leveraged Fund Short Notional",
        linewidth=1.8,
        linestyle="--",
        alpha=0.85
    )

    ax1.plot(
        df.index,
        gross_notional,
        label="Total Synthetic Gross Notional",
        linewidth=2,
        color="tab:green"
    )

    ax1.axhline(
        initial_capital,
        linestyle=":",
        color="gray",
        linewidth=1.2,
        label="Initial Capital"
    )

    ax1.set_title("Synthetic Leveraged-Fund Pair / Notional")
    ax1.set_ylabel("Notional ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # ===== Plot 2: Underlying Price =====
    ax2.plot(
        df.index,
        df[price_col].astype(float),
        color="tab:purple",
        linewidth=1.5,
        label="Underlying Asset Price"
    )

    ax2.set_title("Underlying Asset Price Movement")
    ax2.set_ylabel("Price ($)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # ===== Plot 3: Strategy Return =====
    ax3.plot(
        df.index,
        cumulative_return_pct,
        color="tab:blue",
        linewidth=2,
        label="Cumulative Return"
    )

    ax3.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7
    )

    ax3.set_title("Cumulative Strategy Return")
    ax3.set_ylabel("Return (%)")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def _summarize_vdc(result, summary, initial_capital=1.0, h_interval=1):

    if not isinstance(result, pd.DataFrame):
        raise TypeError("result must be a pandas DataFrame")

    nav = result["strategy_nav"].astype(float)

    # ---------------------------------------------------------
    # Final P&L / Return
    # ---------------------------------------------------------

    final_pnl = float(result["cumulative_pnl"].iloc[-1])
    final_nav = float(nav.iloc[-1])

    final_return_pct = (
        final_pnl / initial_capital
    ) * 100.0

    # ---------------------------------------------------------
    # Maximum Drawdown
    # ---------------------------------------------------------

    running_peak = nav.cummax()
    drawdown = (nav / running_peak) - 1.0

    mdd_pct = float(drawdown.min() * 100.0)

    # ---------------------------------------------------------
    # Hedge-period returns
    # ---------------------------------------------------------

    pnl = result["net_pnl"].astype(float).to_numpy()

    hedge_returns = []

    for start in range(1, len(pnl), h_interval):
        end = min(start + h_interval, len(pnl))

        period_pnl = pnl[start:end].sum()

        hedge_returns.append(
            period_pnl / initial_capital
        )

    hedge_returns = np.asarray(hedge_returns)

    avg_return_pct_per_hedge = (
        np.mean(hedge_returns) * 100.0
        if len(hedge_returns) > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # Sortino
    # ---------------------------------------------------------

    downside = np.minimum(hedge_returns, 0.0)

    downside_deviation = (
        np.sqrt(np.mean(downside ** 2))
        if len(hedge_returns) > 0
        else np.nan
    )

    if np.isfinite(downside_deviation) and downside_deviation > 0:
        sortino = (
            np.mean(hedge_returns)
            / downside_deviation
        )
    elif len(hedge_returns) > 0 and np.mean(hedge_returns) > 0:
        sortino = np.inf
    else:
        sortino = np.nan

    # ---------------------------------------------------------
    # Win rate
    # ---------------------------------------------------------

    win_rate_pct = (
        np.mean(hedge_returns > 0) * 100.0
        if len(hedge_returns) > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # Hedge turnover
    # ---------------------------------------------------------

    total_hedge_turnover = float(
        result["hedge_trade_notional"].abs().sum()
    )

    num_hedge_periods = len(hedge_returns)

    avg_hedge_turnover_per_period = (
        total_hedge_turnover / num_hedge_periods
        if num_hedge_periods > 0
        else np.nan
    )

    # Turnover relative to capital
    turnover_to_capital = (
        total_hedge_turnover / initial_capital
    )

    # ---------------------------------------------------------
    # Build clean report
    # ---------------------------------------------------------

    report = {
        # Original summary
        **{
            k: (
                v.item()
                if isinstance(v, np.generic)
                else v
            )
            for k, v in summary.items()
        },

        # P&L
        "final_pnl": final_pnl,
        "final_nav": final_nav,
        "final_return_pct": final_return_pct,

        # Hedge performance
        "avg_return_pct_per_hedge":
            avg_return_pct_per_hedge,

        "num_hedge_periods":
            num_hedge_periods,

        "win_rate_pct":
            win_rate_pct,

        # Risk
        "mdd_pct":
            mdd_pct,

        "sortino":
            float(sortino),

        "downside_deviation":
            float(downside_deviation),

        # Costs
        "total_transaction_cost":
            float(result["transaction_cost"].sum()),

        # Turnover
        "total_hedge_turnover":
            total_hedge_turnover,

        "avg_hedge_turnover_per_period":
            float(avg_hedge_turnover_per_period),

        "turnover_to_capital":
            float(turnover_to_capital),
    }

    return report


def _output_report(report):

    display_report = {
        "Initial Capital":
            report["initial_capital"],

        "Final NAV":
            report["final_nav"],

        "Final P&L":
            report["final_pnl"],

        "Final Return %":
            report["final_return_pct"],

        "Avg Return / Hedge %":
            report["avg_return_pct_per_hedge"],

        "Max Drawdown %":
            report["mdd_pct"],

        "Sortino":
            report["sortino"],

        "Win Rate %":
            report["win_rate_pct"],

        "Total Hedge Turnover":
            report["total_hedge_turnover"],

        "Avg Turnover / Hedge":
            report["avg_hedge_turnover_per_period"],

        "Turnover / Capital":
            report["turnover_to_capital"],

        "Transaction Cost":
            report["total_transaction_cost"],

        "Hedge Periods":
            report["num_hedge_periods"],
    }

    report_table = pd.DataFrame(
        display_report.items(),
        columns=["Metric", "Value"],
    )

    print("\n" + "=" * 50)
    print("VOLATILITY DECAY CAPTURE REPORT")
    print("=" * 50)

    print(
        report_table.to_string(
            index=False,
            formatters={
                "Value": lambda x: (
                    f"{x:.6f}"
                    if isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    else str(x)
                )
            },
        )
    )

    print("=" * 50)


def _construct_leveraged_funds(
        df,
        price_column=["mid_price"],
        h_lev=3,
    ):
    """
    Construct the synthetic leveraged-fund pair.

    The two funds are reset every observation:

        Leveraged Fund:
            F[t+1] = F[t] * (1 + L * r[t+1])

        Inverse Leveraged Fund:
            G[t+1] = G[t] * (1 - L * r[t+1])

    Both start at 1.0.

    This intentionally models an intraday-reset leveraged product.
    It is therefore NOT a daily-reset ETF replication.

    Returns
    -------
    x : pd.DataFrame
        Underlying price plus both synthetic leveraged-fund paths.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if isinstance(price_column, (list, tuple)):
        if len(price_column) != 1:
            raise ValueError(
                "price_column must contain exactly one column"
            )
        price_column = price_column[0]

    if price_column not in df.columns:
        raise ValueError(
            f"{price_column!r} not found in df"
        )

    if h_lev <= 0:
        raise ValueError(
            "h_lev must be > 0"
        )

    # ---------------------------------------------------------
    # Reformating
    # ---------------------------------------------------------

    x = df[[price_column]].copy()

    x = x.rename(
        columns={
            price_column: "price"
        }
    )

    x["price"] = pd.to_numeric(
        x["price"],
        errors="coerce"
    )

    x = x.dropna()

    if len(x) < 2:
        raise ValueError(
            "Need at least 2 valid price observations"
        )

    if (x["price"] <= 0).any():
        raise ValueError(
            "Prices must be > 0"
        )

    x = x.sort_index()

    # ---------------------------------------------------------
    # Underlying returns
    # ---------------------------------------------------------

    x["underlying_ret"] = (
        x["price"].pct_change()
    )

    # ---------------------------------------------------------
    # Construct leveraged-fund pair
    #
    # At each observation:
    #
    # Leveraged fund:
    #   F[t+1] = F[t] * (1 + L * r[t+1])
    #
    # Inverse leveraged fund:
    #   G[t+1] = G[t] * (1 - L * r[t+1])
    #
    # These are the synthetic +L and -L products.
    #
    # This is intentionally reset every row because the research
    # objective is to isolate intraday volatility decay.
    # ---------------------------------------------------------

    leveraged_fund = np.full(
        len(x),
        np.nan
    )

    inverse_leveraged_fund = np.full(
        len(x),
        np.nan
    )

    leveraged_fund[0] = 1.0
    inverse_leveraged_fund[0] = 1.0

    for i in range(1, len(x)):

        r = x["underlying_ret"].iloc[i]

        leveraged_fund[i] = (
            leveraged_fund[i - 1]
            * (1.0 + h_lev * r)
        )

        inverse_leveraged_fund[i] = (
            inverse_leveraged_fund[i - 1]
            * (1.0 - h_lev * r)
        )

    # ---------------------------------------------------------
    # Leveraged funds can mathematically hit <= 0 if an
    # underlying interval move is too large for the chosen
    # leverage.
    # ---------------------------------------------------------

    if (leveraged_fund <= 0).any():
        raise ValueError(
            "Leveraged fund became <= 0. "
            "The simple L*r reset model breaks for an interval "
            f"move <= {-1 / h_lev:.4%}."
        )

    if (inverse_leveraged_fund <= 0).any():
        raise ValueError(
            "Inverse leveraged fund became <= 0. "
            "The simple -L*r reset model breaks for an interval "
            f"move >= {1 / h_lev:.4%}."
        )

    x["leveraged_fund"] = leveraged_fund
    x["inverse_leveraged_fund"] = inverse_leveraged_fund

    # ---------------------------------------------------------
    # Pair value
    #
    # Equal-dollar pair:
    #
    #   0.5 * Leveraged Fund
    # + 0.5 * Inverse Leveraged Fund
    #
    # The pair starts at 1.0.
    # Volatility causes the pair to decay below 1.0.
    # ---------------------------------------------------------

    x["pair_value"] = (
        0.5 * x["leveraged_fund"]
        + 0.5 * x["inverse_leveraged_fund"]
    )

    # Difference / divergence between the two funds.
    x["fund_divergence"] = (
        x["leveraged_fund"]
        - x["inverse_leveraged_fund"]
    )

    return x
def _legacy_volatility_decay_capture_test(

        df,

        price_column=["mid_price"],

        h_interval=300,

        h_lev=1,

        friction_bps=4,

        position=-1,

        report_output=False,

        execution_mode="minimum_margin",

    ):

    """
    vdc computes leveraged fund calculation to capture volatility decay.
    H1: Does leveraged fund exhibit consistent, negative expectancy?
    H2: Can orderflow, or other features be used to predict volatility
    crashes before strategy fails?

    ===================================================================

    Params
    ----------
        h_interval: Hedging per unit time.
        h_lev: hedging leverage aggressiveness (ex: 3x).
        friction_bps: total in-and-out, assumes 0.04%.
        execution_mode:
            "pair"           = directly hold the +L / -L leveraged-fund pair.
            "minimum_margin" = dynamically replicate the pair using the
                               minimum underlying delta exposure.

    Metrics
    ----------
        Average move%/hedging interval: If -move% > friction, edge is proven.
        Neg-targeted Sortino ratio    : higher is better.
        MDD / Freq_rate               : Lower is better.

    Assumptions
    -----------
        - price_column is the underlying mid-price.
        - h_lev is the leveraged-fund target leverage.
        - Leveraged and inverse leveraged funds are reset every observation.
        - The strategy shorts the equal-dollar leveraged-fund pair.
        - In minimum_margin mode, the pair is dynamically replicated using
          the underlying delta:
              delta = L * (F - G) / (2 * S)
        - Minimum-margin replication uses only information available at t-1
          for the P&L from t-1 -> t.
        - friction_bps is charged on actual hedge turnover.
        - Initial capital = 1.0.
        - Initial synthetic short-pair notional = 1.0.

    Returns
    -------
    result : pd.DataFrame
        Full path of the simulation.
    summary : dict
        Final cumulative return and basic accounting quantities.
    """

    # ---------------------------------------------------------
    # Validation
    # - Ensure all parameters entered fulfills the requirement.
    # ---------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if position not in (-1, 1):
        raise ValueError("position must be either -1 or +1")

    if isinstance(price_column, (list, tuple)):
        if len(price_column) != 1:
            raise ValueError("price_column must contain exactly one column")
        price_column = price_column[0]

    if h_interval <= 0:
        raise ValueError("h_interval must be > 0")

    if h_lev <= 0:
        raise ValueError("h_lev must be > 0")

    if friction_bps < 0:
        raise ValueError("friction_bps must be >= 0")

    if execution_mode not in ("pair", "minimum_margin"):
        raise ValueError(
            "execution_mode must be either "
            "'pair' or 'minimum_margin'"
        )

    # ---------------------------------------------------------
    # Reformatting
    # ---------------------------------------------------------

    x = df[[price_column]].copy()

    x = x.rename(
        columns={
            price_column: "price"
        }
    )

    x["price"] = pd.to_numeric(
        x["price"],
        errors="coerce"
    )

    x = x.dropna()

    if len(x) < 2:
        raise ValueError(
            "Need at least 2 valid price observations"
        )

    if (x["price"] <= 0).any():
        raise ValueError(
            "Prices must be > 0"
        )

    x = x.sort_index()

    # ---------------------------------------------------------
    # 2. Threshold Filter (Remove Sub-Tick HFT Micro-Flicker)
    #
    # Ignores price moves smaller than min_change_pct (e.g., 0.02% or 2 bps)
    # relative to the LAST ACCEPTED PRICE level to prevent artificial
    # volatility decay compounding on bid-ask noise.
    # ---------------------------------------------------------

    min_change_pct = 0.0002  # 0.02% (2 bps). Adjust based on asset tick size.

    raw_prices = x["price"].to_numpy()

    filtered_prices = np.empty_like(raw_prices)

    filtered_prices[0] = raw_prices[0]

    last_valid_price = raw_prices[0]

    for i in range(1, len(raw_prices)):

        current_p = raw_prices[i]

        rel_change = (
            abs(current_p - last_valid_price)
            / last_valid_price
        )

        if rel_change >= min_change_pct:
            last_valid_price = current_p

        filtered_prices[i] = last_valid_price

    x["price"] = filtered_prices

    # ---------------------------------------------------------
    # Underlying returns (Calculated on cleaned price path)
    # ---------------------------------------------------------

    x["underlying_ret"] = (
        x["price"].pct_change()
    )

    # ---------------------------------------------------------
    # Construct leveraged-fund pair
    #
    # At each observation:
    #
    # Leveraged fund:
    #   F[t+1] = F[t] * (1 + L * r[t+1])
    #
    # Inverse leveraged fund:
    #   G[t+1] = G[t] * (1 - L * r[t+1])
    #
    # These are the synthetic +L and -L products.
    #
    # This is intentionally reset every row because the research
    # objective is to isolate intraday volatility decay.
    # ---------------------------------------------------------

    leveraged_fund = np.full(
        len(x),
        np.nan
    )

    inverse_leveraged_fund = np.full(
        len(x),
        np.nan
    )

    leveraged_fund[0] = 1.0
    inverse_leveraged_fund[0] = 1.0

    for i in range(1, len(x)):

        r = x["underlying_ret"].iloc[i]

        leveraged_fund[i] = (
            leveraged_fund[i - 1]
            * (1.0 + h_lev * r)
        )

        inverse_leveraged_fund[i] = (
            inverse_leveraged_fund[i - 1]
            * (1.0 - h_lev * r)
        )

    # ---------------------------------------------------------
    # Leveraged funds can mathematically hit <= 0 if an
    # underlying interval move is too large for the chosen
    # leverage.
    # ---------------------------------------------------------

    if (leveraged_fund <= 0).any():
        raise ValueError(
            "Leveraged fund became <= 0. "
            "The simple L*r reset model breaks for an interval "
            f"move <= {-1 / h_lev:.4%}."
        )

    if (inverse_leveraged_fund <= 0).any():
        raise ValueError(
            "Inverse leveraged fund became <= 0. "
            "The simple -L*r reset model breaks for an interval "
            f"move >= {1 / h_lev:.4%}."
        )

    x["leveraged_fund"] = leveraged_fund
    x["inverse_leveraged_fund"] = inverse_leveraged_fund

    # ---------------------------------------------------------
    # Pair value
    #
    # Equal-dollar pair:
    #
    #   0.5 * Leveraged Fund
    # + 0.5 * Inverse Leveraged Fund
    #
    # The pair starts at 1.0.
    # Volatility causes the pair to decay below 1.0.
    # ---------------------------------------------------------

    x["pair_value"] = (
        0.5 * x["leveraged_fund"]
        + 0.5 * x["inverse_leveraged_fund"]
    )

    # Difference / divergence between the two funds.

    x["fund_divergence"] = (
        x["leveraged_fund"]
        - x["inverse_leveraged_fund"]
    )

    # ---------------------------------------------------------
    # Initial capital / reference notional
    # ---------------------------------------------------------

    initial_capital = 1.0

    # ---------------------------------------------------------
    # Portfolio P&L
    #
    # There are 2 ways to implement the same economic exposure:
    #
    # 1. "pair":
    #       Short 0.5 leveraged fund
    #       Short 0.5 inverse leveraged fund
    #
    # 2. "minimum_margin":
    #       Dynamically trade only the underlying delta of the pair.
    #
    # For zero costs and one-bar rehedging, both produce the same
    # P&L mathematically.
    # ---------------------------------------------------------

    fund_pnl = np.zeros(len(x))
    hedge_pnl = np.zeros(len(x))
    net_pnl = np.zeros(len(x))

    cumulative_pnl = np.zeros(len(x))
    cumulative_return = np.zeros(len(x))
    strategy_nav = np.zeros(len(x))

    # ---------------------------------------------------------
    # Pair / hedge accounting
    # ---------------------------------------------------------

    hedge_units = np.zeros(len(x))
    hedge_notional = np.zeros(len(x))
    hedge_trade_units = np.zeros(len(x))
    hedge_trade_notional = np.zeros(len(x))
    transaction_cost = np.zeros(len(x))

    # ---------------------------------------------------------
    # Initial position
    # ---------------------------------------------------------

    previous_hedge_units = 0.0

    # ---------------------------------------------------------
    # Initial pair entry
    #
    # The normalized pair consists of:
    #
    #     0.5 leveraged fund
    #     0.5 inverse leveraged fund
    #
    # Total pair notional = 1.0.
    #
    # In minimum_margin mode, the initial pair delta is zero because
    # both funds start at 1.0.
    # ---------------------------------------------------------

    if execution_mode == "pair":

        initial_trade_notional = (
            initial_capital
        )

        transaction_cost[0] = (
            initial_trade_notional
            * friction_bps
            / 10_000.0
        )

    else:

        initial_delta_units = (
            position
            * 0.5
            * h_lev
            * (
                leveraged_fund[0]
                - inverse_leveraged_fund[0]
            )
            / x["price"].iloc[0]
        )

        hedge_units[0] = (
            initial_delta_units
        )

        hedge_notional[0] = (
            abs(initial_delta_units)
            * x["price"].iloc[0]
        )

        hedge_trade_units[0] = (
            initial_delta_units
        )

        hedge_trade_notional[0] = (
            abs(initial_delta_units)
            * x["price"].iloc[0]
        )

        transaction_cost[0] = (
            hedge_trade_notional[0]
            * friction_bps
            / 10_000.0
        )

        previous_hedge_units = (
            initial_delta_units
        )

    # ---------------------------------------------------------
    # Start NAV after paying initial transaction cost.
    # ---------------------------------------------------------

    strategy_nav[0] = (
        initial_capital
        - transaction_cost[0]
    )

    cumulative_pnl[0] = (
        -transaction_cost[0]
    )

    cumulative_return[0] = (
        strategy_nav[0]
        / initial_capital
    ) - 1.0

    # ---------------------------------------------------------
    # Continuous calculation at t=(1...n)
    #
    # Utilize for loop to compute previous, stateful hedge calculation.
    # Vectorization cannot be used for the stateful minimum-margin
    # implementation.
    # ---------------------------------------------------------

    for i in range(1, len(x)):

        current_price = (
            x["price"].iloc[i]
        )

        previous_price = (
            x["price"].iloc[i - 1]
        )

        previous_fund = (
            leveraged_fund[i - 1]
        )

        current_fund = (
            leveraged_fund[i]
        )

        previous_inverse_fund = (
            inverse_leveraged_fund[i - 1]
        )

        current_inverse_fund = (
            inverse_leveraged_fund[i]
        )

        # -----------------------------------------------------
        # 1. P&L during t-1 -> t
        #
        # Pair implementation:
        #
        #   -(0.5 * ΔF)
        #   -(0.5 * ΔG)
        #
        # Minimum-margin implementation:
        #
        #   previous_delta_units * ΔS
        #
        # The minimum-margin position uses only t-1 information.
        # Hence ZERO lookahead.
        # -----------------------------------------------------

        if execution_mode == "pair":

            fund_pnl[i] = (
                position
                * 0.5
                * (
                    current_fund
                    - previous_fund
                )
            )

            hedge_pnl[i] = (
                position
                * 0.5
                * (
                    current_inverse_fund
                    - previous_inverse_fund
                )
            )

            gross_interval_pnl = (
                fund_pnl[i]
                + hedge_pnl[i]
            )

        else:

            gross_interval_pnl = (
                previous_hedge_units
                * (
                    current_price
                    - previous_price
                )
            )

            hedge_pnl[i] = (
                gross_interval_pnl
            )

        # -----------------------------------------------------
        # 2. Decide whether to rebalance
        #
        # Here h_interval is interpreted as NUMBER OF ROWS.
        #
        # If your dataframe is 1-second data, h_interval=60
        # means 60 seconds.
        #
        # If your dataframe is already 1-minute data, h_interval=60
        # means 60 minutes.
        # -----------------------------------------------------

        if execution_mode == "pair":

            # The pair itself does not need rebalancing.

            hedge_units[i] = 0.0
            hedge_notional[i] = 0.0
            hedge_trade_units[i] = 0.0
            hedge_trade_notional[i] = 0.0
            transaction_cost[i] = 0.0

        else:

            rebalance = (
                i % h_interval == 0
            )

            if rebalance:

                # -------------------------------------------------
                # Target hedge calculated using current information.
                #
                # This is deliberately AFTER P&L for the interval
                # has been computed.
                #
                # Hence ZERO lookahead.
                #
                # Pair delta:
                #
                #   L * (F - G) / (2S)
                # -------------------------------------------------

                target_hedge_units = (
                    position
                    * 0.5
                    * h_lev
                    * (
                        current_fund
                        - current_inverse_fund
                    )
                    / current_price
                )

                hedge_units[i] = (
                    target_hedge_units
                )

                hedge_notional[i] = (
                    abs(target_hedge_units)
                    * current_price
                )

                trade_units = (
                    target_hedge_units
                    - previous_hedge_units
                )

                trade_notional = (
                    abs(trade_units)
                    * current_price
                )

                cost = (
                    trade_notional
                    * friction_bps
                    / 10_000.0
                )

                hedge_trade_units[i] = (
                    trade_units
                )

                hedge_trade_notional[i] = (
                    trade_notional
                )

                transaction_cost[i] = (
                    cost
                )

                previous_hedge_units = (
                    target_hedge_units
                )

            else:

                # Carry existing hedge forward.

                hedge_units[i] = (
                    previous_hedge_units
                )

                hedge_notional[i] = (
                    abs(previous_hedge_units)
                    * current_price
                )

                hedge_trade_units[i] = 0.0
                hedge_trade_notional[i] = 0.0
                transaction_cost[i] = 0.0

        # ---------------------------------------------------------
        # Net P&L after transaction cost
        # ---------------------------------------------------------

        net_pnl[i] = (
            gross_interval_pnl
            - transaction_cost[i]
        )

        cumulative_pnl[i] = (
            cumulative_pnl[i - 1]
            + net_pnl[i]
        )

        strategy_nav[i] = (
            initial_capital
            + cumulative_pnl[i]
        )

        cumulative_return[i] = (
            strategy_nav[i]
            / initial_capital
        ) - 1.0

    # ---------------------------------------------------------
    # Add results
    # ---------------------------------------------------------

    x["hedge_units"] = hedge_units
    x["hedge_notional"] = hedge_notional
    x["hedge_trade_units"] = hedge_trade_units
    x["hedge_trade_notional"] = hedge_trade_notional

    x["fund_pnl"] = fund_pnl
    x["hedge_pnl"] = hedge_pnl

    x["gross_pnl"] = (
        fund_pnl + hedge_pnl
        if execution_mode == "pair"
        else x["hedge_pnl"]
    )

    x["transaction_cost"] = (
        transaction_cost
    )

    x["net_pnl"] = net_pnl

    x["cumulative_pnl"] = (
        cumulative_pnl
    )

    x["strategy_nav"] = (
        strategy_nav
    )

    x["cumulative_return"] = (
        cumulative_return
    )

    # ---------------------------------------------------------
    # Basic summary
    # ---------------------------------------------------------

    summary = {

        "initial_capital":
            initial_capital,

        "final_cumulative_pnl":
            cumulative_pnl[-1],

        "final_cumulative_return":
            cumulative_return[-1],

        "leveraged_fund_return": (
            leveraged_fund[-1]
            / leveraged_fund[0]
            - 1.0
        ),

        "underlying_return": (
            x["price"].iloc[-1]
            / x["price"].iloc[0]
            - 1.0
        ),

        "gross_pnl":
            np.sum(x["gross_pnl"]),

        "transaction_cost":
            np.sum(transaction_cost),

        "net_pnl":
            np.sum(net_pnl),

        "total_hedge_turnover":
            np.sum(hedge_trade_notional),

        "num_rebalances":
            int(
                np.count_nonzero(
                    hedge_trade_notional
                )
            ),

        "final_pair_value":
            float(
                x["pair_value"].iloc[-1]
            ),

        "final_fund_divergence":
            float(
                x["fund_divergence"].iloc[-1]
            ),

        "execution_mode":
            execution_mode,
    }

    if report_output:

        report = _summarize_vdc(
            x,
            summary,
            initial_capital=initial_capital,
            h_interval=h_interval,
        )

        _output_report(report)

    return x, summary

def test_vdc_mathematical_equivalence(
        df,
        price_column=["mid_price"],
        h_lev=3,
    ):
    """
    Test whether:

        1. Short +L / -L leveraged-fund pair

    and

        2. Minimum-margin dynamic underlying replication

    produce the same mathematical P&L.

    Transaction costs are deliberately set to zero.
    Rehedging is done every observation.

    The two paths should be identical up to floating-point precision.
    """

    pair_result, pair_summary = (
        volatility_decay_capture_test(
            df=df,
            price_column=price_column,
            h_interval=1,
            h_lev=h_lev,
            friction_bps=0,
            position=-1,
            report_output=False,
            execution_mode="pair",
        )
    )

    minimum_margin_result, minimum_margin_summary = (
        volatility_decay_capture_test(
            df=df,
            price_column=price_column,
            h_interval=1,
            h_lev=h_lev,
            friction_bps=0,
            position=-1,
            report_output=False,
            execution_mode="minimum_margin",
        )
    )

    pnl_difference = (
        pair_result["cumulative_pnl"]
        - minimum_margin_result["cumulative_pnl"]
    )

    max_abs_difference = float(
        np.max(np.abs(pnl_difference))
    )

    final_difference = float(
        pnl_difference.iloc[-1]
    )

    mathematically_equivalent = bool(
        np.allclose(
            pair_result["cumulative_pnl"].to_numpy(),
            minimum_margin_result["cumulative_pnl"].to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    print("\n" + "=" * 60)
    print("VDC MATHEMATICAL EQUIVALENCE TEST")
    print("=" * 60)

    print(
        f"Mathematically equivalent: "
        f"{mathematically_equivalent}"
    )

    print(
        f"Max absolute P&L difference: "
        f"{max_abs_difference:.12e}"
    )

    print(
        f"Final P&L difference: "
        f"{final_difference:.12e}"
    )

    print("=" * 60)

    return {
        "mathematically_equivalent":
            mathematically_equivalent,
        "max_abs_pnl_difference":
            max_abs_difference,
        "final_pnl_difference":
            final_difference,
        "pair_summary":
            pair_summary,
        "minimum_margin_summary":
            minimum_margin_summary,
    }


# Keep the original research import path working while routing callers to the
# time-based, modular implementation in strategy_system.strategy.vdc.
def volatility_decay_capture_test(
        df,
        price_column=["mid_price"],
        h_interval=300,
        h_lev=1,
        friction_bps=4,
        position=-1,
        report_output=False,
        execution_mode="minimum_margin",
        min_change_pct=0.0002,
        timestamp_column=None,
        sessionize=False,
        hedge_interval=None,
        initial_capital=1.0,
        plot_output=False,
    ):
    from strategy_system.strategy.vdc.policies.vdc.vdc_runner import run_vdc_strategy

    return run_vdc_strategy(
        df=df,
        price_column=price_column,
        h_interval=h_interval,
        h_lev=h_lev,
        friction_bps=friction_bps,
        position=position,
        execution_mode=execution_mode,
        min_change_pct=min_change_pct,
        timestamp_column=timestamp_column,
        sessionize=sessionize,
        hedge_interval=hedge_interval,
        initial_capital=initial_capital,
        report_output=report_output,
        plot_output=plot_output,
    )
