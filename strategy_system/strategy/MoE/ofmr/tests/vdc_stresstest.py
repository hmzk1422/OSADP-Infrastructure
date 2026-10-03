import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from strategy_system.strategy.MoE.ofmr.hypothesis.vdc.vdc_runner import (
    run_vdc_strategy,
)
from components.pipelines.ingestion.CryptoL1.price import get_ohlcv
from components.pipelines.ingestion.synthetic.synth2 import simulate_market_stochastic 

def vdc_monte_carlo(result_df, n_paths=40):
    strategy_paths = []
    final_returns = []
    invalid_paths = []

    for seed in tqdm(range(n_paths), desc="Simulating Monte Carlo Paths"):
        # 1. Generate stochastic market path
        df_path = simulate_market_stochastic(
            n_bars=5000,
            n_features=50,
            n_states=4,
            seed=seed,
            feature_noise=0.8,
            initial_price=100,
        )

        # Ensure DataFrame has a DatetimeIndex (required for sessionize=True)
        if not isinstance(df_path.index, pd.DatetimeIndex):
            df_path.index = pd.date_range("2026-01-01 00:00", periods=len(df_path), freq="1s", tz="UTC")

        # 2. Run VDC strategy per path. A path whose interval return violates
        # 1 +/- L*r > 0 is mathematically undefined for this fund model, so
        # reject it instead of clipping or fabricating a fund value.
        try:
            result_df, summary = run_vdc_strategy(
                df=df_path,
                position=1,
                price_column="price",   # Matches df_path["price"]
                h_interval=1,         # 1-second fund reset / hedge interval
                h_lev=3.0,              # 3x target leverage
                friction_bps=0.0,       # Ignore costs for pair/hedge parity check
                execution_mode="minimum_margin",  # Underlying-only replication
                min_change_pct=0,       # 0 noise filter
                sessionize=True,        # Reset fund state daily
                report_output=False,    # Disable individual report prints
                plot_output=False,     # Keep Monte Carlo to its aggregate plot
            )
        except ValueError as exc:
            if "A leveraged fund became <= 0" not in str(exc):
                raise
            invalid_paths.append({"seed": seed, "reason": str(exc)})
            continue

        # 3. Store path cumulative return trajectory (%)
        cum_ret = result_df["cumulative_return"].to_numpy() * 100.0
        strategy_paths.append(cum_ret)
        final_returns.append(summary["final_cumulative_return"] * 100.0)

    strategy_paths = np.array(strategy_paths)
    final_returns = np.array(final_returns)

    if len(final_returns) == 0:
        raise RuntimeError(
            "All Monte Carlo paths were invalid for the selected VDC interval "
            "and leverage. No research P&L was produced."
        )

    # 4. Calculate how many ended up on top (profitable)
    profitable_count = int(np.sum(final_returns > 0))
    unprofitable_count = len(final_returns) - profitable_count
    win_rate = (profitable_count / len(final_returns)) * 100.0

    # 5. Plot Monte Carlo Results
    plt.figure(figsize=(14, 7))

    for i, path in enumerate(strategy_paths):
        color = "tab:green" if final_returns[i] > 0 else "tab:red"
        plt.plot(path, alpha=0.25, color=color, linewidth=1)

    # Highlight breakeven baseline
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Breakeven (0%)")

    plt.xlabel("Bar")
    plt.ylabel("Cumulative Return (%)")
    plt.title(
        f"VDC Strategy Monte Carlo Simulation ({len(final_returns)} Valid Paths)\n"
        f"Profitable Paths: {profitable_count}/{len(final_returns)} ({win_rate:.1f}%) | "
        f"Unprofitable Paths: {unprofitable_count}/{len(final_returns)} | "
        f"Rejected: {len(invalid_paths)}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Print summary metrics
    print("=" * 50)
    print("MONTE CARLO SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total Paths Simulated : {n_paths}")
    print(f"Valid Paths           : {len(final_returns)}")
    print(f"Rejected Paths        : {len(invalid_paths)}")
    print(f"Profitable Paths      : {profitable_count} ({win_rate:.1f}%)")
    print(f"Average Final Return  : {np.mean(final_returns):.2f}%")
    print(f"Median Final Return   : {np.median(final_returns):.2f}%")
    print(f"Best Path Return      : {np.max(final_returns):.2f}%")
    print(f"Worst Path Return     : {np.min(final_returns):.2f}%")
    print("=" * 50)
    if invalid_paths:
        print("Rejected-path example:", invalid_paths[0])

    return {
        "strategy_paths": strategy_paths,
        "final_returns": final_returns,
        "invalid_paths": invalid_paths,
    }
def vdc_asset_test(
    time_slice=1,
    train_days=30,
    purge_days=1,
    test_days=7,
    step_days=7,
    since="2025-01-01T00:00:00Z",
    until="2025-06-01T00:00:00Z",
    timeframe="1m",
    h_interval=1,
    h_lev=3.0,
    friction_bps=0.0,
    position=-1,
    execution_mode="pair",
    min_change_pct=0.0,
    sessionize=True,
    report_output=False,
    plot_output=True,
):
    """
    Test VDC across multiple assets using chronological walk-forward windows.

    Each asset is evaluated independently.

    Walk-forward structure:

        TRAIN
          ↓
        PURGE
          ↓
        TEST
          ↓
        move forward
          ↓
        TRAIN
          ↓
        PURGE
          ↓
        TEST

    VDC itself has no fitted model, so the training period is used only to
    preserve chronological market/fund context before the out-of-sample
    test period. The purge period is excluded from reported test P&L.

    ``time_slice`` controls how many most-recent walk-forward test windows
    are evaluated for each asset.

    Returns
    -------
    dict
        folds:
            every asset × walk-forward test result.

        asset_summary:
            aggregated performance by asset.

        class_summary:
            aggregated performance by asset class.

        overall_summary:
            aggregate statistics across all valid test paths.

        invalid_paths:
            failed/invalid asset-window combinations.
    """

    # =========================================================
    # Asset universe
    #
    # Binance symbols are grouped by broad market role rather
    # than pretending every asset belongs to the same economic
    # class.
    # =========================================================

    assets = {
        "BTCUSDT": "large_cap",
        "ETHUSDT": "large_cap",
        "BNBUSDT": "exchange",
        "SOLUSDT": "layer_1",
        "XRPUSDT": "payments",
        "LINKUSDT": "infrastructure",
        "UNIUSDT": "defi",
        "DOGEUSDT": "meme",
    }

    if time_slice <= 0:
        raise ValueError(
            "time_slice must be > 0"
        )

    if train_days <= 0:
        raise ValueError(
            "train_days must be > 0"
        )

    if purge_days < 0:
        raise ValueError(
            "purge_days must be >= 0"
        )

    if test_days <= 0:
        raise ValueError(
            "test_days must be > 0"
        )

    if step_days <= 0:
        raise ValueError(
            "step_days must be > 0"
        )

    # =========================================================
    # Generate the chronological walk-forward boundaries.
    #
    # The final ``time_slice`` windows are used.
    # =========================================================

    start_time = pd.Timestamp(
        since,
        tz="UTC",
    )

    end_time = pd.Timestamp(
        until,
        tz="UTC",
    )

    train_delta = pd.Timedelta(
        days=train_days
    )

    purge_delta = pd.Timedelta(
        days=purge_days
    )

    test_delta = pd.Timedelta(
        days=test_days
    )

    step_delta = pd.Timedelta(
        days=step_days
    )

    possible_folds = []

    fold_start = start_time

    while True:

        train_start = fold_start

        train_end = (
            train_start
            + train_delta
        )

        purge_start = train_end

        purge_end = (
            purge_start
            + purge_delta
        )

        test_start = purge_end

        test_end = (
            test_start
            + test_delta
        )

        if test_end > end_time:
            break

        possible_folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "purge_start": purge_start,
                "purge_end": purge_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        fold_start += step_delta

    if not possible_folds:
        raise ValueError(
            "No valid walk-forward folds fit inside the requested "
            "date range."
        )

    folds = possible_folds[
        -time_slice:
    ]

    # =========================================================
    # Storage
    # =========================================================

    fold_results = []
    invalid_paths = []
    plotted_paths = []

    # =========================================================
    # Asset loop
    # =========================================================

    total_jobs = (
        len(assets)
        * len(folds)
    )

    with tqdm(
        total=total_jobs,
        desc="VDC Asset Walk-Forward",
    ) as progress:

        for ticker, asset_class in assets.items():

            # =================================================
            # Fetch the asset ONCE.
            #
            # This avoids repeatedly hitting the exchange for
            # overlapping walk-forward windows.
            # =================================================

            try:

                df = get_ohlcv(
                    symbol=ticker,
                    timeframe=timeframe,
                    since=since,
                    until=until,
                    exchange_id="binance",
                )

                if not isinstance(
                    df,
                    pd.DataFrame,
                ):
                    raise TypeError(
                        "get_ohlcv did not return a DataFrame"
                    )

                if df.empty:
                    raise ValueError(
                        "get_ohlcv returned an empty DataFrame"
                    )

                # ---------------------------------------------
                # Ensure DatetimeIndex.
                # ---------------------------------------------

                if not isinstance(
                    df.index,
                    pd.DatetimeIndex,
                ):

                    if "timestamp" in df.columns:

                        df.index = pd.to_datetime(
                            df["timestamp"],
                            utc=True,
                        )

                    else:

                        raise TypeError(
                            f"{ticker}: OHLCV data must have a "
                            "DatetimeIndex or timestamp column."
                        )

                else:

                    if df.index.tz is None:

                        df.index = (
                            df.index
                            .tz_localize("UTC")
                        )

                    else:

                        df.index = (
                            df.index
                            .tz_convert("UTC")
                        )

                df = (
                    df
                    .sort_index()
                    .loc[
                        ~df.index.duplicated(
                            keep="last"
                        )
                    ]
                )

            except Exception as exc:

                for fold_number, _ in enumerate(
                    folds,
                    start=1,
                ):

                    invalid_paths.append(
                        {
                            "ticker": ticker,
                            "asset_class": asset_class,
                            "fold": fold_number,
                            "reason": str(exc),
                        }
                    )

                    progress.update(1)

                continue

            # =================================================
            # Walk-forward folds
            # =================================================

            for fold_number, fold in enumerate(
                folds,
                start=1,
            ):

                try:

                    train_start = fold[
                        "train_start"
                    ]

                    train_end = fold[
                        "train_end"
                    ]

                    purge_start = fold[
                        "purge_start"
                    ]

                    purge_end = fold[
                        "purge_end"
                    ]

                    test_start = fold[
                        "test_start"
                    ]

                    test_end = fold[
                        "test_end"
                    ]

                    # -----------------------------------------
                    # Include train + purge + test in the
                    # simulation so state can evolve naturally.
                    #
                    # Only TEST P&L is reported.
                    # -----------------------------------------

                    window_df = df.loc[
                        train_start:test_end
                    ].copy()

                    if window_df.empty:
                        raise ValueError(
                            "No data in walk-forward window."
                        )

                    test_mask = (
                        (window_df.index >= test_start)
                        & (window_df.index < test_end)
                    )

                    if not test_mask.any():
                        raise ValueError(
                            "No observations in test window."
                        )

                    # -----------------------------------------
                    # Run VDC across the full chronological
                    # train + purge + test window.
                    # -----------------------------------------

                    result, summary = run_vdc_strategy(

                        df=window_df,

                        price_column="close",

                        h_interval=h_interval,

                        h_lev=h_lev,

                        friction_bps=friction_bps,

                        position=position,

                        execution_mode=execution_mode,

                        min_change_pct=min_change_pct,

                        sessionize=sessionize,

                        report_output=False,

                        plot_output=False,
                    )

                    # -----------------------------------------
                    # Locate the point immediately before TEST.
                    #
                    # This lets us isolate test P&L from the
                    # full chronological simulation.
                    # -----------------------------------------

                    test_positions = np.flatnonzero(
                        test_mask
                    )

                    test_start_position = (
                        test_positions[0]
                    )

                    test_end_position = (
                        test_positions[-1]
                    )

                    if test_start_position == 0:

                        base_cumulative_pnl = 0.0

                    else:

                        base_cumulative_pnl = float(
                            result[
                                "cumulative_pnl"
                            ].iloc[
                                test_start_position - 1
                            ]
                        )

                    test_result = result.loc[
                        test_mask
                    ].copy()

                    # -----------------------------------------
                    # Rebase P&L/NAV to beginning of TEST.
                    # -----------------------------------------

                    test_result[
                        "test_cumulative_pnl"
                    ] = (
                        test_result[
                            "cumulative_pnl"
                        ]
                        - base_cumulative_pnl
                    )

                    test_result[
                        "test_strategy_nav"
                    ] = (
                        1.0
                        + test_result[
                            "test_cumulative_pnl"
                        ]
                    )

                    test_result[
                        "test_cumulative_return"
                    ] = (
                        test_result[
                            "test_cumulative_pnl"
                        ]
                    )

                    # -----------------------------------------
                    # Test-period P&L.
                    # -----------------------------------------

                    final_test_pnl = float(
                        test_result[
                            "test_cumulative_pnl"
                        ].iloc[-1]
                    )

                    final_test_return = (
                        final_test_pnl
                    )

                    # -----------------------------------------
                    # Test underlying return.
                    # -----------------------------------------

                    test_price = (
                        test_result[
                            "price"
                        ].astype(float)
                    )

                    underlying_return = (
                        float(
                            test_price.iloc[-1]
                            / test_price.iloc[0]
                            - 1.0
                        )
                    )

                    # -----------------------------------------
                    # Test average absolute underlying move.
                    # -----------------------------------------

                    test_underlying_returns = (
                        test_price
                        .pct_change()
                        .dropna()
                    )

                    average_abs_move = (
                        float(
                            test_underlying_returns.abs().mean()
                            * 100.0
                        )
                        if len(test_underlying_returns)
                        else np.nan
                    )

                    # -----------------------------------------
                    # Test interval return.
                    # -----------------------------------------

                    test_interval_return = (
                        test_result[
                            "net_pnl"
                        ]
                        .astype(float)
                        .sum()
                    )

                    # -----------------------------------------
                    # Test drawdown.
                    # -----------------------------------------

                    test_nav = (
                        test_result[
                            "test_strategy_nav"
                        ]
                        .astype(float)
                    )

                    running_peak = (
                        test_nav.cummax()
                    )

                    drawdown = (
                        test_nav
                        / running_peak
                        - 1.0
                    )

                    mdd_pct = float(
                        drawdown.min()
                        * 100.0
                    )

                    # -----------------------------------------
                    # Test transaction costs.
                    # -----------------------------------------

                    transaction_cost = float(
                        test_result[
                            "transaction_cost"
                        ]
                        .astype(float)
                        .sum()
                    )

                    gross_pnl = float(
                        test_result[
                            "gross_pnl"
                        ]
                        .astype(float)
                        .sum()
                    )

                    net_pnl = float(
                        test_result[
                            "net_pnl"
                        ]
                        .astype(float)
                        .sum()
                    )

                    # -----------------------------------------
                    # Test turnover.
                    # -----------------------------------------

                    execution_turnover = float(
                        test_result[
                            "trade_notional"
                        ]
                        .astype(float)
                        .sum()
                    )

                    # -----------------------------------------
                    # Number of observations.
                    # -----------------------------------------

                    n_test = int(
                        len(test_result)
                    )

                    # -----------------------------------------
                    # Append fold result.
                    # -----------------------------------------

                    fold_record = {

                        "ticker":
                            ticker,

                        "asset_class":
                            asset_class,

                        "fold":
                            fold_number,

                        "train_start":
                            train_start,

                        "train_end":
                            train_end,

                        "purge_start":
                            purge_start,

                        "purge_end":
                            purge_end,

                        "test_start":
                            test_start,

                        "test_end":
                            test_end,

                        "n_test":
                            n_test,

                        "final_return_pct":
                            final_test_return * 100.0,

                        "underlying_return_pct":
                            underlying_return * 100.0,

                        "average_abs_move_pct":
                            average_abs_move,

                        "gross_pnl":
                            gross_pnl,

                        "net_pnl":
                            net_pnl,

                        "transaction_cost":
                            transaction_cost,

                        "execution_turnover":
                            execution_turnover,

                        "mdd_pct":
                            mdd_pct,

                        "final_pair_value":
                            float(
                                test_result[
                                    "pair_value"
                                ].iloc[-1]
                            ),

                        "final_fund_divergence":
                            float(
                                test_result[
                                    "fund_divergence"
                                ].iloc[-1]
                            ),

                        "max_hedge_notional":
                            float(
                                test_result[
                                    "hedge_notional"
                                ]
                                .astype(float)
                                .max()
                            )
                            if "hedge_notional"
                            in test_result.columns
                            else np.nan,
                    }

                    fold_results.append(
                        fold_record
                    )

                    # -----------------------------------------
                    # Save path for aggregate plot.
                    # -----------------------------------------

                    normalized_path = (
                        test_result[
                            "test_cumulative_return"
                        ]
                        .astype(float)
                        .to_numpy()
                        * 100.0
                    )

                    plotted_paths.append(
                        {
                            "ticker": ticker,
                            "asset_class": asset_class,
                            "fold": fold_number,
                            "path": normalized_path,
                            "final_return_pct":
                                final_test_return * 100.0,
                        }
                    )

                except Exception as exc:

                    invalid_paths.append(
                        {
                            "ticker": ticker,
                            "asset_class": asset_class,
                            "fold": fold_number,
                            "reason": str(exc),
                        }
                    )

                finally:

                    progress.update(1)

    # =========================================================
    # Convert results
    # =========================================================

    folds_df = pd.DataFrame(
        fold_results
    )

    if folds_df.empty:

        raise RuntimeError(
            "No valid VDC walk-forward paths were produced."
        )

    # =========================================================
    # Profitable paths
    # =========================================================

    profitable = (
        folds_df[
            "final_return_pct"
        ]
        > 0.0
    )

    folds_df[
        "profitable"
    ] = profitable

    # =========================================================
    # Asset summary
    # =========================================================

    asset_summary = (
        folds_df
        .groupby(
            [
                "ticker",
                "asset_class",
            ],
            as_index=False,
        )
        .agg(
            paths=(
                "final_return_pct",
                "count",
            ),
            profitable_paths=(
                "profitable",
                "sum",
            ),
            average_final_return_pct=(
                "final_return_pct",
                "mean",
            ),
            median_final_return_pct=(
                "final_return_pct",
                "median",
            ),
            best_final_return_pct=(
                "final_return_pct",
                "max",
            ),
            worst_final_return_pct=(
                "final_return_pct",
                "min",
            ),
            average_underlying_return_pct=(
                "underlying_return_pct",
                "mean",
            ),
            average_abs_move_pct=(
                "average_abs_move_pct",
                "mean",
            ),
            average_mdd_pct=(
                "mdd_pct",
                "mean",
            ),
            average_transaction_cost=(
                "transaction_cost",
                "mean",
            ),
            average_execution_turnover=(
                "execution_turnover",
                "mean",
            ),
        )
    )

    asset_summary[
        "profitable_path_pct"
    ] = (
        asset_summary[
            "profitable_paths"
        ]
        / asset_summary["paths"]
        * 100.0
    )

    # =========================================================
    # Asset-class summary
    # =========================================================

    class_summary = (
        folds_df
        .groupby(
            "asset_class",
            as_index=False,
        )
        .agg(
            paths=(
                "final_return_pct",
                "count",
            ),
            profitable_paths=(
                "profitable",
                "sum",
            ),
            average_final_return_pct=(
                "final_return_pct",
                "mean",
            ),
            median_final_return_pct=(
                "final_return_pct",
                "median",
            ),
            best_final_return_pct=(
                "final_return_pct",
                "max",
            ),
            worst_final_return_pct=(
                "final_return_pct",
                "min",
            ),
            average_underlying_return_pct=(
                "underlying_return_pct",
                "mean",
            ),
            average_abs_move_pct=(
                "average_abs_move_pct",
                "mean",
            ),
            average_mdd_pct=(
                "mdd_pct",
                "mean",
            ),
            average_transaction_cost=(
                "transaction_cost",
                "mean",
            ),
            average_execution_turnover=(
                "execution_turnover",
                "mean",
            ),
        )
    )

    class_summary[
        "profitable_path_pct"
    ] = (
        class_summary[
            "profitable_paths"
        ]
        / class_summary["paths"]
        * 100.0
    )

    # =========================================================
    # Overall summary
    # =========================================================

    total_paths = len(
        folds_df
    )

    profitable_paths = int(
        folds_df[
            "profitable"
        ].sum()
    )

    final_returns = (
        folds_df[
            "final_return_pct"
        ]
        .astype(float)
        .to_numpy()
    )

    overall_summary = {

        "total_paths":
            total_paths,

        "profitable_paths":
            profitable_paths,

        "unprofitable_paths":
            total_paths
            - profitable_paths,

        "profitable_path_pct":
            (
                profitable_paths
                / total_paths
                * 100.0
            ),

        "average_final_return_pct":
            float(
                np.mean(final_returns)
            ),

        "median_final_return_pct":
            float(
                np.median(final_returns)
            ),

        "best_path_return_pct":
            float(
                np.max(final_returns)
            ),

        "worst_path_return_pct":
            float(
                np.min(final_returns)
            ),

        "average_mdd_pct":
            float(
                folds_df[
                    "mdd_pct"
                ].mean()
            ),

        "average_underlying_return_pct":
            float(
                folds_df[
                    "underlying_return_pct"
                ].mean()
            ),

        "average_abs_move_pct":
            float(
                folds_df[
                    "average_abs_move_pct"
                ].mean()
            ),

        "total_transaction_cost":
            float(
                folds_df[
                    "transaction_cost"
                ].sum()
            ),

        "total_execution_turnover":
            float(
                folds_df[
                    "execution_turnover"
                ].sum()
            ),

        "execution_mode":
            execution_mode,

        "h_interval":
            h_interval,

        "h_lev":
            h_lev,

        "friction_bps":
            friction_bps,

        "n_assets":
            len(assets),

        "n_folds_per_asset":
            len(folds),
    }

    # =========================================================
    # Aggregate plot
    # =========================================================

    if plot_output:

        plt.figure(
            figsize=(15, 8)
        )

        for path_info in plotted_paths:

            path = (
                path_info[
                    "path"
                ]
            )

            label = (
                f"{path_info['ticker']} "
                f"F{path_info['fold']}"
            )

            plt.plot(
                path,
                alpha=0.15,
                linewidth=1.0,
                label=label,
            )

        plt.axhline(
            0.0,
            linestyle="--",
            linewidth=1.5,
        )

        plt.xlabel(
            "Test Observation"
        )

        plt.ylabel(
            "Cumulative Return (%)"
        )

        plt.title(
            "VDC Purged Chronological Walk-Forward "
            f"Across {len(assets)} Assets"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        plt.show()

    # =========================================================
    # Report
    # =========================================================

    if report_output:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "VDC ASSET WALK-FORWARD SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Assets                  : "
            f"{len(assets)}"
        )

        print(
            f"Folds / Asset           : "
            f"{len(folds)}"
        )

        print(
            f"Valid Paths             : "
            f"{total_paths}"
        )

        print(
            f"Profitable Paths        : "
            f"{profitable_paths} "
            f"({overall_summary['profitable_path_pct']:.2f}%)"
        )

        print(
            f"Average Final Return    : "
            f"{overall_summary['average_final_return_pct']:.4f}%"
        )

        print(
            f"Median Final Return     : "
            f"{overall_summary['median_final_return_pct']:.4f}%"
        )

        print(
            f"Best Path Return        : "
            f"{overall_summary['best_path_return_pct']:.4f}%"
        )

        print(
            f"Worst Path Return       : "
            f"{overall_summary['worst_path_return_pct']:.4f}%"
        )

        print(
            f"Average MDD             : "
            f"{overall_summary['average_mdd_pct']:.4f}%"
        )

        print(
            f"Average Underlying Move : "
            f"{overall_summary['average_abs_move_pct']:.6f}%"
        )

        print(
            f"Total Transaction Cost  : "
            f"{overall_summary['total_transaction_cost']:.6f}"
        )

        print(
            f"Total Execution Turnover:"
            f" {overall_summary['total_execution_turnover']:.6f}"
        )

        print(
            "=" * 70
        )

        print(
            "\nPER-ASSET RESULTS"
        )

        print(
            asset_summary.to_string(
                index=False,
                formatters={
                    "average_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "median_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "best_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "worst_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "profitable_path_pct":
                        lambda x: f"{x:.2f}%",

                    "average_mdd_pct":
                        lambda x: f"{x:.4f}%",

                    "average_abs_move_pct":
                        lambda x: f"{x:.6f}%",

                    "average_underlying_return_pct":
                        lambda x: f"{x:.4f}%",
                },
            )
        )

        print(
            "\nPER-ASSET-CLASS RESULTS"
        )

        print(
            class_summary.to_string(
                index=False,
                formatters={
                    "average_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "median_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "best_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "worst_final_return_pct":
                        lambda x: f"{x:.4f}%",

                    "profitable_path_pct":
                        lambda x: f"{x:.2f}%",

                    "average_mdd_pct":
                        lambda x: f"{x:.4f}%",

                    "average_abs_move_pct":
                        lambda x: f"{x:.6f}%",

                    "average_underlying_return_pct":
                        lambda x: f"{x:.4f}%",
                },
            )
        )

        if invalid_paths:

            print(
                "\nINVALID PATHS"
            )

            print(
                pd.DataFrame(
                    invalid_paths
                ).to_string(
                    index=False
                )
            )

        print(
            "=" * 70
        )

    return {
        "folds": folds_df,
        "asset_summary": asset_summary,
        "class_summary": class_summary,
        "overall_summary": overall_summary,
        "invalid_paths": invalid_paths,
        "plotted_paths": plotted_paths,
    }
def vdc_portfolio_index(
    vdc_results,
    weighting="equal",
    initial_capital=1.0,
):
    """
    Combine multiple VDC asset/fold paths into one portfolio equity index.

    Portfolio construction
    ----------------------
    Each asset/fold path is normalized to begin at 0%.

    At every test observation, the portfolio combines the returns of the
    available asset/fold paths.

    ``weighting="equal"`` gives every active asset the same portfolio weight.

    The objective is NOT to ask whether one asset produces a profitable VDC
    path. It asks whether diversification across different assets makes the
    VDC edge more stable.

    Returns
    -------
    dict
        portfolio_path:
            DataFrame containing the aggregate portfolio index.

        summary:
            Portfolio-level performance statistics.
    """

    if not isinstance(
        vdc_results,
        dict,
    ):
        raise TypeError(
            "vdc_results must be the dictionary returned by vdc_asset_test"
        )

    if weighting != "equal":
        raise ValueError(
            "Currently only weighting='equal' is supported"
        )

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError(
            "initial_capital must be > 0"
        )

    if "plotted_paths" not in vdc_results:
        raise ValueError(
            "vdc_results must contain 'plotted_paths'. "
            "Add plotted_paths to the return dictionary of vdc_asset_test."
        )

    paths = vdc_results[
        "plotted_paths"
    ]

    if not paths:
        raise ValueError(
            "No valid portfolio paths were supplied."
        )

    # =========================================================
    # Group paths by walk-forward fold.
    #
    # Assets belonging to the same fold represent simultaneous
    # portfolio constituents during the same test period.
    # =========================================================

    fold_groups = {}

    for item in paths:

        fold = item[
            "fold"
        ]

        fold_groups.setdefault(
            fold,
            [],
        ).append(item)

    portfolio_paths = []

    fold_records = []

    # =========================================================
    # Build one portfolio path per walk-forward fold.
    # =========================================================

    for fold, fold_items in sorted(
        fold_groups.items()
    ):

        raw_paths = [
            np.asarray(
                item["path"],
                dtype=np.float64,
            )
            for item in fold_items
        ]

        if not raw_paths:
            continue

        # -----------------------------------------------------
        # Paths should normally have identical length because
        # the test window is shared across assets.
        #
        # If data availability differs, truncate to the common
        # length rather than inventing observations.
        # -----------------------------------------------------

        common_length = min(
            len(path)
            for path in raw_paths
        )

        if common_length == 0:
            continue

        matrix = np.vstack(
            [
                path[:common_length]
                for path in raw_paths
            ]
        )

        # -----------------------------------------------------
        # Convert cumulative returns into per-bar returns.
        #
        # This is critical.
        #
        # We do NOT average cumulative returns directly.
        # We first recover each constituent's bar return, then
        # average those returns to create the portfolio return.
        # -----------------------------------------------------

        constituent_returns = (
            matrix[:, 1:]
            - matrix[:, :-1]
        )

        initial_returns = (
            matrix[:, :1]
        )

        # Since each path begins at 0 cumulative return,
        # the first bar's return is its first cumulative value.
        constituent_returns = np.hstack(
            [
                initial_returns,
                constituent_returns,
            ]
        )

        # -----------------------------------------------------
        # Equal-weight portfolio.
        #
        # Each asset receives identical weight within the fold.
        # -----------------------------------------------------

        portfolio_returns = (
            constituent_returns.mean(
                axis=0
            )
        )

        # -----------------------------------------------------
        # Reconstruct portfolio cumulative return.
        # -----------------------------------------------------

        portfolio_cumulative = (
            np.cumsum(
                portfolio_returns
            )
        )

        portfolio_path = {
            "fold": fold,
            "path": portfolio_cumulative,
            "n_assets": len(fold_items),
        }

        portfolio_paths.append(
            portfolio_path
        )

        # -----------------------------------------------------
        # Fold-level record.
        # -----------------------------------------------------

        fold_records.append(
            {
                "fold":
                    fold,

                "n_assets":
                    len(fold_items),

                "final_return_pct":
                    float(
                        portfolio_cumulative[-1]
                    ),

                "average_asset_return_pct":
                    float(
                        matrix[:, -1].mean()
                    ),

                "median_asset_return_pct":
                    float(
                        np.median(
                            matrix[:, -1]
                        )
                    ),

                "profitable_asset_pct":
                    float(
                        np.mean(
                            matrix[:, -1] > 0
                        )
                        * 100.0
                    ),
            }
        )

    if not portfolio_paths:
        raise RuntimeError(
            "Unable to construct portfolio paths."
        )

    # =========================================================
    # Combine chronological folds into one portfolio index.
    #
    # Folds are kept sequential rather than overlapping.
    # Each fold starts from the previous portfolio NAV.
    # =========================================================

    portfolio_returns = []

    for fold_path in portfolio_paths:

        path = fold_path[
            "path"
        ]

        if len(path) == 0:
            continue

        # Convert cumulative path back into period returns.
        fold_returns = np.empty_like(
            path
        )

        fold_returns[0] = path[0]

        if len(path) > 1:

            fold_returns[1:] = (
                np.diff(path)
            )

        portfolio_returns.extend(
            fold_returns.tolist()
        )

    portfolio_returns = np.asarray(
        portfolio_returns,
        dtype=np.float64,
    )

    portfolio_cumulative_return = (
        np.cumsum(
            portfolio_returns
        )
    )

    strategy_nav = (
        initial_capital
        * (
            1.0
            + portfolio_cumulative_return
        )
    )

    # =========================================================
    # Portfolio MDD
    # =========================================================

    running_peak = (
        np.maximum.accumulate(
            strategy_nav
        )
    )

    drawdown = (
        strategy_nav
        / running_peak
        - 1.0
    )

    mdd_pct = float(
        drawdown.min()
        * 100.0
    )

    # =========================================================
    # Portfolio return statistics
    # =========================================================

    profitable_periods = (
        portfolio_returns > 0
    )

    win_rate_pct = float(
        np.mean(
            profitable_periods
        )
        * 100.0
    )

    downside = np.minimum(
        portfolio_returns,
        0.0,
    )

    downside_deviation = float(
        np.sqrt(
            np.mean(
                downside ** 2
            )
        )
    )

    if downside_deviation > 0:

        sortino = float(
            np.mean(
                portfolio_returns
            )
            / downside_deviation
        )

    elif np.mean(
        portfolio_returns
    ) > 0:

        sortino = np.inf

    else:

        sortino = np.nan

    # =========================================================
    # Portfolio output
    # =========================================================

    portfolio_index = pd.DataFrame(
        {
            "portfolio_return":
                portfolio_returns,

            "cumulative_return":
                portfolio_cumulative_return,

            "strategy_nav":
                strategy_nav,

            "drawdown":
                drawdown,
        }
    )

    # =========================================================
    # Summary
    # =========================================================

    all_asset_returns = np.asarray(
        [
            item["final_return_pct"]
            for item in paths
        ],
        dtype=np.float64,
    )

    summary = {

        "initial_capital":
            float(initial_capital),

        "n_assets":
            len(
                set(
                    item["ticker"]
                    for item in paths
                )
            ),

        "n_paths":
            len(paths),

        "n_folds":
            len(portfolio_paths),

        "final_portfolio_return_pct":
            float(
                portfolio_cumulative_return[-1]
                * 100.0
            ),

        "final_nav":
            float(
                strategy_nav[-1]
            ),

        "average_asset_final_return_pct":
            float(
                np.mean(
                    all_asset_returns
                )
            ),

        "median_asset_final_return_pct":
            float(
                np.median(
                    all_asset_returns
                )
            ),

        "profitable_asset_path_pct":
            float(
                np.mean(
                    all_asset_returns > 0
                )
                * 100.0
            ),

        "portfolio_mdd_pct":
            mdd_pct,

        "portfolio_sortino":
            sortino,

        "portfolio_win_rate_pct":
            win_rate_pct,

        "portfolio_downside_deviation":
            downside_deviation,

        "best_fold_return_pct":
            float(
                max(
                    record[
                        "final_return_pct"
                    ]
                    for record in fold_records
                )
            ),

        "worst_fold_return_pct":
            float(
                min(
                    record[
                        "final_return_pct"
                    ]
                    for record in fold_records
                )
            ),
    }

    # =========================================================
    # Plot
    # =========================================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        portfolio_index.index,
        portfolio_index[
            "cumulative_return"
        ].to_numpy()
        * 100.0,
        linewidth=2.0,
        label="Equal-Weighted VDC Portfolio",
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="Breakeven",
    )

    plt.xlabel(
        "Portfolio Test Observation"
    )

    plt.ylabel(
        "Cumulative Return (%)"
    )

    plt.title(
        "VDC Diversified Portfolio Index"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend(
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

    # =========================================================
    # Report
    # =========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VDC PORTFOLIO INDEX SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Assets                     : "
        f"{summary['n_assets']}"
    )

    print(
        f"Asset/Fold Paths            : "
        f"{summary['n_paths']}"
    )

    print(
        f"Walk-Forward Folds          : "
        f"{summary['n_folds']}"
    )

    print(
        f"Final Portfolio Return      : "
        f"{summary['final_portfolio_return_pct']:.4f}%"
    )

    print(
        f"Final NAV                   : "
        f"{summary['final_nav']:.6f}"
    )

    print(
        f"Average Asset Return        : "
        f"{summary['average_asset_final_return_pct']:.4f}%"
    )

    print(
        f"Median Asset Return         : "
        f"{summary['median_asset_final_return_pct']:.4f}%"
    )

    print(
        f"Profitable Asset Paths      : "
        f"{summary['profitable_asset_path_pct']:.2f}%"
    )

    print(
        f"Portfolio MDD               : "
        f"{summary['portfolio_mdd_pct']:.4f}%"
    )

    print(
        f"Portfolio Sortino           : "
        f"{summary['portfolio_sortino']:.4f}"
    )

    print(
        f"Portfolio Win Rate          : "
        f"{summary['portfolio_win_rate_pct']:.2f}%"
    )

    print(
        f"Best Fold Return            : "
        f"{summary['best_fold_return_pct']:.4f}%"
    )

    print(
        f"Worst Fold Return           : "
        f"{summary['worst_fold_return_pct']:.4f}%"
    )

    print(
        "=" * 60
    )

    return {
        "portfolio_index":
            portfolio_index,

        "fold_results":
            pd.DataFrame(
                fold_records
            ),

        "portfolio_paths":
            portfolio_paths,

        "summary":
            summary,
    }