from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Added progress bar import

from components.pipelines.diagnostic.lossless import lob_report
from strategy_system.exchange.loader import load_L1
from strategy_system.strategy.vdc.features import compute_feature
from strategy_system.strategy.vdc.outcome.triple_barrier import add_triple_barrier_sequence
from strategy_system.strategy.vdc.policies.orderflow_prediction import orderflow_study_1
from components.pipelines.ingestion.synthetic.synth2 import simulate_market_stochastic
from components.pipelines.ingestion.synthetic.uncorrelated_filter import (
    generate_uncorrelated_synthetic_assets,
)
from strategy_system.strategy.vdc.policies.vdc.vdc_runner import run_vdc_strategy
from strategy_system.strategy.vdc.policies.vdc_stresstest import (
    vdc_monte_carlo,
    vdc_asset_test,
    vdc_synthetic_asset_test,
)
from strategy_system.strategy.vdc.policies.vdc_portfolio import vdc_portfolio
from components.pipelines.event_merger.matched.reconstruct import (
    reconstruct,
    rule,
    join_timeframes,
)
from components.pipelines.event_merger.matched.lob import (
    LobReplayState,
    build_multilevel_lob,
    resume_from_day_snapshot,
)

from components.pipelines.event_merger.matched.parquet import (
    list_parquet,
    read_parquet,
    read_batched_parquet,
    write_parquet,
    write_parquet_batched,
)
from components.pipelines.ingestion.CryptoL1.price import get_ohlcv

VDC_EXAMPLE_MODE = "asset_portfolio"


def simple_vdc_regime_detection(snapshot):
    """Example hook: give every currently eligible asset equal score."""
    return pd.Series(1.0, index=snapshot.index)


def simple_vdc_stop_loss(snapshot, portfolio_state):
    """Example hook: stop new exposure after a 10% portfolio drawdown."""
    return portfolio_state["drawdown"] <= -0.10


def simple_vdc_executor(orders):
    """Example hook: fill target weights immediately with no extra fee."""
    return orders["target_weight"]


def run_vdc_monte_carlo_portfolio_example(
    n_scenarios=40,
    n_assets_per_scenario=1,
    n_bars=5_000,
    generator="synth2",
    seed=0,
    plot_output=True,
):
    """Standard MC test; scenarios stay separate, assets diversify within each."""
    mc_results = vdc_monte_carlo(
        n_paths=n_scenarios,
        n_assets_per_scenario=n_assets_per_scenario,
        n_bars=n_bars,
        generator=generator,
        seed=seed,
        plot_output=plot_output,
        strategy_kwargs={"report_output": False, "plot_output": False},
    )
    portfolio_results = vdc_portfolio(mc_results, plot_output=plot_output)
    return {"monte_carlo": mc_results, "portfolio": portfolio_results}


def run_vdc_asset_portfolio_example(asset_data=None, assets=None, plot_output=True):
    """Walk-forward real-asset test with small example risk/execution hooks."""
    asset_results = vdc_asset_test(
        asset_data=asset_data,
        assets=assets,
        plot_output=plot_output,
    )
    portfolio_results = vdc_portfolio(
        asset_results,
        regime_detection=simple_vdc_regime_detection,
        stop_loss=simple_vdc_stop_loss,
        executor=simple_vdc_executor,
        plot_output=plot_output,
    )
    return {"assets": asset_results, "portfolio": portfolio_results}


def run_vdc_uncorrelated_synthetic_portfolio_example(
    n_assets=100,
    n_bars=5_000,
    generator="synth2",
    max_abs_correlation=0.10,
    seed=0,
    bar_seconds=1.0,
    plot_output=True,
):
    """Generate, correlation-filter, and portfolio-test synthetic VDC assets."""
    filtered = generate_uncorrelated_synthetic_assets(
        n_assets=n_assets,
        generator=generator,
        max_abs_correlation=max_abs_correlation,
        max_candidates=n_assets * 10,
        seed=seed,
        bar_seconds=bar_seconds,
        generator_kwargs={
            "n_bars": n_bars,
            "n_features": 50,
            "n_states": 4,
            "feature_noise": 0.8,
            "volatility_scale": 1.0,
            "directional_bias": 0.0,
        },
    )
    asset_results = vdc_synthetic_asset_test(
        filtered["asset_data"],
        h_interval=bar_seconds,
        bar_seconds=bar_seconds,
        plot_output=plot_output,
    )
    portfolio_results = vdc_portfolio(
        asset_results,
        regime_detection=simple_vdc_regime_detection,
        stop_loss=simple_vdc_stop_loss,
        executor=simple_vdc_executor,
        plot_output=plot_output,
    )
    return {
        "correlation_filter": filtered,
        "assets": asset_results,
        "portfolio": portfolio_results,
    }


def build_state(df, vdc_example_mode=VDC_EXAMPLE_MODE):
    """
    Builds the internal strategy mechanism for strategy.py
    to backtest and interact with.
    """

    #df, feat_col = compute_feature(df, delay=0)
    #result = orderflow_study_1(df, feat_col)
    #vdc_summarized_report(df)
    # Automatically loops to pull > 1k bars

    # Select one research path here:
    #   "monte_carlo"            -> independent MC scenarios (and optional assets/scenario)
    #   "asset_portfolio"        -> real assets, walk-forward, custom example hooks
    #   "uncorrelated_synthetic" -> synthetic generator -> correlation filter -> portfolio
    if vdc_example_mode == "monte_carlo":
        research_results = run_vdc_monte_carlo_portfolio_example(
            n_scenarios=5,
            n_assets_per_scenario=100,
        )
    elif vdc_example_mode == "asset_portfolio":
        research_results = run_vdc_asset_portfolio_example()
    elif vdc_example_mode == "uncorrelated_synthetic":
        research_results = run_vdc_uncorrelated_synthetic_portfolio_example(
            n_assets=100,
            max_abs_correlation=0.10,
        )
    else:
        raise ValueError(
            "vdc_example_mode must be 'monte_carlo', 'asset_portfolio', "
            "or 'uncorrelated_synthetic'"
        )

    df, summary = run_vdc_strategy(
        df=df,
        price_column="close",   # Matches df_path["price"]
        h_interval=1,           # Synthetic fund reset interval, in seconds
        h_lev=3.0,              # 3x target leverage
        friction_bps=0.0,       # 0.04% one-way transaction cost
        execution_mode="minimum_margin",   # Direct short pair mode
        min_change_pct=0,       # 0 noise filter
        sessionize=True,        # Reset fund state daily
        report_output=True,     # Print the VDC summary
        plot_output=True,       # Show the individual VDC plot
    )
    print(summary)
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
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_paths = 100

    paths = []

    for seed in range(n_paths):
        df_path = simulate_market_stochastic(
            n_bars=5000,
            n_features=50,
            n_states=4,
            seed=seed,
            feature_noise=0.8,
            initial_price=100,
        )

        paths.append(df_path["price"].to_numpy())

    paths = np.array(paths)

    plt.figure(figsize=(14, 7))

    for path in paths:
        plt.plot(path, alpha=0.15)

    plt.xlabel("Bar")
    plt.ylabel("Price")
    plt.title(f"{n_paths} Independent Stochastic Price Paths")
    plt.show()
    #return result
    """

if __name__ == "__main__":
    data = load_L1()
    build_state(data)