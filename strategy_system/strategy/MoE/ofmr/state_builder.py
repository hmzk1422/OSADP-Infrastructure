from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Added progress bar import

from components.pipelines.diagnostic.lossless import lob_report
from strategy_system.strategy.MoE.ofmr.feature_builder import compute_feature
from strategy_system.strategy.MoE.ofmr.transform.triple_barrier import add_triple_barrier_sequence
from strategy_system.strategy.MoE.ofmr.hypothesis.orderflow_prediction import orderflow_study_1
from components.pipelines.ingestion.synthetic.synth2 import simulate_market_stochastic 
from strategy_system.strategy.MoE.ofmr.hypothesis.vdc.vdc_runner import run_vdc_strategy
from strategy_system.strategy.MoE.ofmr.tests.vdc_stresstest import vdc_monte_carlo, vdc_asset_test, vdc_portfolio_index
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

def build_state(df):
    """
    Builds the internal strategy mechanism for strategy.py
    to backtest and interact with.
    """

    #df, feat_col = compute_feature(df, delay=0)
    #result = orderflow_study_1(df, feat_col)
    #vdc_summarized_report(df)
    # Automatically loops to pull > 1k bars

    results = vdc_asset_test()
    portfolio = vdc_portfolio_index(
        results
    )
    df, summary = run_vdc_strategy(
        df=df,
        price_column="close",   # Matches df_path["price"]
        h_interval=1,         # 5-minute fund reset / hedge interval
        h_lev=3.0,              # 3x target leverage
        friction_bps=0.0,       # 0.04% one-way transaction cost
        execution_mode="pair",   # Direct short pair mode
        min_change_pct=0,       # 0 noise filter
        sessionize=True,        # Reset fund state daily
        report_output=True,    # Disable individual report prints
        plot_output=True,      # Disable individual plot popups
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
