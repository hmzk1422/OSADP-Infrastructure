"""
run_persistence_study.py - build the rolling-origin dataset, print every diagnostic, fit the bundle, then (optionally)
run your Monte Carlo purged walk-forward with the persistence rule.

Folder layout (flat):  spread_discoverer.py  strategy.py (patched)  mc_stress_test.py  persistence.py  this file

STEP 1  python run_persistence_study.py            -> dataset parquet + study tables + persistence_bundle.pkl
STEP 2  set RUN_MC = True and run again            -> mc_stress_test with retrain_rule=PersistenceRetrainRule(bundle)

LEAKAGE RULE (the one thing that can fake an edge here): the gate/half-life models are trained on GATE_RANGES only.
The Monte Carlo must draw its random segments from AFTER them (MC_HISTORY_START) so no test bar was ever seen by the
models.  Keep a final stretch you never touch until the last run.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from persistence import DatasetConfig, PersistenceBundle, PersistenceRetrainRule, build_dataset, run_study
from spread_discoverer import DiscoverConfig, get_assets

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- settings
SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
TARGET = "ETH/USDT"
BAR, BAR_MINUTES = "1min", 1.0
# calendar blocks used ONLY to train the models (several disjoint blocks = several regimes); about 30-60 days each
GATE_RANGES = [("2021-03-01", "2021-04-15"), ("2022-05-01", "2022-06-15"), ("2023-01-10", "2023-02-25"),
               ("2023-08-01", "2023-09-15")]
MC_HISTORY_START = "2024-01-01"          # Monte Carlo draws segments from here on (after every gate range)
DATASET_PATH = "persistence_dataset.parquet"
BUNDLE_PATH = "persistence_bundle.pkl"
HORIZON = 2_500                          # forward horizon the half-life model and the gate are trained for
GATE_W = 720                             # z window whose structural payoff the gate learns (the grid is evaluated anyway)
RUN_MC = False

dcfg = DatasetConfig(train_bars=5_000, fwd_bars=(500, 1_000, 2_500, 5_000), step=1_000, top_k=20,
                     max_origins=250, n_jobs=1, seed=7)       # raise max_origins / n_jobs when happy with the plumbing
discover_cfg = DiscoverConfig(use_features=True)


def make_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        return pd.read_parquet(DATASET_PATH)
    frames = []
    for k, (s, e) in enumerate(GATE_RANGES):
        panel = get_assets(SYMBOLS, s, e, bar=BAR, market="um", cfg=discover_cfg)
        frames.append(build_dataset(panel, TARGET, discover_cfg, dcfg, segment=k))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(DATASET_PATH)
    return df


def show(res: dict) -> None:
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
    titles = {
        "adf_power": "1) ADF vs variance-ratio power on SIMULATED stationary spreads (why 'failed ADF OOS' can still be reverting)",
        "ex_post_forward_metric_vs_payoff": "2) EX POST: which forward property co-moves with mean-reversion payoff (Spearman, block-bootstrap CI)",
        "ex_ante_train_metric_vs_payoff": "3) EX ANTE: which TRAIN statistic predicts forward payoff",
        "ex_ante_train_metric_vs_persistence": "4) EX ANTE: which TRAIN statistic predicts forward half-life ratio (log, + = slower)",
        "halflife_cv": "5) half-life predictor vs baselines (purged time-blocked CV; log-space error, lower = better)",
        "window_policies": "6) z-window policies, realised structural R (oracle row = ceiling for any half-life predictor)",
        "gate_precision_curve": "7) gate: cumulative confidence cuts -> coverage / P(persist) / mean R with CI (lo_R > 0 opens the gate)",
    }
    for k, t in titles.items():
        print(f"\n=== {t}")
        print(res[k].head(15).to_string(index=False))
    g = res["bundle"].gate
    print(f"\ngate threshold (raw score): {g.threshold}   -> {'OPEN above it' if g.threshold < float('inf') else 'CLOSED: no cut had a positive lower CI'}")


if __name__ == "__main__":
    df = make_dataset()
    print(f"dataset: {len(df):,} rows | origins {df['origin'].nunique():,} | segments {df['segment'].nunique()}")
    res = run_study(df, dcfg, BAR_MINUTES, horizon=HORIZON, W=GATE_W, target="edge", model="ridge", k_window=3.0)
    show(res)
    res["bundle"].save(BUNDLE_PATH)

    if RUN_MC:
        from mc_stress_test import MCConfig, build_experiment, compare, make_loader
        from strategy import TradeConfig
        bundle = PersistenceBundle.load(BUNDLE_PATH)
        MC = MCConfig(total_bars=100_000, chunk_bars=5_000, min_segment_bars=10_000, history_start=MC_HISTORY_START,
                      history_end="2026-07-31", bar=BAR, seed=7, n_boot=2_000)
        loader = make_loader(SYMBOLS, discover_cfg, bar=MC.bar)
        base = dict(entry_z=2.0, capital=10_000, risk_fraction=0.02, rr=1.0, barrier_size=1.0, fee_bps=2.0, slippage_bps=1.0,
                    feature_score_mode="filter", use_tree=False, tree_diagnostics=False, strategy_side=1,
                    zscore_mode="rolling", barrier_mode="hilo")
        configs = {"rolling z, persistence rule": TradeConfig.for_bar(MC.bar_minutes, **base)}
        # same draw, default screening, for the apples-to-apples comparison (both see identical segments and costs)
        exp_default = build_experiment(TARGET, loader, MC, discover_cfg, n_jobs=-1)
        exp_rule = build_experiment(TARGET, loader, MC, discover_cfg, n_jobs=-1, retrain_rule=PersistenceRetrainRule(bundle))
        for name, exp in (("default screening", exp_default), ("persistence rule", exp_rule)):
            table, results = compare(exp, {f"{name}": configs["rolling z, persistence rule"]}, n_jobs=-1)
            print(f"\n##### {name}")
            print(table.to_string())
            for r in results.values():
                print(r.report)