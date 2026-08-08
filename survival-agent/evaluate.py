"""
Evaluate the trained survival agent against baseline strategies.

Usage:
    python evaluate.py [--seeds N]

Runs the trained Q-table (greedy policy) and each baseline across N
independent market seeds and reports:
  - survival rate (fraction of episodes not ruined)
  - median / mean final wealth
  - median / mean max drawdown
  - median / mean Sharpe
  - median / mean total return

Results are saved to results.csv and summary.json.
"""

import argparse
import json
import os

import numpy as np

from config import (
    EvalConfig,
    MarketConfig,
    PathConfig,
    RewardConfig,
    TradingConfig,
)
from environment import SurvivalEnv
from agent import QLearningAgent, make_baseline


def run_strategy(strategy, env, n_episodes: int) -> list:
    """Run a strategy (callable obs->action) over fresh episodes."""
    outcomes = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = strategy(obs)
            obs, _, done, _ = env.step(action)
        outcomes.append({
            "final_wealth": env.final_wealth,
            "max_drawdown": env.max_drawdown,
            "sharpe": env.sharpe,
            "total_return": env.total_return,
            "ruined": env.ruined,
            "survived": env.survived,
        })
    return outcomes


def summarize(outcomes: list) -> dict:
    """Aggregate outcomes into summary statistics."""
    fw = np.array([o["final_wealth"] for o in outcomes])
    dd = np.array([o["max_drawdown"] for o in outcomes])
    sh = np.array([o["sharpe"] for o in outcomes])
    tr = np.array([o["total_return"] for o in outcomes])
    surv = np.mean([o["survived"] for o in outcomes])
    ruin = np.mean([o["ruined"] for o in outcomes])

    return {
        "survival_rate": float(surv),
        "ruin_rate": float(ruin),
        "median_final_wealth": float(np.median(fw)),
        "mean_final_wealth": float(np.mean(fw)),
        "p10_final_wealth": float(np.percentile(fw, 10)),
        "p90_final_wealth": float(np.percentile(fw, 90)),
        "median_max_drawdown": float(np.median(dd)),
        "mean_max_drawdown": float(np.mean(dd)),
        "median_sharpe": float(np.median(sh)),
        "mean_sharpe": float(np.mean(sh)),
        "median_total_return": float(np.median(tr)),
        "mean_total_return": float(np.mean(tr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the survival agent.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override number of evaluation seeds.")
    args = parser.parse_args()

    market_cfg = MarketConfig()
    trading_cfg = TradingConfig()
    reward_cfg = RewardConfig()
    eval_cfg = EvalConfig()
    if args.seeds is not None:
        eval_cfg.n_seeds = args.seeds
    paths = PathConfig()

    os.makedirs(paths.output_dir, exist_ok=True)

    # --- Load trained agent ------------------------------------------------
    agent_cfg = __import__("config", fromlist=["AgentConfig"]).AgentConfig()
    env = SurvivalEnv(market_cfg, trading_cfg, reward_cfg, seed=0)
    agent = QLearningAgent(agent_cfg, trading_cfg, env.n_obs)
    agent.q = np.load(paths.model_path)
    edges = np.load(paths.model_path.replace(".npy", "_edges.npy"),
                    allow_pickle=True)
    agent.discretizer.edges = list(edges)

    def agent_strategy(obs):
        return agent.act_greedy(obs)

    # --- Run all strategies ------------------------------------------------
    strategies = {"trained_agent": agent_strategy}
    for name in eval_cfg.baselines:
        strategies[name] = make_baseline(name, trading_cfg).act

    results = {}
    for name, strat in strategies.items():
        print(f"Evaluating {name} over {eval_cfg.n_seeds} seeds...")
        # One episode per seed (fresh market each time).
        outcomes = run_strategy(strat, env, eval_cfg.n_seeds)
        results[name] = summarize(outcomes)

    # --- Save ---------------------------------------------------------------
    import csv
    with open(paths.results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy"] + list(results["trained_agent"].keys()))
        for name, stats in results.items():
            writer.writerow([name] + [f"{v:.6f}" for v in stats.values()])

    with open(paths.summary_json, "w") as f:
        json.dump(results, f, indent=2)

    # --- Print --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SURVIVAL AGENT STUDY — EVALUATION RESULTS")
    print("=" * 80)
    header = f"{'strategy':<16} {'survive':>8} {'ruin':>8} {'medWealth':>10} "
    header += f"{'p10':>8} {'p90':>8} {'medDD':>8} {'medSharpe':>10}"
    print(header)
    print("-" * 80)
    for name, s in results.items():
        print(
            f"{name:<16} {s['survival_rate']:8.3f} {s['ruin_rate']:8.3f} "
            f"{s['median_final_wealth']:10.4f} {s['p10_final_wealth']:8.4f} "
            f"{s['p90_final_wealth']:8.4f} {s['median_max_drawdown']:8.3f} "
            f"{s['median_sharpe']:10.3f}"
        )
    print("=" * 80)
    print(f"Saved results to {paths.results_csv} and {paths.summary_json}")


if __name__ == "__main__":
    main()