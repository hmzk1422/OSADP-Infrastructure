"""
Generate analysis plots for the survival agent study.

Usage:
    python analyze.py

Produces:
  - equity_curves.png:   sample equity curves (agent vs baselines) on one market
  - learning_curve.png:  training reward / wealth / survival over episodes
  - regime_example.png:  example market with hidden regimes + observable vars
  - qtable_heatmap.png:  heatmap of the learned Q-table (action preference)
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    AgentConfig,
    MarketConfig,
    PathConfig,
    RewardConfig,
    TradingConfig,
)
from environment import SurvivalEnv
from agent import QLearningAgent, make_baseline


def plot_equity(paths: PathConfig):
    """Sample equity curves for agent vs baselines on one market seed."""
    market_cfg = MarketConfig()
    trading_cfg = TradingConfig()
    reward_cfg = RewardConfig()
    agent_cfg = AgentConfig()

    env = SurvivalEnv(market_cfg, trading_cfg, reward_cfg, seed=7)

    # Load agent.
    agent = QLearningAgent(agent_cfg, trading_cfg, env.n_obs)
    agent.q = np.load(paths.model_path)
    edges = np.load(paths.model_path.replace(".npy", "_edges.npy"),
                    allow_pickle=True)
    agent.discretizer.edges = list(edges)

    strategies = {
        "trained_agent": lambda obs: agent.act_greedy(obs),
        "always_cash": make_baseline("always_cash", trading_cfg).act,
        "always_full": make_baseline("always_full", trading_cfg).act,
        "always_2x": make_baseline("always_2x", trading_cfg).act,
        "ema_rule": make_baseline("ema_rule", trading_cfg).act,
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, strat in strategies.items():
        obs = env.reset()
        done = False
        wealth = [1.0]
        while not done:
            obs, _, done, _ = env.step(strat(obs))
            wealth.append(env.wealth)
        ax.plot(wealth, label=name, linewidth=1.5)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Wealth (log scale)")
    ax.set_yscale("log")
    ax.set_title("Equity Curves on One Market Seed (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.plot_equity, dpi=150)
    plt.close(fig)
    print(f"Saved {paths.plot_equity}")


def plot_learning(paths: PathConfig):
    """Plot the training learning curve."""
    import csv
    csv_path = os.path.join(paths.output_dir, "learning_curve.csv")
    if not os.path.exists(csv_path):
        print(f"Learning curve CSV not found: {csv_path}. Run train.py first.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    ep = np.array([int(r["episode"]) for r in rows])
    reward = np.array([float(r["total_reward"]) for r in rows])
    wealth = np.array([float(r["final_wealth"]) for r in rows])
    survived = np.array([r["survived"] == "True" for r in rows])

    # Rolling means.
    window = 50
    def roll(x):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(ep[window - 1:], roll(reward), color="tab:blue")
    axes[0].set_ylabel("Total reward (rolling 50)")
    axes[0].set_title("Learning Curve")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep[window - 1:], roll(wealth), color="tab:green")
    axes[1].set_ylabel("Final wealth (rolling 50)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, survived.astype(float), color="tab:red", alpha=0.3)
    axes[2].plot(ep[window - 1:], roll(survived.astype(float)),
                 color="tab:red")
    axes[2].set_ylabel("Survival rate (rolling 50)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(paths.plot_learning, dpi=150)
    plt.close(fig)
    print(f"Saved {paths.plot_learning}")


def plot_regime_example(paths: PathConfig):
    """Show an example market with hidden regimes and observable variables."""
    market_cfg = MarketConfig()
    from market import RegimeSwitchingMarket
    mkt = RegimeSwitchingMarket(market_cfg, seed=3).generate()

    T = market_cfg.n_steps
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Price.
    axes[0].plot(mkt["price"], color="tab:blue", linewidth=1.0)
    axes[0].set_ylabel("Price")
    axes[0].set_title("Example Market: Hidden Regimes + Observable Variables")
    axes[0].grid(True, alpha=0.3)

    # True bias vs observable EMA.
    axes[1].plot(mkt["true_bias"] / market_cfg.base_vol,
                 color="black", linewidth=1.0, label="True bias (hidden)")
    axes[1].plot(mkt["obs"][:, 0], color="tab:red", linewidth=1.0,
                 label="Short EMA (observable)")
    axes[1].plot(mkt["obs"][:, 1], color="tab:orange", linewidth=1.0,
                 label="Long EMA (observable)")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Bias (in vol units)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Regime shading.
    regime = mkt["regime"]
    labels = market_cfg.regime_labels
    colors = ["tab:red", "tab:gray", "tab:green"]
    for r, lab, col in zip(range(len(labels)), labels, colors):
        mask = regime == r
        # Shade contiguous runs.
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # Find run boundaries.
        boundaries = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[idx[0]], idx[boundaries + 1]])
        ends = np.concatenate([idx[boundaries], [idx[-1]]])
        for s, e in zip(starts, ends):
            axes[2].axvspan(s, e, color=col, alpha=0.3)
    axes[2].set_ylabel("Regime")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Step")
    axes[2].set_title("Hidden Regime (negative=red, neutral=gray, positive=green)")
    axes[2].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(paths.plot_regimes, dpi=150)
    plt.close(fig)
    print(f"Saved {paths.plot_regimes}")


def plot_qtable_heatmap(paths: PathConfig):
    """Heatmap of the learned Q-table: which action is preferred per state."""
    agent_cfg = AgentConfig()
    trading_cfg = TradingConfig()
    env = SurvivalEnv(MarketConfig(), trading_cfg, RewardConfig(), seed=0)
    agent = QLearningAgent(agent_cfg, trading_cfg, env.n_obs)
    agent.q = np.load(paths.model_path)
    edges = np.load(paths.model_path.replace(".npy", "_edges.npy"),
                    allow_pickle=True)
    agent.discretizer.edges = list(edges)

    # Preferred action per state.
    best = np.argmax(agent.q, axis=1)
    n_states = agent.n_states
    n_actions = agent.n_actions

    # Reshape to 2D for heatmap: use first two used vars as axes.
    n_bins = agent_cfg.n_bins
    n_used = len(agent_cfg.use_vars)
    if n_used < 2:
        print("Need at least 2 used vars for heatmap.")
        return

    # Flatten remaining dims.
    grid = best.reshape([n_bins] * n_used)
    # Average over trailing dims to get 2D.
    while grid.ndim > 2:
        grid = grid.mean(axis=-1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=n_actions - 1)
    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels([f"b{i}" for i in range(n_bins)])
    ax.set_yticklabels([f"b{i}" for i in range(n_bins)])
    ax.set_xlabel(f"Var: {agent_cfg.use_vars[0]}")
    ax.set_ylabel(f"Var: {agent_cfg.use_vars[1]}")
    ax.set_title("Learned Preferred Action per State (0=cash, 4=2x)")
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_actions))
    cbar.set_ticklabels([f"{a}x" for a in trading_cfg.actions])
    fig.tight_layout()
    fig.savefig(paths.plot_heatmap, dpi=150)
    plt.close(fig)
    print(f"Saved {paths.plot_heatmap}")


def main():
    paths = PathConfig()
    os.makedirs(paths.output_dir, exist_ok=True)
    plot_equity(paths)
    plot_learning(paths)
    plot_regime_example(paths)
    plot_qtable_heatmap(paths)
    print("\nAll plots generated.")


if __name__ == "__main__":
    main()