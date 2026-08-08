"""
Train the Q-learning survival agent.

Usage:
    python train.py [--episodes N] [--seed S]

Trains the agent on the survival-first environment and saves:
  - the learned Q-table (numpy .npy)
  - a learning curve CSV (episode, reward, wealth, survived, epsilon)
  - a learning curve plot
"""

import argparse
import os

import numpy as np

from config import (
    AgentConfig,
    MarketConfig,
    PathConfig,
    RewardConfig,
    TradingConfig,
)
from environment import SurvivalEnv
from agent import QLearningAgent


def main():
    parser = argparse.ArgumentParser(description="Train the survival agent.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    market_cfg = MarketConfig()
    trading_cfg = TradingConfig()
    reward_cfg = RewardConfig()
    agent_cfg = AgentConfig()
    if args.episodes is not None:
        agent_cfg.n_episodes = args.episodes
    paths = PathConfig()

    os.makedirs(paths.output_dir, exist_ok=True)

    # --- Environment & agent ------------------------------------------------
    env = SurvivalEnv(market_cfg, trading_cfg, reward_cfg, seed=args.seed)
    agent = QLearningAgent(agent_cfg, trading_cfg, env.n_obs)

    print(f"State space: {agent.n_states} states x {agent.n_actions} actions")
    print(f"Training for {agent_cfg.n_episodes} episodes...")

    # Fit the discretizer on random exploration.
    agent.fit_discretizer(env)

    # --- Training loop ------------------------------------------------------
    history = []
    for ep in range(agent_cfg.n_episodes):
        result = agent.train_episode(env)
        history.append(result)

        if (ep + 1) % 200 == 0 or ep == 0:
            print(
                f"Ep {ep+1:5d} | reward={result['total_reward']:8.3f} "
                f"| wealth={result['final_wealth']:7.4f} "
                f"| survived={result['survived']} "
                f"| eps={result['epsilon']:.3f}"
            )

    # --- Save ---------------------------------------------------------------
    np.save(paths.model_path, agent.q)
    np.save(paths.model_path.replace(".npy", "_edges.npy"),
            np.array(agent.discretizer.edges, dtype=object), allow_pickle=True)

    # Learning curve CSV.
    import csv
    with open(os.path.join(paths.output_dir, "learning_curve.csv"), "w",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"\nSaved Q-table to {paths.model_path}")
    print(f"Saved learning curve to {os.path.join(paths.output_dir, 'learning_curve.csv')}")


if __name__ == "__main__":
    main()