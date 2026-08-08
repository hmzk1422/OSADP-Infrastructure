"""
Q-learning agent and baseline strategies.

The agent discretizes the continuous observation space into bins and learns
a Q-table mapping (discretized state, action) -> value. The action space is
the set of capital sizes. The agent must learn that "cash is a position" and
that the optimal size is often 0.

Baselines are simple, interpretable strategies used as a benchmark:
  - always_cash:  never trade (pure survival)
  - always_full:  always 1x (naive active trader)
  - always_2x:    always 2x (reckless)
  - fixed_half:   always 0.5x
  - ema_rule:     trade 1x if short EMA > 0, else cash (simple momentum rule)
"""

from typing import List, Optional

import numpy as np

from config import AgentConfig, TradingConfig
from environment import SurvivalEnv


# ---------------------------------------------------------------------------
# State discretizer
# ---------------------------------------------------------------------------
class Discretizer:
    """Discretizes continuous observations into a flat integer state index."""

    def __init__(self, n_vars: int, n_bins: int, use_vars: List[int]):
        self.n_vars = n_vars
        self.n_bins = n_bins
        self.use_vars = use_vars          # indices of obs to use
        self.n_used = len(use_vars)
        # Per-variable bin edges, learned from data (percentile-based).
        self.edges: List[np.ndarray] = [None] * n_vars

    def fit(self, obs_samples: np.ndarray):
        """Fit bin edges from a sample of observations (percentiles)."""
        for i in self.use_vars:
            col = obs_samples[:, i]
            qs = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            self.edges[i] = np.percentile(col, qs)

    def encode(self, obs: np.ndarray) -> int:
        """Map an observation to a flat state index."""
        idx = 0
        for k, i in enumerate(self.use_vars):
            edges = self.edges[i]
            bin_i = int(np.searchsorted(edges, obs[i], side="right"))
            idx += bin_i * (self.n_bins ** k)
        return idx

    @property
    def n_states(self) -> int:
        return self.n_bins ** self.n_used


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------
class QLearningAgent:
    """Tabular Q-learning agent for the survival environment."""

    def __init__(
        self,
        agent_cfg: AgentConfig,
        trading_cfg: TradingConfig,
        n_obs: int,
    ):
        self.cfg = agent_cfg
        self.n_actions = len(trading_cfg.actions)

        # Map variable names to observation indices.
        obs_names = ["ema_short", "ema_long", "vol", "momentum",
                     "drawdown", "wealth_norm"]
        self.use_var_idx = [obs_names.index(v) for v in agent_cfg.use_vars]

        self.discretizer = Discretizer(n_obs, agent_cfg.n_bins, self.use_var_idx)
        self.n_states = self.discretizer.n_states
        self.q = np.zeros((self.n_states, self.n_actions))

        self.epsilon = agent_cfg.epsilon_start
        self.episode = 0

    def fit_discretizer(self, env: SurvivalEnv, n_samples: int = 20000):
        """Collect observations from random episodes to fit bin edges."""
        samples = []
        for _ in range(20):
            obs = env.reset()
            done = False
            while not done:
                samples.append(obs)
                action = np.random.randint(self.n_actions)
                obs, _, done, _ = env.step(action)
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break
        self.discretizer.fit(np.array(samples))

    def _act(self, state: int, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[state]))

    def train_episode(self, env: SurvivalEnv) -> dict:
        """Run one training episode and update the Q-table."""
        cfg = self.cfg
        obs = env.reset()
        state = self.discretizer.encode(obs)
        done = False
        total_reward = 0.0
        n_steps = 0

        while not done:
            action = self._act(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = self.discretizer.encode(next_obs)

            # Q-learning update.
            best_next = np.max(self.q[next_state]) if not done else 0.0
            td_target = reward + cfg.gamma * best_next
            self.q[state, action] += cfg.alpha * (
                td_target - self.q[state, action]
            )

            state = next_state
            total_reward += reward
            n_steps += 1

        # Epsilon decay.
        self.epsilon = max(
            cfg.epsilon_end,
            self.epsilon * cfg.epsilon_decay,
        )
        self.episode += 1

        return {
            "episode": self.episode,
            "total_reward": total_reward,
            "n_steps": n_steps,
            "final_wealth": env.final_wealth,
            "ruined": env.ruined,
            "survived": env.survived,
            "epsilon": self.epsilon,
        }

    def act_greedy(self, obs: np.ndarray) -> int:
        """Greedy action for evaluation (no exploration)."""
        state = self.discretizer.encode(obs)
        return self._act(state, greedy=True)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class Baseline:
    """Base class for baseline strategies."""

    def __init__(self, trading_cfg: TradingConfig):
        self.actions = trading_cfg.actions

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError


class AlwaysCash(Baseline):
    def act(self, obs: np.ndarray) -> int:
        return self.actions.index(0.0)


class AlwaysFull(Baseline):
    def act(self, obs: np.ndarray) -> int:
        return self.actions.index(1.0)


class Always2x(Baseline):
    def act(self, obs: np.ndarray) -> int:
        return self.actions.index(2.0)


class FixedHalf(Baseline):
    def act(self, obs: np.ndarray) -> int:
        return self.actions.index(0.5)


class EMARule(Baseline):
    """Trade 1x if short EMA > 0, else cash. Simple momentum rule."""

    def act(self, obs: np.ndarray) -> int:
        # obs[0] is ema_short (normalized by base_vol).
        if obs[0] > 0.0:
            return self.actions.index(1.0)
        return self.actions.index(0.0)


def make_baseline(name: str, trading_cfg: TradingConfig) -> Baseline:
    mapping = {
        "always_cash": AlwaysCash,
        "always_full": AlwaysFull,
        "always_2x": Always2x,
        "fixed_half": FixedHalf,
        "ema_rule": EMARule,
    }
    return mapping[name](trading_cfg)