"""
Survival-first trading environment (Gym-style interface, numpy-only).

The agent's ONLY job is risk management: each step it chooses a capital size
(action) from a discrete set. The environment applies that size to the next
market return, charges commission drag proportional to exposure, tracks
wealth and drawdown, and terminates the episode on ruin.

The reward function is deliberately NOT "maximize profit". It is built to
teach survival:

  reward = log_return
           - risk_aversion * (log_return - mean_log_return)^2   # variance penalty
           - dd_penalty_coef * max(0, drawdown - dd_threshold)  # drawdown penalty
           + survival_bonus (terminal, if you survive the whole episode)
           + ruin_penalty (terminal, if you hit the ruin threshold)

This makes "sitting in cash" a legitimate, often optimal, action. Cash is a
position. The agent must learn WHEN NOT to trade.
"""

from typing import List, Optional, Tuple

import numpy as np

from config import MarketConfig, RewardConfig, TradingConfig
from market import RegimeSwitchingMarket


class SurvivalEnv:
    """Gym-style environment for the survival agent study."""

    def __init__(
        self,
        market_cfg: MarketConfig,
        trading_cfg: TradingConfig,
        reward_cfg: RewardConfig,
        seed: Optional[int] = None,
    ):
        self.market_cfg = market_cfg
        self.trading_cfg = trading_cfg
        self.reward_cfg = reward_cfg
        self.seed = seed

        self.n_actions = len(trading_cfg.actions)
        self.action_space = list(range(self.n_actions))

        # Observation: market vars (4) + portfolio state (drawdown, wealth_norm)
        self.n_obs = 4 + 2
        self.obs_names = ["ema_short", "ema_long", "vol", "momentum",
                          "drawdown", "wealth_norm"]

        self._market = RegimeSwitchingMarket(market_cfg, seed=seed)
        self._reset_episode()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def _reset_episode(self):
        self.data = self._market.generate()
        self.t = 0
        self.wealth = 1.0
        self.peak_wealth = 1.0
        self.initial_wealth = 1.0
        self.episode_returns = []
        self.episode_actions = []
        self.episode_wealth = [1.0]
        self.episode_drawdowns = []
        self.ruined = False
        self.survived = False

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation."""
        self._reset_episode()
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply the chosen capital size to the next market step.

        Returns (obs, reward, done, info).
        """
        cfg_t = self.trading_cfg
        cfg_r = self.reward_cfg

        exposure = cfg_t.actions[action]  # capital size (0..2x)

        # Market return for this step (bias + noise).
        r = self.data["returns"][self.t]

        # Commission drag: proportional to exposure, in return units.
        commission = cfg_t.commission_rr * self.market_cfg.base_vol * exposure

        # Net portfolio return this step.
        net_r = exposure * r - commission

        # Update wealth (log-return accumulation).
        self.wealth *= np.exp(net_r)
        self.peak_wealth = max(self.peak_wealth, self.wealth)

        # Track.
        self.episode_returns.append(net_r)
        self.episode_actions.append(action)
        self.episode_wealth.append(self.wealth)

        # Drawdown from peak.
        dd = 1.0 - self.wealth / self.peak_wealth
        self.episode_drawdowns.append(dd)

        # --- Reward (survival-first) -------------------------------------
        reward = self._compute_reward(net_r, dd)

        # --- Termination --------------------------------------------------
        done = False
        if self.wealth <= self.initial_wealth * cfg_r.ruin_fraction:
            done = True
            self.ruined = True
            reward += cfg_r.ruin_penalty

        self.t += 1
        if self.t >= self.market_cfg.n_steps:
            done = True
            if not self.ruined:
                self.survived = True
                reward += cfg_r.survival_bonus

        info = {
            "wealth": self.wealth,
            "drawdown": dd,
            "exposure": exposure,
            "ruined": self.ruined,
            "survived": self.survived,
            "true_regime": self.data["regime"][self.t - 1],
            "true_bias": self.data["true_bias"][self.t - 1],
        }

        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, net_r: float, dd: float) -> float:
        cfg = self.reward_cfg

        # Base: log return.
        reward = net_r

        # Variance penalty: penalize the squared deviation from the running
        # mean of returns. This is a simple proxy for risk-aversion.
        if len(self.episode_returns) > 1:
            mean_r = float(np.mean(self.episode_returns[:-1]))
            reward -= cfg.risk_aversion * (net_r - mean_r) ** 2

        # Drawdown penalty: kicks in once drawdown exceeds the threshold.
        excess_dd = max(0.0, dd - cfg.dd_penalty_threshold)
        reward -= cfg.dd_penalty_coef * excess_dd

        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the observation vector for the current step."""
        # Clamp index: after the final step, self.t == n_steps (out of bounds).
        idx = min(self.t, self.market_cfg.n_steps - 1)
        obs = self.data["obs"][idx].copy()
        dd = 1.0 - self.wealth / self.peak_wealth
        wealth_norm = self.wealth / self.initial_wealth
        return np.concatenate([obs, [dd, wealth_norm]])

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    @property
    def final_wealth(self) -> float:
        return self.episode_wealth[-1]

    @property
    def max_drawdown(self) -> float:
        return max(self.episode_drawdowns) if self.episode_drawdowns else 0.0

    @property
    def total_return(self) -> float:
        return self.final_wealth - 1.0

    @property
    def sharpe(self) -> float:
        """Annualized-ish Sharpe from per-step returns (scaled by sqrt(T))."""
        if len(self.episode_returns) < 2:
            return 0.0
        arr = np.array(self.episode_returns)
        std = arr.std()
        if std == 0:
            return 0.0
        return arr.mean() / std * np.sqrt(len(arr))