"""
Markov-Switching GBM Environment and Data Generator.

This module is the single source of truth for the sandbox_msgbm environment.
Both the notebook and the validation test import from here to prevent
lookahead bias and ensure consistent scoring.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


def generate_dynamic_regime_market(n_assets=5, n_steps=2000, dt=1/252, seed=42):
    """
    Generates a highly unpredictable market where the DURATION, DRIFT, and VOLATILITY
    of every regime are completely randomized per asset.
    """
    np.random.seed(seed)
    returns = np.zeros((n_steps, n_assets))

    for i in range(n_assets):
        t = 0
        while t < n_steps:
            # 1. Randomize Regime Duration (Log-Normal gives heavy tails)
            duration = int(np.random.lognormal(mean=3.0, sigma=1.0))
            duration = max(5, min(duration, 500))

            end_t = min(t + duration, n_steps)
            actual_duration = end_t - t

            # 2. Randomize Drift and Volatility for this specific block
            mu = np.random.uniform(-0.60, 0.60)
            sigma = np.random.uniform(0.05, 0.60)

            # 3. Generate Returns for this block
            drift_term = (mu - 0.5 * sigma**2) * dt
            shock_term = sigma * np.sqrt(dt) * np.random.randn(actual_duration)
            returns[t:end_t, i] = drift_term + shock_term

            t = end_t

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return returns, prices


def compute_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Compute Sortino ratio (higher is better).

    Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
    where Downside Deviation only considers negative returns.
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No downside - perfect situation
        mean_ret = np.mean(excess_returns)
        return mean_ret / 1e-8 if mean_ret > 0 else 0.0

    downside_std = np.std(downside_returns)
    if downside_std < 1e-8:
        downside_std = 1e-8

    return float(np.mean(excess_returns) / downside_std)


def compute_trade_expectancy(portfolio_values):
    """
    Compute trade expectancy (edge per period).

    If portfolio goes from 1.0 to 1.4 in 20 periods, expectancy = 0.02.

    Args:
        portfolio_values: Array of portfolio values

    Returns:
        Average edge per period
    """
    n = len(portfolio_values) - 1
    if n <= 0:
        return 0.0

    total_return = portfolio_values[-1] / portfolio_values[0] - 1.0
    return float(total_return / n)

class SyntheticPortfolioEnv(gym.Env):
    """
    Custom Gymnasium Environment for Dynamic Asset Allocation.

    ML Objective: Maximize Sortino Ratio + Trade Expectancy.

    The reward function returns the Sortino ratio (risk-adjusted return),
    and the info dict contains trade expectancy for tracking.
    """

    def __init__(self, returns_matrix, window_size=20, transaction_cost=0.0005,
                 sortino_lookback=20, risk_free_rate=0.0):
        super(SyntheticPortfolioEnv, self).__init__()

        self.returns = returns_matrix  # Shape: (n_steps, n_assets)
        self.n_steps, self.n_assets = returns_matrix.shape
        self.window_size = window_size
        self.tc = transaction_cost
        self.sortino_lookback = sortino_lookback
        self.risk_free_rate = risk_free_rate

        # Action Space: Unnormalized weight preference for each asset (0.0 to 1.0)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation Space: Rolling Means (5) + Rolling Std Devs (5) + Current Portfolio Weights (5)
        obs_dim = self.n_assets * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.portfolio_value = 1.0
        self.returns_history = []
        self.portfolio_values_history = [1.0]

        return self._get_observation(), {}

    def _get_observation(self):
        """
        Extract rolling statistical metrics over window_size.
        CRITICAL: Only uses PAST data [current_step - window_size, current_step).
        No lookahead bias.
        """
        window = self.returns[self.current_step - self.window_size : self.current_step]
        rolling_mean = np.mean(window, axis=0)
        rolling_std = np.std(window, axis=0) + 1e-8  # Add epsilon to prevent div zero

        obs = np.concatenate([rolling_mean, rolling_std, self.current_weights]).astype(np.float32)
        return obs

    def step(self, action):
        # Normalize actions using Softmax / L1-norm so weights sum exactly to 1.0
        action = np.clip(action, 1e-5, 1.0)
        target_weights = action / np.sum(action)

        # Calculate Turnover Penalty
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        rebalance_cost = turnover * self.tc

        # Realize Step Returns for time t
        step_returns = self.returns[self.current_step]
        net_portfolio_return = np.dot(target_weights, step_returns) - rebalance_cost

        # Update Portfolio Value
        self.portfolio_value *= (1.0 + net_portfolio_return)
        self.returns_history.append(net_portfolio_return)
        self.portfolio_values_history.append(self.portfolio_value)

        # Update portfolio weights drift due to market movement
        self.current_weights = target_weights * (1.0 + step_returns)
        self.current_weights /= np.sum(self.current_weights)

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # ML Objective: Sortino Ratio
        # For early steps with insufficient history, use raw return.
        # Once we have enough data, use Sortino ratio as the reward.
        if len(self.returns_history) >= self.sortino_lookback:
            recent_returns = np.array(self.returns_history[-self.sortino_lookback:])
            sortino = compute_sortino_ratio(recent_returns, self.risk_free_rate)
            reward = float(sortino)
        else:
            reward = float(net_portfolio_return)

        # Trade Expectancy: edge per period
        expectancy = compute_trade_expectancy(np.array(self.portfolio_values_history))

        info = {
            "portfolio_value": self.portfolio_value,
            "step_return": net_portfolio_return,
            "weights": target_weights,
            "sortino_ratio": reward,
            "trade_expectancy": expectancy,
            "returns_history": np.array(self.returns_history).copy(),
        }

        return self._get_observation(), reward, terminated, truncated, info