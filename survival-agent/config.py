"""
Central configuration for the Survival Agent study.

This study answers one question: can a reinforcement learning agent learn to
SURVIVE (and ideally profit) in a hostile, non-stationary market where the
house has a built-in edge (commission drag) and the true bias is hidden under
noise and shifts back and forth between regimes?

The agent's ONLY job is risk management: choose a capital size each step.
Cash is a position. The optimal size is often 0.
"""

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Random walk / market generator
# ---------------------------------------------------------------------------
@dataclass
class MarketConfig:
    n_steps: int = 2000                 # steps per episode
    base_vol: float = 0.01              # per-step noise std (1%)
    # Regime biases expressed in return units. A "0.10 RR" drag means
    # 0.10 * base_vol per step.
    regime_biases: List[float] = field(
        default_factory=lambda: [-0.10 * 0.01, 0.0, 0.10 * 0.01]
    )                                   # negative / neutral / positive
    regime_labels: List[str] = field(
        default_factory=lambda: ["negative", "neutral", "positive"]
    )
    # Markov switching probabilities (stay probability per regime).
    # Lower = more frequent regime flips (more non-stationary).
    stay_probs: List[float] = field(
        default_factory=lambda: [0.985, 0.985, 0.985]
    )
    start_regime: int = 1               # start neutral

    # The "variables" the agent can read. These are noisy, lagged estimates
    # of the true bias. They are informative but NEVER perfect, because the
    # market is non-stationary (the bias shifts under the agent's feet).
    ema_short_span: int = 20
    ema_long_span: int = 100
    vol_span: int = 50
    momentum_span: int = 50


# ---------------------------------------------------------------------------
# Trading / cost model
# ---------------------------------------------------------------------------
@dataclass
class TradingConfig:
    # Commission drag: 0.10 RR per step you are in the market.
    # In return units this is 0.10 * base_vol.
    commission_rr: float = 0.10
    # Discrete capital-size actions the agent can choose.
    # 0.0 = fully in cash (the survival move).
    actions: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 2.0]
    )
    # Leverage cap (2.0 above means up to 2x).
    max_leverage: float = 2.0


# ---------------------------------------------------------------------------
# Survival-first reward / episode termination
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    # Ruin: if wealth drops below this fraction of initial, episode ends
    # with a large negative penalty. This is the "survival" hard constraint.
    ruin_fraction: float = 0.50
    # Drawdown penalty kicks in once drawdown exceeds this fraction.
    dd_penalty_threshold: float = 0.20
    # Per-step drawdown penalty coefficient (scaled by excess drawdown).
    dd_penalty_coef: float = 3.0
    # Terminal bonus for surviving the full episode (reward for survival).
    survival_bonus: float = 0.5
    # Penalty for hitting ruin.
    ruin_penalty: float = -5.0
    # Risk-aversion on per-step log-return (penalizes variance of wealth).
    # Higher = more conservative.
    risk_aversion: float = 1.0


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    # Number of bins per continuous state variable (discretization).
    n_bins: int = 4
    # Which state variables to use (must match env observation order).
    use_vars: List[str] = field(
        default_factory=lambda: [
            "ema_short",
            "ema_long",
            "vol",
            "momentum",
            "drawdown",
            "wealth_norm",
        ]
    )
    alpha: float = 0.1                  # learning rate
    gamma: float = 0.99                 # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995       # per episode
    n_episodes: int = 3000
    # Reward shaping: scale the raw reward before learning.
    reward_scale: float = 1.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    n_seeds: int = 200                  # number of independent market seeds
    eval_episodes_per_seed: int = 1
    # Baselines to compare against.
    baselines: List[str] = field(
        default_factory=lambda: [
            "always_cash",      # never trade (pure survival)
            "always_full",      # always 1x (naive active trader)
            "always_2x",        # always 2x (reckless)
            "fixed_half",       # always 0.5x
            "ema_rule",         # simple rule: trade if ema_short > 0
        ]
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    output_dir: str = "survival-agent/outputs"
    model_path: str = "survival-agent/outputs/qtable.npy"
    results_csv: str = "survival-agent/outputs/results.csv"
    summary_json: str = "survival-agent/outputs/summary.json"
    plot_equity: str = "survival-agent/outputs/equity_curves.png"
    plot_learning: str = "survival-agent/outputs/learning_curve.png"
    plot_regimes: str = "survival-agent/outputs/regime_example.png"
    plot_heatmap: str = "survival-agent/outputs/qtable_heatmap.png"