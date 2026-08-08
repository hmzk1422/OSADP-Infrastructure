# Survival Agent — Risk Management RL Study

A self-contained research sandbox that answers one question:

> **Can a reinforcement learning agent learn to SURVIVE (and ideally profit) in a hostile, non-stationary market where the house has a built-in edge (commission drag) and the true bias is hidden under noise and shifts back and forth between regimes?**

This is **not** a price-prediction study. The agent's only job is **risk management**: choose a capital size each step. The core lesson it must learn is that **cash is a position** — the optimal size is often `0`.

---

## Why This Formulation Is Mathematically Sound

Unlike trying to predict absolute price movements, this is a well-posed problem:

- The market is a **regime-switching random walk** with a hidden Markov chain bias (negative / neutral / positive).
- The agent pays a **commission drag** proportional to exposure (`0.10 RR` per step in the market).
- The agent reads **noisy, lagged variables** (EMAs, volatility, momentum) that are informative but never perfect, because the market is non-stationary — by the time the variables confirm a regime shift, the regime may already be shifting back.

If an agent can survive and eke out positive expectancy in a simulated environment where the house has a mathematical edge, it has learned a profound risk-management lesson: **when to sit on its hands.**

---

## The #1 Survival Rule

With a negative commission drag, the agent's primary survival mechanism **cannot** be "how to size winning trades." It must be **learning when not to trade**.

If the agent feels compelled to take trades because it's an "active trader," the commission drag chews it to pieces no matter how good its sizing is. A true risk-management agent must learn that **cash is a position** — when statistical confidence of a positive bias is low, the optimal capital size is `0`.

---

## The Non-Stationarity Lag

The variables the agent reads are **lagged**. By the time they say "the bias just turned positive," the agent is looking backward. By the time it scales up, the regime may already be shifting back to negative. In non-stationary environments, reactive risk models are always fighting yesterday's war. This study measures how well the agent copes with that lag.

---

## Architecture

```
survival-agent/
├── config.py        # All hyperparameters (market, trading, reward, agent, eval)
├── market.py        # Regime-switching random walk + noisy observable variables
├── environment.py   # Survival-first Gym-style environment (numpy-only)
├── agent.py         # Q-learning agent + baseline strategies
├── train.py         # Train the agent, save Q-table + learning curve
├── evaluate.py      # Benchmark agent vs baselines across many seeds
├── analyze.py       # Generate plots (equity, learning, regimes, Q-heatmap)
└── outputs/         # Generated results and plots
```

### The Environment

- **State** (observation): short EMA, long EMA, volatility, momentum (all normalized), plus portfolio drawdown and normalized wealth.
- **Action**: discrete capital size — `[0, 0.25, 0.5, 1.0, 2.0]` (0 = fully in cash).
- **Reward** (survival-first, deliberately NOT "maximize profit"):
  ```
  reward = log_return
           - risk_aversion * (log_return - mean)^2   # variance penalty
           - dd_penalty_coef * max(0, drawdown - threshold)  # drawdown penalty
           + survival_bonus (terminal, if you survive the whole episode)
           + ruin_penalty (terminal, if you hit the ruin threshold)
  ```
- **Termination**: episode ends on ruin (wealth < 50% of initial) or after all steps.

### The Agent

A **tabular Q-learning** agent (numpy-only, no heavy dependencies) that discretizes the continuous observation space into bins and learns a Q-table. This is ideal for a first study: the state space is small and interpretable, and it directly demonstrates the "when to sit on your hands" survival lesson. The environment interface is Gym-style, so it can be swapped to Stable-Baselines3 (PPO) later if desired.

### Baselines

| Strategy | Description |
|----------|-------------|
| `always_cash` | Never trade (pure survival) |
| `always_full` | Always 1x (naive active trader) |
| `always_2x` | Always 2x (reckless) |
| `fixed_half` | Always 0.5x |
| `ema_rule` | Trade 1x if short EMA > 0, else cash (simple momentum rule) |

---

## How to Run

All scripts run from the `survival-agent/` directory. **No extra installs needed** — uses only numpy + matplotlib (already in your environment).

```bash
cd survival-agent

# 1. Train the agent (default 3000 episodes)
python train.py

# 2. Evaluate against baselines across 200 market seeds
python evaluate.py

# 3. Generate analysis plots
python analyze.py
```

### Quick smoke test (fast, few episodes)

```bash
python train.py --episodes 200
python evaluate.py --seeds 20
python analyze.py
```

---

## What the Outputs Tell You

- **`outputs/summary.json` / `results.csv`** — the headline table: survival rate, ruin rate, median/p10/p90 final wealth, median max drawdown, median Sharpe for the trained agent vs every baseline.
- **`outputs/equity_curves.png`** — sample equity curves on one market seed (log scale). Watch the reckless `always_2x` blow up while the agent survives.
- **`outputs/learning_curve.png`** — training reward, final wealth, and survival rate over episodes. A rising survival rate is the key signal the agent is learning to sit on its hands.
- **`outputs/regime_example.png`** — an example market showing the hidden true bias vs the lagged observable variables, with regime shading. This visualizes the non-stationarity lag.
- **`outputs/qtable_heatmap.png`** — the learned preferred action per state. A good agent will show large regions where the preferred action is `0x` (cash).

---

## Interpreting the Results (What "Good" Looks Like)

- **Survival rate should be high** (near 1.0). This is the primary objective.
- **Median final wealth should be ≥ 1.0** (you didn't lose money on average despite the commission drag).
- **Max drawdown should be low** (the agent avoids deep holes).
- **The agent should beat `always_cash`** — if it can't beat "do nothing," it hasn't learned to exploit the positive regimes well enough.
- **The agent should massively beat `always_2x`** — the reckless baseline should have a high ruin rate. If it doesn't, the reward function isn't punishing risk hard enough.

---

## Extending the Study

- **Swap to PPO / SB3**: the environment is Gym-style; wrap it and train with Stable-Baselines3 for a continuous-action variant.
- **Tune the reward**: increase `risk_aversion` or `dd_penalty_coef` to make the agent more conservative; lower them to allow more aggression.
- **Change regime dynamics**: lower `stay_probs` for more frequent regime flips (more non-stationary); widen the bias spread for stronger regimes.
- **Add more variables**: e.g., a "regime confidence" feature, or a longer-horizon momentum.
- **Add transaction costs on position changes** (not just per-step exposure) to model turnover drag.

---

## The Verdict

This is ambitious, but testing a risk-management model on synthetic random walks first is the exact right sandbox. **If your risk-management model can't survive a simulated random walk with a negative fee drag, it has zero business touching a live exchange.**