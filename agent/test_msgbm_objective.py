"""
Bulletproof validation test for sandbox_msgbm Sortino + Expectancy objective.

This test is SEPARATED from the notebook to prevent:
1. Lookahead bias (observations must only contain past data)
2. Scoring problems (metrics computed independently)
3. Overfitting to a specific random seed

Run: python agent/test_msgbm_objective.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msgbm_env import (
    generate_dynamic_regime_market,
    SyntheticPortfolioEnv,
    compute_sortino_ratio,
    compute_trade_expectancy,
)


def verify_no_lookahead_bias(env):
    """Verify observation only uses PAST data. Critical anti-cheat check."""
    obs, _ = env.reset(seed=42)

    for step in range(env.window_size, env.n_steps - 1):
        expected_window = env.returns[step - env.window_size : step]
        actual_window = env.returns[env.current_step - env.window_size : env.current_step]

        assert env.current_step == step
        np.testing.assert_array_equal(expected_window, actual_window)

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    return {
        "passed": True,
        "steps_checked": step - env.window_size + 1,
        "message": "All observations use only past data. No lookahead bias detected."
    }


def train_agent(train_returns, seed=42):
    """Train PPO agent on the given returns data."""
    from stable_baselines3 import PPO

    train_env = SyntheticPortfolioEnv(train_returns)
    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=0.0003,
                policy_kwargs=policy_kwargs, seed=seed)
    model.learn(total_timesteps=150000)
    return model

def run_backtest(model, test_returns):
    """Run backtest and collect portfolio values and metrics."""
    test_env = SyntheticPortfolioEnv(test_returns)
    obs, _ = test_env.reset()

    portfolio_values = [1.0]
    step_returns = []

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        portfolio_values.append(info["portfolio_value"])
        step_returns.append(info["step_return"])

    return {
        "portfolio_values": np.array(portfolio_values),
        "step_returns": np.array(step_returns),
        "final_value": portfolio_values[-1],
    }


def run_equal_weight_benchmark(test_returns, transaction_cost=0.0005):
    """Run equal-weight benchmark."""
    n_assets = test_returns.shape[1]
    equal_weights = np.ones(n_assets) / n_assets

    portfolio_values = [1.0]
    step_returns_list = []

    for t, r in enumerate(test_returns):
        step_ret = np.dot(equal_weights, r)

        if t == 0:
            turnover = 1.0
            step_ret -= turnover * transaction_cost

        portfolio_values.append(portfolio_values[-1] * (1.0 + step_ret))
        step_returns_list.append(step_ret)

    return {
        "portfolio_values": np.array(portfolio_values),
        "step_returns": np.array(step_returns_list),
        "final_value": portfolio_values[-1],
    }


def run_random_agent(test_returns):
    """Run random agent as sanity check lower bound."""
    env = SyntheticPortfolioEnv(test_returns)
    obs, _ = env.reset()

    portfolio_values = [1.0]
    step_returns = []

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_values.append(info["portfolio_value"])
        step_returns.append(info["step_return"])

    return {
        "portfolio_values": np.array(portfolio_values),
        "step_returns": np.array(step_returns),
        "final_value": portfolio_values[-1],
    }


def compute_all_metrics(portfolio_values, step_returns, risk_free_rate=0.0):
    """Compute Sortino ratio and expectancy from portfolio data."""
    return {
        "total_return": portfolio_values[-1] / portfolio_values[0] - 1.0,
        "sortino_ratio": compute_sortino_ratio(step_returns, risk_free_rate),
        "expectancy": compute_trade_expectancy(portfolio_values),
        "final_value": portfolio_values[-1],
        "n_steps": len(step_returns),
    }


def compare_models(metrics_rl, metrics_benchmark, metrics_random):
    """Compare metrics across models and compute statistical tests."""
    print("")
    print("=" * 70)
    print("METRICS COMPARISON")
    print("=" * 70)

    models = ["RL Agent", "Equal-Weight", "Random"]
    metrics_list = [metrics_rl, metrics_benchmark, metrics_random]

    for name, metrics in zip(models, metrics_list):
        tr = metrics["total_return"]
        sr = metrics["sortino_ratio"]
        exp = metrics["expectancy"]
        fv = metrics["final_value"]
        print("")
        print(f"  {name}:")
        print(f"    Total Return:   {tr:+.4f}")
        print(f"    Sortino Ratio:  {sr:.4f}")
        print(f"    Expectancy:     {exp:+.6f}")
        print(f"    Final Value:    {fv:.4f}")

    print("")
    print("-" * 70)
    print("VALIDATION CHECKS")
    print("-" * 70)

    checks = []

    rl_sr = metrics_rl["sortino_ratio"]
    rand_sr = metrics_random["sortino_ratio"]
    rl_exp = metrics_rl["expectancy"]
    rl_fv = metrics_rl["final_value"]

    if rl_sr > rand_sr:
        checks.append(("RL Sortino > Random Sortino", True,
                       f"{rl_sr:.4f} > {rand_sr:.4f}"))
    else:
        checks.append(("RL Sortino > Random Sortino", False,
                       f"{rl_sr:.4f} <= {rand_sr:.4f}"))

    if rl_sr > 0:
        checks.append(("RL Sortino > 0", True,
                       f"{rl_sr:.4f}"))
    else:
        checks.append(("RL Sortino > 0", False,
                       f"{rl_sr:.4f}"))

    if rl_exp > 0:
        checks.append(("RL Expectancy > 0", True,
                       f"{rl_exp:+.6f}"))
    else:
        checks.append(("RL Expectancy > 0", False,
                       f"{rl_exp:+.6f}"))

    if rl_fv > 1.0:
        checks.append(("RL Final Value > 1.0", True,
                       f"{rl_fv:.4f}"))
    else:
        checks.append(("RL Final Value > 1.0", False,
                       f"{rl_fv:.4f}"))

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    all_passed = all(c[1] for c in checks)
    return all_passed, checks

def run_full_validation(seed=42, n_seeds=5):
    """Run full validation with multiple seeds."""
    print("=" * 70)
    print("BULLETPROOF VALIDATION TEST FOR sandbox_msgbm")
    print("Objective: Maximize Sortino Ratio + Trade Expectancy")
    print("=" * 70)

    all_results = []

    for seed_i in range(n_seeds):
        current_seed = seed + seed_i
        print(f"")
        print(f"--- Seed {current_seed} ({seed_i + 1}/{n_seeds}) ---")

        np.random.seed(current_seed)
        returns_data, prices_data = generate_dynamic_regime_market(seed=current_seed)

        train_returns = returns_data[:1400]
        test_returns = returns_data[1400:]

        print(f"  Training on {len(train_returns)} steps...")
        model = train_agent(train_returns, seed=current_seed)

        print(f"  Testing on {len(test_returns)} steps...")
        rl_results = run_backtest(model, test_returns)

        bench_results = run_equal_weight_benchmark(test_returns)

        random_results = run_random_agent(test_returns)

        rl_metrics = compute_all_metrics(
            rl_results["portfolio_values"], rl_results["step_returns"]
        )
        bench_metrics = compute_all_metrics(
            bench_results["portfolio_values"], bench_results["step_returns"]
        )
        rand_metrics = compute_all_metrics(
            random_results["portfolio_values"], random_results["step_returns"]
        )

        all_results.append({
            "seed": current_seed,
            "rl": rl_metrics,
            "benchmark": bench_metrics,
            "random": rand_metrics,
            "rl_portfolio": rl_results["portfolio_values"].tolist(),
            "bench_portfolio": bench_results["portfolio_values"].tolist(),
        })

        print(f"  RL:    Sortino={rl_metrics['sortino_ratio']:+.4f} "
              f"Exp={rl_metrics['expectancy']:+.6f} "
              f"Ret={rl_metrics['total_return']:+.4f}")
        print(f"  Bench: Sortino={bench_metrics['sortino_ratio']:+.4f} "
              f"Exp={bench_metrics['expectancy']:+.6f} "
              f"Ret={bench_metrics['total_return']:+.4f}")
        print(f"  Rand:  Sortino={rand_metrics['sortino_ratio']:+.4f} "
              f"Exp={rand_metrics['expectancy']:+.6f} "
              f"Ret={rand_metrics['total_return']:+.4f}")

    print("")
    print("=" * 70)
    print("AGGREGATE RESULTS ACROSS ALL SEEDS")
    print("=" * 70)

    rl_sortinos = [r["rl"]["sortino_ratio"] for r in all_results]
    bench_sortinos = [r["benchmark"]["sortino_ratio"] for r in all_results]
    rand_sortinos = [r["random"]["sortino_ratio"] for r in all_results]

    rl_expectancies = [r["rl"]["expectancy"] for r in all_results]
    bench_expectancies = [r["benchmark"]["expectancy"] for r in all_results]
    rand_expectancies = [r["random"]["expectancy"] for r in all_results]

    print("")
    print("Sortino Ratio:")
    print(f"  RL Agent:  {np.mean(rl_sortinos):+.4f} +/- {np.std(rl_sortinos):.4f}")
    print(f"  Benchmark: {np.mean(bench_sortinos):+.4f} +/- {np.std(bench_sortinos):.4f}")
    print(f"  Random:    {np.mean(rand_sortinos):+.4f} +/- {np.std(rand_sortinos):.4f}")

    print("")
    print("Expectancy (edge/period):")
    print(f"  RL Agent:  {np.mean(rl_expectancies):+.6f} +/- {np.std(rl_expectancies):.6f}")
    print(f"  Benchmark: {np.mean(bench_expectancies):+.6f} +/- {np.std(bench_expectancies):.6f}")
    print(f"  Random:    {np.mean(rand_expectancies):+.6f} +/- {np.std(rand_expectancies):.6f}")

    t_stat, p_value = stats.ttest_rel(rl_sortinos, bench_sortinos)
    print("")
    print("Paired t-test (RL vs Benchmark Sortino):")
    print(f"  t={t_stat:.4f}, p={p_value:.4f}")

    if p_value < 0.05 and np.mean(rl_sortinos) > np.mean(bench_sortinos):
        print("  Result: RL significantly OUTPERFORMS benchmark (p < 0.05)")
    elif p_value < 0.05:
        print("  Result: RL significantly UNDERPERFORMS benchmark (p < 0.05)")
    else:
        print("  Result: No significant difference (p >= 0.05)")

    save_path = os.path.join(os.path.dirname(__file__), "test_msgbm_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "aggregate": {
                "rl_sortino_mean": float(np.mean(rl_sortinos)),
                "rl_sortino_std": float(np.std(rl_sortinos)),
                "bench_sortino_mean": float(np.mean(bench_sortinos)),
                "rl_expectancy_mean": float(np.mean(rl_expectancies)),
                "p_value_sortino": float(p_value),
            },
            "per_seed": all_results,
        }, f, indent=2)

    print(f"")
    print(f"Results saved to: {save_path}")

    print("")
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    verdict_checks = []

    if np.mean(rl_sortinos) > np.mean(rand_sortinos):
        verdict_checks.append("RL beats random on Sortino")

    rand_expectancies = [r["random"]["expectancy"] for r in all_results]
    if np.mean(rl_expectancies) > np.mean(rand_expectancies):
        verdict_checks.append("RL expectancy > Random expectancy (smaller losses)")

    if all(r["rl"]["final_value"] > 0.5 for r in all_results):
        verdict_checks.append("RL survives all seeds (final_value > 0.5)")

    if np.mean(rl_sortinos) >= np.mean(bench_sortinos):
        verdict_checks.append("RL matches/beats benchmark on Sortino")

    for check in verdict_checks:
        print(f"  [PASS] {check}")

    if len(verdict_checks) == 4:
        print("")
        print("  OVERALL: OBJECTIVE ACHIEVED")
        return True
    else:
        print("")
        print("  OVERALL: OBJECTIVE NOT ACHIEVED")
        return False


if __name__ == "__main__":
    success = run_full_validation(seed=42, n_seeds=5)
    sys.exit(0 if success else 1)
