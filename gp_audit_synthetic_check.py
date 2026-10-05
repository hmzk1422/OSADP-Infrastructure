"""Synthetic GP smoke check: DatetimeIndex + power operator + walk-forward."""
import numpy as np
import pandas as pd
from strategy_system.strategy.MoE.ofmr_scalp.gp.grammar import Node, evaluate_expression
from strategy_system.strategy.MoE.ofmr_scalp.gp.evaluation import test_feature_set
from strategy_system.strategy.MoE.ofmr_scalp.gp import extrapolate_features

rng = np.random.default_rng(7)
n = 400
idx = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=n, freq="s"))
frame = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)}, index=idx)

# 1. power node must not raise AttributeError
power_node = Node("power", (Node("var", (), "x0"), Node("var", (), "x1")))
vals = evaluate_expression(power_node, frame)
print("power node OK, finite:", int(np.isfinite(vals.to_numpy(dtype=float)).sum()), "/", n)

# 2. walk-forward on DatetimeIndex must not raise
y = frame["x0"].shift(-1).fillna(0).to_numpy() * 0.5 + rng.normal(scale=0.1, size=n)
res = test_feature_set(frame[["x0"]], y, model_type="ridge", purge_rows=2, min_samples=30)
print("walk-forward OK: expectancy=", res["expectancy"], "stability=", res["stability"])

# 3. tiny end-to-end GP run with DatetimeIndex (exercises evolve_population path)
targets = pd.DataFrame({"tgt": y}, index=idx)
states = pd.Series(np.where(np.arange(n) < n // 2, "L1", "L2"), index=idx)
train_mask = pd.Series(np.arange(n) < int(n * 0.7), index=idx)
out = extrapolate_features(
    frame, targets, states, train_mask,
    target_columns=["tgt"], population_size=6, generations=2, max_height=3,
    min_state_samples=20, block_count=3, min_block_samples=10, min_valid_blocks=2,
    evaluation_models=["ridge", "ols"], seed=7, purge_rows=2,
)
print("extrapolate_features OK, report rows:", len(out.report))
print("selected:", len(out.model.selected))
print("library regimes:", list((out.feature_library or {}).get("regimes", {}).keys()))
print("ALL SYNTHETIC CHECKS PASSED")
