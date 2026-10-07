"""Temporary S3E refactor smoke test (deleted after the run)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_system.strategy.moe.strategy import (
    StrategyResult,
    load_settings,
    run_triple_barrier,
)

# --- 1. Public API + settings mapping (S4) ------------------------------------
assert StrategyResult._fields == (
    "features", "regimes", "events", "outcomes", "policy", "gp", "metalabel",
), StrategyResult._fields
cfg = load_settings()
assert "model" in cfg["outcome"], "outcome.model section missing"
assert "example" not in cfg, "legacy [example] section still present"
for key in (
    "feature_columns", "metalabel_features", "train_fraction", "purge_rows",
    "ridge", "min_state_rows", "side", "tree_max_depth",
    "tree_min_samples_leaf", "tree_min_train_rows", "seed",
):
    assert key in cfg["outcome"]["model"], f"outcome.model.{key} missing"
print("API + settings mapping OK")

# --- Synthetic model frame ----------------------------------------------------
n = 600
rng = np.random.default_rng(7)
idx = pd.date_range("2025-01-01", periods=n, freq="100ms", tz="UTC")
price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 1e-4, n))
frame = pd.DataFrame(
    {
        "price_close": price,
        "htf_rsi_d2_signal": rng.choice([-1, 0, 1], n, p=[0.4, 0.2, 0.4]),
        "feature_a": rng.normal(0.0, 1.0, n),
        "latent_id": np.where(np.arange(n) < n // 2, "state-a", "state-b"),
        "similarity": rng.uniform(0.5, 1.0, n),
        "trade_allowed": rng.random(n) < 0.3,
        "state_mature": True,
        "model_ready": True,
    },
    index=idx,
)
targets = pd.DataFrame(
    {
        "future_return__1s": rng.normal(0.0, 1e-3, n),
        "future_return__5s": rng.normal(0.0, 1e-3, n),
        "future_timestamp__1s": idx + pd.Timedelta("1s"),
        "future_timestamp__5s": idx + pd.Timedelta("5s"),
    },
    index=idx,
)

ANN = ("latent_id", "ood_similarity", "trade_allowed", "state_mature", "model_ready")

# build_model_frame joins targets alongside; mirror that assembly here.
model_frame = frame.join(targets)

# --- 2. Outcome path (run_outcomes composition) -------------------------------
outcome = run_triple_barrier(model_frame, targets, cfg)
assert not outcome.events.empty, "outcome path produced no events"
assert all(column in outcome.events.columns for column in ANN), "event annotation missing"
assert not outcome.report.empty, "linreg report empty"
print(
    f"outcome path OK: {len(outcome.events)} events, report {len(outcome.report)} rows, "
    f"metalabel {'fitted' if outcome.metalabel is not None else 'unfit'}"
)

# --- 3. Legacy barriers path --------------------------------------------------
legacy_cfg = {
    "barriers": {
        "signal_col": "htf_rsi_d2_signal",
        "price_col": "price_close",
        "pt": 0.005,
        "sl": 0.003,
        "horizon_bars": 30,
    }
}
legacy = run_triple_barrier(model_frame, pd.DataFrame(index=idx), legacy_cfg)
assert not legacy.events.empty, "legacy path produced no events"
assert legacy.metalabel is None, "legacy path must not fit a metalabel"
assert all(column in legacy.events.columns for column in ANN), "legacy annotation missing"
print(f"legacy barriers OK: {len(legacy.events)} events")

print("SMOKE PASSED")
