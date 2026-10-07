"""Temporary diagnostics for the linreg state-fit failure (deleted after)."""
import pandas as pd

from strategy_system.strategy.moe.strategy import (
    load_data,
    load_settings,
    run_gp,
    run_ood,
)
from strategy_system.strategy.moe.features import compute_features
from strategy_system.strategy.moe.outcome import (
    build_model_frame,
    build_supervision_targets,
    chronological_train_mask,
)

cfg = load_settings()
cfg["data"]["orderbook"]["file_limit"] = 1

raw = load_data(cfg)
feat_df, feat_cols = compute_features(raw, **cfg["features"])
regimes = run_ood(raw, cfg)
targets = build_supervision_targets(raw, feat_df, cfg.get("outcome"))
gp_result = run_gp(raw, feat_df, targets, regimes, cfg)
model_frame, _regimes, _gp_cols = build_model_frame(
    raw, feat_df, regimes, gp_result.model if gp_result is not None else None, targets=targets
)

model_cfg = cfg["outcome"]["model"]
train_mask = chronological_train_mask(
    model_frame, float(model_cfg["train_fraction"]), int(model_cfg["purge_rows"])
)
print("model_frame:", model_frame.shape, "index:", type(model_frame.index).__name__)
print("raw index:", type(raw.index).__name__, raw.index[:2].tolist())
print("feat_df index:", type(feat_df.index).__name__, feat_df.index[:2].tolist())
print("regimes index:", type(regimes.index).__name__, regimes.index[:2].tolist())
print("regimes shape:", regimes.shape)
print("regimes latent_id (unaligned):")
print(regimes["latent_id"].value_counts(dropna=False).to_string())
overlap = feat_df.index.intersection(regimes.index)
print("index overlap:", len(overlap), "/", len(feat_df))
print("regimes columns:", list(regimes.columns))
print("regimes status:")
print(regimes["status"].value_counts(dropna=False).to_string())
print("embedding_ready:", int(regimes["embedding_ready"].sum()), "/", len(regimes))
print("known_state_count max:", regimes["known_state_count"].max())
print("similarity describe:")
print(regimes["similarity"].describe().to_string())
print("raw_latent_id:")
print(regimes["raw_latent_id"].value_counts(dropna=False).head(12).to_string())
print("train_mask True:", int(train_mask.sum()), "/", len(train_mask))
print("latent_id counts:")
print(model_frame["latent_id"].value_counts(dropna=False).to_string())
print("targets columns:", list(targets.columns))
print("future_timestamp dtypes:", targets.filter(like="future_timestamp").dtypes.to_dict())

endpoint_columns = [c for c in targets if c.startswith("future_timestamp__")]
print("endpoint_columns:", endpoint_columns)
if endpoint_columns:
    boundary_pos = int((~train_mask.to_numpy()).argmax())
    print("boundary index label:", model_frame.index[boundary_pos])
print("DIAG DONE")
