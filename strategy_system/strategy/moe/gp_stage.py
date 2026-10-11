"""GP stage machinery (everything ``run_gp`` delegates to).

features -> supervision targets -> per-latent GP search -> per-latent horizon
predictor (t+1..t+n) refit on the same training rows.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .features import compute_features
from .gp import GPEvolutionResult, extrapolate_features
from .gp.evolution import build_gp_inputs
from .outcome import (
    align_regimes_to_index,
    build_gp_source,
    build_supervision_targets,
    fit_state_linregs,
)

_GP_RESERVED_KEYS = ("enabled", "target_columns", "train_fraction", "purge_rows")


class GPStage(NamedTuple):
    """Everything run_tb and the exporter need from the GP stage."""

    gp: GPEvolutionResult
    horizon_models: dict            # latent_id -> StateLinregModel (gp features -> t+1..t+n)
    horizon_report: pd.DataFrame
    predictions: pd.DataFrame       # future_return__<h> predictions, only on each latent's own rows
    targets: pd.DataFrame           # future_return__<h> / future_timestamp__<h> (targets only)
    latent_id: pd.Series            # latent label per row, aligned to the feature index
    feat_df: pd.DataFrame
    terminal_names: list            # GP terminal (x__...) names, for the formula export
    horizons: list                  # ["1", "2", ...] suffixes of future_return__<h>


# =========================================================================== #
# TRAINING MECHANISM  --  the ONE place that decides which rows GP trains on.
# Currently: ALL episodes of every latent (every row), no holdout.
# To swap (walk-forward, per-episode CV, last-N-episodes, ...): edit this
# function only. ``gp.train_fraction`` < 1 restores a chronological split with
# the GP validation veto; ``None`` (default) = all episodes.
# =========================================================================== #
def select_training_rows(index: pd.Index, latent_id: pd.Series, train_fraction: float | None) -> pd.Series:
    mask = np.ones(len(index), dtype=bool)
    if train_fraction is not None and 0.0 < float(train_fraction) < 1.0:
        mask[int(len(index) * float(train_fraction)):] = False
    return pd.Series(mask, index=index)


def fit_latent_gp(df: pd.DataFrame, regimes: pd.DataFrame, cfg: dict) -> GPStage:
    """Build features/targets, run the per-latent GP search, fit the horizon predictors."""
    gp_cfg = dict(cfg.get("gp", {}))
    reserved = {k: gp_cfg.pop(k, None) for k in _GP_RESERVED_KEYS}

    feat_df, _ = compute_features(df, **cfg["features"])
    targets = build_supervision_targets(df, feat_df, cfg.get("outcome"))
    return_targets = targets.loc[:, [c for c in targets.columns if c.startswith("future_return__")]]
    if return_targets.empty:
        raise ValueError("no future_return__* targets were built; check outcome.forward_returns")
    horizons = [c.split("__", 1)[1] for c in return_targets.columns]

    latent_id = align_regimes_to_index(regimes, feat_df.index)["latent_id"]
    ts_col = (cfg.get("ood") or {}).get("timestamp_col", "timestamp")
    source = build_gp_source(df, feat_df, ts_col)

    train_mask = select_training_rows(feat_df.index, latent_id, reserved["train_fraction"])
    train_end_time = None
    if not train_mask.all() and isinstance(feat_df.index, pd.DatetimeIndex):
        train_end_time = feat_df.index[int(np.flatnonzero(~train_mask.to_numpy())[0])]

    # threads (pairs run in a pool; GIL-free numba kernels) + JIT: see gp/evolution.py
    gp_result = extrapolate_features(
        source, targets, latent_id, train_mask,
        target_columns=reserved["target_columns"],
        train_end_time=train_end_time,
        purge_rows=reserved["purge_rows"],
        **gp_cfg,
    )
    if gp_result.features.empty or not gp_result.model.selected:
        raise ValueError("GP selected no features for any latent; relax the gp.* gates")

    terminals = build_gp_inputs(
        source,
        input_columns=gp_cfg.get("input_columns"),
        normalization_window=gp_cfg.get("normalization_window", 500),
        normalization_min_periods=gp_cfg.get("normalization_min_periods", 50),
        clip_z=gp_cfg.get("clip_z", 8.0),
    )

    model_cfg = dict((cfg.get("outcome") or {}).get("model", {}))
    lin = fit_state_linregs(
        gp_result.features, return_targets, latent_id,
        train_mask=train_mask,                       # same TRAINING MECHANISM as GP
        feature_columns=list(gp_result.features.columns),
        price_col=None,
        ridge=float(model_cfg.get("ridge", 1.0e-6)),
        min_state_rows=int(model_cfg.get("min_state_rows", 60)),
        purge_rows=0,
    )
    return GPStage(
        gp=gp_result, horizon_models=lin.models, horizon_report=lin.report,
        predictions=lin.predictions, targets=targets, latent_id=latent_id,
        feat_df=feat_df, terminal_names=list(terminals.columns), horizons=horizons,
    )