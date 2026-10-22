"""GP-FIRST stage machinery:  terminals -> GP features (target-free) -> [OOD on them] -> per-latent horizon models."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from strategy_system.strategy.moe.features import compute_features
from strategy_system.strategy.moe.outcome.alignment import align_regimes_to_index, build_gp_source
from strategy_system.strategy.moe.outcome.outcomes import build_supervision_targets, fit_state_linregs
from .api import GPFeatureSet, enforce_correlation_screen, prepare_terminals, search_features

_GP_RESERVED_KEYS = ("enabled", "train_fraction", "fit_rows", "input_columns", "normalization_window",
                     "normalization_min_periods", "clip_z", "correlation_threshold")


class GPFeatures(NamedTuple):
    """Output of the GP stage, BEFORE any regime exists."""
    feature_set: GPFeatureSet
    feat_df: pd.DataFrame        # compute_features output (used by tb stage for barrier prices)
    terminals: pd.DataFrame
    features: pd.DataFrame       # selected GP features on every row (gp_0000, ...)


class GPStage(NamedTuple):
    """Everything run_tb and the exporter need (same field names as before)."""
    gp: GPFeatureSet
    horizon_models: dict
    horizon_report: pd.DataFrame
    predictions: pd.DataFrame
    targets: pd.DataFrame
    latent_id: pd.Series
    feat_df: pd.DataFrame
    terminal_names: list
    horizons: list


def fit_gp_features(df: pd.DataFrame, cfg: dict) -> GPFeatures:
    gp_cfg = dict(cfg.get("gp", {}))
    reserved = {k: gp_cfg.pop(k, None) for k in _GP_RESERVED_KEYS}
    feat_df, _ = compute_features(df, **cfg["features"])
    ts_col = (cfg.get("ood") or {}).get("timestamp_col", "timestamp")
    source = build_gp_source(df, feat_df, ts_col)
    terminals = prepare_terminals(
        source, input_columns=reserved["input_columns"],
        normalization_window=reserved["normalization_window"] or 500,
        normalization_min_periods=reserved["normalization_min_periods"] or 50,
        clip_z=reserved["clip_z"] or 8.0)
    # search only on the leading rows when fit_rows is set (leak guard); features are evaluated on ALL rows
    fs = search_features(terminals, gp_cfg, fit_rows=reserved["fit_rows"] or reserved["train_fraction"])
    if not fs.names:
        raise ValueError("GP selected no features; relax gp.threshold / gp.gate")
    fs = enforce_correlation_screen(
        fs, terminals, threshold=float(reserved["correlation_threshold"] or 0.5)
    )
    fs.normalization = {
        "normalization_window": reserved["normalization_window"] or 500,
        "normalization_min_periods": reserved["normalization_min_periods"] or 50,
        "clip_z": reserved["clip_z"] or 8.0,
    }
    return GPFeatures(fs, feat_df, terminals, fs.transform(terminals))


def fit_horizon_models(df: pd.DataFrame, gpf: GPFeatures, regimes: pd.DataFrame, cfg: dict) -> GPStage:
    """After OOD: per-latent linear t+1..t+n predictors on the GLOBAL GP feature set."""
    targets = build_supervision_targets(df, gpf.feat_df, cfg.get("outcome"))
    return_targets = targets.loc[:, [c for c in targets.columns if c.startswith("future_return__")]]
    if return_targets.empty:
        raise ValueError("no future_return__* targets were built; check outcome.forward_returns")
    horizons = [c.split("__", 1)[1] for c in return_targets.columns]
    latent_id = align_regimes_to_index(regimes, gpf.feat_df.index)["latent_id"]
    gp_cfg = cfg.get("gp", {})
    frac = gp_cfg.get("fit_rows") or gp_cfg.get("train_fraction")
    mask = np.ones(len(gpf.feat_df), dtype=bool)
    if frac:
        k = int(len(mask) * frac) if frac < 1 else int(frac)
        mask[k:] = False
    train_mask = pd.Series(mask, index=gpf.feat_df.index)
    model_cfg = dict((cfg.get("outcome") or {}).get("model", {}))
    lin = fit_state_linregs(
        gpf.features, return_targets, latent_id, train_mask=train_mask,
        feature_columns=list(gpf.features.columns), price_col=None,
        ridge=float(model_cfg.get("ridge", 1.0e-6)),
        min_state_rows=int(model_cfg.get("min_state_rows", 60)), purge_rows=0)
    return GPStage(gp=gpf.feature_set, horizon_models=lin.models, horizon_report=lin.report,
                   predictions=lin.predictions, targets=targets, latent_id=latent_id,
                   feat_df=gpf.feat_df, terminal_names=list(gpf.terminals.columns), horizons=horizons)