"""OFMR scalp MoE pipeline: Rust GP -> OOD -> horizon models -> TB.

The GP search/evaluation is native Rust; this Python module wires strategy stages.
Output: moe/models/{regime/, latent_<id>/, manifest.json}  (the Rust contract).

    python -m strategy_system.strategy.moe.strategy [settings.yaml]
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import pandas as pd

from ...exchange.loader import load_L2
from .ops.export import (
    export_gp_features,
    export_latent,
    export_manifest,
    export_regime_bundle,
    models_dir_from_cfg,
    verify_regime_contract,
)
from components.pipelines.gp.gp_stage import (
    GPFeatures,
    GPStage as RustGPStage,
    fit_gp_features,
    fit_horizon_models,
)
from .ops.settings import load_settings
from .ood import OODSystem
from .outcome.tb_stage import ERR_RULE, build_tb_base_frame, fit_latent_tb, tb_params


class StrategyResult(NamedTuple):
    regimes: pd.DataFrame      # filter with regimes[regimes["latent_id"] == "1"]
    gp: RustGPStage
    tb_report: pd.DataFrame    # one row per latent: exported / skipped + reason
    manifest_path: object      # Path | None


def run_gp_new(df: pd.DataFrame, cfg: dict) -> GPFeatures:
    """Run the native Rust GP before OOD and persist its formula/tree library."""
    gpf = fit_gp_features(df, cfg)
    gpf.feature_set.save(models_dir_from_cfg(cfg) / "gp" / "library.json")
    return gpf


def run_ood(df: pd.DataFrame, gpf: GPFeatures, cfg: dict) -> tuple[pd.DataFrame, OODSystem]:
    """Fit OOD on Rust-GP features, not on raw market columns."""
    ood_cfg = dict(cfg.get("ood", {}))
    ts_col = ood_cfg.get("timestamp_col") or "timestamp"
    ood_cfg["primitive_columns"] = list(gpf.features.columns)
    frame = gpf.features.copy()
    if ts_col in df:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame[ts_col] = frame.index
        elif frame.index.isin(df.index).all():
            frame[ts_col] = df[ts_col].reindex(frame.index).to_numpy()
        else:
            raise ValueError("cannot align raw timestamps to Rust GP feature rows")
    detector = OODSystem(**ood_cfg)
    regimes = detector.detect(frame)
    verify_regime_contract(detector, frame, regimes)
    export_regime_bundle(models_dir_from_cfg(cfg), detector)
    return regimes, detector


def run_horizon(
    df: pd.DataFrame, gpf: GPFeatures, regimes: pd.DataFrame, cfg: dict
) -> RustGPStage:
    """Fit downstream return predictors from the globally selected Rust features."""
    stage = fit_horizon_models(df, gpf, regimes, cfg)
    models_dir = models_dir_from_cfg(cfg)
    for latent, horizon_model in stage.horizon_models.items():
        export_gp_features(
            models_dir, latent, stage.gp, horizon_model,
            stage.terminal_names, stage.horizons,
        )
    return stage


def run_tb(df: pd.DataFrame, gp_stage: RustGPStage, cfg: dict) -> tuple[pd.DataFrame, object]:
    """
    tb

    hypothesis:
    Given a latent, the t+1..t+n predictions and the recent forecast errors
    (realized - predicted) carry enough information for tb_regressor to size
    TP/SL barriers (MFE_R / MAE_R) better than a fixed width.

    assumptions:
    Features are ONLY pred__<h> and err__<h> for the latent's horizons. err__<h>
    is the most recent ELAPSED same-latent residual (never the not-yet-known
    actual), 0.0 until one exists. Training uses all episodes of the latent
    (TB_TRAIN_FRACTION_ALL in tb_stage.py). Latents below moe.min_latent_rows
    are skipped and reported. Each latent exports tb_regressor.onnx +
    metadata.json; manifest.json indexes the experts for the Rust router.
    Every latent is fitted on the SHARED full base frame (barrier paths, the
    train split and excursions span the global row timeline -- never the
    latent's own subset), so fits are independent and run on a thread pool:
    moe.tb_workers threads (default: min(latents, cpu_count, 4); 1 = serial).

    args:
    =================================
    df: raw table from load_L2 (price / high / low columns for the barriers).
    gp_stage: run_gp output.
    cfg: full settings mapping (reads "outcome.triple_barrier", "moe").
    returns: (report DataFrame, manifest path or None).
    =================================
    """
    models_dir = models_dir_from_cfg(cfg)
    params = tb_params(cfg)
    moe_cfg = cfg.get("moe") or {}
    min_rows = int(moe_cfg.get("min_latent_rows", 200))
    base = build_tb_base_frame(df, gp_stage.feat_df, params)
    latent_ids = gp_stage.latent_id.astype("object")  # once, not once per latent
    latents = list(gp_stage.horizon_models)

    def _fit_one(latent: str):
        """Fit one latent on the shared frame; returns (row, fit|None)."""
        n_rows = int((latent_ids == latent).sum())
        if n_rows < min_rows:
            return {"latent_id": latent,
                    "status": f"skipped: {n_rows} rows < min_latent_rows={min_rows}"}, None
        try:
            fit = fit_latent_tb(base, gp_stage.predictions, gp_stage.targets,
                                gp_stage.latent_id, latent, gp_stage.horizons, params)
        except (ValueError, KeyError) as exc:
            return {"latent_id": latent, "status": f"skipped: {exc}"}, None
        return {"latent_id": latent, "status": "exported",
                "n_rows": n_rows, "n_fit": fit.n_fit}, fit

    workers = int(moe_cfg.get("tb_workers", 0) or 0)
    if workers <= 0:
        workers = min(len(latents) or 1, os.cpu_count() or 1, 4)
    workers = max(1, min(workers, len(latents) or 1))
    if workers > 1 and latents:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fitted = list(pool.map(_fit_one, latents))
    else:
        fitted = [_fit_one(latent) for latent in latents]

    rows, exported = [], []
    for latent, (row, fit) in zip(latents, fitted):
        if fit is None:
            rows.append(row)
            continue
        try:
            export_latent(models_dir, latent, tp_model=fit.tp_model, sl_model=fit.sl_model,
                          tb_feature_order=fit.feature_order, horizons=gp_stage.horizons,
                          barrier=fit.barrier, err_rule=ERR_RULE, n_fit=fit.n_fit,
                          parity_x=fit.parity_x)
        except (ValueError, KeyError) as exc:
            rows.append({"latent_id": latent, "status": f"skipped: {exc}"})
            continue
        exported.append(latent)
        rows.append(row)

    manifest = export_manifest(models_dir, exported, {"horizons": gp_stage.horizons, "tb": params}) if exported else None
    return pd.DataFrame(rows), manifest


def strategy(df: pd.DataFrame, settings: dict | None = None) -> StrategyResult:
    cfg = load_settings() if settings is None else settings
    gp_features = run_gp_new(df, cfg)
    regimes, _ = run_ood(df, gp_features, cfg)
    gp_stage = run_horizon(df, gp_features, regimes, cfg)
    report, manifest = run_tb(df, gp_stage, cfg)
    return StrategyResult(regimes, gp_stage, report, manifest)


if __name__ == "__main__":
    settings = load_settings(sys.argv[1] if len(sys.argv) > 1 else None)
    data = load_L2(**settings["data"].get("orderbook", {}))
    result = strategy(data, settings)
    print(result.regimes["latent_id"].value_counts(dropna=False).to_string())
    print(result.tb_report.to_string(index=False))
    print(f"manifest: {result.manifest_path}")