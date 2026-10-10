"""OFMR scalp strategy engine: thin orchestration layer.

Pipeline: features -> OOD states -> supervision targets -> optional GP ->
model frame (S1 contract) -> outcomes / triple barrier.

Column contract (S1): [causal features] + [raw__ primitives] + [gp_feat*]
+ [regime flags] + [targets]. ``future_*`` columns are targets only; GP and
policy never consume them.

Run from the repository root:
    python -m strategy_system.strategy.moe.strategy [path/to/settings.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml
from tqdm import tqdm

from ...exchange.loader import load_L1, load_L2
from .features import compute_features
from .gp import GPEvolutionResult, extrapolate_features
from .ood import OODSystem
from .outcome import (
    MetalabelResult,
    OutcomeResult,
    align_regimes_to_index,
    build_gp_source,
    build_model_frame,
    build_supervision_targets,
    build_triple_barrier_sequence,
    chronological_train_mask,
    run_outcomes,
)
from .policies.orderflow_prediction import format_report

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"
DATA_LOADERS = {"orderbook": load_L2, "ohlcv": load_L1}


class StrategyResult(NamedTuple):
    features: pd.DataFrame              # S1 model frame
    regimes: pd.DataFrame               # causal OOD state assignments
    events: pd.DataFrame                # triple-barrier events (one row per signal)
    outcomes: pd.DataFrame              # forward-return targets
    policy: pd.DataFrame                # policy report (currently disabled -> empty)
    gp: GPEvolutionResult | None
    metalabel: MetalabelResult | None


def load_settings(path: str | Path | None = None) -> dict:
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    with settings_path.open("r", encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    if not isinstance(settings, dict):
        raise ValueError(f"{settings_path} must contain a YAML mapping.")
    return settings


def load_data(settings: dict) -> pd.DataFrame:
    data_cfg = settings["data"]
    source = data_cfg["source"]
    if source not in DATA_LOADERS:
        raise ValueError(f"Unknown data source {source!r}; expected one of {sorted(DATA_LOADERS)}.")
    return DATA_LOADERS[source](**data_cfg.get(source, {}))


def run_gp(raw, feat_df, targets, regimes, cfg) -> GPEvolutionResult | None:
    """Opt-in per-state GP search; returns None unless ``gp.enabled``."""
    gp_cfg = dict(cfg.get("gp", {}))
    if not gp_cfg.pop("enabled", False):
        return None

    ts_col = (cfg.get("ood") or {}).get("timestamp_col", "timestamp")
    source = build_gp_source(raw, feat_df, ts_col)
    target_columns = gp_cfg.pop("target_columns", None)
    train_fraction = float(gp_cfg.pop("train_fraction", 0.70))
    purge_rows = int(gp_cfg.pop("purge_rows", 0))

    split = int(len(feat_df) * train_fraction)
    train_end_time = (
        feat_df.index[split]
        if isinstance(feat_df.index, pd.DatetimeIndex) and split < len(feat_df)
        else None
    )
    train_mask = chronological_train_mask(feat_df, train_fraction, purge_rows)
    latent_id = align_regimes_to_index(regimes, feat_df.index)["latent_id"]

    return extrapolate_features(
        source, targets, latent_id, train_mask,
        target_columns=target_columns,
        train_end_time=train_end_time,
        purge_rows=purge_rows,
        **gp_cfg,
    )


def run_triple_barrier(model_frame, targets, cfg, *, raw=None) -> OutcomeResult:
    """linreg -> metalabel -> barriers, then tag events with their entry state."""
    if cfg.get("outcome") is None:
        # Legacy settings: fixed barriers, no fits.
        barrier_cfg = dict(cfg["barriers"])
        horizon = barrier_cfg.pop("horizon", barrier_cfg.pop("horizon_bars", 30))
        outcome = OutcomeResult(
            predictions=pd.DataFrame(index=model_frame.index),
            excursions=pd.DataFrame(index=model_frame.index),
            metalabel=None,
            events=build_triple_barrier_sequence(
                model_frame, mode="fixed", horizon_bars=horizon, **barrier_cfg
            ),
            report=pd.DataFrame(columns=["latent_id", "n_train", "n_features", "rank"]),
        )
    else:
        outcome = run_outcomes(model_frame, targets, cfg, raw=raw)

    events = outcome.events
    if not events.empty and "t0" in events.columns:
        entry = align_regimes_to_index(model_frame, pd.Index(events["t0"]))
        events["latent_id"] = entry["latent_id"].to_numpy()
        events["ood_similarity"] = entry["similarity"].to_numpy()
        for col in ("trade_allowed", "state_mature", "model_ready"):
            events[col] = entry[col].fillna(False).to_numpy(dtype=bool)
    return outcome


def strategy(
    df: pd.DataFrame,
    settings: dict | None = None,
    ood_detector: OODSystem | None = None,
    ood_process_from: int = 0,
    show_progress: bool = True,
    compare_forecasts: bool = False,
) -> StrategyResult:
    """Run the OFMR pipeline on a raw state table.

    ``ood_detector`` / ``ood_process_from``: pass the same detector across data
    chunks to keep learned latent prototypes (``state_dict()`` persists them);
    ``ood_process_from`` is the first row that should update it (earlier rows
    are warmup context).

    ``compare_forecasts``: also run the linreg/ridge/kalman research harness
    (results on ``strategy.forecast_df`` / ``strategy.barrier_summary``).
    """
    cfg = load_settings() if settings is None else settings
    bar = tqdm(total=6, desc="OFMR pipeline", unit="step", disable=not show_progress)

    def stage(name):
        bar.set_postfix_str(name, refresh=True)

    try:
        stage("Features")
        feat_df, feat_cols = compute_features(df, **cfg["features"])
        bar.update()

        stage("OOD states")
        detector = ood_detector if ood_detector is not None else OODSystem(**cfg.get("ood", {}))
        regimes = detector.detect(df, process_from=ood_process_from)
        bar.update()

        stage("Supervision targets")
        targets = build_supervision_targets(df, feat_df, cfg.get("outcome"))
        bar.update()

        stage("GP search")
        gp_result = run_gp(df, feat_df, targets, regimes, cfg)
        bar.update()

        stage("Model frame")
        model_frame, _, _ = build_model_frame(
            df, feat_df, regimes,
            gp_result.model if gp_result is not None else None,
            targets=targets,
        )
        bar.update()

        stage("Triple barrier / outcomes")
        outcome = run_triple_barrier(model_frame, targets, cfg, raw=df)
        bar.update()
    finally:
        bar.close()

    return StrategyResult(
        features=model_frame,
        regimes=regimes,
        events=outcome.events,
        outcomes=targets,
        policy=pd.DataFrame(),  # evaluate_orderflow_policy is disabled
        gp=gp_result,
        metalabel=outcome.metalabel,
    )


if __name__ == "__main__":
    settings = load_settings(sys.argv[1] if len(sys.argv) > 1 else None)
    result = strategy(load_data(settings), settings, compare_forecasts=True)

    print(f"\n=== features ===\n{len(result.features)} rows, {len(result.features.columns)} columns")

    print("\n=== OOD latent states ===")
    print(result.regimes["latent_id"].value_counts(dropna=False).to_string())
    print(f"tradeable rows: {int(result.regimes['trade_allowed'].sum())}/{len(result.regimes)}")

    ev = result.events
    print(f"\n=== triple-barrier events ===\n{len(ev)} events")
    if not ev.empty:
        print(ev["barrier"].value_counts().to_string())
        if "trade_allowed" in ev:
            print(f"known-state eligible events: {int(ev['trade_allowed'].sum())}/{len(ev)}")

    print(f"\n=== targets ===\n{len(result.outcomes)} rows, {len(result.outcomes.columns)} columns")

    print("\n=== metalabel ===")
    ml = result.metalabel
    print("(no state had enough labeled training rows)" if ml is None or ml.report.empty
          else ml.report.to_string(index=False))

    print("\n=== policy report ===")
    print("(not enough data for the policy study)" if result.policy.empty
          else format_report(result.policy).to_string(index=False))

    if result.gp is not None:
        print("\n=== GP evolved features ===")
        print("(no state/target pair had enough complete training samples)" if result.gp.report.empty
              else result.gp.report.to_string(index=False))