"""OFMR scalp strategy engine.

Responsibilities, in order:

    1. Load ``settings.yaml``                       -> ``load_settings``
    2. Load raw market data through ``exchange/``   -> ``load_data``
    3. Compute features                             -> ``features.compute_features``
    4. Detect OOD states from raw temporal primitives -> ``ood/``
    5. Build outcomes and optional GP features      -> ``outcome/`` + ``gp/``
    6. Apply the strategy rules                     -> ``policies/``

Run from the repository root:

    python -m strategy_system.strategy.MoE.ofmr_scalp.strategy [path/to/settings.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

from .outcome import (
    build_forward_excursion_targets,
    build_forward_return_targets,
    build_triple_barrier_sequence,
)

from .exchange.loader import load_ohlcv_state, load_orderbook_state
from .features import compute_features
from .gp import GPEvolutionResult, extrapolate_features
from .ood import OODSystem, build_primitive_observations
from .policies.orderflow_prediction import (
    evaluate_orderflow_policy,
    format_report,
)

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"

DATA_LOADERS = {
    "orderbook": load_orderbook_state,
    "ohlcv": load_ohlcv_state,
}


class StrategyResult(NamedTuple):
    """Everything the engine produced in one run."""

    features: pd.DataFrame  # compute_features output (price_close + features)
    regimes: pd.DataFrame   # causal raw-market OOD state assignments
    events: pd.DataFrame    # triple-barrier sequence (one row per signal)
    outcomes: pd.DataFrame  # all-row forward-return targets for supervised learning
    policy: pd.DataFrame    # policy report (the strategy's rules)
    gp: GPEvolutionResult | None  # optional state-specific evolved features


def _align_regimes_to_index(regimes: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    """Causally align raw OOD states to reconstructed feature/event times."""
    columns = ["latent_id", "similarity", "trade_allowed", "state_mature", "model_ready"]
    if target_index.isin(regimes.index).all():
        return regimes.reindex(target_index)[columns]
    if "timestamp" not in regimes.columns or pd.api.types.is_numeric_dtype(target_index.dtype):
        aligned = pd.DataFrame(index=target_index)
        aligned["latent_id"] = "null"
        aligned["similarity"] = 0.0
        aligned["trade_allowed"] = False
        aligned["state_mature"] = False
        aligned["model_ready"] = False
        return aligned

    left = pd.DataFrame({
        "_time": pd.to_datetime(target_index, utc=True, errors="coerce"),
        "_position": np.arange(len(target_index)),
    })
    right = pd.DataFrame({
        "_time": pd.to_datetime(regimes["timestamp"], utc=True, errors="coerce"),
        "latent_id": regimes["latent_id"].to_numpy(),
        "similarity": regimes["similarity"].to_numpy(),
        "trade_allowed": regimes["trade_allowed"].to_numpy(),
        "state_mature": regimes["state_mature"].to_numpy(),
        "model_ready": regimes["model_ready"].to_numpy(),
    }).dropna(subset=["_time"])
    matched = pd.merge_asof(
        left.sort_values("_time", kind="mergesort"),
        right.sort_values("_time", kind="mergesort"),
        on="_time",
        direction="backward",
    ).sort_values("_position")
    aligned = matched[columns].copy()
    aligned["latent_id"] = aligned["latent_id"].fillna("null")
    aligned["similarity"] = aligned["similarity"].fillna(0.0)
    aligned["trade_allowed"] = aligned["trade_allowed"].fillna(False).astype(bool)
    aligned["state_mature"] = aligned["state_mature"].fillna(False).astype(bool)
    aligned["model_ready"] = aligned["model_ready"].fillna(False).astype(bool)
    aligned.index = target_index
    return aligned


def _build_gp_source(raw: pd.DataFrame, features: pd.DataFrame, timestamp_col: str | None) -> pd.DataFrame:
    """Join causal raw primitives to the engineered frame for GP terminals."""
    primitives = build_primitive_observations(raw).add_prefix("raw__")
    if features.index.isin(primitives.index).all():
        aligned = primitives.reindex(features.index)
    else:
        if isinstance(raw.index, pd.DatetimeIndex):
            raw_times = pd.to_datetime(raw.index, utc=True, errors="coerce")
        elif (timestamp_col if timestamp_col in raw.columns else "timestamp" if "timestamp" in raw.columns else None) is not None:
            raw_time_name = timestamp_col if timestamp_col in raw.columns else "timestamp"
            raw_times = pd.to_datetime(raw[raw_time_name], utc=True, errors="coerce")
        else:
            raise ValueError("GP raw primitives need a timestamp column or DatetimeIndex to align to features")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise ValueError("GP feature frame needs a DatetimeIndex to align raw primitives")
        left = pd.DataFrame({"_time": pd.to_datetime(features.index, utc=True), "_position": np.arange(len(features))})
        right = pd.DataFrame({"_time": raw_times, **{column: primitives[column].to_numpy() for column in primitives}})
        right = right.dropna(subset=["_time"])
        matched = pd.merge_asof(
            left.sort_values("_time", kind="mergesort"),
            right.sort_values("_time", kind="mergesort"),
            on="_time",
            direction="backward",
        ).sort_values("_position")
        aligned = matched[list(primitives.columns)].copy()
        aligned.index = features.index
    return pd.concat([features, aligned], axis=1)


def _align_raw_outcome_columns(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    columns: set[str],
    timestamp_col: str | None,
) -> pd.DataFrame:
    """Attach configured raw OHLC/quote fields to feature rows without lookahead."""
    requested = [column for column in sorted(columns) if column in raw.columns and column not in features.columns]
    if not requested:
        return features
    if features.index.equals(raw.index):
        aligned = raw.loc[:, requested].copy()
        aligned.index = features.index
        return pd.concat([features, aligned], axis=1)

    raw_time_col = (
        timestamp_col if timestamp_col in raw.columns
        else "timestamp" if "timestamp" in raw.columns
        else None
    )
    if raw_time_col is not None:
        raw_times = pd.to_datetime(raw[raw_time_col], utc=True, errors="coerce")
    elif isinstance(raw.index, pd.DatetimeIndex):
        raw_times = pd.to_datetime(raw.index, utc=True, errors="coerce")
    else:
        raise ValueError("Raw outcome columns need matching indexes or timestamps for causal alignment")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("Feature rows need a DatetimeIndex to align raw outcome columns by time")
    left = pd.DataFrame({
        "_time": pd.to_datetime(features.index, utc=True, errors="coerce"),
        "_position": np.arange(len(features)),
    })
    right = pd.DataFrame({"_time": raw_times, **{column: raw[column].to_numpy() for column in requested}})
    right = right.dropna(subset=["_time"])
    matched = pd.merge_asof(
        left.sort_values("_time", kind="mergesort"),
        right.sort_values("_time", kind="mergesort"),
        on="_time",
        direction="backward",
    ).sort_values("_position")
    aligned = matched[requested].copy()
    aligned.index = features.index
    return pd.concat([features, aligned], axis=1)


def _gp_train_mask(
    features: pd.DataFrame,
    train_fraction: float,
    purge_rows: int,
) -> pd.Series:
    """Build a chronological prefix; GP purges target horizons separately."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("gp.train_fraction must be strictly between 0 and 1")
    if not isinstance(purge_rows, int) or purge_rows < 0:
        raise ValueError("gp.purge_rows must be an integer >= 0")
    split = int(len(features) * train_fraction)
    if split < 1 or split >= len(features):
        raise ValueError("not enough rows for the configured GP train/validation split")
    # Keep this as a pure time split. extrapolate_features separately purges
    # each target horizon, so embargo rows are not mistaken for validation.
    mask = np.arange(len(features)) < split
    return pd.Series(mask, index=features.index, name="gp_train_mask")


def load_settings(path: str | Path | None = None) -> dict:
    """Read the engine's ``settings.yaml`` (defaults to the sibling file)."""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH

    with settings_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    if not isinstance(settings, dict):
        raise ValueError(f"{settings_path} must contain a YAML mapping.")

    return settings


def load_data(settings: dict) -> pd.DataFrame:
    """Load raw market data through the exchange layer.

    Uses ``settings['data']['source']`` to pick the loader and forwards the
    matching parameter section to it.
    """
    data_cfg = settings["data"]
    source = data_cfg["source"]

    try:
        loader = DATA_LOADERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown data source {source!r}; expected one of {sorted(DATA_LOADERS)}."
        ) from None

    return loader(**data_cfg.get(source, {}))


def strategy(
    df: pd.DataFrame,
    settings: dict | None = None,
    ood_detector: OODSystem | None = None,
    ood_process_from: int = 0,
) -> StrategyResult:
    """Run the OFMR scalp pipeline on a raw state table.

    Steps
    -----
    1. ``compute_features``          raw frame -> causal feature frame
    2. OOD detector                  raw temporal primitives -> latent states
    3. outcome builders              future returns + fixed/dynamic barriers
    4. optional GP                   state-specific evolved features
    5. ``policies``                  evaluate the strategy's rules

    Parameters
    ----------
    df:
        Raw state table from ``load_data``.
    settings:
        Parsed ``settings.yaml``; loaded from disk when omitted.
    ood_detector:
        Optional stateful detector. Pass the same instance across data chunks
        to preserve learned latent prototypes; pass its ``state_dict()`` to
        persist and later restore those prototypes.
    ood_process_from:
        First row in ``df`` that should update a supplied detector. Include a
        raw context prefix for encoder warmup and set this to its first new row.
    """
    cfg = load_settings() if settings is None else settings

    # 1. Features.
    feat_df, feat_cols = compute_features(df, **cfg["features"])

    # 2. OOD states are built from the original raw stream, not the GP or
    # engineered feature frame. Null states are explicitly non-tradeable.
    detector = ood_detector if ood_detector is not None else OODSystem(**cfg.get("ood", {}))
    regimes = detector.detect(df, process_from=ood_process_from)

    # 3. Outcomes. Forward returns are targets for model fitting/evaluation;
    #    keep them separate from the causal feature frame to avoid leakage.
    outcome_cfg = cfg.get("outcome")
    if outcome_cfg is None:
        # Backward compatibility for settings files that still use `barriers`.
        barrier_cfg = dict(cfg["barriers"])
        horizon_bars = barrier_cfg.pop("horizon", barrier_cfg.pop("horizon_bars", 30))
        events = build_triple_barrier_sequence(
            feat_df,
            mode="fixed",
            horizon_bars=horizon_bars,
            **barrier_cfg,
        )
        outcomes = pd.DataFrame(index=feat_df.index)
    else:
        target_cfg = dict(outcome_cfg.get("forward_returns", {}))
        excursion_cfg = dict(target_cfg.pop("excursions", {}))
        target_source_columns = {
            column for column in (excursion_cfg.get("high_col"), excursion_cfg.get("low_col"))
            if column is not None
        }
        if target_cfg.get("timestamp_col") is not None:
            target_source_columns.add(target_cfg["timestamp_col"])
        target_source = _align_raw_outcome_columns(
            df,
            feat_df,
            target_source_columns,
            target_cfg.get("timestamp_col"),
        ) if target_source_columns else feat_df
        outcomes = build_forward_return_targets(target_source, **target_cfg)
        if excursion_cfg.pop("enabled", False):
            excursion_targets = build_forward_excursion_targets(
                target_source,
                horizons=target_cfg.get("horizons", ("1s", "5s", "10s")),
                price_col=target_cfg.get("price_col", "price_close"),
                timestamp_col=target_cfg.get("timestamp_col"),
                max_lag=target_cfg.get("max_lag"),
                **excursion_cfg,
            )
            duplicate_endpoints = [
                column for column in excursion_targets
                if column.startswith("future_timestamp__") and column in outcomes
            ]
            if duplicate_endpoints:
                excursion_targets = excursion_targets.drop(columns=duplicate_endpoints)
            for column in excursion_targets:
                outcomes[column] = excursion_targets[column].to_numpy()

        barrier_cfg = dict(outcome_cfg.get("triple_barrier", {}))
        mode = barrier_cfg.pop("mode", "fixed")
        fixed_cfg = dict(barrier_cfg.pop("fixed", {}))
        dynamic_cfg = dict(barrier_cfg.pop("dynamic", {}))
        predicted_cfg = dict(barrier_cfg.pop("predicted", {}))
        mode_cfg = {
            "fixed": fixed_cfg,
            "dynamic": dynamic_cfg,
            "predicted": predicted_cfg,
        }.get(mode)
        if mode_cfg is None:
            raise ValueError("outcome.triple_barrier.mode must be 'fixed', 'dynamic', or 'predicted'")
        selected_cfg = {**barrier_cfg, **mode_cfg}
        timestamp_col = selected_cfg.get("timestamp_col")
        raw_columns = {
            selected_cfg.get(name)
            for name in (
                "price_col", "signal_col", "high_col", "low_col", "bid_col", "ask_col", "volatility_col",
                "predicted_mfe_col", "predicted_mae_col",
            )
            if selected_cfg.get(name) is not None
        }
        if timestamp_col is not None:
            raw_columns.add(timestamp_col)
        barrier_frame = _align_raw_outcome_columns(df, feat_df, raw_columns, timestamp_col)
        events = build_triple_barrier_sequence(
            barrier_frame,
            mode=mode,
            **selected_cfg,
        )

    # Align raw OOD states causally to reconstructed bar/index timestamps.
    aligned_regimes = _align_regimes_to_index(regimes, feat_df.index)
    # Keep all-row labels for research, while making the entry state and its
    # execution eligibility available as independent regime filters to fits.
    for column in ("latent_id", "trade_allowed", "state_mature", "model_ready"):
        outcomes[column] = aligned_regimes[column].to_numpy()

    # 4. Genetic programming is opt-in because each search evaluates a bounded
    # population over the selected state/target combinations.
    gp_result = None
    gp_cfg = dict(cfg.get("gp", {}))
    gp_enabled = bool(gp_cfg.pop("enabled", False))
    if gp_enabled:
        ood_cfg = cfg.get("ood") or {}
        gp_source = _build_gp_source(df, feat_df, ood_cfg.get("timestamp_col", "timestamp"))
        target_columns = gp_cfg.get("target_columns")
        train_fraction = float(gp_cfg.pop("train_fraction", 0.70))
        purge_rows = int(gp_cfg.pop("purge_rows", 0))
        split_position = int(len(feat_df) * train_fraction)
        train_end_time = feat_df.index[split_position] if isinstance(feat_df.index, pd.DatetimeIndex) and split_position < len(feat_df) else None
        train_mask = _gp_train_mask(
            feat_df,
            train_fraction,
            purge_rows,
        )
        # Control columns configure the split but are not arguments to GP.
        gp_cfg.pop("target_columns", None)
        gp_result = extrapolate_features(
            gp_source,
            outcomes,
            aligned_regimes["latent_id"],
            train_mask,
            target_columns=target_columns,
            train_end_time=train_end_time,
            purge_rows=purge_rows,
            **gp_cfg,
        )

    # Keep labeled events for model research, while exposing whether their
    # entry state was known and therefore eligible for execution.
    if not events.empty and "t0" in events.columns:
        entry_regimes = _align_regimes_to_index(regimes, pd.Index(events["t0"]))
        events["latent_id"] = entry_regimes["latent_id"].to_numpy()
        events["ood_similarity"] = entry_regimes["similarity"].to_numpy()
        events["trade_allowed"] = entry_regimes["trade_allowed"].fillna(False).to_numpy(dtype=bool)
        events["state_mature"] = entry_regimes["state_mature"].fillna(False).to_numpy(dtype=bool)
        events["model_ready"] = entry_regimes["model_ready"].fillna(False).to_numpy(dtype=bool)

    # 4. Policies: the rules of the strategy.
    policy = evaluate_orderflow_policy(feat_df, feat_cols, **cfg["policy"])

    return StrategyResult(features=feat_df, regimes=regimes, events=events, outcomes=outcomes, policy=policy, gp=gp_result)


if __name__ == "__main__":
    settings = load_settings(sys.argv[1] if len(sys.argv) > 1 else None)
    raw = load_data(settings)
    result = strategy(raw, settings)

    print("\n=== features ===")
    print(f"{len(result.features)} rows, {len(result.features.columns)} columns")

    print("\n=== OOD latent states ===")
    print(result.regimes["latent_id"].value_counts(dropna=False).to_string())
    print(f"tradeable rows: {int(result.regimes['trade_allowed'].sum())}/{len(result.regimes)}")

    print("\n=== triple-barrier events ===")
    print(f"{len(result.events)} events")
    if not result.events.empty:
        print(result.events["barrier"].value_counts().to_string())
        if "trade_allowed" in result.events:
            print(f"known-state eligible events: {int(result.events['trade_allowed'].sum())}/{len(result.events)}")

    print("\n=== forward-return targets ===")
    print(f"{len(result.outcomes)} rows, {len(result.outcomes.columns)} target columns")

    print("\n=== policy report ===")
    if result.policy.empty:
        print("(not enough data for the policy study)")
    else:
        print(format_report(result.policy).to_string(index=False))

    if result.gp is not None:
        print("\n=== GP evolved features ===")
        if result.gp.report.empty:
            print("(no state/target pair had enough complete training samples)")
        else:
            print(result.gp.report.to_string(index=False))
