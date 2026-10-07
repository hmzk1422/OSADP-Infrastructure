"""Causal index mechanics for the OFMR scalp outcome layer.

Every timestamp/index shift in the strategy lives here. Nothing in this
module fits a model; it only aligns frames causally (backward merge_asof
or exact reindex) and assembles the single modeling frame via
``build_model_frame``.

Column contract of the assembled frame (S1 of the refactor plan):
    price_close,
    feat_* (compute_features output),
    raw__* (GP terminals from build_primitive_observations),
    gp_feat1..gp_featN (GeneticFeatureModel.transform, state-masked),
    latent_id, similarity, trade_allowed, state_mature, model_ready,
    future_return__* / future_mfe|mae__* supervision targets (alongside only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ood import build_primitive_observations


def align_regimes_to_index(regimes: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
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

def build_gp_source(raw: pd.DataFrame, features: pd.DataFrame, timestamp_col: str | None) -> pd.DataFrame:
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


def attach_raw_columns(
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


def chronological_train_mask(
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


def build_model_frame(
    raw: pd.DataFrame,
    feat_df: pd.DataFrame,
    regimes: pd.DataFrame,
    gp_model=None,
    latent_col: str = "latent_id",
    targets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Assemble the single modeling frame for the strategy.

    Order: engineered features -> raw__ primitives -> state-masked gp_feat*
    -> regime flags -> supervision targets (alongside, never live inputs).
    Returns (frame, aligned_regimes, gp_feature_names).
    ``gp_model`` is a fitted GeneticFeatureModel or None (GP disabled);
    ``targets`` is the forward-return/excursion frame from
    :func:`outcomes.build_supervision_targets` (S1: future_return__* /
    future_mfe|mae__* live in the frame alongside only).
    """
    frame = build_gp_source(raw, feat_df, None)
    aligned_regimes = align_regimes_to_index(regimes, feat_df.index)
    gp_feature_names: list[str] = []
    if gp_model is not None:
        gp_frame = gp_model.transform(feat_df, aligned_regimes[latent_col])
        gp_feature_names = [c for c in gp_frame.columns if c not in frame.columns]
        frame = frame.join(gp_frame[gp_feature_names])
    for column in ("latent_id", "similarity", "trade_allowed", "state_mature", "model_ready"):
        frame[column] = aligned_regimes[column].to_numpy()
    if targets is not None and not targets.empty:
        target_columns = [column for column in targets.columns if column not in frame.columns]
        if target_columns:
            frame = frame.join(targets[target_columns])
    return frame, aligned_regimes, gp_feature_names
