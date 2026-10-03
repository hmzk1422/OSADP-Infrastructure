"""State-conditioned genetic feature search with purged time validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .grammar import Node, evaluate_expression, random_tree
from .population import crossover, initialize_population, mutate, tournament_select


_FORBIDDEN_NAME_PARTS = ("future_", "target", "label", "barrier", "latent", "trade_allowed", "embedding")

def _is_price_level_name(name: str) -> bool:
    """Recognize raw price-level names, including bid/ask and namespaced fields."""
    normalized = str(name).lower()
    parts = normalized.replace("-", "_").split("_")
    if any(part in {
        "return", "returns", "ret", "change", "pct", "percent", "qty", "quantity",
        "size", "volume", "depth", "imbalance", "flow", "count", "ratio",
    } for part in parts):
        return False
    if {"price", "close", "mid", "last"}.intersection(parts):
        return True
    return bool(parts and parts[-1] in {"bid", "ask"})


def build_gp_inputs(
    frame: pd.DataFrame,
    input_columns: tuple[str, ...] | list[str] | None = None,
    normalization_window: int = 500,
    normalization_min_periods: int = 50,
    clip_z: float = 8.0,
) -> pd.DataFrame:
    """Create causal, scale-normalized GP terminals from available features.

    Price levels are converted to log returns. Other numeric columns are
    standardized against preceding rows only. Target-like and registry columns
    are excluded by default; explicit columns are still checked for leakage.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a nonempty DataFrame")
    if normalization_window < 2 or not 2 <= normalization_min_periods <= normalization_window:
        raise ValueError("normalization_min_periods must be in [2, normalization_window]")
    if not np.isfinite(clip_z) or clip_z <= 0:
        raise ValueError("clip_z must be finite and > 0")

    if input_columns is None:
        columns = [
            column for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
            and not any(part in str(column).lower() for part in _FORBIDDEN_NAME_PARTS)
            and "timestamp" not in str(column).lower()
            and not str(column).lower().endswith("_id")
        ]
    else:
        columns = list(input_columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing GP input columns: {missing}")
        forbidden = [column for column in columns if any(part in str(column).lower() for part in _FORBIDDEN_NAME_PARTS)]
        if forbidden:
            raise ValueError(f"target/label/state columns cannot be GP inputs: {forbidden}")
    if not columns:
        raise ValueError("no numeric causal GP input columns were found")

    output: dict[str, pd.Series] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
        if _is_price_level_name(str(column)):
            logged = np.log(values.where(values > 0.0))
            values = logged.diff()
            input_name = f"x__log_return__{column}"
        else:
            input_name = f"x__{column}"
        prior = values.shift(1)
        history_mean = prior.rolling(normalization_window, min_periods=normalization_min_periods).mean()
        history_std = prior.rolling(normalization_window, min_periods=normalization_min_periods).std(ddof=0)
        delta = values - history_mean
        zscore = delta.div(history_std.where(history_std > 1e-12))
        flat = history_std.le(1e-12) & history_mean.notna()
        zscore = zscore.mask(flat & delta.abs().le(1e-12), 0.0)
        zscore = zscore.mask(flat & delta.gt(1e-12), clip_z)
        zscore = zscore.mask(flat & delta.lt(-1e-12), -clip_z)
        output[input_name] = zscore.clip(-clip_z, clip_z)

    result = pd.DataFrame(output, index=frame.index).replace([np.inf, -np.inf], np.nan)
    result = result.loc[:, result.notna().any(axis=0)]
    if result.empty:
        raise ValueError("GP inputs contain no usable values after causal normalization")
    return result


@dataclass(frozen=True)
class SelectedExpression:
    state_id: str
    target_column: str
    output_column: str
    expression: Node
    train_ic: float
    validation_ic: float
    fitness: float
    n_train: int
    n_validation: int
    direction: int = 0


@dataclass
class GeneticFeatureModel:
    """Selected state-specific expression trees and their input transform."""

    selected: tuple[SelectedExpression, ...]
    input_columns: tuple[str, ...] | None
    normalization_window: int
    normalization_min_periods: int
    clip_z: float

    def transform(self, frame: pd.DataFrame, state_labels: pd.Series) -> pd.DataFrame:
        """Evaluate selected features causally and mask them to their state."""
        if not frame.index.equals(state_labels.index):
            raise ValueError("frame and state_labels must have identical indexes")
        inputs = build_gp_inputs(
            frame,
            input_columns=self.input_columns,
            normalization_window=self.normalization_window,
            normalization_min_periods=self.normalization_min_periods,
            clip_z=self.clip_z,
        )
        state_values = state_labels.astype("object").astype(str)
        output = pd.DataFrame(index=frame.index)
        for item in self.selected:
            values = evaluate_expression(item.expression, inputs)
            output[item.output_column] = values.where(state_values == item.state_id)
        return output

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted expression inventory without executable code."""
        return {
            "input_columns": list(self.input_columns) if self.input_columns is not None else None,
            "normalization_window": self.normalization_window,
            "normalization_min_periods": self.normalization_min_periods,
            "clip_z": self.clip_z,
            "selected": [
                {
                    "state_id": item.state_id,
                    "target_column": item.target_column,
                    "output_column": item.output_column,
                    "expression": _node_to_dict(item.expression),
                    "train_ic": item.train_ic,
                    "validation_ic": item.validation_ic,
                    "fitness": item.fitness,
                    "direction": item.direction,
                    "n_train": item.n_train,
                    "n_validation": item.n_validation,
                }
                for item in self.selected
            ],
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "GeneticFeatureModel":
        selected = tuple(
            SelectedExpression(
                state_id=item["state_id"],
                target_column=item["target_column"],
                output_column=item["output_column"],
                expression=_node_from_dict(item["expression"]),
                train_ic=float(item["train_ic"]),
                validation_ic=float(item["validation_ic"]),
                fitness=float(item["fitness"]),
                n_train=int(item["n_train"]),
                n_validation=int(item["n_validation"]),
                direction=int(item.get("direction", np.sign(item["train_ic"]))),
            )
            for item in state["selected"]
        )
        return cls(
            selected=selected,
            input_columns=tuple(state["input_columns"]) if state.get("input_columns") is not None else None,
            normalization_window=int(state["normalization_window"]),
            normalization_min_periods=int(state["normalization_min_periods"]),
            clip_z=float(state["clip_z"]),
        )


@dataclass(frozen=True)
class GPEvolutionResult:
    features: pd.DataFrame
    report: pd.DataFrame
    model: GeneticFeatureModel


def _node_to_dict(node: Node) -> dict[str, Any]:
    return {"op": node.op, "value": node.value, "children": [_node_to_dict(child) for child in node.children]}


def _node_from_dict(value: dict[str, Any]) -> Node:
    return Node(
        op=str(value["op"]),
        value=value.get("value"),
        children=tuple(_node_from_dict(child) for child in value.get("children", ())),
    )


def _spearman(left: np.ndarray, right: np.ndarray, min_samples: int) -> tuple[float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(valid.sum())
    if count < min_samples:
        return np.nan, count
    x = pd.Series(left[valid]).rank(method="average").to_numpy(dtype=np.float64)
    y = pd.Series(right[valid]).rank(method="average").to_numpy(dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom <= 1e-15:
        return np.nan, count
    return float(np.dot(x, y) / denom), count


def _validate_train_mask(train_mask: pd.Series | np.ndarray, index: pd.Index) -> np.ndarray:
    if isinstance(train_mask, pd.Series):
        if not train_mask.index.equals(index):
            raise ValueError("train_mask Series must have the same index as features")
        mask = train_mask.fillna(False).to_numpy(dtype=bool)
    else:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(index):
            raise ValueError("train_mask must be one-dimensional and match the rows")
    seen_validation = np.maximum.accumulate(~mask)
    if np.any(mask & seen_validation):
        raise ValueError("train_mask must be a chronological prefix")
    if not mask.any() or mask.all():
        raise ValueError("train_mask must contain both training and validation rows")
    return mask


def _purge_target_overlap(
    train: np.ndarray,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_columns: tuple[str, ...],
    train_end_time=None,
    purge_rows: int | None = None,
) -> np.ndarray:
    """Remove train rows whose forward target can reach the validation period."""
    if purge_rows is not None and (not isinstance(purge_rows, (int, np.integer)) or purge_rows < 0):
        raise ValueError("purge_rows must be an integer >= 0 or None")
    validation_positions = np.flatnonzero(~train)
    if not validation_positions.size:
        raise ValueError("train_mask must leave validation rows after the training prefix")
    split = int(validation_positions[0])
    cutoff = train_end_time
    if cutoff is None and isinstance(features.index, pd.DatetimeIndex):
        if features.index.isna().any() or not features.index.is_monotonic_increasing:
            raise ValueError("DatetimeIndex must be valid and increasing for target purging")
        cutoff = features.index[split]
    result = train.copy()
    if cutoff is not None:
        cutoff = pd.to_datetime(cutoff, utc=True)
        needs_row_purge = False
        endpoint_columns = [
            f"future_timestamp__{column.rsplit('__', 1)[-1]}"
            if column.startswith(("future_return__", "future_mfe_", "future_mae_")) else None
            for column in target_columns
        ]
        for target_column, endpoint_column in zip(target_columns, endpoint_columns):
            if endpoint_column is not None and endpoint_column in targets.columns:
                endpoints = pd.to_datetime(targets[endpoint_column], utc=True, errors="coerce")
                result &= (endpoints < cutoff).to_numpy()
                continue
            horizon_label = target_column.rsplit("__", 1)[-1]
            if isinstance(features.index, pd.DatetimeIndex):
                if horizon_label.endswith("bar"):
                    try:
                        bars = int(horizon_label[:-3])
                    except ValueError:
                        bars = 0
                    positions = np.arange(len(features))
                    result &= (positions + bars < split) if bars > 0 else False
                else:
                    try:
                        horizon = pd.to_timedelta(horizon_label)
                    except (TypeError, ValueError):
                        horizon = pd.NaT
                    if pd.isna(horizon) or horizon <= pd.Timedelta(0):
                        raise ValueError(f"cannot infer target horizon from {target_column!r}; provide future timestamps or purge_rows")
                    times = pd.to_datetime(features.index, utc=True, errors="coerce")
                    result &= (times + horizon < cutoff).to_numpy()
            elif purge_rows is None:
                needs_row_purge = True
            else:
                needs_row_purge = True
        if isinstance(features.index, pd.DatetimeIndex):
            feature_times = pd.to_datetime(features.index, utc=True, errors="coerce")
            result &= (feature_times < cutoff).to_numpy()
        elif needs_row_purge:
            row_horizons = []
            for target_column in target_columns:
                label = target_column.rsplit("__", 1)[-1]
                if label.endswith("bar"):
                    try:
                        row_horizons.append(int(label[:-3]))
                    except ValueError:
                        pass
            embargo = int(purge_rows or 0) or (max(row_horizons) if row_horizons else 0)
            if embargo <= 0:
                raise ValueError(
                    "cannot purge target overlap without timestamp endpoints; provide purge_rows"
                )
            result[max(0, split - embargo):split] = False
        return result

    if purge_rows is None or purge_rows == 0:
        # Integer bar horizons are enough to derive a conservative embargo.
        bar_horizons = []
        for target_column in target_columns:
            label = target_column.rsplit("__", 1)[-1]
            if label.endswith("bar"):
                try:
                    bar_horizons.append(int(label[:-3]))
                except ValueError:
                    pass
        if bar_horizons:
            purge_rows = max(bar_horizons)
        else:
            raise ValueError("cannot purge target overlap; provide target timestamps, a DatetimeIndex, or purge_rows")
    if purge_rows:
        result[max(0, split - int(purge_rows)):split] = False
    return result


def _blockwise_spearman(
    values: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    train_end: int,
    block_count: int,
    min_block_samples: int,
) -> np.ndarray:
    """Compute separate train ICs over chronological blocks."""
    blocks = np.array_split(np.arange(train_end), block_count)
    correlations = []
    for positions in blocks:
        mask = train_mask[positions]
        if int(mask.sum()) < min_block_samples:
            continue
        correlation, count = _spearman(values[positions][mask], target[positions][mask], min_block_samples)
        if count >= min_block_samples and np.isfinite(correlation):
            correlations.append(correlation)
    return np.asarray(correlations, dtype=np.float64)


def _is_correlated_duplicate(
    candidate: np.ndarray,
    selected: list[np.ndarray],
    mask: np.ndarray,
    threshold: float,
    min_samples: int,
) -> bool:
    for prior in selected:
        valid = mask & np.isfinite(candidate) & np.isfinite(prior)
        if int(valid.sum()) < min_samples:
            continue
        left = candidate[valid]
        right = prior[valid]
        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(correlation) and abs(correlation) >= threshold:
            return True
    return False


def extrapolate_features(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    state_labels: pd.Series,
    train_mask: pd.Series | np.ndarray,
    *,
    target_columns: tuple[str, ...] | list[str] | None = None,
    input_columns: tuple[str, ...] | list[str] | None = None,
    normalization_window: int = 500,
    normalization_min_periods: int = 50,
    clip_z: float = 8.0,
    max_height: int = 5,
    population_size: int = 32,
    generations: int = 8,
    tournament_size: int = 4,
    elite_count: int = 2,
    mutation_rate: float = 0.30,
    crossover_rate: float = 0.70,
    max_features_per_state_target: int = 2,
    min_state_samples: int = 200,
    max_states: int = 10,
    temporal_windows: tuple[int, ...] | list[int] = (2, 4, 8, 16),
    complexity_penalty: float = 0.0005,
    block_count: int = 5,
    min_block_samples: int = 20,
    min_valid_blocks: int = 3,
    require_block_sign_agreement: bool = True,
    block_std_penalty: float = 0.5,
    min_train_ic: float = 0.03,
    min_validation_ic: float = 0.02,
    validation_ic_retention: float = 0.25,
    correlation_threshold: float = 0.98,
    immigrant_fraction: float = 0.10,
    train_end_time=None,
    purge_rows: int | None = None,
    seed: int = 7,
) -> GPEvolutionResult:
    """Evolve causal features per known latent state and target.

    Fitness uses mean IC across chronological training blocks, penalizes
    instability and tree size, and requires block sign agreement by default.
    A held-out validation veto rejects weak or sign-flipped features but never
    ranks candidates. Forward-target overlap is purged at the train boundary.
    Unknown/null states are excluded.
    """
    if not features.index.equals(targets.index) or not features.index.equals(state_labels.index):
        raise ValueError("features, targets, and state_labels must have identical indexes")
    if isinstance(features.index, pd.DatetimeIndex) and (
        features.index.isna().any() or not features.index.is_monotonic_increasing
    ):
        raise ValueError("DatetimeIndex must be valid and increasing for chronological GP validation")
    if not isinstance(max_height, int) or max_height < 2:
        raise ValueError("max_height must be an integer >= 2")
    if not isinstance(population_size, int) or population_size < 4:
        raise ValueError("population_size must be an integer >= 4")
    if not isinstance(generations, int) or generations < 1:
        raise ValueError("generations must be an integer >= 1")
    if not 1 <= elite_count < population_size:
        raise ValueError("elite_count must be in [1, population_size - 1]")
    if not 0.0 <= mutation_rate <= 1.0 or not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("mutation_rate and crossover_rate must be in [0, 1]")
    if max_features_per_state_target < 1 or min_state_samples < 2 or max_states < 1:
        raise ValueError("feature, sample, and state limits must be positive")
    if not np.isfinite(complexity_penalty) or complexity_penalty < 0.0:
        raise ValueError("complexity_penalty must be finite and >= 0")
    if not isinstance(block_count, int) or block_count < 2:
        raise ValueError("block_count must be an integer >= 2")
    if not isinstance(min_block_samples, int) or min_block_samples < 3:
        raise ValueError("min_block_samples must be an integer >= 3")
    if not isinstance(min_valid_blocks, int) or not 1 <= min_valid_blocks <= block_count:
        raise ValueError("min_valid_blocks must be in [1, block_count]")
    if not np.isfinite(block_std_penalty) or block_std_penalty < 0.0:
        raise ValueError("block_std_penalty must be finite and >= 0")
    if not np.isfinite(min_train_ic) or not 0.0 <= min_train_ic <= 1.0:
        raise ValueError("min_train_ic must be in [0, 1]")
    if not np.isfinite(min_validation_ic) or not 0.0 <= min_validation_ic <= 1.0:
        raise ValueError("min_validation_ic must be in [0, 1]")
    if not np.isfinite(validation_ic_retention) or not 0.0 <= validation_ic_retention <= 1.0:
        raise ValueError("validation_ic_retention must be in [0, 1]")
    if not np.isfinite(correlation_threshold) or not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    if not np.isfinite(immigrant_fraction) or not 0.0 <= immigrant_fraction < 0.5:
        raise ValueError("immigrant_fraction must be in [0, 0.5)")

    inputs = build_gp_inputs(
        features,
        input_columns=input_columns,
        normalization_window=normalization_window,
        normalization_min_periods=normalization_min_periods,
        clip_z=clip_z,
    )
    variables = tuple(inputs.columns)
    initial_train = _validate_train_mask(train_mask, features.index)
    targets_to_use = tuple(
        target_columns
        if target_columns is not None
        else [column for column in targets if column.startswith("future_return__")]
    )
    if not targets_to_use:
        raise ValueError("target_columns must select at least one target column")
    missing_targets = [column for column in targets_to_use if column not in targets]
    if missing_targets:
        raise KeyError(f"GP target columns missing or empty: {missing_targets}")
    split_positions = np.flatnonzero(~initial_train)
    split_boundary = int(split_positions[0])
    if train_end_time is not None and isinstance(features.index, pd.DatetimeIndex):
        ordered_times = pd.to_datetime(features.index, utc=True, errors="coerce")
        split_boundary = int(np.searchsorted(ordered_times.asi8, pd.to_datetime(train_end_time, utc=True).value, side="left"))
    train = _purge_target_overlap(
        initial_train,
        features,
        targets,
        targets_to_use,
        train_end_time=train_end_time,
        purge_rows=purge_rows,
    )
    if not train.any():
        raise ValueError("target-overlap purge removed all GP training rows")
    validation = ~initial_train
    state_values = state_labels.astype("object").to_numpy()
    known_state_mask = pd.notna(state_values) & (state_values != "null") & (state_values != "")
    allowed_states = [str(value) for value in pd.unique(state_values[known_state_mask])]
    state_counts = {
        state: int(np.sum(known_state_mask & (state_values.astype(str) == state) & train))
        for state in allowed_states
    }
    states = [state for state, _ in sorted(state_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:max_states]]
    if not states:
        raise ValueError("no known latent states are available for GP search")

    windows = tuple(int(window) for window in temporal_windows)
    if not windows or any(window < 1 for window in windows):
        raise ValueError("temporal_windows must contain positive integers")
    rng = np.random.default_rng(seed)
    target_values = {column: pd.to_numeric(targets[column], errors="coerce").to_numpy(dtype=np.float64) for column in targets_to_use}
    selected: list[SelectedExpression] = []
    report_rows: list[dict[str, Any]] = []
    train_end = split_boundary
    train_inputs = inputs.iloc[:train_end]

    # Each state-target pair has an independent population and objective.
    for state_id in states:
        state_mask = state_values.astype(str) == state_id
        for target_column in targets_to_use:
            y = target_values[target_column]
            train_mask_pair = train & state_mask & np.isfinite(y)
            validation_mask_pair = validation & state_mask & np.isfinite(y)
            if int(train_mask_pair.sum()) < min_state_samples:
                continue
            population = initialize_population(rng, variables, population_size, max_height, windows)
            final_scored: list[tuple[Node, float, float, int, float, float, int]] = []
            score_cache: dict[str, tuple[float, float, int, float, float, int]] = {}

            for generation in range(generations):
                scored: list[tuple[Node, float, float, int, float, float, int]] = []
                for node in population:
                    key = str(node)
                    if key not in score_cache:
                        try:
                            values = evaluate_expression(node, train_inputs).to_numpy(dtype=np.float64)
                        except (FloatingPointError, OverflowError, ValueError, KeyError):
                            values = np.full(train_end, np.nan, dtype=np.float64)
                        train_pair = train_mask_pair[:train_end]
                        train_ic, count = _spearman(values[train_pair], y[:train_end][train_pair], min_state_samples)
                        block_ics = _blockwise_spearman(
                            values,
                            y[:train_end],
                            train_pair,
                            train_end,
                            block_count,
                            min_block_samples,
                        )
                        if block_ics.size:
                            block_mean = float(block_ics.mean())
                            block_std = float(block_ics.std(ddof=0))
                        else:
                            block_mean = np.nan
                            block_std = np.nan
                        signs_agree = (
                            block_ics.size > 0
                            and np.all(np.sign(block_ics) == np.sign(block_ics[0]))
                            and np.sign(train_ic) == np.sign(block_ics[0])
                        )
                        eligible = (
                            np.isfinite(train_ic)
                            and block_ics.size >= min_valid_blocks
                            and abs(block_mean) >= min_train_ic
                            and (signs_agree or not require_block_sign_agreement)
                        )
                        if eligible:
                            fitness = (
                                abs(block_mean)
                                - block_std_penalty * block_std
                                - complexity_penalty * np.log1p(node.size)
                            )
                        else:
                            fitness = -np.inf
                        score_cache[key] = (
                            float(fitness), float(train_ic), count,
                            float(block_mean), float(block_std), int(block_ics.size),
                        )
                    fitness, train_ic, count, block_mean, block_std, n_blocks = score_cache[key]
                    scored.append((node, fitness, train_ic, count, block_mean, block_std, n_blocks))
                scored.sort(key=lambda item: item[1], reverse=True)
                final_scored = scored
                if generation + 1 == generations:
                    break

                scored_for_selection = [(node, fitness) for node, fitness, *_ in scored]
                next_population = [item[0] for item in scored[:elite_count]]
                immigrant_count = int(round(population_size * immigrant_fraction))
                if immigrant_fraction > 0.0:
                    immigrant_count = max(1, immigrant_count)
                offspring_target = population_size - immigrant_count
                while len(next_population) < offspring_target:
                    if rng.random() < crossover_rate:
                        left = tournament_select(rng, scored_for_selection, tournament_size)
                        right = tournament_select(rng, scored_for_selection, tournament_size)
                        child = crossover(rng, left, right, max_height)
                    else:
                        child = tournament_select(rng, scored_for_selection, tournament_size)
                    if rng.random() < mutation_rate:
                        child = mutate(rng, child, variables, max_height, windows)
                    next_population.append(child)
                while len(next_population) < population_size:
                    next_population.append(random_tree(
                        rng,
                        variables,
                        max_height,
                        temporal_windows=windows,
                        grow=True,
                    ))
                population = next_population

            unique_selected = set()
            selected_values: list[np.ndarray] = []
            selected_count = 0
            for node, fitness, train_ic, n_train, block_mean, block_std, n_blocks in final_scored:
                if not np.isfinite(fitness) or str(node) in unique_selected:
                    continue
                values = evaluate_expression(node, inputs).to_numpy(dtype=np.float64)
                validation_ic, n_validation = _spearman(
                    values[validation_mask_pair],
                    y[validation_mask_pair],
                    min_state_samples,
                )
                validation_passes = (
                    np.isfinite(validation_ic)
                    and abs(validation_ic) >= min_validation_ic
                    and abs(validation_ic) >= validation_ic_retention * abs(train_ic)
                    and np.sign(validation_ic) == np.sign(train_ic)
                )
                if not validation_passes:
                    continue
                train_values = values[:train_end]
                if _is_correlated_duplicate(
                    train_values,
                    selected_values,
                    train_mask_pair[:train_end],
                    correlation_threshold,
                    min_state_samples,
                ):
                    continue
                unique_selected.add(str(node))
                output_column = f"gp__{state_id}__{target_column.removeprefix('future_return__')}__{selected_count:02d}"
                item = SelectedExpression(
                    state_id=state_id,
                    target_column=target_column,
                    output_column=output_column,
                    expression=node,
                    train_ic=float(train_ic),
                    validation_ic=float(validation_ic),
                    fitness=float(fitness),
                    n_train=int(n_train),
                    n_validation=int(n_validation),
                    direction=int(np.sign(train_ic)),
                )
                selected.append(item)
                selected_values.append(train_values)
                report_rows.append({
                    "state_id": state_id,
                    "target_column": target_column,
                    "output_column": output_column,
                    "expression": str(node),
                    "tree_height": node.height,
                    "tree_size": node.size,
                    "train_ic": float(train_ic),
                    "validation_ic": float(validation_ic),
                    "fitness": float(fitness),
                    "block_ic_mean": float(block_mean),
                    "block_ic_std": float(block_std),
                    "n_train_blocks": int(n_blocks),
                    "direction": item.direction,
                    "n_train": int(n_train),
                    "n_validation": int(n_validation),
                })
                selected_count += 1
                if selected_count >= max_features_per_state_target:
                    break

    model = GeneticFeatureModel(
        selected=tuple(selected),
        input_columns=tuple(input_columns) if input_columns is not None else None,
        normalization_window=normalization_window,
        normalization_min_periods=normalization_min_periods,
        clip_z=clip_z,
    )
    feature_frame = model.transform(features, state_labels) if selected else pd.DataFrame(index=features.index)
    report = pd.DataFrame(report_rows)
    if not report.empty:
        report = report.sort_values(["state_id", "target_column", "fitness"], ascending=[True, True, False]).reset_index(drop=True)
    return GPEvolutionResult(features=feature_frame, report=report, model=model)
