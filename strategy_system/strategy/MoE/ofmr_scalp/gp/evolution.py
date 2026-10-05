"""State-conditioned genetic feature search with purged time validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .grammar import Node, evaluate_expression, node_from_dict as _node_from_dict, node_to_dict as _node_to_dict, random_tree
from .population import (crossover, create_initial_population, deduplicate_population,
                         filter_constant_features, filter_invalid_features,
                         filter_redundant_features, initialize_population, mutate,
                         tournament_select)
from .evaluation import (calculate_fitness, prefilter_features, ridge_select_features,
                         test_feature_set, validate_evaluation_models)
from .feature_library import (build_feature_library, save_feature_library,
                              select_validated_features)


_FORBIDDEN_NAME_PARTS = ("future_", "target", "label", "barrier", "latent", "trade_allowed", "embedding")


def evaluate_population(population, evaluator):
    """Evaluate an expression population through a caller-provided objective."""
    return [(expression, float(evaluator(expression))) for expression in population]


def select_parents(rng, scored_population, count=2, tournament_size=4):
    return [tournament_select(rng, scored_population, tournament_size) for _ in range(count)]


def evolve_generation(rng, scored_population, primitive_features, *, population_size, max_depth,
                     temporal_windows=(2, 4, 8, 16), elite_count=2, tournament_size=4,
                     crossover_rate=0.7, mutation_rate=0.3, immigrant_fraction=0.0):
    """Create one bounded next generation from a scored expression population."""
    scored = sorted(scored_population, key=lambda item: item[1], reverse=True)
    next_generation = [node for node, _ in scored[:max(0, elite_count)]]
    immigrant_count = int(round(population_size * immigrant_fraction))
    target = population_size - immigrant_count
    while len(next_generation) < target:
        left = select_parents(rng, scored, 1, tournament_size)[0]
        if rng.random() < crossover_rate:
            right = select_parents(rng, scored, 1, tournament_size)[0]
            child = crossover(rng, left, right, max_depth)
        else:
            child = left
        if rng.random() < mutation_rate:
            child = mutate(rng, child, tuple(primitive_features), max_depth, tuple(temporal_windows))
        next_generation.append(child)
    while len(next_generation) < population_size:
        next_generation.append(random_tree(rng, tuple(primitive_features), max_depth,
                                           temporal_windows=tuple(temporal_windows), grow=True))
    return next_generation


def stop_evolution(best_scores, *, patience=3, min_improvement=1e-6):
    """Stop after a stable fitness plateau or a non-finite score sequence."""
    if len(best_scores) <= patience:
        return False
    recent = np.asarray(best_scores[-(patience + 1):], dtype=float)
    if not np.isfinite(recent[-1]):
        return True
    return bool(np.nanmax(recent[1:]) - np.nanmax(recent[:-1]) <= min_improvement)


def evolve_population(rng, primitive_features, evaluator, *, population_size=32, max_depth=5,
                      generations=8, temporal_windows=(2, 4, 8, 16), initial_population=None,
                      population_filter=None, patience=3, **generation_kwargs):
    """Generic offline GP loop; evaluator and optional screening control the objective."""
    population = list(initial_population) if initial_population is not None else create_initial_population(
        rng, primitive_features, population_size, max_depth, temporal_windows)
    history = []
    hall_of_fame = {}
    for _ in range(generations):
        if population_filter is not None:
            population = population_filter(population)
        while len(population) < population_size:
            population.append(random_tree(rng, tuple(primitive_features), max_depth,
                                          temporal_windows=tuple(temporal_windows), grow=True))
        scored = evaluate_population(population, evaluator)
        for expression, score in scored:
            hall_of_fame[str(expression)] = (expression, score)
        history.append(max((score for _, score in scored), default=-np.inf))
        if _ == generations - 1 or stop_evolution(history, patience=patience):
            break
        population = evolve_generation(rng, scored, primitive_features, population_size=population_size,
                                       max_depth=max_depth, temporal_windows=temporal_windows,
                                       **generation_kwargs)
    return sorted(hall_of_fame.values(), key=lambda item: item[1], reverse=True)

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
    model_metrics: dict[str, Any] = field(default_factory=dict)


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
        state_values = state_labels.astype("object").map(
            lambda value: "null" if pd.isna(value) else str(value).casefold())
        output = pd.DataFrame(index=frame.index)
        for item in self.selected:
            values = evaluate_expression(item.expression, inputs)
            output[item.output_column] = values.where(state_values == item.state_id.casefold())
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
                    "model_metrics": item.model_metrics,
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
                model_metrics=dict(item.get("model_metrics", {})),
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
    feature_library: dict[str, Any] | None = None


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


def _correlation_tstat(correlation: float, sample_count: int) -> float:
    if sample_count <= 2 or not np.isfinite(correlation):
        return np.nan
    bounded = min(abs(float(correlation)), 1.0 - 1e-12)
    return bounded * np.sqrt((sample_count - 2) / max(1e-12, 1.0 - bounded * bounded))


def _multiple_testing_tstat_threshold(test_count: int, alpha: float) -> float:
    """Conservative normal-tail Bonferroni threshold for a GP search family."""
    return float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * max(1, test_count) / alpha))))


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
                result &= np.asarray(endpoints < cutoff)
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
                        if purge_rows is None:
                            raise ValueError(f"cannot infer target horizon from {target_column!r}; provide future timestamps or purge_rows")
                        result[max(0, split - int(purge_rows)):split] = False
                        continue
                    times = pd.to_datetime(features.index, utc=True, errors="coerce")
                    result &= np.asarray(times + horizon < cutoff)
            elif purge_rows is None:
                needs_row_purge = True
            else:
                needs_row_purge = True
        if isinstance(features.index, pd.DatetimeIndex):
            feature_times = pd.to_datetime(features.index, utc=True, errors="coerce")
            result &= np.asarray(feature_times < cutoff)
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


def _infer_purge_rows(target_column: str, index: pd.Index) -> int:
    """Infer an inner-fold embargo from the encoded target horizon when possible."""
    label = str(target_column).rsplit("__", 1)[-1].lower()
    if label.endswith("bar"):
        try:
            return max(1, int(label[:-3]))
        except ValueError:
            pass
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        try:
            horizon = pd.to_timedelta(label)
            cadence = pd.Series(index).diff().dropna().median()
            if horizon > pd.Timedelta(0) and cadence > pd.Timedelta(0):
                return max(1, int(np.ceil(horizon / cadence)))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return 1


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
    min_state_samples: int = 60,
    include_null: bool = False,
    min_null_samples: int = 60,
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
    min_train_tstat: float = 2.0,
    min_validation_tstat: float = 1.96,
    multiple_testing_alpha: float = 0.05,
    validation_ic_retention: float = 0.25,
    correlation_threshold: float = 0.98,
    immigrant_fraction: float = 0.10,
    train_end_time=None,
    purge_rows: int | None = None,
    seed: int = 7,
    evaluation_models: tuple[str, ...] | list[str] = ("ridge", "ols", "tree", "neural_network"),
    evaluation_model_params: dict[str, dict[str, Any]] | None = None,
    feature_library_path: str | None = None,
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
    if min_null_samples < 2:
        raise ValueError("min_null_samples must be >= 2")
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
    if not np.isfinite(min_train_tstat) or min_train_tstat < 0:
        raise ValueError("min_train_tstat must be finite and >= 0")
    if not np.isfinite(min_validation_tstat) or min_validation_tstat < 0:
        raise ValueError("min_validation_tstat must be finite and >= 0")
    if not np.isfinite(multiple_testing_alpha) or not 0.0 < multiple_testing_alpha < 1.0:
        raise ValueError("multiple_testing_alpha must be in (0, 1)")
    if not np.isfinite(validation_ic_retention) or not 0.0 <= validation_ic_retention <= 1.0:
        raise ValueError("validation_ic_retention must be in [0, 1]")
    if not np.isfinite(correlation_threshold) or not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    if not np.isfinite(immigrant_fraction) or not 0.0 <= immigrant_fraction < 0.5:
        raise ValueError("immigrant_fraction must be in [0, 0.5)")
    evaluation_models = validate_evaluation_models(evaluation_models)
    model_params_by_name = {
        validate_evaluation_models([name])[0]: params
        for name, params in (evaluation_model_params or {}).items()
    }

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
    state_keys = np.asarray([str(value).casefold() if pd.notna(value) else "null" for value in state_values], dtype=object)
    known_state_mask = pd.notna(state_values) & (state_keys != "null") & (state_keys != "")
    state_display = {}
    for value, key, known in zip(state_values, state_keys, known_state_mask):
        if known:
            state_display.setdefault(key, str(value))
    allowed_state_keys = list(state_display)
    if include_null and np.any((state_keys == "null") & train):
        state_display["null"] = "null"
        allowed_state_keys.append("null")
    state_counts = {
        key: int(np.sum(pd.notna(state_values) & (state_keys == key) & train))
        for key in allowed_state_keys
    }
    states = [state_display[key] for key, _ in sorted(state_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:max_states]]
    if not states:
        raise ValueError("no known latent states are available for GP search")

    windows = tuple(int(window) for window in temporal_windows)
    if not windows or any(window < 1 for window in windows):
        raise ValueError("temporal_windows must contain positive integers")
    rng = np.random.default_rng(seed)
    search_tstat = _multiple_testing_tstat_threshold(population_size * generations, multiple_testing_alpha)
    required_train_tstat = max(min_train_tstat, search_tstat)
    required_validation_tstat = max(min_validation_tstat, search_tstat)
    target_values = {column: pd.to_numeric(targets[column], errors="coerce").to_numpy(dtype=np.float64) for column in targets_to_use}
    selected: list[SelectedExpression] = []
    report_rows: list[dict[str, Any]] = []
    train_end = split_boundary
    train_inputs = inputs.iloc[:train_end]

    # Each state-target pair has an independent population and objective.
    for state_id in states:
        state_mask = state_keys == state_id.casefold()
        for target_column in targets_to_use:
            y = target_values[target_column]
            train_mask_pair = train & state_mask & np.isfinite(y)
            validation_mask_pair = validation & state_mask & np.isfinite(y)
            required_samples = min_null_samples if state_id.casefold() == "null" else min_state_samples
            if int(train_mask_pair.sum()) < required_samples:
                continue
            # Cheap target correlation, followed by Ridge coefficient and
            # redundancy filtering, is fit on this latent's training rows only.
            train_rows = np.flatnonzero(train_mask_pair[:train_end])
            primitive_frame = inputs.iloc[train_rows]
            primitive_target = y[train_rows]
            cheap_names = prefilter_features(primitive_frame, primitive_target, max_features=64)
            if not cheap_names:
                continue
            ridge_names = ridge_select_features(primitive_frame[cheap_names], primitive_target,
                                                max_features=min(32, len(cheap_names)))
            search_variables = tuple(ridge_names or cheap_names)
            population = initialize_population(rng, search_variables, population_size, max_height, windows)

            def screen_expression_population(candidates):
                candidates = deduplicate_population(filter_invalid_features(candidates, search_variables, max_height))
                sample_mask = train_mask_pair[:train_end]
                candidates = filter_constant_features(candidates, train_inputs, sample_mask=sample_mask)
                if not candidates:
                    return []
                expression_frame = pd.DataFrame(
                    {str(node): evaluate_expression(node, train_inputs).iloc[train_rows].to_numpy(dtype=float)
                     for node in candidates},
                    index=features.index[train_rows],
                )
                cheap = prefilter_features(expression_frame, primitive_target, max_features=64)
                if not cheap:
                    return []
                ridge = ridge_select_features(expression_frame[cheap], primitive_target,
                                              max_features=min(16, len(cheap)))
                survivors = [node for node in candidates if str(node) in set(ridge)]
                survivors = filter_redundant_features(survivors, train_inputs, threshold=correlation_threshold,
                                                      min_samples=min_block_samples, sample_mask=sample_mask)
                return survivors

            population = screen_expression_population(population)
            while len(population) < population_size:
                population.append(random_tree(rng, search_variables, max_height, temporal_windows=windows, grow=True))
            final_scored: list[tuple[Node, float, float, int, float, float, int]] = []
            score_cache: dict[str, tuple[float, float, int, float, float, int]] = {}

            def score_expression(node):
                key = str(node)
                if key not in score_cache:
                    try:
                        values = evaluate_expression(node, train_inputs).to_numpy(dtype=np.float64)
                    except (FloatingPointError, OverflowError, ValueError, KeyError):
                        values = np.full(train_end, np.nan, dtype=np.float64)
                    train_pair = train_mask_pair[:train_end]
                    train_ic, count = _spearman(values[train_pair], y[:train_end][train_pair], min_state_samples)
                    block_ics = _blockwise_spearman(values, y[:train_end], train_pair, train_end,
                                                     block_count, min_block_samples)
                    block_mean = float(block_ics.mean()) if block_ics.size else np.nan
                    block_std = float(block_ics.std(ddof=0)) if block_ics.size else np.nan
                    signs_agree = (block_ics.size > 0 and np.all(np.sign(block_ics) == np.sign(block_ics[0]))
                                   and np.sign(train_ic) == np.sign(block_ics[0]))
                    eligible = (np.isfinite(train_ic)
                                and _correlation_tstat(train_ic, count) >= required_train_tstat
                                and block_ics.size >= min_valid_blocks
                                and abs(block_mean) >= min_train_ic
                                and (signs_agree or not require_block_sign_agreement))
                    objective_scores = []
                    if eligible:
                        masked_y = np.where(train_pair, y[:train_end], np.nan)
                        candidate_frame = pd.DataFrame({"candidate": values}, index=features.index[:train_end])
                        for model_name in evaluation_models:
                            try:
                                wf = test_feature_set(
                                    candidate_frame, masked_y, model_type=model_name,
                                    model_params=model_params_by_name.get(model_name),
                                    purge_rows=int(purge_rows or _infer_purge_rows(target_column, features.index[:train_end])),
                                    min_samples=max(20, required_samples // 5),
                                    regime_labels=state_labels.iloc[:train_end].to_numpy(),
                                    regime_id=state_id,
                                )
                                score = calculate_fitness(wf, complexity=node.size,
                                                          complexity_penalty=complexity_penalty)
                                if np.isfinite(score):
                                    objective_scores.append((model_name, score))
                            except (ValueError, KeyError, TypeError, AttributeError,
                                    FloatingPointError, OverflowError, np.linalg.LinAlgError):
                                continue
                    family_scores = []
                    families = dict.fromkeys("linear" if name in {"ridge", "ols"} else name
                                             for name in evaluation_models)
                    for family in families:
                        family_values = [score for name, score in objective_scores
                                         if ("linear" if name in {"ridge", "ols"} else name) == family]
                        if family_values:
                            family_scores.append(float(np.mean(family_values)))
                    fitness = float(np.mean(family_scores)) if family_scores else -np.inf
                    score_cache[key] = (fitness, float(train_ic), count, block_mean, block_std, int(block_ics.size))
                return score_cache[key][0]

            scored_population = evolve_population(
                rng, search_variables, score_expression, population_size=population_size,
                max_depth=max_height, generations=generations, temporal_windows=windows,
                initial_population=population, population_filter=screen_expression_population,
                elite_count=elite_count, tournament_size=tournament_size,
                crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                immigrant_fraction=max(immigrant_fraction, 1 / population_size if immigrant_fraction > 0 else 0),
            )
            final_scored = [
                (node, score, *score_cache[str(node)][1:])
                for node, score in scored_population
            ]

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
                    and _correlation_tstat(validation_ic, n_validation) >= required_validation_tstat
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
                # Compare feature usefulness across model families on the
                # training prefix only. The outer validation block above stays
                # a final veto and never enters these walk-forward folds.
                masked_target = np.where(train_mask_pair[:train_end], y[:train_end], np.nan)
                candidate_frame = pd.DataFrame({"candidate": values[:train_end]}, index=features.index[:train_end])
                model_results = []
                model_metrics: dict[str, Any] = {}
                for model_name in evaluation_models:
                    try:
                        result = test_feature_set(
                            candidate_frame,
                            masked_target,
                            model_type=model_name,
                                        model_params=model_params_by_name.get(model_name),
                                        purge_rows=int(purge_rows or _infer_purge_rows(target_column, features.index[:train_end])),
                                        min_samples=max(20, required_samples // 5),
                                        regime_labels=state_labels.iloc[:train_end].to_numpy(),
                                        regime_id=state_id,
                        )
                    except (ValueError, KeyError, TypeError, AttributeError,
                            FloatingPointError, OverflowError, np.linalg.LinAlgError):
                        continue
                    if np.isfinite(result.get("expectancy", np.nan)):
                        model_results.append(result)
                        prefix = str(model_name).lower().replace(" ", "_")
                        model_metrics[f"{prefix}_prediction_error"] = result["prediction_error"]
                        model_metrics[f"{prefix}_directional_accuracy"] = result["directional_accuracy"]
                        model_metrics[f"{prefix}_expectancy"] = result["expectancy"]
                        model_metrics[f"{prefix}_stability"] = result["stability"]
                        model_metrics[f"{prefix}_episode_expectancy_std"] = result.get("episode_expectancy_std", np.nan)
                        model_metrics[f"{prefix}_recurrence_consistency"] = result.get("recurrence_consistency", np.nan)
                        model_metrics[f"{prefix}_episode_count"] = result.get("episode_count", 0)
                if model_results:
                    score_by_family = {}
                    for result in model_results:
                        family = "linear" if result["model"] in {"ridge", "ols"} else result["model"]
                        score_by_family.setdefault(family, []).append(
                            calculate_fitness(result, complexity=node.size, complexity_penalty=complexity_penalty))
                    fitness = float(np.mean([np.mean(scores) for scores in score_by_family.values()]))
                else:
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
                    model_metrics=model_metrics,
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
                    "models_evaluated": tuple(result["model"] for result in model_results),
                    **model_metrics,
                })
                selected_count += 1
                if selected_count >= max_features_per_state_target:
                    break

    # Validate selected features jointly as a candidate feature set for each
    # latent-target pair; this catches redundancy and interaction effects that
    # single-expression scores cannot reveal.
    selected_groups: dict[tuple[str, str], list[int]] = {}
    for i, item in enumerate(selected):
        selected_groups.setdefault((item.state_id, item.target_column), []).append(i)
    report_by_output = {row.get("output_column"): row for row in report_rows}
    for (state_id, target_column), indexes in selected_groups.items():
        bundle = pd.DataFrame({
            selected[i].output_column: evaluate_expression(selected[i].expression, inputs).iloc[:train_end]
            for i in indexes
        }, index=features.index[:train_end])
        target = target_values[target_column][:train_end]
        state_train = train[:train_end] & (state_keys[:train_end] == state_id.casefold()) & np.isfinite(target)
        masked_target = np.where(state_train, target, np.nan)
        bundle_metrics: dict[str, Any] = {}
        bundle_scores: dict[str, list[float]] = {}
        for model_name in evaluation_models:
            try:
                result = test_feature_set(
                    bundle, masked_target, model_type=model_name,
                    model_params=model_params_by_name.get(model_name),
                    purge_rows=int(purge_rows or _infer_purge_rows(target_column, features.index[:train_end])),
                    min_samples=max(20, min_state_samples // 5),
                    regime_labels=state_labels.iloc[:train_end].to_numpy(),
                    regime_id=state_id,
                )
            except (ValueError, KeyError, TypeError, AttributeError,
                    FloatingPointError, OverflowError, np.linalg.LinAlgError):
                continue
            prefix = f"feature_set_{model_name}"
            for metric in ("prediction_error", "directional_accuracy", "expectancy", "stability", "episode_count",
                           "episode_expectancy_std", "recurrence_consistency"):
                bundle_metrics[f"{prefix}_{metric}"] = result.get(metric, np.nan)
            score = calculate_fitness(result, complexity=sum(selected[i].expression.size for i in indexes),
                                      complexity_penalty=complexity_penalty)
            if np.isfinite(score):
                family = "linear" if model_name in {"ridge", "ols"} else model_name
                bundle_scores.setdefault(family, []).append(score)
        set_fitness_values = [float(np.mean(values)) for values in bundle_scores.values()]
        set_fitness = float(np.mean(set_fitness_values)) if set_fitness_values else -np.inf
        for i in indexes:
            selected[i] = replace(selected[i], fitness=set_fitness if np.isfinite(set_fitness) else selected[i].fitness,
                                  model_metrics={**selected[i].model_metrics, **bundle_metrics,
                                                 "feature_set_fitness": set_fitness})
            report_by_output[selected[i].output_column].update({"feature_set_fitness": set_fitness, **bundle_metrics})

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
        report = select_validated_features(report, min_validation_ic=min_validation_ic)
    library = build_feature_library(model.selected, metadata={"source_input_columns": list(input_columns) if input_columns is not None else None,
                                                              "encoded_terminal_columns": list(variables),
                                                              "normalization_window": normalization_window,
                                                              "normalization_min_periods": normalization_min_periods,
                                                              "clip_z": clip_z})
    if feature_library_path:
        save_feature_library(library, feature_library_path)
    return GPEvolutionResult(features=feature_frame, report=report, model=model, feature_library=library)
