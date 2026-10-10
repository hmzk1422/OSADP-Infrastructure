"""Genetic-programming feature generation and leakage-aware evaluation.

Public API
----------
generate_gp_features(df, ...)
    Fit a supervised gplearn SymbolicTransformer on an early, purged training
    slice and append ``gp_feature_1``, ``gp_feature_2``, ... to a copy of df.
    Pass ``return_state=True`` to retain a fitted state for transforming later
    data without refitting. Pass that state back as ``state=...`` to transform.

evaluate_gp_features(df, ...)
    Rank candidate features with training-only mutual information, fit a
    swappable scikit-learn-style regressor separately for each future horizon,
    and append actual future prices/returns, test-period predictions and errors.

Important
---------
* Input rows must already be in chronological order. This module does not sort.
* GP is supervised: its target is a future return or future price, but GP fitting
  uses only rows whose full target horizon ends strictly before the training/test
  boundary.
* The evaluation split is chronological and purged separately for each horizon.
* ``actual_*`` and ``error_*`` columns are diagnostics/labels, never inputs to a
  model selected by this module. Do not use them as predictors in other code.
* ``gplearn`` is imported lazily; evaluation can be used without it if the frame
  already contains GP features.


Example use case
------------------
from gp import generate_gp_features, evaluate_gp_features

# 1. Generate GP features
df_gp, gp_state = generate_gp_features(
    df,
    price_col="price_close",
    target_horizon=1,
    train_fraction=0.7,
    n_components=10,
    return_state=True,
)

# 2. Evaluate future horizons using mutual information
df_evaluated, results = evaluate_gp_features(
    df_gp,
    price_col="price_close",
    horizons=(1, 2, 3, 5, 10),
    target_kind="return",
    train_fraction=0.7,
    top_k=5,
)

print(results["metrics"])
print(results["mutual_information"][1])
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


_DEFAULT_FUNCTION_SET = (
    "add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min"
)
_DEFAULT_FORBIDDEN_TOKENS = {
    "timestamp", "datetime", "date", "time", "timeframe", "tf",
    "id", "uuid", "symbol", "ticker", "target", "label", "future",
    "forward", "lead", "outcome", "actual", "prediction", "predicted",
    "error", "residual", "targetreturn", "targetprice",
}


def _validate_frame(df: pd.DataFrame, *, timestamp_col: str | None = "timestamp") -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df must contain at least one row")
    if df.columns.has_duplicates:
        raise ValueError("df must have unique column names")
    if timestamp_col is not None and timestamp_col in df.columns:
        times = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        if times.isna().any():
            raise ValueError(f"{timestamp_col!r} contains invalid timestamps")
        if not times.is_monotonic_increasing:
            raise ValueError("df must be in chronological order (timestamps are not increasing)")


def _is_excluded_name(name: Any, *, exclude_gp_generated: bool = True) -> bool:
    """Conservatively detect common metadata, target, and post-outcome names."""
    text = str(name).strip().lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    if exclude_gp_generated and text.startswith("gp_feature_"):
        return True
    if tokens & _DEFAULT_FORBIDDEN_TOKENS:
        return True
    # Catch frequent compound names without treating `mid_price` as an ID.
    suspicious_substrings = (
        "timestamp", "datetime", "future_", "future-", "forward_", "forward-",
        "target_", "target-", "label_", "label-", "outcome_", "outcome-",
        "prediction_", "predicted_", "actual_", "error_", "residual_",
    )
    if any(part in text for part in suspicious_substrings):
        return True
    # IDs should be tokens / suffixes, not arbitrary occurrences (e.g. `mid_price`).
    if re.search(r"(^|[_\- ])id($|[_\- ])", text) or text.endswith("_id"):
        return True
    return False


def _select_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None,
    exclude_cols: Sequence[str] | None = None,
) -> list[str]:
    explicitly_excluded = set(exclude_cols or ())
    missing_excluded = explicitly_excluded.difference(df.columns)
    if missing_excluded:
        raise KeyError(f"exclude_cols contain missing columns: {sorted(missing_excluded)}")

    if feature_cols is None:
        selected = [
            str(col) for col in df.columns
            if col not in explicitly_excluded
            and pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
            and not _is_excluded_name(col)
        ]
    else:
        selected = list(feature_cols)
        if not selected:
            raise ValueError("feature_cols must not be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("feature_cols must not contain duplicates")
        missing = [col for col in selected if col not in df.columns]
        if missing:
            raise KeyError(f"feature_cols contain missing columns: {missing}")
        forbidden = [col for col in selected if col in explicitly_excluded or _is_excluded_name(col)]
        if forbidden:
            raise ValueError(
                "feature_cols contains metadata/target-like columns that must not be predictors: "
                f"{forbidden}"
            )

    if not selected:
        raise ValueError(
            "no primitive numeric features found; pass feature_cols explicitly or check column names"
        )
    return selected


def _numeric_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    # Explicit coercion makes object columns fail safe as NaNs rather than leaking strings.
    frame = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _future_targets(price: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(price)
    future_price = np.full(n, np.nan, dtype=np.float64)
    if horizon < n:
        future_price[:-horizon] = price[horizon:]
    future_return = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(price) & (price > 0) & np.isfinite(future_price) & (future_price > 0)
    future_return[valid] = future_price[valid] / price[valid] - 1.0
    return future_price, future_return


def _target_vector(df: pd.DataFrame, price_col: str, horizon: int, target_kind: str) -> np.ndarray:
    if price_col not in df.columns:
        raise KeyError(f"price column {price_col!r} is missing")
    price = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=np.float64)
    price[~np.isfinite(price)] = np.nan
    future_price, future_return = _future_targets(price, horizon)
    if target_kind == "return":
        return future_return
    if target_kind == "price":
        return future_price
    raise ValueError("target_kind must be 'return' or 'price'")


def _column_medians(values: np.ndarray, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return active column indices and medians using only the specified train rows.

    Column-at-a-time reductions avoid making a second full training-matrix copy
    for large datasets.
    """
    source_rows = np.arange(values.shape[0]) if rows is None else np.asarray(rows, dtype=np.int64)
    active = np.zeros(values.shape[1], dtype=bool)
    medians_all = np.full(values.shape[1], np.nan, dtype=np.float64)
    for j in range(values.shape[1]):
        col = values[source_rows, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            active[j] = True
            medians_all[j] = float(np.median(finite))
    if not active.any():
        raise ValueError("all candidate features are missing/non-finite in the training period")
    return np.flatnonzero(active), medians_all[active]


def _impute(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    invalid = ~np.isfinite(result)
    if invalid.any():
        result[invalid] = np.broadcast_to(medians, result.shape)[invalid]
    return result


def _chunks(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def _append_generated(df: pd.DataFrame, generated: np.ndarray, output_columns: Sequence[str]) -> pd.DataFrame:
    result = df.copy()
    # These names are reserved for this module's generated columns. Reapplying a
    # stored state replaces its own outputs instead of creating duplicate columns.
    for j, name in enumerate(output_columns):
        result[name] = generated[:, j]
    return result


@dataclass
class GPFeatureState:
    """Fitted GP expression set and training-only preprocessing for reuse."""

    feature_columns: tuple[str, ...]
    medians: np.ndarray
    scaler: StandardScaler
    transformer: Any
    output_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    target_kind: str
    target_horizon: int
    price_col: str
    train_fraction: float
    fit_end_row: int

    def transform(self, df: pd.DataFrame, *, batch_size: int = 100_000) -> pd.DataFrame:
        """Apply already-fitted expressions; no refit and no target access."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"input frame is missing fitted GP feature columns: {missing}")

        n = len(df)
        generated = np.full((n, len(self.output_columns)), np.nan, dtype=np.float64)
        for start, stop in _chunks(n, batch_size):
            raw = _numeric_matrix(df.iloc[start:stop], self.feature_columns)
            row_has_data = np.isfinite(raw).any(axis=1)
            if not row_has_data.any():
                continue
            clean = _impute(raw[row_has_data], self.medians)
            scaled = self.scaler.transform(clean)
            # Keep temporary transformed data bounded for large input frames.
            transformed = np.asarray(self.transformer.transform(scaled), dtype=np.float64)
            if transformed.ndim == 1:
                transformed = transformed.reshape(-1, 1)
            transformed[~np.isfinite(transformed)] = np.nan
            generated[start:stop][row_has_data] = transformed
        return _append_generated(df, generated, self.output_columns)


def generate_gp_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    *,
    price_col: str = "price_close",
    target_horizon: int = 1,
    target_kind: str = "return",
    train_fraction: float = 0.7,
    n_components: int = 10,
    generations: int = 10,
    population_size: int = 500,
    hall_of_fame: int = 100,
    max_train_samples: int = 50_000,
    function_set: Sequence[str] = _DEFAULT_FUNCTION_SET,
    metric: str = "pearson",
    parsimony_coefficient: float = 0.001,
    n_jobs: int = 1,
    random_state: int = 7,
    verbose: int = 0,
    exclude_cols: Sequence[str] | None = None,
    timestamp_col: str | None = "timestamp",
    transform_batch_size: int = 100_000,
    state: GPFeatureState | None = None,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, GPFeatureState]:
    """Append supervised gplearn features without fitting on the test period.

    By default GP evolves expressions against the future ``target_kind`` at
    ``target_horizon`` rows ahead (future return is recommended over absolute
    future price). The GP expressions and preprocessing are fitted using only
    the first ``train_fraction`` of chronologically ordered data, purged so the
    GP label ends before the split. Features are then generated for all rows.

    Args:
        df: Original, oldest-to-newest input DataFrame. All original columns are
            preserved in the returned copy.
        feature_cols: Primitive numeric inputs. If omitted, numeric columns are
            selected automatically while excluding common metadata, IDs and
            target/outcome-like columns.
        price_col: Current price series used to create the GP training target.
        target_horizon: Number of rows ahead for the GP fitness target.
        target_kind: ``"return"`` (default) or ``"price"``.
        train_fraction: Chronological boundary used to fit GP expressions.
        n_components: Number of generated expressions.
        generations, population_size, hall_of_fame: gplearn search controls.
        max_train_samples: Cap GP fitting rows for practical runtime.
        function_set: gplearn primitive functions.
        metric: gplearn fitness metric, usually ``"pearson"`` or ``"spearman"``.
        n_jobs: gplearn worker count; use -1 to request all available workers.
        state: If provided, apply an existing fitted state without refitting.
        return_state: If true, return ``(frame, fitted_state)`` for later reuse.

    Returns:
        A DataFrame preserving the input index and original columns, plus
        ``gp_feature_1`` ... ``gp_feature_N``. If ``return_state=True``, returns
        the DataFrame and the fitted GPFeatureState.

    Raises:
        ImportError: if fitting a new state and gplearn is not installed.
    """
    _validate_frame(df, timestamp_col=timestamp_col)
    if not isinstance(target_horizon, (int, np.integer)) or target_horizon < 1:
        raise ValueError("target_horizon must be a positive integer")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if transform_batch_size < 1 or max_train_samples < 2:
        raise ValueError("transform_batch_size must be >= 1 and max_train_samples >= 2")

    if state is not None:
        if feature_cols is not None and tuple(feature_cols) != state.feature_columns:
            raise ValueError("feature_cols must match the fitted state's feature_columns when state= is supplied")
        result = state.transform(df, batch_size=transform_batch_size)
        return (result, state) if return_state else result

    if target_kind not in ("return", "price"):
        raise ValueError("target_kind must be 'return' or 'price'")
    if price_col not in df.columns:
        raise KeyError(f"price column {price_col!r} is missing")
    if n_components < 1 or population_size < 2 or generations < 1:
        raise ValueError("n_components, population_size and generations must be positive")
    if hall_of_fame < n_components or hall_of_fame > population_size:
        raise ValueError("require n_components <= hall_of_fame <= population_size")
    if not function_set:
        raise ValueError("function_set must contain at least one function")
    if max_train_samples < 2:
        raise ValueError("max_train_samples must be >= 2")

    try:
        from gplearn.genetic import SymbolicTransformer
    except ImportError as exc:
        raise ImportError(
            "generate_gp_features() requires gplearn. Install it with `pip install gplearn`."
        ) from exc

    selected = _select_feature_columns(df, feature_cols, exclude_cols)
    raw = _numeric_matrix(df, selected)
    n = len(df)
    split = int(n * train_fraction)
    if split <= int(target_horizon) + 2:
        raise ValueError("not enough rows before the chronological split for the GP target horizon")

    y = _target_vector(df, price_col, int(target_horizon), target_kind)
    row_has_data = np.isfinite(raw).any(axis=1)
    row_index = np.arange(n)
    # Strict purge: each GP target must end before the first test row.
    eligible = (row_index < split - int(target_horizon)) & np.isfinite(y) & row_has_data
    eligible_rows = np.flatnonzero(eligible)
    if len(eligible_rows) < 20:
        raise ValueError(f"not enough valid purged GP training rows: {len(eligible_rows)} (need >= 20)")

    active_indices, medians = _column_medians(raw, eligible_rows)
    active_columns = tuple(selected[j] for j in active_indices)
    dropped_columns = tuple(selected[j] for j in range(len(selected)) if j not in set(active_indices.tolist()))
    if len(eligible_rows) > max_train_samples:
        rng = np.random.default_rng(random_state)
        fit_rows = np.sort(rng.choice(eligible_rows, size=max_train_samples, replace=False))
    else:
        fit_rows = eligible_rows

    x_fit = _impute(raw[fit_rows][:, active_indices], medians)
    # The full numeric matrix is no longer needed; release it before applying
    # the expression set to the complete frame in bounded chunks.
    del raw
    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit)
    y_fit = y[fit_rows]
    if not np.isfinite(y_fit).all():
        raise ValueError("internal error: non-finite GP targets reached fitting")

    transformer = SymbolicTransformer(
        population_size=int(population_size),
        hall_of_fame=int(hall_of_fame),
        n_components=int(n_components),
        generations=int(generations),
        function_set=tuple(function_set),
        metric=metric,
        parsimony_coefficient=float(parsimony_coefficient),
        max_samples=1.0,
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        verbose=int(verbose),
        low_memory=True,
    )
    transformer.fit(x_fit_scaled, y_fit)

    output_columns = tuple(f"gp_feature_{i + 1}" for i in range(int(n_components)))
    gp_state = GPFeatureState(
        feature_columns=active_columns,
        medians=medians.copy(),
        scaler=scaler,
        transformer=transformer,
        output_columns=output_columns,
        dropped_columns=dropped_columns,
        target_kind=target_kind,
        target_horizon=int(target_horizon),
        price_col=price_col,
        train_fraction=float(train_fraction),
        fit_end_row=int(split),
    )
    result = gp_state.transform(df, batch_size=transform_batch_size)
    return (result, gp_state) if return_state else result


def _default_evaluation_features(df: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in df.columns if str(c).startswith("gp_feature_")]
    if not columns:
        raise ValueError(
            "No gp_feature_* columns found. Run generate_gp_features() first, "
            "or pass feature_cols explicitly."
        )
    return columns


def _validate_evaluation_features(df: pd.DataFrame, feature_cols: Sequence[str] | None) -> list[str]:
    selected = _default_evaluation_features(df) if feature_cols is None else list(feature_cols)
    if not selected:
        raise ValueError("feature_cols must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("feature_cols must not contain duplicates")
    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise KeyError(f"feature_cols contain missing columns: {missing}")
    forbidden = [col for col in selected if _is_excluded_name(col, exclude_gp_generated=False)]
    if forbidden:
        raise ValueError(
            "feature_cols contains target/metadata-like columns; these must not be model inputs: "
            f"{forbidden}"
        )
    return selected


def evaluate_gp_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    *,
    price_col: str = "price_close",
    horizons: Sequence[int] = (1, 2, 3, 5, 10),
    target_kind: str = "return",
    model: Any = None,
    train_fraction: float = 0.7,
    top_k: int | None = None,
    mi_neighbors: int = 3,
    mi_n_jobs: int = 1,
    random_state: int = 7,
    timestamp_col: str | None = "timestamp",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank features by train-only MI and evaluate swappable regressors by horizon.

    Each horizon gets its own forward target and a chronological, purged split.
    Training rows satisfy ``t + horizon < split`` so their target cannot touch
    the test period. MI ranks are estimated only on those training rows. If
    ``top_k`` is given, only the top-k MI features are passed to that horizon's
    model; otherwise all non-dead candidate features are used.

    The returned DataFrame contains ``actual_price_t_plus_H`` and
    ``actual_return_t_plus_H`` labels, plus test-period predictions and errors.
    Prediction/error values are NaN outside the test period; actual future labels
    are available wherever the future price exists. Never pass those actual/error
    columns into a later model.

    Args:
        df: Input DataFrame; expected to contain GP features from
            ``generate_gp_features()``.
        feature_cols: Candidate predictors. Defaults to all ``gp_feature_*``
            columns. Explicit columns can include safe primitive features too.
        price_col: Price series that defines future price and forward-return targets.
        horizons: Positive row horizons such as ``(1, 2, 3, 5, 10)``.
        target_kind: Regress either forward ``"return"`` (recommended) or future
            absolute ``"price"``. Future prices are reported in either mode.
        model: Any cloneable sklearn-style regressor with ``fit`` and ``predict``.
            Defaults to StandardScaler + Ridge regression. A fresh clone is fit
            independently for every horizon.
        train_fraction: Chronological split fraction; the remainder is test.
        top_k: Optional number of highest-MI predictors to use per horizon.
        mi_neighbors: Neighbors for sklearn's continuous mutual-information estimate.
        mi_n_jobs: Worker count for mutual_info_regression.

    Returns:
        ``(evaluated_df, results)``. ``results`` contains a per-horizon metrics
        DataFrame, per-horizon MI rankings, selected features, and fitted models.
    """
    _validate_frame(df, timestamp_col=timestamp_col)
    if price_col not in df.columns:
        raise KeyError(f"price column {price_col!r} is missing")
    if target_kind not in ("return", "price"):
        raise ValueError("target_kind must be 'return' or 'price'")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not isinstance(mi_neighbors, (int, np.integer)) or mi_neighbors < 1:
        raise ValueError("mi_neighbors must be an integer >= 1")
    if top_k is not None and (not isinstance(top_k, (int, np.integer)) or top_k < 1):
        raise ValueError("top_k must be None or an integer >= 1")
    horizon_list = [int(h) for h in horizons]
    if not horizon_list or any(h < 1 for h in horizon_list):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizon_list)) != len(horizon_list):
        raise ValueError("horizons must be unique")

    selected = _validate_evaluation_features(df, feature_cols)
    raw = _numeric_matrix(df, selected)
    n = len(df)
    split = int(n * train_fraction)
    if split < 3 or split >= n:
        raise ValueError("chronological split leaves too few training or test rows")
    price = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=np.float64)
    price[~np.isfinite(price)] = np.nan

    if model is None:
        base_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("model must implement sklearn-style fit() and predict()")
        base_model = model

    evaluated = df.copy()
    results_by_horizon: list[dict[str, Any]] = []
    mi_rankings: dict[int, pd.DataFrame] = {}
    selected_by_horizon: dict[int, list[str]] = {}
    fitted_models: dict[int, Any] = {}
    dropped_by_horizon: dict[int, list[str]] = {}
    row_index = np.arange(n)
    row_has_data = np.isfinite(raw).any(axis=1)

    for horizon in horizon_list:
        future_price, future_return = _future_targets(price, horizon)
        target_values = future_return if target_kind == "return" else future_price
        # Purge labels that touch or cross the first test row.
        train_mask = (row_index < split - horizon) & np.isfinite(target_values) & row_has_data
        test_predict_mask = (row_index >= split) & row_has_data
        test_score_mask = test_predict_mask & np.isfinite(target_values)

        train_rows = np.flatnonzero(train_mask)
        if len(train_rows) < max(10, mi_neighbors + 2):
            raise ValueError(
                f"horizon {horizon}: not enough purged training rows ({len(train_rows)})"
            )
        active_indices, medians = _column_medians(raw, train_rows)
        active_names = [selected[j] for j in active_indices]
        dropped = [selected[j] for j in range(len(selected)) if j not in set(active_indices.tolist())]
        active_row_has_data = np.isfinite(raw[:, active_indices]).any(axis=1)
        test_predict_mask = (row_index >= split) & active_row_has_data
        train_rows = train_rows[active_row_has_data[train_rows]]
        if len(train_rows) < max(10, mi_neighbors + 2):
            raise ValueError(
                f"horizon {horizon}: too few training rows have data in usable features ({len(train_rows)})"
            )
        X_train_all = _impute(raw[train_rows][:, active_indices], medians)
        y_train = target_values[train_rows]
        if not np.isfinite(y_train).all():
            raise ValueError(f"horizon {horizon}: non-finite target reached MI/model fitting")

        n_neighbors = min(int(mi_neighbors), max(1, len(train_rows) - 1))
        mi = mutual_info_regression(
            X_train_all,
            y_train,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=int(random_state),
            n_jobs=int(mi_n_jobs),
        )
        mi = np.asarray(mi, dtype=np.float64)
        mi[~np.isfinite(mi)] = 0.0
        order = np.argsort(-mi, kind="stable")
        ranking = pd.DataFrame({
            "feature": np.asarray(active_names, dtype=object)[order],
            "mutual_information": mi[order],
        })
        mi_rankings[horizon] = ranking.reset_index(drop=True)
        n_keep = len(active_names) if top_k is None else min(int(top_k), len(active_names))
        chosen_local = order[:n_keep]
        chosen_names = [active_names[j] for j in chosen_local]
        selected_by_horizon[horizon] = chosen_names
        dropped_by_horizon[horizon] = dropped

        X_train = X_train_all[:, chosen_local]
        model_fit = clone(base_model)
        model_fit.fit(X_train, y_train)

        # Impute prediction rows using medians learned from training only.
        X_pred_raw = raw[test_predict_mask][:, active_indices]
        X_pred_all = _impute(X_pred_raw, medians)
        X_pred = X_pred_all[:, chosen_local]
        predictions = np.asarray(model_fit.predict(X_pred), dtype=np.float64).reshape(-1)
        predictions[~np.isfinite(predictions)] = np.nan
        test_rows = np.flatnonzero(test_predict_mask)
        predicted_target = np.full(n, np.nan, dtype=np.float64)
        predicted_target[test_rows] = predictions

        target_suffix = f"t_plus_{horizon}"
        actual_price_col = f"actual_price_{target_suffix}"
        actual_return_col = f"actual_return_{target_suffix}"
        predicted_target_col = f"predicted_{target_kind}_{target_suffix}"
        predicted_price_col = f"predicted_price_{target_suffix}"
        predicted_return_col = f"predicted_return_{target_suffix}"
        error_col = f"error_{target_suffix}"
        price_error_col = f"price_error_{target_suffix}"
        return_error_col = f"return_error_{target_suffix}"

        evaluated[actual_price_col] = future_price
        evaluated[actual_return_col] = future_return
        evaluated[predicted_target_col] = predicted_target
        if target_kind == "return":
            predicted_return = predicted_target
            predicted_price = price * (1.0 + predicted_return)
        else:
            predicted_price = predicted_target.copy()
            predicted_return = np.full(n, np.nan, dtype=np.float64)
            valid_current = np.isfinite(price) & (price > 0) & np.isfinite(predicted_price)
            predicted_return[valid_current] = predicted_price[valid_current] / price[valid_current] - 1.0
        evaluated[predicted_price_col] = predicted_price
        evaluated[predicted_return_col] = predicted_return
        evaluated[error_col] = target_values - predicted_target
        evaluated[price_error_col] = future_price - predicted_price
        evaluated[return_error_col] = future_return - predicted_return

        score_mask = test_score_mask & np.isfinite(predicted_target)
        y_true_test = target_values[score_mask]
        y_pred_test = predicted_target[score_mask]
        if len(y_true_test):
            mae = float(mean_absolute_error(y_true_test, y_pred_test))
            rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
            r2 = float(r2_score(y_true_test, y_pred_test)) if len(y_true_test) >= 2 else np.nan
            corr = float(np.corrcoef(y_true_test, y_pred_test)[0, 1]) if len(y_true_test) >= 2 and np.std(y_true_test) > 0 and np.std(y_pred_test) > 0 else np.nan
            direction = float(np.mean(np.sign(y_true_test) == np.sign(y_pred_test))) if target_kind == "return" else np.nan
        else:
            mae = rmse = r2 = corr = direction = np.nan

        results_by_horizon.append({
            "horizon": horizon,
            "target_kind": target_kind,
            "n_train": int(len(train_rows)),
            "n_test_predictions": int(test_predict_mask.sum()),
            "n_test_scored": int(score_mask.sum()),
            "n_features_available": int(len(active_names)),
            "n_features_selected": int(len(chosen_names)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "prediction_correlation": corr,
            "directional_accuracy": direction,
        })
        fitted_models[horizon] = model_fit

    summary = pd.DataFrame(results_by_horizon).set_index("horizon")
    results: dict[str, Any] = {
        "metrics": summary,
        "mutual_information": mi_rankings,
        "selected_features": selected_by_horizon,
        "dropped_all_missing_features": dropped_by_horizon,
        "models": fitted_models,
        "train_fraction": float(train_fraction),
        "split_row": int(split),
        "target_kind": target_kind,
        "price_column": price_col,
    }
    return evaluated, results


__all__ = [
    "GPFeatureState",
    "generate_gp_features",
    "evaluate_gp_features",
]
