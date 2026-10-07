"""Clock-time forward-return targets and timeout-selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


def _horizon_key(horizon) -> str:
    if isinstance(horizon, (int, np.integer)):
        if horizon <= 0:
            raise ValueError("bar horizons must be positive integers")
        label = f"{int(horizon)}bar"
    else:
        delta = pd.to_timedelta(horizon)
        if pd.isna(delta) or delta <= pd.Timedelta(0):
            raise ValueError(f"horizon {horizon!r} must be a positive duration")
        label = str(horizon).strip().lower().replace(" ", "")
    return "".join(character if character.isalnum() else "_" for character in label)


def build_forward_return_targets(
    df: pd.DataFrame,
    horizons: Sequence = ("1s", "5s", "10s", "20s", "40s", "60s", "2min", "3min", "4min", "5min"),
    price_col: str = "price_close",
    timestamp_col: str | None = None,
    max_lag: str | pd.Timedelta | Mapping[str, str | pd.Timedelta] | None = None,
) -> pd.DataFrame:
    """Create raw future simple-return targets at configurable horizons.

    String/Timedelta horizons use clock time and select the first observation
    at or after ``timestamp + horizon``. Integer horizons mean that many rows
    ahead. By default, a clock target tolerates overshoot up to one additional
    requested horizon; ``max_lag`` can set a fixed duration or per-target
    tolerances. Targets are labels for training/evaluation only; callers must keep
    them out of live feature inputs. The returned target frame has the same
    rows/index as ``df`` and is intentionally separate from feature columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if price_col not in df.columns:
        raise KeyError(f"Missing price column: {price_col!r}")
    if not horizons:
        raise ValueError("horizons must contain at least one duration or bar count")
    data = df.copy()
    position_col = "__forward_target_input_position__"
    while position_col in data.columns:
        position_col = f"_{position_col}"
    data[position_col] = np.arange(len(data), dtype=np.int64)
    if timestamp_col is not None:
        if timestamp_col not in data.columns:
            raise KeyError(f"Missing timestamp column: {timestamp_col!r}")
        order_col = "__forward_target_timestamp_order__"
        while order_col in data.columns:
            order_col = f"_{order_col}"
        data[order_col] = pd.to_datetime(data[timestamp_col], utc=True, errors="coerce")
        data = data.sort_values(order_col, kind="mergesort").drop(columns=order_col)
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index(kind="mergesort")
    else:
        # Non-datetime indexes are treated as the caller's chronological row order.
        data = data.copy()
    prices = pd.to_numeric(data[price_col], errors="coerce").to_numpy(dtype=np.float64)
    if timestamp_col is not None:
        if timestamp_col not in data.columns:
            raise KeyError(f"Missing timestamp column: {timestamp_col!r}")
        timestamps = pd.DatetimeIndex(pd.to_datetime(data[timestamp_col], utc=True, errors="coerce"))
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(data.index)
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize("UTC")
        else:
            timestamps = timestamps.tz_convert("UTC")
    else:
        timestamps = None

    if timestamps is not None and (timestamps.isna().any() or not timestamps.is_monotonic_increasing):
        raise ValueError("timestamps must be valid and monotonically increasing")
    timestamps = timestamps.as_unit("ns") if timestamps is not None else None
    time_ns = timestamps.asi8 if timestamps is not None else None
    result = pd.DataFrame(index=data.index)
    used_keys = set()

    for horizon in horizons:
        key = _horizon_key(horizon)
        if key in used_keys:
            raise ValueError(f"horizons create duplicate target key {key!r}")
        used_keys.add(key)
        target_indices = np.full(len(data), -1, dtype=np.int64)
        if isinstance(horizon, (int, np.integer)):
            if horizon <= 0:
                raise ValueError("bar horizons must be positive integers")
            rows = np.arange(len(data), dtype=np.int64) + int(horizon)
            valid = rows < len(data)
            target_indices[valid] = rows[valid]
            label = f"{int(horizon)}bar"
            horizon_delta = None
        else:
            if timestamps is None:
                raise ValueError("DatetimeIndex or timestamp_col is required for duration horizons")
            delta = pd.to_timedelta(horizon)
            if pd.isna(delta) or delta <= pd.Timedelta(0):
                raise ValueError(f"horizon {horizon!r} must be a positive duration")
            targets_ns = time_ns + int(delta.value)
            rows = np.searchsorted(time_ns, targets_ns, side="left")
            valid = rows < len(data)
            target_indices[valid] = rows[valid]
            label = str(horizon).strip().lower().replace(" ", "")
            horizon_delta = delta

        future_price = np.full(len(data), np.nan, dtype=np.float64)
        future_time = np.full(len(data), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = (target_indices >= 0) & np.isfinite(prices) & (prices > 0.0)
        valid_rows = np.flatnonzero(valid)
        future_rows = target_indices[valid]
        if timestamps is not None and horizon_delta is not None and valid_rows.size:
            if isinstance(max_lag, Mapping):
                allowed_lag = max_lag.get(key, max_lag.get(label, horizon_delta))
            elif max_lag is None:
                allowed_lag = horizon_delta
            else:
                allowed_lag = max_lag
            allowed_lag = pd.to_timedelta(allowed_lag)
            if pd.isna(allowed_lag) or allowed_lag < pd.Timedelta(0):
                raise ValueError("max_lag must be a nonnegative duration or a mapping of target keys to durations")
            requested_ns = time_ns[valid_rows] + int(horizon_delta.value)
            overshoot_ns = time_ns[future_rows] - requested_ns
            within_tolerance = overshoot_ns <= int(allowed_lag.value)
            valid_rows = valid_rows[within_tolerance]
            future_rows = future_rows[within_tolerance]
        valid_future_price = np.isfinite(prices[future_rows]) & (prices[future_rows] > 0.0)
        valid_rows = valid_rows[valid_future_price]
        future_rows = future_rows[valid_future_price]
        future_price[valid_rows] = prices[future_rows]
        if timestamps is not None:
            future_time[valid_rows] = timestamps.to_numpy(dtype="datetime64[ns]")[future_rows]
        future_return = np.full(len(data), np.nan, dtype=np.float64)
        future_return[valid_rows] = future_price[valid_rows] / prices[valid_rows] - 1.0
        result[f"future_return__{key}"] = future_return
        result[f"future_price__{key}"] = future_price
        if timestamps is not None:
            result[f"future_timestamp__{key}"] = pd.to_datetime(future_time, utc=True)
        result.attrs.setdefault("horizons", {})[key] = label

    original_positions = data[position_col].to_numpy(dtype=np.int64)
    return result.iloc[np.argsort(original_positions, kind="mergesort")]


def build_forward_excursion_targets(
    df: pd.DataFrame,
    horizons: Sequence = ("1s", "5s", "10s"),
    price_col: str = "price_close",
    high_col: str | None = None,
    low_col: str | None = None,
    timestamp_col: str | None = None,
    max_lag: str | pd.Timedelta | Mapping[str, str | pd.Timedelta] | None = None,
) -> pd.DataFrame:
    """Build per-row long/short MFE and MAE targets over each horizon.

    Excursions are simple signed returns from entry price to the maximum or
    minimum observed price in ``(entry, horizon_end]``. If high/low columns
    are omitted, ``price_col`` supplies both extremes. Incomplete or stale
    horizons remain NaN. Target timestamps are included for purging.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if price_col not in df:
        raise KeyError(f"Missing price column: {price_col!r}")
    if (high_col is None) != (low_col is None):
        raise ValueError("high_col and low_col must either both be supplied or both be omitted")
    for column in (high_col, low_col):
        if column is not None and column not in df:
            raise KeyError(f"Missing excursion column: {column!r}")
    if not horizons:
        raise ValueError("horizons must contain at least one duration or bar count")

    data = df.copy()
    position_col = "__excursion_input_position__"
    while position_col in data.columns:
        position_col = f"_{position_col}"
    data[position_col] = np.arange(len(data), dtype=np.int64)
    if timestamp_col is not None:
        if timestamp_col not in data:
            raise KeyError(f"Missing timestamp column: {timestamp_col!r}")
        order_col = "__excursion_timestamp_order__"
        while order_col in data.columns:
            order_col = f"_{order_col}"
        data[order_col] = pd.to_datetime(data[timestamp_col], utc=True, errors="coerce")
        data = data.sort_values(order_col, kind="mergesort").drop(columns=order_col)
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index(kind="mergesort")

    prices = pd.to_numeric(data[price_col], errors="coerce").to_numpy(dtype=np.float64)
    highs = pd.to_numeric(data[high_col], errors="coerce").to_numpy(dtype=np.float64) if high_col is not None else prices
    lows = pd.to_numeric(data[low_col], errors="coerce").to_numpy(dtype=np.float64) if low_col is not None else prices
    if timestamp_col is not None:
        timestamps = pd.DatetimeIndex(pd.to_datetime(data[timestamp_col], utc=True, errors="coerce")).as_unit("ns")
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True, errors="coerce")).as_unit("ns")
    else:
        timestamps = None
    if timestamps is not None and (timestamps.isna().any() or not timestamps.is_monotonic_increasing):
        raise ValueError("timestamps must be valid and monotonically increasing")

    result = pd.DataFrame(index=data.index)
    time_ns = timestamps.asi8 if timestamps is not None else None
    for horizon in horizons:
        key = _horizon_key(horizon)
        end_indices = np.full(len(data), -1, dtype=np.int64)
        if isinstance(horizon, (int, np.integer)):
            if horizon <= 0:
                raise ValueError("bar horizons must be positive integers")
            rows = np.arange(len(data), dtype=np.int64) + int(horizon)
            end_indices[rows < len(data)] = rows[rows < len(data)]
            duration = None
        else:
            if timestamps is None:
                raise ValueError("DatetimeIndex or timestamp_col is required for duration horizons")
            duration = pd.to_timedelta(horizon)
            if pd.isna(duration) or duration <= pd.Timedelta(0):
                raise ValueError(f"horizon {horizon!r} must be a positive duration")
            requested_ns = time_ns + int(duration.value)
            rows = np.searchsorted(time_ns, requested_ns, side="left")
            end_indices[rows < len(data)] = rows[rows < len(data)]

        valid = (end_indices > np.arange(len(data))) & np.isfinite(prices) & (prices > 0.0)
        if duration is not None and valid.any():
            if isinstance(max_lag, Mapping):
                allowed_lag = max_lag.get(key, max_lag.get(str(horizon).strip().lower().replace(" ", ""), duration))
            else:
                allowed_lag = duration if max_lag is None else max_lag
            allowed_lag = pd.to_timedelta(allowed_lag)
            if pd.isna(allowed_lag) or allowed_lag < pd.Timedelta(0):
                raise ValueError("max_lag must be a nonnegative duration or per-horizon mapping")
            rows = np.flatnonzero(valid)
            overshoot = time_ns[end_indices[rows]] - (time_ns[rows] + int(duration.value))
            valid[rows[overshoot > int(allowed_lag.value)]] = False

        mfe_long = np.full(len(data), np.nan, dtype=np.float64)
        mae_long = np.full(len(data), np.nan, dtype=np.float64)
        mfe_short = np.full(len(data), np.nan, dtype=np.float64)
        mae_short = np.full(len(data), np.nan, dtype=np.float64)
        
        for row in tqdm(np.flatnonzero(valid), desc=f"Excursions ({key})", leave=False):
            end = int(end_indices[row])
            path_high = highs[row + 1:end + 1]
            path_low = lows[row + 1:end + 1]
            finite_high = path_high[np.isfinite(path_high)]
            finite_low = path_low[np.isfinite(path_low)]
            if finite_high.size and finite_low.size:
                entry = prices[row]
                positive_high = finite_high[finite_high > 0.0]
                positive_low = finite_low[finite_low > 0.0]
                if positive_high.size and positive_low.size:
                    mfe_long[row] = max(float(np.max(positive_high) / entry - 1.0), 0.0)
                    mae_long[row] = min(float(np.min(positive_low) / entry - 1.0), 0.0)
                    mfe_short[row] = max(float(1.0 - np.min(positive_low) / entry), 0.0)
                    mae_short[row] = min(float(1.0 - np.max(positive_high) / entry), 0.0)
        result[f"future_mfe_long__{key}"] = mfe_long
        result[f"future_mae_long__{key}"] = mae_long
        result[f"future_mfe_short__{key}"] = mfe_short
        result[f"future_mae_short__{key}"] = mae_short
        if timestamps is not None:
            target_time = np.full(len(data), np.datetime64("NaT"), dtype="datetime64[ns]")
            target_rows = np.flatnonzero(valid)
            target_time[target_rows] = timestamps.to_numpy(dtype="datetime64[ns]")[end_indices[target_rows]]
            result[f"future_timestamp__{key}"] = pd.to_datetime(target_time, utc=True)

    original_positions = data[position_col].to_numpy(dtype=np.int64)
    return result.iloc[np.argsort(original_positions, kind="mergesort")]


@dataclass(frozen=True)
class LinearTimeoutModel:
    """OLS multi-output model, fitted only on rows explicitly marked for train."""

    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    coefficients: np.ndarray
    intercept: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    n_train: int
    rank: int

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise KeyError(f"Missing model feature columns: {missing}")
        values = features.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64)
        predictions = np.full((len(features), len(self.target_columns)), np.nan, dtype=np.float64)
        valid = np.isfinite(values).all(axis=1)
        standardized = (values - self.feature_mean) / self.feature_scale
        predictions[valid] = standardized[valid] @ self.coefficients + self.intercept
        return pd.DataFrame(predictions, index=features.index, columns=self.target_columns)


def fit_timeout_regression(
    features: pd.DataFrame,
    forward_targets: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str] | None = None,
    train_mask: pd.Series | np.ndarray | None = None,
    ridge: float = 0.0,
    regime_mask: pd.Series | np.ndarray | None = None,
    train_end_time=None,
    purge_rows: int | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> LinearTimeoutModel:
    """Fit multi-horizon linear regression from features to future returns.

    ``train_mask`` declares a chronological time prefix, while ``regime_mask``
    independently selects rows for a state-specific model. The model
    predicts raw signed underlying returns for each target horizon; a trade
    side can then be applied with :func:`select_timeout_horizon`. Optional
    ``sample_weight`` supports event uniqueness weights for overlapping labels.
    """
    if not features.index.equals(forward_targets.index):
        raise ValueError("features and forward_targets must have identical indexes")
    feature_columns = tuple(feature_columns)
    target_columns = tuple(
        target_columns
        if target_columns is not None
        else [c for c in forward_targets if c.startswith("future_return__")]
    )
    if not feature_columns or not target_columns:
        raise ValueError("feature_columns and target_columns must be nonempty")
    missing_features = [column for column in feature_columns if column not in features]
    missing_targets = [column for column in target_columns if column not in forward_targets]
    if missing_features or missing_targets:
        raise KeyError(f"Missing features {missing_features}; missing targets {missing_targets}")
    if train_mask is None:
        raise ValueError("train_mask is required; pass an explicit chronological training mask")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and >= 0")

    if isinstance(train_mask, pd.Series):
        if not train_mask.index.equals(features.index):
            raise ValueError("train_mask Series must have the same index as features")
        mask = train_mask.fillna(False).to_numpy(dtype=bool)
    else:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(features):
            raise ValueError("train_mask must be one-dimensional and match the rows")
    seen_validation_row = np.maximum.accumulate(~mask)
    if np.any(mask & seen_validation_row):
        raise ValueError("train_mask must be a chronological prefix; training cannot resume after validation rows")
    if purge_rows is not None and (not isinstance(purge_rows, int) or purge_rows < 0):
        raise ValueError("purge_rows must be an integer >= 0 or None")
    if sample_weight is None:
        weights = np.ones(len(features), dtype=np.float64)
    elif isinstance(sample_weight, pd.Series):
        if not sample_weight.index.equals(features.index):
            raise ValueError("sample_weight Series must have the same index as features")
        weights = pd.to_numeric(sample_weight, errors="coerce").to_numpy(dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(features):
            raise ValueError("sample_weight must be one-dimensional and match the rows")
    if np.any(np.isfinite(weights) & (weights < 0.0)):
        raise ValueError("sample_weight values must be nonnegative")

    if regime_mask is None:
        regime = np.ones(len(features), dtype=bool)
    elif isinstance(regime_mask, pd.Series):
        if not regime_mask.index.equals(features.index):
            raise ValueError("regime_mask Series must have the same index as features")
        regime = regime_mask.fillna(False).to_numpy(dtype=bool)
    else:
        regime = np.asarray(regime_mask, dtype=bool)
        if regime.ndim != 1 or len(regime) != len(features):
            raise ValueError("regime_mask must be one-dimensional and match the rows")

    # Purge training labels whose outcome endpoint reaches into validation.
    train = mask.copy()
    false_positions = np.flatnonzero(~mask)
    boundary_pos = int(false_positions[0]) if false_positions.size else len(features)
    cutoff = train_end_time
    if cutoff is None and isinstance(features.index, pd.DatetimeIndex) and boundary_pos < len(features):
        cutoff = features.index[boundary_pos]
    endpoint_columns = [
        target.replace("future_return__", "future_timestamp__", 1)
        for target in target_columns
        if target.startswith("future_return__")
        and target.replace("future_return__", "future_timestamp__", 1) in forward_targets
    ]
    if cutoff is not None and len(endpoint_columns) == len(target_columns):
        cutoff = pd.to_datetime(cutoff, utc=True)
        if isinstance(features.index, pd.DatetimeIndex):
            feature_times = pd.to_datetime(features.index, utc=True, errors="coerce")
            train &= (feature_times < cutoff).to_numpy()
        for column in endpoint_columns:
            endpoint = pd.to_datetime(forward_targets[column], utc=True, errors="coerce")
            train &= (endpoint < cutoff).to_numpy()
    elif purge_rows is not None:
        train[max(0, boundary_pos - purge_rows):boundary_pos] = False
    else:
        raise ValueError(
            "cannot purge target overlap: provide future_timestamp__* columns, "
            "train_end_time, or an explicit purge_rows embargo"
        )
    x = features.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
    y = forward_targets.loc[:, list(target_columns)].to_numpy(dtype=np.float64)
    valid = train & regime & np.isfinite(weights) & (weights > 0.0) & np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if int(valid.sum()) <= len(feature_columns) + 1:
        raise ValueError("not enough complete training rows for the selected features and intercept")

    x_train = x[valid]
    y_train = y[valid]
    train_weights = weights[valid]
    feature_mean = np.average(x_train, axis=0, weights=train_weights)
    feature_variance = np.average((x_train - feature_mean) ** 2, axis=0, weights=train_weights)
    feature_scale = np.sqrt(feature_variance)
    feature_scale[feature_scale <= 1e-12] = 1.0
    x_train = (x_train - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(x_train)), x_train])
    sqrt_weights = np.sqrt(train_weights)[:, None]
    weighted_design = design * sqrt_weights
    weighted_targets = y_train * sqrt_weights
    if ridge > 0.0:
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0  # Do not shrink the intercept.
        beta = np.linalg.solve(weighted_design.T @ weighted_design + penalty, weighted_design.T @ weighted_targets)
        rank = int(np.linalg.matrix_rank(design))
    else:
        beta, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)
    return LinearTimeoutModel(
        feature_columns=feature_columns,
        target_columns=target_columns,
        intercept=np.asarray(beta[0], dtype=np.float64),
        coefficients=np.asarray(beta[1:], dtype=np.float64),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        n_train=int(valid.sum()),
        rank=int(rank),
    )


def select_timeout_horizon(
    predicted_returns: pd.DataFrame,
    side: int | pd.Series,
    min_expected_return: float = 0.0,
    cost_bps_per_side: float = 0.0,
    horizon_seconds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Choose a duration-normalized, cost-adjusted positive-return timeout.

    The score is estimated net directional return divided by square-root
    duration. Bar-based target names need an explicit ``horizon_seconds`` entry.
    This is a ranking rule, not a guarantee of profit.
    """
    if not np.isfinite(min_expected_return):
        raise ValueError("min_expected_return must be finite")
    if not np.isfinite(cost_bps_per_side) or cost_bps_per_side < 0.0:
        raise ValueError("cost_bps_per_side must be finite and >= 0")
    columns = [column for column in predicted_returns if column.startswith("future_return__")]
    if not columns:
        raise ValueError("predicted_returns must contain future_return__* columns")
    values = predicted_returns[columns].to_numpy(dtype=np.float64)
    if isinstance(side, pd.Series):
        if not side.index.equals(predicted_returns.index):
            raise ValueError("side Series must have the same index as predicted_returns")
        sides = side.to_numpy(dtype=np.float64)
    else:
        sides = np.full(len(predicted_returns), side, dtype=np.float64)
    if not np.isfinite(sides).all() or not np.isin(sides, (-1.0, 0.0, 1.0)).all():
        raise ValueError("side must contain only -1, 0, or +1")
    labels = np.asarray([column.removeprefix("future_return__") for column in columns], dtype=object)
    durations = []
    for column, label in zip(columns, labels):
        duration = None
        if horizon_seconds is not None:
            duration = horizon_seconds.get(column, horizon_seconds.get(str(label)))
        if duration is None and not str(label).endswith("bar"):
            try:
                duration = pd.to_timedelta(str(label)).total_seconds()
            except (TypeError, ValueError):
                duration = None
        if duration is None or not np.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"provide a positive horizon_seconds value for target {column!r}")
        durations.append(float(duration))

    directional = values * sides[:, None]
    total_cost = 2.0 * float(cost_bps_per_side) / 10_000.0
    net = directional - total_cost
    score = net / np.sqrt(np.asarray(durations, dtype=np.float64))[None, :]
    eligible = np.isfinite(net) & (net >= min_expected_return) & (sides[:, None] != 0.0)
    score[~eligible] = -np.inf
    best_index = np.argmax(score, axis=1)
    row_index = np.arange(len(directional))
    best_score = score[row_index, best_index]
    best_gross = directional[row_index, best_index]
    best_net = net[row_index, best_index]
    selected = np.isfinite(best_score)
    result = pd.DataFrame(index=predicted_returns.index)
    result["timeout_horizon"] = np.where(selected, labels[best_index], None)
    result["predicted_strategy_return"] = np.where(selected, best_net, np.nan)
    result["predicted_gross_return"] = np.where(selected, best_gross, np.nan)
    result["estimated_cost_return"] = np.where(selected, total_cost, np.nan)
    result["timeout_score"] = np.where(selected, best_score, np.nan)
    return result
