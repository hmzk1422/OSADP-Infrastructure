"""Live supervision, regression, metalabeling, and barrier construction.

This is the live outcome path. ``timeout.py`` in this package is the
dormant idea-only reference and is imported by nothing; the builders below
are the live home of the forward-target and regression mechanics, plus the
metalabel tree and the predicted-barrier assembly.

Pipeline owned by :func:`run_outcomes`:
    1. ``build_supervision_targets``  causal future-return / excursion labels
    2. ``fit_state_linregs``          one multi-output OLS per latent state (t+1..t+n)
    3. ``predict_excursions``         side-adjusted t+1..t+n -> predicted MFE/MAE (TP/SL)
    4. ``fit_metalabel_tree``         tree on linreg predictions + linreg errors ->
                                      take_probability / take_flag (expectancy objective)
    5. ``build_predicted_barriers``   predicted-mode triple barrier on gated rows
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .alignment import attach_raw_columns, chronological_train_mask
from .triple_barrier import build_triple_barrier_sequence


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


def _ordered_prices(df, price_col, timestamp_col):
    """Sort a copy chronologically; return (data, prices, timestamps|None, position)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if price_col not in df.columns:
        raise KeyError(f"Missing price column: {price_col!r}")
    data = df.copy()
    position_col = "__outcome_input_position__"
    while position_col in data.columns:
        position_col = f"_{position_col}"
    data[position_col] = np.arange(len(data), dtype=np.int64)
    if timestamp_col is not None:
        if timestamp_col not in data.columns:
            raise KeyError(f"Missing timestamp column: {timestamp_col!r}")
        order_col = "__outcome_timestamp_order__"
        while order_col in data.columns:
            order_col = f"_{order_col}"
        data[order_col] = pd.to_datetime(data[timestamp_col], utc=True, errors="coerce")
        data = data.sort_values(order_col, kind="mergesort").drop(columns=order_col)
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index(kind="mergesort")
    prices = pd.to_numeric(data[price_col], errors="coerce").to_numpy(dtype=np.float64)
    timestamps = None
    if timestamp_col is not None:
        timestamps = pd.DatetimeIndex(pd.to_datetime(data[timestamp_col], utc=True, errors="coerce"))
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(data.index)
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize("UTC")
        else:
            timestamps = timestamps.tz_convert("UTC")
    if timestamps is not None:
        if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be valid and monotonically increasing")
        timestamps = timestamps.as_unit("ns")
    return data, prices, timestamps, position_col

def build_forward_return_targets(
    df: pd.DataFrame,
    horizons: Sequence = ("1s", "5s", "10s", "20s", "40s", "60s", "2min", "3min", "4min", "5min"),
    price_col: str = "price_close",
    timestamp_col: str | None = None,
    max_lag: str | pd.Timedelta | Mapping[str, str | pd.Timedelta] | None = None,
) -> pd.DataFrame:
    """Create raw future simple-return targets at configurable horizons.

    Live home of the builder formerly hosted in dormant ``timeout.py``.
    String/Timedelta horizons use clock time (first observation at/after
    ``timestamp + horizon``); integer horizons mean that many rows ahead.
    Targets are labels for training/evaluation only; keep them out of live
    feature inputs.
    """
    if not horizons:
        raise ValueError("horizons must contain at least one duration or bar count")
    data, prices, timestamps, position_col = _ordered_prices(df, price_col, timestamp_col)
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
            allowed_lag = _allowed_lag(max_lag, key, label, horizon_delta)
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


def _allowed_lag(max_lag, key, label, duration):
    if isinstance(max_lag, Mapping):
        allowed = max_lag.get(key, max_lag.get(label, duration))
    elif max_lag is None:
        allowed = duration
    else:
        allowed = max_lag
    allowed = pd.to_timedelta(allowed)
    if pd.isna(allowed) or allowed < pd.Timedelta(0):
        raise ValueError("max_lag must be a nonnegative duration or per-horizon mapping")
    return allowed
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

    Live home of the builder formerly hosted in dormant ``timeout.py``.
    Excursions are signed simple returns to the max/min observed price in
    ``(entry, horizon_end]``. Incomplete or stale horizons stay NaN.
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
    data, prices, timestamps, position_col = _ordered_prices(df, price_col, timestamp_col)
    highs = pd.to_numeric(data[high_col], errors="coerce").to_numpy(dtype=np.float64) if high_col else prices
    lows = pd.to_numeric(data[low_col], errors="coerce").to_numpy(dtype=np.float64) if low_col else prices
    time_ns = timestamps.asi8 if timestamps is not None else None
    result = pd.DataFrame(index=data.index)
    for horizon in horizons:
        key = _horizon_key(horizon)
        end_indices = np.full(len(data), -1, dtype=np.int64)
        if isinstance(horizon, (int, np.integer)):
            if horizon <= 0:
                raise ValueError("bar horizons must be positive integers")
            rows = np.arange(len(data), dtype=np.int64) + int(horizon)
            end_indices[rows < len(data)] = rows[rows < len(data)]
            duration = None
            label = f"{int(horizon)}bar"
        else:
            if timestamps is None:
                raise ValueError("DatetimeIndex or timestamp_col is required for duration horizons")
            duration = pd.to_timedelta(horizon)
            if pd.isna(duration) or duration <= pd.Timedelta(0):
                raise ValueError(f"horizon {horizon!r} must be a positive duration")
            requested_ns = time_ns + int(duration.value)
            rows = np.searchsorted(time_ns, requested_ns, side="left")
            end_indices[rows < len(data)] = rows[rows < len(data)]
            label = str(horizon).strip().lower().replace(" ", "")

        valid = (end_indices > np.arange(len(data))) & np.isfinite(prices) & (prices > 0.0)
        if duration is not None and valid.any():
            allowed_lag = _allowed_lag(max_lag, key, label, duration)
            rows = np.flatnonzero(valid)
            overshoot = time_ns[end_indices[rows]] - (time_ns[rows] + int(duration.value))
            valid[rows[overshoot > int(allowed_lag.value)]] = False

        mfe_long = np.full(len(data), np.nan, dtype=np.float64)
        mae_long = np.full(len(data), np.nan, dtype=np.float64)
        mfe_short = np.full(len(data), np.nan, dtype=np.float64)
        mae_short = np.full(len(data), np.nan, dtype=np.float64)

        valid_rows = np.flatnonzero(valid)

        for row in tqdm(valid_rows, desc=f"Calculating Excursions ({key})", leave=False):
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


class StateLinregModel(NamedTuple):
    """Per-state multi-output ridge linreg (hoisted from dormant timeout.py).

    Standardized closed-form least squares: one model maps a state's causal
    features to every ``future_return__*`` target at t+1..t+n.
    """

    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    intercept: np.ndarray  # (targets,)
    coefficients: np.ndarray  # (features, targets)
    feature_mean: np.ndarray  # (features,)
    feature_scale: np.ndarray  # (features,)
    n_train: int
    rank: int

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        x = features.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64)
        x = (x - self.feature_mean) / self.feature_scale
        y = x @ self.coefficients + self.intercept
        return pd.DataFrame(y, index=features.index, columns=list(self.target_columns))


def _fit_linreg(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    train_mask: np.ndarray,
    regime_mask: np.ndarray,
    ridge: float,
) -> StateLinregModel:
    """Closed-form weighted ridge on complete causal training rows."""
    x = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    y = frame.loc[:, target_columns].to_numpy(dtype=np.float64)
    valid = train_mask & regime_mask & np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if int(valid.sum()) <= len(feature_columns) + 1:
        raise ValueError("not enough complete training rows for the selected features and intercept")
    x_train = x[valid]
    y_train = y[valid]
    feature_mean = x_train.mean(axis=0)
    feature_scale = x_train.std(axis=0, ddof=0)
    feature_scale[feature_scale <= 1e-12] = 1.0
    x_train = (x_train - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(x_train)), x_train])
    if ridge > 0.0:
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0  # Do not shrink the intercept.
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
        rank = int(np.linalg.matrix_rank(design))
    else:
        beta, _, rank, _ = np.linalg.lstsq(design, y_train, rcond=None)
    return StateLinregModel(
        feature_columns=tuple(feature_columns),
        target_columns=tuple(target_columns),
        intercept=np.asarray(beta[0], dtype=np.float64),
        coefficients=np.asarray(beta[1:], dtype=np.float64),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        n_train=int(valid.sum()),
        rank=int(rank),
    )


def finite_state_columns(frame: pd.DataFrame, columns: list[str], mask: np.ndarray) -> list[str]:
    """Keep only columns that are fully finite on a state's training rows."""
    subset = frame.loc[mask, columns]
    if subset.empty:
        return []
    values = subset.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=0)
    return [column for column, keep in zip(columns, finite) if keep]


class StateLinregResult(NamedTuple):
    """Outputs of :func:`fit_state_linregs`."""

    predictions: pd.DataFrame  # per-state linreg predictions (one column per target)
    report: pd.DataFrame  # one row per fitted latent state
    models: dict[str, StateLinregModel]  # latent_id -> fitted model


def fit_state_linregs(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    state_labels: pd.Series,
    *,
    train_mask: pd.Series | np.ndarray,
    feature_columns: list[str] | None = None,
    price_col: str | None = "price_close",
    ridge: float = 1.0e-6,
    min_state_rows: int = 60,
    purge_rows: int = 0,
) -> StateLinregResult:
    """Fit one multi-output linreg per known latent state (t+1..t+n).

    Rows of states that never reached ``min_state_rows`` stay NaN. Raises if
    no state could be fitted. ``purge_rows`` drops training rows near the
    chronological split boundary; when the targets carry
    ``future_timestamp__*`` columns the purge instead drops every training
    row whose target endpoint reaches into validation.
    """
    if not targets.columns.str.startswith("future_return__").all():
        raise ValueError("linreg targets must all be future_return__* columns")
    if not isinstance(purge_rows, int) or purge_rows < 0:
        raise ValueError("purge_rows must be an integer >= 0")
    if feature_columns is None:
        excluded = ("future_", "predicted_", "raw__", "__outcome_")
        flags = ("similarity", "trade_allowed", "state_mature", "model_ready", "latent_id")
        feature_columns = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column != price_col and column not in flags and not column.startswith(excluded)
        ]
    if not feature_columns:
        raise ValueError("fit_state_linregs found no numeric feature columns")
    mask = (
        train_mask.to_numpy(dtype=bool)
        if isinstance(train_mask, pd.Series)
        else np.asarray(train_mask, dtype=bool)
    )
    if mask.ndim != 1 or len(mask) != len(frame):
        raise ValueError("train_mask must be one-dimensional and match the frame rows")
    if mask.any():
        # Purge labels whose outcome endpoint reaches into validation.
        boundary_pos = int(np.flatnonzero(~mask)[0]) if (~mask).any() else len(frame)
        timestamp_columns = [
            column for column in targets.columns if column.startswith("future_timestamp__")
        ]
        cutoff = (
            frame.index[boundary_pos]
            if isinstance(frame.index, pd.DatetimeIndex) and boundary_pos < len(frame)
            else None
        )
        purged = mask.copy()
        if timestamp_columns and cutoff is not None:
            # pandas 3: pd.to_datetime rejects DataFrame input for some dtypes,
            # convert per column.
            endpoints = targets[timestamp_columns].apply(
                pd.to_datetime, utc=True, errors="coerce"
            )
            max_endpoint = endpoints.max(axis=1)
            purged &= (max_endpoint < cutoff).to_numpy(dtype=bool)
        elif purge_rows > 0:
            purged[max(0, boundary_pos - purge_rows):boundary_pos] = False
        mask = purged

    predictions = pd.DataFrame(index=frame.index, columns=targets.columns, dtype=np.float64)
    report_rows = []
    models: dict[str, StateLinregModel] = {}
    for state in pd.unique(state_labels):
        if state == "null":
            continue
        state_mask = (state_labels == state).to_numpy()
        fit_mask = mask & state_mask
        if int(fit_mask.sum()) < min_state_rows:
            continue
        state_columns = finite_state_columns(frame, feature_columns, fit_mask)
        if not state_columns:
            continue
        model = _fit_linreg(frame, state_columns, list(targets.columns), mask, state_mask, ridge)
        predicted = model.predict(frame).to_numpy(dtype=np.float64)
        predictions.loc[state_mask] = predicted[state_mask]
        models[str(state)] = model
        report_rows.append(
            {
                "latent_id": state,
                "n_train": model.n_train,
                "n_features": len(state_columns),
                "rank": model.rank,
            }
        )
    report = pd.DataFrame(report_rows, columns=["latent_id", "n_train", "n_features", "rank"])
    if report.empty:
        raise ValueError(
            "fit_state_linregs could not fit any latent state; lower outcome.model.min_state_rows"
        )
    return StateLinregResult(predictions=predictions, report=report, models=models)


def predict_excursions(predictions: pd.DataFrame, side: int = 1) -> pd.DataFrame:
    """Side-adjust t+1..t+n predictions into predicted MFE/MAE barriers.

    The largest side-adjusted predicted return becomes the take-profit width
    (``predicted_mfe``); the smallest becomes the stop-loss width
    (``predicted_mae``, <= 0). Both are added alongside the raw predictions.
    """
    if side not in (-1, 1):
        raise ValueError("side must be +1 (long) or -1 (short)")
    adjusted = pd.DataFrame(
        predictions.to_numpy(dtype=np.float64) * float(side),
        index=predictions.index,
        columns=predictions.columns,
    )
    output = predictions.copy()
    output["predicted_mfe"] = adjusted.max(axis=1)
    output["predicted_mae"] = adjusted.min(axis=1)
    return output


class MetalabelResult(NamedTuple):
    """Outputs of :func:`fit_metalabel_tree`."""

    labels: pd.DataFrame  # take_probability / take_flag for every row
    report: pd.DataFrame  # fit + OOS expectancy diagnostics (single row)
    importances: pd.Series  # tree feature importances (empty when unfit)
    tree: DecisionTreeClassifier | None  # fitted classifier (None when unfit)


def fit_metalabel_tree(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    actual_targets: pd.DataFrame,
    *,
    train_mask: pd.Series | np.ndarray,
    endpoint_times: pd.Series | None = None,
    feature_columns: Sequence[str] | None = None,
    price_col: str | None = "price_close",
    side: int = 1,
    max_depth: int = 3,
    min_samples_leaf: int = 50,
    min_train_rows: int = 200,
    seed: int = 7,
) -> MetalabelResult:
    """Fit the metalabel tree: linreg predictions + linreg errors + features -> take probability.

    Objective (expectancy, not hit-rate): a training row "worked" when the
    realized side-adjusted return across the target horizons met or beat the
    side-adjusted *predicted* return, i.e. the linreg was not optimistically
    wide. ``take_probability >= 0.5`` sets ``take_flag``.

    Features, all causal at prediction time:
      * ``linreg_pred``  side-adjusted linreg point prediction (mean across horizons);
      * ``linreg_err``   most recent *elapsed* linreg error (realized - predicted of a
        row whose full target horizon already ended, located by backward merge on
        ``endpoint_times``, which should be the row-wise **max** future_timestamp so
        the error is fully known);
      * frame features   ``feature_columns`` (e.g. feat_*/gp_feat*); ``None`` selects
        every numeric column except price/targets/predictions. Columns with no finite
        training value are dropped; partially-missing values (state-masked gp_feat*
        outside their latent state, early rows without an elapsed error) are
        zero-filled, so the tree can split on them without emptying the fit.

    ``report`` carries one row: fit counts/depth plus OOS expectancy of taken vs
    skipped validation rows. Returns ``tree=None`` with a diagnostic row when
    fewer than ``min_train_rows`` rows are usable.
    """
    if side not in (-1, 1):
        raise ValueError("side must be +1 (long) or -1 (short)")
    if not predictions.columns.equals(actual_targets.columns):
        raise ValueError("predictions and actual_targets must share the same columns")
    mask = (
        train_mask.to_numpy(dtype=bool)
        if isinstance(train_mask, pd.Series)
        else np.asarray(train_mask, dtype=bool)
    )
    pred_values = predictions.to_numpy(dtype=np.float64)
    actual_values = actual_targets.to_numpy(dtype=np.float64)
    complete = np.isfinite(pred_values).all(axis=1) & np.isfinite(actual_values).all(axis=1)
    predicted = np.full(len(frame), np.nan, dtype=np.float64)
    realized = np.full(len(frame), np.nan, dtype=np.float64)
    predicted[complete] = np.nanmean(pred_values[complete] * float(side), axis=1)
    realized[complete] = np.nanmean(actual_values[complete] * float(side), axis=1)
    worked = np.zeros(len(frame), dtype=np.int64)
    worked[complete] = (realized[complete] >= predicted[complete]).astype(np.int64)

    # --- Feature matrix: linreg scalar(s) + elapsed error + frame features. ---
    feature_frame = pd.DataFrame({"linreg_pred": predicted}, index=frame.index)
    if endpoint_times is not None:
        endpoint = pd.to_datetime(endpoint_times, utc=True, errors="coerce")
        error = pd.Series(realized - predicted, index=frame.index)
        known = pd.DataFrame({"_endpoint": endpoint, "_error": error}).dropna()
        if not known.empty and isinstance(frame.index, pd.DatetimeIndex):
            clock = pd.to_datetime(frame.index, utc=True, errors="coerce")
            merged = pd.merge_asof(
                pd.DataFrame({"_time": clock}).sort_values("_time", kind="mergesort"),
                known.sort_values("_endpoint", kind="mergesort"),
                left_on="_time",
                right_on="_endpoint",
                direction="backward",
            )
            feature_frame["linreg_err"] = merged["_error"].to_numpy(dtype=np.float64)

    if feature_columns is None:
        excluded = ("future_", "predicted_", "__outcome_")
        feature_columns = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column != price_col and not column.startswith(excluded)
        ]
    for column in feature_columns:
        if column not in frame:
            raise KeyError(f"Missing metalabel feature column: {column!r}")
    frame_features = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    feature_frame = pd.concat([feature_frame, frame_features], axis=1)

    labels = pd.DataFrame(index=frame.index)
    labels["take_probability"] = np.nan
    labels["take_flag"] = False
    features = feature_frame.to_numpy(dtype=np.float64)
    fit_ok = mask & complete
    # Drop columns with no finite value among training candidates (e.g. all-NaN
    # gp_feat* when GP never ran on a state); zero-fill partially-missing values
    # so state-masked gp_feat* reads as 0.0 outside its latent state and early
    # rows without an elapsed linreg error read as a 0.0 error.
    if fit_ok.any():
        usable = np.isfinite(features[fit_ok]).any(axis=0)
        if not usable.all():
            features = features[:, usable]
            feature_frame = feature_frame.iloc[:, usable.tolist()]
    features = np.where(np.isfinite(features), features, 0.0)
    n_train = int(fit_ok.sum())
    take_rate = float(worked[fit_ok].mean()) if n_train else np.nan

    def _report(depth: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "n_train": n_train,
                    "depth": depth,
                    "take_rate": take_rate,
                    "n_oos": 0,
                    "oos_take_rate": np.nan,
                    "oos_expectancy_take": np.nan,
                    "oos_expectancy_skip": np.nan,
                    "oos_expectancy_all": np.nan,
                }
            ]
        )

    if n_train < max(int(min_train_rows), 2):
        return MetalabelResult(
            labels=labels,
            report=_report(0),
            importances=pd.Series(dtype=np.float64, name="importance"),
            tree=None,
        )

    tree = DecisionTreeClassifier(
        max_depth=max(1, int(max_depth)),
        min_samples_leaf=max(1, int(min_samples_leaf)),
        random_state=int(seed),
    )
    tree.fit(features[fit_ok], worked[fit_ok])
    classes = list(tree.classes_)
    scores = np.full(len(features), np.nan, dtype=np.float64)
    scorable = np.isfinite(features).all(axis=1)
    probability = tree.predict_proba(features[scorable])
    if 1 in classes:
        scores[scorable] = probability[:, classes.index(1)]
    else:
        scores[scorable] = 0.0
    take_flag = np.zeros(len(features), dtype=bool)
    take_flag[scorable] = scores[scorable] >= 0.5
    labels["take_probability"] = scores
    labels["take_flag"] = take_flag

    importances = pd.Series(
        tree.feature_importances_, index=list(feature_frame.columns), name="importance"
    )

    # OOS expectancy on validation rows the tree could score (side-adjusted mean target).
    oos = (~mask) & complete & scorable
    n_oos = int(oos.sum())
    if n_oos:
        taken = oos & take_flag
        skipped = oos & ~take_flag
        oos_take = float(realized[taken].mean()) if taken.any() else np.nan
        oos_skip = float(realized[skipped].mean()) if skipped.any() else np.nan
        oos_all = float(realized[oos].mean())
        oos_rate = float(take_flag[oos].mean())
    else:
        oos_take = oos_skip = oos_all = oos_rate = np.nan
    report = pd.DataFrame(
        [
            {
                "n_train": n_train,
                "depth": int(tree.get_depth()),
                "take_rate": take_rate,
                "n_oos": n_oos,
                "oos_take_rate": oos_rate,
                "oos_expectancy_take": oos_take,
                "oos_expectancy_skip": oos_skip,
                "oos_expectancy_all": oos_all,
            }
        ]
    )
    return MetalabelResult(labels=labels, report=report, importances=importances, tree=tree)


def build_supervision_targets(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    outcome_cfg: dict | None,
) -> pd.DataFrame:
    """Build the forward-return (+ excursion) supervision targets.

    This is the strategy's ``targets`` step, moved out of ``strategy()``
    (S3E: no config surgery in strategy.py). Returns an empty frame for
    settings files that still use the legacy ``barriers`` section.
    """
    if outcome_cfg is None:
        return pd.DataFrame(index=features.index)

    target_cfg = dict(outcome_cfg.get("forward_returns", {}))
    excursion_cfg = dict(target_cfg.pop("excursions", {}))
    target_source_columns = {
        column for column in (excursion_cfg.get("high_col"), excursion_cfg.get("low_col"))
        if column is not None
    }
    if target_cfg.get("timestamp_col") is not None:
        target_source_columns.add(target_cfg["timestamp_col"])
    target_source = attach_raw_columns(
        raw,
        features,
        target_source_columns,
        target_cfg.get("timestamp_col"),
    ) if target_source_columns else features
    targets = build_forward_return_targets(target_source, **target_cfg)
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
            if column.startswith("future_timestamp__") and column in targets
        ]
        if duplicate_endpoints:
            excursion_targets = excursion_targets.drop(columns=duplicate_endpoints)
        for column in excursion_targets:
            targets[column] = excursion_targets[column].to_numpy()
    return targets


def build_predicted_barriers(
    model_frame: pd.DataFrame,
    excursions: pd.DataFrame,
    metalabel: MetalabelResult | None,
    barrier_cfg: dict,
) -> pd.DataFrame:
    """Gate rows by the metalabel, then build the predicted-mode barrier sequence.

    Thin wrapper over the unchanged ``triple_barrier.py`` (S3D): attaches
    ``predicted_mfe`` / ``predicted_mae`` (from :func:`predict_excursions`),
    drops rows the fitted metalabel rejects (``take_flag == False``; when the
    tree is unfit or unscored the row is kept, since there is no trusted
    decision), then calls ``build_triple_barrier_sequence(mode="predicted")``.
    """
    frame = model_frame.join(excursions[["predicted_mfe", "predicted_mae"]], how="left")
    if metalabel is not None and metalabel.tree is not None:
        take = metalabel.labels["take_flag"].reindex(frame.index)
        # Keep rows without a metalabel decision (NaN probability) or accepted.
        gate = take.fillna(True).to_numpy(dtype=bool)
        frame = frame.loc[gate]
    finite = np.isfinite(frame["predicted_mfe"].to_numpy(dtype=np.float64)) & np.isfinite(
        frame["predicted_mae"].to_numpy(dtype=np.float64)
    )
    frame = frame.loc[finite]
    return build_triple_barrier_sequence(frame, mode="predicted", **barrier_cfg)


class OutcomeResult(NamedTuple):
    """Outputs of :func:`run_outcomes`."""

    predictions: pd.DataFrame  # per-state linreg predictions (future_return__* columns)
    excursions: pd.DataFrame   # side-adjusted predicted_mfe / predicted_mae
    metalabel: MetalabelResult | None  # fitted tree + labels + OOS report
    events: pd.DataFrame       # triple-barrier sequence for the configured mode
    report: pd.DataFrame       # per-state linreg fit report


def run_outcomes(
    model_frame: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    *,
    raw: pd.DataFrame | None = None,
) -> OutcomeResult:
    """Compose the live outcome pipeline (S3B): linreg -> excursions -> metalabel -> barriers.

    Owns the ``outcome.model`` settings section (the old ``[example]``
    knobs). ``cfg`` is the full settings mapping; ``raw`` is only needed to
    causally attach configured barrier columns (high/low/bid/ask) that are
    not already on the model frame.
    """
    outcome_cfg = dict(cfg.get("outcome") or {})
    _inner = outcome_cfg.get("outcome")
    if isinstance(_inner, dict) and ("triple_barrier" in _inner or "forward_returns" in _inner or "model" in _inner):
        outcome_cfg = {**outcome_cfg, **_inner}
    model_cfg = dict(outcome_cfg.get("model") or {})
    barrier_cfg = dict(outcome_cfg.get("triple_barrier") or {})

    # --- 1-2. Per-state linreg on future_return__* (t+1..t+n). ---
    forward_columns = [column for column in targets if column.startswith("future_return__")]
    linreg_targets = targets[forward_columns] if forward_columns else targets.iloc[:, 0:0]
    feature_columns = model_cfg.get("feature_columns") or None
    price_col = barrier_cfg.get("price_col", "price_close")
    train_fraction = float(model_cfg.get("train_fraction", 0.70))
    purge_rows = int(model_cfg.get("purge_rows", 0))
    ridge = float(model_cfg.get("ridge", 1.0e-6))
    min_state_rows = int(model_cfg.get("min_state_rows", 60))
    side = int(model_cfg.get("side", 1))
    if side not in (-1, 1):
        raise ValueError("outcome.model.side must be +1 (long) or -1 (short)")
    train_mask = chronological_train_mask(model_frame, train_fraction, purge_rows)

    predictions = pd.DataFrame(index=model_frame.index, dtype=np.float64)
    report = pd.DataFrame(columns=["latent_id", "n_train", "n_features", "rank"])
    metalabel: MetalabelResult | None = None
    excursions = pd.DataFrame(
        {
            "predicted_mfe": np.full(len(model_frame), np.nan),
            "predicted_mae": np.full(len(model_frame), np.nan),
        },
        index=model_frame.index,
    )
    if not linreg_targets.empty:
        linreg = fit_state_linregs(
            model_frame,
            linreg_targets,
            model_frame["latent_id"],
            train_mask=train_mask,
            feature_columns=feature_columns,
            price_col=price_col,
            ridge=ridge,
            min_state_rows=min_state_rows,
            purge_rows=purge_rows,
        )
        predictions = linreg.predictions
        report = linreg.report

        # --- 3. Side-adjusted t+1..t+n -> predicted MFE/MAE (TP/SL widths). ---
        excursion_predictions = predict_excursions(predictions, side=side)
        excursions = excursion_predictions[["predicted_mfe", "predicted_mae"]].copy()

        # --- 4. Metalabel tree: can the regression be trusted here? ---
        endpoint_columns = [
            column for column in targets if column.startswith("future_timestamp__")
        ]
        endpoint_times = None
        if endpoint_columns:
            # pandas 3: pd.to_datetime rejects DataFrame input, convert per column.
            endpoint_times = (
                targets[endpoint_columns]
                .apply(pd.to_datetime, utc=True, errors="coerce")
                .max(axis=1)
            )
        metalabel = fit_metalabel_tree(
            model_frame,
            predictions,
            linreg_targets,
            train_mask=train_mask,
            endpoint_times=endpoint_times,
            feature_columns=model_cfg.get("metalabel_features") or None,
            price_col=price_col,
            side=side,
            max_depth=int(model_cfg.get("tree_max_depth", 3)),
            min_samples_leaf=int(model_cfg.get("tree_min_samples_leaf", 50)),
            min_train_rows=int(model_cfg.get("tree_min_train_rows", 200)),
            seed=int(model_cfg.get("seed", 7)),
        )

    # --- 5. Barriers: new-style evaluator config OR legacy mode dispatch. ---
    # New style: outcome.triple_barrier carries classifier:/regressor: sections
    # (your settings.yaml) -> model-frame bridge with linreg->metalabel gate.
    # Legacy: mode fixed/dynamic/predicted -> price-action builders (untouched).
    from .triple_barrier import (
        build_model_barriers,
        evaluate_classifier,
        evaluate_regressor,
    )
    if "classifier" in barrier_cfg or "regressor" in barrier_cfg:
        which = barrier_cfg.pop("evaluator", "classifier")
        evaluator = {"classifier": evaluate_classifier,
                     "regressor": evaluate_regressor}[which] if isinstance(which, str) else which
        gate_frame = model_frame
        if metalabel is not None and metalabel.tree is not None:
            take = metalabel.labels["take_flag"].reindex(gate_frame.index)
            gate_frame = gate_frame.loc[take.fillna(True).to_numpy(dtype=bool)]
        result = build_model_barriers(
            gate_frame, raw, cfg, evaluator,
            timestamp_col=barrier_cfg.get("timestamp_col"),
        )
        events = result.trades.rename(columns={
            "r": "return", "rr": "return_rr",
        })
        events["t0"] = result.trades.index
        if "t1" not in events.columns:
            events["t1"] = events["t0"]
        if "barrier" not in events.columns:
            events["barrier"] = np.where(
                events.get("outcome", 0) == 1, "profit",
                np.where(events.get("outcome", 0) == -1, "stop", "vertical"))
        if "side" not in events.columns:
            events["side"] = int(model_cfg.get("side", 1))
    else:
        mode = barrier_cfg.pop("mode", "fixed")
        fixed_cfg = dict(barrier_cfg.pop("fixed", {}))
        dynamic_cfg = dict(barrier_cfg.pop("dynamic", {}))
        predicted_cfg = dict(barrier_cfg.pop("predicted", {}))
        mode_cfg = {"fixed": fixed_cfg, "dynamic": dynamic_cfg, "predicted": predicted_cfg}.get(mode)
        if mode_cfg is None:
            raise ValueError("outcome.triple_barrier.mode must be 'fixed', 'dynamic', or 'predicted'")
        selected_cfg = {**barrier_cfg, **mode_cfg}
        timestamp_col = selected_cfg.get("timestamp_col")
        raw_columns = {
            selected_cfg.get(name)
            for name in (
                "price_col", "signal_col", "high_col", "low_col", "bid_col", "ask_col",
                "volatility_col", "predicted_mfe_col", "predicted_mae_col",
            )
            if selected_cfg.get(name) is not None
        }
        if timestamp_col is not None:
            raw_columns.add(timestamp_col)
        frame = model_frame
        if raw is not None and raw_columns:
            frame = attach_raw_columns(raw, model_frame, raw_columns, timestamp_col)
        if mode == "predicted":
            events = build_predicted_barriers(frame, excursions, metalabel, selected_cfg)
        else:
            events = build_triple_barrier_sequence(frame, mode=mode, **selected_cfg)

    return OutcomeResult(
        predictions=predictions,
        excursions=excursions,
        metalabel=metalabel,
        events=events,
        report=report,
    )




