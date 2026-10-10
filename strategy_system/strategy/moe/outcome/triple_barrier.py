"""Fixed, volatility-scaled, and predicted triple-barrier event outcomes.

These builders consume a causal side/entry column and raw price observations.
They do not depend on GP-generated features, OOD embeddings, or a model.

Two evaluation layers live here:

* ``tb_classifier()`` / ``tb_regressor()`` -- the current entry points.
  Give them a DataFrame (they resolve ``timeout`` once via
  ``models.timeout_function`` and build their own purged ``BarrierData``)
  or a prebuilt ``BarrierData``. All width math delegates to ``models.py``
  so it stays swappable per run or per edit.
* ``evaluate_classifier()`` / ``evaluate_regressor()`` -- compatibility
  wrappers that keep the legacy settings keys (rr/vol_col/vol_window) and
  legacy defaults while delegating to the ``tb_*`` core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import models as _barrier_models


#: BTCUSDT instrument tick -- same formula as loader.quantize_price and
#: uets VenueSpec::round_price: (price / tick).round() * tick.
TICK_SIZE = 0.01


def quantize_price(price, tick: float = TICK_SIZE):
    """Snap price(s) to the instrument tick (exact Decimal grid value)."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(str(tick))

    def _one(v):
        import math
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        if not math.isfinite(f):
            return v
        if isinstance(v, float) and math.isnan(f):
            return v
        return float(Decimal(str(f)).quantize(q, rounding=ROUND_HALF_UP))

    try:
        if hasattr(price, "apply"):
            return price.apply(_one)
        return _one(price)
    except Exception:
        return price


_EVENT_COLUMNS = [
    "t0", "t1", "entry_price", "reference_entry_price", "side",
    "profit_barrier", "stop_barrier", "exit_price", "reference_exit_price",
    "gross_return", "fee_return", "return", "label", "meta_label",
    "barrier", "bars_held", "holding_time", "pt_width", "sl_width",
    "mfe", "mae", "uniqueness_weight",
]


def _ordered_frame(df: pd.DataFrame, timestamp_col: str | None = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if timestamp_col is not None and timestamp_col in df.columns:
        order_col = "__outcome_timestamp_order__"
        while order_col in df.columns:
            order_col = f"_{order_col}"
        data = df.assign(**{order_col: pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")})
        data = data.sort_values(order_col, kind="mergesort").drop(columns=order_col)
    else:
        if timestamp_col is not None and not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError(f"Missing timestamp column: {timestamp_col!r}")
        data = df.sort_index(kind="mergesort").copy()
    if timestamp_col is None and not data.index.is_monotonic_increasing:
        raise ValueError("df index must be sortable in increasing order")
    return data


def _timestamps(data: pd.DataFrame, timestamp_col: str | None) -> pd.DatetimeIndex | None:
    if timestamp_col is not None:
        if timestamp_col in data:
            values = pd.to_datetime(data[timestamp_col], utc=True, errors="coerce")
        elif isinstance(data.index, pd.DatetimeIndex):
            values = pd.to_datetime(data.index, utc=True, errors="coerce")
        else:
            raise KeyError(f"Missing timestamp column: {timestamp_col!r}")
    elif isinstance(data.index, pd.DatetimeIndex):
        values = pd.to_datetime(data.index, utc=True, errors="coerce")
    else:
        return None
    timestamps = pd.DatetimeIndex(values).as_unit("ns")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be valid and monotonically increasing")
    return timestamps


def _parse_horizon_value(value) -> tuple[pd.Timedelta | None, int | None]:
    if isinstance(value, str) and value.strip().lower().endswith("bar"):
        try:
            bars = int(value.strip()[:-3])
        except ValueError:
            return None, None
        return (None, bars) if bars > 0 else (None, None)
    if isinstance(value, (int, np.integer)):
        bars = int(value)
        return (None, bars) if bars > 0 else (None, None)
    try:
        duration = pd.to_timedelta(value)
    except (TypeError, ValueError):
        return None, None
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        return None, None
    return duration, None


def _validate_inputs(
    data: pd.DataFrame,
    signal_col: str,
    price_col: str,
    high_col: str | None,
    low_col: str | None,
    horizon_bars: int | None,
    horizon: pd.Timedelta | None = None,
    horizon_col: str | None = None,
    timestamp_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    fee_bps_per_side: float = 0.0,
) -> None:
    if horizon is None and horizon_col is None:
        if not isinstance(horizon_bars, (int, np.integer)) or horizon_bars <= 0:
            raise ValueError("horizon_bars must be a positive integer when no duration horizon is set")
    if horizon is not None and (pd.isna(horizon) or horizon <= pd.Timedelta(0)):
        raise ValueError("horizon must be a positive duration")
    if (bid_col is None) != (ask_col is None):
        raise ValueError("bid_col and ask_col must either both be supplied or both be omitted")
    if not np.isfinite(fee_bps_per_side) or fee_bps_per_side < 0.0:
        raise ValueError("fee_bps_per_side must be finite and >= 0")
    if (high_col is None) != (low_col is None):
        raise ValueError("high_col and low_col must either both be supplied or both be omitted")
    required = [signal_col, price_col]
    if horizon_col is not None:
        required.append(horizon_col)
    if bid_col is not None:
        required.extend([bid_col, ask_col])
    if high_col is not None:
        required.extend([high_col, low_col])
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if timestamp_col is not None and timestamp_col not in data and not isinstance(data.index, pd.DatetimeIndex):
        raise KeyError(f"Missing timestamp column: {timestamp_col!r}")
    if horizon_col is not None:
        horizons = data[horizon_col].dropna()
        if any(_parse_horizon_value(value) == (None, None) for value in horizons):
            raise ValueError(f"{horizon_col!r} must contain positive bar counts or positive durations")
        if any(_parse_horizon_value(value)[0] is not None for value in horizons) and _timestamps(data, timestamp_col) is None:
            raise ValueError("duration values in horizon_col require a DatetimeIndex or timestamp_col")


def _entry_events(
    data: pd.DataFrame,
    *,
    signal_col: str,
    price_col: str,
    pt_widths: np.ndarray,
    sl_widths: np.ndarray,
    horizon_bars: int | None,
    horizon: pd.Timedelta | None,
    horizon_col: str | None,
    timestamps: pd.DatetimeIndex | None,
    high_col: str | None,
    low_col: str | None,
    drop_incomplete: bool,
    bid_col: str | None,
    ask_col: str | None,
    fee_bps_per_side: float,
) -> pd.DataFrame:
    prices = pd.to_numeric(data[price_col], errors="coerce").to_numpy(dtype=np.float64)
    sides_raw = pd.to_numeric(data[signal_col], errors="coerce").fillna(0).to_numpy()
    highs = (
        pd.to_numeric(data[high_col], errors="coerce").to_numpy(dtype=np.float64)
        if high_col is not None else prices
    )
    lows = (
        pd.to_numeric(data[low_col], errors="coerce").to_numpy(dtype=np.float64)
        if low_col is not None else prices
    )
    bids = pd.to_numeric(data[bid_col], errors="coerce").to_numpy(dtype=np.float64) if bid_col else prices
    asks = pd.to_numeric(data[ask_col], errors="coerce").to_numpy(dtype=np.float64) if ask_col else prices
    times = data.index.to_numpy()
    events = []
    event_spans: list[tuple[int, int]] = []
    time_ns = timestamps.asi8 if timestamps is not None else None
    fee_rate = float(fee_bps_per_side) / 10_000.0

    for i, raw_side in enumerate(tqdm(sides_raw, desc="Evaluating Triple Barriers")):
        if raw_side == 0 or not np.isfinite(raw_side):
            continue
        if raw_side not in (-1, 1):
            raise ValueError(f"{signal_col} must contain only -1, 0, +1; found {raw_side!r} at row {i}")
        side = int(raw_side)
        entry = prices[i]
        pt_width = float(pt_widths[i])
        sl_width = float(sl_widths[i])
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        if not np.isfinite(pt_width) or not np.isfinite(sl_width) or pt_width <= 0.0 or sl_width <= 0.0:
            continue
        if horizon_col is not None:
            raw_horizon = data[horizon_col].iloc[i]
            if pd.isna(raw_horizon):
                continue
            event_horizon, event_horizon_bars = _parse_horizon_value(raw_horizon)
            if event_horizon is None and event_horizon_bars is None:
                continue
        else:
            event_horizon = horizon
            event_horizon_bars = None if horizon is not None else horizon_bars

        if event_horizon is not None:
            if timestamps is None:
                raise ValueError("duration horizons require a DatetimeIndex or timestamp_col")
            target_time = time_ns[i] + int(event_horizon.value)
            end = int(np.searchsorted(time_ns, target_time, side="left"))
            if end >= len(data):
                if drop_incomplete:
                    continue
                end = len(data) - 1
        else:
            if event_horizon_bars is None:
                raise ValueError("horizon_bars is required when no duration horizon is set")
            if drop_incomplete and i + event_horizon_bars >= len(data):
                continue
            end = min(i + event_horizon_bars, len(data) - 1)
        if end <= i:
            continue

        reference_entry = entry
        entry_fill = quantize_price(asks[i] if side > 0 else bids[i], TICK_SIZE)
        if not np.isfinite(entry_fill) or entry_fill <= 0.0:
            continue
        profit_level = quantize_price(entry_fill * (1.0 + side * pt_width), TICK_SIZE)
        stop_level = quantize_price(entry_fill * (1.0 - side * sl_width), TICK_SIZE)
        if profit_level <= 0.0 or stop_level <= 0.0:
            continue
        exit_idx = end
        barrier = "vertical"

        for j in range(i + 1, end + 1):
            high, low = highs[j], lows[j]
            if not np.isfinite(high) or not np.isfinite(low):
                continue
            profit_hit = high >= profit_level if side == 1 else low <= profit_level
            stop_hit = low <= stop_level if side == 1 else high >= stop_level
            if stop_hit:  # Conservative tie-break if both barriers touch in one bar.
                exit_idx, barrier = j, "stop"
                break
            if profit_hit:
                exit_idx, barrier = j, "profit"
                break

        reference_exit = profit_level if barrier == "profit" else stop_level if barrier == "stop" else prices[exit_idx]
        exit_fill = quantize_price(bids[exit_idx] if side > 0 else asks[exit_idx], TICK_SIZE)
        if not np.isfinite(reference_exit) or reference_exit <= 0.0 or not np.isfinite(exit_fill) or exit_fill <= 0.0:
            continue

        path_high = highs[i + 1:exit_idx + 1]
        path_low = lows[i + 1:exit_idx + 1]
        finite_high = path_high[np.isfinite(path_high)]
        finite_low = path_low[np.isfinite(path_low)]
        if side == 1:
            mfe = float(np.max(finite_high) / reference_entry - 1.0) if finite_high.size else 0.0
            mae = float(np.min(finite_low) / reference_entry - 1.0) if finite_low.size else 0.0
        else:
            mfe = float(-(np.min(finite_low) / reference_entry - 1.0)) if finite_low.size else 0.0
            mae = float(-(np.max(finite_high) / reference_entry - 1.0)) if finite_high.size else 0.0
        mfe = max(mfe, 0.0)
        mae = min(mae, 0.0)

        gross_return = side * (exit_fill / entry_fill - 1.0)
        fee_return = fee_rate * (1.0 + exit_fill / entry_fill)
        strategy_return = gross_return - fee_return
        label = 1 if barrier == "profit" else (-1 if barrier == "stop" else 0)
        if timestamps is not None:
            holding_time = timestamps[exit_idx] - timestamps[i]
        else:
            holding_time = pd.NaT
        events.append({
            "t0": times[i], "t1": times[exit_idx], "entry_price": entry_fill,
            "reference_entry_price": reference_entry,
            "side": side, "profit_barrier": profit_level, "stop_barrier": stop_level,
            "exit_price": exit_fill, "reference_exit_price": reference_exit,
            "gross_return": gross_return, "fee_return": fee_return, "return": strategy_return,
            "label": label, "meta_label": int(strategy_return > 0.0),
            "barrier": barrier, "bars_held": exit_idx - i, "holding_time": holding_time,
            "pt_width": pt_width, "sl_width": sl_width, "mfe": mfe, "mae": mae,
            "uniqueness_weight": np.nan,
        })
        event_spans.append((i, exit_idx))

    result = pd.DataFrame(events, columns=_EVENT_COLUMNS)
    if event_spans:
        concurrency_delta = np.zeros(len(data) + 1, dtype=np.int64)
        for start, end in event_spans:
            concurrency_delta[start] += 1
            concurrency_delta[end + 1] -= 1
        concurrency = np.cumsum(concurrency_delta[:-1])
        inverse_concurrency = np.divide(
            1.0,
            concurrency,
            out=np.zeros(len(data), dtype=np.float64),
            where=concurrency > 0,
        )
        prefix = np.concatenate([[0.0], np.cumsum(inverse_concurrency)])
        result["uniqueness_weight"] = [
            (prefix[end + 1] - prefix[start]) / (end - start + 1)
            for start, end in event_spans
        ]
    return result


def build_triple_barrier_fixed(
    df: pd.DataFrame,
    signal_col: str = "side",
    price_col: str = "price_close",
    pt: float = 0.005,
    sl: float = 0.003,
    horizon_bars: int | None = 30,
    horizon: str | pd.Timedelta | None = None,
    horizon_col: str | None = None,
    timestamp_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    fee_bps_per_side: float = 0.0,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build fixed-percentage PT/SL labels with a bar or clock-time timeout.

    ``side`` in ``signal_col`` is +1 for long, -1 for short, and 0 for no
    event. When both barriers touch within one observation, the stop wins as
    a conservative tie-break. Incomplete vertical horizons are omitted by
    default so end-of-sample events do not get artificially shorter timeouts.
    """
    if not np.isfinite(pt) or not 0.0 < pt < 1.0 or not np.isfinite(sl) or not 0.0 < sl < 1.0:
        raise ValueError("pt and sl must be finite and between 0 and 1")
    duration = pd.to_timedelta(horizon) if horizon is not None else None
    data = _ordered_frame(df, timestamp_col)
    timestamps = _timestamps(data, timestamp_col)
    if duration is not None and timestamps is None:
        raise ValueError("duration horizons require a DatetimeIndex or timestamp_col")
    _validate_inputs(
        data, signal_col, price_col, high_col, low_col, horizon_bars,
        duration, horizon_col, timestamp_col, bid_col, ask_col, fee_bps_per_side,
    )
    size = len(data)
    return _entry_events(
        data,
        signal_col=signal_col,
        price_col=price_col,
        pt_widths=np.full(size, float(pt)),
        sl_widths=np.full(size, float(sl)),
        horizon_bars=int(horizon_bars) if horizon_bars is not None else None,
        horizon=duration,
        horizon_col=horizon_col,
        timestamps=timestamps,
        high_col=high_col,
        low_col=low_col,
        drop_incomplete=drop_incomplete,
        bid_col=bid_col,
        ask_col=ask_col,
        fee_bps_per_side=fee_bps_per_side,
    )


def build_triple_barrier_dynamic(
    df: pd.DataFrame,
    signal_col: str = "side",
    price_col: str = "price_close",
    horizon_bars: int | None = 30,
    horizon: str | pd.Timedelta | None = None,
    horizon_col: str | None = None,
    timestamp_col: str | None = None,
    volatility_col: str | None = None,
    volatility_lookback: int = 60,
    volatility_min_periods: int = 20,
    pt_sigma: float = 2.0,
    sl_sigma: float = 1.0,
    high_col: str | None = None,
    low_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    fee_bps_per_side: float = 0.0,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build PT/SL widths from entry volatility, scaled to the timeout.

    If ``volatility_col`` is omitted, volatility is the rolling standard
    deviation of simple bar returns. Barrier widths are
    ``multiplier * entry_sigma * sqrt(horizon_bars)``. For clock horizons,
    volatility per observation is scaled by ``sqrt(horizon_seconds / median
    observation_seconds)``. This changes barrier
    distance to the requested risk horizon; it does not claim predicted MFE or
    guarantee profitable timeouts. Supply causal OHLC high/low columns when
    available; otherwise touches are determined from the close price only.
    """
    if not isinstance(volatility_lookback, (int, np.integer)) or volatility_lookback <= 0:
        raise ValueError("volatility_lookback must be a positive integer")
    if not isinstance(volatility_min_periods, (int, np.integer)) or not 1 <= volatility_min_periods <= volatility_lookback:
        raise ValueError("volatility_min_periods must be in [1, volatility_lookback]")
    if not np.isfinite(pt_sigma) or pt_sigma <= 0.0 or not np.isfinite(sl_sigma) or sl_sigma <= 0.0:
        raise ValueError("pt_sigma and sl_sigma must be finite and > 0")
    duration = pd.to_timedelta(horizon) if horizon is not None else None
    data = _ordered_frame(df, timestamp_col)
    timestamps = _timestamps(data, timestamp_col)
    if duration is not None and timestamps is None:
        raise ValueError("duration horizons require a DatetimeIndex or timestamp_col")
    _validate_inputs(
        data, signal_col, price_col, high_col, low_col, horizon_bars,
        duration, horizon_col, timestamp_col, bid_col, ask_col, fee_bps_per_side,
    )
    if volatility_col is not None:
        if volatility_col not in data.columns:
            raise KeyError(f"Missing volatility column: {volatility_col!r}")
        sigma = pd.to_numeric(data[volatility_col], errors="coerce")
    else:
        prices = pd.to_numeric(data[price_col], errors="coerce")
        returns = prices.pct_change(fill_method=None)
        sigma = returns.rolling(
            window=int(volatility_lookback),
            min_periods=int(volatility_min_periods),
        ).std(ddof=0)
    sigma = sigma.clip(lower=0.0).to_numpy(dtype=np.float64)
    horizon_scales = np.full(len(data), np.nan, dtype=np.float64)
    if timestamps is not None and len(timestamps) > 1:
        deltas = np.diff(timestamps.asi8).astype(np.float64) / 1e9
        positive_deltas = deltas[deltas > 0.0]
        median_interval_seconds = float(np.median(positive_deltas)) if positive_deltas.size else np.nan
    else:
        median_interval_seconds = np.nan
    for i in range(len(data)):
        if horizon_col is not None:
            raw_horizon = data[horizon_col].iloc[i]
            if pd.isna(raw_horizon):
                continue
            duration_i, bars_i = _parse_horizon_value(raw_horizon)
        else:
            duration_i = duration
            bars_i = None if duration is not None else horizon_bars
        if duration_i is not None:
            if not np.isfinite(median_interval_seconds) or median_interval_seconds <= 0.0:
                continue
            horizon_scales[i] = np.sqrt(duration_i.total_seconds() / median_interval_seconds)
        elif bars_i is not None and bars_i > 0:
            horizon_scales[i] = np.sqrt(bars_i)
    pt_widths = sigma * horizon_scales * float(pt_sigma)
    sl_widths = sigma * horizon_scales * float(sl_sigma)
    return _entry_events(
        data,
        signal_col=signal_col,
        price_col=price_col,
        pt_widths=pt_widths,
        sl_widths=sl_widths,
        horizon_bars=int(horizon_bars) if horizon_bars is not None else None,
        horizon=duration,
        horizon_col=horizon_col,
        timestamps=timestamps,
        high_col=high_col,
        low_col=low_col,
        drop_incomplete=drop_incomplete,
        bid_col=bid_col,
        ask_col=ask_col,
        fee_bps_per_side=fee_bps_per_side,
    )


def build_triple_barrier_predicted(
    df: pd.DataFrame,
    signal_col: str = "side",
    price_col: str = "price_close",
    predicted_mfe_col: str = "predicted_mfe",
    predicted_mae_col: str = "predicted_mae",
    pt_scale: float = 1.0,
    sl_scale: float = 1.0,
    horizon_bars: int | None = 30,
    horizon: str | pd.Timedelta | None = None,
    horizon_col: str | None = None,
    timestamp_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    bid_col: str | None = None,
    ask_col: str | None = None,
    fee_bps_per_side: float = 0.0,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build barriers from causal, out-of-sample predicted MFE and MAE.

    ``predicted_mfe_col`` is a predicted favorable excursion (nonnegative);
    ``predicted_mae_col`` is a predicted signed adverse excursion (normally
    nonpositive). The MAE magnitude sets the stop width. Predictions must be
    generated without using the event's future path.
    """
    if not np.isfinite(pt_scale) or pt_scale <= 0.0 or not np.isfinite(sl_scale) or sl_scale <= 0.0:
        raise ValueError("pt_scale and sl_scale must be finite and > 0")
    duration = pd.to_timedelta(horizon) if horizon is not None else None
    data = _ordered_frame(df, timestamp_col)
    timestamps = _timestamps(data, timestamp_col)
    if duration is not None and timestamps is None:
        raise ValueError("duration horizons require a DatetimeIndex or timestamp_col")
    _validate_inputs(
        data, signal_col, price_col, high_col, low_col, horizon_bars,
        duration, horizon_col, timestamp_col, bid_col, ask_col, fee_bps_per_side,
    )
    missing_predictions = [
        column for column in (predicted_mfe_col, predicted_mae_col)
        if column not in data.columns
    ]
    if missing_predictions:
        raise KeyError(f"Missing predicted excursion columns: {missing_predictions}")
    predicted_mfe = pd.to_numeric(data[predicted_mfe_col], errors="coerce").clip(lower=0.0).to_numpy(dtype=np.float64)
    predicted_mae = pd.to_numeric(data[predicted_mae_col], errors="coerce").abs().to_numpy(dtype=np.float64)
    return _entry_events(
        data,
        signal_col=signal_col,
        price_col=price_col,
        pt_widths=predicted_mfe * float(pt_scale),
        sl_widths=predicted_mae * float(sl_scale),
        horizon_bars=int(horizon_bars) if horizon_bars is not None else None,
        horizon=duration,
        horizon_col=horizon_col,
        timestamps=timestamps,
        high_col=high_col,
        low_col=low_col,
        drop_incomplete=drop_incomplete,
        bid_col=bid_col,
        ask_col=ask_col,
        fee_bps_per_side=fee_bps_per_side,
    )


def build_triple_barrier_sequence(
    df: pd.DataFrame,
    mode: str = "fixed",
    **kwargs,
) -> pd.DataFrame:
    """Select the fixed or volatility-scaled event outcome builder."""
    builders = {
        "fixed": build_triple_barrier_fixed,
        "dynamic": build_triple_barrier_dynamic,
        "predicted": build_triple_barrier_predicted,
    }
    try:
        builder = builders[mode]
    except KeyError:
        raise ValueError("mode must be 'fixed', 'dynamic', or 'predicted'") from None
    return builder(df, **kwargs)


# --------------------------------------------------------------------------
# Model-based evaluators: classifier + regressor on the model frame.
# Your new API (rebuilt): feature_cols auto-derived when None, NaN-safe.
# --------------------------------------------------------------------------
_EXCLUDE_PREFIXES = ("future_", "predicted_", "raw__", "__outcome_")
_EXCLUDE_COLUMNS = frozenset({
    "latent_id", "similarity", "trade_allowed", "state_mature", "model_ready",
    "timestamp",
})


def resolve_feature_cols(df: pd.DataFrame, price_col: str = "price_close") -> list[str]:
    """Auto-pick tradeable feature columns (your rule: everything useful).

    Keeps numeric columns, drops price_col, regime flags, future_/predicted_/
    targets and raw__/bookkeeping. Callers may still pass feature_cols
    explicitly — those are used verbatim (truth).
    """
    cols: list[str] = []
    for column in df.columns:
        if column == price_col or column in _EXCLUDE_COLUMNS:
            continue
        if column.startswith(_EXCLUDE_PREFIXES):
            continue
        if column in ("trade_id",):
            continue
        if "timestamp" in str(column).lower() or "_id" in str(column).lower():
            continue
        if pd.api.types.is_numeric_dtype(df[column].dtype):
            cols.append(column)
    return cols


def _causal_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build the model matrix with zero lookahead.

    - Uses ONLY the passed feature_cols (truth, no re-derivation).
    - Drops columns with NO finite value on train rows (e.g. gp_feat* for a
      state absent from train). Decision uses train rows ONLY.
    - Zero-fills partial NaN (state-masked gp_feat* outside its state reads
      as 0.0 — same rule as outcomes.py metalabel). Fill value is the
      constant 0.0, no mean/std computed on any data, so no leakage.
    Returns (X, kept_cols).
    """
    raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if raw.shape[1] == 0:
        raise ValueError("no feature columns to build the model matrix")
    train_finite = np.isfinite(raw[train_idx]).any(axis=0) if len(train_idx) else np.zeros(raw.shape[1], bool)
    keep = np.flatnonzero(train_finite)
    if len(keep) == 0:
        raise ValueError("not enough complete training rows")
    return (np.where(np.isfinite(raw[:, keep]), raw[:, keep], 0.0),
            [feature_cols[i] for i in keep.tolist()])


@dataclass
class BarrierData:
    df: pd.DataFrame
    X: np.ndarray
    kept_cols: list
    price: np.ndarray
    high: np.ndarray
    low: np.ndarray
    side: np.ndarray
    horizon: int
    path_ok: np.ndarray
    train: np.ndarray
    test: np.ndarray
    #: column of df that produced ``price`` (rr_unit needs it for U_t)
    price_col: str = "price_close"


class TripleBarrierResult(NamedTuple):
    stats: dict
    trades: pd.DataFrame
    model: object


def simulate_barriers(d: BarrierData, tp_width: np.ndarray, sl_width: np.ndarray) -> pd.DataFrame:
    """Run TP/SL/timeout for every row. Widths are fractions of entry price.

    Barriers sit at close[i]; path starts at i+1. Stop wins ties.
    outcome: +1 TP first, -1 SL first (or tie), 0 timeout.
    r: result in R (1R = stop distance). valid: full path + sane widths.
    """
    n, H = len(d.price), d.horizon
    p, s = d.price, d.side.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tp_level = p * (1.0 + s * tp_width)
        sl_level = p * (1.0 - s * sl_width)
    outcome = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    for k in range(1, H + 1):
        m = n - k
        if m <= 0:
            break
        hi, lo, side = d.high[k:], d.low[k:], s[:m]
        hit_sl = np.where(side > 0, lo <= sl_level[:m], hi >= sl_level[:m])
        hit_tp = np.where(side > 0, hi >= tp_level[:m], lo <= tp_level[:m])
        live = ~done[:m]
        sl_now = live & hit_sl
        tp_now = live & ~hit_sl & hit_tp
        outcome[:m][sl_now] = -1
        outcome[:m][tp_now] = 1
        done[:m] |= sl_now | tp_now
    timeout_price = np.full(n, np.nan)
    timeout_price[: n - H] = p[H:]
    timeout_ret = s * (timeout_price / p - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = tp_width / sl_width
        r = np.where(outcome == 1, rr, np.where(outcome == -1, -1.0, timeout_ret / sl_width))
    valid = (
        d.path_ok
        & np.isfinite(p) & (p > 0)
        & np.isfinite(tp_width) & (tp_width > 0)
        & np.isfinite(sl_width) & (sl_width > 0)
        & np.isfinite(tp_level) & (tp_level > 0)
        & np.isfinite(sl_level) & (sl_level > 0)
        & np.isfinite(r)
        & (d.side != 0)
    )
    return pd.DataFrame(
        {"tp_width": tp_width, "sl_width": sl_width, "rr": rr,
         "outcome": outcome, "r": r, "valid": valid}, index=d.df.index)


def forward_excursions(d: BarrierData) -> tuple[np.ndarray, np.ndarray]:
    """Side-adjusted MFE/MAE (positive fractions) over next H bars; NaN if path bad."""
    n, H = len(d.price), d.horizon
    p, s = d.price, d.side
    max_hi = np.full(n, -np.inf)
    min_lo = np.full(n, np.inf)
    for k in range(1, H + 1):
        m = n - k
        if m <= 0:
            break
        max_hi[:m] = np.maximum(max_hi[:m], d.high[k:])
        min_lo[:m] = np.minimum(min_lo[:m], d.low[k:])
    up = max_hi / p - 1.0
    down = 1.0 - min_lo / p
    mfe = np.where(s > 0, up, down)
    mae = np.where(s > 0, down, up)
    mfe[~d.path_ok] = np.nan
    mae[~d.path_ok] = np.nan
    return np.maximum(mfe, 0.0), np.maximum(mae, 0.0)


def _check_positive(name: str, value) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")


def _first_not_none(*values, default=None):
    """First argument that is not None (explicit kwargs > settings > default)."""
    for value in values:
        if value is not None:
            return value
    return default


def _summarize(d: BarrierData, sim: pd.DataFrame, trades: pd.DataFrame) -> dict:
    n = len(trades)
    stats = {
        "n_eligible": int(d.test.sum()),
        "n_scored": int((d.test & sim["valid"].to_numpy()).sum()),
        "n_trades": n,
    }
    stats["n_invalid_barriers"] = stats["n_eligible"] - stats["n_scored"]
    if n == 0:
        stats.update(win_rate=np.nan, avg_rr=np.nan, expectancy_r=np.nan,
                     tp_rate=np.nan, sl_rate=np.nan, timeout_rate=np.nan)
        return stats
    stats.update(
        win_rate=float((trades["outcome"] == 1).mean()),
        avg_rr=float(trades["rr"].mean()),
        expectancy_r=float(trades["r"].mean()),
        tp_rate=float((trades["outcome"] == 1).mean()),
        sl_rate=float((trades["outcome"] == -1).mean()),
        timeout_rate=float((trades["outcome"] == 0).mean()),
    )
    return stats


def _prepare_barrier(
    data,  # pd.DataFrame | BarrierData
    *,
    settings,
    which: str,
    timeout: int | None,
    timeout_fn,
    tp_size=None,
    rr=None,
    barrier_size=None,
    threshold=None,
    rr_window=None,
    vol_window=None,
    rr_col=None,
    vol_col=None,
    sqrt_horizon=None,
    feature_cols=None,
    side=None,
    price_col=None,
    high_col=None,
    low_col=None,
    train_fraction=None,
    default_tp_size: float = 1.0,
    default_rr_window: int = 14,
) -> tuple[BarrierData, dict]:
    """Shared config resolution for tb_classifier / tb_regressor.

    Precedence: explicit kwarg > evaluator settings section > triple_barrier
    section > default. ``timeout`` resolves ONCE through ``timeout_fn`` and
    drives the purge, ``path_ok`` and the simulation from a single number;
    for a prebuilt BarrierData it must match ``d.horizon``. Returns
    ``(d, resolved)`` with resolved keys tp_size, barrier_size, threshold,
    rr_window, rr_col, sqrt_horizon (the factor handed to rr_unit).
    """
    tb_cfg = load_settings(settings) if settings is not None else {}
    section = dict(tb_cfg.get(which, None) or {})
    tp_size = _first_not_none(
        tp_size, rr, section.get("tp_size"), section.get("rr"), default=default_tp_size)
    barrier_size = _first_not_none(barrier_size, section.get("barrier_size"), default=1.0)
    threshold = _first_not_none(threshold, section.get("threshold"), default=None)
    rr_window = _first_not_none(
        rr_window, vol_window, section.get("rr_window"), section.get("vol_window"),
        default=default_rr_window)
    rr_col = _first_not_none(
        rr_col, vol_col, section.get("rr_col"), section.get("vol_col"), default=None)
    sqrt_factor = _first_not_none(sqrt_horizon, section.get("sqrt_horizon"), default=None)
    _check_positive("tp_size", tp_size)
    _check_positive("barrier_size", barrier_size)

    raw_timeout = _first_not_none(
        timeout, section.get("timeout"), tb_cfg.get("timeout"), tb_cfg.get("horizon"),
        default=None)
    if isinstance(data, BarrierData):
        d = data
        build_args = (feature_cols, side, price_col, high_col, low_col, train_fraction)
        if any(v is not None for v in build_args):
            raise ValueError(
                "build arguments (feature_cols/side/price_col/high_col/low_col/"
                "train_fraction) apply only when data is a DataFrame, not a "
                "prebuilt BarrierData")
        resolved_timeout = d.horizon if raw_timeout is None else timeout_fn(raw_timeout)
        if resolved_timeout != d.horizon:
            raise ValueError(
                f"timeout {raw_timeout!r} resolves to {resolved_timeout} bars but the "
                f"BarrierData was built with horizon={d.horizon}; the purge, path mask "
                "and simulation would disagree -- rebuild it with that timeout")
    else:
        horizon = timeout_fn(raw_timeout if raw_timeout is not None else _DEFAULTS["horizon"])
        price_col = _first_not_none(
            price_col, tb_cfg.get("price_col"), default=_DEFAULTS["price_col"])
        side = _first_not_none(side, tb_cfg.get("side"), default=_DEFAULTS["side"])
        high_col = _first_not_none(
            high_col, tb_cfg.get("high_col"), default=_DEFAULTS["high_col"])
        low_col = _first_not_none(
            low_col, tb_cfg.get("low_col"), default=_DEFAULTS["low_col"])
        train_fraction = _first_not_none(
            train_fraction, tb_cfg.get("train_fraction"), default=_DEFAULTS["train_fraction"])
        if feature_cols is None:
            feature_cols = resolve_feature_cols(data, price_col=price_col)
        d = _build_barrier_data(
            data, feature_cols, side=side, horizon=horizon, price_col=price_col,
            high_col=high_col, low_col=low_col, train_fraction=train_fraction)
    if sqrt_factor is None:
        sqrt_factor = np.sqrt(d.horizon)  # per-bar unit widened to the holding path
    return d, {
        "tp_size": tp_size,
        "barrier_size": barrier_size,
        "threshold": threshold,
        "rr_window": rr_window,
        "rr_col": rr_col,
        "sqrt_horizon": sqrt_factor,
    }


def tb_classifier(
    data,  # pd.DataFrame | BarrierData
    model=None,
    *,
    settings=None,
    tp_size: float | None = None,
    rr: float | None = None,  # legacy settings key for tp_size
    barrier_size: float | None = None,
    timeout: int | None = None,
    threshold: float | None = None,
    rr_window: int | None = None,
    vol_window: int | None = None,  # legacy settings key for rr_window
    rr_col: str | None = None,
    vol_col: str | None = None,  # legacy settings key for rr_col
    sqrt_horizon: float | None = None,  # None -> np.sqrt(resolved timeout)
    rr_unit_fn=None,
    loss_fn=None,
    tp_fn=None,
    timeout_fn=None,
    model_fn=None,
    feature_cols: list[str] | None = None,
    side=None,
    price_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    train_fraction: float | None = None,
) -> TripleBarrierResult:
    """Classifier triple-barrier run: label = TP hit first; trade on P(win).

    Barrier math (all delegated to ``models.py``):

        U_t   = rr_unit(df, price_col, window=rr_window, vol_col=rr_col,
                        sqrt_horizon=sqrt(timeout))   # baseline risk unit
        sl_t  = loss_function(U_t, barrier_size)      # stop width
        tp_t  = tp_function(sl_w, tp_size)            # 1 -> 1R, 2 -> 2R

    Labels come from ``simulate_barriers`` on those widths; the model
    trains ONLY on purged train rows and trades test rows with
    ``P(win) >= threshold`` (default break-even ``1 / (1 + tp_size)``).

    ``timeout`` (bars) resolves once through ``models.timeout_function``
    and then drives everything consistently: the purged split distance,
    ``path_ok`` (rows near the end cannot enter), the simulation horizon,
    and therefore the labels.

    ``data`` may be a ``pd.DataFrame`` (this function builds its own
    ``BarrierData``; the ``feature_cols/side/price_col/...`` build
    arguments and ``settings`` then apply) or a prebuilt ``BarrierData``
    (build arguments must be omitted and ``timeout`` must match
    ``d.horizon``).

    Every helper default is resolved at call time: pass ``rr_unit_fn`` /
    ``loss_fn`` / ``tp_fn`` / ``timeout_fn`` / ``model_fn`` (or monkeypatch
    ``models.rr_unit``) to change the math per run. Settings keys accepted
    (also by ``run_triple_barrier``): tp_size|rr, barrier_size, timeout,
    threshold, rr_window|vol_window, rr_col|vol_col, sqrt_horizon.
    """
    d, cfg_res = _prepare_barrier(
        data, settings=settings,
        which=getattr(tb_classifier, "settings_key", "classifier"),
        timeout=timeout,
        timeout_fn=timeout_fn if timeout_fn is not None
        else _barrier_models.timeout_function,
        tp_size=tp_size, rr=rr, barrier_size=barrier_size, threshold=threshold,
        rr_window=rr_window, vol_window=vol_window, rr_col=rr_col, vol_col=vol_col,
        sqrt_horizon=sqrt_horizon, feature_cols=feature_cols, side=side,
        price_col=price_col, high_col=high_col, low_col=low_col,
        train_fraction=train_fraction,
    )
    tp_size = cfg_res["tp_size"]
    barrier_size = cfg_res["barrier_size"]
    threshold = cfg_res["threshold"]
    threshold = 1.0 / (1.0 + tp_size) if threshold is None else threshold
    rr_unit_fn = rr_unit_fn if rr_unit_fn is not None else _barrier_models.rr_unit
    loss_fn = loss_fn if loss_fn is not None else _barrier_models.loss_function
    tp_fn = tp_fn if tp_fn is not None else _barrier_models.tp_function
    model_fn = model_fn if model_fn is not None else _barrier_models.classifier_model

    unit = rr_unit_fn(
        d.df, price_col=d.price_col, window=cfg_res["rr_window"],
        vol_col=cfg_res["rr_col"], sqrt_horizon=cfg_res["sqrt_horizon"])
    sl_width = loss_fn(unit, barrier_size)
    tp_width = tp_fn(sl_width, tp_size)
    sim = simulate_barriers(d, tp_width, sl_width)
    valid = sim["valid"].to_numpy()
    label = (sim["outcome"] == 1).astype(int).to_numpy()
    fit_rows = d.train & valid
    if len(np.unique(label[fit_rows])) < 2:
        raise ValueError("training labels contain only one class; barriers never/always hit")
    model = model if model is not None else model_fn()
    model.fit(d.X[fit_rows], label[fit_rows])
    test_rows = d.test & valid
    prob = np.full(len(d.X), np.nan)
    if test_rows.any():
        prob[test_rows] = model.predict_proba(d.X[test_rows])[:, list(model.classes_).index(1)]
    take = test_rows & (prob >= threshold)
    trades = sim.loc[take].drop(columns="valid").assign(prob=prob[take])
    stats = _summarize(d, sim, trades)
    stats.update(threshold=float(threshold),
                 base_win_rate=float(label[test_rows].mean()) if test_rows.any() else np.nan)
    return TripleBarrierResult(stats, trades, model)


def evaluate_classifier(
    d: BarrierData, model=None, *, rr: float = 1.0, barrier_size: float = 1.0,
    vol_col: str | None = None, vol_window: int = 60,
    threshold: float | None = None,
    tp_size: float | None = None,
    rr_col: str | None = None, rr_window: int | None = None,
    timeout: int | None = None, timeout_fn=None, sqrt_horizon=None,
) -> TripleBarrierResult:
    """Compatibility wrapper over :func:`tb_classifier` (legacy keys).

    Keeps the historical defaults and settings keys -- ``rr`` (maps to
    tp_size; break-even threshold 1/(1+rr)) and ``vol_window`` (maps to
    rr_window, default 60) -- so existing settings.yaml sections reproduce
    the old math exactly: sl = vol * sqrt(H) * barrier_size, tp = sl * rr.
    New-style keys (tp_size/rr_window/rr_col/timeout) win when present.
    """
    return tb_classifier(
        d, model,
        tp_size=rr if tp_size is None else tp_size,
        barrier_size=barrier_size,
        rr_window=vol_window if rr_window is None else rr_window,
        rr_col=vol_col if rr_col is None else rr_col,
        threshold=threshold, timeout=timeout, timeout_fn=timeout_fn,
        sqrt_horizon=sqrt_horizon,
    )


def tb_regressor(
    data,  # pd.DataFrame | BarrierData
    model=None,
    *,
    settings=None,
    tp_size: float | None = None,
    rr: float | None = None,  # legacy settings key for tp_size
    barrier_size: float | None = None,
    timeout: int | None = None,
    threshold: float | None = None,  # accepted, ignored (no gating here)
    rr_window: int | None = None,
    vol_window: int | None = None,  # legacy settings key for rr_window
    rr_col: str | None = None,
    vol_col: str | None = None,  # legacy settings key for rr_col
    sqrt_horizon: float | None = None,  # None -> np.sqrt(resolved timeout)
    rr_unit_fn=None,
    loss_fn=None,
    tp_fn=None,
    timeout_fn=None,
    model_fn=None,
    feature_cols: list[str] | None = None,
    side=None,
    price_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    train_fraction: float | None = None,
) -> TripleBarrierResult:
    """Regressor triple-barrier run: predict MFE/MAE in RR-unit multiples.

    Targets and widths (the model learns MFE_R/MAE_R = excursion / U_t):

        U_t      = rr_unit(df, ...)                    # baseline risk unit
        sl_width = loss_function(MAE_R * U_t, barrier_size)
        tp_width = tp_function(loss_function(MFE_R * U_t, barrier_size),
                               tp_size)                # default tp_size = 1

    i.e. TP = MFE_R x U_t x barrier_size, SL = MAE_R x U_t x barrier_size;
    realized R is measured against the trade's actual stop (1R = sl).
    One clone of ``model`` per target, fitted on purged train rows only;
    rows with non-finite/non-positive predicted widths are unscored
    (counted n_invalid_barriers).

    Same timeout / DataFrame-vs-BarrierData / runtime-resolution contract
    as :func:`tb_classifier` (see there). The regressor has no gating:
    every valid test row is a trade. Settings keys: barrier_size, timeout,
    tp_size|rr, rr_window|vol_window, rr_col|vol_col, sqrt_horizon.
    """
    d, cfg_res = _prepare_barrier(
        data, settings=settings,
        which=getattr(tb_regressor, "settings_key", "regressor"),
        timeout=timeout,
        timeout_fn=timeout_fn if timeout_fn is not None
        else _barrier_models.timeout_function,
        tp_size=tp_size, rr=rr, barrier_size=barrier_size, threshold=threshold,
        rr_window=rr_window, vol_window=vol_window, rr_col=rr_col, vol_col=vol_col,
        sqrt_horizon=sqrt_horizon, feature_cols=feature_cols, side=side,
        price_col=price_col, high_col=high_col, low_col=low_col,
        train_fraction=train_fraction,
    )
    tp_size = cfg_res["tp_size"]
    barrier_size = cfg_res["barrier_size"]
    rr_unit_fn = rr_unit_fn if rr_unit_fn is not None else _barrier_models.rr_unit
    loss_fn = loss_fn if loss_fn is not None else _barrier_models.loss_function
    tp_fn = tp_fn if tp_fn is not None else _barrier_models.tp_function
    model_fn = model_fn if model_fn is not None else _barrier_models.regressor_model

    unit = rr_unit_fn(
        d.df, price_col=d.price_col, window=cfg_res["rr_window"],
        vol_col=cfg_res["rr_col"], sqrt_horizon=cfg_res["sqrt_horizon"])
    mfe, mae = forward_excursions(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        mfe_r = mfe / unit  # excursions in R (multiples of the unit)
        mae_r = mae / unit
    fit_rows = d.train & np.isfinite(mfe_r) & np.isfinite(mae_r)
    if fit_rows.sum() < d.X.shape[1] + 2:
        raise ValueError("not enough complete training rows")
    model = model if model is not None else model_fn()
    tp_model = clone(model).fit(d.X[fit_rows], mfe_r[fit_rows])
    sl_model = clone(model).fit(d.X[fit_rows], mae_r[fit_rows])
    tp_width = np.full(len(d.X), np.nan)
    sl_width = np.full(len(d.X), np.nan)
    rows = d.test
    if rows.any():
        tp_width[rows] = tp_fn(
            loss_fn(tp_model.predict(d.X[rows]) * unit[rows], barrier_size), tp_size)
        sl_width[rows] = loss_fn(sl_model.predict(d.X[rows]) * unit[rows], barrier_size)
    sim = simulate_barriers(d, tp_width, sl_width)
    take = d.test & sim["valid"].to_numpy()
    trades = sim.loc[take].drop(columns="valid")
    stats = _summarize(d, sim, trades)
    return TripleBarrierResult(stats, trades, {"tp_model": tp_model, "sl_model": sl_model})


def evaluate_regressor(
    d: BarrierData, model=None, *, rr: float = 1.0, barrier_size: float = 1.0,
    tp_size: float | None = None,
    rr_col: str | None = None, rr_window: int | None = None,
    vol_col: str | None = None, vol_window: int | None = None,
    timeout: int | None = None, timeout_fn=None, sqrt_horizon=None,
) -> TripleBarrierResult:
    """Compatibility wrapper over :func:`tb_regressor` (legacy keys).

    Historical behaviour trained MFE/MAE as *raw price fractions* (unit
    U_t = 1) and scaled the predicted MFE by ``rr``:
    sl = pred_mae * barrier_size, tp = pred_mfe * barrier_size * rr --
    keep it whenever only legacy keys arrive (the U_t = 1 shim is applied
    automatically). Passing a vol key (rr_window/rr_col/vol_window/vol_col)
    opts into the new RR-unit normalisation instead.
    """
    kwargs = dict(
        tp_size=rr if tp_size is None else tp_size,
        barrier_size=barrier_size,
        rr_window=rr_window, rr_col=rr_col,
        timeout=timeout, timeout_fn=timeout_fn, sqrt_horizon=sqrt_horizon,
    )
    if rr_window is None and rr_col is None and vol_window is None and vol_col is None:
        kwargs["rr_unit_fn"] = _barrier_models.raw_fraction_unit
    else:
        kwargs["rr_window"] = rr_window if rr_window is not None else vol_window
        kwargs["rr_col"] = rr_col if rr_col is not None else vol_col
    return tb_regressor(d, model, **kwargs)


tb_classifier.settings_key = "classifier"
tb_regressor.settings_key = "regressor"
evaluate_classifier.settings_key = "classifier"
evaluate_regressor.settings_key = "regressor"


_DEFAULTS = {"horizon": 30, "price_col": "price_close", "train_fraction": 0.7,
             "side": 1, "high_col": None, "low_col": None}


def load_settings(settings=None) -> dict:
    """Return the outcome.triple_barrier (or top-level triple_barrier) section."""
    if isinstance(settings, dict):
        cfg = settings
    else:
        path = Path(settings) if settings is not None else Path("settings.yaml")
        if not path.exists():
            if settings is None:
                return {}
            raise FileNotFoundError(path)
        import yaml
        cfg = yaml.safe_load(path.read_text()) or {}
    section = cfg.get("triple_barrier")
    if section is None:
        outcome = cfg.get("outcome") or {}
        inner = outcome.get("outcome")
        if isinstance(inner, dict) and ("triple_barrier" in inner or "forward_returns" in inner):
            outcome = {**outcome, **inner}
        section = outcome.get("triple_barrier")
    return dict(section or {})


def _build_barrier_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    side,
    horizon: int,
    price_col: str,
    high_col: str | None,
    low_col: str | None,
    train_fraction: float,
) -> BarrierData:
    """Validate the frame and build the purged BarrierData (shared).

    Verbatim construction extracted from run_triple_barrier: missing-column
    and split checks, the path mask (row i may enter only while i+horizon < n
    and the path is clean), the chronological purge (train rows end
    ``horizon`` bars before the split) and the causal feature matrix.
    """
    required = [*feature_cols, price_col, high_col, low_col,
                side if isinstance(side, str) else None]
    missing = [c for c in required if c is not None and c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted oldest-first (DatetimeIndex is not increasing)")
    n = len(df)
    price = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=float) if high_col else price
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=float) if low_col else price
    side_arr = (pd.to_numeric(df[side], errors="coerce").fillna(0).to_numpy()
                if isinstance(side, str) else np.full(n, side))
    if not np.isin(side_arr, (-1, 0, 1)).all():
        raise ValueError("side must contain only -1, 0 (no event), or +1")
    event = side_arr != 0
    idx = np.arange(n)
    bad = ~(np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0))
    bad_cum = np.concatenate([[0], np.cumsum(bad)])
    path_ok = (idx + horizon < n) & ((bad_cum[np.minimum(idx + horizon + 1, n)] - bad_cum[idx + 1]) == 0)
    split = int(n * train_fraction)
    train = (idx < split - horizon) & event
    X, kept = _causal_feature_matrix(df, list(feature_cols), np.flatnonzero(train))
    return BarrierData(
        df=df, X=X, kept_cols=kept, price=price, high=high, low=low,
        side=side_arr, horizon=horizon, path_ok=path_ok,
        train=train,
        test=(idx >= split) & event & path_ok,
        price_col=price_col,
    )


def run_triple_barrier(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    evaluator: Callable[..., TripleBarrierResult] = evaluate_classifier,
    model=None,
    *,
    settings=None,
    side=None,
    horizon: int | None = None,
    price_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    train_fraction: float | None = None,
    **evaluator_kwargs,
) -> TripleBarrierResult:
    """Run a triple-barrier evaluation. feature_cols=None auto-picks (everything useful).

    df: time-ordered frame, oldest first. Barriers at close[i], path from i+1,
    stop wins ties. Train rows purged (split - horizon); no train label sees test.
    feature_cols: explicit list = truth; None = resolve_feature_cols(df).
    Only the resolved cols reach the model; price/flags/future_/raw__ never do.

    If the evaluator section (or evaluator_kwargs) carries a ``timeout``
    (bars), it wins over the shared ``horizon``: the single resolved value
    is applied to the purge, the path mask and the simulation alike (it is
    resolved through the evaluator's ``timeout_fn`` when supplied, else
    models.timeout_function).
    """
    cfg = load_settings(settings)

    def pick(name, value):
        return value if value is not None else cfg.get(name, _DEFAULTS[name])

    side = pick("side", side)
    horizon = pick("horizon", horizon)
    price_col = pick("price_col", price_col)
    high_col = pick("high_col", high_col)
    low_col = pick("low_col", low_col)
    train_fraction = pick("train_fraction", train_fraction)
    section = dict(cfg.get(getattr(evaluator, "settings_key", ""), None) or {})
    params = {**section, **evaluator_kwargs}
    # One timeout, one horizon: resolve it BEFORE the BarrierData exists so
    # purge and path_ok can never drift from the simulation.
    if params.get("timeout") is not None:
        timeout_fn = params.get("timeout_fn")
        if timeout_fn is None:
            timeout_fn = _barrier_models.timeout_function  # resolved at call time
        horizon = timeout_fn(params["timeout"])
    if feature_cols is None:
        feature_cols = resolve_feature_cols(df, price_col=price_col)
    d = _build_barrier_data(
        df, feature_cols, side=side, horizon=horizon, price_col=price_col,
        high_col=high_col, low_col=low_col, train_fraction=train_fraction,
    )
    return evaluator(d, model, **params)


def build_model_barriers(
    model_frame: pd.DataFrame,
    raw: pd.DataFrame | None,
    cfg: dict,
    evaluator=evaluate_classifier,
    model=None,
    *,
    feature_cols: list[str] | None = None,
    timestamp_col: str | None = None,
    **overrides,
) -> TripleBarrierResult:
    """Bridge the S1 model frame to run_triple_barrier (causal).

    Attaches raw side/high/low via backward merge_asof (same as outcomes.py),
    auto-picks feature_cols when None, forwards cfg + overrides. feature_cols
    passed explicitly are truth; everything in them is used as a feature.
    """
    from .alignment import attach_raw_columns
    outcome = cfg.get("outcome") or {}
    # Tolerate a doubled nesting (outcome.outcome.*) seen in some working trees.
    inner = outcome.get("outcome")
    if isinstance(inner, dict) and ("triple_barrier" in inner or "forward_returns" in inner):
        outcome = {**outcome, **inner}
    tb_cfg = dict(outcome.get("triple_barrier", cfg.get("triple_barrier", {})))
    tb_cfg.update(overrides)
    side = tb_cfg.get("side", "htf_rsi_d2_signal")
    price_col = tb_cfg.get("price_col", "price_close")
    high_col, low_col = tb_cfg.get("high_col"), tb_cfg.get("low_col")
    need = {c for c in (price_col, high_col, low_col) if c is not None}
    if isinstance(side, str):
        need.add(side)
    if timestamp_col is not None:
        need.add(timestamp_col)
    frame = model_frame
    if raw is not None:
        missing_raw = [c for c in need if c not in frame.columns]
        if missing_raw:
            frame = attach_raw_columns(raw, model_frame, set(missing_raw), timestamp_col)
    if feature_cols is None:
        feature_cols = resolve_feature_cols(frame, price_col=price_col)
    return run_triple_barrier(
        frame, feature_cols, evaluator, model, settings={"triple_barrier": tb_cfg},
        side=side, price_col=price_col, high_col=high_col, low_col=low_col,
        horizon=tb_cfg.get("horizon"), train_fraction=tb_cfg.get("train_fraction"),
        **{k: v for k, v in tb_cfg.get(getattr(evaluator, "settings_key", ""), {}).items()},
    )
