"""Fixed, volatility-scaled, and predicted triple-barrier event outcomes.

These builders consume a causal side/entry column and raw price observations.
They do not depend on GP-generated features, OOD embeddings, or a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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

    for i, raw_side in enumerate(sides_raw):
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
        entry_fill = asks[i] if side > 0 else bids[i]
        if not np.isfinite(entry_fill) or entry_fill <= 0.0:
            continue
        profit_level = entry_fill * (1.0 + side * pt_width)
        stop_level = entry_fill * (1.0 - side * sl_width)
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
        exit_fill = bids[exit_idx] if side > 0 else asks[exit_idx]
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
