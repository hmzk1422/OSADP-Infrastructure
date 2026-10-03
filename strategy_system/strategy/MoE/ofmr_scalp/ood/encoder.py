"""Causal primitive extraction and rolling temporal embedding."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _first_existing(columns, names):
    return next((name for name in names if name in columns), None)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float64)


def build_primitive_observations(
    df: pd.DataFrame,
    primitive_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a stable raw-observation set, independent of GP features.

    Auto mode derives only from present raw fields: mid-price return, spread,
    top-of-book queue imbalance, available depth imbalances, signed trade flow,
    activity, and numeric liquidation primitives. Explicit ``primitive_columns``
    uses exactly those input columns, converted to numeric. Derived features
    here are fixed measurement primitives; they are not GP-evolved features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("raw market frame is empty")

    if primitive_columns is not None:
        columns = list(primitive_columns)
        if not columns:
            raise ValueError("primitive_columns must not be empty")
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise KeyError(f"Missing OOD primitive columns: {missing}")
        primitive = pd.DataFrame({column: _numeric(df, column) for column in columns}, index=df.index)
    else:
        primitive_data: dict[str, pd.Series] = {}
        columns = df.columns
        bid_price_col = _first_existing(columns, ("best_bid_price", "best_bid_price_close", "bid_price"))
        ask_price_col = _first_existing(columns, ("best_ask_price", "best_ask_price_close", "ask_price"))
        price_col = _first_existing(columns, ("mid_price", "price_close", "price", "close"))

        if bid_price_col is not None and ask_price_col is not None:
            bid_price = _numeric(df, bid_price_col)
            ask_price = _numeric(df, ask_price_col)
            mid = (bid_price + ask_price) / 2.0
            primitive_data["mid_return"] = np.log(mid.where(mid > 0.0)).diff()
            primitive_data["spread_bps"] = (ask_price - bid_price) / mid.where(mid > 0.0) * 10_000.0
        elif price_col is not None:
            price = _numeric(df, price_col)
            primitive_data["mid_return"] = np.log(price.where(price > 0.0)).diff()

        bid_qty_col = _first_existing(columns, ("best_bid_qty", "bid_qty_1", "bid_size"))
        ask_qty_col = _first_existing(columns, ("best_ask_qty", "ask_qty_1", "ask_size"))
        if bid_qty_col is not None and ask_qty_col is not None:
            bid_qty = _numeric(df, bid_qty_col).clip(lower=0.0)
            ask_qty = _numeric(df, ask_qty_col).clip(lower=0.0)
            total = bid_qty + ask_qty
            primitive_data["queue_imbalance"] = (bid_qty - ask_qty) / total.replace(0.0, np.nan)
            primitive_data["top_depth_log"] = np.log1p(total)

        # Aggregate matching bid/ask depth levels without assuming every feed
        # has the same maximum depth.
        levels = set()
        for column in columns:
            if column.startswith("bid_qty_") and column[len("bid_qty_"):].isdigit():
                levels.add(column[len("bid_qty_"):])
            elif column.startswith("ask_qty_") and column[len("ask_qty_"):].isdigit():
                levels.add(column[len("ask_qty_"):])
        for level in sorted(levels, key=lambda value: int(value)):
            bid_name, ask_name = f"bid_qty_{level}", f"ask_qty_{level}"
            if bid_name in columns and ask_name in columns:
                # Level 1 is already represented by queue_imbalance/top_depth_log.
                if level == "1" and bid_qty_col is not None and ask_qty_col is not None:
                    continue
                bid_depth = _numeric(df, bid_name).clip(lower=0.0)
                ask_depth = _numeric(df, ask_name).clip(lower=0.0)
                total_depth = bid_depth + ask_depth
                primitive_data[f"depth_imbalance_{level}"] = (
                    (bid_depth - ask_depth) / total_depth.replace(0.0, np.nan)
                )
                primitive_data[f"depth_log_{level}"] = np.log1p(total_depth)

        quantity_col = _first_existing(columns, ("quantity", "trade_qty", "volume"))
        maker_col = _first_existing(columns, ("is_buyer_maker", "buyer_is_maker"))
        if quantity_col is not None:
            quantity = _numeric(df, quantity_col).clip(lower=0.0)
            primitive_data["trade_activity_log"] = np.log1p(quantity)
            if maker_col is not None:
                maker = df[maker_col].fillna(False).astype(bool)
                signed_qty = quantity.where(~maker, -quantity)
                primitive_data["signed_trade_flow"] = signed_qty
                primitive_data["trade_direction"] = np.sign(signed_qty)

        # Keep user-provided liquidation primitives as raw measurements. Their
        # causal rolling normalization is applied by the temporal encoder.
        for column in columns:
            if "liquidat" in str(column).lower() and pd.api.types.is_numeric_dtype(df[column]):
                primitive_data[f"raw_{column}"] = _numeric(df, column)

        if not primitive_data:
            numeric_columns = [
                column for column in columns
                if pd.api.types.is_numeric_dtype(df[column])
                and not any(token in str(column).lower() for token in ("timestamp", "_id", "trade_id"))
            ]
            if not numeric_columns:
                raise ValueError("no usable raw numeric market primitives were found")
            primitive_data = {f"raw_{column}": _numeric(df, column) for column in numeric_columns}

        primitive = pd.DataFrame(primitive_data, index=df.index)

    primitive = primitive.replace([np.inf, -np.inf], np.nan)
    primitive = primitive.loc[:, ~primitive.columns.duplicated()]
    if primitive.empty or primitive.shape[1] == 0:
        raise ValueError("no OOD primitive observations could be constructed")
    return primitive


class TemporalPrimitiveEncoder:
    """Encode causal rolling sequences of fixed raw primitives.

    Current primitives are standardized against preceding observations only.
    Each embedding contains chronological sub-window means plus full-window
    dispersion and linear slope, so order and temporal shape affect similarity.
    """

    def __init__(
        self,
        sequence_window: int = 32,
        embedding_bins: int = 8,
        normalization_window: int = 500,
        normalization_min_periods: int = 50,
        clip_z: float = 8.0,
        slow_feature_window: int = 5000,
        slow_min_periods: int = 50,
        slow_level_weight: float = 1.0,
        primitive_columns: Sequence[str] | None = None,
    ):
        if not isinstance(sequence_window, int) or sequence_window < 2:
            raise ValueError("sequence_window must be an integer >= 2")
        if not isinstance(embedding_bins, int) or not 1 <= embedding_bins <= sequence_window:
            raise ValueError("embedding_bins must be in [1, sequence_window]")
        if not isinstance(normalization_window, int) or normalization_window < 2:
            raise ValueError("normalization_window must be an integer >= 2")
        if not isinstance(normalization_min_periods, int) or not 2 <= normalization_min_periods <= normalization_window:
            raise ValueError("normalization_min_periods must be in [2, normalization_window]")
        if not np.isfinite(clip_z) or clip_z <= 0.0:
            raise ValueError("clip_z must be finite and > 0")
        if not isinstance(slow_feature_window, int) or slow_feature_window < 2:
            raise ValueError("slow_feature_window must be an integer >= 2")
        if not isinstance(slow_min_periods, int) or not 2 <= slow_min_periods <= slow_feature_window:
            raise ValueError("slow_min_periods must be in [2, slow_feature_window]")
        if not np.isfinite(slow_level_weight) or slow_level_weight < 0.0:
            raise ValueError("slow_level_weight must be finite and >= 0")
        self.sequence_window = sequence_window
        self.embedding_bins = embedding_bins
        self.normalization_window = normalization_window
        self.normalization_min_periods = normalization_min_periods
        self.clip_z = float(clip_z)
        self.slow_feature_window = slow_feature_window
        self.slow_min_periods = slow_min_periods
        self.slow_level_weight = float(slow_level_weight)
        self.primitive_columns = tuple(primitive_columns) if primitive_columns is not None else None
        self.primitive_names: tuple[str, ...] = ()
        self.embedding_columns: tuple[str, ...] = ()

    def encode(self, df: pd.DataFrame, timestamp_col: str | None = None) -> pd.DataFrame:
        """Return one embedding row per input row; warmup rows are NaN."""
        timestamp_name = timestamp_col or ("timestamp" if "timestamp" in df.columns else None)
        if timestamp_name is not None:
            if timestamp_name in df.columns:
                times = pd.to_datetime(df[timestamp_name], utc=True, errors="coerce")
            elif isinstance(df.index, pd.DatetimeIndex):
                times = pd.to_datetime(df.index, utc=True, errors="coerce")
            else:
                raise KeyError(f"timestamp_col {timestamp_name!r} is missing and df has no DatetimeIndex")
            if times.isna().any() or not times.is_monotonic_increasing:
                raise ValueError("OOD input timestamps must be valid and monotonically increasing")
        elif isinstance(df.index, pd.DatetimeIndex):
            if df.index.isna().any() or not df.index.is_monotonic_increasing:
                raise ValueError("OOD DatetimeIndex must be valid and monotonically increasing")

        primitives = build_primitive_observations(df, self.primitive_columns)
        self.primitive_names = tuple(primitives.columns)

        history = primitives.shift(1)
        mean = history.rolling(
            self.normalization_window,
            min_periods=self.normalization_min_periods,
        ).mean()
        scale = history.rolling(
            self.normalization_window,
            min_periods=self.normalization_min_periods,
        ).std(ddof=0)
        delta = primitives - mean
        z = delta.div(scale.where(scale > 1e-12))
        flat = scale.le(1e-12) & mean.notna()
        z = z.mask(flat & delta.abs().le(1e-12), 0.0)
        z = z.mask(flat & delta.gt(1e-12), self.clip_z)
        z = z.mask(flat & delta.lt(-1e-12), -self.clip_z)
        z = z.clip(-self.clip_z, self.clip_z)
        matrix = z.to_numpy(dtype=np.float64)

        # Slow level channels retain persistent volatility/liquidity conditions
        # that a rolling z-score alone would eventually normalize away. Means
        # summarize levels; return dispersion summarizes realized volatility.
        slow_values = {}
        for column in primitives.columns:
            series = primitives[column]
            if column == "mid_return":
                level = series.rolling(
                    self.slow_feature_window,
                    min_periods=self.slow_min_periods,
                ).std(ddof=0)
            else:
                level = series.rolling(
                    self.slow_feature_window,
                    min_periods=self.slow_min_periods,
                ).mean()
            slow_values[column] = level
        slow = pd.DataFrame(slow_values, index=primitives.index)
        slow_history = slow.shift(1)
        slow_mean = slow_history.expanding(min_periods=self.slow_min_periods).mean()
        slow_scale = slow_history.expanding(min_periods=self.slow_min_periods).std(ddof=0)
        slow_delta = slow - slow_mean
        slow_z = slow_delta.div(slow_scale.where(slow_scale > 1e-12))
        slow_flat = slow_scale.le(1e-12) & slow_mean.notna()
        slow_z = slow_z.mask(slow_flat & slow_delta.abs().le(1e-12), 0.0)
        slow_z = slow_z.mask(slow_flat & slow_delta.gt(1e-12), self.clip_z)
        slow_z = slow_z.mask(slow_flat & slow_delta.lt(-1e-12), -self.clip_z)
        slow_matrix = slow_z.clip(-self.clip_z, self.clip_z).to_numpy(dtype=np.float64)

        n_rows, n_features = matrix.shape
        n_fast_components = n_features * (self.embedding_bins + 2)
        n_components = n_fast_components + n_features
        embeddings = np.full((n_rows, n_components), np.nan, dtype=np.float64)
        time_axis = np.linspace(-1.0, 1.0, self.sequence_window, dtype=np.float64)
        centered_time = time_axis - time_axis.mean()
        time_denom = float(np.dot(centered_time, centered_time))
        bin_indices = np.array_split(np.arange(self.sequence_window), self.embedding_bins)

        if n_rows >= self.sequence_window:
            windows = np.lib.stride_tricks.sliding_window_view(
                matrix,
                self.sequence_window,
                axis=0,
            ).transpose(0, 2, 1)
            # Count complete rows with a cumulative sum; avoid allocating an
            # N x window x features boolean tensor for large tick histories.
            complete_rows = np.isfinite(matrix).all(axis=1).astype(np.int64)
            complete_prefix = np.concatenate([[0], np.cumsum(complete_rows)])
            valid = (complete_prefix[self.sequence_window:] - complete_prefix[:-self.sequence_window]) == self.sequence_window
            valid &= np.isfinite(slow_matrix[self.sequence_window - 1:]).all(axis=1)
            # Build each complete embedding in blocks, retaining time order.
            for start in range(0, len(windows), 8192):
                stop = min(start + 8192, len(windows))
                batch = windows[start:stop, :, :]
                target = np.arange(start, stop) + self.sequence_window - 1
                target_valid = valid[start:stop]
                if not target_valid.any():
                    continue
                selected = batch[target_valid]
                binned_means = np.stack(
                    [selected[:, indices, :].mean(axis=1) for indices in bin_indices],
                    axis=1,
                )
                dispersion = selected.std(axis=1, ddof=0)
                slope = np.einsum("t,ntf->nf", centered_time, selected) / time_denom
                fast = np.concatenate([binned_means.reshape(len(selected), -1), dispersion, slope], axis=1)
                embeddings[target[target_valid]] = np.concatenate(
                    [fast, self.slow_level_weight * slow_matrix[target[target_valid]]], axis=1,
                )

        names = (
            tuple(f"fast_embedding_{index}" for index in range(n_fast_components))
            + tuple(f"slow_level_{column}" for column in primitives.columns)
        )
        self.embedding_columns = names
        return pd.DataFrame(embeddings, index=df.index, columns=names)
