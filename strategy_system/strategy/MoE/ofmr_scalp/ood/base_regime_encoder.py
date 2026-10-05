"""Swappable causal encoders for recurring market-regime discovery.

The encoder contract deliberately knows nothing about regime IDs or outcomes.
Inputs are raw/primitive observations, and outputs are one embedding per input
row, anchored to that row's timestamp.  Encoders must only use observations
available at or before the anchor row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def prepare_regime_input(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    price_column: str | None = None,
) -> pd.DataFrame:
    """Select numeric causal inputs, deriving log returns from price if needed.

    Pass ``feature_columns`` to make the schema explicit. Auto-selection drops
    timestamps, identifiers, and obvious forward/target columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("regime input is empty")
    if feature_columns is not None:
        columns = list(feature_columns)
        if not columns:
            raise ValueError("feature_columns must not be empty")
        missing = [column for column in columns if column not in df]
        if missing:
            raise KeyError(f"Missing regime input columns: {missing}")
        values = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = [
            column for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
            and not any(token in str(column).lower() for token in
                        ("timestamp", "_id", "trade_id", "future", "target", "label"))
        ]
        price = price_column or next(
            (name for name in ("mid_price", "price_close", "close", "price") if name in df), None
        )
        if price is not None and len(numeric) <= 1:
            series = pd.to_numeric(df[price], errors="coerce")
            values = pd.DataFrame({"log_return": np.log(series.where(series > 0)).diff()}, index=df.index)
        elif numeric:
            values = df.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
        else:
            raise ValueError("no numeric regime input columns found")
    values = values.replace([np.inf, -np.inf], np.nan)
    if values.columns.has_duplicates:
        raise ValueError("regime input feature names must be unique")
    return values.astype(np.float64)


def build_temporal_windows(values: pd.DataFrame | np.ndarray, window: int) -> np.ndarray:
    """Build row-aligned trailing windows; leading incomplete windows are NaN."""
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    matrix = values.to_numpy(dtype=np.float64) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    result = np.full((len(matrix), window, matrix.shape[1]), np.nan, dtype=np.float64)
    if len(matrix) >= window:
        raw = np.lib.stride_tricks.sliding_window_view(matrix, window, axis=0).transpose(0, 2, 1)
        valid = np.isfinite(raw).all(axis=(1, 2))
        result[window - 1:][valid] = raw[valid]
    return result


class BaseRegimeEncoder(ABC):
    """Interface for fit/transform temporal encoders."""

    schema_version = 1

    def __init__(self, feature_columns: Sequence[str] | None = None, timestamp_col: str | None = None):
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.timestamp_col = timestamp_col
        self.input_columns_: tuple[str, ...] | None = None
        self.embedding_columns_: tuple[str, ...] | None = None
        self.is_fitted_ = False

    @property
    def window_size(self) -> int | None:
        """Temporal input length, when this encoder uses fixed windows."""
        value = getattr(self, "window", None)
        return int(value) if value is not None else None

    def _validate_time(self, df: pd.DataFrame) -> None:
        name = self.timestamp_col or ("timestamp" if "timestamp" in df else None)
        times = None
        if name is not None:
            if name not in df and isinstance(df.index, pd.DatetimeIndex):
                times = pd.to_datetime(df.index, utc=True, errors="coerce")
            elif name not in df:
                raise KeyError(f"timestamp column {name!r} is missing")
            else:
                times = pd.to_datetime(df[name], utc=True, errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            times = pd.to_datetime(df.index, utc=True, errors="coerce")
        if times is not None and (times.isna().any() or not times.is_monotonic_increasing):
            raise ValueError("regime input timestamps must be valid and increasing")

    def prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_time(df)
        timestamp_name = self.timestamp_col or ("timestamp" if "timestamp" in df else None)
        source = df.drop(columns=[timestamp_name]) if timestamp_name is not None and timestamp_name in df else df
        values = prepare_regime_input(source, self.feature_columns)
        if self.input_columns_ is not None and tuple(values.columns) != self.input_columns_:
            raise ValueError("regime input schema changed; feature names and order must remain fixed")
        return values

    def fit(self, df: pd.DataFrame, y=None) -> "BaseRegimeEncoder":
        return self.fit_encoder(df, y=y)

    @abstractmethod
    def fit_encoder(self, df: pd.DataFrame, y=None) -> "BaseRegimeEncoder":
        """Fit on historical data only; never consume forward outcomes."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return causal embeddings aligned one-to-one with ``df``."""

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit_encoder(df, y=y)
        return self.transform(df)

    def save_encoder(self, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load_encoder(cls, path: str | Path) -> "BaseRegimeEncoder":
        with Path(path).open("rb") as stream:
            encoder = pickle.load(stream)
        if not isinstance(encoder, cls) or encoder.schema_version != cls.schema_version:
            raise ValueError("saved encoder type or schema version is incompatible")
        return encoder


class TemporalWindowEncoder(BaseRegimeEncoder):
    """Dependency-free temporal baseline using causal scaling and ordered bins.

    This baseline preserves coarse within-window order, dispersion, and slope.
    Learned TCN/TS2Vec/Transformer implementations can implement the same base
    contract and be injected into :class:`OODSystem` without changing it.
    """

    def __init__(
        self, window: int = 32, bins: int = 8, normalization_window: int = 500,
        normalization_min_periods: int = 50, clip_z: float = 8.0,
        slow_feature_window: int = 5000, slow_min_periods: int = 50,
        slow_level_weight: float = 1.0,
        feature_columns: Sequence[str] | None = None, timestamp_col: str | None = None,
    ):
        super().__init__(feature_columns, timestamp_col)
        if window < 2 or not 1 <= bins <= window:
            raise ValueError("require window >= 2 and 1 <= bins <= window")
        if normalization_window < 2 or not 2 <= normalization_min_periods <= normalization_window:
            raise ValueError("invalid causal normalization window/minimum")
        if not np.isfinite(clip_z) or clip_z <= 0:
            raise ValueError("clip_z must be finite and positive")
        if slow_feature_window < 2 or not 2 <= slow_min_periods <= slow_feature_window:
            raise ValueError("invalid slow feature window/minimum")
        if not np.isfinite(slow_level_weight) or slow_level_weight < 0:
            raise ValueError("slow_level_weight must be finite and nonnegative")
        self.window, self.bins = int(window), int(bins)
        self.normalization_window = int(normalization_window)
        self.normalization_min_periods = int(normalization_min_periods)
        self.clip_z = float(clip_z)
        self.slow_feature_window = int(slow_feature_window)
        self.slow_min_periods = int(slow_min_periods)
        self.slow_level_weight = float(slow_level_weight)

    def fit_encoder(self, df: pd.DataFrame, y=None) -> "TemporalWindowEncoder":
        if y is not None:
            raise ValueError("regime encoders do not train from forward outcomes")
        values = self.prepare_input(df)
        self.input_columns_ = tuple(values.columns)
        self.embedding_columns_ = tuple(
            [f"bin_{i}__{name}" for i in range(self.bins) for name in values.columns]
            + [f"dispersion__{name}" for name in values.columns]
            + [f"slope__{name}" for name in values.columns]
            + [f"slow_level__{name}" for name in values.columns]
        )
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("fit_encoder must be called before transform")
        values = self.prepare_input(df)
        history = values.shift(1)
        mean = history.rolling(self.normalization_window, min_periods=self.normalization_min_periods).mean()
        scale = history.rolling(self.normalization_window, min_periods=self.normalization_min_periods).std(ddof=0)
        delta = values - mean
        z = delta.div(scale.where(scale > 1e-12)).clip(-self.clip_z, self.clip_z)
        flat = scale.le(1e-12) & mean.notna()
        z = z.mask(flat & delta.abs().le(1e-12), 0.0)
        z = z.mask(flat & delta.gt(1e-12), self.clip_z).mask(flat & delta.lt(-1e-12), -self.clip_z)
        slow_levels = {}
        for column in values.columns:
            series = values[column]
            slow_levels[column] = (
                series.rolling(self.slow_feature_window, min_periods=self.slow_min_periods).std(ddof=0)
                if column in {"mid_return", "log_return"}
                else series.rolling(self.slow_feature_window, min_periods=self.slow_min_periods).mean()
            )
        slow = pd.DataFrame(slow_levels, index=values.index)
        slow_history = slow.shift(1)
        slow_mean = slow_history.expanding(min_periods=self.slow_min_periods).mean()
        slow_scale = slow_history.expanding(min_periods=self.slow_min_periods).std(ddof=0)
        slow_delta = slow - slow_mean
        slow_z = slow_delta.div(slow_scale.where(slow_scale > 1e-12))
        slow_flat = slow_scale.le(1e-12) & slow_mean.notna()
        slow_z = slow_z.mask(slow_flat & slow_delta.abs().le(1e-12), 0.0)
        slow_z = slow_z.mask(slow_flat & slow_delta.gt(1e-12), self.clip_z)
        slow_z = slow_z.mask(slow_flat & slow_delta.lt(-1e-12), -self.clip_z)
        slow_z = (self.slow_level_weight * slow_z).clip(-self.clip_z, self.clip_z)
        windows = build_temporal_windows(z, self.window)
        out = np.full((len(values), len(self.embedding_columns_)), np.nan)
        if len(values) >= self.window:
            bin_indices = np.array_split(np.arange(self.window), self.bins)
            for row in range(self.window - 1, len(values)):
                sequence = windows[row]
                if not np.isfinite(sequence).all():
                    continue
                means = [sequence[idx].mean(axis=0) for idx in bin_indices]
                dispersion = sequence.std(axis=0, ddof=0)
                centered = np.linspace(-1.0, 1.0, self.window)
                centered -= centered.mean()
                slope = centered @ sequence / max(float(centered @ centered), 1e-12)
                slow_row = slow_z.iloc[row].to_numpy(dtype=np.float64)
                if not np.isfinite(slow_row).all():
                    continue
                out[row] = np.concatenate([*means, dispersion, slope, slow_row])
        return pd.DataFrame(out, index=df.index, columns=self.embedding_columns_)


def encode_windows(encoder: BaseRegimeEncoder, df: pd.DataFrame) -> pd.DataFrame:
    """Convenience adapter preserving the common embedding contract."""
    return encoder.transform(df)


class _TorchTemporalEncoder(BaseRegimeEncoder):
    """Optional trainable sequence encoder; torch is imported only when fitted."""

    architecture = "tcn"

    def __init__(self, window: int = 32, embedding_dim: int = 16, epochs: int = 10,
                 batch_size: int = 256, learning_rate: float = 1e-3, seed: int = 7,
                 feature_columns: Sequence[str] | None = None, timestamp_col: str | None = None):
        super().__init__(feature_columns, timestamp_col)
        if window < 2 or embedding_dim < 2 or epochs < 1 or batch_size < 2:
            raise ValueError("invalid temporal encoder dimensions or training settings")
        self.window, self.embedding_dim, self.epochs = int(window), int(embedding_dim), int(epochs)
        self.batch_size, self.learning_rate, self.seed = int(batch_size), float(learning_rate), int(seed)
        self.normalization_window = max(500, self.window)
        self.normalization_min_periods = min(50, self.normalization_window)
        self._torch_model = None
        self._torch_features = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("_torch_model", None)
        state["_torch_model_state"] = (
            {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
            if model is not None else None
        )
        return state

    def __setstate__(self, state):
        model_state = state.pop("_torch_model_state", None)
        self.__dict__.update(state)
        self._torch_model = None
        if model_state is not None:
            import torch
            self._torch_model = self._build_model(self._torch_features)
            self._torch_model.load_state_dict({key: torch.as_tensor(value) for key, value in model_state.items()})
            self._torch_model.eval()

    def _scaled_windows(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        values = self.prepare_input(df)
        history = values.shift(1)
        mean = history.rolling(self.normalization_window, min_periods=self.normalization_min_periods).mean()
        scale = history.rolling(self.normalization_window, min_periods=self.normalization_min_periods).std(ddof=0)
        z = ((values - mean) / scale.where(scale > 1e-12)).clip(-8, 8)
        flat = scale.le(1e-12) & mean.notna()
        z = z.mask(flat & (values - mean).abs().le(1e-12), 0.0)
        return build_temporal_windows(z, self.window), values

    def _build_model(self, feature_count: int):
        import torch
        from torch import nn

        arch, emb = self.architecture, self.embedding_dim
        width = max(32, int(np.ceil((self.embedding_dim * 2) / 4) * 4))

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Linear(feature_count, width)
                if arch in {"tcn", "ts2vec"}:
                    self.sequence = nn.Sequential(
                        nn.Conv1d(width, width, 3, dilation=1), nn.GELU(),
                        nn.Conv1d(width, width, 3, dilation=2), nn.GELU(),
                        nn.Conv1d(width, width, 3, dilation=4), nn.GELU(),
                    )
                else:
                    layer = nn.TransformerEncoderLayer(width, 4, width * 2, batch_first=True, norm_first=True)
                    self.sequence = nn.TransformerEncoder(layer, num_layers=2)
                    self.position = nn.Parameter(torch.zeros(1, self_window, width))
                    nn.init.normal_(self.position, std=0.02)
                self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, emb))
                self.decode = nn.Linear(emb, self_window * feature_count)

            def forward(self, x):
                h = self.input(x)
                if arch in {"tcn", "ts2vec"}:
                    import torch.nn.functional as F
                    h = h.transpose(1, 2)
                    for layer in self.sequence:
                        if isinstance(layer, nn.Conv1d):
                            left = 2 * layer.dilation[0]
                            h = layer(F.pad(h, (left, 0)))
                        else:
                            h = layer(h)
                    h = h.transpose(1, 2)
                else:
                    h = self.sequence(h + self.position)
                latent = self.project(h.mean(dim=1))
                return latent, self.decode(latent).view(-1, self_window, feature_count)

        self_window = self.window
        return Network()

    def _loss(self, latent, reconstructed, target, second_view=None):
        import torch
        import torch.nn.functional as F
        if self.architecture != "ts2vec":
            return F.mse_loss(reconstructed, target)
        second, _ = self._torch_model(second_view)
        left, right = F.normalize(latent, dim=1), F.normalize(second, dim=1)
        logits = left @ right.T / 0.1
        labels = torch.arange(len(left), device=left.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def fit_encoder(self, df: pd.DataFrame, y=None):
        if y is not None:
            raise ValueError("regime encoders do not train from forward outcomes")
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        windows, values = self._scaled_windows(df)
        valid = np.isfinite(windows).all(axis=(1, 2))
        samples = windows[valid].astype(np.float32)
        if len(samples) < 2:
            raise ValueError("not enough complete temporal windows to fit encoder")
        self.input_columns_ = tuple(values.columns)
        self.embedding_columns_ = tuple(f"embedding_{i}" for i in range(self.embedding_dim))
        self._torch_features = samples.shape[2]
        self._torch_model = self._build_model(self._torch_features)
        optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=self.learning_rate)
        batches = DataLoader(TensorDataset(torch.from_numpy(samples)), batch_size=self.batch_size, shuffle=True)
        self._torch_model.train()
        for _ in range(self.epochs):
            for (batch,) in batches:
                optimizer.zero_grad()
                view = None
                if self.architecture == "ts2vec":
                    scale = 0.8 + 0.4 * torch.rand((len(batch), 1, 1))
                    view = batch * scale + torch.randn_like(batch) * 0.03
                    # Mask a short contiguous span as a second temporal view.
                    width = max(1, batch.shape[1] // 8)
                    for sample in range(len(view)):
                        start = int(torch.randint(0, max(batch.shape[1] - width + 1, 1), (1,)).item())
                        view[sample, start:start + width] = 0.0
                latent, reconstructed = self._torch_model(batch)
                loss = self._loss(latent, reconstructed, batch, view)
                loss.backward()
                optimizer.step()
        self._torch_model.eval()
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_ or self._torch_model is None:
            raise RuntimeError("fit_encoder must be called before transform")
        import torch
        windows, _ = self._scaled_windows(df)
        output = np.full((len(df), self.embedding_dim), np.nan, dtype=np.float64)
        valid_rows = np.flatnonzero(np.isfinite(windows).all(axis=(1, 2)))
        self._torch_model.eval()
        with torch.no_grad():
            for start in range(0, len(valid_rows), self.batch_size):
                rows = valid_rows[start:start + self.batch_size]
                latent, _ = self._torch_model(torch.as_tensor(windows[rows], dtype=torch.float32))
                output[rows] = latent.cpu().numpy()
        return pd.DataFrame(output, index=df.index, columns=self.embedding_columns_)


class TCNRegimeEncoder(_TorchTemporalEncoder):
    """Temporal convolutional reconstruction encoder."""
    architecture = "tcn"


class TS2VecRegimeEncoder(_TorchTemporalEncoder):
    """Temporal contrastive sequence encoder inspired by TS2Vec objectives."""
    architecture = "ts2vec"


class TransformerRegimeEncoder(_TorchTemporalEncoder):
    """Transformer sequence reconstruction encoder."""
    architecture = "transformer"
