"""
base_regime_encoder.py
===================================================================================

Swappable causal encoders for recurring market-regime discovery.

The encoder contract deliberately knows nothing about regime IDs or outcomes.
Inputs are raw/primitive observations, and outputs are one embedding per input
row, anchored to that row's timestamp.  Encoders must only use observations
available at or before the anchor row.

Candidates: temporal_window (dependency-free baseline), vq-vae, ts2vec, tcn,
transformer (PyTorch), pca (scikit-learn).  Pick one with ``make_encoder``.
"""

from __future__ import annotations

from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from ._jit import NUMBA_AVAILABLE, temporal_window_kernel


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
    """Build row-aligned trailing windows; leading incomplete windows are NaN.

    Materializes the full ``(rows, window, features)`` float64 cube, so peak
    memory is ``rows * window * features * 8`` bytes (12+ GiB at ~1.1M rows of
    45 features with window 32), plus a transient copy while filling it. Only
    the unused torch encoder path still calls this; ``TemporalWindowEncoder``
    reads trailing windows as views instead of paying that cost.
    """
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
        # Memory fix: never materialize the (rows, window, features) cube. At
        # ~1.1M rows x window 32 x 45 features that is a 12.1 GiB allocation
        # (build_temporal_windows) plus an equally large fancy-index copy --
        # the exact ArrayMemoryError seen in production. One float64 matrix
        # plus per-row trailing-window *views* gives identical values: a row
        # is skipped iff its window is not all-finite, which is exactly what
        # the materialized path left as an all-NaN row.
        matrix = z.to_numpy(dtype=np.float64)
        out = np.full((len(values), len(self.embedding_columns_)), np.nan)
        if len(values) >= self.window:
            bin_indices = np.array_split(np.arange(self.window), self.bins)
            centered = np.linspace(-1.0, 1.0, self.window)
            centered -= centered.mean()
            centered_norm = max(float(centered @ centered), 1e-12)
            slow_matrix = slow_z.to_numpy(dtype=np.float64)
            # A row is encoded iff its whole trailing window AND its slow row
            # are finite -- the same two ``continue`` conditions as before,
            # evaluated for every row at once in O(rows).
            valid = complete_window_mask(matrix, self.window) & np.isfinite(slow_matrix).all(axis=1)
            rows = np.flatnonzero(valid)
            if NUMBA_AVAILABLE:
                # NumPy's summation order depends on memory layout (see the
                # kernel docstring); read it off the original array first.
                single = matrix.shape[1] == 1
                temporal_window_kernel(
                    np.ascontiguousarray(matrix), np.ascontiguousarray(slow_matrix), rows,
                    np.array([idx[0] for idx in bin_indices], dtype=np.int64),
                    np.array([idx[-1] + 1 for idx in bin_indices], dtype=np.int64),
                    centered, centered_norm, self.window,
                    single, bool(matrix.flags.f_contiguous or single), out)
            else:  # original reference loop (no numba installed)
                for row in rows:
                    sequence = matrix[row - self.window + 1: row + 1]
                    means = [sequence[idx].mean(axis=0) for idx in bin_indices]
                    out[row] = np.concatenate([*means, sequence.std(axis=0, ddof=0),
                                               centered @ sequence / centered_norm, slow_matrix[row]])
        return pd.DataFrame(out, index=df.index, columns=self.embedding_columns_)


def encode_windows(encoder: BaseRegimeEncoder, df: pd.DataFrame) -> pd.DataFrame:
    """Convenience adapter preserving the common embedding contract."""
    return encoder.transform(df)


# ---------------------------------------------------------------------------
# Shared causal plumbing for learned encoders (no (rows, window, features) cube)
# ---------------------------------------------------------------------------

def causal_zscore_matrix(values: pd.DataFrame, window: int = 500, min_periods: int = 50,
                         clip_z: float = 8.0) -> np.ndarray:
    """Rolling z-score using only *strictly past* rows (shifted by one).

    Same rule as ``TemporalWindowEncoder``: constant history maps to 0 / +-clip.
    Rows before ``min_periods`` of history are NaN.
    """
    history = values.shift(1)
    mean = history.rolling(window, min_periods=min_periods).mean()
    scale = history.rolling(window, min_periods=min_periods).std(ddof=0)
    delta = values - mean
    z = delta.div(scale.where(scale > 1e-12)).clip(-clip_z, clip_z)
    flat = scale.le(1e-12) & mean.notna()
    z = z.mask(flat & delta.abs().le(1e-12), 0.0)
    z = z.mask(flat & delta.gt(1e-12), clip_z).mask(flat & delta.lt(-1e-12), -clip_z)
    return z.to_numpy(dtype=np.float64)


def complete_window_mask(matrix: np.ndarray, window: int) -> np.ndarray:
    """True at row r iff rows r-window+1..r are all finite (O(rows), no cube)."""
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(matrix).all(axis=1))])
    mask = np.zeros(len(matrix), dtype=bool)
    if len(matrix) >= window:
        mask[window - 1:] = (bad[window:] - bad[:-window]) == 0
    return mask


def gather_trailing_windows(matrix: np.ndarray, rows: np.ndarray, window: int) -> np.ndarray:
    """Gather ``(len(rows), window, features)`` float32 windows for the given end rows."""
    index = np.asarray(rows)[:, None] + np.arange(-window + 1, 1)[None, :]
    return matrix[index].astype(np.float32)


class _ScaledWindowEncoder(BaseRegimeEncoder):
    """Common base: causal scaling -> trailing windows -> subclass-specific model."""

    def __init__(self, window: int, embedding_dim: int, max_train_windows: int, seed: int,
                 normalization_window: int, normalization_min_periods: int, clip_z: float,
                 feature_columns, timestamp_col):
        super().__init__(feature_columns, timestamp_col)
        if window < 2 or embedding_dim < 1 or max_train_windows < 2:
            raise ValueError("invalid window / embedding_dim / max_train_windows")
        self.window, self.embedding_dim = int(window), int(embedding_dim)
        self.max_train_windows, self.seed = int(max_train_windows), int(seed)
        self.normalization_window = int(max(normalization_window, window))
        self.normalization_min_periods = int(min(normalization_min_periods, self.normalization_window))
        self.clip_z = float(clip_z)

    def _scaled(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        values = self.prepare_input(df)
        matrix = causal_zscore_matrix(values, self.normalization_window,
                                      self.normalization_min_periods, self.clip_z)
        return matrix, np.flatnonzero(complete_window_mask(matrix, self.window)), values

    def _training_rows(self, valid: np.ndarray) -> np.ndarray:
        if len(valid) < 2:
            raise ValueError("not enough complete temporal windows to fit encoder")
        if len(valid) > self.max_train_windows:  # seeded, evenly random subsample
            rng = np.random.default_rng(self.seed)
            valid = np.sort(rng.choice(valid, self.max_train_windows, replace=False))
        return valid


class TorchRegimeEncoder(_ScaledWindowEncoder):
    """Wrap ANY torch network that exposes ``encode(x)`` and ``loss(x)``.

    ``model`` is a name understood by ``encoder_networks.build_network`` ("vq-vae",
    "ts2vec", "tcn", "transformer") or your own callable
    ``factory(n_features, window, embedding_dim) -> nn.Module`` (must be importable
    at module level if you want to pickle the encoder).

    Quant hygiene baked in: unsupervised only; causal scaling; train/validation split
    is chronological with a ``window``-row embargo; seeded; weights saved as arrays.
    """

    def __init__(self, model="vq-vae", window: int = 32, embedding_dim: int = 16,
                 epochs: int = 10, batch_size: int = 256, learning_rate: float = 1e-3,
                 val_fraction: float = 0.1, max_train_windows: int = 500_000, seed: int = 7,
                 device: str = "cpu", model_params: dict | None = None,
                 normalization_window: int = 500, normalization_min_periods: int = 50,
                 clip_z: float = 8.0, feature_columns: Sequence[str] | None = None,
                 timestamp_col: str | None = None):
        super().__init__(window, embedding_dim, max_train_windows, seed, normalization_window,
                         normalization_min_periods, clip_z, feature_columns, timestamp_col)
        if epochs < 1 or batch_size < 2 or not 0.0 <= val_fraction < 0.5:
            raise ValueError("invalid epochs / batch_size / val_fraction")
        self.model, self.model_params = model, dict(model_params or {})
        self.epochs, self.batch_size = int(epochs), int(batch_size)
        self.learning_rate, self.val_fraction, self.device = float(learning_rate), float(val_fraction), device
        self.history_: list[dict] = []
        self.net_ = None
        self._n_features = 0
        self._embedding_width = embedding_dim

    # -- construction / persistence ------------------------------------------
    def _build(self, n_features: int):
        if callable(self.model):
            return self.model(n_features, self.window, self.embedding_dim, **self.model_params)
        from .encoder_networks import build_network
        return build_network(self.model, n_features, self.window, self.embedding_dim, **dict(self.model_params))

    def _probe_width(self, matrix) -> int:
        """Run one dummy window through ``encode`` to learn the true embedding width.

        Pooling variants like VQ-VAE ``histogram`` (num_codes) or ``mean_std``
        (2 * embedding_dim) change the output dimensionality, so we cannot assume it
        equals ``embedding_dim``.  Default ``mean`` pooling keeps it at ``embedding_dim``.
        """
        import torch
        device = self._device()
        self.net_.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.window, self._n_features, device=device)
            return int(self.net_.encode(dummy).shape[-1])

    def _device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        net = state.pop("net_", None)
        state["_net_weights"] = ({k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
                                 if net is not None else None)
        return state

    def __setstate__(self, state):
        weights = state.pop("_net_weights", None)
        self.__dict__.update(state)
        self.net_ = None
        if "_embedding_width" not in self.__dict__:      # back-compat for older pickles
            self._embedding_width = self.embedding_dim
        if weights is not None:
            import torch
            self.net_ = self._build(self._n_features)
            self.net_.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
            self.net_.to(self._device()).eval()

    # -- fit / transform ---------------------------------------------------
    def fit_encoder(self, df: pd.DataFrame, y=None) -> "TorchRegimeEncoder":
        if y is not None:
            raise ValueError("regime encoders do not train from forward outcomes")
        import torch
        torch.manual_seed(self.seed)
        matrix, valid, values = self._scaled(df)
        rows = self._training_rows(valid)
        split = int(len(rows) * (1.0 - self.val_fraction))
        train_rows = rows[:split] if self.val_fraction > 0 else rows
        val_rows = rows[rows >= (rows[split - 1] + self.window)] if self.val_fraction > 0 and split > 0 else rows[:0]
        self.input_columns_ = tuple(values.columns)
        self._n_features = matrix.shape[1]
        device = self._device()
        self.net_ = self._build(self._n_features).to(device)
        self._embedding_width = self._probe_width(matrix)   # allows pooling that changes width (histogram/mean_std)
        self.embedding_columns_ = tuple(f"embedding_{i}" for i in range(self._embedding_width))
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        generator = np.random.default_rng(self.seed)
        self.history_ = []
        for epoch in tqdm(range(self.epochs), desc=f"Training {self.model}"):
            self.net_.train()
            # reset dataset-level codebook diagnostics so epoch_stats reflect THIS epoch
            if hasattr(self.net_, "reset_epoch_stats"):
                self.net_.reset_epoch_stats()
            _q = getattr(self.net_, "quantizer", None)
            resets_before = int(_q.dead_code_resets.item()) if _q is not None else 0
            order = generator.permutation(train_rows)
            losses = []
            for start in range(0, len(order), self.batch_size):
                batch_rows = order[start:start + self.batch_size]
                if len(batch_rows) < 2:
                    continue
                batch = torch.from_numpy(gather_trailing_windows(matrix, batch_rows, self.window)).to(device)
                optimizer.zero_grad()
                loss = self.net_.loss(batch)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            entry = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else np.nan}
            entry.update(getattr(self.net_, "last_stats", {}) or {})
            if len(val_rows) >= 2:
                self.net_.eval()
                with torch.no_grad():
                    val = [float(self.net_.loss(torch.from_numpy(
                        gather_trailing_windows(matrix, val_rows[s:s + self.batch_size], self.window)).to(device)))
                        for s in range(0, len(val_rows), self.batch_size)]
                entry["val_loss"] = float(np.mean(val))
            # attach dataset-level (this-epoch) codebook diagnostics + per-epoch reset delta
            if hasattr(self.net_, "epoch_stats"):
                entry.update(self.net_.epoch_stats)
            if _q is not None:
                entry["dead_code_resets_epoch"] = int(_q.dead_code_resets.item()) - resets_before
            self.history_.append(entry)
        self.net_.eval()
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_ or self.net_ is None:
            raise RuntimeError("fit_encoder must be called before transform")
        import torch
        matrix, valid, _ = self._scaled(df)
        width = getattr(self, "_embedding_width", self.embedding_dim)
        out = np.full((len(df), width), np.nan)
        device = self._device()
        self.net_.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(valid), self.batch_size), desc=f"Encoding ({self.model})"):
                rows = valid[start:start + self.batch_size]
                batch = torch.from_numpy(gather_trailing_windows(matrix, rows, self.window)).to(device)
                out[rows] = self.net_.encode(batch).cpu().numpy()
        return pd.DataFrame(out, index=df.index, columns=self.embedding_columns_)


class SklearnRegimeEncoder(_ScaledWindowEncoder):
    """Wrap ANY scikit-learn transformer (PCA, FastICA, KernelPCA, ...) as an encoder.

    The estimator sees each trailing window flattened to ``window * features``
    columns.  It is cloned and fitted on (a seeded subsample of) training windows;
    ``transform`` is a pure function of the window, so it is causal by construction.
    """

    def __init__(self, estimator, window: int = 32, max_train_windows: int = 100_000, seed: int = 7,
                 normalization_window: int = 500, normalization_min_periods: int = 50,
                 clip_z: float = 8.0, feature_columns: Sequence[str] | None = None,
                 timestamp_col: str | None = None):
        super().__init__(window, 1, max_train_windows, seed, normalization_window,
                         normalization_min_periods, clip_z, feature_columns, timestamp_col)
        if not (hasattr(estimator, "fit") and hasattr(estimator, "transform")):
            raise TypeError("estimator must implement fit and transform")
        self.estimator = estimator
        self.estimator_ = None

    def fit_encoder(self, df: pd.DataFrame, y=None) -> "SklearnRegimeEncoder":
        if y is not None:
            raise ValueError("regime encoders do not train from forward outcomes")
        from sklearn.base import clone
        matrix, valid, values = self._scaled(df)
        rows = self._training_rows(valid)
        flat = gather_trailing_windows(matrix, rows, self.window).reshape(len(rows), -1)
        self.estimator_ = clone(self.estimator).fit(flat)
        self.input_columns_ = tuple(values.columns)
        dim = int(np.asarray(self.estimator_.transform(flat[:2])).shape[1])
        self.embedding_dim = dim
        self.embedding_columns_ = tuple(f"embedding_{i}" for i in range(dim))
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("fit_encoder must be called before transform")
        matrix, valid, _ = self._scaled(df)
        out = np.full((len(df), self.embedding_dim), np.nan)
        for start in range(0, len(valid), 50_000):
            rows = valid[start:start + 50_000]
            flat = gather_trailing_windows(matrix, rows, self.window).reshape(len(rows), -1)
            out[rows] = self.estimator_.transform(flat)
        return pd.DataFrame(out, index=df.index, columns=self.embedding_columns_)


# ---------------------------------------------------------------------------
# THE switch: pick the encoder by name.  Everything downstream is unchanged.
# ---------------------------------------------------------------------------

def make_encoder(model: str = "temporal_window", *, window: int = 32, embedding_dim: int = 16,
                 feature_columns: Sequence[str] | None = None, timestamp_col: str | None = None,
                 **params) -> BaseRegimeEncoder:
    """Build an encoder.  ``params`` are the chosen model's own hyperparameters."""
    common = dict(feature_columns=feature_columns, timestamp_col=timestamp_col)
    name = str(model).lower().replace("_", "-")
    if name in ("temporal-window", "baseline"):
        return TemporalWindowEncoder(window=window, bins=params.pop("bins", 8), **common, **params)
    elif name in ("vq-vae", "vqvae"):
        return TorchRegimeEncoder(
            model="vq-vae", window=window, embedding_dim=embedding_dim,
            epochs=params.pop("epochs", 10), batch_size=params.pop("batch_size", 256),
            learning_rate=params.pop("learning_rate", 1e-3),
            model_params={"num_codes": params.pop("num_codes", 64), "hidden": params.pop("hidden", 64),
                          "commitment": params.pop("commitment", 0.25),
                          "pooling": params.pop("pooling", "mean"),
                          "usage_decay": params.pop("usage_decay", 0.99),
                          "dead_threshold": params.pop("dead_threshold", 0.01)},
            **common, **params)
    elif name == "ts2vec":
        return TorchRegimeEncoder(
            model="ts2vec", window=window, embedding_dim=embedding_dim,
            epochs=params.pop("epochs", 10), batch_size=params.pop("batch_size", 256),
            learning_rate=params.pop("learning_rate", 1e-3),
            model_params={"hidden": params.pop("hidden", 64), "depth": params.pop("depth", 3),
                          "mask_prob": params.pop("mask_prob", 0.5)},
            **common, **params)
    elif name in ("tcn", "transformer"):
        return TorchRegimeEncoder(
            model=name, window=window, embedding_dim=embedding_dim,
            epochs=params.pop("epochs", 10), batch_size=params.pop("batch_size", 256),
            learning_rate=params.pop("learning_rate", 1e-3),
            model_params={"hidden": params.pop("hidden", 64)}, **common, **params)
    elif name == "pca":
        from sklearn.decomposition import PCA
        return SklearnRegimeEncoder(PCA(n_components=embedding_dim, random_state=params.get("seed", 7)),
                                    window=window, **common, **params)
    elif name == "sklearn":  # bring your own: make_encoder("sklearn", estimator=FastICA(8))
        return SklearnRegimeEncoder(params.pop("estimator"), window=window, **common, **params)
    raise ValueError(f"unknown encoder {model!r}; expected temporal_window, vq-vae, ts2vec, "
                     "tcn, transformer, pca, sklearn")


# Backwards-compatible class names (now thin presets over TorchRegimeEncoder).
def TCNRegimeEncoder(**kw): return make_encoder("tcn", **kw)
def TS2VecRegimeEncoder(**kw): return make_encoder("ts2vec", **kw)
def TransformerRegimeEncoder(**kw): return make_encoder("transformer", **kw)
def VQVAERegimeEncoder(**kw): return make_encoder("vq-vae", **kw)