"""Distance and similarity functions used by the OOD state registry."""

from __future__ import annotations

import numpy as np


def normalized_euclidean(left, right) -> float:
    """Root-mean-square Euclidean distance between equal-sized embeddings."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("embeddings must be one-dimensional arrays with equal shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nearest_prototype(embedding, prototypes: dict[str, np.ndarray]) -> tuple[str | None, float]:
    """Return nearest state ID and RMS distance, or (None, NaN) if empty."""
    if not prototypes:
        return None, np.nan
    distances = {
        state_id: normalized_euclidean(embedding, center)
        for state_id, center in prototypes.items()
    }
    finite = [(state_id, distance) for state_id, distance in distances.items() if np.isfinite(distance)]
    if not finite:
        return None, np.nan
    return min(finite, key=lambda pair: pair[1])


def diagonal_mahalanobis(embedding, center, scale, scale_floor: float = 0.1) -> float:
    """RMS diagonal-Mahalanobis distance using a state's observed spread."""
    value = np.asarray(embedding, dtype=np.float64)
    mean = np.asarray(center, dtype=np.float64)
    spread = np.asarray(scale, dtype=np.float64)
    if value.ndim != 1 or mean.shape != value.shape or spread.shape != value.shape:
        raise ValueError("embedding, center, and scale must be equal-sized vectors")
    if not np.isfinite(value).all() or not np.isfinite(mean).all() or not np.isfinite(spread).all():
        return np.nan
    normalized = (value - mean) / np.maximum(spread, float(scale_floor))
    return float(np.sqrt(np.mean(normalized ** 2)))


def similarity_from_distance(distance: float, scale: float) -> float:
    """Map nonnegative RMS distance to [0, 1], with 1 denoting a match."""
    if not np.isfinite(distance) or not np.isfinite(scale) or scale <= 0.0:
        return 0.0
    return float(np.exp(-max(float(distance), 0.0) / float(scale)))
