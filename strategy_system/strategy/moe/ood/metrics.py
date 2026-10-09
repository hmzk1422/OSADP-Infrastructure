"""Swappable similarity metrics for regime validation.

Each validation slot maps to a named implementation:

    slot                             default      alternatives
    -------------------------------  -----------  ------------------------------
    embedding_similarity             cosine       euclidean | pearson
    feature_similarity               cosine       euclidean | pearson
    temporal_pattern_similarity      resampled    dtw | flattened_cosine
    episode_duration_similarity      quantile     mean | overlap
    prototype_pearson_corr           pearson      spearman | kendall

``RegimeValidator`` dispatches to the selected implementation with an explicit
``if/elif`` block; ``settings.yaml`` selects the name under ``ood.metrics``.

The defaults reproduce the original hard-coded behaviour exactly, so running
with no ``ood.metrics`` override changes no numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Vector metrics: compare two aggregated 1-D vectors -> similarity in ~[0, 1].
# --------------------------------------------------------------------------- #
def cosine_similarity(left, right) -> float:
    """Cosine similarity remapped from [-1, 1] to [0, 1] (original behaviour)."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return np.nan
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.clip((np.dot(left, right) / denom + 1.0) / 2.0, 0.0, 1.0)) if denom > 1e-12 else float(np.allclose(left, right))


def euclidean_similarity(left, right) -> float:
    """exp(-normalized RMS distance) between two vectors, in (0, 1]."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return np.nan
    scale = max(float(np.sqrt(np.mean(np.concatenate([left ** 2, right ** 2])))), 1e-12)
    distance = float(np.sqrt(np.mean((left - right) ** 2))) / scale
    return float(np.exp(-distance))


def pearson_similarity(left, right) -> float:
    """Centred (Pearson) correlation remapped from [-1, 1] to [0, 1]."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return np.nan
    left_c, right_c = left - left.mean(), right - right.mean()
    denom = np.linalg.norm(left_c) * np.linalg.norm(right_c)
    return float(np.clip((np.dot(left_c, right_c) / denom + 1.0) / 2.0, 0.0, 1.0)) if denom > 1e-12 else float(np.allclose(left, right))


VECTOR_METRICS: dict[str, Callable[[Sequence, Sequence], float]] = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_similarity,
    "pearson": pearson_similarity,
}


# --------------------------------------------------------------------------- #
# Sequence metrics: compare two time-ordered 2-D paths -> similarity in ~[0, 1].
# --------------------------------------------------------------------------- #
def _resample(array: np.ndarray, points: int) -> np.ndarray:
    """Resample a (T, D) path onto a common relative-time grid of ``points``."""
    source = np.linspace(0.0, 1.0, len(array))
    grid = np.linspace(0.0, 1.0, points)
    return np.stack([np.interp(grid, source, array[:, col]) for col in range(array.shape[1])], axis=1)


def resampled_distance_similarity(left, right) -> float:
    """exp(-normalized RMS distance) of equal-length resampled paths (original)."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        return np.nan
    points = min(max(len(a), len(b)), 64)
    aa, bb = _resample(a, points), _resample(b, points)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        return np.nan
    scale = np.sqrt(np.mean(np.concatenate([aa, bb], axis=0) ** 2))
    distance = np.sqrt(np.mean((aa - bb) ** 2)) / max(float(scale), 1e-12)
    return float(np.exp(-distance))


def dtw_similarity(left, right) -> float:
    """exp(-mean warping cost) of resampled paths under dynamic time warping."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        return np.nan
    points = min(max(len(a), len(b)), 64)
    aa, bb = _resample(a, points), _resample(b, points)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        return np.nan
    n, m = len(aa), len(bb)
    cost = np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    scale = max(float(np.sqrt(np.mean(np.concatenate([aa, bb], axis=0) ** 2))), 1e-12)
    return float(np.exp(-(acc[n, m] / max(n, m)) / scale))


def flattened_cosine_similarity(left, right) -> float:
    """Cosine similarity of the flattened resampled paths."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        return np.nan
    points = min(max(len(a), len(b)), 64)
    return cosine_similarity(_resample(a, points).ravel(), _resample(b, points).ravel())


SEQUENCE_METRICS: dict[str, Callable[[Sequence, Sequence], float]] = {
    "resampled": resampled_distance_similarity,
    "dtw": dtw_similarity,
    "flattened_cosine": flattened_cosine_similarity,
}


# --------------------------------------------------------------------------- #
# Distribution metrics: compare two duration samples -> similarity in ~[0, 1].
# --------------------------------------------------------------------------- #
def quantile_similarity(left, right) -> float:
    """exp(-mean |quantile gap| / mean level) over 21 quantiles (original)."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if not len(left) or not len(right):
        return np.nan
    quantiles = np.linspace(0.0, 1.0, 21)
    a, b = np.quantile(left, quantiles), np.quantile(right, quantiles)
    distance = float(np.mean(np.abs(a - b))) / max(float(np.mean(np.concatenate([left, right]))), 1.0)
    return float(np.exp(-distance))


def mean_similarity(left, right) -> float:
    """Compare distribution location and spread jointly via exp(-normalized gap)."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if not len(left) or not len(right):
        return np.nan
    mean_l, mean_r = float(np.mean(left)), float(np.mean(right))
    std_l, std_r = float(np.std(left)), float(np.std(right))
    mean_gap = abs(mean_l - mean_r) / max(abs(mean_l) + abs(mean_r), 1.0)
    std_gap = abs(std_l - std_r) / max(std_l + std_r, 1e-12)
    return float(np.exp(-0.5 * (mean_gap + std_gap)))


def overlap_similarity(left, right) -> float:
    """Histogram intersection (overlap coefficient) in [0, 1]."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if not len(left) or not len(right):
        return np.nan
    edges = np.unique(np.concatenate([left, right]))
    if len(edges) < 2:
        return 1.0
    hist_l, _ = np.histogram(left, bins=edges)
    hist_r, _ = np.histogram(right, bins=edges)
    prob_l, prob_r = hist_l / hist_l.sum(), hist_r / hist_r.sum()
    return float(np.minimum(prob_l, prob_r).sum())


DISTRIBUTION_METRICS: dict[str, Callable[[Sequence, Sequence], float]] = {
    "quantile": quantile_similarity,
    "mean": mean_similarity,
    "overlap": overlap_similarity,
}


# --------------------------------------------------------------------------- #
# Prototype metrics: correlate a {state: prototype-vector} mapping -> DataFrame.
# --------------------------------------------------------------------------- #
def prototype_pearson(prototypes) -> pd.DataFrame:
    return pd.DataFrame(prototypes).corr(method="pearson")


def prototype_spearman(prototypes) -> pd.DataFrame:
    return pd.DataFrame(prototypes).corr(method="spearman")


def prototype_kendall(prototypes) -> pd.DataFrame:
    return pd.DataFrame(prototypes).corr(method="kendall")


PROTOTYPE_METRICS: dict[str, Callable[[dict], pd.DataFrame]] = {
    "pearson": prototype_pearson,
    "spearman": prototype_spearman,
    "kendall": prototype_kendall,
}


# --------------------------------------------------------------------------- #
# Slot registry, defaults, and override resolution.
# --------------------------------------------------------------------------- #
DEFAULT_METRICS: dict[str, str] = {
    "embedding_similarity": "cosine",
    "feature_similarity": "cosine",
    "temporal_pattern_similarity": "resampled",
    "episode_duration_similarity": "quantile",
    "prototype_pearson_corr": "pearson",
}

SLOT_METRICS: dict[str, dict[str, Callable]] = {
    "embedding_similarity": VECTOR_METRICS,
    "feature_similarity": VECTOR_METRICS,
    "temporal_pattern_similarity": SEQUENCE_METRICS,
    "episode_duration_similarity": DISTRIBUTION_METRICS,
    "prototype_pearson_corr": PROTOTYPE_METRICS,
}


def available_metrics() -> dict[str, list[str]]:
    """Return every slot with its sorted list of implementation names."""
    return {slot: sorted(impls) for slot, impls in SLOT_METRICS.items()}


def resolve_metrics(overrides) -> dict[str, str]:
    """Merge ``overrides`` (ood.metrics) onto the defaults, validating names."""
    resolved = dict(DEFAULT_METRICS)
    if not overrides:
        return resolved
    if not isinstance(overrides, dict):
        raise TypeError("ood.metrics must be a mapping of slot -> metric name")
    unknown_slots = sorted(set(overrides) - set(DEFAULT_METRICS))
    if unknown_slots:
        raise ValueError(f"unknown metric slots: {unknown_slots}; choose from {sorted(DEFAULT_METRICS)}")
    for slot, name in overrides.items():
        if name is None:
            continue
        impls = SLOT_METRICS[slot]
        if name not in impls:
            raise ValueError(f"metric '{name}' is not valid for slot '{slot}'; choose from {sorted(impls)}")
        resolved[slot] = name
    return resolved

