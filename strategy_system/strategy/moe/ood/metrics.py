"""Swappable similarity metrics for regime validation.

Each validation slot maps to a named implementation:

    slot                             default      alternatives
    -------------------------------  -----------  ------------------------------
    embedding_similarity             cosine       euclidean | pearson
    feature_similarity               cosine       euclidean | pearson
    temporal_pattern_similarity      resampled    dtw | flattened_cosine | tdmi
    episode_duration_similarity      quantile     mean | overlap
    prototype_pearson_corr           pearson      spearman | kendall

``RegimeValidator`` dispatches to the selected implementation with an explicit
``if/elif`` block; ``settings.yaml`` selects the name under ``ood.metrics``.

The defaults reproduce the original hard-coded behaviour exactly, so running
with no ``ood.metrics`` override changes no numbers.

``tdmi`` (time-delayed mutual information) is the temporality-aware alternative
for ``temporal_pattern_similarity``: it compares how strongly each episode's
features depend on their own past (a null-calibrated lag x feature fingerprint)
instead of comparing path shapes through an ``exp(-distance)`` kernel. Estimation
is within-episode; see the TDMI section below for the design and its caveats.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import NamedTuple

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


# --------------------------------------------------------------------------- #
# Time-delayed mutual information (TDMI).
#
#   TDMI_f(tau) = I( x_f[t] ; x_f[t + tau] | episode )          (nats, per feature f)
#
# Design (each point exists because the naive version fails without it):
#   * WITHIN-EPISODE estimation. Each episode is rank-binned on its own and the
#     per-episode estimates are averaged, weighted by pair count. Pooling raw
#     samples across episodes would let a shift in the feature's level between
#     recurrences masquerade as temporal dependence (a mixture is dependent even
#     when every component is i.i.d.).
#   * SIGNED aggregation. Per-episode Miller-Madow estimates are NOT clipped at 0
#     before averaging (clipping adds upward bias); the aggregate is clipped.
#   * NULL calibration by circular-shift surrogates: lead/lag pairs are re-paired
#     after a circular shift of the lagged series, preserving each marginal and
#     its serial structure while destroying the pairing. Raw MI is reported
#     alongside its surrogate null; a fixed noise floor is not used.
#   * Surrogate k of a segment uses ONE shift fraction shared by every (lag,
#     feature) cell, so the null keeps the cross-cell dependence that the real
#     estimates have. Independent shifts per cell would make the null of any
#     cross-cell mean (tdmi_p, tdmi_null) too narrow and its p-values too small.
#   * Lags are in bars; paths are never resampled; no pair crosses an episode
#     boundary. Rank binning is invariant to monotone transforms.
#   * The feature axis is kept: fingerprints are (lag x feature).
#
# Caveats: a trend or level drift *inside* an episode still looks like temporal
# dependence; shifts are >= n/4 pairs, so for very persistent series the null is
# conservative; TDMI is within-feature self-dependence, not predictive power.
# --------------------------------------------------------------------------- #
TDMI_MAX_LAG = 8           # lags 1..TDMI_MAX_LAG (bars)
TDMI_MIN_PAIRS = 30        # a segment needs this many lag pairs to contribute at a lag
TDMI_MAX_BINS = 16         # cap on bins per axis (auto rule: floor(sqrt(n_pairs / 5)))
TDMI_SURROGATES = 49       # circular-shift surrogates for the null (p-value resolution 1/50)
TDMI_NULL_QUANTILE = 0.95  # 1 - alpha: a cell is "above null" at false-positive rate <= 5%
TDMI_SEED = 0


class TDMIEstimate(NamedTuple):
    """Pair-count-weighted TDMI per (lag, feature) with its surrogate null."""
    mi: np.ndarray            # (L, D) signed Miller-Madow estimate; NaN where undefined
    weight: np.ndarray        # (L, D) number of lag pairs behind each cell
    null: np.ndarray | None   # (K, L, D) surrogate estimates, or None when K == 0


def _average_rank(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (matches pandas 'average')."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    new_group = np.concatenate([[True], sorted_values[1:] != sorted_values[:-1]])
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.concatenate([starts, [len(values)]]))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (starts + (counts + 1.0) / 2.0)[np.cumsum(new_group) - 1]
    return ranks


def _rank_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Equiprobable bin index in [0, bins) from average ranks (ties share a bin)."""
    ranks = _average_rank(values)
    return np.minimum(((ranks - 1.0) / len(values) * bins).astype(np.int64), bins - 1)


def _mi_batch(bx: np.ndarray, by_batch: np.ndarray, bins: int) -> np.ndarray:
    """Signed Miller-Madow MI (nats) of ``bx`` against each row of ``by_batch`` (K, n)."""
    n, k = len(bx), by_batch.shape[0]
    offsets = (np.arange(k, dtype=np.int64) * bins * bins)[:, None]
    flat = (offsets + bx[None, :] * bins + by_batch).ravel()
    joint = np.bincount(flat, minlength=k * bins * bins).reshape(k, bins, bins).astype(np.float64)
    px, py = joint.sum(axis=2), joint.sum(axis=1)
    occupied = joint > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(occupied, np.log(joint * n / (px[:, :, None] * py[:, None, :])), 0.0)
    mi = (joint / n * log_ratio).sum(axis=(1, 2))
    # Miller-Madow: H_MM = H_ML + (K - 1) / 2n  =>  I_MM = I_ML + (Kx + Ky - Kxy - 1) / 2n
    return mi + ((px > 0).sum(axis=1) + (py > 0).sum(axis=1) - occupied.sum(axis=(1, 2)) - 1) / (2.0 * n)


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Miller-Madow histogram MI (nats) of two equal-length samples, clipped at 0."""
    return max(float(_mi_batch(_rank_bins(x, bins), _rank_bins(y, bins)[None, :], bins)[0]), 0.0)


def tdmi_bins(n_pairs: float) -> int:
    """Automatic bins per axis: floor(sqrt(n_pairs / 5)), i.e. >= ~5 pairs per joint cell.

    Sparser joint tables break the Miller-Madow correction (it under-corrects when cells
    are empty), which leaves a positive bias on short episodes.
    """
    return int(np.clip(np.floor(np.sqrt(float(n_pairs) / 5.0)), 2, TDMI_MAX_BINS))


def _as_segments(paths) -> list[np.ndarray] | None:
    """Normalise paths to contiguous (T, D) runs, splitting on all-invalid rows.

    An all-invalid (all-NaN) row is a *separator*, not a value to drop and bridge
    over: a caller may mask rows belonging to another latent with NaNs, and
    concatenating across such a separator would let a level shift between the two
    runs masquerade as within-episode temporal dependence. Partial feature
    missingness is retained here; each feature/lag pair is filtered independently
    later. ``None`` is returned only when the input is unusable.
    """
    if isinstance(paths, (np.ndarray, pd.DataFrame, pd.Series)):
        paths = [paths]
    segments: list[np.ndarray] = []
    width = None
    for path in paths:
        array = np.asarray(path, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            return None
        if width is None:
            width = array.shape[1]
        elif array.shape[1] != width:
            return None
        # Split on rows that are entirely non-finite; never bridge across them.
        row_has_value = np.isfinite(array).any(axis=1)
        starts = np.flatnonzero(row_has_value & np.r_[True, ~row_has_value[:-1]])
        ends = np.flatnonzero(row_has_value & np.r_[~row_has_value[1:], True]) + 1
        segments.extend(array[a:b] for a, b in zip(starts, ends) if b > a)
    if not segments or len({segment.shape[1] for segment in segments}) != 1:
        return None
    return segments


def _check_tdmi_args(max_lag, bins, min_pairs) -> tuple[int, int | None, int]:
    max_lag = TDMI_MAX_LAG if max_lag is None else int(max_lag)
    min_pairs = TDMI_MIN_PAIRS if min_pairs is None else int(min_pairs)
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    if min_pairs < 4:
        raise ValueError("min_pairs must be >= 4")
    if bins is not None and int(bins) < 2:
        raise ValueError("bins must be >= 2 (or None for the automatic rule)")
    return max_lag, (None if bins is None else int(bins)), min_pairs


def tdmi_estimate(paths, max_lag=None, bins=None, min_pairs=None,
                  n_surrogates: int = 0, seed=TDMI_SEED) -> TDMIEstimate:
    """Within-episode TDMI per (lag, feature), pair-count weighted across ``paths``.

    ``paths`` is one (T, D) path or a list of them (e.g. every episode of one
    latent). Each path is rank-binned on its own, so differences in level or scale
    between episodes cannot create dependence; lag pairs never cross a path
    boundary; non-finite values drop only the pairs they touch. With
    ``n_surrogates`` > 0 each path also gets circular-shift surrogates, aggregated
    the same way, giving the null distribution of every cell. Surrogate k uses one
    shift fraction for all cells of a path, so cross-cell dependence is preserved.
    """
    max_lag, bins, min_pairs = _check_tdmi_args(max_lag, bins, min_pairs)
    n_surrogates = int(n_surrogates)
    if n_surrogates < 0:
        raise ValueError("n_surrogates must be >= 0")
    segments = _as_segments(paths)
    if segments is None:
        nan = np.full((max_lag, 1), np.nan)
        return TDMIEstimate(nan, np.zeros((max_lag, 1)), np.full((n_surrogates, max_lag, 1), np.nan) if n_surrogates else None)
    width = segments[0].shape[1]
    if bins is None:
        sizes = [len(seg) - 1 for seg in segments if len(seg) - 1 >= min_pairs]
        bins = tdmi_bins(np.median(sizes)) if sizes else 2
    rng = np.random.default_rng(seed)
    mi_sum = np.zeros((max_lag, width))
    weight = np.zeros((max_lag, width))
    null_sum = np.zeros((n_surrogates, max_lag, width)) if n_surrogates else None
    for segment in segments:
        # One shift fraction per surrogate, shared by every (lag, feature) cell of this
        # segment, so the null keeps the cross-cell dependence of the real estimates.
        shared = rng.uniform(0.25, 0.75, size=n_surrogates) if n_surrogates else None
        for lag in range(1, max_lag + 1):
            if len(segment) <= lag:
                continue
            for col in range(width):
                a, b = segment[:-lag, col], segment[lag:, col]
                ok = np.isfinite(a) & np.isfinite(b)
                a, b = a[ok], b[ok]
                n = len(a)
                if n < min_pairs:
                    continue
                bx, by = _rank_bins(a, bins), _rank_bins(b, bins)
                mi_sum[lag - 1, col] += n * _mi_batch(bx, by[None, :], bins)[0]
                weight[lag - 1, col] += n
                if n_surrogates:
                    shifts = np.clip(np.rint(shared * n).astype(np.int64), n // 4, n - n // 4)
                    index = (np.arange(n)[None, :] + shifts[:, None]) % n
                    null_sum[:, lag - 1, col] += n * _mi_batch(bx, by[index], bins)
    defined = weight > 0
    mi = np.divide(mi_sum, weight, out=np.full_like(mi_sum, np.nan), where=defined)
    null = None
    if n_surrogates:
        null = np.divide(null_sum, weight[None], out=np.full_like(null_sum, np.nan),
                         where=np.broadcast_to(defined, null_sum.shape))
    return TDMIEstimate(mi, weight, null)


def combine_tdmi_estimates(estimates: Sequence[TDMIEstimate]) -> TDMIEstimate:
    """Pair-count-weighted combination of estimates (e.g. one per episode -> one per latent)."""
    estimates = list(estimates)
    if not estimates:
        raise ValueError("no estimates to combine")
    weight = sum(e.weight for e in estimates)
    mi_sum = sum(np.where(e.weight > 0, e.mi * e.weight, 0.0) for e in estimates)
    mi = np.divide(mi_sum, weight, out=np.full_like(mi_sum, np.nan), where=weight > 0)
    null = None
    if all(e.null is not None for e in estimates):
        null_sum = sum(np.where(e.weight[None] > 0, e.null * e.weight[None], 0.0) for e in estimates)
        null = np.divide(null_sum, weight[None], out=np.full_like(null_sum, np.nan),
                         where=np.broadcast_to(weight > 0, null_sum.shape))
    return TDMIEstimate(mi, weight, null)


def tdmi_calibrate(estimate: TDMIEstimate, alpha: float = 1.0 - TDMI_NULL_QUANTILE) -> np.ndarray:
    """Evidence above the null: max(mi - threshold, 0) per cell; NaN where undefined.

    The threshold is the permutation-test order statistic: a cell is non-zero only if its
    estimate exceeds all but ``floor(alpha * (K + 1)) - 1`` of the K surrogates, which gives
    a false-positive rate of at most ``alpha`` for exchangeable nulls (an interpolated
    quantile of a few dozen draws is noticeably anti-conservative). Needs K >= 1/alpha - 1.
    Cells whose raw MI is undefined (NaN) stay NaN rather than being reported as 0.
    """
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if estimate.null is None:
        raise ValueError("estimate has no surrogates; call tdmi_estimate with n_surrogates > 0")
    k = estimate.null.shape[0]
    if k + 1 < 1.0 / alpha:
        raise ValueError(f"n_surrogates={k} is too few for alpha={alpha}; need >= {int(np.ceil(1.0 / alpha)) - 1}")
    allowed = int(np.floor(alpha * (k + 1)))                 # p = (1 + #null >= obs) / (K + 1) <= alpha
    threshold = np.sort(estimate.null, axis=0)[k - allowed]  # the `allowed`-th largest surrogate
    result = np.maximum(estimate.mi - threshold, 0.0)
    result[~np.isfinite(estimate.mi)] = np.nan
    return result


def tdmi_matrix(paths, max_lag=None, bins=None, min_pairs=None, n_surrogates: int = 0,
                seed=TDMI_SEED, calibrate: bool = False) -> np.ndarray:
    """TDMI per (lag, feature), shape (max_lag, D). Raw (clipped at 0) or null-calibrated."""
    estimate = tdmi_estimate(paths, max_lag, bins, min_pairs, n_surrogates, seed)
    return tdmi_calibrate(estimate) if calibrate else np.maximum(estimate.mi, 0.0)


def tdmi_profile(paths, max_lag=None, bins=None, min_pairs=None) -> np.ndarray:
    """Raw TDMI by lag, averaged over features: a *summary* (the feature axis is lost).

    Use ``tdmi_fingerprint`` when features must stay distinguishable.
    """
    matrix = tdmi_matrix(paths, max_lag, bins, min_pairs)
    finite = np.isfinite(matrix)
    count = finite.sum(axis=1)
    total = np.where(finite, matrix, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def tdmi_score(paths, max_lag=None, bins=None, min_pairs=None) -> float:
    """Scalar TDMI: mean of the raw lag profile over its defined lags (nats)."""
    profile = tdmi_profile(paths, max_lag, bins, min_pairs)
    profile = profile[np.isfinite(profile)]
    return float(profile.mean()) if len(profile) else np.nan


def tdmi_fingerprint(paths, max_lag=None, bins=None, min_pairs=None,
                     n_surrogates: int = TDMI_SURROGATES, seed=TDMI_SEED) -> np.ndarray:
    """Lag-by-feature TDMI fingerprint, flattened lag-major to length ``max_lag * D``.

    Null-calibrated when surrogates are enabled; with ``n_surrogates=0`` the raw
    nonnegative fingerprint is returned instead of trying to calibrate against a
    null distribution that does not exist.
    """
    estimate = tdmi_estimate(paths, max_lag, bins, min_pairs, n_surrogates, seed)
    cells = tdmi_calibrate(estimate) if estimate.null is not None else np.maximum(estimate.mi, 0.0)
    return cells.ravel()


def tdmi_fingerprint_similarity(left, right, floor: float = 0.0) -> float:
    """Weighted Jaccard of two fingerprints, in [0, 1] (no exponential kernel).

    ``sum(min) / sum(max)`` over cells defined in both. Fingerprints are null-calibrated,
    so insignificant cells are exactly 0 and two episodes with no detectable temporal
    structure are identical by convention (1.0): that is *agreement about absence*, not
    evidence of a recurring pattern -- read it together with the share of cells above
    null (``tdmi_sig_frac`` in the validator tables). ``floor`` > 0 optionally adds a
    constant to every cell before the ratio.
    """
    p, q = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and >= 0")
    if p.shape != q.shape or p.ndim != 1:
        return np.nan
    ok = np.isfinite(p) & np.isfinite(q)
    if not ok.any():
        return np.nan
    p, q = p[ok] + floor, q[ok] + floor
    high = float(np.maximum(p, q).sum())
    return float(np.minimum(p, q).sum() / high) if high > 1e-12 else 1.0


tdmi_profile_similarity = tdmi_fingerprint_similarity  # backwards-compatible alias


def _valid_path_pair(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        return None
    return a, b


def tdmi_similarity(left, right, max_lag=None, bins=None, min_pairs=None,
                    n_surrogates: int = TDMI_SURROGATES, seed=TDMI_SEED, floor: float = 0.0) -> float:
    """Similarity of two episode paths' temporal-dependence fingerprints, in [0, 1].

    Each (T, D) path is summarised by its null-calibrated lag x feature TDMI
    fingerprint (features stay distinguishable); fingerprints are compared with
    ``tdmi_fingerprint_similarity``. Paths are NOT resampled: lag is in bars.
    All-invalid rows split a path into independent contiguous runs. When ``bins``
    is omitted, both paths use the same automatic bin count for comparability.
    """
    pair = _valid_path_pair(left, right)
    if pair is None:
        return np.nan
    a, b = pair
    max_lag, bins, min_pairs = _check_tdmi_args(max_lag, bins, min_pairs)
    if bins is None:
        both = _as_segments([a, b])
        sizes = [len(seg) - 1 for seg in (both or []) if len(seg) - 1 >= min_pairs]
        bins = tdmi_bins(np.median(sizes)) if sizes else 2
    fa = tdmi_fingerprint(a, max_lag, bins, min_pairs, n_surrogates, seed)
    fb = tdmi_fingerprint(b, max_lag, bins, min_pairs, n_surrogates, seed)
    return tdmi_fingerprint_similarity(fa, fb, floor)


def make_tdmi_similarity(max_lag=None, bins=None, min_pairs=None, n_surrogates: int = TDMI_SURROGATES,
                         seed=TDMI_SEED, floor: float = 0.0) -> Callable[[Sequence, Sequence], float]:
    """``tdmi_similarity`` with fixed settings and a per-path fingerprint cache.

    ``build_temporal_similarity_matrix`` compares the same episode against many
    others; caching its fingerprint (keyed by content) computes each one once.
    Pass ``bins`` explicitly to use one bin count for every episode.
    """
    _check_tdmi_args(max_lag, bins, min_pairs)
    cache: dict[bytes, np.ndarray] = {}

    def fingerprint(array: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(array)
        key = hashlib.blake2b(array.tobytes() + repr(array.shape).encode(), digest_size=16).digest()
        if key not in cache:
            cache[key] = tdmi_fingerprint(array, max_lag, bins, min_pairs, n_surrogates, seed)
        return cache[key]

    def similarity(left, right) -> float:
        pair = _valid_path_pair(left, right)
        return np.nan if pair is None else tdmi_fingerprint_similarity(fingerprint(pair[0]), fingerprint(pair[1]), floor)

    similarity.__doc__ = tdmi_similarity.__doc__
    return similarity


SEQUENCE_METRICS: dict[str, Callable[[Sequence, Sequence], float]] = {
    "resampled": resampled_distance_similarity,
    "dtw": dtw_similarity,
    "flattened_cosine": flattened_cosine_similarity,
    "tdmi": tdmi_similarity,
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