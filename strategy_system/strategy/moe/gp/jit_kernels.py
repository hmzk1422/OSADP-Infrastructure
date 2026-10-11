"""Numba-compiled hot kernels for the GP search.

Every kernel is ``nogil=True`` so the (state, target) thread pool in
``gp/parallel.py`` runs them truly in parallel (pandas ``rank`` holds the GIL).
Results match the previous pandas implementations to float precision.
If numba is not installed the same functions run as plain Python (slow, but
correct), so imports never break.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from numba import njit
except ImportError:  # graceful fallback
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn

__all__ = ["spearman_kernel", "blockwise_spearman_kernel", "abs_corr_exceeds"]


@njit(cache=True, nogil=True)
def _average_ranks(values):
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # 1-based average rank of the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


@njit(cache=True, nogil=True)
def spearman_kernel(left, right):
    """Return (rho, count) over rows where both inputs are finite; rho=nan if undefined."""
    n = left.shape[0]
    valid = np.empty(n, dtype=np.bool_)
    count = 0
    for i in range(n):
        ok = np.isfinite(left[i]) and np.isfinite(right[i])
        valid[i] = ok
        if ok:
            count += 1
    if count < 2:
        return np.nan, count
    x = np.empty(count, dtype=np.float64)
    y = np.empty(count, dtype=np.float64)
    k = 0
    for i in range(n):
        if valid[i]:
            x[k] = left[i]
            y[k] = right[i]
            k += 1
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    mx = rx.mean()
    my = ry.mean()
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in range(count):
        dx = rx[i] - mx
        dy = ry[i] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    denom = np.sqrt(sxx * syy)
    if denom <= 1e-15:
        return np.nan, count
    return sxy / denom, count


@njit(cache=True, nogil=True)
def blockwise_spearman_kernel(values, target, train_mask, train_end, block_count, min_block_samples):
    """Spearman IC per chronological block of ``[0, train_end)`` (np.array_split rules)."""
    out = np.empty(block_count, dtype=np.float64)
    used = 0
    base = train_end // block_count
    extra = train_end % block_count
    start = 0
    for b in range(block_count):
        size = base + (1 if b < extra else 0)
        stop = start + size
        m = 0
        for i in range(start, stop):
            if train_mask[i]:
                m += 1
        if m >= min_block_samples:
            xs = np.empty(m, dtype=np.float64)
            ys = np.empty(m, dtype=np.float64)
            k = 0
            for i in range(start, stop):
                if train_mask[i]:
                    xs[k] = values[i]
                    ys[k] = target[i]
                    k += 1
            rho, cnt = spearman_kernel(xs, ys)
            if cnt >= min_block_samples and np.isfinite(rho):
                out[used] = rho
                used += 1
        start = stop
    return out[:used]


@njit(cache=True, nogil=True)
def abs_corr_exceeds(candidate, prior, mask, threshold, min_samples):
    """True if |Pearson(candidate, prior)| >= threshold on finite masked rows."""
    n = candidate.shape[0]
    count = 0
    for i in range(n):
        if mask[i] and np.isfinite(candidate[i]) and np.isfinite(prior[i]):
            count += 1
    if count < min_samples:
        return False
    sa = 0.0
    sb = 0.0
    for i in range(n):
        if mask[i] and np.isfinite(candidate[i]) and np.isfinite(prior[i]):
            sa += candidate[i]
            sb += prior[i]
    ma = sa / count
    mb = sb / count
    saa = 0.0
    sbb = 0.0
    sab = 0.0
    for i in range(n):
        if mask[i] and np.isfinite(candidate[i]) and np.isfinite(prior[i]):
            da = candidate[i] - ma
            db = prior[i] - mb
            saa += da * da
            sbb += db * db
            sab += da * db
    if saa / count <= 1e-24 or sbb / count <= 1e-24:  # std <= 1e-12
        return False
    corr = sab / np.sqrt(saa * sbb)
    return bool(np.isfinite(corr) and abs(corr) >= threshold)