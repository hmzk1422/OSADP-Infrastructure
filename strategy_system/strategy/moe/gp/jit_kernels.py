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
    from numba import njit, prange
except ImportError:  # graceful fallback
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn

    def prange(*args):
        return range(*args)

__all__ = ["spearman_kernel", "blockwise_spearman_kernel", "abs_corr_exceeds", "batch_spearman_kernel", "rolling_mean_kernel", "rolling_std_kernel", "lag_kernel", "diff_kernel", "quantize_tick_kernel", "barrier_sim_kernel", "excursion_kernel", "elapsed_err_kernel", "barrier_outcome_kernel"]


@njit(cache=True, nogil=True)
def _average_ranks(values):
    # NOTE: kept as serial core; parallel fan-out happens in spearman_kernel / batch kernel via prange.
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


@njit(cache=True, nogil=True, parallel=True)
def batch_spearman_kernel(vals, tgts, min_samples):
    """Batched (rho, n) over stacked candidate matrix. Heavy: parallel fan-out."""
    n_cand = vals.shape[0]
    out_rho = np.empty(n_cand, dtype=np.float64)
    out_n = np.empty(n_cand, dtype=np.int64)
    for c in prange(n_cand):
        rho, cnt = spearman_kernel(vals[c], tgts[c])
        if cnt < min_samples or not np.isfinite(rho):
            out_rho[c] = np.nan
        else:
            out_rho[c] = rho
        out_n[c] = cnt
    return out_rho, out_n


@njit(cache=True, nogil=True, parallel=True)
def rolling_mean_kernel(x, window):
    """Causal rolling mean, min_periods=window. Heavy: parallel."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        if i + 1 < window:
            out[i] = np.nan
            continue
        s = 0.0
        c = 0
        for j in range(i - window + 1, i + 1):
            v = x[j]
            if np.isfinite(v):
                s += v
                c += 1
        if c < window:
            out[i] = np.nan
        else:
            out[i] = np.float32(s / c)
    return out


@njit(cache=True, nogil=True, parallel=True)
def rolling_std_kernel(x, window):
    """Causal population rolling std (ddof=0), min_periods=window. Heavy: parallel."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        if i + 1 < window:
            out[i] = np.nan
            continue
        s = 0.0
        s2 = 0.0
        c = 0
        for j in range(i - window + 1, i + 1):
            v = x[j]
            if np.isfinite(v):
                s += v
                s2 += v * v
                c += 1
        if c < window:
            out[i] = np.nan
        else:
            m = s / c
            var = s2 / c - m * m
            out[i] = np.float32(np.sqrt(var)) if var > 0.0 else np.float32(0.0)
    return out


@njit(cache=True, nogil=True)
def lag_kernel(x, lag):
    """Causal lag; NaN for i < lag. Lightweight: serial (no parallel)."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan if i < lag else x[i - lag]
    return out


@njit(cache=True, nogil=True)
def diff_kernel(x, lag):
    """Causal diff x[i]-x[i-lag]; NaN for i < lag. Lightweight: serial."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        if i < lag:
            out[i] = np.nan
        else:
            a = x[i]
            b = x[i - lag]
            out[i] = np.float32(a - b) if (np.isfinite(a) and np.isfinite(b)) else np.float32(np.nan)
    return out



def _tick_decimals(tick) -> int:
    """Decimal places implied by Decimal(str(tick)) -- the exponent .quantize() uses.

    quantize() ignores the tick's magnitude entirely: str(0.01) -> 2 decimals,
    str(1.0) -> 1 decimal (grid 0.1), str(100) -> 0. This mirrors that rule
    exactly for int and float ticks alike.
    """
    from decimal import Decimal
    return max(0, -int(Decimal(str(tick)).as_tuple().exponent))


def quantize_tick_kernel(price, tick):
    """Tick snap with exact Decimal(str(v)).quantize(str(tick), ROUND_HALF_UP) parity.

    Decimal .quantize() rounds to the tick's decimal EXPONENT (grid = 10^-d,
    d from str(tick)), ties (exact .5) away from zero. The njit kernel
    computes k = floor(|v|*10^d + 0.5 + bias) with bias = 1e-6 + count*1.1e-15
    absorbing float error of v*10^d (canonical .5-tie doubles land within
    ~half-ulp of the tie; values in that sub-tie window also resolve up --
    indistinguishable from the tie in float64), then k/10^d is correctly
    rounded to the nearest grid double = float(Decimal) bit for bit.
    Non-finite values pass through. The wrapper resolves d from str(tick)
    before entering numba (str() is unavailable in nopython mode).
    """
    arr = np.ascontiguousarray(np.asarray(price, dtype=np.float64).reshape(-1))
    return _quantize_tick_kernel(arr, _tick_decimals(tick))


@njit(cache=True, nogil=True)
def _quantize_tick_kernel(price, decimals):
    """Numba core: round-half-up to the 10^-decimals grid; ties away from zero."""
    n = price.shape[0]
    out = np.empty(n, dtype=np.float64)
    s = 10.0 ** decimals
    for i in range(n):
        v = price[i]
        if not np.isfinite(v):
            out[i] = v
            continue
        c = np.abs(v) * s
        k = np.floor(c + 0.5 + c * 1.1e-15 + 1e-6)
        out[i] = (k / s) if v >= 0.0 else -(k / s)
    return out


def quantize_tick_scalar(v, tick: float) -> float:
    """Python twin of quantize_tick_kernel: same str(tick) decimals, same IEEE
    op order, same Decimal ROUND_HALF_UP parity (ties away from zero).

    Non-numeric inputs and non-finite floats pass through unchanged (the
    Decimal path also leaves non-finite values alone).
    """
    import math
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return v
    s = 10.0 ** _tick_decimals(tick)
    c = abs(f) * s
    k = math.floor(c + 0.5 + c * 1.1e-15 + 1e-6)
    return (k / s) if f >= 0.0 else -(k / s)


@njit(cache=True, nogil=True, parallel=True)
def barrier_sim_kernel(entry, high, low, tp_w, sl_w, timeout, side):
    """Per-event TP/SL/timeout sim. Heavy O(E*H): parallel over events.

    Matches simulate_barriers semantics: side-aware barriers, stop wins ties,
    timeout after timeout bars, exit = last available entry when timed out.
    """
    n = entry.shape[0]
    outcome = np.empty(n, dtype=np.int64)
    exit_px = np.empty(n, dtype=np.float64)
    bars = np.empty(n, dtype=np.int64)
    for e in prange(n):
        if not (np.isfinite(entry[e]) and np.isfinite(tp_w[e]) and np.isfinite(sl_w[e])):
            outcome[e] = 0
            exit_px[e] = np.nan
            bars[e] = 0
            continue
        s = side[e]
        p = entry[e]
        tp = p * (1.0 + tp_w[e]) if s > 0 else p * (1.0 - tp_w[e])
        sl = p * (1.0 - sl_w[e]) if s > 0 else p * (1.0 + sl_w[e])
        hit = 0
        xp = p
        lim = e + timeout
        if lim > n:
            lim = n
        for t in range(e, lim):
            h = high[t]
            lo = low[t]
            if s > 0:
                sl_hit = np.isfinite(lo) and lo <= sl
                tp_hit = np.isfinite(h) and h >= tp
            else:
                sl_hit = np.isfinite(h) and h >= sl
                tp_hit = np.isfinite(lo) and lo <= tp
            if sl_hit:
                hit = -1
                xp = sl
                break
            if tp_hit:
                hit = 1
                xp = tp
                break
        if hit == 0:
            xp = entry[lim - 1] if (lim - 1 >= e and np.isfinite(entry[lim - 1])) else p
        outcome[e] = hit
        exit_px[e] = xp
        bars[e] = lim - e
    return outcome, exit_px, bars



@njit(cache=True, nogil=True, parallel=True)
def excursion_kernel(entry, high, low, timeout, side):
    """Per-event max favorable/adverse excursion fractions. Heavy: parallel.

    Matches forward_excursions semantics: future path is bars e+1..e+H
    (entry bar excluded); tail uses whatever future bars remain.
    """
    n = entry.shape[0]
    mfe = np.empty(n, dtype=np.float64)
    mae = np.empty(n, dtype=np.float64)
    for e in prange(n):
        p = entry[e]
        if not np.isfinite(p) or p == 0.0:
            mfe[e] = np.nan
            mae[e] = np.nan
            continue
        best_up = 0.0
        best_dn = 0.0
        lim = e + timeout + 1
        if lim > n:
            lim = n
        for t in range(e + 1, lim):
            h = high[t]
            lo = low[t]
            if np.isfinite(h):
                r = (h - p) / p
                if r > best_up:
                    best_up = r
            if np.isfinite(lo):
                r = (lo - p) / p
                if r < best_dn:
                    best_dn = r
        if side[e] > 0:
            mfe[e] = best_up
            mae[e] = -best_dn
        else:
            mfe[e] = -best_dn
            mae[e] = best_up
    return mfe, mae


@njit(cache=True, nogil=True, parallel=True)
def elapsed_err_kernel(clock_ns, end_ns, resid, pos_idx):
    """Backward-asof gather: latest resid with end_ns <= clock. Heavy: parallel."""
    n = pos_idx.shape[0]
    m = end_ns.shape[0]
    out = np.empty(n, dtype=np.float64)
    for k in prange(n):
        t = clock_ns[pos_idx[k]]
        lo = 0
        hi = m - 1
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if end_ns[mid] <= t:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        out[k] = 0.0 if ans < 0 else (resid[ans] if np.isfinite(resid[ans]) else 0.0)
    return out


@njit(cache=True, nogil=True, parallel=True)
def barrier_outcome_kernel(high, low, side, tp_level, sl_level, timeout):
    """TP/SL/timeout outcome for EVERY row. Exact mirror of the vectorized
    O(n*H) k-loop in outcome/triple_barrier.simulate_barriers, early-exiting
    per row and parallel over rows.

    Contract (must stay bit-identical to the k-loop):
      - path = bars e+1 .. min(e+timeout, n-1); the entry bar is excluded;
      - stop is checked before TP on each bar (stop wins ties);
      - raw IEEE comparisons (NaN level or NaN bar never hits, +inf hits);
      - both levels NaN (side-0 / unscored rows) -> outcome 0 immediately;
      - outcome: +1 TP first, -1 SL first (or tie), 0 timeout/no event.
    """
    n = side.shape[0]
    outcome = np.empty(n, dtype=np.int64)
    for e in prange(n):
        sl = sl_level[e]
        tp = tp_level[e]
        if np.isnan(sl) and np.isnan(tp):
            # comparisons against NaN are always False in the k-loop too
            outcome[e] = 0
            continue
        s = side[e]
        hit = 0
        for k in range(1, timeout + 1):
            t = e + k
            if t >= n:
                break
            h = high[t]
            lo = low[t]
            if s > 0.0:
                sl_hit = lo <= sl
                tp_hit = h >= tp
            else:
                sl_hit = h >= sl
                tp_hit = lo <= tp
            if sl_hit:
                hit = -1
                break
            if tp_hit:
                hit = 1
                break
        outcome[e] = hit
    return outcome

