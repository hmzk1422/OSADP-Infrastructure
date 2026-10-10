"""Numba kernels that replace the per-row Python loops in the regime pipeline.

The kernels reproduce the original NumPy/pandas arithmetic operation for
operation, and on the configurations tested (see the differential tests) the
segmenter output and end state are *exactly* equal to the pure-Python
reference, while the encoder matches exactly except ``slope``, which agrees to
~1e-15 (the original uses a BLAS gemv whose summation order is not
reproducible).  Treat this as "numerically equivalent on the tested NumPy
build", not a guarantee of bit-identity on every platform:

* Reductions that NumPy runs as a 1-D pairwise sum (``np.mean`` of a vector)
  use ``_pairwise_sum``, a port of NumPy's algorithm.  A NumPy build with a
  different reduction path (other SIMD dispatch / version) could differ in
  the last bits.
* Reductions NumPy runs row-by-row over ``axis=0`` are sequential loops.
* ``fastmath`` is never enabled, so LLVM may not reassociate floating point.
* ``exp`` is *not* evaluated in-kernel: kernels return its argument and the
  caller applies ``np.exp`` so the same implementation is used as before.

The pure-Python methods remain in the segmenter/encoder as the reference
(``_predict_custom_python`` etc.); re-run the differential tests when you
change NumPy/numba/CPU.  If numba is missing, ``NUMBA_AVAILABLE`` is False and
callers keep using the original pure-Python paths.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba optional
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func


# Status codes shared with the segmenter wrapper.
ST_WARMUP, ST_KNOWN, ST_LEARNING_DISABLED, ST_MERGED, ST_PROMOTED, ST_CAPACITY, ST_LEARNING = range(7)
STATUS_NAMES = ("warmup", "known", "learning_disabled", "merged", "promoted", "capacity", "learning")


@njit(cache=True)
def _pairwise_leaf(a, lo, n):
    """Blocks of at most 128 elements, exactly as NumPy's ``pairwise_sum``."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = 8
    limit = n - (n % 8)
    while i < limit:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(cache=True)
def _pairwise_sum(a, lo, n, wi, wf):
    """Port of NumPy's ``pairwise_sum`` (what ``np.add.reduce`` does on 1-D).

    NumPy recurses for n > 128 (split at ``n//2`` rounded down to a multiple of
    8).  numba's LLVM backend aborts on this recursion when caching, so it is
    run with an explicit stack instead; ``wi`` (64, 3) int64 and ``wf`` (64,)
    float64 are caller-owned scratch so the hot path never allocates.
    """
    if n <= 128:
        return _pairwise_leaf(a, lo, n)
    sp = 0
    wi[0, 0] = lo
    wi[0, 1] = n
    wi[0, 2] = 0
    ret = 0.0
    while sp >= 0:
        cl = wi[sp, 0]
        cn = wi[sp, 1]
        if cn <= 128:
            ret = _pairwise_leaf(a, cl, cn)
            sp -= 1
        else:
            n2 = cn // 2
            n2 -= n2 % 8
            stage = wi[sp, 2]
            if stage == 0:
                wi[sp, 2] = 1
                sp += 1
                wi[sp, 0] = cl
                wi[sp, 1] = n2
                wi[sp, 2] = 0
            elif stage == 1:
                wf[sp] = ret
                wi[sp, 2] = 2
                sp += 1
                wi[sp, 0] = cl + n2
                wi[sp, 1] = cn - n2
                wi[sp, 2] = 0
            else:
                ret = wf[sp] + ret
                sp -= 1
    return ret


# --------------------------------------------------------------------------- #
# TemporalWindowEncoder
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def temporal_window_kernel(matrix, slow, rows, starts, ends, centered, centered_norm, window,
                           pairwise_bins, pairwise_disp, out):
    """Fill ``out[row]`` for every complete row in ``rows``.

    Layout per row: ``bins`` ordered bin means, dispersion, slope, slow level
    (each ``n_features`` wide), identical to the original ``np.concatenate``.
    Rows are independent, so they run in parallel without changing any value.

    NumPy sums along ``axis=0`` sequentially when the slice is C-ordered but
    pairwise when the reduced axis is contiguous (F-ordered ``to_numpy()``
    output, or a single feature).  ``pairwise_bins`` / ``pairwise_disp`` select
    whichever order the original expression used so the sums stay bit-exact.
    """
    n_features = matrix.shape[1]
    n_bins = starts.shape[0]
    for k in prange(rows.shape[0]):
        row = rows[k]
        base = row - window + 1
        buf = np.empty(window)
        wi = np.empty((64, 3), np.int64)
        wf = np.empty(64)
        offset = 0
        for b in range(n_bins):
            first = base + starts[b]
            last = base + ends[b]
            count = ends[b] - starts[b]
            for f in range(n_features):
                if pairwise_bins:
                    for r in range(count):
                        buf[r] = matrix[first + r, f]
                    total = _pairwise_sum(buf, 0, count, wi, wf)
                else:
                    total = matrix[first, f]
                    for r in range(first + 1, last):
                        total += matrix[r, f]
                out[row, offset + f] = total / count
            offset += n_features
        for f in range(n_features):                       # dispersion (ddof=0)
            if pairwise_disp:
                for r in range(window):
                    buf[r] = matrix[base + r, f]
                mean = _pairwise_sum(buf, 0, window, wi, wf) / window
                for r in range(window):
                    delta = buf[r] - mean
                    buf[r] = delta * delta
                out[row, offset + f] = np.sqrt(_pairwise_sum(buf, 0, window, wi, wf) / window)
            else:
                total = matrix[base, f]
                for r in range(base + 1, base + window):
                    total += matrix[r, f]
                mean = total / window
                delta = matrix[base, f] - mean
                acc = delta * delta
                for r in range(base + 1, base + window):
                    delta = matrix[r, f] - mean
                    acc += delta * delta
                out[row, offset + f] = np.sqrt(acc / window)
        offset += n_features
        for f in range(n_features):                       # slope
            total = centered[0] * matrix[base, f]
            for r in range(1, window):
                total += centered[r] * matrix[base + r, f]
            out[row, offset + f] = total / centered_norm
        offset += n_features
        for f in range(n_features):                       # slow level
            out[row, offset + f] = slow[row, f]


# --------------------------------------------------------------------------- #
# HysteresisClusterRegimeSegmenter
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _rms_distance(value, center, scale, use_scale, floor, tmp, wi, wf):
    """``sqrt(mean(((value-center)/max(scale,floor))**2))`` as in ``_distance``."""
    size = value.shape[0]
    for j in range(size):
        if use_scale:
            s = scale[j]
            if s < floor:
                s = floor
            t = (value[j] - center[j]) / s
        else:
            t = value[j] - center[j]
        tmp[j] = t * t
    return np.sqrt(_pairwise_sum(tmp, 0, size, wi, wf) / size)


@njit(cache=True)
def _drop_state(k, nk, P, S, M2, SC, CNT, LAST, UID):
    for r in range(k, nk - 1):
        P[r] = P[r + 1]
        S[r] = S[r + 1]
        M2[r] = M2[r + 1]
        SC[r] = SC[r + 1]
        CNT[r] = CNT[r + 1]
        LAST[r] = LAST[r + 1]
        UID[r] = UID[r + 1]


@njit(cache=True)
def _drop_candidate(k, nc, CC, CM2, CCNT, CEP, CLAST):
    for r in range(k, nc - 1):
        CC[r] = CC[r + 1]
        CM2[r] = CM2[r + 1]
        CCNT[r] = CCNT[r + 1]
        CEP[r] = CEP[r + 1]
        CLAST[r] = CLAST[r + 1]


@njit(cache=True)
def segment_chunk(values, start, stop, pos0,
                  P, S, M2, SC, CNT, LAST, UID,
                  CC, CM2, CCNT, CEP, CLAST,
                  ints, cf, ci,
                  uid_out, sim_arg, dist_out, cand_count, cand_eps, support, status, known_count,
                  pruned_log):
    """The ``update_state`` state machine over ``values[start:stop]``.

    State arrays are mutated in place, so a run can be split into chunks.
    ``ints`` = [n_states, n_candidates, next_uid, current_uid (-1 = None),
    n_pruned, n_promoted].  Prototype/candidate slots keep dict/list *order*
    (removal shifts down, promotion appends) so tie-breaking is unchanged.
    """
    nk = ints[0]
    nc = ints[1]
    next_uid = ints[2]
    cur = ints[3]
    n_pruned = ints[4]
    n_promoted = ints[5]

    match_thr = cf[0]
    cand_thr = cf[1]
    margin = cf[2]
    rate = cf[3]
    floor = cf[4]
    max_regimes = ci[0]
    min_obs = ci[1]
    min_eps = ci[2]
    gap = ci[3]
    max_cand = ci[4]
    cand_expiry = ci[5]
    state_expiry = ci[6]
    learn = ci[7]

    size = values.shape[1]
    tmp = np.empty(size)
    dummy = np.empty(size)
    wi = np.empty((64, 3), np.int64)
    wf = np.empty(64)

    for i in range(start, stop):
        position = pos0 + i
        value = values[i]

        # -- _prune_states ----------------------------------------------------
        w = 0
        for r in range(nk):
            if position - LAST[r] > state_expiry:
                pruned_log[n_pruned] = UID[r]
                n_pruned += 1
            else:
                if w != r:
                    P[w] = P[r]
                    S[w] = S[r]
                    M2[w] = M2[r]
                    SC[w] = SC[r]
                    CNT[w] = CNT[r]
                    LAST[w] = LAST[r]
                    UID[w] = UID[r]
                w += 1
        nk = w
        if cur >= 0:
            found = False
            for r in range(nk):
                if UID[r] == cur:
                    found = True
                    break
            if not found:
                cur = -1

        # -- warm-up (non-finite) row -------------------------------------------
        finite = True
        for j in range(size):
            if not np.isfinite(value[j]):
                finite = False
                break
        if not finite:
            uid_out[i] = -1
            sim_arg[i] = -np.inf
            dist_out[i] = np.nan
            cand_count[i] = 0
            cand_eps[i] = 0
            support[i] = 0
            status[i] = ST_WARMUP
            known_count[i] = nk
            continue

        # -- assign_regime ----------------------------------------------------
        best = -1
        best_d = np.inf
        for r in range(nk):
            d = _rms_distance(value, P[r], S[r], True, floor, tmp, wi, wf)
            if best == -1 or d < best_d:
                best = r
                best_d = d
        allowed = match_thr
        if best >= 0 and UID[best] == cur:
            allowed += margin
        elif best < 0 and cur < 0:
            allowed += margin

        if best >= 0 and best_d <= allowed:
            # ---------------- known state: update prototype ------------------
            CNT[best] += 1
            for j in range(size):
                residual = value[j] - P[best, j]
                updated = P[best, j] + rate * residual
                adjusted = value[j] - updated
                M2[best, j] += residual * adjusted
                P[best, j] = updated
            SC[best] += 1
            denom = SC[best] - 1
            if denom < 1:
                denom = 1
            for j in range(size):
                m = M2[best, j]
                if m < 0.0:
                    m = 0.0
                v = np.sqrt(m / denom)
                if v < floor:
                    v = floor
                S[best, j] = v
            LAST[best] = position
            cur = UID[best]
            uid_out[i] = cur
            denom_allowed = allowed if allowed > 1e-12 else 1e-12
            sim_arg[i] = -best_d / denom_allowed
            dist_out[i] = best_d
            cand_count[i] = 0
            cand_eps[i] = 0
            support[i] = CNT[best]
            status[i] = ST_KNOWN
            known_count[i] = nk
            continue

        # ---------------- novel observation ------------------------------------
        if np.isfinite(best_d):
            sim_arg[i] = -(best_d if best_d > 0.0 else 0.0) / match_thr
            dist_out[i] = best_d
        else:
            sim_arg[i] = -np.inf
            dist_out[i] = np.nan

        count = 0
        episodes = 0
        code = ST_LEARNING_DISABLED
        if learn:
            # expire stale candidates (order preserved)
            w = 0
            for r in range(nc):
                if position - CLAST[r] <= cand_expiry:
                    if w != r:
                        CC[w] = CC[r]
                        CM2[w] = CM2[r]
                        CCNT[w] = CCNT[r]
                        CEP[w] = CEP[r]
                        CLAST[w] = CLAST[r]
                    w += 1
            nc = w
            idx = -1
            dist = np.inf
            for r in range(nc):
                d = _rms_distance(value, CC[r], dummy, False, floor, tmp, wi, wf)
                if np.isfinite(d):
                    if idx == -1 or d < dist:
                        idx = r
                        dist = d
            if idx == -1 or dist > cand_thr:
                if nc >= max_cand:
                    oldest = 0
                    for r in range(1, nc):
                        if CLAST[r] < CLAST[oldest]:
                            oldest = r
                    _drop_candidate(oldest, nc, CC, CM2, CCNT, CEP, CLAST)
                    nc -= 1
                for j in range(size):
                    CC[nc, j] = value[j]
                    CM2[nc, j] = 0.0
                CCNT[nc] = 1
                CEP[nc] = 1
                CLAST[nc] = position
                idx = nc
                nc += 1
            else:
                if position - CLAST[idx] > gap:
                    CEP[idx] += 1
                CCNT[idx] += 1
                cnt = CCNT[idx]
                for j in range(size):
                    delta = value[j] - CC[idx, j]
                    CC[idx, j] += delta / cnt
                    CM2[idx, j] += delta * (value[j] - CC[idx, j])
                CLAST[idx] = position
            count = CCNT[idx]
            episodes = CEP[idx]
            code = ST_LEARNING
            if count >= min_obs and episodes >= min_eps:
                merged = False
                if nk > 0:
                    nearest = -1
                    nearest_d = np.inf
                    for r in range(nk):
                        d = _rms_distance(CC[idx], P[r], S[r], True, floor, tmp, wi, wf)
                        if nearest == -1 or d < nearest_d:
                            nearest = r
                            nearest_d = d
                    if nearest_d <= match_thr:
                        merged = True
                if merged:
                    _drop_candidate(idx, nc, CC, CM2, CCNT, CEP, CLAST)
                    nc -= 1
                    code = ST_MERGED
                elif nk >= max_regimes:
                    code = ST_CAPACITY
                else:
                    pdenom = count - 1
                    if pdenom < 1:
                        pdenom = 1
                    for j in range(size):
                        P[nk, j] = CC[idx, j]
                        var = CM2[idx, j] / pdenom
                        if var < 0.0:
                            var = 0.0
                        v = np.sqrt(var)
                        if v < floor:
                            v = floor
                        S[nk, j] = v
                        M2[nk, j] = CM2[idx, j]
                    SC[nk] = count
                    CNT[nk] = 0
                    LAST[nk] = CLAST[idx]
                    UID[nk] = next_uid
                    next_uid += 1
                    nk += 1
                    n_promoted += 1
                    _drop_candidate(idx, nc, CC, CM2, CCNT, CEP, CLAST)
                    nc -= 1
                    code = ST_PROMOTED
        cur = -1
        uid_out[i] = -1
        cand_count[i] = count
        cand_eps[i] = episodes
        support[i] = 0
        status[i] = code
        known_count[i] = nk

    ints[0] = nk
    ints[1] = nc
    ints[2] = next_uid
    ints[3] = cur
    ints[4] = n_pruned
    ints[5] = n_promoted


@njit(cache=True)
def persistence_kernel(codes, state, min_dwell, out, pending_rows):
    """Causal min-dwell debounce on integer label codes.

    ``state`` = [persistent_code, pending_code (-1 = None), pending_count].
    """
    persistent = state[0]
    pending = state[1]
    pending_count = state[2]
    for i in range(codes.shape[0]):
        label = codes[i]
        if label == persistent:
            pending = -1
            pending_count = 0
        elif label == pending:
            pending_count += 1
            if pending_count >= min_dwell:
                persistent = label
                pending = -1
                pending_count = 0
        else:
            pending = label
            pending_count = 1
            if min_dwell <= 1:
                persistent = label
                pending = -1
                pending_count = 0
        out[i] = persistent
        pending_rows[i] = label != persistent
    state[0] = persistent
    state[1] = pending
    state[2] = pending_count