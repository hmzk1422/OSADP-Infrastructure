"""Precomputed float32 temporal features for GP. Build once, pointer-lookup in loop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .jit_kernels import diff_kernel, lag_kernel, rolling_mean_kernel, rolling_std_kernel


class TemporalFeatureCache:
    """Contiguous float32 base + precomputed (var, window) temporal variants."""

    def __init__(self, index, base: dict[str, np.ndarray], windows: tuple[int, ...]):
        self.index = index
        self.n = len(index)
        self.base = base
        self.arrays: dict[str, np.ndarray] = dict(base)
        self.windows = tuple(windows)

    def get(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    def prefix(self, n: int) -> "TemporalFeatureCache":
        """Zero-copy view over the first ``n`` rows.

        Every cached op is causal (lag/diff/rolling/window/block segment
        boundaries anchor at row 0), so the prefix of a full-frame build
        equals a build over the prefixed frame; slicing views adds no copy.
        """
        if not 0 <= n <= self.n:
            raise ValueError(f"prefix length must be in [0, {self.n}]; got {n}")
        if n == self.n:
            return self
        sub = TemporalFeatureCache(self.index[:n],
                                   {k: v[:n] for k, v in self.base.items()},
                                   self.windows)
        for k, v in self.arrays.items():
            if k not in sub.arrays:
                sub.arrays[k] = v[:n]
        return sub


def build_temporal_cache(inputs: pd.DataFrame, temporal_windows: tuple[int, ...] = (2, 4, 8, 16)) -> TemporalFeatureCache:
    """Precompute every (var, window) lag/rollmean/rollstd/diff/windowsum once."""
    wins = tuple(sorted(set(int(w) for w in temporal_windows)))
    if any(w <= 0 for w in wins):
        raise ValueError(f"temporal_windows must be positive; got {temporal_windows!r}")
    import polars as pl

    base: dict[str, np.ndarray] = {}
    try:
        plf = pl.from_pandas(pd.DataFrame(inputs))
    except Exception:
        plf = None
    if plf is not None:
        for c in plf.columns:
            try:
                col = plf[c].cast(pl.Float32, strict=False).to_numpy()
            except Exception:
                col = pd.to_numeric(inputs[c], errors="coerce").to_numpy(dtype=np.float32)
            base[str(c)] = np.ascontiguousarray(np.asarray(col, dtype=np.float32).reshape(-1))
    else:
        for c in list(inputs.columns):
            col = pd.to_numeric(inputs[c], errors="coerce").to_numpy(dtype=np.float32)
            base[str(c)] = np.ascontiguousarray(col.reshape(-1))
    cache = TemporalFeatureCache(inputs.index, base, wins)
    for name, arr in list(base.items()):
        xf = np.ascontiguousarray(arr.astype(np.float32, copy=False))
        for w in wins:
            cache.arrays["__lag__" + name + "__" + str(w)] = lag_kernel(xf, int(w))
            cache.arrays["__rollmean__" + name + "__" + str(w)] = rolling_mean_kernel(xf, int(w))
            cache.arrays["__rollstd__" + name + "__" + str(w)] = rolling_std_kernel(xf, int(w))
            cache.arrays["__diff__" + name + "__" + str(w)] = diff_kernel(xf, int(w))
            f = arr.astype(np.float64)
            n = len(f)
            ws = np.empty(n, dtype=np.float32)
            for i in range(n):
                if i + 1 < w:
                    ws[i] = np.nan
                    continue
                seg = f[i - w + 1: i + 1]
                ws[i] = np.float32(seg.sum()) if np.isfinite(seg).all() else np.nan
            cache.arrays["__windowsum__" + name + "__" + str(w)] = ws
    return cache



def evaluate_expression_cached(node, cache: TemporalFeatureCache, index: pd.Index) -> pd.Series:
    """Numpy evaluator with cache pointer-lookup; kernels for computed children."""
    from . import grammar as _g

    def ev(nd) -> np.ndarray:
        op = nd.op
        if op == "var":
            return cache.get(str(nd.value)).astype(np.float64, copy=False)
        if op == "const":
            return np.full(cache.n, float(nd.value), dtype=np.float64)
        if op == "partial":
            tmp = pd.DataFrame({k: cache.arrays[k] for k in cache.base.keys()}, index=index)
            return _g.evaluate(nd.children[0], tmp, str(nd.value))[1].to_numpy(dtype=np.float64)
        ch = [ev(c) for c in nd.children]
        x = ch[0]
        if op in ("lag", "diff", "derivative", "time_derivative"):
            lag = int(nd.value)
            if lag <= 0:
                raise ValueError(f"lag must be positive; got {lag!r}")
            b0 = nd.children[0]
            if b0.op == "var" and op in ("lag", "diff"):
                key = ("__lag__" if op == "lag" else "__diff__") + str(b0.value) + "__" + str(lag)
                if key in cache:
                    return cache.get(key).astype(np.float64)
            n = len(x)
            out = np.empty(n, dtype=np.float64)
            out[:lag] = np.nan
            if op == "lag":
                out[lag:] = x[:-lag]
            else:
                out[lag:] = x[lag:] - x[:-lag]
                if op in ("derivative", "time_derivative"):
                    out[lag:] = out[lag:] / float(lag)
            return _g._bounded(pd.Series(out, index=index)).to_numpy(dtype=np.float64)
        if op in ("rolling_mean", "window", "rolling_std", "block"):
            w = int(nd.value)
            if w <= 0:
                raise ValueError(f"window must be positive; got {w!r}")
            b0 = nd.children[0]
            if b0.op == "var":
                km = {"rolling_mean": "__rollmean__", "window": "__windowsum__", "rolling_std": "__rollstd__"}
                if op in km:
                    key = km[op] + str(b0.value) + "__" + str(w)
                    if key in cache:
                        return cache.get(key).astype(np.float64)
            xf = np.ascontiguousarray(x.astype(np.float32))
            if op == "rolling_mean":
                return rolling_mean_kernel(xf, w).astype(np.float64)
            if op == "window":
                f = x.astype(np.float64)
                n = len(f)
                ws = np.empty(n, dtype=np.float64)
                for i in range(n):
                    if i + 1 < w:
                        ws[i] = np.nan
                        continue
                    seg = f[i - w + 1: i + 1]
                    ws[i] = seg.sum() if np.isfinite(seg).all() else np.nan
                return ws
            if op == "rolling_std":
                return rolling_std_kernel(xf, w).astype(np.float64)
            n = len(x)
            out = np.empty(n, dtype=np.float64)
            for s in range(0, n, w):
                e = min(s + w, n)
                seg = x[s:e]
                cs = np.cumsum(np.where(np.isfinite(seg), seg, 0.0))
                cn = np.cumsum(np.isfinite(seg).astype(np.int64))
                out[s:e] = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)
            return out
        if op == "neg":
            return -x
        if op == "abs":
            return np.abs(x)
        if op == "signed_log":
            return np.sign(x) * np.log1p(np.abs(x))
        if op == "log":
            return _g._bounded(pd.Series(np.log(np.clip(np.abs(x), 1e-12, None)), index=index)).to_numpy(dtype=np.float64)
        if op == "sqrt":
            return _g._bounded(pd.Series(np.sqrt(np.abs(x)), index=index)).to_numpy(dtype=np.float64)
        if op == "signed_sqrt":
            return _g._bounded(pd.Series(np.sign(x) * np.sqrt(np.abs(x)), index=index)).to_numpy(dtype=np.float64)
        y = ch[1]
        if op == "add":
            return _g._bounded(pd.Series(x + y, index=index)).to_numpy(dtype=np.float64)
        if op == "sub":
            return _g._bounded(pd.Series(x - y, index=index)).to_numpy(dtype=np.float64)
        if op == "mul":
            return _g._bounded(pd.Series(x * y, index=index)).to_numpy(dtype=np.float64)
        if op == "div":
            den = np.where(np.abs(y) >= 1e-12, y, np.nan)
            return _g._bounded(pd.Series(x / den, index=index)).to_numpy(dtype=np.float64)
        if op == "power":
            e = np.clip(np.round(y), -4, 4)
            with np.errstate(all="ignore"):
                return _g._bounded(pd.Series(np.power(x, e), index=index)).to_numpy(dtype=np.float64)
        raise ValueError("unknown GP operator " + repr(op))

    vals = ev(node)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    vals = np.clip(vals, -1e6, 1e6)
    return pd.Series(vals, index=index)
