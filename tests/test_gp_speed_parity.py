"""Parity tests for gp_speed_fixes CHECK flags (part 1: kernels)."""
import numpy as np
import pandas as pd


def test_spearman_ties_matches_scipy():
    from scipy.stats import spearmanr
    from strategy_system.strategy.moe.gp.jit_kernels import spearman_kernel
    rng = np.random.default_rng(7)
    a = np.round(rng.normal(size=2000) * 2) / 2
    b = np.round(2 * a + rng.normal(size=2000) * 0.5)
    rho, cnt = spearman_kernel(a, b)
    ref, _ = spearmanr(a, b)
    assert cnt == 2000
    assert abs(rho - ref) < 1e-9, (rho, ref)


def test_spearman_nan_handling_matches():
    from strategy_system.strategy.moe.gp.jit_kernels import spearman_kernel
    a = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    b = np.array([2.0, 1.0, np.nan, 4.0, 5.0])
    rho, cnt = spearman_kernel(a, b)
    assert cnt == 2
    assert abs(rho - 1.0) < 1e-12


def test_spearman_constant_returns_nan():
    from strategy_system.strategy.moe.gp.jit_kernels import spearman_kernel
    rho, cnt = spearman_kernel(np.ones(100), np.arange(100, dtype=float))
    assert cnt == 100
    assert not np.isfinite(rho)


def test_blockwise_matches_per_block_scipy():
    from scipy.stats import spearmanr
    from strategy_system.strategy.moe.gp.jit_kernels import blockwise_spearman_kernel
    rng = np.random.default_rng(3)
    n = 1000
    v = rng.normal(size=n)
    t = v * 0.5 + rng.normal(size=n)
    out = blockwise_spearman_kernel(v, t, np.ones(n, bool), n, 4, 30)
    ref = []
    base, extra, s = n // 4, n % 4, 0
    for b in range(4):
        size = base + (1 if b < extra else 0)
        r, _ = spearmanr(v[s:s + size], t[s:s + size])
        ref.append(r)
        s += size
    assert len(out) == 4
    assert np.allclose(out, ref, atol=1e-9)


def test_batch_matches_serial():
    from strategy_system.strategy.moe.gp.jit_kernels import batch_spearman_kernel, spearman_kernel
    rng = np.random.default_rng(11)
    vals = rng.normal(size=(8, 500))
    tgts = vals * 0.7 + rng.normal(size=(8, 500)) * 0.3
    rhos, cnts = batch_spearman_kernel(vals, tgts, 30)
    for i in range(8):
        r, c = spearman_kernel(vals[i], tgts[i])
        assert cnts[i] == c
        assert (np.isnan(rhos[i]) and np.isnan(r)) or abs(rhos[i] - r) < 1e-12


def test_rolling_mean_matches_pandas():
    from strategy_system.strategy.moe.gp.jit_kernels import rolling_mean_kernel
    rng = np.random.default_rng(5)
    x = (rng.normal(size=500) * 10).astype(np.float32)
    x[::37] = np.nan
    for w in (2, 4, 8, 16):
        got = rolling_mean_kernel(x, w).astype(float)
        ref = pd.Series(x.astype(float)).rolling(w, min_periods=w).mean().to_numpy()
        assert np.allclose(got, ref, rtol=1e-6, atol=1e-6, equal_nan=True), w


def test_rolling_std_matches_pandas():
    from strategy_system.strategy.moe.gp.jit_kernels import rolling_std_kernel
    rng = np.random.default_rng(5)
    x = (rng.normal(size=500) * 10).astype(np.float32)
    x[::37] = np.nan
    for w in (2, 4, 8, 16):
        got = rolling_std_kernel(x, w).astype(float)
        ref = pd.Series(x.astype(float)).rolling(w, min_periods=w).std(ddof=0).to_numpy()
        assert np.allclose(got, ref, rtol=1e-5, atol=1e-4, equal_nan=True), w


def test_lag_diff_match_pandas():
    from strategy_system.strategy.moe.gp.jit_kernels import lag_kernel, diff_kernel
    x = np.arange(20, dtype=np.float32)
    assert np.allclose(lag_kernel(x, 3).astype(float),
                       pd.Series(x.astype(float)).shift(3).to_numpy(), equal_nan=True)
    assert np.allclose(diff_kernel(x, 4).astype(float),
                       (pd.Series(x.astype(float)) - pd.Series(x.astype(float)).shift(4)).to_numpy(),
                       equal_nan=True)


def test_windowsum_nan_aware():
    from strategy_system.strategy.moe.gp.feature_cache import build_temporal_cache
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]})
    c = build_temporal_cache(df, (3,))
    got = c.get("__windowsum__a__3")
    ref = pd.Series(df["a"]).rolling(3, min_periods=3).sum().to_numpy(dtype=np.float32)
    assert np.allclose(got.astype(float), ref.astype(float), equal_nan=True)


def test_quantize_ties_documented():
    """Ties follow Decimal ROUND_HALF_UP (pack CHECK 861), not numpy half-even."""
    from decimal import ROUND_HALF_UP, Decimal
    from strategy_system.strategy.moe.gp.jit_kernels import quantize_tick_kernel
    ties = np.array([100.005, 100.015, 100.025])
    got = quantize_tick_kernel(ties, 0.01)
    assert list(got) == [100.01, 100.02, 100.03]
    rng = np.random.default_rng(0)
    px = 100000 + rng.normal(size=10000) * 500
    ref = np.array([float(Decimal(str(float(v))).quantize(Decimal("0.01"),
                                                         rounding=ROUND_HALF_UP))
                    for v in px])
    assert np.array_equal(quantize_tick_kernel(px, 0.01), ref)


def test_barrier_sim_matches_python_loop():
    from strategy_system.strategy.moe.gp.jit_kernels import barrier_sim_kernel
    rng = np.random.default_rng(9)
    n, H = 500, 30
    entry = 100 + rng.normal(size=n).cumsum() * 0.05
    high = entry + np.abs(rng.normal(size=n)) * 0.3
    low = entry - np.abs(rng.normal(size=n)) * 0.3
    tp = np.full(n, 0.002)
    sl = np.full(n, 0.001)
    side = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    outcome, exit_px, bars = barrier_sim_kernel(entry, high, low, tp, sl, H, side)
    for e in range(0, n, 37):
        s, p = side[e], entry[e]
        tp_px = p * (1 + tp[e]) if s > 0 else p * (1 - tp[e])
        sl_px = p * (1 - sl[e]) if s > 0 else p * (1 + sl[e])
        hit, xp = 0, p
        for t in range(e, min(e + H, n)):
            if s > 0:
                s_hit, t_hit = low[t] <= sl_px, high[t] >= tp_px
            else:
                s_hit, t_hit = high[t] >= sl_px, low[t] <= tp_px
            if s_hit:
                hit, xp = -1, sl_px
                break
            if t_hit:
                hit, xp = 1, tp_px
                break
        if hit == 0:
            xp = entry[min(e + H, n) - 1]
        assert outcome[e] == hit
        assert abs(exit_px[e] - xp) < 1e-9
        assert bars[e] == min(e + H, n) - e


def test_excursion_matches_vectorized():
    from strategy_system.strategy.moe.gp.jit_kernels import excursion_kernel
    rng = np.random.default_rng(1)
    n, H = 2000, 30
    px = 100 + rng.normal(size=n).cumsum() * 0.1
    hi = px + np.abs(rng.normal(size=n)) * 0.2
    lo = px - np.abs(rng.normal(size=n)) * 0.2
    side = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    mfe, mae = excursion_kernel(px, hi, lo, H, side)
    max_hi = np.full(n, -np.inf)
    min_lo = np.full(n, np.inf)
    for k in range(1, H + 1):
        m = n - k
        max_hi[:m] = np.maximum(max_hi[:m], hi[k:])
        min_lo[:m] = np.minimum(min_lo[:m], lo[k:])
    ref_up = max_hi / px - 1.0
    ref_dn = 1.0 - min_lo / px
    ref_mfe = np.where(side > 0, ref_up, ref_dn)
    ref_mae = np.where(side > 0, ref_dn, ref_up)
    m = n - H
    assert np.allclose(mfe[:m], np.maximum(ref_mfe[:m], 0.0), atol=1e-9)
    assert np.allclose(mae[:m], np.maximum(ref_mae[:m], 0.0), atol=1e-9)


def test_elapsed_err_matches_merge_asof():
    from strategy_system.strategy.moe.gp.jit_kernels import elapsed_err_kernel
    rng = np.random.default_rng(4)
    n = 2000
    idx = pd.date_range("2025-01-01", periods=n, freq="100ms", tz="UTC")
    clock = idx.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    is_pred = np.zeros(n, bool)
    is_pred[::7] = True
    resid = np.where(is_pred, rng.normal(size=n), np.nan)
    end = clock + 30 * 100_000_000
    ok = is_pred & np.isfinite(resid)
    order = np.argsort(end[ok], kind="mergesort")
    end_s = np.ascontiguousarray(end[ok][order])
    resid_s = np.ascontiguousarray(resid[ok][order])
    pos = np.ascontiguousarray(np.arange(n).astype(np.int64))
    got = elapsed_err_kernel(np.ascontiguousarray(clock), end_s, resid_s, pos)
    known = pd.DataFrame({"_endpoint": pd.to_datetime(end[ok], utc=True).as_unit("ns"),
                          "_resid": resid[ok]}).sort_values("_endpoint", kind="mergesort")
    left = pd.DataFrame({"_time": idx.astype("datetime64[ns, UTC]")}).sort_values(
        "_time", kind="mergesort")
    ref = pd.merge_asof(left, known, left_on="_time", right_on="_endpoint",
                        direction="backward")["_resid"].to_numpy(float)
    ref = np.where(np.isfinite(ref), ref, 0.0)
    assert np.allclose(got, ref, atol=1e-12)


def test_gp_cache_matches_pandas_eval():
    from strategy_system.strategy.moe.gp.grammar import Node, evaluate_expression
    from strategy_system.strategy.moe.gp.feature_cache import (
        build_temporal_cache, evaluate_expression_cached)
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"a": rng.normal(size=n) * 10, "b": rng.normal(size=n) + 5})
    cache = build_temporal_cache(df, (2, 4, 8))
    lag_a = Node("lag", (Node("var", (), "a"),), 2)
    mul_b = Node("mul", (Node("var", (), "b"), Node("const", (), 2.0)))
    win_a = Node("window", (Node("var", (), "a"),), 4)
    diff_b = Node("diff", (Node("var", (), "b"),), 2)
    trees = [
        Node("rolling_mean", (Node("var", (), "a"),), 4),
        Node("rolling_std", (Node("var", (), "b"),), 8),
        Node("add", (lag_a, mul_b)),
        Node("div", (win_a, diff_b)),
    ]
    for t in trees:
        o = evaluate_expression(t, df).to_numpy(float)
        f = evaluate_expression_cached(t, cache, df.index).to_numpy(float)
        assert (np.isnan(o) == np.isnan(f)).all(), t
        if str(t).startswith("(window"):
            # div by near-zero diff amplifies float32 cache ULP (design bound):
            # assert tight parity only where the denominator is well-conditioned
            den = evaluate_expression(t.children[1], df).to_numpy(float)
            good = ~np.isnan(o) & (np.abs(den) > 0.05)
            assert np.allclose(o[good], f[good], rtol=1e-4, atol=1e-3), t
        else:
            assert np.allclose(o[~np.isnan(o)], f[~np.isnan(o)], rtol=1e-5, atol=1e-6), t


# ===================== part 2: fix-pack integration parity =======================

import pytest


def _decimal_ref(v, tick):
    """Reference: exact Decimal(str(v)).quantize(tick, ROUND_HALF_UP) (pack CHECK 861)."""
    import math
    from decimal import ROUND_HALF_UP, Decimal
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return v
    return float(Decimal(str(f)).quantize(Decimal(str(tick)), rounding=ROUND_HALF_UP))


def _quantize_values():
    rng = np.random.default_rng(17)
    return np.concatenate([
        np.round(rng.uniform(0.01, 200000, 20000), 6),
        np.round(-rng.uniform(0.01, 200000, 5000), 6),
        np.array([0.005, -0.005, 100.005, -100.005, 100.015, 123456.785,
                  -123456.785, 0.004999, 100.004999, 100.005001,
                  108613.205, -108613.205, 1.0, -1.0, 0.0, 1e-9, 108613.2]),
        np.array([np.nan, np.inf, -np.inf]),
    ])


def test_quantize_tick_kernel_matches_decimal_round_half_up():
    from strategy_system.strategy.moe.gp.jit_kernels import quantize_tick_kernel
    vals = _quantize_values()
    for tick in (0.01, 0.1, 1.0, 0.5, 0.25):
        got = quantize_tick_kernel(vals, tick)
        ref = np.array([_decimal_ref(v, tick) for v in vals], dtype=np.float64)
        finite = np.isfinite(vals)
        assert np.array_equal(np.isnan(got), np.isnan(ref)), tick
        assert np.array_equal(got[finite], ref[finite]), tick


def test_quantize_tick_scalar_matches_kernel_and_decimal():
    from strategy_system.strategy.moe.gp.jit_kernels import (
        quantize_tick_kernel, quantize_tick_scalar)
    vals = _quantize_values()[::7]
    for tick in (0.01, 0.5, 1.0):
        ker = quantize_tick_kernel(vals, tick)
        for v, kv in zip(vals, ker):
            s = quantize_tick_scalar(float(v), tick)
            if np.isnan(kv):
                assert np.isnan(s), (v, tick)
            else:
                assert s == kv == _decimal_ref(float(v), tick), (v, tick)
    assert quantize_tick_scalar("abc", 0.01) == "abc"
    assert quantize_tick_scalar(None, 0.01) is None


def test_loader_quantize_price_matches_decimal():
    from strategy_system.exchange.loader import quantize_price
    vals = _quantize_values()
    ref = np.array([_decimal_ref(v, 0.01) for v in vals], dtype=np.float64)
    out = quantize_price(pd.Series(vals), 0.01)
    assert isinstance(out, pd.Series)
    fin = np.isfinite(vals)
    assert np.array_equal(np.isnan(out.to_numpy()), np.isnan(ref))
    assert np.array_equal(out.to_numpy()[fin], ref[fin])
    lst = quantize_price(list(vals[:500]), 0.01)
    assert np.array_equal(np.asarray(lst, dtype=float), ref[:500], equal_nan=True)
    assert quantize_price(100.005, 0.01) == 100.01
    assert quantize_price(-100.005, 0.01) == -100.01
    assert np.isnan(quantize_price(float("nan"), 0.01))
    assert quantize_price("abc", 0.01) == "abc"


def test_triple_barrier_quantize_price_matches_decimal():
    from strategy_system.strategy.moe.outcome.triple_barrier import quantize_price
    vals = _quantize_values()
    ref = np.array([_decimal_ref(v, 0.01) for v in vals], dtype=np.float64)
    fin = np.isfinite(vals)
    out = quantize_price(pd.Series(vals), 0.01)
    assert np.array_equal(out.to_numpy()[fin], ref[fin])
    arr = quantize_price(vals, 0.01)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr[fin], ref[fin])
    assert quantize_price(100.005, 0.01) == _decimal_ref(100.005, 0.01)
    assert quantize_price("abc", 0.01) == "abc"


def test_elapsed_errors_matches_legacy_pandas():
    from strategy_system.strategy.moe.outcome.tb_stage import (
        _elapsed_errors_pandas, elapsed_errors)
    rng = np.random.default_rng(9)
    n = 3000
    idx = pd.date_range("2025-03-01", periods=n, freq="100ms", tz="UTC")
    states = pd.Series(np.where(rng.integers(0, 2, n) == 0, "alpha", "beta"))
    realized = pd.Series(rng.normal(size=n))
    predicted = pd.Series(rng.normal(size=n))
    endpoint = pd.Series(idx + pd.to_timedelta(rng.integers(0, 300, n), unit="s"))
    endpoint.iloc[::23] = pd.NaT
    predicted.iloc[::17] = np.nan
    predicted.iloc[5] = np.inf
    realized.iloc[::29] = np.nan
    for st in ("alpha", "beta", "missing"):
        got = elapsed_errors(idx, states, realized, predicted, endpoint, st)
        ref = _elapsed_errors_pandas(idx, states, realized, predicted, endpoint, st)
        assert got.shape == (n,)
        assert np.allclose(got, ref, atol=1e-12, equal_nan=True), st
    # non-datetime index -> zeros on both paths
    ridx = pd.RangeIndex(n)
    assert not elapsed_errors(ridx, states, realized, predicted, endpoint, "alpha").any()
    # causality: the first row of a state cannot see any residual (no same-state
    # endpoint exists at or before it)
    first_a = int(np.flatnonzero((states == "alpha").to_numpy())[0])
    assert elapsed_errors(idx, states, realized, predicted, endpoint, "alpha")[first_a] == 0.0


def test_spearman_wrapper_matches_pandas_reference():
    from strategy_system.strategy.moe.gp.evolution import _spearman
    rng = np.random.default_rng(21)
    n = 1500
    a = np.round(rng.normal(size=n) * 3) / 3.0
    b = np.round(rng.normal(size=n))
    a[::13] = np.nan
    b[::17] = np.inf
    valid = np.isfinite(a) & np.isfinite(b)
    x = pd.Series(a[valid]).rank(method="average").to_numpy(dtype=np.float64)
    y = pd.Series(b[valid]).rank(method="average").to_numpy(dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    ref = float(np.dot(x, y) / denom) if denom > 1e-15 else np.nan
    rho, cnt = _spearman(a, b, 30)
    assert cnt == int(valid.sum())
    assert (np.isnan(rho) and np.isnan(ref)) or abs(rho - ref) < 1e-9
    rho2, cnt2 = _spearman(a, b, int(valid.sum()) + 5)
    assert np.isnan(rho2) and cnt2 == int(valid.sum())


def test_blockwise_wrapper_matches_legacy_blocks():
    from strategy_system.strategy.moe.gp.evolution import _blockwise_spearman
    rng = np.random.default_rng(23)
    n = 1200
    v = rng.normal(size=n)
    t = v * 0.6 + rng.normal(size=n) * 0.4
    t[::31] = np.nan
    mask = rng.random(n) > 0.25

    def legacy(values, target, train_mask, train_end, block_count, min_block_samples):
        corrs = []
        for positions in np.array_split(np.arange(train_end), block_count):
            m = train_mask[positions]
            if int(m.sum()) < min_block_samples:
                continue
            va, vb = values[positions][m], target[positions][m]
            ok = np.isfinite(va) & np.isfinite(vb)
            if int(ok.sum()) < min_block_samples:
                continue
            x = pd.Series(va[ok]).rank().to_numpy(dtype=np.float64)
            y = pd.Series(vb[ok]).rank().to_numpy(dtype=np.float64)
            x = x - x.mean()
            y = y - y.mean()
            d = np.sqrt(np.dot(x, x) * np.dot(y, y))
            if d <= 1e-15:
                continue
            r = float(np.dot(x, y) / d)
            if np.isfinite(r):
                corrs.append(r)
        return np.asarray(corrs, dtype=np.float64)

    got = _blockwise_spearman(v, t, mask, n, 5, 40)
    ref = legacy(v, t, mask, n, 5, 40)
    assert np.allclose(got, ref, atol=1e-9)


def test_cache_prefix_equals_rebuild():
    from strategy_system.strategy.moe.gp.feature_cache import build_temporal_cache
    rng = np.random.default_rng(29)
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n) * 10, "b": rng.normal(size=n)})
    df.iloc[::11, 0] = np.nan
    wins = (2, 4, 8)
    full = build_temporal_cache(df, wins)
    cut = 257
    view = full.prefix(cut)
    ref = build_temporal_cache(df.iloc[:cut].reset_index(drop=True), wins)
    assert view.n == cut
    assert set(view.arrays) == set(ref.arrays)
    for key, arr in ref.arrays.items():
        assert np.array_equal(view.get(key), arr, equal_nan=True), key
    with pytest.raises(ValueError):
        full.prefix(n + 1)
    with pytest.raises(ValueError):
        full.prefix(-1)


def test_constant_and_redundant_filters_cache_parity(monkeypatch):
    from strategy_system.strategy.moe.gp import population as pop_mod
    from strategy_system.strategy.moe.gp.feature_cache import build_temporal_cache
    from strategy_system.strategy.moe.gp.grammar import Node
    rng = np.random.default_rng(31)
    n = 600
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    mask = np.ones(n, dtype=bool)
    mask[::9] = False
    nodes = [
        Node("var", (), "a"),
        Node("rolling_mean", (Node("var", (), "a"),), 4),
        Node("const", (), 3.0),
        Node("add", (Node("var", (), "a"), Node("const", (), 0.0))),
        Node("mul", (Node("var", (), "a"), Node("const", (), 2.0))),
        Node("var", (), "b"),
    ]
    cache = build_temporal_cache(df, (4,))
    kept_const = pop_mod.filter_constant_features(nodes, df, sample_mask=mask, cache=cache)
    assert kept_const == [nodes[0], nodes[1], nodes[3], nodes[4], nodes[5]]
    kept_red = pop_mod.filter_redundant_features(
        nodes, df, threshold=0.98, min_samples=30, sample_mask=mask, cache=cache)
    # exact linear duplicates of 'a' drop out; the constant node stays here
    # (zero-variance never counts as a correlated duplicate; the constant
    # filter is the stage that removes it)
    assert kept_red == [nodes[0], nodes[1], nodes[2], nodes[5]]

    # cache build fails -> pure pandas evaluator must decide identically
    def _boom(*_a, **_k):
        raise RuntimeError("cache disabled in test")

    monkeypatch.setattr(
        "strategy_system.strategy.moe.gp.feature_cache.build_temporal_cache", _boom)
    pd_const = pop_mod.filter_constant_features(nodes, df, sample_mask=mask, cache=None)
    pd_red = pop_mod.filter_redundant_features(
        nodes, df, threshold=0.98, min_samples=30, sample_mask=mask, cache=None)
    assert pd_const == kept_const
    assert pd_red == kept_red


def test_entry_events_prefilter_validation():
    from strategy_system.strategy.moe.outcome.triple_barrier import _entry_events
    n = 50
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    data = pd.DataFrame({"px": np.full(n, 100.0), "sig": np.zeros(n)}, index=idx)
    kw = dict(signal_col="sig", price_col="px", pt_widths=np.full(n, 0.02),
              sl_widths=np.full(n, 0.01), horizon_bars=5, horizon=None,
              horizon_col=None, timestamps=idx, high_col=None, low_col=None,
              drop_incomplete=True, bid_col=None, ask_col=None,
              fee_bps_per_side=0.0)
    bad = data.copy()
    bad.iloc[3, bad.columns.get_loc("sig")] = 0.5
    with pytest.raises(ValueError, match=r"row 3"):
        _entry_events(bad, **kw)
    good = data.copy()
    good.iloc[[1, 2, 10], good.columns.get_loc("sig")] = [1.0, -1.0, 1.0]
    out = _entry_events(good, **kw)
    assert len(out) == 3
    assert list(out["side"]) == [1, -1, 1]
    assert np.allclose(out["entry_price"],
                       np.round(out["entry_price"] / 0.01) * 0.01,
                       rtol=0, atol=1e-9)


def test_simulate_barriers_kernel_matches_vectorized():
    """barrier_outcome_kernel (wired into simulate_barriers) must reproduce the
    original vectorized O(n*H) k-loop exactly: same path window, stop-wins-ties,
    NaN never hits, +inf hits, side-0/unscored rows stay outcome 0."""
    from strategy_system.strategy.moe.outcome.triple_barrier import BarrierData, simulate_barriers
    rng = np.random.default_rng(21)
    n, H = 1500, 30
    idx = pd.date_range("2025-01-01", periods=n, freq="1s", tz="UTC")
    price = 100 + rng.normal(size=n).cumsum() * 0.1
    high = price * (1.0 + np.abs(rng.normal(size=n)) * 0.002)
    low = price * (1.0 - np.abs(rng.normal(size=n)) * 0.002)
    high[::211] = np.nan          # NaN bar: never hits
    low[::197] = np.nan
    high[::503] = np.inf          # +inf bar: hits, exactly like numpy >=
    side = rng.choice([-1.0, 0.0, 1.0], size=n).astype(float)
    ev = side != 0
    tp_width = np.full(n, np.nan)
    sl_width = np.full(n, np.nan)
    tp_width[ev] = 0.002          # widths finite only on event rows (as tb_regressor)
    sl_width[ev] = 0.001
    tp_width[ev & (rng.random(n) < 0.05)] = np.nan   # unscored: one-sided NaN too
    sl_width[ev & (rng.random(n) < 0.03)] = np.nan
    df = pd.DataFrame({"price_close": price}, index=idx)
    d = BarrierData(df=df, X=np.zeros((n, 1)), kept_cols=["x"], price=price,
                    high=high, low=low, side=side, horizon=H,
                    path_ok=np.ones(n, bool), train=np.zeros(n, bool),
                    test=ev.copy(), price_col="price_close")

    # reference = the original vectorized k-loop, verbatim
    p, s = d.price, d.side.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tp_level = p * (1.0 + s * tp_width)
        sl_level = p * (1.0 - s * sl_width)
    outcome = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    for k in range(1, H + 1):
        m = n - k
        if m <= 0:
            break
        hi, lo, side_v = d.high[k:], d.low[k:], s[:m]
        hit_sl = np.where(side_v > 0, lo <= sl_level[:m], hi >= sl_level[:m])
        hit_tp = np.where(side_v > 0, hi >= tp_level[:m], lo <= tp_level[:m])
        live = ~done[:m]
        sl_now = live & hit_sl
        tp_now = live & ~hit_sl & hit_tp
        outcome[:m][sl_now] = -1
        outcome[:m][tp_now] = 1
        done[:m] |= sl_now | tp_now
    timeout_price = np.full(n, np.nan)
    timeout_price[: n - H] = p[H:]
    timeout_ret = s * (timeout_price / p - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = tp_width / sl_width
        r = np.where(outcome == 1, rr, np.where(outcome == -1, -1.0, timeout_ret / sl_width))
    valid = (d.path_ok & np.isfinite(p) & (p > 0) & np.isfinite(tp_width) & (tp_width > 0)
             & np.isfinite(sl_width) & (sl_width > 0) & np.isfinite(tp_level) & (tp_level > 0)
             & np.isfinite(sl_level) & (sl_level > 0) & np.isfinite(r) & (d.side != 0))
    ref = pd.DataFrame({"tp_width": tp_width, "sl_width": sl_width, "rr": rr,
                        "outcome": outcome, "r": r, "valid": valid}, index=d.df.index)

    got = simulate_barriers(d, tp_width, sl_width)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)


def test_run_tb_serial_and_threaded_identical(tmp_path):
    """moe.tb_workers=1 (serial) vs >1 (thread pool) must produce identical
    reports and identical exported latent metadata."""
    import json as _json
    from strategy_system.strategy.moe.gp.gp_stage import GPStage
    from strategy_system.strategy.moe.ops.export import (
        GP_FEATURES_FILE, METADATA_FILE, REGIME_METADATA_FILE, REGIME_MODEL_FILE,
        latent_dir_name,
    )
    from strategy_system.strategy.moe.strategy import run_tb

    rng = np.random.default_rng(5)
    n = 4000
    idx = pd.date_range("2025-02-01", periods=n, freq="100ms", tz="UTC").as_unit("ns")
    price = 100 + rng.normal(size=n).cumsum() * 0.05
    feat = pd.DataFrame({"price_close": price,
                         "high": price * 1.001,
                         "low": price * 0.999}, index=idx)
    horizons = ["1", "2"]
    preds = pd.DataFrame({f"future_return__{h}": rng.normal(size=n) * 1e-3
                          for h in horizons}, index=idx)
    targets = pd.DataFrame(index=idx)
    for h in horizons:
        targets[f"future_return__{h}"] = rng.normal(size=n) * 1e-3
        targets[f"future_timestamp__{h}"] = idx + pd.to_timedelta(int(h) * 100_000_000, unit="ns")
    latent = pd.Series(np.where(np.arange(n) % 2 == 0, "1", "2"),
                       index=idx, name="latent_id")
    stage = GPStage(gp=None,
                    horizon_models={"1": None, "2": None},
                    horizon_report=pd.DataFrame(),
                    predictions=preds, targets=targets,
                    latent_id=latent, feat_df=feat,
                    terminal_names=[], horizons=horizons)

    def _run(workers: int):
        out = tmp_path / f"models_{workers}"
        (out / "regime").mkdir(parents=True, exist_ok=True)
        (out / "regime" / REGIME_MODEL_FILE).write_bytes(b"dummy")
        (out / "regime" / REGIME_METADATA_FILE).write_text("{}", encoding="utf-8")
        for lat in ("1", "2"):
            lat_dir = out / latent_dir_name(lat)
            lat_dir.mkdir(parents=True, exist_ok=True)
            (lat_dir / GP_FEATURES_FILE).write_text(
                _json.dumps({"placeholder": True}), encoding="utf-8")
        cfg = {"moe": {"models_dir": str(out), "tb_workers": workers},
               "outcome": {"triple_barrier": {"horizon": 10, "rr_window": 20,
                                              "high_col": "high", "low_col": "low"}}}
        report, manifest = run_tb(feat, stage, cfg)
        return out, report, manifest

    out1, report1, manifest1 = _run(1)
    out3, report3, manifest3 = _run(3)   # 2 latents -> 2 pool workers

    pd.testing.assert_frame_equal(report1, report3)
    assert report1["status"].tolist() == ["exported", "exported"]
    assert manifest1 is not None and manifest3 is not None
    for lat in ("1", "2"):
        meta1 = _json.loads((out1 / latent_dir_name(lat) / METADATA_FILE).read_text())
        meta3 = _json.loads((out3 / latent_dir_name(lat) / METADATA_FILE).read_text())
        for key in ("n_fit_rows", "input_features", "horizons", "barrier", "model"):
            assert meta1[key] == meta3[key], (lat, key)
        # same ONNX bytes: the fit is deterministic and independent of workers
        onnx1 = (out1 / latent_dir_name(lat) / "tb_regressor.onnx").read_bytes()
        onnx3 = (out3 / latent_dir_name(lat) / "tb_regressor.onnx").read_bytes()
        assert onnx1 == onnx3


def test_repair_one_file_matches_pandas_reference():
    from strategy_system.exchange.loader import (
        BTCTICK, MAX_SPREAD, _book_columns, _repair_one_file, quantize_price)
    rows = 40
    bid = np.full(rows, 108612.2)
    ask = np.full(rows, 108613.4)
    bid[5:7] = np.nan
    ask[5:7] = np.nan
    bid[10] = 5000.0          # errant needle
    ask[10] = 109753.4
    ev = ["LOB"] * rows
    ev[1], ev[2], ev[10] = "trade", "BookDepth", "lob"   # mixed case
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-07-03", periods=rows, freq="100ms", tz="UTC"),
        "event_type": ev,
        "best_bid_price": bid,
        "best_ask_price": ask,
        "bid_price_1": bid - 1.0,
        "ask_price_1": ask + 1.0,
    })
    # reference = the original pandas repair body, verbatim
    ref = df.copy()
    ref_cols = _book_columns(ref)
    ref.loc[ref["event_type"].astype(str).str.upper() != "LOB", ref_cols] = np.nan
    for c in ref_cols:
        ref[c] = pd.to_numeric(ref[c], errors="coerce")
    _b, _a = ref["best_bid_price"], ref["best_ask_price"]
    _errant = _b.notna() & _a.notna() & (
        (_b <= 0) | (_a <= _b) | ((_a - _b) > MAX_SPREAD * (_b + _a) / 2.0))
    ref.loc[_errant, ref_cols] = np.nan
    ref[ref_cols] = ref[ref_cols].ffill().bfill()
    ref["mid_price"] = quantize_price(
        (ref["best_bid_price"] + ref["best_ask_price"]) / 2.0, BTCTICK)

    got = _repair_one_file(df.copy())
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), ref.reset_index(drop=True), check_dtype=False)


def test_causal_feature_matrix_matches_pandas_reference():
    from strategy_system.strategy.moe.outcome.triple_barrier import _causal_feature_matrix
    rng = np.random.default_rng(37)
    n = 300
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n) * 3.0,
        "all_nan_train": np.where(np.arange(n) < 200, np.nan, 1.0),
        "words": ["abc"] * n,
    })
    df.loc[::7, "x"] = np.nan
    cols = ["x", "y", "all_nan_train", "words"]
    train_idx = np.arange(0, 200)
    raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    keep = np.flatnonzero(np.isfinite(raw[train_idx]).any(axis=0))
    ref_X = np.where(np.isfinite(raw[:, keep]), raw[:, keep], 0.0)
    ref_cols = [cols[i] for i in keep.tolist()]
    got_X, got_cols = _causal_feature_matrix(df, cols, train_idx)
    assert got_cols == ref_cols
    assert np.array_equal(got_X, ref_X, equal_nan=True)
