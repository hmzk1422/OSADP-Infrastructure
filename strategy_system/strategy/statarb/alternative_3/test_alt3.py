"""Unit tests for spread_discoverer.py / strategy.py / backtest.py.     Run:  python -m pytest test_alt3.py -q
Everything runs on synthetic data (no network, no binance_vision): a cointegrated ETH = 0.9 BTC + 0.2 SOL + OU spread."""
from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
import backtest as bt_mod
import spread_discoverer as sd
import strategy as st
from spread_discoverer import AssetPanel, DiscoverConfig, ft, px

SYMS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
Y = "ETH/USDT"


# --------------------------------------------------------------------------- synthetic data
def ou(n, hl, rng, sd_=1.0):
    phi = np.exp(-np.log(2) / hl)
    e = rng.normal(size=n) * np.sqrt(1 - phi ** 2) * sd_
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def make_panel(n=6000, seed=0, hl=60.0, start="2024-01-01", feats=True, spread_sd=0.01, drop=150):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min")
    common = np.cumsum(rng.normal(0, 4e-4, n))
    lp = {s: 5 + 0.3 * k + common * (0.8 + 0.1 * k) + np.cumsum(rng.normal(0, 3e-4, n)) for k, s in enumerate(SYMS)}
    lp[Y] = 0.5 + 0.9 * lp["BTC/USDT"] + 0.2 * lp["SOL/USDT"] + ou(n, hl, rng, spread_sd)
    logp = pd.DataFrame(lp, index=idx)[SYMS]
    ret = logp.diff()
    fe = {}
    if feats:
        noise = lambda: pd.DataFrame(rng.normal(size=(n, len(SYMS))), index=idx, columns=SYMS)
        r = ret.fillna(0.0)
        fe["ofi"] = (r / (r.std() + 1e-12)) * 0.5 + noise()
        fe["tfi"] = np.tanh(fe["ofi"] * 0.5 + noise() * 0.5)
        fe["net_flow"] = fe["ofi"] * 100 + noise() * 10
        fe["cvd"] = fe["net_flow"].cumsum()
        fe["qi1"] = np.tanh(noise())
    return AssetPanel(logp, ret, ret.rolling(100).std(), fe).iloc(drop, n)


def make_hilo(panel, seed=1, amp=4e-4):
    rng = np.random.default_rng(seed)
    c = np.exp(panel.logp)
    return st.HiLo(c * (1 + np.abs(rng.normal(0, amp, c.shape))), c * (1 - np.abs(rng.normal(0, amp, c.shape))))


@pytest.fixture(scope="module")
def panel():
    return make_panel(7000, seed=1)


@pytest.fixture(scope="module")
def hilo(panel):
    return make_hilo(panel)


@pytest.fixture(scope="module")
def dcfg():
    return DiscoverConfig(k=10, max_terms=3, use_features=True)


@pytest.fixture(scope="module")
def selection(panel, dcfg):
    return st.Screener(panel.iloc(0, 3500), Y, bars=3500, cfg=dcfg)


# =============================================================================== spread_discoverer
def test_variance_ratio_random_walk_vs_ou():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.normal(size=20000))
    assert abs(sd.variance_ratio(rw, 30) - 1.0) < 0.15
    x = ou(20000, 40, rng)
    assert sd.variance_ratio(x, 60) < 0.75 and 0.62 < sd.variance_ratio(x, 15) < 0.95 and sd.variance_ratio(x, 15) > sd.variance_ratio(x, 60) and sd.vr_mean(x, (5, 15, 30, 60)) < sd.vr_mean(rw, (5, 15, 30, 60))
    assert np.isnan(sd.variance_ratio(x[:50], 30))                       # too short
    assert sd.vr_robust_z(x, 60) < -3 and abs(sd.vr_robust_z(rw, 60)) < 3.5


def test_half_life_matches_ou_truth():
    x = ou(60000, 80, np.random.default_rng(1))
    assert abs(sd.half_life(x) - 80) / 80 < 0.15


def test_poolpeval_equals_reference(panel, dcfg):
    pool = sd._pool(panel, Y, dcfg)
    ev = sd.PoolEval(panel, Y, pool, dcfg)
    subsets = [(px("BTC/USDT"),), tuple(sorted((px("BTC/USDT"), px("SOL/USDT")))),
               tuple(sorted((px("BTC/USDT"), px("SOL/USDT"), ft("BTC/USDT", "ofi")))),
               tuple(sorted((px("ADA/USDT"), ft("SOL/USDT", "cvd"), ft("XRP/USDT", "tfi"))))]
    for s in subsets:
        a, b = ev.evaluate(s), sd._evaluate(panel, Y, s, dcfg)
        assert a.score == pytest.approx(b.score, rel=1e-5)
        assert a.vr == pytest.approx(b.vr, rel=1e-6) and a.half_life == pytest.approx(b.half_life, rel=1e-6)
        assert a.drift == pytest.approx(b.drift, rel=1e-4)            # the closed-form piecewise-constant chain == block refits
        assert a.feat_share == pytest.approx(b.feat_share, rel=1e-6) and a.max_share == pytest.approx(b.max_share, rel=1e-6)
        for t in s:
            assert a.betas[t] == pytest.approx(b.betas[t], rel=1e-5, abs=1e-8)


def test_beam_search_does_not_use_adf(panel, dcfg, monkeypatch):
    calls = {"n": 0}
    real = sd.adf_stat

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(sd, "adf_stat", counting)
    res = sd.create_pair(panel.iloc(0, 3500), Y, cfg=dcfg)
    assert len(res) > 0
    assert calls["n"] <= len(res)                       # ADF only on the final shortlist (thousands of candidates were scored)
    assert all(np.isfinite(c.p_eg) for c in res)        # ... and the shortlist still carries its Engle-Granger p for reports


def test_score_is_vr_and_halflife_driven():
    cfg = DiscoverConfig()
    good = sd._score(0.30, 80.0, 2, 0.5, 1.0, cfg)
    worse_vr = sd._score(0.95, 80.0, 2, 0.5, 1.0, cfg)
    slow = sd._score(0.30, 900.0, 2, 0.5, 1.0, cfg)
    noise = sd._score(0.02, 0.5, 2, 0.5, 1.0, cfg)      # white-noise spread: VR tiny but half-life < 2 bars -> rejected
    unit_root = sd._score(1.0, float("inf"), 2, 0.5, 1.0, cfg)
    assert good < slow < unit_root and good < worse_vr and noise > good
    assert sd._score(0.30, 80.0, 3, 0.5, 1.0, cfg) > good                       # extra term must pay for itself
    assert sd._score(0.7, 80.0, 1, 1.0, 1.0, cfg) > sd._score(0.55, 80.0, 1, 1.0, 1.0, cfg)


def test_beam_finds_planted_cointegration(panel, dcfg):
    res = sd.create_pair(panel.iloc(0, 4000), Y, cfg=replace(dcfg, k=15))
    best_px = {t for c in res[:5] for t in c.terms if t.startswith("px|")}
    assert px("BTC/USDT") in best_px or px("SOL/USDT") in best_px
    assert res[0].vr < 0.9 and np.isfinite(res[0].half_life)


def test_test_pair_gate_is_vr_not_adf(panel, dcfg):
    cands = sd.create_pair(panel.iloc(0, 3500), Y, cfg=dcfg)
    r = sd.test_pair(panel.iloc(0, 7000), cands[0], dcfg, split=3500)
    assert np.isfinite(r.vr_oos) and r.vr_ok == (r.vr_oos <= dcfg.vr_oos_max)
    assert r.passed == (r.vr_ok and r.hl_ok and r.dom_ok and r.drift_ok and r.feat_ok)   # p_ok is informative only


def test_discover_cfg_for_bar():
    base = DiscoverConfig()
    one, five = DiscoverConfig.for_bar(1), DiscoverConfig.for_bar(5)
    assert one.vr_lags == base.vr_lags and one.hl_ref == base.hl_ref and one.z_window == 288
    assert max(five.vr_lags) < max(one.vr_lags) and five.hl_ref == pytest.approx(20.0) and min(five.vr_lags) >= 2


def test_resample_panel(panel):
    p = panel.iloc(0, 1503)
    r = sd.resample_panel(p, 5, vol_window=20)
    assert r.bar_minutes == 5.0
    key = p.logp.index.floor("5min")
    full = pd.Series(1, index=p.logp.index).groupby(key).sum()
    full = full.index[full.to_numpy() >= 5]
    exp_last = p.logp.groupby(key).last().loc[full]
    assert np.allclose(r.logp.to_numpy(), exp_last.iloc[20:].to_numpy())
    assert np.allclose(r.ret.iloc[1:].to_numpy(), r.logp.diff().iloc[1:].to_numpy())
    # feature rules: running total -> last, per-bar flow -> sum, ratio -> mean
    assert sd.feature_agg_rule("cvd") == "last" and sd.feature_agg_rule("net_flow") == "sum" and sd.feature_agg_rule("tfi") == "mean"
    exp_nf = p.feats["net_flow"].groupby(key).sum().loc[full].iloc[20:]
    assert np.allclose(r.feats["net_flow"].to_numpy(), exp_nf.to_numpy())
    assert np.allclose(r.feats["cvd"].to_numpy(), p.feats["cvd"].groupby(key).last().loc[full].iloc[20:].to_numpy())
    assert not sd.resample_panel(p, 5, vol_window=20, features=False).feats                       # price-only option
    assert sd.resample_panel(p, 1) is p


def test_resample_hilo_and_concat(panel, hilo):
    r = st.resample_hilo(hilo, 5)
    key = hilo.high.index.floor("5min")
    assert np.allclose(r.high.iloc[3].to_numpy(), hilo.high[key == r.high.index[3]].max().to_numpy())
    assert np.allclose(r.low.iloc[3].to_numpy(), hilo.low[key == r.low.index[3]].min().to_numpy())
    a, b = panel.iloc(0, 500), panel.iloc(500, 900)
    c = AssetPanel.concat([a, b])
    assert len(c.logp) == 900 and np.allclose(c.logp.to_numpy(), panel.iloc(0, 900).logp.to_numpy())
    assert set(c.feats) == set(panel.feats)


# =============================================================================== structural trades (+ MAE / MFE, smart barriers)
def _ref_structural(x, z, mu, sd_, start, end, ez=2.0, tp_z=0.5, sl_z=3.5, T=576):
    """The per-bar python loop of the study (reference implementation)."""
    out, armed, i = [], True, int(start)
    while i < end - 1:
        zi = z[i]
        if not np.isfinite(zi):
            i += 1
            continue
        if abs(zi) < ez:
            armed = True
            i += 1
            continue
        if (not armed) or abs(zi) >= sl_z:
            i += 1
            continue
        sgn = 1.0 if zi > 0 else -1.0
        m, s0 = mu[i], sd_[i]
        tp, sl = m + sgn * tp_z * s0, m + sgn * sl_z * s0
        risk = (sl_z - abs(zi)) * s0
        seg = x[i + 1: min(i + 1 + T, end)]
        htp = seg <= tp if sgn > 0 else seg >= tp
        hsl = seg >= sl if sgn > 0 else seg <= sl
        big = 10 ** 9
        ktp = int(htp.argmax()) if htp.any() else big
        ksl = int(hsl.argmax()) if hsl.any() else big
        if ktp == big and ksl == big:
            j, pxe, reason = len(seg) - 1, seg[-1], ("time" if len(seg) == T else "end")
        elif ksl <= ktp:
            j, pxe, reason = ksl, sl, "sl"
        else:
            j, pxe, reason = ktp, tp, "tp"
        e = i + 1 + j
        out.append((i, float(-sgn * (pxe - x[i]) / risk), reason))
        armed = bool(np.isfinite(z[e]) and abs(z[e]) < ez)
        i = e + 1
    return out


@pytest.mark.parametrize("seed,hl,T", [(0, 40, 120), (1, 150, 300), (2, 15, 60)])
def test_structural_trades_fast_equals_reference(seed, hl, T):
    rng = np.random.default_rng(seed)
    x = ou(6000, hl, rng, 0.01)
    z, mu, s_ = st.rolling_z(x, 300)
    ref = _ref_structural(x, z, mu, s_, 300, len(x), T=T)
    new = st.structural_trades(x, z, mu, s_, 300, len(x), T=T)
    assert len(ref) == len(new) > 5
    for a, b in zip(ref, new):
        assert a[0] == b[0] and a[2] == b[2] and a[1] == pytest.approx(b[1])


def test_structural_trade_hand_path_mae_mfe():
    # spread: mean 0, sigma 1.  z crosses 2 at bar 3 -> short the spread.  path then goes UP 0.5 (adverse) then collapses to the TP
    x = np.array([0, 0, 1.0, 2.0, 2.5, 1.0, -0.2, 0, 0, 0])
    z = x.copy()
    mu, s_ = np.zeros(10), np.ones(10)
    tr = st.structural_trades(x, z, mu, s_, 0, 10, ez=2.0, tp_z=0.5, sl_z=3.5, T=6)
    assert len(tr) == 1
    i, R, reason, e, side, mae, mfe = tr[0]
    assert (i, reason, e, side) == (3, "tp", 6, -1)                       # TP level = +0.5 sigma is crossed at bar 5 (x=1.0)? 1.0 > 0.5 -> bar 6
    assert mae == pytest.approx(0.5) and mfe == pytest.approx(2.2)       # worst close +0.5 against us, best close 2.0 - (-0.2) in our favour
    assert R == pytest.approx((2.0 - 0.5) / (3.5 - 2.0))                  # pays (|z| - tp_z) / (sl_z - |z|)


def test_structural_stop_wins_ties_and_time_barrier():
    x = np.array([0, 0, 2.2, 3.6, 0.0, 0, 0, 0])
    z, mu, s_ = x.copy(), np.zeros(8), np.ones(8)
    tr = st.structural_trades(x, z, mu, s_, 0, 8, T=5)
    assert tr[0][2] == "sl" and tr[0][1] == pytest.approx(-1.0)
    x2 = np.array([0, 0, 2.2, 2.1, 2.2, 2.1, 2.2, 2.1, 2.0])
    tr2 = st.structural_trades(x2, x2.copy(), np.zeros(9), np.ones(9), 0, 9, T=4)
    assert tr2[0][2] == "time"


def test_structural_smart_barriers_use_stop_dist_and_rr():
    x = np.array([0, 0, 0, 2.0, 1.8, 1.6, 1.0, 0.5, 0.2, 0])
    z, mu, s_ = x.copy(), np.zeros(10), np.ones(10)
    # short at 2.0 with stop 0.5 above, TP rr*stop = 1.0 below -> TP at 1.0: R = rr
    tr = st.structural_trades(x, z, mu, s_, 0, 10, T=8, stop_dist=np.full(10, 0.5), rr=2.0)
    assert tr[0][2] == "tp" and tr[0][1] == pytest.approx(2.0)
    x2 = np.array([0, 0, 0, 2.0, 2.3, 2.6, 1.0, 0.5, 0.2, 0])
    tr2 = st.structural_trades(x2, x2.copy(), mu, s_, 0, 10, T=8, stop_dist=np.full(10, 0.5), rr=2.0)
    assert tr2[0][2] == "sl" and tr2[0][1] == pytest.approx(-1.0)
    ok = np.zeros(10, bool)                                              # entry model says no -> no trade, and it does not disarm
    assert st.structural_trades(x, z, mu, s_, 0, 10, T=8, stop_dist=np.full(10, 0.5), rr=2.0, entry_ok=ok) == []


def test_structural_independent_events_overlap():
    rng = np.random.default_rng(3)
    x = ou(4000, 60, rng, 0.01)
    z, mu, s_ = st.rolling_z(x, 300)
    seq = st.structural_trades(x, z, mu, s_, 300, len(x), T=200)
    ev = st.structural_trades(x, z, mu, s_, 300, len(x), T=200, sequential=False)
    ev3 = st.structural_trades(x, z, mu, s_, 300, len(x), T=200, sequential=False, stride=3)
    assert len(ev) > 2 * len(seq) and len(ev3) == len(range(0, len(ev), 3))
    assert all(t[5] >= 0 and t[6] >= 0 for t in ev)


# =============================================================================== order-flow features
def test_orderflow_features_isolated_per_leg_and_causal(panel):
    legs = ["ETH/USDT", "BTC/USDT", "XRP/USDT"]
    of = st.orderflow_features(panel, legs, lookback=30)
    assert {c.split("|")[1] for c in of.columns} == set(legs)
    for name in ("ofi", "tfi", "net_flow", "qi1"):
        for sym in legs:
            assert f"{name}|{sym}" in of.columns and f"{name}|{sym}|m30" in of.columns       # one block per leg, never netted
    assert not any(c.startswith(("net|", "sum|", "total|")) for c in of.columns)
    assert not any("ADA/USDT" in c for c in of.columns)
    # each leg's column is the leg's own feature, not a mix
    assert np.allclose(of["ofi|BTC/USDT"].to_numpy(), panel.feats["ofi"]["BTC/USDT"].to_numpy())
    # causal: perturbing the future never changes the past
    p2 = panel.iloc(0, len(panel.logp))
    p2.feats["ofi"] = p2.feats["ofi"].copy()
    p2.feats["ofi"].iloc[3000:] += 100.0
    of2 = st.orderflow_features(p2, legs, lookback=30)
    assert np.allclose(of.iloc[:3000].fillna(0).to_numpy(), of2.iloc[:3000].fillna(0).to_numpy())
    # price-only panel: only the price context survives
    po = st.orderflow_features(make_panel(1000, feats=False), legs, lookback=30)
    assert all(c.startswith(("ret|", "vol|")) for c in po.columns)


def test_orderflow_window_stats_marks_legs(panel):
    of = st.orderflow_features(panel, panel.symbols, lookback=30)
    s = st.orderflow_window_stats(of, 0, 2000, ["ETH/USDT", "BTC/USDT"], panel.symbols)
    assert s["of_leg_ETHUSDT"] == 1 and s["of_leg_BTCUSDT"] == 1 and s["of_leg_ADAUSDT"] == 0
    assert np.isfinite(s["of_ofi_BTCUSDT_std"]) and np.isnan(s["of_ofi_ADAUSDT_std"])           # absent leg stays NaN, isolated


# =============================================================================== HalfLifePredictor + half-life multiplier
@pytest.fixture(scope="module")
def dataset():
    p = make_panel(9000, seed=7, hl=70)
    d = st.DatasetConfig(train_bars=1500, fwd_bars=(300, 600), step=1500, top_k=4, max_origins=4, vr_lags=(5, 15, 30, 60), orderflow_lookback=30)
    return st.build_dataset(p, Y, DiscoverConfig(k=8, max_terms=3, use_features=True), d), d


def test_halflife_predictor_orderflow_flag(dataset):
    df, d = dataset
    assert any(c.startswith("of_") for c in df.columns) and any(c.startswith("f_") for c in df.columns)
    base = st.HalfLifePredictor(600, orderflow_features=False).fit(df, d)
    flow = st.HalfLifePredictor(600, orderflow_features=True, alpha=100.0).fit(df, d)
    assert not any(c.startswith("of_") for c in base.features)                       # False: the study's spread statistics only
    assert set(base.features) <= set(st.DEFAULT_FEATURES)
    of_cols = [c for c in flow.features if c.startswith("of_")]
    assert of_cols and any("ofi_BTCUSDT" in c for c in of_cols) and any("ofi_ETHUSDT" in c for c in of_cols)   # per-leg, isolated
    syms = {x.replace("/", "") for x in SYMS}
    assert all(c.startswith("of_leg_") or any(f"_{sy}_" in c for sy in syms) for c in of_cols)   # every column belongs to exactly one leg
    F = df[df["horizon"] == 600].head(5)
    for m in (base, flow):
        hl = m.predict_hl(F)
        assert hl.shape == (5,) and np.all(hl > 0) and np.all(np.isfinite(hl))
    with pytest.raises(ValueError):
        st.HalfLifePredictor(600, orderflow_features=True).fit(df[[c for c in df.columns if not c.startswith("of_")]], d)
    rep = st.compare_predictors(df, d, 1.0, horizon=600)
    assert set(rep["orderflow_features"]) <= {False, True}


def test_half_life_multiplier_is_two_and_sets_z_window(panel, dcfg):
    assert st.HALF_LIFE_MULTIPLIER == 2.0 and st.TradeConfig().half_life_multiplier == 2.0
    sel = st.Screener(panel.iloc(0, 3500), Y, bars=3500, cfg=dcfg)
    assert sel.z_window == int(np.clip(round(2.0 * sel.half_life_asset), 30, 2880))
    sel3 = st.Screener(panel.iloc(0, 3500), Y, bars=3500, cfg=dcfg, half_life_mult=3.0)
    assert sel3.z_window == int(np.clip(round(3.0 * sel3.half_life_asset), 30, 2880))


def test_retrain_rule_uses_predicted_half_life(dataset, panel, dcfg):
    df, d = dataset
    pred = st.HalfLifePredictor(600, orderflow_features=True, alpha=100.0).fit(df, d)
    rule = st.HalfLifeRetrainRule(pred)
    sel = rule(panel.iloc(0, 1500), Y, dcfg)
    assert np.isfinite(sel.hl_pred) and sel.z_window == int(np.clip(round(2.0 * sel.hl_pred), 30, 2880))
    assert sel.hl_trade == sel.hl_pred
    same = st.screen_fold(panel.iloc(0, 1500), Y, dcfg, retrain_rule=rule, ranked=st.beam_rank(panel.iloc(0, 1500), Y, dcfg))
    assert same.z_window == sel.z_window and same.hedge_symbols == sel.hedge_symbols      # shared beam search -> same selection


# =============================================================================== risk rules
def test_no_sl_vol_mult_and_formula():
    cfg = st.TradeConfig(rr=2.0, barrier_size=1.5, risk_fraction=0.02)
    assert not hasattr(cfg, "sl_vol_mult")
    lv = st.stoploss(0.001, 1.5, cfg, 100)
    assert lv.stop_pct == pytest.approx(1.5 * 0.001 * 10 / 1.5) and lv.tp_pct == pytest.approx(2.0 * lv.stop_pct)
    assert lv.stop_dollars == pytest.approx(cfg.risk_dollars) and lv.tp_dollars == pytest.approx(2 * cfg.risk_dollars)
    capped = st.levels_from_stop_pct(1e-5, replace(cfg, max_leverage=10.0))
    assert capped.capped and capped.leverage == 10.0 and capped.stop_dollars < cfg.risk_dollars
    assert st.levels_from_stop_pct(0.0, cfg) is None and st.stoploss(0.0, 1.5, cfg, 10) is None


def test_trade_config_validation_and_for_bar():
    for bad in (dict(smart_mode="x"), dict(half_life_multiplier=0), dict(struct_sl_z=1.0), dict(smart_oof_folds=1), dict(rr=0)):
        with pytest.raises(ValueError):
            st.TradeConfig(**bad)
    c5 = st.TradeConfig.for_bar(5.0)
    assert c5.vol_window == 288 and c5.smart_lookback_bars == 12 and c5.max_hold_bars == 576


# =============================================================================== smart trading
def test_rl_policy_learns_planted_signal_and_abstains_without_edge():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 8))
    R = np.where(X[:, 0] > 0.3, 1.0, -1.0) + rng.normal(0, 0.3, n)               # a trade pays only when feature 0 is high
    t = np.arange(n)
    pol = st.train_rl_policy(X, R, t, t + 5, epochs=200, lr=0.05, min_cov=0.1)
    p = pol.prob(X)
    taken = p >= 0.5
    assert not pol.abstain and R[taken].mean() > R.mean() + 0.6 and taken.mean() >= 0.1
    assert 0 in pol.cols
    R_bad = -0.4 + rng.normal(0, 0.5, n)                                          # no feature predicts anything, mean negative
    pol2 = st.train_rl_policy(X, R_bad, t, t + 5, epochs=200, lr=0.05, min_cov=0.1)
    assert pol2.abstain


def test_smart_models_fit_and_decide(panel, hilo, selection, dcfg):
    cfg = st.TradeConfig.for_bar(1.0, smart_trading=True, smart_mode="classifier", rr=1.5, barrier_size=2.0, smart_min_events=30,
                                 smart_event_stride=2)
    tp = panel.iloc(0, 3500)
    m = st.fit_smart_models(selection, tp, cfg, st._hilo_window(hilo, tp))
    assert m is not None and m.mode == "classifier" and m.stop_hi > m.stop_lo > 0
    assert any(c.startswith("ofi|") for c in m.columns) and any(c.startswith("ofi|XRP") or c.startswith("ofi|BTC") or c.startswith("ofi|SOL") or c.startswith("ofi|ADA") or c.startswith("ofi|BNB") for c in m.columns)
    assert {c.split("|")[1] for c in m.columns if "|" in c} <= set(selection.symbols) | {"ETH/USDT"}   # only the legs of THIS spread, isolated
    X = st._smart_frame(selection, cfg, st._signals(panel.iloc(3500, 5500), st._close_frame(panel.iloc(3500, 5500), selection.symbols), selection, cfg),
                        np.array([300, 800, 1200]), np.array([1, -1, 1]), st.orderflow_features(panel.iloc(3500, 5500), selection.symbols, lookback=60))
    d1 = m.decide(X, cfg)
    d2 = m.decide(X, replace(cfg, barrier_size=1.0))
    assert np.allclose(d1["stop_pct"], 2.0 * d2["stop_pct"])                      # stop_pct = predicted MAE * barrier_size
    lv = st.levels_from_stop_pct(float(d1["stop_pct"][0]), cfg)
    assert lv.tp_pct == pytest.approx(cfg.rr * lv.stop_pct) and (lv.stop_dollars == pytest.approx(cfg.risk_dollars) or lv.capped)
    assert np.all(d1["mae_pct"] > 0) and np.all(d1["mfe_pct"] > 0)


def test_cost_aware_threshold():
    m = st.SmartModels("classifier", [], pd.Series(dtype=float), None, None, 0.001, 0.01)
    cfg = st.TradeConfig(rr=1.0, fee_bps=2.0, slippage_bps=1.0, smart_margin=0.0)
    stop = np.array([0.0004, 0.003])
    thr = m.threshold_for(stop, cfg)
    assert thr[0] == pytest.approx((1 + 2 * 3e-4 / 0.0004) / 2) and thr[0] > 1.0 > thr[1]           # c = 1.5R: no probability can pay
    assert m.threshold_for(stop, replace(cfg, smart_threshold=0.6)) .tolist() == [0.6, 0.6]


@pytest.mark.parametrize("mode", ["classifier", "rl"])
def test_smart_forward_ledger(panel, hilo, selection, mode):
    tp, fp = panel.iloc(0, 3500), panel.iloc(3500, 7000)
    cfg = st.TradeConfig.for_bar(1.0, smart_trading=True, smart_mode=mode, rr=1.0, smart_min_events=30, smart_event_stride=2,
                                 smart_margin=-0.5, rl_min_coverage=0.3)
    plan = st.FoldPlan(0, "A", 0, 3500, 3501, 7000)
    fold = st.trade_fold(selection, tp, fp, plan, cfg, hilo=st._hilo_window(hilo, tp, fp))
    assert fold.selection.smart is not None
    tr = fold.forward.trades
    if mode == "classifier":
        assert len(tr) > 0
    if len(tr):
        assert np.allclose(tr["stop_pct"], tr["mae_pred"])                         # barrier_size = 1 -> stop IS the predicted MAE
        assert tr["p_enter"].notna().all()
        assert np.allclose(tr["risk_dollars"], cfg.risk_dollars, rtol=1e-6) or (tr["leverage"] >= cfg.max_leverage - 1e-9).any()


def test_formula_mode_has_no_smart_outputs(panel, hilo, selection):
    fp = panel.iloc(3500, 7000)
    cfg = st.TradeConfig.for_bar(1.0)
    fwd = st.Forward(selection, fp, cfg=cfg, hilo=st._hilo_window(hilo, fp))
    assert len(fwd.trades) > 0 and fwd.trades["p_enter"].isna().all() and fwd.trades["mae_pred"].isna().all()


def test_smart_trading_without_models_trades_nothing(panel, hilo, selection):
    fp = panel.iloc(3500, 7000)
    cfg = st.TradeConfig.for_bar(1.0, smart_trading=True)
    fwd = st.Forward(selection, fp, cfg=cfg, hilo=st._hilo_window(hilo, fp))
    assert len(fwd.trades) == 0 and np.allclose(fwd.quantities.to_numpy(), 0)


# =============================================================================== execution invariants
def test_signals_are_causal(panel, selection):
    cfg = st.TradeConfig.for_bar(1.0)
    fp = panel.iloc(3500, 6000)
    cut = 4500 - 3500
    full = st._signals(fp, st._close_frame(fp, selection.symbols), selection, cfg)
    pre = fp.iloc(0, cut)
    part = st._signals(pre, st._close_frame(pre, selection.symbols), selection, cfg)
    for col in ("z", "vol", "dev", "asset_spread"):
        assert np.allclose(full[col].iloc[:cut].fillna(-9).to_numpy(), part[col].fillna(-9).to_numpy())


def test_trade_has_no_lookahead(panel, hilo, selection):
    cfg = st.TradeConfig.for_bar(1.0)
    fp = panel.iloc(3500, 6500)
    cut = 1800
    h = st._hilo_window(hilo, fp)
    full = st.trade(fp, selection, cfg, panel=fp, hilo=h)
    pre = fp.iloc(0, cut)
    part = st.trade(pre, selection, cfg, panel=pre, hilo=h)
    T = st._time_barrier(selection, cfg)
    idx = full.index
    safe = {t: m for t, m in full.attrs["trade_meta"].items() if idx.get_loc(t) + T + 2 < cut}      # trades that are over long before the cut
    assert len(safe) > 0
    for t, m in safe.items():
        assert t in part.attrs["trade_meta"] and part.attrs["trade_meta"][t]["reason"] == m["reason"]


def test_ledger_pnl_adds_up_and_costs_hurt(panel, hilo, selection):
    fp = panel.iloc(3500, 7000)
    h = st._hilo_window(hilo, fp)
    free = st.Forward(selection, fp, cfg=st.TradeConfig.for_bar(1.0), hilo=h)
    paid = st.Forward(selection, fp, cfg=st.TradeConfig.for_bar(1.0, fee_bps=3.0, slippage_bps=1.0), hilo=h)
    assert free.trades["pnl"].sum() == pytest.approx(free.net_pnl.sum(), rel=1e-9, abs=1e-6)
    assert len(free.trades) == len(paid.trades) and paid.net_pnl.sum() < free.net_pnl.sum() and paid.costs.sum() > 0
    risk = free.trades["risk_dollars"]
    assert (free.trades.loc[free.trades["reason"] == "sl", "r_multiple"] < -0.99).all() or free.trades["reason"].ne("sl").all()
    tp = free.trades[free.trades["reason"] == "tp"]
    assert (tp["r_multiple"] > 0).all()
    assert (free.quantities.iloc[-1].abs() == 0).all()                                      # window end is flat


def test_strategy_side_inverts_direction(panel, hilo, selection):
    fp = panel.iloc(3500, 6000)
    h = st._hilo_window(hilo, fp)
    a = st.Forward(selection, fp, cfg=st.TradeConfig.for_bar(1.0, strategy_side=1), hilo=h).trades
    b = st.Forward(selection, fp, cfg=st.TradeConfig.for_bar(1.0, strategy_side=2), hilo=h).trades
    m = a.merge(b, on="entry_time", suffixes=("_a", "_b"))             # paths diverge after the first exit, so compare shared entries
    assert len(m) > 0 and (m["side_a"] == -m["side_b"]).all() and (m["strategy_side_a"] == m["strategy_side_b"]).all()


def test_flip_flop_plan_is_always_trading():
    plans = st.plan_folds(60_000, train_bars=5_000, forward_bars=5_000, purge_bars=1)
    assert len(plans) == 10 and [p.instance for p in plans[:4]] == ["A", "B", "A", "B"]
    for a, b in zip(plans[:-1], plans[1:]):
        assert (b.train_start, b.train_end) == (a.forward_start, a.forward_end)               # next instance trains on the window just traded
        assert b.forward_start == a.forward_end + 1                                          # ... and takes over right after the purge
    assert plans[0].train_end + 1 == plans[0].forward_start and plans[-1].forward_end <= 60_000
    with pytest.raises(ValueError):
        st.plan_folds(60_000, train_bars=5_000, forward_bars=3_000)


def test_algorithm_end_to_end(panel, hilo, dcfg):
    res = st.Algorithm(panel, Y, train_bars=2000, forward_bars=2000, discover_cfg=dcfg, trade_cfg=st.TradeConfig.for_bar(1.0),
                       hilo=hilo, print_report=False)
    assert len(res.folds) == 2 and len(res.trades) > 0 and {"vr_train", "hl_pred", "z_window", "vr_oos"} <= set(res.fold_table.columns)
    assert res.pnl.sum() == pytest.approx(res.total_pnl)


# =============================================================================== backtest.py
def _bt_cfg(tmp, **kw):
    base = dict(symbols=tuple(SYMS), history_start="2024-01-20", history_end="2024-04-05", total_bars=8000, chunk_bars=2000,
                min_segment_bars=4000, n_iterations=2, seed=3, n_workers=1, cache_dir=tmp, report_dir=None, n_boot=100, lead_days=1)
    base.update(kw)
    return bt_mod.BacktestConfig(**base)


@pytest.fixture(scope="module")
def big():
    p = make_panel(140_000, seed=5, hl=80, start="2024-01-01")
    return p, make_hilo(p, seed=2)


def _loader(big, hole=None):
    p, h = big

    def loader(start, end):
        idx = p.logp.index
        a, b = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end))
        q = p.iloc(a, b)
        hi = st.HiLo(h.high.loc[q.logp.index[0]:q.logp.index[-1]], h.low.loc[q.logp.index[0]:q.logp.index[-1]])
        if hole is not None:
            keep = ~((q.logp.index >= hole[0]) & (q.logp.index < hole[1]))
            q = AssetPanel(q.logp[keep], q.ret[keep], q.vol[keep], {k: v[keep] for k, v in q.feats.items()})
        return q, hi
    return loader


def test_segment_composition_random_and_reproducible():
    cfg = _bt_cfg(tempfile.mkdtemp(), total_bars=100_000, chunk_bars=5_000, min_segment_bars=10_000)
    seen = set()
    for it in range(6):
        segs = bt_mod.draw_iteration(cfg, it)
        assert sum(s.bars for s in segs) == 100_000 and all(s.bars % 5_000 == 0 and s.bars >= 10_000 for s in segs)
        assert all(s.n_folds == s.bars // 5_000 - 1 and s.need_bars == s.bars + s.n_folds for s in segs)
        assert all(cfg.history_start <= str(s.start.date()) for s in segs)
        seen.add(tuple((s.bars, str(s.start.date())) for s in segs))
        assert [(s.bars, s.start) for s in segs] == [(s.bars, s.start) for s in bt_mod.draw_iteration(cfg, it)]   # same (seed, it) -> same draw
    assert len(seen) == 6                                                                                    # iterations differ


def test_phase2_mixes_timeframes():
    cfg = _bt_cfg(tempfile.mkdtemp(), total_bars=40_000, chunk_bars=2_000, min_segment_bars=4_000, timeframes=(1, 5, 15), n_iterations=6,
                  history_start="2023-01-01")
    tfs = {s.tf for it in range(6) for s in bt_mod.draw_iteration(cfg, it)}
    assert tfs == {1, 5, 15}
    assert {s.tf for s in bt_mod.draw_iteration(_bt_cfg(tempfile.mkdtemp()), 0)} == {1}              # phase_1 is 1-minute only


def test_datastore_roundtrip_month_boundary_and_gaps(big):
    p, _ = big
    tmp = tempfile.mkdtemp()
    store = bt_mod.DataStore(tmp, SYMS, "1min", _loader(big), lead_days=1, max_gap_bars=30)
    a, b = pd.Timestamp("2024-01-28"), pd.Timestamp("2024-02-03")
    assert store.ensure(a, b) and store.range_ok(a, b)
    q, h = store.read(a, b)
    idx = q.logp.index
    assert idx[0] == a and idx[-1] == b - pd.Timedelta(minutes=1) and (idx.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all()   # no hole at Feb 1
    ref = p.logp.loc[a:b - pd.Timedelta(minutes=1)]
    assert np.allclose(q.logp.to_numpy(), ref.to_numpy()) and len(h.high.loc[idx[0]:idx[-1]]) == len(idx)
    assert pickled_without_loader(store)
    holey = bt_mod.DataStore(tempfile.mkdtemp(), SYMS, "1min", _loader(big, hole=(pd.Timestamp("2024-02-10 00:00"), pd.Timestamp("2024-02-10 06:00"))),
                             lead_days=1, max_gap_bars=30)
    assert holey.ensure("2024-02-01", "2024-02-28")
    assert not holey.range_ok("2024-02-09", "2024-02-11") and holey.range_ok("2024-02-12", "2024-02-20")


def pickled_without_loader(store):
    import pickle
    s2 = pickle.loads(pickle.dumps(store))
    return s2.loader is None and s2.dir == store.dir and s2._lru == {}


def test_leakage_guard():
    cfg = _bt_cfg(tempfile.mkdtemp(), history_start="2023-06-01")
    with pytest.raises(ValueError):
        bt_mod.check_no_leakage(cfg, [("2023-01-01", "2023-09-15")])
    bt_mod.check_no_leakage(replace(cfg, history_start="2023-10-01"), [("2023-01-01", "2023-09-15")])


@pytest.fixture(scope="module")
def tiny_run(big):
    tmp = tempfile.mkdtemp()
    cfg = _bt_cfg(tmp, timeframes=(1, 5), report_dir=tmp + "/rep")
    store = bt_mod.DataStore(tmp, SYMS, "1min", _loader(big), 1, 30)
    d = DiscoverConfig(k=8, max_terms=3, use_features=True)
    common = dict(entry_z=2.0, rr=1.0, fee_bps=0.5)
    strategies = {"formula": bt_mod.StrategySpec(None, dict(common)),
                  "formula b2": bt_mod.StrategySpec(None, dict(common, barrier_size=2.0)),
                  "smart clf": bt_mod.StrategySpec(None, dict(common, smart_trading=True, smart_mode="classifier", smart_min_events=40))}
    return cfg, store, d, strategies, bt_mod.run_backtest(cfg, store, d, strategies, {}, verbose=False)


def test_run_backtest_structure_and_shared_beam(tiny_run):
    cfg, store, d, strategies, res = tiny_run
    assert not res.errors, res.errors
    assert set(res.strategies) == set(strategies) and set(res.segments["tf"]) <= {1, 5}
    a, b = res.strategies["formula"].chunks, res.strategies["formula b2"].chunks
    assert len(a) == len(b) > 0 and (a["hedge"].to_numpy() == b["hedge"].to_numpy()).all()     # one beam search per fold, shared by every strategy
    assert (a["z_window"].to_numpy() == b["z_window"].to_numpy()).all()
    assert set(res.strategies["formula"].by_timeframe["tf"]) <= {1, 5}
    assert os.path.exists(os.path.join(cfg.report_dir, "summary.csv")) and any(f.endswith("_chunks.csv") for f in res.report_files)
    assert {"iteration", "segment", "tf", "chunk", "hl_train", "hl_oos", "vr_train", "vr_oos", "mean_r"} <= set(a.columns)


def test_run_backtest_parallel_equals_serial(big, tiny_run):
    cfg, store, d, strategies, serial = tiny_run
    two = bt_mod.run_backtest(replace(cfg, n_workers=2, report_dir=None), store, d, strategies, {}, verbose=False)
    pd.testing.assert_frame_equal(serial.summary, two.summary)
    s1, s2 = serial.strategies["formula"].trades.reset_index(drop=True), two.strategies["formula"].trades.reset_index(drop=True)
    assert len(s1) == len(s2) and np.allclose(s1["r_multiple"].to_numpy(), s2["r_multiple"].to_numpy())


def test_chunk_bootstrap_stats():
    r = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    st_ = bt_mod._stats(r, np.array([2.0, 0.0, 2.0, 0.0]), np.array([2.0, 2.0, 2.0, 2.0]), 500, 1)
    assert st_["mean_r"] == pytest.approx(0.5) and st_["n_chunks"] == 4 and st_["ci_lo"] <= st_["boot_mean"] <= st_["ci_hi"]
    assert bt_mod._stats(np.array([]), np.array([]), np.array([]), 10, 1)["n_trades"] == 0