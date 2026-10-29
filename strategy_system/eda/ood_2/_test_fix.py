#!/usr/bin/env python
"""
test_infonet_study.py -- direct-run tests for infonet_study / run_infonet_study / synthetic_loader_stub.

    python test_infonet_study.py            # everything that can run without your package (fake backends)
    python test_infonet_study.py -v         # verbose
    python test_infonet_study.py TestOracleControls.test_sign_flip_is_recovered   # one test

Put this file next to the three modules (or set INFONET_DIR).  No pytest needed, only numpy / pandas / sklearn.

TWO MODES
  FAKE (default)  infonet_study.py uses relative imports into your package (strategy.moe.*, exchange.loader).  Those
                  are replaced by small fakes (add_orderflow, encoder, segmenter, metrics) registered in sys.modules,
                  so what is tested is infonet_study's OWN logic: targets, costs, episodes, experts, cells, deviation
                  statistics, checkpointing, reporting, fold geometry, leakage.
  REAL            set INFONET_MODULE to the dotted path of the real module, e.g.
                      INFONET_MODULE=mypkg.research.infonet_study python test_infonet_study.py
                  The same tests then run against your real add_orderflow / encoders / segmenters / metrics (tests that
                  need fakes are skipped).  Add INFONET_REAL_DATA=1 to also run the real-data health check, which
                  prints per-fold volatility coverage -- this is the check for "volatility-filtered == test rows".

The regression tests pin down previously observed problems. FAKE mode runs
synthetic controls; REAL mode plus INFONET_REAL_DATA=1 also checks each market-data fold.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib
import importlib.util
import io
import math
import os
import pickle
import sys
import tempfile
import types
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(os.environ.get("INFONET_DIR", Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REAL_MODULE = os.environ.get("INFONET_MODULE")           # e.g. "mypkg.research.infonet_study"
FAKE = REAL_MODULE is None
FAKE_ROOT = "_infonet_fake"
PKG = f"{FAKE_ROOT}.a.b"                                   # infonet_study lives 3 levels below the package root


def _path(env: str, name: str) -> Path:
    p = Path(os.environ.get(env, HERE / name))
    if not p.exists() and (Path.cwd() / name).exists():
        p = Path.cwd() / name
    return p


# =================================================================================================================
# Fake backends (FAKE mode only).  Deliberately simple; they only need to honour the call signatures that
# infonet_study uses.
# =================================================================================================================
_ORDERFLOW_19 = [
    "trade_imbalance_5", "trade_notional_z_5", "spread_bps_z", "queue_imbalance", "microprice_displacement",
    "ofi_z_5", "depth_imbalance_5", "depth_imbalance_20", "bid_depth_change_z_5", "flow_price_efficiency_z_5",
    "trade_imbalance_1", "trade_imbalance_20", "buy_notional_z_5", "sell_notional_z_5", "ofi_z_1", "ofi_z_20",
    "depth_imbalance_1", "depth_imbalance_10", "ask_depth_change_z_5"]


def _fake_robust_zscore(s, baseline_window=1000, min_periods=100):
    s = pd.Series(s)
    mu = s.rolling(baseline_window, min_periods=min_periods).mean()
    sd = s.rolling(baseline_window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sd.where(sd > 1e-12)


def _fake_add_orderflow(df, windows=(1, 5, 20, 100), baseline_window=1000, min_periods=100):
    out = df.copy()
    idx = out.index
    z = lambda s: _fake_robust_zscore(s, baseline_window, min_periods)
    qty = lambda side, k: out["best_%s_qty" % side] if k == 1 else out[f"{side}_qty_{k}"]
    depth = lambda side, k: sum(qty(side, j) for j in range(1, k + 1))
    price = out["price"].astype(float)
    q = out["quantity"].astype(float).fillna(0.0)
    sgn = pd.Series(np.where(out["is_buyer_maker"].astype(bool), -1.0, 1.0), index=idx) * out["new_trade"].astype(float)
    notional = q * price
    out["aggressive_signed_notional"] = sgn * notional
    buy, sell = notional.where(sgn > 0, 0.0), notional.where(sgn < 0, 0.0)
    for w in windows:
        b, s = buy.rolling(w, min_periods=1).sum(), sell.rolling(w, min_periods=1).sum()
        out[f"trade_imbalance_{w}"] = ((b - s) / (b + s)).where((b + s) > 0)
    ofi = out["best_bid_qty"].diff() - out["best_ask_qty"].diff()
    for w in windows:
        out[f"ofi_z_{w}"] = z(ofi.rolling(w, min_periods=1).sum())
    mid = (out["best_bid_price"] + out["best_ask_price"]) / 2
    out["spread_bps"] = (out["best_ask_price"] - out["best_bid_price"]) / mid * 1e4
    out["spread_bps_z"] = z(out["spread_bps"])
    bq, aq = out["best_bid_qty"], out["best_ask_qty"]
    out["queue_imbalance"] = (bq - aq) / (bq + aq)
    micro = (out["best_bid_price"] * aq + out["best_ask_price"] * bq) / (bq + aq)
    out["microprice_displacement"] = (micro - mid) / mid * 1e4
    for k in (1, 5, 10, 20):
        bd, ad = depth("bid", k), depth("ask", k)
        out[f"depth_imbalance_{k}"] = (bd - ad) / (bd + ad)
        if k in (5, 20):
            out[f"bid_depth_{k}"], out[f"ask_depth_{k}"] = bd, ad
    out["trade_notional_z_5"] = z(notional.rolling(5, min_periods=1).sum())
    out["buy_notional_z_5"] = z(buy.rolling(5, min_periods=1).sum())
    out["sell_notional_z_5"] = z(sell.rolling(5, min_periods=1).sum())
    out["bid_depth_change_z_5"] = z(out["bid_depth_5"].diff(5))
    out["ask_depth_change_z_5"] = z(out["ask_depth_5"].diff(5))
    out["flow_price_efficiency_z_5"] = z(out["mid_price"].diff(5)
                                         / (out["aggressive_signed_notional"].rolling(5, min_periods=1).sum().abs() + 1.0))
    return out, list(_ORDERFLOW_19)


class _FakeEncoder:
    """Causal rolling-mean + PCA projection; accepts and ignores encoder_kwargs (epochs, batch_size...)."""

    def __init__(self, window=32, embedding_dim=8, feature_columns=None, **_):
        self.window, self.dim, self.cols = window, embedding_dim, list(feature_columns or [])

    def _smooth(self, Z):
        return pd.DataFrame(Z).rolling(self.window, min_periods=1).mean().to_numpy()

    def fit_encoder(self, frame):
        X = frame[self.cols].to_numpy(float)
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        Zs = self._smooth((X - self.mu) / self.sd)
        _, _, vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
        self.P = vt[: self.dim].T
        return self

    def transform(self, frame):
        Z = (frame[self.cols].to_numpy(float) - self.mu) / self.sd
        E = self._smooth(Z) @ self.P
        return pd.DataFrame(E, index=frame.index, columns=[f"e{i}" for i in range(E.shape[1])])


def _fake_make_encoder(name, **kw):
    return _FakeEncoder(**kw)


class _FakeSegmenter:
    backend = "kmeans"
    learn_novel = False

    def __init__(self, name="kmeans", max_regimes=6, min_dwell=50, backend_params=None, **_):
        self.k = (backend_params or {}).get("n_clusters", max_regimes)

    def fit(self, emb):
        from sklearn.cluster import KMeans
        self.km = KMeans(n_clusters=self.k, n_init=3, random_state=0).fit(emb.to_numpy(float))
        return self

    def predict(self, emb):
        lab = self.km.predict(emb.to_numpy(float))
        return pd.DataFrame({"latent_id": [f"latent_{int(x) + 1:03d}" for x in lab]}, index=emb.index)


def _fake_make_segmenter(name, **kw):
    return _FakeSegmenter(name, **kw)


def _fake_cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _fake_tdmi_fp(x, lag, bins, _none=None, surrogates=0):
    xs = x if isinstance(x, (list, tuple)) else [x]
    X = np.vstack([np.asarray(v, float) for v in xs])
    return np.array([abs(np.corrcoef(X[:-lag, j], X[lag:, j])[0, 1]) if X[:, j].std() > 0 else 0.0
                     for j in range(X.shape[1])])


def _fake_infonet(F, rr, block, alpha, min_features, min_run, n_surrogates, overlap):
    F, rr = np.asarray(F, float), np.asarray(rr, float)
    rr = rr if rr.ndim == 2 else rr[:, None]
    hits = np.zeros((rr.shape[1], F.shape[1]))
    for h in range(rr.shape[1]):
        ok = np.isfinite(rr[:, h])
        for j in range(F.shape[1]):
            if ok.sum() > 10 and F[ok, j].std() > 0 and rr[ok, h].std() > 0:
                hits[h, j] = float(abs(np.corrcoef(F[ok, j], rr[ok, h])[0, 1]) > 0.05)
    nb = max(1, len(F) // block)
    return SimpleNamespace(persistent=np.ones((nb, F.shape[1]), bool), feature_hits=hits,
                           score=float(hits.mean()), info=float(hits.sum()))


def _install_fake_package() -> types.ModuleType:
    def pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        return m

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    for n in ("", ".a", ".a.b", ".strategy", ".strategy.moe", ".strategy.moe.ood", ".strategy.moe.feature", ".exchange"):
        pkg(FAKE_ROOT + n)

    def _no_loader(*a, **k):
        raise RuntimeError("real load_L2 is not available in FAKE mode; pass DataConfig(loader=...)")

    mod(f"{FAKE_ROOT}.strategy.moe.ood.base_regime_encoder", make_encoder=_fake_make_encoder)
    mod(f"{FAKE_ROOT}.strategy.moe.ood.base_regime_segmenter", make_segmenter=_fake_make_segmenter)
    mod(f"{FAKE_ROOT}.strategy.moe.ood.metrics", cosine_similarity=_fake_cosine, infonet_estimate=_fake_infonet,
        resampled_distance_similarity=lambda a, b: float(1 / (1 + abs(np.nanmean(a) - np.nanmean(b)))),
        tdmi_fingerprint=_fake_tdmi_fp, tdmi_fingerprint_similarity=_fake_cosine)
    mod(f"{FAKE_ROOT}.strategy.moe.feature.orderflow", _robust_zscore=_fake_robust_zscore, add_orderflow=_fake_add_orderflow)
    mod(f"{FAKE_ROOT}.exchange.loader", load_L2=_no_loader)

    name = f"{PKG}.infonet_study"
    path = _path("INFONET_FILE", "infonet_study.py")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m                       # dataclasses needs the module registered before exec
    spec.loader.exec_module(m)
    return m


def _load_stub() -> types.ModuleType:
    path = _path("INFONET_STUB", "synthetic_loader_stub.py")
    spec = importlib.util.spec_from_file_location("synthetic_loader_stub", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["synthetic_loader_stub"] = m
    spec.loader.exec_module(m)
    return m


def _load_runner():
    if FAKE:
        name = f"{PKG}.run_infonet_study"
        spec = importlib.util.spec_from_file_location(name, _path("INFONET_RUNNER_FILE", "run_infonet_study.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    return importlib.import_module(REAL_MODULE.rsplit(".", 1)[0] + ".run_infonet_study")


S = _install_fake_package() if FAKE else importlib.import_module(REAL_MODULE)
STUB = _load_stub()

only_fake = unittest.skipUnless(FAKE, "needs the fake backends (FAKE mode only)")


# =================================================================================================================
# Shared fixtures
# =================================================================================================================
def quiet(fn, *a, **k):
    """Run fn, swallow its prints, return (result, printed_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*a, **k)
    return res, buf.getvalue()


def make_fold(world="sign_flip", seed=2, beta=0.35, mean_dwell=1500, rows_per_file=10_000, train=6, test=4,
              gap=0, decimate=2, fold=0, start=0, step=3):
    STUB.set_world(world, seed=seed, mean_dwell=mean_dwell, beta=beta, rows_per_file=rows_per_file)
    d = S.DataConfig(train_files=train, gap_files=gap, test_files=test, step_files=step, n_folds=1, start=start,
                     decimate=decimate, feature_sets=("20",), loader=STUB.load_L2)
    return S.load_fold(STUB.load_L2, fold, d), d


def oracle_cfg(**kw):
    base = dict(name="oracle", feature_set="20", encoder="pca", segmenter="oracle", horizons=(4, 16, 32),
                trade_horizon=16, block=100, min_episode_len=200, min_dwell=50, metrics=(),
                infonet_surrogates=19, tdmi_surrogates=19, max_test_episodes=6, max_ref_episodes=2)
    base.update(kw)
    return S.RunConfig(**base)


def cell_means(cells: pd.DataFrame) -> pd.DataFrame:
    """Trade-weighted mean net RR, rows = expert, columns = regime of the test episode."""
    w = cells.assign(w=cells["rr_net"] * cells["n_trades"]).groupby(["expert_latent", "assigned_latent"]).agg(
        w=("w", "sum"), n=("n_trades", "sum"))
    return (w["w"] / w["n"]).unstack("assigned_latent")


def vol_health(fd, horizons=(4, 16, 32), vol_window=100) -> dict:
    """Volatility / target coverage per segment.  This is what went wrong in folds 1-3 of the real run."""
    _, vol = S.stable_rr_targets(fd.mid, horizons, vol_window)
    fin = np.isfinite(vol)
    out = {}
    for code, name in ((0, "train"), (2, "test")):
        m = fd.seg == code
        out[f"{name}_rows"] = int(m.sum())
        out[f"{name}_vol_finite"] = float(fin[m].mean()) if m.any() else np.nan
        dm = np.diff(fd.mid[m]) if m.sum() > 1 else np.array([])
        out[f"{name}_mid_nan"] = int(np.isnan(fd.mid[m]).sum())
        out[f"{name}_mid_unique"] = int(len(np.unique(fd.mid[m][np.isfinite(fd.mid[m])]))) if m.any() else 0
        out[f"{name}_moving_share"] = float((dm != 0).mean()) if len(dm) else np.nan
    return out


# =================================================================================================================
# 1. Static facts and small pure helpers
# =================================================================================================================
class TestFeatureSets(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual({k: len(v) for k, v in S.FEATURE_SETS.items()}, {"10": 10, "20": 20, "40": 40})

    def test_nested_and_unique(self):
        f = S.FEATURE_SETS
        for k, v in f.items():
            self.assertEqual(len(v), len(set(v)), f"duplicate feature in set {k}")
        self.assertTrue(set(f["10"]) < set(f["20"]) < set(f["40"]))

    @only_fake
    def test_fake_orderflow_covers_the_19_names(self):
        self.assertEqual(_ORDERFLOW_19, S.CORE_10 + S.REST_OF_19)


class TestAutoMinFeatures(unittest.TestCase):
    def test_null_pass_rate_is_controlled(self):
        for d in (1, 10, 20, 40):
            m = S.auto_min_features(d, 0.05, 0.05)
            tail = sum(math.comb(d, k) * 0.05 ** k * 0.95 ** (d - k) for k in range(m, d + 1))
            self.assertLessEqual(tail, 0.05 + 1e-12, f"d={d}")
            if m > 1:     # and it is the SMALLEST such m
                tail_prev = sum(math.comb(d, k) * 0.05 ** k * 0.95 ** (d - k) for k in range(m - 1, d + 1))
                self.assertGreater(tail_prev, 0.05)

    def test_monotone_in_d(self):
        ms = [S.auto_min_features(d, 0.05) for d in range(1, 60)]
        self.assertEqual(ms, sorted(ms))


class TestLabelsAndRuns(unittest.TestCase):
    def test_as_labels_maps_noise_to_null(self):
        out = S._as_labels(np.array([-1.0, 0.0, 3.0]))
        self.assertEqual(list(out), ["null", "0.0", "3.0"])
        out = S._as_labels(np.array(["noise", "latent_001", np.nan], dtype=object))
        self.assertEqual(list(out), ["null", "latent_001", "null"])

    def test_runs_of(self):
        lab = np.array(["a", "a", "null", "b", "b", "b", "gap", "a"], dtype=object)
        self.assertEqual(S.runs_of(lab), [(0, 2, "a"), (3, 6, "b"), (7, 8, "a")])
        self.assertEqual(S.runs_of(lab, 1, 6), [(1, 2, "a"), (3, 6, "b")])
        self.assertEqual(S.runs_of(lab, 4, 4), [])
        self.assertEqual([k for *_, k in S.runs_of(lab, skip=())], ["a", "null", "b", "gap", "a"])

    def test_persistence_stats(self):
        lab = np.array(["a"] * 4 + ["b"] * 4 + ["a"] * 4 + ["null"] * 2, dtype=object)
        cfg = S.RunConfig(min_episode_len=3, min_dwell=1)
        st = S.persistence_stats(lab, np.arange(14, dtype=float), 0, 14, cfg, "x_")
        self.assertEqual(st["x_n_episodes"], 3)
        self.assertEqual(st["x_recurring_latents"], 1)               # only "a" recurs
        self.assertAlmostEqual(st["x_tradable_share"], 12 / 14)
        self.assertAlmostEqual(st["x_share_null"], 2 / 14)
        self.assertAlmostEqual(st["x_transitions_per_10k"], 3 / 14 * 1e4)


class TestTargets(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.p = 100 + np.cumsum(rng.normal(0, 0.05, 2000))

    def test_vol_finite_after_warmup(self):
        _, vol = S.stable_rr_targets(self.p, (4, 16), 100)
        self.assertTrue(np.isnan(vol[:49]).all())                    # min_periods = vol_window // 2
        self.assertTrue(np.isfinite(vol[60:]).all())
        self.assertTrue((vol[60:] > 0).all())

    def test_rr_formula(self):
        rr, vol = S.stable_rr_targets(self.p, (4, 16), 100)
        t = 500
        self.assertAlmostEqual(rr["rr_16"].iloc[t], (self.p[t + 16] - self.p[t]) / (vol[t] * 4.0), places=9)
        self.assertAlmostEqual(rr["rr_4"].iloc[t], (self.p[t + 4] - self.p[t]) / (vol[t] * 2.0), places=9)
        self.assertTrue(rr["rr_16"].iloc[-16:].isna().all())          # no forward window at the end

    def test_vol_is_causal(self):
        _, v1 = S.stable_rr_targets(self.p, (4,), 100)
        q = self.p.copy()
        q[1200:] += 5.0 * np.sin(np.arange(len(q) - 1200))
        _, v2 = S.stable_rr_targets(q, (4,), 100)
        np.testing.assert_array_equal(v1[:1200], v2[:1200])

    def test_flat_price_gives_no_volatility(self):
        _, vol = S.stable_rr_targets(np.full(800, 100.0), (4,), 100)
        self.assertTrue(np.isnan(vol).all())

    def test_nan_price_gives_no_volatility(self):
        rr, vol = S.stable_rr_targets(np.full(800, np.nan), (4,), 100)
        self.assertTrue(np.isnan(vol).all() and rr.isna().all().all())

    def test_tick_bounded_price_keeps_most_rows(self):
        rng = np.random.default_rng(1)
        steps = rng.choice([-0.01, 0.0, 0.01], size=5000, p=[0.03, 0.94, 0.03])
        _, vol = S.stable_rr_targets(100 + np.cumsum(steps), (4,), 100)
        self.assertGreater(np.isfinite(vol[100:]).mean(), 0.9)

    def test_forward_return_targets_are_cumulative_log_bps(self):
        mid = np.array([100.0, 101.0, 102.01, np.nan, 104.0, 105.0])
        y = S.forward_return_targets(mid, (1, 2))
        self.assertAlmostEqual(y.loc[0, "logret_1"], 1e4 * np.log(1.01), places=8)
        self.assertAlmostEqual(y.loc[0, "logret_2"], 1e4 * np.log(1.0201), places=8)
        self.assertTrue(np.isnan(y.loc[2, "logret_1"]))
        self.assertTrue(y["logret_2"].iloc[-2:].isna().all())

    def test_forward_targets_do_not_depend_on_volatility(self):
        y1 = S.forward_return_targets(self.p, (1, 2, 3, 4, 5))
        changed = self.p.copy()
        changed[1000:] *= 10
        y2 = S.forward_return_targets(changed, (1, 2, 3, 4, 5))
        pd.testing.assert_frame_equal(y1.iloc[:995], y2.iloc[:995])

    def test_trade_horizon_must_be_a_target_horizon(self):
        fd, _ = make_fold(rows_per_file=3000, train=3, test=2, step=2)
        with self.assertRaises((KeyError, AssertionError, ValueError)):
            quiet(S.run_simple_fold, fd, oracle_cfg(horizons=(4, 32), trade_horizon=16))


class TestStrategyHelpers(unittest.TestCase):
    def test_fit_expert_recovers_signal(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 3))
        X[:, 2] = 1.0                                                # zero-variance column must not blow up
        y = 2 * X[:, 0] + rng.normal(size=600)
        e = S.fit_expert(X, y, lam=0.1, q=0.6)
        self.assertIsNotNone(e)
        self.assertTrue(np.isfinite(e.w).all())
        self.assertGreater(e.w[0], 5 * abs(e.w[1]))
        self.assertGreater(np.corrcoef(e.predict(X), y)[0, 1], 0.8)
        self.assertAlmostEqual(np.mean(np.abs(e.predict(X)) >= e.thr), 0.4, delta=0.03)

    def test_multioutput_ridge_recovers_each_horizon_signal(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(900, 3))
        Y = np.column_stack([2 * X[:, 0] + rng.normal(0, .2, len(X)),
                             -X[:, 1] + rng.normal(0, .2, len(X))])
        expert = S.fit_forecast_expert(X, Y, lam=.01)
        self.assertIsNotNone(expert)
        pred = expert.predict(X)
        self.assertEqual(pred.shape, Y.shape)
        self.assertLess(np.mean(np.abs(pred - Y)), np.mean(np.abs(Y - Y.mean(0))))

    @only_fake
    def test_forecast_fold_reports_mae_and_baselines(self):
        fd, _ = make_fold("sign_flip", rows_per_file=4_000, train=5, test=3, step=3,
                          mean_dwell=500, decimate=1)
        cfg = oracle_cfg(strategy="forecast_mae", horizons=(1, 2, 3, 4, 5),
                         min_expert_rows=50, min_episode_len=100)
        result, _ = quiet(S.run_simple_fold, fd, cfg)
        metrics = result["forecast_metrics"]
        self.assertEqual(result["strategy"], "forecast_mae")
        self.assertEqual(set(metrics["horizon"]), {1, 2, 3, 4, 5})
        self.assertTrue({"mae_bps", "zero_mae_bps", "global_mae_bps", "skill_vs_zero"} <= set(metrics))
        self.assertTrue(np.isfinite(metrics["mae_bps"]).any())
        self.assertGreater(result["health"]["forecast_test_rows"], 0)

    def test_fit_expert_needs_enough_finite_rows(self):
        X = np.random.default_rng(0).normal(size=(40, 3))
        self.assertIsNone(S.fit_expert(X, X[:, 0], 0.1, 0.6))
        X = np.random.default_rng(0).normal(size=(200, 3))
        y = X[:, 0].copy()
        y[:160] = np.nan
        self.assertIsNone(S.fit_expert(X, y, 0.1, 0.6))

    def test_spearman(self):
        a = np.arange(50.0)
        self.assertAlmostEqual(S.spearman(a, a ** 3), 1.0)
        self.assertAlmostEqual(S.spearman(a, -a), -1.0)
        self.assertTrue(np.isnan(S.spearman(a, np.ones(50))))
        self.assertTrue(np.isnan(S.spearman(a[:5], a[:5])))

    def test_trade_rows_non_overlapping_and_net_of_cost(self):
        rows = np.arange(100)
        pred, rr, cost = np.full(100, 2.0), np.full(100, 0.5), np.full(100, 0.1)
        take, net, gross = S.trade_rows(pred, 1.0, rr, cost, rows, 10)
        self.assertEqual(list(take), list(range(0, 100, 10)))
        np.testing.assert_allclose(net, 0.4)
        take, net, _ = S.trade_rows(-pred, 1.0, rr, cost, rows, 10)           # short: sign flips, cost does not
        np.testing.assert_allclose(net, -0.6)

    def test_trade_rows_skips_nan_cost_and_small_signals(self):
        rows = np.arange(40)
        pred, rr, cost = np.full(40, 2.0), np.full(40, 0.5), np.full(40, 0.1)
        cost[10] = np.nan
        take, *_ = S.trade_rows(pred, 1.0, rr, cost, rows, 10)
        self.assertEqual(list(take[:3]), [0, 11, 21])
        pred[:] = 0.5
        take, net, _ = S.trade_rows(pred, 1.0, rr, cost, rows, 10)
        self.assertEqual(len(take), 0)

    def test_heterogeneity_homogeneous_vs_heterogeneous(self):
        rng = np.random.default_rng(3)
        n, k = np.full(60, 40.0), 60
        sd = np.ones(k)
        homo = S.heterogeneity(rng.normal(0.0, 1 / np.sqrt(40), k), n, sd)
        self.assertLess(homo["tau"], 0.12)
        self.assertLess(homo["I2"], 0.4)
        het = S.heterogeneity(rng.normal(0.0, 0.5, k) + rng.normal(0, 1 / np.sqrt(40), k), n, sd)
        self.assertGreater(het["tau"], 0.35)
        self.assertLess(het["tau"], 0.7)
        self.assertGreater(het["I2"], 0.6)
        few = S.heterogeneity([0.1, 0.2], [40, 40], [1, 1])
        self.assertTrue(np.isnan(few["tau"]) and few["k"] == 2)

    def test_clean_sd_removes_sampling_noise(self):
        rng = np.random.default_rng(4)
        n = np.full(300, 11.0)
        g = pd.DataFrame({"rr_net": rng.normal(0, 1 / np.sqrt(11), 300), "rr_sd": 1.0, "n_trades": n})
        self.assertGreater(S._sd(g["rr_net"].to_numpy()), 0.25)
        self.assertLess(S._clean_sd(g), 0.5 * S._sd(g["rr_net"].to_numpy()))
        self.assertTrue(np.isnan(S._clean_sd(g.iloc[:2])))

    def test_wmean_ci(self):
        m, lo, hi = S._wmean_ci([1.0, 3.0], [1, 3])
        self.assertAlmostEqual(m, 2.5)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))               # < 3 episodes: no CI
        m, lo, hi = S._wmean_ci(np.ones(20), np.full(20, 5))
        self.assertEqual((m, lo, hi), (1.0, 1.0, 1.0))
        m, lo, hi = S._wmean_ci(np.random.default_rng(0).normal(0.5, 0.2, 80), np.full(80, 10))
        self.assertLess(lo, m < hi or m)
        self.assertTrue(lo < m < hi)

    def test_tail_average(self):
        v = np.arange(101.0)
        self.assertAlmostEqual(S._tail_average(v, upper=False), np.mean(v[v <= 5]))
        self.assertAlmostEqual(S._tail_average(v, upper=True), np.mean(v[v >= 95]))
        self.assertTrue(np.isnan(S._tail_average(np.array([np.nan]), upper=True)))


class TestCellTables(unittest.TestCase):
    @staticmethod
    def _cells(own_mu=0.2, cross_mu=-0.1, own_sd=0.1, cross_sd=0.3, per=12, seed=0):
        rng = np.random.default_rng(seed)
        rows, ep = [], 0
        for a in ("A", "B"):
            for _ in range(per):
                for e in ("A", "B"):
                    mu, sd = (own_mu, own_sd) if e == a else (cross_mu, cross_sd)
                    rows.append(dict(ep=ep, start=ep * 10, end=ep * 10 + 9, assigned_latent=a, expert_latent=e,
                                     n_trades=30, rr_net=rng.normal(mu, sd), rr_sd=1.0))
                ep += 1
        return pd.DataFrame(rows)

    def test_edge_matrix_marks_own_regime_only(self):
        t = S._edge_matrix(self._cells(per=3))
        self.assertEqual(list(t.index), ["A (model)", "B (model)"])
        self.assertEqual(list(t.columns), ["A (n=3)", "B (n=3)"])
        self.assertTrue(t.loc["A (model)", "A (n=3)"].endswith("*"))
        self.assertTrue(t.loc["B (model)", "B (n=3)"].endswith("*"))
        self.assertFalse(t.loc["A (model)", "B (n=3)"].endswith("*"))

    def test_fmt_cell(self):
        self.assertIn("[n<3]", S._fmt_cell(0.1, np.nan, np.nan, False))
        self.assertTrue(S._fmt_cell(0.1, 0.0, 0.2, True).endswith(" *"))

    def test_episode_correlation(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=10)
        cells = pd.DataFrame(
            [dict(ep=i, expert_latent=e, rr_net=v) for i in range(10)
             for e, v in (("A", a[i]), ("B", 2 * a[i] + 1), ("C", 0.5))])
        m = S._episode_correlation(cells)
        self.assertAlmostEqual(m.loc["A", "B"], 1.0)
        self.assertEqual(m.loc["A", "C"], "constant")
        self.assertEqual(m.loc["C", "C"], "constant")
        m2 = S._episode_correlation(cells[cells["ep"] < 2])
        self.assertEqual(m2.loc["A", "B"], "n<3")

    def test_run_deviation_prefers_consistent_own_regime(self):
        d = S.run_deviation({"folds": [{"cells": self._cells()}]})
        self.assertTrue(d["eligible"], d["why"])
        self.assertEqual(d["own_eps"], 24)
        self.assertLess(d["own_sd"], d["cross_sd"])
        self.assertGreater(d["own_exp"], d["cross_exp"])

    def test_run_deviation_not_rankable_cases(self):
        d = S.run_deviation({"folds": [{"cells": pd.DataFrame()}, {"cells": None}, {}]})
        self.assertFalse(d["eligible"])
        self.assertIn("regime", d["why"])
        d = S.run_deviation({"folds": [{"cells": self._cells(per=2)}]})
        self.assertFalse(d["eligible"])
        self.assertIn("own-regime episodes", d["why"])


# =================================================================================================================
# 2. Data -> features
# =================================================================================================================
class TestPrepareL2(unittest.TestCase):
    def test_new_trade_flag_and_quantity(self):
        n = np.nan
        raw = pd.DataFrame({
            "best_bid_price": [1.0] * 7, "best_ask_price": [1.1] * 7, "best_bid_qty": [5.0] * 7, "best_ask_qty": [4.0] * 7,
            "agg_trade_id": [n, 1, n, 2, n, 3, n], "price": [n, 1.1, n, 1.0, n, 1.1, n],
            "is_buyer_maker": [n, 0, n, 1, n, 0, n], "quantity": [n, 2.0, n, 3.0, n, 1.0, n]})
        df, first = S.prepare_l2(raw)
        self.assertEqual(first, 1)                                   # leading row without trade context dropped
        self.assertEqual(len(df), 6)
        self.assertEqual(list(df["new_trade"]), [False, False, True, False, True, False])
        self.assertEqual(list(df["quantity"]), [0.0, 0.0, 3.0, 0.0, 1.0, 0.0])    # ffilled rows do not recount trades
        self.assertEqual(df["is_buyer_maker"].dtype, bool)
        np.testing.assert_allclose(df["mid_price"], 1.05)
        self.assertTrue((df["price_close"] == df["price"]).all())

    def test_stale_mid_is_rebuilt_from_live_top_of_book(self):
        raw = pd.DataFrame({
            "best_bid_price": [100.0] * 4, "best_ask_price": [100.1] * 4,
            "best_bid_qty": [5.0] * 4, "best_ask_qty": [4.0] * 4,
            "agg_trade_id": [1.0, np.nan, 2.0, np.nan], "price": [100.1, np.nan, 100.0, np.nan],
            "is_buyer_maker": [0.0, np.nan, 1.0, np.nan], "quantity": [1.0, np.nan, 2.0, np.nan],
            "mid_price": [50.0] * 4,
        })
        df, _ = S.prepare_l2(raw)
        np.testing.assert_allclose(df["mid_price"], 100.05)

    def test_bookless_rows_use_last_trade_price_instead_of_stale_quote(self):
        raw = pd.DataFrame({
            "best_bid_price": [100.0, np.nan, np.nan, np.nan],
            "best_ask_price": [100.1, np.nan, np.nan, np.nan],
            "best_bid_qty": [5.0, np.nan, np.nan, np.nan],
            "best_ask_qty": [4.0, np.nan, np.nan, np.nan],
            "agg_trade_id": [1.0, 2.0, 3.0, 4.0], "price": [100.0, 100.1, 100.2, 100.3],
            "is_buyer_maker": [0.0, 0.0, 0.0, 0.0], "quantity": [1.0] * 4,
        })
        df, _ = S.prepare_l2(raw)
        np.testing.assert_allclose(df["mid_price"], [100.05, 100.1, 100.2, 100.3])


class TestFoldBuilding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        STUB.set_world("sign_flip", seed=1, mean_dwell=1500, beta=0.35, rows_per_file=10_000)
        cls.raw = STUB.load_L2(file_limit=3, start=0)
        cls.seg = np.r_[np.zeros(20_000, int), np.full(2_000, 1), np.full(8_000, 2)]

    def test_fold_invariants(self):
        fd = S.build_fold_features(self.raw, self.seg, ("20", "decoy40"), decimate=1)
        base = fd.frames["_base"]
        n = len(base)
        self.assertTrue(len(fd.mid) == len(fd.spread) == len(fd.ts_sec) == len(fd.seg) == n)
        self.assertEqual(fd.meta["rows"], n)
        self.assertEqual(base.shape[1], 40)
        self.assertTrue(np.isfinite(base.to_numpy()).all(), "FoldData frames must be finite")
        self.assertTrue(np.isfinite(fd.mid).all() and (fd.spread > 0).all())
        self.assertTrue((np.diff(fd.ts_sec) >= 0).all())
        self.assertEqual(list(np.unique(fd.seg)), [0, 1, 2])
        self.assertTrue((np.diff(fd.seg) >= 0).all())
        self.assertIsNotNone(fd.truth)                                # regime_truth passes through from the stub
        self.assertEqual(len(fd.truth), n)

    def test_decimation(self):
        a = S.build_fold_features(self.raw, self.seg, ("20",), decimate=1)
        b = S.build_fold_features(self.raw, self.seg, ("20",), decimate=4)
        self.assertAlmostEqual(len(b.mid) / len(a.mid), 0.25, delta=0.01)
        np.testing.assert_array_equal(b.mid, a.mid[::4])              # decimation happens AFTER feature calculation

    def test_bookless_file_does_not_inherit_stale_book_features(self):
        raw = self.raw.copy()
        missing_from = 2_000
        quote_cols = [c for c in raw.columns if c in ("best_bid_price", "best_ask_price")
                      or c.startswith(("best_bid_qty", "best_ask_qty", "bid_price_", "bid_qty_",
                                       "ask_price_", "ask_qty_"))]
        raw.loc[missing_from:, quote_cols] = np.nan
        fd = S.build_fold_features(raw, self.seg, ("20",), decimate=1,
                                   baseline_window=100, min_periods=20)
        pos = missing_from + 200 - fd.meta["dropped_warmup"]
        frame = fd.frames["_base"]
        for col in ("spread_bps_z", "queue_imbalance", "microprice_displacement", "ofi_z_5", "depth_imbalance_5"):
            self.assertEqual(float(frame.loc[pos, col]), 0.0, col)
        self.assertTrue(np.isnan(fd.spread[pos]))

    def test_decoy_features_are_shifted_copies(self):
        fd = S.build_fold_features(self.raw, self.seg, ("20", "decoy40"), decimate=1)
        dec, n = fd.frames["decoy40"], len(fd.mid)
        self.assertEqual(dec.shape[1], 40)
        for i in (1, 2, 3):
            np.testing.assert_array_equal(dec[f"decoy{i}_queue_imbalance"].to_numpy(),
                                          np.roll(dec["queue_imbalance"].to_numpy(), n * i // 4))

    def test_features_are_causal(self):
        """Changing raw rows after r0 must not change any feature row before r0 (no look-ahead in the pipeline)."""
        a = S.build_fold_features(self.raw, self.seg, ("20",), decimate=1)
        r0 = int(0.8 * len(self.raw))
        raw2 = self.raw.copy()
        cols = [c for c in raw2.columns if c.startswith(("best_", "bid_", "ask_")) or c in ("price", "quantity", "mid_price")]
        raw2.loc[raw2.index[r0:], cols] = raw2.loc[raw2.index[r0:], cols].to_numpy() * 1.37
        b = S.build_fold_features(raw2, self.seg, ("20",), decimate=1)
        keep = int(0.7 * len(a.mid))
        pd.testing.assert_frame_equal(a.frames["_base"].iloc[:keep], b.frames["_base"].iloc[:keep])

    def test_load_fold_geometry(self):
        STUB.set_world("null", seed=1, mean_dwell=1500, rows_per_file=2_000)
        calls = []

        def rec(file_limit, start, **_):
            calls.append((start, file_limit))
            return STUB.load_L2(file_limit=file_limit, start=start)

        d = S.DataConfig(train_files=3, gap_files=1, test_files=2, step_files=2, n_folds=3, start=1, decimate=2,
                         feature_sets=("20",), loader=rec)
        fd = S.load_fold(rec, 2, d)
        self.assertEqual(calls, [(5, 3), (8, 1), (9, 2)])             # s = start + fold*step; gap and test follow train
        self.assertEqual(list(np.unique(fd.seg)), [0, 1, 2])
        self.assertTrue((np.diff(fd.seg) >= 0).all())
        ratio = np.array([(fd.seg == c).sum() for c in (0, 1, 2)]) / len(fd.seg)
        self.assertAlmostEqual(ratio[0], 3 / 6, delta=0.12)
        self.assertAlmostEqual(ratio[2], 2 / 6, delta=0.12)

    def test_no_empty_fold_without_gap(self):
        STUB.set_world("null", seed=1, mean_dwell=1500, rows_per_file=2_000)
        d = S.DataConfig(train_files=2, gap_files=0, test_files=1, step_files=1, n_folds=1, decimate=1,
                         feature_sets=("20",), loader=STUB.load_L2)
        fd = S.load_fold(STUB.load_L2, 0, d)
        self.assertNotIn(1, set(fd.seg))


# =================================================================================================================
# 3. Synthetic loader stub (the test fixture itself must be trustworthy)
# =================================================================================================================
class TestSyntheticStub(unittest.TestCase):
    def setUp(self):
        STUB.set_world("sign_flip", seed=3, mean_dwell=1500, beta=0.35, rows_per_file=4_000)

    def test_schema_and_deterministic(self):
        a = STUB.load_L2(file_limit=2, start=1)
        STUB.set_world("sign_flip", seed=3, mean_dwell=1500, beta=0.35, rows_per_file=4_000)
        b = STUB.load_L2(file_limit=2, start=1)
        pd.testing.assert_frame_equal(a, b)
        need = {"timestamp", "best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty", "price", "quantity",
                "is_buyer_maker", "agg_trade_id", "mid_price", "regime_truth", "bid_price_20", "ask_qty_20"}
        self.assertTrue(need <= set(a.columns))
        self.assertEqual(len(a), 8_000)
        self.assertTrue((a["best_ask_price"] > a["best_bid_price"]).all())
        self.assertTrue((a["timestamp"].diff().dropna() > pd.Timedelta(0)).all())

    def test_regimes_are_persistent_and_three(self):
        t = STUB.load_L2(file_limit=10, start=0)["regime_truth"].to_numpy()
        self.assertEqual(set(np.unique(t)), {0, 1, 2})
        self.assertLess((np.diff(t) != 0).sum(), 80)                  # ~dwell 1500 on 40k rows
        self.assertTrue((np.diff(t)[np.diff(t) != 0] != 0).all())

    def test_worlds_differ_only_in_signal(self):
        def ic(world):
            STUB.set_world(world, seed=3, mean_dwell=1500, beta=0.35, rows_per_file=4_000)
            d = STUB.load_L2(file_limit=10, start=0)
            mid = d["mid_price"].to_numpy()
            fwd = np.r_[mid[16:] - mid[:-16], np.full(16, np.nan)]
            qi = ((d["best_bid_qty"] - d["best_ask_qty"]) / (d["best_bid_qty"] + d["best_ask_qty"])).to_numpy()
            out = {}
            for r in (0, 1, 2):
                m = (d["regime_truth"].to_numpy() == r) & np.isfinite(fwd)
                out[r] = np.corrcoef(qi[m], fwd[m])[0, 1]
            return out
        sf, nl = ic("sign_flip"), ic("null")
        self.assertGreater(sf[0], 0.05)
        self.assertLess(sf[1], -0.05)
        self.assertLess(abs(sf[2]), 0.05)
        self.assertTrue(all(abs(v) < 0.05 for v in nl.values()))

    def test_requests_are_consistent_across_calls(self):
        """Fold k loads train / gap / test with separate calls that must share one deterministic master.

        Out-of-range requests may be rejected explicitly by the stub.
        """
        STUB.set_world("sign_flip", seed=3, mean_dwell=1500, beta=0.35, rows_per_file=1_000)
        try:
            a = STUB.load_L2(file_limit=10, start=12)
            b = STUB.load_L2(file_limit=4, start=22)
            whole = STUB.load_L2(file_limit=14, start=12)
        except ValueError:
            return                                                    # explicit refusal is fine
        glued = pd.concat([a, b], ignore_index=True)
        np.testing.assert_array_equal(glued["mid_price"].to_numpy(), whole["mid_price"].to_numpy(),
                                      "train and test pieces come from different synthetic worlds")
        np.testing.assert_array_equal(glued["regime_truth"].to_numpy(), whole["regime_truth"].to_numpy())

    def test_in_range_requests_are_consistent(self):
        STUB.set_world("sign_flip", seed=3, mean_dwell=1500, beta=0.35, rows_per_file=1_000)
        a = STUB.load_L2(file_limit=6, start=2)
        b = STUB.load_L2(file_limit=4, start=8)
        whole = STUB.load_L2(file_limit=10, start=2)
        pd.testing.assert_frame_equal(pd.concat([a, b], ignore_index=True), whole.reset_index(drop=True))


# =================================================================================================================
# 4. Regime discovery
# =================================================================================================================
class TestDiscoverRegimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fd, _ = make_fold(rows_per_file=5_000, train=4, test=2, step=2, mean_dwell=800, decimate=2)
        cls.frame = cls.fd.frames["_base"][S.FEATURE_SETS["20"]]
        cls.n_fit = max(2, int((cls.fd.seg == 0).sum() // 2))

    def _discover(self, cfg, truth=None):
        return quiet(S.discover_regimes, self.frame, self.fd.seg, cfg, self.n_fit, truth=truth)

    def test_oracle_uses_truth_and_marks_gap(self):
        fd, _ = make_fold(rows_per_file=5_000, train=3, gap=1, test=2, step=2, mean_dwell=800, decimate=2)
        frame = fd.frames["_base"][S.FEATURE_SETS["20"]]
        (labels, emb), _ = quiet(S.discover_regimes, frame, fd.seg, oracle_cfg(), 100, truth=fd.truth)
        self.assertEqual(len(labels), len(frame))
        self.assertTrue((labels[fd.seg == 1] == "gap").all())
        nongap = fd.seg != 1
        self.assertTrue(all(str(x).startswith("truth_") for x in labels[nongap]))
        self.assertEqual(len(emb), len(frame))

    def test_oracle_requires_truth(self):
        with self.assertRaises(AssertionError):
            self._discover(oracle_cfg(), truth=None)

    @staticmethod
    def _hdbscan_factory(sizes_tried, clusters_by_size):
        class Seg:
            backend = "hdbscan"
            learn_novel = False

            def __init__(self, size):
                self.size = size

            def fit(self, emb):
                if clusters_by_size(self.size) == 0:
                    raise ValueError("HDBSCAN found no clusters")
                return self

            def predict(self, emb):
                k = clusters_by_size(self.size)
                col = emb.iloc[:, 0].to_numpy()
                ids = np.where(col > np.median(col), "1", "0") if k >= 2 else np.where(col > np.percentile(col, 80), "-1", "0")
                return pd.DataFrame({"latent_id": ids}, index=emb.index)

        def make(name, **kw):
            size = kw["backend_params"]["min_cluster_size"]
            sizes_tried.append(size)
            return Seg(size)
        return make

    def test_hdbscan_retries_with_smaller_clusters_until_enough_latents(self):
        tried = []
        cfg = oracle_cfg(segmenter="hdbscan", min_latents=2)
        with mock.patch.object(S, "make_segmenter", self._hdbscan_factory(tried, lambda s: 0 if s >= 100 else (1 if s > 40 else 2))):
            (labels, _), _ = self._discover(cfg)
        self.assertEqual(tried, [200, 100, 40])                       # stops as soon as >= min_latents latents exist
        self.assertEqual(len(set(labels) - {"null", "gap"}), 2)

    def test_hdbscan_with_no_clusters_records_null_without_crashing(self):
        tried = []
        with mock.patch.object(S, "make_segmenter", self._hdbscan_factory(tried, lambda s: 0)):
            (labels, _), out = self._discover(oracle_cfg(segmenter="hdbscan"))
        self.assertEqual(tried, [200, 100, 40, 10])
        self.assertTrue(set(labels) <= {"null", "gap"})
        self.assertIn("no clusters", out)

    def test_hdbscan_single_cluster_is_kept_but_cannot_be_ranked(self):
        """The situation seen in the real run: 1 latent -> 1 expert -> no cross-regime comparison is possible."""
        tried = []
        with mock.patch.object(S, "make_segmenter", self._hdbscan_factory(tried, lambda s: 1)):
            (labels, _), out = self._discover(oracle_cfg(segmenter="hdbscan", min_latents=2))
        self.assertEqual(tried, [200, 100, 40, 10])                   # kept shrinking, then kept the best it had
        self.assertEqual(len(set(labels) - {"null", "gap"}), 1)
        self.assertIn("wanted >= 2", out)

    def test_hdbscan_unrelated_errors_are_not_swallowed(self):
        class Boom:
            backend = "hdbscan"

            def fit(self, emb):
                raise ValueError("shape mismatch in backend")
        with mock.patch.object(S, "make_segmenter", lambda name, **kw: Boom()):
            with self.assertRaises(ValueError):
                self._discover(oracle_cfg(segmenter="hdbscan"))

    @only_fake
    def test_kmeans_labels_train_and_test_rows(self):
        (labels, _), _ = self._discover(oracle_cfg(segmenter="kmeans", n_latents=4))
        seg = self.fd.seg
        for code in (0, 2):
            self.assertGreater((labels[seg == code] != "null").mean(), 0.9)
        self.assertLessEqual(len(set(labels) - {"null", "gap"}), 4)

    def test_discovery_uses_train_rows_only_for_fitting(self):
        seen = {}

        class Spy(_FakeEncoder if FAKE else object):
            pass
        if not FAKE:
            self.skipTest("spy encoder only in FAKE mode")
        orig = _FakeEncoder.fit_encoder

        def spy(self_, frame):
            seen["rows"] = len(frame)
            return orig(self_, frame)
        with mock.patch.object(_FakeEncoder, "fit_encoder", spy):
            self._discover(oracle_cfg(), truth=self.fd.truth)
        self.assertEqual(seen["rows"], self.n_fit)                    # only the discovery half of TRAIN


# =================================================================================================================
# 5. End-to-end controls on the synthetic worlds (this is the scientific validation of the design)
# =================================================================================================================
class TestOracleControls(unittest.TestCase):
    """Ground-truth regimes from the stub: if the pipeline cannot recover a known sign-flip here, nothing real can be
    trusted.  Oracle labels isolate the strategy / episode / cell machinery from encoder+segmenter quality."""

    @classmethod
    def setUpClass(cls):
        cls.fd_sf, _ = make_fold("sign_flip")
        cls.res_sf, _ = quiet(S.run_simple_fold, cls.fd_sf, oracle_cfg())
        cls.fd_null, _ = make_fold("null")
        cls.res_null, _ = quiet(S.run_simple_fold, cls.fd_null, oracle_cfg())

    def test_result_contract(self):
        r = self.res_sf
        for key in ("fold", "recurring_latents", "expectancy", "persistence", "parity", "correlation", "cells",
                    "sim_pairs", "cost_percentiles", "health", "rejected_costs", "vol_filtered_rows"):
            self.assertIn(key, r)
        self.assertGreaterEqual(len(r["recurring_latents"]), 2)
        c = r["cells"]
        self.assertTrue({"ep", "start", "end", "assigned_latent", "expert_latent", "n_trades", "rr_net", "rr_sd"} <= set(c.columns))
        self.assertTrue((c["n_trades"] >= 1).all())
        self.assertTrue((c["end"] - c["start"] >= 200).all())          # only tradable episodes
        self.assertTrue(c["assigned_latent"].isin(r["recurring_latents"]).all())

    def test_test_episodes_live_in_the_test_segment(self):
        te = np.flatnonzero(self.fd_sf.seg == 2)
        c = self.res_sf["cells"]
        self.assertTrue((c["start"] >= te[0]).all() and (c["end"] <= te[-1] + 1).all())

    def test_volatility_coverage_is_healthy_on_synthetic_data(self):
        h = vol_health(self.fd_sf)
        self.assertGreater(h["test_vol_finite"], 0.9, h)
        self.assertGreater(h["train_vol_finite"], 0.9, h)

    def test_costs_are_positive_and_bounded(self):
        p = self.res_sf["cost_percentiles"]
        self.assertGreater(p["p50"], 0)
        self.assertLessEqual(p["p100"], 5.0)

    def test_sign_flip_is_recovered(self):
        """A is +flow, B is -flow, C is noise.  Each expert earns its own regime and LOSES in the flipped one."""
        m = cell_means(self.res_sf["cells"])
        a, b, c = "truth_0", "truth_1", "truth_2"
        self.assertIn(a, m.index)
        self.assertIn(b, m.index)
        self.assertGreater(m.loc[a, a], 0.05, f"\n{m}")
        self.assertGreater(m.loc[b, b], 0.05, f"\n{m}")
        self.assertLess(m.loc[a, b], -0.05, f"\n{m}")                  # same information, opposite sign
        self.assertLess(m.loc[b, a], -0.05, f"\n{m}")
        self.assertGreater(m.loc[a, a] - m.loc[a, b], 0.2)
        if c in m.index:
            self.assertLess(m.loc[c, c], 0.05, f"\n{m}")               # no signal in C (cost makes it negative)

    def test_null_world_has_no_edge(self):
        m = cell_means(self.res_null["cells"])
        self.assertLess(np.nanmax(m.to_numpy()), 0.1, f"\n{m}")

    def test_deviation_report_row_is_rankable_and_prefers_own_regime(self):
        d = S.run_deviation({"folds": [self.res_sf]})
        self.assertGreater(d["own_exp"], d["cross_exp"])

    def test_expert_fit_uses_only_identification_rows(self):
        """Experts must be fitted on the second half of TRAIN only (never on discovery-half or test rows)."""
        tr = np.flatnonzero(self.fd_sf.seg == 0)
        r = self.res_sf
        self.assertEqual(r["discovery_rows"], max(2, len(tr) // 2))
        self.assertEqual(r["identification_rows"], len(tr) - r["discovery_rows"])

    @only_fake
    def test_target_rows_never_use_prices_from_the_test_segment(self):
        """LEAKAGE CHECK.

        rr_h at train row t uses the price at t+h.  For the last h train rows that price lies in the TEST segment, and
        Perturbing test prices must leave expert weights bit-identical because targets crossing a segment boundary
        are purged before fitting.
        """
        def weights(fd):
            got = []
            orig = S.fit_expert

            def spy(X, y, lam, q):
                e = orig(X, y, lam, q)
                got.append(None if e is None else e.w.copy())
                return e
            with mock.patch.object(S, "fit_expert", spy):
                quiet(S.run_simple_fold, fd, oracle_cfg())
            return got
        base = weights(self.fd_sf)
        mid = self.fd_sf.mid.copy()
        te = self.fd_sf.seg == 2
        mid[te] = mid[te] + np.cumsum(np.random.default_rng(0).normal(0, 0.05, te.sum()))
        pert = weights(dataclasses.replace(self.fd_sf, mid=mid))
        self.assertEqual(len(base), len(pert))
        for w0, w1 in zip(base, pert):
            np.testing.assert_array_equal(w0, w1, "expert weights changed when only TEST prices changed -> label leakage")

    def test_perturbing_test_prices_changes_only_test_results(self):
        """Complement of the leakage check: the test cells MUST react to test prices (the check has teeth)."""
        mid = self.fd_sf.mid.copy()
        te = self.fd_sf.seg == 2
        mid[te] = mid[te] + np.cumsum(np.random.default_rng(0).normal(0, 0.05, te.sum()))
        res2, _ = quiet(S.run_simple_fold, dataclasses.replace(self.fd_sf, mid=mid), oracle_cfg())
        self.assertFalse(np.allclose(cell_means(res2["cells"]).to_numpy(), cell_means(self.res_sf["cells"]).to_numpy(),
                                     equal_nan=True))


class TestSimilarityRows(unittest.TestCase):
    """The InfoNET / cosine / TDMI table was EMPTY in the real run.  This test fails loudly (with the run log) when
    run_simple_fold produces no similarity pairs; run it in REAL mode to test the real metric functions."""

    @classmethod
    def setUpClass(cls):
        cls.fd, _ = make_fold("sign_flip", rows_per_file=5_000, train=4, test=2, step=2, mean_dwell=800)
        cls.cfg = oracle_cfg(metrics=("infonet", "cosine", "tdmi"), min_episode_len=150)
        cls.res, cls.log = quiet(S.run_simple_fold, cls.fd, cls.cfg)

    def test_similarity_pairs_are_produced(self):
        sp = self.res["sim_pairs"]
        self.assertGreater(len(sp), 0, f"no similarity rows -- run log:\n{self.log}")

    def test_similarity_pair_structure(self):
        sp = self.res["sim_pairs"]
        if not len(sp):
            self.skipTest("no similarity rows (see test_similarity_pairs_are_produced)")
        for m in self.cfg.metrics:
            self.assertIn(m, sp.columns)
        n_refs = len(self.res["recurring_latents"])
        self.assertTrue((sp.groupby("ep").size() <= n_refs).all())
        self.assertTrue(set(sp["ref_latent"]) <= set(self.res["recurring_latents"]))
        self.assertLessEqual(sp["ep"].nunique(), self.cfg.max_test_episodes)
        self.assertTrue(np.isfinite(sp[list(self.cfg.metrics)].to_numpy(float)).any())

    def test_run_similarity_summary(self):
        rows = S.run_similarity({"cfg": self.cfg, "folds": [self.res]})
        for r in rows:
            self.assertTrue({"metric", "episodes", "own", "other", "gap", "top1", "chance"} <= set(r))
            self.assertTrue(0.0 <= r["top1"] <= 1.0 and 0.0 < r["chance"] <= 1.0)


# =================================================================================================================
# 6. Study driver: checkpointing, resume, failure handling, report
# =================================================================================================================
class TestStudyDriver(unittest.TestCase):
    def setUp(self):
        STUB.set_world("sign_flip", seed=2, mean_dwell=800, beta=0.35, rows_per_file=3_000)
        # Keep test artifacts in the writable workspace; OS temp is outside
        # the sandbox on some Windows runners and makes atomic replace fail.
        self.tmp = tempfile.TemporaryDirectory(dir=HERE)
        self.ckpt = os.path.join(self.tmp.name, "ck.pkl")
        self.calls = []

        def loader(file_limit, start, **kw):
            self.calls.append((start, file_limit))
            return STUB.load_L2(file_limit=file_limit, start=start)
        self.loader = loader
        self.data = S.DataConfig(train_files=3, test_files=2, step_files=2, n_folds=2, decimate=2,
                                 feature_sets=("20",), loader=loader)
        self.runs = [oracle_cfg(name="r1", min_episode_len=150)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_all_folds_and_writes_checkpoint(self):
        res, _ = quiet(S.run_simple_study, self.data, self.runs, self.ckpt)
        self.assertEqual(set(res), {"r1"})
        self.assertEqual([f["fold"] for f in res["r1"]["folds"]], [0, 1])
        self.assertTrue(os.path.exists(self.ckpt))
        self.assertFalse(os.path.exists(self.ckpt + ".tmp"))          # atomic replace leaves no temp file
        with open(self.ckpt, "rb") as fh:
            saved = pickle.load(fh)
        self.assertEqual(set(saved), {"key", "results"})
        self.assertIs(res["r1"]["cfg"], self.runs[0])

    def test_resume_does_not_reload_or_recompute(self):
        quiet(S.run_simple_study, self.data, self.runs, self.ckpt)
        self.calls.clear()
        with mock.patch.object(S, "run_simple_fold", side_effect=AssertionError("recomputed")):
            res, out = quiet(S.run_simple_study, self.data, self.runs, self.ckpt)
        self.assertEqual(self.calls, [])
        self.assertIn("Resuming 2", out)
        self.assertEqual(len(res["r1"]["folds"]), 2)

    def test_checkpoint_from_other_settings_is_refused(self):
        quiet(S.run_simple_study, self.data, self.runs, self.ckpt)
        with self.assertRaises(ValueError):
            quiet(S.run_simple_study, replace(self.data, n_folds=1), self.runs, self.ckpt)
        with self.assertRaises(ValueError):
            quiet(S.run_simple_study, self.data, [replace(self.runs[0], block=50)], self.ckpt)

    def test_failed_run_is_recorded_then_retried_on_resume(self):
        real = S.run_simple_fold
        with mock.patch.object(S, "run_simple_fold", side_effect=RuntimeError("boom")):
            res, out = quiet(S.run_simple_study, self.data, self.runs, self.ckpt)
        self.assertTrue(all("boom" in f.get("error", "") for f in res["r1"]["folds"]))
        self.assertIn("FAILED", out)
        with mock.patch.object(S, "run_simple_fold", wraps=real) as spy:
            res, _ = quiet(S.run_simple_study, self.data, self.runs, self.ckpt)
        self.assertEqual(spy.call_count, 2)                            # both failed folds re-run
        self.assertTrue(all("error" not in f for f in res["r1"]["folds"]))

    def test_fold_without_test_rows_is_skipped_not_fatal(self):
        data = replace(self.data, test_files=0, n_folds=1)
        res, out = quiet(S.run_simple_study, data, self.runs, self.ckpt)
        f = res["r1"]["folds"][0]
        self.assertIn("skipped", f)
        self.assertIn("skipped", out)

    def test_report_on_real_results(self):
        res, _ = quiet(S.run_simple_study, self.data, self.runs, self.ckpt)
        # final_report only knows kmeans / hdbscan; relabel the oracle run so it is rendered
        res = {"x": dict(cfg=replace(res["r1"]["cfg"], segmenter="kmeans", encoder="tcn"), folds=res["r1"]["folds"])}
        txt = S.final_report(res)
        for needle in ("With KNN/Kmeans as Segmenter", "TCN by edge", "[fold 0]", "Edge deviation", "Summary"):
            self.assertIn(needle, txt)

    def test_report_handles_skipped_error_and_empty_folds(self):
        empty = dict(expectancy=pd.DataFrame(), persistence=pd.DataFrame(), parity=pd.DataFrame(),
                     correlation=pd.DataFrame(), cells=pd.DataFrame(), sim_pairs=pd.DataFrame())
        cfg = oracle_cfg(segmenter="hdbscan", encoder="ts2vec", metrics=("infonet", "cosine"))
        res = {"n": dict(cfg=cfg, folds=[
            dict(fold=0, skipped="no usable rows", **empty),
            dict(fold=1, error="ValueError: x", **empty),
            dict(fold=2, recurring_latents=[], **empty),
            dict(fold=3, recurring_latents=["l1"], health=dict(test_rows=10, test_vol_nan=10, test_cost_rejected=0,
                                                                 max_cost_rr=5.0), eligible_test_rows=0, **empty)])}
        txt = S.final_report(res)
        self.assertIn("skipped: no usable rows", txt)
        self.assertIn("error: ValueError: x", txt)
        self.assertIn("no regime recurred", txt)
        self.assertIn("volatility-filtered=10", txt)
        self.assertIn("Winner: none", txt)
        self.assertIn("NOT RANKABLE", txt)

    def test_forecast_report_uses_mae_not_trade_rr(self):
        cfg = oracle_cfg(strategy="forecast_mae", horizons=(1, 2, 3, 4, 5), encoder="tcn")
        metrics = pd.DataFrame([dict(fold=0, expert_latent="global", horizon=1, n=50, mae_bps=.8,
                                     zero_mae_bps=1., global_mae_bps=.8, skill_vs_zero=.2,
                                     skill_vs_global=0.)])
        txt = S.final_report({"forecast": {"cfg": cfg, "folds": [dict(fold=0, forecast_metrics=metrics)]}})
        self.assertIn("mean absolute error (MAE)", txt)
        self.assertIn("mae_bps", txt)
        self.assertNotIn("net RR per trade", txt)


# =================================================================================================================
# 7. The runner script
# =================================================================================================================
class TestRunnerConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = _load_runner()

    def test_runs_are_valid_and_unique(self):
        names = [r.name for r in self.R.RUNS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), len(self.R.ENCODERS) * len(self.R.SEGMENTERS))
        for r in self.R.RUNS:
            self.assertIn(r.encoder, self.R.ENCODERS)
            self.assertIn(r.segmenter, self.R.SEGMENTERS)

    def test_horizons_blocks_and_episode_lengths_are_coherent(self):
        for r in self.R.RUNS:
            self.assertIn(r.trade_horizon, r.horizons, "run_simple_fold indexes rr_<trade_horizon>")
            self.assertGreaterEqual(r.min_episode_len, 2 * r.block, "runner comment: min_episode_len = 2 blocks")
            self.assertGreater(r.min_episode_len, max(r.horizons))
            self.assertGreaterEqual(r.min_episode_len, 2 * r.min_dwell)
            self.assertLessEqual(r.min_latents, r.n_latents)
            self.assertIn(r.feature_set, S.FEATURE_SETS)
            self.assertTrue(set(r.metrics) <= {"cosine", "geometric", "tdmi", "infonet"})
            if r.strategy == "forecast_mae":
                self.assertEqual(r.horizons, (1, 2, 3, 4, 5))

    def test_all_encoders_and_segmenters_appear_in_the_report(self):
        """final_report silently ignores runs whose encoder/segmenter it does not list."""
        self.assertTrue(set(self.R.SEGMENTERS) <= {"hdbscan", "kmeans"})
        self.assertTrue(set(self.R.ENCODERS) <= {"ts2vec", "tcn", "vq-vae"})
        self.assertTrue(set(self.R.ENCODERS) <= set(S.ENC_NAMES))
        self.assertTrue(set(self.R.SEGMENTERS) <= set(S.SEG_NAMES))

    def test_data_config_geometry(self):
        d = self.R.DATA
        self.assertGreaterEqual(d.train_files, 1)
        self.assertGreaterEqual(d.test_files, 1)
        self.assertGreaterEqual(d.step_files, d.test_files, "test windows of consecutive folds should not overlap")
        self.assertGreaterEqual(d.decimate, 1)
        self.assertIn(d.ts_unit, ("ms", "s", "us", "ns"))
        files_needed = d.start + (d.n_folds - 1) * d.step_files + d.train_files + d.gap_files + d.test_files
        print(f"\n   [runner] folds need files 0..{files_needed - 1}", end=" ")
        self.assertGreater(d.train_files, 10)
        self.assertGreaterEqual(files_needed, 700, "runner should use most of the expanded matched dataset")

    def test_checkpoint_name_matches_current_settings(self):
        """A stale checkpoint raises ValueError on resume; this just documents the file that will be used."""
        self.assertTrue(self.R.CHECKPOINT.endswith(".pkl"))
        key = repr(({k: v for k, v in vars(self.R.DATA).items() if k != "loader"}, self.R.RUNS))
        self.assertIsInstance(key, str)

    def test_runconfig_repr_is_stable_for_checkpoint_keys(self):
        """The checkpoint key is repr(settings): it must not contain memory addresses (that would break resume)."""
        key = repr(({k: v for k, v in vars(self.R.DATA).items() if k != "loader"}, self.R.RUNS))
        self.assertNotIn(" at 0x", key)


class TestRebuildSafety(unittest.TestCase):
    def test_orderbook_id_normalizer_fixes_only_redundant_lower_bound(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from components.pipelines.event_merger.matched.normalize_orderbook_ids import normalize_hour
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            source, output = Path(td) / "source.parquet", Path(td) / "normalized.parquet"
            table = pa.table({"event_type": pa.array(["snapshot", "update", "update"], type=pa.string()),
                              "event_time": pa.array([3, 1, 2], type=pa.int64()),
                              "first_update_id": pa.array([None, 30, 15], type=pa.int64()),
                              "final_update_id": pa.array([10, 30, 20], type=pa.int64()),
                              "prev_final_update_id": pa.array([None, 20, 10], type=pa.int64())})
            pq.write_table(table, source)
            result = normalize_hour(source, output)
            self.assertEqual(result["before"]["jump_forward"], 2)
            self.assertEqual(result["after"]["jump_forward"], 0)
            normalized = pq.read_table(output)
            self.assertEqual(normalized["first_update_id"].to_pylist(), [None, 11, 21])
            self.assertEqual(normalized.schema.field("event_type").type, pa.string())
            self.assertEqual(normalized["event_time"].to_pylist(), [3, 2, 1])
            self.assertEqual(pq.read_table(source)["first_update_id"].to_pylist(), [None, 30, 15])

    def test_invalid_day_start_cannot_reuse_stale_seed(self):
        from components.pipelines.event_merger.matched import snapshots
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            root = Path(td)
            source = root / "ingest_parquet" / "BTCUSDT" / "orderbook" / "2025-01-01" / "00.parquet"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"placeholder")
            seed_dir = root / "matched" / "BTCUSDT" / "_pipeline" / "day_seed"
            seed_dir.mkdir(parents=True)
            stale_seed = seed_dir / "2025-01-01.parquet"
            stale_seed.write_bytes(b"stale")
            with mock.patch.object(snapshots, "_run_snapshot_policy",
                                   return_value={"emitted_rows": 0, "malformed_events": 42}):
                with self.assertRaisesRegex(RuntimeError, "day-start seed was not created"):
                    snapshots.build_day_start_snapshots(root=root, symbol="BTCUSDT",
                                                        start="2025-01-01", end="2025-01-01")
            self.assertFalse(stale_seed.exists(), "invalid replay must remove an old seed before checking output")

    def test_bulk_pipeline_skips_bad_day_and_continues(self):
        from components.pipelines.ingestion.CryptoL2 import matched_pipeline
        days = ["2025-01-01", "2025-01-02"]
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            with mock.patch.object(matched_pipeline.paths, "binance_parquet_days",
                                   side_effect=lambda root, symbol, dataset: days), \
                 mock.patch.object(matched_pipeline.paths, "orderbook_parquet_days",
                                   return_value=days), \
                 mock.patch.object(matched_pipeline, "pipeline_work_dir",
                                   return_value=Path(td) / "_pipeline"), \
                 mock.patch.object(matched_pipeline, "build_day_start_snapshots"), \
                 mock.patch.object(matched_pipeline, "fetch_source_data"), \
                 mock.patch.object(matched_pipeline, "build_per_row_book_state"), \
                 mock.patch.object(matched_pipeline, "run_merge") as merge:
                def side_effect(*, root, symbol, source, date, **kw):
                    if date == days[0]:
                        raise RuntimeError("no usable snapshot")
                merge.side_effect = side_effect
                rc = matched_pipeline.main(["--root", td, "--start", days[0], "--end", days[-1]])
        self.assertEqual(rc, 0)
        # run_merge attempted for both days; the bad day is isolated, the good day merged.
        self.assertEqual(merge.call_count, 2)
        self.assertEqual({call.kwargs.get("date") for call in merge.call_args_list}, set(days))


# =================================================================================================================
# 8. Real-data health check (opt-in)
# =================================================================================================================
@unittest.skipUnless(REAL_MODULE and os.environ.get("INFONET_REAL_DATA"),
                     "set INFONET_MODULE and INFONET_REAL_DATA=1 to run the real-data health check")
class TestRealDataHealth(unittest.TestCase):
    def test_every_fold_has_usable_forecast_targets(self):
        """Check forward-return target coverage instead of the retired RR volatility filter."""
        R = _load_runner()
        root = ".".join(REAL_MODULE.split(".")[:-3])
        loader = importlib.import_module(root + ".exchange.loader").load_L2
        bad = []
        for k in range(R.DATA.n_folds):
            fd = S.load_fold(loader, k, R.DATA)
            Y = S.forward_return_targets(fd.mid, R.COMMON["horizons"]).to_numpy()
            coverage = {}
            for code, name in ((0, "train"), (2, "test")):
                mask = fd.seg == code
                coverage[name] = float(np.isfinite(Y[mask]).all(axis=1).mean()) if mask.any() else 0.0
            print(f"\n   fold {k}: train_target_coverage={coverage['train']:.3f}, test_target_coverage={coverage['test']:.3f}")
            if not (coverage["train"] > 0.5 and coverage["test"] > 0.5):
                bad.append(k)
        self.assertFalse(bad, f"folds with (almost) no usable forward returns: {bad} -- check quote coverage and flat/stale mid")


if __name__ == "__main__":
    print(f"mode: {'FAKE backends' if FAKE else 'REAL ' + REAL_MODULE}   dir: {HERE}")
    unittest.main(verbosity=2)
