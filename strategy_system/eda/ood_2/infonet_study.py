"""
infonet_study.py
===================================================================================

Does InfoNet-guided OOD regime discovery find recurring regimes in which a FIXED
trading strategy keeps behaving the same way out of sample?

Pipeline per purged walk-forward fold (everything fitted on TRAIN only)

    load_L2 (train | gap | test)  ->  ffill  ->  add_orderflow (+extras, 10/20/40 features)
      -> encoder (fit on train) -> embeddings (causal)
      -> segmenter (fit on train, frozen novelty on test) -> latent labels -> episodes
      -> one ridge "expert" per latent (fit on the first part of train, scored on the rest = "backtest")
      -> every TEST episode is traded by EVERY expert  (own-regime = diagonal, cross-regime = off-diagonal)
      -> every (episode, regime) pair is scored by 4 similarity metrics (cosine, geometric, TDMI, InfoNet)

What is measured (all OUT OF SAMPLE)

  consistency   between-episode heterogeneity of net RR/trade AFTER removing sampling noise
                (DerSimonian-Laird tau and I^2).  A raw std of per-episode means is dominated by
                trade-count noise: rr has unit variance, so 10 trades give a standard error of ~0.3 RR.
  own vs cross  does a regime's own expert beat other regimes' experts on the same episodes, and with less tau?
  controls      global (regime-free) expert, circularly-shifted placebo labels, decoy features
  persistence   episode length (events + seconds), tradable share, flicker, recurrence
  metric value  does similarity rank experts correctly inside an episode (rank corr, top-1 lift)?
                does high similarity => lower tau?  does high InfoNet similarity hide SIGN flips?

Caveats baked into the design (also printed by ``report``)

  * InfoNet is an EX-POST objective (it reads forward rr) so it can only validate, never gate live trades.
  * Mutual information is sign-blind: a regime where flow predicts +rr and one where it predicts -rr have the
    same InfoNet fingerprint.  ``sign_flip_rate_high_sim`` measures exactly how often that happens.
  * InfoNet's default ``min_features=1`` passes ~1-(1-alpha)^D of PURE-NOISE blocks, so "more features => more
    informative" can be a multiplicity artefact.  ``infonet_min_features="auto"`` (default here) holds the
    block-level null pass rate fixed as D grows, and the ``decoy40`` arm adds 30 time-shifted copies of the
    10 core features (same marginals, no relation to price) to separate "more information" from "more columns".
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from ...strategy.moe.ood.base_regime_encoder import make_encoder
from ...strategy.moe.ood.base_regime_segmenter import make_segmenter
from ...strategy.moe.ood.metrics import (cosine_similarity, infonet_estimate, infonet_target, resampled_distance_similarity,
                      tdmi_fingerprint, tdmi_fingerprint_similarity)
from ...strategy.moe.feature.orderflow import _robust_zscore, add_orderflow

# ----------------------------------------------------------------------------------------------
# 1. Data -> features
# ----------------------------------------------------------------------------------------------
CORE_10 = ["trade_imbalance_5", "trade_notional_z_5", "spread_bps_z", "queue_imbalance",
           "microprice_displacement", "ofi_z_5", "depth_imbalance_5", "depth_imbalance_20",
           "bid_depth_change_z_5", "flow_price_efficiency_z_5"]
# remaining 9 of add_orderflow's 19 (order = priority), then 21 extras (first one completes the 20-set)
REST_OF_19 = ["trade_imbalance_1", "trade_imbalance_20", "buy_notional_z_5", "sell_notional_z_5",
              "ofi_z_1", "ofi_z_20", "depth_imbalance_1", "depth_imbalance_10", "ask_depth_change_z_5"]
EXTRA_ORDER = [
    "ret_z_20",                                                                   # -> 20-set
    "ret_z_100", "ret_z_500", "vol_ratio_20_200", "vol_z_100", "efficiency_50", "sign_persistence_20",   # momentum / vol
    "trade_imbalance_100", "ofi_z_100", "trade_rate_z_100", "avg_trade_size_z_20", "flow_ret_corr_100",  # flow
    "spread_mean_z_100", "touch_concentration", "bid_dist_bps", "ask_dist_bps", "slope_imbalance",       # book shape
    "wdepth_imbalance", "imb5_change_20", "total_depth_5_z", "queue_size_z",
]
FEATURE_SETS = {"10": CORE_10, "20": CORE_10 + REST_OF_19 + EXTRA_ORDER[:1],
                "40": CORE_10 + REST_OF_19 + EXTRA_ORDER}
assert len(FEATURE_SETS["20"]) == 20 and len(FEATURE_SETS["40"]) == 40
ZERO_FILL = ("trade_imbalance_1", "trade_imbalance_5", "trade_imbalance_20", "trade_imbalance_100")


def _level(side: str, kind: str, k: int) -> str:
    return f"best_{side}_{kind}" if k == 1 else f"{side}_{kind}_{k}"


def prepare_l2(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """ffill the raw L2 frame and add the ``*_close`` aliases ``add_orderflow`` expects.

    NOTE on trades: after ffill every book row carries the LAST trade, so using ``quantity`` as is would count
    each trade once per following book update.  A row is a *new* trade only when ``agg_trade_id`` changes, and
    ``quantity`` is zeroed elsewhere.  Without this, all trade-flow features are inflated and autocorrelated.
    Returns (frame, number_of_leading_rows_dropped).
    """
    df = raw.reset_index(drop=True).ffill()
    core = ["best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty", "price", "is_buyer_maker"]
    first = int(df[core].notna().all(axis=1).to_numpy().argmax())
    df = df.iloc[first:].reset_index(drop=True)
    tid = df["agg_trade_id"]
    new_trade = (tid.ne(tid.shift()) & tid.notna())
    new_trade.iloc[0] = False
    df["new_trade"] = new_trade.to_numpy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").where(new_trade, 0.0)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(float).astype(bool)
    if "mid_price" not in df or df["mid_price"].isna().any():
        df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2
    df["price_close"] = df["price"]
    df["best_bid_price_close"] = df["best_bid_price"]
    df["best_ask_price_close"] = df["best_ask_price"]
    return df, first


def _extra_features(out: pd.DataFrame, bw: int, mp: int) -> dict[str, pd.Series]:
    z = lambda s: _robust_zscore(s, baseline_window=bw, min_periods=mp)
    mid = out["mid_price"]
    lm = np.log(mid)
    d1 = lm.diff()
    ex: dict[str, pd.Series] = {}
    ex["ret_z_20"], ex["ret_z_100"], ex["ret_z_500"] = z(lm.diff(20)), z(lm.diff(100)), z(lm.diff(500))
    vs = d1.rolling(20, min_periods=20).std(ddof=0)
    vl = d1.rolling(200, min_periods=100).std(ddof=0)
    ex["vol_ratio_20_200"] = np.log(vs.where(vs > 0) / vl.where(vl > 0))
    ex["vol_z_100"] = z(d1.rolling(100, min_periods=50).std(ddof=0))
    path = d1.abs().rolling(50, min_periods=50).sum()
    ex["efficiency_50"] = lm.diff(50).abs() / path.where(path > 0)
    ex["sign_persistence_20"] = np.sign(d1).rolling(20, min_periods=20).mean()
    ex["trade_imbalance_100"] = out["trade_imbalance_100"]
    ex["ofi_z_100"] = out["ofi_z_100"]
    nt = out["new_trade"].astype(float)
    ex["trade_rate_z_100"] = z(nt.rolling(100, min_periods=100).sum())
    cnt20 = nt.rolling(20, min_periods=20).sum()
    ex["avg_trade_size_z_20"] = z(out["quantity"].rolling(20, min_periods=20).sum() / cnt20.where(cnt20 > 0))
    ex["flow_ret_corr_100"] = out["aggressive_signed_notional"].rolling(100, min_periods=100).corr(d1)
    ex["spread_mean_z_100"] = z(out["spread_bps"].rolling(100, min_periods=100).mean())
    d5 = out["bid_depth_5"] + out["ask_depth_5"]
    d20 = out["bid_depth_20"] + out["ask_depth_20"]
    ex["touch_concentration"] = d5 / d20.where(d20 > 0)
    mid_v = mid.to_numpy()[:, None]
    qb = np.column_stack([out[_level("bid", "qty", k)].to_numpy() for k in range(1, 21)])
    qa = np.column_stack([out[_level("ask", "qty", k)].to_numpy() for k in range(1, 21)])
    pb = np.column_stack([out[_level("bid", "price", k)].to_numpy() for k in range(1, 21)])
    pa = np.column_stack([out[_level("ask", "price", k)].to_numpy() for k in range(1, 21)])
    with np.errstate(divide="ignore", invalid="ignore"):
        bd = (qb * (mid_v - pb)).sum(1) / qb.sum(1) / mid.to_numpy() * 1e4
        ad = (qa * (pa - mid_v)).sum(1) / qa.sum(1) / mid.to_numpy() * 1e4
        w = 1.0 / np.arange(1, 21)[None, :]
        wb, wa = (qb * w).sum(1), (qa * w).sum(1)
        wimb = (wb - wa) / (wb + wa)
        slope = (ad - bd) / (ad + bd)
    ex["bid_dist_bps"], ex["ask_dist_bps"] = pd.Series(bd, index=out.index), pd.Series(ad, index=out.index)
    ex["slope_imbalance"], ex["wdepth_imbalance"] = pd.Series(slope, index=out.index), pd.Series(wimb, index=out.index)
    ex["imb5_change_20"] = out["depth_imbalance_5"].diff(20)
    ex["total_depth_5_z"] = z(d5)
    ex["queue_size_z"] = z(out["best_bid_qty"] + out["best_ask_qty"])
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in ex.items()}


@dataclass
class FoldData:
    fold: int
    frames: dict                 # feature_set -> DataFrame (RangeIndex, decimated, finite)
    mid: np.ndarray
    spread: np.ndarray
    ts_sec: np.ndarray
    seg: np.ndarray              # 0 train, 1 gap, 2 test
    meta: dict = field(default_factory=dict)
    truth: np.ndarray | None = None   # ground-truth regimes (synthetic positive control only)


def build_fold_features(raw: pd.DataFrame, seg_raw: np.ndarray, feature_sets=("10", "20", "40", "decoy40"),
                        decimate: int = 1, baseline_window: int = 1000, min_periods: int = 100,
                        ts_unit: str = "ms", fold: int = 0) -> FoldData:
    seg_raw = np.asarray(seg_raw)
    df, dropped = prepare_l2(raw)
    seg_raw = seg_raw[dropped:][: len(df)]
    ts = df["timestamp"] if "timestamp" in df else pd.Series(np.arange(len(df)), index=df.index)
    if np.issubdtype(ts.dtype, np.datetime64):        # unit-safe (pandas >= 2 keeps ms/us resolution)
        ts_sec = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    else:
        ts_sec = pd.to_numeric(ts, errors="coerce").to_numpy(dtype=float) / {"ms": 1e3, "s": 1.0, "us": 1e6, "ns": 1e9}[ts_unit]
    out, feat19 = add_orderflow(df, windows=(1, 5, 20, 100), baseline_window=baseline_window, min_periods=min_periods)
    out["new_trade"] = df["new_trade"].to_numpy()
    allf = out[feat19].copy()
    for name, col in _extra_features(out, baseline_window, min_periods).items():
        allf[name] = col
    for c in allf.columns:
        if c in ZERO_FILL:                       # no trade in the window == no net flow, not "unknown"
            allf[c] = allf[c].fillna(0.0)
    need = [c for c in FEATURE_SETS["40"]]
    ok = allf[need].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
    warm = int(ok.argmax()) if ok.any() else len(allf)
    allf = allf.iloc[warm:].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    keep = slice(None, None, decimate)
    allf = allf.iloc[keep].reset_index(drop=True)
    mid = out["mid_price"].to_numpy()[warm:][keep]
    spread = (out["best_ask_price"] - out["best_bid_price"]).to_numpy()[warm:][keep]
    ts_sec = ts_sec[warm:][keep]
    seg = seg_raw[warm:][keep]
    truth = df["regime_truth"].to_numpy()[warm:][keep] if "regime_truth" in df else None
    frames = {}
    n = len(allf)
    for fs in feature_sets:
        if fs == "decoy40":
            base = allf[CORE_10].copy()
            for i in (1, 2, 3):
                shifted = np.roll(allf[CORE_10].to_numpy(), n * i // 4, axis=0)
                for j, c in enumerate(CORE_10):
                    base[f"decoy{i}_{c}"] = shifted[:, j]
            frames[fs] = base
        else:
            frames[fs] = allf[FEATURE_SETS[fs]].copy()
    return FoldData(fold, frames, mid, spread, ts_sec, seg,
                    meta=dict(rows=n, dropped_warmup=dropped + warm, decimate=decimate), truth=truth)


def load_fold(loader: Callable, fold: int, d: "DataConfig") -> FoldData:
    s = d.start + fold * d.step_files
    parts, seg = [], []
    for code, (n_files, offset) in enumerate([(d.train_files, 0), (d.gap_files, d.train_files),
                                              (d.test_files, d.train_files + d.gap_files)]):
        if n_files <= 0:
            continue
        piece = loader(file_limit=n_files, start=s + offset)
        parts.append(piece)
        seg.append(np.full(len(piece), code))
    raw = pd.concat(parts, ignore_index=True)
    return build_fold_features(raw, np.concatenate(seg), d.feature_sets, d.decimate, d.baseline_window,
                               d.min_periods, d.ts_unit, fold)


# ----------------------------------------------------------------------------------------------
# 2. Configs
# ----------------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    train_files: int = 6
    gap_files: int = 0              # extra purge by whole files (row-level purge is always applied too)
    test_files: int = 2
    step_files: int = 2
    n_folds: int = 5
    start: int = 0
    decimate: int = 1               # keep every k-th event after feature computation (speed knob)
    baseline_window: int = 1000
    min_periods: int = 100
    ts_unit: str = "ms"
    feature_sets: tuple = ("10", "20", "40", "decoy40")
    loader: Callable | None = None  # default: load_L2


@dataclass
class RunConfig:
    name: str = "run"
    feature_set: str = "20"
    encoder: str = "pca"
    encoder_kwargs: dict = field(default_factory=dict)
    window: int = 32
    embedding_dim: int = 8
    segmenter: str = "kmeans"
    segmenter_kwargs: dict = field(default_factory=dict)
    n_latents: int = 6
    min_dwell: int = 50
    learn_novel_oos: bool = False   # False: test regimes are limited to latents seen in train
    standardize_embeddings: bool = True   # z-score embeddings with TRAIN mean/std; segmenter thresholds assume unit scale
    metrics: tuple = ("cosine", "geometric", "tdmi", "infonet")
    # InfoNet objective
    horizons: tuple = (4, 16, 64)
    block: int = 200
    infonet_alpha: float = 0.05
    infonet_surrogates: int = 49
    infonet_min_features: int | str = "auto"
    infonet_min_run: int = 2
    # TDMI
    tdmi_lag: int = 8
    tdmi_bins: int = 8
    tdmi_surrogates: int = 19
    # strategy
    trade_horizon: int = 16
    entry_quantile: float = 0.6
    ridge_lambda: float = 0.1
    val_fraction: float = 0.25
    fee_bps: float = 0.0            # per side, on top of spread crossing
    # episode / sampling rules
    min_episode_len: int = 400
    min_trades: int = 5
    min_val_trades: int = 20
    min_expert_rows: int = 500
    max_test_episodes: int = 60
    max_ref_episodes: int = 6
    parity_tol: float = 0.05
    n_placebo: int = 20
    seed: int = 0


def auto_min_features(d: int, alpha: float, tail: float = 0.05) -> int:
    """Smallest m with P(Binomial(d, alpha) >= m) <= tail: holds the null block pass-rate fixed as D grows."""
    for m in range(1, d + 1):
        p = sum(math.comb(d, k) * alpha ** k * (1 - alpha) ** (d - k) for k in range(m, d + 1))
        if p <= tail:
            return m
    return d


# ----------------------------------------------------------------------------------------------
# 3. Regime discovery
# ----------------------------------------------------------------------------------------------
def _seg_kwargs(cfg: RunConfig) -> tuple[str, dict]:
    kw = dict(max_regimes=cfg.n_latents, min_dwell=cfg.min_dwell)
    user = dict(cfg.segmenter_kwargs)
    if cfg.segmenter == "kmeans":
        bp = {"n_clusters": cfg.n_latents, **user.pop("backend_params", {})}
        return "kmeans", {**kw, "backend_params": bp, **user}
    if cfg.segmenter == "hdbscan":
        bp = {"min_cluster_size": 500, **user.pop("backend_params", {})}
        return "hdbscan", {**kw, "backend_params": bp, **user}
    return cfg.segmenter, {**kw, **user}


def discover_regimes(frame: pd.DataFrame, seg: np.ndarray, cfg: RunConfig, n_tr_fit_end: int, truth=None):
    """Fit encoder + segmenter on TRAIN rows only; label train and test rows.  Returns (labels, embeddings)."""
    cols = list(frame.columns)
    tr = np.flatnonzero(seg == 0)
    te = np.flatnonzero(seg == 2)
    enc = make_encoder(cfg.encoder, window=cfg.window, embedding_dim=cfg.embedding_dim,
                       feature_columns=cols, **cfg.encoder_kwargs)
    enc.fit_encoder(frame.iloc[tr[:n_tr_fit_end]])
    emb = enc.transform(frame)                       # causal normalisation; weights learnt on train only
    if cfg.standardize_embeddings:
        ref = emb.iloc[tr].dropna()
        emb = (emb - ref.mean()) / ref.std(ddof=0).where(lambda x: x > 1e-12, 1.0)
    labels = np.full(len(frame), "null", dtype=object)
    if cfg.segmenter == "oracle":                    # positive control: needs a 'regime_truth' column from the loader
        assert truth is not None, "segmenter='oracle' needs ground-truth regimes (synthetic loader only)"
        labels[:] = [f"truth_{int(x)}" for x in truth]
        labels[seg == 1] = "gap"
        return labels, emb
    sg_name, sg_kw = _seg_kwargs(cfg)
    sg = make_segmenter(sg_name, **sg_kw)
    e_tr = emb.iloc[tr].dropna()
    e_te = emb.iloc[te].dropna()
    if sg.backend == "custom":
        pred_tr = sg.predict(e_tr)                   # online learning pass == fit
    else:
        sg.fit(e_tr)
        pred_tr = sg.predict(e_tr)
    labels[e_tr.index.to_numpy()] = pred_tr["latent_id"].to_numpy()
    sg.learn_novel = bool(cfg.learn_novel_oos)
    if len(e_te):
        pred_te = sg.predict(e_te)
        labels[e_te.index.to_numpy()] = pred_te["latent_id"].to_numpy()
    labels[seg == 1] = "gap"
    return labels, emb


def runs_of(labels: np.ndarray, lo: int = 0, hi: int | None = None, skip=("null", "gap")):
    """Contiguous runs of equal label within [lo, hi): list of (start, end, label)."""
    hi = len(labels) if hi is None else hi
    seg = labels[lo:hi]
    if not len(seg):
        return []
    b = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
    return [(lo + int(a), lo + int(c), seg[a]) for a, c in zip(b[:-1], b[1:]) if seg[a] not in skip]


def persistence_stats(labels: np.ndarray, ts_sec: np.ndarray, lo: int, hi: int, cfg: RunConfig, prefix="") -> dict:
    runs = runs_of(labels, lo, hi)
    n = hi - lo
    lens = np.array([e - s for s, e, _ in runs], dtype=float)
    secs = np.array([ts_sec[min(e, len(ts_sec)) - 1] - ts_sec[s] for s, e, _ in runs], dtype=float)
    all_runs = runs_of(labels, lo, hi, skip=())
    per_latent = {}
    for s, e, k in runs:
        if e - s >= cfg.min_episode_len:
            per_latent[k] = per_latent.get(k, 0) + 1
    tradable = lens[lens >= cfg.min_episode_len].sum() if len(lens) else 0.0
    d = dict(rows=n, share_null=float(np.mean(labels[lo:hi] == "null")) if n else np.nan,
             n_episodes=len(runs), n_latents=len({k for *_, k in runs}),
             median_len=float(np.median(lens)) if len(lens) else np.nan,
             mean_len=float(lens.mean()) if len(lens) else np.nan,
             p25_len=float(np.percentile(lens, 25)) if len(lens) else np.nan,
             p75_len=float(np.percentile(lens, 75)) if len(lens) else np.nan,
             median_sec=float(np.median(secs)) if len(secs) else np.nan,
             tradable_share=float(tradable / n) if n else np.nan,
             flicker_share=float(np.mean(lens < 2 * cfg.min_dwell)) if len(lens) else np.nan,
             surv_1x=float(np.mean(lens >= cfg.min_episode_len)) if len(lens) else np.nan,
             surv_4x=float(np.mean(lens >= 4 * cfg.min_episode_len)) if len(lens) else np.nan,
             recurring_latents=int(sum(v >= 2 for v in per_latent.values())),
             transitions_per_10k=float((len(all_runs) - 1) / n * 1e4) if n else np.nan)
    return {prefix + k: v for k, v in d.items()}


# ----------------------------------------------------------------------------------------------
# 4. Strategy: one ridge expert per latent; trades are greedy non-overlapping, net of spread (+fee)
# ----------------------------------------------------------------------------------------------
@dataclass
class Expert:
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    thr: float
    ref_rr: float = np.nan      # backtest (validation block) net RR per trade
    ref_n: int = 0
    ref_ic: float = np.nan

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w


def fit_expert(X: np.ndarray, y: np.ndarray, lam: float, q: float) -> Expert | None:
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 50:
        return None
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    n, d = Z.shape
    w = np.linalg.solve(Z.T @ Z / n + lam * np.eye(d), Z.T @ (y - y.mean()) / n)
    thr = float(np.quantile(np.abs(Z @ w), q))
    return Expert(mu, sd, w, thr)


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def spearman(a: np.ndarray, b: np.ndarray, min_n: int = 10) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def trade_rows(pred: np.ndarray, thr: float, rr: np.ndarray, cost: np.ndarray, rows: np.ndarray, h: int):
    """Greedy non-overlapping entries among ``rows``; returns (entry_rows, net_rr, gross_rr)."""
    p = pred[rows]
    ok = np.isfinite(p) & (np.abs(p) >= thr) & np.isfinite(rr[rows]) & np.isfinite(cost[rows])
    cand = rows[ok]
    take, i = [], 0
    while i < len(cand):
        take.append(cand[i])
        i = int(np.searchsorted(cand, cand[i] + h, side="left"))
    take = np.asarray(take, dtype=int)
    gross = np.sign(pred[take]) * rr[take]
    return take, gross - cost[take], gross


def heterogeneity(means, ns, sds) -> dict:
    """DerSimonian-Laird between-episode SD (tau) and I^2, i.e. dispersion BEYOND trade-count sampling noise."""
    m, n, s = (np.asarray(x, dtype=float) for x in (means, ns, sds))
    ok = (n >= 2) & np.isfinite(m) & np.isfinite(s) & (s > 0)
    k = int(ok.sum())
    if k < 3:
        return dict(k=k, pooled=np.nan, tau=np.nan, I2=np.nan, raw_sd=np.nan)
    v = s[ok] ** 2 / n[ok]
    w = 1 / v
    mw = float((w * m[ok]).sum() / w.sum())
    q = float((w * (m[ok] - mw) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c)
    return dict(k=k, pooled=mw, tau=math.sqrt(tau2), I2=max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0,
                raw_sd=float(np.std(m[ok], ddof=1)))


# ----------------------------------------------------------------------------------------------
# 5. Similarity engine (4 metrics)
# ----------------------------------------------------------------------------------------------
class Similarity:
    def __init__(self, cfg: RunConfig, F: np.ndarray, Fz: np.ndarray, emb: np.ndarray, rr: np.ndarray):
        self.cfg, self.F, self.Fz, self.emb, self.rr = cfg, F, Fz, emb, rr
        self.D = F.shape[1]
        self.min_feat = (auto_min_features(self.D, cfg.infonet_alpha) if cfg.infonet_min_features == "auto"
                         else int(cfg.infonet_min_features))
        self.refs: dict = {}

    # --- per-episode summaries -------------------------------------------------------------
    def infonet(self, s: int, e: int):
        c = self.cfg
        return infonet_estimate(self.F[s:e], self.rr[s:e], block=c.block, alpha=c.infonet_alpha,
                                min_features=self.min_feat, min_run=c.infonet_min_run,
                                n_surrogates=c.infonet_surrogates, overlap=max(c.horizons))

    def describe(self, s: int, e: int) -> dict:
        c, d = self.cfg, {}
        if "cosine" in c.metrics:
            d["emb"] = np.nanmean(self.emb[s:e], axis=0)
        if "geometric" in c.metrics:
            d["path"] = self.Fz[s:e]
        if "tdmi" in c.metrics:
            d["tdmi"] = tdmi_fingerprint(self.F[s:e], c.tdmi_lag, c.tdmi_bins, None, c.tdmi_surrogates)
        if "infonet" in c.metrics:
            r = self.infonet(s, e)
            npers = r.persistent.sum(axis=0).astype(float)
            d["infonet_res"] = (np.nan_to_num(r.feature_hits) * npers, npers)
            d["infonet_fp"] = np.nan_to_num(r.feature_hits).ravel()
            d["infonet_score"] = r.score
            d["infonet_info"] = r.info
        return d

    def build_ref(self, episodes: list[tuple[int, int]]):
        """Reference for one latent from its (longest) TRAIN episodes."""
        c = self.cfg
        eps = sorted(episodes, key=lambda x: x[0] - x[1])[:c.max_ref_episodes]
        descs = [self.describe(s, e) for s, e in eps]
        ref = {"descs": descs, "n": len(descs)}
        if "tdmi" in c.metrics:
            ref["tdmi_pooled"] = tdmi_fingerprint([self.F[s:e] for s, e in eps], c.tdmi_lag, c.tdmi_bins,
                                                   None, c.tdmi_surrogates)
        if "infonet" in c.metrics:
            cnt = sum(d["infonet_res"][0] for d in descs)
            npers = sum(d["infonet_res"][1] for d in descs)
            fp = np.divide(cnt, npers[None, :], out=np.zeros_like(cnt), where=npers[None, :] > 0)
            ref["infonet_fp"] = fp.ravel()
            ref["infonet_score"] = float(np.nanmean([d["infonet_score"] for d in descs]))
        if "cosine" in c.metrics:
            ref["emb"] = np.mean([d["emb"] for d in descs], axis=0)
        return ref

    def sim(self, ep: dict, ref: dict) -> dict:
        out = {}
        if "cosine" in self.cfg.metrics:
            out["cosine"] = cosine_similarity(ep["emb"], ref["emb"])
        if "geometric" in self.cfg.metrics:
            vals = [resampled_distance_similarity(ep["path"], d["path"]) for d in ref["descs"]]
            out["geometric"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        if "tdmi" in self.cfg.metrics:
            out["tdmi"] = tdmi_fingerprint_similarity(ep["tdmi"], ref["tdmi_pooled"])
        if "infonet" in self.cfg.metrics:
            a, b = ep["infonet_fp"], ref["infonet_fp"]
            if a.sum() == 0 and b.sum() == 0:
                out["infonet"] = np.nan          # agreement about absence is not evidence of recurrence
            elif a.sum() == 0 or b.sum() == 0:
                out["infonet"] = 0.0
            else:
                out["infonet"] = tdmi_fingerprint_similarity(a, b)
        return out


# ----------------------------------------------------------------------------------------------
# 6. One fold of one run
# ----------------------------------------------------------------------------------------------
def run_fold(fd: FoldData, cfg: RunConfig, verbose: bool = True) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed + fd.fold)
    frame = fd.frames[cfg.feature_set]
    F = frame.to_numpy(dtype=float)
    T, D = F.shape
    seg = fd.seg
    tr = np.flatnonzero(seg == 0)
    te = np.flatnonzero(seg == 2)
    n_tr = int(tr[-1]) + 1
    h = cfg.trade_horizon
    maxh = max(cfg.horizons)
    assert h in cfg.horizons, "trade_horizon must be one of the InfoNet horizons"

    # ---- targets, cost ------------------------------------------------------------------
    rr_df = infonet_target(pd.Series(fd.mid), cfg.horizons)
    rr = rr_df.to_numpy()
    rr_h = rr_df[f"rr_{h}"].to_numpy()
    vol = pd.Series(fd.mid).diff().rolling(100, min_periods=50).std(ddof=0).to_numpy()
    vol = np.where(vol > 1e-12, vol, np.nan)
    exit_spread = np.full(T, np.nan)
    exit_spread[:-h] = fd.spread[h:]
    cost = ((fd.spread + exit_spread) / 2 + 2 * cfg.fee_bps * 1e-4 * fd.mid) / (vol * math.sqrt(h))

    # ---- purge: training rows whose forward window crosses the train boundary are unusable -------
    usable_tr = np.zeros(T, bool)
    usable_tr[: max(n_tr - maxh, 0)] = seg[: max(n_tr - maxh, 0)] == 0
    split = int(tr[0] + (1 - cfg.val_fraction) * (n_tr - tr[0]))
    fit_rows = usable_tr & (np.arange(T) < split - maxh)       # embargo between fit and validation blocks
    val_rows = usable_tr & (np.arange(T) >= split)

    labels, emb_df = discover_regimes(frame, seg, cfg, n_tr_fit_end=len(tr), truth=fd.truth)
    emb = emb_df.to_numpy(dtype=float)
    lab_ref = labels.copy()
    lab_ref[~usable_tr] = "null"                                # references come from purged train rows only

    # ---- persistence (H2) -------------------------------------------------------------
    pers = {**persistence_stats(labels, fd.ts_sec, 0, n_tr, cfg, "train_"),
            **persistence_stats(labels, fd.ts_sec, int(te[0]), int(te[-1]) + 1, cfg, "test_")}
    known = set(np.unique(lab_ref[usable_tr])) - {"null", "gap"}
    pers["test_share_unseen"] = float(np.mean(~np.isin(labels[te], list(known) + ["null"]))) if len(te) else np.nan

    # ---- experts: fit on fit_rows, "backtest" on val_rows --------------------------------------
    mu_all = F[usable_tr & np.isfinite(F).all(1)].mean(0)
    sd_all = F[usable_tr & np.isfinite(F).all(1)].std(0)
    sd_all = np.where(sd_all > 1e-12, sd_all, 1.0)
    Fz = (F - mu_all) / sd_all
    experts: dict[str, Expert] = {}
    for k in sorted(known):
        m = fit_rows & (labels == k)
        if m.sum() < cfg.min_expert_rows:
            continue
        ex = fit_expert(F[m], rr_h[m], cfg.ridge_lambda, cfg.entry_quantile)
        if ex is None:
            continue
        pred = ex.predict(F)
        vrows = np.flatnonzero(val_rows & (labels == k))
        take, net, _ = trade_rows(pred, ex.thr, rr_h, cost, vrows, h)
        if len(take) < cfg.min_val_trades:
            continue
        ex.ref_rr, ex.ref_n = float(net.mean()), len(take)
        ex.ref_ic = spearman(pred[vrows], rr_h[vrows])
        experts[k] = ex
    gex = fit_expert(F[fit_rows], rr_h[fit_rows], cfg.ridge_lambda, cfg.entry_quantile)
    if gex is not None:
        pred = gex.predict(F)
        vrows = np.flatnonzero(val_rows)
        take, net, _ = trade_rows(pred, gex.thr, rr_h, cost, vrows, h)
        gex.ref_rr, gex.ref_n, gex.ref_ic = (float(net.mean()) if len(take) else np.nan), len(take), spearman(pred[vrows], rr_h[vrows])
    preds = {k: e.predict(F) for k, e in experts.items()}
    if gex is not None:
        preds["global"] = gex.predict(F)
    allx = dict(experts, **({"global": gex} if gex is not None else {}))

    out = dict(fold=fd.fold, persistence=pers, n_experts=len(experts),
               experts={k: dict(ref_rr=e.ref_rr, ref_n=e.ref_n, ref_ic=e.ref_ic) for k, e in allx.items()})
    if not experts:
        out.update(pairs=pd.DataFrame(), trades=pd.DataFrame(), placebo=pd.DataFrame(), secs=time.time() - t0)
        if verbose:
            print(f"  [{cfg.name}] fold {fd.fold}: NO EXPERTS -- {len(known)} latent(s) known in train "
                  f"(train latents={pers.get('train_n_latents')}, null share={pers.get('train_share_null'):.2f}); each needs "
                  f">= {cfg.min_expert_rows} fit rows and >= {cfg.min_val_trades} validation trades. Persistence still recorded.")
        return out

    # ---- test episodes ---------------------------------------------------------------------------
    eps = [(s, e, k) for s, e, k in runs_of(labels, int(te[0]), int(te[-1]) + 1)
           if e - s >= cfg.min_episode_len and k in experts]
    sample = set(rng.choice(len(eps), min(len(eps), cfg.max_test_episodes), replace=False).tolist()) if eps else set()

    sim = None
    refs: dict = {}
    if cfg.metrics and eps:
        sim = Similarity(cfg, F, Fz, emb, rr)
        by_latent: dict = {}
        for s, e, k in runs_of(lab_ref, 0, n_tr):
            if k in experts and e - s >= cfg.min_episode_len:
                by_latent.setdefault(k, []).append((s, e))
        refs = {k: sim.build_ref(v) for k, v in by_latent.items()}

    rows, trades = [], []
    for ei, (s, e, k) in enumerate(eps):
        idx = np.arange(s, e)
        desc = sim.describe(s, e) if (sim is not None and ei in sample) else None
        for j, ex in allx.items():
            take, net, gross = trade_rows(preds[j], ex.thr, rr_h, cost, idx, h)
            row = dict(fold=fd.fold, ep=ei, start=s, end=e, length=e - s, assigned=k, expert=j,
                       own=(j == k), n_trades=len(take), rr_net=float(net.mean()) if len(take) else np.nan,
                       rr_gross=float(gross.mean()) if len(take) else np.nan,
                       rr_sd=float(net.std(ddof=1)) if len(take) > 1 else np.nan,
                       ic=spearman(preds[j][idx], rr_h[idx]), ref_rr=ex.ref_rr, ref_ic=ex.ref_ic,
                       episode_infonet=desc["infonet_score"] if desc and "infonet_score" in desc else np.nan)
            if desc is not None and j in refs:
                row.update(sim.sim(desc, refs[j]))
            rows.append(row)
            if j in (k, "global") and len(take):
                trades.extend(dict(fold=fd.fold, arm="own" if j == k else "global", row=int(r), rr_net=float(x))
                              for r, x in zip(take, net))
    pairs = pd.DataFrame(rows)

    # ---- placebo: same episode-length structure, labels circularly shifted vs. price ------------------
    plc = []
    for draw in range(cfg.n_placebo):
        shifted = np.roll(labels[te[0]:te[-1] + 1], int(rng.integers(len(te) // 4, 3 * len(te) // 4)))
        lab2 = np.full(T, "null", dtype=object)
        lab2[te[0]:te[-1] + 1] = shifted
        ep_rows, pooled = [], []
        for s, e, k in runs_of(lab2, int(te[0]), int(te[-1]) + 1):
            if e - s < cfg.min_episode_len or k not in experts:
                continue
            take, net, _ = trade_rows(preds[k], experts[k].thr, rr_h, cost, np.arange(s, e), h)
            if len(take):
                ep_rows.append(dict(fold=fd.fold, expert=k, assigned=k, n_trades=len(take), rr_net=float(net.mean()),
                                    rr_sd=float(net.std(ddof=1)) if len(take) > 1 else np.nan))
                pooled.extend(net.tolist())
        het = _het(pd.DataFrame(ep_rows)) if ep_rows else dict(tau=np.nan, I2=np.nan, k=0)
        plc.append(dict(fold=fd.fold, draw=draw, mean_rr=float(np.mean(pooled)) if pooled else np.nan,
                        n_trades=len(pooled), tau=het["tau"], I2=het["I2"], k=het["k"]))
    out.update(pairs=pairs, trades=pd.DataFrame(trades), placebo=pd.DataFrame(plc), secs=time.time() - t0)
    if verbose:
        print(f"  [{cfg.name}] fold {fd.fold}: {len(eps)} test episodes, {len(experts)} experts, "
              f"{len(pairs)} pairs, {out['secs']:.1f}s")
    return out


# ----------------------------------------------------------------------------------------------
# 7. Driver + analysis
# ----------------------------------------------------------------------------------------------
def run_study(data: DataConfig, runs: list[RunConfig], verbose: bool = True) -> dict:
    """Fold-major loop (each fold is loaded and featurised ONCE, then every run config is executed on it)."""
    if data.loader is None:
        from ...exchange.loader import load_L2
        data = replace(data, loader=load_L2)
    res = {r.name: [] for r in runs}
    for f in range(data.n_folds):
        t0 = time.time()
        fd = load_fold(data.loader, f, data)
        if verbose:
            print(f"fold {f}: {fd.meta['rows']} rows after warm-up/decimation ({time.time() - t0:.1f}s)")
        for r in runs:
            try:
                res[r.name].append(run_fold(fd, r, verbose))
            except Exception as exc:                          # keep the grid alive; surface the failure
                print(f"  [{r.name}] fold {f} FAILED: {type(exc).__name__}: {exc}")
                res[r.name].append(dict(fold=f, error=str(exc), pairs=pd.DataFrame(), trades=pd.DataFrame(),
                                        placebo=pd.DataFrame(), persistence={}))
        del fd
    return {r.name: collect(res[r.name], r) for r in runs}


def collect(folds: list[dict], cfg: RunConfig) -> dict:
    cat = lambda key: (pd.concat([f[key] for f in folds if len(f.get(key, []))], ignore_index=True)
                       if any(len(f.get(key, [])) for f in folds) else pd.DataFrame())
    pers = pd.DataFrame([{**f["persistence"], "fold": f["fold"]} for f in folds if f.get("persistence")])
    return dict(cfg=cfg, pairs=cat("pairs"), trades=cat("trades"), placebo=cat("placebo"), persistence=pers,
                experts=pd.DataFrame([dict(fold=f["fold"], latent=k, **v) for f in folds
                                      for k, v in f.get("experts", {}).items()]))


def boot_ci(x, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _het(df: pd.DataFrame) -> dict:
    """Between-episode heterogeneity pooled over (fold, expert, assigned-regime) CELLS.

    Latent ids are re-learned in every fold, so a cell is only meaningful inside one fold.  Within each cell
    (>= 2 episodes) we take Cochran's Q against the cell's own inverse-variance mean; Q, its df and the DL scaling
    constant are then summed over cells.  tau is therefore the SD of true episode-mean RR *inside* a regime/expert
    cell (the quantity in "0.15-0.25 around 0.20"), free of between-regime differences in edge and of trade-count noise.
    """
    d = df[(df["n_trades"] >= 2) & df["rr_net"].notna() & (df["rr_sd"] > 0)]
    q_sum = df_sum = c_sum = 0.0
    resid, noise = [], []
    cells = episodes = 0
    for _, g in d.groupby(["fold", "expert", "assigned"]):
        if len(g) < 2:
            continue
        v = g["rr_sd"].to_numpy() ** 2 / g["n_trades"].to_numpy()
        w = 1 / v
        m = g["rr_net"].to_numpy()
        mw = (w * m).sum() / w.sum()
        q_sum += float((w * (m - mw) ** 2).sum())
        df_sum += len(g) - 1
        c_sum += float(w.sum() - (w ** 2).sum() / w.sum())
        resid.extend((m - m.mean()).tolist())
        noise.extend(v.tolist())
        cells += 1
        episodes += len(g)
    if cells == 0 or df_sum < 3:
        return dict(k=episodes, cells=cells, pooled=np.nan, tau=np.nan, I2=np.nan, raw_sd=np.nan, noise_sd=np.nan, spread_p10_p90=np.nan)
    tau2 = max(0.0, (q_sum - df_sum) / c_sum)
    return dict(k=episodes, cells=cells, pooled=_pooled(d), tau=math.sqrt(tau2),
                I2=max(0.0, (q_sum - df_sum) / q_sum) if q_sum > 0 else 0.0,
                raw_sd=float(np.std(resid)), noise_sd=float(math.sqrt(np.mean(noise))),
                spread_p10_p90=float(np.percentile(resid, 90) - np.percentile(resid, 10)))


def _pooled(df: pd.DataFrame) -> float:
    d = df[df["n_trades"] > 0]
    return float((d["rr_net"] * d["n_trades"]).sum() / d["n_trades"].sum()) if len(d) else np.nan


def max_drawdown(rr: np.ndarray) -> float:
    c = np.cumsum(rr)
    return float((np.maximum.accumulate(np.r_[0, c])[1:] - c).max()) if len(c) else np.nan


def economics(run: dict) -> pd.DataFrame:
    """Own-regime vs cross-regime vs global vs placebo (H0: regime-specific experts are no better)."""
    p, cfg = run["pairs"], run["cfg"]
    if p.empty:
        return pd.DataFrame()
    p = p[(p["n_trades"] >= cfg.min_trades)]
    reg = p[p["expert"] != "global"]
    own, cross, glob = reg[reg["own"]], reg[~reg["own"]], p[p["expert"] == "global"]
    # paired by episode: own minus mean of cross experts on the SAME episode
    paired = []
    for _, g in reg.groupby(["fold", "ep"]):
        o, c = g.loc[g["own"], "rr_net"], g.loc[~g["own"], "rr_net"]
        if len(o) and len(c):
            paired.append(o.mean() - c.mean())
    piv = pd.Series(paired, dtype=float)
    diff = boot_ci(piv.to_numpy())
    rows = []
    for name, d in (("own_regime", own), ("cross_regime", cross), ("global_expert", glob)):
        h = _het(d)
        rows.append(dict(arm=name, episodes=len(d), trades=int(d["n_trades"].sum()), mean_rr=_pooled(d),
                         cells=h["cells"], tau=h["tau"], I2=h["I2"], raw_sd_within_cell=h["raw_sd"],
                         noise_sd=h["noise_sd"], spread_p10_p90=h["spread_p10_p90"],
                         share_ep_positive=float((d["rr_net"] > 0).mean()) if len(d) else np.nan,
                         gross_rr=float(np.average(d["rr_gross"], weights=d["n_trades"])) if len(d) else np.nan,
                         within_tol_of_backtest=float(((d["rr_net"] - d["ref_rr"]).abs() <= cfg.parity_tol).mean()) if len(d) else np.nan))
    plc = run["placebo"]
    if len(plc):
        rows.append(dict(arm="placebo_own (shifted labels)", episodes=int(plc["k"].sum()), trades=int(plc["n_trades"].sum()),
                         mean_rr=float(plc["mean_rr"].mean()), tau=float(plc["tau"].mean()), I2=float(plc["I2"].mean())))
    tab = pd.DataFrame(rows)
    tab.attrs["own_minus_cross_paired"] = diff
    tab.attrs["own_beats_placebo_p"] = (float(((plc["mean_rr"] >= _pooled(own)).sum() + 1) / (len(plc) + 1))
                                        if len(plc) else np.nan)
    tr = run["trades"]
    if len(tr):
        dd = {a: np.mean([max_drawdown(g.sort_values("row")["rr_net"].to_numpy())
                          for _, g in d.groupby("fold")]) for a, d in tr.groupby("arm")}
        tab.attrs["max_drawdown_rr_mean_over_folds"] = dd
    return tab


def metric_value(run: dict) -> pd.DataFrame:
    """H3: does similarity rank experts inside an episode, and does high similarity mean consistent economics?"""
    p, cfg = run["pairs"], run["cfg"]
    rows = []
    reg = p[(p["expert"] != "global") & (p["n_trades"] >= cfg.min_trades)] if len(p) else p
    for m in cfg.metrics:
        if m not in reg or reg[m].notna().sum() < 10:
            continue
        d = reg[reg[m].notna()]
        rc, lift, acc, gap = [], [], [], []
        for _, g in d.groupby(["fold", "ep"]):
            if len(g) < 3:
                continue
            rc.append(spearman(g[m].to_numpy(), g["rr_net"].to_numpy(), min_n=3))
            lift.append(g.loc[g[m].idxmax(), "rr_net"] - g["rr_net"].mean())
            acc.append(float(g.loc[g[m].idxmax(), "own"]))
            if g["own"].any() and (~g["own"]).any():
                gap.append(g.loc[g["own"], m].mean() - g.loc[~g["own"], m].mean())
        lo_q, hi_q = d[m].quantile([1 / 3, 2 / 3])
        hi, lo = d[d[m] >= hi_q], d[d[m] <= lo_q]
        hh, hl = _het(hi), _het(lo)
        flip = np.nan
        sg = hi[np.isfinite(hi["ic"]) & np.isfinite(hi["ref_ic"])]
        if len(sg):
            flip = float((np.sign(sg["ic"]) != np.sign(sg["ref_ic"])).mean())
        rows.append(dict(metric=m, pairs=len(d), episodes=len(acc),
                         within_episode_rank_corr=float(np.nanmean(rc)) if np.isfinite(rc).any() else np.nan,
                         top1_lift_rr=boot_ci(lift)[0], top1_lift_lo=boot_ci(lift)[1], top1_lift_hi=boot_ci(lift)[2],
                         top1_is_assigned=float(np.mean(acc)) if acc else np.nan,
                         sim_gap_own_minus_cross=float(np.mean(gap)) if gap else np.nan,
                         rr_high_sim=_pooled(hi), rr_low_sim=_pooled(lo), tau_high_sim=hh["tau"], tau_low_sim=hl["tau"],
                         within_tol_high=float(((hi["rr_net"] - hi["ref_rr"]).abs() <= cfg.parity_tol).mean()),
                         within_tol_low=float(((lo["rr_net"] - lo["ref_rr"]).abs() <= cfg.parity_tol).mean()),
                         sign_flip_rate_high_sim=flip))
    return pd.DataFrame(rows).set_index("metric") if rows else pd.DataFrame()


def compare(results: dict) -> pd.DataFrame:
    """One row per run: persistence + economics + own-vs-cross.  Use for H1 (feature sets) and H2 (encoder x segmenter)."""
    rows = []
    for name, run in results.items():
        cfg = run["cfg"]
        row = dict(run=name, feature_set=cfg.feature_set, encoder=cfg.encoder, segmenter=cfg.segmenter)
        if len(run["persistence"]):
            row.update(run["persistence"].drop(columns="fold").mean(numeric_only=True)[
                ["test_median_len", "test_median_sec", "test_tradable_share", "test_flicker_share", "test_n_latents",
                 "test_share_null", "test_recurring_latents", "test_transitions_per_10k"]].to_dict())
        eco = economics(run)
        if len(eco):
            e = eco.set_index("arm")
            row.update(own_rr=e.loc["own_regime", "mean_rr"], cross_rr=e.loc["cross_regime", "mean_rr"],
                       global_rr=e.loc["global_expert", "mean_rr"] if "global_expert" in e.index else np.nan,
                       own_tau=e.loc["own_regime", "tau"], cross_tau=e.loc["cross_regime", "tau"],
                       own_minus_cross=eco.attrs["own_minus_cross_paired"][0],
                       omc_lo=eco.attrs["own_minus_cross_paired"][1], omc_hi=eco.attrs["own_minus_cross_paired"][2],
                       p_vs_placebo=eco.attrs["own_beats_placebo_p"])
        mv = metric_value(run)
        for m in mv.index:
            row[f"gap_{m}"] = mv.loc[m, "sim_gap_own_minus_cross"]
            row[f"top1_{m}"] = mv.loc[m, "top1_is_assigned"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


def report(results: dict) -> None:
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
    print("\n=== COMPARISON (one row per run) ===")
    print(compare(results).T)
    for name, run in results.items():
        print(f"\n=== {name}: economics (net RR / trade, OOS) ===")
        eco = economics(run)
        print(eco.to_string(index=False) if len(eco) else "no pairs")
        if len(eco):
            d = eco.attrs["own_minus_cross_paired"]
            print(f"own - cross, paired by episode: {d[0]:+.3f} RR  [95% CI {d[1]:+.3f}, {d[2]:+.3f}]   "
                  f"P(placebo >= own) = {eco.attrs['own_beats_placebo_p']:.3f}")
            print("max drawdown (RR, mean over folds):", {k: round(v, 2) for k, v in eco.attrs.get('max_drawdown_rr_mean_over_folds', {}).items()})
        mv = metric_value(run)
        if len(mv):
            print(f"--- {name}: similarity metrics ---")
            print(mv.T)
    print("\nREAD ME: InfoNet is ex-post and sign-blind; see sign_flip_rate_high_sim and the global/placebo arms before "
          "concluding anything from a high similarity score.  Episode-level RR has SE ~ 1/sqrt(n_trades): use tau/I2, "
          "not the raw spread, to judge consistency.")