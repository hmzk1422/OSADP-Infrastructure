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
import gc
import os
import pickle
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from ...strategy.moe.ood.base_regime_encoder import make_encoder
from ...strategy.moe.ood.base_regime_segmenter import make_segmenter
from ...strategy.moe.ood.metrics import (cosine_similarity, infonet_estimate, resampled_distance_similarity,
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
NOISE_IDS = {"-1", "-1.0", "noise", "nan", "None", "NaN"}


def _as_labels(ids) -> np.ndarray:
    labels = pd.Series(np.asarray(ids, dtype=object)).astype("string")
    labels = labels.mask(labels.isna() | labels.isin(NOISE_IDS), "null")
    return labels.to_numpy(dtype=object)


def _level(side: str, kind: str, k: int) -> str:
    return f"best_{side}_{kind}" if k == 1 else f"{side}_{kind}_{k}"


def prepare_l2(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """ffill the raw L2 frame and add the ``*_close`` aliases ``add_orderflow`` expects.

    NOTE on trades: after ffill every book row carries the LAST trade, so using ``quantity`` as is would count
    each trade once per following book update.  A row is a *new* trade only when ``agg_trade_id`` changes, and
    ``quantity`` is zeroed elsewhere.  Without this, all trade-flow features are inflated and autocorrelated.
    Returns (frame, number_of_leading_rows_dropped).
    """
    raw = raw.reset_index(drop=True)
    # Keep the row-level availability signal before ffill. Once files with no
    # book snapshots are concatenated, ffill alone would carry an old quote
    # across the entire file and make volatility appear identically zero.
    book_present = (pd.to_numeric(raw["best_bid_price"], errors="coerce").notna() &
                    pd.to_numeric(raw["best_ask_price"], errors="coerce").notna()).to_numpy()
    df = raw.ffill()
    core = ["best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty", "price", "is_buyer_maker"]
    first = int(df[core].notna().all(axis=1).to_numpy().argmax())
    df = df.iloc[first:].reset_index(drop=True)
    book_present = book_present[first:]
    tid = df["agg_trade_id"]
    new_trade = (tid.ne(tid.shift()) & tid.notna())
    new_trade.iloc[0] = False
    df["new_trade"] = new_trade.to_numpy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").where(new_trade, 0.0)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(float).astype(bool)
    # Cached mid_price can be stale even when fully populated. Use current
    # quotes when available; bookless rows use the causally ffilled trade
    # price instead of a quote carried across an entire bookless file.
    bid = pd.to_numeric(df["best_bid_price"], errors="coerce")
    ask = pd.to_numeric(df["best_ask_price"], errors="coerce")
    quote_mid = (bid + ask) / 2
    valid_quote = book_present & np.isfinite(quote_mid) & (bid > 0) & (ask >= bid)
    df["_book_available"] = valid_quote
    trade_mid = pd.to_numeric(df["price"], errors="coerce")
    df["mid_price"] = quote_mid.where(valid_quote, trade_mid)
    # Do not let the global ffill above invent a book state for a whole file
    # that had no replayed LOB rows. The trade price remains usable as the
    # forecast reference, while book-derived features are neutralized below.
    book_cols = [c for c in df.columns if c in ("best_bid_price", "best_ask_price")
                 or c.startswith(("bid_price_", "bid_qty_", "ask_price_", "ask_qty_", "best_bid_qty", "best_ask_qty"))]
    df.loc[~valid_quote, book_cols] = np.nan
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
    # Pre-extract df-only columns so the full-resolution frame can be freed
    # immediately after add_orderflow deep-copies it — otherwise both the
    # original (~1.5 GiB) and the featurized copy (~2.6 GiB) live at once.
    new_trade = df["new_trade"].to_numpy()
    mid_price = df["mid_price"].to_numpy()
    book_available = df["_book_available"].to_numpy(dtype=bool)
    regime_truth = df["regime_truth"].to_numpy() if "regime_truth" in df else None
    out, feat19 = add_orderflow(df, windows=(1, 5, 20, 100), baseline_window=baseline_window, min_periods=min_periods)
    del raw, df                                          # free original full-resolution frames (~1.5 GiB)
    out["new_trade"] = new_trade
    # add_orderflow reconstructs mid from the last ffilled quote. Restore the
    # row-aware mid chosen above so bookless files use their trade-price path.
    out["mid_price"] = mid_price
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_efficiency = out["mid_price"].diff(5) / (
            out["aggressive_signed_notional"].rolling(5, min_periods=1).sum().abs() + 1.0)
    out["flow_price_efficiency_z_5"] = _robust_zscore(
        flow_efficiency, baseline_window=baseline_window, min_periods=min_periods)
    allf = out[feat19].copy()
    for name, col in _extra_features(out, baseline_window, min_periods).items():
        allf[name] = col
    book_dependent = [c for c in allf.columns if c.startswith((
        "spread_bps_z", "queue_", "microprice_", "ofi_", "depth_imbalance_",
        "bid_depth_change_", "ask_depth_change_", "spread_mean_", "touch_concentration",
        "bid_dist_", "ask_dist_", "slope_imbalance", "wdepth_imbalance", "imb5_change_",
        "total_depth_", "bid_depth_", "ask_depth_"))]
    book_missing = ~book_available
    if book_missing.any() and book_dependent:
        allf.loc[book_missing, book_dependent] = 0.0
    for c in allf.columns:
        if c in ZERO_FILL:                       # no trade in the window == no net flow, not "unknown"
            allf[c] = allf[c].fillna(0.0)
    need = [c for c in FEATURE_SETS["40"]]
    ok = allf[need].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
    warm = int(ok.argmax()) if ok.any() else len(allf)
    allf = allf.iloc[warm:].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    keep = slice(None, None, decimate)
    allf = allf.iloc[keep].reset_index(drop=True)
    mid = mid_price[warm:][keep]
    _ask = np.where(book_available, out["best_ask_price"].to_numpy(dtype=float), np.nan)
    _bid = np.where(book_available, out["best_bid_price"].to_numpy(dtype=float), np.nan)
    spread = (_ask - _bid)[warm:][keep]
    ts_sec = ts_sec[warm:][keep]
    seg = seg_raw[warm:][keep]
    truth = regime_truth[warm:][keep] if regime_truth is not None else None
    frames = {"_base": allf}
    n = len(allf)
    if "decoy40" in feature_sets:
        base = allf[CORE_10].copy()
        core_values = allf[CORE_10].to_numpy()
        for i in (1, 2, 3):
            shifted = np.roll(core_values, n * i // 4, axis=0)
            for j, c in enumerate(CORE_10):
                base[f"decoy{i}_{c}"] = shifted[:, j]
        frames["decoy40"] = base
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
    del parts
    fold_data = build_fold_features(raw, np.concatenate(seg), d.feature_sets, d.decimate, d.baseline_window,
                                    d.min_periods, d.ts_unit, fold)
    del raw
    return fold_data


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
    min_latents: int = 2            # HDBSCAN is re-fitted with smaller clusters until it yields >= this many latents
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
    strategy: str = "rr"               # "rr" legacy trade evaluation or "forecast_mae"
    trade_horizon: int = 16
    entry_quantile: float = 0.6
    ridge_lambda: float = 0.1
    val_fraction: float = 0.25
    fee_bps: float = 0.0            # per side, on top of spread crossing
    max_cost_rr: float = 5.0        # reject entries whose spread + fee costs exceed 5 volatility units
    # episode / sampling rules
    min_episode_len: int = 400
    min_trades: int = 5
    min_val_trades: int = 20
    min_expert_rows: int = 500
    max_test_episodes: int = 60
    max_ref_episodes: int = 6
    parity_tol: float = 0.05
    n_placebo: int = 20
    vol_window: int = 100
    seed: int = 0


@dataclass
class ForecastExpert:
    """Standardized ridge forecaster with one output per forward horizon."""
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    y_mean: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w + self.y_mean


def forward_return_targets(mid: np.ndarray, horizons=(1, 2, 3, 4, 5), scale: float = 1e4) -> pd.DataFrame:
    """Cumulative log returns from t to t+h, in basis points; no volatility scaling."""
    p = np.asarray(mid, dtype=float)
    targets = {}
    valid_mid = np.isfinite(p) & (p > 0)
    for h in horizons:
        y = np.full(len(p), np.nan, dtype=float)
        if h > 0 and h < len(p):
            ok = valid_mid[:-h] & valid_mid[h:]
            y[:-h][ok] = scale * np.log(p[h:][ok] / p[:-h][ok])
        targets[f"logret_{h}"] = y
    return pd.DataFrame(targets)


def fit_forecast_expert(X: np.ndarray, Y: np.ndarray, lam: float = 0.1,
                        min_rows: int = 50) -> ForecastExpert | None:
    """Fit a multi-output ridge forecaster, retaining rows finite at every horizon."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if int(ok.sum()) < min_rows:
        return None
    X, Y = X[ok], Y[ok]
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    y_mean = Y.mean(0)
    n, d = Z.shape
    w = np.linalg.solve(Z.T @ Z / n + lam * np.eye(d), Z.T @ (Y - y_mean) / n)
    return ForecastExpert(mu, sd, w, y_mean)


def auto_min_features(d: int, alpha: float, tail: float = 0.05) -> int:
    """Smallest m with P(Binomial(d, alpha) >= m) <= tail: holds the null block pass-rate fixed as D grows."""
    for m in range(1, d + 1):
        p = sum(math.comb(d, k) * alpha ** k * (1 - alpha) ** (d - k) for k in range(m, d + 1))
        if p <= tail:
            return m
    return d


def stable_rr_targets(price, horizons, vol_window: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    """Forward RR targets and cost vol, rejecting numerical flats with a causal local floor."""
    series = pd.Series(np.asarray(price, dtype=np.float64))
    min_periods = max(vol_window // 2, 2)
    vol = series.diff().rolling(vol_window, min_periods=min_periods).std(ddof=0)
    price_scale = float(np.nanmedian(np.abs(series.to_numpy()))) if len(series) else 0.0
    numeric_floor = np.finfo(float).eps * max(price_scale, 1.0)
    change = series.diff()
    move_floor = max(10 * numeric_floor, 1e-10 * max(price_scale, 1.0))
    meaningful_steps = change.abs().where(change.abs() > move_floor)
    local_step = meaningful_steps.rolling(vol_window, min_periods=1).median()
    has_real_move = meaningful_steps.notna().rolling(vol_window, min_periods=1).sum() > 0
    floor = np.maximum(numeric_floor, 0.2 * local_step / math.sqrt(vol_window))
    vol = vol.where(np.isfinite(vol) & (vol > floor) & has_real_move)
    out = {f"rr_{int(h)}": series.diff(int(h)).shift(-int(h)) / (vol * math.sqrt(int(h)))
           for h in horizons}
    targets = pd.DataFrame(out, index=series.index).replace([np.inf, -np.inf], np.nan)
    return targets, vol.to_numpy(dtype=float)


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
        # Bound HDBSCAN's expensive fit sample and use a less restrictive
        # default so sparse-but-real clusters are less likely to be all noise.
        bp = {"min_cluster_size": 200, "max_fit_samples": 20_000,
              **user.pop("backend_params", {})}
        return "hdbscan", {**kw, "backend_params": bp, **user}
    return cfg.segmenter, {**kw, **user}


def discover_regimes(frame: pd.DataFrame, seg: np.ndarray, cfg: RunConfig, n_tr_fit_end: int, truth=None):
    """Fit encoder + segmenter on TRAIN rows only; label train and test rows.  Returns (labels, embeddings)."""
    cols = list(frame.columns)
    tr = np.flatnonzero(seg == 0)
    te = np.flatnonzero(seg == 2)
    fit_tr = tr[:n_tr_fit_end]
    enc = make_encoder(cfg.encoder, window=cfg.window, embedding_dim=cfg.embedding_dim,
                       feature_columns=cols, **cfg.encoder_kwargs)
    enc.fit_encoder(frame.iloc[tr[:n_tr_fit_end]])
    emb = enc.transform(frame)                       # causal normalisation; weights learnt on train only
    if cfg.standardize_embeddings:
        ref = emb.iloc[fit_tr].dropna()
        emb = (emb - ref.mean()) / ref.std(ddof=0).where(lambda x: x > 1e-12, 1.0)
    labels = np.full(len(frame), "null", dtype=object)
    if cfg.segmenter == "oracle":                    # positive control: needs a 'regime_truth' column from the loader
        assert truth is not None, "segmenter='oracle' needs ground-truth regimes (synthetic loader only)"
        labels[:] = [f"truth_{int(x)}" for x in truth]
        labels[seg == 1] = "gap"
        return labels, emb
    sg_name, sg_kw = _seg_kwargs(cfg)
    e_fit = emb.iloc[fit_tr].dropna()
    e_tr = emb.iloc[tr].dropna()
    e_te = emb.iloc[te].dropna()
    # HDBSCAN can legitimately find no clusters when its default minimum
    # cluster size is too large for the embedding. Retry at smaller sizes;
    # if the data still has no clusters, keep the fold usable and record null
    # labels instead of failing the whole run.
    if cfg.segmenter == "hdbscan":
        if not len(e_fit):
            labels[seg == 1] = "gap"
            print(f"  [{cfg.name}]: HDBSCAN has no finite training embeddings; recording null regimes")
            return labels, emb
        backend_params = dict(sg_kw.get("backend_params", {}))
        initial_size = int(backend_params.get("min_cluster_size", 200))
        sizes = list(dict.fromkeys([initial_size, max(20, initial_size // 2),
                                    max(10, initial_size // 5), 10]))
        sg, pred_tr, best_n, used = None, None, -1, initial_size
        for min_size in sizes:
            retry_kw = {**sg_kw, "backend_params": {**backend_params, "min_cluster_size": min_size}}
            candidate = make_segmenter(sg_name, **retry_kw)
            try:
                candidate.fit(e_fit)
                pred_try = candidate.predict(e_tr)
            except ValueError as exc:
                if "found no clusters" not in str(exc).lower():
                    raise
                continue
            n_lat = len(set(_as_labels(pred_try["latent_id"].to_numpy())) - {"null"})
            if n_lat > best_n:                       # keep the most informative solution seen so far
                sg, pred_tr, best_n, used = candidate, pred_try, n_lat, min_size
            if n_lat >= cfg.min_latents:             # a single cluster cannot separate regimes: keep shrinking
                break
        if sg is None:
            labels[seg == 1] = "gap"
            print(f"  [{cfg.name}]: HDBSCAN found no clusters after retries (min_cluster_size={sizes}); recording null regimes")
            return labels, emb
        if used != initial_size or best_n < cfg.min_latents:
            print(f"  [{cfg.name}] HDBSCAN used min_cluster_size={used} -> {best_n} latent(s) "
                  f"(wanted >= {cfg.min_latents}, started at {initial_size})")
    else:
        sg = make_segmenter(sg_name, **sg_kw)
        pred_tr = None
    if pred_tr is None:
        if sg.backend != "custom":                   # custom backends learn online inside predict()
            sg.fit(e_fit)
        pred_tr = sg.predict(e_tr)
    labels[e_tr.index.to_numpy()] = _as_labels(pred_tr["latent_id"].to_numpy())
    sg.learn_novel = bool(cfg.learn_novel_oos)
    if len(e_te):
        pred_te = sg.predict(e_te)
        labels[e_te.index.to_numpy()] = _as_labels(pred_te["latent_id"].to_numpy())
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
    if cfg.feature_set == "decoy40":
        frame = fd.frames["decoy40"]
    elif "_base" in fd.frames:
        frame = fd.frames["_base"][FEATURE_SETS[cfg.feature_set]]
    else:  # compatibility with FoldData built by older callers
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
    rr_df, vol = stable_rr_targets(fd.mid, cfg.horizons, cfg.vol_window)
    rr = rr_df.to_numpy()
    rr_h = rr_df[f"rr_{h}"].to_numpy()
    exit_spread = np.full(T, np.nan)
    exit_spread[:-h] = fd.spread[h:]
    cost = ((fd.spread + exit_spread) / 2 + 2 * cfg.fee_bps * 1e-4 * fd.mid) / (vol * math.sqrt(h))
    cost = np.where(np.isfinite(cost) & (cost >= 0) & (cost <= cfg.max_cost_rr), cost, np.nan)

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


def _tail_average(values: np.ndarray, upper: bool) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    q = np.quantile(values, 0.95 if upper else 0.05)
    tail = values[values >= q] if upper else values[values <= q]
    return float(tail.mean()) if len(tail) else np.nan


def _episode_correlation(cells: pd.DataFrame) -> pd.DataFrame:
    """Pairwise expert correlations with explicit reasons where correlation is undefined."""
    experts = sorted(cells["expert_latent"].unique()) if len(cells) else []
    matrix = pd.DataFrame(index=experts, columns=experts, dtype=object)
    by_episode = cells.pivot(index="ep", columns="expert_latent", values="rr_net") if len(cells) else pd.DataFrame()
    for left in experts:
        for right in experts:
            pair = by_episode[[left, right]].dropna() if left != right else by_episode[[left]].dropna()
            a = pair[left].to_numpy(dtype=float)
            if left == right:
                matrix.loc[left, right] = 1.0 if len(a) >= 2 and np.ptp(a) > 0 else ("constant" if len(a) >= 2 else "n<2")
                continue
            b = pair[right].to_numpy(dtype=float)
            if len(a) < 3:
                value = "n<3"
            elif np.ptp(a) == 0 or np.ptp(b) == 0:
                value = "constant"
            else:
                value = float(np.corrcoef(a, b)[0, 1])
            matrix.loc[left, right] = value
    return matrix


def run_simple_fold(fd: FoldData, cfg: RunConfig) -> dict:
    """Half-train discovery, half-train identification, and compact OOS metrics."""
    if cfg.strategy == "forecast_mae":
        return run_forecast_fold(fd, cfg)
    frame = fd.frames["decoy40"] if cfg.feature_set == "decoy40" else fd.frames["_base"][FEATURE_SETS[cfg.feature_set]]
    F = frame.to_numpy(dtype=float)
    T = len(F)
    seg = fd.seg
    tr, te = np.flatnonzero(seg == 0), np.flatnonzero(seg == 2)
    if len(tr) < 4 or not len(te):
        return dict(fold=fd.fold, skipped="too few train or test rows", expectancy=pd.DataFrame(),
                    persistence=pd.DataFrame(), parity=pd.DataFrame(), correlation=pd.DataFrame(),
                    cells=pd.DataFrame(), sim_pairs=pd.DataFrame())

    discovery_n = max(2, len(tr) // 2)
    rr_df, vol = stable_rr_targets(fd.mid, cfg.horizons, cfg.vol_window)
    h = cfg.trade_horizon
    rr = rr_df[f"rr_{h}"].to_numpy(dtype=float, copy=True)
    # A training target at t reads price[t+h]. Purge targets whose forward
    # interval leaves the training segment, including into a gap or test set.
    train_rows = seg == 0
    future_train = np.zeros(T, dtype=bool)
    if h < T:
        window_counts = np.convolve(train_rows.astype(np.int8), np.ones(h + 1, dtype=np.int8), mode="valid")
        future_train[:T - h] = window_counts[:T - h] == h + 1
    rr[train_rows & ~future_train] = np.nan
    exit_spread = np.full(T, np.nan)
    exit_spread[:-h] = fd.spread[h:]
    cost = ((fd.spread + exit_spread) / 2 + 2 * cfg.fee_bps * 1e-4 * fd.mid) / (vol * math.sqrt(h))
    cost_raw = cost.copy()
    rejected_costs = int(np.sum(np.isfinite(cost) & ((cost < 0) | (cost > cfg.max_cost_rr))))
    cost = np.where(np.isfinite(cost) & (cost >= 0) & (cost <= cfg.max_cost_rr), cost, np.nan)
    finite_cost = cost[np.isfinite(cost)]
    cost_diag = ({f"p{p}": float(x) for p, x in zip((50, 99, 99.9, 100),
                  np.percentile(finite_cost, (50, 99, 99.9, 100))) } if len(finite_cost) else {})

    labels, emb_df = discover_regimes(frame, seg, cfg, n_tr_fit_end=discovery_n, truth=fd.truth)
    ident = tr[discovery_n:]
    ident_runs = runs_of(labels, int(ident[0]), int(tr[-1]) + 1) if len(ident) else []
    counts: dict[str, int] = {}
    for s, e, k in ident_runs:
        if e - s >= cfg.min_episode_len:
            counts[k] = counts.get(k, 0) + 1
    recurring = {k for k, n in counts.items() if n >= 2}

    experts: dict[str, Expert] = {}
    ident_mask = np.zeros(T, dtype=bool)
    ident_mask[ident] = True
    for k in sorted(recurring):
        train_rows = ident_mask & (labels == k) & np.isfinite(rr)
        if train_rows.sum() < cfg.min_expert_rows:
            continue
        expert = fit_expert(F[train_rows], rr[train_rows], cfg.ridge_lambda, cfg.entry_quantile)
        if expert is not None:
            experts[k] = expert

    episodes = [(s, e, k) for s, e, k in runs_of(labels, int(te[0]), int(te[-1]) + 1)
                if e - s >= cfg.min_episode_len and k in experts]
    cell_rows = []
    predictions = {k: expert.predict(F) for k, expert in experts.items()}
    for ep, (s, e, assigned) in enumerate(episodes):
        rows = np.arange(s, e)
        for expert_id, expert in experts.items():
            take, net, _ = trade_rows(predictions[expert_id], expert.thr, rr, cost, rows, h)
            if len(net):
                cell_rows.append(dict(ep=ep, start=s, end=e, assigned_latent=assigned, expert_latent=expert_id,
                                      n_trades=len(net), rr_net=float(net.mean()),
                                      rr_sd=float(net.std(ddof=1)) if len(net) > 1 else np.nan))
    cells = pd.DataFrame(cell_rows)

    # ---- similarity of every sampled test episode to every latent's reference (train identification half) ----
    sim_rows = []
    if cfg.metrics and experts and episodes:
        try:
            ok_tr = np.isfinite(F[tr]).all(axis=1)
            mu_z, sd_z = F[tr][ok_tr].mean(0), F[tr][ok_tr].std(0)
            sd_z = np.where(sd_z > 1e-12, sd_z, 1.0)
            sim = Similarity(cfg, F, (F - mu_z) / sd_z, emb_df.to_numpy(dtype=float), rr_df.to_numpy(dtype=float))
            by_latent: dict = {}
            for s, e, k in ident_runs:
                if k in experts and e - s >= cfg.min_episode_len:
                    by_latent.setdefault(k, []).append((s, e))
            refs = {}
            for k, v in by_latent.items():
                try:
                    refs[k] = sim.build_ref(v)
                except Exception as exc:
                    print(f"  [{cfg.name}] fold {fd.fold}: reference for {k} failed ({type(exc).__name__}: {exc})")
            pick_rng = np.random.default_rng(cfg.seed + fd.fold)
            pick = set(pick_rng.choice(len(episodes), min(len(episodes), cfg.max_test_episodes),
                                       replace=False).tolist())
            for ep, (s, e, assigned) in enumerate(episodes):
                if ep not in pick:
                    continue
                try:
                    desc = sim.describe(s, e)
                except Exception:
                    continue
                for j, ref in refs.items():
                    sim_rows.append(dict(ep=ep, assigned_latent=assigned, ref_latent=j, **sim.sim(desc, ref)))
        except Exception as exc:
            print(f"  [{cfg.name}] fold {fd.fold}: similarity skipped ({type(exc).__name__}: {exc})")
    sim_pairs = pd.DataFrame(sim_rows)
    if len(cells):
        weighted = cells.assign(weighted_rr=cells["rr_net"] * cells["n_trades"])
        grouped = weighted.groupby(["expert_latent", "assigned_latent"], sort=True).agg(
            weighted_rr=("weighted_rr", "sum"), trades=("n_trades", "sum"))
        parity = (grouped["weighted_rr"] / grouped["trades"]).unstack("assigned_latent")
        correlation = _episode_correlation(cells)
    else:
        parity = pd.DataFrame()
        correlation = pd.DataFrame()

    expectancy_rows = []
    for k in sorted(experts):
        own = cells[(cells["expert_latent"] == k) & (cells["assigned_latent"] == k)] if len(cells) else pd.DataFrame()
        if own.empty:
            continue
        means = own["rr_net"].to_numpy(dtype=float) if len(own) else np.array([])
        weights = own["n_trades"].to_numpy(dtype=float) if len(own) else np.array([])
        expectancy_rows.append(dict(
            latent=k, episodes=len(own), trades=int(weights.sum()) if len(weights) else 0,
            expectancy=float(np.average(means, weights=weights)) if len(means) else np.nan,
            cvar005=_tail_average(means, upper=False), cvar095=_tail_average(means, upper=True)))
    expectancy = pd.DataFrame(expectancy_rows)

    duration_rows = []
    duration_runs = [(s, e, k) for s, e, k in runs_of(labels, int(te[0]), int(te[-1]) + 1)
                     if e - s >= cfg.min_episode_len and k in experts]
    for k in sorted(experts):
        subset = [(s, e) for s, e, label in duration_runs if label == k]
        if not subset:
            continue
        row_durations = np.asarray([e - s for s, e in subset], dtype=float)
        sec_durations = np.asarray([fd.ts_sec[e - 1] - fd.ts_sec[s] for s, e in subset], dtype=float)
        duration_rows.append(dict(
            latent=k, episodes=len(subset), mean_duration_rows=float(row_durations.mean()) if len(row_durations) else np.nan,
            cvar005_rows=_tail_average(row_durations, upper=False), cvar095_rows=_tail_average(row_durations, upper=True),
            mean_duration_sec=float(sec_durations.mean()) if len(sec_durations) else np.nan,
            cvar005_sec=_tail_average(sec_durations, upper=False), cvar095_sec=_tail_average(sec_durations, upper=True)))

    return dict(fold=fd.fold, discovery_rows=discovery_n, identification_rows=len(ident),
                recurring_latents=sorted(experts), expectancy=expectancy,
                persistence=pd.DataFrame(duration_rows), parity=parity, correlation=correlation,
                cost_percentiles=cost_diag, rejected_costs=rejected_costs,
                vol_filtered_rows=int((~np.isfinite(vol)).sum()),
                eligible_test_rows=int(np.sum(np.isfinite(rr[te]) & np.isfinite(cost[te]))),
                cells=cells, sim_pairs=sim_pairs,
                health=dict(test_rows=int(len(te)), test_vol_nan=int((~np.isfinite(vol[te])).sum()),
                            test_cost_rejected=int(np.sum(np.isfinite(cost_raw[te]) &
                                                          ((cost_raw[te] < 0) | (cost_raw[te] > cfg.max_cost_rr)))),
                            test_rr_nan=int((~np.isfinite(rr[te])).sum()), max_cost_rr=cfg.max_cost_rr))


def run_forecast_fold(fd: FoldData, cfg: RunConfig) -> dict:
    """Regime-conditioned multi-horizon forecasting, scored by out-of-sample MAE."""
    frame = fd.frames["decoy40"] if cfg.feature_set == "decoy40" else fd.frames["_base"][FEATURE_SETS[cfg.feature_set]]
    F = frame.to_numpy(dtype=float)
    seg = np.asarray(fd.seg)
    tr, te = np.flatnonzero(seg == 0), np.flatnonzero(seg == 2)
    empty = dict(fold=fd.fold, forecast_metrics=pd.DataFrame(), forecast_cells=pd.DataFrame(),
                 sim_pairs=pd.DataFrame())
    if len(tr) < 4 or not len(te):
        return dict(**empty, skipped="too few train or test rows")
    horizons = tuple(int(h) for h in cfg.horizons)
    if not horizons or min(horizons) <= 0:
        raise ValueError("forecast horizons must be positive integers")
    maxh = max(horizons)
    Y = forward_return_targets(fd.mid, horizons).to_numpy(dtype=float)
    labels, emb_df = discover_regimes(frame, seg, cfg, n_tr_fit_end=max(2, len(tr) // 2), truth=fd.truth)
    ident_start = int(tr[len(tr) // 2])
    train_end = int(tr[-1]) + 1
    fit_rows = (np.arange(len(F)) >= ident_start) & (np.arange(len(F)) + maxh < train_end)
    fit_rows &= seg == 0
    # Require the entire forward path to remain in the training segment.
    fit_rows &= np.isfinite(Y).all(axis=1)
    experts: dict[str, ForecastExpert] = {}
    for latent in sorted(set(labels[fit_rows]) - {"null", "gap"}):
        mask = fit_rows & (labels == latent)
        expert = fit_forecast_expert(F[mask], Y[mask], cfg.ridge_lambda, cfg.min_expert_rows)
        if expert is not None:
            experts[latent] = expert
    global_expert = fit_forecast_expert(F[fit_rows], Y[fit_rows], cfg.ridge_lambda,
                                        cfg.min_expert_rows)
    eligible = np.zeros(len(F), dtype=bool)
    eligible[te] = np.arange(len(te)) < max(len(te) - maxh, 0)
    eligible &= np.isfinite(Y).all(axis=1)
    test_rows = np.flatnonzero(eligible)
    zero_mae = np.mean(np.abs(Y[test_rows]), axis=0) if len(test_rows) else np.full(len(horizons), np.nan)
    global_pred = global_expert.predict(F[test_rows]) if global_expert is not None and len(test_rows) else None
    global_mae = np.mean(np.abs(Y[test_rows] - global_pred), axis=0) if global_pred is not None else np.full(len(horizons), np.nan)
    cell_rows, metric_rows = [], []
    for expert_id, expert in experts.items():
        pred = expert.predict(F[test_rows]) if len(test_rows) else np.empty((0, len(horizons)))
        groups = sorted(set(labels[test_rows]) - {"null", "gap"})
        for assigned in groups:
            local = np.flatnonzero(labels[test_rows] == assigned)
            if not len(local):
                continue
            actual = Y[test_rows[local]]
            error = np.abs(actual - pred[local])
            for col, horizon in enumerate(horizons):
                base = float(np.mean(np.abs(actual[:, col])))
                mae = float(np.mean(error[:, col]))
                cell_rows.append(dict(fold=fd.fold, expert_latent=expert_id, assigned_latent=assigned,
                                      own=(expert_id == assigned), horizon=horizon, n=len(local),
                                      mae_bps=mae, zero_mae_bps=base,
                                      skill_vs_zero=1.0 - mae / base if base > 0 else np.nan))
        own = np.flatnonzero(labels[test_rows] == expert_id)
        for col, horizon in enumerate(horizons):
            mae = float(np.mean(np.abs(Y[test_rows[own], col] - pred[own, col]))) if len(own) else np.nan
            own_zero_mae = float(np.mean(np.abs(Y[test_rows[own], col]))) if len(own) else np.nan
            own_global_mae = (float(np.mean(np.abs(Y[test_rows[own], col] - global_pred[own, col])))
                              if len(own) and global_pred is not None else np.nan)
            metric_rows.append(dict(fold=fd.fold, expert_latent=expert_id, horizon=horizon, n=len(own),
                                    mae_bps=mae,
                                    zero_mae_bps=own_zero_mae,
                                    global_mae_bps=own_global_mae,
                                    skill_vs_zero=1.0 - mae / own_zero_mae if len(own) and own_zero_mae > 0 else np.nan,
                                    skill_vs_global=1.0 - mae / own_global_mae
                                    if len(own) and np.isfinite(own_global_mae) and own_global_mae > 0 else np.nan))
    cells = pd.DataFrame(cell_rows)
    metrics = pd.DataFrame(metric_rows)
    if global_expert is not None:
        global_rows = []
        for col, horizon in enumerate(horizons):
            global_rows.append(dict(fold=fd.fold, expert_latent="global", horizon=horizon, n=len(test_rows),
                                    mae_bps=float(global_mae[col]), zero_mae_bps=float(zero_mae[col]),
                                    global_mae_bps=float(global_mae[col]),
                                    skill_vs_zero=1.0 - global_mae[col] / zero_mae[col] if zero_mae[col] > 0 else np.nan,
                                    skill_vs_global=0.0))
        metrics = pd.concat([metrics, pd.DataFrame(global_rows)], ignore_index=True)
    print(f"  [{cfg.name}] fold {fd.fold}: forecast rows={len(test_rows)}, experts={len(experts)}, "
          f"horizons={horizons}, scored by MAE (bps)")
    return dict(fold=fd.fold, strategy="forecast_mae", horizons=horizons,
                recurring_latents=sorted(experts), forecast_metrics=metrics,
                forecast_cells=cells, sim_pairs=pd.DataFrame(),
                health=dict(train_rows=len(tr), test_rows=len(te), forecast_test_rows=len(test_rows),
                            train_target_rows=int(fit_rows.sum()), target_scale="cumulative log return, bps"))


def run_simple_study(data: DataConfig, runs: list[RunConfig], checkpoint_path: str,
                     verbose: bool = True) -> dict:
    """Run and checkpoint each encoder/segmenter/fold independently for safe resume."""
    if data.loader is None:
        from ...exchange.loader import load_L2
        data = replace(data, loader=load_L2)
    key = repr(({k: v for k, v in vars(data).items() if k != "loader"}, runs))
    results = {r.name: [] for r in runs}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as fh:
            saved = pickle.load(fh)
        if saved.get("key") != key:
            raise ValueError(f"Checkpoint {checkpoint_path} belongs to different settings; choose another path")
        results = {name: [x for x in saved.get("results", {}).get(name, []) if "error" not in x]
                   for name in results}
        if verbose:
            print(f"Resuming {sum(map(len, results.values()))} completed run-folds from {checkpoint_path}")

    def checkpoint():
        parent = os.path.dirname(os.path.abspath(checkpoint_path))
        os.makedirs(parent, exist_ok=True)
        temp = checkpoint_path + ".tmp"
        with open(temp, "wb") as fh:
            pickle.dump({"key": key, "results": results}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, checkpoint_path)

    for fold in range(data.n_folds):
        if all(any(x.get("fold") == fold for x in results[r.name]) for r in runs):
            continue
        t0 = time.time()
        fd = load_fold(data.loader, fold, data)
        if verbose:
            print(f"fold {fold}: {fd.meta['rows']} rows after warm-up/decimation ({time.time() - t0:.1f}s)")
        if not len(fd.seg) or not np.any(fd.seg == 0) or not np.any(fd.seg == 2):
            reason = "no usable rows" if not len(fd.seg) else "missing train or test rows"
            if verbose:
                print(f"fold {fold}: skipped ({reason})")
            for run in runs:
                results[run.name].append(dict(fold=fold, skipped=reason, expectancy=pd.DataFrame(),
                                              persistence=pd.DataFrame(), parity=pd.DataFrame(),
                                              correlation=pd.DataFrame()))
            checkpoint()
            del fd
            gc.collect()
            continue
        for run_idx, run in enumerate(runs):
            if any(x.get("fold") == fold for x in results[run.name]):
                continue
            try:
                result = run_simple_fold(fd, run)
                if verbose:
                    if run_idx == 0:
                        p = result.get("cost_percentiles", {})
                        if p:
                            print(f"  accepted cost RR: p50={p['p50']:.4g}, p99={p['p99']:.4g}, "
                                  f"p99.9={p['p99.9']:.4g}, max={p['p100']:.4g}")
                        else:
                            print("  accepted cost RR: no eligible rows")
                        print(f"  volatility-filtered rows={result.get('vol_filtered_rows', 0)}, "
                              f"cost-rejected rows={result.get('rejected_costs', 0)}")
                    print(f"  [{run.name}] recurring train latents={len(result.get('recurring_latents', []))}; "
                          f"test expectancy rows={len(result.get('expectancy', []))}")
                results[run.name].append(result)
            except Exception as exc:
                print(f"  [{run.name}] fold {fold} FAILED: {type(exc).__name__}: {exc}")
                results[run.name].append(dict(fold=fold, error=f"{type(exc).__name__}: {exc}", expectancy=pd.DataFrame(),
                                              persistence=pd.DataFrame(), parity=pd.DataFrame(),
                                              correlation=pd.DataFrame(), cells=pd.DataFrame(),
                                              sim_pairs=pd.DataFrame()))
            checkpoint()
            gc.collect()
        del fd
        gc.collect()
    return {r.name: dict(cfg=r, folds=results[r.name]) for r in runs}


# ----------------------------------------------------------------------------------------------
# 7. Driver + analysis
# ----------------------------------------------------------------------------------------------
def run_study(data: DataConfig, runs: list[RunConfig], verbose: bool = True,
              checkpoint_path: str | None = None) -> dict:
    """Fold-major loop (each fold is loaded and featurised ONCE, then every run config is executed on it)."""
    if data.loader is None:
        from ...exchange.loader import load_L2
        data = replace(data, loader=load_L2)
    res = {r.name: [] for r in runs}
    checkpoint_key = repr(({k: v for k, v in vars(data).items() if k != "loader"}, runs))
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as fh:
            saved = pickle.load(fh)
        if saved.get("key") != checkpoint_key:
            raise ValueError(f"Checkpoint {checkpoint_path} belongs to different study settings; choose another checkpoint path")
        res = {name: [row for row in saved.get("results", {}).get(name, []) if "error" not in row]
               for name in res}
        if verbose:
            done = sum(len(rows) for rows in res.values())
            print(f"Resuming from {checkpoint_path}: {done} completed run-folds already saved")

    def save_checkpoint():
        if not checkpoint_path:
            return
        parent = os.path.dirname(os.path.abspath(checkpoint_path))
        os.makedirs(parent, exist_ok=True)
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, "wb") as fh:
            pickle.dump({"key": checkpoint_key, "results": res}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, checkpoint_path)

    for f in range(data.n_folds):
        if all(any(row.get("fold") == f for row in res[r.name]) for r in runs):
            if verbose:
                print(f"fold {f}: all run-folds already checkpointed; skipping")
            continue
        t0 = time.time()
        fd = load_fold(data.loader, f, data)
        if verbose:
            print(f"fold {f}: {fd.meta['rows']} rows after warm-up/decimation ({time.time() - t0:.1f}s)")
        if not len(fd.seg) or not np.any(fd.seg == 0) or not np.any(fd.seg == 2):
            reason = "no usable rows" if not len(fd.seg) else "missing train or test rows"
            if verbose:
                print(f"fold {f}: skipped ({reason})")
            for r in runs:
                res[r.name].append(dict(fold=f, skipped=reason, pairs=pd.DataFrame(), trades=pd.DataFrame(),
                                        placebo=pd.DataFrame(), persistence={}))
            save_checkpoint()
            del fd
            gc.collect()
            continue
        for r in runs:
            if any(row.get("fold") == f for row in res[r.name]):
                continue
            try:
                result = run_fold(fd, r, verbose)
                res[r.name].append(result)
            except Exception as exc:                          # keep the grid alive; surface the failure
                print(f"  [{r.name}] fold {f} FAILED: {type(exc).__name__}: {exc}")
                res[r.name].append(dict(fold=f, error=str(exc), pairs=pd.DataFrame(), trades=pd.DataFrame(),
                                        placebo=pd.DataFrame(), persistence={}))
            save_checkpoint()
            gc.collect()
        del fd
        gc.collect()
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


# ----------------------------------------------------------------------------------------------
# 8. Final report: edge matrices, similarity-metric check, deviation summary
# ----------------------------------------------------------------------------------------------
ENC_NAMES = {"ts2vec": "TS2Vec", "tcn": "TCN", "vq-vae": "VQ-VAE"}
SEG_NAMES = {"hdbscan": "HDBSCAN", "kmeans": "KNN/Kmeans"}
_RULE = "-" * 96
METRIC_NAMES = {"infonet": "InfoNET", "cosine": "Cosine", "tdmi": "TDMI", "geometric": "Geometric"}


def _wmean_ci(rr, n, boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Trade-weighted mean of episode RR with a 95% bootstrap CI over EPISODES (NaN CI if < 3 episodes)."""
    rr, n = np.asarray(rr, dtype=float), np.asarray(n, dtype=float)
    m = float(np.average(rr, weights=n))
    if len(rr) < 3:
        return m, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(rr), (boot, len(rr)))
    bm = (rr[idx] * n[idx]).sum(1) / n[idx].sum(1)
    return m, float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def _fmt_cell(m: float, lo: float, hi: float, own: bool) -> str:
    s = f"{m:+.2f} +- [n<3]" if not np.isfinite(lo) else f"{m:+.2f} +- [{lo:+.2f}, {hi:+.2f}]"
    return s + (" *" if own else "")


def _edge_matrix(cells: pd.DataFrame) -> pd.DataFrame:
    experts = sorted(cells["expert_latent"].unique())
    assigned = sorted(cells["assigned_latent"].unique())
    n_ep = cells.drop_duplicates("ep").groupby("assigned_latent").size()
    table = pd.DataFrame("-", index=[f"{k} (model)" for k in experts],
                         columns=[f"{a} (n={int(n_ep.get(a, 0))})" for a in assigned])
    table.index.name = "model used / episodes"
    for (j, a), g in cells.groupby(["expert_latent", "assigned_latent"]):
        m, lo, hi = _wmean_ci(g["rr_net"], g["n_trades"])
        table.loc[f"{j} (model)", f"{a} (n={int(n_ep.get(a, 0))})"] = _fmt_cell(m, lo, hi, j == a)
    return table


def _why_empty(f: dict) -> str:
    if f.get("error"):
        return f"error: {f['error']}"
    if f.get("skipped"):
        return f"skipped: {f['skipped']}"
    if not f.get("recurring_latents"):
        return ("no regime recurred (>= 2 episodes of >= min_episode_len rows) in the identification half with "
                ">= min_expert_rows rows, so no expert could be fitted")
    h = f.get("health", {})
    return (f"{len(f['recurring_latents'])} expert(s) fitted but no tradable test trades: test rows={h.get('test_rows')}, "
            f"volatility-filtered={h.get('test_vol_nan')}, cost>{h.get('max_cost_rr')}R rejected={h.get('test_cost_rejected')}, "
            f"rows with valid RR and cost={f.get('eligible_test_rows')}")


def _sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) >= 3 else np.nan


def _clean_sd(g: pd.DataFrame) -> float:
    """SD of episode means minus the part explained by trade-count noise (rr_sd^2 / n_trades)."""
    se2 = float(np.nanmean((g["rr_sd"] ** 2 / g["n_trades"]).to_numpy())) if g["rr_sd"].notna().any() else 0.0
    return math.sqrt(max(0.0, float(np.var(g["rr_net"].to_numpy(), ddof=1)) - se2)) if len(g) >= 3 else np.nan


def _wavg(vals, weights) -> float:
    v, w = np.asarray(vals, dtype=float), np.asarray(weights, dtype=float)
    ok = np.isfinite(v)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan


def run_deviation(run: dict, min_own_episodes: int = 10) -> dict:
    """Edge deviation = SD of episode-level net RR/trade.

    own   : expert k on test episodes assigned to regime k   (SD inside each fold x expert cell, episode-weighted)
    cross : expert j on test episodes assigned to other regimes (SD inside each fold x expert, episode-weighted)
    SD instead of max-min: a range grows with the number of episodes and is set by one outlier.
    ``*_clean`` removes the share of the SD that is just trade-count noise.
    """
    own = dict(sd=[], clean=[], w=[]); cross = dict(sd=[], clean=[], w=[])
    own_rr, own_n, cross_rr, cross_n = [], [], [], []
    max_latents = 0
    for f in run["folds"]:
        c = f.get("cells")
        if c is None or c.empty:
            continue
        max_latents = max(max_latents, c["expert_latent"].nunique())
        for j, gj in c.groupby("expert_latent"):
            o, x = gj[gj["assigned_latent"] == j], gj[gj["assigned_latent"] != j]
            for g, bucket, rr_l, n_l in ((o, own, own_rr, own_n), (x, cross, cross_rr, cross_n)):
                if len(g):
                    rr_l.extend(g["rr_net"]); n_l.extend(g["n_trades"])
                if len(g) >= 3:
                    bucket["sd"].append(_sd(g["rr_net"].to_numpy())); bucket["clean"].append(_clean_sd(g)); bucket["w"].append(len(g))
    res = dict(own_eps=len(own_rr), cross_eps=len(cross_rr), latents=max_latents,
               own_exp=_wavg(own_rr, own_n) if own_rr else np.nan, cross_exp=_wavg(cross_rr, cross_n) if cross_rr else np.nan,
               own_sd=_wavg(own["sd"], own["w"]), own_clean=_wavg(own["clean"], own["w"]),
               cross_sd=_wavg(cross["sd"], cross["w"]), cross_clean=_wavg(cross["clean"], cross["w"]))
    why = []
    if max_latents < 2:
        why.append(f"only {max_latents} regime(s) with an expert in any fold")
    if res["own_eps"] < min_own_episodes:
        why.append(f"only {res['own_eps']} own-regime episodes (< {min_own_episodes})")
    if not np.isfinite(res["own_sd"]):
        why.append("no expert/regime cell with >= 3 episodes")
    res["eligible"], res["why"] = not why, "; ".join(why)
    return res


def run_similarity(run: dict) -> list[dict]:
    """Per metric: is an episode more similar to ITS OWN latent's reference than to the other latents' references?"""
    parts = [f["sim_pairs"].assign(fold=f["fold"]) for f in run["folds"]
             if f.get("sim_pairs") is not None and len(f["sim_pairs"])]
    if not parts:
        return []
    sp = pd.concat(parts, ignore_index=True)
    rows = []
    for m in run["cfg"].metrics:
        if m not in sp:
            continue
        own_v, cross_v, hit, chance = [], [], [], []
        for _, g in sp.dropna(subset=[m]).groupby(["fold", "ep"]):
            same = g["ref_latent"] == g["assigned_latent"]
            if not same.any() or same.all():
                continue
            own_v.append(float(g.loc[same, m].iloc[0])); cross_v.append(float(g.loc[~same, m].mean()))
            hit.append(bool(g.loc[g[m].idxmax(), "ref_latent"] == g["assigned_latent"].iloc[0])); chance.append(1 / len(g))
        if not own_v:
            continue
        gap = np.asarray(own_v) - np.asarray(cross_v)
        gm, glo, ghi = boot_ci(gap)
        rows.append(dict(metric=METRIC_NAMES.get(m, m), episodes=len(gap), own=float(np.mean(own_v)), other=float(np.mean(cross_v)),
                         gap=gm, lo=glo, hi=ghi, top1=float(np.mean(hit)), chance=float(np.mean(chance))))
    return rows


def final_report(results: dict) -> str:
    cfgs = [r.get("cfg") for r in results.values()]
    if cfgs and all(getattr(c, "strategy", "rr") == "forecast_mae" for c in cfgs):
        out = ["Forecast target: cumulative log return from t to t+h, in basis points.",
               "Scoring: out-of-sample mean absolute error (MAE); lower is better.",
               "Zero and global ridge forecasts are reported as baselines. Skill = 1 - model MAE / baseline MAE."]
        for name, run in results.items():
            out.append(f"\n{_RULE}\n{name}\n{_RULE}")
            fold_results = []
            for fold in run.get("folds", []):
                fm = fold.get("forecast_metrics")
                if fm is not None and len(fm):
                    fold_results.append(fm)
                    out.append(f"[fold {fold.get('fold')}]")
                    show = fm[["expert_latent", "horizon", "n", "mae_bps", "zero_mae_bps", "global_mae_bps",
                               "skill_vs_zero", "skill_vs_global"]]
                    out.append(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
                else:
                    out.append(f"[fold {fold.get('fold')}] no forecast rows: {fold.get('skipped', fold.get('error', 'no expert met minimum rows'))}")
            if fold_results:
                joined = pd.concat(fold_results, ignore_index=True)
                out.append("\nAcross folds (mean by expert and horizon):")
                out.append(joined.groupby(["expert_latent", "horizon"], as_index=False).agg(
                    folds=("fold", "nunique"), rows=("n", "sum"), mae_bps=("mae_bps", "mean"),
                    zero_mae_bps=("zero_mae_bps", "mean"), global_mae_bps=("global_mae_bps", "mean"),
                    skill_vs_zero=("skill_vs_zero", "mean"), skill_vs_global=("skill_vs_global", "mean")
                ).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return "\n".join(out)
    out: list[str] = []
    p = out.append
    runs = {(r["cfg"].segmenter, r["cfg"].encoder): r for r in results.values()}
    segs = [s for s in ("hdbscan", "kmeans") if any(k[0] == s for k in runs)]
    encs = [e for e in ("ts2vec", "tcn", "vq-vae") if any(k[1] == e for k in runs)]
    p("Cell = trade-weighted mean net RR per trade  +- [95% bootstrap CI over test episodes].  * = the expert's own regime.")
    p("Rows: model (expert) used.  Columns: regime the test episode was assigned to (n = tradable test episodes).")
    p("Latent ids are re-learned in every fold, so each fold gets its own matrix (folds are never pooled by name).")
    for seg in segs:
        p("\n" + _RULE); p(f"With {SEG_NAMES.get(seg, seg)} as Segmenter (InfoNET)"); p(_RULE)
        for enc in encs:
            run = runs.get((seg, enc))
            if run is None:
                continue
            p(f"\n{ENC_NAMES.get(enc, enc)} by edge (expectancy per trade)")
            for f in run["folds"]:
                c = f.get("cells")
                if c is not None and len(c):
                    p(f"[fold {f['fold']}]")
                    p(_edge_matrix(c).to_string())
                else:
                    p(f"[fold {f['fold']}] no matrix: {_why_empty(f)}")
    p("\n" + _RULE); p("InfoNET vs cosine similarity vs TDMI"); p(_RULE)
    p("Is a test episode more similar to its own latent's reference than to other latents'?  gap = own - other (+- 95% CI);")
    p("top1 = share of episodes whose most-similar reference is their own latent (chance = 1/#latents).")
    p(f"{'segmenter':<11}{'encoder':<8}{'metric':<9}{'ep':>4}{'own':>8}{'other':>8}{'gap':>9}{'gap 95% CI':>18}{'top1':>7}{'chance':>8}")
    any_sim = False
    for seg in segs:
        for enc in encs:
            run = runs.get((seg, enc))
            for r in (run_similarity(run) if run else []):
                any_sim = True
                ci = f"[{r['lo']:+.2f}, {r['hi']:+.2f}]" if np.isfinite(r["lo"]) else "[n<3]"
                p(f"{SEG_NAMES.get(seg, seg):<11}{ENC_NAMES.get(enc, enc):<8}{r['metric']:<9}{r['episodes']:>4}{r['own']:>8.2f}"
                  f"{r['other']:>8.2f}{r['gap']:>+9.2f}{ci:>18}{r['top1']:>7.2f}{r['chance']:>8.2f}")
    if not any_sim:
        p("no similarity rows (needs >= 2 recurring latents with tradable test episodes, and RunConfig.metrics non-empty)")
    p("\n" + _RULE); p("Edge deviation (SD of episode-level net RR per trade; lower = more consistent)"); p(_RULE)
    p("own   = model k on episodes of its own regime k;  cross = model j on episodes of the other regimes.")
    p(f"{'run':<22}{'own exp':>8}{'own SD':>8}{'(clean)':>8}{'cross exp':>10}{'cross SD':>9}{'(clean)':>8}{'own ep':>7}  status")
    devs = {}
    for seg in segs:
        for enc in encs:
            run = runs.get((seg, enc))
            if run is None:
                continue
            d = devs[(seg, enc)] = run_deviation(run)
            label = f"{ENC_NAMES.get(enc, enc)}+{SEG_NAMES.get(seg, seg)}"
            f2 = lambda v: f"{v:.2f}" if np.isfinite(v) else "  -"
            p(f"{label:<22}{f2(d['own_exp']):>8}{f2(d['own_sd']):>8}{f2(d['own_clean']):>8}{f2(d['cross_exp']):>10}"
              f"{f2(d['cross_sd']):>9}{f2(d['cross_clean']):>8}{d['own_eps']:>7}  {'ok' if d['eligible'] else 'NOT RANKABLE: ' + d['why']}")
    p("\n" + _RULE); p("Summary"); p(_RULE)
    ok = {k: d for k, d in devs.items() if d["eligible"]}
    if not ok:
        p("Winner: none.  No run has >= 2 regimes with experts and >= 10 own-regime episodes, so deviation cannot be compared.")
    else:
        (seg, enc), d = min(ok.items(), key=lambda kv: kv[1]["own_sd"])
        p(f"Winner: {ENC_NAMES.get(enc, enc).upper()} + {SEG_NAMES.get(seg, seg).upper()}")
        p(f"* Deviation: own-regime {d['own_sd']:.2f} RR (noise-corrected {d['own_clean']:.2f}); cross-regime "
          f"{d['cross_sd']:.2f} RR.  It returned the lowest own-regime deviation among rankable runs.")
        p(f"* Own-regime expectancy {d['own_exp']:+.3f} RR/trade over {d['own_eps']} episodes"
          + ("  <-- NEGATIVE: consistent, but consistently losing." if d["own_exp"] <= 0 else "."))
        if np.isfinite(d["cross_sd"]) and d["cross_sd"] <= d["own_sd"]:
            p("* Cross-regime deviation is NOT larger than own-regime deviation: the regime label does not isolate this expert's edge.")
        skipped = [f"{ENC_NAMES.get(e, e)}+{SEG_NAMES.get(s, s)}" for (s, e), x in devs.items() if not x["eligible"]]
        if skipped:
            p("* Not rankable: " + ", ".join(skipped))
    return "\n".join(out)
