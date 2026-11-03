"""
persistence.py - decide WHICH spread to trade, WHEN to trade it, and how long the z-score memory should be,
from what the spread did in the past, without ever tuning on the data it is judged on.

Why this exists (see the diagnosis in the chat):
    * Beam search picks the best of thousands of combinations on in-sample ADF, so the in-sample statistic is
      winner's-curse inflated (train 93% p<0.05, OOS 20%; half-life 102 -> 373 bars).
    * ADF is a unit-root test with low power for slow reverters: at a 373-bar half-life and 5,000 bars it rejects
      only ~27% of the time even though the spread IS mean reverting.  "Failed ADF OOS" mostly means "slower than
      train", not "not mean reverting".
    * What pays is reversion at the TRADE horizon relative to the entry distance, which ADF does not measure.

So this module measures reversion with several tools (AR(1) half-life, variance-ratio profile, Hurst, split-half
stability, an in-sample structural payoff), builds a dataset of (candidate at origin t) -> (what happened after),
and learns three things with deliberately small models and purged, time-blocked validation:

    1. which train-side statistics actually predict forward payoff              (metric_report)
    2. the forward half-life from train-side statistics -> the z-score window   (HalfLifePredictor)
    3. P(persist) / expected R of each candidate -> pick, gate and size         (PersistenceGate)

Everything is computed by the SAME functions in research and in live trading (candidate_features), so there is no
train/serve skew.  The structural payoff (structural_trades) has no learned parameters: TP at the mean (+tp_z),
SL at sl_z, so R varies per trade, which is the "stochastic rr" you asked for, without a hindsight-fitted barrier.

Public entry points
    build_dataset(panel, y, discover_cfg, dcfg)       rolling-origin dataset (expensive: one beam search per origin)
    run_study(df, dcfg)                               every table + a fitted PersistenceBundle
    PersistenceRetrainRule(bundle)                    drop-in `retrain_rule` for strategy.screen_fold / mc_stress_test
    adf_power_curve()                                 ADF vs variance-ratio power on simulated OU spreads
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from spread_discoverer import adf_p, adf_stat, create_pair
except ImportError:  # package layout of your repo
    from strategy_system.strategy.statarb.alternative.spread_discoverer import adf_p, adf_stat, create_pair

HL_CAP = 10_000.0          # half-life used when phi >= 1 (a unit root has no half-life; log scale keeps it finite)
_LN2 = float(np.log(2.0))


# =============================================================================== configuration
@dataclass
class DatasetConfig:
    train_bars: int = 5_000
    fwd_bars: tuple = (500, 1_000, 2_500, 5_000)      # forward horizons evaluated for every (origin, candidate)
    purge: int = 1
    step: int = 1_000                                  # bars between origins (overlap is fine; CV purges it)
    top_k: int = 20                                    # candidates per origin taken from the beam ranking
    z_windows: tuple = (60, 120, 240, 480, 720, 1_440, 2_880)
    feat_struct_windows: tuple = (240, 1_440)          # in-sample structural payoff used as a FEATURE
    entry_z: float = 2.0
    tp_z: float = 0.5                                  # TP level = rolling mean + tp_z*sigma (toward the mean)
    sl_z: float = 3.5                                  # SL level = rolling mean + sl_z*sigma
    hold_mult: float = 2.0                             # time barrier = clip(hold_mult * train half-life)
    hold_min: int = 60
    hold_max: int = 2_880
    vr_lags: tuple = (5, 15, 30, 60, 120)
    max_origins: int | None = None                     # random subsample of origins (None = all)
    seed: int = 7
    n_jobs: int = 1

    @property
    def span(self) -> int:
        return self.train_bars + self.purge + max(self.fwd_bars)


# =============================================================================== spread statistics
def _phi(x: np.ndarray) -> float:
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    return float((a * b).sum() / ((a * a).sum() + 1e-18))


def hl_from_phi(phi: float, cap: float = HL_CAP, floor: float = 1.0) -> float:
    if not np.isfinite(phi) or phi >= 1.0:
        return cap
    if phi <= 0.0:
        return floor
    return float(min(-_LN2 / np.log(phi), cap))


def variance_ratio(x: np.ndarray, q: int) -> float:
    """VR(q) = Var(x_t - x_{t-q}) / (q Var(x_t - x_{t-1})).  <1 mean reversion, =1 random walk, >1 trending."""
    d1 = np.diff(x)
    if len(d1) < 4 * q or d1.var() <= 0:
        return float("nan")
    return float((x[q:] - x[:-q]).var() / (q * d1.var()))


def vr_robust_z(x: np.ndarray, q: int) -> float:
    """Lo-MacKinlay heteroskedasticity-robust z of VR(q) - 1 (very negative = significant reversion)."""
    d = np.diff(x)
    n = len(d)
    if n < 4 * q:
        return float("nan")
    d = d - d.mean()
    d2 = d * d
    denom = float(d2.sum() ** 2)
    if denom <= 0:
        return float("nan")
    theta = 0.0
    for j in range(1, q):
        theta += (2.0 * (q - j) / q) ** 2 * n * float(np.sum(d2[j:] * d2[:-j])) / denom
    return float(np.sqrt(n) * (variance_ratio(x, q) - 1.0) / np.sqrt(theta + 1e-18))


def hurst_agg(x: np.ndarray, max_lag: int) -> float:
    """Aggregated-variance Hurst of the LEVEL series: slope of log std(x_{t+l}-x_t) on log l.  0.5 = random walk,
    <0.5 reverting over [2, max_lag].  Anchoring max_lag to the trade horizon avoids the long-lag flattening of OU."""
    top = int(max(8, min(max_lag, len(x) // 10)))
    lags = np.unique(np.round(np.geomspace(2, top, 8)).astype(int))
    sds = np.array([np.std(x[l:] - x[:-l]) for l in lags])
    if (sds <= 0).any():
        return float("nan")
    return float(np.polyfit(np.log(lags), np.log(sds), 1)[0])


def spread_stats(x: np.ndarray, hold: int, qs: Sequence[int], adf: bool = True) -> dict:
    x = np.asarray(x, float)
    phi = _phi(x)
    hl = hl_from_phi(phi)
    sd = float(x.std()) + 1e-18
    out = {"phi": phi, "loghl": float(np.log(hl)), "logsd": float(np.log(sd)),
           "noise": float(np.diff(x).std() / sd)}
    if adf:
        try:
            stat, _ = adf_stat(x, 6, True)
            out["adf_stat"] = float(stat)
            out["adf_logp"] = float(np.log10(max(adf_p(stat, 1), 1e-12)))
        except Exception:
            out["adf_stat"] = out["adf_logp"] = float("nan")
    for q in qs:
        out[f"vr{q}"] = variance_ratio(x, q)
    out["vrz_hold"] = vr_robust_z(x, int(np.clip(hold, 2, 240)))
    out["hurst"] = hurst_agg(x, hold)
    return out


# =============================================================================== structural payoff (no learned parameters)
def rolling_z(x: np.ndarray, window: int):
    s = pd.Series(x)
    mp = max(30, window // 4)
    mu = s.rolling(window, min_periods=mp).mean().to_numpy()
    sd = s.rolling(window, min_periods=mp).std(ddof=0).to_numpy()
    with np.errstate(all="ignore"):
        z = (x - mu) / np.where(sd > 1e-12, sd, np.nan)
    return z, mu, sd


def structural_trades(x, z, mu, sd, start: int, end: int, ez=2.0, tp_z=0.5, sl_z=3.5, T=576):
    """Mean-reversion rule with fixed, model-free barriers (close-based, one position at a time, armed re-entry).

    Entry  first bar with |z| >= ez after |z| < ez.   Direction: fade the deviation.
    TP     rolling mean + tp_z*sigma (levels frozen at entry).   SL  rolling mean + sl_z*sigma.   Time barrier T bars.
    R      pnl / (distance from entry to the stop) -> TP pays (|z|-tp_z)/(sl_z-|z|): >1 on deeper entries.
    Stop is assumed first when both levels are touched on the same bar.  Returns [(entry_idx, R, reason)].
    """
    out = []
    armed, i = True, int(start)
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
        sgn = 1.0 if zi > 0 else -1.0                      # +1: spread rich -> we short it
        m, s0 = mu[i], sd[i]
        tp, sl = m + sgn * tp_z * s0, m + sgn * sl_z * s0
        risk = (sl_z - abs(zi)) * s0
        seg = x[i + 1: min(i + 1 + T, end)]
        htp = seg <= tp if sgn > 0 else seg >= tp
        hsl = seg >= sl if sgn > 0 else seg <= sl
        big = 10 ** 9
        ktp = int(htp.argmax()) if htp.any() else big
        ksl = int(hsl.argmax()) if hsl.any() else big
        if ktp == big and ksl == big:
            j, px_exit, reason = len(seg) - 1, seg[-1], ("time" if len(seg) == T else "end")
        elif ksl <= ktp:
            j, px_exit, reason = ksl, sl, "sl"
        else:
            j, px_exit, reason = ktp, tp, "tp"
        exit_i = i + 1 + j
        out.append((i, float(-sgn * (px_exit - x[i]) / risk), reason))
        armed = bool(np.isfinite(z[exit_i]) and abs(z[exit_i]) < ez)
        i = exit_i + 1
    return out


# =============================================================================== features / labels (shared research <-> live)
def spread_from_betas(logp: pd.DataFrame, y: str, betas: dict, a: int, b: int) -> np.ndarray:
    x = logp[y].to_numpy(float)[a:b].copy()
    for sym, beta in betas.items():
        x -= beta * logp[sym].to_numpy(float)[a:b]
    return x


def _px_betas(c) -> dict:
    return {t.split("|", 1)[1]: float(v) for t, v in c.betas.items() if t.startswith("px|")}


def hold_from_hl(hl: float, d: DatasetConfig) -> int:
    return int(np.clip(round(d.hold_mult * hl), d.hold_min, d.hold_max))


def candidate_features(c, xt: np.ndarray, d: DatasetConfig, rank: int = 0) -> dict:
    """Everything known at the END of the train window about candidate `c` (xt = its asset-only train spread)."""
    n = len(xt)
    h = n // 2
    hl_t = hl_from_phi(_phi(xt))
    hold = hold_from_hl(hl_t, d)
    s = spread_stats(xt, hold, d.vr_lags)
    a = spread_stats(xt[:h], hold, (), adf=False)
    b = spread_stats(xt[h:], hold, (), adf=False)
    f = {f"f_{k}": v for k, v in s.items()}
    sd = float(xt.std()) + 1e-18
    f.update(
        f_hold=hold,
        f_loghl_a=a["loghl"], f_loghl_b=b["loghl"],
        f_hl_split=abs(a["loghl"] - b["loghl"]),
        f_phi_split=abs(a["phi"] - b["phi"]),
        f_mean_shift=abs(float(xt[h:].mean() - xt[:h].mean())) / sd,
        f_sd_ratio=float(np.log((xt[h:].std() + 1e-18) / (xt[:h].std() + 1e-18))),
        f_trend=abs(float(np.polyfit(np.arange(n), xt, 1)[0])) * n / sd,
        f_rank=rank,
        f_log_p_eg=float(np.log10(max(c.p_eg, 1e-12))),
        f_score=float(c.score),
        f_n_px=sum(t.startswith("px|") for t in c.terms),
        f_n_terms=len(c.terms),
        f_gross_beta=float(sum(abs(v) for v in _px_betas(c).values())),
        f_max_share=float(c.max_share),
        f_drift=float(c.drift) if np.isfinite(c.drift) else 0.0,
    )
    for W in d.feat_struct_windows:                       # in-sample structural payoff (optimistic, but a usable feature)
        z, mu, sg = rolling_z(xt, W)
        tr = structural_trades(xt, z, mu, sg, W, n, d.entry_z, d.tp_z, d.sl_z, hold)
        f[f"f_sR_{W}"] = float(np.mean([t[1] for t in tr])) if tr else float("nan")
        f[f"f_sN_{W}"] = len(tr)
    return f


def forward_labels(xt: np.ndarray, xf_full: np.ndarray, f: dict, d: DatasetConfig) -> list:
    """One dict per forward horizon.  Betas are the TRAIN betas (frozen), z uses the train tail as warm start."""
    hold = int(f["f_hold"])
    seed = xt[-max(d.z_windows):]
    arr = np.r_[seed, xf_full]
    zs = {W: rolling_z(arr, W) for W in d.z_windows}       # causal: a prefix of arr gives the same values
    rows = []
    for h in d.fwd_bars:
        xf = xf_full[:h]
        s = spread_stats(xf, hold, d.vr_lags, adf=h >= 1_000)
        y = {f"y_{k}": v for k, v in s.items()}
        y["y_logratio"] = float(np.clip(s["loghl"] - f["f_loghl"], -2.0, 4.0))
        y["y_persist"] = float(y["y_logratio"] < _LN2)    # forward half-life within 2x of train
        for W, (z, mu, sg) in zs.items():
            tr = structural_trades(arr, z, mu, sg, len(seed), len(seed) + h, d.entry_z, d.tp_z, d.sl_z, hold)
            R = [t[1] for t in tr]
            y[f"y_n_{W}"] = len(R)
            y[f"y_R_{W}"] = float(np.mean(R)) if R else float("nan")
        y["horizon"] = h
        rows.append(y)
    return rows


# =============================================================================== dataset
def _origin_rows(panel, y: str, discover_cfg, d: DatasetConfig, o: int) -> list:
    tr = panel.iloc(o, o + d.train_bars)
    ranked = create_pair(tr, y, cfg=discover_cfg)
    cands = [c for c in ranked if any(t.startswith("px|") for t in c.terms)][: d.top_k]
    a = o + d.train_bars + d.purge
    b = a + max(d.fwd_bars)
    rows = []
    for rank, c in enumerate(cands):
        betas = _px_betas(c)
        x_all = spread_from_betas(panel.logp, y, betas, o, b)
        xt, xf = x_all[: d.train_bars], x_all[d.train_bars + d.purge:]
        f = candidate_features(c, xt, d, rank)
        meta = {"origin": o, "origin_time": panel.logp.index[o], "hedge": " + ".join(betas)}
        for lab in forward_labels(xt, xf, f, d):
            rows.append({**meta, **f, **lab})
    return rows


def build_dataset(panel, y: str, discover_cfg, d: DatasetConfig | None = None, segment: int = 0) -> pd.DataFrame:
    """Rolling-origin dataset.  One row per (origin, candidate, horizon).  Expensive part = one beam search per origin."""
    d = d or DatasetConfig()
    n = len(panel.logp)
    origins = np.arange(0, n - d.span + 1, d.step)
    if d.max_origins and len(origins) > d.max_origins:
        origins = np.sort(np.random.default_rng(d.seed).choice(origins, d.max_origins, replace=False))
    try:
        from tqdm import tqdm
        it = tqdm(origins, desc=f"origins seg{segment}")
    except ImportError:
        it = origins
    if d.n_jobs != 1:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=d.n_jobs)(delayed(_origin_rows)(panel, y, discover_cfg, d, int(o)) for o in it)
    else:
        chunks = [_origin_rows(panel, y, discover_cfg, d, int(o)) for o in it]
    df = pd.DataFrame([r for ch in chunks for r in ch])
    df["segment"] = segment
    return df


# =============================================================================== validation helpers
def _minutes(ts: pd.Series) -> np.ndarray:
    """Timestamps -> integer minutes (robust to pandas datetime resolution and tz-awareness)."""
    s = pd.to_datetime(ts)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy().astype("datetime64[m]").astype("int64").astype(float)


def purged_time_splits(df: pd.DataFrame, d: DatasetConfig, bar_minutes: float = 1.0, n_splits: int = 5):
    """Contiguous time blocks as test sets; train rows whose [origin, origin+span) window touches a test window are dropped."""
    t = _minutes(df["origin_time"])
    span = d.span * bar_minutes
    edges = np.quantile(np.unique(t), np.linspace(0, 1, n_splits + 1))
    out = []
    for k in range(n_splits):
        lo, hi = edges[k], edges[k + 1] + (1e-9 if k == n_splits - 1 else 0)
        te = (t >= lo) & (t < hi)
        tr = (t + span <= lo) | (t >= hi + span)
        if te.sum() and tr.sum():
            out.append((np.flatnonzero(tr), np.flatnonzero(te)))
    return out


def _blocks(df: pd.DataFrame, n_blocks: int = 20) -> np.ndarray:
    t = pd.to_datetime(df["origin_time"]).astype("int64").to_numpy()
    return np.minimum((pd.Series(t).rank(pct=True).to_numpy() * n_blocks).astype(int), n_blocks - 1)


def block_bootstrap_mean(v: np.ndarray, w: np.ndarray, blocks: np.ndarray, n_boot=400, alpha=0.10, seed=7):
    ok = np.isfinite(v) & (w > 0)
    v, w, blocks = v[ok], w[ok], blocks[ok]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    ids = np.unique(blocks)
    num = np.array([np.sum(v[blocks == b] * w[blocks == b]) for b in ids])
    den = np.array([np.sum(w[blocks == b]) for b in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (n_boot, len(ids)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / den.sum()), float(np.quantile(bs, alpha)), float(np.quantile(bs, 1 - alpha))


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 20 or a[ok].nunique() < 3:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def metric_report(df: pd.DataFrame, label: str, feature_cols: Sequence[str] | None = None, n_blocks=20,
                  n_boot=300, seed=7) -> pd.DataFrame:
    """Spearman IC of every column vs `label`, with a time-block bootstrap CI.  Use label=f"y_R_{W}" (forward payoff).
    feature_cols = f_* answers "what predicts payoff ex ante"; y_* answers "what co-moves with payoff ex post"."""
    cols = feature_cols or [c for c in df.columns if c.startswith("f_")]
    blocks = _blocks(df, n_blocks)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, n_blocks, n_blocks) for _ in range(n_boot)]
    rows = []
    for c in cols:
        base = _rank_corr(df[c], df[label])
        if not np.isfinite(base):
            continue
        bs = []
        for dr in draws:
            sel = np.concatenate([np.flatnonzero(blocks == b) for b in dr])
            bs.append(_rank_corr(df[c].iloc[sel], df[label].iloc[sel]))
        rows.append({"metric": c, "ic": base, "lo5": np.nanquantile(bs, 0.05), "hi95": np.nanquantile(bs, 0.95),
                     "n": int(df[[c, label]].dropna().shape[0])})
    r = pd.DataFrame(rows)
    return r.reindex(r["ic"].abs().sort_values(ascending=False).index).reset_index(drop=True) if len(r) else r


# =============================================================================== models
def _pipeline(kind: str, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=kw.get("alpha", 30.0)))
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=kw.get("C", 0.05), max_iter=500))
    if kind == "gbr":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=80,
                                             l2_regularization=5.0, random_state=7)
    if kind == "gbc":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=80,
                                              l2_regularization=5.0, random_state=7)
    raise ValueError(kind)


DEFAULT_FEATURES = ["f_loghl", "f_loghl_a", "f_loghl_b", "f_hl_split", "f_phi_split", "f_mean_shift", "f_sd_ratio",
                    "f_trend", "f_noise", "f_adf_logp", "f_vr15", "f_vr60", "f_vr120", "f_vrz_hold", "f_hurst",
                    "f_log_p_eg", "f_n_px", "f_gross_beta", "f_drift", "f_rank", "f_sR_240", "f_sR_1440"]


class HalfLifePredictor:
    """Predicts log(forward half-life / train half-life) from train-side statistics (ridge: ~20 numbers, can't overfit much).
    Judge it against two baselines in cv_report: 'train' (ratio 1) and 'shrunk' (the median ratio, one constant)."""

    def __init__(self, horizon: int = 2_500, features: Sequence[str] | None = None, alpha: float = 30.0):
        self.horizon, self.features, self.alpha = horizon, list(features or DEFAULT_FEATURES), alpha
        self.model = None
        self.median_ratio = 0.0

    def _data(self, df):
        s = df[df["horizon"] == self.horizon]
        s = s[s["y_logratio"].notna()]
        return s

    def fit(self, df):
        s = self._data(df)
        self.features = [c for c in self.features if c in s.columns]
        self.median_ratio = float(s["y_logratio"].median())
        self.model = _pipeline("ridge", alpha=self.alpha).fit(s[self.features], s["y_logratio"])
        return self

    def predict_logratio(self, F: pd.DataFrame) -> np.ndarray:
        return self.model.predict(F[self.features])

    def predict_hl(self, F: pd.DataFrame) -> np.ndarray:
        return np.exp(F["f_loghl"].to_numpy() + self.predict_logratio(F))

    def oof(self, df, d: DatasetConfig, bar_minutes=1.0, n_splits=5) -> pd.Series:
        s = self._data(df)
        pred = pd.Series(np.nan, index=s.index)
        for tr, te in purged_time_splits(s, d, bar_minutes, n_splits):
            m = _pipeline("ridge", alpha=self.alpha).fit(s.iloc[tr][self.features], s.iloc[tr]["y_logratio"])
            pred.iloc[te] = m.predict(s.iloc[te][self.features])
        return pred

    def cv_report(self, df, d: DatasetConfig, bar_minutes=1.0) -> pd.DataFrame:
        s = self._data(df)
        o = self.oof(df, d, bar_minutes).reindex(s.index)
        ok = o.notna()
        y = s.loc[ok, "y_logratio"]
        const = pd.Series(0.0, index=y.index)
        shr = pd.Series(float(s["y_logratio"].median()), index=y.index)
        rows = []
        for name, p in (("train half-life", const), ("shrunk (constant ratio)", shr), ("ridge on stats", o[ok])):
            rows.append({"predictor": name, "mae_log": float((y - p).abs().mean()),
                         "rmse_log": float(np.sqrt(((y - p) ** 2).mean())),
                         "rank_corr": _rank_corr(p, y) if p.nunique() > 1 else float("nan")})
        return pd.DataFrame(rows)


def snap_window(W: float, grid: Sequence[int]) -> int:
    g = np.asarray(grid)
    return int(g[np.abs(np.log(g) - np.log(max(W, 1.0))).argmin()])


def evaluate_window_policies(df: pd.DataFrame, d: DatasetConfig, hl_oof: pd.Series | None = None, horizon=2_500,
                             ks=(2.0, 3.0, 4.0)) -> pd.DataFrame:
    """Realised structural payoff when the z window is chosen by: a fixed value | k x train half-life | k x predicted
    half-life (out-of-fold) | k x the TRUE forward half-life (oracle = upper bound on what a perfect predictor could add)."""
    s = df[df["horizon"] == horizon].copy()
    blocks = _blocks(s)
    pols = {f"fixed {W}": (lambda r, W=W: W) for W in d.z_windows}
    for k in ks:
        pols[f"{k:g} x train hl"] = lambda r, k=k: k * np.exp(r["f_loghl"])
        if hl_oof is not None:
            pols[f"{k:g} x predicted hl"] = lambda r, k=k: k * np.exp(r["f_loghl"] + r["_pred"])
        pols[f"{k:g} x true hl (oracle)"] = lambda r, k=k: k * np.exp(r["y_loghl"])
    if hl_oof is not None:
        s["_pred"] = hl_oof.reindex(s.index)
    rows = []
    for name, fn in pols.items():
        vals, ws = [], []
        for _, r in s.iterrows():
            if "predicted" in name and not np.isfinite(r["_pred"]):
                vals.append(np.nan); ws.append(0); continue
            W = snap_window(fn(r), d.z_windows)
            vals.append(r[f"y_R_{W}"]); ws.append(r[f"y_n_{W}"])
        m, lo, hi = block_bootstrap_mean(np.array(vals, float), np.array(ws, float), blocks)
        rows.append({"policy": name, "mean_R": m, "lo10": lo, "hi90": hi, "trades": int(np.nansum(ws))})
    return pd.DataFrame(rows).sort_values("mean_R", ascending=False).reset_index(drop=True)


class PersistenceGate:
    """Scores a candidate by expected structural payoff ('edge') or P(persist) ('persist'), calibrated out-of-fold with
    isotonic regression.  precision_curve() is your 'increasing confidence thresholds' table: for each cumulative cut
    (score >= q-th quantile) it reports coverage, P(persist) and weighted mean R with a time-block bootstrap CI.
    The gate threshold is the lowest cut whose LOWER CI bound of mean R is > 0; if no cut qualifies the gate stays
    CLOSED (threshold = inf) - the honest answer is 'no gate adds edge'."""

    def __init__(self, target: str = "edge", model: str = "ridge", horizon: int = 2_500, W: int = 720,
                 features: Sequence[str] | None = None, min_trades: int = 2, ci_alpha: float = 0.10,
                 min_coverage: float = 0.05):
        self.target, self.model_kind, self.horizon, self.W = target, model, horizon, W
        self.features = list(features or DEFAULT_FEATURES)
        self.min_trades, self.ci_alpha, self.min_coverage = min_trades, ci_alpha, min_coverage
        self.model = self.iso = None
        self.threshold = float("inf")
        self.s_hi = float("nan")
        self.curve: pd.DataFrame | None = None

    def _rows(self, df):
        s = df[df["horizon"] == self.horizon]
        if self.target == "edge":
            s = s[(s[f"y_n_{self.W}"] >= self.min_trades) & s[f"y_R_{self.W}"].notna()]
        else:
            s = s[s["y_persist"].notna()]
        return s

    def _y(self, s):
        return s[f"y_R_{self.W}"] if self.target == "edge" else s["y_persist"]

    def _fit_model(self, s):
        kind = {"edge": {"ridge": "ridge", "gbm": "gbr"}, "persist": {"ridge": "logit", "gbm": "gbc"}}[self.target][self.model_kind]
        m = _pipeline(kind)
        if kind == "ridge":                              # weight by number of trades behind each realised mean R (capped)
            m.fit(s[self.features], self._y(s), ridge__sample_weight=s[f"y_n_{self.W}"].clip(upper=10).to_numpy())
        else:
            m.fit(s[self.features], self._y(s))
        return m

    def _raw(self, m, X):
        return m.predict(X) if self.target == "edge" else m.predict_proba(X)[:, 1]

    def fit(self, df: pd.DataFrame, d: DatasetConfig, bar_minutes=1.0, n_splits=5):
        from sklearn.isotonic import IsotonicRegression
        s = self._rows(df)
        self.features = [c for c in self.features if c in s.columns]
        oof = pd.Series(np.nan, index=s.index)
        for tr, te in purged_time_splits(s, d, bar_minutes, n_splits):
            m = self._fit_model(s.iloc[tr])
            oof.iloc[te] = self._raw(m, s.iloc[te][self.features])
        ok = oof.notna()
        self.model = self._fit_model(s)
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(oof[ok], self._y(s)[ok])
        self.oof_score = oof
        self._oof_rows = s
        self.curve = self.precision_curve(s, oof)
        good = self.curve[(self.curve["lo_R"] > 0) & (self.curve["coverage"] >= self.min_coverage)]
        self.threshold = float(good["score_cut"].min()) if len(good) else float("inf")
        self.s_hi = float(np.nanquantile(oof[ok], 0.90))
        return self

    def precision_curve(self, s: pd.DataFrame, score: pd.Series, qs=(0, .25, .5, .7, .8, .9, .95)) -> pd.DataFrame:
        blocks = _blocks(s)
        ok = score.notna().to_numpy()
        R = s[f"y_R_{self.W}"].to_numpy(float)
        n = s[f"y_n_{self.W}"].to_numpy(float)
        rows = []
        for q in qs:
            cut = float(np.nanquantile(score[ok], q))
            sel = ok & (score.to_numpy() >= cut)
            m, lo, hi = block_bootstrap_mean(R[sel], np.maximum(n[sel], 1.0), blocks[sel], alpha=self.ci_alpha)
            rows.append({"quantile": q, "score_cut": cut, "coverage": float(sel.mean()),
                         "P_persist": float(np.nanmean(s["y_persist"].to_numpy(float)[sel])),
                         "mean_R": m, "lo_R": lo, "hi_R": hi, "trades": int(np.nansum(n[sel]))})
        return pd.DataFrame(rows)

    def score(self, F: pd.DataFrame) -> np.ndarray:
        """Calibrated expected value (R for target='edge', probability for 'persist')."""
        return self.iso.predict(self._raw(self.model, F[self.features]))

    def raw_threshold_score(self, F):                    # the gate compares RAW score to the OOF-derived cut
        return self._raw(self.model, F[self.features])

    def passes(self, raw: np.ndarray) -> np.ndarray:
        return raw >= self.threshold

    def size_mult(self, raw: float, floor: float = 0.3) -> float:
        if not np.isfinite(self.threshold) or raw < self.threshold:
            return 0.0
        span = max(self.s_hi - self.threshold, 1e-9)
        return float(floor + (1.0 - floor) * np.clip((raw - self.threshold) / span, 0.0, 1.0))


# =============================================================================== bundle + live rule
@dataclass
class PersistenceBundle:
    dcfg: DatasetConfig
    hl_model: HalfLifePredictor
    gate: PersistenceGate
    k_window: float = 3.0
    w_lo: int = 60
    w_hi: int = 2_880

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "PersistenceBundle":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class PersistenceRetrainRule:
    """retrain_rule(train_panel, y, discover_cfg, previous_selection) -> ScreenSelection

    1 beam search -> top_k candidates -> candidate_features (same function as research) -> score every candidate ->
    trade the best one if it passes the gate (else size_mult = 0 = sit out the whole forward window) with
    z window = clip(k * predicted half-life).  Needs the two small hooks in strategy.py (ranked=, z_window, size_mult)."""

    def __init__(self, bundle: PersistenceBundle, y_hint: str | None = None):
        self.b = bundle

    def __call__(self, train_panel, y, discover_cfg, previous_selection=None):
        try:
            from strategy import Screener
        except ImportError:
            from strategy_system.strategy.statarb.alternative.strategy import Screener
        b, d = self.b, self.b.dcfg
        ranked = create_pair(train_panel, y, cfg=discover_cfg)
        execu = [c for c in ranked if any(t.startswith("px|") for t in c.terms)][: d.top_k]
        if not execu:
            raise RuntimeError("beam search returned no candidate with a tradable price leg")
        n = len(train_panel.logp)
        rows = []
        for rank, c in enumerate(execu):
            xt = spread_from_betas(train_panel.logp, y, _px_betas(c), 0, n)
            rows.append(candidate_features(c, xt, d, rank))
        F = pd.DataFrame(rows)
        raw = b.gate.raw_threshold_score(F)
        i = int(np.argmax(raw))
        hl_pred = float(b.hl_model.predict_hl(F.iloc[[i]])[0])
        order = [execu[i]] + [c for j, c in enumerate(execu) if j != i]
        sel = Screener(train_panel, y, bars=n, cfg=discover_cfg, ranked=order)
        W = int(np.clip(round(b.k_window * hl_pred), b.w_lo, b.w_hi))
        return replace(sel, z_window=W, size_mult=b.gate.size_mult(float(raw[i])), gate_p=float(raw[i]), hl_pred=hl_pred)


# =============================================================================== why ADF fails: power on simulated OU spreads
def _simulate_ou(n: int, hl: float, rng) -> np.ndarray:
    phi = np.exp(-_LN2 / hl)
    e = rng.normal(size=n) * np.sqrt(1 - phi ** 2)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def adf_power_curve(hls=(50, 100, 200, 373, 700, 1500), n_bars=5_000, n_sim=150, hold=60, seed=0) -> pd.DataFrame:
    """Share of simulated TRULY stationary OU spreads each test flags, by half-life.  (ADF: p<0.05;  VR: robust z < -1.645.)
    Also reports the share of the deviation that has reverted after `hold` bars: that, not the root, is what pays."""
    rng = np.random.default_rng(seed)
    rows = []
    for hl in hls:
        adf_ok = vr_ok = 0
        for _ in range(n_sim):
            x = _simulate_ou(n_bars, hl, rng)
            adf_ok += adf_p(adf_stat(x, 6, True)[0], 1) < 0.05
            vr_ok += vr_robust_z(x, hold) < -1.645
        rows.append({"half_life": hl, "ADF_flags_stationary": adf_ok / n_sim, f"VR({hold})_flags_reversion": vr_ok / n_sim,
                     f"reverted_in_{hold}_bars": 1 - 0.5 ** (hold / hl)})
    return pd.DataFrame(rows)


# =============================================================================== one call: every table + fitted bundle
def run_study(df: pd.DataFrame, d: DatasetConfig, bar_minutes: float = 1.0, horizon: int = 2_500, W: int = 720,
              target: str = "edge", model: str = "ridge", k_window: float = 3.0) -> dict:
    out = {}
    out["adf_power"] = adf_power_curve()
    s = df[df["horizon"] == horizon]
    label = f"y_R_{W}"
    out["ex_post_forward_metric_vs_payoff"] = metric_report(
        s, label, [c for c in s.columns if c.startswith("y_") and not c.startswith(("y_R_", "y_n_", "y_persist"))])
    out["ex_ante_train_metric_vs_payoff"] = metric_report(s, label)
    out["ex_ante_train_metric_vs_persistence"] = metric_report(s, "y_logratio")
    hl = HalfLifePredictor(horizon).fit(df)
    out["halflife_cv"] = hl.cv_report(df, d, bar_minutes)
    out["window_policies"] = evaluate_window_policies(df, d, hl.oof(df, d, bar_minutes), horizon)
    gate = PersistenceGate(target, model, horizon, W).fit(df, d, bar_minutes)
    out["gate_precision_curve"] = gate.curve
    out["bundle"] = PersistenceBundle(d, hl, gate, k_window)
    return out


__all__ = ["DatasetConfig", "build_dataset", "run_study", "PersistenceRetrainRule", "PersistenceBundle", "HalfLifePredictor",
           "PersistenceGate", "metric_report", "adf_power_curve", "evaluate_window_policies", "candidate_features",
           "structural_trades", "variance_ratio", "vr_robust_z", "hurst_agg", "spread_stats"]