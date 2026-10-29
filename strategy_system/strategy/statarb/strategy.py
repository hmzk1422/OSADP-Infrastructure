"""
Multivariate spread discoverer: asset legs AND flow/depth terms, static (or ultra-slow) betas.

    spread_t = y_t - a - sum_j beta_j * term_j,t
    term     = log price of an asset ("px|SOL/USDT")  or  a per-asset feature ("ft|BTC/USDT|cvd")

Stages
    get_assets()   -> bars -> features -> AssetPanel
    create_pair()  -> beam search on the FIRST part of the data (betas = ONE static OLS fit)
    test_pair()    -> out-of-sample on the REST with the betas FROZEN (or a Kalman with tiny q)
    analyze()      -> feature ablation, per-term dominance table, z-score bounce statistics

Objective (lower is better)
    score = p_weight * max(0, log10(p_EG / p_train_max))     # first get under the ADF threshold
          + half_life / hl_ref                                # then shrink the half-life
          * (1 + term_penalty * (n_terms - 1))                # (half-life part only) each extra term must cut it by ~term_penalty
          + dom_weight * max(0, max_share - dominance_max)    # one term carrying the whole hedge
          + beta_weight * (|beta_px| above beta_abs_max)      # huge price betas

Statistics, and why they are done this way
    * In-sample p-value = Engle-Granger: ADF on OLS residuals using MacKinnon coint critical values with
      N = 1 + (number of I(1) regressors). A plain ADF p-value on fitted residuals is far too optimistic,
      because OLS picks the most stationary-looking combination.
      Stationary feature terms (tfi, di1, ...) are I(0), do not add to N; cumulative ones (cvd, ofi_cum) do.
    * Out-of-sample p-value = plain ADF (with constant) on the spread built from FROZEN train betas.
      No parameter was estimated on that sample, so the ordinary tables apply.
    * Half-life = -ln2 / ln(phi), phi = AR(1) coefficient of the spread (Ornstein-Uhlenbeck discretisation).
      phi >= 1 -> inf. phi <= 0 is NOT "instant": it is treated as noise (half_life_min_bars) and rejected.
    * Beam search over thousands of combinations selects on in-sample p-values, so the in-sample number is
      optimistic by construction. Trust test_pair().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp
from tqdm import tqdm

try:  # package layout of your repo
    from strategy_system.strategy.statarb.binance_vision import load_bars
    from strategy_system.strategy.statarb.flow_features import (
        INTEGRATED_FEATURES, FeatureConfig, features, warmup_bars)
except ImportError:  # flat folder
    from binance_vision import load_bars
    from flow_features import INTEGRATED_FEATURES, FeatureConfig, features, warmup_bars


# --------------------------------------------------------------------------- config
@dataclass
class DiscoverConfig:
    train_frac: float = 0.5
    k: int = 20                       # beam width = number of ranked constructions returned
    max_terms: int = 4                # RHS terms allowed in total (seed included)
    use_features: bool = True
    # per-asset features that may enter the spread (missing ones are skipped).
    # cvd / ofi_cum are I(1) and can cointegrate with a price level; the rest are I(0) state variables.
    feature_pool: tuple[str, ...] = ("cvd", "ofi_cum", "tfi_l", "di1", "depth_slope")

    # --- ADF / objective
    adf_maxlag: int = 12              # ADF augmentation lags chosen by AIC up to this
    p_train_max: float = 0.05         # in-sample Engle-Granger p target
    p_oos_max: float = 0.10           # out-of-sample ADF p required to pass
    p_weight: float = 100.0
    hl_ref: float = 100.0             # bars; half_life/hl_ref enters the score
    hl_cap: float = 5000.0            # bars; used when half-life is inf / invalid
    half_life_min_bars: float = 2.0   # faster than this = bid-ask-bounce noise, rejected
    half_life_max_bars: float = 288.0 # OOS must revert at least this fast to pass
    term_penalty: float = 0.35        # relative: an extra term must shorten the half-life by roughly this fraction
    min_improve: float = 1e-3

    # --- overfitting guards
    dominance_max: float = 0.85       # max share of |beta|*std(term) among terms (only when >= 2 terms)
    dom_weight: float = 5.0
    beta_abs_max: float = 5.0         # |beta| on a log-price leg above this is suspicious
    beta_weight: float = 1.0
    term_stationary_p: float = 0.05   # a price leg that is itself stationary below this p is flagged

    # --- beta handling in the OOS test
    beta_mode: str = "static"         # "static" (frozen train OLS) or "kalman"
    kalman_q: float = 1e-10           # state-noise variance per bar, in STANDARDISED coefficient units
    kalman_p0_scale: float = 1e-2     # prior covariance = scale * (OLS covariance of the train fit). 1.0 ~ recursive
                                      # least squares (betas keep absorbing estimation error); ->0 = exactly static

    # --- z-score stats
    z_mode: str = "static"            # "static": train mean/std frozen; "rolling": causal rolling mean/std
    z_window: int = 288
    z_entry: float = 2.0
    z_exit: float = 0.0
    z_max_hold: int = 576             # bars before an open excursion is counted as a failed bounce


# --------------------------------------------------------------------------- terms
def px(symbol: str) -> str:
    return f"px|{symbol}"


def ft(symbol: str, name: str) -> str:
    return f"ft|{symbol}|{name}"


def kind(term: str) -> str:
    return term.split("|", 1)[0]


def label(term: str) -> str:
    p = term.split("|")
    return p[1] if p[0] == "px" else f"{p[1]}:{p[2]}"


# --------------------------------------------------------------------------- data container
@dataclass
class AssetPanel:
    logp: pd.DataFrame
    ret: pd.DataFrame
    vol: pd.DataFrame
    feats: dict[str, pd.DataFrame] = field(default_factory=dict)  # feature -> (time x asset)
    integrated: frozenset = INTEGRATED_FEATURES
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def symbols(self) -> list[str]:
        return list(self.logp.columns)

    @property
    def bar_minutes(self) -> float:
        return float(pd.Series(self.logp.index).diff().median().total_seconds() / 60)

    def iloc(self, a, b) -> "AssetPanel":
        return AssetPanel(self.logp.iloc[a:b], self.ret.iloc[a:b], self.vol.iloc[a:b],
                          {k: v.iloc[a:b] for k, v in self.feats.items()}, self.integrated)

    def split_index(self, frac: float) -> int:
        return int(len(self.logp) * frac)

    def has(self, term: str) -> bool:
        p = term.split("|")
        return (p[1] in self.logp.columns) if p[0] == "px" else (p[2] in self.feats and p[1] in self.feats[p[2]].columns)

    def is_i1(self, term: str) -> bool:
        p = term.split("|")
        return p[0] == "px" or p[2] in self.integrated

    def series(self, term: str) -> np.ndarray:
        if term not in self._cache:
            p = term.split("|")
            s = self.logp[p[1]] if p[0] == "px" else self.feats[p[2]][p[1]]
            self._cache[term] = s.to_numpy(float)
        return self._cache[term]

    def matrix(self, terms: Sequence[str]) -> np.ndarray:
        return np.column_stack([self.series(t) for t in terms])

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame], warmup: int = 0, require: Sequence[str] = ()) -> "AssetPanel":
        idx = None
        for f in frames.values():
            idx = f.index if idx is None else idx.intersection(f.index)
        names = sorted(set.intersection(*(set(f.columns) for f in frames.values())))
        wide = {n: pd.DataFrame({s: f[n].reindex(idx) for s, f in frames.items()}) for n in names}
        ok = wide["logp"].notna().all(axis=1) & wide["ret"].notna().all(axis=1)
        for n in require:
            if n in wide:
                ok &= wide[n].notna().all(axis=1)
        ok.iloc[:warmup] = False
        wide = {n: w[ok] for n, w in wide.items()}
        return cls(logp=wide.pop("logp"), ret=wide.pop("ret"), vol=wide.pop("vol"), feats=wide)

    @classmethod
    def from_close(cls, close: pd.DataFrame, vol_window: int = 100) -> "AssetPanel":
        frames = {}
        for s in close.columns:
            lp = np.log(close[s].astype(float))
            r = lp.diff()
            frames[s] = pd.DataFrame({"logp": lp, "ret": r, "vol": r.rolling(vol_window).std()})
        return cls.from_frames(frames)


def get_assets(symbols, start, end, bar="5min", market="um", with_depth=True, with_ticker=False,
               cache_dir="~/.cache/binance_vision", workers=4, fcfg: FeatureConfig = FeatureConfig(),
               cfg: DiscoverConfig = DiscoverConfig(), fetch=None) -> AssetPanel:
    kw = {} if fetch is None else {"fetch": fetch}
    frames = {}
    for s in tqdm(symbols, desc="assets"):
        bars = load_bars(s, start, end, bar=bar, market=market, with_depth=with_depth, with_ticker=with_ticker,
                         cache_dir=cache_dir, workers=workers, **kw)
        frames[s] = features(bars, fcfg)
    return AssetPanel.from_frames(frames, warmup=warmup_bars(fcfg), require=cfg.feature_pool if cfg.use_features else ())


# --------------------------------------------------------------------------- statistics
def ols_fit(y: np.ndarray, X: np.ndarray):
    """
    Function used to form statistical arbitrage relationships.
    CHANGE THIS TO CHANGE THE METHOD
    """
    X = X if X.ndim == 2 else X[:, None]
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    slopes = b[1:] / sd
    return float(b[0] - slopes @ mu), slopes, y - Z @ b


def adf_stat(e: np.ndarray, maxlag: int, const: bool) -> tuple[float, int]:
    e = np.asarray(e, float)
    de = np.diff(e)
    n = len(de)
    maxlag = int(max(0, min(maxlag, (n - 10) // 3)))

    def fit(L: int, s: int):
        y = de[s:]
        cols = [e[s:n]] + [de[s - i:n - i] for i in range(1, L + 1)]
        if const:
            cols.append(np.ones(n - s))
        Z = np.column_stack(cols)
        ZtZ = Z.T @ Z
        b = np.linalg.solve(ZtZ, Z.T @ y)
        r = y - Z @ b
        return b, ZtZ, float(r @ r), len(y), Z.shape[1]

    best_L, best_aic = 0, np.inf
    for L in range(maxlag + 1):
        _, _, rss, N, k = fit(L, maxlag)
        aic = N * np.log(max(rss, 1e-300) / N) + 2 * k
        if aic < best_aic:
            best_L, best_aic = L, aic
    b, ZtZ, rss, N, k = fit(best_L, best_L)
    se = np.sqrt(rss / (N - k) * np.linalg.inv(ZtZ)[0, 0])
    return float(b[0] / se), best_L


def adf_p(stat: float, n_series: int = 1) -> float:
    return float(mackinnonp(stat, regression="c", N=int(min(max(n_series, 1), 12))))


def half_life(s: np.ndarray, floor: float = 1.0) -> float:
    a, b = s[:-1] - s[:-1].mean(), s[1:] - s[1:].mean()
    phi = float((a * b).sum() / ((a * a).sum() + 1e-18))
    if phi >= 1.0:
        return float("inf")
    if phi <= 0.0:
        return floor
    return float(-np.log(2) / np.log(phi))


def bounce_stats(z: np.ndarray, entry: float = 2.0, exit_: float = 0.0, max_hold: int = 576) -> dict:
    pos, t0, peak, armed = 0, 0, 0.0, True
    dur, peaks = [], []
    n_entry = n_timeout = 0
    for t, zt in enumerate(np.asarray(z, float)):
        if not np.isfinite(zt):
            continue
        if pos == 0:
            if abs(zt) < entry:
                armed = True
            elif armed:
                pos, t0, peak = (1 if zt > 0 else -1), t, abs(zt)
                n_entry += 1
        else:
            m = pos * zt
            peak = max(peak, m)
            if m <= exit_:
                dur.append(t - t0)
                peaks.append(peak)
                pos = 0
            elif t - t0 >= max_hold:
                n_timeout += 1
                pos, armed = 0, False
    open_end = int(pos != 0)
    d = np.asarray(dur, float)
    done = max(n_entry - open_end, 1)
    return {
        "n_entries": n_entry, "n_bounces": len(dur), "n_timeouts": n_timeout, "open_at_end": open_end,
        "success_rate": len(dur) / done,
        "dur_mean": float(d.mean()) if len(d) else np.nan,
        "dur_median": float(np.median(d)) if len(d) else np.nan,
        "dur_p90": float(np.quantile(d, 0.9)) if len(d) else np.nan,
        "peak_z_mean": float(np.mean(peaks)) if peaks else np.nan,
        "n_zero_cross": int((np.diff(np.sign(np.nan_to_num(z))) != 0).sum()),
    }


def kalman_spread(y, Zs, theta0, P0, R, q):
    T, p = Zs.shape
    theta, P = theta0.copy(), P0.copy()
    Q = q * np.eye(p + 1)
    e, path = np.empty(T), np.empty((T, p + 1))
    for t in range(T):
        h = np.r_[1.0, Zs[t]]
        P = P + Q
        Ph = P @ h
        e[t] = y[t] - h @ theta
        K = Ph / (h @ Ph + R)
        theta = theta + K * e[t]
        P = P - np.outer(K, Ph)
        path[t] = theta
    return e, path


# --------------------------------------------------------------------------- candidate evaluation
@dataclass
class Candidate:
    y: str
    terms: tuple[str, ...]
    score: float
    adf_stat: float
    p_eg: float
    half_life: float
    resid_std: float
    max_share: float
    betas: dict[str, float]
    shares: dict[str, float]
    n_i1: int


def _score(p, hl, n_terms, max_share, max_abs_beta_px, cfg: DiscoverConfig) -> float:
    viol = max(0.0, np.log10(max(p, 1e-12) / cfg.p_train_max))
    hl_eff = cfg.hl_cap if (not np.isfinite(hl) or hl < cfg.half_life_min_bars) else min(hl, cfg.hl_cap)
    dom = max(0.0, max_share - cfg.dominance_max) if n_terms >= 2 else 0.0
    big = max(0.0, max_abs_beta_px - cfg.beta_abs_max)
    return (cfg.p_weight * viol + hl_eff / cfg.hl_ref * (1 + cfg.term_penalty * (n_terms - 1))
            + cfg.dom_weight * dom + cfg.beta_weight * big)


def _evaluate(panel: AssetPanel, y: str, terms: tuple[str, ...], cfg: DiscoverConfig) -> Candidate | None:
    yv = panel.series(px(y))
    X = panel.matrix(terms)
    if len(yv) < 200 or not np.isfinite(X).all():
        return None
    n_i1 = sum(panel.is_i1(t) for t in terms)
    icpt, slopes, e = ols_fit(yv, X)
    stat, _ = adf_stat(e, cfg.adf_maxlag, const=False)
    if not np.isfinite(stat):
        return None
    p = adf_p(stat, 1 + n_i1)
    hl = half_life(e)
    contrib = np.abs(slopes) * X.std(0)
    share = contrib / (contrib.sum() + 1e-18)
    px_beta = [abs(b) for t, b in zip(terms, slopes) if kind(t) == "px"]
    sc = _score(p, hl, len(terms), float(share.max()), max(px_beta) if px_beta else 0.0, cfg)
    return Candidate(y, terms, float(sc), stat, p, hl, float(e.std()), float(share.max()),
                     dict(zip(terms, map(float, slopes))), dict(zip(terms, map(float, share))), n_i1)


def _pool(panel: AssetPanel, y: str, cfg: DiscoverConfig) -> list[str]:
    pool = [px(s) for s in panel.symbols if s != y]
    if cfg.use_features:
        for name in cfg.feature_pool:
            pool += [ft(s, name) for s in panel.symbols if panel.has(ft(s, name))]
    return pool


# --------------------------------------------------------------------------- 1) create_pair
def create_pair(panel_train: AssetPanel, y: str, seed_x: Sequence[str], cfg: DiscoverConfig = DiscoverConfig()) -> list[Candidate]:
    cache: dict[tuple[str, ...], Candidate | None] = {}

    def score(terms: tuple[str, ...]) -> Candidate | None:
        key = tuple(sorted(terms))
        if key not in cache:
            cache[key] = _evaluate(panel_train, y, key, cfg)
        return cache[key]

    seed = tuple(sorted(px(s) for s in seed_x))
    root = score(seed)
    if root is None:
        raise ValueError("Seed could not be evaluated (too little train data or NaNs).")
    pool = _pool(panel_train, y, cfg)

    beam, best = [root], root.score
    for depth in range(cfg.max_terms - len(seed)):
        new: dict[tuple[str, ...], Candidate] = {}
        for node in tqdm(beam, desc=f"beam depth {depth + 1}", leave=False):
            for t in pool:
                if t in node.terms:
                    continue
                key = tuple(sorted(node.terms + (t,)))
                if key in new:
                    continue
                c = score(key)
                if c is not None:
                    new[key] = c
        if not new:
            break
        beam = sorted(new.values(), key=lambda c: c.score)[: cfg.k]
        if beam[0].score > best - cfg.min_improve:
            break
        best = beam[0].score

    pruned: dict[tuple[str, ...], Candidate] = {}
    for cand in sorted((c for c in cache.values() if c is not None), key=lambda c: c.score)[: cfg.k]:
        cur, improved = cand, True
        while improved:
            improved = False
            for leg in [t for t in cur.terms if t not in seed]:
                trial = score(tuple(t for t in cur.terms if t != leg))
                if trial is not None and trial.score <= cur.score + 1e-12:
                    cur, improved = trial, True
                    break
        pruned[tuple(sorted(cur.terms))] = cur
    return sorted(pruned.values(), key=lambda c: c.score)[: cfg.k]


# --------------------------------------------------------------------------- 2) test_pair
@dataclass
class TestResult:
    __test__ = False
    candidate: Candidate
    adf_oos: float
    p_oos: float
    hl_oos: float
    mean_shift_sigma: float
    beta_block_range: float
    max_beta_move: float
    bounces_train: dict
    bounces_oos: dict
    p_ok: bool
    hl_ok: bool
    dom_ok: bool
    passed: bool


def _spread_parts(panel: AssetPanel, cand: Candidate, split: int, cfg: DiscoverConfig):
    yv, X = panel.series(px(cand.y)), panel.matrix(cand.terms)
    icpt, slopes, e_tr = ols_fit(yv[:split], X[:split])
    sigma = float(e_tr.std())
    move = 0.0
    if cfg.beta_mode == "static":
        s_te = yv[split:] - icpt - X[split:] @ slopes
    elif cfg.beta_mode == "kalman":
        mu, sd = X[:split].mean(0), np.where(X[:split].std(0) > 1e-12, X[:split].std(0), 1.0)
        Ztr = np.column_stack([np.ones(split), (X[:split] - mu) / sd])
        theta0 = np.linalg.lstsq(Ztr, yv[:split], rcond=None)[0]
        R = float(e_tr.var())
        P0 = cfg.kalman_p0_scale * R * np.linalg.inv(Ztr.T @ Ztr)
        s_te, path = kalman_spread(yv[split:], (X[split:] - mu) / sd, theta0, P0, R, cfg.kalman_q)
        move = float(np.abs(path[:, 1:] - theta0[1:]).max() / yv[:split].std())
    else:
        raise ValueError("beta_mode must be 'static' or 'kalman'")
    return icpt, slopes, e_tr, s_te, sigma, move


def _z(e_tr, s_te, sigma, cfg: DiscoverConfig):
    if cfg.z_mode == "static":
        return e_tr / sigma, s_te / sigma
    full = pd.Series(np.r_[e_tr, s_te])
    r = full.rolling(cfg.z_window, min_periods=cfg.z_window // 4)
    z = ((full - r.mean()) / r.std()).to_numpy()
    return z[: len(e_tr)], z[len(e_tr):]


def _block_range(yv, X, split, blocks=4) -> float:
    edges = np.linspace(0, split, blocks + 1).astype(int)
    B = np.array([ols_fit(yv[a:b], X[a:b])[1] * X[:split].std(0) for a, b in zip(edges[:-1], edges[1:])])
    return float((B.max(0) - B.min(0)).max() / yv[:split].std())


def test_pair(panel: AssetPanel, cand: Candidate, cfg: DiscoverConfig = DiscoverConfig(), split: int | None = None) -> TestResult:
    split = panel.split_index(cfg.train_frac) if split is None else split
    yv, X = panel.series(px(cand.y)), panel.matrix(cand.terms)
    _, _, e_tr, s_te, sigma, move = _spread_parts(panel, cand, split, cfg)
    if len(s_te) < 100:
        raise ValueError("Not enough OOS data.")
    stat, _ = adf_stat(s_te, cfg.adf_maxlag, const=True)
    p = adf_p(stat, 1)
    hl = half_life(s_te)
    z_tr, z_te = _z(e_tr, s_te, sigma, cfg)
    b_tr = bounce_stats(z_tr, cfg.z_entry, cfg.z_exit, cfg.z_max_hold)
    b_te = bounce_stats(z_te, cfg.z_entry, cfg.z_exit, cfg.z_max_hold)
    
    p_ok = p <= cfg.p_oos_max
    hl_ok = cfg.half_life_min_bars <= hl <= cfg.half_life_max_bars
    dom_ok = len(cand.terms) < 2 or cand.max_share <= cfg.dominance_max
    
    # Bouncer / trade count rule removed from passed criteria!
    passed = bool(p_ok and hl_ok and dom_ok)
    
    return TestResult(
        cand, stat, p, hl, float(s_te.mean() / sigma), _block_range(yv, X, split), move, b_tr, b_te,
        p_ok, hl_ok, dom_ok, passed)


# --------------------------------------------------------------------------- 3) analysis
@dataclass
class Analysis:
    result: TestResult
    terms: pd.DataFrame
    ablation: pd.DataFrame
    bounces: pd.DataFrame


def _variant_row(panel, y, terms, split, cfg, name) -> dict:
    yv, X = panel.series(px(y)), panel.matrix(terms)
    n_i1 = sum(panel.is_i1(t) for t in terms)
    icpt, slopes, e_tr = ols_fit(yv[:split], X[:split])
    st, _ = adf_stat(e_tr, cfg.adf_maxlag, const=False)
    s_te = yv[split:] - icpt - X[split:] @ slopes
    so, _ = adf_stat(s_te, cfg.adf_maxlag, const=True)
    z = s_te / e_tr.std()
    b = bounce_stats(z, cfg.z_entry, cfg.z_exit, cfg.z_max_hold)
    return {"variant": name, "n_terms": len(terms), "adf_train": st, "p_eg_train": adf_p(st, 1 + n_i1),
            "hl_train": half_life(e_tr), "adf_oos": so, "p_oos": adf_p(so, 1), "hl_oos": half_life(s_te),
            "bounces_oos": b["n_bounces"]}


def analyze(panel: AssetPanel, cand: Candidate, cfg: DiscoverConfig = DiscoverConfig(), split: int | None = None) -> Analysis:
    split = panel.split_index(cfg.train_frac) if split is None else split
    res = test_pair(panel, cand, cfg, split)
    y, terms = cand.y, cand.terms
    px_terms = tuple(t for t in terms if kind(t) == "px")
    ft_terms = tuple(t for t in terms if kind(t) == "ft")

    rows = [_variant_row(panel, y, px_terms, split, cfg, "assets only")]
    if ft_terms:
        rows.append(_variant_row(panel, y, terms, split, cfg, "assets + features (full)"))
        for t in ft_terms:
            rows.append(_variant_row(panel, y, tuple(x for x in terms if x != t), split, cfg, f"full - {label(t)}"))
    ab = pd.DataFrame(rows)
    base = ab.iloc[0]
    for c in ("adf_train", "adf_oos", "hl_oos"):
        ab[f"d_{c}"] = ab[c] - base[c]
    ab["d_log10p_oos"] = np.log10(ab["p_oos"].clip(lower=1e-12)) - np.log10(max(base["p_oos"], 1e-12))

    yv, X = panel.series(px(y)), panel.matrix(terms)
    full = adf_stat(ols_fit(yv[:split], X[:split])[2], cfg.adf_maxlag, False)[0]
    sy = yv[:split].std()
    trs = []
    for j, t in enumerate(terms):
        xs = X[:split, j]
        st_term, _ = adf_stat(xs, cfg.adf_maxlag, const=True)
        rest = tuple(x for x in terms if x != t)
        loo = np.nan
        if rest and any(kind(x) == "px" for x in rest):
            Xr = panel.matrix(rest)
            loo = adf_stat(ols_fit(yv[:split], Xr[:split])[2], cfg.adf_maxlag, False)[0] - full
        trs.append({
            "term": label(t), "kind": kind(t), "i1": panel.is_i1(t), "beta": cand.betas[t],
            "share": cand.shares[t], "gross_vs_y": abs(cand.betas[t]) * xs.std() / sy,
            "p_term_alone": adf_p(st_term, 1),
            "term_stationary": bool(kind(t) == "px" and adf_p(st_term, 1) < cfg.term_stationary_p),
            "huge_beta": bool(kind(t) == "px" and abs(cand.betas[t]) > cfg.beta_abs_max),
            "loo_d_adf": loo,
        })
    tt = pd.DataFrame(trs)

    bn = pd.DataFrame({"train": res.bounces_train, "oos": res.bounces_oos})
    bm = panel.bar_minutes
    bn.loc["dur_median_hours"] = bn.loc["dur_median"] * bm / 60
    bn.loc["dur_p90_hours"] = bn.loc["dur_p90"] * bm / 60
    return Analysis(res, tt, ab, bn)


# --------------------------------------------------------------------------- plotting with beta movement
def plot_spread(panel: AssetPanel, res: TestResult, cfg: DiscoverConfig = DiscoverConfig(), split: int | None = None):
    """Plots the spread, z-score, and rolling/dynamic beta movement across train and out-of-sample periods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    split = panel.split_index(cfg.train_frac) if split is None else split
    cand = res.candidate
    yv, X = panel.series(px(cand.y)), panel.matrix(cand.terms)
    _, _, e_tr, s_te, sigma, _ = _spread_parts(panel, cand, split, cfg)
    
    times = panel.logp.index
    train_times = times[:split]
    oos_times = times[split:]
    
    z_tr, z_te = _z(e_tr, s_te, sigma, cfg)
    
    # Compute rolling OLS betas across the full time series to visualize beta evolution/movement
    window = min(288, len(yv) // 4)
    T, p = X.shape
    rolling_betas = np.full((T, p), np.nan)
    for t in range(window, T):
        _, slopes, _ = ols_fit(yv[t-window:t], X[t-window:t])
        rolling_betas[t] = slopes

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    
    # Top subplot: Spread values (Train vs OOS)
    ax1.plot(train_times, e_tr, label="Train Spread", color="#1f77b4", alpha=0.8)
    ax1.plot(oos_times, s_te, label="OOS Spread", color="#ff7f0e", alpha=0.8)
    ax1.axvline(times[split], color="red", linestyle="--", label="Train/OOS Split")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Spread (Raw Units)")
    ax1.set_title(f"StatArb Spread & Beta Movement: {cand.y} vs {' + '.join(label(t) for t in cand.terms)}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Middle subplot: Z-Score with entry/exit bands
    ax2.plot(train_times, z_tr, label="Train Z-Score", color="#1f77b4", alpha=0.8)
    ax2.plot(oos_times, z_te, label="OOS Z-Score", color="#ff7f0e", alpha=0.8)
    ax2.axvline(times[split], color="red", linestyle="--")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(cfg.z_entry, color="green", linestyle="--", alpha=0.7, label=f"+{cfg.z_entry} Entry")
    ax2.axhline(-cfg.z_entry, color="green", linestyle="--", alpha=0.7, label=f"-{cfg.z_entry} Entry")
    ax2.axhline(cfg.z_exit, color="purple", linestyle=":", alpha=0.7, label=f"Exit ({cfg.z_exit})")
    ax2.axhline(-cfg.z_exit, color="purple", linestyle=":", alpha=0.7)
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Bottom subplot: Beta Movement over time
    for j, t_name in enumerate(cand.terms):
        lbl = label(t_name)
        ax3.plot(times, rolling_betas[:, j], label=f"Rolling Beta ({lbl})", alpha=0.8)
        # Plot horizontal dashed line for the static train beta as reference
        ax3.axhline(cand.betas[t_name], linestyle=":", alpha=0.5, label=f"Static Beta ({lbl})")
    
    ax3.axvline(times[split], color="red", linestyle="--")
    ax3.set_ylabel("Beta Value")
    ax3.set_xlabel("Time")
    ax3.legend(loc="upper left", ncol=2)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- run
def run(panel: AssetPanel, y: str, seed_x: Sequence[str], cfg: DiscoverConfig = DiscoverConfig()):
    split = panel.split_index(cfg.train_frac)
    ranked = create_pair(panel.iloc(0, split), y, seed_x, cfg)
    results = [test_pair(panel, c, cfg, split) for c in ranked]
    table = pd.DataFrame([{
        "terms": " + ".join(label(t) for t in r.candidate.terms),
        "p_eg_train": r.candidate.p_eg, "hl_train": r.candidate.half_life,
        "p_oos": r.p_oos, "hl_oos": r.hl_oos,
        "max_share": r.candidate.max_share, "beta_blk_rng": r.beta_block_range,
        "p_ok": r.p_ok, "hl_ok": r.hl_ok, "dom_ok": r.dom_ok, "passed": r.passed,
    } for r in results])
    return results, table


if __name__ == "__main__":
    SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
    Y, SEED_X = "ETH/USDT", ["BTC/USDT"]
    cfg = DiscoverConfig()
    panel = get_assets(SYMBOLS, start="2024-01-01", end="2024-01-31", bar="5min", market="um", cfg=cfg)
    results, table = run(panel, Y, SEED_X, cfg)
    pd.set_option("display.width", 250)
    print(table.to_string())
    best = next((r for r in results if r.passed), results[0])
    a = analyze(panel, best.candidate, cfg)
    print("\n== terms ==\n", a.terms.to_string(), "\n\n== ablation ==\n", a.ablation.to_string(), "\n\n== bounces ==\n", a.bounces.to_string())
    
    # Plot the spread, z-score, and beta movement chart
    plot_spread(panel, best, cfg)