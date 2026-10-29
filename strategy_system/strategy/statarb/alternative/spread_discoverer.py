"""
Multivariate spread discoverer: asset legs AND flow/depth terms, static (or ultra-slow) betas.

    spread_t = y_t - a - sum_j beta_j * term_j,t
    term     = log price of an asset ("px|SOL/USDT")  or  a per-asset feature ("ft|BTC/USDT|cvd")

Stages
    get_assets()   -> bars -> features -> AssetPanel
    create_pair()  -> beam search on the FIRST part of the data (unseeded or anchored search)
    test_pair()    -> out-of-sample on the REST with train betas FROZEN (or a Kalman with tiny q)
    analyze()      -> feature ablation, per-term dominance table, z-score bounce statistics

Objective (lower is better)
    score = p_weight * max(0, log10(p_EG / p_train_max))     # first get under the ADF threshold
          + half_life / hl_ref                                # then shrink the half-life
          * (1 + term_penalty * (n_terms - 1))                # (half-life part only) each extra term must cut it by ~term_penalty
          + drift_weight * max(0, log10(drift / drift_max))   # beta-drift impact as % of typical TRAIN spread RMS
          + dom_weight * max(0, max_share - dominance_max)    # one term carrying the whole hedge
          + beta_weight * (|beta_px| above beta_abs_max)      # huge price betas

Beta drift, measured directly (the dBeta * underlying idea)
    The spread is  s_t = y_t - a - sum_j beta_j x_j,t.  If beta_j is re-estimated to beta_j + dbeta_j, the spread moves by
        impact_j,t = dbeta_j * (x_j,t - mu_j)          (mu_j = train mean of x_j; for a log price x - mu is "how far the
                                                        asset has wandered from where beta was fitted")
    expressed as a percentage of the TRAIN spread's RMS deviation (the standard deviation of its centred residuals).
    This denominator is frozen from training so an OOS spread explosion cannot inflate the denominator and hide drift.
    A term with a tiny dbeta can still move the spread substantially if x wanders far, so raw beta values are not the
    right thing to look at. drift = 100 * max over blocks of RMS_t(sum_j impact_j,t) / train_spread_rms. It is part of
    the objective, a pass criterion in test_pair, and the bottom panel of plot_spread. It is a relative percentage,
    not a percentage of price and not a percentage of the spread's instantaneous value (which can be near zero).

Fitting method
    fit(y, X, method=...) picks the estimator; DiscoverConfig.fit_method / fit_kw set it everywhere.
    "ols" (default; conventional Engle-Granger first stage), "ridge", "stable" (block-consensus), "kalman".
    Non-OLS estimators are available for exploratory fits, but their in-sample Engle-Granger p-values are not classical.

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

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp
from tqdm import tqdm

try:  # package layout of your repo
    from strategy_system.strategy.statarb.alternative.binance_vision import load_bars
    from strategy_system.strategy.statarb.alternative.flow_features import (
        INTEGRATED_FEATURES, FeatureConfig, features, warmup_bars)
except ImportError:  # flat folder
    from binance_vision import load_bars
    from flow_features import INTEGRATED_FEATURES, FeatureConfig, features, warmup_bars


# --------------------------------------------------------------------------- config
@dataclass
class DiscoverConfig:
    train_frac: float = 0.5
    k: int = 100                      # beam width: much wider than the original 20
    max_terms: int = 6                # RHS terms allowed in total; use OOS validation to control overfit
    use_features: bool = True
    # Broad per-asset feature pool. Missing/unavailable features are dropped by get_assets().
    # cvd / ofi_cum / oi_log / liq_cum are treated as I(1)-like; most others are I(0) state variables.
    # Broadening this pool increases data-mining risk; trust OOS results, not the best training score.
    feature_pool: tuple[str, ...] = (
        "cvd", "ofi_cum", "oi_log", "liq_cum",
        "cvd_s", "cvd_l", "net_flow", "tfi", "tfi_s", "tfi_l", "tfi_cnt",
        "ofi", "ofi_s", "ofi_l", "qi1", "rel_spread",
        "d_oi", "oi_chg_s", "oi_chg_l", "oi_z", "oi_dir_s", "oi_dir_l",
        "ls_top", "ls_acct", "taker_ls", "ls_gap",
        "funding_bps", "prem_bps", "prem_chg", "prem_s", "prem_z",
        "liq_net", "liq_tot", "liq_imb", "liq_s", "liq_l",
        "notional_z", "trade_size_ratio", "large_share", "kyle_lambda", "illiq",
        "di1", "ddi1", "liq1", "dliq1", "depth_flow1",
        "di5", "ddi5", "liq5", "dliq5", "depth_flow5", "depth_slope", "absorb",
    )
    min_feature_coverage: float = 0.95  # a pool feature that is NaN on more of the sample than this is dropped (liquidation
                                        # files, for example, are absent for some periods) instead of deleting the rows

    # --- how the spread is fitted
    fit_method: str = "ols"           # OLS keeps the conventional Engle-Granger screen calibrated; alternatives are exploratory
    fit_kw: dict = field(default_factory=dict)   # e.g. {"between": 0.1} for stable, {"lam": 0.05} for ridge

    # --- beta-drift impact: percent of typical TRAIN spread RMS, using a frozen training denominator
    drift_blocks: int = 6             # data blocks used to re-estimate beta
    drift_max: float = 100.0          # beam: penalise impacts above 100% of typical training-spread RMS (same as old 1 sigma)
    drift_weight: float = 100.0       # same footing as p_weight: large beta drift is strongly penalised
    drift_oos_max: float = 200.0      # OOS limit: 200% of typical training-spread RMS (same as old 2 sigma)
    drift_window: int = 2016          # rolling re-fit window for the chart / beta_drift_impact (bars; 864 = 3 days of 5min)
    drift_stride: int = 12            # re-fit every this many bars for the chart (value held in between: causal)

    # --- ADF / objective
    adf_maxlag: int = 12              # ADF augmentation lags chosen by AIC up to this
    p_train_max: float = 0.05         # in-sample Engle-Granger p target
    p_oos_max: float = 0.10           # out-of-sample ADF p required to pass
    p_weight: float = 100.0
    hl_ref: float = 100.0             # bars; half_life/hl_ref enters the score
    hl_cap: float = 5000.0            # bars; used when half-life is inf / invalid
    half_life_min_bars: float = 2.0   # faster than this = bid-ask-bounce noise, rejected
    half_life_max_bars: float = 576.0 # OOS must revert within this many bars (48h for 5-minute bars)
    term_penalty: float = 0.35        # relative: an extra term must shorten the half-life by roughly this fraction
    min_improve: float = 1e-3

    # --- overfitting guards
    dominance_max: float = 0.85       # max share of |beta|*std(term) among terms (only when >= 2 terms)
    dom_weight: float = 5.0
    beta_abs_max: float = 5.0         # |beta| on a log-price leg above this is suspicious
    beta_weight: float = 1.0
    term_stationary_p: float = 0.05   # a price leg that is itself stationary below this p is flagged

    # --- beta handling in the OOS test
    beta_mode: str = "static"         # "static" (freeze train fit_method betas) or "kalman"
    kalman_q: float = 1e-12           # state-noise variance per bar, in STANDARDISED coefficient units
    kalman_p0_scale: float = 1e-4     # prior covariance = scale * (OLS covariance of the train fit). 1.0 ~ recursive
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


def _loader_kwargs_compatible(loader, kwargs: dict) -> tuple[dict, list[str]]:
    """Filter optional keyword arguments against the imported load_bars signature.

    This keeps the discoverer compatible with older/local binance_vision.py versions
    that predate metrics, funding, premium, liquidation, or bookTicker support. When a
    loader accepts **kwargs, we trust its advertised forwarding interface.
    """
    try:
        params = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return kwargs, []
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []
    accepted = {
        name: value for name, value in kwargs.items()
        if name in params and params[name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    ignored = sorted(set(kwargs) - set(accepted))
    return accepted, ignored


def get_assets(symbols, start, end, bar="5min", market="um", with_depth=True, with_ticker=False,
               cache_dir="~/.cache/binance_vision", workers=4, fcfg: FeatureConfig = FeatureConfig(),
               cfg: DiscoverConfig = DiscoverConfig(), fetch=None,
               with_metrics=True, with_funding=True, with_premium=True, with_liq=True) -> AssetPanel:
    requested = dict(
        bar=bar, market=market, with_depth=with_depth, with_ticker=with_ticker,
        cache_dir=cache_dir, workers=workers,
        with_metrics=with_metrics, with_funding=with_funding,
        with_premium=with_premium, with_liq=with_liq,
    )
    if fetch is not None:
        requested["fetch"] = fetch
    loader_kw, ignored = _loader_kwargs_compatible(load_bars, requested)
    if ignored:
        warnings.warn(
            "The imported binance_vision.load_bars() does not support these arguments: "
            + ", ".join(ignored)
            + ". Continuing with the arguments it supports; features requiring unavailable "
              "datasets (for example OI/funding/premium/liquidations or bookTicker OFI) "
              "will not be built. Update binance_vision.py if you need those datasets.",
            RuntimeWarning,
            stacklevel=2,
        )
    frames = {}
    for s in tqdm(symbols, desc="assets"):
        bars = load_bars(s, start, end, **loader_kw)
        frames[s] = features(bars, fcfg)
    if not cfg.use_features:
        return AssetPanel.from_frames(frames, warmup=warmup_bars(fcfg))

    # Filter unavailable/low-coverage features out of the pool, but do NOT require every candidate feature
    # to be present on every row. Candidate-specific finite checks below decide whether a term is usable.
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    warm = warmup_bars(fcfg)
    keep, dropped = [], {}
    for name in cfg.feature_pool:
        if not all(name in f.columns for f in frames.values()):
            dropped[name] = "not built for every symbol (dataset missing)"
            continue
        cov = min(float(f[name].reindex(idx).iloc[warm:].notna().mean()) for f in frames.values())
        if cov >= cfg.min_feature_coverage:
            keep.append(name)
        else:
            dropped[name] = f"coverage {cov:.0%}"
    if dropped:
        warnings.warn("feature_pool entries dropped: " + "; ".join(f"{k} ({v})" for k, v in dropped.items()))
    panel = AssetPanel.from_frames(frames, warmup=warm)
    for name in dropped:
        panel.feats.pop(name, None)
    return panel


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


# --------------------------------------------------------------------------- fitting methods
# Every estimator has the signature  (y, X, **kw) -> (intercept, slopes, residuals)  in RAW units, like ols_fit.
def _prep(X: np.ndarray):
    X = X if X.ndim == 2 else X[:, None]
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return X, mu, sd, (X - mu) / sd


def _finish(y, X, mu, sd, beta_std):
    slopes = beta_std / sd
    icpt = float(y.mean() - slopes @ mu)
    return icpt, slopes, y - icpt - X @ slopes


def ridge_fit(y: np.ndarray, X: np.ndarray, lam: float = 1e-2, beta0: np.ndarray | None = None):
    """Standardised ridge. lam is on the CORRELATION scale (columns have unit variance), so lam=0.01 is a light touch
    that stops two collinear legs from taking huge opposite-signed betas, and lam=1 is heavy.
    beta0 (RAW units) is what to shrink TOWARD: None = zero, or pass a reference beta to anchor a re-fit to it
    (lam -> inf freezes beta at beta0)."""
    X, mu, sd, Z = _prep(X)
    n, p = Z.shape
    b0 = np.zeros(p) if beta0 is None else np.asarray(beta0, float) * sd
    b = np.linalg.solve(Z.T @ Z / n + lam * np.eye(p), Z.T @ (y - y.mean()) / n + lam * b0)
    return _finish(y, X, mu, sd, b)


def stable_fit(y: np.ndarray, X: np.ndarray, n_blocks: int = 6, between: float = 0.25, lam: float = 1e-3):
    """Block-consensus fit: beta that explains the data INSIDE every block, not just in aggregate.

        minimise   (1-between) * sum_b RSS_b(beta; free intercept per block)  +  between * RSS(beta; one intercept)

    between=1 is plain OLS. Lower values stop slow between-block level drift (the part that makes OLS betas swing when
    the sample is cut differently) from steering beta, so the beta you freeze is the one that hedges in every block.
    Closed form, one linear solve. lam is a spectral ridge (fraction of the mean eigenvalue) for conditioning only.
    The intercept is global, so a level drift between blocks is NOT removed from the spread (use z_mode='rolling' or a
    slow intercept for that); only its influence on beta is.
    """
    X, mu, sd, Z = _prep(X)
    n, p = Z.shape
    nb = int(max(1, min(n_blocks, n // 50)))
    yc = y - y.mean()
    Sg, cg = Z.T @ Z, Z.T @ yc
    Sw, cw = np.zeros((p, p)), np.zeros(p)
    e = np.linspace(0, n, nb + 1).astype(int)
    for a, b in zip(e[:-1], e[1:]):
        Zb, yb = Z[a:b] - Z[a:b].mean(0), y[a:b] - y[a:b].mean()
        Sw += Zb.T @ Zb
        cw += Zb.T @ yb
    A = ((1 - between) * Sw + between * Sg) / n
    c = ((1 - between) * cw + between * cg) / n
    A = A + lam * np.trace(A) / p * np.eye(p)
    return _finish(y, X, mu, sd, np.linalg.solve(A, c))


def kalman_fit(y: np.ndarray, X: np.ndarray, q: float = 1e-8, p0_scale: float = 1e-2):
    """Random-walk-beta Kalman filter (state noise q per bar, standardised units). Returns the FINAL state as the
    betas and the one-step-ahead innovations as the residual. The prior is an OLS fit on the same data, so in-sample
    residuals are optimistic; this is meant for the final/OOS leg (cfg.beta_mode='kalman'), not for a big beam
    (python loop). q -> 0 gives back a static beta."""
    X, mu, sd, Z = _prep(X)
    n, p = Z.shape
    H = np.column_stack([np.ones(n), Z])
    theta0 = np.linalg.lstsq(H, y, rcond=None)[0]
    R = float(((y - H @ theta0) ** 2).mean())
    P0 = p0_scale * R * np.linalg.inv(H.T @ H)
    e, path = kalman_spread(y, Z, theta0, P0, R, q)
    slopes = path[-1, 1:] / sd
    return float(path[-1, 0] - slopes @ mu), slopes, e


FIT_METHODS = {"ols": ols_fit, "ridge": ridge_fit, "stable": stable_fit, "kalman": kalman_fit}


def fit(y: np.ndarray, X: np.ndarray, method: str = "stable", **kw):
    """Single entry point for forming a stat-arb relationship. Returns (intercept, slopes, residuals).
    method: "ols" | "ridge" | "stable" | "kalman". Extra kwargs go to the estimator. Add a method by adding it to FIT_METHODS."""
    try:
        f = FIT_METHODS[method]
    except KeyError:
        raise ValueError(f"unknown fit method {method!r}; choose from {sorted(FIT_METHODS)}") from None
    return f(y, X, **kw)


def _fit(y, X, cfg: "DiscoverConfig"):
    return fit(y, X, cfg.fit_method, **cfg.fit_kw)


# --------------------------------------------------------------------------- beta drift: dBeta * underlying
def _edges(n: int, blocks: int) -> np.ndarray:
    return np.linspace(0, n, max(1, blocks) + 1).astype(int)


def block_betas(y: np.ndarray, X: np.ndarray, blocks: int, lam: float = 1e-4) -> np.ndarray:
    """Local betas, one fit per contiguous block (light ridge so a degenerate block stays finite). Shape (blocks, p)."""
    X = X if X.ndim == 2 else X[:, None]
    e = _edges(len(y), blocks)
    return np.array([ridge_fit(y[a:b], X[a:b], lam=lam)[1] for a, b in zip(e[:-1], e[1:])])


def drift_ratio(y, X, slopes, spread_rms: float, blocks: int, mu_ref=None) -> float:
    """Worst-block RMS beta-drift impact as % of typical TRAIN spread RMS.

    `spread_rms` is the centred RMS/standard deviation of the training residual and is
    held fixed for both training and OOS calculations. The returned value is a percent:
    100 means beta re-estimation would shift the spread by an RMS amount equal to its
    typical training fluctuation. This is not a percent of price or of the instantaneous
    spread value (which may be close to zero).
    """
    X = X if X.ndim == 2 else X[:, None]
    mu = X.mean(0) if mu_ref is None else mu_ref
    B = block_betas(y, X, blocks)
    e = _edges(len(y), blocks)
    worst = 0.0
    for (a, b), bb in zip(zip(e[:-1], e[1:]), B):
        imp = ((X[a:b] - mu) * (bb - slopes)).sum(1)
        worst = max(worst, float(np.sqrt(np.mean(imp * imp))))
    return 100.0 * worst / max(float(spread_rms), 1e-12)


def drift_by_term(y, X, slopes, spread_rms: float, blocks: int, mu_ref=None) -> np.ndarray:
    """Per-term worst-block RMS impact as % of typical TRAIN spread RMS.

    Term impacts do not add linearly because they can offset one another; their sum is
    an upper bound on the absolute combined drift, not equal to the combined drift.
    """
    X = X if X.ndim == 2 else X[:, None]
    mu = X.mean(0) if mu_ref is None else mu_ref
    B, e = block_betas(y, X, blocks), _edges(len(y), blocks)
    out = np.zeros(X.shape[1])
    for (a, b), bb in zip(zip(e[:-1], e[1:]), B):
        out = np.maximum(out, np.sqrt((((X[a:b] - mu) * (bb - slopes)) ** 2).mean(0)))
    return 100.0 * out / max(float(spread_rms), 1e-12)


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
    drift: float = float("nan")      # beta-drift impact as % of typical train-spread RMS


def _score(p, hl, n_terms, max_share, max_abs_beta_px, cfg: DiscoverConfig, drift: float = 0.0) -> float:
    viol = max(0.0, np.log10(max(p, 1e-12) / cfg.p_train_max))
    hl_eff = cfg.hl_cap if (not np.isfinite(hl) or hl < cfg.half_life_min_bars) else min(hl, cfg.hl_cap)
    dom = max(0.0, max_share - cfg.dominance_max) if n_terms >= 2 else 0.0
    big = max(0.0, max_abs_beta_px - cfg.beta_abs_max)
    dr = max(0.0, np.log10(max(drift, 1e-12) / cfg.drift_max)) if np.isfinite(drift) else 0.0
    return (cfg.p_weight * viol + hl_eff / cfg.hl_ref * (1 + cfg.term_penalty * (n_terms - 1))
            + cfg.dom_weight * dom + cfg.beta_weight * big + cfg.drift_weight * dr)


def _evaluate(panel: AssetPanel, y: str, terms: tuple[str, ...], cfg: DiscoverConfig) -> Candidate | None:
    yv = panel.series(px(y))
    X = panel.matrix(terms)
    if len(yv) < 200 or not np.isfinite(X).all():
        return None
    n_i1 = sum(panel.is_i1(t) for t in terms)
    icpt, slopes, e = _fit(yv, X, cfg)
    stat, _ = adf_stat(e, cfg.adf_maxlag, const=False)
    if not np.isfinite(stat):
        return None
    p = adf_p(stat, 1 + n_i1)
    hl = half_life(e)
    contrib = np.abs(slopes) * X.std(0)
    share = contrib / (contrib.sum() + 1e-18)
    px_beta = [abs(b) for t, b in zip(terms, slopes) if kind(t) == "px"]
    drift = drift_ratio(yv, X, slopes, e.std(), cfg.drift_blocks)
    sc = _score(p, hl, len(terms), float(share.max()), max(px_beta) if px_beta else 0.0, cfg, drift)
    return Candidate(y, terms, float(sc), stat, p, hl, float(e.std()), float(share.max()),
                     dict(zip(terms, map(float, slopes))), dict(zip(terms, map(float, share))), n_i1, float(drift))


def _pool(panel: AssetPanel, y: str, cfg: DiscoverConfig) -> list[str]:
    pool = [px(s) for s in panel.symbols if s != y]
    if cfg.use_features:
        for name in cfg.feature_pool:
            pool += [ft(s, name) for s in panel.symbols if panel.has(ft(s, name))]
    return pool


# --------------------------------------------------------------------------- 1) create_pair
def create_pair(panel_train: AssetPanel, y: str, seed_x: Sequence[str] | None = None,
                cfg: DiscoverConfig = DiscoverConfig()) -> list[Candidate]:
    """Search for spread constructions on the training panel only.

    If seed_x is non-empty, its price legs are mandatory anchors (legacy behavior).
    If seed_x is None or empty, search starts from every usable I(1)-like RHS term,
    so no single asset is forced into every candidate.
    """
    cache: dict[tuple[str, ...], Candidate | None] = {}

    def score(terms: tuple[str, ...]) -> Candidate | None:
        key = tuple(sorted(terms))
        if key not in cache:
            cache[key] = _evaluate(panel_train, y, key, cfg)
        return cache[key]

    pool = _pool(panel_train, y, cfg)
    seed = tuple(sorted(px(s) for s in (seed_x or ())))
    if len(seed) > cfg.max_terms:
        raise ValueError(f"seed has {len(seed)} terms but max_terms={cfg.max_terms}.")

    if seed:
        root = score(seed)
        if root is None:
            raise ValueError("Seed could not be evaluated (too little train data or NaNs).")
        beam = [root]
        max_additions = cfg.max_terms - len(seed)
    else:
        # A level regression should start with an I(1)-like term; I(0) features can be added later.
        starters = [t for t in pool if panel_train.is_i1(t)]
        roots = [c for t in starters if (c := score((t,))) is not None]
        if not roots:
            raise ValueError("No usable I(1)-like starting terms. Add assets or enable integrated features.")
        beam = sorted(roots, key=lambda c: c.score)[:cfg.k]
        max_additions = cfg.max_terms - 1

    # Do not stop merely because one extra term fails to improve the score: a second term can
    # make a useful combination even when the intermediate model is worse.
    for depth in range(max_additions):
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
        beam = sorted(new.values(), key=lambda c: c.score)[:cfg.k]

    # Greedily remove non-anchor terms only where the objective does not get worse.
    # In unanchored mode this can reduce a candidate to a singleton, but never to an empty model.
    pruned: dict[tuple[str, ...], Candidate] = {}
    for cand in sorted((c for c in cache.values() if c is not None), key=lambda c: c.score)[:cfg.k]:
        cur, improved = cand, True
        while improved:
            improved = False
            for leg in [t for t in cur.terms if t not in seed]:
                rest = tuple(t for t in cur.terms if t != leg)
                if not rest:
                    continue
                trial = score(rest)
                if trial is not None and trial.score <= cur.score + 1e-12:
                    cur, improved = trial, True
                    break
        pruned[tuple(sorted(cur.terms))] = cur
    return sorted(pruned.values(), key=lambda c: c.score)[:cfg.k]


# --------------------------------------------------------------------------- 2) test_pair
@dataclass
class TestResult:
    __test__ = False
    candidate: Candidate
    adf_oos: float
    p_oos: float
    hl_oos: float
    mean_shift_pct_train_spread_rms: float
    beta_block_range: float
    max_beta_move: float
    bounces_train: dict
    bounces_oos: dict
    p_ok: bool
    hl_ok: bool
    dom_ok: bool
    passed: bool
    drift_train: float = float("nan")   # beta-drift impact on train blocks, % of typical train-spread RMS
    drift_oos: float = float("nan")     # OOS beta-drift impact, % of typical train-spread RMS
    drift_ok: bool = True


def _spread_parts(panel: AssetPanel, cand: Candidate, split: int, cfg: DiscoverConfig):
    yv, X = panel.series(px(cand.y)), panel.matrix(cand.terms)
    icpt, slopes, e_tr = _fit(yv[:split], X[:split], cfg)
    sigma = float(e_tr.std())
    move = 0.0
    if cfg.beta_mode == "static":
        s_te = yv[split:] - icpt - X[split:] @ slopes
    elif cfg.beta_mode == "kalman":
        mu, sd = X[:split].mean(0), np.where(X[:split].std(0) > 1e-12, X[:split].std(0), 1.0)
        Ztr = np.column_stack([np.ones(split), (X[:split] - mu) / sd])
        # start the filter at the fitted static betas, in standardised coordinates
        theta0 = np.r_[icpt + slopes @ mu, slopes * sd]
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
    B = block_betas(yv[:split], X[:split], blocks) * X[:split].std(0)
    return float((B.max(0) - B.min(0)).max() / yv[:split].std())


def test_pair(panel: AssetPanel, cand: Candidate, cfg: DiscoverConfig = DiscoverConfig(), split: int | None = None) -> TestResult:
    split = panel.split_index(cfg.train_frac) if split is None else split
    yv, X = panel.series(px(cand.y)), panel.matrix(cand.terms)
    if not np.isfinite(X).all() or not np.isfinite(yv).all():
        raise ValueError("Candidate target/terms contain NaNs or infinities over train+OOS; skip this candidate or use better-covered terms.")
    _, slopes, e_tr, s_te, sigma, move = _spread_parts(panel, cand, split, cfg)
    if len(s_te) < 100:
        raise ValueError("Not enough OOS data.")
    stat, _ = adf_stat(s_te, cfg.adf_maxlag, const=True)
    p = adf_p(stat, 1)
    hl = half_life(s_te)
    z_tr, z_te = _z(e_tr, s_te, sigma, cfg)
    b_tr = bounce_stats(z_tr, cfg.z_entry, cfg.z_exit, cfg.z_max_hold)
    b_te = bounce_stats(z_te, cfg.z_entry, cfg.z_exit, cfg.z_max_hold)

    mu_tr = X[:split].mean(0)
    d_tr = drift_ratio(yv[:split], X[:split], slopes, sigma, cfg.drift_blocks)
    nb_oos = int(max(1, min(cfg.drift_blocks, len(s_te) // 100)))
    d_oos = drift_ratio(yv[split:], X[split:], slopes, sigma, nb_oos, mu_ref=mu_tr)

    p_ok = p <= cfg.p_oos_max
    hl_ok = cfg.half_life_min_bars <= hl <= cfg.half_life_max_bars
    dom_ok = len(cand.terms) < 2 or cand.max_share <= cfg.dominance_max
    drift_ok = d_oos <= cfg.drift_oos_max

    # Bouncer / trade count rule removed from passed criteria!
    passed = bool(p_ok and hl_ok and dom_ok and drift_ok)

    return TestResult(
        cand, stat, p, hl, float(100.0 * s_te.mean() / max(sigma, 1e-12)), _block_range(yv, X, split), move, b_tr, b_te,
        p_ok, hl_ok, dom_ok, passed, float(d_tr), float(d_oos), bool(drift_ok))


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
    icpt, slopes, e_tr = _fit(yv[:split], X[:split], cfg)
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
    full = adf_stat(_fit(yv[:split], X[:split], cfg)[2], cfg.adf_maxlag, False)[0]
    sy = yv[:split].std()
    trs = []
    for j, t in enumerate(terms):
        xs = X[:split, j]
        st_term, _ = adf_stat(xs, cfg.adf_maxlag, const=True)
        rest = tuple(x for x in terms if x != t)
        loo = np.nan
        if rest and any(kind(x) == "px" for x in rest):
            Xr = panel.matrix(rest)
            loo = adf_stat(_fit(yv[:split], Xr[:split], cfg)[2], cfg.adf_maxlag, False)[0] - full
        trs.append({
            "term": label(t), "kind": kind(t), "i1": panel.is_i1(t), "beta": cand.betas[t],
            "share": cand.shares[t], "gross_vs_y": abs(cand.betas[t]) * xs.std() / sy,
            "p_term_alone": adf_p(st_term, 1),
            "term_stationary": bool(kind(t) == "px" and adf_p(st_term, 1) < cfg.term_stationary_p),
            "huge_beta": bool(kind(t) == "px" and abs(cand.betas[t]) > cfg.beta_abs_max),
            "loo_d_adf": loo,
        })
    tt = pd.DataFrame(trs)
    sig_tr = float(_fit(yv[:split], X[:split], cfg)[2].std())
    tt["drift_pct_train_spread_rms"] = drift_by_term(
        yv[:split], X[:split], np.array([cand.betas[t] for t in terms]), sig_tr, cfg.drift_blocks)

    bn = pd.DataFrame({"train": res.bounces_train, "oos": res.bounces_oos})
    bm = panel.bar_minutes
    bn.loc["dur_median_hours"] = bn.loc["dur_median"] * bm / 60
    bn.loc["dur_p90_hours"] = bn.loc["dur_p90"] * bm / 60
    return Analysis(res, tt, ab, bn)


# --------------------------------------------------------------------------- beta-drift impact (dBeta * underlying)
def beta_drift_impact(panel: AssetPanel, cand: Candidate, cfg: DiscoverConfig = DiscoverConfig(),
                      split: int | None = None, window: int | None = None, stride: int | None = None) -> pd.DataFrame:
    """Rolling beta-drift impact as percent of typical TRAIN spread RMS.

        impact_j,t = 100 * (beta_j,t - beta_ref,j) * (x_j,t - mu_ref,j) / spread_rms_train
        total_t    = sum_j impact_j,t

    The denominator is the RMS deviation of the centred TRAIN residuals and remains fixed in OOS.
    Thus total_t = +100 means re-fitting beta would shift the spread by an amount equal to one
    typical training-spread RMS. This is a percentage of spread scale, not price or instantaneous
    spread level. Re-fits are causal: each trailing window uses only bars <= t.
    """
    split = panel.split_index(cfg.train_frac) if split is None else split
    window = cfg.drift_window if window is None else window
    stride = cfg.drift_stride if stride is None else stride
    yv, X = panel.series(px(cand.y)), panel.matrix(cand.terms)
    window = int(min(window, max(50, split // 2)))
    method = "stable" if cfg.fit_method == "kalman" else cfg.fit_method       # python-loop Kalman is too slow to re-run per window
    kw = {} if cfg.fit_method == "kalman" else cfg.fit_kw
    _, ref, e_tr = _fit(yv[:split], X[:split], cfg)
    mu, sigma = X[:split].mean(0), float(e_tr.std())

    T, p = X.shape
    Bt = np.full((T, p), np.nan)
    for t in range(window - 1, T, max(1, stride)):
        Bt[t] = fit(yv[t - window + 1:t + 1], X[t - window + 1:t + 1], method, **kw)[1]
    Bt = pd.DataFrame(Bt).ffill().to_numpy()
    imp = 100.0 * (Bt - ref) * (X - mu) / max(sigma, 1e-12)
    out = pd.DataFrame(imp, index=panel.logp.index, columns=[label(t) for t in cand.terms])
    out["total"] = imp.sum(1)
    out.attrs.update(split=split, sigma=sigma, window=window, stride=stride)
    return out


def drift_summary(imp: pd.DataFrame, spread: pd.Series | None = None) -> pd.DataFrame:
    """Per-term and total RMS / 95th percentile / max impact, in % of TRAIN spread RMS.

    `imp` is already normalised to percentages with a frozen training denominator, so the
    summary must not multiply by 100 again. `spread` is accepted for call compatibility.
    """
    split = imp.attrs["split"]
    rows = []
    for col in imp.columns:
        for name, part in (("train", imp[col].iloc[:split]), ("oos", imp[col].iloc[split:])):
            v = part.dropna().to_numpy()
            if not len(v):
                continue
            rows.append({
                "term": col,
                "sample": name,
                "rms_pct_train_spread_rms": float(np.sqrt((v * v).mean())),
                "p95_pct_train_spread_rms": float(np.quantile(np.abs(v), 0.95)),
                "max_pct_train_spread_rms": float(np.abs(v).max()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plotting
def plot_spread(panel: AssetPanel, res: TestResult, cfg: DiscoverConfig = DiscoverConfig(), split: int | None = None,
                show_betas: bool = False, show: bool = True):
    """Spread, z-score and beta-drift impact as % of typical TRAIN spread RMS (bottom). Returns (fig, impact, summary).
    show_betas=True adds a panel with the raw rolling betas for comparison (not comparable across terms; for reference)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return None

    split = panel.split_index(cfg.train_frac) if split is None else split
    cand = res.candidate
    yv, X = panel.series(px(cand.y)), panel.matrix(cand.terms)
    icpt, slopes, e_tr, s_te, sigma, _ = _spread_parts(panel, cand, split, cfg)
    times = panel.logp.index
    z_tr, z_te = _z(e_tr, s_te, sigma, cfg)
    imp = beta_drift_impact(panel, cand, cfg, split)
    static = pd.Series(np.r_[e_tr, s_te], index=times)
    summ = drift_summary(imp, static)

    n = 4 if show_betas else 3
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.6 * n + 1), sharex=True)
    ax1, ax2, ax3 = axes[:3]
    ax1.plot(times[:split], e_tr, label="Train Spread", color="#1f77b4", alpha=0.8)
    ax1.plot(times[split:], s_te, label="OOS Spread", color="#ff7f0e", alpha=0.8)
    ax1.axvline(times[split], color="red", linestyle="--", label="Train/OOS Split")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Spread (Raw Units)")
    ax1.set_title(f"StatArb Spread & Beta-Drift Impact [{cfg.fit_method}]: {cand.y} vs {' + '.join(label(t) for t in cand.terms)}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(times[:split], z_tr, label="Train Z-Score", color="#1f77b4", alpha=0.8)
    ax2.plot(times[split:], z_te, label="OOS Z-Score", color="#ff7f0e", alpha=0.8)
    ax2.axvline(times[split], color="red", linestyle="--")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(cfg.z_entry, color="green", linestyle="--", alpha=0.7, label=f"+{cfg.z_entry} Entry")
    ax2.axhline(-cfg.z_entry, color="green", linestyle="--", alpha=0.7, label=f"-{cfg.z_entry} Entry")
    ax2.axhline(cfg.z_exit, color="purple", linestyle=":", alpha=0.7, label=f"Exit ({cfg.z_exit})")
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    for col in imp.columns[:-1]:
        ax3.plot(times, imp[col], label=f"{col}", alpha=0.7, linewidth=1)
    ax3.plot(times, imp["total"], color="black", linewidth=1.8, label="TOTAL dBeta*(x-mu)")
    ax3.axhspan(-cfg.drift_max, cfg.drift_max, color="green", alpha=0.08,
                label=f"+/- {cfg.drift_max:g}% train-spread RMS cap")
    ax3.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax3.axvline(times[split], color="red", linestyle="--")
    ax3.set_ylabel("Beta-drift impact\n(% of train-spread RMS)")
    ax3.legend(loc="upper left", ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    if show_betas:
        ax4 = axes[3]
        _, ref, _ = _fit(yv[:split], X[:split], cfg)
        w = imp.attrs["window"]
        method = "stable" if cfg.fit_method == "kalman" else cfg.fit_method
        Bt = np.full(X.shape, np.nan)
        for t in range(w - 1, len(yv), max(1, imp.attrs["stride"])):
            Bt[t] = fit(yv[t - w + 1:t + 1], X[t - w + 1:t + 1], method, **({} if cfg.fit_method == "kalman" else cfg.fit_kw))[1]
        Bt = pd.DataFrame(Bt, index=times).ffill()
        for j, t_name in enumerate(cand.terms):
            ax4.plot(times, Bt[j], label=f"Rolling Beta ({label(t_name)})", alpha=0.8)
            ax4.axhline(ref[j], linestyle=":", alpha=0.5)
        ax4.axvline(times[split], color="red", linestyle="--")
        ax4.set_ylabel("Beta Value")
        ax4.legend(loc="upper left", ncol=2, fontsize=8)
        ax4.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, imp, summ


# --------------------------------------------------------------------------- run
def run(panel: AssetPanel, y: str, seed_x: Sequence[str] | None = None,
        cfg: DiscoverConfig = DiscoverConfig()):
    split = panel.split_index(cfg.train_frac)
    ranked = create_pair(panel.iloc(0, split), y, seed_x, cfg)
    results = []
    for c in ranked:
        # A feature can be well-covered in train but missing in OOS. Do not let one such
        # candidate crash the whole run or silently evaluate a partial sample.
        X = panel.matrix(c.terms)
        if not np.isfinite(X).all():
            continue
        results.append(test_pair(panel, c, cfg, split))
    table = pd.DataFrame([{
        "terms": " + ".join(label(t) for t in r.candidate.terms),
        "p_eg_train": r.candidate.p_eg, "hl_train": r.candidate.half_life,
        "p_oos": r.p_oos, "hl_oos": r.hl_oos,
        "mean_shift_pct_train_spread_rms": r.mean_shift_pct_train_spread_rms,
        "max_share": r.candidate.max_share, "beta_blk_rng": r.beta_block_range,
        "max_beta_move": r.max_beta_move,
        "drift_train_pct_spread_rms": r.drift_train, "drift_oos_pct_spread_rms": r.drift_oos,
        "p_ok": r.p_ok, "hl_ok": r.hl_ok, "dom_ok": r.dom_ok, "drift_ok": r.drift_ok, "passed": r.passed,
    } for r in results])
    return results, table


if __name__ == "__main__":
    SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
    Y, SEED_X = "ETH/USDT", []  # empty = no forced hedge leg; search alternatives across the universe
    cfg = DiscoverConfig()
    panel = get_assets(SYMBOLS, start="2024-01-01", end="2024-01-31", bar="5min", market="um", cfg=cfg)
    results, table = run(panel, Y, SEED_X, cfg)
    pd.set_option("display.width", 250)
    print(table.to_string())
    if not results:
        raise SystemExit("No candidates had complete train+OOS data. Reduce the feature pool or repair data coverage.")
    best = next((r for r in results if r.passed), results[0])
    a = analyze(panel, best.candidate, cfg)
    print("\n== terms ==\n", a.terms.to_string(), "\n\n== ablation ==\n", a.ablation.to_string(), "\n\n== bounces ==\n", a.bounces.to_string())
    
    # Plot the spread, z-score, and the beta-drift impact (dBeta * underlying) chart
    fig, imp, summ = plot_spread(panel, best, cfg)
    print("\n== beta-drift impact (% of typical train-spread RMS) ==\n", summ.to_string())
    print("\nNote: mean_shift_pct_train_spread_rms is the signed OOS mean shift as % of typical train-spread RMS; diagnostic only, not a pass gate.")