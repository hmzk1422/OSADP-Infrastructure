"""Leakage-controlled feature-information study for triple-barrier net-R outcomes.

Question
--------
Which feature metric measured on DEV best ranks features by their *future* information
about realized net triple-barrier R, and which metrics identify incremental model value?

Timeline for every origin
-------------------------
* GP programs are fit only on bars before gp_end, using labels that resolve before gp_end.
* Feature deduplication is fit only on the GP-fit segment, then the kept columns are
  applied unchanged to DEV and FINAL.
* Supervised walk-forward models train only on rows >= gp_end and strictly before each
  validation chunk minus the label horizon purge.
* DEV stops early enough that its barrier labels cannot cross the FINAL boundary.
* Every origin is truncated before labels are rebuilt/invalidated; labels from data beyond
  an origin's endpoint are never reused.
* FINAL diagnostics are pre-specified. Do not tune parameters or repeatedly change metrics
  based on FINAL results; use the next unseen period for a new confirmation.

Run in the project package context so the relative imports below resolve. For a quick smoke
run set TB_STUDY_QUICK=1. gplearn is required when GP_LEVELS uses GP; LightGBM and SHAP are
optional. This script is syntax-checked here, but its project-local market-data and barrier
primitives must be available in your repository for an end-to-end run.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, replace
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from scipy.special import digamma
from scipy.stats import norm, rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ...exchange.loader import load_L1
from ...strategy.moe.outcome import models as bm
from ...strategy.moe.outcome.triple_barrier import BarrierData, forward_excursions, simulate_barriers

try:
    from gplearn.genetic import SymbolicTransformer
    HAS_GPLEARN = True
except Exception:  # pragma: no cover
    HAS_GPLEARN = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False
try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

# ============================================================================
# Configuration
# ============================================================================
QUICK = os.environ.get("TB_STUDY_QUICK", "").strip().lower() in {"1", "true", "yes", "on"}
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
SINCE = "2025-01-01T00:00:00Z"
UNTIL = "2025-06-01T00:00:00Z"

HORIZON = 30
TP_SIZE = 1.0
BARRIER_SIZE = 1.0
RR_WINDOW = 60
SIDE = 1
BLOCK_LEN = 288
GP_FIT_FRAC = 0.35
FINAL_FRAC = 0.25
FEE_BPS_PER_SIDE = 2.0
SEED = 7
MAX_ROWS = 3000 if QUICK else 15000
ABL_ROWS = 1200 if QUICK else 6000
WF_FOLDS = 3 if QUICK else 4
WARM_FRAC = 0.25
MIN_LEAF = 60 if QUICK else 200
ABL_OBJECTIVE = "corr"       # "corr", "topq_expectancy", "r2"
TOPQ = 0.20
N_SUBSETS = 6 if QUICK else 60
N_BOOT = 10 if QUICK else 100
DEDUPE_THR = 0.98
LOO_SEEDS = (SEED,) if QUICK else (SEED, SEED + 1, SEED + 2)
# These are prefix cutoffs, not independent origins; their FINAL windows overlap.
END_FRACS = (1.0,) if QUICK else (1.0, 0.85, 0.70)
ALL_METRICS = ("MI", "IC", "IR", "SHAP", "CMI_gauss", "CMI_knn_top3", "PFI", "GAIN")
GP_LEVELS = {
    "off": None,
    "light": dict(population_size=100 if QUICK else 300, generations=2 if QUICK else 4,
                  n_components=4, parsimony=0.01),
    "medium": dict(population_size=200 if QUICK else 800, generations=3 if QUICK else 8,
                    n_components=8, parsimony=0.002),
    "heavy": dict(population_size=300 if QUICK else 2000, generations=4 if QUICK else 16,
                   n_components=12, parsimony=0.0005),
}


def section(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


# ============================================================================
# Data preparation and triple-barrier labels
# ============================================================================
def prep_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize loader columns; use a timestamp column as index if present."""
    d = raw.copy()
    d.columns = [str(c).lower() for c in d.columns]
    if not isinstance(d.index, pd.DatetimeIndex):
        ts_col = next((c for c in ("timestamp", "datetime", "date", "time") if c in d.columns), None)
        if ts_col is not None:
            d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
            d = d.set_index(ts_col)
    aliases = {
        "price_open": ("price_open", "open", "o"),
        "price_high": ("price_high", "high", "h"),
        "price_low": ("price_low", "low", "l"),
        "price_close": ("price_close", "close", "c"),
        "volume": ("volume", "vol", "v"),
    }
    rename = {}
    for target, candidates in aliases.items():
        source = next((c for c in candidates if c in d.columns), None)
        if source is None:
            raise KeyError(f"load_L1 frame has no column for {target!r}; got {list(d.columns)}")
        rename[source] = target
    d = d.rename(columns=rename)[list(aliases)].apply(pd.to_numeric, errors="coerce")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def build_trades(d: pd.DataFrame, horizon: int = HORIZON, tp_size: float = TP_SIZE,
                 barrier_size: float = BARRIER_SIZE, side: int = SIDE) -> pd.DataFrame:
    """Use the repository's barrier engine; output one candidate per bar.

    `r` is net R. Cost conversion follows the original study's engine convention: if the
    returned SL width looks like a price amount rather than a price fraction, normalize by
    entry price before converting round-trip fees to R. Inspect this assumption against the
    implementation of `bm.loss_function` / `simulate_barriers` in the target repository.
    """
    n = len(d)
    if n == 0:
        raise ValueError("Cannot label an empty OHLCV frame")
    price = d["price_close"].to_numpy(float)
    hi, lo = d["price_high"].to_numpy(float), d["price_low"].to_numpy(float)
    idx = np.arange(n)
    bad = ~(np.isfinite(hi) & np.isfinite(lo) & (hi > 0) & (lo > 0) & np.isfinite(price) & (price > 0))
    bad_cum = np.concatenate(([0], np.cumsum(bad.astype(np.int64))))
    path_end = np.minimum(idx + horizon + 1, n)
    path_ok = (idx + horizon < n) & ((bad_cum[path_end] - bad_cum[idx + 1]) == 0)
    bd = BarrierData(
        df=d, X=np.zeros((n, 1)), kept_cols=[], price=price, high=hi, low=lo,
        side=np.full(n, side), horizon=horizon, path_ok=path_ok,
        train=np.ones(n, bool), test=path_ok, price_col="price_close",
    )
    unit = bm.rr_unit(d, price_col="price_close", window=RR_WINDOW, vol_col=None,
                      sqrt_horizon=float(np.sqrt(horizon)))
    sl = bm.loss_function(unit, barrier_size)
    tp = bm.tp_function(sl, tp_size)
    sim = simulate_barriers(bd, tp, sl)
    required = {"valid", "sl_width", "r", "outcome"}
    missing = required - set(sim.columns)
    if missing:
        raise KeyError(f"simulate_barriers output lacks {sorted(missing)}; got {list(sim.columns)}")
    if len(sim) != n:
        raise ValueError(f"simulate_barriers returned {len(sim)} rows for {n} input bars")
    # Force positional alignment to OHLCV bars even if the engine returns a RangeIndex.
    sim = sim.copy()
    sim.index = d.index
    mfe, mae = forward_excursions(bd)
    sw = sim["sl_width"].to_numpy(float).copy()
    med_sw = np.nanmedian(sw) if np.isfinite(sw).any() else np.nan
    if np.isfinite(med_sw) and med_sw > 0.2:
        print(f"  note: median sl_width={med_sw:.4g}; treating it as price units and dividing by entry price")
        sw = sw / price
    with np.errstate(invalid="ignore", divide="ignore"):
        cost_r = (2.0 * FEE_BPS_PER_SIDE * 1e-4) / sw
        out = sim.copy()
        out["r_gross"] = sim["r"].to_numpy(float)
        out["cost_r"] = cost_r
        out["r"] = out["r_gross"].to_numpy(float) - cost_r
        out["win"] = (sim["outcome"].to_numpy() == 1).astype(int)
        out["mfe_r"] = np.asarray(mfe, float) / np.asarray(unit, float)
        out["mae_r"] = np.asarray(mae, float) / np.asarray(unit, float)
        out["fwd_bps"] = SIDE * (pd.Series(price, index=d.index).shift(-horizon) / price - 1.0) * 1e4
    # Engine validity is authoritative, and also require a complete future path.
    out["valid"] = out["valid"].fillna(False).astype(bool).to_numpy() & path_ok
    return out


def invalidate_labels_past_origin(trades: pd.DataFrame, n_origin: int, horizon: int = HORIZON) -> pd.DataFrame:
    """Drop labels whose possible barrier path extends beyond a truncated origin endpoint."""
    t = trades.iloc[:n_origin].copy()
    first_uncertain = max(0, n_origin - horizon)
    if first_uncertain < len(t):
        t.iloc[first_uncertain:, t.columns.get_loc("valid")] = False
    return t


# ============================================================================
# Causal features / GP feature creation
# ============================================================================
def base_features(d: pd.DataFrame) -> pd.DataFrame:
    """OHLCV features whose row t uses only bars at or before t."""
    c, h, l, o, v = (d[k] for k in ("price_close", "price_high", "price_low", "price_open", "volume"))
    ret = c.pct_change(fill_method=None)
    f: dict[str, pd.Series] = {}
    for k in (1, 3, 6, 12, 24, 48, 96):
        f[f"ret_{k}"] = c.pct_change(k, fill_method=None)
    for w in (12, 48, 144):
        f[f"vol_{w}"] = ret.rolling(w, min_periods=w).std()
    f["vol_ratio"] = f["vol_12"] / f["vol_144"].replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    f["hl_range"] = (h - l) / c
    f["close_loc"] = (c - l) / rng
    f["body"] = (c - o) / c
    f["body_frac"] = (c - o) / rng
    for w in (24, 96):
        f[f"z_{w}"] = (c - c.rolling(w, min_periods=w).mean()) / c.rolling(w, min_periods=w).std().replace(0, np.nan)
    gain = ret.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-ret.clip(upper=0)).rolling(14, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["vol_z"] = (v - v.rolling(96, min_periods=96).mean()) / v.rolling(96, min_periods=96).std().replace(0, np.nan)
    f["vol_chg_12"] = np.log(v.where(v > 0)).diff(12)
    f["dist_hi_48"] = c / h.rolling(48, min_periods=48).max() - 1
    f["dist_lo_48"] = c / l.rolling(48, min_periods=48).min() - 1
    f["skew_48"] = ret.rolling(48, min_periods=48).skew()
    f["ret_acc"] = f["ret_12"] - f["ret_12"].shift(12)
    f["eff_48"] = (c - c.shift(48)).abs() / c.diff().abs().rolling(48, min_periods=48).sum().replace(0, np.nan)
    f["flow_12"] = f["body_frac"].rolling(12, min_periods=12).mean()
    if isinstance(d.index, pd.DatetimeIndex):
        hod = d.index.hour + d.index.minute / 60.0
        f["hod_sin"] = pd.Series(np.sin(2 * np.pi * hod / 24), index=d.index)
        f["hod_cos"] = pd.Series(np.cos(2 * np.pi * hod / 24), index=d.index)
    return pd.DataFrame(f, index=d.index).replace([np.inf, -np.inf], np.nan)


def rank_gauss(a) -> np.ndarray:
    """Within-sample Gaussian-copula transformation, for descriptive estimators."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a[:, None]
    if len(a) == 0:
        return a.copy()
    r = rankdata(a, axis=0, method="average")
    return norm.ppf(np.clip((r - 0.5) / len(a), 0.5 / len(a), 1 - 0.5 / len(a)))


class RankGauss:
    """Fit empirical rank transform on DEV, apply unchanged to FINAL."""
    def fit(self, a):
        a = np.asarray(a, float)
        if a.ndim != 2 or len(a) == 0:
            raise ValueError("RankGauss.fit expects a non-empty 2-D array")
        self.ref = np.sort(a, axis=0)
        return self

    def transform(self, a):
        a = np.asarray(a, float)
        if a.ndim != 2 or a.shape[1] != self.ref.shape[1]:
            raise ValueError("RankGauss.transform feature dimension differs from fit")
        n = len(self.ref)
        out = np.empty_like(a, dtype=float)
        for j in range(a.shape[1]):
            lo = np.searchsorted(self.ref[:, j], a[:, j], side="left")
            hi = np.searchsorted(self.ref[:, j], a[:, j], side="right")
            pct = ((lo + hi) / 2.0) / n
            out[:, j] = norm.ppf(np.clip(pct, 0.5 / n, 1 - 0.5 / n))
        return out


def dedupe_features(X: pd.DataFrame, fit_until: int, thr: float = DEDUPE_THR) -> tuple[pd.DataFrame, dict]:
    """Fit feature validity/correlation filtering only on the early fit slice.

    Constant and mostly-missing features are removed first; they must not be treated as
    perfectly correlated with every other feature just because correlation is undefined.
    The retained column list is then applied unchanged to all later bars.
    """
    fit = X.iloc[:fit_until]
    raw = fit.to_numpy(float)
    n_fit = len(raw)
    min_finite = max(50, int(0.80 * n_fit))
    valid_cols: list[int] = []
    dropped: dict[str, str] = {}
    col_std: dict[int, float] = {}
    for j, name in enumerate(X.columns):
        col = raw[:, j]
        finite = np.isfinite(col)
        if int(finite.sum()) < min_finite:
            dropped[str(name)] = "<insufficient_finite_training_values>"
            continue
        std = float(np.std(col[finite]))
        if not np.isfinite(std) or std < 1e-10:
            dropped[str(name)] = "<constant_training_feature>"
            continue
        valid_cols.append(j)
        col_std[j] = std
    if not valid_cols:
        raise ValueError("No nonconstant, sufficiently finite features remain in the GP-fit period")
    # Compute the correlation matrix only on fit rows complete for the eligible columns.
    candidate = raw[:, valid_cols]
    complete = np.flatnonzero(np.isfinite(candidate).all(axis=1))
    keep = list(valid_cols)
    if len(complete) >= 50 and len(valid_cols) > 1:
        sample = complete[::max(1, int(np.ceil(len(complete) / 5000)))]
        A = rank_gauss(candidate[sample])
        C = np.corrcoef(A.T)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.abs(C)
        np.fill_diagonal(C, 1.0)
        kept_local: list[int] = []
        local_drop: dict[str, str] = {}
        for j, original_j in enumerate(valid_cols):
            twin_local = next((k for k in kept_local if C[j, k] >= thr), None)
            if twin_local is None:
                kept_local.append(j)
            else:
                local_drop[str(X.columns[original_j])] = str(X.columns[valid_cols[twin_local]])
        keep = [valid_cols[j] for j in kept_local]
        dropped.update(local_drop)
    if not keep:
        raise ValueError("Correlation deduplication removed every feature")
    return X.iloc[:, keep], dropped


def create_features(d: pd.DataFrame, trades: pd.DataFrame, gp_level: str = "heavy",
                    seed: int = SEED, fit_until: int | None = None,
                    dedupe_thr: float | None = DEDUPE_THR):
    """Create causal base + GP features, and fit both GP and dedupe only before fit_until."""
    base = base_features(d)
    n = len(d)
    fit_until = int(GP_FIT_FRAC * n) if fit_until is None else int(fit_until)
    if not 0 < fit_until <= n:
        raise ValueError(f"fit_until must be in [1, {n}], got {fit_until}")
    spec = GP_LEVELS[gp_level]
    X = base.copy()
    progs: dict[str, str] = {}
    if spec is not None:
        if not HAS_GPLEARN:
            raise ImportError("gplearn is required for GP features (pip install gplearn)")
        clean = base.fillna(0.0)
        eligible = (
            base.notna().all(axis=1).to_numpy()
            & trades["valid"].to_numpy(bool)
            & np.isfinite(trades["r"].to_numpy(float))
            & (np.arange(n) < fit_until - HORIZON)
        )
        rows = np.flatnonzero(eligible)
        if len(rows) < 100:
            raise ValueError(f"Only {len(rows)} rows available to fit GP before cutoff {fit_until}; need >=100")
        gp = SymbolicTransformer(
            population_size=spec["population_size"], generations=spec["generations"],
            hall_of_fame=max(spec["n_components"] * 3, 20), n_components=spec["n_components"],
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min"),
            metric="spearman", parsimony_coefficient=spec["parsimony"], max_samples=1.0,
            tournament_size=20, init_depth=(2, 5), random_state=seed, n_jobs=1, verbose=0,
        )
        gp.fit(clean.to_numpy(float)[rows], trades["r"].to_numpy(float)[rows])
        g = np.asarray(gp.transform(clean.to_numpy(float)), dtype=float)
        if g.ndim == 1:
            g = g[:, None]
        cols = [f"gp_{i}" for i in range(g.shape[1])]
        gdf = pd.DataFrame(np.where(np.isfinite(g), g, 0.0), index=d.index, columns=cols)
        progs = {c: str(p) for c, p in zip(cols, gp._best_programs)}
        X = pd.concat([base, gdf], axis=1)
    dropped: dict[str, str] = {}
    if dedupe_thr is not None:
        X, dropped = dedupe_features(X, fit_until=fit_until, thr=dedupe_thr)
    X.attrs["dropped"] = dropped
    return X, {k: v for k, v in progs.items() if k in X.columns}


# ============================================================================
# Study windows and chronological validation
# ============================================================================
@dataclass
class Study:
    X: pd.DataFrame
    y: pd.Series           # realized NET R per trade
    win: pd.Series
    fwd: pd.Series          # exact-horizon forward return, bps
    pos: np.ndarray         # original bar positions
    eval_lo: int = 0        # model-fold training cannot include positions below this bar

    def __len__(self):
        return len(self.y)

    def ev(self) -> "Study":
        rows = np.flatnonzero(self.pos >= self.eval_lo)
        return self if len(rows) == len(self) else take(self, rows)


def make_study(X: pd.DataFrame, trades: pd.DataFrame, max_rows: int = MAX_ROWS,
               lo: int = 0, hi: int | None = None, eval_lo: int | None = None) -> Study:
    """Finite candidate rows in [lo, hi); chronological stride subsampling if needed."""
    n = len(X)
    hi = n if hi is None else min(int(hi), n)
    lo = max(0, int(lo))
    if hi <= lo:
        raise ValueError(f"Empty requested Study interval [{lo}, {hi})")
    vals = X.to_numpy(float)
    r = trades["r"].to_numpy(float)
    fwd = trades["fwd_bps"].to_numpy(float)
    valid = trades["valid"].to_numpy(bool)
    ok = valid & np.isfinite(vals).all(axis=1) & np.isfinite(r) & np.isfinite(fwd)
    ok &= (np.arange(n) >= lo) & (np.arange(n) < hi)
    pos = np.flatnonzero(ok)
    if len(pos) > max_rows:
        pos = pos[::int(np.ceil(len(pos) / max_rows))]
    return Study(X=X.iloc[pos].reset_index(drop=True), y=pd.Series(r[pos]),
                 win=pd.Series(trades["win"].to_numpy()[pos]),
                 fwd=pd.Series(fwd[pos]), pos=pos,
                 eval_lo=lo if eval_lo is None else int(eval_lo))


def take(st: Study, rows) -> Study:
    rows = np.asarray(rows, dtype=int)
    return Study(st.X.iloc[rows].reset_index(drop=True), st.y.iloc[rows].reset_index(drop=True),
                 st.win.iloc[rows].reset_index(drop=True), st.fwd.iloc[rows].reset_index(drop=True),
                 st.pos[rows], st.eval_lo)


def timeline(n: int) -> tuple[int, int]:
    """Absolute bar cutoffs (GP end, FINAL start)."""
    return int(GP_FIT_FRAC * n), int((1.0 - FINAL_FRAC) * n)


def warm_cut(pos: np.ndarray, eval_lo: int, warm: float = WARM_FRAC) -> int:
    ev = np.flatnonzero(pos >= eval_lo)
    if not len(ev):
        return eval_lo
    return int(pos[ev[min(int(len(ev) * warm), len(ev) - 1)]])


def wf_splits(pos: np.ndarray, eval_lo: int, k: int = WF_FOLDS,
              purge: int = HORIZON, warm: float = WARM_FRAC) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding chronological folds; purge is in original bar positions, not sampled rows."""
    ev = np.flatnonzero(pos >= eval_lo)
    ev = ev[int(len(ev) * warm):]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for te in np.array_split(ev, k):
        if not len(te):
            continue
        boundary = int(pos[te[0]]) - purge
        tr = np.flatnonzero((pos >= eval_lo) & (pos < boundary))
        if len(tr) >= 200:
            out.append((tr, te))
    return out


# ============================================================================
# Models and utility measurements
# ============================================================================
def make_model(small: bool = False, seed: int = SEED):
    if HAS_LGBM:
        return LGBMRegressor(
            n_estimators=80 if small else 200,
            learning_rate=0.08 if small else 0.05,
            num_leaves=7 if small else 15,
            min_child_samples=MIN_LEAF,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=80 if small else 200, learning_rate=0.08 if small else 0.05,
        max_leaf_nodes=7 if small else 15, min_samples_leaf=MIN_LEAF, random_state=seed,
    )


def make_r_model():
    """Low-capacity direct-R model used in the GP-overfit diagnostic."""
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


def wf_predict(A: np.ndarray, y: np.ndarray, pos: np.ndarray, eval_lo: int,
               folds: int = WF_FOLDS, small: bool = True, seed: int = SEED) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for tr, te in wf_splits(pos, eval_lo, folds):
        model = make_model(small=small, seed=seed)
        model.fit(A[tr], y[tr])
        pred[te] = model.predict(A[te])
    return pred


def utility(pred: np.ndarray, y: np.ndarray) -> dict:
    """Score predictions. `topq_expectancy` is a batch ranking diagnostic, not a live threshold policy."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 20:
        return dict(topq_expectancy=np.nan, corr=np.nan, r2=np.nan, n=int(m.sum()))
    pp, yy = pred[m], y[m]
    threshold = np.quantile(pp, 1 - TOPQ)
    return dict(topq_expectancy=float(yy[pp >= threshold].mean()), corr=_spear(pp, yy),
                r2=float(r2_score(yy, pp)), n=int(m.sum()))


def _spear(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return np.nan
    return float(spearmanr(a[m], b[m]).statistic)


def _sub(idx: np.ndarray, m: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    return idx[::max(1, int(np.ceil(len(idx) / max(int(m), 1))))]


def _block_rows(pos: np.ndarray) -> int:
    pos = np.asarray(pos, float)
    step = max(float(np.median(np.diff(pos))), 1.0) if len(pos) > 2 else 1.0
    return max(5, int(BLOCK_LEN / step))


def _block_idx(n: int, b: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = max(1, min(int(b), n))
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, n_blocks)
    return np.concatenate([np.arange(s, s + b) for s in starts])[:n]


def boot_ci(stat_fn, n: int, b: int, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Moving-block bootstrap interval; resampling preserves local serial dependence approximately."""
    if n < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = [stat_fn(_block_idx(n, b, rng)) for _ in range(n_boot)]
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


# ============================================================================
# Feature-importance estimators
# ============================================================================
def permutation_shapley(predict, Xe: np.ndarray, Xb: np.ndarray,
                         n_perm: int = 16, seed: int = SEED) -> np.ndarray:
    """Monte-Carlo permutation SHAP approximation, using a DEV/background sample."""
    rng = np.random.default_rng(seed)
    n, n_features = Xe.shape
    phi = np.zeros((n, n_features), dtype=float)
    for _ in range(n_perm):
        cur = Xb[rng.integers(0, len(Xb), n)].copy()
        prev = np.asarray(predict(cur), float)
        for j in rng.permutation(n_features):
            cur[:, j] = Xe[:, j]
            new = np.asarray(predict(cur), float)
            phi[:, j] += new - prev
            prev = new
    return phi / n_perm


def shap_values(model, Xe: np.ndarray, Xb: np.ndarray):
    """Exact TreeSHAP for compatible LightGBM models; otherwise permutation SHAP."""
    if HAS_SHAP and HAS_LGBM and isinstance(model, LGBMRegressor):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xe)
        base = float(np.ravel(explainer.expected_value)[0])
        return np.asarray(values), base
    values = permutation_shapley(model.predict, Xe, Xb)
    return values, float(np.asarray(model.predict(Xb), float).mean())


def pfi(model, A: np.ndarray, y: np.ndarray, n_repeats: int = 3, seed: int = SEED) -> np.ndarray:
    """Ordinary held-out permutation importance, measured as increase in MSE."""
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    if len(A) < 10:
        return np.full(A.shape[1], np.nan)
    rng = np.random.default_rng(seed)
    base = mean_squared_error(y, model.predict(A))
    out = np.zeros(A.shape[1], dtype=float)
    Ap = A.copy()
    for j in range(A.shape[1]):
        original = Ap[:, j].copy()
        for _ in range(n_repeats):
            Ap[:, j] = rng.permutation(original)
            out[j] += mean_squared_error(y, model.predict(Ap)) - base
        Ap[:, j] = original
    return out / n_repeats


def _gain(model, n_features: int) -> np.ndarray:
    if HAS_LGBM and isinstance(model, LGBMRegressor):
        gain = model.booster_.feature_importance(importance_type="gain").astype(float)
        return gain / max(float(gain.sum()), 1e-12)
    return np.full(n_features, np.nan)


def model_importances(st: Study, mode: str = "wf", n_shap: int = 1000,
                      n_pfi: int = 2500, folds: int = WF_FOLDS) -> pd.DataFrame:
    """Average PFI/|SHAP|/GAIN over walk-forward held-out chunks on DEV.

    `mode='insample'` is a deliberately leaky diagnostic only. It must never be used to
    make claims about generalization.
    """
    A, y = st.X.to_numpy(float), st.y.to_numpy(float)
    F = A.shape[1]
    all_rows = np.arange(len(A))
    splits = [(all_rows, all_rows)] if mode == "insample" else wf_splits(st.pos, st.eval_lo, folds)
    if not splits:
        return pd.DataFrame(np.nan, index=st.X.columns, columns=["PFI", "SHAP", "GAIN"])
    pf, sh, gn = [], [], []
    for fold_i, (tr, te) in enumerate(splits):
        if len(tr) < 200 or len(te) < 20:
            continue
        model = make_model(seed=SEED + fold_i).fit(A[tr], y[tr])
        tp, ts = _sub(te, n_pfi), _sub(te, n_shap)
        pf.append(pfi(model, A[tp], y[tp], seed=SEED + fold_i))
        background = A[_sub(tr, 100)]
        sv, _ = shap_values(model, A[ts], background)
        sh.append(np.abs(sv).mean(0))
        gn.append(_gain(model, F))
    if not pf:
        return pd.DataFrame(np.nan, index=st.X.columns, columns=["PFI", "SHAP", "GAIN"])
    return pd.DataFrame({"PFI": np.nanmean(np.stack(pf), axis=0),
                         "SHAP": np.nanmean(np.stack(sh), axis=0),
                         "GAIN": np.nanmean(np.stack(gn), axis=0)}, index=st.X.columns)


# ============================================================================
# Per-feature metrics
# ============================================================================
def metric_mi(X: pd.DataFrame, y: pd.Series, n: int = 6000) -> pd.Series:
    rows = _sub(np.arange(len(X)), n)
    mi = mutual_info_regression(X.to_numpy(float)[rows], np.asarray(y, float)[rows],
                                n_neighbors=5, random_state=SEED)
    return pd.Series(mi, index=X.columns, dtype=float)


def metric_ic(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    return X.rank(method="average").corrwith(pd.Series(np.asarray(y)).rank(method="average"))


def metric_ir(X: pd.DataFrame, y: pd.Series, pos: np.ndarray, min_rows: int = 30) -> pd.Series:
    """Absolute mean daily rank-IC / daily rank-IC standard deviation (a stability proxy)."""
    blocks = np.asarray(pos) // BLOCK_LEN
    ics = []
    for block in np.unique(blocks):
        mask = blocks == block
        if mask.sum() >= min_rows:
            ics.append(metric_ic(X.iloc[np.flatnonzero(mask)], pd.Series(np.asarray(y)[mask])))
    if len(ics) < 2:
        return pd.Series(np.nan, index=X.columns)
    frame = pd.DataFrame(ics)
    return frame.mean() / frame.std(ddof=0).replace(0, np.nan)


def gauss_cmi_all(Xg: np.ndarray, yg: np.ndarray, cols) -> pd.Series:
    """Per-feature Gaussian-copula I(X_j; R | all other features), using a ridged precision matrix."""
    Z = np.column_stack([np.asarray(Xg, float), np.asarray(yg, float).reshape(-1)])
    C = np.corrcoef(Z.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.linalg.pinv(C + 1e-4 * np.eye(len(C)))
    denom = np.sqrt(np.abs(np.diag(P)[:-1] * P[-1, -1]) + 1e-12)
    rho = -P[:-1, -1] / denom
    return pd.Series(-0.5 * np.log(1 - np.clip(rho, -0.999, 0.999) ** 2), index=cols)


def gauss_joint_mi(Xg: np.ndarray, yg: np.ndarray) -> float:
    """Gaussian-copula joint MI proxy based on multiple-correlation R^2."""
    Xg, yg = np.asarray(Xg, float), np.asarray(yg, float).reshape(-1)
    if Xg.shape[1] == 0:
        return 0.0
    A = np.column_stack([Xg, np.ones(len(Xg))])
    beta = np.linalg.lstsq(A, yg, rcond=None)[0]
    r2 = 1 - np.var(yg - A @ beta) / max(np.var(yg), 1e-12)
    return float(-0.5 * np.log(1 - np.clip(r2, 0, 0.999)))


def cmi_chain(Xg: np.ndarray, yg: np.ndarray, cols) -> pd.Series:
    """Greedy conditional Gaussian-information increments; order-dependent attribution."""
    Xg, yg = np.asarray(Xg, float), np.asarray(yg, float).reshape(-1)
    n, F = Xg.shape
    Xc = Xg - Xg.mean(0)
    y_center = yg - yg.mean()
    syy = max(float(y_center @ y_center), 1e-12)
    Q = np.zeros((n, 0))
    available, increments = list(range(F)), {}
    r2_prev = 0.0
    residual = y_center.copy()
    while available:
        Xr = Xc[:, available] - Q @ (Q.T @ Xc[:, available])
        nrm = (Xr ** 2).sum(0)
        gain = np.where(nrm > 1e-10, (Xr.T @ residual) ** 2 / np.maximum(nrm, 1e-12), 0.0)
        b = int(np.argmax(gain))
        j = available.pop(b)
        if nrm[b] > 1e-10:
            q = Xr[:, b] / np.sqrt(nrm[b])
            Q = np.column_stack([Q, q])
            residual = residual - q * (q @ residual)
        r2_new = 1 - float(residual @ residual) / syy
        increments[cols[j]] = -0.5 * np.log(max(1 - r2_new, 1e-9)) + 0.5 * np.log(max(1 - r2_prev, 1e-9))
        r2_prev = r2_new
    return pd.Series(increments).reindex(cols)


def cmi_knn(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None, k: int = 5) -> float:
    """Frenzel-Pompe/KSG-style kNN conditional MI estimator; inputs are 2-D."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    parts = [x, y] + ([np.asarray(z, float)] if z is not None else [])
    pts = np.hstack(parts)
    if len(pts) <= k + 2:
        return np.nan
    eps = np.maximum(cKDTree(pts).query(pts, k=k + 1, p=np.inf)[0][:, -1] - 1e-10, 0.0)

    def cnt(a):
        return np.asarray(cKDTree(a).query_ball_point(a, eps, p=np.inf, return_length=True)) - 1

    if z is None:
        return float(digamma(k) + digamma(len(x))
                     - np.mean(digamma(cnt(x) + 1) + digamma(cnt(y) + 1)))
    z = np.asarray(z, float)
    return float(digamma(k) - np.mean(digamma(cnt(np.hstack([x, z])) + 1)
                                     + digamma(cnt(np.hstack([y, z])) + 1)
                                     - digamma(cnt(z) + 1)))


def metric_cmi_knn_top3(Xg: np.ndarray, yg: np.ndarray, cols, m: int = 3,
                        n: int = 2500, seed: int = SEED) -> pd.Series:
    """I(feature; R | its 3 most-correlated peers), not conditioned on every feature."""
    rows = _sub(np.arange(len(Xg)), n)
    Xs = np.asarray(Xg, float)[rows]
    ys = np.asarray(yg, float)[rows].reshape(-1, 1)
    rng = np.random.default_rng(seed)
    ys = ys + 1e-5 * rng.normal(size=len(ys),)[:, None]
    C = np.abs(np.corrcoef(Xs.T))
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, -np.inf)
    out = []
    for j in range(Xs.shape[1]):
        peers = np.argsort(C[j])[::-1][:min(m, Xs.shape[1] - 1)]
        value = cmi_knn(Xs[:, [j]], ys, Xs[:, peers], k=5)
        out.append(max(value, 0.0) if np.isfinite(value) else np.nan)
    return pd.Series(out, index=cols, dtype=float)


def score_all(st: Study, metrics=ALL_METRICS, mode: str = "wf") -> pd.DataFrame:
    """Rank feature information scores on the DEV evaluation portion only."""
    ev = st.ev()
    if len(ev) < 100:
        raise ValueError(f"Too few GP-unseen DEV rows to score metrics: {len(ev)}")
    out: dict[str, pd.Series] = {}
    cols = list(ev.X.columns)
    need_copula = bool({"CMI_gauss", "CMI_knn_top3"}.intersection(metrics))
    if need_copula:
        Xg = rank_gauss(ev.X.to_numpy(float))
        yg = rank_gauss(ev.y.to_numpy(float)[:, None])[:, 0]
    if "MI" in metrics:
        out["MI"] = metric_mi(ev.X, ev.y)
    if "IC" in metrics:
        out["IC"] = metric_ic(ev.X, ev.y).abs()
    if "IR" in metrics:
        out["IR"] = metric_ir(ev.X, ev.y, ev.pos).abs()
    if "CMI_gauss" in metrics:
        out["CMI_gauss"] = gauss_cmi_all(Xg, yg, cols)
    if "CMI_knn_top3" in metrics:
        out["CMI_knn_top3"] = metric_cmi_knn_top3(Xg, yg, cols)
    if {"PFI", "SHAP", "GAIN"}.intersection(metrics):
        imp = model_importances(st, mode=mode)
        for metric in ("PFI", "SHAP", "GAIN"):
            if metric in metrics:
                out[metric] = imp[metric]
    return pd.DataFrame(out, index=cols).replace([np.inf, -np.inf], np.nan)


def truth_table(st: Study, n_bins: int = 10) -> pd.DataFrame:
    """Marginal outcome truth using decile eta² and top-minus-bottom mean net R.

    Constant/near-constant features are explicitly assigned zero truth instead of being
    arbitrarily split by row order (which otherwise creates spurious decile effects).
    """
    y = st.y.to_numpy(float)
    overall = float(np.mean(y))
    sst = max(float(np.sum((y - overall) ** 2)), 1e-12)
    rows = {}
    for col in st.X.columns:
        x = st.X[col].to_numpy(float)
        if not np.isfinite(x).all() or np.nanmax(x) - np.nanmin(x) < 1e-12:
            rows[col] = dict(eta2=0.0, spread=0.0, abs_spread=0.0, n_bins=1)
            continue
        bins = pd.qcut(pd.Series(x), q=min(n_bins, len(np.unique(x))), labels=False, duplicates="drop")
        frame = pd.DataFrame({"bin": bins.to_numpy(), "y": y}).dropna()
        g = frame.groupby("bin", observed=True)["y"].agg(["mean", "size"])
        if len(g) < 2:
            rows[col] = dict(eta2=0.0, spread=0.0, abs_spread=0.0, n_bins=len(g))
            continue
        eta2 = float((g["size"] * (g["mean"] - overall) ** 2).sum() / sst)
        spread = float(g["mean"].iloc[-1] - g["mean"].iloc[0])
        rows[col] = dict(eta2=eta2, spread=spread, abs_spread=abs(spread), n_bins=len(g))
    return pd.DataFrame.from_dict(rows, orient="index")


def _topk_overlap(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    aa = set(a.dropna().nlargest(min(k, a.notna().sum())).index)
    bb = set(b.dropna().nlargest(min(k, b.notna().sum())).index)
    return len(aa & bb) / max(1, min(k, len(aa), len(bb)))


def vif(X: pd.DataFrame) -> pd.Series:
    Z = rank_gauss(X.to_numpy(float))
    C = np.corrcoef(Z.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(np.diag(np.linalg.pinv(C + 1e-6 * np.eye(C.shape[0]))), index=X.columns)


# ============================================================================
# Experiment 1: do dev information metrics rank future marginal expectancy?
# ============================================================================
def cross_test(dev: Study, final: Study, scores: pd.DataFrame,
               n_dropped: int = 0, n_boot: int = N_BOOT) -> dict:
    section("CROSS TEST: DEV metric ranking vs untouched FINAL outcome truth")
    ev = dev.ev()
    truth_dev, truth_final = truth_table(ev), truth_table(final)
    all_tables = pd.concat([
        scores,
        truth_dev["eta2"].rename("TRUTH_eta2_dev"),
        truth_final["eta2"].rename("TRUTH_eta2_FINAL"),
        truth_final["abs_spread"].rename("TRUTH_abs_spread_FINAL"),
    ], axis=1)
    corr = all_tables.corr(method="spearman")
    summary = pd.DataFrame({
        "rho_eta2_dev": {m: _spear(scores[m], truth_dev["eta2"]) for m in scores},
        "rho_spread_dev": {m: _spear(scores[m], truth_dev["abs_spread"]) for m in scores},
        "rho_eta2_FINAL": {m: _spear(scores[m], truth_final["eta2"]) for m in scores},
        "rho_spread_FINAL": {m: _spear(scores[m], truth_final["abs_spread"]) for m in scores},
        "top5_overlap_FINAL": {m: _topk_overlap(scores[m], truth_final["eta2"]) for m in scores},
    })
    print("\nDev score -> future truth. FINAL columns are the main ranking test:")
    print(summary.sort_values("rho_eta2_FINAL", ascending=False).to_string())
    print("\nMetric/truth correlation matrix:")
    print(corr.round(2).to_string())

    persistence = _spear(truth_dev["eta2"], truth_final["eta2"])
    parts = [take(ev, p) for p in np.array_split(np.arange(len(ev)), 3) if len(p)]
    ic_parts = [metric_ic(p.X, p.y).abs() for p in parts if len(p) >= 20]
    eta_parts = [truth_table(p)["eta2"] for p in parts if len(p) >= 20]
    def pair_avg(values):
        return float(np.nanmean([_spear(values[i], values[j])
                                 for i in range(len(values)) for j in range(i + 1, len(values))])) if len(values) >= 2 else np.nan
    print(f"\nMarginal-truth persistence dev->FINAL: eta2 rho={persistence:.3f}; "
          f"within-DEV rank stability: |IC|={pair_avg(ic_parts):.3f}, eta2={pair_avg(eta_parts):.3f}")

    # Cheap, prespecified block-bootstrap intervals for two reference estimators.
    rng = np.random.default_rng(SEED)
    b_dev, b_fin = _block_rows(ev.pos), _block_rows(final.pos)
    boot_values = {m: [] for m in ("IC", "MI", "CMI_gauss") if m in scores}
    for _ in range(n_boot):
        id_dev = _block_idx(len(ev), b_dev, rng)
        id_fin = _block_idx(len(final), b_fin, rng)
        dboot, fboot = take(ev, id_dev), take(final, id_fin)
        truth_b = truth_table(fboot)["eta2"]
        for metric in boot_values:
            if metric == "IC":
                metric_b = metric_ic(dboot.X, dboot.y).abs()
            elif metric == "MI":
                metric_b = metric_mi(dboot.X, dboot.y)
            else:
                xg = rank_gauss(dboot.X.to_numpy(float))
                yg = rank_gauss(dboot.y.to_numpy(float)[:, None])[:, 0]
                metric_b = gauss_cmi_all(xg, yg, dboot.X.columns)
            boot_values[metric].append(_spear(metric_b, truth_b))
    ci = pd.DataFrame({m: {"lo": np.nanpercentile(v, 2.5), "hi": np.nanpercentile(v, 97.5)}
                       for m, v in boot_values.items() if np.isfinite(v).any()}).T
    print(f"\nBlock bootstrap {100 - 2.5 * 2:.0f}% CI for rho(metric on bootstrapped DEV, eta2 on bootstrapped FINAL):")
    print(ci.to_string() if len(ci) else "No finite bootstrap estimates")

    # Correlation diagnostics are target-free, but they are fit only on DEV.
    v = vif(ev.X)
    ac = np.abs(np.corrcoef(rank_gauss(ev.X.to_numpy(float)).T))
    ac = np.nan_to_num(ac, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(ac, 0.0)
    if len(ev.X.columns) >= 2:
        dist = np.clip(1.0 - ac, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="average")
        groups = pd.Series(fcluster(link, t=0.3, criterion="distance"))
        sizes = groups.value_counts()
        n_groups = int((sizes > 1).sum())
        n_group_features = int(sizes[sizes > 1].sum())
        pairs_09 = int(np.triu(ac > 0.9, k=1).sum())
        pairs_07 = int(np.triu(ac > 0.7, k=1).sum())
    else:
        n_groups = n_group_features = pairs_09 = pairs_07 = 0
    print(f"\nDeduplication / DEV dependence: earlier-fit dedupe dropped {n_dropped}; "
          f"median VIF={v.median():.2f}, max VIF={v.max():.1f} ({v.idxmax()}); "
          f"pairs |rho|>0.9={pairs_09}, >0.7={pairs_07}; correlated groups={n_groups} "
          f"covering {n_group_features} features")
    return dict(corr=corr, summary=summary, truth_dev=truth_dev, truth_final=truth_final,
                vif=v, ci=ci, eta2_persistence=persistence)


# ============================================================================
# Experiment 2: do individual scores add up to useful set/trade scores?
# ============================================================================
def additive_test(dev: Study, final: Study, scores: pd.DataFrame,
                  n_subsets: int = N_SUBSETS, seed: int = SEED) -> dict:
    section("ADDITIVITY TEST: heuristic score sums vs walk-forward subset utility")
    rng = np.random.default_rng(seed)
    ev = dev.ev()
    cols = list(ev.X.columns)
    if len(cols) < 2:
        raise ValueError("Additivity test needs at least two features")
    A, y = dev.X.to_numpy(float), dev.y.to_numpy(float)
    Xg = rank_gauss(ev.X.to_numpy(float))
    yg = rank_gauss(ev.y.to_numpy(float)[:, None])[:, 0]
    add_scores = {m: scores[m].reindex(cols) for m in scores.columns}
    if "IC" in add_scores:
        r = add_scores["IC"].clip(0, 0.99)
        add_scores["IC"] = -0.5 * np.log(1 - r ** 2)
    add_scores["CMI_chain"] = cmi_chain(Xg, yg, cols)
    nat_family = [m for m in ("MI", "IC", "CMI_gauss", "CMI_chain") if m in add_scores]

    records = []
    for _ in range(n_subsets):
        k = int(rng.integers(2, min(12, len(cols)) + 1))
        idx = rng.choice(len(cols), size=k, replace=False)
        selected = [cols[i] for i in idx]
        util = utility(wf_predict(A[:, idx], y, dev.pos, dev.eval_lo), y)
        rec = {"k": k, "real_ic": util["corr"], "real_topq": util["topq_expectancy"],
               "GaussJoint": gauss_joint_mi(Xg[:, idx], yg)}
        for name, score in add_scores.items():
            rec[name] = float(np.nansum(score.reindex(selected)))
        records.append(rec)
    raw = pd.DataFrame(records)

    def residualize_size(v):
        v = np.asarray(v, float)
        m = np.isfinite(v) & np.isfinite(raw["k"].to_numpy(float))
        if m.sum() < 3:
            return np.full_like(v, np.nan)
        coef = np.polyfit(raw.loc[m, "k"].to_numpy(float), v[m], 1)
        return v - np.polyval(coef, raw["k"].to_numpy(float))

    subset_rows = {}
    for name in [c for c in raw.columns if c not in ("k", "real_ic", "real_topq")]:
        v = raw[name].to_numpy(float)
        subset_rows[name] = dict(
            rho_real_ic=_spear(v, raw["real_ic"]),
            rho_real_ic_given_k=_spear(residualize_size(v), residualize_size(raw["real_ic"].to_numpy(float))),
            rho_real_topq_given_k=_spear(residualize_size(v), residualize_size(raw["real_topq"].to_numpy(float))),
            nats_over_GaussJoint=(float(np.nanmedian(raw[name] / np.maximum(raw["GaussJoint"], 1e-6)))
                                  if name in nat_family else np.nan),
        )
    subset_summary = pd.DataFrame(subset_rows).T
    print(f"Random subsets: {n_subsets}; subset utilities are chronological walk-forward diagnostics.")
    print(subset_summary.sort_values("rho_real_ic_given_k", ascending=False).to_string())

    # Composite per-trade scores: all weights/transforms come from DEV only.
    rg = RankGauss().fit(ev.X.to_numpy(float))
    Zd = rg.transform(ev.X.to_numpy(float))
    Zf = rg.transform(final.X.to_numpy(float))
    signed_ic = np.sign(metric_ic(ev.X, ev.y)).replace(0, 1).fillna(1).to_numpy(float)
    composites: dict[str, tuple[np.ndarray, np.ndarray]] = {"EqualWeightSignedIC": (Zd @ signed_ic, Zf @ signed_ic)}
    for metric in scores.columns:
        s = scores[metric].reindex(cols).fillna(0.0).clip(lower=0).to_numpy(float)
        composites[f"{metric}_weighted"] = (Zd @ (s * signed_ic), Zf @ (s * signed_ic))

    # Additive decile-R lift with empirical-Bayes shrinkage towards DEV unconditional mean.
    dev_mean = float(ev.y.mean())
    nb_dev, nb_final = np.zeros(len(ev)), np.zeros(len(final))
    for feature in cols:
        edges = np.unique(np.quantile(ev.X[feature].to_numpy(float), np.linspace(0, 1, 11)[1:-1]))
        bd = np.searchsorted(edges, ev.X[feature].to_numpy(float))
        bf = np.searchsorted(edges, final.X[feature].to_numpy(float))
        stats = pd.DataFrame({"bin": bd, "y": ev.y.to_numpy(float)}).groupby("bin")["y"].agg(["sum", "count"])
        lift = (stats["sum"] + 20.0 * dev_mean) / (stats["count"] + 20.0) - dev_mean
        nb_dev += pd.Series(bd).map(lift).fillna(0).to_numpy(float)
        nb_final += pd.Series(bf).map(lift).fillna(0).to_numpy(float)
    composites["DecileShrinkageSum(R)"] = (nb_dev, nb_final)

    # Flexible joint-model reference: in-sample DEV score is explicitly descriptive; FINAL is OOS.
    joint = make_model().fit(ev.X.to_numpy(float), ev.y.to_numpy(float))
    composites["GBM_joint(ref)"] = (joint.predict(ev.X.to_numpy(float)), joint.predict(final.X.to_numpy(float)))
    # Verify SHAP additivity only as an implementation diagnostic.
    sample_f = _sub(np.arange(len(final)), 1000)
    sample_b = _sub(np.arange(len(ev)), 100)
    sv, base = shap_values(joint, final.X.to_numpy(float)[sample_f], ev.X.to_numpy(float)[sample_b])
    max_gap = float(np.max(np.abs(sv.sum(1) + base - joint.predict(final.X.to_numpy(float)[sample_f]))))
    print(f"SHAP additivity max abs gap on FINAL sample: {max_gap:.5g}")

    yd, yf = ev.y.to_numpy(float), final.y.to_numpy(float)
    b_final = _block_rows(final.pos)
    comp_rows = {}
    for name, (sd, sf) in composites.items():
        sf = np.asarray(sf, float)
        qhi, qlo = np.nanquantile(sf, 0.9), np.nanquantile(sf, 0.1)
        lo, hi = boot_ci(lambda ii, s=sf: _spear(s[ii], yf[ii]), len(yf), b_final)
        comp_rows[name] = dict(rho_dev=_spear(sd, yd), rho_final=_spear(sf, yf),
                               rho_final_lo=lo, rho_final_hi=hi,
                               top10_R_final=float(np.mean(yf[sf >= qhi])) if np.any(sf >= qhi) else np.nan,
                               bot10_R_final=float(np.mean(yf[sf <= qlo])) if np.any(sf <= qlo) else np.nan)
    composite_summary = pd.DataFrame(comp_rows).T
    print("\nComposite score vs net R (DEV fit; FINAL evaluation):")
    print(composite_summary.sort_values("rho_final", ascending=False).to_string())
    return dict(subset=subset_summary, subset_raw=raw, composite=composite_summary)


# ============================================================================
# Experiment 3: held-out importance diagnostic (not used to choose DEV metrics)
# ============================================================================
def feature_importance_test(dev: Study, final: Study) -> dict:
    """Train on DEV, measure true held-out PFI and SHAP on FINAL as a diagnostic only."""
    section("HELD-OUT IMPORTANCE DIAGNOSTIC (no selection/tuning from these numbers)")
    ev = dev.ev()
    A, y = ev.X.to_numpy(float), ev.y.to_numpy(float)
    Af, yf = final.X.to_numpy(float), final.y.to_numpy(float)
    model = make_model().fit(A, y)
    ts, tp = _sub(np.arange(len(Af)), 1000), _sub(np.arange(len(Af)), 3000)
    bg = A[_sub(np.arange(len(A)), 100)]
    sv, _ = shap_values(model, Af[ts], bg)
    tbl = pd.DataFrame({"GAIN": _gain(model, A.shape[1]), "PFI_FINAL": pfi(model, Af[tp], yf[tp]),
                        "SHAP_FINAL": np.abs(sv).mean(0)}, index=ev.X.columns)
    tbl["SHAP_dir"] = [_spear(Af[ts, j], sv[:, j]) for j in range(A.shape[1])]
    tbl["IC_fwd_ret_FINAL"] = final.X.rank().corrwith(final.fwd.rank())
    tbl["IC_R_FINAL"] = metric_ic(final.X, final.y)
    tbl["TRUTH_eta2_FINAL"] = truth_table(final)["eta2"]
    print(tbl.sort_values("SHAP_FINAL", ascending=False).head(20).to_string())
    agree = pd.Series({
        "GAIN~eta2_FINAL": _spear(tbl["GAIN"], tbl["TRUTH_eta2_FINAL"]),
        "PFI_FINAL~eta2_FINAL": _spear(tbl["PFI_FINAL"], tbl["TRUTH_eta2_FINAL"]),
        "SHAP_FINAL~eta2_FINAL": _spear(tbl["SHAP_FINAL"], tbl["TRUTH_eta2_FINAL"]),
        "PFI_FINAL~SHAP_FINAL": _spear(tbl["PFI_FINAL"], tbl["SHAP_FINAL"]),
        "sign(SHAP_dir)==sign(IC_R_FINAL)": float(np.mean(np.sign(tbl["SHAP_dir"]) == np.sign(tbl["IC_R_FINAL"]))),
    })
    print("\nFINAL-only descriptive agreement (do not use to choose parameters):")
    print(agree.to_string())
    return dict(table=tbl, agreement=agree)


# ============================================================================
# Experiment 4: collinearity stress test
# ============================================================================
def collinearity_damage_test(dev: Study, base_scores: pd.DataFrame, n_targets: int = 3,
                             noise_levels=(0.05, 0.3, 1.0), n_copies: int = 2,
                             metrics=("MI", "IC", "CMI_gauss", "SHAP", "PFI"),
                             seed: int = SEED) -> pd.DataFrame:
    """Clone highly scored features with noise; measure retention, double-counting, and rank stability."""
    section("COLLINEARITY DAMAGE TEST")
    rng = np.random.default_rng(seed)
    metrics = [m for m in metrics if m in base_scores.columns]
    if not metrics:
        return pd.DataFrame()
    seed_metric = "SHAP" if "SHAP" in base_scores else "MI"
    top = base_scores[seed_metric].dropna().nlargest(min(n_targets, base_scores[seed_metric].notna().sum())).index.tolist()
    if not top:
        return pd.DataFrame()
    others = [c for c in dev.X.columns if c not in top]
    rows = []
    for noise in noise_levels:
        X_aug = dev.X.copy()
        duplicate_map: dict[str, list[str]] = {}
        for feature in top:
            scale = float(dev.X[feature].std())
            scale = scale if np.isfinite(scale) and scale > 0 else 1.0
            for j in range(n_copies):
                name = f"{feature}__dup{j}"
                X_aug[name] = dev.X[feature].to_numpy(float) + noise * scale * rng.normal(size=len(dev))
                duplicate_map.setdefault(feature, []).append(name)
        new_scores = score_all(replace(dev, X=X_aug), tuple(metrics), mode="wf")
        for metric in metrics:
            base_max = max(float(base_scores[metric].max(skipna=True)), 1e-12)
            per_feature_floor = 0.05 * base_max
            den = {f: max(float(base_scores.loc[f, metric]) if pd.notna(base_scores.loc[f, metric]) else 0.0,
                          per_feature_floor, 1e-12) for f in top}
            ret = np.nanmedian([
                max(float(new_scores.loc[f, metric]), 0.0) / den[f]
                if pd.notna(new_scores.loc[f, metric]) else np.nan for f in top
            ])
            inflation = np.nanmedian([
                float(new_scores.loc[[f] + duplicate_map[f], metric].clip(lower=0).sum()) / den[f]
                for f in top
            ])
            rows.append(dict(noise=noise, metric=metric, retention=ret, inflation=inflation,
                             rank_stability=_spear(base_scores.loc[others, metric], new_scores.loc[others, metric])
                             if others else np.nan))
    out = pd.DataFrame(rows)
    print(f"Features cloned: {top}; clone = source + noise * source std * Gaussian noise")
    print(out.pivot(index="metric", columns="noise", values=["retention", "inflation", "rank_stability"])
          .round(3).to_string())
    return out


# ============================================================================
# Experiment 5: feature ablation + leave-one-out conditional utility
# ============================================================================
def pfi_ablation_test(dev: Study, final: Study, scores: pd.DataFrame | None = None,
                      key: str = ABL_OBJECTIVE, max_passes: int = 3 if QUICK else 8,
                      min_gain: float = 1e-4) -> dict:
    """Choose features using DEV walk-forward utility; evaluate selected sets on FINAL once.

    LOO values are conditional utility changes after removing one feature. Positive means
    removing the feature makes performance worse, so the feature added value in that fitted set.
    """
    section(f"PFI ABLATION: select on DEV WF {key}; evaluate selected sets on FINAL")
    if key not in {"corr", "topq_expectancy", "r2"}:
        raise ValueError(f"Unsupported ablation objective {key!r}")
    sample = take(dev, _sub(np.arange(len(dev)), ABL_ROWS))
    cols = list(sample.X.columns)
    A, y = sample.X.to_numpy(float), sample.y.to_numpy(float)
    dv = dev.ev()
    Ad, yd = dv.X.to_numpy(float), dv.y.to_numpy(float)
    Af, yf = final.X.to_numpy(float), final.y.to_numpy(float)
    cache: dict[tuple[tuple[str, ...], int], dict] = {}
    final_cache: dict[tuple[tuple[str, ...], int], dict] = {}

    def obj_dev(subset, seed: int = SEED) -> dict:
        selected = tuple(sorted(set(subset)))
        ck = (selected, seed)
        if ck not in cache:
            if not selected:
                cache[ck] = dict(corr=np.nan, r2=np.nan, topq_expectancy=np.nan, n=0)
            else:
                idx = [cols.index(c) for c in selected]
                cache[ck] = utility(wf_predict(A[:, idx], y, sample.pos, sample.eval_lo, seed=seed), y)
        return cache[ck]

    def obj_final(subset, seed: int = SEED) -> dict:
        selected = tuple(sorted(set(subset)))
        ck = (selected, seed)
        if ck not in final_cache:
            if not selected:
                final_cache[ck] = dict(corr=np.nan, r2=np.nan, topq_expectancy=np.nan, n=0)
            else:
                idx = [cols.index(c) for c in selected]
                model = make_model(small=True, seed=seed).fit(Ad[:, idx], yd)
                final_cache[ck] = utility(model.predict(Af[:, idx]), yf)
        return final_cache[ck]

    cut = warm_cut(sample.pos, sample.eval_lo)
    train_idx = np.flatnonzero((sample.pos >= sample.eval_lo) & (sample.pos < cut - HORIZON))
    eval_idx = np.flatnonzero(sample.pos >= cut)
    if len(train_idx) < 200 or len(eval_idx) < 100:
        raise ValueError("Not enough GP-unseen warm-up/evaluation data for PFI ablation")

    def warm_pfi(selected):
        idx = [cols.index(c) for c in selected]
        model = make_model(small=True).fit(A[train_idx][:, idx], y[train_idx])
        rows = _sub(eval_idx, 2500)
        return pfi(model, A[rows][:, idx], y[rows], seed=SEED)

    full = obj_dev(cols)
    initial_order = pd.Series(warm_pfi(cols), index=cols).sort_values(ascending=False).index.tolist()
    forward_records = [dict(k=k, **obj_dev(initial_order[:k])) for k in range(1, len(cols) + 1)]
    forward = pd.DataFrame(forward_records)

    current = list(cols)
    backward_records, backward_sets = [], {}
    while True:
        backward_records.append(dict(k=len(current), **obj_dev(current)))
        backward_sets[len(current)] = list(current)
        if len(current) <= 1:
            break
        importance = pd.Series(warm_pfi(current), index=current)
        current.pop(int(np.nanargmin(importance.to_numpy(float))))
    backward = pd.DataFrame(backward_records)

    def best_row(frame):
        vals = pd.to_numeric(frame[key], errors="coerce")
        if not vals.notna().any():
            return frame.iloc[0]
        return frame.loc[vals.idxmax()]
    best_f = best_row(forward)
    best_b = best_row(backward)
    f_set = initial_order[:max(1, int(best_f["k"]))]
    b_set = backward_sets[max(1, int(best_b["k"]))]
    start_set = set(f_set if float(best_f[key]) >= float(best_b[key]) else b_set)
    current_set = set(start_set)
    best_value = obj_dev(current_set)[key]
    floating_log = []
    for _ in range(max_passes):
        candidates = []
        for feature in cols:
            candidate = current_set ^ {feature}
            if not candidate:
                continue
            value = obj_dev(candidate)[key]
            candidates.append((value if np.isfinite(value) else -np.inf, feature))
        if not candidates:
            break
        value, feature = max(candidates, key=lambda pair: pair[0])
        if value <= best_value + min_gain:
            break
        action = "drop" if feature in current_set else "add"
        current_set ^= {feature}
        best_value = value
        floating_log.append(dict(action=action, feature=feature, objective=value, k=len(current_set)))

    sets = {
        "all": cols,
        "forward_best": f_set,
        "backward_best": b_set,
        "floating_best": sorted(current_set),
        "pfi_top5": initial_order[:5],
    }
    final_tbl = pd.DataFrame({
        name: dict(k=len(sset), dev_wf=obj_dev(sset)[key], FINAL=obj_final(sset)[key],
                   FINAL_corr=obj_final(sset)["corr"], FINAL_topq=obj_final(sset)["topq_expectancy"],
                   FINAL_n=obj_final(sset)["n"])
        for name, sset in sets.items()
    }).T
    final_tbl["optimism_gap"] = final_tbl["dev_wf"] - final_tbl["FINAL"]
    print(f"full DEV WF {key}={full[key]:.4f}; forward={best_f[key]:.4f}; backward={best_b[key]:.4f}; "
          f"floating={best_value:.4f}")
    if floating_log:
        print(pd.DataFrame(floating_log).to_string(index=False))
    print("\nSelected sets: DEV selection then FINAL evaluation")
    print(final_tbl.to_string())
    print("Floating set:", sorted(current_set))

    # Leave-one-out conditional value. Seed variability is reported as a noise floor.
    def loo_frame(fn):
        measures = sorted({"corr", key})
        frame = {m: pd.DataFrame(np.nan, index=cols, columns=range(len(LOO_SEEDS))) for m in measures}
        for seed_idx, seed in enumerate(LOO_SEEDS):
            full_u = fn(cols, seed)
            for feature in cols:
                reduced = set(cols) - {feature}
                reduced_u = fn(reduced, seed)
                for measure in measures:
                    frame[measure].loc[feature, seed_idx] = full_u[measure] - reduced_u[measure]
        return frame
    loo_dev, loo_final = loo_frame(obj_dev), loo_frame(obj_final)
    loo = pd.DataFrame({
        f"LOO_dev_{key}": loo_dev[key].mean(1),
        "LOO_dev_corr": loo_dev["corr"].mean(1),
        "LOO_dev_corr_sd": loo_dev["corr"].std(1, ddof=0),
        f"LOO_FINAL_{key}": loo_final[key].mean(1),
        "LOO_FINAL_corr": loo_final["corr"].mean(1),
        "LOO_FINAL_corr_sd": loo_final["corr"].std(1, ddof=0),
    })
    mean_noise = float(loo["LOO_FINAL_corr_sd"].mean())
    loo_snr = float(loo["LOO_FINAL_corr"].std() / max(mean_noise, 1e-12))
    print(f"\nLOO over {len(LOO_SEEDS)} model seed(s): cross-feature spread / mean seed noise={loo_snr:.2f}")
    if scores is not None:
        agreement = pd.DataFrame({
            metric: {
                "rho_LOO_dev_corr": _spear(scores[metric].reindex(cols), loo["LOO_dev_corr"]),
                "rho_LOO_FINAL_corr": _spear(scores[metric].reindex(cols), loo["LOO_FINAL_corr"]),
            }
            for metric in scores.columns
        }).T
        print("\nDEV metric vs feature leave-one-out value (FINAL is descriptive confirmation):")
        print(agreement.to_string())
    else:
        agreement = pd.DataFrame()
    return dict(forward=forward, backward=backward, floating_log=pd.DataFrame(floating_log),
                final_set=sorted(current_set), final_value=best_value, final_check=final_tbl,
                loo=loo, loo_snr=loo_snr, full=full, pfi_order=initial_order,
                loo_agreement=agreement)


# ============================================================================
# Experiment 6: does more GP search inflate apparent information / trading results?
# ============================================================================
def overfit_wr_test(d: pd.DataFrame, trades: pd.DataFrame,
                    levels=("off", "light", "medium", "heavy"),
                    seeds=(1, 2) if not QUICK else (1,)) -> dict:
    """Diagnostic: GP fit-region metrics/predictions vs later FINAL.

    This deliberately fits GP programs to the early region. It asks whether increasing GP
    search strength inflates early apparent performance/information and whether that transfers.
    FINAL importance metrics are either univariate target metrics on FINAL or importance from
    an early-trained model evaluated on FINAL; the model is never refit on FINAL here.
    """
    section("GP SEARCH OVERFIT TEST: fit-region vs FINAL")
    n = len(d)
    gp_end, final_lo = timeline(n)
    records = []
    for level in levels:
        for seed in seeds:
            if level == "off" and seed != seeds[0]:
                continue
            t0 = time.time()
            X, _ = create_features(d, trades, gp_level=level, seed=seed, fit_until=gp_end)
            st_in = make_study(X, trades, MAX_ROWS, lo=0, hi=gp_end - HORIZON, eval_lo=0)
            st_final = make_study(X, trades, MAX_ROWS, lo=final_lo, eval_lo=final_lo)
            if len(st_in) < 200 or len(st_final) < 100:
                print(f"  skip {level}/{seed}: insufficient rows")
                continue
            model = make_r_model().fit(st_in.X.to_numpy(float), st_in.y.to_numpy(float))
            pred_in = model.predict(st_in.X.to_numpy(float))
            pred_final = model.predict(st_final.X.to_numpy(float))
            threshold = float(np.quantile(pred_in, 1 - TOPQ))
            take_in, take_final = pred_in >= threshold, pred_final >= threshold
            yi, yf = st_in.y.to_numpy(float), st_final.y.to_numpy(float)
            ci = (np.nan, np.nan)
            if take_final.sum() >= 20:
                selected = np.flatnonzero(take_final)
                block = max(5, _block_rows(st_final.pos[selected]) // 2)
                ci = boot_ci(lambda ii: float(yf[selected][ii].mean()), len(selected), block)

            # Cheap information metrics measured separately in early and future windows.
            cheap_metrics = ("MI", "IC", "CMI_gauss")
            sc_in = score_all(st_in, cheap_metrics, mode="insample")
            sc_final = score_all(st_final, cheap_metrics, mode="insample")
            rec = dict(level=level, seed=seed, n_feat=X.shape[1],
                       WR_in=float(st_in.win.to_numpy()[take_in].mean()) if take_in.any() else np.nan,
                       EXP_in=float(yi[take_in].mean()) if take_in.any() else np.nan,
                       n_in=int(take_in.sum()),
                       WR_FINAL=float(st_final.win.to_numpy()[take_final].mean()) if take_final.any() else np.nan,
                       EXP_FINAL=float(yf[take_final].mean()) if take_final.any() else np.nan,
                       EXP_FINAL_lo=ci[0], EXP_FINAL_hi=ci[1], n_FINAL=int(take_final.sum()),
                       base_EXP_FINAL=float(yf.mean()))
            for metric in cheap_metrics:
                rec[f"{metric}_fit_top5"] = float(sc_in[metric].nlargest(5).mean())
                rec[f"{metric}_FINAL_top5"] = float(sc_final[metric].nlargest(5).mean())
            records.append(rec)
            print(f"  {level:7s} seed={seed}: in E[R]={rec['EXP_in']:+.4f}, WR={rec['WR_in']:.3f}; "
                  f"FINAL E[R]={rec['EXP_FINAL']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}], "
                  f"n={rec['n_FINAL']} ({time.time() - t0:.0f}s)")
    runs = pd.DataFrame(records)
    if len(runs):
        print("\n", runs.to_string(index=False))
        metric_cols = [c for c in runs.columns if c.endswith("_fit_top5")]
        agreement = pd.DataFrame({
            "rho~EXP_in": {c: _spear(runs[c], runs["EXP_in"]) for c in metric_cols},
            "rho~EXP_FINAL": {c: _spear(runs[c], runs["EXP_FINAL"]) for c in metric_cols},
            "rho~WR_FINAL": {c: _spear(runs[c], runs["WR_FINAL"]) for c in metric_cols},
        })
        print("\nDo fit-period metric levels track early/future realized performance across runs? (descriptive, low N)")
        print(agreement.to_string())
    else:
        agreement = pd.DataFrame()
    return dict(runs=runs, corr=agreement)


# ============================================================================
# Scoreboard, plot, complete origin
# ============================================================================
SB_ORDER = list(ALL_METRICS) + ["CMI_chain", "EqualWeightSignedIC", "DecileShrinkageSum(R)", "GBM_joint(ref)"]


def scoreboard(res: dict) -> pd.DataFrame:
    section("SCOREBOARD: DEV-measured metrics vs FINAL truth/utility")
    summary = res["cross"]["summary"]
    columns = {
        "MARG rho eta2 FINAL": summary["rho_eta2_FINAL"],
        "MARG rho |spread| FINAL": summary["rho_spread_FINAL"],
        "COND rho LOO FINAL": res["ablation"]["loo_agreement"].get("rho_LOO_FINAL_corr", pd.Series(dtype=float)),
        "ADDITIVE rho|k": res["additive"]["subset"]["rho_real_ic_given_k"],
        "TRADE rho FINAL": res["additive"]["composite"]["rho_final"].rename(
            index=lambda name: name[:-9] if name.endswith("_weighted") else name),
    }
    coll = res.get("collinearity", pd.DataFrame())
    if len(coll):
        noise_min = coll["noise"].min()
        columns["COLLIN rank stability"] = coll[coll["noise"] == noise_min].set_index("metric")["rank_stability"]
    out = pd.DataFrame(columns).reindex([m for m in SB_ORDER if m in set().union(*(set(s.index) for s in columns.values()))])
    print(out.to_string())
    print(f"\neta2 dev->FINAL rank persistence={res['cross']['eta2_persistence']:.3f}; "
          f"LOO seed-noise ratio={res['ablation']['loo_snr']:.2f}.")
    return out


def plot_report(res: dict, path: str = "feature_metric_study.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    cm = res["cross"]["corr"]
    im = axes[0].imshow(cm.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(range(len(cm.columns)), cm.columns, rotation=65, ha="right")
    axes[0].set_yticks(range(len(cm.index)), cm.index)
    axes[0].set_title("Metric vs metric vs truth")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    subset = res["additive"]["subset"]["rho_real_ic_given_k"].sort_values()
    axes[1].barh(subset.index, subset.to_numpy(float))
    axes[1].set_title("Score sums vs WF subset utility (|S| controlled)")
    ablation = res["ablation"]
    axes[2].plot(ablation["forward"]["k"], ablation["forward"][ABL_OBJECTIVE], label="forward add, DEV WF")
    axes[2].plot(ablation["backward"]["k"], ablation["backward"][ABL_OBJECTIVE], label="backward drop, DEV WF")
    axes[2].axhline(ablation["full"][ABL_OBJECTIVE], ls="--", color="grey", label="all features, DEV")
    if "floating_best" in ablation["final_check"].index:
        axes[2].axhline(ablation["final_check"].loc["floating_best", "FINAL"], ls=":", color="green",
                        label="floating set, FINAL")
    axes[2].set_xlabel("Number of features")
    axes[2].set_ylabel(ABL_OBJECTIVE)
    axes[2].legend(fontsize=8)
    axes[2].set_title("Feature ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved plot: {path}")


def run_origin(d_full: pd.DataFrame, trades_full: pd.DataFrame,
               end_frac: float, first: bool) -> dict:
    m = int(end_frac * len(d_full))
    d = d_full.iloc[:m].copy()
    # IMPORTANT: don't reuse future labels from the full timeline beyond this origin.
    trades = invalidate_labels_past_origin(trades_full, m, HORIZON)
    tag = f"origin end_frac={end_frac:.2f}"
    section(f"{tag.upper()} ({m} bars)")
    gp_end, final_lo = timeline(len(d))
    print(f"Timeline: GP fit [0,{gp_end}); DEV [{gp_end},{final_lo - HORIZON}); "
          f"FINAL [{final_lo},{m}); barrier horizon={HORIZON} bars")
    X, programs = create_features(d, trades, gp_level="heavy", fit_until=gp_end)
    n_dropped = len(X.attrs.get("dropped", {}))
    print(f"{X.shape[1]} kept features ({len(programs)} GP programs, {n_dropped} duplicates removed "
          "using GP-fit rows only)")
    if not programs:
        print("WARNING: no GP columns survived; reduce GP duplication or loosen DEDUPE_THR")
    for name, program in list(programs.items())[:3]:
        print(f"  {name} = {program[:110]}")
    dev = make_study(X, trades, MAX_ROWS, lo=0, hi=final_lo - HORIZON, eval_lo=gp_end)
    final = make_study(X, trades, MAX_ROWS, lo=final_lo, eval_lo=final_lo)
    if len(dev.ev()) < 500 or len(final) < 200:
        raise ValueError(f"Insufficient DEV/FINAL data: DEV eval={len(dev.ev())}, FINAL={len(final)}")
    print(f"Rows: DEV total={len(dev)}, DEV GP-unseen evaluation={len(dev.ev())}, FINAL={len(final)}")

    results: dict = {}
    scores = score_all(dev, ALL_METRICS, mode="wf")
    section("DEV FEATURE METRICS (top 15 by SHAP; SHAP/PFI/GAIN are walk-forward held-out)")
    print(scores.sort_values("SHAP", ascending=False).head(15).to_string())
    results["scores"] = scores
    results["cross"] = cross_test(dev, final, scores, n_dropped=n_dropped)
    results["additive"] = additive_test(dev, final, scores)
    # This fixed FINAL diagnostic is not used to choose/modify the DEV metrics.
    results["importance"] = feature_importance_test(dev, final)
    results["collinearity"] = collinearity_damage_test(dev, scores)
    results["ablation"] = pfi_ablation_test(dev, final, scores)
    if first:
        results["overfit"] = overfit_wr_test(d, trades)
    results["scoreboard"] = scoreboard(results)
    plot_report(results, "feature_metric_study.png" if first else f"feature_metric_study_{end_frac:.2f}.png")
    return results


def main() -> None:
    started = time.time()
    raw = load_L1(symbol=SYMBOL, timeframe=TIMEFRAME, since=SINCE, until=UNTIL)
    d = prep_ohlcv(raw)
    if len(d) < 2000:
        raise ValueError(f"Only {len(d)} bars loaded; not enough for this study")
    # Base labels are calculated once for the largest origin; each shorter origin masks
    # outcomes whose full barrier path would reach beyond its own endpoint.
    trades = build_trades(d)
    valid = trades[trades["valid"].astype(bool)]
    print(f"{len(d)} bars | valid candidate trades={len(valid)} | base WR={valid['win'].mean():.3f} | "
          f"net E[R]={valid['r'].mean():+.4f} (gross={valid['r_gross'].mean():+.4f}; "
          f"fees={FEE_BPS_PER_SIDE} bps/side; median cost={valid['cost_r'].median():.3f}R) | "
          f"TP/SL/timeout={(valid['outcome'] == 1).mean():.2f}/"
          f"{(valid['outcome'] == -1).mean():.2f}/{(valid['outcome'] == 0).mean():.2f}")
    run_results = {}
    for i, end_frac in enumerate(END_FRACS):
        run_results[end_frac] = run_origin(d, trades, end_frac, first=(i == 0))
    if len(run_results) > 1:
        section("SCOREBOARD ACROSS PREFIX ORIGINS (mean/std; origins overlap and are NOT independent)")
        stacked = pd.concat({f"{frac:.2f}": result["scoreboard"] for frac, result in run_results.items()})
        aggregate = stacked.groupby(level=1).agg(["mean", "std"])
        print(aggregate.to_string())
        print("Origin variability is descriptive only because the prefix windows share observations.")
    print(f"\nFinished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
