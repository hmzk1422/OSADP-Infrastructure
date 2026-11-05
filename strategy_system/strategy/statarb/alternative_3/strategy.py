"""strategy.py - spread -> edge: half-life prediction, order-flow features, smart (MAE/MFE) stops, entry model, flip-flop folds.

Layout of the three files
    spread_discoverer.py   beam search on VARIANCE RATIO + HALF-LIFE (not ADF), fast Gram-matrix evaluator, panel utilities
    strategy.py            (this file) everything between "a spread was found" and "trades were taken"
    backtest.py            phase_1 (random chunks x many iterations, cached 1-minute data, parallel) / phase_2 (mixed timeframes)

What this file does
-------------------
1. HalfLifePredictor(orderflow_features=False|True)
       Ridge model of log(forward half-life / train half-life) from train-window statistics of the candidate spread.
       orderflow_features=False -> spread statistics only (identical to the study).
       orderflow_features=True  -> + the order-flow features of every underlying LEG, kept isolated per leg (ofi_ETH, ofi_BTC, ...,
                                   never netted), produced by orderflow_features() / orderflow_window_stats().
2. HALF_LIFE_MULTIPLIER = 2     the rolling z-score window of the traded period is  multiplier x half-life  (predicted when a
                                predictor is plugged in through HalfLifeRetrainRule, otherwise the train half-life).
3. smart_trading=False  (default)  risk rule = stoploss():   stop_pct = barrier_size * spread_vol * sqrt(horizon) / gross_weight
                                                              tp_pct   = rr * stop_pct
   smart_trading=True                the stop is MODELLED:   stop_pct = predicted_MAE * barrier_size,   tp_pct = rr * stop_pct
       * a tree regression predicts the MAE and MFE (fractions of gross notional) of the structural trades
         (structural_trades(): model-free mean-reversion trades with their excursions) from the same orderflow_features()
         plus the spread state;
       * an entry model then decides WHETHER to trade.  It reads the regression's outputs, its tree-dispersion (prediction
         uncertainty) and is trained on the outcomes of trades that use the regression's OUT-OF-FOLD stops, so it learns exactly
         where the regression's errors hurt.  smart_mode:
             "classifier"  random forest, objective = "will this trade win" (net R > 0)
             "rl"          policy-gradient agent, objective = mean R per EXECUTED trade (Dinkelbach / fractional programme)
4. Flip-flop folds (unchanged): instance A trades a window while the next instance trains on the window just traded.

Conventions that are unchanged
    spread (asset only) = log P_target - sum(beta_j log P_hedge_j);  z > 0 -> short the spread, z < 0 -> long it.
    Signals are observed at bar close, PnL starts on the next bar, quantities are fixed between entry and exit.
    Barriers default to INTRA-BAR high/low touches (barrier_mode="hilo", fill at the barrier); the stop is first when both are touched.
    Every *_bars setting is counted in bars of the panel traded; TradeConfig.for_bar(bar_minutes) rescales them (defaults = 1-minute bars).
"""
from __future__ import annotations

import inspect
import os
import pickle
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

try:  # package layout
    from .spread_discoverer import (AssetPanel, Candidate, DiscoverConfig, adf_p, adf_stat, create_pair, fit, half_life, px,
                                    variance_ratio, vr_mean, vr_robust_z)
except ImportError:  # flat folder
    from spread_discoverer import (AssetPanel, Candidate, DiscoverConfig, adf_p, adf_stat, create_pair, fit, half_life, px,
                                   variance_ratio, vr_mean, vr_robust_z)
try:
    from .binance_vision import load_bars
except ImportError:
    try:
        from binance_vision import load_bars
    except ImportError:
        load_bars = None

HALF_LIFE_MULTIPLIER = 2.0       # rolling z window of the traded period = HALF_LIFE_MULTIPLIER x half-life
DEFAULT_BAR = "1min"
HL_CAP = 10_000.0                # half-life used when phi >= 1 (a unit root has no half-life)
_LN2 = float(np.log(2.0))

# per-leg order-flow features used by orderflow_features(); whatever the panel does not carry is skipped
ORDERFLOW_FEATURES = ("ofi", "tfi", "net_flow", "qi1", "di1", "depth_flow1", "cvd_s", "kyle_lambda", "notional_z", "taker_ls")


def _bars_per_day(bar_minutes: float) -> float:
    return 1440.0 / float(bar_minutes)


# =============================================================================== configuration
@dataclass(frozen=True)
class StopLevels:
    """Everything a trade needs to size and bracket a position."""

    stop_pct: float        # stop distance, fraction of GROSS notional
    tp_pct: float          # take-profit distance, fraction of gross notional (= rr * stop_pct)
    leverage: float        # gross notional / capital
    gross_notional: float
    stop_dollars: float    # loss if the stop is hit
    tp_dollars: float
    capped: bool           # True when max_leverage limited the size (risk < risk_fraction)


@dataclass
class TradeConfig:
    """Execution and signal rules.  All price legs are traded; features only decide whether / how to enter.

    smart_trading=False : stop_pct = barrier_size * spread vol (stoploss()), tp = rr * stop, entries gated by the feature-sign
                          filter and the optional OU edge check.
    smart_trading=True  : stop_pct = predicted MAE * barrier_size, tp = rr * stop; the entry model (smart_mode) decides.
    """

    # --- signal
    entry_z: float = 2.0
    signal_spread: str = "asset"        # "asset": z of the traded (price-only) spread | "full": z of the discovered spread incl. features
    zscore_mode: str = "rolling"        # "static": train-window mean/std | "rolling": causal mean/std over the selection's z window
    zscore_window: int = 1_440          # bars; fallback window when the selection has none (static selections / failed half-life)
    half_life_multiplier: float = HALF_LIFE_MULTIPLIER   # z window = multiplier x half-life (applied by Screener / HalfLifeRetrainRule)

    # --- risk
    capital: float = 10_000.0
    risk_fraction: float = 0.02         # max loss per trade as a fraction of capital
    rr: float = 2.0                     # take profit = rr * stop
    barrier_size: float = 1.0           # multiplies the stop distance (formula: x vol-stop, smart: x predicted MAE); TP = rr * stop
    vol_window: int = 1_440             # bars for the rolling std of the asset-only spread changes (non-smart stop)
    stop_horizon_bars: int | None = None  # None -> half-life of the asset spread (clipped); volatility is scaled by sqrt(horizon)
    max_leverage: float = 100.0
    stop_fn: Callable[..., StopLevels | None] | None = None   # replaces stoploss() entirely if you want a different rule

    # --- strategy direction switch
    strategy_side: int = 1              # 1 = trade what the strategy says | 2 = INVERT it

    # --- barrier evaluation
    barrier_mode: str = "hilo"          # "hilo": intra-bar high/low touch, fill AT the barrier | "close": bar-close check
    hilo_extremes: str = "comove"       # how the legs' highs/lows combine inside one bar (see _resolve)

    # --- time barrier
    max_hold_bars: int = 2_880
    ou_time_barrier: bool = True        # time barrier = min(max_hold_bars, ou_hold_mult * half-life)
    ou_hold_mult: float = 3.0
    ou_min_edge_ratio: float | None = None   # optional gate (non-smart): expected OU reversion over the time barrier / stop distance

    # --- feature weighted-sum gate (features that sit inside the discovered spread)
    feature_score_mode: str = "filter"  # "off" | "filter" (sign must agree with the trade) | "feature" (smart input only) | "both"
    min_feature_score: float = 0.0

    # --- smart trading
    smart_trading: bool = False
    smart_mode: str = "classifier"      # "classifier" (will it win) | "rl" (maximise mean R per executed trade)
    smart_event_z: float | None = None  # |z| of the training events; None -> entry_z
    smart_event_stride: int = 3         # every k-th training event (neighbouring bars are near-duplicates)
    smart_min_events: int = 60
    smart_lookback_bars: int = 60       # look-back of the order-flow rolling means / z-move feature (1 hour at 1m)
    smart_oof_folds: int = 4            # purged time-blocked folds that produce out-of-fold MAE/MFE predictions
    struct_tp_z: float = 0.5            # structural trade: TP at rolling mean + tp_z sigma
    struct_sl_z: float = 3.5            # structural trade: SL at rolling mean + sl_z sigma (events at |z| >= sl_z are skipped)
    smart_threshold: float | None = None   # classifier: P(win) needed; None -> cost-aware breakeven (1 + c)/(1 + rr) + smart_margin
    smart_margin: float = 0.03
    smart_diagnostics: bool = True      # purged hold-out AUC of the classifier
    orderflow_names: tuple | None = None   # None -> ORDERFLOW_FEATURES
    reg_params: dict = field(default_factory=lambda: dict(n_estimators=120, max_depth=5, max_features=0.7, min_samples_leaf=12))
    clf_params: dict = field(default_factory=lambda: dict(n_estimators=200, max_depth=4, max_features=0.8, min_samples_leaf=20))
    rl_epochs: int = 300
    rl_lr: float = 0.05
    rl_l2: float = 1e-2
    rl_min_coverage: float = 0.10       # the policy is pushed to keep at least this share of the events
    rl_max_features: int = 30
    smart_n_jobs: int = 1               # forest threads (parallel runners keep this at 1)
    seed: int = 7

    # --- costs
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    close_at_window_end: bool = True

    def __post_init__(self):
        if self.entry_z <= 0:
            raise ValueError("entry_z must be > 0")
        if self.signal_spread not in ("asset", "full"):
            raise ValueError("signal_spread must be 'asset' or 'full'")
        if self.zscore_mode not in ("static", "rolling"):
            raise ValueError("zscore_mode must be 'static' or 'rolling'")
        if self.zscore_window < 30:
            raise ValueError("zscore_window must be >= 30 bars")
        if self.half_life_multiplier <= 0:
            raise ValueError("half_life_multiplier must be > 0")
        if self.strategy_side not in (1, 2):
            raise ValueError("strategy_side must be 1 (follow the strategy) or 2 (invert it)")
        if self.barrier_mode not in ("hilo", "close"):
            raise ValueError("barrier_mode must be 'hilo' or 'close'")
        if self.hilo_extremes not in ("comove", "worst"):
            raise ValueError("hilo_extremes must be 'comove' or 'worst'")
        if self.max_hold_bars < 2:
            raise ValueError("max_hold_bars must be >= 2")
        if self.vol_window < 10:
            raise ValueError("vol_window must be >= 10")
        if self.capital <= 0 or self.max_leverage <= 0:
            raise ValueError("capital and max_leverage must be > 0")
        if not (0 < self.risk_fraction <= 1):
            raise ValueError("risk_fraction must be in (0, 1]")
        if self.rr <= 0 or self.barrier_size <= 0:
            raise ValueError("rr and barrier_size must be > 0")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")
        if self.feature_score_mode not in ("off", "filter", "feature", "both"):
            raise ValueError("feature_score_mode must be off|filter|feature|both")
        if self.smart_mode not in ("classifier", "rl"):
            raise ValueError("smart_mode must be 'classifier' or 'rl'")
        if self.smart_event_stride < 1 or self.smart_lookback_bars < 2 or self.smart_oof_folds < 2:
            raise ValueError("smart_event_stride >= 1, smart_lookback_bars >= 2, smart_oof_folds >= 2 required")
        if self.struct_tp_z < 0 or self.struct_sl_z <= self.entry_z:
            raise ValueError("need struct_tp_z >= 0 and struct_sl_z > entry_z")

    @classmethod
    def for_bar(cls, bar_minutes: float, **overrides) -> "TradeConfig":
        """Same wall-clock settings (1 day vol window, 2 day max hold, 1 hour order-flow look-back) on any bar size."""
        day = _bars_per_day(bar_minutes)
        base = dict(
            vol_window=max(10, int(round(day))),
            zscore_window=max(30, int(round(day))),
            max_hold_bars=max(2, int(round(2 * day))),
            smart_lookback_bars=max(2, int(round(60.0 / bar_minutes))),
        )
        base.update(overrides)
        return cls(**base)

    @property
    def side_sign(self) -> int:
        """+1 when the strategy is followed (strategy_side=1), -1 when it is inverted (strategy_side=2)."""
        return 1 if self.strategy_side == 1 else -1

    @property
    def risk_dollars(self) -> float:
        return float(self.capital * self.risk_fraction)

    @property
    def cost_rate(self) -> float:
        return float((self.fee_bps + self.slippage_bps) * 1e-4)

    @property
    def breakeven_win_rate(self) -> float:
        return 1.0 / (1.0 + self.rr)


# =============================================================================== result objects
@dataclass
class ScreenSelection:
    """Frozen selection returned by Screener(): executable price legs + the feature terms of the spread."""

    target: str
    terms: tuple[str, ...]                 # ALL terms of the discovered spread (px| and ft|)
    hedge_symbols: tuple[str, ...]         # traded hedge legs (px terms only)
    intercept: float
    betas: dict[str, float]                # keyed by hedge symbol
    train_mean: float                      # full spread (incl. features)
    train_std: float
    score: float
    p_eg_train: float
    half_life_train: float
    train_start: Any
    train_end: Any
    train_bars: int
    candidate: Candidate
    ranked_candidates: tuple[Candidate, ...] = ()
    feature_terms: tuple[str, ...] = ()
    feature_coefs: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_std: dict[str, float] = field(default_factory=dict)
    asset_mean: float = 0.0
    asset_std: float = 1.0
    phi_asset: float = float("nan")        # AR(1) coefficient (OU discretisation)
    half_life_asset: float = float("nan")
    vol_seed: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    asset_seed: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    full_seed: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    z_window: int | None = None            # rolling z window (bars) = half_life_multiplier x half-life; None -> TradeConfig.zscore_window
    hl_pred: float = float("nan")          # predicted forward half-life (bars) from a HalfLifePredictor (diagnostic + time barrier)
    smart: Any = None                      # SmartModels (fitted per fold by trade_fold when smart_trading=True)

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.target,) + self.hedge_symbols

    @property
    def gross_weight(self) -> float:
        return 1.0 + sum(abs(self.betas[s]) for s in self.hedge_symbols)

    @property
    def hl_trade(self) -> float:
        """The half-life the trade rules use: the predicted one when a predictor produced it, else the train half-life."""
        return self.hl_pred if np.isfinite(self.hl_pred) else self.half_life_asset

    def feature_table(self) -> pd.DataFrame:
        rows = [{"term": t, "coef": self.feature_coefs[t], "std": self.feature_std[t],
                 "std_coef": self.feature_coefs[t] * self.feature_std[t]} for t in self.feature_terms]
        return pd.DataFrame(rows)


@dataclass
class ForwardResult:
    selection: ScreenSelection
    close: pd.DataFrame
    spread: pd.Series                 # asset-only spread; diagnostic only, NEVER used for PnL
    zscore: pd.Series
    quantities: pd.DataFrame          # base-asset units, one column per traded asset
    gross_pnl: pd.Series
    costs: pd.Series
    net_pnl: pd.Series
    trades: pd.DataFrame
    adf_stat_oos: float
    p_oos: float
    half_life_oos: float
    vr_oos: float = float("nan")


@dataclass
class FoldResult:
    cycle: int
    instance: str
    train_start: Any
    train_end: Any
    forward_start: Any
    forward_end: Any
    selection: ScreenSelection
    forward: ForwardResult


@dataclass
class AlgorithmResult:
    folds: list[FoldResult]
    pnl: pd.Series
    quantities: pd.DataFrame
    trades: pd.DataFrame
    total_pnl: float
    total_r: float
    expectancy_r: float
    sortino: float
    max_drawdown: float
    report: str
    fold_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================== panel / price utilities
def _as_panel(data: AssetPanel | pd.DataFrame) -> AssetPanel:
    if isinstance(data, AssetPanel):
        return data
    if isinstance(data, pd.DataFrame):
        return AssetPanel.from_close(data.astype(float))
    raise TypeError("data must be an AssetPanel or a close-price DataFrame")


def _timestamp_for_index(value, index: pd.Index) -> pd.Timestamp:
    t = pd.Timestamp(value)
    tz = getattr(index, "tz", None)
    if tz is not None:
        t = t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)
    elif t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def _start_offset(panel: AssetPanel, start=None) -> int:
    if start is None:
        return 0
    t = _timestamp_for_index(start, panel.logp.index)
    return int(panel.logp.index.searchsorted(t, side="left"))


def _close_frame(data: AssetPanel | pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Positive close prices. AssetPanel.logp is converted back to prices."""
    symbols = list(dict.fromkeys(symbols))
    if isinstance(data, AssetPanel):
        missing = [s for s in symbols if s not in data.logp.columns]
        if missing:
            raise ValueError(f"missing price columns in panel: {missing}")
        close = np.exp(data.logp.loc[:, symbols])
    elif isinstance(data, pd.DataFrame):
        missing = [s for s in symbols if s not in data.columns]
        if missing:
            raise ValueError(f"missing close-price columns: {missing}")
        close = data.loc[:, symbols].astype(float).copy()
    else:
        raise TypeError("forward prices must be an AssetPanel or close-price DataFrame")
    close = close.replace([np.inf, -np.inf], np.nan)
    if close.empty or close.isna().any().any() or (close <= 0).any().any():
        raise ValueError("all traded assets must have finite, strictly positive close prices")
    return close


def _slice_panel(panel: AssetPanel, start: int, stop: int) -> AssetPanel:
    return panel.iloc(start, stop)


# =============================================================================== intra-bar highs / lows
@dataclass
class HiLo:
    """Per-bar high and low of every traded asset (price units).  Used only for barrier checks (stop / take-profit touches
    inside a bar); entries, PnL between entry and exit and the signal still use bar closes."""

    high: pd.DataFrame
    low: pd.DataFrame

    def between(self, start, end) -> "HiLo":
        s = _timestamp_for_index(start, self.high.index)
        e = _timestamp_for_index(end, self.high.index)
        return HiLo(self.high.loc[s:e], self.low.loc[s:e])

    def arrays(self, close: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(high, low) float arrays aligned to close.index / close.columns, hi >= close >= lo enforced; missing bars -> close."""
        cols = list(close.columns)
        missing = [c for c in cols if c not in self.high.columns or c not in self.low.columns]
        if missing:
            raise ValueError(f"HiLo has no high/low columns for: {missing}")
        c = close.to_numpy(float)
        out = []
        for frame in (self.high, self.low):
            f = frame.loc[:, cols]
            ci, fi = close.index, f.index
            if getattr(ci, "tz", None) is None and getattr(fi, "tz", None) is not None:
                f = f.tz_convert("UTC").tz_localize(None)
            elif getattr(ci, "tz", None) is not None and getattr(fi, "tz", None) is None:
                f = f.tz_localize(ci.tz)
            elif getattr(ci, "tz", None) is not None:
                f = f.tz_convert(ci.tz)
            a = f.reindex(close.index).to_numpy(float)
            nan_frac = float(np.isnan(a).mean())
            if nan_frac > 0.5:
                raise ValueError("HiLo index does not line up with the price index (check timezone / bar size)")
            if nan_frac > 0.01:
                warnings.warn(f"HiLo is missing {nan_frac:.1%} of bars; those bars fall back to the close")
            out.append(np.where(np.isfinite(a), a, c))
        hi, lo = out
        return np.maximum(hi, c), np.minimum(lo, c)


def resample_hilo(hilo: HiLo, factor: int) -> HiLo:
    """High = max of the highs, low = min of the lows inside each clock-aligned bucket of `factor` base bars."""
    f = int(factor)
    if f <= 1:
        return hilo
    key = hilo.high.index.floor(f"{f}min")
    return HiLo(hilo.high.groupby(key).max(), hilo.low.groupby(key).min())


def load_hilo(symbols: Sequence[str], start, end, bar: str = DEFAULT_BAR, market: str = "um",
              cache_dir: str = "~/.cache/binance_vision", workers: int = 4) -> HiLo:
    """Per-bar high/low for `symbols` from binance_vision.load_bars."""
    if load_bars is None:
        raise ImportError("Could not import binance_vision.load_bars")
    if not symbols:
        raise ValueError("symbols cannot be empty")
    highs: dict[str, pd.Series] = {}
    lows: dict[str, pd.Series] = {}
    for symbol in symbols:
        raw = load_bars(symbol, start=pd.Timestamp(start), end=pd.Timestamp(end), bar=bar, market=market,
                        with_depth=False, with_ticker=False, with_metrics=False, with_funding=False,
                        with_premium=False, with_liq=False, cache_dir=cache_dir, workers=workers)
        for col in ("high", "low"):
            if col not in raw:
                raise ValueError(f"load_bars returned no '{col}' column for {symbol}; pass your own HiLo or use barrier_mode='close'")
        highs[symbol] = raw["high"].astype(float)
        lows[symbol] = raw["low"].astype(float)
    return HiLo(pd.DataFrame(highs).sort_index(), pd.DataFrame(lows).sort_index())


def _barrier_arrays(close: pd.DataFrame, cfg: TradeConfig, hilo: HiLo | None):
    if cfg.barrier_mode != "hilo":
        return None, None
    if hilo is None:
        raise ValueError("barrier_mode='hilo' needs per-bar highs/lows: pass hilo=HiLo(...) or set TradeConfig(barrier_mode='close')")
    return hilo.arrays(close)


# =============================================================================== spread statistics (research == live: same functions)
def _phi(x: np.ndarray) -> float:
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    return float((a * b).sum() / ((a * a).sum() + 1e-18))


def hl_from_phi(phi: float, cap: float = HL_CAP, floor: float = 1.0) -> float:
    if not np.isfinite(phi) or phi >= 1.0:
        return cap
    if phi <= 0.0:
        return floor
    return float(min(-_LN2 / np.log(phi), cap))


def hurst_agg(x: np.ndarray, max_lag: int) -> float:
    """Aggregated-variance Hurst of the LEVEL series (0.5 random walk, < 0.5 reverting over [2, max_lag])."""
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
    out = {"phi": phi, "loghl": float(np.log(hl)), "logsd": float(np.log(sd)), "noise": float(np.diff(x).std() / sd)}
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


def rolling_z(x: np.ndarray, window: int):
    s = pd.Series(x)
    mp = max(30, window // 4)
    mu = s.rolling(window, min_periods=mp).mean().to_numpy()
    sd = s.rolling(window, min_periods=mp).std(ddof=0).to_numpy()
    with np.errstate(all="ignore"):
        z = (x - mu) / np.where(sd > 1e-12, sd, np.nan)
    return z, mu, sd


# =============================================================================== structural trades (+ MAE / MFE, + smart barriers)
def structural_trades(x, z, mu, sd, start: int, end: int, ez: float = 2.0, tp_z: float = 0.5, sl_z: float = 3.5, T: int = 576,
                      *, stop_dist: np.ndarray | None = None, rr: float | None = None, entry_ok: np.ndarray | None = None,
                      sequential: bool = True, stride: int = 1) -> list:
    """Mean-reversion trades on a SPREAD series with close-based barriers.  Returns
        [(entry_idx, R, reason, exit_idx, side, mae, mfe), ...]      side = position on the spread (+1 long); mae/mfe in spread units.

    Fixed (model-free) barriers, the default:
        entry  first bar with ez <= |z| < sl_z after |z| < ez (armed), fade the deviation
        TP     rolling mean + tp_z*sigma, SL rolling mean + sl_z*sigma (both frozen at entry), time barrier T bars
        R      pnl / distance to the stop
    SMART barriers (smart_trading): pass `stop_dist` (per-bar stop distance in spread units, from the MAE model) and `rr`:
        SL = entry -/+ stop_dist,  TP = entry +/- rr*stop_dist,  R = pnl / stop_dist   (TP pays rr, SL pays -1)
        `entry_ok` (bool per bar) is the entry model's decision; a rejected bar does not disarm.
    sequential=True  one position at a time with armed re-entry (the real strategy).
    sequential=False every qualifying bar (every `stride`-th) is simulated independently: the sample used to FIT the MAE/MFE trees.
    MAE / MFE: worst / best close-to-close excursion of the position between entry and exit (>= 0).
    If one bar touches both barriers the stop is assumed first.
    """
    x = np.asarray(x, float)
    n = len(x)
    start, end = max(int(start), 0), min(int(end), n)
    out: list = []
    if end - start < 2:
        return out
    z, mu, sd = np.asarray(z, float), np.asarray(mu, float), np.asarray(sd, float)
    az = np.abs(z)
    fin = np.isfinite(z)
    inside = fin & (az < ez)
    smart = stop_dist is not None
    if smart:
        if rr is None:
            raise ValueError("smart barriers need rr")
        stop_dist = np.asarray(stop_dist, float)
        hit = fin & (az >= ez) & np.isfinite(stop_dist) & (stop_dist > 0)
    else:
        hit = fin & (az >= ez) & (az < sl_z) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    if entry_ok is not None:
        hit &= np.asarray(entry_ok, bool)
    lim = end - 1
    hp = np.flatnonzero(hit[start:lim]) + start
    ip = np.flatnonzero(inside[start:lim]) + start
    big = 10 ** 9

    def sim(j: int):
        sgn = 1.0 if z[j] > 0 else -1.0                       # +1: spread rich -> we short it
        if smart:
            d = float(stop_dist[j])
            sl, tp, risk = x[j] + sgn * d, x[j] - sgn * rr * d, d
        else:
            m0, s0 = mu[j], sd[j]
            tp, sl, risk = m0 + sgn * tp_z * s0, m0 + sgn * sl_z * s0, (sl_z - az[j]) * s0
        seg = x[j + 1: min(j + 1 + T, end)]
        htp = seg <= tp if sgn > 0 else seg >= tp
        hsl = seg >= sl if sgn > 0 else seg <= sl
        ktp = int(htp.argmax()) if htp.any() else big
        ksl = int(hsl.argmax()) if hsl.any() else big
        if ktp == big and ksl == big:
            k, p_exit, reason = len(seg) - 1, seg[-1], ("time" if len(seg) == T else "end")
        elif ksl <= ktp:
            k, p_exit, reason = ksl, sl, "sl"
        else:
            k, p_exit, reason = ktp, tp, "tp"
        adv = sgn * (seg[: k + 1] - x[j])
        mae, mfe = max(float(adv.max()), 0.0), max(float((-adv).max()), 0.0)
        return (j, float(-sgn * (p_exit - x[j]) / risk), reason, j + 1 + k, int(-sgn), mae, mfe)

    if not sequential:
        for j in hp[:: max(int(stride), 1)]:
            out.append(sim(int(j)))
        return out
    i, armed = start, True
    while True:
        if not armed:
            k = int(ip.searchsorted(i))
            if k >= len(ip):
                break
            i, armed = int(ip[k]), True
        m = int(hp.searchsorted(i))
        if m >= len(hp):
            break
        tr = sim(int(hp[m]))
        out.append(tr)
        e = tr[3]
        armed = bool(fin[e] and az[e] < ez)
        i = e + 1
    return out


# =============================================================================== order-flow features (shared by HalfLifePredictor and smart trading)
def orderflow_features(panel: AssetPanel, symbols: Sequence[str], *, names: Sequence[str] | None = None, lookback: int = 60,
                       price_context: bool = True) -> pd.DataFrame:
    """Causal order-flow features of the UNDERLYING legs, one block per leg, never netted across legs.

    Columns (time x column), for every leg `sym` and every feature `name` the panel carries:
        "{name}|{sym}"            the feature at the bar
        "{name}|{sym}|m{L}"       its rolling mean over the last L bars
        "ret|{sym}|{L}"           the leg's return over the last L bars   (price_context)
        "vol|{sym}"               the leg's rolling volatility            (price_context)
    e.g. for a spread built from ETH, BTC and XRP:  ofi|ETH/USDT, ofi|BTC/USDT, ofi|XRP/USDT  (three separate columns).
    Every value at bar t only uses bars <= t.  `names=None` -> ORDERFLOW_FEATURES (missing ones are skipped).
    """
    L = max(int(lookback), 2)
    mp = max(2, L // 4)
    have = [n for n in (names or ORDERFLOW_FEATURES) if n in panel.feats]
    cols: dict[str, np.ndarray] = {}
    for sym in symbols:
        for nme in have:
            df = panel.feats[nme]
            if sym not in df.columns:
                continue
            s = df[sym].astype(float)
            cols[f"{nme}|{sym}"] = s.to_numpy()
            cols[f"{nme}|{sym}|m{L}"] = s.rolling(L, min_periods=mp).mean().to_numpy()
        if price_context and sym in panel.ret.columns:
            cols[f"ret|{sym}|{L}"] = panel.ret[sym].rolling(L, min_periods=mp).sum().to_numpy()
            cols[f"vol|{sym}"] = panel.vol[sym].to_numpy(float)
    return pd.DataFrame(cols, index=panel.logp.index)


def _safe(sym: str) -> str:
    return sym.replace("/", "").replace("|", "")


def orderflow_window_stats(of: pd.DataFrame, a: int, b: int, legs: Sequence[str], all_symbols: Sequence[str]) -> dict:
    """Candidate-level summary of orderflow_features() over the train window [a, b): per leg and per feature the mean, the
    std and the latest rolling mean.  Legs that are not in the candidate stay NaN (+ an of_leg_<sym> indicator), so the
    ridge sees isolated, symbol-specific columns."""
    out: dict[str, float] = {}
    legs = set(legs)
    sub = of.iloc[a:b]
    for sym in all_symbols:
        out[f"of_leg_{_safe(sym)}"] = float(sym in legs)
    for col in of.columns:
        parts = col.split("|")
        if len(parts) != 2 or parts[0] in ("vol",):         # plain "{name}|{sym}" columns only
            continue
        name, sym = parts
        key = f"of_{name}_{_safe(sym)}"
        if sym in legs:
            v = sub[col].to_numpy(float)
            v = v[np.isfinite(v)]
            out[key + "_mean"] = float(v.mean()) if len(v) else np.nan
            out[key + "_std"] = float(v.std()) if len(v) else np.nan
            mcols = [c for c in of.columns if c.startswith(f"{name}|{sym}|m")]
            out[key + "_end"] = float(sub[mcols[0]].iloc[-1]) if mcols else np.nan
        else:
            out[key + "_mean"] = out[key + "_std"] = out[key + "_end"] = np.nan
    return out


# =============================================================================== HalfLifePredictor + rolling-origin dataset
@dataclass
class DatasetConfig:
    train_bars: int = 5_000
    fwd_bars: tuple = (500, 1_000, 2_500, 5_000)      # forward horizons evaluated for every (origin, candidate)
    purge: int = 1
    step: int = 1_000
    top_k: int = 20                                    # candidates per origin taken from the beam ranking
    z_windows: tuple = (60, 120, 240, 480, 720, 1_440, 2_880)
    feat_struct_windows: tuple = (240, 1_440)          # in-sample structural payoff used as a FEATURE
    entry_z: float = 2.0
    tp_z: float = 0.5
    sl_z: float = 3.5
    hold_mult: float = 2.0                             # time barrier = clip(hold_mult * train half-life)
    hold_min: int = 60
    hold_max: int = 2_880
    vr_lags: tuple = (5, 15, 30, 60, 120)
    label_payoff: bool = False                         # also label y_R_<W> (structural payoff per z window); slower
    orderflow_lookback: int = 60
    orderflow_names: tuple | None = None
    max_origins: int | None = None
    seed: int = 7
    n_jobs: int = 1

    @property
    def span(self) -> int:
        return self.train_bars + self.purge + max(self.fwd_bars)


def spread_from_betas(logp: pd.DataFrame, y: str, betas: dict, a: int, b: int) -> np.ndarray:
    x = logp[y].to_numpy(float)[a:b].copy()
    for sym, beta in betas.items():
        x -= beta * logp[sym].to_numpy(float)[a:b]
    return x


def _px_betas(c) -> dict:
    return {t.split("|", 1)[1]: float(v) for t, v in c.betas.items() if t.startswith("px|")}


def hold_from_hl(hl: float, d: DatasetConfig) -> int:
    return int(np.clip(round(d.hold_mult * hl), d.hold_min, d.hold_max))


def candidate_features(c, xt: np.ndarray, d: DatasetConfig, rank: int = 0, *, of: pd.DataFrame | None = None,
                       legs: Sequence[str] | None = None, all_symbols: Sequence[str] = ()) -> dict:
    """Everything known at the END of the train window about candidate `c` (xt = its asset-only train spread).
    f_* = spread statistics (the study).  of_* = isolated per-leg order-flow statistics, added when `of` is given."""
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
        f_hold=hold, f_loghl_a=a["loghl"], f_loghl_b=b["loghl"],
        f_hl_split=abs(a["loghl"] - b["loghl"]), f_phi_split=abs(a["phi"] - b["phi"]),
        f_mean_shift=abs(float(xt[h:].mean() - xt[:h].mean())) / sd,
        f_sd_ratio=float(np.log((xt[h:].std() + 1e-18) / (xt[:h].std() + 1e-18))),
        f_trend=abs(float(np.polyfit(np.arange(n), xt, 1)[0])) * n / sd,
        f_rank=rank,
        f_log_p_eg=float(np.log10(max(c.p_eg, 1e-12))) if np.isfinite(c.p_eg) else float("nan"),
        f_score=float(c.score), f_vr_search=float(c.vr),
        f_n_px=sum(t.startswith("px|") for t in c.terms), f_n_terms=len(c.terms),
        f_gross_beta=float(sum(abs(v) for v in _px_betas(c).values())),
        f_max_share=float(c.max_share), f_drift=float(c.drift) if np.isfinite(c.drift) else 0.0,
    )
    for W in d.feat_struct_windows:
        z, mu, sg = rolling_z(xt, W)
        tr = structural_trades(xt, z, mu, sg, W, n, d.entry_z, d.tp_z, d.sl_z, hold)
        f[f"f_sR_{W}"] = float(np.mean([t[1] for t in tr])) if tr else float("nan")
        f[f"f_sN_{W}"] = len(tr)
    if of is not None:
        f.update(orderflow_window_stats(of, 0, n, legs if legs is not None else _px_betas(c).keys(), all_symbols))
    return f


def forward_labels(xt: np.ndarray, xf_full: np.ndarray, f: dict, d: DatasetConfig) -> list:
    """One dict per forward horizon.  Betas are the TRAIN betas (frozen), z uses the train tail as warm start."""
    hold = int(f["f_hold"])
    rows = []
    zs = None
    if d.label_payoff:
        seed = xt[-max(d.z_windows):]
        arr = np.r_[seed, xf_full]
        zs = {W: rolling_z(arr, W) for W in d.z_windows}
    for h in d.fwd_bars:
        xf = xf_full[:h]
        s = spread_stats(xf, hold, d.vr_lags, adf=h >= 1_000)
        y = {f"y_{k}": v for k, v in s.items()}
        y["y_logratio"] = float(np.clip(s["loghl"] - f["f_loghl"], -2.0, 4.0))
        y["y_persist"] = float(y["y_logratio"] < _LN2)
        if zs is not None:
            for W, (z, mu, sg) in zs.items():
                tr = structural_trades(arr, z, mu, sg, len(seed), len(seed) + h, d.entry_z, d.tp_z, d.sl_z, hold)
                R = [t[1] for t in tr]
                y[f"y_n_{W}"], y[f"y_R_{W}"] = len(R), (float(np.mean(R)) if R else float("nan"))
        y["horizon"] = h
        rows.append(y)
    return rows


def _origin_rows(panel, y: str, discover_cfg, d: DatasetConfig, o: int) -> list:
    tr = panel.iloc(o, o + d.train_bars)
    cfg = discover_cfg if tr.feats else replace(discover_cfg, use_features=False)
    ranked = create_pair(tr, y, cfg=cfg)
    cands = [c for c in ranked if any(t.startswith("px|") for t in c.terms)][: d.top_k]
    a = o + d.train_bars + d.purge
    b = a + max(d.fwd_bars)
    of = orderflow_features(tr, tr.symbols, names=d.orderflow_names, lookback=d.orderflow_lookback) if tr.feats else None
    rows = []
    for rank, c in enumerate(cands):
        betas = _px_betas(c)
        x_all = spread_from_betas(panel.logp, y, betas, o, b)
        xt, xf = x_all[: d.train_bars], x_all[d.train_bars + d.purge:]
        f = candidate_features(c, xt, d, rank, of=of, legs=(y, *betas), all_symbols=tr.symbols)
        meta = {"origin": o, "origin_time": panel.logp.index[o], "hedge": " + ".join(betas)}
        for lab in forward_labels(xt, xf, f, d):
            rows.append({**meta, **f, **lab})
    return rows


def build_dataset(panel, y: str, discover_cfg, d: DatasetConfig | None = None, segment: int = 0) -> pd.DataFrame:
    """Rolling-origin dataset: one row per (origin, candidate, horizon).  The expensive part is one beam search per origin."""
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


def _minutes(ts: pd.Series) -> np.ndarray:
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


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 20 or a[ok].nunique() < 3:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def _ridge_pipeline(alpha: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


DEFAULT_FEATURES = ["f_loghl", "f_loghl_a", "f_loghl_b", "f_hl_split", "f_phi_split", "f_mean_shift", "f_sd_ratio",
                    "f_trend", "f_noise", "f_adf_logp", "f_vr15", "f_vr60", "f_vr120", "f_vrz_hold", "f_hurst",
                    "f_log_p_eg", "f_n_px", "f_gross_beta", "f_drift", "f_rank", "f_sR_240", "f_sR_1440"]


class HalfLifePredictor:
    """Predicts log(forward half-life / train half-life) of a candidate spread from train-side statistics (ridge).

    orderflow_features=False : the spread statistics of the study (DEFAULT_FEATURES).
    orderflow_features=True  : + the isolated per-leg order-flow statistics (columns "of_*", see orderflow_window_stats) of every
                               underlying the spread uses; features are never netted across legs.
    The half-life the trade rules use is  predict_hl(F)  and the rolling z window is HALF_LIFE_MULTIPLIER x that.
    """

    def __init__(self, horizon: int = 2_500, features: Sequence[str] | None = None, alpha: float = 30.0,
                 orderflow_features: bool = False):
        self.horizon, self.alpha, self.orderflow_features = horizon, alpha, bool(orderflow_features)
        self._base = list(features or DEFAULT_FEATURES)
        self.features = list(self._base)
        self.model = None
        self.median_ratio = 0.0
        self.dcfg: DatasetConfig = DatasetConfig()

    def _columns(self, df: pd.DataFrame) -> list[str]:
        cols = list(self._base)
        if self.orderflow_features:
            of = [c for c in df.columns if c.startswith("of_")]
            if not of:
                raise ValueError("orderflow_features=True but the dataset has no of_* columns "
                                 "(build it from a panel that carries order-flow features)")
            cols += of
        return [c for c in cols if c in df.columns]

    def _data(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[df["horizon"] == self.horizon]
        return s[s["y_logratio"].notna()]

    def fit(self, df: pd.DataFrame, d: DatasetConfig | None = None) -> "HalfLifePredictor":
        s = self._data(df)
        self.features = self._columns(s)
        self.dcfg = d or self.dcfg
        self.median_ratio = float(s["y_logratio"].median())
        self.model = _ridge_pipeline(self.alpha).fit(s[self.features], s["y_logratio"])
        return self

    def predict_logratio(self, F: pd.DataFrame) -> np.ndarray:
        return self.model.predict(F.reindex(columns=self.features))

    def predict_hl(self, F: pd.DataFrame) -> np.ndarray:
        return np.exp(F["f_loghl"].to_numpy(float) + self.predict_logratio(F))

    def oof(self, df: pd.DataFrame, d: DatasetConfig, bar_minutes: float = 1.0, n_splits: int = 5) -> pd.Series:
        s = self._data(df)
        feats = self._columns(s)
        pred = pd.Series(np.nan, index=s.index)
        for tr, te in purged_time_splits(s, d, bar_minutes, n_splits):
            m = _ridge_pipeline(self.alpha).fit(s.iloc[tr][feats], s.iloc[tr]["y_logratio"])
            pred.iloc[te] = m.predict(s.iloc[te][feats])
        return pred

    def cv_report(self, df: pd.DataFrame, d: DatasetConfig, bar_minutes: float = 1.0) -> pd.DataFrame:
        """Purged time-blocked CV against two baselines: the train half-life (ratio 1) and one constant median ratio."""
        s = self._data(df)
        o = self.oof(df, d, bar_minutes).reindex(s.index)
        ok = o.notna()
        y = s.loc[ok, "y_logratio"]
        const = pd.Series(0.0, index=y.index)
        shr = pd.Series(float(s["y_logratio"].median()), index=y.index)
        rows = []
        tag = "ridge + order flow" if self.orderflow_features else "ridge on spread stats"
        for name, p in (("train half-life", const), ("shrunk (constant ratio)", shr), (tag, o[ok])):
            rows.append({"predictor": name, "mae_log": float((y - p).abs().mean()),
                         "rmse_log": float(np.sqrt(((y - p) ** 2).mean())),
                         "rank_corr": _rank_corr(p, y) if p.nunique() > 1 else float("nan")})
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "HalfLifePredictor":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def compare_predictors(df: pd.DataFrame, d: DatasetConfig, bar_minutes: float = 1.0, horizon: int = 2_500) -> pd.DataFrame:
    """orderflow_features False vs True on the same purged CV (the two rows to read: do the order-flow stats beat the spread stats?)."""
    rows = []
    for flag in (False, True):
        try:
            rep = HalfLifePredictor(horizon, orderflow_features=flag).cv_report(df, d, bar_minutes)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        rep.insert(0, "orderflow_features", flag)
        rows.append(rep)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================== 1) Screener
def Screener(
    panel: AssetPanel | pd.DataFrame | None = None, y: str | None = None, *,
    symbols: Sequence[str] | None = None, start=None, bars: int = 5_000, bar: str = DEFAULT_BAR, market: str = "um",
    cache_dir: str = "~/.cache/binance_vision", workers: int = 4, seed_x: Sequence[str] | None = None,
    cfg: DiscoverConfig | None = None, ranked: Sequence[Candidate] | None = None,
    half_life_mult: float = HALF_LIFE_MULTIPLIER, z_bounds: tuple[int, int] = (30, 2_880),
) -> ScreenSelection:
    """Beam search (variance ratio + half-life objective) on exactly the requested training bars; freeze the best candidate
    that has a tradable price leg.  The selection's rolling z window is  half_life_mult x half-life of its asset-only spread."""
    if y is None:
        raise ValueError("y (the target asset symbol) is required")
    cfg0 = cfg or DiscoverConfig()
    if panel is None:
        raise ValueError("provide a panel (load it with spread_discoverer.get_assets)")
    panel0 = _as_panel(panel)
    offset = _start_offset(panel0, start)
    if offset >= len(panel0.logp):
        raise ValueError(f"start={start!r} is after the available panel")
    if len(panel0.logp) - offset < bars:
        raise ValueError(f"not enough bars from start: need {bars}, have {len(panel0.logp) - offset}")
    panel0 = panel0.iloc(offset, offset + bars)
    screen_cfg = cfg0 if panel0.feats else replace(cfg0, use_features=False)
    if y not in panel0.logp.columns:
        raise ValueError(f"target {y!r} is not in the panel")
    train_panel = panel0
    if ranked is None:
        ranked = create_pair(train_panel, y, seed_x=seed_x, cfg=screen_cfg)
    executable = [c for c in ranked if c.terms and any(t.startswith("px|") for t in c.terms)]
    if not executable:
        raise RuntimeError("beam search returned no candidate with a tradable price leg")
    chosen = executable[0]
    terms = tuple(chosen.terms)
    X = train_panel.matrix(terms)
    yv = train_panel.series(px(y))
    intercept, slopes, residual = fit(yv, X, method=screen_cfg.fit_method, **screen_cfg.fit_kw)
    sigma = float(np.std(residual))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        raise RuntimeError("selected candidate has a degenerate training spread")
    betas: dict[str, float] = {}
    f_terms, f_coef, f_mean, f_std = [], {}, {}, {}
    for j, (t, b) in enumerate(zip(terms, slopes)):
        if t.startswith("px|"):
            betas[t.split("|", 1)[1]] = float(b)
        else:
            f_terms.append(t)
            f_coef[t], f_mean[t], f_std[t] = float(b), float(X[:, j].mean()), float(X[:, j].std())
    asset = train_panel.series(px(y)).copy()
    for s_, b_ in betas.items():
        asset = asset - b_ * train_panel.series(px(s_))
    a_std = float(np.std(asset))
    if not np.isfinite(a_std) or a_std <= 1e-12:
        raise RuntimeError("selected candidate has a degenerate asset-only training spread")
    phi = _phi(asset)
    hl = half_life(asset, floor=1.0)
    asset_arr = np.asarray(asset, dtype=float)
    full_arr = asset_arr - float(intercept)
    for j, t in enumerate(terms):
        if not t.startswith("px|"):
            full_arr = full_arr - f_coef[t] * (X[:, j] - f_mean[t])
    zw = int(np.clip(round(half_life_mult * hl), z_bounds[0], z_bounds[1])) if np.isfinite(hl) else None
    return ScreenSelection(
        target=y, terms=terms, hedge_symbols=tuple(betas), intercept=float(intercept), betas=betas,
        train_mean=float(np.mean(residual)), train_std=sigma, score=float(chosen.score), p_eg_train=float(chosen.p_eg),
        half_life_train=float(chosen.half_life), train_start=train_panel.logp.index[0], train_end=train_panel.logp.index[-1],
        train_bars=len(train_panel.logp), candidate=chosen, ranked_candidates=tuple(executable),
        feature_terms=tuple(f_terms), feature_coefs=f_coef, feature_mean=f_mean, feature_std=f_std,
        asset_mean=float(np.mean(asset)), asset_std=a_std, phi_asset=phi, half_life_asset=float(hl),
        vol_seed=np.diff(asset_arr)[-4096:].astype(float), asset_seed=asset_arr[-4096:].copy(), full_seed=full_arr[-4096:].copy(),
        z_window=zw)


class HalfLifeRetrainRule:
    """retrain_rule(train_panel, y, discover_cfg, previous_selection) -> ScreenSelection

    beam search -> top_k candidates -> candidate_features (the SAME function as the research dataset) -> predicted forward
    half-life of each -> trade the beam's best candidate (select="beam") or the one with the shortest predicted half-life
    (select="predicted_hl") with  z window = clip(multiplier x predicted half-life)."""

    def __init__(self, predictor: HalfLifePredictor, multiplier: float = HALF_LIFE_MULTIPLIER, top_k: int = 20,
                 w_lo: int = 30, w_hi: int = 2_880, select: str = "beam"):
        if select not in ("beam", "predicted_hl"):
            raise ValueError("select must be 'beam' or 'predicted_hl'")
        self.predictor, self.multiplier, self.top_k, self.w_lo, self.w_hi, self.select = predictor, multiplier, top_k, w_lo, w_hi, select

    def from_ranked(self, train_panel: AssetPanel, y: str, discover_cfg: DiscoverConfig, ranked: Sequence[Candidate]) -> ScreenSelection:
        d, pred = self.predictor.dcfg, self.predictor
        execu = [c for c in ranked if any(t.startswith("px|") for t in c.terms)][: self.top_k]
        if not execu:
            raise RuntimeError("beam search returned no candidate with a tradable price leg")
        n = len(train_panel.logp)
        of = None
        if pred.orderflow_features and train_panel.feats:
            of = orderflow_features(train_panel, train_panel.symbols, names=d.orderflow_names, lookback=d.orderflow_lookback)
        rows = []
        for rank, c in enumerate(execu):
            betas = _px_betas(c)
            xt = spread_from_betas(train_panel.logp, y, betas, 0, n)
            rows.append(candidate_features(c, xt, d, rank, of=of, legs=(y, *betas), all_symbols=train_panel.symbols))
        F = pd.DataFrame(rows)
        hl_all = pred.predict_hl(F)
        i = 0 if self.select == "beam" else int(np.argmin(np.where(hl_all >= 2.0, hl_all, np.inf)))
        order = [execu[i]] + [c for j, c in enumerate(execu) if j != i]
        sel = Screener(train_panel, y, bars=n, cfg=discover_cfg, ranked=order, half_life_mult=self.multiplier)
        hl_pred = float(hl_all[i])
        W = int(np.clip(round(self.multiplier * hl_pred), self.w_lo, self.w_hi))
        return replace(sel, z_window=W, hl_pred=hl_pred)

    def __call__(self, train_panel, y, discover_cfg, previous_selection=None) -> ScreenSelection:
        cfg = discover_cfg if train_panel.feats else replace(discover_cfg, use_features=False)
        return self.from_ranked(train_panel, y, discover_cfg, create_pair(train_panel, y, cfg=cfg))


# =============================================================================== 2) risk rules
def levels_from_stop_pct(stop_pct: float, cfg: TradeConfig) -> StopLevels | None:
    """Stop distance (fraction of GROSS notional) -> size and brackets.  Shared by the formula rule and the smart rule:
        leverage = risk_fraction / stop_pct (<= max_leverage);  loss at stop = risk_fraction * capital;  TP = rr * stop."""
    if not (np.isfinite(stop_pct) and stop_pct > 0):
        return None
    lev_raw = cfg.risk_fraction / stop_pct
    lev = min(lev_raw, cfg.max_leverage)
    gross = cfg.capital * lev
    stop_d = gross * stop_pct
    return StopLevels(stop_pct=float(stop_pct), tp_pct=float(cfg.rr * stop_pct), leverage=float(lev), gross_notional=float(gross),
                      stop_dollars=float(stop_d), tp_dollars=float(cfg.rr * stop_d), capped=bool(lev_raw > cfg.max_leverage))


def stoploss(spread_vol: float, gross_weight: float, cfg: TradeConfig, horizon_bars: int) -> StopLevels | None:
    """The FORMULA risk rule (smart_trading=False).  Edit this function (or pass TradeConfig.stop_fn) to change it.

        vol_pct  = spread_vol * sqrt(horizon) / gross_weight        # % of gross notional
        stop_pct = barrier_size * vol_pct          tp_pct = rr * stop_pct
    With smart_trading=True the stop is stop_pct = predicted MAE * barrier_size instead (see SmartModels)."""
    if not (np.isfinite(spread_vol) and spread_vol > 0 and gross_weight > 0):
        return None
    vol_pct = spread_vol * np.sqrt(max(int(horizon_bars), 1)) / gross_weight
    return levels_from_stop_pct(cfg.barrier_size * vol_pct, cfg)


def _levels(spread_vol: float, sel: ScreenSelection, cfg: TradeConfig) -> StopLevels | None:
    fn = cfg.stop_fn or stoploss
    return fn(spread_vol, sel.gross_weight, cfg, _stop_horizon(sel, cfg))


def _stop_horizon(sel: ScreenSelection, cfg: TradeConfig) -> int:
    if cfg.stop_horizon_bars:
        return int(cfg.stop_horizon_bars)
    hl = sel.hl_trade
    h = hl if np.isfinite(hl) else 48.0
    return int(np.clip(round(h), 6, cfg.max_hold_bars))


def _time_barrier(sel: ScreenSelection, cfg: TradeConfig) -> int:
    T = int(cfg.max_hold_bars)
    hl = sel.hl_trade
    if cfg.ou_time_barrier and np.isfinite(hl) and hl > 0:
        T = int(min(T, max(2, np.ceil(cfg.ou_hold_mult * hl))))
    return T


# =============================================================================== 3) signals
def _asset_spread(close: pd.DataFrame, sel: ScreenSelection) -> pd.Series:
    logp = np.log(close.loc[:, list(sel.symbols)])
    s = logp[sel.target].copy()
    for h in sel.hedge_symbols:
        s = s - sel.betas[h] * logp[h]
    return s.rename("asset_spread")


def _feature_part(panel: AssetPanel | None, sel: ScreenSelection, index: pd.Index) -> pd.Series:
    """F_t = sum_j coef_j * (x_j,t - mean_j)   (spread = asset - icpt - F)."""
    f = np.zeros(len(index))
    if not sel.feature_terms:
        return pd.Series(f, index=index, name="feature_part")
    if panel is None or any(not panel.has(t) for t in sel.feature_terms):
        raise ValueError("the selected spread contains feature terms; pass an AssetPanel that carries those features")
    for t in sel.feature_terms:
        f = f + sel.feature_coefs[t] * (panel.series(t) - sel.feature_mean[t])
    return pd.Series(f, index=index, name="feature_part")


def _clean(df: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    df = df.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = df.median().fillna(0.0)
    return df.fillna(medians).fillna(0.0), medians


def _candidate_bars(sig: pd.DataFrame, z_level: float, n: int) -> np.ndarray:
    z = sig["z"].to_numpy()
    ok = np.isfinite(z) & (np.abs(z) >= z_level) & np.isfinite(sig["vol"].to_numpy())
    ok[max(n - 2, 0):] = False          # need at least one forward bar
    return np.flatnonzero(ok)


def _rolling_z(x: np.ndarray, seed, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal rolling z over the `window` bars ending at each bar; `seed` = tail of the previous window as warm start.
    Returns (z, rolling_mean, rolling_std)."""
    x = np.asarray(x, float)
    w = int(window)
    seed = np.asarray(seed, float)
    s = seed[-(w - 1):] if (len(seed) and w > 1) else np.zeros(0)
    roll = pd.Series(np.r_[s, x]).rolling(w, min_periods=max(30, w // 4))
    mu = roll.mean().to_numpy()[len(s):]
    sd = roll.std(ddof=0).to_numpy()[len(s):]
    sd = np.where(sd > 1e-12, sd, np.nan)
    return (x - mu) / sd, mu, sd


def _signals(panel: AssetPanel | None, close: pd.DataFrame, sel: ScreenSelection, cfg: TradeConfig,
             use_seed: bool = True) -> pd.DataFrame:
    idx = close.index
    asset = _asset_spread(close, sel)
    F = _feature_part(panel, sel, idx)
    full = asset - sel.intercept - F
    a, fu = asset.to_numpy(float), full.to_numpy(float)
    if cfg.zscore_mode == "rolling":
        zw = int(sel.z_window or cfg.zscore_window)
        z_asset, mu, sd_a = _rolling_z(a, sel.asset_seed if use_seed else (), zw)
        z_full, _, _ = _rolling_z(fu, sel.full_seed if use_seed else (), zw)
        dev = a - mu
    else:
        mu, sd_a = np.full(len(a), sel.asset_mean), np.full(len(a), sel.asset_std)
        z_asset = (a - sel.asset_mean) / sel.asset_std
        z_full = (fu - sel.train_mean) / max(sel.train_std, 1e-12)
        dev = a - sel.asset_mean
    d = np.diff(a)
    seed = sel.vol_seed[-(cfg.vol_window - 1):] if (use_seed and len(sel.vol_seed)) else np.zeros(0)
    arr = np.r_[seed, d]
    rolled = pd.Series(arr).rolling(cfg.vol_window, min_periods=max(10, cfg.vol_window // 4)).std().to_numpy()
    vol = np.full(len(idx), np.nan)
    vol[1:] = rolled[len(seed):]
    if len(seed):
        vol[0] = rolled[len(seed) - 1]
    return pd.DataFrame({
        "asset_spread": a, "z_asset": z_asset, "z_full": z_full, "z": z_asset if cfg.signal_spread == "asset" else z_full,
        "dev": dev, "mu": mu, "sd": sd_a, "feat_score": F.to_numpy(float) / sel.asset_std, "vol": vol}, index=idx)


# =============================================================================== 4) barrier simulation
def _legs(close_row: np.ndarray, sel: ScreenSelection, side: int, gross: float) -> np.ndarray:
    """Base-asset quantities (column order = sel.symbols) for a given gross notional."""
    tn = gross / sel.gross_weight
    q = np.zeros(len(sel.symbols))
    q[0] = side * tn / close_row[0]
    for k, h in enumerate(sel.hedge_symbols, start=1):
        q[k] = -side * sel.betas[h] * tn / close_row[k]
    return q


def _resolve(close_arr: np.ndarray, i: int, q: np.ndarray, stop_d: float, tp_d: float, T: int,
             hi: np.ndarray | None = None, lo: np.ndarray | None = None, extremes: str = "comove"):
    """First barrier touched after entry at the close of bar i.  Returns None (no forward bar) or
    (exit_index, reason, fill_pnl, close_pnl); reason 'sl' | 'tp' | 'time' | 'window_end'.
    hi/lo None  -> barriers tested on bar closes, fill at the close.
    hi/lo given -> a barrier is touched when the bar's high/low reaches it; the fill is EXACTLY at the barrier.
      'comove': all legs at their highs together / lows together (+ the close); 'worst': every leg at its adverse extreme.
    If one bar touches both barriers the stop is assumed first."""
    last = len(close_arr) - 1
    end = min(i + T, last)
    if end <= i:
        return None
    ref = close_arr[i]
    path = (close_arr[i + 1:end + 1] - ref) @ q
    if hi is None:
        lowest = highest = path
    else:
        dh, dl = hi[i + 1:end + 1] - ref, lo[i + 1:end + 1] - ref
        if extremes == "comove":
            vh, vl = dh @ q, dl @ q
            lowest = np.minimum(np.minimum(vh, vl), path)
            highest = np.maximum(np.maximum(vh, vl), path)
        else:
            up = q > 0
            lowest = np.minimum(np.where(up, dl, dh) @ q, path)
            highest = np.maximum(np.where(up, dh, dl) @ q, path)
    sl = np.flatnonzero(lowest <= -stop_d)
    tp = np.flatnonzero(highest >= tp_d)
    k_sl = int(sl[0]) if len(sl) else None
    k_tp = int(tp[0]) if len(tp) else None
    if k_sl is None and k_tp is None:
        reason = "time" if i + T <= last else "window_end"
        return end, reason, float(path[-1]), float(path[-1])
    intrabar = hi is not None
    if k_tp is None or (k_sl is not None and k_sl <= k_tp):
        return i + k_sl + 1, "sl", (-float(stop_d) if intrabar else float(path[k_sl])), float(path[k_sl])
    return i + k_tp + 1, "tp", (float(tp_d) if intrabar else float(path[k_tp])), float(path[k_tp])


# =============================================================================== 5) smart models: MAE/MFE regression + entry model
def _forest_mean_std(model, arr32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and dispersion (std across trees) of a fitted forest's prediction."""
    preds = np.stack([t.predict(arr32, check_input=False) for t in model.estimators_])
    return preds.mean(0), preds.std(0)


def _smart_frame(sel: ScreenSelection, cfg: TradeConfig, sig: pd.DataFrame, idxs: np.ndarray, sides: np.ndarray,
                 of: pd.DataFrame | None) -> pd.DataFrame:
    """Decision features at event bars: spread state + the isolated per-leg order-flow features of orderflow_features()."""
    z = sig["z"].to_numpy()
    L = int(cfg.smart_lookback_bars)
    zs = np.r_[np.full(L, np.nan), z[:-L]] if len(z) > L else np.full(len(z), np.nan)
    vol = sig["vol"].to_numpy()
    cols: dict[str, np.ndarray] = {
        "side": sides.astype(float),
        "abs_z": np.abs(z[idxs]),
        "dz_side": sides * (z[idxs] - zs[idxs]),                              # >0: z already moving the trade's way
        "dev_sigma": sig["dev"].to_numpy()[idxs] / sel.asset_std,
        "log_vol": np.log(np.maximum(vol[idxs] / sel.asset_std, 1e-12)),
    }
    if cfg.feature_score_mode in ("feature", "both"):
        cols["feat_score_side"] = sides * sig["feat_score"].to_numpy()[idxs]
    if of is not None and not of.empty:
        for c in of.columns:
            cols[c] = of[c].to_numpy(float)[idxs]
    return pd.DataFrame(cols)


def _decision_matrix(Xc: np.ndarray, m_mae, s_mae, m_mfe, s_mfe, cfg: TradeConfig, lo: float, hi: float):
    """Entry-model input = base features + the regression's outputs and its tree dispersion (its own uncertainty).
    Returns (matrix, stop_pct) with stop_pct = clip(exp(pred MAE)) * barrier_size."""
    stop = np.clip(np.exp(m_mae), lo, hi) * cfg.barrier_size
    reach = m_mfe - np.log(cfg.rr * stop)            # >0: the predicted MFE exceeds the take-profit distance
    return np.hstack([Xc, np.column_stack([m_mae, m_mfe, s_mae, s_mfe, reach])]), stop


_EXTRA_COLS = ["pred_log_mae", "pred_log_mfe", "unc_mae", "unc_mfe", "reach"]


@dataclass
class RLPolicy:
    """Logistic policy pi(enter | features) trained by policy gradient to maximise the MEAN R PER EXECUTED trade."""

    cols: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    hurdle: float = 0.0
    epochs: int = 0
    val_j: float = float("nan")    # hold-out expectancy per executed trade at the chosen epoch
    abstain: bool = False          # True: the hold-out showed no positive expectancy, so the agent stays out of the market

    def prob(self, X: np.ndarray) -> np.ndarray:
        z = ((np.asarray(X, float)[:, self.cols] - self.mu) / self.sd) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _rl_run(Z: np.ndarray, R: np.ndarray, epochs: int, lr: float, l2: float, min_cov: float, checkpoints: int = 0):
    """Full-batch Adam ascent on the EXACT expected-reward policy gradient of a one-step MDP (state = event features,
    action in {skip, enter}, reward = R if entered).  The per-trade objective  J = E[pi R] / E[pi]  is a fractional programme:
    Dinkelbach turns it into  max E[pi (R - kappa)]  with kappa <- J(pi) refreshed every few epochs (kappa = the policy's current
    expectancy, so an event is entered only when it beats the average trade it would be mixed with).  The coverage floor is HARD:
    after every step the bias is projected so that at least min_cov of the training events are entered (logit >= 0), otherwise
    the fractional objective collapses onto the one luckiest trade."""
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    m_w, v_w, m_b, v_b = np.zeros(d), np.zeros(d), 0.0, 0.0
    kappa = float(R.mean())
    hist = []
    for ep in range(1, epochs + 1):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        if ep % 5 == 1:
            kappa = float((p * R).sum() / (p.sum() + 1e-9))
        g = p * (1 - p) * (R - kappa) / n
        gw, gb = Z.T @ g - 2.0 * l2 * w, float(g.sum())
        m_w, v_w = 0.9 * m_w + 0.1 * gw, 0.999 * v_w + 0.001 * gw * gw
        m_b, v_b = 0.9 * m_b + 0.1 * gb, 0.999 * v_b + 0.001 * gb * gb
        c1, c2 = 1 - 0.9 ** ep, 1 - 0.999 ** ep
        w = w + lr * (m_w / c1) / (np.sqrt(v_w / c2) + 1e-8)
        b = b + lr * (m_b / c1) / (np.sqrt(v_b / c2) + 1e-8)
        zw = Z @ w
        if float(((zw + b) >= 0).mean()) < min_cov:                       # projection onto {coverage >= min_cov}
            b = -float(np.quantile(zw, 1.0 - min_cov)) + 1e-9
        if checkpoints and ep % checkpoints == 0:
            hist.append((ep, w.copy(), b))
    return w, b, kappa, hist


def train_rl_policy(X: np.ndarray, R: np.ndarray, entry: np.ndarray, exit_: np.ndarray, *, epochs: int = 300, lr: float = 0.05,
                    l2: float = 1e-2, min_cov: float = 0.10, max_features: int = 30) -> RLPolicy:
    """Features are standardised, the top `max_features` by |rank correlation with R| are kept, the number of epochs is chosen on a
    purged chronological hold-out (last 25% of events) by the hold-out expectancy per executed trade, then the policy is refit on all."""
    X = np.asarray(X, float)
    R = np.asarray(R, float)
    lo_, hi_ = np.quantile(R, [0.01, 0.99])
    Rw = np.clip(R, lo_, hi_)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Zall = (X - mu) / sd
    rc = pd.DataFrame(Zall).rank().corrwith(pd.Series(Rw).rank()).abs().fillna(0.0).to_numpy()
    cols = np.sort(np.argsort(-rc)[: max(1, min(max_features, X.shape[1]))])
    Z = Zall[:, cols]
    n = len(R)
    best_ep = epochs
    best_j, validated = -np.inf, False
    cut = int(n * 0.75)
    if n >= 60 and cut >= 30:
        tr = exit_ < entry[cut]
        if tr.sum() >= 30:
            validated = True
            _, _, _, hist = _rl_run(Z[tr], Rw[tr], epochs, lr, l2, min_cov, checkpoints=max(epochs // 15, 1))
            for ep, w, b in hist:
                p = 1.0 / (1.0 + np.exp(-np.clip(Z[cut:] @ w + b, -30, 30)))
                if (p >= 0.5).mean() < 0.5 * min_cov:
                    continue
                j = float((p * Rw[cut:]).sum() / (p.sum() + 1e-9))
                if j > best_j:
                    best_j, best_ep = j, ep
    w, b, kappa, _ = _rl_run(Z, Rw, best_ep, lr, l2, min_cov)
    return RLPolicy(cols, mu[cols], sd[cols], w, float(b), float(kappa), int(best_ep),
                    float(best_j) if np.isfinite(best_j) else float("nan"), bool(validated and not best_j > 0))


@dataclass
class SmartModels:
    """The fitted smart-trading stack of ONE fold."""

    mode: str
    columns: list[str]
    medians: pd.Series
    mae: Any                       # RandomForestRegressor on log(MAE as fraction of gross notional)
    mfe: Any                       # same for the MFE
    stop_lo: float                 # clip of the predicted MAE (train 5% / 95% quantiles of the realised MAE)
    stop_hi: float
    clf: Any = None
    policy: RLPolicy | None = None
    threshold: float = 0.5
    diag: dict = field(default_factory=dict)

    def _clean32(self, X: pd.DataFrame) -> np.ndarray:
        Xc, _ = _clean(X.reindex(columns=self.columns), self.medians)
        return Xc[self.columns].to_numpy(float)

    def threshold_for(self, stop_pct: np.ndarray, cfg: TradeConfig) -> np.ndarray:
        """P(win) needed per trade.  Explicit cfg.smart_threshold wins; otherwise the COST-AWARE breakeven: with round-trip cost
        c = 2 * cost_rate / stop_pct (in R) a win pays rr - c and a loss costs 1 + c, so E[R] > 0  <=>  p > (1 + c) / (1 + rr)."""
        if cfg.smart_threshold is not None:
            return np.full(len(stop_pct), float(cfg.smart_threshold))
        c = 2.0 * cfg.cost_rate / np.maximum(stop_pct, 1e-12)
        return (1.0 + c) / (1.0 + cfg.rr) + cfg.smart_margin

    def decide(self, X: pd.DataFrame, cfg: TradeConfig) -> dict[str, np.ndarray]:
        """stop_pct = predicted MAE * barrier_size (fraction of gross notional), accept = the entry model's decision."""
        Xc = self._clean32(X)
        x32 = Xc.astype(np.float32)
        m_mae, s_mae = _forest_mean_std(self.mae, x32)
        m_mfe, s_mfe = _forest_mean_std(self.mfe, x32)
        D, stop = _decision_matrix(Xc, m_mae, s_mae, m_mfe, s_mfe, cfg, self.stop_lo, self.stop_hi)
        if self.mode == "classifier":
            p = self.clf.predict_proba(D)[:, 1]
            acc = p >= self.threshold_for(stop, cfg)
        else:
            p = self.policy.prob(D)
            acc = (p >= 0.5) & (not self.policy.abstain)
        return dict(stop_pct=stop, p=p, accept=acc, mae_pct=np.exp(m_mae), mfe_pct=np.exp(m_mfe))


@dataclass
class _Stage:
    """Everything of the smart stack that does not depend on rr / barrier_size / smart_mode (so it is fitted once per selection)."""

    sel: ScreenSelection
    idxs: np.ndarray
    sides: np.ndarray
    entry: np.ndarray
    exit_: np.ndarray
    Xc: pd.DataFrame
    medians: pd.Series
    m_mae: np.ndarray
    s_mae: np.ndarray
    m_mfe: np.ndarray
    s_mfe: np.ndarray
    reg_mae: Any
    reg_mfe: Any
    lo: float
    hi: float
    mae_lab: np.ndarray
    closeA: np.ndarray
    hi_arr: np.ndarray | None
    lo_arr: np.ndarray | None
    T: int


def _oof_forest(X32: np.ndarray, y: np.ndarray, entry: np.ndarray, exit_: np.ndarray, k: int, params: dict):
    """Out-of-fold prediction (mean, tree std) with PURGED time blocks: training events whose [entry, exit] touches the test block
    are dropped.  Blocks that end up with too little training data fall back to the full-sample model."""
    from sklearn.ensemble import RandomForestRegressor
    n = len(y)
    mu, sd = np.full(n, np.nan), np.full(n, np.nan)
    edges = np.linspace(0, n, k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        tr = np.flatnonzero((exit_ < entry[a]) | (entry > exit_[a:b].max()))
        if len(tr) < 20:
            continue
        m = RandomForestRegressor(**params).fit(X32[tr], y[tr])
        mu[a:b], sd[a:b] = _forest_mean_std(m, X32[a:b])
    full = RandomForestRegressor(**params).fit(X32, y)
    fm, fs = _forest_mean_std(full, X32)
    miss = ~np.isfinite(mu)
    mu[miss], sd[miss] = fm[miss], fs[miss]
    return mu, sd, full


def _stage_key(cfg: TradeConfig) -> tuple:
    return (cfg.zscore_mode, cfg.zscore_window, cfg.vol_window, cfg.entry_z, cfg.smart_event_z, cfg.smart_event_stride,
            cfg.smart_lookback_bars, cfg.struct_tp_z, cfg.struct_sl_z, cfg.hilo_extremes, cfg.barrier_mode, cfg.max_hold_bars,
            cfg.ou_time_barrier, cfg.ou_hold_mult, repr(sorted(cfg.reg_params.items())), cfg.seed, cfg.smart_oof_folds,
            cfg.signal_spread, cfg.strategy_side, cfg.feature_score_mode, cfg.orderflow_names, cfg.stop_horizon_bars)


def _fit_stage(sel: ScreenSelection, train_panel, cfg: TradeConfig, hilo: HiLo | None) -> _Stage | None:
    """Structural trades on the train window -> MAE / MFE labels -> tree regression (out-of-fold + final)."""
    pan = _as_panel(train_panel)
    close = _close_frame(pan, sel.symbols)
    hi, lo = _barrier_arrays(close, cfg, hilo)
    sig = _signals(pan, close, sel, cfg, use_seed=False)
    n = len(close)
    T = _time_barrier(sel, cfg)
    z = sig["z"].to_numpy()
    ez = cfg.smart_event_z or cfg.entry_z
    ev = structural_trades(sig["asset_spread"].to_numpy(float), z, sig["mu"].to_numpy(float), sig["sd"].to_numpy(float), 0, n,
                           ez, cfg.struct_tp_z, cfg.struct_sl_z, T, sequential=False, stride=cfg.smart_event_stride)
    ev = [t for t in ev if t[2] != "end" and np.isfinite(sig["vol"].iloc[t[0]])]
    if len(ev) < cfg.smart_min_events:
        warnings.warn(f"smart trading skipped: {len(ev)} structural events (need >= {cfg.smart_min_events})")
        return None
    idxs = np.array([t[0] for t in ev])
    exit_ = np.array([t[3] for t in ev])
    sides = np.where(z[idxs] < 0, 1, -1)
    gw = sel.gross_weight
    mae, mfe = np.array([t[5] for t in ev]) / gw, np.array([t[6] for t in ev]) / gw
    if cfg.side_sign < 0:                          # inverted strategy: the position we hold sees the opposite excursions
        mae, mfe = mfe, mae
    mae, mfe = np.maximum(mae, 1e-6), np.maximum(mfe, 1e-6)
    of = orderflow_features(pan, sel.symbols, names=cfg.orderflow_names, lookback=cfg.smart_lookback_bars)
    Xdf = _smart_frame(sel, cfg, sig, idxs, sides, of)
    Xc, med = _clean(Xdf)
    X32 = Xc.to_numpy(np.float32)
    params = dict(cfg.reg_params)
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", cfg.smart_n_jobs)
    m_mae, s_mae, reg_mae = _oof_forest(X32, np.log(mae), idxs, exit_, cfg.smart_oof_folds, params)
    m_mfe, s_mfe, reg_mfe = _oof_forest(X32, np.log(mfe), idxs, exit_, cfg.smart_oof_folds, params)
    lo_, hi_ = (float(v) for v in np.quantile(mae, [0.05, 0.95]))
    return _Stage(sel, idxs, sides, idxs, exit_, Xc, med, m_mae, s_mae, m_mfe, s_mfe, reg_mae, reg_mfe, lo_, hi_, mae,
                  close.to_numpy(float), hi, lo, T)


def _finish_smart(st: _Stage, cfg: TradeConfig) -> SmartModels | None:
    """Entry model on the outcomes of trades that use the regression's OUT-OF-FOLD stops (so its labels contain the regression's
    errors): realised net R of every event, win = R > 0.  classifier -> P(win);  rl -> policy gradient on mean R per trade."""
    sel = st.sel
    D, stops = _decision_matrix(st.Xc.to_numpy(float), st.m_mae, st.s_mae, st.m_mfe, st.s_mfe, cfg, st.lo, st.hi)
    keep, Rn = [], []
    for k, i in enumerate(st.idxs):
        lv = levels_from_stop_pct(float(stops[k]), cfg)
        if lv is None:
            continue
        exec_side = int(st.sides[k]) * cfg.side_sign
        q = _legs(st.closeA[i], sel, exec_side, lv.gross_notional)
        res = _resolve(st.closeA, int(i), q, lv.stop_dollars, lv.tp_dollars, st.T, st.hi_arr, st.lo_arr, cfg.hilo_extremes)
        if res is None or res[1] == "window_end":
            continue
        keep.append(k)
        Rn.append(res[2] / lv.stop_dollars - 2.0 * cfg.cost_rate / lv.stop_pct)
    if len(keep) < cfg.smart_min_events:
        warnings.warn(f"smart trading skipped: {len(keep)} resolved events (need >= {cfg.smart_min_events})")
        return None
    keep, R = np.array(keep), np.array(Rn)
    Dk, ent, ex = D[keep], st.entry[keep], st.exit_[keep]
    win = (R > 0).astype(int)
    diag = dict(n_events=int(len(st.idxs)), n_labelled=int(len(keep)), base_win=float(win.mean()), mean_r_all=float(R.mean()),
                mae_med=float(np.median(st.mae_lab)), val_auc=float("nan"), rl_epochs=0, rl_hurdle=float("nan"))
    columns = list(st.Xc.columns)
    mdl = SmartModels(cfg.smart_mode, columns, st.medians, st.reg_mae, st.reg_mfe, st.lo, st.hi, diag=diag)
    if cfg.smart_mode == "classifier":
        from sklearn.ensemble import RandomForestClassifier
        if len(set(win)) < 2:
            warnings.warn("smart classifier skipped: only one outcome class among the training events")
            return None
        params = dict(cfg.clf_params)
        params.setdefault("random_state", cfg.seed)
        params.setdefault("n_jobs", cfg.smart_n_jobs)
        if cfg.smart_diagnostics:
            from sklearn.metrics import roc_auc_score
            cut = ent[int(len(ent) * 0.75)]
            tr, va = ex < cut, ent >= cut
            if tr.sum() >= cfg.smart_min_events // 2 and va.sum() >= 10 and len(set(win[tr])) == 2 and len(set(win[va])) == 2:
                probe = RandomForestClassifier(**params).fit(Dk[tr], win[tr])
                diag["val_auc"] = float(roc_auc_score(win[va], probe.predict_proba(Dk[va])[:, 1]))
        mdl.clf = RandomForestClassifier(**params).fit(Dk, win)
        mdl.threshold = float(cfg.smart_threshold) if cfg.smart_threshold is not None else float("nan")   # nan = cost-aware per trade
    else:
        mdl.policy = train_rl_policy(Dk, R, ent, ex, epochs=cfg.rl_epochs, lr=cfg.rl_lr, l2=cfg.rl_l2,
                                     min_cov=cfg.rl_min_coverage, max_features=cfg.rl_max_features)
        diag["rl_epochs"], diag["rl_hurdle"] = mdl.policy.epochs, mdl.policy.hurdle
        diag["rl_val_j"], diag["rl_abstain"] = mdl.policy.val_j, mdl.policy.abstain
        mdl.threshold = 0.5
    mdl.columns = columns
    return mdl


def fit_smart_models(sel: ScreenSelection, train_panel, cfg: TradeConfig, hilo: HiLo | None = None, *,
                     stage_cache: dict | None = None) -> SmartModels | None:
    """Fit the smart stack on the train window.  `stage_cache` shares the MAE/MFE regression between configs that only differ in
    rr / barrier_size / smart_mode / costs (the regression is the expensive part)."""
    key = (id(sel), _stage_key(cfg))
    hit = stage_cache.get(key) if stage_cache is not None else None
    if hit is not None:
        stage = hit[1]
    else:
        stage = _fit_stage(sel, train_panel, cfg, hilo)
        if stage_cache is not None:
            stage_cache[key] = (sel, stage)      # keep `sel` alive so id() cannot be reused
    return None if stage is None else _finish_smart(stage, cfg)


# =============================================================================== 6) trade: triple-barrier entry/exit rule
def trade(prices: pd.DataFrame, selection: ScreenSelection, cfg: TradeConfig | None = None,
          panel: AssetPanel | None = None, hilo: HiLo | None = None) -> pd.DataFrame:
    """Triple-barrier mean-reversion rule.  Returns base-asset quantity at every bar.

    Entry (bar close): |z| >= entry_z, armed (z has been back inside entry_z since the last trade), and
        non-smart:  feature-sign filter [feature_score_mode], optional OU edge ratio; stop from stoploss()
        smart:      feature-sign filter + the entry model (classifier / RL) says yes; stop_pct = predicted MAE * barrier_size
    Direction: z>0 -> short the spread, z<0 -> long it (strategy_side=2 inverts the executed direction only).
    Exit: stop-loss / take-profit (rr * stop) / time barrier, whichever comes first (hilo: intra-bar touch, fill at the barrier).
    Trade metadata is attached as .attrs['trade_meta'], barrier-fill PnL adjustments as .attrs['pnl_adjust']."""
    cfg = cfg or TradeConfig()
    close = _close_frame(prices, selection.symbols)
    if panel is None and isinstance(prices, AssetPanel):
        panel = prices
    hi, lo = _barrier_arrays(close, cfg, hilo)
    sig = _signals(panel, close, selection, cfg)
    n = len(close)
    closeA = close.to_numpy(float)
    qty = np.zeros((n, len(selection.symbols)))
    z, vol, dev, fs = sig["z"].to_numpy(), sig["vol"].to_numpy(), sig["dev"].to_numpy(), sig["feat_score"].to_numpy()
    T = _time_barrier(selection, cfg)
    phi = float(np.clip(selection.phi_asset, 0.0, 1.0)) if np.isfinite(selection.phi_asset) else 1.0
    info: dict[int, dict] = {}
    cand = _candidate_bars(sig, cfg.entry_z, n)
    smart = None
    if cfg.smart_trading:
        if panel is None:
            raise ValueError("smart_trading needs the AssetPanel (order-flow features), not a price DataFrame")
        smart = selection.smart
        if smart is None:
            warnings.warn("smart_trading=True but the selection has no fitted SmartModels: no trades in this window")
            cand = cand[:0]
    if len(cand):
        sides = np.where(z[cand] < 0, 1, -1)
        if smart is not None:
            of = orderflow_features(panel, selection.symbols, names=cfg.orderflow_names, lookback=cfg.smart_lookback_bars)
            dec = smart.decide(_smart_frame(selection, cfg, sig, cand, sides, of), cfg)
        for k, i in enumerate(cand):
            i, side = int(i), int(sides[k])
            if cfg.feature_score_mode in ("filter", "both") and selection.feature_terms and not side * fs[i] >= cfg.min_feature_score:
                continue
            if smart is not None:
                if not dec["accept"][k]:
                    continue
                lv = levels_from_stop_pct(float(dec["stop_pct"][k]), cfg)
                extra = dict(p_enter=float(dec["p"][k]), mae_pred=float(dec["mae_pct"][k]))
            else:
                lv = _levels(float(vol[i]), selection, cfg)
                extra = dict(p_enter=float("nan"), mae_pred=float("nan"))
                if lv is not None and cfg.ou_min_edge_ratio is not None:
                    if abs(dev[i]) * (1.0 - phi ** T) / (lv.stop_pct * selection.gross_weight) < cfg.ou_min_edge_ratio:
                        continue
            if lv is None:
                continue
            info[i] = dict(side=side, lv=lv, **extra)

    meta: dict[Any, dict] = {}
    adjust: dict[Any, float] = {}
    armed = True
    i = 0
    while i < n:
        zt = z[i]
        if not np.isfinite(zt):
            i += 1
            continue
        if abs(zt) < cfg.entry_z:
            armed = True
            i += 1
            continue
        if not armed or i not in info:
            i += 1
            continue
        d = info[i]
        lv: StopLevels = d["lv"]
        exec_side = d["side"] * cfg.side_sign
        q = _legs(closeA[i], selection, exec_side, lv.gross_notional)
        res = _resolve(closeA, i, q, lv.stop_dollars, lv.tp_dollars, T, hi, lo, cfg.hilo_extremes)
        if res is None:
            i += 1
            continue
        exit_i, reason, fill_pnl, close_pnl = res
        if reason == "window_end" and not cfg.close_at_window_end:
            qty[i:n, :] = q
            exit_i = n
        else:
            qty[i:exit_i, :] = q
            if reason in ("sl", "tp") and fill_pnl != close_pnl:
                adjust[close.index[exit_i]] = fill_pnl - close_pnl
        meta[close.index[i]] = dict(side=exec_side, strategy_side=d["side"], reason=reason, leverage=lv.leverage,
                                    stop_pct=lv.stop_pct, tp_pct=lv.tp_pct, risk_dollars=lv.stop_dollars,
                                    p_enter=d["p_enter"], mae_pred=d["mae_pred"], capped=lv.capped)
        armed = bool(exit_i < n and abs(z[exit_i]) < cfg.entry_z)
        i = exit_i + 1
    out = pd.DataFrame(qty, index=close.index, columns=list(selection.symbols))
    out.attrs["trade_meta"] = meta
    out.attrs["pnl_adjust"] = adjust
    return out


def _trade_ledger(quantities: pd.DataFrame, net_pnl: pd.Series, cfg: TradeConfig, meta: dict | None = None) -> pd.DataFrame:
    """One row per trade (a run of non-zero quantities plus the exit bar)."""
    meta = meta or {}
    n = len(quantities)
    if n == 0:
        return pd.DataFrame([])
    active = quantities.abs().sum(axis=1).to_numpy() > 1e-12
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    cs = np.cumsum(net_pnl.to_numpy(float))
    index = quantities.index
    rows: list[dict[str, Any]] = []
    for trade_no, (s, e) in enumerate(zip(starts, stops), start=1):
        open_end = e >= n
        end_i = n - 1 if open_end else int(e)
        pnl = float(cs[end_i] - (cs[s - 1] if s > 0 else 0.0))
        m = meta.get(index[s], {})
        risk = float(m.get("risk_dollars", cfg.risk_dollars))
        r = {"trade": trade_no, "entry_time": index[s], "exit_time": index[end_i], "bars_held": int(end_i - s),
             "side": m.get("side", np.nan), "strategy_side": m.get("strategy_side", np.nan),
             "reason": m.get("reason", "open_at_end" if open_end else ""), "leverage": m.get("leverage", np.nan),
             "stop_pct": m.get("stop_pct", np.nan), "p_enter": m.get("p_enter", np.nan), "mae_pred": m.get("mae_pred", np.nan),
             "pnl": pnl, "risk_dollars": risk, "r_multiple": pnl / risk}
        if open_end:
            r["open_at_end"] = True
        rows.append(r)
    return pd.DataFrame(rows)


def _accepts(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def Forward(selection: ScreenSelection, forward_prices: AssetPanel | pd.DataFrame, *, cfg: TradeConfig | None = None,
            trade_fn: Callable[..., pd.DataFrame] | None = None, hilo: HiLo | None = None) -> ForwardResult:
    """Run the frozen selection over one forward window.  PnL = previous-bar base units x price change (the spread is NOT used)."""
    cfg = cfg or TradeConfig()
    pan = forward_prices if isinstance(forward_prices, AssetPanel) else None
    close = _close_frame(forward_prices, selection.symbols)
    if pan is None and selection.feature_terms:
        raise ValueError("the selected spread has feature terms; Forward needs an AssetPanel with features")
    sig = _signals(pan, close, selection, cfg)
    spread, z = sig["asset_spread"].rename("spread_diagnostic"), sig["z"].rename("zscore")
    fn = trade if trade_fn is None else trade_fn
    kw: dict[str, Any] = {}
    if _accepts(fn, "panel"):
        kw["panel"] = pan
    if hilo is not None and _accepts(fn, "hilo"):
        kw["hilo"] = hilo
    quantities = fn(close.copy(), selection, cfg, **kw)
    if not isinstance(quantities, pd.DataFrame):
        raise TypeError("trade_fn must return a DataFrame of base-asset quantities")
    meta = dict(getattr(quantities, "attrs", {}).get("trade_meta", {}))
    adjust = dict(getattr(quantities, "attrs", {}).get("pnl_adjust", {}))
    quantities = quantities.reindex(index=close.index)
    unknown = [c for c in quantities.columns if c not in selection.symbols]
    if unknown:
        raise ValueError(f"trade_fn returned quantities for unselected assets: {unknown}")
    quantities = quantities.reindex(columns=list(selection.symbols), fill_value=0.0).astype(float)
    quantities = quantities.replace([np.inf, -np.inf], np.nan)
    if quantities.isna().any().any():
        raise ValueError("trade_fn returned NaN or infinite quantities")
    if cfg.close_at_window_end and len(quantities):
        quantities.iloc[-1, :] = 0.0
    dprice = close.diff().fillna(0.0)
    gross_pnl = (quantities.shift(1).fillna(0.0) * dprice).sum(axis=1)
    if adjust:                                   # barrier fills: replace the exit bar's close-marked PnL by the fill
        pos = close.index.get_indexer(list(adjust.keys()))
        vals = np.array(list(adjust.values()), dtype=float)
        good = pos >= 0
        adj = np.zeros(len(close))
        np.add.at(adj, pos[good], vals[good])
        gross_pnl = gross_pnl + adj
    dqty = quantities.diff()
    if len(dqty):
        dqty.iloc[0] = quantities.iloc[0]
    costs = (dqty.abs() * close).sum(axis=1) * cfg.cost_rate
    net_pnl = (gross_pnl - costs).rename("net_pnl")
    gross_pnl, costs = gross_pnl.rename("gross_asset_pnl"), costs.rename("trading_costs")
    ledger = _trade_ledger(quantities, net_pnl, cfg, meta)
    try:
        oos_stat, _ = adf_stat(spread.to_numpy(float), maxlag=12, const=True)
        oos_p = adf_p(oos_stat, 1)
        oos_hl = half_life(spread.to_numpy(float), floor=1.0)
        oos_vr = vr_mean(spread.to_numpy(float), (5, 15, 30, 60, 120))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        oos_stat = oos_p = oos_hl = oos_vr = float("nan")
    return ForwardResult(selection, close, spread, z, quantities, gross_pnl, costs, net_pnl, ledger,
                         float(oos_stat), float(oos_p), float(oos_hl), float(oos_vr))


# =============================================================================== 7) folds / flip-flop / Algorithm
def _sortino(pnl: pd.Series, bars_per_year: float) -> float:
    r = pnl.astype(float).to_numpy()
    if not len(r):
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside <= 1e-15:
        return float("inf") if float(np.mean(r)) > 0 else float("nan")
    return float(np.mean(r) / downside * np.sqrt(max(bars_per_year, 1.0)))


def _max_drawdown(pnl: pd.Series, capital: float) -> float:
    equity = capital + pnl.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else float("nan")


@dataclass(frozen=True)
class FoldPlan:
    """Bar positions (half-open, relative to the panel being walked) of one train -> forward cycle."""

    cycle: int
    instance: str          # "A" / "B" flip-flop slot
    train_start: int
    train_end: int
    forward_start: int
    forward_end: int


def plan_folds(n_bars: int, *, offset: int = 0, train_bars: int = 5_000, forward_bars: int = 5_000, purge_bars: int = 1,
               max_cycles: int | None = None) -> list[FoldPlan]:
    """Flip-flop schedule: cycle 0 trains on the first `train_bars`, then every next cycle trains on the window that was just
    traded, so one instance is always trading while the other is being (re)trained.  Each cycle consumes forward_bars + purge_bars."""
    plans: list[FoldPlan] = []
    prev: tuple[int, int] | None = None
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        if prev is None:
            ts = offset
            te = ts + train_bars
        else:
            if prev[1] - prev[0] < train_bars:
                raise ValueError("forward_bars must be at least train_bars so the next instance can train on the previous traded window")
            te = prev[1]
            ts = te - train_bars
        fs = te + purge_bars
        fe = fs + forward_bars
        if te > n_bars or fe > n_bars:
            break
        plans.append(FoldPlan(cycle, "A" if cycle % 2 == 0 else "B", ts, te, fs, fe))
        prev = (fs, fe)
        cycle += 1
    return plans


def beam_rank(train_panel: AssetPanel, y: str, discover_cfg: DiscoverConfig) -> list[Candidate]:
    """The expensive, rule-independent part of a screening: the ranked candidate list of the beam search."""
    cfg = discover_cfg if train_panel.feats else replace(discover_cfg, use_features=False)
    return create_pair(train_panel, y, cfg=cfg)


def retrain_rules(train_panel: AssetPanel, y: str, discover_cfg: DiscoverConfig, previous_selection: ScreenSelection | None = None,
                  *, ranked: Sequence[Candidate] | None = None) -> ScreenSelection:
    """Default retrain rule: beam search on the latest train window; z window = HALF_LIFE_MULTIPLIER x train half-life."""
    del previous_selection
    return Screener(train_panel, y, bars=len(train_panel.logp), cfg=discover_cfg, ranked=ranked)


def screen_fold(train_panel: AssetPanel, y: str, discover_cfg: DiscoverConfig, *, retrain_rule: Callable[..., ScreenSelection] | None = None,
                previous_selection: ScreenSelection | None = None, ranked: Sequence[Candidate] | None = None) -> ScreenSelection:
    """Beam search (or a precomputed `ranked` list) + the retrain rule -> frozen selection."""
    if retrain_rule is not None and hasattr(retrain_rule, "from_ranked"):
        sel = retrain_rule.from_ranked(train_panel, y, discover_cfg, ranked if ranked is not None else beam_rank(train_panel, y, discover_cfg))
    elif retrain_rule is not None:
        sel = retrain_rule(train_panel, y, discover_cfg, previous_selection)
    else:
        sel = retrain_rules(train_panel, y, discover_cfg, previous_selection, ranked=ranked)
    if not isinstance(sel, ScreenSelection):
        raise TypeError("retrain_rule must return ScreenSelection")
    if sel.target != y or not sel.hedge_symbols:
        raise ValueError("retrain_rule returned a target mismatch or a selection with no tradable hedge leg")
    return sel


def trade_fold(selection: ScreenSelection, train_panel: AssetPanel, forward_panel: AssetPanel, plan: FoldPlan, tcfg: TradeConfig, *,
               hilo: HiLo | None = None, trade_fn: Callable[..., pd.DataFrame] | None = None,
               stage_cache: dict | None = None) -> FoldResult:
    """The TradeConfig-dependent half of a fold: (smart_trading) fit the MAE/MFE regression + entry model on the train window,
    then trade the forward window.  `hilo` should cover both windows."""
    if tcfg.smart_trading:
        selection = replace(selection, smart=fit_smart_models(selection, train_panel, tcfg, hilo, stage_cache=stage_cache))
    fwd = Forward(selection, forward_panel, cfg=tcfg, trade_fn=trade_fn, hilo=hilo)
    return FoldResult(plan.cycle, plan.instance, train_panel.logp.index[0], train_panel.logp.index[-1],
                      forward_panel.logp.index[0], forward_panel.logp.index[-1], selection, fwd)


def run_fold(train_panel: AssetPanel, forward_panel: AssetPanel, y: str, plan: FoldPlan, discover_cfg: DiscoverConfig,
             tcfg: TradeConfig, *, hilo: HiLo | None = None, trade_fn: Callable[..., pd.DataFrame] | None = None,
             retrain_rule: Callable[..., ScreenSelection] | None = None, previous_selection: ScreenSelection | None = None) -> FoldResult:
    sel = screen_fold(train_panel, y, discover_cfg, retrain_rule=retrain_rule, previous_selection=previous_selection)
    return trade_fold(sel, train_panel, forward_panel, plan, tcfg, hilo=hilo, trade_fn=trade_fn)


def parallel_map(fn: Callable, jobs: Sequence[dict], n_jobs: int = 1, backend: str = "loky", verbose: int = 0) -> list:
    """fn(**job) for every job, in order, on `n_jobs` workers (-1 = all cores).  backend: "loky" (processes, default: the beam
    search is Python/numpy heavy so processes scale) | "threading".  fn must be a module-level function for processes."""
    jobs = list(jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    n_jobs = min(n_jobs, max(len(jobs), 1))
    if n_jobs <= 1:
        return [fn(**kw) for kw in jobs]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        Ex = ThreadPoolExecutor if backend == "threading" else ProcessPoolExecutor
        with Ex(max_workers=n_jobs) as ex:
            return [f.result() for f in [ex.submit(fn, **kw) for kw in jobs]]
    tasks = (delayed(fn)(**kw) for kw in jobs)
    if backend == "loky":
        from joblib import parallel_config

        with parallel_config(backend=backend, n_jobs=n_jobs, inner_max_num_threads=1):
            return Parallel(n_jobs=n_jobs, verbose=verbose)(tasks)
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(tasks)


def _hilo_window(hilo: HiLo | None, *panels: AssetPanel) -> HiLo | None:
    if hilo is None:
        return None
    return hilo.between(min(p.logp.index[0] for p in panels), max(p.logp.index[-1] for p in panels))


def assemble_algorithm(folds: list[FoldResult], *, index: pd.Index, symbols: Sequence[str], bar_minutes: float, tcfg: TradeConfig,
                       train_bars: int, forward_bars: int, purge_bars: int, print_report: bool = True) -> AlgorithmResult:
    """Stitch fold results into one PnL series / trade ledger / report."""
    if not folds:
        raise ValueError("no folds to assemble")
    first, last = folds[0].forward.close.index[0], folds[-1].forward.close.index[-1]
    full_index = index[(index >= first) & (index <= last)]
    pnl = pd.Series(0.0, index=full_index, name="net_pnl")
    quantities = pd.DataFrame(0.0, index=full_index, columns=list(symbols))
    trade_frames, fold_rows = [], []
    for fold in folds:
        idx = fold.forward.net_pnl.index
        pnl.loc[idx] = pnl.loc[idx].to_numpy() + fold.forward.net_pnl.to_numpy()
        q = fold.forward.quantities
        quantities.loc[idx, q.columns] = q
        tr = fold.forward.trades
        if not tr.empty:
            tr = tr.copy()
            tr.insert(0, "instance", fold.instance)
            tr.insert(0, "cycle", fold.cycle)
            tr.insert(0, "target", fold.selection.target)
            tr.insert(0, "hedge", " + ".join(fold.selection.hedge_symbols))
            trade_frames.append(tr)
        sm = fold.selection.smart
        reasons = tr["reason"] if len(tr) else pd.Series(dtype=object)
        fold_rows.append({
            "cycle": fold.cycle, "instance": fold.instance, "hedge": " + ".join(fold.selection.hedge_symbols),
            "feature_terms": ", ".join(fold.selection.feature_terms),
            "vr_train": fold.selection.candidate.vr, "hl_train": fold.selection.half_life_asset, "hl_pred": fold.selection.hl_pred,
            "z_window": fold.selection.z_window, "vr_oos": fold.forward.vr_oos, "hl_oos": fold.forward.half_life_oos,
            "p_oos": fold.forward.p_oos, "trades": len(tr),
            "tp_rate": float((reasons == "tp").mean()) if len(tr) else np.nan,
            "sl_rate": float((reasons == "sl").mean()) if len(tr) else np.nan,
            "time_rate": float((reasons == "time").mean()) if len(tr) else np.nan,
            "gross_pnl": float(fold.forward.gross_pnl.sum()), "net_pnl": float(fold.forward.net_pnl.sum()),
            "smart_events": sm.diag.get("n_labelled", np.nan) if sm else np.nan,
            "smart_val_auc": sm.diag.get("val_auc", np.nan) if sm else np.nan,
        })
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(
        columns=["cycle", "instance", "target", "hedge", "trade", "entry_time", "exit_time", "bars_held", "side", "strategy_side",
                 "reason", "leverage", "stop_pct", "p_enter", "mae_pred", "pnl", "risk_dollars", "r_multiple"])
    total_pnl = float(pnl.sum())
    total_r = float(trades["r_multiple"].sum()) if len(trades) else 0.0
    expectancy = float(trades["r_multiple"].mean()) if len(trades) else float("nan")
    bpy = 365.25 * 24.0 * 60.0 / bar_minutes if np.isfinite(bar_minutes) and bar_minutes > 0 else 1.0
    sortino, max_dd = _sortino(pnl, bpy), _max_drawdown(pnl, tcfg.capital)
    gross_total = float(sum(f.forward.gross_pnl.sum() for f in folds))
    cost_total = float(sum(f.forward.costs.sum() for f in folds))
    if len(trades):
        rs = trades["reason"]
        mix = " | ".join(f"{k} {float((rs == k).mean()):.1%}" for k in ("tp", "sl", "time", "window_end") if (rs == k).any())
        lev = pd.to_numeric(trades["leverage"], errors="coerce").mean()
    else:
        mix, lev = "n/a", np.nan
    mode = f"SMART ({tcfg.smart_mode}): stop = predicted MAE x {tcfg.barrier_size:g}" if tcfg.smart_trading else \
        f"formula: stop = {tcfg.barrier_size:g} x spread vol"
    report = (
        "PURGED WALK-FORWARD / FLIP-FLOP REPORT (triple barrier)\n"
        f"Folds: {len(folds)} | train={train_bars:,} bars | forward={forward_bars:,} bars | purge={purge_bars:,} bars\n"
        f"z-score: {tcfg.zscore_mode}" + (f" (window = {tcfg.half_life_multiplier:g} x half-life)" if tcfg.zscore_mode == "rolling" else " (train mean/std)")
        + f" | barriers: {tcfg.barrier_mode}\n"
        f"Risk rule: {mode} | TP = rr {tcfg.rr:g} x stop\n"
        f"Instances: A/B alternating | Closed trades: {len(trades):,} | Exits: {mix} | mean leverage: {lev:.2f}x\n"
        f"Gross asset PnL: {gross_total:,.4f} | Costs: {cost_total:,.4f} | Net asset PnL: {total_pnl:,.4f}\n"
        f"Accumulated R: {total_r:,.4f}R | Expectancy: {expectancy:,.4f}R/trade\n"
        f"Sortino (annualised, bar PnL): {sortino:,.4f} | Max drawdown: {max_dd:.2%}\n"
        f"Median forward VR: {np.nanmedian([f.forward.vr_oos for f in folds]):.3f} | "
        f"Median forward half-life: {np.nanmedian([f.forward.half_life_oos for f in folds]):.2f} bars")
    if print_report:
        print(report)
    return AlgorithmResult(folds, pnl, quantities, trades, total_pnl, total_r, expectancy, sortino, max_dd, report, pd.DataFrame(fold_rows))


def Algorithm(panel: AssetPanel | pd.DataFrame, y: str, *, start=None, train_bars: int = 5_000, forward_bars: int = 5_000,
              purge_bars: int = 1, discover_cfg: DiscoverConfig | None = None, trade_cfg: TradeConfig | None = None,
              trade_fn: Callable[..., pd.DataFrame] | None = None, retrain_rule: Callable[..., ScreenSelection] | None = None,
              hilo: HiLo | None = None, max_cycles: int | None = None, n_jobs: int = 1, print_report: bool = True) -> AlgorithmResult:
    """Two alternating strategy instances in a causal flip-flop schedule (see plan_folds).  n_jobs != 1 runs the folds in
    parallel processes (previous_selection is then None for every fold)."""
    if train_bars < 200 or forward_bars < 100:
        raise ValueError("train_bars must be >= 200 and forward_bars must be >= 100")
    if purge_bars < 0:
        raise ValueError("purge_bars must be >= 0")
    p = _as_panel(panel)
    offset = _start_offset(p, start)
    if offset >= len(p.logp):
        raise ValueError(f"start={start!r} is after the available panel")
    y = str(y)
    if y not in p.logp.columns:
        raise ValueError(f"target {y!r} is not in panel")
    dcfg = discover_cfg or DiscoverConfig()
    if dcfg.use_features and not p.feats:
        warnings.warn("panel carries no features: running the beam search price-only")
        dcfg = replace(dcfg, use_features=False)
    tcfg = trade_cfg or TradeConfig()
    if tcfg.barrier_mode == "hilo" and hilo is None:
        raise ValueError("barrier_mode='hilo' needs hilo=HiLo(...) (see load_hilo), or set TradeConfig(barrier_mode='close')")
    plans = plan_folds(len(p.logp), offset=offset, train_bars=train_bars, forward_bars=forward_bars, purge_bars=purge_bars,
                       max_cycles=max_cycles)
    if not plans:
        raise ValueError(f"not enough data for one train/forward cycle; need at least {train_bars + purge_bars + forward_bars} bars")

    def _slices(plan: FoldPlan):
        tp, fp = _slice_panel(p, plan.train_start, plan.train_end), _slice_panel(p, plan.forward_start, plan.forward_end)
        return tp, fp, _hilo_window(hilo, tp, fp)

    if n_jobs == 1 or len(plans) == 1:
        slots: dict[str, ScreenSelection | None] = {"A": None, "B": None}
        folds: list[FoldResult] = []
        for plan in plans:
            tp, fp, hl = _slices(plan)
            fold = run_fold(tp, fp, y, plan, dcfg, tcfg, hilo=hl, trade_fn=trade_fn, retrain_rule=retrain_rule,
                            previous_selection=slots[plan.instance])
            slots[plan.instance] = fold.selection
            folds.append(fold)
    else:
        jobs = []
        for plan in plans:
            tp, fp, hl = _slices(plan)
            jobs.append(dict(train_panel=tp, forward_panel=fp, y=y, plan=plan, discover_cfg=dcfg, tcfg=tcfg, hilo=hl,
                             trade_fn=trade_fn, retrain_rule=retrain_rule))
        folds = parallel_map(run_fold, jobs, n_jobs)
    return assemble_algorithm(folds, index=p.logp.index, symbols=list(p.logp.columns), bar_minutes=p.bar_minutes, tcfg=tcfg,
                              train_bars=train_bars, forward_bars=forward_bars, purge_bars=purge_bars, print_report=print_report)


__all__ = [
    "HALF_LIFE_MULTIPLIER", "ORDERFLOW_FEATURES", "TradeConfig", "StopLevels", "ScreenSelection", "ForwardResult", "FoldResult",
    "AlgorithmResult", "HiLo", "load_hilo", "resample_hilo", "orderflow_features", "orderflow_window_stats", "structural_trades",
    "DatasetConfig", "build_dataset", "candidate_features", "HalfLifePredictor", "HalfLifeRetrainRule", "compare_predictors",
    "Screener", "stoploss", "levels_from_stop_pct", "fit_smart_models", "SmartModels", "RLPolicy", "train_rl_policy", "trade",
    "Forward", "FoldPlan", "plan_folds", "beam_rank", "screen_fold", "trade_fold", "run_fold", "assemble_algorithm", "parallel_map",
    "retrain_rules", "Algorithm", "DEFAULT_BAR",
]

if __name__ == "__main__":
    print("strategy.py is a library.  Run backtest.py for the phase_1 / phase_2 backtests and test_alt3.py for the unit tests.")