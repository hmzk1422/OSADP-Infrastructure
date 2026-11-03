"""Triple-barrier stat-arb execution and walk-forward orchestration for spread_discoverer.

What changed versus the price-only version
------------------------------------------
* Commissions / slippage default to 0 so the raw edge is visible first (fee_bps, slippage_bps are still there).
* stoploss() is the single place that defines the risk rules.  It turns the volatility of the TRADED spread
  (asset legs only, no features) into a stop distance, a take-profit distance (rr * stop) and a leverage:
      stop_pct (of gross notional) = sl_vol_mult * barrier_size * spread_vol_pct
      leverage                     = risk_fraction / stop_pct          (capped by max_leverage)
      loss at stop = risk_fraction * capital,  profit at TP = rr * that
  e.g. risk 2%, spread vol 0.50% -> 4x, lose 2% / win 4%;  vol 0.10% -> 20x.
* Exits are a triple barrier: stop-loss, take-profit (rr * stop), time barrier (OU based, capped by max_hold_bars).
  There is no z-score exit any more.
* Entry rule = z-score beyond entry_z, then up to three gates:
    1. feature weighted-sum direction  sum_j coef_j * z_j  (features that sit inside the discovered spread)
    2. optional OU edge check (expected reversion over the time barrier vs stop distance)
    3. a tree (random forest) trained on the previous window to predict "this trade hits TP"
* Features only ever decide WHETHER / WHICH WAY to enter.  PnL is still traded asset quantities x asset price
  changes; feature terms in the spread are never traded.

Conventions that are unchanged
------------------------------
* spread (asset only) = log(P_target) - sum(beta_j * log(P_hedge_j));  z>0 -> short the spread, z<0 -> long it.
* Signals are observed at bar close, PnL starts on the next bar; quantities are fixed between entry and exit.
* Barriers default to INTRA-BAR high/low touches (TradeConfig.barrier_mode='hilo'): a stop/TP is hit as soon as the
  bar's high/low reaches it and fills exactly at the barrier, so a big candle can no longer blow through the stop.
  barrier_mode='close' keeps the legacy close-only check.  R uses the dollar risk set by stoploss() for that trade.
* z-score modes: 'static' (train-window mean/std) or 'rolling' (causal mean/std over zscore_window bars).
* All bar-denominated settings default to 1-minute bars; TradeConfig.for_bar(bar_minutes) rescales them.
* The Monte Carlo / stress test lives in mc_stress_test.py (this file only holds the strategy and the fold runners).

Expected repository layout:
    spread_discoverer.py   binance_vision.py   flow_features.py   strategy.py (this file)   mc_stress_test.py
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence, Any
import warnings

import numpy as np
import pandas as pd

try:  # package layout
    from .spread_discoverer import (
        AssetPanel,
        Candidate,
        DiscoverConfig,
        create_pair,
        fit,
        px,
        adf_stat,
        adf_p,
        half_life,
    )
except ImportError:  # flat folder
    from spread_discoverer import (
        AssetPanel,
        Candidate,
        DiscoverConfig,
        create_pair,
        fit,
        px,
        adf_stat,
        adf_p,
        half_life,
    )

try:
    from .spread_discoverer import get_assets
except ImportError:
    try:
        from spread_discoverer import get_assets
    except ImportError:
        get_assets = None

try:
    from .binance_vision import load_bars
except ImportError:
    try:
        from binance_vision import load_bars
    except ImportError:
        load_bars = None


# --------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class StopLevels:
    """Output of stoploss(): everything the trade needs to size and bracket a position."""

    stop_pct: float        # stop distance, fraction of GROSS notional
    tp_pct: float          # take-profit distance, fraction of gross notional
    leverage: float        # gross notional / capital
    gross_notional: float
    stop_dollars: float    # loss if the stop is hit
    tp_dollars: float
    capped: bool           # True when max_leverage limited the size (risk < risk_fraction)


DEFAULT_BAR = "1min"          # bar size used when a loader is called without an explicit `bar`


def _bars_per_day(bar_minutes: float) -> float:
    return 1440.0 / float(bar_minutes)


@dataclass
class TradeConfig:
    """Execution and signal rules. All price legs are traded; features only gate entries.

    Every *_bars setting is counted in BARS of the panel being traded.  The defaults describe 1-minute bars
    (1,440 bars = 1 day).  Use TradeConfig.for_bar(bar_minutes, ...) to get the same wall-clock settings on any
    other bar size (e.g. for_bar(5) gives vol_window=288, max_hold_bars=576, tree_lookback_bars=12).
    """

    # --- signal
    entry_z: float = 2.0
    signal_spread: str = "asset"        # "asset": z of the traded (price-only) spread | "full": z of the discovered spread incl. features
    zscore_mode: str = "static"         # "static": train-window mean/std | "rolling": causal mean/std of the last zscore_window bars
    zscore_window: int = 1_440          # bars; only used by zscore_mode="rolling" (seeded with the tail of the train window)

    # --- risk (see stoploss())
    capital: float = 10_000.0
    risk_fraction: float = 0.02         # max loss per trade as a fraction of capital
    rr: float = 2.0                     # take profit = rr * stop
    barrier_size: float = 1.0           # multiplies the stop distance (1 = standard). TP = rr * stop, so it scales both barriers; risk per trade stays risk_fraction
    sl_vol_mult: float = 1.0            # stop distance = sl_vol_mult * barrier_size * spread volatility over the stop horizon (leave at 1; use barrier_size)
    vol_window: int = 1_440             # bars for the rolling std of the asset-only spread changes
    stop_horizon_bars: int | None = None  # None -> OU half-life of the asset spread (clipped); volatility is scaled by sqrt(horizon)
    max_leverage: float = 100.0
    stop_fn: Callable[..., StopLevels | None] | None = None   # replaces stoploss() entirely if you want a different rule

    # --- strategy direction switch
    strategy_side: int = 1              # 1 = trade what the strategy says | 2 = INVERT it (strategy says long -> we short, and vice versa)

    # --- barrier evaluation
    barrier_mode: str = "hilo"          # "hilo": intra-bar high/low touch, fill AT the barrier | "close": legacy bar-close check
    hilo_extremes: str = "comove"       # how the legs' highs/lows combine inside one bar (see _resolve)

    # --- time barrier
    max_hold_bars: int = 2_880
    ou_time_barrier: bool = True        # time barrier = min(max_hold_bars, ou_hold_mult * OU half-life)
    ou_hold_mult: float = 3.0
    ou_min_edge_ratio: float | None = None   # optional gate: expected OU reversion over the time barrier / stop distance >= this

    # --- feature weighted-sum gate / feature
    feature_score_mode: str = "filter"  # "off" | "filter" (sign must agree with trade) | "feature" (tree input only) | "both"
    min_feature_score: float = 0.0      # |score| (in asset-spread sigmas) required in the trade direction when filtering

    # --- tree gate
    use_tree: bool = True
    tree_features: str = "all"          # "all": spread state + features/returns/vol of EVERY asset in the panel | "state": spread state only
    tree_event_z: float | None = None   # |z| level of training events; None -> entry_z
    tree_event_stride: int = 1          # use every k-th training event (consecutive bars are near-duplicates; k>1 is faster)
    tree_min_events: int = 60
    tree_lookback_bars: int = 60        # bars for the "z already moving my way" and short-return features (1 hour at 1m)
    tree_prob_threshold: float | None = None   # None -> breakeven win rate 1/(1+rr) + tree_margin
    tree_margin: float = 0.05
    tree_diagnostics: bool = True       # extra forest fit for a purged hold-out AUC; switch off for speed
    tree_n_jobs: int = -1               # forest threads (parallel runners set this to 1 to avoid oversubscription)
    tree_params: dict = field(default_factory=lambda: dict(
        n_estimators=200, max_depth=4, max_features=0.8, min_samples_leaf=20))
    seed: int = 7

    # --- costs (0 while looking for the raw edge)
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
        if self.rr <= 0 or self.sl_vol_mult <= 0 or self.barrier_size <= 0:
            raise ValueError("rr, sl_vol_mult and barrier_size must be > 0")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")
        if self.feature_score_mode not in ("off", "filter", "feature", "both"):
            raise ValueError("feature_score_mode must be off|filter|feature|both")
        if self.tree_features not in ("all", "state"):
            raise ValueError("tree_features must be 'all' or 'state'")
        if self.tree_event_stride < 1 or self.tree_lookback_bars < 1:
            raise ValueError("tree_event_stride and tree_lookback_bars must be >= 1")

    @classmethod
    def for_bar(cls, bar_minutes: float, **overrides) -> "TradeConfig":
        """Same wall-clock settings (1 day vol / z-score window, 2 day max hold, 1 hour tree lookback) on any bar size."""
        day = _bars_per_day(bar_minutes)
        base = dict(
            vol_window=max(10, int(round(day))),
            zscore_window=max(30, int(round(day))),
            max_hold_bars=max(2, int(round(2 * day))),
            tree_lookback_bars=max(1, int(round(60.0 / bar_minutes))),
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


# --------------------------------------------------------------------------- result objects
@dataclass
class TPModel:
    """Tree that predicts P(trade hits take-profit before stop / time barrier)."""

    clf: Any
    columns: list[str]
    medians: pd.Series
    n_events: int
    base_rate: float
    val_auc: float
    threshold: float


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
    # --- feature part of the spread
    feature_terms: tuple[str, ...] = ()
    feature_coefs: dict[str, float] = field(default_factory=dict)   # raw coefficient per ft| term
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_std: dict[str, float] = field(default_factory=dict)
    # --- traded (asset-only) spread statistics, train window
    asset_mean: float = 0.0
    asset_std: float = 1.0
    phi_asset: float = float("nan")        # AR(1) coefficient (OU discretisation)
    half_life_asset: float = float("nan")
    vol_seed: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)   # last train bar-to-bar asset-spread changes
    asset_seed: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)  # last train asset-spread levels (rolling z warm start)
    full_seed: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)   # last train full-spread levels (rolling z warm start)
    tp_model: TPModel | None = None
    # --- persistence.py hooks (all default to "behave exactly as before")
    z_window: int | None = None            # per-selection rolling z window (bars); None -> TradeConfig.zscore_window
    size_mult: float = 1.0                 # position-size multiplier from the persistence gate; 0 -> sit out this window
    gate_p: float = float("nan")           # raw gate score of the selected candidate (diagnostic)
    hl_pred: float = float("nan")          # predicted forward half-life (bars) of the selected candidate (diagnostic)

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.target,) + self.hedge_symbols

    @property
    def gross_weight(self) -> float:
        return 1.0 + sum(abs(self.betas[s]) for s in self.hedge_symbols)

    def feature_table(self) -> pd.DataFrame:
        """Standardised coefficient (coef * std) of every feature term in the spread."""
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


# --------------------------------------------------------------------------- panel / price utilities
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
        if t.tzinfo is None:
            t = t.tz_localize(tz)
        else:
            t = t.tz_convert(tz)
    elif t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def _start_offset(panel: AssetPanel, start=None) -> int:
    if start is None:
        return 0
    t = _timestamp_for_index(start, panel.logp.index)
    return int(panel.logp.index.searchsorted(t, side="left"))


def _close_frame(data: AssetPanel | pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Return positive close prices. AssetPanel.logp is converted back to prices."""
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


def load_price_panel(
    symbols: Sequence[str],
    start,
    bars: int | None = None,
    bar: str = DEFAULT_BAR,
    market: str = "um",
    cache_dir: str = "~/.cache/binance_vision",
    workers: int = 4,
    end=None,
) -> AssetPanel:
    """Load only aggTrades close prices; no flow/depth/feature calculation is run.

    Set `bars` for an exact-length panel, or set `end` with `bars=None` to load
    the full date range (useful as the input to Algorithm).
    """
    if load_bars is None:
        raise ImportError("Could not import binance_vision.load_bars")
    if not symbols:
        raise ValueError("symbols cannot be empty")
    if bars is not None and bars < 2:
        raise ValueError("bars must be >= 2 when specified")
    if bars is None and end is None:
        raise ValueError("supply either bars or end (or both)")
    start_ts = pd.Timestamp(start)
    if end is None:
        # Add a small calendar buffer because Data Vision is fetched in whole daily files.
        end_ts = start_ts + pd.Timedelta(bar) * int(bars) + pd.Timedelta(days=2)
    else:
        end_ts = pd.Timestamp(end)

    closes: dict[str, pd.Series] = {}
    for symbol in symbols:
        raw = load_bars(
            symbol,
            start=start_ts,
            end=end_ts,
            bar=bar,
            market=market,
            with_depth=False,
            with_ticker=False,
            with_metrics=False,
            with_funding=False,
            with_premium=False,
            with_liq=False,
            cache_dir=cache_dir,
            workers=workers,
        )
        if "close" not in raw:
            raise ValueError(f"load_bars returned no close column for {symbol}")
        closes[symbol] = raw["close"].astype(float)
    close = pd.DataFrame(closes).sort_index()
    panel = AssetPanel.from_close(close)
    offset = _start_offset(panel, start_ts)
    available = len(panel.logp) - offset
    if bars is None:
        return panel.iloc(offset, len(panel.logp))
    if available < bars:
        raise ValueError(
            f"loaded only {available} aligned bars from {start}; need {bars}. "
            "Pass a later end date or use a larger fetch range."
        )
    return panel.iloc(offset, offset + bars)




# --------------------------------------------------------------------------- intra-bar highs / lows
@dataclass
class HiLo:
    """Per-bar high and low of every traded asset (price units; columns = asset symbols, index = bar timestamps).

    Used only for barrier checks (stop / take-profit touches inside a bar).  Entries, PnL between entry and exit and
    the signal still use bar closes.
    """

    high: pd.DataFrame
    low: pd.DataFrame

    def between(self, start, end) -> "HiLo":
        """Rows from `start` to `end` inclusive (timestamps may be tz-naive or tz-aware)."""
        s = _timestamp_for_index(start, self.high.index)
        e = _timestamp_for_index(end, self.high.index)
        return HiLo(self.high.loc[s:e], self.low.loc[s:e])

    def arrays(self, close: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(high, low) as float arrays aligned to close.index / close.columns, with hi >= close >= lo enforced.

        Bars with no high/low row fall back to the close (so they can only touch a barrier on the close).
        """
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


def load_hilo(
    symbols: Sequence[str],
    start,
    end,
    bar: str = DEFAULT_BAR,
    market: str = "um",
    cache_dir: str = "~/.cache/binance_vision",
    workers: int = 4,
) -> HiLo:
    """Load per-bar high/low for `symbols` from binance_vision.load_bars (same call as load_price_panel)."""
    if load_bars is None:
        raise ImportError("Could not import binance_vision.load_bars")
    if not symbols:
        raise ValueError("symbols cannot be empty")
    highs: dict[str, pd.Series] = {}
    lows: dict[str, pd.Series] = {}
    for symbol in symbols:
        raw = load_bars(
            symbol, start=pd.Timestamp(start), end=pd.Timestamp(end), bar=bar, market=market,
            with_depth=False, with_ticker=False, with_metrics=False, with_funding=False,
            with_premium=False, with_liq=False, cache_dir=cache_dir, workers=workers,
        )
        for col in ("high", "low"):
            if col not in raw:
                raise ValueError(
                    f"load_bars returned no '{col}' column for {symbol}; pass your own HiLo "
                    "(HiLo(high_df, low_df)) or set TradeConfig.barrier_mode='close'")
        highs[symbol] = raw["high"].astype(float)
        lows[symbol] = raw["low"].astype(float)
    return HiLo(pd.DataFrame(highs).sort_index(), pd.DataFrame(lows).sort_index())


def _barrier_arrays(close: pd.DataFrame, cfg: "TradeConfig", hilo: HiLo | None):
    """(hi, lo) arrays for intra-bar barrier checks, or (None, None) for the legacy close-only mode."""
    if cfg.barrier_mode != "hilo":
        return None, None
    if hilo is None:
        raise ValueError(
            "barrier_mode='hilo' needs per-bar highs/lows: pass hilo=HiLo(...) (see load_hilo) "
            "or set TradeConfig(barrier_mode='close')")
    return hilo.arrays(close)


# --------------------------------------------------------------------------- 1) Screener
def _phi(s: np.ndarray) -> float:
    a, b = s[:-1] - s[:-1].mean(), s[1:] - s[1:].mean()
    return float((a * b).sum() / ((a * a).sum() + 1e-18))


def Screener(
    panel: AssetPanel | pd.DataFrame | None = None,
    y: str | None = None,
    *,
    symbols: Sequence[str] | None = None,
    start=None,
    bars: int = 5_000,
    bar: str = DEFAULT_BAR,
    market: str = "um",
    cache_dir: str = "~/.cache/binance_vision",
    workers: int = 4,
    seed_x: Sequence[str] | None = None,
    cfg: DiscoverConfig | None = None,
    ranked: Sequence[Candidate] | None = None,
) -> ScreenSelection:
    """Run the existing beam search on exactly the requested training bars.

    The search runs with the caller's DiscoverConfig (use_features is respected when the panel carries
    features), i.e. with the same constraints / ticker selection as spread_discoverer.  The first
    (lowest-score) candidate with at least one price leg is selected.  Its price legs are what gets traded;
    its feature terms (if any) are kept with their coefficients so strategy code can build the weighted-sum
    direction score.  If the panel has no features the search is price-only automatically.
    """
    if y is None:
        raise ValueError("y (the target asset symbol) is required")
    cfg0 = cfg or DiscoverConfig()

    if panel is None:
        if not symbols or start is None:
            raise ValueError("provide panel, or both symbols and start")
        panel0 = load_price_panel(
            symbols, start=start, bars=bars, bar=bar, market=market,
            cache_dir=cache_dir, workers=workers,
        )
    else:
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
        raise RuntimeError(
            "beam search returned no candidate with a tradable price leg. "
            "Check the asset universe, sample length, and DiscoverConfig thresholds."
        )

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
            f_coef[t] = float(b)
            f_mean[t] = float(X[:, j].mean())
            f_std[t] = float(X[:, j].std())
    hedge_symbols = tuple(betas)

    # asset-only (traded) spread statistics on the train window
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

    return ScreenSelection(
        target=y,
        terms=terms,
        hedge_symbols=hedge_symbols,
        intercept=float(intercept),
        betas=betas,
        train_mean=float(np.mean(residual)),
        train_std=sigma,
        score=float(chosen.score),
        p_eg_train=float(chosen.p_eg),
        half_life_train=float(chosen.half_life),
        train_start=train_panel.logp.index[0],
        train_end=train_panel.logp.index[-1],
        train_bars=len(train_panel.logp),
        candidate=chosen,
        ranked_candidates=tuple(executable),
        feature_terms=tuple(f_terms),
        feature_coefs=f_coef,
        feature_mean=f_mean,
        feature_std=f_std,
        asset_mean=float(np.mean(asset)),
        asset_std=a_std,
        phi_asset=phi,
        half_life_asset=float(hl),
        vol_seed=np.diff(asset_arr)[-4096:].astype(float),
        asset_seed=asset_arr[-4096:].copy(),
        full_seed=full_arr[-4096:].copy(),
    )


# --------------------------------------------------------------------------- 2) risk rules: stoploss()
def stoploss(spread_vol: float, gross_weight: float, cfg: TradeConfig, horizon_bars: int) -> StopLevels | None:
    """THE risk rule. Edit this function (or pass TradeConfig.stop_fn) to change stops, targets and sizing.

    spread_vol    std of ONE-BAR changes of the asset-only log spread (no features), rolling and causal
    gross_weight  1 + sum|beta_px|; converts a log-spread move into a % move of gross notional
    horizon_bars  volatility is scaled by sqrt(horizon) to the horizon the stop should live on

        vol_pct   = spread_vol * sqrt(horizon) / gross_weight          # % of gross notional
        stop_pct  = sl_vol_mult * vol_pct
        tp_pct    = rr * stop_pct
        leverage  = risk_fraction / stop_pct   (<= max_leverage)       # gross notional / capital
    Example: risk 2%, vol_pct 0.50% -> 4x, loses 2% at the stop, makes 4% at the TP; vol_pct 0.10% -> 20x.
    """
    if not (np.isfinite(spread_vol) and spread_vol > 0 and gross_weight > 0):
        return None
    vol_pct = spread_vol * np.sqrt(max(int(horizon_bars), 1)) / gross_weight
    stop_pct = cfg.sl_vol_mult * cfg.barrier_size * vol_pct
    if stop_pct <= 0:
        return None
    lev_raw = cfg.risk_fraction / stop_pct
    lev = min(lev_raw, cfg.max_leverage)
    gross = cfg.capital * lev
    stop_d = gross * stop_pct                 # == capital * risk_fraction unless leverage was capped
    return StopLevels(
        stop_pct=float(stop_pct), tp_pct=float(cfg.rr * stop_pct), leverage=float(lev),
        gross_notional=float(gross), stop_dollars=float(stop_d), tp_dollars=float(cfg.rr * stop_d),
        capped=bool(lev_raw > cfg.max_leverage),
    )


def _levels(spread_vol: float, sel: ScreenSelection, cfg: TradeConfig) -> StopLevels | None:
    if sel.size_mult <= 0.0:                       # persistence gate closed: no trades in this forward window
        return None
    fn = cfg.stop_fn or stoploss
    lv = fn(spread_vol, sel.gross_weight, cfg, _stop_horizon(sel, cfg))
    if lv is not None and sel.size_mult != 1.0:    # scale notional and dollar risk; the % barriers do not change
        m = float(sel.size_mult)
        lv = replace(lv, leverage=lv.leverage * m, gross_notional=lv.gross_notional * m,
                     stop_dollars=lv.stop_dollars * m, tp_dollars=lv.tp_dollars * m)
    return lv


def _stop_horizon(sel: ScreenSelection, cfg: TradeConfig) -> int:
    if cfg.stop_horizon_bars:
        return int(cfg.stop_horizon_bars)
    hl = sel.half_life_asset
    h = hl if np.isfinite(hl) else 48.0
    return int(np.clip(round(h), 6, cfg.max_hold_bars))


def _time_barrier(sel: ScreenSelection, cfg: TradeConfig) -> int:
    T = int(cfg.max_hold_bars)
    hl = sel.half_life_asset
    if cfg.ou_time_barrier and np.isfinite(hl) and hl > 0:
        T = int(min(T, max(2, np.ceil(cfg.ou_hold_mult * hl))))
    return T


# --------------------------------------------------------------------------- 3) signals
def _asset_spread(close: pd.DataFrame, sel: ScreenSelection) -> pd.Series:
    logp = np.log(close.loc[:, list(sel.symbols)])
    s = logp[sel.target].copy()
    for h in sel.hedge_symbols:
        s = s - sel.betas[h] * logp[h]
    return s.rename("asset_spread")


def _feature_part(panel: AssetPanel | None, sel: ScreenSelection, index: pd.Index) -> pd.Series:
    """F_t = sum_j coef_j * (x_j,t - mean_j) = sum_j (coef_j*std_j) * zscore_j,t   (spread = asset - icpt - F)."""
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


def _rolling_z(x: np.ndarray, seed, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling z-score: mean/std over the `window` bars ending at (and including) each bar.

    `seed` is the tail of the previous (train) window so the first forward bars already have a full window.
    Returns (z, rolling_mean).  z is NaN where the rolling std is ~0 or too few bars exist.
    """
    x = np.asarray(x, float)
    w = int(window)
    seed = np.asarray(seed, float)
    s = seed[-(w - 1):] if (len(seed) and w > 1) else np.zeros(0)
    roll = pd.Series(np.r_[s, x]).rolling(w, min_periods=max(30, w // 4))
    mu = roll.mean().to_numpy()[len(s):]
    sd = roll.std(ddof=0).to_numpy()[len(s):]
    sd = np.where(sd > 1e-12, sd, np.nan)
    return (x - mu) / sd, mu


def _signals(panel: AssetPanel | None, close: pd.DataFrame, sel: ScreenSelection, cfg: TradeConfig,
             use_seed: bool = True) -> pd.DataFrame:
    idx = close.index
    asset = _asset_spread(close, sel)
    F = _feature_part(panel, sel, idx)
    full = asset - sel.intercept - F
    a = asset.to_numpy(float)
    fu = full.to_numpy(float)
    if cfg.zscore_mode == "rolling":
        zw = int(sel.z_window or cfg.zscore_window)
        z_asset, mu = _rolling_z(a, sel.asset_seed if use_seed else (), zw)
        z_full, _ = _rolling_z(fu, sel.full_seed if use_seed else (), zw)
        dev = a - mu                                    # distance of the spread from its (rolling) mean
    else:
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
    sig = pd.DataFrame({
        "asset_spread": a,
        "z_asset": z_asset,
        "z_full": z_full,
        "z": z_asset if cfg.signal_spread == "asset" else z_full,
        "dev": dev,
        "feat_score": F.to_numpy(float) / sel.asset_std,   # sigma units; >0 -> feature part expected to fall -> spread expected UP
        "vol": vol,
    }, index=idx)
    return sig


def _spread_and_z(close: pd.DataFrame, selection: ScreenSelection, cfg: TradeConfig | None = None,
                  panel: AssetPanel | None = None) -> tuple[pd.Series, pd.Series]:
    cfg = cfg or TradeConfig()
    sig = _signals(panel, close, selection, cfg)
    return sig["asset_spread"].rename("spread_diagnostic"), sig["z"].rename("zscore")


# --------------------------------------------------------------------------- 4) barrier simulation + position building
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
    """First barrier touched after entry at the close of bar i.

    Returns None (no forward bar) or (exit_index, reason, fill_pnl, close_pnl):
      reason     'sl' | 'tp' | 'time' | 'window_end' (data ended first - unresolved)
      fill_pnl   position PnL actually realised at the exit
      close_pnl  position PnL marked at the exit bar's close  (fill_pnl - close_pnl = intra-bar fill adjustment)

    hi/lo None  -> legacy: barriers tested on bar closes, fill at the close (can overshoot the barrier).
    hi/lo given -> a barrier is touched if the bar's high/low reaches it; the fill is EXACTLY at the barrier
                   (-stop_d / +tp_d), never at the close.  The position's value inside a bar is unknowable from
                   per-leg highs/lows, so two conventions are offered:
      'comove'  all legs sit at their highs together or at their lows together (correlated majors move together);
                extremes = the position value in the all-highs state, the all-lows state, and at the close.
      'worst'   every leg independently at its adverse (stop test) / favourable (TP test) extreme; a strict bound
                that over-triggers stops on hedged spreads.
    If one bar touches both barriers the stop is assumed to come first.
    """
    last = len(close_arr) - 1
    end = min(i + T, last)
    if end <= i:
        return None
    ref = close_arr[i]
    path = (close_arr[i + 1:end + 1] - ref) @ q
    if hi is None:
        lowest = highest = path
    else:
        dh = hi[i + 1:end + 1] - ref
        dl = lo[i + 1:end + 1] - ref
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


def _tree_frame(panel: AssetPanel | None, sel: ScreenSelection, cfg: TradeConfig, sig: pd.DataFrame,
                idxs: np.ndarray, sides: np.ndarray, levels: list[StopLevels]) -> pd.DataFrame:
    z = sig["z"].to_numpy()
    zf = sig["z_full"].to_numpy()
    dev = sig["dev"].to_numpy()
    L = int(cfg.tree_lookback_bars)
    zs = np.r_[np.full(L, np.nan), z[:-L]] if len(z) > L else np.full(len(z), np.nan)
    T = _time_barrier(sel, cfg)
    phi = float(np.clip(sel.phi_asset, 0.0, 1.0)) if np.isfinite(sel.phi_asset) else 1.0
    stop_spread = np.array([l.stop_pct * sel.gross_weight for l in levels])
    cols: dict[str, np.ndarray] = {
        "side": sides.astype(float),
        "abs_z": np.abs(z[idxs]),
        "abs_z_full": np.abs(zf[idxs]),
        "dz_side": sides * (z[idxs] - zs[idxs]),                   # >0: z already moving the trade's way
        "ou_edge": np.abs(dev[idxs]) * (1.0 - phi ** T) / stop_spread,
        "sl_pct": np.array([l.stop_pct for l in levels]),
        "leverage": np.array([l.leverage for l in levels]),
    }
    if cfg.feature_score_mode in ("feature", "both"):
        cols["feat_score_side"] = sides * sig["feat_score"].to_numpy()[idxs]
    if cfg.tree_features == "all" and panel is not None:
        for name, df in panel.feats.items():
            for sym in df.columns:
                cols[f"{sym}:{name}"] = df[sym].to_numpy(float)[idxs]
        for sym in panel.symbols:
            cols[f"{sym}:ret"] = panel.ret[sym].to_numpy(float)[idxs]
            cols[f"{sym}:vol"] = panel.vol[sym].to_numpy(float)[idxs]
            cols[f"{sym}:retL"] = panel.ret[sym].rolling(L).sum().to_numpy(float)[idxs]
    return pd.DataFrame(cols)


def fit_tp_model(selection: ScreenSelection, train_panel: AssetPanel | pd.DataFrame,
                 cfg: TradeConfig, hilo: HiLo | None = None) -> TPModel | None:
    """Train the tree on the train window: for every bar with |z| >= tree_event_z, simulate the triple barrier with the
    same stoploss()/time barrier/barrier_mode used live and label 1 if the take-profit is hit first.

    Caveat: the spread was fitted on this window, so its z-scores mean-revert more here than they will out of sample;
    the tree's absolute probabilities are optimistic.  Judge it by the per-fold OOS TP rate vs breakeven, not by p.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:                          # pragma: no cover
        raise ImportError("fit_tp_model needs scikit-learn (pip install scikit-learn)") from exc

    pan = _as_panel(train_panel)
    close = _close_frame(pan, selection.symbols)
    hi, lo = _barrier_arrays(close, cfg, hilo)
    sig = _signals(pan, close, selection, cfg, use_seed=False)
    n = len(close)
    closeA = close.to_numpy(float)
    z = sig["z"].to_numpy()
    vol = sig["vol"].to_numpy()
    T = _time_barrier(selection, cfg)
    z_level = cfg.tree_event_z or cfg.entry_z
    events = _candidate_bars(sig, z_level, n)[::cfg.tree_event_stride]
    ev_i, ev_side, ev_lv, ev_exit, ev_y = [], [], [], [], []
    for i in events:
        i = int(i)
        lv = _levels(float(vol[i]), selection, cfg)
        if lv is None:
            continue
        side = 1 if z[i] < 0 else -1
        q = _legs(closeA[i], selection, side, lv.gross_notional)
        res = _resolve(closeA, i, q, lv.stop_dollars, lv.tp_dollars, T, hi, lo, cfg.hilo_extremes)
        if res is None or res[1] == "window_end":
            continue                                    # unresolved at the end of the window: no label
        ev_i.append(i); ev_side.append(side); ev_lv.append(lv); ev_exit.append(res[0]); ev_y.append(int(res[1] == "tp"))
    if len(ev_i) < cfg.tree_min_events or len(set(ev_y)) < 2:
        warnings.warn(f"tree skipped: {len(ev_i)} labelled events (need >= {cfg.tree_min_events} and both classes)")
        return None

    idxs, sides, y = np.array(ev_i), np.array(ev_side), np.array(ev_y)
    Xdf = _tree_frame(pan, selection, cfg, sig, idxs, sides, ev_lv)
    Xc, med = _clean(Xdf)
    params = dict(cfg.tree_params)
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", cfg.tree_n_jobs)

    # purged chronological hold-out (last 25% of events, training labels must resolve before it starts) -> diagnostic AUC
    val_auc = float("nan")
    if cfg.tree_diagnostics:
        cut = idxs[int(len(idxs) * 0.75)]
        tr = np.array(ev_exit) < cut
        va = idxs >= cut
        if tr.sum() >= cfg.tree_min_events // 2 and va.sum() >= 10 and len(set(y[tr])) == 2 and len(set(y[va])) == 2:
            probe = RandomForestClassifier(**params).fit(Xc[tr], y[tr])
            val_auc = float(roc_auc_score(y[va], probe.predict_proba(Xc[va])[:, 1]))

    clf = RandomForestClassifier(**params).fit(Xc, y)
    thr = cfg.tree_prob_threshold if cfg.tree_prob_threshold is not None else cfg.breakeven_win_rate + cfg.tree_margin
    return TPModel(clf=clf, columns=list(Xc.columns), medians=med, n_events=len(y), base_rate=float(y.mean()),
                   val_auc=val_auc, threshold=float(thr))


# --------------------------------------------------------------------------- 5) trade: triple-barrier entry/exit rule
def trade(
    prices: pd.DataFrame,
    selection: ScreenSelection,
    cfg: TradeConfig | None = None,
    panel: AssetPanel | None = None,
    hilo: HiLo | None = None,
) -> pd.DataFrame:
    """Triple-barrier mean-reversion rule. Returns base-asset quantity at every bar.

    Entry (bar close):  |z| >= entry_z, armed (z has been back inside entry_z since the last trade), and
        (a) feature weighted-sum direction agrees   [feature_score_mode in filter|both]
        (b) OU edge ratio >= ou_min_edge_ratio       [if set]
        (c) tree P(TP) >= threshold                  [use_tree and selection.tp_model]
    Direction: z>0 -> short the spread, z<0 -> long it.  Size: stoploss() -> gross notional.
    cfg.strategy_side=2 INVERTS the executed direction only (strategy says long -> we short, and vice versa).  Every gate
    (feature score, OU edge, tree) is still evaluated on the strategy's own direction; the stop / TP / time barrier and the
    sizing are then built for the position actually held, so risk per trade and the rr stay as configured.
    Exit: stop-loss / take-profit (rr*stop) / time barrier, whichever comes first.  With barrier_mode='hilo' a stop or
    TP is hit as soon as the bar's high/low reaches it and fills exactly at the barrier (see _resolve); with 'close'
    it is checked on closes and fills at the close.
    Trade metadata (risk, reason, leverage, p_tp...) is attached as `.attrs['trade_meta']`; the PnL difference between
    a barrier fill and the exit bar's close is attached as `.attrs['pnl_adjust']` ({exit timestamp: dollars}).
    """
    cfg = cfg or TradeConfig()
    close = _close_frame(prices, selection.symbols)
    if panel is None and isinstance(prices, AssetPanel):
        panel = prices
    hi, lo = _barrier_arrays(close, cfg, hilo)
    sig = _signals(panel, close, selection, cfg)
    n = len(close)
    closeA = close.to_numpy(float)
    qty = np.zeros((n, len(selection.symbols)))
    z = sig["z"].to_numpy()
    vol = sig["vol"].to_numpy()
    dev = sig["dev"].to_numpy()
    fs = sig["feat_score"].to_numpy()
    T = _time_barrier(selection, cfg)
    phi = float(np.clip(selection.phi_asset, 0.0, 1.0)) if np.isfinite(selection.phi_asset) else 1.0

    # ---- batch-evaluate every candidate bar (all inputs are known at that bar's close)
    cand = _candidate_bars(sig, cfg.entry_z, n)
    info: dict[int, dict] = {}
    keep_i, keep_side, keep_lv = [], [], []
    for i in cand:
        lv = _levels(float(vol[i]), selection, cfg)
        if lv is None:
            continue
        keep_i.append(int(i)); keep_side.append(1 if z[i] < 0 else -1); keep_lv.append(lv)
    probs = np.full(len(keep_i), np.nan)
    if keep_i:
        idxs, sides = np.array(keep_i), np.array(keep_side)
        model = selection.tp_model if cfg.use_tree else None
        if model is not None:
            Xdf = _tree_frame(panel, selection, cfg, sig, idxs, sides, keep_lv)
            Xc, _ = _clean(Xdf.reindex(columns=model.columns), model.medians)
            probs = model.clf.predict_proba(Xc[model.columns])[:, 1]
        for k, i in enumerate(keep_i):
            lv, side = keep_lv[k], keep_side[k]
            ok = True
            if cfg.feature_score_mode in ("filter", "both") and selection.feature_terms:
                ok &= side * fs[i] >= cfg.min_feature_score
            if cfg.ou_min_edge_ratio is not None:
                edge = abs(dev[i]) * (1.0 - phi ** T) / (lv.stop_pct * selection.gross_weight)
                ok &= edge >= cfg.ou_min_edge_ratio
            if model is not None:
                ok &= probs[k] >= model.threshold
            if ok:
                info[i] = dict(side=side, lv=lv, p_tp=float(probs[k]) if model is not None else float("nan"))

    # ---- sequential walk: arming, one position at a time
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
        exec_side = d["side"] * cfg.side_sign              # strategy_side=2 flips the position actually taken
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
        meta[close.index[i]] = dict(
            side=exec_side, strategy_side=d["side"], reason=reason, leverage=lv.leverage, stop_pct=lv.stop_pct, tp_pct=lv.tp_pct,
            risk_dollars=lv.stop_dollars, p_tp=d["p_tp"], capped=lv.capped)
        armed = bool(exit_i < n and abs(z[exit_i]) < cfg.entry_z)
        i = exit_i + 1

    out = pd.DataFrame(qty, index=close.index, columns=list(selection.symbols))
    out.attrs["trade_meta"] = meta
    out.attrs["pnl_adjust"] = adjust
    return out


def _trade_ledger(
    quantities: pd.DataFrame,
    net_pnl: pd.Series,
    cfg: TradeConfig,
    meta: dict | None = None,
) -> pd.DataFrame:
    """One row per trade (a run of non-zero quantities plus the exit bar).  Vectorised: PnL = cumsum differences."""
    meta = meta or {}
    n = len(quantities)
    if n == 0:
        return pd.DataFrame([])
    active = quantities.abs().sum(axis=1).to_numpy() > 1e-12
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)            # first inactive bar after the run (== n if still open at the end)
    cs = np.cumsum(net_pnl.to_numpy(float))
    index = quantities.index
    rows: list[dict[str, Any]] = []
    for trade_no, (s, e) in enumerate(zip(starts, stops), start=1):
        open_end = e >= n
        end_i = n - 1 if open_end else int(e)
        pnl = float(cs[end_i] - (cs[s - 1] if s > 0 else 0.0))   # entry costs + all bar PnL + exit bar PnL + exit costs
        m = meta.get(index[s], {})
        risk = float(m.get("risk_dollars", cfg.risk_dollars))
        r = {
            "trade": trade_no,
            "entry_time": index[s],
            "exit_time": index[end_i],
            "bars_held": int(end_i - s),
            "side": m.get("side", np.nan),
            "strategy_side": m.get("strategy_side", np.nan),
            "reason": m.get("reason", "open_at_end" if open_end else ""),
            "leverage": m.get("leverage", np.nan),
            "stop_pct": m.get("stop_pct", np.nan),
            "p_tp": m.get("p_tp", np.nan),
            "pnl": pnl,
            "risk_dollars": risk,
            "r_multiple": pnl / risk,
        }
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


def Forward(
    selection: ScreenSelection,
    forward_prices: AssetPanel | pd.DataFrame,
    *,
    cfg: TradeConfig | None = None,
    trade_fn: Callable[..., pd.DataFrame] | None = None,
    hilo: HiLo | None = None,
) -> ForwardResult:
    """Run the frozen selection over one forward window (normally 5,000 bars).

    trade_fn(prices, selection, cfg[, panel=AssetPanel][, hilo=HiLo]) -> DataFrame of base-asset quantities.
    """
    cfg = cfg or TradeConfig()
    pan = forward_prices if isinstance(forward_prices, AssetPanel) else None
    close = _close_frame(forward_prices, selection.symbols)
    if pan is None and selection.feature_terms:
        raise ValueError("the selected spread has feature terms; Forward needs an AssetPanel with features, not a price DataFrame")
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

    # Actual asset PnL: previous-bar base units * current asset price change. The spread is NOT used here.
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
    turnover = (dqty.abs() * close).sum(axis=1)
    costs = turnover * cfg.cost_rate
    net_pnl = (gross_pnl - costs).rename("net_pnl")
    gross_pnl = gross_pnl.rename("gross_asset_pnl")
    costs = costs.rename("trading_costs")
    ledger = _trade_ledger(quantities, net_pnl, cfg, meta)
    try:
        oos_stat, _ = adf_stat(spread.to_numpy(float), maxlag=12, const=True)
        oos_p = adf_p(oos_stat, 1)
        oos_hl = half_life(spread.to_numpy(float), floor=1.0)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        oos_stat = oos_p = oos_hl = float("nan")

    return ForwardResult(
        selection=selection, close=close, spread=spread, zscore=z, quantities=quantities,
        gross_pnl=gross_pnl, costs=costs, net_pnl=net_pnl, trades=ledger,
        adf_stat_oos=float(oos_stat), p_oos=float(oos_p), half_life_oos=float(oos_hl),
    )


# --------------------------------------------------------------------------- 6) folds / Algorithm / flip-flop retraining

def retrain_rules(
    train_panel: AssetPanel,
    y: str,
    discover_cfg: DiscoverConfig,
    previous_selection: ScreenSelection | None = None,
) -> ScreenSelection:
    """Default retrain rule: re-run the beam search on the latest train window.

    Replace this callable with custom if/else logic. Return `previous_selection`
    to keep the old hedge, or call Screener(...) to discover a new one.
    """
    del previous_selection  # default rule always re-screens the latest window
    return Screener(train_panel, y, bars=len(train_panel.logp), cfg=discover_cfg)


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


def plan_folds(
    n_bars: int,
    *,
    offset: int = 0,
    train_bars: int = 5_000,
    forward_bars: int = 5_000,
    purge_bars: int = 1,
    max_cycles: int | None = None,
) -> list[FoldPlan]:
    """Flip-flop schedule: cycle 0 trains on the first `train_bars`, then every next cycle trains on the window that
    was just traded.  Each cycle consumes forward_bars + purge_bars of fresh data."""
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


def screen_fold(
    train_panel: AssetPanel,
    y: str,
    discover_cfg: DiscoverConfig,
    *,
    retrain_rule: Callable[..., ScreenSelection] | None = None,
    previous_selection: ScreenSelection | None = None,
) -> ScreenSelection:
    """The expensive, TradeConfig-independent half of a fold: beam search on the train window."""
    sel = (retrain_rule or retrain_rules)(train_panel, y, discover_cfg, previous_selection)
    if not isinstance(sel, ScreenSelection):
        raise TypeError("retrain_rule must return ScreenSelection")
    if sel.target != y or not sel.hedge_symbols:
        raise ValueError("retrain_rule returned a target mismatch or a selection with no tradable hedge leg")
    return sel


def trade_fold(
    selection: ScreenSelection,
    train_panel: AssetPanel,
    forward_panel: AssetPanel,
    plan: FoldPlan,
    tcfg: TradeConfig,
    *,
    hilo: HiLo | None = None,
    trade_fn: Callable[..., pd.DataFrame] | None = None,
) -> FoldResult:
    """The TradeConfig-dependent half of a fold: fit the tree on the train window, trade the forward window.
    `hilo` should cover both windows (train for tree labels, forward for the live barriers)."""
    selection = replace(selection, tp_model=fit_tp_model(selection, train_panel, tcfg, hilo=hilo) if tcfg.use_tree else None)
    fwd = Forward(selection, forward_panel, cfg=tcfg, trade_fn=trade_fn, hilo=hilo)
    return FoldResult(
        cycle=plan.cycle,
        instance=plan.instance,
        train_start=train_panel.logp.index[0],
        train_end=train_panel.logp.index[-1],
        forward_start=forward_panel.logp.index[0],
        forward_end=forward_panel.logp.index[-1],
        selection=selection,
        forward=fwd,
    )


def run_fold(
    train_panel: AssetPanel,
    forward_panel: AssetPanel,
    y: str,
    plan: FoldPlan,
    discover_cfg: DiscoverConfig,
    tcfg: TradeConfig,
    *,
    hilo: HiLo | None = None,
    trade_fn: Callable[..., pd.DataFrame] | None = None,
    retrain_rule: Callable[..., ScreenSelection] | None = None,
    previous_selection: ScreenSelection | None = None,
) -> FoldResult:
    sel = screen_fold(train_panel, y, discover_cfg, retrain_rule=retrain_rule, previous_selection=previous_selection)
    return trade_fold(sel, train_panel, forward_panel, plan, tcfg, hilo=hilo, trade_fn=trade_fn)


def parallel_map(fn: Callable, jobs: Sequence[dict], n_jobs: int = 1) -> list:
    """fn(**job) for every job, in order, on `n_jobs` processes (-1 = all cores).  fn must be a module-level function."""
    jobs = list(jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    n_jobs = min(n_jobs, len(jobs))
    if n_jobs <= 1:
        return [fn(**kw) for kw in jobs]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            return [f.result() for f in [ex.submit(fn, **kw) for kw in jobs]]
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(**kw) for kw in jobs)


def _hilo_window(hilo: HiLo | None, *panels: AssetPanel) -> HiLo | None:
    if hilo is None:
        return None
    return hilo.between(min(p.logp.index[0] for p in panels), max(p.logp.index[-1] for p in panels))


def assemble_algorithm(
    folds: list[FoldResult],
    *,
    index: pd.Index,
    symbols: Sequence[str],
    bar_minutes: float,
    tcfg: TradeConfig,
    train_bars: int,
    forward_bars: int,
    purge_bars: int,
    print_report: bool = True,
) -> AlgorithmResult:
    """Stitch fold results into one PnL series / trade ledger / report."""
    if not folds:
        raise ValueError("no folds to assemble")
    first = folds[0].forward.close.index[0]
    last = folds[-1].forward.close.index[-1]
    full_index = index[(index >= first) & (index <= last)]
    pnl = pd.Series(0.0, index=full_index, name="net_pnl")
    quantities = pd.DataFrame(0.0, index=full_index, columns=list(symbols))
    trade_frames = []
    fold_rows = []
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
        m = fold.selection.tp_model
        reasons = tr["reason"] if len(tr) else pd.Series(dtype=object)
        fold_rows.append({
            "cycle": fold.cycle, "instance": fold.instance,
            "hedge": " + ".join(fold.selection.hedge_symbols),
            "feature_terms": ", ".join(fold.selection.feature_terms),
            "p_eg_train": fold.selection.p_eg_train, "hl_train": fold.selection.half_life_train,
            "p_oos": fold.forward.p_oos, "hl_oos": fold.forward.half_life_oos,
            "trades": len(tr),
            "tp_rate": float((reasons == "tp").mean()) if len(tr) else np.nan,
            "sl_rate": float((reasons == "sl").mean()) if len(tr) else np.nan,
            "time_rate": float((reasons == "time").mean()) if len(tr) else np.nan,
            "gross_pnl": float(fold.forward.gross_pnl.sum()), "net_pnl": float(fold.forward.net_pnl.sum()),
            "tree_events": m.n_events if m else np.nan, "tree_base_rate": m.base_rate if m else np.nan,
            "tree_val_auc": m.val_auc if m else np.nan,
        })
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(
        columns=["cycle", "instance", "target", "hedge", "trade", "entry_time", "exit_time", "bars_held", "side",
                 "strategy_side", "reason", "leverage", "stop_pct", "p_tp", "pnl", "risk_dollars", "r_multiple"]
    )
    fold_table = pd.DataFrame(fold_rows)

    total_pnl = float(pnl.sum())
    total_r = float(trades["r_multiple"].sum()) if len(trades) else 0.0
    expectancy = float(trades["r_multiple"].mean()) if len(trades) else float("nan")
    bars_per_year = 365.25 * 24.0 * 60.0 / bar_minutes if np.isfinite(bar_minutes) and bar_minutes > 0 else 1.0
    sortino = _sortino(pnl, bars_per_year)
    max_dd = _max_drawdown(pnl, tcfg.capital)
    gross_total = float(sum(f.forward.gross_pnl.sum() for f in folds))
    cost_total = float(sum(f.forward.costs.sum() for f in folds))
    if len(trades):
        rs = trades["reason"]
        mix = " | ".join(f"{k} {float((rs == k).mean()):.1%}" for k in ("tp", "sl", "time", "window_end") if (rs == k).any())
        p_taken = pd.to_numeric(trades["p_tp"], errors="coerce").mean()
        lev = pd.to_numeric(trades["leverage"], errors="coerce").mean()
    else:
        mix, p_taken, lev = "n/a", np.nan, np.nan

    report = (
        "PURGED WALK-FORWARD / FLIP-FLOP REPORT (triple barrier)\n"
        f"Folds: {len(folds)} | train={train_bars:,} bars | forward={forward_bars:,} bars | purge={purge_bars:,} bars\n"
        f"z-score: {tcfg.zscore_mode}" + (f" ({tcfg.zscore_window:,} bars)" if tcfg.zscore_mode == "rolling" else " (train mean/std)")
        + f" | barriers: {tcfg.barrier_mode}" + (f" ({tcfg.hilo_extremes})" if tcfg.barrier_mode == "hilo" else "") + "\n"
        f"Strategy direction: {'FOLLOWED (strategy_side=1)' if tcfg.strategy_side == 1 else 'INVERTED (strategy_side=2: strategy long -> we short)'}\n"
        f"Instances: A/B alternating | Closed trades: {len(trades):,} | Exits: {mix}\n"
        f"Barriers: stop = {tcfg.sl_vol_mult * tcfg.barrier_size:g} x spread vol (barrier_size={tcfg.barrier_size:g}), TP = rr {tcfg.rr:g} x stop\n"
        f"Breakeven TP rate at rr={tcfg.rr:g}: {tcfg.breakeven_win_rate:.1%} (zero-cost, ignoring time exits) | "
        f"mean tree P(TP) of taken trades: {p_taken:.3f} | mean leverage: {lev:.2f}x\n"
        f"Gross asset PnL: {gross_total:,.4f} | Costs: {cost_total:,.4f} | Net asset PnL: {total_pnl:,.4f}\n"
        f"Accumulated R: {total_r:,.4f}R | Expectancy: {expectancy:,.4f}R/trade\n"
        f"Sortino (annualised, bar PnL): {sortino:,.4f} | Max drawdown: {max_dd:.2%}\n"
        f"Median forward ADF p: {np.nanmedian([f.forward.p_oos for f in folds]):.4g} | "
        f"Median forward half-life: {np.nanmedian([f.forward.half_life_oos for f in folds]):.2f} bars\n"
        "PnL uses traded asset quantities x asset price changes; the statistical spread and features are signal-only."
    )
    if print_report:
        print(report)
    return AlgorithmResult(folds, pnl, quantities, trades, total_pnl, total_r, expectancy, sortino, max_dd, report, fold_table)


def Algorithm(
    panel: AssetPanel | pd.DataFrame,
    y: str,
    *,
    start=None,
    train_bars: int = 5_000,
    forward_bars: int = 5_000,
    purge_bars: int = 1,
    discover_cfg: DiscoverConfig | None = None,
    trade_cfg: TradeConfig | None = None,
    trade_fn: Callable[..., pd.DataFrame] | None = None,
    retrain_rule: Callable[[AssetPanel, str, DiscoverConfig, ScreenSelection | None], ScreenSelection] | None = None,
    hilo: HiLo | None = None,
    max_cycles: int | None = None,
    n_jobs: int = 1,
    print_report: bool = True,
) -> AlgorithmResult:
    """Run two alternating strategy instances in a causal flip-flop schedule.

    Cycle 0 / instance A: train on the initial `train_bars` (beam search + tree), trade the next
    `forward_bars`. Cycle 1 / instance B: train on the just-traded window, trade the next window.
    Then A, B, ... alternate. A purge gap is left between each training window and its forward window.

    n_jobs != 1 runs the folds in parallel processes (-1 = all cores).  Folds are independent given the data, but
    in parallel mode `retrain_rule` receives previous_selection=None (no state is carried between folds).
    hilo: per-bar highs/lows (required when trade_cfg.barrier_mode == 'hilo', the default).
    """
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

    plans = plan_folds(len(p.logp), offset=offset, train_bars=train_bars, forward_bars=forward_bars,
                       purge_bars=purge_bars, max_cycles=max_cycles)
    if not plans:
        needed = train_bars + purge_bars + forward_bars
        raise ValueError(f"not enough data for one train/forward cycle; need at least {needed} bars from start")

    def _slices(plan: FoldPlan):
        tp = _slice_panel(p, plan.train_start, plan.train_end)
        fp = _slice_panel(p, plan.forward_start, plan.forward_end)
        return tp, fp, _hilo_window(hilo, tp, fp)

    if n_jobs == 1 or len(plans) == 1:
        slots: dict[str, ScreenSelection | None] = {"A": None, "B": None}
        folds: list[FoldResult] = []
        for plan in plans:
            tp, fp, hl = _slices(plan)
            fold = run_fold(tp, fp, y, plan, dcfg, tcfg, hilo=hl, trade_fn=trade_fn,
                            retrain_rule=retrain_rule, previous_selection=slots[plan.instance])
            slots[plan.instance] = fold.selection
            folds.append(fold)
    else:
        par_cfg = replace(tcfg, tree_n_jobs=1)
        jobs = []
        for plan in plans:
            tp, fp, hl = _slices(plan)
            jobs.append(dict(train_panel=tp, forward_panel=fp, y=y, plan=plan, discover_cfg=dcfg, tcfg=par_cfg,
                             hilo=hl, trade_fn=trade_fn, retrain_rule=retrain_rule))
        folds = parallel_map(run_fold, jobs, n_jobs)

    return assemble_algorithm(
        folds, index=p.logp.index, symbols=list(p.logp.columns), bar_minutes=p.bar_minutes, tcfg=tcfg,
        train_bars=train_bars, forward_bars=forward_bars, purge_bars=purge_bars, print_report=print_report)


__all__ = [
    "TradeConfig", "StopLevels", "TPModel", "ScreenSelection", "ForwardResult", "FoldResult", "AlgorithmResult",
    "HiLo", "load_hilo", "load_price_panel", "Screener", "stoploss", "fit_tp_model", "trade", "Forward",
    "FoldPlan", "plan_folds", "screen_fold", "trade_fold", "run_fold", "assemble_algorithm", "parallel_map",
    "retrain_rules", "Algorithm", "DEFAULT_BAR",
]

if __name__ == "__main__":
    print("=== STRATEGY STARTED ===", flush=True)

    try:
        SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
        TARGET = "ETH/USDT"
        START = "2024-01-01"
        END = "2024-07-01"

        print("[1/3] Loading prices + features...", flush=True)
        if get_assets is None:
            raise ImportError("could not import spread_discoverer.get_assets")
        discover_cfg = DiscoverConfig(use_features=True)
        panel = get_assets(SYMBOLS, START, END, bar="5min", market="um", cfg=discover_cfg)
        print(f"Loaded {len(panel.logp)} bars, {len(panel.feats)} features.", flush=True)

        print("[2/3] Running walk-forward stress test (commissions = 0)...", flush=True)
        result = stress_test(
            panel=panel,
            y=TARGET,
            n_paths=100,
            seed=42,
            train_bars=5000,
            forward_bars=5000,
            purge_bars=1,
            discover_cfg=discover_cfg,
            trade_cfg=TradeConfig(
                entry_z=2.0,
                capital=10_000,
                risk_fraction=0.02,       # max loss per trade = 2% of capital
                rr=2.0,                   # take profit = 2 x stop
                fee_bps=0.0,
                slippage_bps=0.0,
                feature_score_mode="filter",
                use_tree=True,
            ),
            plot=True,
            print_report=True,
        )
        print(result.algorithm.fold_table.to_string(), flush=True)
        print("[3/3] Completed successfully.", flush=True)

        if result.figure is not None:
            import matplotlib.pyplot as plt
            plt.show()

    except Exception:
        import traceback
        traceback.print_exc()
        raise