"""Price-only execution and walk-forward orchestration for spread_discoverer.

The screener is reused from spread_discoverer.py.  This module deliberately does
not import flow_features or calculate indicators.  Only close prices are needed
for screening, signal generation, position sizing, and realised PnL.

Execution convention
-------------------
* The discovered model is a static log-price relationship:
      spread = log(P_target) - intercept - sum(beta_j * log(P_hedge_j))
* A positive z-score sells the target spread (short target, long beta-weighted
  hedge legs); a negative z-score buys it.
* Positions are sized to a fixed gross notional and quantities are fixed between
  entry and exit.  Signals are observed at bar close; PnL starts on the next bar.
* PnL is calculated only from actual asset close-price changes and held asset
  quantities.  The statistical spread is used only to create entry/exit signals.
* R is normalised by a configurable fixed risk budget (capital * risk_fraction).
  It is a risk-budget multiple, not a stop-distance-defined R unless the user
  separately calibrates risk_fraction to their stop model.

Expected repository layout:
    spread_discoverer.py
    binance_vision.py
    strategy.py                 # this file

The module can also be imported from a package; relative imports are attempted
first and flat-folder imports are the fallback.
"""
from __future__ import annotations

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
    from .binance_vision import load_bars
except ImportError:
    try:
        from binance_vision import load_bars
    except ImportError:
        load_bars = None


# --------------------------------------------------------------------------- configuration
@dataclass
class TradeConfig:
    """Execution and signal rules. All price legs are traded; features are not."""

    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float | None = None
    max_hold_bars: int = 576
    capital: float = 10_000.0
    gross_leverage: float = 1.0
    risk_fraction: float = 0.01
    fee_bps: float = 5.0
    slippage_bps: float = 0.0
    close_at_window_end: bool = True

    def __post_init__(self):
        if self.entry_z <= 0:
            raise ValueError("entry_z must be > 0")
        if self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")
        if self.capital <= 0 or self.gross_leverage <= 0:
            raise ValueError("capital and gross_leverage must be > 0")
        if not (0 < self.risk_fraction <= 1):
            raise ValueError("risk_fraction must be in (0, 1]")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")

    @property
    def risk_dollars(self) -> float:
        return float(self.capital * self.risk_fraction)

    @property
    def cost_rate(self) -> float:
        return float((self.fee_bps + self.slippage_bps) * 1e-4)


# --------------------------------------------------------------------------- result objects
@dataclass
class ScreenSelection:
    """Frozen, executable price-only selection returned by Screener()."""

    target: str
    terms: tuple[str, ...]
    hedge_symbols: tuple[str, ...]
    intercept: float
    betas: dict[str, float]              # keyed by hedge symbol
    train_mean: float
    train_std: float
    score: float
    p_eg_train: float
    half_life_train: float
    train_start: Any
    train_end: Any
    train_bars: int
    candidate: Candidate
    ranked_candidates: tuple[Candidate, ...] = ()

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.target,) + self.hedge_symbols


@dataclass
class ForwardResult:
    selection: ScreenSelection
    close: pd.DataFrame
    spread: pd.Series                 # diagnostic only; NEVER used for PnL
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


@dataclass
class StressTestResult:
    algorithm: AlgorithmResult
    expectancy_paths: pd.Series
    cumulative_r_paths: pd.DataFrame
    expectancy_mean: float
    expectancy_p05: float
    expectancy_p95: float
    cvar05: float
    cvar95: float
    report: str
    figure: Any = None


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
    bar: str = "5min",
    market: str = "um",
    cache_dir: str = "~/.cache/binance_vision",
    workers: int = 4,
    end=None,
) -> AssetPanel:
    """Load only aggTrades close prices; no flow/depth/feature calculation is run.

    Set `bars` for an exact-length panel, or set `end` with `bars=None` to load
    the full date range (useful as the input to Algorithm / stress_test).
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


# --------------------------------------------------------------------------- 1) Screener

def Screener(
    panel: AssetPanel | pd.DataFrame | None = None,
    y: str | None = None,
    *,
    symbols: Sequence[str] | None = None,
    start=None,
    bars: int = 5_000,
    bar: str = "5min",
    market: str = "um",
    cache_dir: str = "~/.cache/binance_vision",
    workers: int = 4,
    seed_x: Sequence[str] | None = None,
    cfg: DiscoverConfig | None = None,
) -> ScreenSelection:
    """Run the existing beam search on exactly the requested training bars.

    The feature pool is disabled here on purpose: forward execution is price-only,
    so a feature-based hedge could not be reconstructed without calculating features.
    Candidates are ranked by spread_discoverer's existing objective; the first
    (lowest-score) executable candidate is selected. The remaining ranked candidates
    are retained on the returned object for inspection.

    Supply either `panel` (preloaded prices) or `symbols` + `start` (fetches prices).
    """
    if y is None:
        raise ValueError("y (the target asset symbol) is required")
    cfg0 = cfg or DiscoverConfig()
    # Force price-only search irrespective of the caller's feature setting.
    screen_cfg = replace(cfg0, use_features=False)

    if panel is None:
        if not symbols or start is None:
            raise ValueError("provide panel, or both symbols and start")
        panel0 = load_price_panel(
            symbols, start=start, bars=bars, bar=bar, market=market,
            cache_dir=cache_dir, workers=workers,
        )
        offset = 0
    else:
        panel0 = _as_panel(panel)
        offset = _start_offset(panel0, start)
        if offset >= len(panel0.logp):
            raise ValueError(f"start={start!r} is after the available panel")
        if len(panel0.logp) - offset < bars:
            raise ValueError(f"not enough bars from start: need {bars}, have {len(panel0.logp) - offset}")
        panel0 = panel0.iloc(offset, offset + bars)

    if y not in panel0.logp.columns:
        raise ValueError(f"target {y!r} is not in the panel")
    train_panel = panel0
    ranked = create_pair(train_panel, y, seed_x=seed_x, cfg=screen_cfg)
    executable = [c for c in ranked if c.terms and all(t.startswith("px|") for t in c.terms)]
    if not executable:
        raise RuntimeError(
            "beam search returned no price-only executable candidate. "
            "Check the asset universe, sample length, and DiscoverConfig thresholds."
        )

    chosen = executable[0]
    hedge_symbols = tuple(t.split("|", 1)[1] for t in chosen.terms)
    X = train_panel.matrix(chosen.terms)
    yv = train_panel.series(px(y))
    intercept, slopes, residual = fit(yv, X, method=screen_cfg.fit_method, **screen_cfg.fit_kw)
    sigma = float(np.std(residual))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        raise RuntimeError("selected candidate has a degenerate training spread")
    betas = {s: float(b) for s, b in zip(hedge_symbols, slopes)}

    return ScreenSelection(
        target=y,
        terms=tuple(chosen.terms),
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
    )


# --------------------------------------------------------------------------- 2) Forward / trade

def _spread_and_z(close: pd.DataFrame, selection: ScreenSelection) -> tuple[pd.Series, pd.Series]:
    symbols = selection.symbols
    logp = np.log(close.loc[:, symbols])
    spread = logp[selection.target] - selection.intercept
    for symbol in selection.hedge_symbols:
        spread = spread - selection.betas[symbol] * logp[symbol]
    z = (spread - selection.train_mean) / max(selection.train_std, 1e-12)
    return spread.rename("spread_diagnostic"), z.rename("zscore")


def trade(
    prices: pd.DataFrame,
    selection: ScreenSelection,
    cfg: TradeConfig | None = None,
) -> pd.DataFrame:
    """Default z-score mean-reversion rule. Returns base-asset quantity at every bar.

    The callback sees only close prices and the frozen selection. It does not receive
    feature frames. Long spread = long target / short beta-weighted hedge; short spread
    = the inverse. Quantity is fixed from entry until exit (no hidden rebalancing).
    """
    cfg = cfg or TradeConfig()
    close = _close_frame(prices, selection.symbols)
    _, z = _spread_and_z(close, selection)
    qty = pd.DataFrame(0.0, index=close.index, columns=list(selection.symbols))

    side = 0                    # +1 long spread; -1 short spread
    entry_i = -1
    entry_qty: dict[str, float] = {}
    armed = True

    for i, (ts, zt) in enumerate(z.items()):
        if not np.isfinite(zt):
            continue

        if side == 0:
            if abs(zt) < cfg.entry_z:
                armed = True
            elif armed:
                side = 1 if zt < -cfg.entry_z else -1
                entry_i = i
                gross = cfg.capital * cfg.gross_leverage
                gross_weight = 1.0 + sum(abs(selection.betas[s]) for s in selection.hedge_symbols)
                target_notional = gross / gross_weight
                entry_qty = {selection.target: side * target_notional / float(close.at[ts, selection.target])}
                for s in selection.hedge_symbols:
                    hedge_notional = -side * selection.betas[s] * target_notional
                    entry_qty[s] = hedge_notional / float(close.at[ts, s])
                armed = False

        if side != 0:
            for s in selection.symbols:
                qty.at[ts, s] = entry_qty.get(s, 0.0)

            held = i - entry_i + 1
            # For a long spread (+1), z reverting upward to exit_z closes it.
            # For a short spread (-1), z reverting downward to exit_z closes it.
            reverted = side * (zt - cfg.exit_z) >= 0
            stopped = cfg.stop_z is not None and (-side * zt) >= float(cfg.stop_z)
            timed_out = held >= cfg.max_hold_bars
            if reverted or stopped or timed_out:
                # Exit at this bar's close; this row is flat and the previous holding
                # earns the move into this close through qty.shift(1) in PnL accounting.
                qty.loc[ts, :] = 0.0
                side, entry_i, entry_qty = 0, -1, {}
                armed = abs(zt) < cfg.entry_z

    if cfg.close_at_window_end and len(qty):
        # Do not carry a position into a segment with a newly trained hedge.
        qty.iloc[-1, :] = 0.0
    return qty


def _trade_ledger(
    quantities: pd.DataFrame,
    net_pnl: pd.Series,
    cfg: TradeConfig,
) -> pd.DataFrame:
    active = quantities.abs().sum(axis=1).to_numpy() > 1e-12
    rows: list[dict[str, Any]] = []
    in_trade = False
    trade_pnl = 0.0
    start_i = -1
    trade_no = 0
    for i, is_active in enumerate(active):
        if is_active and not in_trade:
            in_trade = True
            trade_no += 1
            start_i = i
            trade_pnl = 0.0
        if in_trade:
            # Includes entry costs, all bar-to-bar asset PnL, and exit costs.
            trade_pnl += float(net_pnl.iloc[i])
        if in_trade and not is_active:
            risk = cfg.risk_dollars
            rows.append({
                "trade": trade_no,
                "entry_time": quantities.index[start_i],
                "exit_time": quantities.index[i],
                "bars_held": int(i - start_i),
                "pnl": trade_pnl,
                "risk_dollars": risk,
                "r_multiple": trade_pnl / risk,
            })
            in_trade = False
            trade_pnl = 0.0
            start_i = -1
    if in_trade:
        # Normally prevented by close_at_window_end; retain a defensive open-trade row.
        risk = cfg.risk_dollars
        rows.append({
            "trade": trade_no,
            "entry_time": quantities.index[start_i],
            "exit_time": quantities.index[-1],
            "bars_held": int(len(quantities) - 1 - start_i),
            "pnl": trade_pnl,
            "risk_dollars": risk,
            "r_multiple": trade_pnl / risk,
            "open_at_end": True,
        })
    return pd.DataFrame(rows)


def Forward(
    selection: ScreenSelection,
    forward_prices: AssetPanel | pd.DataFrame,
    *,
    cfg: TradeConfig | None = None,
    trade_fn: Callable[[pd.DataFrame, ScreenSelection, TradeConfig], pd.DataFrame] | None = None,
) -> ForwardResult:
    """Run the frozen selection over one forward window (normally 5,000 bars)."""
    cfg = cfg or TradeConfig()
    if any(not t.startswith("px|") for t in selection.terms):
        raise ValueError("Forward only accepts price legs (px|...). Feature terms cannot be executed here.")
    close = _close_frame(forward_prices, selection.symbols)
    spread, z = _spread_and_z(close, selection)
    fn = trade if trade_fn is None else trade_fn
    quantities = fn(close.copy(), selection, cfg)
    if not isinstance(quantities, pd.DataFrame):
        raise TypeError("trade_fn must return a DataFrame of base-asset quantities")
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

    # Actual asset PnL: previous-bar base units * current asset price change.
    # The fitted spread is NOT used in this calculation.
    dprice = close.diff().fillna(0.0)
    gross_pnl = (quantities.shift(1).fillna(0.0) * dprice).sum(axis=1)
    dqty = quantities.diff()
    if len(dqty):
        dqty.iloc[0] = quantities.iloc[0]
    turnover = (dqty.abs() * close).sum(axis=1)
    costs = turnover * cfg.cost_rate
    net_pnl = (gross_pnl - costs).rename("net_pnl")
    gross_pnl = gross_pnl.rename("gross_asset_pnl")
    costs = costs.rename("trading_costs")
    ledger = _trade_ledger(quantities, net_pnl, cfg)
    # OOS diagnostics use the frozen train parameters and ordinary ADF tables;
    # these diagnostics never enter the asset-PnL calculation.
    try:
        oos_stat, _ = adf_stat(spread.to_numpy(float), maxlag=12, const=True)
        oos_p = adf_p(oos_stat, 1)
        oos_hl = half_life(spread.to_numpy(float), floor=1.0)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        oos_stat = oos_p = oos_hl = float("nan")

    return ForwardResult(
        selection=selection,
        close=close,
        spread=spread,
        zscore=z,
        quantities=quantities,
        gross_pnl=gross_pnl,
        costs=costs,
        net_pnl=net_pnl,
        trades=ledger,
        adf_stat_oos=float(oos_stat),
        p_oos=float(oos_p),
        half_life_oos=float(oos_hl),
    )


# --------------------------------------------------------------------------- 3) Algorithm / flip-flop retraining

def retrain_rules(
    train_panel: AssetPanel,
    y: str,
    discover_cfg: DiscoverConfig,
    previous_selection: ScreenSelection | None = None,
) -> ScreenSelection:
    """Default retrain rule: re-run the price-only beam on the latest train window.

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
    trade_fn: Callable[[pd.DataFrame, ScreenSelection, TradeConfig], pd.DataFrame] | None = None,
    retrain_rule: Callable[[AssetPanel, str, DiscoverConfig, ScreenSelection | None], ScreenSelection] | None = None,
    max_cycles: int | None = None,
    print_report: bool = True,
) -> AlgorithmResult:
    """Run two alternating strategy instances in a causal flip-flop schedule.

    Cycle 0 / instance A: train on the initial `train_bars`, trade the next
    `forward_bars`. Cycle 1 / instance B: train on the just-traded window, trade
    the next window. Then A, B, ... alternate. A purge gap is left between each
    training window and its forward window. The two instances alternate ownership;
    they are not simultaneously trading the same timestamps.
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

    dcfg = replace(discover_cfg or DiscoverConfig(), use_features=False)
    tcfg = trade_cfg or TradeConfig()
    rule = retrain_rule or retrain_rules
    slots: dict[str, ScreenSelection | None] = {"A": None, "B": None}
    folds: list[FoldResult] = []
    previous_forward: tuple[int, int] | None = None
    cycle = 0

    while max_cycles is None or cycle < max_cycles:
        if previous_forward is None:
            train_start = offset
            train_end = train_start + train_bars
        else:
            prev_start, prev_end = previous_forward
            if prev_end - prev_start < train_bars:
                raise ValueError(
                    "forward_bars must be at least train_bars so the next instance can train "
                    "on the previous traded window"
                )
            train_start = prev_end - train_bars
            train_end = prev_end

        forward_start = train_end + purge_bars
        forward_end = forward_start + forward_bars
        if train_end > len(p.logp) or forward_end > len(p.logp):
            break

        train_panel = _slice_panel(p, train_start, train_end)
        forward_panel = _slice_panel(p, forward_start, forward_end)
        instance = "A" if cycle % 2 == 0 else "B"
        selection = rule(train_panel, y, dcfg, slots[instance])
        if not isinstance(selection, ScreenSelection):
            raise TypeError("retrain_rule must return ScreenSelection")
        # A custom retrain rule may keep an earlier selection. It must still be executable.
        if selection.target != y or any(not t.startswith("px|") for t in selection.terms):
            raise ValueError("retrain_rule returned a target mismatch or non-price feature term")
        slots[instance] = selection

        fwd = Forward(selection, forward_panel, cfg=tcfg, trade_fn=trade_fn)
        folds.append(FoldResult(
            cycle=cycle,
            instance=instance,
            train_start=p.logp.index[train_start],
            train_end=p.logp.index[train_end - 1],
            forward_start=p.logp.index[forward_start],
            forward_end=p.logp.index[forward_end - 1],
            selection=selection,
            forward=fwd,
        ))
        previous_forward = (forward_start, forward_end)
        cycle += 1

    if not folds:
        needed = train_bars + purge_bars + forward_bars
        raise ValueError(f"not enough data for one train/forward cycle; need at least {needed} bars from start")

    first = folds[0].forward.close.index[0]
    last = folds[-1].forward.close.index[-1]
    full_index = p.logp.index[(p.logp.index >= first) & (p.logp.index <= last)]
    pnl = pd.Series(0.0, index=full_index, name="net_pnl")
    all_symbols = list(p.logp.columns)
    quantities = pd.DataFrame(0.0, index=full_index, columns=all_symbols)
    trade_frames = []
    for fold in folds:
        idx = fold.forward.net_pnl.index
        pnl.loc[idx] = pnl.loc[idx].to_numpy() + fold.forward.net_pnl.to_numpy()
        q = fold.forward.quantities
        quantities.loc[idx, q.columns] = q
        if not fold.forward.trades.empty:
            tr = fold.forward.trades.copy()
            tr.insert(0, "instance", fold.instance)
            tr.insert(0, "cycle", fold.cycle)
            tr.insert(0, "target", fold.selection.target)
            tr.insert(0, "hedge", " + ".join(fold.selection.hedge_symbols))
            trade_frames.append(tr)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(
        columns=["cycle", "instance", "target", "hedge", "trade", "entry_time", "exit_time", "bars_held", "pnl", "risk_dollars", "r_multiple"]
    )

    total_pnl = float(pnl.sum())
    total_r = float(trades["r_multiple"].sum()) if len(trades) else 0.0
    expectancy = float(trades["r_multiple"].mean()) if len(trades) else float("nan")
    bar_minutes = p.bar_minutes
    bars_per_year = 365.25 * 24.0 * 60.0 / bar_minutes if np.isfinite(bar_minutes) and bar_minutes > 0 else 1.0
    sortino = _sortino(pnl, bars_per_year)
    max_dd = _max_drawdown(pnl, tcfg.capital)

    report = (
        "PURGED WALK-FORWARD / FLIP-FLOP REPORT\n"
        f"Folds: {len(folds)} | train={train_bars:,} bars | forward={forward_bars:,} bars | purge={purge_bars:,} bars\n"
        f"Instances: A/B alternating | Closed trades: {len(trades):,}\n"
        f"Net asset PnL: {total_pnl:,.4f} | Accumulated R: {total_r:,.4f}R | "
        f"Expectancy: {expectancy:,.4f}R/trade\n"
        f"Sortino (annualised, bar PnL): {sortino:,.4f} | Max drawdown: {max_dd:.2%}\n"
        f"Median forward ADF p: {np.nanmedian([f.forward.p_oos for f in folds]):.4g} | "
        f"Median forward half-life: {np.nanmedian([f.forward.half_life_oos for f in folds]):.2f} bars\n"
        "PnL uses traded asset quantities x asset price changes; statistical spread is signal-only."
    )
    if print_report:
        print(report)
    return AlgorithmResult(folds, pnl, quantities, trades, total_pnl, total_r, expectancy, sortino, max_dd, report)


# --------------------------------------------------------------------------- 4) stress_test: purged walk-forward + bootstrap path distribution

def stress_test(
    panel: AssetPanel | pd.DataFrame,
    y: str,
    *,
    n_paths: int = 100,
    seed: int = 7,
    train_bars: int = 5_000,
    forward_bars: int = 5_000,
    purge_bars: int = 1,
    discover_cfg: DiscoverConfig | None = None,
    trade_cfg: TradeConfig | None = None,
    trade_fn: Callable[[pd.DataFrame, ScreenSelection, TradeConfig], pd.DataFrame] | None = None,
    retrain_rule: Callable[[AssetPanel, str, DiscoverConfig, ScreenSelection | None], ScreenSelection] | None = None,
    max_cycles: int | None = None,
    plot: bool = True,
    print_report: bool = True,
) -> StressTestResult:
    """Purged walk-forward stress test with bootstrap Monte Carlo trade paths.

    CVaR05 is the mean of the worst 5% of path-level average-R outcomes; CVaR95
    is the mean of the best 5%. P05/P95 are the corresponding path-expectancy
    quantiles. Each path resamples the realised OOS trade R-multiples with
    replacement, preserving the number of trades in the observed walk-forward.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    alg = Algorithm(
        panel, y,
        train_bars=train_bars,
        forward_bars=forward_bars,
        purge_bars=purge_bars,
        discover_cfg=discover_cfg,
        trade_cfg=trade_cfg,
        trade_fn=trade_fn,
        retrain_rule=retrain_rule,
        max_cycles=max_cycles,
        print_report=False,
    )
    r = alg.trades["r_multiple"].to_numpy(float) if len(alg.trades) else np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    n_trades = len(r)
    paths = np.zeros((n_paths, n_trades), dtype=float)
    expectancies = np.full(n_paths, np.nan, dtype=float)
    if n_trades:
        samples = rng.choice(r, size=(n_paths, n_trades), replace=True)
        paths = np.cumsum(samples, axis=1)
        expectancies = samples.mean(axis=1)
        p05, p95 = np.quantile(expectancies, [0.05, 0.95])
        c05 = float(expectancies[expectancies <= p05].mean())
        c95 = float(expectancies[expectancies >= p95].mean())
        mean_exp = float(expectancies.mean())
    else:
        p05 = p95 = c05 = c95 = mean_exp = float("nan")

    path_cols = [f"path_{i + 1:03d}" for i in range(n_paths)]
    if n_trades:
        cumulative_paths = pd.DataFrame(paths.T, columns=path_cols, index=pd.RangeIndex(1, n_trades + 1, name="trade"))
    else:
        cumulative_paths = pd.DataFrame(columns=path_cols, index=pd.RangeIndex(0, name="trade"))
    expectancy_series = pd.Series(expectancies, name="mean_R_per_trade")

    fig = None
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 5))
            if n_trades:
                ax.plot(cumulative_paths.index, cumulative_paths.to_numpy(), alpha=0.20, linewidth=0.9)
                mean_path = cumulative_paths.mean(axis=1)
                ax.plot(mean_path.index, mean_path.to_numpy(), linewidth=2.2, label="Mean bootstrap path")
                ax.legend(loc="best")
            else:
                ax.text(0.5, 0.5, "No closed OOS trades", ha="center", va="center", transform=ax.transAxes)
            ax.axhline(0.0, linewidth=0.8, linestyle="--")
            ax.set_title("Purged walk-forward: bootstrap cumulative R paths")
            ax.set_xlabel("Closed OOS trade")
            ax.set_ylabel("Cumulative R")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        except ImportError:
            warnings.warn("matplotlib is not installed; Monte Carlo paths are returned without a figure")

    report = (
        "PURGED WALK-FORWARD STRESS TEST\n"
        f"Bootstrap paths: {n_paths:,} | OOS trades: {n_trades:,} | Folds: {len(alg.folds):,}\n"
        f"Net asset PnL: {alg.total_pnl:,.4f} | Accumulated R: {alg.total_r:,.4f}R | "
        f"Sortino: {alg.sortino:,.4f}\n"
        f"RR expectancy: {mean_exp:,.4f}R/trade | P05/P95: [{p05:,.4f}, {p95:,.4f}]R\n"
        f"CVaR05/CVaR95: [{c05:,.4f}, {c95:,.4f}]R\n"
        "CVaR05/95 are tail means of bootstrap path-level average R, not spread returns."
    )
    if print_report:
        print(report)
    return StressTestResult(alg, expectancy_series, cumulative_paths, mean_exp, float(p05), float(p95), c05, c95, report, fig)


__all__ = [
    "TradeConfig", "ScreenSelection", "ForwardResult", "FoldResult", "AlgorithmResult",
    "StressTestResult", "load_price_panel", "Screener", "trade", "Forward",
    "retrain_rules", "Algorithm", "stress_test",
]

if __name__ == "__main__":
    print("=== STRATEGY STARTED ===", flush=True)

    try:
        SYMBOLS = [
            "ETH/USDT",
            "BTC/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
        ]

        TARGET = "ETH/USDT"

        START = "2024-01-01"
        END = "2024-07-01"

        print("[1/3] Loading price data...", flush=True)

        panel = load_price_panel(
            symbols=SYMBOLS,
            start=START,
            end=END,
            bars=None,
            bar="5min",
            market="um",
            workers=4,
        )

        print(f"Loaded {len(panel.logp)} bars.", flush=True)

        print("[2/3] Running walk-forward stress test...", flush=True)

        # DiscoverConfig is already imported at module level

        result = stress_test(
            panel=panel,
            y=TARGET,
            n_paths=100,
            seed=42,
            train_bars=5000,
            forward_bars=5000,
            purge_bars=1,
            discover_cfg=DiscoverConfig(use_features=False),
            trade_cfg=TradeConfig(
                entry_z=2.0,
                exit_z=0.0,
                stop_z=None,
                capital=10_000,
                risk_fraction=0.01,
                fee_bps=5,
            ),
            plot=True,
            print_report=True,
        )

        print("[3/3] Completed successfully.", flush=True)

        if result.figure is not None:
            import matplotlib.pyplot as plt
            plt.show()

    except Exception:
        import traceback
        traceback.print_exc()
        raise