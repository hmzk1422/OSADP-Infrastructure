"""backtest.py - the backtesting system (replaces mc_stress_test.py).

phase_1   Randomised chunks, many iterations, one parallel pool
          * every iteration draws its own random composition of `total_bars` into segments (10k, 25k, 30k, 15k, ... multiples of
            chunk_bars, each >= min_segment_bars) at random start dates; inside a segment the flip-flop runs (train chunk -> trade the
            next chunk -> train on the chunk just traded -> ...);  n_iterations independent draws (e.g. 20) give a DISTRIBUTION of
            outcomes instead of one lucky / unlucky draw.
          * data is built ONCE, month by month, at the base 1-minute bar (features included) and cached on disk; every iteration,
            every run and (phase_2) every timeframe reads the same cache.  Workers read the cache themselves, so no panel is pickled.
          * the unit of work is a SEGMENT: a worker loads it, runs the beam search ONCE per fold, derives every strategy variant's
            selection from that single ranked list (half-life predictor variants only add a cheap post-processing step), then trades
            every strategy on it (the smart MAE/MFE regression is shared between strategies that differ only in rr / mode / costs).
            All segments of all iterations are scheduled together, biggest first, on `n_workers` processes (or threads).
          * complexity: the beam search is the only super-linear piece and uses the Gram-matrix evaluator of spread_discoverer
            (O(n p) per candidate instead of lstsq + 13-regression ADF + ridge refits); the rest is linear in bars.
phase_2   Same engine, plus mixed timeframes: each segment draws its bar size from `timeframes` (1, 5, 15, ... minutes); the cached
          1-minute panel (features, highs/lows) is aggregated into that bar size (spread_discoverer.resample_panel), so no feature is
          recomputed.  Optional: leave timeframes=(1,) for phase_1 only.

Flat folder:  spread_discoverer.py  strategy.py  backtest.py  (test_alt3.py = unit tests)
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

try:
    from .spread_discoverer import AssetPanel, DiscoverConfig, get_assets, resample_panel
    from .strategy import (HiLo, TradeConfig, beam_rank, load_hilo, parallel_map, plan_folds, resample_hilo, screen_fold,
                           trade_fold, _hilo_window, DEFAULT_BAR, DatasetConfig, HalfLifePredictor, HalfLifeRetrainRule,
                           build_dataset, compare_predictors)
except ImportError:
    from spread_discoverer import AssetPanel, DiscoverConfig, get_assets, resample_panel
    from strategy import (HiLo, TradeConfig, beam_rank, load_hilo, parallel_map, plan_folds, resample_hilo, screen_fold,
                          trade_fold, _hilo_window, DEFAULT_BAR, DatasetConfig, HalfLifePredictor, HalfLifeRetrainRule,
                          build_dataset, compare_predictors)


# =============================================================================== configuration
@dataclass
class BacktestConfig:
    symbols: tuple = ("ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT")
    target: str = "ETH/USDT"
    history_start: str = "2024-01-01"      # segments are drawn from [history_start, history_end]; must be AFTER every model-training range
    history_end: str = "2026-07-31"
    # --- phase_1: random chunks x iterations
    total_bars: int = 100_000              # data budget of ONE iteration (train windows included), in bars of the segment's timeframe
    chunk_bars: int = 5_000                # train window = traded window = one measured chunk
    min_segment_bars: int = 10_000
    max_segment_bars: int | None = None
    purge_bars: int = 1
    n_iterations: int = 20
    seed: int | None = 7                   # None -> a fresh draw every run (the seed used is stored back and printed)
    n_boot: int = 2_000
    # --- parallelism
    n_workers: int = 2                     # conservative default; raise explicitly for larger-memory hosts
    backend: str = "loky"                  # "loky" (processes) | "threading"
    # --- data cache (1-minute base bars, month granularity)
    cache_dir: str = "bt_cache"
    base_bar: str = DEFAULT_BAR
    lead_days: int = 3                     # extra days loaded before each month so rolling features are warm at the month start
    max_gap_bars: int = 30                 # a hole longer than this (in base bars) makes a range unusable (the segment is redrawn)
    max_draw_retries: int = 25
    # --- phase_2 (optional): mixed timeframes
    timeframes: tuple = (1,)               # minutes; (1, 5, 15) mixes bar sizes per segment
    feature_mode: str = "aggregate"        # "aggregate" (features resampled) | "drop" (price-only panels above 1 minute)
    predictor_timeframes: tuple = (1,)     # timeframes where a pre-trained HalfLifePredictor may be used (others: train half-life)
    report_dir: str | None = "bt_reports"

    def __post_init__(self):
        if self.seed is None:
            self.seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        c = self.chunk_bars
        if c < 200:
            raise ValueError("chunk_bars must be >= 200")
        for name in ("total_bars", "min_segment_bars"):
            if getattr(self, name) % c:
                raise ValueError(f"{name} must be a multiple of chunk_bars ({c:,})")
        if self.max_segment_bars is not None and (self.max_segment_bars % c or self.max_segment_bars < self.min_segment_bars):
            raise ValueError("max_segment_bars must be a multiple of chunk_bars and >= min_segment_bars")
        if self.min_segment_bars < 2 * c:
            raise ValueError("min_segment_bars must be >= 2 * chunk_bars (one train chunk + one traded chunk)")
        if self.total_bars < self.min_segment_bars:
            raise ValueError("total_bars must be >= min_segment_bars")
        if pd.Timestamp(self.history_end) <= pd.Timestamp(self.history_start):
            raise ValueError("history_end must be after history_start")
        if self.backend not in ("loky", "threading"):
            raise ValueError("backend must be 'loky' or 'threading'")
        if self.feature_mode not in ("aggregate", "drop"):
            raise ValueError("feature_mode must be 'aggregate' or 'drop'")
        if any(int(t) < 1 for t in self.timeframes):
            raise ValueError("timeframes are whole minutes >= 1")

    @property
    def base_minutes(self) -> float:
        return pd.Timedelta(self.base_bar).total_seconds() / 60.0


@dataclass
class StrategySpec:
    """One row of the comparison: which half-life variant (key into `variants`, None = train half-life) and which TradeConfig."""

    variant: str | None = None
    trade: dict = field(default_factory=dict)       # TradeConfig kwargs; TradeConfig.for_bar(timeframe, **trade) is applied per segment


@dataclass
class SegSpec:
    iteration: int
    segment: int
    tf: int                  # minutes per bar of this segment
    bars: int                # nominal length in bars (multiple of chunk_bars)
    n_folds: int
    need_bars: int           # bars that must exist (adds one purge gap per fold)
    start: pd.Timestamp      # requested start


# =============================================================================== random segments
def draw_segment_lengths(total_units: int, min_units: int, max_units: int | None, rng: np.random.Generator) -> list[int]:
    """Uniformly random composition of `total_units` into parts in [min_units, max_units] (units = chunks)."""
    cap = max_units or total_units
    ways = [0] * (total_units + 1)
    ways[0] = 1
    for r in range(1, total_units + 1):
        ways[r] = sum(ways[r - k] for k in range(min_units, min(r, cap) + 1))
    if ways[total_units] == 0:
        raise ValueError("total_bars cannot be split with these min/max segment sizes")
    out, rem = [], total_units
    while rem > 0:
        ks = [k for k in range(min_units, min(rem, cap) + 1) if ways[rem - k] > 0]
        w = np.array([ways[rem - k] for k in ks], dtype=float)
        k = int(rng.choice(ks, p=w / w.sum()))
        out.append(k)
        rem -= k
    return out


def _span_minutes(bt: BacktestConfig, need_bars: int, tf: int) -> float:
    return float(need_bars * tf + (100 + 3) * tf)          # + resample warm-up (vol window) and bucket slack


def _random_start(bt: BacktestConfig, span_min: float, rng: np.random.Generator) -> pd.Timestamp:
    h0, h1 = pd.Timestamp(bt.history_start).normalize(), pd.Timestamp(bt.history_end)
    free_days = int((h1 - pd.Timedelta(minutes=span_min) - h0) / pd.Timedelta(days=1))
    if free_days < 0:
        raise ValueError(f"a {span_min / 1440:.0f}-day segment does not fit inside {bt.history_start} .. {bt.history_end}")
    return h0 + pd.Timedelta(days=int(rng.integers(0, free_days + 1)))


def draw_iteration(bt: BacktestConfig, iteration: int, store: "DataStore | None" = None, verbose: bool = False) -> list[SegSpec]:
    """One iteration's random composition of segments (lengths, timeframes, start dates).  Deterministic in (seed, iteration).
    With a store, a start whose data range has holes / is unavailable is redrawn (and the months are built into the cache)."""
    rng = np.random.default_rng([int(bt.seed), int(iteration)])
    units = draw_segment_lengths(bt.total_bars // bt.chunk_bars, bt.min_segment_bars // bt.chunk_bars,
                                 None if bt.max_segment_bars is None else bt.max_segment_bars // bt.chunk_bars, rng)
    segs = []
    for k, u in enumerate(units):
        tf = int(rng.choice(list(bt.timeframes)))
        n_folds = u - 1
        need = u * bt.chunk_bars + n_folds * bt.purge_bars
        span = _span_minutes(bt, need, tf)
        for attempt in range(bt.max_draw_retries + 1):
            start = _random_start(bt, span, rng)
            if store is None:
                break
            end = start + pd.Timedelta(minutes=span) + pd.Timedelta(days=1)
            if store.ensure(start - pd.Timedelta(minutes=(100 + 3) * tf), end, verbose=verbose) and \
                    store.range_ok(start - pd.Timedelta(minutes=(100 + 3) * tf), end):
                break
            if attempt == bt.max_draw_retries:
                raise RuntimeError(f"iteration {iteration} segment {k + 1}: no loadable start in {bt.max_draw_retries} draws")
        segs.append(SegSpec(iteration, k + 1, tf, u * bt.chunk_bars, n_folds, need, start))
    return segs


# =============================================================================== data cache: 1-minute panels, one file per month
def make_loader(symbols: Sequence[str], discover_cfg: DiscoverConfig, *, bar: str = DEFAULT_BAR, market: str = "um",
                cache_dir: str = "~/.cache/binance_vision", workers: int = 4) -> Callable:
    """Default loader (start, end) -> (AssetPanel with features, HiLo): spread_discoverer.get_assets + binance_vision highs/lows.
    Any callable with that signature works (your own parquet reader, a synthetic generator for tests)."""
    def loader(start: pd.Timestamp, end: pd.Timestamp):
        s, e = pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")
        panel = get_assets(list(symbols), s, e, bar=bar, market=market, cfg=discover_cfg)
        return panel, load_hilo(symbols, s, e, bar=bar, market=market, cache_dir=cache_dir, workers=workers)
    return loader


def _ts(value, index: pd.Index) -> pd.Timestamp:
    t = pd.Timestamp(value)
    tz = getattr(index, "tz", None)
    if tz is not None:
        return t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)
    return t.tz_localize(None) if t.tzinfo is not None else t


class DataStore:
    """Month-granular cache of base-bar panels (+ high/low).  The parent builds missing months (`ensure`); workers only `read`.

    Each month is loaded with `lead_days` of history before it and then trimmed, so rolling-feature warm-up never leaves a hole at a
    month boundary.  A month's metadata (first/last bar, holes longer than max_gap_bars) lets the sampler reject unusable ranges
    without touching the data."""

    def __init__(self, cache_dir: str, symbols: Sequence[str], base_bar: str = DEFAULT_BAR, loader: Callable | None = None,
                 lead_days: int = 3, max_gap_bars: int = 30, feature_tag: str = ""):
        self.symbols, self.base_bar, self.loader = tuple(symbols), base_bar, loader
        self.lead_days, self.max_gap_bars = int(lead_days), int(max_gap_bars)
        key = hashlib.md5(("|".join(self.symbols) + base_bar + feature_tag).encode()).hexdigest()[:10]
        self.dir = os.path.join(os.path.expanduser(cache_dir), key)
        self._lru: dict[str, Any] = {}

    def __getstate__(self):                       # workers only read: never ship the loader or the in-memory months
        d = dict(self.__dict__)
        d["loader"], d["_lru"] = None, {}
        return d

    # ---- paths / month arithmetic
    def _path(self, ym: str, kind: str = "data") -> str:
        return os.path.join(self.dir, f"{ym}.{kind}.pkl")

    @staticmethod
    def months(start, end) -> list[str]:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s, e = (s.tz_localize(None) if s.tzinfo else s), (e.tz_localize(None) if e.tzinfo else e)
        return [str(p) for p in pd.period_range(s.to_period("M"), (e - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")]

    @property
    def bar_td(self) -> pd.Timedelta:
        return pd.Timedelta(self.base_bar)

    # ---- build (parent process)
    def _build_month(self, ym: str, verbose: bool) -> bool:
        if self.loader is None:
            return False
        ms = pd.Period(ym, "M").start_time
        me = (pd.Period(ym, "M") + 1).start_time
        try:
            t0 = time.time()
            panel, hilo = self.loader(ms - pd.Timedelta(days=self.lead_days), me)
            idx = panel.logp.index
            a, b = idx.searchsorted(_ts(ms, idx)), idx.searchsorted(_ts(me, idx))
            if b - a < 100:
                raise ValueError(f"only {b - a} bars in {ym}")
            panel = panel.iloc(a, b)
            hilo = hilo.between(panel.logp.index[0], panel.logp.index[-1])
            gaps = []
            d = panel.logp.index.to_series().diff()
            for t, dt in d[d > self.bar_td * self.max_gap_bars].items():
                gaps.append((t - dt, t))
            os.makedirs(self.dir, exist_ok=True)
            with open(self._path(ym), "wb") as fh:
                pickle.dump(dict(panel=panel, hilo=hilo), fh, protocol=pickle.HIGHEST_PROTOCOL)
            meta = dict(ok=True, first=panel.logp.index[0], last=panel.logp.index[-1], gaps=gaps, bars=len(panel.logp))
            if verbose:
                print(f"[store] built {ym}: {len(panel.logp):,} bars, {len(gaps)} gap(s) in {time.time() - t0:,.0f}s", flush=True)
        except Exception as exc:                  # a missing month must not kill the run: it is recorded and the range redrawn
            meta = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
            os.makedirs(self.dir, exist_ok=True)
            if verbose:
                print(f"[store] {ym} unavailable: {meta['error']}", flush=True)
        with open(self._path(ym, "meta"), "wb") as fh:
            pickle.dump(meta, fh)
        return bool(meta["ok"])

    def _meta(self, ym: str) -> dict | None:
        p = self._path(ym, "meta")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as fh:
            return pickle.load(fh)

    def ensure(self, start, end, verbose: bool = False) -> bool:
        """Make sure every month touching [start, end) is cached.  False if any month is unavailable."""
        ok = True
        for ym in self.months(start, end):
            m = self._meta(ym)
            if m is None:
                self._build_month(ym, verbose)
                m = self._meta(ym)
            ok &= bool(m and m["ok"])
        return ok

    def range_ok(self, start, end) -> bool:
        """Metadata-only check: all months present and no hole (> max_gap_bars) inside [start, end)."""
        s, e = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start), \
            pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)
        tol = self.bar_td * self.max_gap_bars
        for ym in self.months(s, e):
            m = self._meta(ym)
            if not m or not m["ok"]:
                return False
            lo = max(s, pd.Period(ym, "M").start_time)                      # the part of [s, e) that falls in this month must be covered
            hi = min(e, (pd.Period(ym, "M") + 1).start_time)
            first, last = pd.Timestamp(m["first"]), pd.Timestamp(m["last"])
            first, last = (first.tz_localize(None) if first.tzinfo else first), (last.tz_localize(None) if last.tzinfo else last)
            if first > lo + tol or last < hi - tol - self.bar_td:
                return False
            for g0, g1 in m["gaps"]:
                g0, g1 = pd.Timestamp(g0).tz_localize(None) if g0.tzinfo else g0, pd.Timestamp(g1).tz_localize(None) if g1.tzinfo else g1
                if g0 < e and g1 > s:
                    return False
        return True

    # ---- read (workers)
    def _month(self, ym: str):
        if ym not in self._lru:
            if len(self._lru) >= 3:
                self._lru.pop(next(iter(self._lru)))
            with open(self._path(ym), "rb") as fh:
                self._lru[ym] = pickle.load(fh)
        return self._lru[ym]

    def read(self, start, end) -> tuple[AssetPanel, HiLo]:
        """Panel + HiLo for [start, end) at the base bar, stitched from the cached months."""
        mons = [self._month(ym) for ym in self.months(start, end)]
        panel = AssetPanel.concat([m["panel"] for m in mons])
        hilo = HiLo(pd.concat([m["hilo"].high for m in mons]), pd.concat([m["hilo"].low for m in mons]))
        idx = panel.logp.index
        a, b = idx.searchsorted(_ts(start, idx)), idx.searchsorted(_ts(end, idx))
        if b - a < 10:
            raise ValueError(f"no data in {start} .. {end}")
        panel = panel.iloc(a, b)
        return panel, hilo.between(panel.logp.index[0], panel.logp.index[-1])


# =============================================================================== segment job (runs in a worker)
_SCALE_FIELDS = ("vr_lags", "hl_ref", "hl_cap", "half_life_max_bars", "z_window", "z_max_hold", "drift_window", "drift_stride")


def discover_cfg_for_bar(base: DiscoverConfig, bar_minutes: float, features: bool = True) -> DiscoverConfig:
    """Rescale the bar-denominated settings of the user's DiscoverConfig to another bar size (1-minute = unchanged)."""
    cfg = replace(base, use_features=base.use_features and features)
    if abs(bar_minutes - 1.0) < 1e-9:
        return cfg
    scaled = DiscoverConfig.for_bar(bar_minutes)
    return replace(cfg, **{f: getattr(scaled, f) for f in _SCALE_FIELDS})


def _fold_row(spec: SegSpec, plan, fold, trades: pd.DataFrame) -> dict:
    sel, fwd = fold.selection, fold.forward
    r = trades["r_multiple"].to_numpy(float) if len(trades) else np.array([])
    rs = trades["reason"] if len(trades) else pd.Series(dtype=object)
    pnl = fwd.net_pnl.to_numpy(float)
    eq = np.cumsum(pnl)
    sm = sel.smart
    return dict(
        iteration=spec.iteration, segment=spec.segment, tf=spec.tf, chunk=plan.cycle, instance=plan.instance,
        train_start=fold.train_start, oos_start=fold.forward_start, oos_end=fold.forward_end,
        hedge=" + ".join(sel.hedge_symbols), z_window=sel.z_window, hl_train=sel.half_life_asset, hl_pred=sel.hl_pred,
        vr_train=sel.candidate.vr, vr_oos=fwd.vr_oos, hl_oos=fwd.half_life_oos, adf_p_oos=fwd.p_oos,
        trades=len(r), sum_r=float(r.sum()), mean_r=float(r.mean()) if len(r) else np.nan,
        tp_rate=float((rs == "tp").mean()) if len(r) else np.nan, sl_rate=float((rs == "sl").mean()) if len(r) else np.nan,
        time_rate=float((rs == "time").mean()) if len(r) else np.nan, net_pnl=float(pnl.sum()),
        max_dd_pnl=float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0,
        smart_events=sm.diag.get("n_labelled", np.nan) if sm else np.nan, smart_val_auc=sm.diag.get("val_auc", np.nan) if sm else np.nan,
        smart_abstain=bool(sm.diag.get("rl_abstain", False)) if sm else False)


def run_segment(spec: SegSpec, store: DataStore, y: str, bt: BacktestConfig, base_dcfg: DiscoverConfig,
                variants: dict[str, Any], strategies: dict[str, StrategySpec]) -> dict:
    """Load the segment, run the flip-flop folds.  Per fold: ONE beam search; every strategy then trades the same windows."""
    out: dict[str, Any] = dict(spec=spec, folds={k: [] for k in strategies}, trades={k: [] for k in strategies}, errors=[])
    try:
        warm = pd.Timedelta(minutes=(100 + 3) * spec.tf)
        start = pd.Timestamp(spec.start)
        end = start + pd.Timedelta(minutes=spec.need_bars * spec.tf) + pd.Timedelta(days=1)
        panel, hilo = store.read(start - warm, end)
        if spec.tf > 1:
            panel = resample_panel(panel, spec.tf, features=bt.feature_mode == "aggregate")
            hilo = resample_hilo(hilo, spec.tf)
            idx = panel.logp.index
            panel = panel.iloc(idx.searchsorted(_ts(start, idx)), len(idx))
        if len(panel.logp) < spec.need_bars:
            raise ValueError(f"loaded {len(panel.logp):,} bars, need {spec.need_bars:,}")
        panel = panel.iloc(0, spec.need_bars)
        dcfg = discover_cfg_for_bar(base_dcfg, spec.tf, features=bool(panel.feats))
        tcfgs = {k: TradeConfig.for_bar(float(spec.tf), **s.trade) for k, s in strategies.items()}
        plans = plan_folds(len(panel.logp), train_bars=bt.chunk_bars, forward_bars=bt.chunk_bars, purge_bars=bt.purge_bars)
        for plan in plans:
            tp = panel.iloc(plan.train_start, plan.train_end)
            fp = panel.iloc(plan.forward_start, plan.forward_end)
            hl = _hilo_window(hilo, tp, fp)
            try:
                ranked = beam_rank(tp, y, dcfg)
                sels: dict[str | None, Any] = {}
                for name in {s.variant for s in strategies.values()}:
                    rule = variants.get(name) if name is not None else None
                    if rule is not None and spec.tf not in bt.predictor_timeframes:
                        rule = None                       # a 1-minute predictor is not applied to other bar sizes
                    sels[name] = screen_fold(tp, y, dcfg, retrain_rule=rule, ranked=ranked)
            except Exception as exc:
                out["errors"].append(f"it{spec.iteration} seg{spec.segment} chunk{plan.cycle}: screen failed ({type(exc).__name__}: {exc})")
                continue
            cache: dict = {}
            for k, s in strategies.items():
                try:
                    fold = trade_fold(sels[s.variant], tp, fp, plan, tcfgs[k], hilo=hl, stage_cache=cache)
                except Exception as exc:
                    out["errors"].append(f"it{spec.iteration} seg{spec.segment} chunk{plan.cycle} [{k}]: trade failed ({type(exc).__name__}: {exc})")
                    continue
                tr = fold.forward.trades
                if len(tr):
                    tr = tr[["entry_time", "exit_time", "bars_held", "side", "reason", "leverage", "stop_pct", "p_enter", "pnl",
                             "risk_dollars", "r_multiple"]].copy()
                    tr.insert(0, "chunk", plan.cycle)
                    tr.insert(0, "segment", spec.segment)
                    tr.insert(0, "iteration", spec.iteration)
                    tr.insert(3, "tf", spec.tf)
                    out["trades"][k].append(tr)
                out["folds"][k].append(_fold_row(spec, plan, fold, fold.forward.trades))
    except Exception as exc:
        out["errors"].append(f"it{spec.iteration} seg{spec.segment}: {type(exc).__name__}: {exc}")
    return out


# =============================================================================== statistics (chunk bootstrap, per-iteration distribution)
def _tail_means(x: np.ndarray):
    p05, p95 = np.quantile(x, [0.05, 0.95])
    return float(p05), float(p95), float(x[x <= p05].mean()), float(x[x >= p95].mean())


def _stats(r: np.ndarray, chunk_sum: np.ndarray, chunk_n: np.ndarray, n_boot: int, seed: int) -> dict:
    """Trade-level stats + chunk-level bootstrap (resample whole OOS chunks: trades inside a chunk are correlated)."""
    n, K = len(r), len(chunk_sum)
    nan = float("nan")
    st = dict(n_trades=n, n_chunks=K, mean_r=nan, sd_r=nan, se_r=nan, t_stat=nan, win_rate=nan, boot_mean=nan, p05=nan, p95=nan,
              cvar05=nan, cvar95=nan, ci_lo=nan, ci_hi=nan, p_positive=nan, pos_chunks=nan, n_required=nan, total_r=float(r.sum()) if n else 0.0)
    if n == 0 or K == 0:
        return st
    st["mean_r"] = float(r.mean())
    st["sd_r"] = float(r.std(ddof=1)) if n > 1 else nan
    st["se_r"] = st["sd_r"] / np.sqrt(n) if n > 1 else nan
    st["t_stat"] = st["mean_r"] / st["se_r"] if n > 1 and st["se_r"] > 0 else nan
    st["win_rate"] = float((r > 0).mean())
    has = chunk_n > 0
    st["pos_chunks"] = float(((chunk_sum[has] / chunk_n[has]) > 0).sum())
    if st["mean_r"] != 0 and np.isfinite(st["sd_r"]):
        st["n_required"] = float((1.96 * st["sd_r"] / st["mean_r"]) ** 2)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, K, size=(n_boot, K))
    sums, cnt = chunk_sum[idx], chunk_n[idx]
    tot = cnt.sum(axis=1)
    ok = tot > 0
    e = sums[ok].sum(axis=1) / tot[ok]
    if len(e):
        st["boot_mean"] = float(e.mean())
        st["p05"], st["p95"], st["cvar05"], st["cvar95"] = _tail_means(e)
        st["ci_lo"], st["ci_hi"] = (float(v) for v in np.quantile(e, [0.025, 0.975]))
        st["p_positive"] = float((e > 0).mean())
    return st


@dataclass
class StrategyResult:
    name: str
    spec: StrategySpec
    chunks: pd.DataFrame          # one row per OOS chunk (train vs OOS statistics + trading stats)
    trades: pd.DataFrame
    iterations: pd.DataFrame      # one row per iteration
    by_timeframe: pd.DataFrame
    stats: dict
    report: str


@dataclass
class BacktestResult:
    bt: BacktestConfig
    segments: pd.DataFrame
    strategies: dict[str, StrategyResult]
    summary: pd.DataFrame
    errors: list[str]
    seconds: float
    report_files: list[str] = field(default_factory=list)


def _agg_group(g: pd.DataFrame) -> pd.Series:
    r_n = g["trades"].sum()
    return pd.Series(dict(chunks=len(g), trades=int(r_n), sum_r=g["sum_r"].sum(), mean_r=g["sum_r"].sum() / r_n if r_n else np.nan,
                          pos_chunks=int((g["mean_r"] > 0).sum()), net_pnl=g["net_pnl"].sum()))


def _summarise(name: str, spec: StrategySpec, folds: list[dict], trades: list[pd.DataFrame], bt: BacktestConfig) -> StrategyResult:
    chunks = pd.DataFrame(folds)
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["iteration", "r_multiple", "reason", "tf"])
    if chunks.empty:
        st = _stats(np.array([]), np.array([]), np.array([]), bt.n_boot, int(bt.seed) + 1)
        return StrategyResult(name, spec, chunks, tr, pd.DataFrame(), pd.DataFrame(), st, f"[{name}] no chunk produced a result")
    r = tr["r_multiple"].to_numpy(float) if len(tr) else np.array([])
    st = _stats(r, chunks["sum_r"].to_numpy(float), chunks["trades"].to_numpy(float), bt.n_boot, int(bt.seed) + 1)
    its = chunks.groupby("iteration").apply(_agg_group, include_groups=False).reset_index()
    by_tf = chunks.groupby("tf").apply(_agg_group, include_groups=False).reset_index()
    it_e = its["mean_r"].dropna().to_numpy(float)
    L = [f"=== {name}  (variant={spec.variant}, smart={spec.trade.get('smart_trading', False)}"
         + (f"/{spec.trade.get('smart_mode', 'classifier')}" if spec.trade.get("smart_trading") else "") + ")",
         f"iterations {len(its)} | segments {chunks[['iteration', 'segment']].drop_duplicates().shape[0]} | OOS chunks {int(st['n_chunks'])}"
         f" | trades {int(st['n_trades']):,} | abstaining smart folds {int(chunks['smart_abstain'].sum())}"]
    if st["n_trades"]:
        rs = tr["reason"]
        mix = " | ".join(f"{k} {float((rs == k).mean()):.1%}" for k in ("tp", "sl", "time", "window_end") if (rs == k).any())
        L += [f"exits: {mix}",
              f"Mean R/trade {st['mean_r']:+.4f} | sd {st['sd_r']:.3f} | trade-level t {st['t_stat']:+.2f} | win rate {st['win_rate']:.1%} | total {st['total_r']:+.1f}R",
              f"Chunk bootstrap ({bt.n_boot:,}): E[R] {st['boot_mean']:+.4f}  95% CI [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]  "
              f"P(E[R]>0) {st['p_positive']:.1%}  P05/P95 [{st['p05']:+.4f}, {st['p95']:+.4f}]  CVaR05 {st['cvar05']:+.4f}",
              f"chunks with positive expectancy: {int(st['pos_chunks'])}/{int((chunks['trades'] > 0).sum())}"]
        if len(it_e):
            L.append(f"Per-iteration E[R] ({len(it_e)} independent draws): mean {it_e.mean():+.4f} | median {np.median(it_e):+.4f} | "
                     f"P05/P95 [{np.quantile(it_e, 0.05):+.4f}, {np.quantile(it_e, 0.95):+.4f}] | iterations > 0: {(it_e > 0).mean():.0%}")
        if np.isfinite(st["n_required"]):
            L.append(f"Trades needed for 95% significance (trade-level): ~{st['n_required']:,.0f} | have {int(st['n_trades']):,} -> "
                     f"{'enough' if st['n_trades'] >= st['n_required'] else 'NOT enough'}")
    else:
        L.append("No closed OOS trades.")
    return StrategyResult(name, spec, chunks, tr, its, by_tf, st, "\n".join(L))


# =============================================================================== phase runner
def run_backtest(bt: BacktestConfig, store: DataStore, base_dcfg: DiscoverConfig, strategies: dict[str, StrategySpec],
                 variants: dict[str, Any] | None = None, *, verbose: bool = True) -> BacktestResult:
    """Draw `n_iterations` random compositions, build any missing months of the cache, run every segment of every iteration
    (all strategies, one beam search per fold) on one worker pool, aggregate.  Same BacktestConfig.seed -> same draws."""
    t0 = time.time()
    variants = variants or {}
    for k, s in strategies.items():
        if s.variant is not None and s.variant not in variants:
            raise KeyError(f"strategy {k!r} uses variant {s.variant!r}, which is not in `variants`")
    if verbose:
        print(f"[bt] seed {bt.seed} | {bt.n_iterations} iterations x {bt.total_bars:,} bars | timeframes {bt.timeframes} | "
              f"strategies {list(strategies)}", flush=True)
    segs: list[SegSpec] = []
    for it in range(bt.n_iterations):
        segs += draw_iteration(bt, it, store, verbose=verbose)
    if verbose:
        tot = sum(s.bars for s in segs)
        print(f"[bt] drew {len(segs)} segments ({tot:,} bars in total) in {time.time() - t0:,.0f}s; running them on "
              f"{bt.n_workers} worker(s) [{bt.backend}]", flush=True)
    order = sorted(range(len(segs)), key=lambda i: -segs[i].bars)            # biggest first: best load balance
    jobs = [dict(spec=segs[i], store=store, y=bt.target, bt=bt, base_dcfg=base_dcfg, variants=variants, strategies=strategies) for i in order]
    t1 = time.time()
    done = parallel_map(run_segment, jobs, bt.n_workers, bt.backend, verbose=10 if verbose else 0)
    results = [None] * len(segs)
    for pos, i in enumerate(order):
        results[i] = done[pos]
    errors = [e for r in results for e in r["errors"]]
    seg_df = pd.DataFrame([dict(iteration=s.iteration, segment=s.segment, tf=s.tf, bars=s.bars, start=s.start,
                                chunks=len(r["folds"][next(iter(strategies))])) for s, r in zip(segs, results)])
    out = {}
    for k, spec in strategies.items():
        folds = [f for r in results for f in r["folds"][k]]
        trades = [t for r in results for t in r["trades"][k]]
        out[k] = _summarise(k, spec, folds, trades, bt)
    summary = pd.DataFrame([{
        "strategy": k, "variant": v.spec.variant, "smart": v.spec.trade.get("smart_mode") if v.spec.trade.get("smart_trading") else "-",
        "trades": int(v.stats["n_trades"]), "mean_R": v.stats["mean_r"], "ci_lo": v.stats["ci_lo"], "ci_hi": v.stats["ci_hi"],
        "P(E>0)": v.stats["p_positive"], "win": v.stats["win_rate"], "total_R": v.stats["total_r"],
        "iter_pos": float((v.iterations["mean_r"].dropna() > 0).mean()) if len(v.iterations) else np.nan,
        "pos_chunks": f"{int(v.stats['pos_chunks'])}/{int(v.stats['n_chunks'])}" if np.isfinite(v.stats["pos_chunks"]) else "n/a",
    } for k, v in out.items()]).set_index("strategy")
    res = BacktestResult(bt, seg_df, out, summary, errors, time.time() - t0)
    if verbose:
        print(f"[bt] {len(jobs)} segments traded in {time.time() - t1:,.0f}s (total {res.seconds:,.0f}s) | errors: {len(errors)}", flush=True)
        for e in errors[:10]:
            print(f"[bt]   ! {e}", flush=True)
    if bt.report_dir:
        res.report_files = write_reports(res, bt.report_dir)
    return res


def write_reports(res: BacktestResult, report_dir: str) -> list[str]:
    """<dir>/summary.csv, <dir>/<strategy>_chunks.csv (one row per OOS chunk), _iterations.csv, _trades.csv, report.txt."""
    os.makedirs(report_dir, exist_ok=True)
    paths = []
    p = os.path.join(report_dir, "summary.csv")
    res.summary.to_csv(p)
    paths.append(p)
    p = os.path.join(report_dir, "segments.csv")
    res.segments.to_csv(p, index=False)
    paths.append(p)
    text = [f"seed {res.bt.seed} | {res.bt.n_iterations} iterations | timeframes {res.bt.timeframes}", ""]
    for k, v in res.strategies.items():
        slug = "".join(c if c.isalnum() else "_" for c in k).strip("_").lower() or "strategy"
        for kind, df in (("chunks", v.chunks), ("iterations", v.iterations), ("trades", v.trades), ("by_timeframe", v.by_timeframe)):
            p = os.path.join(report_dir, f"{slug}_{kind}.csv")
            df.to_csv(p, index=False)
            paths.append(p)
        text += [v.report, ""]
    p = os.path.join(report_dir, "report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(text))
    paths.append(p)
    return paths


# =============================================================================== stage 0: half-life models (trained on GATE_RANGES only)
def build_halflife_models(store: DataStore, symbols: Sequence[str], target: str, ranges: Sequence[tuple[str, str]],
                          discover_cfg: DiscoverConfig, dcfg=None, *, dataset_path: str = "halflife_dataset.parquet",
                          models_path: str = "halflife_models.pkl", horizon: int = 2_500, verbose: bool = True) -> dict:
    """Rolling-origin dataset on the calendar blocks `ranges` -> two HalfLifePredictors (orderflow_features False / True).
    LEAKAGE RULE: these models must never see a bar the backtest trades: BacktestConfig.history_start has to be after every range
    (check_no_leakage enforces it)."""
    dcfg = dcfg or DatasetConfig()
    if os.path.exists(dataset_path):
        df = pd.read_parquet(dataset_path)
    else:
        frames = []
        for k, (s, e) in enumerate(ranges):
            store.ensure(s, e, verbose=verbose)
            panel, _ = store.read(s, e)
            frames.append(build_dataset(panel, target, discover_cfg, dcfg, segment=k))
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(dataset_path)
    if verbose:
        print(f"[halflife] dataset: {len(df):,} rows | origins {df['origin'].nunique():,} | segments {df['segment'].nunique()}", flush=True)
        with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
            print(compare_predictors(df, dcfg, store.bar_td.total_seconds() / 60.0, horizon).to_string(index=False), flush=True)
    models = {"spread stats": HalfLifePredictor(horizon, orderflow_features=False).fit(df, dcfg),
              "spread + order flow": HalfLifePredictor(horizon, orderflow_features=True).fit(df, dcfg)}
    with open(models_path, "wb") as fh:
        pickle.dump(models, fh)
    return {k: HalfLifeRetrainRule(m) for k, m in models.items()}


def check_no_leakage(bt: BacktestConfig, ranges: Sequence[tuple[str, str]]) -> None:
    last = max(pd.Timestamp(e) for _, e in ranges)
    if pd.Timestamp(bt.history_start) <= last:
        raise ValueError(f"LEAKAGE: history_start {bt.history_start} is not after the last model-training range end {last:%Y-%m-%d}")


__all__ = ["BacktestConfig", "StrategySpec", "DataStore", "make_loader", "draw_segment_lengths", "draw_iteration", "run_segment",
           "run_backtest", "write_reports", "build_halflife_models", "check_no_leakage", "discover_cfg_for_bar", "BacktestResult"]


# =============================================================================== settings + entry point
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    SYMBOLS = ("ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT")
    TARGET = "ETH/USDT"

    # calendar blocks used ONLY to train the half-life models (several disjoint blocks = several regimes)
    GATE_RANGES = [("2021-03-01", "2021-04-15"), ("2022-05-01", "2022-06-15"), ("2023-01-10", "2023-02-25"), ("2023-08-01", "2023-09-15")]
    TRAIN_HALFLIFE = True            # build/load the dataset + fit the predictors (cached on disk); False -> no predictor variants
    RUN_PHASE_2 = False              # True: mixed timeframes (needs the same 1-minute cache; features are aggregated, not recomputed)

    discover_cfg = DiscoverConfig(use_features=True)
    bt = BacktestConfig(
        symbols=SYMBOLS, target=TARGET, history_start="2024-01-01", history_end="2026-07-31",
        total_bars=100_000, chunk_bars=5_000, min_segment_bars=10_000, n_iterations=20, seed=7,
        n_workers=2, backend="loky", cache_dir="bt_cache", report_dir="bt_reports")
    check_no_leakage(bt, GATE_RANGES)

    loader = make_loader(SYMBOLS, discover_cfg, bar=bt.base_bar)
    store = DataStore(bt.cache_dir, SYMBOLS, bt.base_bar, loader, bt.lead_days, bt.max_gap_bars)

    variants: dict[str, Any] = {}
    if TRAIN_HALFLIFE:
        ds = DatasetConfig(train_bars=5_000, fwd_bars=(500, 1_000, 2_500, 5_000), step=1_000, top_k=20, max_origins=250, n_jobs=1, seed=7)
        variants = build_halflife_models(store, SYMBOLS, TARGET, GATE_RANGES, discover_cfg, ds)

    common = dict(entry_z=2.0, capital=10_000, risk_fraction=0.02, rr=1.0, barrier_size=1.0, fee_bps=0.0, slippage_bps=0.0,
                  feature_score_mode="filter", strategy_side=1, zscore_mode="rolling", barrier_mode="hilo")
    strategies = {
        "formula stop | train half-life": StrategySpec(None, dict(common)),
        "smart classifier | train half-life": StrategySpec(None, dict(common, smart_trading=True, smart_mode="classifier")),
        "smart RL | train half-life": StrategySpec(None, dict(common, smart_trading=True, smart_mode="rl")),
    }
    for vname in variants:
        strategies[f"formula stop | {vname}"] = StrategySpec(vname, dict(common))
        strategies[f"smart classifier | {vname}"] = StrategySpec(vname, dict(common, smart_trading=True, smart_mode="classifier"))
        strategies[f"smart RL | {vname}"] = StrategySpec(vname, dict(common, smart_trading=True, smart_mode="rl"))

    print("\n########## PHASE 1: random chunks x iterations (1-minute bars) ##########")
    r1 = run_backtest(bt, store, discover_cfg, strategies, variants)
    with pd.option_context("display.width", 220, "display.float_format", "{:,.4f}".format):
        print(r1.summary.to_string())
    for v in r1.strategies.values():
        print("\n" + v.report)

    if RUN_PHASE_2:
        print("\n########## PHASE 2: mixed timeframes ##########")
        bt2 = replace(bt, timeframes=(1, 5, 15), total_bars=50_000, n_iterations=20, seed=bt.seed + 1, report_dir="bt_reports_phase2")
        r2 = run_backtest(bt2, store, discover_cfg, strategies, variants)
        with pd.option_context("display.width", 220, "display.float_format", "{:,.4f}".format):
            print(r2.summary.to_string())
            for v in r2.strategies.values():
                print(f"\n[{v.name}] by timeframe")
                print(v.by_timeframe.to_string(index=False))