"""Parametric Monte Carlo walk-forward stress test for strategy.py.

Instead of one fixed purged walk-forward over a single date range, the test draws RANDOM segments of history:

    total_bars     = 100,000      # data budget (train windows included)
    chunk_bars     = 5,000        # one flip-flop window: train 5k -> trade 5k.  Every segment is a multiple of this
    min_segment    = 10,000       # smallest segment = one train chunk + one traded chunk

    e.g. one draw:   20k [2026-07 .. 2026-08] | 50k [2024-01 .. 2024-06] | 10k [2023-10 .. 2023-11] | 20k [2025-01 .. 2025-03]

Segment lengths are a random composition of total_bars into multiples of chunk_bars (each >= min_segment); each segment
gets a random start day anywhere in [history_start, history_end]; segments may overlap.  Inside a segment the normal
flip-flop runs (train chunk -> trade next chunk -> train on the chunk just traded -> ...), so a segment of S bars yields
S/chunk - 1 out-of-sample chunks.  The unit of measurement is the OOS CHUNK: chunk statistics, a chunk-level bootstrap and
the pooled trade list are what the report is built from.

Speed
-----
* only the drawn segments are loaded (never years of 1-minute bars with features);
* beam-search screening is independent of TradeConfig, so build_experiment() does it ONCE (in parallel) and
  evaluate()/compare() re-use it - changing entry_z / rr / zscore_mode / barrier rules costs only the tree + trading;
* every fold of every segment is an independent job and runs on all cores.

Same MCConfig.seed -> same segments, so two TradeConfigs can be compared on identical data (compare()).  Once you have
tuned on one seed, confirm on a different seed before believing the number.  MCConfig(seed=None) draws a fresh set of
random segments AND random dates on every run (the seed that was used is printed and written into the report, so any
draw can be reproduced).

Per-chunk report
----------------
evaluate(..., report_dir="mc_reports") (or compare(..., report_dir=...)) writes, per TradeConfig, a CSV with ONE ROW PER
5,000-BAR CHUNK (= one parallel fold: train window -> OOS window) and a Markdown report built from it.  Each row has the
training-window and the OOS-window statistics side by side: Engle-Granger p, ADF stat / p, half-life, AR(1) phi, spread
std, z drift, then trades / TP rate / mean R.  Train and OOS ADF / half-life are computed on the SAME traded (asset-only)
spread with the SAME function, so they are directly comparable.  `strategy_side` (TradeConfig) flips the strategy.
"""
from __future__ import annotations

import time
import warnings
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

try:  # package layout
    from .strategy import (
        DEFAULT_BAR, AlgorithmResult, FoldPlan, FoldResult, HiLo, ScreenSelection, TradeConfig,
        assemble_algorithm, load_hilo, parallel_map, plan_folds, screen_fold, trade_fold,
        _asset_spread, _close_frame, _phi,
    )
except ImportError:  # flat folder
    from strategy import (
        DEFAULT_BAR, AlgorithmResult, FoldPlan, FoldResult, HiLo, ScreenSelection, TradeConfig,
        assemble_algorithm, load_hilo, parallel_map, plan_folds, screen_fold, trade_fold,
        _asset_spread, _close_frame, _phi,
    )
try:
    from .spread_discoverer import DiscoverConfig, adf_p, adf_stat, half_life
except ImportError:
    from spread_discoverer import DiscoverConfig, adf_p, adf_stat, half_life
try:
    from .spread_discoverer import get_assets
except ImportError:
    try:
        from spread_discoverer import get_assets
    except ImportError:
        get_assets = None


# --------------------------------------------------------------------------- configuration
@dataclass
class MCConfig:
    total_bars: int = 100_000            # data budget across all segments (train windows included)
    chunk_bars: int = 5_000              # train window = traded window = one measured chunk
    min_segment_bars: int = 10_000       # >= 2 * chunk_bars (one train + one traded chunk)
    max_segment_bars: int | None = None  # optional cap on a single segment
    history_start: str = "2023-10-01"    # segments are drawn from [history_start, history_end]
    history_end: str = "2026-07-31"
    bar: str = DEFAULT_BAR
    purge_bars: int = 1
    seed: int | None = 7                 # None -> fresh random segments + dates every run (the drawn seed is stored back here)
    n_boot: int = 2_000                  # chunk-bootstrap paths
    max_load_retries: int = 5            # redraw a segment's start if its data cannot be loaded

    def __post_init__(self):
        if self.seed is None:                         # new draw each run; keep the value so the run can be reproduced
            self.seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        c = self.chunk_bars
        if c < 200:
            raise ValueError("chunk_bars must be >= 200")
        for name in ("total_bars", "min_segment_bars"):
            if getattr(self, name) % c:
                raise ValueError(f"{name} must be a multiple of chunk_bars ({c:,})")
        if self.max_segment_bars is not None and self.max_segment_bars % c:
            raise ValueError("max_segment_bars must be a multiple of chunk_bars")
        if self.min_segment_bars < 2 * c:
            raise ValueError("min_segment_bars must be >= 2 * chunk_bars (one train chunk + one traded chunk)")
        if self.total_bars < self.min_segment_bars:
            raise ValueError("total_bars must be >= min_segment_bars")
        if self.max_segment_bars is not None and self.max_segment_bars < self.min_segment_bars:
            raise ValueError("max_segment_bars must be >= min_segment_bars")
        if pd.Timestamp(self.history_end) <= pd.Timestamp(self.history_start):
            raise ValueError("history_end must be after history_start")

    @property
    def bar_minutes(self) -> float:
        return pd.Timedelta(self.bar).total_seconds() / 60.0


@dataclass
class Segment:
    id: int
    bars: int                 # nominal length (multiple of chunk_bars)
    n_folds: int              # OOS chunks it yields = bars / chunk_bars - 1
    need_bars: int            # bars to load (adds one purge gap per fold)
    start: pd.Timestamp       # requested start day (updated to the real first bar once loaded)
    end: pd.Timestamp | None = None   # real last bar once loaded


# --------------------------------------------------------------------------- random segment sampler
def draw_segment_lengths(total_units: int, min_units: int, max_units: int | None, rng: np.random.Generator) -> list[int]:
    """Uniformly random composition of `total_units` into parts in [min_units, max_units] (units = chunks)."""
    cap = max_units or total_units
    ways = [0] * (total_units + 1)          # number of valid compositions of r units
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


def _random_start(mc: MCConfig, need_bars: int, rng: np.random.Generator) -> pd.Timestamp:
    h0 = pd.Timestamp(mc.history_start).normalize()
    h1 = pd.Timestamp(mc.history_end)
    span = pd.Timedelta(mc.bar) * need_bars
    free_days = int((h1 - span - h0) / pd.Timedelta(days=1))
    if free_days < 0:
        raise ValueError(f"a {need_bars:,}-bar segment ({span}) does not fit inside {mc.history_start} .. {mc.history_end}")
    return h0 + pd.Timedelta(days=int(rng.integers(0, free_days + 1)))


def sample_segments(mc: MCConfig, rng: np.random.Generator) -> list[Segment]:
    units = draw_segment_lengths(
        mc.total_bars // mc.chunk_bars, mc.min_segment_bars // mc.chunk_bars,
        None if mc.max_segment_bars is None else mc.max_segment_bars // mc.chunk_bars, rng)
    segs = []
    for k, u in enumerate(units):
        n_folds = u - 1
        need = u * mc.chunk_bars + n_folds * mc.purge_bars
        segs.append(Segment(id=k + 1, bars=u * mc.chunk_bars, n_folds=n_folds, need_bars=need,
                            start=_random_start(mc, need, rng)))
    return segs


# --------------------------------------------------------------------------- data loading
Loader = Callable[[pd.Timestamp, pd.Timestamp, int], "tuple[Any, HiLo | None]"]


def make_loader(
    symbols: Sequence[str],
    discover_cfg: "DiscoverConfig",
    *,
    bar: str = DEFAULT_BAR,
    market: str = "um",
    cache_dir: str = "~/.cache/binance_vision",
    workers: int = 4,
) -> Loader:
    """Default loader: spread_discoverer.get_assets for the panel (+features) and binance_vision for bar highs/lows.
    Any callable (start, end, n_bars) -> (AssetPanel, HiLo) works, e.g. one that reads your own parquet files."""
    if get_assets is None:
        raise ImportError("could not import spread_discoverer.get_assets")

    def loader(start: pd.Timestamp, end: pd.Timestamp, n_bars: int):
        s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        panel = get_assets(list(symbols), s, e, bar=bar, market=market, cfg=discover_cfg)
        hilo = load_hilo(symbols, s, e, bar=bar, market=market, cache_dir=cache_dir, workers=workers)
        return panel, hilo

    return loader


# --------------------------------------------------------------------------- experiment (screening done once)
@dataclass
class SegmentInfo:
    segment: Segment
    index: pd.Index
    symbols: list[str]
    bar_minutes: float


@dataclass
class FoldTask:
    segment: int
    plan: FoldPlan
    train_panel: Any
    forward_panel: Any
    hilo: HiLo | None
    selection: ScreenSelection | None = None
    error: str | None = None


@dataclass
class Experiment:
    mc: MCConfig
    y: str
    discover_cfg: Any
    segments: dict[int, SegmentInfo]
    tasks: list[FoldTask]
    skipped: list[str] = field(default_factory=list)


def _screen_job(**kw):
    try:
        return screen_fold(**kw), None
    except Exception as exc:                    # one degenerate window must not kill the whole Monte Carlo
        return None, f"{type(exc).__name__}: {exc}"


def _trade_job(**kw):
    try:
        return trade_fold(**kw), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _unique_coverage(segs: Sequence[Segment]) -> float:
    iv = sorted((s.start, s.end) for s in segs if s.end is not None)
    if not iv:
        return float("nan")
    total = sum((e - s).total_seconds() for s, e in iv)
    merged, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
    merged += (cur_e - cur_s).total_seconds()
    return merged / total if total > 0 else float("nan")


def build_experiment(
    y: str,
    loader: Loader,
    mc: MCConfig | None = None,
    discover_cfg: "DiscoverConfig | None" = None,
    *,
    retrain_rule: Callable[..., ScreenSelection] | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Experiment:
    """Draw the random segments, load them, and run the beam-search screening of every fold (in parallel, once)."""
    mc = mc or MCConfig()
    dcfg = discover_cfg or DiscoverConfig()
    rng = np.random.default_rng(mc.seed)
    t0 = time.time()
    segments = sample_segments(mc, rng)
    if verbose:
        print(f"[mc] seed {mc.seed}: drew {len(segments)} segments (random lengths AND random start dates):", flush=True)
        for s in segments:
            print(f"[mc]   #{s.id}: {s.bars // 1000}k bars from {s.start:%Y-%m-%d} "
                  f"(~{s.start + pd.Timedelta(mc.bar) * s.need_bars:%Y-%m-%d})", flush=True)

    infos: dict[int, SegmentInfo] = {}
    tasks: list[FoldTask] = []
    for seg in segments:
        for attempt in range(mc.max_load_retries + 1):
            try:
                end = seg.start + pd.Timedelta(mc.bar) * seg.need_bars + pd.Timedelta(days=2)
                panel, hilo = loader(seg.start, end, seg.need_bars)
                if len(panel.logp) < seg.need_bars:
                    raise ValueError(f"loaded {len(panel.logp):,} bars, need {seg.need_bars:,}")
                panel = panel.iloc(0, seg.need_bars)
                break
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                if attempt == mc.max_load_retries:
                    raise
                warnings.warn(f"segment {seg.id}: load failed ({exc}); redrawing its start")
                seg.start = _random_start(mc, seg.need_bars, rng)
        idx = panel.logp.index
        seg.start, seg.end = idx[0], idx[-1]
        plans = plan_folds(len(idx), train_bars=mc.chunk_bars, forward_bars=mc.chunk_bars, purge_bars=mc.purge_bars)
        if len(plans) != seg.n_folds:
            raise RuntimeError(f"segment {seg.id}: planned {len(plans)} folds, expected {seg.n_folds}")
        infos[seg.id] = SegmentInfo(seg, idx, list(panel.logp.columns), panel.bar_minutes)
        for plan in plans:
            tp = panel.iloc(plan.train_start, plan.train_end)
            fp = panel.iloc(plan.forward_start, plan.forward_end)
            hl = None if hilo is None else hilo.between(tp.logp.index[0], fp.logp.index[-1])
            tasks.append(FoldTask(seg.id, plan, tp, fp, hl))
        if verbose:
            print(f"[mc] segment {seg.id}: {seg.bars:,} bars {seg.start:%Y-%m-%d} -> {seg.end:%Y-%m-%d} "
                  f"({seg.n_folds} chunks)  loaded", flush=True)

    if verbose:
        print(f"[mc] screening {len(tasks)} folds (beam search, runs once per experiment)...", flush=True)
    dcfg_run = dcfg
    if dcfg.use_features and tasks and not tasks[0].train_panel.feats:
        warnings.warn("panel carries no features: running the beam search price-only")
        dcfg_run = replace(dcfg, use_features=False)
    jobs = [dict(train_panel=t.train_panel, y=y, discover_cfg=dcfg_run, retrain_rule=retrain_rule) for t in tasks]
    results = parallel_map(_screen_job, jobs, n_jobs)
    skipped = []
    for t, (sel, err) in zip(tasks, results):
        t.selection, t.error = sel, err
        if err:
            skipped.append(f"segment {t.segment} chunk {t.plan.cycle}: screen failed ({err})")
    if verbose:
        for s in skipped:
            print(f"[mc] SKIPPED {s}", flush=True)
        print(f"[mc] screening done in {time.time() - t0:,.1f}s total", flush=True)
    return Experiment(mc, y, dcfg_run, infos, tasks, skipped)


# --------------------------------------------------------------------------- statistics
@dataclass
class MCResult:
    mc: MCConfig
    trade_cfg: TradeConfig
    segments: pd.DataFrame
    chunk_table: pd.DataFrame
    trades: pd.DataFrame
    algorithms: dict[int, AlgorithmResult]
    stats: dict[str, float]
    expectancy_paths: pd.Series
    cumulative_r_paths: pd.DataFrame
    report: str
    figure: Any = None
    skipped: list[str] = field(default_factory=list)
    chunk_report: pd.DataFrame = field(default_factory=pd.DataFrame)   # one row per 5k chunk: train vs OOS statistics
    report_files: list[str] = field(default_factory=list)              # CSV / MD paths written by evaluate(report_dir=...)


def _tail_means(x: np.ndarray) -> tuple[float, float, float, float]:
    p05, p95 = np.quantile(x, [0.05, 0.95])
    return float(p05), float(p95), float(x[x <= p05].mean()), float(x[x >= p95].mean())


def _stats(r: np.ndarray, chunk_sum: np.ndarray, chunk_n: np.ndarray, n_boot: int, seed: int):
    """Trade-level stats + chunk-level bootstrap (resample whole OOS chunks with replacement)."""
    n, K = len(r), len(chunk_sum)
    nan = float("nan")
    st: dict[str, float] = dict(n_trades=n, n_chunks=K, mean_r=nan, sd_r=nan, se_r=nan, t_stat=nan, win_rate=nan,
                                boot_mean=nan, p05=nan, p95=nan, cvar05=nan, cvar95=nan, ci_lo=nan, ci_hi=nan,
                                p_positive=nan, pos_chunks=nan, n_required=nan, total_r=float(r.sum()) if n else 0.0)
    cum = np.zeros((0, 0))
    exp = np.array([])
    if n == 0 or K == 0:
        return st, exp, cum
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
    exp = np.full(n_boot, np.nan)
    exp[ok] = sums[ok].sum(axis=1) / tot[ok]
    e = exp[ok]
    if len(e):
        st["boot_mean"] = float(e.mean())
        st["p05"], st["p95"], st["cvar05"], st["cvar95"] = _tail_means(e)
        st["ci_lo"], st["ci_hi"] = (float(v) for v in np.quantile(e, [0.025, 0.975]))
        st["p_positive"] = float((e > 0).mean())
    cum = np.cumsum(sums, axis=1)
    return st, exp, cum


def _fmt_report(mc: MCConfig, tcfg: TradeConfig, seg_df: pd.DataFrame, chunk_df: pd.DataFrame, trades: pd.DataFrame,
                st: dict[str, float], coverage: float, skipped: list[str], name: str) -> str:
    L = []
    L.append(f"MONTE CARLO PURGED WALK-FORWARD{f' [{name}]' if name else ''}")
    L.append(f"bar={mc.bar} | data budget {mc.total_bars:,} bars | chunk {mc.chunk_bars:,} | min segment {mc.min_segment_bars:,} | seed {mc.seed}")
    L.append(f"z-score: {tcfg.zscore_mode}" + (f" ({tcfg.zscore_window:,} bars)" if tcfg.zscore_mode == "rolling" else " (train mean/std)")
             + f" | barriers: {tcfg.barrier_mode}" + (f" ({tcfg.hilo_extremes})" if tcfg.barrier_mode == 'hilo' else "")
             + f" | entry_z {tcfg.entry_z:g} | barrier_size {tcfg.barrier_size:g} | rr {tcfg.rr:g}")
    L.append(f"strategy direction: {'FOLLOWED (strategy_side=1)' if tcfg.strategy_side == 1 else 'INVERTED (strategy_side=2: strategy long -> we short)'}")
    L.append(f"Segments: {len(seg_df)} | OOS chunks measured: {int(st['n_chunks'])} | unique time coverage: {coverage:.0%}")
    for _, s in seg_df.iterrows():
        L.append(f"  #{int(s['segment'])}  {int(s['bars']):>7,} bars  [{s['start']:%Y-%m-%d} .. {s['end']:%Y-%m-%d}]  {int(s['chunks'])} chunk(s)")
    for s in skipped:
        L.append(f"  skipped: {s}")
    if not st["n_trades"]:
        L.append("No closed OOS trades.")
        return "\n".join(L)
    rs = trades["reason"]
    mix = " | ".join(f"{k} {float((rs == k).mean()):.1%}" for k in ("tp", "sl", "time", "window_end") if (rs == k).any())
    L.append(f"Trades: {int(st['n_trades']):,} | exits: {mix} | breakeven TP rate {tcfg.breakeven_win_rate:.1%}")
    L.append(f"Mean R/trade: {st['mean_r']:+.4f} | sd {st['sd_r']:.3f} | trade-level t-stat {st['t_stat']:+.2f} | accumulated {st['total_r']:+.2f}R")
    L.append(f"Chunk bootstrap ({mc.n_boot:,} paths): E[R] {st['boot_mean']:+.4f}  P05/P95 [{st['p05']:+.4f}, {st['p95']:+.4f}]"
             f"  CVaR05/95 [{st['cvar05']:+.4f}, {st['cvar95']:+.4f}]")
    L.append(f"95% CI of mean R: [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}] | P(E[R] > 0): {st['p_positive']:.1%} | "
             f"chunks with positive expectancy: {int(st['pos_chunks'])}/{int((chunk_df['trades'] > 0).sum())}")
    if np.isfinite(st["n_required"]):
        verdict = "enough" if st["n_trades"] >= st["n_required"] else "NOT enough"
        L.append(f"Trades needed for this mean to be significant at 95% (trade-level): ~{st['n_required']:,.0f} | have {int(st['n_trades']):,} -> {verdict}")
    L.append("Chunks are the resampling unit (trades inside a chunk are correlated); overlapping segments share bars, so coverage < 100% means fewer independent samples.")
    return "\n".join(L)


def _plot(res_paths: np.ndarray, exp: np.ndarray, st: dict[str, float], title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed; Monte Carlo paths are returned without a figure")
        return None
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    if res_paths.size:
        x = np.arange(1, res_paths.shape[1] + 1)
        ax.plot(x, res_paths[:200].T, alpha=0.12, linewidth=0.8)
        ax.plot(x, res_paths.mean(axis=0), linewidth=2.2, label="mean bootstrap path")
        ax.legend(loc="best")
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_title(f"{title}: bootstrap cumulative R by OOS chunk")
    ax.set_xlabel("OOS chunk (resampled)")
    ax.set_ylabel("Cumulative R")
    ax.grid(True, alpha=0.25)
    e = exp[np.isfinite(exp)]
    if len(e):
        bx.hist(e, bins=40)
        for v in (st["p05"], st["p95"]):
            bx.axvline(v, linestyle=":", linewidth=1.2)
        bx.axvline(0, linewidth=1.0)
        bx.axvline(st["boot_mean"], linewidth=2.0)
    bx.set_title("Per-path expectancy (R/trade)")
    bx.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- per-chunk (per-fold) report
def _spread_stats(spread: np.ndarray) -> dict[str, float]:
    """ADF stat / p, half-life, AR(1) phi and std of a spread, computed the same way for train and OOS windows."""
    nan = float("nan")
    out = dict(adf_stat=nan, adf_p=nan, half_life=nan, phi=nan, std=nan)
    x = np.asarray(spread, float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return out
    out["std"] = float(x.std())
    try:
        out["phi"] = float(_phi(x))
        out["half_life"] = float(half_life(x, floor=1.0))
        st, _ = adf_stat(x, maxlag=12, const=True)
        out["adf_stat"], out["adf_p"] = float(st), float(adf_p(st, 1))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    return out


def _chunk_report_row(segment: int, fold: FoldResult, task: FoldTask, trades: pd.DataFrame, net_pnl: float) -> dict[str, Any]:
    """Everything about one 5k chunk: the TRAIN window the pair was selected on and the OOS window it was traded on."""
    sel, fwd = fold.selection, fold.forward
    tr_close = _close_frame(task.train_panel, sel.symbols)
    tr = _spread_stats(_asset_spread(tr_close, sel).to_numpy(float))        # traded (asset-only) spread, train window
    oo = _spread_stats(fwd.spread.to_numpy(float))                          # same spread, OOS window
    z = fwd.zscore.to_numpy(float)
    z = z[np.isfinite(z)]
    r = trades["r_multiple"].to_numpy(float) if len(trades) else np.array([])
    rs = trades["reason"] if len(trades) else pd.Series(dtype=object)
    m = sel.tp_model
    ratio = oo["half_life"] / tr["half_life"] if np.isfinite(oo["half_life"]) and np.isfinite(tr["half_life"]) and tr["half_life"] > 0 else np.nan
    return {
        "segment": segment, "chunk": fold.cycle, "instance": fold.instance,
        "train_start": fold.train_start, "train_end": fold.train_end,
        "oos_start": fold.forward_start, "oos_end": fold.forward_end,
        "train_bars": len(task.train_panel.logp), "oos_bars": len(fwd.close),
        "hedge": " + ".join(sel.hedge_symbols),
        "betas": ", ".join(f"{h}={sel.betas[h]:.3f}" for h in sel.hedge_symbols),
        "feature_terms": len(sel.feature_terms),
        # --- training window
        "p_eg_train": sel.p_eg_train, "adf_stat_train": tr["adf_stat"], "adf_p_train": tr["adf_p"],
        "half_life_train": tr["half_life"], "half_life_screen": sel.half_life_train, "phi_train": tr["phi"],
        "spread_std_train": tr["std"],
        # --- OOS window
        "adf_stat_oos": fwd.adf_stat_oos, "adf_p_oos": fwd.p_oos, "half_life_oos": fwd.half_life_oos,
        "phi_oos": oo["phi"], "spread_std_oos": oo["std"], "hl_oos_over_train": ratio,
        "z_mean_oos": float(z.mean()) if len(z) else np.nan, "z_std_oos": float(z.std()) if len(z) else np.nan,
        "z_absmax_oos": float(np.abs(z).max()) if len(z) else np.nan,
        # --- trading
        "trades": len(r), "tp_rate": float((rs == "tp").mean()) if len(r) else np.nan,
        "sl_rate": float((rs == "sl").mean()) if len(r) else np.nan,
        "time_rate": float((rs == "time").mean()) if len(r) else np.nan,
        "sum_r": float(r.sum()), "mean_r": float(r.mean()) if len(r) else np.nan,
        "net_pnl": net_pnl,
        "tree_events": m.n_events if m else np.nan, "tree_val_auc": m.val_auc if m else np.nan,
    }


def _md_cell(v: Any) -> str:
    if isinstance(v, (pd.Timestamp,)):
        return f"{v:%Y-%m-%d %H:%M}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:,.1f}"
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def format_chunk_report_md(res: "MCResult", name: str = "") -> str:
    """Markdown report: segment dates, a train-vs-OOS table per segment, pooled stationarity summary, the MC summary."""
    mc, tcfg, cr = res.mc, res.trade_cfg, res.chunk_report
    L = [f"# Monte Carlo walk-forward: per-chunk report{f' - {name}' if name else ''}", ""]
    L.append(f"- bar `{mc.bar}` | chunk (train = OOS window) **{mc.chunk_bars:,}** bars | purge {mc.purge_bars} | seed `{mc.seed}`")
    L.append(f"- strategy direction: **{'FOLLOWED (strategy_side=1)' if tcfg.strategy_side == 1 else 'INVERTED (strategy_side=2)'}**"
             f" | entry_z {tcfg.entry_z:g} | barrier_size {tcfg.barrier_size:g} | rr {tcfg.rr:g} | z-score {tcfg.zscore_mode} | barriers {tcfg.barrier_mode}")
    L.append(f"- segments {len(res.segments)} | chunks {len(cr)}")
    L.append("")
    L.append("Train and OOS ADF / half-life are measured on the same traded (asset-only) spread with the same function. "
             "`p_eg_train` / `half_life_screen` are the beam-search values of the selected (full, feature-adjusted) spread.")
    L.append("")
    if cr.empty:
        L.append("No chunks.")
        return "\n".join(L)

    L += ["## Segments (random dates)", ""]
    seg = res.segments.copy()
    seg["start"] = seg["start"].map(lambda v: f"{v:%Y-%m-%d %H:%M}")
    seg["end"] = seg["end"].map(lambda v: f"{v:%Y-%m-%d %H:%M}")
    L += [_md_table(seg), ""]

    L += ["## Train vs OOS stationarity (all chunks)", ""]
    def share(col):
        v = cr[col].to_numpy(float)
        v = v[np.isfinite(v)]
        return f"{(v < 0.05).mean():.0%}" if len(v) else "n/a"
    summary = pd.DataFrame([
        {"window": "train", "median_adf_p": cr["adf_p_train"].median(), "share_adf_p<0.05": share("adf_p_train"),
         "median_half_life": cr["half_life_train"].median(), "median_phi": cr["phi_train"].median(),
         "median_spread_std": cr["spread_std_train"].median()},
        {"window": "OOS", "median_adf_p": cr["adf_p_oos"].median(), "share_adf_p<0.05": share("adf_p_oos"),
         "median_half_life": cr["half_life_oos"].median(), "median_phi": cr["phi_oos"].median(),
         "median_spread_std": cr["spread_std_oos"].median()},
    ])
    L += [_md_table(summary), ""]

    cols_main = ["chunk", "train_start", "oos_start", "hedge", "p_eg_train", "adf_p_train", "adf_p_oos",
                 "half_life_train", "half_life_oos", "phi_train", "phi_oos", "z_mean_oos", "trades", "tp_rate", "mean_r", "sum_r"]
    L += [f"## Per segment: one row per {mc.chunk_bars:,}-bar chunk", ""]
    for sid, g in cr.groupby("segment", sort=True):
        row = res.segments[res.segments["segment"] == sid].iloc[0]
        L += [f"### Segment {sid}: {int(row['bars']):,} bars, {row['start']:%Y-%m-%d} .. {row['end']:%Y-%m-%d}", "",
              _md_table(g[cols_main].sort_values("chunk")), ""]

    L += ["## Full-width chunk statistics", "", "See the companion CSV for every column (train/OOS window dates, ADF stats, "
          "betas, z drift, SL/time rates, net PnL, tree AUC).", ""]
    L += ["## Pooled Monte Carlo summary", "", "```", res.report, "```", ""]
    return "\n".join(L)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "mc"


def write_chunk_reports(res: "MCResult", report_dir: str, name: str = "", fmt: str = "both") -> list[str]:
    """Write <dir>/<name>_chunks.csv and/or <dir>/<name>_report.md.  fmt: 'csv' | 'md' | 'both'."""
    if fmt not in ("csv", "md", "both"):
        raise ValueError("report_format must be 'csv', 'md' or 'both'")
    os.makedirs(report_dir, exist_ok=True)
    stem = os.path.join(report_dir, _slug(name or "mc"))
    paths = []
    if fmt in ("csv", "both"):
        paths.append(stem + "_chunks.csv")
        res.chunk_report.to_csv(paths[-1], index=False)
    if fmt in ("md", "both"):
        paths.append(stem + "_report.md")
        with open(paths[-1], "w", encoding="utf-8") as fh:
            fh.write(format_chunk_report_md(res, name))
    return paths


# --------------------------------------------------------------------------- evaluate / compare
def evaluate(
    exp: Experiment,
    trade_cfg: TradeConfig | None = None,
    *,
    trade_fn: Callable[..., pd.DataFrame] | None = None,
    n_jobs: int = -1,
    name: str = "",
    plot: bool = True,
    print_report: bool = True,
    report_dir: str | None = None,
    report_format: str = "both",
    print_chunks: bool | None = None,
) -> MCResult:
    """Trade every OOS chunk of the experiment with `trade_cfg` (all folds in parallel) and build the statistics.

    report_dir: when set, writes the per-chunk train-vs-OOS report (CSV and/or Markdown, see report_format) there.
    print_chunks: also print the per-chunk table (defaults to print_report)."""
    mc = exp.mc
    tcfg = trade_cfg or TradeConfig.for_bar(mc.bar_minutes)
    if tcfg.barrier_mode == "hilo" and any(t.hilo is None for t in exp.tasks):
        raise ValueError("barrier_mode='hilo' but the loader returned no HiLo; return one or use barrier_mode='close'")
    par_cfg = tcfg if n_jobs == 1 else replace(tcfg, tree_n_jobs=1)
    live = [t for t in exp.tasks if t.selection is not None]
    jobs = [dict(selection=t.selection, train_panel=t.train_panel, forward_panel=t.forward_panel, plan=t.plan,
                 tcfg=par_cfg, hilo=t.hilo, trade_fn=trade_fn) for t in live]
    t0 = time.time()
    results = parallel_map(_trade_job, jobs, n_jobs)
    skipped = list(exp.skipped)
    by_seg: dict[int, list[FoldResult]] = {}
    for t, (fold, err) in zip(live, results):
        if err:
            skipped.append(f"segment {t.segment} chunk {t.plan.cycle}: trade failed ({err})")
            continue
        by_seg.setdefault(t.segment, []).append(fold)

    algs: dict[int, AlgorithmResult] = {}
    seg_rows, chunk_rows, trade_frames, report_rows = [], [], [], []
    task_by = {(t.segment, t.plan.cycle): t for t in live}
    for sid, folds in sorted(by_seg.items()):
        info = exp.segments[sid]
        folds.sort(key=lambda f: f.cycle)
        alg = assemble_algorithm(folds, index=info.index, symbols=info.symbols, bar_minutes=info.bar_minutes,
                                 tcfg=tcfg, train_bars=mc.chunk_bars, forward_bars=mc.chunk_bars,
                                 purge_bars=mc.purge_bars, print_report=False)
        algs[sid] = alg
        s = info.segment
        seg_rows.append(dict(segment=sid, bars=s.bars, start=s.start, end=s.end, chunks=len(folds)))
        tr = alg.trades
        for f in folds:
            sub = tr[tr["cycle"] == f.cycle] if len(tr) else tr
            r = sub["r_multiple"].to_numpy(float) if len(sub) else np.array([])
            chunk_rows.append(dict(segment=sid, chunk=f.cycle, start=f.forward_start, end=f.forward_end,
                                   hedge=" + ".join(f.selection.hedge_symbols), trades=len(r), sum_r=float(r.sum()),
                                   mean_r=float(r.mean()) if len(r) else np.nan,
                                   tp_rate=float((sub["reason"] == "tp").mean()) if len(sub) else np.nan,
                                   net_pnl=float(f.forward.net_pnl.sum())))
            report_rows.append(_chunk_report_row(sid, f, task_by[(sid, f.cycle)], sub, float(f.forward.net_pnl.sum())))
        if len(tr):
            tr = tr.copy()
            tr.insert(0, "segment", sid)
            trade_frames.append(tr)

    seg_df = pd.DataFrame(seg_rows)
    chunk_df = pd.DataFrame(chunk_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=["segment", "reason", "r_multiple"])
    if chunk_df.empty:
        raise RuntimeError("no chunk produced a result; see Experiment.skipped / MCResult.skipped")
    r = trades["r_multiple"].to_numpy(float) if len(trades) else np.array([])
    st, exp_paths, cum = _stats(r, chunk_df["sum_r"].to_numpy(float), chunk_df["trades"].to_numpy(float), mc.n_boot, mc.seed + 1)
    coverage = _unique_coverage([exp.segments[i].segment for i in algs])
    report = _fmt_report(mc, tcfg, seg_df, chunk_df, trades, st, coverage, skipped, name)
    fig = _plot(cum, exp_paths, st, name or "Monte Carlo") if (plot and cum.size) else None
    if print_report:
        print(report, flush=True)
        print(f"[mc] trading {len(jobs)} chunks took {time.time() - t0:,.1f}s", flush=True)
    path_cols = [f"path_{i + 1:04d}" for i in range(len(cum))]
    cum_df = pd.DataFrame(cum.T, columns=path_cols, index=pd.RangeIndex(1, cum.shape[1] + 1, name="chunk")) if cum.size else pd.DataFrame()
    chunk_report = pd.DataFrame(report_rows)
    res = MCResult(mc, tcfg, seg_df, chunk_df, trades, algs, st, pd.Series(exp_paths, name="mean_R_per_trade"),
                   cum_df, report, fig, skipped, chunk_report)
    if report_dir:
        res.report_files = write_chunk_reports(res, report_dir, name, report_format)
        for pth in res.report_files:
            print(f"[mc] wrote {pth}", flush=True)
    if print_chunks if print_chunks is not None else print_report:
        show = ["segment", "chunk", "oos_start", "hedge", "p_eg_train", "adf_p_train", "adf_p_oos", "half_life_train",
                "half_life_oos", "trades", "tp_rate", "mean_r"]
        with pd.option_context("display.width", 220, "display.max_rows", 500, "display.float_format", "{:,.4f}".format):
            print(chunk_report[show].to_string(index=False), flush=True)
    return res


def compare(
    exp: Experiment,
    configs: dict[str, TradeConfig],
    *,
    n_jobs: int = -1,
    plot: bool = False,
    print_reports: bool = False,
    report_dir: str | None = None,
    report_format: str = "both",
) -> tuple[pd.DataFrame, dict[str, MCResult]]:
    """Evaluate several TradeConfigs on the SAME random segments (screening is shared, so each extra config is cheap).
    With report_dir set, every config gets its own per-chunk CSV / Markdown report there."""
    results: dict[str, MCResult] = {}
    rows = []
    for name, cfg in configs.items():
        res = evaluate(exp, cfg, n_jobs=n_jobs, name=name, plot=plot, print_report=print_reports,
                       report_dir=report_dir, report_format=report_format, print_chunks=False)
        results[name] = res
        s = res.stats
        rs = res.trades["reason"] if len(res.trades) else pd.Series(dtype=object)
        rows.append({
            "config": name, "side": "follow" if cfg.strategy_side == 1 else "INVERT", "barrier": cfg.barrier_size, "rr": cfg.rr, "trades": int(s["n_trades"]), "mean_R": s["mean_r"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
            "P(E>0)": s["p_positive"], "total_R": s["total_r"], "pos_chunks": f"{int(s['pos_chunks'])}/{int(s['n_chunks'])}" if np.isfinite(s["pos_chunks"]) else "n/a",
            "tp": float((rs == "tp").mean()) if len(rs) else np.nan, "sl": float((rs == "sl").mean()) if len(rs) else np.nan,
            "time": float((rs == "time").mean()) if len(rs) else np.nan,
        })
    table = pd.DataFrame(rows).set_index("config")
    return table, results


def monte_carlo_test(
    y: str,
    loader: Loader,
    mc: MCConfig | None = None,
    discover_cfg: "DiscoverConfig | None" = None,
    trade_cfg: TradeConfig | None = None,
    **kw,
) -> MCResult:
    """One-call convenience: draw + load + screen + trade + report."""
    n_jobs = kw.pop("n_jobs", -1)
    exp = build_experiment(y, loader, mc, discover_cfg, n_jobs=n_jobs, retrain_rule=kw.pop("retrain_rule", None))
    return evaluate(exp, trade_cfg, n_jobs=n_jobs, **kw)


__all__ = ["MCConfig", "Segment", "Experiment", "MCResult", "make_loader", "sample_segments", "draw_segment_lengths",
           "build_experiment", "evaluate", "compare", "monte_carlo_test", "write_chunk_reports", "format_chunk_report_md"]


if __name__ == "__main__":
    print("=== MONTE CARLO STRESS TEST STARTED ===", flush=True)

    SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
    TARGET = "ETH/USDT"

    MC = MCConfig(
        total_bars=100_000,          # data budget; must be a multiple of chunk_bars
        chunk_bars=5_000,            # train 5k -> trade 5k
        min_segment_bars=10_000,
        history_start="2020-10-01",
        history_end="2026-07-31",
        bar="1min",                  # <- bar size (1min / 5min / ...). Everything bar-denominated in TradeConfig follows via for_bar()
        seed=7,                      # 7 = reproducible draw | None = brand-new random segments + dates every run
        n_boot=2_000,
    )
    N_JOBS = -1                      # all cores
    BARRIER_SIZE = 2.0               # stop distance multiplier (1 = standard); TP = RR x stop, so this widens/narrows both barriers
    RR = 0.5                         # take profit = RR x stop
    STRATEGY_SIDE = 2                # 1 = trade what the strategy says | 2 = INVERT it (strategy long -> we short, and vice versa)
    REPORT_DIR = "mc_reports"        # per-config, per-5k-chunk CSV + Markdown (train vs OOS: half-life, ADF p, ...). None = off
    REPORT_FORMAT = "both"           # "csv" | "md" | "both"

    discover_cfg = DiscoverConfig(use_features=True)
    loader = make_loader(SYMBOLS, discover_cfg, bar=MC.bar)

    exp = build_experiment(TARGET, loader, MC, discover_cfg, n_jobs=N_JOBS)

    base = dict(entry_z=2.0, capital=10_000, risk_fraction=0.02, rr=RR, barrier_size=BARRIER_SIZE, fee_bps=0.0, slippage_bps=0.0,
                feature_score_mode="filter", use_tree=True, tree_diagnostics=False,
                strategy_side=STRATEGY_SIDE)
    configs = {
        "static z / close barriers (old)": TradeConfig.for_bar(MC.bar_minutes, zscore_mode="static", barrier_mode="close", **base),
        "static z / hi-lo barriers": TradeConfig.for_bar(MC.bar_minutes, zscore_mode="static", barrier_mode="hilo", **base),
        "rolling z / hi-lo barriers": TradeConfig.for_bar(MC.bar_minutes, zscore_mode="rolling", barrier_mode="hilo", **base),
    }
    table, results = compare(exp, configs, n_jobs=N_JOBS, report_dir=REPORT_DIR, report_format=REPORT_FORMAT)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.4f}".format):
        print(table.to_string(), flush=True)

    first = next(iter(results.values()))
    for name, res in results.items():
        print("\n" + res.report, flush=True)
    print(first.chunk_table.to_string(), flush=True)
    for name, res in results.items():
        with pd.option_context("display.width", 220, "display.max_rows", 500, "display.float_format", "{:,.4f}".format):
            print(f"\n[{name}] per-chunk train vs OOS", flush=True)
            print(res.chunk_report[["segment", "chunk", "oos_start", "hedge", "p_eg_train", "adf_p_train", "adf_p_oos",
                                    "half_life_train", "half_life_oos", "trades", "tp_rate", "mean_r"]].to_string(index=False), flush=True)
        for pth in res.report_files:
            print(f"[mc] report file: {pth}", flush=True)

    try:
        import matplotlib.pyplot as plt
        for name, res in results.items():
            res.figure = _plot(res.cumulative_r_paths.to_numpy().T, res.expectancy_paths.to_numpy(), res.stats, name)
        plt.show()
    except ImportError:
        pass