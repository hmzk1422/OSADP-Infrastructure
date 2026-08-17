"""
Monte Carlo Stress Testing — Circular Block Bootstrap on Trade Return Sequences
=================================================================================
Built to consume the per-trade R-multiple output of ml_state_optimized.py's
walk-forward loop (`trade_log[t]` / `master_trade_df`, or the exported
`build/walk_forward_trade_log_v3.csv`) and answer one question: "how bad can this
get, given only the trades I already observed?"

DESIGN NOTES — READ BEFORE TRUSTING THE OUTPUT
------------------------------------------------
1. INPUT UNITS. `returns` is assumed to be the R-multiple sequence already
   produced by the walk-forward loop (`normalized_return * side`), i.e. wins
   are ~+RR_RATIO, losses are ~-1.0, timeouts are fractional, and fees are
   already netted in via `fee_in_r`. This module does NOT know your dollar
   risk sizing — that's `MonteCarloConfig.risk_per_trade` (fraction of
   current equity risked on 1.0R), which YOU must set to match your real
   position sizing. Getting this wrong doesn't change the R-multiple stats
   (drawdown-in-R, ruin-in-R) but WILL change the equity-curve/dollar plots.

2. BLOCK SIZE. A single block size is drawn per BATCH (not per path) from
   [block_size_min, block_size_max] — this keeps index generation fully
   vectorized (numpy, no per-path Python loop) while still varying block
   size across the full run, since n_simulations / batch_size batches each
   get an independent draw. With the default batch_size=2000 and
   n_simulations=20000 that's 10 independent block-size draws, which is
   enough to avoid the whole run being anchored to one arbitrary block size
   without paying for a fully per-path vectorization scheme.

3. CIRCULARITY. Blocks are sampled by drawing a random start index into the
   historical return array and taking `block_size` consecutive trades,
   WRAPPING AROUND (modulo n) if the block runs off the end. This is what
   makes it a *circular* block bootstrap (Politis & Romano, 1992) rather
   than a plain moving-block bootstrap — every trade in history has an equal
   chance of being a block's start, including the last (block_size - 1)
   trades, which a non-circular version would systematically under-sample.

4. STRESSORS ARE ADDITIVE PER PATH, NOT PER HISTORICAL TRADE. Each stressor
   below draws its own random parameters independently for every simulated
   path, then applies them to that path's already-bootstrapped return
   sequence. They stack: a path can simultaneously get a bad block-sampling
   draw, a harsh slippage draw, a harsh win-rate-shock draw, etc. That
   stacking is the point — it's what makes this an adversarial stress test
   instead of a plain bootstrap.

5. RUIN is defined on MAX DRAWDOWN IN R (or in equity %, same thing under
   fixed-fractional sizing) breaching `ruin_drawdown_pct`, checked at every
   trade in the path — not just at the end. If `profit_target_multiple` is
   set, a breach only counts as ruin if it happens BEFORE the path first
   reaches that equity multiple; otherwise every breach counts. Default is
   None (every breach counts), which is the more conservative reading of
   "probability of ruin" and the one worth defaulting to.
"""

from __future__ import annotations

import os

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# All generated report artifacts go under the project's build/ folder (never CWD).
_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))


def _build_output(name: str) -> str:
    os.makedirs(_BUILD_DIR, exist_ok=True)
    return os.path.join(_BUILD_DIR, name)


# ============================================================================
# CONFIG
# ============================================================================
@dataclass
class MonteCarloConfig:
    # --- resampling ---
    n_simulations: int = 20_000
    block_size_min: int = 10
    block_size_max: int = 100
    path_length: Optional[int] = None      # None -> same length as historical trade count
    batch_size: int = 2_000                # sims per batch (memory control)
    random_seed: Optional[int] = 42

    # --- capital / equity model ---
    starting_equity: float = 100_000.0
    risk_per_trade: float = 0.0005           # fraction of equity risked per 1.0R
    compounding: bool = True               # fixed-fractional compounding vs fixed-dollar additive

    # --- stressor toggles ---
    apply_stressors: bool = False

    # Slippage inflation: each trade's assumed baseline slippage cost
    # (in R) gets multiplied by a random factor in this range and the
    # EXTRA cost beyond baseline is subtracted from that trade's return.
    # For now, keep it at 0 for analysis of edge expectancy
    slippage_r_base: float = 0
    slippage_mult_range: tuple = (0, 0)

    # Win-rate sensitivity shock: per path, this fraction of trades gets
    # flipped from win -> loss (applied to currently-winning trades only).
    win_rate_shock_range: tuple = (0.02, 0.05)

    # Returns shuffling with volatility scaling: trades whose ORIGINAL
    # historical return sits above this percentile are "outlier winners";
    # in each simulated path they get scaled down by a random factor in
    # winner_scale_down_range (e.g. 0.10 -> keep 90% of the outlier's edge).
    winner_percentile: float = 90.0
    winner_scale_down_range: tuple = (0.10, 0.20)

    # --- tail-risk metric parameters ---
    ruin_drawdown_pct: float = 0.20        # e.g. 0.20 = 20% equity drawdown
    profit_target_multiple: Optional[float] = None  # e.g. 1.5 -> only breaches before hitting +50% count as ruin
    cvar_alpha: float = 0.05               # worst 5% of terminal outcomes

    # --- plotting ---
    n_sample_paths_for_fan_chart: int = 2_000
    n_spaghetti_paths: int = 25


@dataclass
class MonteCarloResults:
    config: MonteCarloConfig
    n_historical_trades: int
    max_drawdown_pct: np.ndarray           # (n_simulations,) worst peak-to-trough dd per path, as a positive fraction
    terminal_equity: np.ndarray            # (n_simulations,)
    terminal_return_pct: np.ndarray        # (n_simulations,)
    ruined: np.ndarray                     # (n_simulations,) bool
    sample_curves: np.ndarray              # (n_sample, path_length+1) equity curves, for plotting only
    summary: dict = field(default_factory=dict)


# ============================================================================
# CORE SIMULATION
# ============================================================================
class MonteCarloStressTester:
    """
    Circular block bootstrap + adversarial stressors over a historical
    R-multiple trade return sequence.

    Usage:
        tester = MonteCarloStressTester(returns_r_multiples, MonteCarloConfig())
        results = tester.run()
        tester.print_summary(results)
        tester.plot(results)          # -> build/monte_carlo_stress_report.html
    """

    def __init__(self, returns: Sequence[float], config: Optional[MonteCarloConfig] = None):
        returns = np.asarray(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        if len(returns) < 30:
            raise ValueError(
                f"Only {len(returns)} historical trades supplied — block bootstrap "
                f"with block sizes up to {config.block_size_max if config else 20} needs "
                f"materially more history than that to be meaningful. Pool more folds/thresholds "
                f"or lower block_size_max before trusting the output."
            )
        self.returns = returns
        self.n = len(returns)
        self.config = config or MonteCarloConfig()
        if self.config.block_size_max > self.n:
            raise ValueError(
                f"block_size_max ({self.config.block_size_max}) exceeds the number of "
                f"historical trades ({self.n})."
            )
        self.path_length = self.config.path_length or self.n

        # Fixed reference point for "what counts as an outlier winner" — computed
        # ONCE off the real historical distribution, not re-derived per path/batch,
        # so the definition of "outlier" doesn't drift with what got resampled.
        self._winner_cut = np.percentile(returns, self.config.winner_percentile)
        self._historical_losses = returns[returns < 0]

    # ------------------------------------------------------------------
    # Circular block bootstrap — vectorized index generation for a batch
    # ------------------------------------------------------------------
    def _bootstrap_batch(self, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        n, L = self.n, self.path_length

        block_size = int(rng.integers(cfg.block_size_min, cfg.block_size_max + 1))
        num_blocks = int(np.ceil(L / block_size))

        starts = rng.integers(0, n, size=(n_paths, num_blocks))          # (P, num_blocks)
        offsets = np.arange(block_size)                                   # (block_size,)
        idx = (starts[:, :, None] + offsets[None, None, :]) % n           # (P, num_blocks, block_size)
        idx = idx.reshape(n_paths, num_blocks * block_size)[:, :L]        # (P, L)

        return self.returns[idx]  # (P, L) bootstrapped R-multiple returns

    # ------------------------------------------------------------------
    # Adversarial stressors — applied in-place on a batch's return matrix
    # ------------------------------------------------------------------
    def _apply_stressors(self, batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        P, L = batch.shape

        # 3. Returns shuffling with volatility scaling — shrink outlier winners.
        #    Applied first so downstream win-rate accounting reflects the
        #    already-shrunk (still-positive, in virtually all cases) values.
        is_outlier_winner = batch > self._winner_cut
        shrink = rng.uniform(cfg.winner_scale_down_range[0], cfg.winner_scale_down_range[1], size=(P, L))
        batch = np.where(is_outlier_winner, batch * (1.0 - shrink), batch)

        # 2. Win-rate sensitivity shock — flip a random slice of each path's
        #    *currently* winning trades into losses, degrading realized win
        #    rate by win_rate_shock_range percentage points per path.
        degrade_pct = rng.uniform(cfg.win_rate_shock_range[0], cfg.win_rate_shock_range[1], size=P)
        n_to_flip = np.round(degrade_pct * L).astype(int)
        is_win = batch > 0
        # Rank each row's winning trades randomly so we can take a random subset
        # per row without a Python loop: argsort of a random score, restricted
        # to winners by pushing losers' scores to +inf so they sort last.
        rand_scores = np.where(is_win, rng.random(size=(P, L)), np.inf)
        flip_rank = np.argsort(rand_scores, axis=1)  # column 0 = most "eligible" winner to flip
        col_rank = np.empty_like(flip_rank)
        rows = np.arange(P)[:, None]
        col_rank[rows, flip_rank] = np.arange(L)[None, :]
        flip_mask = is_win & (col_rank < n_to_flip[:, None])
        replacement_loss = (
            rng.choice(self._historical_losses, size=(P, L))
            if len(self._historical_losses) > 0
            else -1.0
        )
        batch = np.where(flip_mask, replacement_loss, batch)

        # 1. Slippage inflation — extra cost beyond the assumed baseline,
        #    scaled by a random 1.5x-3.0x multiplier, drawn per trade.
        slip_mult = rng.uniform(cfg.slippage_mult_range[0], cfg.slippage_mult_range[1], size=(P, L))
        extra_slippage_cost = cfg.slippage_r_base * (slip_mult - 1.0)
        batch = batch - extra_slippage_cost

        return batch

    # ------------------------------------------------------------------
    # Equity curve construction + per-path tail metrics for one batch
    # ------------------------------------------------------------------
    def _equity_and_metrics(self, batch: np.ndarray):
        cfg = self.config
        if cfg.compounding:
            equity = cfg.starting_equity * np.cumprod(1.0 + cfg.risk_per_trade * batch, axis=1)
        else:
            equity = cfg.starting_equity * (1.0 + cfg.risk_per_trade * np.cumsum(batch, axis=1))
        equity = np.concatenate(
            [np.full((batch.shape[0], 1), cfg.starting_equity), equity], axis=1
        )  # prepend t=0 so drawdown/ruin can be checked from the start

        running_peak = np.maximum.accumulate(equity, axis=1)
        drawdown = (running_peak - equity) / running_peak
        max_dd = drawdown.max(axis=1)

        if cfg.profit_target_multiple is not None:
            target_equity = cfg.starting_equity * cfg.profit_target_multiple
            hit_target = equity >= target_equity
            # first index (per row) at which target is hit; L+1 (never) if not hit
            first_hit = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1), equity.shape[1])
            breach = drawdown >= cfg.ruin_drawdown_pct
            step_idx = np.arange(equity.shape[1])[None, :]
            breach_before_target = breach & (step_idx < first_hit[:, None])
            ruined = breach_before_target.any(axis=1)
        else:
            ruined = max_dd >= cfg.ruin_drawdown_pct

        terminal_equity = equity[:, -1]
        terminal_return_pct = (terminal_equity - cfg.starting_equity) / cfg.starting_equity
        return equity, max_dd, terminal_equity, terminal_return_pct, ruined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> MonteCarloResults:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)

        max_dd_all, terminal_eq_all, terminal_ret_all, ruined_all = [], [], [], []
        sample_curves = []
        n_sample_target = min(cfg.n_sample_paths_for_fan_chart, cfg.n_simulations)

        remaining = cfg.n_simulations
        while remaining > 0:
            batch_n = min(cfg.batch_size, remaining)
            raw = self._bootstrap_batch(batch_n, rng)
            stressed = self._apply_stressors(raw, rng) if cfg.apply_stressors else raw
            equity, max_dd, term_eq, term_ret, ruined = self._equity_and_metrics(stressed)

            max_dd_all.append(max_dd)
            terminal_eq_all.append(term_eq)
            terminal_ret_all.append(term_ret)
            ruined_all.append(ruined)

            if len(sample_curves) < n_sample_target:
                take = min(n_sample_target - len(sample_curves), batch_n)
                sample_curves.append(equity[:take])

            remaining -= batch_n

        max_dd_all = np.concatenate(max_dd_all)
        terminal_eq_all = np.concatenate(terminal_eq_all)
        terminal_ret_all = np.concatenate(terminal_ret_all)
        ruined_all = np.concatenate(ruined_all)
        sample_curves = np.concatenate(sample_curves, axis=0)

        sorted_terminal_ret = np.sort(terminal_ret_all)
        cutoff = max(1, int(len(sorted_terminal_ret) * cfg.cvar_alpha))
        cvar = float(sorted_terminal_ret[:cutoff].mean())

        summary = {
            "n_simulations": cfg.n_simulations,
            "n_historical_trades": self.n,
            "path_length": self.path_length,
            "median_max_drawdown_pct": float(np.median(max_dd_all)),
            "p95_max_drawdown_pct": float(np.percentile(max_dd_all, 95)),
            "p99_max_drawdown_pct": float(np.percentile(max_dd_all, 99)),
            "worst_max_drawdown_pct": float(max_dd_all.max()),
            "cvar_terminal_return_pct": cvar,
            "median_terminal_return_pct": float(np.median(terminal_ret_all)),
            "prob_negative_terminal_pct": float(np.mean(terminal_ret_all < 0) * 100),
            "prob_ruin_pct": float(np.mean(ruined_all) * 100),
            "ruin_drawdown_threshold_pct": cfg.ruin_drawdown_pct * 100,
        }

        return MonteCarloResults(
            config=cfg,
            n_historical_trades=self.n,
            max_drawdown_pct=max_dd_all,
            terminal_equity=terminal_eq_all,
            terminal_return_pct=terminal_ret_all,
            ruined=ruined_all,
            sample_curves=sample_curves,
            summary=summary,
        )

    # ------------------------------------------------------------------
    def print_summary(self, results: MonteCarloResults) -> None:
        s = results.summary
        print("\n" + "=" * 70)
        print("MONTE CARLO STRESS TEST — CIRCULAR BLOCK BOOTSTRAP")
        print("=" * 70)
        print(f"Historical trades pooled:      {s['n_historical_trades']}")
        print(f"Simulated paths:               {s['n_simulations']:,}  "
              f"(block size {self.config.block_size_min}-{self.config.block_size_max} trades)")
        print(f"Path length:                   {s['path_length']} trades")
        print(f"Stressors applied:              {self.config.apply_stressors}")
        print("-" * 70)
        print(f"Median Max Drawdown:            {s['median_max_drawdown_pct']:.1%}")
        print(f"95th Pct Max Drawdown:          {s['p95_max_drawdown_pct']:.1%}")
        print(f"99th Pct Max Drawdown (worst 1%): {s['p99_max_drawdown_pct']:.1%}")
        print(f"Absolute Worst Max Drawdown:    {s['worst_max_drawdown_pct']:.1%}")
        print("-" * 70)
        print(f"Median Terminal Return:         {s['median_terminal_return_pct']:+.1%}")
        print(f"CVaR / Expected Shortfall ({self.config.cvar_alpha:.0%}):  {s['cvar_terminal_return_pct']:+.1%}")
        print(f"P(terminal return < 0):         {s['prob_negative_terminal_pct']:.1f}%")
        print("-" * 70)
        print(f"Probability of Ruin (breach {s['ruin_drawdown_threshold_pct']:.0f}% DD): "
              f"{s['prob_ruin_pct']:.2f}%")
        print("=" * 70)

    # ------------------------------------------------------------------
    def plot(self, results: MonteCarloResults, output_html: Optional[str] = None) -> str:
        if output_html is None:
            output_html = _build_output("monte_carlo_stress_report.html")
        cfg = results.config
        curves = results.sample_curves  # (n_sample, L+1)
        x = np.arange(curves.shape[1])

        pcts = [5, 25, 50, 75, 95]
        bands = {p: np.percentile(curves, p, axis=0) for p in pcts}

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Simulated Equity Curves (percentile bands)",
                "Max Drawdown Distribution",
                "Terminal Return Distribution",
            ),
            column_widths=[0.42, 0.29, 0.29],
        )

        # --- Panel 1: fan chart ---
        spaghetti_n = min(cfg.n_spaghetti_paths, curves.shape[0])
        for i in range(spaghetti_n):
            fig.add_trace(go.Scatter(
                x=x, y=curves[i], mode="lines",
                line=dict(width=0.6, color="rgba(120,120,120,0.25)"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([bands[95], bands[5][::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0), name="5th-95th pct", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([bands[75], bands[25][::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.30)",
            line=dict(width=0), name="25th-75th pct", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=bands[50], mode="lines", line=dict(width=2, color="rgb(31,119,180)"),
            name="Median path",
        ), row=1, col=1)
        fig.add_hline(y=cfg.starting_equity, line_dash="dash", line_color="black", opacity=0.4, row=1, col=1)

        # --- Panel 2: drawdown distribution ---
        fig.add_trace(go.Histogram(
            x=results.max_drawdown_pct * 100, nbinsx=60, name="Max DD",
            marker_color="rgba(214,39,40,0.7)", showlegend=False,
        ), row=1, col=2)
        fig.add_vline(x=results.summary["p95_max_drawdown_pct"] * 100, line_dash="dash",
                      line_color="orange", annotation_text="95th", row=1, col=2)
        fig.add_vline(x=results.summary["p99_max_drawdown_pct"] * 100, line_dash="dash",
                      line_color="darkred", annotation_text="99th", row=1, col=2)
        fig.add_vline(x=cfg.ruin_drawdown_pct * 100, line_dash="dot",
                      line_color="black", annotation_text="Ruin", row=1, col=2)

        # --- Panel 3: terminal return distribution ---
        fig.add_trace(go.Histogram(
            x=results.terminal_return_pct * 100, nbinsx=60, name="Terminal Return",
            marker_color="rgba(44,160,44,0.7)", showlegend=False,
        ), row=1, col=3)
        fig.add_vline(x=results.summary["cvar_terminal_return_pct"] * 100, line_dash="dash",
                      line_color="darkred", annotation_text=f"CVaR {cfg.cvar_alpha:.0%}", row=1, col=3)
        fig.add_vline(x=0, line_dash="dot", line_color="black", row=1, col=3)

        title = (
            f"Monte Carlo Stress Test — {cfg.n_simulations:,} paths, "
            f"block size {cfg.block_size_min}-{cfg.block_size_max}, "
            f"{'stressed' if cfg.apply_stressors else 'unstressed'} | "
            f"P(ruin @ {cfg.ruin_drawdown_pct:.0%} DD) = {results.summary['prob_ruin_pct']:.2f}% | "
            f"99th pct DD = {results.summary['p99_max_drawdown_pct']:.1%} | "
            f"CVaR({cfg.cvar_alpha:.0%}) = {results.summary['cvar_terminal_return_pct']:+.1%}"
        )
        fig.update_layout(title=title, template="plotly_white", height=520, hovermode="closest")
        fig.update_xaxes(title_text="Trade #", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=1, col=2)
        fig.update_xaxes(title_text="Terminal Return (%)", row=1, col=3)

        fig.write_html(output_html)
        return output_html


# ============================================================================
# Convenience: run directly off the walk-forward trade log CSV this pipeline
# already exports (STEP 10 of ml_state_optimized.py -> build/walk_forward_trade_log_v3.csv)
# ============================================================================
def run_monte_carlo_from_csv(
    csv_path: str,
    threshold: Optional[float] = None,
    ret_col: str = "ret",
    threshold_col: str = "threshold",
    datetime_col: str = "datetime",
    config: Optional[MonteCarloConfig] = None,
    output_html: Optional[str] = None,
):
    """
    Loads `build/walk_forward_trade_log_v3.csv` (or any CSV with the same columns),
    optionally filters to one threshold, sorts by datetime, and runs the
    full stress test. Returns (tester, results).
    """
    import polars as pl

    if output_html is None:
        output_html = _build_output("monte_carlo_stress_report.html")

    df = pl.read_csv(csv_path, try_parse_dates=True)
    if threshold is not None and threshold_col in df.columns:
        df = df.filter(pl.col(threshold_col) == threshold)
    if datetime_col in df.columns:
        df = df.sort(datetime_col)

    returns = df[ret_col].to_numpy()
    tester = MonteCarloStressTester(returns, config)
    results = tester.run()
    tester.print_summary(results)
    tester.plot(results, output_html)
    print(f"\nStress test chart saved to: {output_html}")
    return tester, results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python monte_carlo.py <path_to_trade_log.csv> [threshold]")
        sys.exit(1)
    csv_path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run_monte_carlo_from_csv(csv_path, threshold=thr)