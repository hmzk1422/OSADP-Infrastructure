"""
regime_data.py
==============
Shared, leakage-safe data layer for the regime-detection study.

Consumes the realized walk-forward trade log
(`build/walk_forward_trade_log_v3.csv`) and produces a single tidy frame of
ENTRY-TIME-OBSERVABLE state for every trade, plus:

  * corrected chronological ordering,
  * per-trade MFE / MAE / escape statistics (derived from the observational
    `execution_price_*` holding-window windows),
  * volatility / market-state proxies that use ONLY pre-entry information,
  * strategy-state rolling variables built ONLY from strictly-prior trades,
  * an expanding-window (walk-forward) fold splitter used by every learner.

LEAKAGE CONTRACT
----------------
For every entry-condition analysis we use only:

  * `{feature}_open`   -> feature value on the 1-second grid row the trade
                        entered (point-in-time safe; see the comment block in
                        `research/ofmr/ml_state_optimized.py` near
                        `_trade_feature_windows`).
  * `{feature}_high/_low/_close` describe the holding window (entry, exit].
                        They are OUTCOME data for this trade and are NEVER used
                        as entry conditions. They are only used to describe
                        what *already-exited prior trades* did (strictly before
                        the current entry, because entries within a threshold
                        stream are non-overlapping by construction).
  * Rolling strategy-state variables use trades strictly before the current
    trade via `.shift(1)` on the sorted same-threshold stream.

NOTE on trade duration: `entry_dt`/`exit_dt` in the CSV are stored in a corrupt
interning format (year 56282+), so we cannot recover exact durations from them.
Because the research walk-forward enforces LOCKOUT (entries within a threshold
stream are non-overlapping: next entry is at or after the prior exit), the gap
between consecutive entries is a real-time-available UPPER BOUND on the prior
trade's holding time. We use that gap as the time-to-exit proxy.

Every learner (buckets, trees, HMM) is fit on the training segment of an
expanding window and evaluated on the subsequent unseen month, so no fitted
threshold / split / parameter / state definition ever touches the test segment.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ml_shared_config import (  # noqa: E402
    BUILD_DIR, SESSION_BOUNDS, SESSION_ORDER,
    REGIME_FOLD_MODE, REGIME_FOLD_N_SAMPLES, REGIME_FOLD_BLOCK_SIZE,
    REGIME_FOLD_HORIZON_SECONDS, REGIME_FOLD_EMBARGO_SECONDS,
)

# ----------------------------------------------------------------------------
# Study configuration
# ----------------------------------------------------------------------------
TRADE_LOG = os.path.join(BUILD_DIR, "walk_forward_trade_log_v3.csv")
STUDY_OUT = os.path.join(BUILD_DIR, "regime_study")

# Confidence threshold under test (user-specified).
THRESHOLD = 0.20
# ------------------------------------------------------------------------------
# Fold-scheme selection for make_folds() — applies to EVERY learner in the
# study (buckets, trees, HMM all pull their folds from here).
#
#   "monthly" -> expanding window, test = the following calendar month
#                (original scheme, unchanged).
#   "purged"  -> N-purged chronological walk-forward in the style of
#                ml_state_optimized.purged_chronological_splits: an initial
#                train block of 35% of the stream, then `FOLD_N_SAMPLES`
#                non-overlapping test blocks of `FOLD_BLOCK_SIZE` trades
#                spaced evenly over the remainder; every train trade whose
#                outcome horizon + embargo reaches into the test block is
#                purged from training.
#
# Source of truth: research/ml_shared_config.py (REGIME_FOLD_* constants) —
# edit there, no environment variables or code changes needed.
# ------------------------------------------------------------------------------
FOLD_MODE = REGIME_FOLD_MODE
FOLD_N_SAMPLES = REGIME_FOLD_N_SAMPLES
FOLD_BLOCK_SIZE = REGIME_FOLD_BLOCK_SIZE
FOLD_HORIZON_SECONDS = REGIME_FOLD_HORIZON_SECONDS
FOLD_EMBARGO_SECONDS = REGIME_FOLD_EMBARGO_SECONDS

# Minimum number of prior trades required for a rolling strategy-state variable.
MIN_PRIOR = 20

# Trade-time session labels (mirror ml_state_optimized.SESSION_BOUNDS).
SESSION_COLS = [label for label, _, _ in SESSION_BOUNDS]

# Entry-safe feature families (only the `_open` variant is used as an entry
# condition). `cvd_delta_z`/`absorption_z`/`delta_divergence_z` are provably
# ----------------------------------------------------------------------------
# Loading + base cleansing
# ----------------------------------------------------------------------------
def load_trade_log(path: str = TRADE_LOG, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Load the trade log, restrict to one confidence threshold, sort by entry.

    Returns a frame indexed 0..N-1 in strictly chronological entry order for
    the given threshold stream.
    """
    df = pd.read_csv(path)
    df = df[df["threshold"] == threshold].copy()
    df["dt"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    df["trade_id"] = np.arange(len(df), dtype=np.int64)
    return df


# ----------------------------------------------------------------------------
# Entry-state features (open variants only)
# ----------------------------------------------------------------------------
def _dedupe_features(df: pd.DataFrame) -> None:
    """Sanity checks + report provably redundant / dead entry features."""
    n = len(df)
    dead = {"ofi_z_open": (df["ofi_z_open"] == 0).sum() == n}
    for name, is_dead in dead.items():
        print(f"  [data] {name}: {'DEAD (all zeros) - excluded' if is_dead else 'ok'}")

    r_cd_ab = df["cvd_delta_z_open"].corr(df["absorption_z_open"])
    r_cd_dd = df["cvd_delta_z_open"].corr(df["delta_divergence_z_open"])
    print(f"  [data] collinearity: cvd_delta/absorption r={r_cd_ab:.3f}, "
          f"cvd_delta/delta_divergence r={r_cd_dd:.3f} -> keeping cvd_slope only")


DUP_FLOW_COLS = [
    "ofi_z_open", "ofi_z_high", "ofi_z_low", "ofi_z_close",
    "cvd_delta_z_open", "cvd_delta_z_high", "cvd_delta_z_low", "cvd_delta_z_close",
    "absorption_z_open", "absorption_z_high", "absorption_z_low", "absorption_z_close",
    "delta_divergence_z_open", "delta_divergence_z_high",
    "delta_divergence_z_low", "delta_divergence_z_close",
]


def add_entry_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach interpretable entry-state variables (pre-entry / at-entry only)."""
    # Position of price relative to the 14-bar moving average (interpretable).
    df["price_rel_sma14"] = df["execution_price_open"] / df["sma_14_open"] - 1.0

    # Calendar state (knowable at entry).
    df["hour_of_day"] = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    df["weekday"] = df["dt"].dt.weekday

    # Session label from entry time-of-day (identical mapping to research).
    h = df["hour_of_day"]
    df["session"] = SESSION_COLS[-1]
    for label, lo, hi in SESSION_BOUNDS:
        df.loc[(h >= lo / 60.0) & (h < hi / 60.0), "session"] = label
    df["session_ord"] = df["session"].map({s: i for i, s in enumerate(SESSION_COLS)})
    df["week"] = df["dt"].dt.isocalendar().week.astype(int)
    df["year_month"] = df["dt"].dt.strftime("%Y-%m")
    return df
# collinear (r = 1.0 on this log) so only `cvd_slope_z` + `depth_imbalance_z`
# are kept for order-flow; `ofi_z` is all-zeros on this log.
SPREAD_FEATURES = ["spread_bps_z_open"]
TREND_FEATURES = ["ma_slope_5_open", "ma_accel_open"]
MOMENTUM_FEATURES = ["price_roc_3_open", "rsi_2_open"]
FLOW_FEATURES = ["cvd_slope_z_open", "tick_autocorr_open",
                "inst_ratio_open", "depth_imbalance_z_open"]

# Full entry-state variable set used by the tree learner.
ENTRY_STATE_FEATURES = (
    SPREAD_FEATURES + TREND_FEATURES + MOMENTUM_FEATURES + FLOW_FEATURES
)

# Market-state-only set fed to the HMM (compact, interpretable, deduplicated).
HMM_FEATURES = (
    SPREAD_FEATURES + TREND_FEATURES + MOMENTUM_FEATURES + FLOW_FEATURES
)

MIN_PERIODS_ROLLING = 10
MIN_LEAF = 150          # minimum trades for a reported subgroup
RANDOM_STATE = 42
# ----------------------------------------------------------------------------
# Volatility proxies (strictly pre-entry)
# ----------------------------------------------------------------------------
def _realized_range(open_p, high_p, low_p):
    """Normalized price range of a completed trade's holding window."""
    denom = np.where(open_p > 0, open_p, np.nan)
    return (high_p - low_p) / denom


def _mae_pct(open_p, low_p, sign=1.0):
    """Adverse excursion magnitude in % of entry price (long: (open-low)/open)."""
    denom = np.where(open_p > 0, open_p, np.nan)
    return sign * (open_p - low_p) / denom


def _mfe_pct(open_p, high_p, sign=1.0):
    """Favorable excursion magnitude in % of entry price (long: (high-open)/open)."""
    denom = np.where(open_p > 0, open_p, np.nan)
    return sign * (high_p - open_p) / denom


def add_trade_outcome_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-trade range / MFE / MAE from the observational price windows.

    These describe this trade's OWN realized excursion - OUTCOME data, never
    used as entry conditions. Used to describe the outcome distribution inside
    each regime bucket and as strictly-PRIOR info for rolling volatility vars.
    """
    df["range_pct"] = _realized_range(df["execution_price_open"].to_numpy(),
                                      df["execution_price_high"].to_numpy(),
                                      df["execution_price_low"].to_numpy())
    df["mae_pct"] = _mae_pct(df["execution_price_open"].to_numpy(),
                            df["execution_price_low"].to_numpy())
    df["mfe_pct"] = _mfe_pct(df["execution_price_open"].to_numpy(),
                            df["execution_price_high"].to_numpy())
    df["is_timeout"] = (df["ret"] < 0) & (df["ret"] > -0.9)
    df["is_stopout"] = (df["ret"] < 0) & (df["ret"] <= -0.9)
    df["is_winner"] = df["ret"] > 0
    return df


def enterprise_volatility_prior(df: pd.DataFrame, warm=MIN_PRIOR) -> pd.DataFrame:
    """Volatility / activity proxies from PRIOR completed trades only.

    All shifted so trade `i` only sees trades `<= i-1`. Entries within a
    threshold stream are non-overlapping; the prior trade's holding window lies
    strictly before this entry.
    """
    out = pd.DataFrame(index=df.index)
    out["vol_prior20_range"] = df["range_pct"].rolling(20, min_periods=warm).median().shift(1)
    out["vol_prior20_range_std"] = df["range_pct"].rolling(20, min_periods=warm).std().shift(1)
    out["vol_prior20_mae"] = df["mae_pct"].rolling(20, min_periods=warm).median().shift(1)
    out["vol_prior50_range"] = df["range_pct"].rolling(50, min_periods=warm).median().shift(1)
    out["vol_prior20_ret_std"] = df["ret"].rolling(20, min_periods=warm).std().shift(1)
    return out


# ----------------------------------------------------------------------------
# Strategy-state rolling variables (strictly prior trades)
# ----------------------------------------------------------------------------
def add_strategy_state(df: pd.DataFrame, warm=10) -> pd.DataFrame:
    """Rolling behavior of the strategy itself - PRIOR trades only.

    Rolling windows run on the same-threshold chronological stream and are
    shifted by one, so the current trade never contributes.
    """
    r = np.where(df["ret"] > 0, df["ret"], np.where(df["ret"] < 0, df["ret"], 0.0))

    def roll_shift(series, window, fn, min_periods=warm):
        s = series.rolling(window, min_periods=min_periods).agg(fn)
        return s.shift(1)

    df["strat_prev10_mean_r"] = roll_shift(pd.Series(r, index=df.index), 10, "mean")
    df["strat_prev10_median_r"] = roll_shift(pd.Series(r, index=df.index), 10, "median")
    df["strat_prev10_win_rate"] = roll_shift(df["is_winner"].astype(float), 10, "mean")
    df["strat_prev20_win_rate"] = roll_shift(df["is_winner"].astype(float), 20, "mean")
    df["strat_prev50_win_rate"] = roll_shift(df["is_winner"].astype(float), 50, "mean")

    def profit_factor(s):
        w = s[s > 0].sum()
        l = -s[s < 0].sum()
        return w / l if l > 1e-9 else np.nan

    df["strat_prev20_profit_factor"] = (
        pd.Series(r, index=df.index).rolling(20, min_periods=warm).agg(profit_factor).shift(1)
    )
    df["strat_prev20_ret_std"] = roll_shift(pd.Series(r, index=df.index), 20, "std")
    df["strat_prev10_mae"] = roll_shift(df["mae_pct"], 10, "mean")
    df["strat_prev10_mfe"] = roll_shift(df["mfe_pct"], 10, "mean")

    # --- Time-to-exit / stop-out speed (gap proxy; module docstring) ---
    gap_min = df["dt"].diff().dt.total_seconds() / 60.0
    df["prev_gap_min"] = gap_min

    quick_def = 6.0  # minutes: next entry occurred at the earliest possible exit
    # (gap granularity is ~5 min; gap <= 6 min => prior trade exited within its
    # first bar, i.e. an immediate stop-out / very short hold)
    is_quick = (gap_min < quick_def).astype(float)
    df["strat_prev10_median_gap"] = roll_shift(gap_min, 10, "median")
    df["strat_prev20_median_gap"] = roll_shift(gap_min, 20, "median")
    df["strat_prev50_median_gap"] = roll_shift(gap_min, 50, "median")
    df["strat_prev10_quick_frac"] = roll_shift(is_quick, 10, "mean")
    df["strat_prev20_quick_frac"] = roll_shift(is_quick, 20, "mean")
    df["strat_prev10_quick_winless_frac"] = (
        (is_quick * (~df["is_winner"]).astype(float))
        .rolling(10, min_periods=warm).mean().shift(1)
    )
    return df


# ----------------------------------------------------------------------------
# Conditional-outcome metrics
# ----------------------------------------------------------------------------
def ret_metrics(r: np.ndarray) -> dict:
    """Conditional-outcome summary used across every methodology."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        return dict(n=0, mean_r=np.nan, median_r=np.nan, win_rate=np.nan,
                    p_1r=np.nan, p_2r=np.nan, p_3r=np.nan, p_5r=np.nan,
                    pf=np.nan, total_r=np.nan, avg_win=np.nan, avg_loss=np.nan)
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    pf = (len(wins) * avg_win) / (-len(losses) * avg_loss) if len(losses) else np.nan
    return dict(
        n=n,
        mean_r=float(r.mean()),
        median_r=float(np.median(r)),
        win_rate=float((r > 0).mean()),
        p_1r=float((r >= 1.0).mean()),
        p_2r=float((r >= 2.0).mean()),
        p_3r=float((r >= 3.0).mean()),
        p_5r=float((r >= 5.0).mean()),
        pf=float(pf),
        total_r=float(r.sum()),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
    )


def bootstrap_ci(r: np.ndarray, stat: str = "mean_r", n_boot: int = 2000,
                seed: int = RANDOM_STATE) -> tuple[float, float]:
    """Percentile bootstrap CI (90%) for a mean/win-rate statistic."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    boot = r[idx]
    vals = boot.mean(axis=1) if stat == "mean_r" else (boot > 0).mean(axis=1)
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))


def fmt_metrics(m: dict) -> str:
    """Compact single-line rendering of a conditional-outcome dict."""
    pf = m["pf"] if np.isfinite(m["pf"]) else float("nan")
    return (f"n={m['n']:>6d} meanR={m['mean_r']:>7.3f} medR={m['median_r']:>7.3f} "
            f"WR={m['win_rate'] * 100:>5.1f}% PF={pf:>5.2f} "
            f"P(1R)={m['p_1r'] * 100:>5.1f}% P(2R)={m['p_2r'] * 100:>5.1f}% "
            f"P(3R)={m['p_3r'] * 100:>5.1f}% totalR={m['total_r']:>8.1f}")


# ----------------------------------------------------------------------------
# Walk-forward folds (expanding window, monthly test cutoffs)
# ----------------------------------------------------------------------------
def make_folds(df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series, str]]:
    """Fold splitter used by every learner in the study.

    Dispatches on FOLD_MODE (see the configuration block at the top of this
    file): "monthly" -> expanding-window monthly cutoffs (original, unchanged);
    "purged" -> N-purged chronological walk-forward mirroring
    ml_state_optimized.purged_chronological_splits.

    Returns list of (train_mask, test_mask, label). Bucket quantiles / tree
    splits / HMM parameters are fit on `train_mask` and evaluated on the
    untouched `test_mask`, so no fitted threshold / split / parameter / state
    definition ever touches the test segment in either mode.
    """
    if FOLD_MODE == "purged":
        return _make_purged_folds(df)
    return _make_monthly_folds(df)


def _make_monthly_folds(df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series, str]]:
    """Expanding-window folds at monthly cutoffs (original scheme).

    Returns list of (train_mask, test_mask, label). Bucket quantiles / tree
    splits / HMM parameters are fit on `train_mask` and evaluated on the
    untouched `test_mask` (the following month). This is the trade-log-
    granularity counterpart of the purged walk-forward of the model pipeline.
    """
    months = sorted(df["year_month"].unique())
    folds = []
    for k in range(1, len(months)):
        cutoff = months[k]
        train_mask = df["year_month"] < cutoff
        test_mask = df["year_month"] == cutoff
        if int(train_mask.sum()) >= 100 and int(test_mask.sum()) >= 100:
            folds.append((train_mask, test_mask, f"train<{cutoff}->test{cutoff}"))
    return folds


def _make_purged_folds(df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series, str]]:
    """N-purged chronological walk-forward at trade granularity.

    Mirrors ml_state_optimized.purged_chronological_splits:
      * initial train block = 35% of the stream,
      * `FOLD_N_SAMPLES` non-overlapping test blocks of `FOLD_BLOCK_SIZE`
        trades each, spaced evenly over the remaining rows (fold count is
        auto-reduced with a [FOLD COUNT ADJUSTED] notice if fewer independent
        windows fit),
      * the expanding train window is purged of every trade whose outcome
        horizon + embargo reaches into the test block.

    The frame from load_trade_log() is sorted chronologically with a RangeIndex,
    so positional indices == df.index and the boolean masks returned here are
    drop-in compatible with the monthly scheme's consumers.
    """
    n = len(df)
    # tz-aware timestamps -> UTC-naive datetime64[ns] so numpy timedelta64
    # arithmetic (the purge comparison below) works directly on the array.
    dt = df["dt"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    horizon = np.timedelta64(FOLD_HORIZON_SECONDS, "s")
    embargo = np.timedelta64(FOLD_EMBARGO_SECONDS, "s")
    min_initial_train = int(n * 0.35)
    available_test_run = n - min_initial_train - FOLD_BLOCK_SIZE

    if available_test_run <= 0:
        print(f"[folds] purged mode skipped: stream of {n} trades is too short "
              f"for block_size={FOLD_BLOCK_SIZE} after the {min_initial_train}-trade "
              f"initial train block.")
        return []

    max_non_overlapping = max(1, available_test_run // FOLD_BLOCK_SIZE)
    n_samples = FOLD_N_SAMPLES
    if n_samples > max_non_overlapping:
        print(f"[FOLD COUNT ADJUSTED] requested n_samples={n_samples}, but only "
              f"{max_non_overlapping} non-overlapping {FOLD_BLOCK_SIZE}-trade test "
              f"windows fit in the {available_test_run} trades available after the "
              f"{min_initial_train}-trade initial train block. Reducing n_samples "
              f"to {max_non_overlapping} so every fold tests an independent period.")
        n_samples = max_non_overlapping

    step_intervals = available_test_run // n_samples
    folds = []
    for fold in range(n_samples):
        start_idx = min_initial_train + (fold * step_intervals)
        test_idx = np.arange(start_idx, start_idx + FOLD_BLOCK_SIZE)
        test_start_time = dt[test_idx[0]]

        possible_train_idx = np.arange(0, start_idx)
        purge_mask = dt[possible_train_idx] + horizon + embargo >= test_start_time
        train_idx = possible_train_idx[~purge_mask]
        n_purged = len(possible_train_idx) - len(train_idx)

        if len(train_idx) < 100 or len(test_idx) < 100:
            continue

        train_mask = pd.Series(False, index=df.index)
        test_mask = pd.Series(False, index=df.index)
        train_mask.iloc[train_idx] = True
        test_mask.iloc[test_idx] = True
        folds.append((
            train_mask, test_mask,
            f"wf{fold + 1:02d}:train<{pd.Timestamp(test_start_time):%Y-%m-%d %H:%M}"
            f"->test{FOLD_BLOCK_SIZE}trades(purged={n_purged})",
        ))
    return folds


def STRAT_COLS() -> list[str]:
    """All rolling strategy-state / volatility-prior column names (prior only)."""
    return [c for c in [
        "strat_prev10_mean_r", "strat_prev10_median_r", "strat_prev10_win_rate",
        "strat_prev20_win_rate", "strat_prev50_win_rate", "strat_prev20_profit_factor",
        "strat_prev20_ret_std", "strat_prev10_mae", "strat_prev10_mfe",
        "strat_prev10_median_gap", "strat_prev20_median_gap", "strat_prev50_median_gap",
        "strat_prev10_quick_frac", "strat_prev20_quick_frac",
        "strat_prev10_quick_winless_frac",
        "vol_prior20_range", "vol_prior20_range_std", "vol_prior20_mae",
        "vol_prior50_range", "vol_prior20_ret_std",
    ]]


def ensure_out() -> str:
    os.makedirs(STUDY_OUT, exist_ok=True)
    return STUDY_OUT


def load_study_frame() -> pd.DataFrame:
    """End-to-end loader: threshold stream -> entry-state + outcome + state vars."""
    df = load_trade_log()
    print(f"[data] threshold={THRESHOLD} stream: {len(df)} trades "
          f"({df['dt'].min()} .. {df['dt'].max()})")
    _dedupe_features(df)
    drop_cols = [c for c in DUP_FLOW_COLS if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    df = add_entry_state(df)
    df = add_trade_outcome_proxies(df)
    df = pd.concat([df, enterprise_volatility_prior(df)], axis=1)
    df = add_strategy_state(df)
    # rolling state unavailable for the first MIN_PRIOR trades of the stream
    df["_complete"] = ~df[STRAT_COLS()].isna().any(axis=1)
    return df


if __name__ == "__main__":
    d = load_study_frame()
    print(f"[data] frame shape {d.shape}")
    print(f"[data] complete rows (rolling state available): {int(d['_complete'].sum())}")
    print()
    cols = ["ret", "range_pct", "mae_pct", "mfe_pct", "prev_gap_min",
            "vol_prior20_range", "strat_prev10_median_gap", "strat_prev10_win_rate",
            "spread_bps_z_open", "ma_slope_5_open", "rsi_2_open", "cvd_slope_z_open"]
    print(d[cols].describe(percentiles=[0.05, 0.5, 0.95]).T.round(4).to_string())
    print()
    print("session counts:\n", d["session"].value_counts().to_string())
    print()
    print("folds:")
    for tr, te, lab in make_folds(d):
      print(f"  {lab}: train={int(tr.sum())} test={int(te.sum())}")