"""
Enhanced ML State Space — Quant-Grade Feature Engineering for RSI Snapback
==========================================================================
POLARS PORT of the original pandas research script.

WHY THIS PORT EXISTS
---------------------
polars' lazy, multithreaded, Rust-backed engine reads/aggregates large
parquet files (and does the resampling-style feature engineering below)
meaningfully faster than pandas. It does NOT speed up the walk-forward
loop itself (LightGBM fit + SHAP + sklearn permutation_importance across
N folds) — that's CPU-bound model fitting/explainability work on numpy
arrays regardless of which dataframe library built them. This port keeps
that section on numpy arrays at the LightGBM/SHAP/sklearn boundary, same
as the original, and only changes the surrounding plumbing.

WHAT ACTUALLY CHANGES BEHAVIOUR — READ THIS BEFORE TRUSTING THE OUTPUT
------------------------------------------------------------------------
polars has no index concept and several pandas idioms this script relied
on heavily do NOT have 1:1 equivalents. Every one of the following was a
real, silent behavioural mismatch caught by diffing this port's output
against the original pandas script on synthetic data (three datasets:
dense, sparse, and bursty/sparse-with-genuine-empty-bars) column-by-column
until every feature matched to float64 noise (~1e-9). If you change the
feature engineering later, re-run that kind of diff before trusting it —
these are exactly the class of bug this file's own original comments
describe having shipped before (research/live drift).

  1. `df.group_by_dynamic(...)` (polars' nearest `.resample()` equivalent)
     SILENTLY DROPS bins with zero rows; pandas' `.resample(freq).agg()`
     always returns one row per bin, NaN/0 for empty ones. Fixed via the
     `dense_resample()` helper below, which reinserts every bin over a
     regular grid and matches pandas' sum()-defaults-to-0 vs
     last()/mean()-defaults-to-null behaviour per aggregation.
  2. polars distinguishes float NaN from null — `drop_nulls()` does NOT
     remove NaN. Any numpy array with `np.nan` (e.g. from `_pad_front`)
     MUST be constructed with `nan_to_null=True` or it silently survives
     what would have been a pandas `.dropna()`.
  3. The final micro/macro join in the original
     (`df_micro_shifted.join(df_macro_shifted, how='left').ffill()`) does
     two different things at once: (a) broadcasts each bar's shifted
     features across every second until the next bar closes — replicated
     here with `join_asof(strategy="backward")` — AND (b) forward-fills
     any bar whose OWN value was genuinely missing (e.g. `volume` on a
     zero-trade bar, since pandas' `df_trades_1m` drops empty-trade bars
     entirely rather than zero-filling them) from the previous bar. (b) is
     a real quirk of the original pipeline, not a bug introduced here —
     replicated via an explicit `pl.all().forward_fill()` on the
     bar-level frame before the asof broadcast.
  4. VPIN bucket-close times must be BINNED into the BAR_FREQ grid
     (matching pandas' `.resample(BAR_FREQ).last()`, which assigns a
     bucket to whichever bar's WINDOW CONTAINS its close time) rather than
     asof-joined directly against each bar's left-edge timestamp — the
     latter introduces a systematic one-bar lag.
  5. `pandas.groupby(bucket).last()` (used in `deduplicate_to_bar`)
     replaces the row's timestamp with the BUCKET FLOOR, not the actual
     last observation's own timestamp. `polars.group_by(...).last()` keeps
     the real last timestamp unless you explicitly overwrite it.
  6. `calculate_zscore()` / other custom helpers in your real
     `utils/indicators.py` are almost certainly pandas-Series-shaped. This
     port reimplements a rolling z-score natively in polars
     (`_pl_rolling_zscore`) rather than calling your real function, since
     I don't have its source. VERIFY it matches your actual ddof/edge
     behaviour, or port your real function to accept a polars column.
     `get_wilders_rsi()` and `sma()` are untouched — they're called on
     plain numpy arrays (`np.diff(...)` output) in both versions, so they
     don't need to change.

WHAT THIS PORT DOES NOT VALIDATE
----------------------------------
I don't have your real `main/config.py`, `utils/indicators.py`, or
`utils/engine.py`, or your actual parquet data — I validated this against
a stub config + reimplemented indicators on synthetic orderbook/trade
data (see the parity-check notes above). Before trusting this for
anything live, run both the original and this port on the SAME real
session data and diff every column — the `dense_resample`/asof-join
patterns above are exactly the kind of thing that can look right and
still be subtly wrong on data shaped differently than what I tested.

------------------------------------------------------------------------
2026-08-16 PATCH NOTE (see chat): normalized_return had a long-only bias
baked into the triple-barrier labeling, not just in the R-multiple
mapping. See _sequential_triple_barrier's docstring and STEP 5 below for
the fix and why it had to happen in both places.
------------------------------------------------------------------------
"""

import sys
import os
import json
import datetime as _dt
import gc
import atexit
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from utils.indicators import get_wilders_rsi, sma
import warnings

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, classification_report
import shap
from sklearn.inspection import permutation_importance
from research.ml_state_builder import *
from research.ml_state_builder import _session_time_bounds

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress sklearn's missing feature names warning
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="X does not have valid feature names"
)

# 1. Limit thread allocation to keep per-core memory low
os.environ["POLARS_MAX_THREADS"] = "4"

# get_wilders_rsi/sma operate on plain numpy arrays in the original script
# (np.diff(...) output) — untouched by the pandas->polars switch.

from research.ml_shared_config import *

# ============================================================================
# STEP 3 — TRIPLE BARRIER LABELING (pure numpy — unchanged by the polars
# switch; sliding_window_view operates on plain numpy arrays either way)
# ============================================================================
def _infer_trade_side(df: pl.DataFrame) -> pl.Series:
    """Per-row trade side used to orient the triple barrier (+1 = long,
    -1 = short). MUST mirror the zone/direction logic in __main__ STEP 4
    exactly (ENABLE_RSI_ACCEL_FILTER two-sided zone, or the single global
    TRADE_DIRECTION fade/snapback flip) -- keep these two in sync if you
    ever change the hunting-zone rule.

    This runs BEFORE the hunting-zone filter, on every row (not just rows
    that end up traded), so rows outside any zone need *some* side value
    even though it's never used -- they get dropped from df_training_zone
    downstream regardless of what side they were assigned here.
    """
    rsi_high_thr = 100 - RSI_FILTER_THRESHOLD
    if ENABLE_RSI_ACCEL_FILTER:
        zone_side = (
            pl.when((pl.col(RSI_FILTER_COLUMN) < RSI_FILTER_THRESHOLD) & (pl.col("ma_accel_ewma3") > 0)).then(1)
              .when((pl.col(RSI_FILTER_COLUMN) > rsi_high_thr) & (pl.col("ma_accel_ewma3") < 0)).then(-1)
              .otherwise(1)  # non-zone rows: arbitrary, filtered out downstream
        )
        side_expr = zone_side * (-1 if TRADE_DIRECTION == 'short' else 1)
    else:
        # NOTE: pl.lit(x) alone does NOT broadcast to df's row count inside
        # select() -- it silently returns a 1-row Series regardless of len(df).
        # This was the source of the (0,)-shape crash inside
        # _sequential_triple_barrier: side_np came back as length 1, which
        # NumPy silently broadcasts against p0 in the first chunk, then
        # collapses to a genuinely empty array once c0 in the chunk loop
        # exceeds 1. pl.repeat(..., n=pl.len()) ties it to the frame's
        # actual height instead.
        side_expr = pl.repeat(-1 if TRADE_DIRECTION == 'short' else 1, n=pl.len())
    return df.select(side_expr.alias("_side"))["_side"]


def _sequential_triple_barrier(price, profit_thresh, stop_thresh, lookahead, side, chunk_size=BARRIER_CHUNK_SIZE):
    """side: +1 for a long-direction row, -1 for a short-direction row
    (see _infer_trade_side). Determines which raw price direction gets
    the RR_RATIO-scaled TARGET barrier and which gets the 1x STOP barrier.

    FIX (see chat): this previously always placed the RR_RATIO barrier
    above price ("upper") and the 1x barrier below ("lower"), regardless
    of side -- i.e. it silently assumed every row was a long. For a short
    row that put the real stop RR_RATIO x farther away than the real
    target, which inverts the intended win:loss R-multiple once
    TRADE_DIRECTION/side gets applied downstream (10R win / 1R loss on
    paper became 1R win / 10R loss in practice for short-side rows).
    WIN now consistently means "this row's own trade direction reached
    its target first", for both long and short rows.
    """
    price = np.asarray(price, dtype=float)
    profit_thresh = np.asarray(profit_thresh, dtype=float)
    stop_thresh = np.asarray(stop_thresh, dtype=float)
    side = np.asarray(side, dtype=float)
    n = len(price)

    label = np.full(n, LABEL_LOSS, dtype=int)
    exit_offset = np.full(n, -1, dtype=int)
    exit_ret = np.full(n, np.nan)
    terminal_ret = np.full(n, np.nan)
    is_timeout = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)

    usable_n = n - lookahead
    if usable_n <= 0:
        return label, exit_offset, exit_ret, terminal_ret, is_timeout, valid

    for c0 in range(0, usable_n, chunk_size):
        c1 = min(c0 + chunk_size, usable_n)
        p0 = price[c0:c1]
        pt = profit_thresh[c0:c1]
        st = stop_thresh[c0:c1]
        sd = side[c0:c1]
        is_long = sd >= 0

        # target = favourable-direction barrier, RR_RATIO x away.
        # stop   = adverse-direction barrier, 1x away.
        # Long rows: target above price, stop below (old, only, behaviour).
        # Short rows: target below price, stop above (mirror image).
        target = np.where(is_long, p0 * (1.0 + pt * RR_RATIO), p0 * (1.0 - pt * RR_RATIO))
        stop = np.where(is_long, p0 * (1.0 - st), p0 * (1.0 + st))

        window = sliding_window_view(price[c0 + 1: c1 + lookahead], lookahead)

        hit_target = np.where(is_long[:, None], window >= target[:, None], window <= target[:, None])
        hit_stop = np.where(is_long[:, None], window <= stop[:, None], window >= stop[:, None])

        has_target = hit_target.any(axis=1)
        has_stop = hit_stop.any(axis=1)
        first_target = np.argmax(hit_target, axis=1)
        first_stop = np.argmax(hit_stop, axis=1)

        win_mask = has_target & (~has_stop | (first_target < first_stop))
        loss_mask = has_stop & (~has_target | (first_stop < first_target))

        timeout_mask = ~(win_mask | loss_mask)
        # Timeout tie-break: was the final bar favourable for THIS row's
        # own direction (up for longs, down for shorts)?
        final_favorable = np.where(is_long, window[:, -1] > p0, window[:, -1] < p0)
        timeout_win_mask = timeout_mask & final_favorable
        timeout_loss_mask = timeout_mask & ~final_favorable

        thresholds_ok = ~np.isnan(pt) & ~np.isnan(st)

        lab = np.full(c1 - c0, LABEL_LOSS, dtype=int)
        lab[win_mask | timeout_win_mask] = LABEL_WIN
        lab[loss_mask | timeout_loss_mask] = LABEL_LOSS

        off = np.full(c1 - c0, lookahead - 1, dtype=int)
        off[win_mask] = first_target[win_mask]
        off[loss_mask] = first_stop[loss_mask]

        exit_price = window[np.arange(c1 - c0), off]

        label[c0:c1] = lab
        exit_offset[c0:c1] = off
        # exit_ret / terminal_ret stay RAW (unsigned, price-direction)
        # returns, exactly as before -- only the barrier PLACEMENT above
        # is now side-aware. Side is applied once, in STEP 5 below.
        exit_ret[c0:c1] = (exit_price - p0) / p0
        terminal_ret[c0:c1] = (window[:, -1] - p0) / p0
        is_timeout[c0:c1] = timeout_mask
        valid[c0:c1] = thresholds_ok

    return label, exit_offset, exit_ret, terminal_ret, is_timeout, valid


def deduplicate_to_bar(df: pl.DataFrame, bar_freq: str = BAR_FREQ) -> pl.DataFrame:
    """Keep the last row of each bar. See module docstring point 5:
    pandas' groupby(bucket).last() replaces the row's timestamp with the
    BUCKET FLOOR, not the raw last-observation timestamp — replicated here
    by overwriting 'datetime' with the bucket value after aggregating."""
    return (
        df.with_columns(pl.col("datetime").dt.truncate(bar_freq).alias("_bucket"))
          .sort("datetime")
          .group_by("_bucket", maintain_order=True)
          .last()
          .with_columns(pl.col("_bucket").alias("datetime"))
          .drop("_bucket")
          .sort("datetime")
    )


def add_barrier_labels(df: pl.DataFrame, lookahead_seconds: int = LOOKAHEAD_SECONDS,
                        barrier_mult: float = BARRIER_MULTIPLIER) -> pl.DataFrame:
    parts = []
    for _, g in df.sort("datetime").group_by("session_id", maintain_order=True):
        # Bootstrap floor — matches PositionTracker.get_barrier_pct()'s
        # `fallback = 0.001 * BARRIER_MULTIPLIER` branch.
        g = g.with_columns(
            pl.when(pl.col("bar_vol_std") > 0).then(pl.col("bar_vol_std")).otherwise(0.001).alias("_bvs")
        )
        g = g.with_columns([
            (pl.col("_bvs") * barrier_mult).alias("profit_barrier"),
            (pl.col("_bvs") * barrier_mult).alias("stop_barrier"),
        ]).drop("_bvs")

        # FIX (see chat): side must be known BEFORE labeling, so the
        # RR_RATIO barrier can be placed on the correct side of price for
        # short-direction rows too. See _infer_trade_side / STEP 4.
        side_np = _infer_trade_side(g).to_numpy()
        assert len(side_np) == len(g), (
            f"_infer_trade_side returned {len(side_np)} rows for a "
            f"{len(g)}-row group (session_id={g['session_id'][0]!r}) -- "
            "likely a pl.lit() broadcasting bug, see _infer_trade_side."
        )

        label, exit_offset, exit_ret, terminal_ret, is_timeout, valid = _sequential_triple_barrier(
            g["execution_price"].to_numpy(), g["profit_barrier"].to_numpy(),
            g["stop_barrier"].to_numpy(), lookahead_seconds, side_np
        )

        g = g.with_columns([
            pl.Series("target", label.astype("int32")),
            pl.Series("bars_to_exit", exit_offset),
            pl.Series("exit_ret", exit_ret, nan_to_null=True),
            pl.Series("terminal_ret", terminal_ret, nan_to_null=True),
            pl.Series("is_timeout", is_timeout),
            pl.Series("_valid", valid),
        ])
        g = g.filter(pl.col("_valid")).drop("_valid")
        parts.append(g)

    return pl.concat(parts)


# ============================================================================
# purged_chronological_splits — numpy datetime64/timedelta64 in place of
# pandas Timestamp/Timedelta + Index. Positional (numpy) indexing throughout
# instead of .loc/.iloc.
# ============================================================================
def purged_chronological_splits(datetimes: np.ndarray, n_samples: int, block_size: int,
                                 horizon_seconds: int, embargo_seconds: int = 0,
                                 allow_overlap: bool = True):
    n = len(datetimes)
    horizon = np.timedelta64(horizon_seconds, "s")
    embargo = np.timedelta64(embargo_seconds, "s")
    min_initial_train = int(n * 0.35)
    available_test_run = n - min_initial_train - block_size

    if available_test_run <= 0:
        raise ValueError("The dataset length is too narrow for this test configuration block size.")

    max_non_overlapping = max(1, available_test_run // block_size)

    if n_samples > max_non_overlapping and not allow_overlap:
        print(f"[FOLD COUNT ADJUSTED] Requested n_samples={n_samples}, but only "
              f"{max_non_overlapping} non-overlapping {block_size}-bar test windows fit in "
              f"the {available_test_run} bars available after the {min_initial_train}-bar "
              f"initial training block. Reducing n_samples to {max_non_overlapping} so every "
              f"fold tests an independent period. Pass allow_overlap=True to keep the "
              f"original count with overlapping (non-independent) folds instead.")
        n_samples = max_non_overlapping

    step_intervals = available_test_run // n_samples

    if allow_overlap and step_intervals < block_size:
        overlap_pct = 100 * (1 - step_intervals / block_size)
        print(f"[FOLD OVERLAP WARNING] step_intervals={step_intervals} < block_size={block_size} "
              f"-> consecutive test folds overlap by ~{overlap_pct:.0f}% of their length. "
              f"Trade counts/expectancy pooled across folds are NOT independent draws.")

    for fold in range(n_samples):
        start_idx = min_initial_train + (fold * step_intervals)
        test_idx = np.arange(start_idx, start_idx + block_size)
        test_start_time = datetimes[test_idx[0]]

        possible_train_idx = np.arange(0, start_idx)
        train_times = datetimes[possible_train_idx]

        purge_mask = (train_times + horizon + embargo >= test_start_time)
        train_idx = possible_train_idx[~purge_mask]

        n_purged = len(possible_train_idx) - len(train_idx)
        yield train_idx, test_idx, n_purged


def find_debounced_signals(y_proba: np.ndarray, threshold: float, hysteresis: float = HYSTERESIS_BUFFER) -> np.ndarray:
    """Return indices where probability meets or exceeds threshold.

    NOTE: hysteresis is accepted for backward compatibility but NOT used here.
    A simple threshold filter is used instead of a stateful crossing detector.
    The lockout filter (applied separately in the walk-forward loop) handles
    trade spacing, making the stateful debouncer redundant. This avoids the
    counterintuitive non-monotonic trade counts that the old stateful
    debouncer produced (high thresholds re-armed more often than low ones,
    yielding more signals for stricter thresholds).
    """
    return np.where(y_proba >= threshold)[0]


# ============================================================================
# HISTORICAL STATE-SPACE ORCHESTRATION
# ============================================================================
def build_historical_state_space(sessions: list[tuple[str, str, int]] | None = None,
                                 chunk_sessions: bool = True,
                                 chunks_per_session: int = 6,
                                 state_space_dir: str = STATE_SPACE_DIR) -> pl.DataFrame:
    """Build raw historical state space independently from labeling/modeling.

    The returned frame is the exact input to barrier labeling and walk-forward
    evaluation. Keeping it separate gives the backtest and live-parity checks
    one concrete, testable feature boundary.

    State-space partitions are persisted under `state_space_dir` (default:
    ``research/state_space/``) so they never land in the repo root / CWD.
    """
    os.makedirs(state_space_dir, exist_ok=True)
    sessions = sessions if sessions is not None else get_session_paths()
    print(f"Data source: DATA_FORMAT={DATA_FORMAT!r}, DATA_DIR={DATA_DIR!r}")
    if chunk_sessions:
        print(f"CHUNK_SESSIONS=True: computing features continuously per session, "
              f"then writing {chunks_per_session} state-space partitions")
    print(f"Found {len(sessions)} session(s): {[sid for _, _, sid in sessions]}")

    chunk_files = []
    for orderbook_path, trades_path, session_id in sessions:
        print(f"\n=== Session {session_id}: {orderbook_path}, {trades_path} ===")
        session_features = build_session_feature_context(orderbook_path, trades_path)
        print(f"Session {session_id}: precomputed {session_features.rsi_features.height} continuous RSI bars")

        session_start, session_end = _session_time_bounds(trades_path)

        if not chunk_sessions:
            # Eagerly load the whole session (will spike RAM, but expected if
            # chunk_sessions=False).
            session_df = load_and_build_state_space(
                orderbook_path,
                trades_path,
                session_id=session_id,
                session_features=session_features,
            )
            fname = os.path.join(state_space_dir, f"state_space_session{session_id}.parquet")
            session_df.write_parquet(fname)
            chunk_files.append(fname)
            print(f"Session {session_id}: {session_df.height} rows -> {fname}")
            del session_df
            gc.collect()
            continue

        # ---- Chunked computation + persistence path ----
        # IMPORTANT: pass the time boundaries into load_and_build_state_space
        # so the raw orderbook/trades are windowed BEFORE any eager materialization.
        # Each chunk includes CHUNK_BUFFER_SECONDS of warm-up history, then trims
        # back to [start_dt, end_dt) before returning.
        chunk_duration = (session_end - session_start) / chunks_per_session

        print(f"Session {session_id} time range: {session_start} to {session_end} "
              f"-> {chunks_per_session} chunks (all sharing session_id={session_id})")

        for i in range(chunks_per_session):
            start_dt = session_start + (chunk_duration * i)
            end_dt = session_start + (chunk_duration * (i + 1))

            chunk_df = load_and_build_state_space(
                orderbook_path,
                trades_path,
                session_id=session_id,
                start_dt=start_dt,
                end_dt=end_dt,
                session_features=session_features,
            )

            fname = os.path.join(state_space_dir, f"state_space_session{session_id}_chunk{i}.parquet")
            # chunk_df is already bounded; sink the final chunk to SSD.
            chunk_df.lazy().sink_parquet(fname)
            chunk_files.append(fname)

            print(f"  chunk {i+1} -> {fname}. Wiping memory...")
            del chunk_df
            gc.collect()

    print("\nAll sessions processed successfully! Loading final compiled dataset...")

    # Explicit sort by datetime after concatenating sessions -- matches
    # ml_state_optimized_csv.py's pd.concat([...]).sort_index() exactly.
    df_state_space = pl.concat([pl.scan_parquet(f) for f in chunk_files]).sort("datetime").collect()
    print(f"Combined state space: {df_state_space.height} rows across {len(sessions)} session(s)")
    return df_state_space

def _session_expr() -> pl.Expr:
    minute_of_day = (
        pl.col("datetime").dt.hour().cast(pl.Int32) * 60
        + pl.col("datetime").dt.minute().cast(pl.Int32)
    )
    expr = pl.when(minute_of_day < SESSION_BOUNDS[0][2]).then(pl.lit(SESSION_BOUNDS[0][0]))
    for label, _, hi in SESSION_BOUNDS[1:]:
        expr = expr.when(minute_of_day < hi).then(pl.lit(label))
    return expr.otherwise(pl.lit(SESSION_BOUNDS[-1][0]))
 
def _pwe(ret: pl.Series) -> float:
    """Win-rate-weighted expectancy in R. See module docstring point 2."""
    r = ret.to_numpy()
    if len(r) == 0:
        return float("nan")
    wins, losses = r[r > 0], r[r < 0]
    win_rate = len(wins) / len(r)
    loss_rate = len(losses) / len(r)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0  # negative
    return float(win_rate * avg_win + loss_rate * avg_loss)
 
 
def _sortino(ret: pl.Series, mar: float = 0.0) -> float:
    """Raw (non-annualized) per-trade Sortino. See module docstring point 3."""
    r = ret.to_numpy()
    if len(r) == 0:
        return float("nan")
    downside = np.minimum(0.0, r - mar)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev == 0:
        return float("nan")
    return float((r.mean() - mar) / downside_dev)
 
 
def session_regime_breakdown(trade_df: pl.DataFrame,
                              threshold: float | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    trade_df: must have `datetime` (Datetime) and `ret` (float) columns --
        e.g. master_trade_df from STEP 10 of the main script.
    threshold: if trade_df has a `threshold` column (master_trade_df pools
        every tracked threshold together) and you only want one production
        threshold's trades, pass it here to filter first. If trade_df is
        already filtered to a single threshold, leave this None.
 
    Returns (stats_df, count_df):
        stats_df  -- rows: Weekend (PWE), Weekday (PWE), Weekend (Sortino),
                     Weekday (Sortino); columns: Regime/Hour + one per session.
        count_df  -- rows: Weekend, Weekday; columns: Regime/Hour + one per session.
    """
    df = trade_df
    if threshold is not None and "threshold" in df.columns:
        df = df.filter(pl.col("threshold") == threshold)
 
    df = df.with_columns([
        _session_expr().alias("session"),
        pl.col("datetime").dt.weekday().alias("_iso_weekday"),  # polars: Mon=1..Sun=7
    ]).with_columns(
        pl.when(pl.col("_iso_weekday") >= 6).then(pl.lit("Weekend"))
          .otherwise(pl.lit("Weekday")).alias("regime")
    ).drop("_iso_weekday")
 
    pwe_rows, sortino_rows, count_rows = [], [], []
    for regime in ("Weekend", "Weekday"):
        pwe_row = {"Regime/Hour": f"{regime} (PWE)"}
        sortino_row = {"Regime/Hour": f"{regime} (Sortino)"}
        count_row = {"Regime/Hour": regime}
        for session in SESSION_ORDER:
            bucket = df.filter((pl.col("regime") == regime) & (pl.col("session") == session))
            rets = bucket["ret"]
            pwe_row[session] = _pwe(rets)
            sortino_row[session] = _sortino(rets)
            count_row[session] = bucket.height
        pwe_rows.append(pwe_row)
        sortino_rows.append(sortino_row)
        count_rows.append(count_row)
 
    # Screenshot order: Weekend(PWE), Weekday(PWE), Weekend(Sortino), Weekday(Sortino)
    stats_df = pl.DataFrame([pwe_rows[0], pwe_rows[1], sortino_rows[0], sortino_rows[1]])
    count_df = pl.DataFrame(count_rows)
    return stats_df, count_df
 
 
def print_regime_report(stats_df: pl.DataFrame, count_df: pl.DataFrame) -> None:
    label_w, col_w = 22, 24
 
    def _fmt(v, is_count: bool) -> str:
        if is_count:
            return str(v)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.2f}"
 
    def _print_table(df: pl.DataFrame, title: str, is_count: bool) -> None:
        print(f"\n{title}")
        header = f"{'Regime/Hour':<{label_w}}" + "".join(f"{c:<{col_w}}" for c in SESSION_ORDER)
        print(header)
        print("-" * len(header))
        for row in df.iter_rows(named=True):
            line = f"{row['Regime/Hour']:<{label_w}}"
            for c in SESSION_ORDER:
                line += f"{_fmt(row[c], is_count):<{col_w}}"
            print(line)
 
    _print_table(stats_df, "Production-Grade Live Execution (Statistic)", is_count=False)
    _print_table(count_df, "Production-Grade Live Execution (Trade Count)", is_count=True)

from polars.testing import assert_frame_equal

def test_chunking_loss(
    df_chunked: pl.DataFrame,
    df_unchunked: pl.DataFrame,
    key: str = "datetime",
    float_tolerance: float = 1e-9,
    max_row_diff: int = 5,            # absolute number of missing/extra rows to tolerate
    max_row_diff_pct: float = 0.001,  # ...or as a fraction of total rows, whichever is looser
) -> bool:
    """
    Tests if the state space composed from chunked processing matches unchunked
    processing. Tolerates a small number of missing/extra rows (e.g. chunk-boundary
    clipping) and, when values differ, reports every offending column instead of
    stopping at the first one.

    Returns:
        bool: True if lossless (or within tolerance), False if lossy.
    """
    print("Beginning lossy validation check...")

    # 1. Row count check, with tolerance
    n_chunked, n_unchunked = df_chunked.height, df_unchunked.height
    row_diff = abs(n_chunked - n_unchunked)
    row_diff_pct = row_diff / max(n_chunked, n_unchunked)

    if row_diff > max_row_diff and row_diff_pct > max_row_diff_pct:
        print(f"[FAIL] Row count mismatch too large: Chunked ({n_chunked}) vs "
              f"Unchunked ({n_unchunked}), diff={row_diff}")
        return False
    elif row_diff > 0:
        print(f"[WARN] Row count differs by {row_diff} "
              f"({row_diff_pct:.4%}) — within tolerance, checking overlap...")

    # 2. Schema check
    if df_chunked.schema != df_unchunked.schema:
        print("[FAIL] Schema mismatch between dataframes.")
        print(f"  chunked only: {set(df_chunked.schema) - set(df_unchunked.schema)}")
        print(f"  unchunked only: {set(df_unchunked.schema) - set(df_chunked.schema)}")
        return False

    # 3. Identify which keys are missing on either side
    chunked_keys = set(df_chunked[key])
    unchunked_keys = set(df_unchunked[key])
    missing_from_chunked = unchunked_keys - chunked_keys
    missing_from_unchunked = chunked_keys - unchunked_keys

    if missing_from_chunked:
        print(f"[INFO] {len(missing_from_chunked)} row(s) present in unchunked "
              f"but missing from chunked, e.g. {sorted(missing_from_chunked)[:5]}")
    if missing_from_unchunked:
        print(f"[INFO] {len(missing_from_unchunked)} row(s) present in chunked "
              f"but missing from unchunked, e.g. {sorted(missing_from_unchunked)[:5]}")

    # 4. Align by key (not position) over the overlap only
    overlap_keys = chunked_keys & unchunked_keys
    df_c = df_chunked.filter(pl.col(key).is_in(overlap_keys)).sort(key)
    df_u = df_unchunked.filter(pl.col(key).is_in(overlap_keys)).sort(key)

    print(f"Comparing {df_c.height} overlapping rows with float tolerance {float_tolerance}...")

    # 5. Check every column, don't stop at the first failure
    bad_cols = {}
    for col in df_c.columns:
        if col == key:
            continue
        try:
            assert_frame_equal(
                df_c.select(col), df_u.select(col),
                check_exact=False,
                rel_tol=float_tolerance,
                abs_tol=float_tolerance,
            )
        except AssertionError:
            if df_c[col].dtype.is_numeric():
                diff = (df_c[col] - df_u[col]).abs()
                bad_cols[col] = {
                    "max_abs_diff": diff.max(),
                    "n_mismatched": (diff > float_tolerance).sum(),
                    "first_bad_idx": diff.arg_max(),
                }
            else:
                mismatched = (df_c[col] != df_u[col])
                bad_cols[col] = {
                    "n_mismatched": mismatched.sum(),
                    "first_bad_idx": mismatched.arg_max(),
                }

    if bad_cols:
        print("\n[FAIL] Value mismatches found:")
        for col, info in bad_cols.items():
            parts = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in info.items())
            print(f"  {col}: {parts}")
        return False

    if missing_from_chunked or missing_from_unchunked:
        print(f"[SUCCESS*] Overlapping data matches; {row_diff} row(s) differ in "
              f"presence, within tolerance ({max_row_diff} rows / {max_row_diff_pct:.3%}).")
    else:
        print("[SUCCESS] Data is perfectly lossless. Chunked matches unchunked.")
    return True

# ============================================================================
# REPORT LOGGING — mirror everything printed to the console into a single
# Markdown file, so an AI (or a human) can read the full run later without
# re-executing it. This is a stdout tee: it does NOT touch any of the
# print() calls above/below — it just duplicates whatever they write.
# ============================================================================
class _TeeToMarkdown:
    """
    Wraps the real stdout. Every .write() call is sent to the console AND
    appended to a Markdown file, inside a single fenced code block so the
    aligned tables / dataframe reprs / progress lines stay readable as-is.
    """
    def __init__(self, filepath: str, console_stream):
        self.console = console_stream
        self._closed = False
        self.filepath = filepath
        self.md_file = open(filepath, "w", encoding="utf-8")
        self.md_file.write("# Walk-Forward Backtest Report\n\n")
        self.md_file.write(f"_Generated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}_\n\n")
        self.md_file.write(f"_Script: `{os.path.basename(__file__)}`_\n\n")
        self.md_file.write("```text\n")

    def write(self, text):
        self.console.write(text)
        self.md_file.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.md_file.flush()

    def isatty(self):
        return False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.md_file.write("\n```\n")
        self.md_file.close()


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _start_report_capture(filepath: str = None):
    """Redirect sys.stdout through _TeeToMarkdown. Call once, at the top of
    the run. Cleanup is registered via atexit so the file is always closed
    and flushed to disk — including on an uncaught exception or Ctrl-C.

    Defaults to <project>/build/backtest_report_v3.md — the shared build
    folder that holds every generated report artifact — NOT the current
    working directory, so it lands in one predictable place no matter where
    you launch the run from. Override with BACKTEST_REPORT_PATH."""
    if filepath is None:
        filepath = os.environ.get(
            "BACKTEST_REPORT_PATH",
            os.path.join(BUILD_DIR, "backtest_report_v3.md"),
        )
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tee = _TeeToMarkdown(filepath, sys.stdout)
    sys.stdout = tee

    def _cleanup():
        sys.stdout = tee.console
        tee.close()
        print(f"[report] Full run report written to: {os.path.abspath(filepath)}")

    atexit.register(_cleanup)
    return tee


# ============================================================================
# MAIN: historical state space -> labels -> walk-forward evaluation
# ============================================================================
if __name__ == "__main__":
    # Everything printed from here on (including polars' own DataFrame
    # reprs, the walk-forward progress lines, the feature-importance table,
    # and the final summary) is mirrored into REPORT.md as well as stdout.
    _start_report_capture()

    # Chunking partitions the finished state space for downstream storage or
    # transport. Feature calculation itself remains session-continuous.
    df_state_space = []
    if DATA_FORMAT == "csv":
        df_baseline = build_historical_state_space(chunk_sessions=False)
    
        df_state_space = build_historical_state_space(
            chunk_sessions=True,
            chunks_per_session=N_CHUNKS_PER_SESSION,
        )
        is_lossless = test_chunking_loss(df_state_space, df_baseline)
    
        if not is_lossless:
            print("WARNING: Chunking logic is leaking or altering data.")
        else:
            print("Proceeding with walk-forward evaluation on validated data...")
    else:
        df_state_space = build_historical_state_space(
            chunk_sessions=CHUNK_SESSIONS,
            chunks_per_session=N_CHUNKS_PER_SESSION,
        )
    # ==========================================
    # ADD THESE LINES TO INSPECT YOUR TABLE
    # ==========================================
    
    # 1. Force Polars to show ALL columns instead of truncating them
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(10) # Show 10 rows

    print("\n--- DEBUG: hour distribution right after state-space build ---")
    print(df_state_space["datetime"].dt.hour().value_counts().sort("datetime"))
    
    print("\n--- STATE SPACE PREVIEW (FIRST 5 ROWS) ---")
    print(df_state_space.head(5))
    
    print("\n--- STATE SPACE SUMMARY STATISTICS ---")
    # describe() is great for spotting NaN issues or extreme z-scores
    print(df_state_space.describe())
    
    print("\n--- STATE SPACE GLIMPSE ---")
    # glimpse() flips the table sideways, making it very easy to read wide ML datasets
    print(df_state_space.glimpse())
    
    # Optional: Reset config back to default so later prints aren't massive
    pl.Config.set_tbl_cols(8)
    
    # ... [PROCEED WITH YOUR BARRIER LABELING AND LIGHTGBM CODE] ...

    # ------------------------------------------------------------------
    # STEP 3 — Barrier labeling
    # ------------------------------------------------------------------
    df_state_space = add_barrier_labels(df_state_space)
    df_state_space = df_state_space.filter(pl.col("volume") > 0)

    print(f"\nState space before dedup: {df_state_space.height} rows")
    df_state_space = deduplicate_to_bar(df_state_space)
    print(f"State space after dedup (bar-level): {df_state_space.height} rows")

    label_counts = (
        df_state_space.group_by("target").len()
        .with_columns(pl.col("target").replace_strict(CLASS_NAMES).alias("label"))
    )
    label_counts_dict = dict(zip(label_counts["label"].to_list(), label_counts["len"].to_list()))
    print(f"Label distribution (full dataset): {label_counts_dict}")

    df_training_zone = df_state_space.filter(pl.col(RSI_FILTER_COLUMN) < RSI_FILTER_THRESHOLD)
    # --------------------------------------------------------------------
    # Two-sided directional hunting zone (layered on top of the model):
    #   LONG setup : RSI(2) < 30            AND ma_accel_EWA3 > 0
    #   SHORT setup: RSI(2) > 70 (=100-30)  AND ma_accel_EWA3 < 0
    # Once the zone is resolved, TRADE_DIRECTION inverts the whole rule:
    #   'short' flips a LONG setup to an actual SHORT and a SHORT setup to a
    #   LONG (mirrors the executor's fade/snapback inversion).
    #
    # NOTE: this exact rule is duplicated in _infer_trade_side() above,
    # which now runs BEFORE barrier labeling so the triple barrier itself
    # can be side-aware. If you change this block, update that one too.
    rsi_high_thr = 100 - RSI_FILTER_THRESHOLD  # 70
    if ENABLE_RSI_ACCEL_FILTER:
        # NEW: two-sided directional hunting zone (RSI + ma_accel_EWA3), direction
        # per zone; TRADE_DIRECTION inverts the whole rule.
        df_training_zone = (
            df_state_space
            .with_columns(
                pl.when((pl.col(RSI_FILTER_COLUMN) < RSI_FILTER_THRESHOLD) & (pl.col("ma_accel_ewma3") > 0)).then(1)
                  .when((pl.col(RSI_FILTER_COLUMN) > rsi_high_thr) & (pl.col("ma_accel_ewma3") < 0)).then(-1)
                  .otherwise(0).alias("zone_side")
            )
            .filter(pl.col("zone_side") != 0)
            .with_columns(
                (pl.col("zone_side") * (-1 if TRADE_DIRECTION == 'short' else 1)).alias("traded_side")
            )
            .drop("zone_side")
        )
    else:
        # ORIGINAL (very first) behaviour: single-zone RSI(2)<30 filter only —
        # no ma_accel, no RSI>70 short zone. traded_side encodes the GLOBAL
        # direction (+1 = long/snapback, -1 = fade) per TRADE_DIRECTION, and the
        # walk-forward gate stays on the unconditional P(WIN)>=threshold.
        df_training_zone = (
            df_state_space
            .filter(pl.col(RSI_FILTER_COLUMN) < RSI_FILTER_THRESHOLD)
            .with_columns(
                pl.lit(-1 if TRADE_DIRECTION == 'short' else 1).alias("traded_side")
            )
        )

    print("\n--- DEBUG: hour distribution after RSI hunting-zone filter ---")
    print(df_training_zone["datetime"].dt.hour().value_counts().sort("datetime"))
    print(f"Total rows in hunting zone: {df_training_zone.height}")
    hz_counts = (
        df_training_zone.group_by("target").len()
        .with_columns(pl.col("target").replace_strict(CLASS_NAMES).alias("label"))
    )
    print(f"Label distribution (hunting zone): "
          f"{dict(zip(hz_counts['label'].to_list(), hz_counts['len'].to_list()))}")

    # ------------------------------------------------------------------
    # STEP 4 — Features / target configuration
    # ------------------------------------------------------------------
    if USE_ORDERFLOW_FEATURES:
        features = list(FEATURE_COLUMNS) + [
            c for c in ORDERFLOW_FEATURE_COLUMNS if c in df_training_zone.columns
        ]
    else:
        features = list(FEATURE_COLUMNS)
    missing = [c for c in FEATURE_COLUMNS if c not in df_training_zone.columns]
    if missing:
        raise ValueError(f"FEATURE_COLUMNS from config.py missing from pipeline output: {missing}")

    # Numpy at the LightGBM/SHAP/sklearn boundary from here on — this
    # matches the original's own usage pattern (X/y were already just
    # feeding numpy-based estimators) and is where any dataframe library's
    # overhead stops mattering anyway.
    datetimes = df_training_zone["datetime"].to_numpy()
    X = df_training_zone.select(features).to_numpy()
    y = df_training_zone["target"].to_numpy()
    side_arr = df_training_zone["traded_side"].to_numpy()
    stop_barrier_arr = df_training_zone["stop_barrier"].to_numpy()
    bars_to_exit_arr = df_training_zone["bars_to_exit"].to_numpy()
    is_timeout_arr = df_training_zone["is_timeout"].to_numpy()
    exit_ret_arr = df_training_zone["exit_ret"].to_numpy()

    print(f"\nFeatures ({len(features)} total):")
    for f in features:
        print(f"  - {f}")

    # ------------------------------------------------------------------
    # STEP 5 — Risk-Normalized Return
    # ------------------------------------------------------------------
    # FIX (see chat): normalized_return is now computed to ALREADY be
    # correct from the trade's own point of view (positive = good for
    # THIS row's side), for all three branches:
    #   - WIN / LOSS (non-timeout): hardcoded +RR_RATIO / -1.0 as before.
    #     This is now correct as-is because add_barrier_labels/
    #     _sequential_triple_barrier place the target/stop barriers on
    #     the correct side of price per row (see fix above) -- WIN means
    #     "this row's own direction won", not "price went up".
    #   - TIMEOUT: exit_ret_arr is still a RAW (unsigned, price-direction)
    #     return out of the barrier function, so it needs side_arr applied
    #     HERE to convert it into the trade's own R, unlike before.
    # Because normalized_return is now fully side-adjusted at this point,
    # do NOT multiply by side_arr again downstream (see STEP 6 fix) --
    # doing so would double-apply the direction and re-invert short trades.
    is_win = y == LABEL_WIN
    is_loss = y == LABEL_LOSS

    normalized_return = np.full(len(y), np.nan)
    normalized_return[is_win & ~is_timeout_arr] = RR_RATIO
    normalized_return[is_loss & ~is_timeout_arr] = -1.0
    normalized_return[is_timeout_arr] = (
        (exit_ret_arr[is_timeout_arr] * side_arr[is_timeout_arr])
        / (stop_barrier_arr[is_timeout_arr] + EPS)
    )

    fee_in_r = (2.0 * FEE_BPS / 10000.0) / (stop_barrier_arr + EPS)
    normalized_return = normalized_return - fee_in_r

    upper_quantile = np.nanquantile(normalized_return, 0.99)
    lower_quantile = np.nanquantile(normalized_return, 0.01)
    cap = max(abs(upper_quantile), abs(lower_quantile), RR_RATIO)
    normalized_return = np.clip(normalized_return, -cap, cap)

    mean_stop_barrier = np.nanmean(stop_barrier_arr)
    print("\n" + "=" * 50)
    print("STOP BARRIER METRICS")
    print("=" * 50)
    print(f"Average Stop Barrier Size (%): {mean_stop_barrier * 100:.4f}%")
    print("=" * 50)

    # ------------------------------------------------------------------
    # STEP 6 — Purged Chronological Walk-Forward
    # ------------------------------------------------------------------
    TARGET_TRADE_CLASS = LABEL_WIN
    INVERT_RETURNS = (TRADE_DIRECTION == 'short')
    target_name = CLASS_NAMES[TARGET_TRADE_CLASS]
    prob_label = f"P({target_name})"
    print(f"TRADE_DIRECTION={TRADE_DIRECTION!r} -> INVERT_RETURNS={INVERT_RETURNS} "
          f"(mirrors executor.py's actual fade/snapback semantics)")

    splitter = purged_chronological_splits(
        datetimes, n_samples=N_RANDOM_SAMPLES, block_size=TEST_BLOCK_SIZE,
        horizon_seconds=LOOKAHEAD_SECONDS, embargo_seconds=EMBARGO_SECONDS
    )

    print(f"\n--- Starting Walk-Forward (requested up to {N_RANDOM_SAMPLES} non-overlapping folds; "
          f"see [FOLD COUNT ADJUSTED] above if fewer independent windows fit the data) "
          f"| Targeting: {target_name} | Inverting Trades: {INVERT_RETURNS} ---")

    thresholds_to_track = np.arange(0.20, 0.96, 0.05)
    fold_metrics = {t: {'precisions': [], 'win_rates': [], 'rrs': [], 'trades': []} for t in thresholds_to_track}
    fold_importances = []
    trade_log = {t: [] for t in thresholds_to_track}
    fold_detail_records = []

    class_labels_ordered = [LABEL_LOSS, LABEL_WIN]
    class_metrics_by_label = {lbl: {'precision': [], 'recall': []} for lbl in class_labels_ordered}
    class_metrics_records = []
    all_y_test_pooled = []
    all_y_pred_pooled = []

    fold_shap_values = []
    fold_pfi_values = []

    for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            print(f"Fold {fold} | skipped — invalid training data distribution.")
            continue

        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=1, verbose=-1,
        )
        clf.fit(X_train, y_train)

        # --- SHAP --- (feature_names passed explicitly since X is a plain
        # numpy array here, not a named DataFrame)
        try:
            n_sample = min(1000, len(X_test))
            sample_idx = np.random.RandomState(42).choice(len(X_test), n_sample, replace=False)
            X_test_sampled = X_test[sample_idx]
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_test_sampled)

            if isinstance(shap_vals, list):
                shap_array = shap_vals[1]
            elif hasattr(shap_vals, 'values'):
                shap_array = shap_vals.values[:, :, 1] if len(shap_vals.values.shape) == 3 else shap_vals.values
            else:
                shap_array = shap_vals

            fold_shap_values.append(np.abs(shap_array).mean(axis=0))
        except Exception as e:
            print(f"Fold {fold:02d} | SHAP failed: {e}")
            fold_shap_values.append(np.full(len(features), np.nan))

        # --- PFI ---
        if len(np.unique(y_test)) < 2:
            fold_pfi_values.append(np.full(len(features), np.nan))
        else:
            try:
                pi_result = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=42, scoring='roc_auc', n_jobs=1
                )
                fold_pfi_values.append(pi_result.importances_mean)
            except Exception as e:
                print(f"Fold {fold:02d} | PFI failed: {e}")
                fold_pfi_values.append(np.full(len(features), np.nan))

        # --- Class metrics ---
        y_pred_argmax = clf.predict(X_test)
        prec_per_class = precision_score(y_test, y_pred_argmax, labels=class_labels_ordered, average=None, zero_division=0)
        rec_per_class = recall_score(y_test, y_pred_argmax, labels=class_labels_ordered, average=None, zero_division=0)

        for lbl, p, r in zip(class_labels_ordered, prec_per_class, rec_per_class):
            class_metrics_by_label[lbl]['precision'].append(p)
            class_metrics_by_label[lbl]['recall'].append(r)
            class_metrics_records.append({
                'fold': fold, 'class': CLASS_NAMES[lbl], 'precision': p, 'recall': r,
                'support': int((y_test == lbl).sum()),
            })

        all_y_test_pooled.append(y_test)
        all_y_pred_pooled.append(y_pred_argmax)

        if TARGET_TRADE_CLASS not in clf.classes_:
            print(f"Fold {fold} | skipped threshold scoring — no {target_name} samples in training data.")
            continue

        target_idx = list(clf.classes_).index(TARGET_TRADE_CLASS)
        y_proba = clf.predict_proba(X_test)[:, target_idx]

        # Per-test-bar traded side (+1 = LONG / -1 = SHORT; TRADE_DIRECTION
        # already folded into traded_side) and the per-side win probability
        # used as the entry gate: a LONG trades the model's P(up); a SHORT
        # trades P(down) = 1 - P(up). Threshold stays on per-side win prob.
        test_side = side_arr[test_idx]
        if ENABLE_RSI_ACCEL_FILTER:
            # NEW: per-side win probability — a LONG trades the model's P(up); a
            # SHORT trades P(down) = 1 - P(up).
            trade_conf = np.where(test_side > 0, y_proba, 1.0 - y_proba)
        else:
            # ORIGINAL: gate on the unconditional P(WIN) = P(up); direction and
            # inversion are already baked into traded_side (global fade/snapback).
            trade_conf = y_proba

        for t in thresholds_to_track:
            signal_indices = find_debounced_signals(trade_conf, t, HYSTERESIS_BUFFER)

            non_overlapping_idx = []          # positions within X_test/test_idx
            next_available_time = np.datetime64('1677-01-01')  # numpy datetime64 min-ish sentinel
            lockout_enabled = LOCKOUT_SECONDS > 0

            rejected_lockout = rejected_barrier = passed = 0
            for local_idx in signal_indices:
                global_idx = test_idx[local_idx]
                actual_global_time = datetimes[global_idx]
                if lockout_enabled and actual_global_time < next_available_time:
                    rejected_lockout += 1
                    continue
                row_barrier_pct = stop_barrier_arr[global_idx]
                if not (MIN_BARRIER_PCT <= row_barrier_pct <= MAX_BARRIER_PCT):
                    rejected_barrier += 1
                    continue
                passed += 1

                non_overlapping_idx.append(global_idx)

                if lockout_enabled:
                    trade_duration_seconds = int(bars_to_exit_arr[global_idx]) + 1
                    next_available_time = actual_global_time + np.timedelta64(trade_duration_seconds, 's')
            print(f"t={t:.2f} fold={fold}: raw_crossings={len(signal_indices)} "
            f"lockout_rejected={rejected_lockout} barrier_pct_rejected={rejected_barrier} passed={passed}")

            num_trades = len(non_overlapping_idx)

            if num_trades > 0:
                traded_idx = np.array(non_overlapping_idx)

                # FIX (see chat): normalized_return is already side-correct
                # (see STEP 5) -- do NOT multiply by side_arr again here.
                # The old `* side_arr[traded_idx]` double-applied direction
                # on top of the (now-fixed) barrier placement and re-flipped
                # short trades back to the broken 1R-win/RR_RATIO-loss shape.
                traded_returns = normalized_return[traded_idx].copy()

                trade_log[t].extend(zip(datetimes[traded_idx], traded_returns))

                strategy_wins = int((traded_returns > 0).sum())
                strategy_losses = int((traded_returns < 0).sum())
                win_rate = strategy_wins / num_trades

                win_returns = traded_returns[traded_returns > 0]
                loss_returns = traded_returns[traded_returns < 0]
                avg_win_ret = win_returns.mean() if len(win_returns) > 0 else np.nan
                avg_loss_ret = loss_returns.mean() if len(loss_returns) > 0 else np.nan
                if (not np.isnan(avg_win_ret) and not np.isnan(avg_loss_ret) and abs(avg_loss_ret) > 0.001):
                    rr = avg_win_ret / abs(avg_loss_ret)
                else:
                    rr = np.nan

                prec = win_rate  # strategy precision = fraction of winning (side-aware) trades

                fold_metrics[t]['precisions'].append(prec)
                fold_metrics[t]['win_rates'].append(win_rate)
                fold_metrics[t]['rrs'].append(rr)

                fold_detail_records.append({
                    'threshold': round(float(t), 2), 'fold': fold, 'trades': num_trades,
                    'strategy_wins': strategy_wins, 'strategy_losses': strategy_losses,
                    'precision': prec, 'win_rate': win_rate, 'rr': rr,
                    'avg_return': float(traded_returns.mean()),
                })

            fold_metrics[t]['trades'].append(num_trades)

        fold_importances.append(clf.feature_importances_)
        print(f"Fold {fold:02d} | Train Rows: {len(X_train):<6} | Tested on forward {TEST_BLOCK_SIZE} rows. Purged: {n_purged}")

    # ------------------------------------------------------------------
    # STEP 7 — SHAP vs PERMUTATION IMPORTANCE
    # ------------------------------------------------------------------
    print("\n--- Feature Importance: SHAP vs Permutation (PFI) ---")
    print("-" * 80)
    mean_shap = np.nanmean(fold_shap_values, axis=0)
    mean_pfi = np.nanmean(fold_pfi_values, axis=0)
    importance_df = pl.DataFrame({
        "Feature": features, "SHAP_Importance": mean_shap, "PFI_Mean_AUC_Drop": mean_pfi,
    }).sort("PFI_Mean_AUC_Drop", descending=True)
    print(importance_df)
    ensure_dir(BUILD_DIR)
    feature_importance_path = os.path.join(BUILD_DIR, "feature_importance_v3.csv")
    importance_df.write_csv(feature_importance_path)
    print(f"Feature importance saved to: {feature_importance_path}")

    # ------------------------------------------------------------------
    # STEP 8 — Model Class-Level Precision & Recall
    # ------------------------------------------------------------------
    print("\n--- Model Binary Precision & Recall (argmax prediction) — Mean ± Std Across Folds ---")
    print("-" * 70)
    print(f"{'Class':<10} | {'Precision':<20} | {'Recall'}")
    print("-" * 70)
    for lbl in class_labels_ordered:
        p_arr = np.array(class_metrics_by_label[lbl]['precision'])
        r_arr = np.array(class_metrics_by_label[lbl]['recall'])
        p_txt = f"{p_arr.mean():>6.2%} ± {p_arr.std():<6.2%}" if len(p_arr) else "n/a"
        r_txt = f"{r_arr.mean():>6.2%} ± {r_arr.std():<6.2%}" if len(r_arr) else "n/a"
        print(f"{CLASS_NAMES[lbl]:<10} | {p_txt:<20} | {r_txt}")
    print("-" * 70)

    if all_y_test_pooled:
        pooled_y_test = np.concatenate(all_y_test_pooled)
        pooled_y_pred = np.concatenate(all_y_pred_pooled)
        print("\n--- Pooled Out-of-Sample Classification Report ---")
        print(classification_report(
            pooled_y_test, pooled_y_pred, labels=class_labels_ordered,
            target_names=[CLASS_NAMES[l] for l in class_labels_ordered], zero_division=0
        ))

    # ------------------------------------------------------------------
    # STEP 9 — Strategy Rolling Expectancy & CVaR
    # ------------------------------------------------------------------
    print(f"\n--- Strategy Rolling Expectancy & Stability by {prob_label} Threshold ---")
    print("-" * 140)
    print(f"{'Thresh':<8} | {'Trades':<8} | {'Win%':<8} | {'RR':<6} | {'Overall Exp':<12} | {'Exp CVaR (5%)':<14} | {'Stable (CVaR>0)':<16} | {'Total R'}")
    print("-" * 140)


    for t in thresholds_to_track:
        if len(trade_log[t]) == 0:
            continue
        dts = np.array([x[0] for x in trade_log[t]])
        returns = np.array([x[1] for x in trade_log[t]], dtype=float)
        order = np.argsort(dts)
        returns = returns[order]
        num_trades = len(returns)

        win_rate = np.mean(returns > 0)
        avg_win = returns[returns > 0].mean() if np.sum(returns > 0) > 0 else 0
        avg_loss = returns[returns < 0].mean() if np.sum(returns < 0) > 0 else 0
        rr = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan

        overall_exp = np.mean(returns)
        total_r = np.sum(returns)

        if num_trades >= ROLLING_WINDOW:
            rolling_exp = pl.Series(returns).rolling_mean(window_size=ROLLING_WINDOW).drop_nulls().to_numpy()
            alpha = 0.05
            cutoff_idx = max(1, int(len(rolling_exp) * alpha))
            worst_rolling_streaks = np.sort(rolling_exp)[:cutoff_idx]
            cvar_exp = np.mean(worst_rolling_streaks)
            stable_flag = "YES" if cvar_exp > 0 else "NO"
        else:
            cvar_exp = np.nan
            stable_flag = "N/A (low trades)"

        print(f">= {t*100:>3.0f}% | {num_trades:<8} | {win_rate:>5.1%} | {rr:<6.2f} | {overall_exp:>10.2f} R | {cvar_exp:>12.2f} R | {stable_flag:<16} | {total_r:>+6.2f} R")

    print("-" * 140)
    print(f"Note: 'Exp CVaR (5%)' = avg R per trade during worst 5% of {ROLLING_WINDOW}-trade streaks.")

    
    # ------------------------------------------------------------------
    # STEP 10 — Out-of-Sample PnL Chart
    # ------------------------------------------------------------------
    import plotly.graph_objects as go

    print("\n--- Saving Trade Logs to CSV ---")
    all_trades_frames = []
    fig = go.Figure()

    for t in thresholds_to_track:
        if len(trade_log[t]) == 0:
            continue
        dts = np.array([x[0] for x in trade_log[t]])
        returns = np.array([x[1] for x in trade_log[t]], dtype=float)
        order = np.argsort(dts)
        dts, returns = dts[order], returns[order]

        log_df = pl.DataFrame({"datetime": dts, "ret": returns, "threshold": round(float(t), 2)})
        log_df = log_df.with_columns(pl.col("ret").cum_sum().alias("cum_pnl"))
        all_trades_frames.append(log_df)

        trade_numbers = list(range(1, len(log_df) + 1))
        fig.add_trace(go.Scatter(
            x=trade_numbers, y=log_df["cum_pnl"].to_list(), mode='lines', name=f'>= {t:.2f}',
            hovertemplate='Trade: %{x}<br>PnL: %{y:.2f}R<extra></extra>'
        ))

    if all_trades_frames:
        master_trade_df = pl.concat(all_trades_frames)
        ensure_dir(BUILD_DIR)
        trade_log_path = os.path.join(BUILD_DIR, "walk_forward_trade_log_v3.csv")
        master_trade_df.write_csv(trade_log_path)
        print("\n--- DEBUG: hour distribution in final trade log ---")
        print(master_trade_df["datetime"].dt.hour().value_counts().sort("datetime"))

        print(f"Full trade log saved to: {trade_log_path}")

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    chart_title = f'Out-of-Sample PnL (OLS Divergence + DeepLOB) — Target: {target_name} | Inverted: {INVERT_RETURNS}'
    fig.update_layout(
        title=chart_title, xaxis_title='Trade Number', yaxis_title='Cumulative Return (R)',
        legend_title=f'{prob_label} ≥ (Click!)', hovermode="x unified", template="plotly_white"
    )
    pnl_chart_path = os.path.join(BUILD_DIR, "pnl_walkforward_v3.html")
    fig.write_html(pnl_chart_path)
    print(f"Interactive chart saved to: {pnl_chart_path}")

    # ------------------------------------------------------------------
    # STEP 11 — Summary Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — OLS DIVERGENCE + DEEPLOB FEATURES")
    print("=" * 70)
    print(f"Sequential (order-aware) binary barrier labels, full dataset: {label_counts_dict}")
    print(f"Hunting zone ({RSI_FILTER_COLUMN} < {RSI_FILTER_THRESHOLD}) rows: {df_training_zone.height}")
    print(f"Trading Setup: Target {target_name} | Inverting Trade Direction: {INVERT_RETURNS}")
    print(f"Model: n_estimators=200, max_depth=3, learning_rate=0.01 (see clf definition above)")
    print(f"\nActual Features Used ({len(features)} total):")
    for f in features:
        print(f"  - {f}")

    for lbl in class_labels_ordered:
        p_arr = np.array(class_metrics_by_label[lbl]['precision'])
        if len(p_arr) > 0:
            print(f"  {CLASS_NAMES[lbl]:<8} class precision (argmax): {p_arr.mean():.2%} ± {p_arr.std():.2%} across folds")

    best_threshold, most_trades = None, -1
    for t in thresholds_to_track:
        n_trades = len(trade_log[t])
        if n_trades > most_trades:
            most_trades, best_threshold = n_trades, t

    if best_threshold is not None:
        print(f"Threshold with most out-of-sample trades: {most_trades} trades at {prob_label} >= {best_threshold:.2f}")
        production_threshold = round(float(best_threshold), 2)
        stats_df, count_df = session_regime_breakdown(master_trade_df, threshold=production_threshold)
        print_regime_report(stats_df, count_df)
        
    for t in thresholds_to_track:
        if len(trade_log[t]) > 0:
            returns = np.array([x[1] for x in trade_log[t]], dtype=float)
            print(f"  {prob_label} >= {t:.2f}: {len(returns)} trades, cumulative {returns.sum():.2f}R out-of-sample")

    print("\nReminder: this run ignores fees, slippage, latency, and partial fills.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 12 — Export Production Model
    # ------------------------------------------------------------------
    if EXPORT_PRODUCTION_MODEL:
        print("\n" + "=" * 70)
        print("EXPORTING PRODUCTION MODEL")
        print("=" * 70)

        final_clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
        final_clf.fit(X, y)

        model_dir = os.path.dirname(os.path.abspath(MODEL_PATH))
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        final_clf.booster_.save_model(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

        manifest_path = os.path.splitext(MODEL_PATH)[0] + ".features.json"
        label_counts_export = {CLASS_NAMES[k]: int((y == k).sum()) for k in np.unique(y)}
        manifest = {
            "features": features,
            "n_features": len(features),
            "includes_orderflow": USE_ORDERFLOW_FEATURES,
            "orderflow_features": [c for c in ORDERFLOW_FEATURE_COLUMNS if c in features],
            "base_features": list(FEATURE_COLUMNS),
            "trained_rows": int(len(X)),
            "trained_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "label_counts": label_counts_export,
            "rr_ratio": RR_RATIO,
            "lookahead_seconds": LOOKAHEAD_SECONDS,
            "barrier_multiplier": BARRIER_MULTIPLIER,
            "trade_direction": TRADE_DIRECTION,
            "confidence_threshold_note": (
                "This manifest does not set CONFIDENCE_THRESHOLD — that stays a "
                "config.py / CLI setting. See the per-threshold table above (now "
                "computed from non-overlapping folds) before trusting a threshold."
            ),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Feature manifest saved to: {manifest_path}")

        print(f"\nFeatures ({len(features)}), IN THIS EXACT ORDER — "
              f"LiveFeatureEngine.extract_feature_vector() must produce the same order:")
        for i, f in enumerate(features):
            tag = " [orderflow]" if f in ORDERFLOW_FEATURE_COLUMNS else ""
            print(f"  [{i}] {f}{tag}")

        print("\nNOTE: this export model is fit on 100% of available hunting-zone")
        print("data (no holdout) — it is NOT itself a forward-performance estimate.")
        print("Use the walk-forward table above for that; this step only produces")
        print("the artifact executor.py should load.")
        print("=" * 70)