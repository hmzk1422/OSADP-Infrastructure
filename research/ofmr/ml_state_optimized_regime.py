"""
ml_state_optimized_regime.py
=============================
SESSION-LEVEL REGIME-DETECTION EXTENSION on top of research/ofmr/ml_state_optimized.py.

WHY THIS IS A SEPARATE FILE, NOT AN EDIT TO ml_state_optimized.py
-------------------------------------------------------------------
Per your call: this imports research.ofmr.ml_state_optimized unmodified
(`import *` plus the handful of underscore-prefixed helpers it needs) and
only ADDS the new per-session study + regime-detection CSV around the
existing pipeline. ml_state_optimized.py itself is untouched — if this
script has a bug, your existing walk-forward numbers in
backtest_report_v3.md are unaffected.

WHAT THIS ADDS ON TOP OF THE ORIGINAL WALK-FORWARD LOOP
-----------------------------------------------------------
NOTE: granularity is per WALK-FORWARD FOLD, not per raw-data-file
session_id. session_id (0/1/2) is which parquet file a row loaded from —
at most a handful of values, fixed at load time. A "fold" is one
out-of-sample test window produced by purged_chronological_splits() —
up to N_RANDOM_SAMPLES of them (e.g. 400), each a distinct chronological
slice (different volatility regime, different time-of-day mix), which is
the axis that actually matters for regime detection. Everything below
that used to say "session" is keyed by fold number instead.

1. Per-fold performance study (STEP 11.5): for each fold, at the SAME
   "production threshold" the original STEP 11 already selects (the
   threshold with the most out-of-sample trades) — trade count, win rate,
   RR, profit-weighted expectancy (`_pwe`, reused as-is from
   ml_state_optimized), Sortino, and rolling-window Exp CVaR(5%) (identical
   formula to the existing global threshold table in STEP 9, just sliced to
   one fold's trades instead of all trades pooled).

2. Per-fold SHAP / PFI (STEP 6, inside the fold loop): the original already
   computes one SHAP vector and one PFI vector per fold (that's what feeds
   feature_importance_v3.csv, averaged across folds). No second rescoring
   pass is needed here — this file just captures that same per-fold vector
   (plus the fold's test_idx) into `fold_regime_data` instead of only
   averaging it away, so it survives into the per-fold CSV.

   "sum of SHAP" / "sum of PFI" per fold = that fold's own SHAP/PFI vector
   SUMMED across all features into one scalar — i.e. total feature
   importance mass the model placed on THIS fold's out-of-sample window, as
   a single regime-comparable number. This is MY assumption on what "sum"
   means here (you said "sum of PFI, sum of SHAP" without specifying sum
   over what) — if you meant something else (top-N features only, etc.)
   that's a one-line change where fold_sum_shap/fold_sum_pfi are built.

3. fold_regime_detection.csv (also STEP 11.5): one row per fold —
   n_hunting_zone_rows (= size of that fold's OOS test window), n_trades,
   win_rate, rr, pwe, sortino, cvar_exp_5pct, total_r, sum_shap, sum_pfi,
   then mean_<feature> for every feature in `features` (whatever
   FEATURE_COLUMNS + ORDERFLOW_FEATURE_COLUMNS resolves to — RSI,
   volatility, VPIN, whichever you have config'd), averaged over every
   hunting-zone row in that fold's test window (full regime picture, not
   just traded rows) — so you can correlate any feature's fold-average
   against fold-level performance / importance later (e.g. in a
   spreadsheet or a quick polars .corr()), up to N_RANDOM_SAMPLES rows.

WHAT THIS DOES NOT VALIDATE
------------------------------
Same caveat as both files this builds on: I don't have your real
ml_shared_config.py, feature_builder.py, indicators.py, or actual data.
This is written to the exact contract ml_state_optimized.py's __main__
already establishes (same df_training_zone / features / X / y / side_arr /
datetimes construction, copied verbatim from its STEP 1-6, 9, 10, 11, 12),
so it should slot in cleanly — but run it against real data and eyeball
fold_regime_detection.csv before trusting it (row count should match the
number of folds actually completed, i.e. what STEP 6's "[FOLD COUNT
ADJUSTED]" / per-fold print lines report — not necessarily the full
N_RANDOM_SAMPLES if fewer independent windows fit the data).
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import datetime as _dt
import numpy as np
import polars as pl

import lightgbm as lgb
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, classification_report

# Reuse the entire original pipeline unmodified — every public name
# (build_historical_state_space, add_barrier_labels, deduplicate_to_bar,
# purged_chronological_splits, find_debounced_signals,
# session_regime_breakdown, print_regime_report, test_chunking_loss, ...)
# plus everything ml_state_optimized.py itself pulled in transitively from
# ml_state_builder.py and ml_shared_config.py (BAR_FREQ, RR_RATIO, FEE_BPS,
# FEATURE_COLUMNS, BUILD_DIR, ROLLING_WINDOW, CLASS_NAMES, ensure_dir, ...).
from research.ofmr.ml_state_optimized import *
from research.ofmr.ml_state_optimized import _pwe, _sortino, _start_report_capture
from research.ml_shared_config import *


# ============================================================================
# NEW — per-fold CVaR. Identical formula to the global rolling-expectancy
# CVaR already in ml_state_optimized.py's STEP 9 (rolling mean of returns,
# worst alpha=5% of rolling windows, averaged) — just sliced to one fold's
# trades instead of all trades pooled across the strategy.
# ============================================================================
def _fold_cvar(returns: np.ndarray, window: int, alpha: float = 0.05) -> float:
    if len(returns) < window:
        return float("nan")
    rolling_exp = pl.Series(returns).rolling_mean(window_size=window).drop_nulls().to_numpy()
    cutoff_idx = max(1, int(len(rolling_exp) * alpha))
    worst = np.sort(rolling_exp)[:cutoff_idx]
    return float(np.mean(worst))


# ============================================================================
# NEW — per-fold performance study + regime-detection CSV
# ============================================================================
def build_fold_regime_report(
    df_training_zone: pl.DataFrame,
    fold_trades: dict,       # {fold: np.ndarray of R-returns, chronologically sorted}
    fold_regime_data: dict,  # {fold: {"test_idx": np.ndarray, "shap_vec": ..., "pfi_vec": ...}}
    fold_sum_shap: dict,     # {fold: float}
    fold_sum_pfi: dict,      # {fold: float}
    features: list,
    rolling_window: int,
) -> pl.DataFrame:
    """One row per walk-forward fold. Feature means are computed over EVERY
    hunting-zone row in that fold's out-of-sample test window (full regime
    picture, not just traded rows), so a fold with zero trades still gets a
    row with its feature averages (useful for spotting *why* a regime
    produced no trades at all)."""
    rows = []
    for fold in sorted(fold_regime_data.keys()):
        test_idx = fold_regime_data[fold]["test_idx"]
        zone_fold = df_training_zone[test_idx]
        returns = fold_trades.get(fold, np.array([], dtype=float))
        n_trades = len(returns)

        if n_trades > 0:
            win_rate = float(np.mean(returns > 0))
            wins, losses = returns[returns > 0], returns[returns < 0]
            avg_win = float(wins.mean()) if len(wins) else np.nan
            avg_loss = float(losses.mean()) if len(losses) else np.nan
            rr = (avg_win / abs(avg_loss)) if (len(wins) and len(losses) and abs(avg_loss) > 1e-9) else np.nan
            pwe = _pwe(pl.Series(returns))
            sortino = _sortino(pl.Series(returns))
            cvar_exp = _fold_cvar(returns, rolling_window)
            total_r = float(returns.sum())
        else:
            win_rate = rr = pwe = sortino = cvar_exp = total_r = float("nan")

        row = {
            "fold": fold,
            "n_hunting_zone_rows": zone_fold.height,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "rr": rr,
            "pwe": pwe,
            "sortino": sortino,
            "cvar_exp_5pct": cvar_exp,
            "total_r": total_r,
            "sum_shap": fold_sum_shap.get(fold, float("nan")),
            "sum_pfi": fold_sum_pfi.get(fold, float("nan")),
        }
        for feat in features:
            row[f"mean_{feat}"] = zone_fold[feat].mean()
        rows.append(row)

    return pl.DataFrame(rows)


def print_fold_regime_report(fold_df: pl.DataFrame) -> None:
    core_cols = ["fold", "n_hunting_zone_rows", "n_trades", "win_rate", "rr",
                 "pwe", "sortino", "cvar_exp_5pct", "total_r", "sum_shap", "sum_pfi"]
    print("\nPer-Fold Regime / Performance Study (core columns shown; per-feature means are in the CSV only)")
    print("-" * (18 * len(core_cols)))
    print("".join(f"{c:<18}" for c in core_cols))
    print("-" * (18 * len(core_cols)))
    for row in fold_df.select(core_cols).iter_rows(named=True):
        line = ""
        for c in core_cols:
            v = row[c]
            if isinstance(v, float):
                line += f"{v:<18.4f}" if not np.isnan(v) else f"{'n/a':<18}"
            else:
                line += f"{str(v):<18}"
        print(line)


# ============================================================================
# MAIN — identical to ml_state_optimized.py STEP 1-12, with the fold-level
# additions above wired in at:
#   STEP 6    -- inside the fold loop, right after the existing global
#                SHAP/PFI block: capture that same per-fold vector (plus
#                test_idx) into fold_regime_data instead of discarding it
#   STEP 7.5  -- right after the existing global SHAP/PFI CSV export: sum
#                each fold's own SHAP/PFI vector across features
#   STEP 10   -- trade_log entries now carry fold (3-tuples instead of
#                2-tuples); master_trade_df gets a fold_id column
#   STEP 11.5 -- new per-fold study + fold_regime_detection.csv, right
#                after the existing weekday/weekend regime report
# Everything else below is copied verbatim from ml_state_optimized.py's
# __main__ so this script produces the SAME walk-forward numbers, trade
# log, and existing outputs, PLUS the new fold-level outputs.
# ============================================================================
if __name__ == "__main__":
    _start_report_capture(os.path.join(BUILD_DIR, "backtest_report_regime.md"))

    # ------------------------------------------------------------------
    # STEP 1/2 — Build (or load) the historical state space
    # ------------------------------------------------------------------
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

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(10)

    print("\n--- DEBUG: hour distribution right after state-space build ---")
    print(df_state_space["datetime"].dt.hour().value_counts().sort("datetime"))

    print("\n--- STATE SPACE PREVIEW (FIRST 5 ROWS) ---")
    print(df_state_space.head(5))

    print("\n--- STATE SPACE SUMMARY STATISTICS ---")
    print(df_state_space.describe())

    print("\n--- STATE SPACE GLIMPSE ---")
    print(df_state_space.glimpse())

    pl.Config.set_tbl_cols(8)

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
    rsi_high_thr = 100 - RSI_FILTER_THRESHOLD  # 70
    if ENABLE_RSI_ACCEL_FILTER:
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

    # NEW: per-fold regime accumulator (see module docstring point 2/3).
    # {fold: {"test_idx": np.ndarray, "shap_vec": np.ndarray, "pfi_vec": np.ndarray}}
    # populated once per completed fold, right after the existing global
    # per-fold SHAP/PFI block below — no rescoring needed, we just keep
    # what STEP 6 already computes instead of only averaging it away.
    fold_regime_data = {}

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

        # --- SHAP --- (global, fold-level — unchanged from ml_state_optimized.py)
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

            fold_shap_vec = np.abs(shap_array).mean(axis=0)
        except Exception as e:
            print(f"Fold {fold:02d} | SHAP failed: {e}")
            fold_shap_vec = np.full(len(features), np.nan)
        fold_shap_values.append(fold_shap_vec)

        # --- PFI --- (global, fold-level — unchanged from ml_state_optimized.py)
        if len(np.unique(y_test)) < 2:
            fold_pfi_vec = np.full(len(features), np.nan)
        else:
            try:
                pi_result = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=42, scoring='roc_auc', n_jobs=1
                )
                fold_pfi_vec = pi_result.importances_mean
            except Exception as e:
                print(f"Fold {fold:02d} | PFI failed: {e}")
                fold_pfi_vec = np.full(len(features), np.nan)
        fold_pfi_values.append(fold_pfi_vec)

        # NEW: keep this fold's own SHAP/PFI vector + test_idx for the
        # per-fold regime CSV (STEP 11.5) instead of only averaging it
        # away into feature_importance_v3.csv.
        fold_regime_data[fold] = {
            "test_idx": test_idx,
            "shap_vec": fold_shap_vec,
            "pfi_vec": fold_pfi_vec,
        }

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

        test_side = side_arr[test_idx]
        if ENABLE_RSI_ACCEL_FILTER:
            trade_conf = np.where(test_side > 0, y_proba, 1.0 - y_proba)
        else:
            trade_conf = y_proba

        for t in thresholds_to_track:
            signal_indices = find_debounced_signals(trade_conf, t, HYSTERESIS_BUFFER)

            non_overlapping_idx = []
            next_available_time = np.datetime64('1677-01-01')
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

                traded_returns = normalized_return[traded_idx].copy()

                # NEW: fold number tags along with every trade so Step 10/11.5
                # can bucket trades by fold, not just by threshold.
                trade_log[t].extend(zip(datetimes[traded_idx], traded_returns, [fold] * num_trades))

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

                prec = win_rate

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
    # STEP 7 — SHAP vs PERMUTATION IMPORTANCE (global, unchanged)
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
    # STEP 7.5 — Per-Fold SHAP/PFI Summation (NEW)
    # ------------------------------------------------------------------
    # Each fold already has exactly one SHAP vector and one PFI vector
    # (captured into fold_regime_data above) — no cross-fold averaging
    # needed here, just sum each fold's own vector across features.
    fold_sum_shap = {fid: float(np.nansum(d["shap_vec"])) for fid, d in fold_regime_data.items()}
    fold_sum_pfi = {fid: float(np.nansum(d["pfi_vec"])) for fid, d in fold_regime_data.items()}
    print("\nPer-fold summed importance (that fold's own SHAP/PFI vector, summed across all features):")
    for fid in sorted(fold_sum_shap):
        print(f"  fold={fid}: sum_shap={fold_sum_shap[fid]:.4f}  sum_pfi={fold_sum_pfi[fid]:.4f}")

    # ------------------------------------------------------------------
    # STEP 8 — Model Class-Level Precision & Recall (unchanged)
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
    # STEP 9 — Strategy Rolling Expectancy & CVaR (global, unchanged)
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
    # STEP 10 — Out-of-Sample PnL Chart (unchanged except fold number
    # now travels through trade_log into master_trade_df as fold_id)
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
        fold_ids = np.array([x[2] for x in trade_log[t]])
        order = np.argsort(dts)
        dts, returns, fold_ids = dts[order], returns[order], fold_ids[order]

        log_df = pl.DataFrame({
            "datetime": dts, "ret": returns, "fold_id": fold_ids, "threshold": round(float(t), 2),
        })
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
    # STEP 11 — Summary Report (unchanged, up through print_regime_report)
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

        # ------------------------------------------------------------------
        # STEP 11.5 — Per-Fold Regime Detection Report + CSV (NEW)
        # ------------------------------------------------------------------
        prod_trades = master_trade_df.filter(pl.col("threshold") == production_threshold)
        fold_trades = {
            int(fid): prod_trades.filter(pl.col("fold_id") == fid).sort("datetime")["ret"].to_numpy()
            for fid in prod_trades["fold_id"].unique().to_list()
        }
        fold_regime_df = build_fold_regime_report(
            df_training_zone, fold_trades, fold_regime_data, fold_sum_shap, fold_sum_pfi,
            features, ROLLING_WINDOW,
        )
        print_fold_regime_report(fold_regime_df)
        ensure_dir(BUILD_DIR)
        fold_regime_path = os.path.join(BUILD_DIR, "fold_regime_detection.csv")
        fold_regime_df.write_csv(fold_regime_path)
        print(f"\nFold regime-detection CSV saved to: {fold_regime_path} "
              f"({fold_regime_df.height} fold(s), {len(features)} feature-mean columns)")

    for t in thresholds_to_track:
        if len(trade_log[t]) > 0:
            returns = np.array([x[1] for x in trade_log[t]], dtype=float)
            print(f"  {prob_label} >= {t:.2f}: {len(returns)} trades, cumulative {returns.sum():.2f}R out-of-sample")

    print("\nReminder: this run ignores fees, slippage, latency, and partial fills.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 12 — Export Production Model (unchanged)
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