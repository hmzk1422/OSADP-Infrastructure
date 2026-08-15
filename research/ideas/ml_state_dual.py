"""
Dual-Tier ML State Space — Macro Bias + 1-Second Orderflow Trigger
===================================================================
Architecture (Top-Down Cascade):

    [ 5m Macro AI ]  --ARMED when P(WIN) >= threshold-->
           |
           v
    [ 1s Orderflow AI ]  --fires on high-confidence orderflow pattern
           |                 aligned with the macro bias-->
           v
        EXECUTE at that 1-second price

Why two tiers instead of one 1s model?
    The original single model trained on 1-second rows but fed them frozen
    5-minute indicators (rsi_2, ma_slope_5, ...). 300 consecutive rows had
    IDENTICAL features but DIFFERENT labels -> the model learned the bar's
    average label -> degenerate bimodal P(WIN) -> ~10 trades. The fix in
    ml_state_optimized.py deduplicated to bar-level, which fixed the stats
    but threw away 1-second timing.

    This file restores 1-second timing the RIGHT way: a dedicated 1s model
    that sees REAL per-second orderflow features (CVD, OBI, velocity, ...)
    which genuinely change every second, so consecutive rows are distinct.
    The macro model provides the statistical bias; the 1s model provides
    the zero-lag entry trigger.

Output includes the same threshold expectancy table + cumulative-R PnL chart
the original research produced, applied to BOTH:
    A) Micro-only   : 1s AI fires on its own (no macro filter)
    B) Cascade      : 1s AI fires only inside macro-ARMED 5m bars

Anti-overfitting measures (applied throughout):
    1. Purged chronological walk-forward (train never overlaps test in time)
    2. Embargo after each test block (label horizon + buffer)
    3. Macro deduplicated to bar-level (one independent sample per 5m bar)
    4. 1s model: features are backward-looking (no lookahead), labels on future
    5. 1s model: purged by label horizon so train/test don't share outcomes
    6. Class balancing (class_weight='balanced')
    7. Hysteresis debounce + non-overlap filter on the 1s trigger (one
       position per 30s horizon — overlapping trades would double-count the
       same outcome)
    8. Ablation test: macro with vs without the 1s AI feature, compared on
       IC / R-per-trade to quantify orderflow's real contribution
"""

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

import lightgbm as lgb

from utils.indicators import calculate_zscore

# Reuse the tested macro building blocks from the optimized pipeline so the
# two files can't silently drift.
from ml_state_optimized import (
    load_and_build_state_space,
    add_barrier_labels,
    deduplicate_to_bar,
    BAR_FREQ,
    CLASS_NAMES,
)

from main.config import (
    LOOKAHEAD_SECONDS, BARRIER_MULTIPLIER, RSI_FILTER_COLUMN, RSI_FILTER_THRESHOLD,
    RR_RATIO, TRADE_DIRECTION, TIMEFRAME_BARRIER,
    RSI_14_WINDOW, RSI_2_WINDOW, SMA_14_WINDOW, DEEPLOB_Z_WINDOW, RSI_Z_WINDOW,
    MA_SLOPE_WINDOW, PRICE_ROC_WINDOW_3, LABEL_LOSS, LABEL_WIN, FEATURE_COLUMNS,
    FEE_BPS, MIN_BARRIER_PCT, MAX_BARRIER_PCT, EPS,
    HYSTERESIS_BUFFER, LOCKOUT_SECONDS,
)

try:
    from main.config import BARRIER_VOL_BAR_WINDOW
except ImportError:
    BARRIER_VOL_BAR_WINDOW = 20

# ============================================================================
# DUAL-TIER CONFIG
# ============================================================================
# --- 1s Orderflow AI (the "Sniper") ---
MICRO_HORIZON_SECONDS = 30        # label: does price move within next 30s?
MICRO_THRESHOLD_BPS   = 2.0       # label: move >= 2.0 bps (0.02%)
MICRO_EMBARGO_SECONDS = MICRO_HORIZON_SECONDS  # purge + embargo for 1s model
MICRO_N_FOLDS         = 5         # fewer folds (1s data is huge)
MICRO_TEST_BLOCK      = 20000     # 1s rows per test block (~5.5 hours)
MICRO_MIN_TRAIN_FRAC  = 0.30

# --- Macro walk-forward (mirrors ml_state_optimized's __main__ constants) ---
MACRO_N_FOLDS         = 15
MACRO_TEST_BLOCK      = 150       # bar-level rows per test fold
MACRO_EMBARGO_SECONDS = LOOKAHEAD_SECONDS

# --- Cascade / threshold sweep ---
MACRO_ARM_THRESHOLD   = 0.60      # macro P(WIN) to set ARMED
MICRO_FIRE_THRESHOLD  = 0.70      # 1s AI P(up) for the strict cascade check
THRESHOLDS_TO_TRACK   = np.arange(0.20, 0.96, 0.05)  # expectancy sweep
ROLLING_WINDOW        = 30        # CVaR window (must be >= 30 trades)
INVERT_RETURNS = (TRADE_DIRECTION == 'short')

# --- Ablation ---
ABLATION_FOLDS = 5


# ============================================================================
# PURGED CHRONOLOGICAL SPLITTER (local copy — ml_state_optimized's lives
# inside its __main__ block and is NOT importable)
# ============================================================================
def purged_chronological_splits(index, n_samples, block_size, horizon_seconds, embargo_seconds=0):
    n = len(index)
    horizon = pd.Timedelta(seconds=horizon_seconds)
    embargo = pd.Timedelta(seconds=embargo_seconds)
    min_initial_train = int(n * 0.35)
    available_test_run = n - min_initial_train - block_size
    if available_test_run <= 0:
        raise ValueError("The dataset length is too narrow for this test configuration block size.")
    step_intervals = available_test_run // n_samples
    for fold in range(n_samples):
        start_idx = min_initial_train + (fold * step_intervals)
        test_idx = np.arange(start_idx, start_idx + block_size)
        test_start_time = index[test_idx[0]]
        possible_train_idx = np.arange(0, start_idx)
        train_times = index[possible_train_idx]
        purge_mask = (train_times + horizon + embargo >= test_start_time)
        train_idx = possible_train_idx[~purge_mask]
        n_purged = len(possible_train_idx) - len(train_idx)
        yield train_idx, test_idx, n_purged


def find_debounced_signals(y_proba, threshold, hysteresis=HYSTERESIS_BUFFER):
    """Hysteresis-debounced threshold crossings (matches ml_state_optimized)."""
    lower_bound = max(threshold - hysteresis, 0.0)
    armed = True
    out = []
    for i, p in enumerate(y_proba):
        if armed and p >= threshold:
            out.append(i)
            armed = False
        elif not armed and p < lower_bound:
            armed = True
    return np.array(out, dtype=int)


# ============================================================================
# STEP 1 — 1-SECOND ORDERFLOW FEATURE BUILDER
# ============================================================================
def build_1s_orderflow_features(orderbook_path, trades_path):
    """
    Build a TRUE 1-second orderflow feature frame from the raw tape + L1 book.
    Features change every second -> consecutive 1s rows are genuinely distinct
    (fixes the frozen-5m-indicator duplication that broke the old 1s model).
    """
    ob = pd.read_csv(orderbook_path)
    tr = pd.read_csv(trades_path)

    ob['dt'] = pd.to_datetime(ob['timestamp'].astype('int64') // 1000, unit='s')
    tr['dt'] = pd.to_datetime(tr['timestamp'].astype('int64') // 1000, unit='s')
    ob = ob.set_index('dt').sort_index()
    tr = tr.set_index('dt').sort_index()

    # --- Per-second orderbook (last snapshot each second) ---
    ob_1s = ob.resample('1s').agg({
        'bid_price': 'last', 'bid_amount': 'last',
        'ask_price': 'last', 'ask_amount': 'last',
    })
    ob_1s['obi'] = (ob_1s['bid_amount'] - ob_1s['ask_amount']) / (ob_1s['bid_amount'] + ob_1s['ask_amount'] + EPS)
    ob_1s['obi_change'] = ob_1s['obi'].diff()
    ob_1s['bid_size_delta'] = ob_1s['bid_amount'].diff()
    ob_1s['ask_size_delta'] = ob_1s['ask_amount'].diff()
    ob_1s['spread'] = ob_1s['ask_price'] - ob_1s['bid_price']
    ob_1s['mid'] = (ob_1s['bid_price'] + ob_1s['ask_price']) / 2

    # --- Per-second trades ---
    tr['buy_vol'] = tr['amount'].where(tr['side'] == 'buy', 0)
    tr['sell_vol'] = tr['amount'].where(tr['side'] == 'sell', 0)
    tr_1s = tr.resample('1s').agg({
        'price': 'last', 'amount': 'sum',
        'buy_vol': 'sum', 'sell_vol': 'sum', 'id': 'count',
    })
    tr_1s.columns = ['close', 'volume', 'buy_vol', 'sell_vol', 'n_trades']
    tr_1s['cvd_delta'] = tr_1s['buy_vol'] - tr_1s['sell_vol']
    tr_1s['cvd'] = tr_1s['cvd_delta'].cumsum()
    tr_1s['aggressive_ratio'] = tr_1s['buy_vol'] / (tr_1s['volume'] + EPS)
    tr_1s['trade_velocity'] = tr_1s['n_trades']
    tr_1s['volume_velocity'] = tr_1s['volume']

    # --- Join (book persists between updates; empty trade seconds = 0) ---
    df = ob_1s.join(tr_1s, how='outer').sort_index()
    for c in ['obi', 'obi_change', 'bid_size_delta', 'ask_size_delta', 'spread', 'mid']:
        df[c] = df[c].ffill()
    # FIX (bug 1): 'close' (last traded price) was previously left unfilled for
    # seconds with zero trades -> NaN propagated through pct_change/shift ->
    # dropna() in train_micro_ai silently deleted large numbers of rows around
    # every quiet second, biasing the sample toward continuously-trading periods.
    # Forward-fill it the same way as the book columns: "no trade this second"
    # means the last traded price hasn't changed.
    df['close'] = df['close'].ffill()
    df['cvd'] = df['cvd'].ffill().fillna(0)
    df = df.fillna({
        'volume': 0, 'buy_vol': 0, 'sell_vol': 0, 'n_trades': 0,
        'cvd_delta': 0, 'aggressive_ratio': 0.5,
        'trade_velocity': 0, 'volume_velocity': 0,
    })

    # --- Micro momentum / rolling (backward-looking only) ---
    df['ret_1s'] = df['close'].pct_change()
    df['ret_3s'] = df['close'].pct_change(3)
    df['ret_5s'] = df['close'].pct_change(5)
    df['cvd_accel_3s'] = df['cvd_delta'].rolling(3).sum()
    df['cvd_accel_5s'] = df['cvd_delta'].rolling(5).sum()
    df['obi_ma_3s'] = df['obi'].rolling(3).mean()
    df['vol_ma_3s'] = df['volume'].rolling(3).mean()
    df['trade_vel_ma_3s'] = df['trade_velocity'].rolling(3).mean()
    df['obi_z'] = calculate_zscore(df['obi'].fillna(0), window=20)
    df['cvd_delta_z'] = calculate_zscore(df['cvd_delta'].fillna(0), window=20)
    df['vol_z'] = calculate_zscore(df['volume'].fillna(0), window=20)

    return df


# ============================================================================
# STEP 2 — 1-SECOND MICRO LABEL (time-based)
# ============================================================================
def add_micro_labels(df, horizon_seconds=MICRO_HORIZON_SECONDS,
                     threshold_bps=MICRO_THRESHOLD_BPS):
    """Label: price up >= threshold_bps within the next horizon_seconds."""
    df = df.copy()
    fwd_ret = df['close'].shift(-horizon_seconds) / df['close'] - 1.0
    df['fwd_ret'] = fwd_ret
    df['micro_target'] = (fwd_ret >= threshold_bps / 10000.0).astype(int)
    return df


# ============================================================================
# STEP 3 — 1-SECOND ORDERFLOW AI (purged walk-forward)
# ============================================================================
MICRO_FEATURES = [
    'obi', 'obi_change', 'bid_size_delta', 'ask_size_delta', 'spread',
    'cvd_delta', 'cvd', 'aggressive_ratio', 'trade_velocity', 'volume_velocity',
    'ret_1s', 'ret_3s', 'ret_5s', 'cvd_accel_3s', 'cvd_accel_5s',
    'obi_ma_3s', 'vol_ma_3s', 'trade_vel_ma_3s', 'obi_z', 'cvd_delta_z', 'vol_z',
]


def train_micro_ai(df_micro):
    """Train the 1s Orderflow AI; return OOS P(up) + OOS targets (each row
    tested exactly once across folds)."""
    df = df_micro.dropna(subset=MICRO_FEATURES + ['micro_target']).copy()
    X = df[MICRO_FEATURES]
    y = df['micro_target']
    print(f"\n1s Orderflow AI: {len(X)} labeled rows | P(up) rate = {y.mean():.3f}")

    splitter = purged_chronological_splits(
        X.index, n_samples=MICRO_N_FOLDS, block_size=MICRO_TEST_BLOCK,
        horizon_seconds=MICRO_HORIZON_SECONDS, embargo_seconds=MICRO_EMBARGO_SECONDS,
    )

    oos_proba, oos_y = {}, {}
    for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if len(X_train) == 0 or y_train.nunique() < 2:
            print(f"  Micro fold {fold} skipped (bad train distribution)")
            continue
        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        for ts, p, yy in zip(X_test.index, proba, y_test.values):
            oos_proba[ts] = p
            oos_y[ts] = yy
        print(f"  Micro fold {fold:02d} | Train {len(X_train):<7} | "
              f"Test {len(X_test):<7} | Purged {n_purged}")

    oos = pd.Series(oos_proba).sort_index()
    oos_target = pd.Series(oos_y).sort_index()
    return oos, oos_target


# ============================================================================
# STEP 4 — MACRO PIPELINE (dedup-fixed) + OOS P(WIN)
# ============================================================================
def build_macro_state_space():
    print("\nLoading macro sessions...")
    s0 = load_and_build_state_space(
        "research/csv/BTCUSDT_orderbook.csv", "research/csv/BTCUSDT_trades.csv", session_id=0)
    s1 = load_and_build_state_space(
        "research/csv/BTCUSDT_orderbook_2.csv", "research/csv/BTCUSDT_trades_2.csv", session_id=1)
    s2 = load_and_build_state_space(
        "research/csv/BTCUSDT_orderbook_3.csv", "research/csv/BTCUSDT_trades_3.csv", session_id=2)
    df = pd.concat([s0, s1, s2]).sort_index()
    df = add_barrier_labels(df)
    df = df[df['volume'] > 0]
    print(f"Macro state space before dedup: {len(df)} rows")
    df = deduplicate_to_bar(df)
    print(f"Macro state space after dedup (bar-level): {len(df)} rows")
    return df


def macro_features_list(df_macro):
    ORDERFLOW_FEATURE_COLUMNS = [
        'ofi_z', 'cvd_delta_z', 'cvd_slope_z', 'vpin', 'absorption_z',
        'tick_autocorr', 'inst_ratio', 'delta_divergence_z', 'depth_imbalance_z',
    ]
    return list(FEATURE_COLUMNS) + [
        c for c in ORDERFLOW_FEATURE_COLUMNS if c in df_macro.columns
    ]


def get_macro_oos(df_macro, n_folds=None, block_size=None):
    """
    Purged walk-forward on the macro model. Returns:
      macro_p     : Series[bar_ts -> OOS P(WIN)]   (each bar tested once)
      macro_y     : Series[bar_ts -> actual label]
      macro_norm  : Series[bar_ts -> risk-normalized R return]
    """
    n_folds = n_folds or MACRO_N_FOLDS
    block_size = block_size or MACRO_TEST_BLOCK
    features = macro_features_list(df_macro)
    X = df_macro[features]
    y = df_macro['target']

    # Risk-normalized return (Step 5 of ml_state_optimized)
    norm = pd.Series(index=df_macro.index, dtype=float)
    is_timeout = df_macro['is_timeout']
    is_win = df_macro['target'] == LABEL_WIN
    norm[is_win & ~is_timeout] = 1.0
    norm[is_loss := ~is_win & ~is_timeout] = -1.0 / RR_RATIO
    norm[is_timeout] = (
        df_macro.loc[is_timeout, 'exit_ret'] /
        (df_macro.loc[is_timeout, 'stop_barrier'] + EPS)
    )
    fee_in_r = (2.0 * FEE_BPS / 10000.0) / (df_macro['stop_barrier'] + EPS)
    norm = (norm - fee_in_r).clip(-max(abs(norm.quantile(0.99)), abs(norm.quantile(0.01)), RR_RATIO),
                                   max(abs(norm.quantile(0.99)), abs(norm.quantile(0.01)), RR_RATIO))

    splitter = purged_chronological_splits(
        X.index, n_samples=n_folds, block_size=block_size,
        horizon_seconds=LOOKAHEAD_SECONDS, embargo_seconds=MACRO_EMBARGO_SECONDS,
    )
    oos_p, oos_y = {}, {}
    for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if len(X_train) == 0 or y_train.nunique() < 2:
            continue
        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_train, y_train)
        target_idx = list(clf.classes_).index(LABEL_WIN)
        proba = clf.predict_proba(X_test)[:, target_idx]
        for ts, p, yy in zip(X_test.index, proba, y_test.values):
            oos_p[ts] = p
            oos_y[ts] = yy

    macro_p = pd.Series(oos_p).sort_index()
    macro_y = pd.Series(oos_y).sort_index()
    return macro_p, macro_y, norm


# ============================================================================
# STEP 5 — EXPECTANCY TABLE (threshold sweep, both micro-only and cascade)
# ============================================================================
def _realized_r(micro_target):
    """Convert a 0/1 micro label to realized R (win=+RR, loss=-1), inverted for short."""
    r = np.where(micro_target == 1, RR_RATIO, -1.0)
    if INVERT_RETURNS:
        r = -r
    return r


def _build_trade_set(micro_oos, micro_target, threshold, macro_oos=None):
    """
    Hysteresis-debounced 1s signals at `threshold`, with a non-overlap filter
    of MICRO_HORIZON_SECONDS (one position per horizon — overlapping trades
    would double-count the same 30s outcome). If macro_oos is given, only
    signals inside a macro-ARMED (P(WIN) >= MACRO_ARM_THRESHOLD) 5m bar window
    count (cascade). Returns the list of signal timestamps.
    """
    sig_idx = find_debounced_signals(micro_oos.values, threshold, HYSTERESIS_BUFFER)
    sig_ts = micro_oos.index[sig_idx]

    # Build macro-armed window intervals once (if filtering)
    armed_windows = []
    if macro_oos is not None:
        for bar_ts, mp in macro_oos.items():
            if mp >= MACRO_ARM_THRESHOLD:
                armed_windows.append(
                    (bar_ts, bar_ts + pd.Timedelta(seconds=TIMEFRAME_BARRIER)))

    trades = []
    next_available = pd.Timestamp.min
    for ts in sig_ts:
        if ts < next_available:
            continue  # still in a position for the prior trade's horizon
        if macro_oos is not None:
            inside = any(w0 <= ts < w1 for w0, w1 in armed_windows)
            if not inside:
                continue  # macro not ARMED at this second
        trades.append(ts)
        next_available = ts + pd.Timedelta(seconds=MICRO_HORIZON_SECONDS)
    return trades


def _expectancy_row(trade_ts, micro_target):
    """Compute one threshold's stats: Win%, RR, Exp, CVaR(30), Total R."""
    if len(trade_ts) == 0:
        return None
    r = _realized_r(micro_target.loc[trade_ts].values)
    num = len(r)
    win_rate = np.mean(r > 0)
    avg_win = r[r > 0].mean() if (r > 0).sum() > 0 else 0
    avg_loss = r[r < 0].mean() if (r < 0).sum() > 0 else 0
    rr = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan
    overall_exp = np.mean(r)
    total_r = np.sum(r)
    if num >= ROLLING_WINDOW:
        rolling_exp = pd.Series(r).rolling(ROLLING_WINDOW).mean().dropna()
        cutoff = max(1, int(len(rolling_exp) * 0.05))
        cvar_exp = np.mean(np.sort(rolling_exp.values)[:cutoff])
        stable = "YES" if cvar_exp > 0 else "NO"
    else:
        cvar_exp = np.nan
        stable = "N/A (low trades)"
    return {
        'trades': num, 'win_rate': win_rate, 'rr': rr,
        'exp': overall_exp, 'cvar': cvar_exp, 'stable': stable, 'total_r': total_r,
    }


def print_expectancy_table(title, micro_oos, micro_target, macro_oos=None):
    """Print the Strategy Rolling Expectancy table (same format as research)."""
    print(f"\n--- {title} ---")
    print("-" * 140)
    print(f"{'Thresh':<8} | {'Trades':<8} | {'Win%':<8} | {'RR':<6} | "
          f"{'Overall Exp':<12} | {'Exp CVaR (5%)':<14} | {'Stable (CVaR>0)':<16} | {'Total R'}")
    print("-" * 140)
    trade_logs = {}
    for t in THRESHOLDS_TO_TRACK:
        trade_ts = _build_trade_set(micro_oos, micro_target, t, macro_oos)
        trade_logs[t] = trade_ts
        row = _expectancy_row(trade_ts, micro_target)
        if row is None:
            continue
        print(f">= {t*100:>3.0f}% | {row['trades']:<8} | {row['win_rate']:>5.1%} | "
              f"{row['rr']:<6.2f} | {row['exp']:>10.2f} R | {row['cvar']:>12.2f} R | "
              f"{row['stable']:<16} | {row['total_r']:>+6.2f} R")
    print("-" * 140)
    return trade_logs


# ============================================================================
# STEP 6 — CUMULATIVE-R PNL CHART (Plotly, saved to HTML)
# ============================================================================
def plot_cumulative_pnl(trade_logs, micro_target, title, filename):
    import plotly.graph_objects as go

    fig = go.Figure()
    for t, trade_ts in sorted(trade_logs.items()):
        if len(trade_ts) == 0:
            continue
        r = _realized_r(micro_target.loc[trade_ts].values)
        cum = np.cumsum(r)
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum) + 1)), y=cum, mode='lines',
            name=f'>= {t:.2f}',
            hovertemplate='Trade: %{x}<br>PnL: %{y:.2f}R<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title='Trade Number',
        yaxis_title='Cumulative Return (R)',
        legend_title='P(up) ≥ (Click!)',
        hovermode="x unified",
        template="plotly_white",
    )
    fig.write_html(filename)
    print(f"Interactive chart saved to: {filename}")
    fig.show()


# ============================================================================
# STEP 7 — CASCADE LOG (strict thresholds, single line of summary)
# ============================================================================
def run_cascade(df_macro, micro_oos, micro_target, macro_p):
    """
    Strict cascade at MACRO_ARM_THRESHOLD / MICRO_FIRE_THRESHOLD: fires the
    FIRST 1s signal >= fire threshold inside each macro-ARMED bar's window.
    """
    trades = []
    for bar_ts, mp in macro_p.items():
        if mp < MACRO_ARM_THRESHOLD:
            continue
        window_end = bar_ts + pd.Timedelta(seconds=TIMEFRAME_BARRIER)
        in_window = micro_oos[(micro_oos.index >= bar_ts) & (micro_oos.index < window_end)]
        fired = in_window[in_window >= MICRO_FIRE_THRESHOLD]
        if fired.empty:
            continue
        trigger_ts = fired.index[0]
        if trigger_ts in micro_target.index:
            r = _realized_r(micro_target.loc[[trigger_ts]])[0]
            trades.append({'time': trigger_ts, 'macro_p': mp,
                           'micro_p': fired.iloc[0], 'r': r})
    return pd.DataFrame(trades)


# ============================================================================
# STEP 8 — ABLATION: does the 1s AI feature help the macro model?
# ============================================================================
def run_ablation(df_macro, micro_oos):
    # FIX (bug 2): resample(BAR_FREQ).mean() labels each bin by its START, so
    # micro_bar.loc[bar_ts] previously averaged 1s predictions from the bar's
    # OWN future window [bar_ts, bar_ts + BAR_FREQ) -- the same window
    # df_macro['target'] is realized over. That's look-ahead: the "with_micro"
    # variant was partly reading a summary of the label's own window, not
    # information available at decision time bar_ts. shift(1) lags the feature
    # by one full bar so only the PRECEDING, already-closed bar's orderflow is
    # used -- the most recent information genuinely known at bar_ts.
    micro_bar = micro_oos.resample(BAR_FREQ).mean().shift(1).reindex(df_macro.index).ffill()
    base_features = macro_features_list(df_macro)
    X_base = df_macro[base_features]
    y = df_macro['target']

    splitter = purged_chronological_splits(
        X_base.index, n_samples=ABLATION_FOLDS, block_size=MACRO_TEST_BLOCK,
        horizon_seconds=LOOKAHEAD_SECONDS, embargo_seconds=MACRO_EMBARGO_SECONDS,
    )
    results = {'base': {'ic': [], 'r_per_trade': []},
               'with_micro': {'ic': [], 'r_per_trade': []}}

    for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if len(train_idx) == 0 or y_train.nunique() < 2:
            continue
        for variant, X in [('base', X_base),
                           ('with_micro', X_base.assign(micro_p=micro_bar))]:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            clf = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
            )
            clf.fit(X_train, y_train)
            target_idx = list(clf.classes_).index(LABEL_WIN)
            p = clf.predict_proba(X_test)[:, target_idx]
            ic = pd.Series(p, index=X_test.index).corr(
                pd.Series(y_test.values, index=X_test.index), method='spearman')
            results[variant]['ic'].append(ic)
            armed = p >= MACRO_ARM_THRESHOLD
            if armed.sum() > 0:
                r = np.where(y_test.values[armed] == LABEL_WIN, RR_RATIO, -1.0)
                if INVERT_RETURNS:
                    r = -r
                results[variant]['r_per_trade'].append(r.mean())

    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DUAL-TIER ML — 5m Macro Bias + 1s Orderflow Trigger")
    print("=" * 70)
    print(f"Macro ARMED threshold : P(WIN) >= {MACRO_ARM_THRESHOLD}")
    print(f"1s fire threshold     : P(up) >= {MICRO_FIRE_THRESHOLD}")
    print(f"Micro label           : price >= {MICRO_THRESHOLD_BPS} bps in next {MICRO_HORIZON_SECONDS}s")
    print(f"Direction             : {TRADE_DIRECTION} (invert returns = {INVERT_RETURNS})")

    # ---- 1s orderflow features (session 0) ----
    print("\n[1/8] Building 1-second orderflow features (session 0)...")
    df_micro = build_1s_orderflow_features(
        "research/csv/BTCUSDT_orderbook.csv", "research/csv/BTCUSDT_trades.csv")
    print(f"  1s orderflow frame: {len(df_micro)} rows")

    # ---- Micro label ----
    print("[2/8] Adding micro labels...")
    df_micro = add_micro_labels(df_micro)

    # ---- Train the 1s Orderflow AI ----
    print("[3/8] Training 1s Orderflow AI (purged walk-forward)...")
    micro_oos, micro_target = train_micro_ai(df_micro)
    print(f"  Out-of-sample 1s predictions: {len(micro_oos)}")
    if len(micro_oos) > 0:
        print(f"  OOS P(up) mean={micro_oos.mean():.3f} | "
              f"P(up)>=0.7 rate={(micro_oos>=0.7).mean():.3f}")

    # ---- Macro state space + OOS P(WIN) ----
    print("[4/8] Building macro state space (dedup-fixed) + OOS P(WIN)...")
    df_macro = build_macro_state_space()
    macro_p, macro_y, macro_norm = get_macro_oos(df_macro)
    print(f"  Macro OOS bars: {len(macro_p)} | ARM rate (>=0.6): "
          f"{(macro_p>=MACRO_ARM_THRESHOLD).mean():.3f}")

    # ---- Strict cascade summary ----
    print("[5/8] Strict cascade (macro-ARMED + 1s fire)...")
    trade_df = run_cascade(df_macro, micro_oos, micro_target, macro_p)
    if len(trade_df) > 0:
        print(f"  Cascade trades: {len(trade_df)} | "
              f"Win%: {(trade_df['r']>0).mean():.1%} | "
              f"Total R: {trade_df['r'].sum():+.2f}")
    else:
        print("  No cascade trades fired (strict thresholds). See expectancy table below.")

    # ---- Threshold expectancy tables ----
    print("\n" + "=" * 70)
    print("EXPECTANCY BY P(UP) THRESHOLD")
    print("=" * 70)
    print("NOTE: Win/RR/Exp are for the 30s micro horizon (the timing event),")
    print("realized on the micro label and inverted for 'short'. Trades are")
    print("hysteresis-debounced and non-overlapping (one per 30s horizon).")

    micro_only_logs = print_expectancy_table(
        "A) MICRO-ONLY — 1s Orderflow AI standalone", micro_oos, micro_target)
    cascade_logs = print_expectancy_table(
        "B) CASCADE — 1s trigger only inside macro-ARMED bars",
        micro_oos, micro_target, macro_oos=macro_p)

    # ---- Cumulative-R PnL charts ----
    print("\n[7/8] Building cumulative-R PnL charts...")
    plot_cumulative_pnl(micro_only_logs, micro_target,
                        "Dual-Tier — Micro-Only Cumulative R by P(up) Threshold",
                        "pnl_dual_micro_only.html")
    plot_cumulative_pnl(cascade_logs, micro_target,
                        "Dual-Tier — Cascade Cumulative R by P(up) Threshold (Macro-ARMED)",
                        "pnl_dual_cascade.html")

    # ---- Ablation ----
    print("[8/8] Ablation: macro with vs without 1s AI feature...")
    abl = run_ablation(df_macro, micro_oos)
    print("\n--- Ablation Results (out-of-sample) ---")
    for variant in ['base', 'with_micro']:
        ic = np.array(abl[variant]['ic'])
        rpt = np.array(abl[variant]['r_per_trade'])
        print(f"  {variant:<12} | IC mean={np.nanmean(ic):+.4f} | "
              f"R/trade mean={np.nanmean(rpt):+.3f}")

    print("\n" + "=" * 70)
    print("DONE. Compare A vs B to see how much the macro filter improves")
    print("timing-cadence, and the ablation IC/R to see the feature value.")
    print("=" * 70)
