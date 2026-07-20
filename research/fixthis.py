import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
warnings.filterwarnings('ignore')

import sys
import os
# Get the directory where this script lives (research/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)  # cd to research/ so CSV paths resolve
# Add project root to path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from utils.indicators import *
from utils.engine import *

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
from sklearn.inspection import permutation_importance
import shap

# ============================================================================
# CONFIG
# ============================================================================
LOOKAHEAD_SECONDS    = 300
BARRIER_MULTIPLIER   = 2.0
RSI_FILTER_COLUMN    = 'rsi_2'
RSI_FILTER_THRESHOLD = 30
EPS                  = 1e-9
N_RANDOM_SAMPLES     = 15
TEST_BLOCK_SIZE      = 10000
EMBARGO_SECONDS      = LOOKAHEAD_SECONDS
BARRIER_CHUNK_SIZE   = 50_000
RR_RATIO             = 1.0

LABEL_LOSS = 0
LABEL_WIN  = 1
CLASS_NAMES = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}
TARGET_TRADE_CLASS = LABEL_WIN
INVERT_RETURNS     = True
target_name = CLASS_NAMES[TARGET_TRADE_CLASS]

# --- DEFINE FEATURE GROUPS ---
# Price features: derived from price action / RSI / SMA
PRICE_FEATURES = ['rsi_2', 'rsi_14', 'sma_14']
# Orderflow features: everything else that is not price-derived and not metadata
# We'll build the full feature list from the pipeline, then separate programmatically.

print('=' * 70)
print('MULTICOLLINEARITY TEST: PRICE vs ORDERFLOW PREDICTIVENESS')
print('Hypothesis: Orderflow predictive power is masked by multicollinearity')
print('  with price features. Removing orderflow should reduce price feature')
print('  importance if multicollinearity is the cause.')
print('=' * 70)

# ============================================================================
# STEP 1 — Data Loading & State Space Building (from fixthis.py base)
# ============================================================================
def load_and_build_state_space(orderbook_path, trades_path, session_id):
    df_orderbook = pd.read_csv(orderbook_path, low_memory=False)
    df_trades = pd.read_csv(trades_path, low_memory=False)

    num_cols_ob = ['timestamp', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount']
    num_cols_tr = ['timestamp', 'price', 'amount']
    for c in num_cols_ob:
        df_orderbook[c] = pd.to_numeric(df_orderbook[c], errors='coerce')
    for c in num_cols_tr:
        df_trades[c] = pd.to_numeric(df_trades[c], errors='coerce')

    df_orderbook['datetime'] = pd.to_datetime(df_orderbook['timestamp'] // 1000, unit='s')
    df_trades['datetime'] = pd.to_datetime(df_trades['timestamp'] // 1000, unit='s')

    df_orderbook = df_orderbook.set_index('datetime').sort_index()
    df_trades = df_trades.set_index('datetime').sort_index()

    df_orderbook = df_orderbook.dropna(subset=['timestamp'])
    df_trades = df_trades.dropna(subset=['timestamp'])

    df_orderbook['mid_price'] = (df_orderbook['bid_price'] + df_orderbook['ask_price']) / 2

    df_ob_1m = df_orderbook.resample('1min').agg({
        'mid_price': ['first', 'max', 'min', 'last'],
        'bid_price': 'last',
        'ask_price': 'last',
        'bid_amount': 'sum',
        'ask_amount': 'sum',
    })
    df_ob_1m.columns = ['open', 'high', 'low', 'close', 'bid_price', 'ask_price', 'bid_amount', 'ask_amount']
    df_ob_1m['spread'] = df_ob_1m['ask_price'] - df_ob_1m['bid_price']

    df_trades['buy_vol'] = df_trades['amount'].where(df_trades['side'] == 'buy', 0)
    df_trades['sell_vol'] = df_trades['amount'].where(df_trades['side'] == 'sell', 0)

    df_micro = df_trades.resample('1s').agg({
        'price': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum'
    }).ffill()

    df_micro['micro_OFI'] = df_micro['buy_vol'] - df_micro['sell_vol']
    df_micro['micro_CVD'] = df_micro['micro_OFI'].rolling(10).sum()

    df_trades_1m = df_trades.resample('1min').agg({
        'price': ['first', 'max', 'min', 'last', 'count'],
        'amount': 'sum',
        'buy_vol': ['sum', 'max'],
        'sell_vol': ['sum', 'max']
    })
    df_trades_1m.columns = ['open', 'high', 'low', 'close', 'num_trades', 'volume',
                             'agg_buyer', 'largest_buyer', 'agg_seller', 'largest_seller']
    df_trades_1m = df_trades_1m.dropna(subset=['close'])

    df_price_1m = pd.DataFrame({
        'open': df_ob_1m['open'],
        'high': df_ob_1m['high'],
        'low': df_ob_1m['low'],
        'close': df_ob_1m['close'],
        'spread': df_ob_1m['spread'],
        'largest_buyer': df_trades_1m['largest_buyer'] * 100 / (df_trades_1m['volume'] + EPS),
        'largest_seller': df_trades_1m['largest_seller'] * 100 / (df_trades_1m['volume'] + EPS),
        'OFI_LOB': 100 * (df_ob_1m['bid_amount'] - df_ob_1m['ask_amount']) / (df_ob_1m['bid_amount'] + df_ob_1m['ask_amount'] + EPS),
        'OFI_AGG': 100 * (df_trades_1m['agg_buyer'] - df_trades_1m['agg_seller']) / (df_trades_1m['agg_buyer'] + df_trades_1m['agg_seller'] + EPS),
        'CVD_raw': calculate_cvd(df_trades_1m['agg_buyer'], df_trades_1m['agg_seller']),
        'candle_type': abs(df_ob_1m['close'] - df_ob_1m['open']) / (abs(df_ob_1m['high'] - df_ob_1m['low']) + EPS),
        'candle_size': (df_ob_1m['high'] - df_ob_1m['low']) / ((df_ob_1m['high'] - df_ob_1m['low']).rolling(14).mean() + EPS),
        'volume': df_trades_1m['volume']
    })

    cvd_delta = df_price_1m['CVD_raw'].diff()
    df_price_1m['CVD_z'] = (cvd_delta - cvd_delta.rolling(60).mean()) / (cvd_delta.rolling(60).std() + EPS)
    df_price_1m = df_price_1m.drop(columns=['CVD_raw'])

    price_clean = np.asarray(df_price_1m['close'], dtype=float)
    diffs = np.diff(price_clean)

    def _pad_front(arr, target_len):
        out = np.full(target_len, np.nan)
        out[-len(arr):] = arr
        return out

    df_price_1m['rsi_2'] = _pad_front(get_wilders_rsi(diffs, 2), len(df_price_1m))
    rsi_14_raw = get_wilders_rsi(diffs, 14)
    df_price_1m['rsi_14'] = _pad_front(rsi_14_raw, len(df_price_1m))
    df_price_1m['sma_14'] = _pad_front(sma(rsi_14_raw, 14), len(df_price_1m))

    df_micro_shifted = df_micro.shift(1)
    df_macro_shifted = df_price_1m

    df_state_space = df_micro_shifted.join(df_macro_shifted, how='left').ffill()
    df_state_space['execution_price'] = df_micro['price']
    df_state_space = df_state_space.dropna()
    df_state_space['session_id'] = session_id

    return df_state_space

print('\n--- Loading data ---')
s0 = load_and_build_state_space('./csv/BTCUSDT_orderbook.csv', './csv/BTCUSDT_trades.csv', 0)
print(f'Session 0 rows: {len(s0)}')
s1 = load_and_build_state_space('./csv/BTCUSDT_orderbook_2.csv', './csv/BTCUSDT_trades_2.csv', 1)
print(f'Session 1 rows: {len(s1)}')
s2 = load_and_build_state_space('./csv/BTCUSDT_orderbook_3.csv', './csv/BTCUSDT_trades_3.csv', 1)
print(f'Session 2 rows: {len(s2)}')

df_state_space = pd.concat([s0, s1, s2]).sort_index()
print(f'Total combined rows: {len(df_state_space)}')

# ============================================================================
# STEP 2 — Barrier Labeling
# ============================================================================
def _sequential_triple_barrier(price, profit_thresh, stop_thresh, lookahead, chunk_size=BARRIER_CHUNK_SIZE):
    price = np.asarray(price, dtype=float)
    profit_thresh = np.asarray(profit_thresh, dtype=float)
    stop_thresh = np.asarray(stop_thresh, dtype=float)
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
        upper = p0 * (1.0 + (pt * RR_RATIO))
        lower = p0 * (1.0 - st)
        window = sliding_window_view(price[c0 + 1: c1 + lookahead], lookahead)
        hit_upper = window >= upper[:, None]
        hit_lower = window <= lower[:, None]
        has_upper = hit_upper.any(axis=1)
        has_lower = hit_lower.any(axis=1)
        first_upper = np.argmax(hit_upper, axis=1)
        first_lower = np.argmax(hit_lower, axis=1)
        win_mask = has_upper & (~has_lower | (first_upper < first_lower))
        loss_mask = has_lower & (~has_upper | (first_lower < first_upper))
        timeout_mask = ~(win_mask | loss_mask)
        timeout_win_mask = timeout_mask & (window[:, -1] > p0)
        timeout_loss_mask = timeout_mask & (window[:, -1] <= p0)
        thresholds_ok = ~np.isnan(pt) & ~np.isnan(st)
        lab = np.full(c1 - c0, LABEL_LOSS, dtype=int)
        lab[win_mask | timeout_win_mask] = LABEL_WIN
        lab[loss_mask | timeout_loss_mask] = LABEL_LOSS
        off = np.full(c1 - c0, lookahead - 1, dtype=int)
        off[win_mask] = first_upper[win_mask]
        off[loss_mask] = first_lower[loss_mask]
        exit_price = window[np.arange(c1 - c0), off]
        label[c0:c1] = lab
        exit_offset[c0:c1] = off
        exit_ret[c0:c1] = (exit_price - p0) / p0
        terminal_ret[c0:c1] = (window[:, -1] - p0) / p0
        is_timeout[c0:c1] = timeout_mask
        valid[c0:c1] = thresholds_ok
    return label, exit_offset, exit_ret, terminal_ret, is_timeout, valid

def add_barrier_labels(df, lookahead_seconds=LOOKAHEAD_SECONDS, barrier_mult=BARRIER_MULTIPLIER, timeframe=60):
    labeled = []
    for _, g in df.groupby('session_id', sort=False):
        g = g.copy()
        ret_std_1s = g['execution_price'].pct_change().rolling(window=100).std()
        horizon_std = ret_std_1s * np.sqrt(timeframe)
        g['profit_barrier'] = horizon_std * barrier_mult
        g['stop_barrier'] = horizon_std * barrier_mult
        label, exit_offset, exit_ret, terminal_ret, is_timeout, valid = _sequential_triple_barrier(
            g['execution_price'].values, g['profit_barrier'].values, g['stop_barrier'].values, lookahead_seconds
        )
        g['target'] = label
        g['bars_to_exit'] = exit_offset
        g['exit_ret'] = exit_ret
        g['terminal_ret'] = terminal_ret
        g['is_timeout'] = is_timeout
        g['_valid'] = valid
        g = g[g['_valid']].drop(columns=['_valid'])
        g['target'] = g['target'].astype(int)
        labeled.append(g)
    return pd.concat(labeled)

print('\n--- Applying barrier labels ---')
df_state_space = add_barrier_labels(df_state_space)
df_state_space = df_state_space[df_state_space['volume'] > 0]
print(f"Label distribution: {df_state_space['target'].map(CLASS_NAMES).value_counts().to_dict()}")

# Filter to hunting zone
df_training_zone = df_state_space[df_state_space[RSI_FILTER_COLUMN] < RSI_FILTER_THRESHOLD]
print(f"Hunting zone rows: {len(df_training_zone)}")

# ============================================================================
# STEP 3 — Feature Sets
# ============================================================================
drop_cols = ['target', 'price', 'open', 'high', 'low', 'close',
             'profit_barrier', 'stop_barrier', 'session_id',
             'bars_to_exit', 'exit_ret', 'terminal_ret', 'is_timeout',
             'execution_price', 'buy_vol', 'sell_vol']
all_features = [c for c in df_training_zone.columns if c not in drop_cols]

# Classify features
price_features = ['rsi_2', 'rsi_14', 'sma_14', 'spread']
orderflow_features = [f for f in all_features if f not in price_features]

print(f'\n--- Feature Sets ---')
print(f'Price features ({len(price_features)}): {price_features}')
print(f'Orderflow features ({len(orderflow_features)}): {orderflow_features}')
print(f'Total features: {len(all_features)}')

# ============================================================================
# STEP 4 — Risk-Normalized Return (R-Multiple)
# ============================================================================
normalized_return = pd.Series(index=df_training_zone.index, dtype=float)
is_timeout = df_training_zone['is_timeout']
is_win = df_training_zone['target'] == LABEL_WIN
is_loss = df_training_zone['target'] == LABEL_LOSS
normalized_return[is_win & ~is_timeout] = RR_RATIO
normalized_return[is_loss & ~is_timeout] = -1.0
normalized_return[is_timeout] = (
    df_training_zone.loc[is_timeout, 'exit_ret'] /
    (df_training_zone.loc[is_timeout, 'stop_barrier'] + EPS)
)

# ============================================================================
# STEP 5 — Walk-Forward Splitter
# ============================================================================
def purged_chronological_splits(index, n_samples, block_size, horizon_seconds, embargo_seconds=0):
    n = len(index)
    horizon = pd.Timedelta(seconds=horizon_seconds)
    embargo = pd.Timedelta(seconds=embargo_seconds)
    min_initial_train = int(n * 0.35)
    available_test_run = n - min_initial_train - block_size
    if available_test_run <= 0:
        raise ValueError("Dataset too narrow for this config.")
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

# ============================================================================
# STEP 6 — Metrics Helpers
# ============================================================================
def compute_mdd(returns):
    """Maximum drawdown from a series of cumulative returns (in R units)."""
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    mdd = np.min(drawdown)
    return mdd

def compute_sortino(returns, rf=0):
    """Sortino ratio: (mean_return - rf) / downside_std."""
    excess = np.mean(returns) - rf
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return np.nan
    return excess / np.std(downside)

# ============================================================================
# STEP 7 — Run Experiment: Full Features vs Price-Only
# ============================================================================
def run_experiment(X, y, feature_set_name, features_list, normalized_return):
    """Run walkforward with SHAP + PFI for a given feature set."""
    print(f'\n{"=" * 70}')
    print(f'EXPERIMENT: {feature_set_name}')
    print(f'Features ({len(features_list)}): {features_list}')
    print(f'{"=" * 70}')

    X_subset = X[features_list]

    splitter = purged_chronological_splits(
        X_subset.index,
        n_samples=N_RANDOM_SAMPLES,
        block_size=TEST_BLOCK_SIZE,
        horizon_seconds=LOOKAHEAD_SECONDS,
        embargo_seconds=EMBARGO_SECONDS
    )

    fold_shap_values = []
    fold_pfi_values = []
    fold_trade_returns = []

    for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(X_train) == 0 or y_train.nunique() < 2:
            print(f'  Fold {fold:02d} | skipped — invalid distribution.')
            continue

        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            num_leaves=31, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )
        clf.fit(X_train, y_train)

        # --- SHAP ---
        try:
            X_test_sampled = X_test.sample(min(1000, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_test_sampled)
            if isinstance(shap_vals, list):
                shap_array = shap_vals[1]
            elif hasattr(shap_vals, 'values'):
                shap_array = shap_vals.values[:, :, 1] if len(shap_vals.values.shape) == 3 else shap_vals.values
            else:
                shap_array = shap_vals
            fold_mean_shap = np.abs(shap_array).mean(axis=0)
            fold_shap_values.append(fold_mean_shap)
        except Exception as e:
            print(f'  Fold {fold:02d} | SHAP failed: {e}')
            fold_shap_values.append(np.full(len(features_list), np.nan))

        # --- PFI ---
        if y_test.nunique() < 2:
            fold_pfi_values.append(np.full(len(features_list), np.nan))
        else:
            try:
                pi_result = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=42,
                    scoring='roc_auc', n_jobs=1
                )
                fold_pfi_values.append(pi_result.importances_mean)
            except Exception as e:
                print(f'  Fold {fold:02d} | PFI failed: {e}')
                fold_pfi_values.append(np.full(len(features_list), np.nan))

        # --- Trade Returns ---
        if TARGET_TRADE_CLASS not in clf.classes_:
            print(f'  Fold {fold:02d} | skipped trade eval — no {target_name} in training.')
            continue

        target_idx = list(clf.classes_).index(TARGET_TRADE_CLASS)
        y_proba = clf.predict_proba(X_test)[:, target_idx]
        preds = (y_proba >= 0.85).astype(int)  # Use fixed 85% threshold for trade evaluation

        # Non-overlapping filter
        signal_indices = np.where(preds == 1)[0]
        non_overlapping_idx = []
        next_available_time = pd.Timestamp.min
        for idx in signal_indices:
            actual_time = X_test.index[idx]
            if actual_time < next_available_time:
                continue
            non_overlapping_idx.append(actual_time)
            trade_duration = int(df_training_zone.loc[actual_time, 'bars_to_exit']) + 1
            next_available_time = actual_time + pd.Timedelta(seconds=trade_duration)

        if len(non_overlapping_idx) > 0:
            traded_idx = pd.Index(non_overlapping_idx)
            traded_returns = normalized_return.loc[traded_idx].copy()
            if INVERT_RETURNS:
                traded_returns = -traded_returns
            fold_trade_returns.append(traded_returns.values)

        print(f'  Fold {fold:02d} | Train: {len(X_train):<6} | Test: {len(X_test):<6} | Purged: {n_purged:<4} | Trades: {len(non_overlapping_idx):<4}')

    # --- Aggregate Results ---
    result = {}

    # SHAP summary
    if fold_shap_values:
        mean_shap = np.nanmean(fold_shap_values, axis=0)
        std_shap = np.nanstd(fold_shap_values, axis=0)
        result['shap_df'] = pd.DataFrame({
            'Feature': features_list,
            'Mean_SHAP': mean_shap,
            'Std_SHAP': std_shap
        }).sort_values('Mean_SHAP', ascending=False)
    else:
        result['shap_df'] = pd.DataFrame({'Feature': features_list, 'Mean_SHAP': np.nan, 'Std_SHAP': np.nan})

    # PFI summary
    if fold_pfi_values:
        mean_pfi = np.nanmean(fold_pfi_values, axis=0)
        std_pfi = np.nanstd(fold_pfi_values, axis=0)
        pos_count = np.sum(np.array(fold_pfi_values) > 0, axis=0)
        neg_count = np.sum(np.array(fold_pfi_values) < 0, axis=0)
        result['pfi_df'] = pd.DataFrame({
            'Feature': features_list,
            'Mean_PFI_AUC_Drop': mean_pfi,
            'Std_PFI': std_pfi,
            'Positive_PFI_Count': pos_count,
            'Negative_PFI_Count': neg_count
        }).sort_values('Mean_PFI_AUC_Drop', ascending=False)
    else:
        result['pfi_df'] = pd.DataFrame({'Feature': features_list, 'Mean_PFI_AUC_Drop': np.nan})

    # Trade performance
    if fold_trade_returns:
        all_returns = np.concatenate(fold_trade_returns)
        result['total_trades'] = len(all_returns)
        result['total_R'] = np.sum(all_returns)
        result['mean_R_per_trade'] = np.mean(all_returns)
        # profit-weighted precision (= renamed expectancy)
        win_mask = all_returns > 0
        loss_mask = all_returns < 0
        n_wins = win_mask.sum()
        n_losses = loss_mask.sum()
        win_rate = n_wins / len(all_returns) if len(all_returns) > 0 else 0
        avg_win = all_returns[win_mask].mean() if n_wins > 0 else 0
        avg_loss = all_returns[loss_mask].mean() if n_losses > 0 else 0
        rr = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan
        result['win_rate'] = win_rate
        result['avg_win_R'] = avg_win
        result['avg_loss_R'] = abs(avg_loss) if avg_loss != 0 else 0
        result['rr'] = rr
        result['profit_weighted_precision'] = np.mean(all_returns)  # = expectancy renamed
        # MDD
        result['mdd'] = compute_mdd(all_returns)
        # Sortino
        result['sortino'] = compute_sortino(all_returns)
    else:
        result['total_trades'] = 0
        result['total_R'] = 0
        result['mean_R_per_trade'] = 0
        result['win_rate'] = 0
        result['avg_win_R'] = 0
        result['avg_loss_R'] = 0
        result['rr'] = np.nan
        result['profit_weighted_precision'] = 0
        result['mdd'] = 0
        result['sortino'] = np.nan

    return result

# ============================================================================
# STEP 8 — RUN BOTH EXPERIMENTS
# ============================================================================
X = df_training_zone[all_features]
y = df_training_zone['target']

# --- Experiment A: Full feature set ---
results_full = run_experiment(X, y, 'FULL MODEL (Price + Orderflow)', all_features, normalized_return)

# --- Experiment B: Price features only ---
X_price = df_training_zone[price_features]
# SMA_14 depends on RSI_14 so we need to check it exists
price_features_available = [f for f in price_features if f in all_features]
results_price = run_experiment(X, y, 'PRICE-ONLY MODEL (No Orderflow)', price_features_available, normalized_return)

# ============================================================================
# STEP 9 — COMPARISON REPORT
# ============================================================================
print('\n\n')
print('=' * 70)
print('FINAL MULTICOLLINEARITY ANALYSIS')
print('=' * 70)

# --- Table A: Feature Importance Comparison ---
print('\n--- SHAP Importance: Full Model vs Price-Only Model ---')
print(f"{'Feature':<20} | {'Full SHAP':<12} | {'Full PFI':<12} | {'Price-Only SHAP':<16} | {'Price-Only PFI':<16}")

# Build comparison table for price features
for feat in price_features_available:
    full_shap = results_full['shap_df'].loc[results_full['shap_df']['Feature'] == feat, 'Mean_SHAP'].values
    full_pfi = results_full['pfi_df'].loc[results_full['pfi_df']['Feature'] == feat, 'Mean_PFI_AUC_Drop'].values
    price_shap = results_price['shap_df'].loc[results_price['shap_df']['Feature'] == feat, 'Mean_SHAP'].values
    price_pfi = results_price['pfi_df'].loc[results_price['pfi_df']['Feature'] == feat, 'Mean_PFI_AUC_Drop'].values

    fs = f'{full_shap[0]:.5f}' if len(full_shap) > 0 else 'N/A'
    fp = f'{full_pfi[0]:.5f}' if len(full_pfi) > 0 else 'N/A'
    ps = f'{price_shap[0]:.5f}' if len(price_shap) > 0 else 'N/A'
    pp = f'{price_pfi[0]:.5f}' if len(price_pfi) > 0 else 'N/A'
    print(f'{feat:<20} | {fs:<12} | {fp:<12} | {ps:<16} | {pp:<16}')

# --- Table B: Multi-collinearity Diagnoses ---
print('\n--- Multicollinearity Diagnosis ---')
print('If price feature SHAP/PFI DROPS when orderflow is removed:')
print('  -> Orderflow predictiveness was partially absorbed by price features (multicollinearity).')
print('If price feature SHAP/PFI STAYS THE SAME when orderflow is removed:')
print('  -> Orderflow has independent predictive power beyond price features.')
print()

for feat in price_features_available:
    full_shap = results_full['shap_df'].loc[results_full['shap_df']['Feature'] == feat, 'Mean_SHAP'].values
    full_pfi = results_full['pfi_df'].loc[results_full['pfi_df']['Feature'] == feat, 'Mean_PFI_AUC_Drop'].values
    price_shap = results_price['shap_df'].loc[results_price['shap_df']['Feature'] == feat, 'Mean_SHAP'].values
    price_pfi = results_price['pfi_df'].loc[results_price['pfi_df']['Feature'] == feat, 'Mean_PFI_AUC_Drop'].values

    fs = full_shap[0] if len(full_shap) > 0 else 0
    fp = full_pfi[0] if len(full_pfi) > 0 else 0
    ps = price_shap[0] if len(price_shap) > 0 else 0
    pp = price_pfi[0] if len(price_pfi) > 0 else 0

    # SHAP change
    shap_change = (ps - fs) / (abs(fs) + EPS) * 100 if abs(fs) > 1e-12 else float('inf')
    pfi_change = (pp - fp) / (abs(fp) + EPS) * 100 if abs(fp) > 1e-12 else float('inf')

    diagnosis = ''
    if shap_change < -20:
        diagnosis += 'HIGH multicollinearity (SHAP ↓ >20%) '
    elif shap_change > 20 and fp > 0:
        diagnosis += 'Orderflow was NOISE-diluting SHAP (SHAP ↑) '
    else:
        diagnosis += 'Low multicollinearity effect '

    if pfi_change < -20 and pp <= 0:
        diagnosis += '| PFI confirms: price lost predictiveness'
    elif pfi_change < -20 and pp > 0:
        diagnosis += '| PFI confirms but price still somewhat predictive'
    elif pfi_change > 20:
        diagnosis += '| Orderflow was damping price PFI'
    else:
        diagnosis += '| PFI stable'

    print(f'{feat:<20} | SHAP: {fs:.5f} -> {ps:.5f} ({shap_change:+.1f}%) | PFI: {fp:.5f} -> {pp:.5f} ({pfi_change:+.1f}%)')
    print(f'  {" ":<18} Diagnosis: {diagnosis}')

# --- Table C: Orderflow Features Summary (Full Model Only) ---
print('\n--- Orderflow Feature Contribution (Full Model) ---')
orderflow_shap = results_full['shap_df'][results_full['shap_df']['Feature'].isin(orderflow_features)]
orderflow_pfi = results_full['pfi_df'][results_full['pfi_df']['Feature'].isin(orderflow_features)]
print(orderflow_shap.to_string(index=False))

# --- Table D: Strategy Performance Comparison ---
print('\n--- Strategy Performance Comparison ---')
print(f"{'Metric':<35} | {'Full Model':<15} | {'Price-Only':<15}")
print('-' * 70)
for metric in ['total_trades', 'total_R', 'mean_R_per_trade', 'profit_weighted_precision',
               'win_rate', 'avg_win_R', 'avg_loss_R', 'rr', 'mdd', 'sortino']:
    fv = results_full.get(metric, np.nan)
    pv = results_price.get(metric, np.nan)
    if isinstance(fv, float):
        f_str = f'{fv:.4f}' if abs(fv) < 100 else f'{fv:.2f}'
        p_str = f'{pv:.4f}' if abs(pv) < 100 else f'{pv:.2f}'
    else:
        f_str = str(fv)
        p_str = str(pv)
    print(f'{metric:<35} | {f_str:<15} | {p_str:<15}')

# --- Table E: Full PFI Detail ---
print('\n--- Full Model: All Features SHAP & PFI ---')
print(results_full['shap_df'].to_string(index=False))
print()
print(results_full['pfi_df'].to_string(index=False))

# --- Table F: Price-Only PFI Detail ---
print('\n--- Price-Only Model: SHAP & PFI ---')
print(results_price['shap_df'].to_string(index=False))
print()
print(results_price['pfi_df'].to_string(index=False))

# ============================================================================
# STEP 10 — Final Conclusion
# ============================================================================
print('\n')
print('=' * 70)
print('CONCLUSION')
print('=' * 70)

# Determine overall diagnosis
price_shap_full = np.mean([results_full['shap_df'].loc[results_full['shap_df']['Feature'] == f, 'Mean_SHAP'].values[0]
                            for f in price_features_available if len(results_full['shap_df'].loc[results_full['shap_df']['Feature'] == f, 'Mean_SHAP'].values) > 0])
price_shap_priceonly = np.mean([results_price['shap_df'].loc[results_price['shap_df']['Feature'] == f, 'Mean_SHAP'].values[0]
                                 for f in price_features_available if len(results_price['shap_df'].loc[results_price['shap_df']['Feature'] == f, 'Mean_SHAP'].values) > 0])

price_pfi_full = np.mean([results_full['pfi_df'].loc[results_full['pfi_df']['Feature'] == f, 'Mean_PFI_AUC_Drop'].values[0]
                           for f in price_features_available if len(results_full['pfi_df'].loc[results_full['pfi_df']['Feature'] == f, 'Mean_PFI_AUC_Drop'].values) > 0])
price_pfi_priceonly = np.mean([results_price['pfi_df'].loc[results_price['pfi_df']['Feature'] == f, 'Mean_PFI_AUC_Drop'].values[0]
                                for f in price_features_available if len(results_price['pfi_df'].loc[results_price['pfi_df']['Feature'] == f, 'Mean_PFI_AUC_Drop'].values) > 0])

print(f'\nAverage Price Feature SHAP: Full={price_shap_full:.5f} vs Price-Only={price_shap_priceonly:.5f}')
print(f'Average Price Feature PFI:  Full={price_pfi_full:.5f} vs Price-Only={price_pfi_priceonly:.5f}')

shap_drop_pct = (price_shap_priceonly - price_shap_full) / (abs(price_shap_full) + EPS) * 100
pfi_drop_pct = (price_pfi_priceonly - price_pfi_full) / (abs(price_pfi_full) + EPS) * 100

print(f'\nSHAP change when removing orderflow: {shap_drop_pct:+.1f}%')
print(f'PFI change when removing orderflow:  {pfi_drop_pct:+.1f}%')

if shap_drop_pct < -15 and pfi_drop_pct < -15:
    print('\nDiagnosis: STRONG multicollinearity confirmed.')
    print('Orderflow predictive power was absorbed into price features.')
    print('Removing orderflow significantly reduces price feature importance.')
elif shap_drop_pct < -5 or pfi_drop_pct < -5:
    print('\nDiagnosis: MODERATE multicollinearity present.')
    print('Some orderflow information was shared with price features.')
else:
    print('\nDiagnosis: WEAK or NO multicollinearity.')
    print('Orderflow had minimal shared information with price features.')
    print('Price feature predictiveness is largely independent of orderflow.')

print(f'\nStrategy performance drop (total R): Full={results_full["total_R"]:.2f}R vs Price-Only={results_price["total_R"]:.2f}R')
print(f'Strategy MDD: Full={results_full["mdd"]:.2f}R vs Price-Only={results_price["mdd"]:.2f}R')
print(f'Strategy Sortino: Full={results_full["sortino"]:.4f} vs Price-Only={results_price["sortino"]:.4f}')
print(f'Profit-Weighted Precision (ex-expectancy): Full={results_full["profit_weighted_precision"]:.4f}R vs Price-Only={results_price["profit_weighted_precision"]:.4f}R')
print('=' * 70)
print('PASS: Full multicollinearity analysis completed without errors!')