import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from utils.indicators import *
from utils.engine import *

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

from utils.engine import *

# ============================================================================
# CONFIG
# ============================================================================
LOOKAHEAD_SECONDS    = 240     # how far forward we walk the price path looking for a barrier touch
BARRIER_MULTIPLIER   = 1.0     # profit/stop barrier = multiplier * horizon-scaled vol
RSI_FILTER_COLUMN    = 'rsi_14'
RSI_FILTER_THRESHOLD = 30
EPS                  = 1e-9    # guards against division by zero
BARRIER_CHUNK_SIZE   = 50_000  # rows processed per chunk in the vectorized barrier walk (memory control)

# --- Chronological Validation Settings ---
N_RANDOM_SAMPLES    = 20      # Number of sequential walk-forward folds to test
TEST_BLOCK_SIZE     = 100   # Number of rows per test chunk
EMBARGO_SECONDS      = LOOKAHEAD_SECONDS

# --- 2-Class Binary Label Encoding (INVERTED FOR SHORT STRATEGY) ---
# WIN  = Profit barrier (LOWER) touched FIRST, OR timeout occurred with price < entry
# LOSS = Stop barrier (UPPER) touched FIRST, OR timeout occurred with price >= entry
LABEL_SHORT    = 0
LABEL_LONG     = 1
CLASS_NAMES   = {LABEL_SHORT: 'LOSS', LABEL_LONG: 'WIN'}


# ============================================================================
# STEP 1 — Per-session feature pipeline
# ============================================================================
def load_and_build_state_space(orderbook_path, trades_path, session_id):
    df_orderbook = pd.read_csv(orderbook_path)
    df_trades = pd.read_csv(trades_path)

    df_orderbook['datetime'] = pd.to_datetime(df_orderbook['timestamp'], unit='ms')
    df_trades['datetime'] = pd.to_datetime(df_trades['timestamp'], unit='ms')

    df_orderbook = df_orderbook.set_index('datetime').sort_index()
    df_trades = df_trades.set_index('datetime').sort_index()

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
        'largest_buyer': df_trades_1m['largest_buyer'] * 100 / (df_trades_1m['volume'] + EPS),
        'largest_seller': df_trades_1m['largest_seller'] * 100 / (df_trades_1m['volume'] + EPS),
        'OFI_LOB': 100 * (df_ob_1m['bid_amount'] - df_ob_1m['ask_amount']) / (df_ob_1m['bid_amount'] + df_ob_1m['ask_amount'] + EPS),
        'OFI_AGG': 100 * (df_trades_1m['agg_buyer'] - df_trades_1m['agg_seller']) / (df_trades_1m['agg_buyer'] + df_trades_1m['agg_seller'] + EPS),
        'CVD_raw': calculate_cvd(df_trades_1m['agg_buyer'], df_trades_1m['agg_seller']),
        'candle_type': abs(df_ob_1m['close'] - df_ob_1m['open']) / (abs(df_ob_1m['high'] - df_ob_1m['low']) + EPS),
        'candle_size': (df_ob_1m['high'] - df_ob_1m['low']) / ((df_ob_1m['high'] - df_ob_1m['low']).rolling(14).mean() + EPS),
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
    df_macro_shifted = df_price_1m.shift(1)

    df_state_space = df_micro_shifted.join(df_macro_shifted, how='left').ffill()
    df_state_space = df_state_space.dropna()
    df_state_space['session_id'] = session_id

    return df_state_space


# ============================================================================
# STEP 2 — Build combined state space from all sessions
# ============================================================================

session_0 = load_and_build_state_space(
    "./csv/BTCUSDT_orderbook.csv", "./csv/BTCUSDT_trades.csv", session_id=0
)
session_1 = load_and_build_state_space(
    "./csv/BTCUSDT_orderbook_2.csv", "./csv/BTCUSDT_trades_2.csv", session_id=1
)
session_2 = load_and_build_state_space(
    "./csv/BTCUSDT_orderbook_3.csv", "./csv/BTCUSDT_trades_3.csv", session_id=1
)

df_state_space = pd.concat([session_0, session_1, session_2]).sort_index()

# ============================================================================
# STEP 3 — Barrier labeling (INVERTED FOR SHORT STRATEGY)
# ============================================================================
def _bidirectional_barrier(price, barrier_pct, lookahead, chunk_size=BARRIER_CHUNK_SIZE):
    """
    Calculates unified labels (0 for Short win, 1 for Long win) and the exact 
    exit price for every row so we can calculate dynamic R-Multiples later.
    """
    price = np.asarray(price, dtype=float)
    barrier_pct = np.asarray(barrier_pct, dtype=float)
    n = len(price)

    label = np.full(n, -1, dtype=int)
    exit_price_arr = np.full(n, np.nan)
    is_timeout = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)

    usable_n = n - lookahead 
    if usable_n <= 0:
        return label, exit_price_arr, is_timeout, valid

    for c0 in range(0, usable_n, chunk_size):
        c1 = min(c0 + chunk_size, usable_n)

        p0 = price[c0:c1]
        bp = barrier_pct[c0:c1] 

        # Symmetric absolute boundaries based on your volatility calculation
        upper = p0 * (1.0 + bp)
        lower = p0 * (1.0 - bp)

        window = sliding_window_view(price[c0 + 1: c1 + lookahead], lookahead) 

        hit_upper = window >= upper[:, None]
        hit_lower = window <= lower[:, None]

        has_upper = hit_upper.any(axis=1)
        has_lower = hit_lower.any(axis=1)
        first_upper = np.argmax(hit_upper, axis=1) 
        first_lower = np.argmax(hit_lower, axis=1) 

        # Win conditions for BOTH directions
        long_win = has_upper & (~has_lower | (first_upper < first_lower))
        short_win = has_lower & (~has_upper | (first_lower < first_upper))
        
        # Timeouts resolved strictly by where price ended relative to entry
        timeout_mask = ~(long_win | short_win)
        timeout_long = timeout_mask & (window[:, -1] >= p0) 
        timeout_short = timeout_mask & (window[:, -1] < p0)

        # Assign Labels (1 = Market goes UP, 0 = Market goes DOWN)
        lab = np.full(c1 - c0, -1, dtype=int)
        lab[long_win | timeout_long] = LABEL_LONG
        lab[short_win | timeout_short] = LABEL_SHORT

        # Find exact exit prices for R-Multiple math
        off = np.full(c1 - c0, lookahead - 1, dtype=int) 
        off[long_win] = first_upper[long_win]
        off[short_win] = first_lower[short_win]
        
        # Array of exact price we exited at (either barrier hit or timeout end)
        exit_p = window[np.arange(c1 - c0), off]

        label[c0:c1] = lab
        exit_price_arr[c0:c1] = exit_p
        is_timeout[c0:c1] = timeout_mask
        valid[c0:c1] = ~np.isnan(bp)

    return label, exit_price_arr, is_timeout, valid

def add_barrier_labels(df, lookahead_seconds=LOOKAHEAD_SECONDS, barrier_mult=BARRIER_MULTIPLIER):
    labeled = []
    for _, g in df.groupby('session_id', sort=False):
        g = g.copy()
        ret_std_1s = g['price'].pct_change().rolling(window=100).std()
        
        # Single barrier percentage based on volatility (symmetric)
        g['barrier_pct'] = ret_std_1s * np.sqrt(lookahead_seconds) * barrier_mult

        label, exit_price, is_timeout, valid = _bidirectional_barrier(
            g['price'].values, g['barrier_pct'].values, lookahead_seconds
        )

        g['target'] = label
        g['exit_price'] = exit_price 
        g['is_timeout'] = is_timeout
        g['_valid'] = valid

        g = g[g['_valid']].drop(columns=['_valid'])
        labeled.append(g)

    return pd.concat(labeled)

# Build features and subset to hunting zone
df_state_space = add_barrier_labels(df_state_space)
df_training_zone = df_state_space[df_state_space[RSI_FILTER_COLUMN] < RSI_FILTER_THRESHOLD].copy()

# ============================================================================
# REVISED STEP 4 — Dual R-Multiple Bookkeeping (Apply to Full Dataset)
# ============================================================================
barrier_price_dist = df_state_space['price'] * df_state_space['barrier_pct']

df_state_space['R_LONG'] = (df_state_space['exit_price'] - df_state_space['price']) / (barrier_price_dist + EPS)
df_state_space['R_SHORT'] = (df_state_space['price'] - df_state_space['exit_price']) / (barrier_price_dist + EPS)

df_state_space.loc[~df_state_space['is_timeout'], 'R_LONG'] = np.clip(df_state_space.loc[~df_state_space['is_timeout'], 'R_LONG'], -1.0, 1.0)
df_state_space.loc[~df_state_space['is_timeout'], 'R_SHORT'] = np.clip(df_state_space.loc[~df_state_space['is_timeout'], 'R_SHORT'], -1.0, 1.0)


# ============================================================================
# REVISED STEP 5 — Features and Target Setup (Full Dataset)
# ============================================================================
drop_cols = ['target', 'price', 'open', 'high', 'low', 'close', 'barrier_pct', 
             'session_id', 'exit_price', 'is_timeout', 'R_LONG', 'R_SHORT']
features = [c for c in df_state_space.columns if c not in drop_cols]

# Notice we use df_state_space here, NOT df_training_zone
X = df_state_space[features]
y = df_state_space['target']


# ============================================================================
# REVISED STEP 6 — Walk-Forward Execution
# ============================================================================
def PurgedTimeSeriesSplit(index, n_samples, block_size, horizon_seconds, embargo_seconds=0):
    n = len(index)
    horizon = pd.Timedelta(seconds=horizon_seconds)
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

        purge_mask = (train_times + horizon >= test_start_time)
        train_idx = possible_train_idx[~purge_mask]

        n_purged = len(possible_train_idx) - len(train_idx)
        yield train_idx, test_idx, n_purged


print(f"\n--- Starting Bidirectional Validation Folds ---")

thresholds = np.arange(0.05, 0.40, 0.05)

long_metrics = {t: {'trades': 0, 'wins': 0, 'r_returns': []} for t in thresholds}
short_metrics = {t: {'trades': 0, 'wins': 0, 'r_returns': []} for t in thresholds}
long_trade_log = {t: [] for t in thresholds}
short_trade_log = {t: [] for t in thresholds}

# (Replace 'PurgedTimeSeriesSplit' with the actual class name inside your utils.engine)
splitter = PurgedTimeSeriesSplit(
    index=X.index,                      # FIX 1: Pass the actual datetime index, not the whole DataFrame
    n_samples=N_RANDOM_SAMPLES, 
    block_size=TEST_BLOCK_SIZE,         # FIX 2: Corrected argument name to match the function definition
    horizon_seconds=LOOKAHEAD_SECONDS,  # FIX 3: Added the missing required argument
    embargo_seconds=EMBARGO_SECONDS
)

thresholds = np.arange(0.05, 0.40, 0.05)

for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
    # 1. Split the raw, continuous data first
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]

    # 2. Apply the RSI filter ON THE SPLITS to create the "Hunting Zone"
    train_mask = X_train_raw[RSI_FILTER_COLUMN] < RSI_FILTER_THRESHOLD
    test_mask = X_test_raw[RSI_FILTER_COLUMN] < RSI_FILTER_THRESHOLD

    X_train = X_train_raw[train_mask]
    y_train = y_train_raw[train_mask]
    X_test = X_test_raw[test_mask]
    y_test = y_test_raw[test_mask]

    # 3. Guard rails: skip if the filter wiped out the data for this specific fold
    if len(X_train) == 0 or y_train.nunique() < 2:
        print(f"Fold {fold} Skipped: Insufficient data after RSI filter. Rows: {len(X_train)}, Classes: {y_train.nunique()}")
        continue

    # Train LightGBM
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, 
                             class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X_train, y_train)

    long_idx = list(clf.classes_).index(LABEL_LONG)
    short_idx = list(clf.classes_).index(LABEL_SHORT)
    
    # If the test set has zero rows after filtering, skip prediction
    if len(X_test) == 0:
        continue

    p_long = clf.predict_proba(X_test)[:, long_idx]
    p_short = clf.predict_proba(X_test)[:, short_idx]
    
    for t in thresholds:
        # LONGS
        long_signals = (p_long >= t)
        if long_signals.sum() > 0:
            traded_idx = X_test.index[long_signals]
            actual_targets = y_test.loc[traded_idx]
            actual_r = df_state_space.loc[traded_idx, 'R_LONG'] # Use full dataframe
            
            long_metrics[t]['trades'] += len(traded_idx)
            long_metrics[t]['wins'] += (actual_targets == LABEL_LONG).sum()
            long_metrics[t]['r_returns'].extend(actual_r.tolist())
            long_trade_log[t].extend(zip(traded_idx, actual_r.values))

        # SHORTS
        short_signals = (p_short >= t)
        if short_signals.sum() > 0:
            traded_idx = X_test.index[short_signals]
            actual_targets = y_test.loc[traded_idx]
            actual_r = df_state_space.loc[traded_idx, 'R_SHORT'] # Use full dataframe
            
            short_metrics[t]['trades'] += len(traded_idx)
            short_metrics[t]['wins'] += (actual_targets == LABEL_SHORT).sum()
            short_metrics[t]['r_returns'].extend(actual_r.tolist())
            short_trade_log[t].extend(zip(traded_idx, actual_r.values))

    print(f"Fold {fold} Complete | Mean P(Long): {p_long.mean():.4f} | Max P(Long): {p_long.max():.4f}")

# ============================================================================
# STEP 7 — Generate the TWO SEPARATE REPORTS
# ============================================================================
def generate_report(metrics_dict, direction_name):
    print(f"\n" + "="*80)
    print(f" {direction_name.upper()} STRATEGY REPORT (Precision == Win Rate)")
    print("="*80)
    print(f"{'Confidence':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Avg RR per Trade':<18} | {'Total PnL'}")
    print("-" * 80)
    
    for t in thresholds:
        stats = metrics_dict[t]
        trades = stats['trades']
        if trades == 0:
            print(f"P({direction_name}) >= {t:.2f} | {0:<8} | {'n/a':<10} | {'n/a':<18} | 0.00R")
            continue
            
        win_rate = stats['wins'] / trades
        r_arr = np.array(stats['r_returns'])
        avg_r = r_arr.mean()
        std_r = r_arr.std()
        total_r = r_arr.sum()
        
        print(f"P({direction_name}) >= {t:.2f} | {trades:<8} | {win_rate:>6.2%}   | {avg_r:>5.2f}R ± {std_r:<5.2f}R   | {total_r:+.2f}R")

# Output the separate reports
generate_report(long_metrics, "Long")
generate_report(short_metrics, "Short")

# ============================================================================
# STEP 8 — Combined PnL Charting
# ============================================================================
plt.figure(figsize=(15, 6))

# Subplot 1: Long PnL
plt.subplot(1, 2, 1)
for i, t in enumerate(thresholds):
    if len(long_trade_log[t]) > 0:
        df_log = pd.DataFrame(long_trade_log[t], columns=['time', 'r']).sort_values('time')
        plt.plot(df_log['time'], df_log['r'].cumsum(), label=f'>= {t:.2f}')
plt.axhline(0, color='black', linestyle='--')
plt.title('LONG Out-of-Sample PnL')
plt.ylabel('Cumulative Return (R)')
plt.legend()

# Subplot 2: Short PnL
plt.subplot(1, 2, 2)
for i, t in enumerate(thresholds):
    if len(short_trade_log[t]) > 0:
        df_log = pd.DataFrame(short_trade_log[t], columns=['time', 'r']).sort_values('time')
        plt.plot(df_log['time'], df_log['r'].cumsum(), label=f'>= {t:.2f}')
plt.axhline(0, color='black', linestyle='--')
plt.title('SHORT Out-of-Sample PnL')
plt.legend()

plt.tight_layout()
plt.savefig('bidirectional_pnl.png', dpi=150)
print("\nSaved bidirectional dual-chart to bidirectional_pnl.png")

