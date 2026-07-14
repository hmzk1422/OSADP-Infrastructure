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

# ============================================================================
# CONFIG
# ============================================================================
LOOKAHEAD_SECONDS    = 240     # how far forward we walk the price path looking for a barrier touch
BARRIER_MULTIPLIER   = 1.0     # profit/stop barrier = multiplier * horizon-scaled vol
RSI_FILTER_COLUMN    = 'rsi_2'
RSI_FILTER_THRESHOLD = 30
EPS                  = 1e-9    # guards against division by zero
BARRIER_CHUNK_SIZE   = 50_000  # rows processed per chunk in the vectorized barrier walk (memory control)

# --- Chronological Validation Settings ---
N_RANDOM_SAMPLES    = 15      # Number of sequential walk-forward folds to test
TEST_BLOCK_SIZE     = 10000   # Number of rows per test chunk
EMBARGO_SECONDS      = LOOKAHEAD_SECONDS

# --- 2-Class Binary Label Encoding ---
# WIN  = Profit barrier touched FIRST, OR timeout occurred with price > entry
# LOSS = Stop barrier touched FIRST, OR timeout occurred with price <= entry
LABEL_LOSS    = 0
LABEL_WIN     = 1
CLASS_NAMES   = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}
RR_RATIO      = 1.5

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
df_state_space = load_and_build_state_space("./csv/XAUTUSDT_orderbook.csv", "./csv/XAUTUSDT_trades.csv", 0)


# ============================================================================
# STEP 3 — Barrier labeling (sequential / path-dependent binary-barrier)
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

        upper = p0 * (RR_RATIO + pt)
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
        
        # Determine timeouts and resolve them based on terminal price vs entry
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


def add_barrier_labels(df, lookahead_seconds=LOOKAHEAD_SECONDS, barrier_mult=BARRIER_MULTIPLIER):
    labeled = []
    for _, g in df.groupby('session_id', sort=False):
        g = g.copy()
        ret_std_1s = g['price'].pct_change().rolling(window=100).std()
        horizon_std = ret_std_1s * np.sqrt(lookahead_seconds)

        g['profit_barrier'] = horizon_std * barrier_mult
        g['stop_barrier'] = horizon_std * barrier_mult

        label, exit_offset, exit_ret, terminal_ret, is_timeout, valid = _sequential_triple_barrier(
            g['price'].values, g['profit_barrier'].values, g['stop_barrier'].values, lookahead_seconds
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

df_state_space = add_barrier_labels(df_state_space)

label_counts = df_state_space['target'].map(CLASS_NAMES).value_counts()
print(f"Label distribution (full dataset): {label_counts.to_dict()}")

df_training_zone = df_state_space[df_state_space[RSI_FILTER_COLUMN] < RSI_FILTER_THRESHOLD]
print(f"Total rows in hunting zone: {len(df_training_zone)}")
print(f"Label distribution (hunting zone): {df_training_zone['target'].map(CLASS_NAMES).value_counts().to_dict()}")


# ============================================================================
# STEP 4 — Features / target configuration
# ============================================================================
drop_cols = ['target', 'price', 'open', 'high', 'low', 'close',
             'profit_barrier', 'stop_barrier', 'session_id',
             'bars_to_exit', 'exit_ret', 'terminal_ret', 'is_timeout']
features = [c for c in df_training_zone.columns if c not in drop_cols]

X = df_training_zone[features]
y = df_training_zone['target']

# ============================================================================
# STEP 5 — Risk-Normalized Return (R-Multiple Bookkeeping)
# ----------------------------------------------------------------------------
# Accounts for timeouts by mapping barrier hits to +/- 1.0R and timeouts to
# their exact fractional R (e.g. +0.4R or -0.2R) relative to the stop distance.
# ============================================================================
normalized_return = pd.Series(index=df_training_zone.index, dtype=float)

is_timeout = df_training_zone['is_timeout']
is_win = df_training_zone['target'] == LABEL_WIN
is_loss = df_training_zone['target'] == LABEL_LOSS

# Exact barrier hits resolve to exactly +1R or -1R
normalized_return[is_win & ~is_timeout] = RR_RATIO
normalized_return[is_loss & ~is_timeout] = -1.0

# Timeouts resolve to their realized fractional R 
normalized_return[is_timeout] = (
    df_training_zone.loc[is_timeout, 'exit_ret'] /
    (df_training_zone.loc[is_timeout, 'stop_barrier'] + EPS)
)


# ============================================================================
# STEP 6 — Purged Chronological Walk-Forward Splitting
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

splitter = purged_chronological_splits(
    X.index,
    n_samples=N_RANDOM_SAMPLES,
    block_size=TEST_BLOCK_SIZE,
    horizon_seconds=LOOKAHEAD_SECONDS,
    embargo_seconds=EMBARGO_SECONDS
)

print(f"\n--- Starting {N_RANDOM_SAMPLES} Chronological Walk-Forward Validation Folds (Size: {TEST_BLOCK_SIZE}) ---")

thresholds_to_track = np.arange(0.50, 0.96, 0.05)
fold_metrics = {t: {'precisions': [], 'win_rates': [], 'rrs': [], 'trades': []} for t in thresholds_to_track}
fold_importances = []
trade_log = {t: [] for t in thresholds_to_track}       
fold_detail_records = []                                

class_labels_ordered = [LABEL_LOSS, LABEL_WIN]
class_metrics_by_label = {lbl: {'precision': [], 'recall': []} for lbl in class_labels_ordered}
class_metrics_records = []
all_y_test_pooled = []
all_y_pred_pooled = []

for fold, (train_idx, test_idx, n_purged) in enumerate(splitter, start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if len(X_train) == 0 or y_train.nunique() < 2:
        print(f"Fold {fold} | skipped — invalid training data distribution.")
        continue

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    

    clf.fit(X_train, y_train)

    y_pred_argmax = clf.predict(X_test)
    prec_per_class = precision_score(y_test, y_pred_argmax, labels=class_labels_ordered, average=None, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred_argmax, labels=class_labels_ordered, average=None, zero_division=0)

    for lbl, p, r in zip(class_labels_ordered, prec_per_class, rec_per_class):
        class_metrics_by_label[lbl]['precision'].append(p)
        class_metrics_by_label[lbl]['recall'].append(r)
        class_metrics_records.append({
            'fold': fold,
            'class': CLASS_NAMES[lbl],
            'precision': p,
            'recall': r,
            'support': int((y_test == lbl).sum()),
        })

    all_y_test_pooled.append(y_test.values)
    all_y_pred_pooled.append(y_pred_argmax)

    if LABEL_WIN not in clf.classes_:
        print(f"Fold {fold} | skipped threshold scoring — no WIN samples present in this fold's training data.")
        continue

    win_idx = list(clf.classes_).index(LABEL_WIN)
    y_proba = clf.predict_proba(X_test)[:, win_idx]
    
    for t in thresholds_to_track:
        preds = (y_proba >= t).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        num_trades = int(preds.sum())

        if num_trades > 0:
            fold_metrics[t]['precisions'].append(prec)

            traded_idx = X_test.index[preds == 1]
            traded_targets = y_test.loc[traded_idx]
            traded_returns = normalized_return.loc[traded_idx]

            trade_log[t].extend(zip(traded_idx, traded_returns.values))

            num_win = int((traded_targets == LABEL_WIN).sum())
            num_loss = int((traded_targets == LABEL_LOSS).sum())

            win_rate = num_win / (num_win + num_loss) if (num_win + num_loss) > 0 else np.nan

            avg_win_ret = traded_returns[traded_targets == LABEL_WIN].mean() if num_win > 0 else np.nan
            avg_loss_ret = traded_returns[traded_targets == LABEL_LOSS].mean() if num_loss > 0 else np.nan
            rr = (avg_win_ret / abs(avg_loss_ret)) if (num_win > 0 and num_loss > 0 and avg_loss_ret != 0) else np.nan

            fold_metrics[t]['win_rates'].append(win_rate)
            fold_metrics[t]['rrs'].append(rr)

            fold_detail_records.append({
                'threshold': round(float(t), 2),
                'fold': fold,
                'trades': num_trades,
                'wins': num_win,
                'losses': num_loss,
                'precision': prec,
                'win_rate': win_rate,
                'rr': rr,
                'avg_return': traded_returns.mean(),
            })

        fold_metrics[t]['trades'].append(num_trades)

    fold_importances.append(clf.feature_importances_)
    print(f"Fold {fold:02d} | Train Rows: {len(X_train):<6} | Tested on forward {TEST_BLOCK_SIZE} rows. Purged: {n_purged}")


# ============================================================================
# STEP 8 — Model Class-Level Precision & Recall (argmax predictions)
# ============================================================================
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

if class_metrics_records:
    class_metrics_df = pd.DataFrame(class_metrics_records)
    class_metrics_df.to_csv('class_performance_detail.csv', index=False)
    print("Saved full per-fold class precision/recall detail to class_performance_detail.csv")

if all_y_test_pooled:
    pooled_y_test = np.concatenate(all_y_test_pooled)
    pooled_y_pred = np.concatenate(all_y_pred_pooled)
    print("\n--- Pooled Out-of-Sample Classification Report (all folds combined) ---")
    print(classification_report(
        pooled_y_test, pooled_y_pred,
        labels=class_labels_ordered,
        target_names=[CLASS_NAMES[l] for l in class_labels_ordered],
        zero_division=0
    ))


# ============================================================================
# STEP 9 & 10 — Full Trade Performance Breakdown
# ============================================================================
print("\n--- Aggregate Trading Performance by P(WIN) Threshold (Mean ± Std Across Folds) ---")
print("-" * 115)
print(f"{'Threshold':<10} | {'Precision':<18} | {'Error Breakdown (Given trade fired)':<38} | {'Win Rate':<18} | {'RR':<14}")
print("-" * 115)

for t in thresholds_to_track:
    prec_array = np.array(fold_metrics[t]['precisions'])
    wr_array = np.array([w for w in fold_metrics[t]['win_rates'] if not np.isnan(w)])
    rr_array = np.array([r for r in fold_metrics[t]['rrs'] if not np.isnan(r)])
    
    total_wins = sum([r['wins'] for r in fold_detail_records if r['threshold'] == round(t, 2)])
    total_losses = sum([r['losses'] for r in fold_detail_records if r['threshold'] == round(t, 2)])
    total_trades = total_wins + total_losses

    if len(prec_array) > 0 and total_trades > 0:
        mean_prec, std_prec = np.mean(prec_array), np.std(prec_array)
        
        p_win = total_wins / total_trades
        p_loss = total_losses / total_trades

        prec_txt = f"{mean_prec:>6.2%} ± {std_prec:<6.2%}"
        error_txt = f"WIN: {p_win:.0%} | LOSS: {p_loss:.0%}"
        wr_txt = f"{np.mean(wr_array):>6.2%} ± {np.std(wr_array):<6.2%}" if len(wr_array) > 0 else "n/a"
        rr_txt = f"{np.mean(rr_array):>5.2f} ± {np.std(rr_array):<5.2f}" if len(rr_array) > 0 else "n/a"

        print(f">= {t*100:>4.1f}%   | {prec_txt:<18} | {error_txt:<38} | {wr_txt:<18} | {rr_txt:<14}")
    else:
        print(f">= {t*100:>4.1f}%   | No trades fired in any test block.")

print("-" * 115)

# ============================================================================
# STEP 11 — Out-of-Sample PnL Chart & Trade Log Export
# ============================================================================
print("\n--- Saving Trade Logs to CSV ---")

all_trades_list = []

plt.figure(figsize=(14, 8))
cmap = plt.get_cmap('viridis')

for i, t in enumerate(thresholds_to_track):
    if len(trade_log[t]) == 0:
        continue

    # Create DF for this threshold
    log_df = pd.DataFrame(trade_log[t], columns=['datetime', 'ret']).sort_values('datetime')
    log_df['threshold'] = round(t, 2)
    log_df['cum_pnl'] = log_df['ret'].cumsum()
    
    # Collect for master export
    all_trades_list.append(log_df)

    # Plotting
    color = cmap(i / max(len(thresholds_to_track) - 1, 1))
    plt.plot(log_df['datetime'], log_df['cum_pnl'], label=f'>= {t:.2f}', color=color)

# Save Master Trade Log
if all_trades_list:
    master_trade_df = pd.concat(all_trades_list)
    master_trade_df.to_csv('walk_forward_trade_log.csv', index=False)
    print("Full trade log saved to: walk_forward_trade_log.csv")

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Out-of-Sample Cumulative PnL — Binary Walk-Forward Test Folds')
plt.legend(title='P(WIN) ≥', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.savefig('pnl_walkforward.png', dpi=150)
plt.show()

# ============================================================================
# STEP 12 — Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Sequential (order-aware) binary barrier labels, full dataset: {label_counts.to_dict()}")
print(f"Hunting zone ({RSI_FILTER_COLUMN} < {RSI_FILTER_THRESHOLD}) rows: {len(df_training_zone)}")

for lbl in class_labels_ordered:
    p_arr = np.array(class_metrics_by_label[lbl]['precision'])
    if len(p_arr) > 0:
        print(f"  {CLASS_NAMES[lbl]:<8} class precision (argmax): {p_arr.mean():.2%} ± {p_arr.std():.2%} across folds")

best_rr_threshold, best_rr = None, -np.inf
for t in thresholds_to_track:
    rr_arr = np.array([r for r in fold_metrics[t]['rrs'] if not np.isnan(r)])
    if len(rr_arr) > 0 and rr_arr.mean() > best_rr:
        best_rr, best_rr_threshold = rr_arr.mean(), t

if best_rr_threshold is not None:
    print(f"Best mean out-of-sample RR across folds: {best_rr:.2f} at P(WIN) >= {best_rr_threshold:.2f}")

for t in thresholds_to_track:
    if len(trade_log[t]) > 0:
        log_df = pd.DataFrame(trade_log[t], columns=['datetime', 'ret']).sort_values('datetime')
        cum = log_df['ret'].cumsum().iloc[-1]
        print(f"  P(WIN) >= {t:.2f}: {len(log_df)} trades, cumulative {cum:.2f}R out-of-sample")

print("\nReminder: this run ignores fees, slippage, latency, and partial fills.")
print("A positive cumulative R here is a necessary but not sufficient condition")
print("for live profitability — re-run with realistic transaction costs before sizing capital.")
print("=" * 70)

