"""
Enhanced ML State Space — Quant-Grade Feature Engineering for RSI Snapback
==========================================================================
Key quant principles applied:
1. OLS-based CVD-Price divergence: z-score of residual spread (directional divergence)
2. DeepLOB spatial features: order book depth imbalances at multiple levels
3. Trade sign autocorrelation: microstructure momentum (hidden institutional orders)
4. Regime features: MA acceleration determines snapback success
5. Rate-of-change: Speed of entry into oversold predicts bounce strength
6. Simple model: n_estimators=200, max_depth=5 (balanced complexity)
"""

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
import shap
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

# ============================================================================
# CONFIG
# ============================================================================
LOOKAHEAD_SECONDS    = 600
BARRIER_MULTIPLIER   = 3.0
RSI_FILTER_COLUMN    = 'rsi_2'
RSI_FILTER_THRESHOLD = 30
EPS                  = 1e-9
BARRIER_CHUNK_SIZE   = 50_000

N_RANDOM_SAMPLES    = 15
TEST_BLOCK_SIZE     = 10000
EMBARGO_SECONDS      = LOOKAHEAD_SECONDS

LABEL_LOSS    = 0
LABEL_WIN     = 1
CLASS_NAMES   = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}
RR_RATIO      = 1.0

# --- Orderflow Feature Config ---
# FIX: VPIN_BUCKET_SIZE used to be a hardcoded 500 (BTC units, per RETAIL_THRESHOLD
# below being denominated in BTC). That's ~$30-50M notional/bucket — on a short
# sample this could mean fewer than min_periods buckets ever close, so VPIN stays
# NaN for the whole run. Instead we target a fixed NUMBER of buckets per session
# (50/session is the standard VPIN convention) and derive bucket size from each
# session's actual traded volume, so it self-calibrates regardless of venue/liquidity.
VPIN_TARGET_BUCKETS_PER_SESSION = 50
VPIN_WINDOW_BUCKETS = 10   # smoothing window in buckets (was 50 - unreachable given only ~50 buckets/session exist)
VPIN_MIN_PERIODS    = 5
RETAIL_THRESHOLD    = 0.05


# ============================================================================
# TRIPLE BARRIER LABELING
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


# ============================================================================
# STEP 1a — ORDERFLOW FEATURE COMPUTATION
# ============================================================================
def compute_orderflow_features(df_trades, df_orderbook):
    """
    Compute orderflow features:
    1. OFI (Cont et al.) — structural order flow imbalance
    2. CVD — cumulative volume delta
    3. VPIN — informed trading probability
    4. Retail vs Institutional Delta — size granularity
    5. Microstructure — tick rule, trade intensity
    6. Absorption — institutional absorption
    7. DeepLOB — order book depth imbalance
    8. Trade sign autocorrelation — microstructure momentum
    """
    # ================================================================
    # 1. ORDER FLOW IMBALANCE (OFI)
    # ================================================================
    ob = df_orderbook.copy()

    ob['bid_price_prev'] = ob['bid_price'].shift(1)
    ob['ask_price_prev'] = ob['ask_price'].shift(1)
    ob['bid_amount_prev'] = ob['bid_amount'].shift(1)
    ob['ask_amount_prev'] = ob['ask_amount'].shift(1)

    bid_improved = ob['bid_price'] > ob['bid_price_prev']
    bid_worsened = ob['bid_price'] < ob['bid_price_prev']
    bid_unchanged = ob['bid_price'] == ob['bid_price_prev']

    delta_bid = np.where(bid_improved, ob['bid_amount'],
                np.where(bid_worsened, -ob['bid_amount_prev'],
                np.where(bid_unchanged, ob['bid_amount'] - ob['bid_amount_prev'], 0)))

    ask_improved = ob['ask_price'] < ob['ask_price_prev']
    ask_worsened = ob['ask_price'] > ob['ask_price_prev']
    ask_unchanged = ob['ask_price'] == ob['ask_price_prev']

    delta_ask = np.where(ask_improved, ob['ask_amount'],
                np.where(ask_worsened, -ob['ask_amount_prev'],
                np.where(ask_unchanged, ob['ask_amount'] - ob['ask_amount_prev'], 0)))

    ob['ofi_raw'] = delta_bid - delta_ask

    ob['datetime'] = pd.to_datetime(ob['timestamp'].astype('int64') // 1000, unit='s')
    ob = ob.set_index('datetime').sort_index()

    ofi_1s = ob['ofi_raw'].resample('1s').sum()
    ofi_1m = ofi_1s.resample('1min').sum()
    ofi_diff = ofi_1m.diff().fillna(0)

    # ================================================================
    # 2. CVD
    # ================================================================
    trades = df_trades.copy()
    trades['buy_vol'] = trades['amount'].where(trades['side'] == 'buy', 0)
    trades['sell_vol'] = trades['amount'].where(trades['side'] == 'sell', 0)

    trades['cvd_tick'] = trades['buy_vol'] - trades['sell_vol']
    trades['datetime'] = pd.to_datetime(trades['timestamp'].astype('int64') // 1000, unit='s')
    trades = trades.set_index('datetime').sort_index()

    cvd_1m = trades['cvd_tick'].resample('1min').sum().cumsum()
    cvd_delta_1m = cvd_1m.diff().fillna(0)
    cvd_slope_5 = cvd_1m.diff(5).fillna(0)

    # ================================================================
    # 3. MICROSTRUCTURE: Tick Rule & Trade Sign Autocorrelation
    # ================================================================
    trades['price_prev'] = trades['price'].shift(1)
    trades['tick_rule'] = np.where(trades['price'] > trades['price_prev'], 1,
                          np.where(trades['price'] < trades['price_prev'], -1, 0))

    micro = trades.resample('1min').agg({
        'tick_rule': 'mean',
        'amount': ['sum', 'count'],
        'price': 'std',
    })
    micro.columns = ['tick_rule', 'volume', 'n_trades', 'price_volatility']
    micro['trade_intensity'] = micro['n_trades'] / 60.0
    micro['avg_trade_size'] = micro['volume'] / (micro['n_trades'] + EPS)

    # Trade sign autocorrelation: rolling autocorrelation of tick rule
    # High autocorrelation = hidden institutional order splitting
    tick_1s = trades['tick_rule'].resample('1s').last().fillna(0)
    tick_1m = tick_1s.resample('1min').apply(
        lambda x: x.autocorr() if len(x) > 5 and not np.isnan(x.autocorr()) else 0
    )

    # ================================================================
    # 4. VPIN
    # ================================================================
    # FIX: bucket size is now derived from this session's actual total traded
    # volume / VPIN_TARGET_BUCKETS_PER_SESSION, instead of a hardcoded constant
    # that assumed a specific liquidity regime. This guarantees ~50 buckets close
    # per session regardless of how much volume the CSV actually contains.
    trades_sorted = trades.sort_index()
    total_session_volume = trades_sorted['amount'].sum()
    vpin_bucket_size = max(total_session_volume / VPIN_TARGET_BUCKETS_PER_SESSION, EPS)

    cum_vol = trades_sorted['amount'].cumsum()
    bucket_labels = (cum_vol / vpin_bucket_size).astype(int)

    vpin_buckets = trades_sorted.groupby(bucket_labels).agg({
        'buy_vol': 'sum',
        'sell_vol': 'sum',
        'amount': 'sum'
    })

    vpin_buckets['vpin_raw'] = np.abs(vpin_buckets['buy_vol'] - vpin_buckets['sell_vol']) / (vpin_buckets['amount'] + EPS)
    vpin_buckets['vpin_smoothed'] = vpin_buckets['vpin_raw'].rolling(VPIN_WINDOW_BUCKETS, min_periods=VPIN_MIN_PERIODS).mean()

    n_buckets_closed = bucket_labels.nunique()
    if n_buckets_closed < VPIN_MIN_PERIODS:
        print(f"  [VPIN WARNING] Only {n_buckets_closed} buckets closed this session "
              f"(need {VPIN_MIN_PERIODS}+) — VPIN will be NaN/sparse for this session.")

    bucket_end_times = trades_sorted.groupby(bucket_labels).apply(lambda g: g.index[-1])
    if isinstance(bucket_end_times, pd.Series):
        bucket_end_times = bucket_end_times.reset_index(drop=True)

    vpin_aligned = pd.Series(vpin_buckets['vpin_smoothed'].values, index=bucket_end_times)
    vpin_1m = vpin_aligned.resample('1min').last().ffill()

    # ================================================================
    # 5. RETAIL vs INSTITUTIONAL DELTA
    # ================================================================
    trades['is_retail'] = trades['amount'] < RETAIL_THRESHOLD
    trades['is_institutional'] = trades['amount'] >= RETAIL_THRESHOLD

    trades['retail_buy'] = trades['buy_vol'].where(trades['is_retail'], 0)
    trades['retail_sell'] = trades['sell_vol'].where(trades['is_retail'], 0)
    trades['inst_buy'] = trades['buy_vol'].where(trades['is_institutional'], 0)
    trades['inst_sell'] = trades['sell_vol'].where(trades['is_institutional'], 0)

    trade_size = trades.resample('1min').agg({
        'retail_buy': 'sum',
        'retail_sell': 'sum',
        'inst_buy': 'sum',
        'inst_sell': 'sum',
        'amount': ['sum', 'count'],
    })
    trade_size.columns = ['retail_buy', 'retail_sell', 'inst_buy', 'inst_sell', 'volume', 'n_trades']

    trade_size['retail_delta'] = trade_size['retail_buy'] - trade_size['retail_sell']
    trade_size['inst_delta'] = trade_size['inst_buy'] - trade_size['inst_sell']
    trade_size['delta_divergence'] = trade_size['inst_delta'] - trade_size['retail_delta']
    trade_size['inst_ratio'] = (trade_size['inst_buy'] + trade_size['inst_sell']) / (trade_size['volume'] + EPS)

    # ================================================================
    # 6. ABSORPTION RATIO
    # ================================================================
    ob_1m_price = ob[['bid_price', 'ask_price']].resample('1min').agg({
        'bid_price': ['min', 'max'],
        'ask_price': ['min', 'max']
    })
    ob_1m_price.columns = ['bid_low', 'bid_high', 'ask_low', 'ask_high']
    ob_1m_price['price_range'] = (ob_1m_price['ask_high'] - ob_1m_price['bid_low']).ffill()

    abs_ratio_raw = cvd_delta_1m / (ob_1m_price['price_range'] + EPS)

    # ================================================================
    # 7. DeepLOB: Order Book Depth Imbalance
    # ================================================================
    # Simulate depth features from top-of-book dynamics
    # Depth imbalance ratio: (bid_amount - ask_amount) / (bid_amount + ask_amount)
    ob['depth_imbalance'] = (ob['bid_amount'] - ob['ask_amount']) / (ob['bid_amount'] + ob['ask_amount'] + EPS)

    # Rate of change of depth imbalance (liquidity absorption speed)
    ob['depth_imbalance_roc'] = ob['depth_imbalance'].diff().fillna(0)

    # Bid-ask spread pressure: how fast the spread is changing
    ob['mid'] = (ob['bid_price'] + ob['ask_price']) / 2
    ob['spread_bps'] = (ob['ask_price'] - ob['bid_price']) / ob['mid'] * 10000

    depth_1m = ob[['depth_imbalance', 'depth_imbalance_roc', 'spread_bps']].resample('1min').last()

    return (ofi_1m, ofi_diff, cvd_delta_1m, cvd_slope_5, micro,
            vpin_1m, trade_size, abs_ratio_raw, tick_1m, depth_1m)


# ============================================================================
# STEP 1b — Per-session feature pipeline
# ============================================================================
def load_and_build_state_space(orderbook_path, trades_path, session_id):
    df_orderbook = pd.read_csv(orderbook_path)
    df_trades = pd.read_csv(trades_path)

    df_orderbook['datetime'] = pd.to_datetime(df_orderbook['timestamp'].astype('int64') // 1000, unit='s')
    df_trades['datetime'] = pd.to_datetime(df_trades['timestamp'].astype('int64') // 1000, unit='s')

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
    df_ob_1m['spread_pct'] = df_ob_1m['spread'] / df_ob_1m['close']

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

    # ===== COMPUTE ORDERFLOW FEATURES =====
    (ofi_1m, ofi_diff, cvd_delta_1m, cvd_slope_5, micro,
     vpin_1m, trade_size, abs_ratio_raw, tick_autocorr_1m,
     depth_1m) = compute_orderflow_features(df_trades, df_orderbook)

    # ===== BUILD PRICE DATAFRAME =====
    df_price_1m = pd.DataFrame({
        'open': df_ob_1m['open'],
        'high': df_ob_1m['high'],
        'low': df_ob_1m['low'],
        'close': df_ob_1m['close'],
        'volume': df_trades_1m['volume'],
        'spread_pct': df_ob_1m['spread_pct'],
    })

    # ================================================================
    # PRICE-BASED FEATURES
    # ================================================================
    price_clean = np.asarray(df_price_1m['close'], dtype=float)
    diffs = np.diff(price_clean)

    def _pad_front(arr, target_len):
        out = np.full(target_len, np.nan)
        out[-len(arr):] = arr
        return out

    # RSI features
    df_price_1m['rsi_2'] = _pad_front(get_wilders_rsi(diffs, 2), len(df_price_1m))
    rsi_14_raw = get_wilders_rsi(diffs, 14)
    df_price_1m['rsi_14'] = _pad_front(rsi_14_raw, len(df_price_1m))
    df_price_1m['sma_14'] = _pad_front(sma(rsi_14_raw, 14), len(df_price_1m))

    # MA Slope (velocity) - strictly backward looking
    ma_5 = df_price_1m['close'].rolling(window=5).mean()
    df_price_1m['ma_slope_5'] = ma_5.diff().fillna(0)

    # MA acceleration (2nd derivative)
    df_price_1m['ma_accel'] = df_price_1m['ma_slope_5'].diff().fillna(0)
    df_price_1m['ma_accel_regime'] = np.where(df_price_1m['ma_accel'] > 0, 1,
                                      np.where(df_price_1m['ma_accel'] < 0, -1, 0))

    # Price rate of change
    df_price_1m['price_roc_3'] = df_price_1m['close'].pct_change(3)
    df_price_1m['price_roc_5'] = df_price_1m['close'].pct_change(5)

    # RSI rate of change — speed of entry into oversold
    df_price_1m['rsi_roc_3'] = df_price_1m['rsi_2'].diff(3)
    df_price_1m['rsi_roc_5'] = df_price_1m['rsi_2'].diff(5)

    # ================================================================
    # MINIMAL FEATURE SET (only proven high-PFI features)
    # ================================================================
    # From v3 analysis, the top features by PFI are:
    # spread_bps_z(0.081), ma_slope_5(0.024), rsi_x_ma(0.006), 
    # ma_accel(0.005), price_roc_3(0.005), depth_imbalance_z(0.002)
    
    # DeepLOB: spread in basis points (PFI=0.081 — dominant feature)
    df_price_1m['spread_bps_z'] = calculate_zscore(
        depth_1m['spread_bps'].reindex(df_ob_1m.index).fillna(0), window=10)
    
    # DeepLOB: depth imbalance (PFI=0.002)
    df_price_1m['depth_imbalance_z'] = calculate_zscore(
        depth_1m['depth_imbalance'].reindex(df_ob_1m.index).fillna(0), window=10)

    # RSI × MA slope interaction (PFI=0.006 — key signal)
    rsi_z = calculate_zscore(df_price_1m['rsi_2'], window=20)
    df_price_1m['rsi_x_ma'] = rsi_z * df_price_1m['ma_slope_5']

    # ================================================================
    # FIX: PREVIOUSLY-DEAD ORDERFLOW FEATURES
    # ================================================================
    # ofi_1m, ofi_diff, cvd_delta_1m, cvd_slope_5, vpin_1m, trade_size, and
    # abs_ratio_raw were all computed by compute_orderflow_features() above but
    # were never joined into df_price_1m — they were silently discarded, so the
    # model never actually saw them despite the docstring/feature-importance
    # comments describing them as inputs. Wiring them in here for real, using
    # the same reindex-onto-df_ob_1m.index pattern already used for depth_1m.
    # Their actual predictive value (PFI/SHAP) will show up honestly in Step 7
    # on the next run instead of being assumed away.

    # OFI: structural order flow imbalance (change in imbalance per minute)
    df_price_1m['ofi_z'] = calculate_zscore(
        ofi_diff.reindex(df_ob_1m.index).fillna(0), window=10)

    # CVD: cumulative volume delta, level change and short-window slope
    df_price_1m['cvd_delta_z'] = calculate_zscore(
        cvd_delta_1m.reindex(df_ob_1m.index).fillna(0), window=10)
    df_price_1m['cvd_slope_z'] = calculate_zscore(
        cvd_slope_5.reindex(df_ob_1m.index).fillna(0), window=10)

    # VPIN: informed-trading probability. Forward-filled (not zero-filled) on
    # reindex, since VPIN is a volume-clock signal that should persist its last
    # known estimate between bucket closes rather than snap to 0.
    df_price_1m['vpin'] = vpin_1m.reindex(df_ob_1m.index).ffill().fillna(0)

    # Absorption ratio: CVD delta normalized by price range (institutional absorption)
    df_price_1m['absorption_z'] = calculate_zscore(
        abs_ratio_raw.reindex(df_ob_1m.index).fillna(0), window=10)

    # Trade sign autocorrelation: hidden institutional order-splitting signature
    df_price_1m['tick_autocorr'] = tick_autocorr_1m.reindex(df_ob_1m.index).fillna(0)

    # Retail vs institutional flow: ratio and divergence between size cohorts
    df_price_1m['inst_ratio'] = trade_size['inst_ratio'].reindex(df_ob_1m.index).fillna(0)
    df_price_1m['delta_divergence_z'] = calculate_zscore(
        trade_size['delta_divergence'].reindex(df_ob_1m.index).fillna(0), window=10)

    # ================================================================
    # JOIN MICRO (1s) WITH MACRO (1m)
    # ================================================================
    df_micro_shifted = df_micro.shift(1)
    df_macro_shifted = df_price_1m.shift(1)

    df_state_space = df_micro_shifted.join(df_macro_shifted, how='left').ffill()
    df_state_space['execution_price'] = df_micro['price']

    df_state_space = df_state_space.dropna()
    df_state_space['session_id'] = session_id

    return df_state_space


# ============================================================================
# STEP 2 — Build combined state space
# ============================================================================
print("Loading sessions...")
session_0 = load_and_build_state_space(
    "./research/csv/BTCUSDT_orderbook.csv", "./research/csv/BTCUSDT_trades.csv", session_id=0
)
print(f"Session 0: {len(session_0)} rows")
session_1 = load_and_build_state_space(
    "./research/csv/BTCUSDT_orderbook_2.csv", "./research/csv/BTCUSDT_trades_2.csv", session_id=1
)
print(f"Session 1: {len(session_1)} rows")
session_2 = load_and_build_state_space(
    "./research/csv/BTCUSDT_orderbook_3.csv", "./research/csv/BTCUSDT_trades_3.csv", session_id=2
)
print(f"Session 2: {len(session_2)} rows")

#df_state_space = load_and_build_state_space("./research/csv/XAUTUSDT_orderbook.csv", "./research/csv/XAUTUSDT_trades.csv", 0)
df_state_space = pd.concat([session_0, session_1, session_2]).sort_index()
print(f"Combined state space: {len(df_state_space)} rows")

# ============================================================================
# STEP 3 — Barrier labeling
# ============================================================================
df_state_space = add_barrier_labels(df_state_space)
df_state_space = df_state_space[df_state_space['volume'] > 0]

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
             'bars_to_exit', 'exit_ret', 'terminal_ret', 'is_timeout',
             'execution_price',
             # Exclude zero-PFI and negative-PFI features
             'buy_vol', 'sell_vol', 'micro_OFI', 'micro_CVD', 'volume',
             'spread_pct', 'rsi_14', 'rsi_roc_3', 'rsi_roc_5',
             'ma_accel_regime', 'price_roc_5']
features = [c for c in df_training_zone.columns if c not in drop_cols]

X = df_training_zone[features]
y = df_training_zone['target']

print(f"\nFeatures ({len(features)} total):")
for f in features:
    print(f"  - {f}")

# ============================================================================
# STEP 5 — Risk-Normalized Return
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

# Winsorize extreme returns
upper_quantile = normalized_return.quantile(0.99)
lower_quantile = normalized_return.quantile(0.01)
cap = max(abs(upper_quantile), abs(lower_quantile), 5.0)
normalized_return = normalized_return.clip(-cap, cap)

mean_stop_barrier = df_training_zone['stop_barrier'].mean()
print("\n" + "="*50)
print("STOP BARRIER METRICS")
print("="*50)
print(f"Average Stop Barrier Size (%): {mean_stop_barrier * 100:.4f}%")
print("="*50)

# ============================================================================
# STEP 6 — Purged Chronological Walk-Forward
# ============================================================================
TARGET_TRADE_CLASS = LABEL_WIN
INVERT_RETURNS     = False

target_name = CLASS_NAMES[TARGET_TRADE_CLASS]
prob_label = f"P({target_name})"


def purged_chronological_splits(index, n_samples, block_size, horizon_seconds, embargo_seconds=0):
    n = len(index)
    horizon = pd.Timedelta(seconds=horizon_seconds)
    embargo = pd.Timedelta(seconds=embargo_seconds)
    min_initial_train = int(n * 0.35)
    available_test_run = n - min_initial_train - block_size

    if available_test_run <= 0:
        raise ValueError("The dataset length is too narrow for this test configuration block size.")

    step_intervals = available_test_run // n_samples

    # FIX: warn if test windows overlap in time. If step_intervals < block_size,
    # consecutive folds re-test overlapping historical periods with different
    # models — the same real-world trade opportunity can get scored (and logged)
    # more than once across folds, which inflates the apparent trade count and
    # breaks the independence assumption behind the win-rate/CVaR stats later.
    if step_intervals < block_size:
        overlap_pct = 100 * (1 - step_intervals / block_size)
        print(f"[FOLD OVERLAP WARNING] step_intervals={step_intervals} < block_size={block_size} "
              f"-> consecutive test folds overlap by ~{overlap_pct:.0f}% of their length. "
              f"Trade counts/expectancy pooled across folds are NOT independent draws. "
              f"Reduce n_samples or block_size, or get more data, to eliminate this.")

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

print(f"\n--- Starting {N_RANDOM_SAMPLES} Walk-Forward Folds | Targeting: {target_name} | Inverting Trades: {INVERT_RETURNS} ---")

# FIX: HYSTERESIS BUFFER for signal detection
# The original edge-trigger (np.diff(preds, prepend=0) == 1) fires a "fresh"
# signal every time y_proba crosses from below t to >= t. If y_proba is noisy
# right around a given threshold (e.g. the model's output clusters near 0.85),
# it can cross back and forth many times within what is really one continuous
# elevated-probability period, and each crossing gets counted as a separate
# trade. This produced the non-monotonic trade-count spike (58 trades at >=85%
# vs. 27 at >=80% and 6 at >=90%) — a strong signature of flicker, not signal.
# HYSTERESIS_BUFFER requires proba to drop meaningfully below the threshold
# before a new signal is allowed to re-arm, collapsing flicker into one trade.
# NOTE: 0.08 is a starting point, not a proven-correct constant — tested on
# synthetic noise (std=0.02) where 0.03 only cut flicker by ~79% (66->14
# spurious signals) while 0.08 fully collapsed it to the true count. Your
# model's actual y_proba noise level may differ. Before trusting this on your
# real data, plot a histogram of y_proba for the fold(s) that fed the >=85%
# threshold and check its local volatility against this buffer.
HYSTERESIS_BUFFER = 0.08

def find_debounced_signals(y_proba, threshold, hysteresis=HYSTERESIS_BUFFER):
    """Return indices of fresh threshold crossings, ignoring flicker.
    Once armed and proba >= threshold, a signal fires and the detector disarms.
    It only re-arms once proba drops below (threshold - hysteresis)."""
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
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if len(X_train) == 0 or y_train.nunique() < 2:
        print(f"Fold {fold} | skipped — invalid training data distribution.")
        continue

    # ===== MINIMAL HYPERPARAMETERS =====
    # max_depth=3 prevents overfitting on the small feature set
    # n_estimators=200 is sufficient for 6 features
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
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
            if len(shap_vals.values.shape) == 3:
                shap_array = shap_vals.values[:, :, 1]
            else:
                shap_array = shap_vals.values
        else:
            shap_array = shap_vals

        fold_mean_shap = np.abs(shap_array).mean(axis=0)
        fold_shap_values.append(fold_mean_shap)
    except Exception as e:
        print(f"Fold {fold:02d} | SHAP failed: {e}")
        fold_shap_values.append(np.full(len(features), np.nan))

    # --- PFI ---
    if y_test.nunique() < 2:
        fold_pfi_values.append(np.full(len(features), np.nan))
    else:
        try:
            pi_result = permutation_importance(
                clf, X_test, y_test,
                n_repeats=5,
                random_state=42,
                scoring='roc_auc',
                n_jobs=1
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
            'fold': fold,
            'class': CLASS_NAMES[lbl],
            'precision': p,
            'recall': r,
            'support': int((y_test == lbl).sum()),
        })

    all_y_test_pooled.append(y_test.values)
    all_y_pred_pooled.append(y_pred_argmax)

    if TARGET_TRADE_CLASS not in clf.classes_:
        print(f"Fold {fold} | skipped threshold scoring — no {target_name} samples in training data.")
        continue

    target_idx = list(clf.classes_).index(TARGET_TRADE_CLASS)
    y_proba = clf.predict_proba(X_test)[:, target_idx]

    for t in thresholds_to_track:
        # FIX: hysteresis-debounced signal detection (see find_debounced_signals
        # above) replaces the plain edge-trigger, which was vulnerable to
        # probability flicker producing spurious repeated "fresh" signals.
        signal_indices = find_debounced_signals(y_proba, t, HYSTERESIS_BUFFER)

        # --- Chronological Non-Overlapping Position Filter ---
        non_overlapping_idx = []
        next_available_time = pd.Timestamp.min  

        for idx in signal_indices:
            actual_global_time = X_test.index[idx]
            
            # If this fresh signal happens while we are ALREADY in a trade, skip it
            if actual_global_time < next_available_time:
                continue 
                
            non_overlapping_idx.append(actual_global_time)
            
            # Get the duration in seconds
            trade_duration_seconds = int(df_training_zone.loc[actual_global_time, 'bars_to_exit']) + 1
            
            # The next time we can trade is the current time + the trade duration
            next_available_time = actual_global_time + pd.Timedelta(seconds=trade_duration_seconds)

        num_trades = len(non_overlapping_idx)

        if num_trades > 0:
            traded_idx = pd.Index(non_overlapping_idx)
            traded_targets = y_test.loc[traded_idx]

            traded_returns = normalized_return.loc[traded_idx].copy()
            if INVERT_RETURNS:
                traded_returns = -traded_returns

            trade_log[t].extend(zip(traded_idx, traded_returns.values))

            num_original_wins = int((traded_targets == LABEL_WIN).sum())
            num_original_losses = int((traded_targets == LABEL_LOSS).sum())

            strategy_wins = num_original_losses if INVERT_RETURNS else num_original_wins
            strategy_losses = num_original_wins if INVERT_RETURNS else num_original_losses

            win_rate = strategy_wins / num_trades

            win_returns = traded_returns[traded_returns > 0]
            loss_returns = traded_returns[traded_returns < 0]
            avg_win_ret = win_returns.mean() if len(win_returns) > 0 else np.nan
            avg_loss_ret = loss_returns.mean() if len(loss_returns) > 0 else np.nan
            if (not np.isnan(avg_win_ret) and not np.isnan(avg_loss_ret) 
                and abs(avg_loss_ret) > 0.001):
                rr = avg_win_ret / abs(avg_loss_ret)
            else:
                rr = np.nan

            prec = precision_score(y_test.loc[traded_idx] == TARGET_TRADE_CLASS, np.ones(num_trades), zero_division=0)

            fold_metrics[t]['precisions'].append(prec)
            fold_metrics[t]['win_rates'].append(win_rate)
            fold_metrics[t]['rrs'].append(rr)

            fold_detail_records.append({
                'threshold': round(float(t), 2),
                'fold': fold,
                'trades': num_trades,
                'strategy_wins': strategy_wins,
                'strategy_losses': strategy_losses,
                'precision': prec,
                'win_rate': win_rate,
                'rr': rr,
                'avg_return': traded_returns.mean(),
            })

        fold_metrics[t]['trades'].append(num_trades)

    fold_importances.append(clf.feature_importances_)
    print(f"Fold {fold:02d} | Train Rows: {len(X_train):<6} | Tested on forward {TEST_BLOCK_SIZE} rows. Purged: {n_purged}")

# ============================================================================
# STEP 7 — SHAP vs PERMUTATION IMPORTANCE
# ============================================================================
print("\n--- Feature Importance: SHAP vs Permutation (PFI) ---")
print("-" * 80)

mean_shap = np.nanmean(fold_shap_values, axis=0)
mean_pfi = np.nanmean(fold_pfi_values, axis=0)

importance_df = pd.DataFrame({
    'Feature': features,
    'SHAP_Importance': mean_shap,
    'PFI_Mean_AUC_Drop': mean_pfi
}).sort_values('PFI_Mean_AUC_Drop', ascending=False)

print(importance_df.to_string(index=False))
importance_df.to_csv('feature_importance_v3.csv', index=False)

# ============================================================================
# STEP 8 — Model Class-Level Precision & Recall
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

if all_y_test_pooled:
    pooled_y_test = np.concatenate(all_y_test_pooled)
    pooled_y_pred = np.concatenate(all_y_pred_pooled)
    print("\n--- Pooled Out-of-Sample Classification Report ---")
    print(classification_report(
        pooled_y_test, pooled_y_pred,
        labels=class_labels_ordered,
        target_names=[CLASS_NAMES[l] for l in class_labels_ordered],
        zero_division=0
    ))

# ============================================================================
# STEP 9 — Strategy Rolling Expectancy & CVaR
# ============================================================================
print(f"\n--- Strategy Rolling Expectancy & Stability by {prob_label} Threshold ---")
print("-" * 140)
print(f"{'Thresh':<8} | {'Trades':<8} | {'Win%':<8} | {'RR':<6} | {'Overall Exp':<12} | {'Exp CVaR (5%)':<14} | {'Stable (CVaR>0)':<16} | {'Total R'}")
print("-" * 140)

ROLLING_WINDOW = 30

for t in thresholds_to_track:
    if len(trade_log[t]) == 0:
        continue

    log_df = pd.DataFrame(trade_log[t], columns=['datetime', 'ret'])
    returns = log_df['ret'].values
    num_trades = len(returns)

    win_rate = np.mean(returns > 0)
    avg_win = returns[returns > 0].mean() if sum(returns > 0) > 0 else 0
    avg_loss = returns[returns < 0].mean() if sum(returns < 0) > 0 else 0
    rr = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan

    overall_exp = np.mean(returns)
    total_r = np.sum(returns)

    if num_trades >= ROLLING_WINDOW:
        rolling_exp = pd.Series(returns).rolling(window=ROLLING_WINDOW).mean().dropna()
        alpha = 0.05
        cutoff_idx = max(1, int(len(rolling_exp) * alpha))
        worst_rolling_streaks = np.sort(rolling_exp.values)[:cutoff_idx]
        cvar_exp = np.mean(worst_rolling_streaks)
        stable_flag = "YES" if cvar_exp > 0 else "NO"
    else:
        cvar_exp = np.nan
        stable_flag = "N/A (low trades)"

    print(f">= {t*100:>3.0f}% | {num_trades:<8} | {win_rate:>5.1%} | {rr:<6.2f} | {overall_exp:>10.2f} R | {cvar_exp:>12.2f} R | {stable_flag:<16} | {total_r:>+6.2f} R")

print("-" * 140)
print(f"Note: 'Exp CVaR (5%)' = avg R per trade during worst 5% of {ROLLING_WINDOW}-trade streaks.")

# ============================================================================
# STEP 10 — Out-of-Sample PnL Chart
# ============================================================================
import plotly.graph_objects as go

print("\n--- Saving Trade Logs to CSV ---")

all_trades_list = []
fig = go.Figure()

for i, t in enumerate(thresholds_to_track):
    if len(trade_log[t]) == 0:
        continue

    log_df = pd.DataFrame(trade_log[t], columns=['datetime', 'ret']).sort_values('datetime')
    log_df['threshold'] = round(t, 2)
    log_df['cum_pnl'] = log_df['ret'].cumsum()

    all_trades_list.append(log_df)
    trade_numbers = list(range(1, len(log_df) + 1))

    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=log_df['cum_pnl'],
        mode='lines',
        name=f'>= {t:.2f}',
        hovertemplate='Trade: %{x}<br>PnL: %{y:.2f}R<extra></extra>'
    ))

if all_trades_list:
    master_trade_df = pd.concat(all_trades_list)
    master_trade_df.to_csv('walk_forward_trade_log_v3.csv', index=False)
    print("Full trade log saved to: walk_forward_trade_log_v3.csv")

fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

chart_title = f'Out-of-Sample PnL (OLS Divergence + DeepLOB) — Target: {target_name} | Inverted: {INVERT_RETURNS}'
fig.update_layout(
    title=chart_title,
    xaxis_title='Trade Number',
    yaxis_title='Cumulative Return (R)',
    legend_title=f'{prob_label} ≥ (Click!)',
    hovermode="x unified",
    template="plotly_white"
)

fig.write_html('pnl_walkforward_v3.html')
print("Interactive chart saved to: pnl_walkforward_v3.html")
fig.show()

# ============================================================================
# STEP 11 — Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY — OLS DIVERGENCE + DEEPLOB FEATURES")
print("=" * 70)
print(f"Sequential (order-aware) binary barrier labels, full dataset: {label_counts.to_dict()}")
print(f"Hunting zone ({RSI_FILTER_COLUMN} < {RSI_FILTER_THRESHOLD}) rows: {len(df_training_zone)}")
print(f"Trading Setup: Target {target_name} | Inverting Trade Direction: {INVERT_RETURNS}")
print(f"Model: n_estimators=200, max_depth=3, learning_rate=0.01 (see clf definition above)")
# FIX: this used to hand-list feature names that didn't actually exist anywhere
# in the computation (cvd_price_divergence, OFI_z_5, rsi_x_cvd_div, etc. — none
# were ever computed). Printing the real, live `features` list instead so this
# summary can't drift out of sync with what the model is actually trained on again.
print(f"\nActual Features Used ({len(features)} total):")
for f in features:
    print(f"  - {f}")

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
    print(f"Best mean out-of-sample RR across folds: {best_rr:.2f} at {prob_label} >= {best_rr_threshold:.2f}")

for t in thresholds_to_track:
    if len(trade_log[t]) > 0:
        log_df = pd.DataFrame(trade_log[t], columns=['datetime', 'ret']).sort_values('datetime')
        cum = log_df['ret'].cumsum().iloc[-1]
        print(f"  {prob_label} >= {t:.2f}: {len(log_df)} trades, cumulative {cum:.2f}R out-of-sample")

print("\nReminder: this run ignores fees, slippage, latency, and partial fills.")
print("=" * 70)