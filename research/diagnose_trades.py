"""
Diagnostic: Why does the walk-forward produce so few trades, and why is the
trade count flat (~9-11) across thresholds 20%-85% then collapse at 90%?

Hypothesis: the model's P(WIN) is sticky/extreme (bimodal near 0 and near 1)
at 1-second resolution, so the hysteresis detector collapses each sustained
high-probability episode into ONE trade. The threshold sweep is therefore
meaningless — every threshold in [0.2, 0.85] catches the same episodes.

This script reuses the research pipeline but only runs ONE fold and prints
the P(WIN) distribution + signal counts to confirm.
"""

import sys
import os
# Ensure project root is on the path (script may be run from project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from utils.indicators import *
from utils.engine import *

import lightgbm as lgb

from main.config import (
    LOOKAHEAD_SECONDS, BARRIER_MULTIPLIER, RSI_FILTER_COLUMN, RSI_FILTER_THRESHOLD,
    RR_RATIO, TRADE_DIRECTION, TIMEFRAME_BARRIER, BARRIER_VOL_WINDOW,
    RSI_14_WINDOW, RSI_2_WINDOW, SMA_14_WINDOW, DEEPLOB_Z_WINDOW, RSI_Z_WINDOW,
    MA_SLOPE_WINDOW, PRICE_ROC_WINDOW_3, LABEL_LOSS, LABEL_WIN, FEATURE_COLUMNS,
    FEE_BPS, ENABLE_SLIPPAGE, MIN_BARRIER_PCT, MAX_BARRIER_PCT, EPS,
    HYSTERESIS_BUFFER, LOCKOUT_SECONDS,
)

try:
    from main.config import BARRIER_VOL_BAR_WINDOW
except ImportError:
    BARRIER_VOL_BAR_WINDOW = 20

BAR_FREQ = f"{TIMEFRAME_BARRIER}s"
BARRIER_CHUNK_SIZE = 50_000
N_RANDOM_SAMPLES = 15
TEST_BLOCK_SIZE = 10000
EMBARGO_SECONDS = LOOKAHEAD_SECONDS
CLASS_NAMES = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}

VPIN_TARGET_BUCKETS_PER_SESSION = 50
VPIN_WINDOW_BUCKETS = 10
VPIN_MIN_PERIODS = 5
RETAIL_THRESHOLD = 0.05

# Reuse the research functions by importing the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_state_optimized as r


def find_debounced_signals(y_proba, threshold, hysteresis=HYSTERESIS_BUFFER):
    """Return indices of fresh threshold crossings, ignoring flicker."""
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

print("=" * 70)
print("DIAGNOSTIC: Trade count flatness & P(WIN) distribution")
print("=" * 70)

# Load just session 0 (fastest) to keep the diagnostic quick
print("\nLoading session 0...")
session_0 = r.load_and_build_state_space(
    "research/csv/BTCUSDT_orderbook.csv", "research/csv/BTCUSDT_trades.csv", session_id=0
)
print(f"Session 0 state space: {len(session_0)} rows (1-second resolution)")

# Barrier labels
df = r.add_barrier_labels(session_0)
df = df[df['volume'] > 0]
print(f"After barrier labels + volume filter: {len(df)} rows")

label_counts = df['target'].map(CLASS_NAMES).value_counts()
print(f"Label distribution (full): {label_counts.to_dict()}")

df_training_zone = df[df[RSI_FILTER_COLUMN] < RSI_FILTER_THRESHOLD]
print(f"Hunting zone rows: {len(df_training_zone)}")
print(f"Hunting zone labels: {df_training_zone['target'].map(CLASS_NAMES).value_counts().to_dict()}")

# Features
ORDERFLOW_FEATURE_COLUMNS = [
    'ofi_z', 'cvd_delta_z', 'cvd_slope_z', 'vpin', 'absorption_z',
    'tick_autocorr', 'inst_ratio', 'delta_divergence_z', 'depth_imbalance_z',
]
features = list(FEATURE_COLUMNS) + [
    c for c in ORDERFLOW_FEATURE_COLUMNS if c in df_training_zone.columns
]
X = df_training_zone[features]
y = df_training_zone['target']
print(f"\nFeatures ({len(features)}): {features}")

# Build one walk-forward split (first fold)
n = len(X)
min_initial_train = int(n * 0.35)
block_size = TEST_BLOCK_SIZE
available_test_run = n - min_initial_train - block_size
step_intervals = available_test_run // N_RANDOM_SAMPLES
start_idx = min_initial_train
test_idx = np.arange(start_idx, start_idx + block_size)
test_start_time = X.index[test_idx[0]]
possible_train_idx = np.arange(0, start_idx)
train_times = X.index[possible_train_idx]
horizon = pd.Timedelta(seconds=LOOKAHEAD_SECONDS)
embargo = pd.Timedelta(seconds=EMBARGO_SECONDS)
purge_mask = (train_times + horizon + embargo >= test_start_time)
train_idx = possible_train_idx[~purge_mask]

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"\nFold 1: Train {len(X_train)} rows | Test {len(X_test)} rows")
print(f"Test time span: {X_test.index[0]} -> {X_test.index[-1]} "
      f"({(X_test.index[-1]-X_test.index[0]).total_seconds()/3600:.2f} hours)")

# Train model (same params as research)
clf = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
)
clf.fit(X_train, y_train)

target_idx = list(clf.classes_).index(LABEL_WIN)
y_proba = clf.predict_proba(X_test)[:, target_idx]

print("\n" + "=" * 70)
print("P(WIN) DISTRIBUTION ON TEST SET (1-second rows)")
print("=" * 70)
bins = np.arange(0.0, 1.01, 0.05)
hist, edges = np.histogram(y_proba, bins=bins)
print(f"{'P(WIN) range':<16} | {'count':<8} | {'pct':<6} | bar")
for i in range(len(hist)):
    lo, hi = edges[i], edges[i+1]
    bar = '#' * int(hist[i] / max(1, hist.max()) * 40)
    print(f"[{lo:.2f}, {hi:.2f})     | {hist[i]:<8} | {hist[i]/len(y_proba)*100:>4.1f}% | {bar}")

print(f"\nP(WIN) stats: mean={y_proba.mean():.4f} | std={y_proba.std():.4f} | "
      f"min={y_proba.min():.4f} | max={y_proba.max():.4f}")
print(f"P(WIN) >= 0.5: {(y_proba>=0.5).sum()} rows ({(y_proba>=0.5).mean()*100:.1f}%)")
print(f"P(WIN) >= 0.8: {(y_proba>=0.8).sum()} rows ({(y_proba>=0.8).mean()*100:.1f}%)")
print(f"P(WIN) >= 0.9: {(y_proba>=0.9).sum()} rows ({(y_proba>=0.9).mean()*100:.1f}%)")

# How many DISTINCT sustained high-probability episodes?
print("\n" + "=" * 70)
print("EPISODE ANALYSIS (sustained high-P(WIN) periods)")
print("=" * 70)
for t in [0.20, 0.30, 0.50, 0.65, 0.80, 0.85, 0.90]:
    above = y_proba >= t
    # count transitions from below to above = number of episodes
    transitions = np.sum((above[1:] == True) & (above[:-1] == False))
    if above[0]:
        transitions += 1
    # hysteresis signals
    sig = find_debounced_signals(y_proba, t, HYSTERESIS_BUFFER)
    print(f"Thresh >= {t*100:.0f}%: {transitions} raw episodes | "
          f"{len(sig)} hysteresis signals | "
          f"{(above).sum()} rows above threshold")

# Show the actual P(WIN) at hysteresis signal points
print("\n" + "=" * 70)
print("HYSTERESIS SIGNAL DETAIL (threshold = 0.50)")
print("=" * 70)
sig = find_debounced_signals(y_proba, 0.50, HYSTERESIS_BUFFER)
for i in sig:
    ts = X_test.index[i]
    p = y_proba[i]
    lbl = y_test.iloc[i]
    # how long does P(WIN) stay high after this signal?
    j = i
    while j < len(y_proba) and y_proba[j] >= 0.50 - HYSTERESIS_BUFFER:
        j += 1
    duration = j - i
    print(f"  {ts} | P(WIN)={p:.3f} | label={CLASS_NAMES[lbl]} | "
          f"stays>=0.42 for {duration}s ({duration/60:.1f} min)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("If P(WIN) is bimodal (most rows near 0, few near 1), the threshold")
print("sweep is meaningless: every threshold in [0.2, 0.85] catches the same")
print("episodes, giving a flat trade count. Only at 0.90 do some episodes")
print("get filtered out. The hysteresis detector then collapses each episode")
print("to ONE trade -> ~10 trades total.")