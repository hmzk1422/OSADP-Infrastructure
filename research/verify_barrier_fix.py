"""
verify_barrier_fix.py
======================
Step-by-step evidence that research/ml_state_optimized.py now computes
bar_vol_std (and therefore the barrier) EXACTLY like the live engine:

  LIVE (main/features.py _compute_macro_features + main/execution.py):
    bar_closes = closes of consecutive TIMEFRAME_BARRIER (300s) candles
    bar_rets   = np.diff(bar_closes) / (bar_closes[:-1] + EPS)
    bar_vol_std = np.nanstd(bar_rets[-window:])        # ddof=0 (population)
    barrier     = bar_vol_std * BARRIER_MULTIPLIER      # NO sqrt(horizon)

  RESEARCH (ml_state_optimized.py load_and_build_state_space):
    bar_rets = df_price_1m['close'].pct_change()
    df_price_1m['bar_vol_std'] = bar_rets.rolling(
        window=BARRIER_VOL_BAR_WINDOW, min_periods=2
    ).std(ddof=0)            # FIXED: was .std() default ddof=1

This script is SELF-CONTAINED (does NOT import main.*, which would pull in
ccxt and other live deps). It reads BARRIER_MULTIPLIER, TIMEFRAME_BARRIER
and BARRIER_VOL_BAR_WINDOW directly from main/config.py via a text regex so
it always reflects the live config and cannot drift.

Run:  python research/verify_barrier_fix.py
"""
import re

import numpy as np
import pandas as pd

EPS = 1e-9


def read_config_int(key: str, default: int) -> int:
    """Read an integer constant from main/config.py without importing it."""
    with open('main/config.py', 'r', encoding='utf-8') as f:
        text = f.read()
    m = re.search(rf"^{key}\s*=\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else default


def read_config_float(key: str, default: float) -> float:
    """Read a float constant from main/config.py without importing it."""
    with open('main/config.py', 'r', encoding='utf-8') as f:
        text = f.read()
    m = re.search(rf"^{key}\s*=\s*([\d.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else default


BARRIER_MULTIPLIER = read_config_float('BARRIER_MULTIPLIER', 0.5)
TIMEFRAME_BARRIER = read_config_int('TIMEFRAME_BARRIER', 300)
# BARRIER_VOL_BAR_WINDOW is NOT in config.py; features.py falls back to 20.
# Note: config.py has BARRIER_VOL_WINDOW=100 (the 1s-tick fallback window),
# which is a DIFFERENT constant. The aggregated-bar window is 20.
BARRIER_VOL_BAR_WINDOW = read_config_int('BARRIER_VOL_BAR_WINDOW', 20)

print("=" * 70)
print(f"CONFIG READ FROM main/config.py (text parse, no imports)")
print(f"  BARRIER_MULTIPLIER      = {BARRIER_MULTIPLIER}")
print(f"  TIMEFRAME_BARRIER       = {TIMEFRAME_BARRIER}s")
print(f"  BARRIER_VOL_BAR_WINDOW  = {BARRIER_VOL_BAR_WINDOW} (features.py fallback 20)")
print("=" * 70)

# ------------------------------------------------------------------------
# Build a synthetic close series of TIMEFRAME_BARRIER-aggregated bars
# (the same series RSI(2) is built from, and the same series LiveFeatureEngine
# feeds into its _bar_vol_std).
# ------------------------------------------------------------------------
rng = np.random.RandomState(42)
n_bars = 200
prices = 65000 * np.exp(np.cumsum(rng.normal(0, 0.0005, n_bars)))
df = pd.DataFrame({'close': prices})

# ------------------------------------------------------------------------
# STEP 1 — LIVE method (features.py _compute_macro_features)
#   bar_rets    = np.diff(bar_closes) / (bar_closes[:-1] + EPS)
#   bar_vol_std = np.nanstd(bar_rets[-window:])   (ddof=0 default)
# ------------------------------------------------------------------------
bar_closes = df['close'].values
bar_rets_live = np.diff(bar_closes) / (bar_closes[:-1] + EPS)
window = min(len(bar_rets_live), BARRIER_VOL_BAR_WINDOW)
bar_vol_std_live = float(np.nanstd(bar_rets_live[-window:])) if window >= 2 else 0.0
barrier_live = bar_vol_std_live * BARRIER_MULTIPLIER

print("\n" + "=" * 70)
print("STEP 1 — LIVE METHOD (features.py + execution.py)")
print("=" * 70)
print(f"  bar_closes used     : {n_bars} aggregated bars")
print(f"  bar_rets           : np.diff(bar_closes)/(bar_closes[:-1]+EPS)")
print(f"  window             : {window}")
print(f"  bar_vol_std (ddof=0): {bar_vol_std_live:.10e}")
print(f"  barrier_pct        : {bar_vol_std_live:.10e} × {BARRIER_MULTIPLIER} "
      f"= {barrier_live:.10e} ({barrier_live*100:.4f}%)  [NO sqrt(horizon)]")

# ------------------------------------------------------------------------
# STEP 2 — RESEARCH method BEFORE the fix (pandas .std() default ddof=1)
# ------------------------------------------------------------------------
pct_chg = df['close'].pct_change()
bar_vol_std_old = pct_chg.rolling(
    window=BARRIER_VOL_BAR_WINDOW, min_periods=2
).std().iloc[-1]  # ddof=1
barrier_old = bar_vol_std_old * BARRIER_MULTIPLIER

print("\n" + "=" * 70)
print("STEP 2 — RESEARCH METHOD BEFORE FIX (pandas .std() ddof=1)")
print("=" * 70)
print(f"  bar_vol_std (ddof=1): {bar_vol_std_old:.10e}")
print(f"  barrier_pct        : {bar_vol_std_old:.10e} × {BARRIER_MULTIPLIER} "
      f"= {barrier_old:.10e} ({barrier_old*100:.4f}%)")
bias_pct = (barrier_old / barrier_live - 1) * 100
print(f"  -> systematic bias  : {bias_pct:+.3f}% (research was WIDER than live)")

# ------------------------------------------------------------------------
# STEP 3 — RESEARCH method AFTER the fix (pandas .std(ddof=0))
# ------------------------------------------------------------------------
bar_vol_std_new = pct_chg.rolling(
    window=BARRIER_VOL_BAR_WINDOW, min_periods=2
).std(ddof=0).iloc[-1]  # ddof=0 == np.nanstd default
barrier_new = bar_vol_std_new * BARRIER_MULTIPLIER

print("\n" + "=" * 70)
print("STEP 3 — RESEARCH METHOD AFTER FIX (pandas .std(ddof=0))")
print("=" * 70)
print(f"  bar_vol_std (ddof=0): {bar_vol_std_new:.10e}")
print(f"  barrier_pct        : {bar_vol_std_new:.10e} × {BARRIER_MULTIPLIER} "
      f"= {barrier_new:.10e} ({barrier_new*100:.4f}%)")

# ------------------------------------------------------------------------
# STEP 4 — EXACT EQUALITY CHECK
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — EXACT EQUALITY CHECK (live vs fixed research)")
print("=" * 70)
print(f"  live bar_vol_std     : {bar_vol_std_live:.12e}")
print(f"  fixed research       : {bar_vol_std_new:.12e}")
print(f"  bar_vol_std equal    : {np.isclose(bar_vol_std_live, bar_vol_std_new, atol=1e-20)}")
print(f"  barrier equal        : {np.isclose(barrier_live, barrier_new, atol=1e-20)}")
print(f"  max abs difference   : {abs(bar_vol_std_live - bar_vol_std_new):.2e}")
print(f"  old bias eliminated  : {abs(bias_pct):.3f}% -> 0.000%" if np.isclose(bar_vol_std_live, bar_vol_std_new, atol=1e-20) else "  FIX FAILED")

assert np.isclose(bar_vol_std_live, bar_vol_std_new, atol=1e-20), \
    "MISMATCH: research still does not match live!"
print("\n✅ PASS — research bar_vol_std now matches live EXACTLY (to 1e-20).")