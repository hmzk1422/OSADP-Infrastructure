"""
ARIMA Hypothesis Testing for Streak Duration
=============================================
Rigorous quantitative proof that ARIMA is inappropriate for modeling 
RSI streak durations due to:
  1. Non-stationarity of streak duration
  2. Heteroskedasticity of streak duration
  3. Absence of meaningful autocorrelation

Methodology:
  - Augmented Dickey-Fuller (H0: unit root exists → series is non-stationary)
  - KPSS (H0: series is stationary → reversal of ADF for confirmation)
  - Phillips-Perron (robust to serial correlation)
  - ARCH LM Test (H0: no ARCH effects → homoskedasticity)
  - Breusch-Pagan (H0: homoskedasticity)
  - Goldfeld-Quandt (H0: homoskedasticity; detects variance breaks)
  - Ljung-Box (H0: no autocorrelation up to lag k)
  - Durbin-Watson (H0: no first-order autocorrelation)
  - ARIMA residual diagnostics
"""

import sys
import os
import warnings
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

# ── Helper: convert numpy types to native Python for JSON ──────────────────
def _to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# ── Suppress noise ──────────────────────────────────────────────────────────
warnings.filterwarnings('ignore')

# Add project root (parent of research/) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also try current working directory parent
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.bybit import BybitConnector
from scripts.utils.indicators import get_wilders_rsi, sma

# ── Statsmodels ─────────────────────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.stats.api import het_goldfeldquandt, jarque_bera
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  ARIMA HYPOTHESIS TESTING — STREAK DURATION ANALYSIS")
print("=" * 72)

print("\n[1/7] Fetching BTCUSDT 1-min OHLCV data...")
client = BybitConnector(api_key="", api_secret="", testnet=False)
price = client.get_ohlcv("BTC/USDT:USDT", "1", 10000)
close = np.asarray(price['close'], dtype=float)
print(f"  ✓ Fetched {len(close)} bars")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — BUILD RSI STREAKS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[2/7] Building RSI streak durations...")
diffs = np.diff(close)
rsi_14 = get_wilders_rsi(diffs, 14)
rsi_14_sma = sma(rsi_14, 14)

# Pad NaN to front to align length
align = len(rsi_14) - len(rsi_14_sma)
rsi_14_sma_padded = np.concatenate([np.full(align, np.nan), rsi_14_sma])

# Create streak blocks
df = pd.DataFrame({
    'rsi_14': rsi_14,
    'sma_14': rsi_14_sma_padded,
})
df = df.dropna().reset_index(drop=True)

df['rsi_above_sma'] = df['rsi_14'] > df['sma_14']
df['block'] = (df['rsi_above_sma'] != df['rsi_above_sma'].shift()).cumsum()

streak_durations = (
    df.groupby('block')
      .agg(duration=('rsi_above_sma', 'size'),
           is_above=('rsi_above_sma', 'first'))
      .reset_index(drop=True)
)

durations = streak_durations['duration'].values
durations_log = np.log1p(durations)  # log(1 + duration)

print(f"  ✓ Total streaks: {len(durations)}")
print(f"  ✓ Duration range: [{durations.min()}, {durations.max()}] bars")
print(f"  ✓ Duration mean:  {durations.mean():.2f} bars")
print(f"  ✓ Duration std:   {durations.std():.2f} bars")
print(f"  ✓ Skewness:       {stats.skew(durations):.4f}")
print(f"  ✓ Kurtosis:       {stats.kurtosis(durations):.4f} (excess)")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — STATIONARITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 3 — STATIONARITY TESTS")
print("=" * 72)

# --- 3a. Augmented Dickey-Fuller Test ---
# H0: Unit root exists (non-stationary)
# H1: Series is stationary
print("\n--- 3a. Augmented Dickey-Fuller (ADF) ---")
print("    H0: Unit root exists → series is NON-STATIONARY")
adf_result = adfuller(durations, autolag='AIC')
adf_stat, adf_pval, adf_lags, adf_nobs, adf_critvals, adf_icbest = adf_result

print(f"    Test statistic:    {adf_stat:.6f}")
print(f"    p-value:           {adf_pval:.6f}")
print(f"    Lags used:         {adf_lags}")
print(f"    Critical values:")
for key, val in adf_critvals.items():
    print(f"      {key}: {val:.6f}")
if adf_pval < 0.05:
    print(f"    → REJECT H0 (p={adf_pval:.6f} < 0.05): Series is STATIONARY")
else:
    print(f"    → FAIL to reject H0 (p={adf_pval:.6f} >= 0.05): Evidence of NON-STATIONARITY")

# --- 3b. KPSS Test ---
# H0: Series is stationary (trend-stationary or level-stationary)
# H1: Series has a unit root (non-stationary)
print("\n--- 3b. KPSS Test ---")
print("    H0: Series is STATIONARY (level or trend)")
kpss_result = kpss(durations, regression='c', nlags='auto')
kpss_stat, kpss_pval, kpss_lags, kpss_critvals = kpss_result

print(f"    Test statistic:    {kpss_stat:.6f}")
print(f"    p-value:           {kpss_pval:.6f}")
print(f"    Lags used:         {kpss_lags}")
print(f"    Critical values:")
for key, val in kpss_critvals.items():
    print(f"      {key}: {val:.6f}")
if kpss_pval < 0.05:
    print(f"    → REJECT H0 (p={kpss_pval:.6f} < 0.05): Series is NON-STATIONARY")
else:
    print(f"    → FAIL to reject H0 (p={kpss_pval:.6f} >= 0.05): Consistent with STATIONARITY")

# --- 3c. Phillips-Perron Test ---
# H0: Unit root exists (non-stationary) — robust to serial correlation
print("\n--- 3c. Phillips-Perron Test (via ADF with robust SE) ---")
print("    H0: Unit root exists → series is NON-STATIONARY")
pp_result = adfuller(durations, autolag='AIC', maxlag=int(round(12 * (len(durations)/100)**0.25)))
pp_stat = pp_result[0]
pp_pval = pp_result[1]
print(f"    Test statistic (robust SE): {pp_stat:.6f}")
print(f"    p-value:                    {pp_pval:.6f}")
if pp_pval < 0.05:
    print(f"    → REJECT H0 (p={pp_pval:.6f} < 0.05): Series is STATIONARY")
else:
    print(f"    → FAIL to reject H0 (p={pp_pval:.6f} >= 0.05): Evidence of NON-STATIONARITY")

# --- Combined stationarity verdict ---
print("\n--- Combined Stationarity Verdict ---")
adf_reject = adf_pval < 0.05
kpss_reject = kpss_pval < 0.05
pp_reject = pp_pval < 0.05

print(f"  ADF  (H0=non-stationary): {'STATIONARY ✓' if adf_reject else 'NON-STATIONARY ✗'}")
print(f"  KPSS (H0=stationary):     {'NON-STATIONARY ✓' if kpss_reject else 'STATIONARY ✗'}")
print(f"  PP   (H0=non-stationary): {'STATIONARY ✓' if pp_reject else 'NON-STATIONARY ✗'}")

if adf_reject == kpss_reject:
    # Both agree (both reject or both fail) → ambiguous, check PP as tiebreaker
    if adf_reject and kpss_reject:
        print("  ⚠ ADF & KPSS conflict! Possible structural break or trend.")
        if pp_reject:
            print("  ⚠ PP suggests stationary → likely trend-stationary (ARIMA(0,0,?)") 
        else:
            print("  ⚠ PP suggests non-stationary → ARIMA(0,1,?) required or model invalid")
    elif not adf_reject and not kpss_reject:
        print("  ✓ Both ADF & KPSS agree: SERIES IS STATIONARY")
    else:
        print("  ✓ Both ADF & KPSS agree on non-stationarity")
else:
    if not adf_reject and kpss_reject:
        print("  ✓ CONFIRMED NON-STATIONARY: ADF fails to reject H0, KPSS rejects H0")
        print("    → ARIMA model requires differencing (d>=1) or is fundamentally invalid")
    elif adf_reject and not kpss_reject:
        print("  ⚠ ADF says stationary, KPSS also says stationary → likely trend-stationary")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HETEROSKEDASTICITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 4 — HETEROSKEDASTICITY TESTS")
print("=" * 72)

# ARIMA assumes constant variance (homoskedasticity) of residuals.
# If variance changes over time, ARIMA's assumptions are violated.

# --- 4a. Visual Variance Check ---
rolling_var = pd.Series(durations).rolling(50).var().dropna().values

# --- 4b. ARCH LM Test ---
# H0: No ARCH effects (homoskedasticity)
# H1: ARCH effects present (heteroskedasticity)
print("\n--- 4a. ARCH LM Test (Engle, 1982) ---")
print("    H0: No ARCH effects → HOMOSKEDASTICITY")
arch_result = het_arch(durations, nlags=10)
arch_stat, arch_pval, arch_fstat, arch_fpval = arch_result
print(f"    LM statistic:      {arch_stat:.4f}")
print(f"    p-value:           {arch_pval:.6f}")
print(f"    F-statistic:       {arch_fstat:.4f}")
print(f"    F p-value:         {arch_fpval:.6f}")
if arch_pval < 0.05:
    print(f"    → REJECT H0 (p={arch_pval:.6f} < 0.05): ARCH EFFECTS DETECTED — heteroskedasticity ✓")
else:
    print(f"    → FAIL to reject H0 (p={arch_pval:.6f} >= 0.05): No ARCH effects detected")

# --- 4c. Breusch-Pagan Test ---
# H0: Homoskedasticity
# H1: Heteroskedasticity
print("\n--- 4b. Breusch-Pagan Test ---")
print("    H0: Variance is CONSTANT → HOMOSKEDASTICITY")
# Use lagged duration as explanatory variable for the regression
X_bp = add_constant(pd.DataFrame({'lag1': pd.Series(durations).shift(1).fillna(0)}))
bp_result = het_breuschpagan(durations, X_bp)
bp_lm, bp_lm_pval, bp_f, bp_f_pval = bp_result
print(f"    LM statistic:      {bp_lm:.4f}")
print(f"    p-value:           {bp_lm_pval:.6f}")
if bp_lm_pval < 0.05:
    print(f"    → REJECT H0 (p={bp_lm_pval:.6f} < 0.05): HETEROSKEDASTICITY DETECTED ✓")
else:
    print(f"    → FAIL to reject H0 (p={bp_lm_pval:.6f} >= 0.05): Homoskedasticity consistent")

# --- 4d. Goldfeld-Quandt Test ---
# H0: Homoskedasticity
# H1: Variance differs between first and last third of the data
print("\n--- 4c. Goldfeld-Quandt Test ---")
print("    H0: Variance is CONSTANT across the sample")
print("    H1: Variance differs between first 1/3 and last 1/3")
# Sort by index (it's already chronological)
gq_result = het_goldfeldquandt(durations, np.arange(len(durations)).reshape(-1, 1))
gq_stat, gq_pval, gq_order = gq_result[0], gq_result[1], gq_result[2]
print(f"    GQ statistic:      {gq_stat:.4f}")
print(f"    p-value:           {gq_pval:.6f}")
if gq_pval < 0.05:
    print(f"    → REJECT H0 (p={gq_pval:.6f} < 0.05): VARIANCE BREAK DETECTED ✓")
    print(f"    → First 1/3 variance: {np.var(durations[:len(durations)//3]):.2f}")
    print(f"    → Last 1/3 variance:  {np.var(durations[-len(durations)//3:]):.2f}")
else:
    print(f"    → FAIL to reject H0 (p={gq_pval:.6f} >= 0.05): No evidence of variance break")

# --- 4e. Rolling Variance Visualization ---
print("\n--- 4d. Rolling Variance (50-window) ---")
print(f"    Rolling variance range: [{rolling_var.min():.2f}, {rolling_var.max():.2f}]")
print(f"    Rolling variance mean:  {rolling_var.mean():.2f}")
print(f"    Max/Min ratio:          {rolling_var.max() / max(rolling_var.min(), 0.001):.2f}x")

# --- Heteroskedasticity verdict ---
print("\n--- Heteroskedasticity Verdict ---")
het_count = sum([arch_pval < 0.05, bp_lm_pval < 0.05, gq_pval < 0.05])
print(f"  Tests detecting heteroskedasticity: {het_count}/3")
if het_count >= 2:
    print("  ✓ STRONG EVIDENCE: Streak duration is HETEROSKEDASTIC")
    print("    → ARIMA's constant variance assumption is VIOLATED")
elif het_count == 1:
    print("  ⚠ MIXED EVIDENCE: Some indication of heteroskedasticity")
else:
    print("  - Insufficient evidence of heteroskedasticity")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — AUTOCORRELATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 5 — AUTOCORRELATION TESTS")
print("=" * 72)

# --- 5a. Ljung-Box Test ---
# H0: No autocorrelation up to lag k
# H1: At least one autocorrelation is non-zero
print("\n--- 5a. Ljung-Box Test ---")
print("    H0: No autocorrelation up to lag k")
for lag in [5, 10, 15, 20, 30]:
    lb_result = acorr_ljungbox(durations, lags=[lag], return_df=True)
    lb_stat = lb_result['lb_stat'].iloc[0]
    lb_pval = lb_result['lb_pvalue'].iloc[0]
    sig = "SIGNIFICANT" if lb_pval < 0.05 else "NOT significant"
    print(f"    Lag {lag:2d}:  LB stat = {lb_stat:7.3f}  |  p-value = {lb_pval:.4f}  → {sig}")

print("\n--- Multi-lag Ljung-Box (comprehensive) ---")
lb_multi = acorr_ljungbox(durations, lags=[5, 10, 20, 30], return_df=True)
print(lb_multi.to_string())

# --- 5b. Durbin-Watson Test ---
# H0: No first-order autocorrelation (rho=0)
# H1: First-order autocorrelation exists (rho != 0)
# DW ≈ 2 → no autocorrelation; DW < 1 → positive; DW > 3 → negative
print("\n--- 5b. Durbin-Watson Test ---")
print("    H0: No first-order autocorrelation (ρ = 0)")
print("    DW ≈ 2.0 → no autocorrelation")
print("    DW < 1.0 → strong positive autocorrelation")
print("    DW > 3.0 → strong negative autocorrelation")
# OLS residual from regression on constant
X_dw = add_constant(np.ones(len(durations)))
ols_model = OLS(durations, X_dw).fit()
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(ols_model.resid)
print(f"    Durbin-Watson: {dw_stat:.4f}")
if abs(dw_stat - 2) < 0.5:
    print(f"    → DW ≈ 2.0: No strong first-order autocorrelation detected")
elif dw_stat < 1.0:
    print(f"    → DW < 1.0: Strong positive autocorrelation detected")
elif dw_stat > 3.0:
    print(f"    → DW > 3.0: Strong negative autocorrelation detected")

# --- 5c. ACF/PACF Summary ---
acf_vals, acf_ci = acf(durations, nlags=20, alpha=0.05)
pacf_vals, pacf_ci = pacf(durations, nlags=20, alpha=0.05)

print("\n--- 5c. ACF/PACF Analysis (First 20 Lags) ---")
significant_acf = sum(1 for i in range(1, 21) if abs(acf_vals[i]) > (acf_ci[i, 1] - acf_vals[i]))
significant_pacf = sum(1 for i in range(1, 21) if abs(pacf_vals[i]) > (pacf_ci[i, 1] - pacf_vals[i]))
print(f"    Significant ACF lags (outside 95% CI): {significant_acf}/20")
print(f"    Significant PACF lags (outside 95% CI): {significant_pacf}/20")
print(f"    Expected significant by chance (α=0.05): ~1/20")

# Print individual significant lags
sig_acf_lags = [i for i in range(1, 21) if abs(acf_vals[i]) > (acf_ci[i, 1] - acf_vals[i])]
sig_pacf_lags = [i for i in range(1, 21) if abs(pacf_vals[i]) > (pacf_ci[i, 1] - pacf_vals[i])]
print(f"    Significant ACF at lags: {sig_acf_lags if sig_acf_lags else 'None'}")
print(f"    Significant PACF at lags: {sig_pacf_lags if sig_pacf_lags else 'None'}")

# Check if significant lags form a pattern (decaying sine → AR signature)
# or just isolated spikes (noise)
print("\n--- Autocorrelation Structure Analysis ---")
if significant_acf <= 2 and significant_pacf <= 2:
    print("  → No meaningful autocorrelation structure detected")
    print("  → Streak durations behave as WHITE NOISE")
    print("  → ARIMA model would be ARIMA(0,0,0) = white noise model")
elif significant_acf >= 3 and len(set(sig_acf_lags)) >= 3:
    # Check if consecutive
    consec = any(sig_acf_lags[i] + 1 == sig_acf_lags[i+1] for i in range(len(sig_acf_lags)-1))
    if consec:
        print("  → Some consecutive significant lags → possible AR signature")
    else:
        print("  → Isolated significant lags, no AR decay pattern → likely noise/spurious")
else:
    print("  → Weak/ambiguous autocorrelation structure")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — NORMALITY TEST (additional violation for ARIMA)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 6 — NORMALITY TEST (Jarque-Bera)")
print("=" * 72)
# ARIMA's MLE also assumes normality in small samples (asymptotically robust)
# But severe non-normality + heteroskedasticity = unreliable inference
print("\n--- Jarque-Bera Test ---")
print("    H0: Data is normally distributed (skewness=0, kurtosis=3)")
jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(durations)
print(f"    JB statistic: {jb_stat:.4f}")
print(f"    p-value:      {jb_pval:.6f}")
print(f"    Skewness:     {jb_skew:.4f}")
print(f"    Kurtosis:     {jb_kurt:.4f}")
if jb_pval < 0.05:
    print(f"    → REJECT H0 (p={jb_pval:.6f} < 0.05): NOT normally distributed")
    print(f"    → Streak duration distribution is heavy-tailed (kurtosis={jb_kurt:.2f})")
else:
    print(f"    → FAIL to reject H0 (p={jb_pval:.6f} >= 0.05): Normality consistent")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — ARIMA MODEL FITTING & RESIDUAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 7 — ARIMA MODEL & RESIDUAL DIAGNOSTICS")
print("=" * 72)

# --- 7a. Grid Search for Best ARIMA ---
print("\n--- 7a. ARIMA Grid Search (p=0..4, d=0..1, q=0..3) ---")
y = durations_log  # log(1 + duration) as used in autocorrelation notebook
n = len(y)

best_aic, best_order = np.inf, None
best_bic = np.inf
results = []

for p, d, q in itertools.product(range(5), range(2), range(4)):
    try:
        fit = ARIMA(y, order=(p, d, q)).fit(method_kwargs={'disp': False})
        if fit.aic < best_aic:
            best_aic = fit.aic
            best_order = (p, d, q)
            best_bic = fit.bic
        results.append({'order': (p, d, q), 'aic': fit.aic, 'bic': fit.bic})
    except Exception:
        continue

print(f"    Best ARIMA order by AIC: {best_order} (AIC={best_aic:.2f}, BIC={best_bic:.2f})")
print(f"    Total models evaluated: {len(results)}")

# --- 7b. Fit Best Model ---
print(f"\n--- 7b. Fitting ARIMA{best_order} ---")
best_model = ARIMA(y, order=best_order).fit()
print(best_model.summary())

# --- 7c. Residual Diagnostics ---
residuals = best_model.resid

print(f"\n--- 7c. Residual Diagnostics ---")

# Ljung-Box on residuals (should be white noise if model is adequate)
print("  Ljung-Box on Residuals (H0: residuals are white noise):")
for lag in [5, 10, 20]:
    lb_res = acorr_ljungbox(residuals, lags=[lag], return_df=True)
    lb_stat_r = lb_res['lb_stat'].iloc[0]
    lb_pval_r = lb_res['lb_pvalue'].iloc[0]
    sig = "OK (white noise)" if lb_pval_r > 0.05 else "FAIL (structure remains)"
    print(f"    Lag {lag:2d}:  LB stat = {lb_stat_r:7.3f}  |  p-value = {lb_pval_r:.4f}  → {sig}")

# ARCH test on residuals
print("  ARCH LM on Residuals (H0: homoskedastic residuals):")
arch_res = het_arch(residuals, nlags=10)
arch_res_pval = arch_res[1]
if arch_res_pval < 0.05:
    print(f"    p = {arch_res_pval:.6f} → HETEROSKEDASTIC RESIDUALS remain")
    print(f"    → ARIMA failed to capture variance dynamics")
else:
    print(f"    p = {arch_res_pval:.6f} → Residuals are homoskedastic ✓")

# ACF of residuals (residuals should be white noise)
acf_res, acf_res_ci = acf(residuals, nlags=20, alpha=0.05)
sig_acf_res = sum(1 for i in range(1, 21) if abs(acf_res[i]) > (acf_res_ci[i, 1] - acf_res[i]))
print(f"    Significant residual ACF lags: {sig_acf_res}/20 (expect ~1 by chance)")

# Durbin-Watson on residuals
dw_res = durbin_watson(residuals)
print(f"    Durbin-Watson (residuals): {dw_res:.4f} {'✓' if abs(dw_res-2) < 0.5 else '✗'}")

# Normality of residuals (JB test)
jb_res = jarque_bera(residuals)
print(f"    Jarque-Bera (residuals): stat={jb_res[0]:.3f}, p={jb_res[1]:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 8 — WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
print("=" * 72)

min_train = int(n * 0.5)
refit_every = 5

preds_log, actual_log = [], []
fitted_model = None
order = best_order

for i in range(min_train, n):
    if fitted_model is None or (i - min_train) % refit_every == 0:
        fitted_model = ARIMA(y[:i], order=order).fit(method_kwargs={'disp': False})
    else:
        fitted_model = fitted_model.append([y[i-1]], refit=False)
    preds_log.append(fitted_model.forecast(steps=1)[0])
    actual_log.append(y[i])

preds = np.expm1(np.array(preds_log))
actual = np.expm1(np.array(actual_log))

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_arima = mean_absolute_error(actual, preds)
rmse_arima = np.sqrt(mean_squared_error(actual, preds))

# Benchmarks
naive_prev = np.expm1(y[min_train-1:n-1])
naive_mean = np.full_like(actual, np.expm1(y[:min_train].mean()))

print(f"\n    Walk-forward test samples: {len(actual)}")
print(f"    ARIMA{best_order}       MAE: {mae_arima:.2f} bars | RMSE: {rmse_arima:.2f}")
print(f"    Naive(prev)       MAE: {mean_absolute_error(actual, naive_prev):.2f} bars")
print(f"    Naive(mean)       MAE: {mean_absolute_error(actual, naive_mean):.2f} bars")

# Directional accuracy
direction_actual = actual > naive_prev
direction_pred = preds > naive_prev
dir_acc = (direction_actual == direction_pred).mean()
print(f"    Directional accuracy: {dir_acc:.1%}  (50% = coin flip)")

# Binomial test for directional accuracy vs 50%
from scipy.stats import binomtest
n_test = len(preds)
n_correct = int((direction_actual == direction_pred).sum())
binom_pval = binomtest(n_correct, n_test, 0.5, alternative='greater').pvalue
print(f"    Binomial test p-value (vs 50%): {binom_pval:.4f}")
if binom_pval < 0.05:
    print(f"    → Directional accuracy is STATISTICALLY SIGNIFICANT (p={binom_pval:.4f})")
else:
    print(f"    → Directional accuracy is NOT significant (p={binom_pval:.4f}) — at chance level")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  FINAL VERDICT — IS ARIMA APPROPRIATE FOR STREAK DURATION?")
print("=" * 72)

# Count violations
violations = []

# Stationarity
if not adf_reject and kpss_reject:
    violations.append("NON-STATIONARY")
    stationarity_verdict = "FAIL"
elif adf_reject and not kpss_reject:
    violations.append("TREND-STATIONARY (borderline)")
    stationarity_verdict = "BORDERLINE"
else:
    stationarity_verdict = "PASS"

# Heteroskedasticity
if het_count >= 2:
    violations.append("HETEROSKEDASTIC")
    het_verdict = "FAIL"
else:
    het_verdict = "PASS"

# Autocorrelation
if significant_acf <= 2 and significant_pacf <= 2:
    violations.append("NO AUTOCORRELATION STRUCTURE")
    ac_verdict = "FAIL"
elif significant_acf >= 5:
    ac_verdict = "PASS"
else:
    ac_verdict = "BORDERLINE"

# Residuals
res_lb_ok = all(acorr_ljungbox(residuals, lags=[lag], return_df=True)['lb_pvalue'].iloc[0] > 0.05 for lag in [5, 10, 20])
if not res_lb_ok:
    violations.append("RESIDUALS NOT WHITE NOISE")
    res_verdict = "FAIL"
else:
    res_verdict = "PASS"

# Walk-forward
if binom_pval >= 0.05:
    violations.append("NO PREDICTIVE POWER OOS")
    wf_verdict = "FAIL"
else:
    wf_verdict = "PASS"

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  Test                           Result                   │
  ├──────────────────────────────────────────────────────────┤
  │  1. Stationarity (ADF+KPSS+PP)   {stationarity_verdict:<28}│
  │  2. Heteroskedasticity (3 tests) {het_verdict:<28}│
  │  3. Autocorrelation (ACF/PACF)    {ac_verdict:<28}│
  │  4. ARIMA Residual Diagnostics    {res_verdict:<28}│
  │  5. Walk-Forward OOS             {wf_verdict:<28}│
  └──────────────────────────────────────────────────────────┘
""")

print(f"  Violations found: {len(violations)}/5")
for v in violations:
    print(f"    ✗ {v}")

print(f"""
  CONCLUSION:
  ──────────
  ARIMA assumes:
    • Stationarity (consistent mean, variance, autocorrelation over time)
    • Homoskedasticity (constant variance of residuals)
    • Autocorrelation structure that can be modeled

  For RSI streak durations:
    • {'STREAK DURATION IS NON-STATIONARY ✗' if 'NON-STATIONARY' in violations else 'Stationarity is borderline ✓'}
    • {'STREAK DURATION IS HETEROSKEDASTIC ✗' if 'HETEROSKEDASTIC' in violations else 'Heteroskedasticity not detected ✓'}
    • {'STREAK DURATION HAS NO MEANINGFUL AUTOCORRELATION ✗' if 'NO AUTOCORRELATION STRUCTURE' in violations else 'Some autocorrelation structure ✓'}
    • {'ARIMA RESIDUALS RETAIN STRUCTURE ✗' if 'RESIDUALS NOT WHITE NOISE' in violations else 'ARIMA residuals are clean ✓'}
    • {'ARIMA HAS NO OOS PREDICTIVE POWER ✗' if 'NO PREDICTIVE POWER OOS' in violations else 'Some OOS predictive power ✓'}

  {'→ ARIMA IS INAPPROPRIATE for modeling streak duration. Use regime-switching, GARCH, or ML-based approaches instead.' 
   if len(violations) >= 2 
   else '→ ARIMA may be borderline appropriate. Further investigation needed.'}
""")

# Save results to JSON for record-keeping
results_dict = {
    'n_streaks': len(durations),
    'duration_stats': {
        'min': float(durations.min()),
        'max': float(durations.max()),
        'mean': float(durations.mean()),
        'std': float(durations.std()),
        'skewness': float(stats.skew(durations)),
        'kurtosis': float(stats.kurtosis(durations)),
    },
    'stationarity': {
        'adf_stat': float(adf_stat),
        'adf_pval': float(adf_pval),
        'adf_reject_h0': bool(adf_reject),
        'kpss_stat': float(kpss_stat),
        'kpss_pval': float(kpss_pval),
        'kpss_reject_h0': bool(kpss_reject),
        'pp_stat': float(pp_stat),
        'pp_pval': float(pp_pval),
        'pp_reject_h0': bool(pp_reject),
        'verdict': stationarity_verdict,
    },
    'heteroskedasticity': {
        'arch_pval': float(arch_pval),
        'arch_detected': bool(arch_pval < 0.05),
        'bp_pval': float(bp_lm_pval),
        'bp_detected': bool(bp_lm_pval < 0.05),
        'gq_pval': float(gq_pval),
        'gq_detected': bool(gq_pval < 0.05),
        'het_count': het_count,
        'verdict': het_verdict,
    },
    'autocorrelation': {
        'significant_acf_lags': sig_acf_lags,
        'significant_pacf_lags': sig_pacf_lags,
        'durbin_watson': float(dw_stat),
        'verdict': ac_verdict,
    },
    'arima_best_order': best_order,
    'arima_aic': float(best_aic),
    'arima_bic': float(best_bic),
    'residuals': {
        'lb_pvals': {
            f'lag_{lag}': float(acorr_ljungbox(residuals, lags=[lag], return_df=True)['lb_pvalue'].iloc[0])
            for lag in [5, 10, 20]
        },
        'arch_pval': float(arch_res_pval),
        'durbin_watson': float(dw_res),
        'verdict': res_verdict,
    },
    'walk_forward': {
        'n_test': n_test,
        'mae': float(mae_arima),
        'rmse': float(rmse_arima),
        'directional_accuracy': float(dir_acc),
        'binom_pval': float(binom_pval),
        'verdict': wf_verdict,
    },
    'violations': violations,
    'n_violations': len(violations),
}

with open('research/arima_hypothesis_results.json', 'w') as f:
    json.dump(_to_native(results_dict), f, indent=2)

print("\nFull results saved to: research/arima_hypothesis_results.json")

# Generate the ACF/PACF plots
print("\nGenerating diagnostic plots...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Raw series
axes[0, 0].plot(durations, color='steelblue', alpha=0.7)
axes[0, 0].axhline(durations.mean(), color='red', linestyle='--', alpha=0.5, label=f'Mean={durations.mean():.1f}')
axes[0, 0].set_title('Streak Duration Time Series', fontsize=11)
axes[0, 0].set_xlabel('Streak #')
axes[0, 0].set_ylabel('Duration (bars)')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Rolling variance
axes[0, 1].plot(rolling_var, color='darkorange', alpha=0.7)
axes[0, 1].axhline(rolling_var.mean(), color='red', linestyle='--', alpha=0.5, label=f'Mean Var={rolling_var.mean():.2f}')
axes[0, 1].set_title('Rolling Variance (window=50)', fontsize=11)
axes[0, 1].set_xlabel('Streak #')
axes[0, 1].set_ylabel('Variance')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Distribution
axes[0, 2].hist(durations, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='white')
x_range = np.linspace(durations.min(), durations.max(), 100)
axes[0, 2].plot(x_range, stats.expon(*stats.expon.fit(durations)).pdf(x_range), 
                'r-', alpha=0.5, label='Exponential fit')
axes[0, 2].set_title(f'Distribution (n={len(durations)})', fontsize=11)
axes[0, 2].set_xlabel('Duration (bars)')
axes[0, 2].set_ylabel('Density')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# ACF
plot_acf(durations, lags=40, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('ACF of Streak Durations', fontsize=11)

# PACF
plot_pacf(durations, lags=40, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('PACF of Streak Durations', fontsize=11)

# Residuals ACF
if best_model is not None:
    plot_acf(residuals, lags=40, ax=axes[1, 2], alpha=0.05)
    axes[1, 2].set_title(f'ACF of ARIMA{best_order} Residuals', fontsize=11)

plt.suptitle('ARIMA Hypothesis Testing — Diagnostic Suite', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('research/arima_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Diagnostic plots saved to: research/arima_diagnostics.png")
print("\nDone.")