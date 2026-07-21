"""TTR Analysis runner script."""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.bybit import BybitConnector
from utils.indicators import get_wilders_rsi, sma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('=' * 72)
print('  TTR ANALYSIS - TIME TO REVERSION')
print('=' * 72)

print('\n[1/4] Fetching BTCUSDT 1-min OHLCV data...')
client = BybitConnector(api_key='', api_secret='', testnet=False)
price = client.get_ohlcv('BTC/USDT:USDT', '1', 10000)
close = np.asarray(price['close'], dtype=float)
high = np.asarray(price['high'], dtype=float)
low = np.asarray(price['low'], dtype=float)
print(f'  OK Fetched {len(close)} bars')

print('\n[2/4] Computing indicators...')
delta = np.diff(close, prepend=close[0])
rsi_14 = get_wilders_rsi(delta, 14)
rsi_14_sma = sma(rsi_14, 14)
rsi_2 = get_wilders_rsi(delta, 2)
align_sma = len(rsi_14) - len(rsi_14_sma)
rsi_14_sma_padded = np.concatenate([np.full(align_sma, np.nan), rsi_14_sma])
print('  OK Indicators computed')

print('\n[3/4] Detecting TTR events...')
df = pd.DataFrame({
    'rsi_14': rsi_14,
    'rsi_14_sma': rsi_14_sma_padded,
    'rsi_2': rsi_2,
})
df = df.dropna().reset_index(drop=True)
df['regime_bull'] = df['rsi_14'] > df['rsi_14_sma']

MAX_LOOKAHEAD = 200


def detect_ttr_events(df, rsi_col, threshold, direction, regime_col, regime_value,
                      max_lookahead=MAX_LOOKAHEAD):
    rsi = df[rsi_col].values
    regime = df[regime_col].values
    ttr_values = []
    i = 0
    while i < len(df) - 5:
        if regime[i] != regime_value:
            i += 1
            continue
        if direction == 'below' and rsi[i] < threshold:
            pass
        elif direction == 'above' and rsi[i] > threshold:
            pass
        else:
            i += 1
            continue
        lookahead = min(max_lookahead, len(df) - i - 1)
        found = False
        if direction == 'below':
            for j in range(1, lookahead):
                if rsi[i + j] >= 50:
                    ttr_values.append(j)
                    found = True
                    i += j
                    break
        else:
            for j in range(1, lookahead):
                if rsi[i + j] <= 50:
                    ttr_values.append(j)
                    found = True
                    i += j
                    break
        if not found:
            i += 1
    return np.array(ttr_values)


ttr_os_bull = detect_ttr_events(df, 'rsi_2', 30, 'below', 'regime_bull', True)
ttr_os_bear = detect_ttr_events(df, 'rsi_2', 30, 'below', 'regime_bull', False)
ttr_ob_bull = detect_ttr_events(df, 'rsi_2', 70, 'above', 'regime_bull', True)
ttr_ob_bear = detect_ttr_events(df, 'rsi_2', 70, 'above', 'regime_bull', False)

print(f'  OS Bull: {len(ttr_os_bull)}, OS Bear: {len(ttr_os_bear)}, '
      f'OB Bull: {len(ttr_ob_bull)}, OB Bear: {len(ttr_ob_bear)}')

print('\n[4/4] Statistical tests...')


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    )
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else 0.0


def run_test(name, ga, gb, la, lb):
    print(f'\n--- {name} ---')
    print(f'  {la}: n={len(ga):5d}, mean TTR={np.mean(ga):7.2f}, '
          f'median={np.median(ga):7.2f}, std={np.std(ga):.2f}')
    print(f'  {lb}: n={len(gb):5d}, mean TTR={np.mean(gb):7.2f}, '
          f'median={np.median(gb):7.2f}, std={np.std(gb):.2f}')
    if len(ga) > 0 and len(gb) > 0:
        stat, pval = mannwhitneyu(ga, gb, alternative='two-sided')
        d = cohens_d(ga, gb)
        print(f'  MW U: stat={stat:.1f}, p={pval:.6f}')
        print(f'  Cohens d: {d:.4f}')
        if abs(d) < 0.2:
            eff_str = 'negligible'
        elif abs(d) < 0.5:
            eff_str = 'small'
        elif abs(d) < 0.8:
            eff_str = 'medium'
        else:
            eff_str = 'large'
        print(f'  Effect size: {eff_str}')
        if pval < 0.05:
            if np.mean(ga) < np.mean(gb):
                print(f'  CONFIRMED: {la} reverts FASTER - p={pval:.6f}')
            else:
                print(f'  CONTRADICTED: {lb} reverts faster')
        else:
            print(f'  Not significant (p={pval:.4f})')
        return stat, float(pval), d
    return 0, 1.0, 0.0


print('\nHYPOTHESIS 1: OVERSOLD - Bull vs Bear regime')
h1_stat, h1_pval, h1_d = run_test(
    'OS Bull vs Bear', ttr_os_bull, ttr_os_bear,
    'Bullish(RSI14>SMA14)', 'Bearish(RSI14<SMA14)'
)

print('\nHYPOTHESIS 2: OVERBOUGHT - Bear vs Bull regime')
h2_stat, h2_pval, h2_d = run_test(
    'OB Bear vs Bull', ttr_ob_bear, ttr_ob_bull,
    'Bearish(RSI14<SMA14)', 'Bullish(RSI14>SMA14)'
)

# Summary
print('\n' + '=' * 72)
print('  SUMMARY')
print('=' * 72)
h1_str = 'CONFIRMED' if (h1_pval < 0.05 and np.mean(ttr_os_bull) < np.mean(ttr_os_bear)) else \
         'CONTRADICTED' if h1_pval < 0.05 else 'NOT SIGNIFICANT'
h2_str = 'CONFIRMED' if (h2_pval < 0.05 and np.mean(ttr_ob_bear) < np.mean(ttr_ob_bull)) else \
         'CONTRADICTED' if h2_pval < 0.05 else 'NOT SIGNIFICANT'
print(f'\n  H1 (Oversold+Bull->faster): {h1_str}  (p={h1_pval:.6f}, d={h1_d:.3f})')
print(f'  H2 (Overbought+Bear->faster): {h2_str}  (p={h2_pval:.6f}, d={h2_d:.3f})')

header = '\n  DETAILED STATISTICS:'
print(header)
header_line = '    {:>25} {:>6} {:>9} {:>8} {:>7}'.format('', 'n', 'Mean TTR', 'Median', 'Std')
print(header_line)
print('    ' + '-' * 55)
if len(ttr_os_bull) > 0:
    print('    {:>25} {:>6d} {:>8.1f} {:>7.1f} {:>7.1f}'.format(
        'Oversold + Bull regime', len(ttr_os_bull), np.mean(ttr_os_bull),
        np.median(ttr_os_bull), np.std(ttr_os_bull)))
if len(ttr_os_bear) > 0:
    print('    {:>25} {:>6d} {:>8.1f} {:>7.1f} {:>7.1f}'.format(
        'Oversold + Bear regime', len(ttr_os_bear), np.mean(ttr_os_bear),
        np.median(ttr_os_bear), np.std(ttr_os_bear)))
if len(ttr_ob_bull) > 0:
    print('    {:>25} {:>6d} {:>8.1f} {:>7.1f} {:>7.1f}'.format(
        'Overbought + Bull regime', len(ttr_ob_bull), np.mean(ttr_ob_bull),
        np.median(ttr_ob_bull), np.std(ttr_ob_bull)))
if len(ttr_ob_bear) > 0:
    print('    {:>25} {:>6d} {:>8.1f} {:>7.1f} {:>7.1f}'.format(
        'Overbought + Bear regime', len(ttr_ob_bear), np.mean(ttr_ob_bear),
        np.median(ttr_ob_bear), np.std(ttr_ob_bear)))

print('\n' + '=' * 72)
print('  END OF ANALYSIS')
print('=' * 72)