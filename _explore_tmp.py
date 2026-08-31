from research.regime_buckets import run_buckets_all
from research.regime_data import load_study_frame

df = load_study_frame()
r = run_buckets_all(df)
b = r['buckets']
for v in ['vol_prior20_ret_std', 'ma_slope_5_open', 'inst_ratio_open', 'strat_prev10_median_gap']:
    print('===', v)
    print(b['tables'][v][['bucket', 'n', 'mean_r', 'ci_lo', 'ci_hi', 'win_rate',
                          'p_3r', 'pf', 'med_mae_pct', 'med_mfe_pct']].round(4).to_string())
print('--- interactions sample ---')
print(r['interactions'].head(15).to_string())
import pandas as pd
import numpy as np

df = pd.read_csv('build/walk_forward_trade_log_v3.csv')
df['dt'] = pd.to_datetime(df['datetime'])
df = df.sort_values('dt').reset_index(drop=True)

def fix(s):
    return s.str.strip('+').str.replace(r'^56\d{3}-', '2000-', regex=True)

df['entry_p'] = pd.to_datetime(fix(df['entry_dt']), format='mixed')
df['exit_p'] = pd.to_datetime(fix(df['exit_dt']), format='mixed')
df['dur'] = (df['exit_p'] - df['entry_p']).dt.total_seconds()
print('dur desc:')
print(df['dur'].describe())

s = df[df['threshold'] == 0.2].copy()
print('n thr0.2:', len(s))
print('ret desc thr0.2:')
print(s['ret'].describe())
print('by month counts:')
print(s.groupby(s['dt'].dt.to_period('M')).size().to_string())
print('win rate thr0.2:', (s['ret'] > 0).mean().round(4))
print('mean ret thr0.2:', s['ret'].mean().round(4))

prev_exit = df['exit_p'].cummax().shift(1)
print('overlap frac (full):', round((df['entry_p'] < prev_exit).mean(), 4))
sp_exit = s['exit_p'].cummax().shift(1)
print('overlap frac thr0.2:', round((s['entry_p'] < sp_exit).mean(), 4))
print()
print('dur by ret sign (full):')
print(df.assign(win=df['ret'] > 0).groupby('win')['dur'].median())
print('dur by ret sign (thr0.2):')
print(s.assign(win=s['ret'] > 0).groupby('win')['dur'].median())

# quick look at feature distributions at open (entry-safe)
cols = ['ret', 'spread_bps_z_open', 'ma_slope_5_open', 'rsi_x_ma_open',
        'ma_accel_open', 'price_roc_3_open', 'rsi_2_open', 'sma_14_open',
        'ofi_z_open', 'cvd_delta_z_open', 'cvd_slope_z_open', 'absorption_z_open',
        'tick_autocorr_open', 'inst_ratio_open', 'delta_divergence_z_open',
        'depth_imbalance_z_open']
print()
print(s[cols].describe().T[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())

# correlation among open features
print()
print('corr matrix (open features):')
print(s[cols[1:]].corr().round(2).to_string())