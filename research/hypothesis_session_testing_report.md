# Session-Regime Hypothesis Testing - Walk-Forward Trade Expectancy

_Input: `walk_forward_trade_log_v3.csv` (out-of-sample, realized trades)_
_Trades loaded: 11315_

**Filter**: production threshold = `P(WIN) >= 0.50` (3352 trades).

## 1. Headline sample - Weekday, production threshold

Session               | n   | PWE     | Sortino
----------------------+-----+---------+--------
Asia (00:00-08:00)    | 823 | -0.0625 | -0.181 
EU-Asia (08:00-09:00) | 98  | -0.0714 | -0.156 
EU (09:00-13:30)      | 418 | 0.0450  | 0.153  
EU-US (13:30-16:00)   | 210 | -0.0567 | -0.170 
US (16:00-20:00)      | 360 | -0.0494 | -0.148 
Pacific (20:00-00:00) | 449 | -0.0198 | -0.065 

## 2. Kruskal-Wallis H-test (non-parametric equality of medians)

H-statistic = **42.765**, p-value = **4.123e-08**
Conclusion: REJECT H0 (medians not all equal across sessions at alpha=0.05).

K-W only proves 'at least one session differs'. Dunn's below shows which.

## 3. Dunn's post-hoc pairwise test (Bonferroni-corrected)

Pair                                           | z      | p         | p_adj (Bonf) | sig<0.05
-----------------------------------------------+--------+-----------+--------------+---------
Asia (00:00-08:00) vs EU-Asia (08:00-09:00)    | 1.975  | 0.04827   | 0.7241       |         
Asia (00:00-08:00) vs EU (09:00-13:30)         | -4.856 | 1.197e-06 | 1.795e-05    | YES     
Asia (00:00-08:00) vs EU-US (13:30-16:00)      | -3.921 | 8.808e-05 | 0.001321     | YES     
Asia (00:00-08:00) vs US (16:00-20:00)         | -2.652 | 0.007996  | 0.1199       |         
Asia (00:00-08:00) vs Pacific (20:00-00:00)    | -1.965 | 0.04945   | 0.7418       |         
EU-Asia (08:00-09:00) vs EU (09:00-13:30)      | -4.479 | 7.492e-06 | 0.0001124    | YES     
EU-Asia (08:00-09:00) vs EU-US (13:30-16:00)   | -4.203 | 2.632e-05 | 0.0003947    | YES     
EU-Asia (08:00-09:00) vs US (16:00-20:00)      | -3.323 | 0.00089   | 0.01335      | YES     
EU-Asia (08:00-09:00) vs Pacific (20:00-00:00) | -2.927 | 0.003426  | 0.05139      |         
EU (09:00-13:30) vs EU-US (13:30-16:00)        | -0.136 | 0.892     | 1            |         
EU (09:00-13:30) vs US (16:00-20:00)           | 1.726  | 0.08441   | 1            |         
EU (09:00-13:30) vs Pacific (20:00-00:00)      | 2.595  | 0.009448  | 0.1417       |         
EU-US (13:30-16:00) vs US (16:00-20:00)        | 1.561  | 0.1185    | 1            |         
EU-US (13:30-16:00) vs Pacific (20:00-00:00)   | 2.247  | 0.02461   | 0.3692       |         
US (16:00-20:00) vs Pacific (20:00-00:00)      | 0.740  | 0.4595    | 1            |         


## 4. OLS regression + Newey-West (HAC) standard errors

`ret_t = b0(Asia) + sum_j b_j*session_j + e_t` — each `b_j` is the mean edge difference vs Asia. HAC SEs correct for autocorrelation + heteroskedasticity.

_n = 2358 weekday trades | baseline = `Asia (00:00-08:00)` | Newey-West maxlags = 8_

term                  | coef(R) | HAC SE | t      | p-value | 95% CI             | sig<0.05
----------------------+---------+--------+--------+---------+--------------------+---------
(Asia baseline)       | -0.0625 | 0.0312 | -2.002 | 0.04528 | [-0.1236, -0.0013] | YES     
EU-Asia (08:00-09:00) | -0.0090 | 0.1345 | -0.067 | 0.9468  | [-0.2726, +0.2547] |         
EU (09:00-13:30)      | +0.1074 | 0.0582 | +1.846 | 0.06495 | [-0.0067, +0.2215] |         
EU-US (13:30-16:00)   | +0.0058 | 0.0559 | +0.104 | 0.9176  | [-0.1038, +0.1154] |         
US (16:00-20:00)      | +0.0130 | 0.0508 | +0.256 | 0.7978  | [-0.0865, +0.1125] |         
Pacific (20:00-00:00) | +0.0426 | 0.0533 | +0.800 | 0.4236  | [-0.0618, +0.1470] |         

No session is significantly negative relative to baseline at alpha=0.05 (HAC).

### 4b. Robustness note — no session-clustered SEs

Session-clustered (panel) standard errors were intentionally NOT reported: cluster-robust inference is unreliable with only 6 session clusters (needs ~50+), and here it degenerates to SE=0. Newey-West (HAC) is the correct robust method for this autocorrelated microstructure series and is used throughout.
Intercept (Asia mean) = -0.0625 R/trade.

## 5. Sensitivity — Weekday vs All-days expectancy (PWE) by session

Session               | n_wd | PWE_wd  | n_all | PWE_all
----------------------+------+---------+-------+--------
Asia (00:00-08:00)    | 823  | -0.0625 | 1143  | -0.0766
EU-Asia (08:00-09:00) | 98   | -0.0714 | 135   | -0.0274
EU (09:00-13:30)      | 418  | 0.0450  | 651   | 0.0424 
EU-US (13:30-16:00)   | 210  | -0.0567 | 310   | -0.0845
US (16:00-20:00)      | 360  | -0.0494 | 482   | -0.0309
Pacific (20:00-00:00) | 449  | -0.0198 | 631   | -0.0084

## 6. Interpretation notes

- 'Asia is the highest edge' is evidenced if its pairwise Dunn p_adj < 0.05 vs the others (or its OLS coefficient is the positive max / the others are negative-sig).
- 'US/EU/Pacific are noise/negative' needs the OLS column: a significantly negative coefficient relative to Asia means the edge decays to (or below) zero.
- Watch `n`: a small US/Pacific sample gives low power — a non-significant p there is 'not proven different', not 'proven equal'.
- Dunn's already Bonferroni-adjusts all 15 pairwise comparisons (family-wise error).
- Sessions are assigned by the entry bar's time-of-day (same as the live-stat table).

---
_Run: read-only on realized trades; no feature/training code touched._
