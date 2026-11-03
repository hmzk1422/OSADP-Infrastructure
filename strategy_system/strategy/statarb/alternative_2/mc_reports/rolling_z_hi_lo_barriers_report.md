# Monte Carlo walk-forward: per-chunk report - rolling z / hi-lo barriers

- bar `1min` | chunk (train = OOS window) **5,000** bars | purge 1 | seed `7`
- strategy direction: **INVERTED (strategy_side=2)** | entry_z 2 | barrier_size 2 | rr 0.5 | z-score rolling | barriers hilo
- segments 5 | chunks 15

Train and OOS ADF / half-life are measured on the same traded (asset-only) spread with the same function. `p_eg_train` / `half_life_screen` are the beam-search values of the selected (full, feature-adjusted) spread.

## Segments (random dates)

| segment | bars | start | end | chunks |
|---|---|---|---|---|
| 1 | 20000 | 2022-05-27 04:48 | 2022-06-10 02:10 | 3 |
| 2 | 30000 | 2025-10-17 04:48 | 2025-11-07 00:52 | 5 |
| 3 | 25000 | 2026-01-10 04:48 | 2026-01-27 13:31 | 4 |
| 4 | 10000 | 2020-10-12 04:48 | 2020-10-19 03:28 | 1 |
| 5 | 15000 | 2023-08-26 04:48 | 2023-09-05 14:49 | 2 |

## Train vs OOS stationarity (all chunks)

| window | median_adf_p | share_adf_p<0.05 | median_half_life | median_phi | median_spread_std |
|---|---|---|---|---|---|
| train | 0.0029 | 93% | 102.1 | 0.9932 | 0.0044 |
| OOS | 0.2776 | 20% | 373.3 | 0.9981 | 0.0085 |

## Per segment: one row per 5,000-bar chunk

### Segment 1: 20,000 bars, 2022-05-27 .. 2022-06-10

| chunk | train_start | oos_start | hedge | p_eg_train | adf_p_train | adf_p_oos | half_life_train | half_life_oos | phi_train | phi_oos | z_mean_oos | trades | tp_rate | mean_r | sum_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2022-05-27 04:48 | 2022-05-30 16:09 | BTC/USDT + SOL/USDT | 0.0107 | 0.0016 | 0.4925 | 106.7 | 551.1 | 0.9935 | 0.9987 | -0.0895 | 11 | 0.4545 | -0.0672 | -0.7396 |
| 1 | 2022-05-30 16:09 | 2022-06-03 03:30 | SOL/USDT | 0.0411 | 0.0194 | 0.0992 | 186.8 | 227.4 | 0.9963 | 0.997 | 0.0308 | 7 | 0.4286 | -0.2445 | -1.7113 |
| 2 | 2022-06-03 03:30 | 2022-06-06 14:51 | ADA/USDT + SOL/USDT | 0.0483 | 0.0843 | 0.0353 | 195.0 | 301.5 | 0.9965 | 0.9977 | -0.7214 | 6 | 0.1667 | -0.3034 | -1.8206 |

### Segment 2: 30,000 bars, 2025-10-17 .. 2025-11-07

| chunk | train_start | oos_start | hedge | p_eg_train | adf_p_train | adf_p_oos | half_life_train | half_life_oos | phi_train | phi_oos | z_mean_oos | trades | tp_rate | mean_r | sum_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2025-10-17 04:48 | 2025-10-20 16:09 | ADA/USDT + SOL/USDT | 0.0344 | 0.0119 | 0.1741 | 76.062 | 259.2 | 0.9909 | 0.9973 | -0.5661 | 8 | 0.375 | -0.0448 | -0.3587 |
| 1 | 2025-10-20 16:09 | 2025-10-24 03:30 | BNB/USDT + XRP/USDT | 0.000269 | 1.34e-05 | 0.6158 | 74.8351 | 1,245.8 | 0.9908 | 0.9994 | -0.1743 | 11 | 0.7273 | 0.1662 | 1.8282 |
| 2 | 2025-10-24 03:30 | 2025-10-27 14:51 | BNB/USDT + SOL/USDT | 0.0129 | 0.0029 | 0.5167 | 102.1 | 373.3 | 0.9932 | 0.9981 | 0.1866 | 9 | 0.5556 | 0.2072 | 1.8645 |
| 3 | 2025-10-27 14:51 | 2025-10-31 02:12 | BNB/USDT + BTC/USDT | 0.0317 | 0.0029 | 0.2544 | 107.6 | 208.2 | 0.9936 | 0.9967 | 0.2124 | 6 | 0.5 | -0.0464 | -0.2782 |
| 4 | 2025-10-31 02:12 | 2025-11-03 13:33 | BNB/USDT + BTC/USDT | 0.0217 | 0.0029 | 0.3065 | 87.8072 | 469.3 | 0.9921 | 0.9985 | -0.4747 | 9 | 0.5556 | -0.0179 | -0.1615 |

### Segment 3: 25,000 bars, 2026-01-10 .. 2026-01-27

| chunk | train_start | oos_start | hedge | p_eg_train | adf_p_train | adf_p_oos | half_life_train | half_life_oos | phi_train | phi_oos | z_mean_oos | trades | tp_rate | mean_r | sum_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-01-10 04:48 | 2026-01-13 16:09 | ADA/USDT + XRP/USDT | 0.000568 | 3.89e-05 | 0.015 | 56.2448 | 382.0 | 0.9878 | 0.9982 | 0.6776 | 5 | 0.6 | 0.0851 | 0.4253 |
| 1 | 2026-01-13 16:09 | 2026-01-17 03:30 | ADA/USDT + BTC/USDT + SOL/USDT | 0.049 | 0.0268 | 0.9949 | 210.2 | n/a | 0.9967 | 1.0005 | 0.0423 | 6 | 0.6667 | 0.3535 | 2.121 |
| 2 | 2026-01-17 03:30 | 2026-01-20 14:51 | ADA/USDT + BNB/USDT + SOL/USDT + XRP/USDT | 0.0489 | 0.0027 | 0.1594 | 75.1795 | 331.4 | 0.9908 | 0.9979 | -0.3632 | 4 | 0.5 | 0.0969 | 0.3878 |
| 3 | 2026-01-20 14:51 | 2026-01-24 02:12 | XRP/USDT | 0.0359 | 0.0075 | 0.3181 | 204.9 | 390.5 | 0.9966 | 0.9982 | 0.1446 | 4 | 0.25 | -0.5527 | -2.2106 |

### Segment 4: 10,000 bars, 2020-10-12 .. 2020-10-19

| chunk | train_start | oos_start | hedge | p_eg_train | adf_p_train | adf_p_oos | half_life_train | half_life_oos | phi_train | phi_oos | z_mean_oos | trades | tp_rate | mean_r | sum_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2020-10-12 04:48 | 2020-10-15 16:09 | ADA/USDT + BTC/USDT + SOL/USDT | 0.0094 | 0.0011 | 0.0044 | 49.1577 | 85.3577 | 0.986 | 0.9919 | -0.3736 | 3 | 0.3333 | -0.3438 | -1.0314 |

### Segment 5: 15,000 bars, 2023-08-26 .. 2023-09-05

| chunk | train_start | oos_start | hedge | p_eg_train | adf_p_train | adf_p_oos | half_life_train | half_life_oos | phi_train | phi_oos | z_mean_oos | trades | tp_rate | mean_r | sum_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-08-26 04:48 | 2023-08-29 16:09 | ADA/USDT + BNB/USDT | 0.0432 | 0.0051 | 0.8429 | 25.8173 | 949.4 | 0.9735 | 0.9993 | -0.4907 | 8 | 0.375 | -0.0151 | -0.1207 |
| 1 | 2023-08-29 16:09 | 2023-09-02 03:30 | BNB/USDT + BTC/USDT | 0.0249 | 0.0147 | 0.2776 | 103.0 | 253.2 | 0.9933 | 0.9973 | -0.2541 | 3 | 0.3333 | -0.2749 | -0.8246 |

## Full-width chunk statistics

See the companion CSV for every column (train/OOS window dates, ADF stats, betas, z drift, SL/time rates, net PnL, tree AUC).

## Pooled Monte Carlo summary

```
MONTE CARLO PURGED WALK-FORWARD [rolling z / hi-lo barriers]
bar=1min | data budget 100,000 bars | chunk 5,000 | min segment 10,000 | seed 7
z-score: rolling (1,440 bars) | barriers: hilo (comove) | entry_z 2 | barrier_size 2 | rr 0.5
strategy direction: INVERTED (strategy_side=2: strategy long -> we short)
Segments: 5 | OOS chunks measured: 15 | unique time coverage: 100%
  #1   20,000 bars  [2022-05-27 .. 2022-06-10]  3 chunk(s)
  #2   30,000 bars  [2025-10-17 .. 2025-11-07]  5 chunk(s)
  #3   25,000 bars  [2026-01-10 .. 2026-01-27]  4 chunk(s)
  #4   10,000 bars  [2020-10-12 .. 2020-10-19]  1 chunk(s)
  #5   15,000 bars  [2023-08-26 .. 2023-09-05]  2 chunk(s)
Trades: 100 | exits: tp 48.0% | sl 21.0% | time 28.0% | window_end 3.0% | breakeven TP rate 66.7%
Mean R/trade: -0.0263 | sd 0.608 | trade-level t-stat -0.43 | accumulated -2.63R
Chunk bootstrap (2,000 paths): E[R] -0.0284  P05/P95 [-0.1169, +0.0558]  CVaR05/95 [-0.1416, +0.0751]
95% CI of mean R: [-0.1335, +0.0693] | P(E[R] > 0): 29.5% | chunks with positive expectancy: 5/15
Trades needed for this mean to be significant at 95% (trade-level): ~2,051 | have 100 -> NOT enough
Chunks are the resampling unit (trades inside a chunk are correlated); overlapping segments share bars, so coverage < 100% means fewer independent samples.
```
