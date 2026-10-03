# Synthetic Data
OSADP-version :
Version       : v1.0.0 
Author        : Himaska Lin

# Introduction
There are 3 levels. As synth levels progresses, they become harder and harder for strategies to beat.
- Synth1: It passed the EDA test (Hypothesis testing only).
- Synth2: It passed basic research phase (Backtesting only + optionally, Monte Carlo).
- Synth3: It passed production test and is ready to run live paper (Purged chronological walk forwards + Monte Carlo (Parametric AND Trades-based/resampled)).

| Component             | Synth1       | Synth2                   | Synth3                                            |
| --------------------- | ------------ | ------------------------ | ------------------------------------------------- |
| Hidden state          | Clean HMM    | Asymmetric HMM           | **State transitions respond to market stress**    |
| Regime persistence    | Fixed        | Fixed                    | **Variable / state-duration dependent**           |
| Volatility            | Regime fixed | Stochastic around regime | **Persistent stochastic volatility / clustering** |
| Volatility shocks     | Gaussian     | Student-t return shocks  | **Volatility jumps + return jumps**               |
| Price → features      | No           | No                       | **Yes, explicitly**                               |
| Feature loadings      | Stable       | Slow drift               | **Drift + occasional structural breaks**          |
| Feature information   | Stable       | Dropout                  | **Decay, dropout, delay, contamination**          |
| Momentum              | Controlled   | Controlled               | **Time-varying and state-dependent**              |
| Mean reversion        | Controlled   | Controlled               | **Variable strength + nonlinear**                 |
| Liquidity             | None         | None                     | **Latent liquidity state**                        |
| Volume                | None         | None                     | **State/volatility dependent**                    |
| Spread                | None         | None                     | **Stochastic spread**                             |
| Jumps                 | None         | Heavy-tailed shocks      | **Explicit jump process**                         |
| Volatility clustering | Weak         | Moderate                 | **Strong**                                        |
| Intraday seasonality  | None         | None                     | **Optional U-shaped activity**                    |
| Market impact         | None         | None                     | **Weak endogenous impact**                        |
| Observability         | Clean        | Noisy                    | **Partially obscured**                            |

# How to Use?
It is recommended to complement synth data with the following.
- Monte Carlo Simulations: Randomizing all possible price sequences under HMM-GBM assumption, which will return range of profitability. Though the standard mode uses uncorrelated sequences only, it is recommended to compute for all sequences to determine whether similar correlations = similar performance.
- Purged Chronological Walk Forwards: Simulation of live markets, whether or not strategy can survive provided the algorithm runs continuously.
- 