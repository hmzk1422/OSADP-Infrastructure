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

## Shared Stress Controls

Synth1, Synth2, and Synth3 accept the same optional controls. They change one
dimension of the existing generator while leaving its feature/state design
alone; they do not make an easier generator imitate a real market.

```python
simulate_market_stochastic(
    seed=12,
    volatility_scale=1.5,
    directional_bias=-0.0004,
    regime_persistence=0.985,
)
```

- `volatility_scale` multiplies per-bar volatility and must be positive.
- `directional_bias` adds a signed log-return drift per generated bar. Negative
  values stress bearish paths; positive values stress bullish paths.
- `regime_persistence` sets the base probability of staying in the current
  regime. `None` retains the generator's original transition matrix. Synth3
  still adjusts transitions in response to stress.

Synth1's defaults remain its clean benchmark defaults; Synth2/3 retain their
own existing complexities. For VDC Monte Carlo,
`directional_bias_per_hour_range` samples a signed log-return drift per clock
hour (default `(-0.05, 0.05)`) and converts it to per-bar units using
`bar_seconds`. This keeps the control comparable when bar frequency changes.
The legacy `directional_bias_range` remains available in per-bar units and
overrides the per-hour range when provided. Set both ranges to `None` to use
one fixed per-bar `directional_bias` from `generator_kwargs`.
`volatility_scale_range` and `regime_persistence_range` can likewise sample
those controls per path; fixed `volatility_scale` and `regime_persistence`
values remain available.

In `vdc_monte_carlo`, `n_paths` counts independent scenarios. Use
`n_assets_per_scenario` to generate multiple independent assets in each
scenario. `vdc_portfolio` diversifies assets within a scenario and reports
scenario outcomes separately; it does not pool mutually exclusive Monte Carlo
scenarios into one fictitious portfolio. Portfolio NAV is compounded.

For a generated synthetic asset universe, use
`generate_uncorrelated_synthetic_assets` in `uncorrelated_filter.py`. It
selects paths only when their aligned simple-return Pearson correlations stay
within the requested absolute threshold. This is a finite-sample correlation
filter, not a guarantee of statistical independence. The accepted `asset_data`
mapping can be passed to `vdc_synthetic_asset_test` and then `vdc_portfolio`.
