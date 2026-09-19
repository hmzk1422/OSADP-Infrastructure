# Trade Journal
Strategy performance notes and trade-by-trade annotations.

# Portfolio Optimization Management (28 Aug 2026 - 1 Sept 2026)
The model returned high edge expectancy across G1 crypto (0.22 - 0.74rr).
| Asset     |    Trades |     Total R | Avg R / Trade | Win Rate | Max DD (R) | Profit Factor | Status   |
| --------- | --------: | ----------: | ------------: | -------: | ---------: | ------------: | -------- |
| BTC       |       512 |     +159.00 |         +0.31 |        — |      39.00 |             — | LIVE     |
| ETH       |       380 |     +280.00 |         +0.74 |        — |       8.87 |             — | LIVE     |
| SOL       |       601 |     +133.39 |         +0.22 |        — |      34.00 |             — | LIVE     |
| **Total** | **1,493** | **+572.39** |     **+0.38** |        — |          — |             — | **LIVE** |

## Observations

Portfolio risk is currently the main limitation. The strategy has high return variance and a relatively low Sortino, meaning standalone deployment can experience substantial drawdown despite positive expectancy. Diversification across weakly correlated assets or strategies may improve portfolio-level risk-adjusted return. The strongest solution may ultimately be finding a naturally higher-Sortino strategy while retaining this strategy for its return contribution.

TWAP appears beneficial for execution. It allows exposure to scale with changing strategy confidence, reducing the risk of immediate full-size entry during noisy regimes. However, it can also under-size winning trades or continue adding to persistent losses when confidence remains high. Whether the benefit outweighs this cost requires direct TWAP vs. immediate-execution testing.

Asset performance differs materially. ETH significantly outperformed BTC and SOL while also experiencing substantially lower drawdown. The cause is currently unknown and may relate to asset-specific volatility, liquidity, order flow, or regime characteristics. Further investigation is required.

## Next Objective

[X] Regime detection + metalabeller implementation on live execution (Simultaneous execution of current model vs metalabeller to see performance difference).

[X] Implementing Modern Portfolio Theory. Diversify through assets or strategies to reduce variance but rigorous testing must show positive expectancy.

# OOD Identifer as Regime Detection (9 Sept 2026 - 13 Sept 2026)
Possibility of edge decay (unproven, still running). Strategy suffers from terrible MDD (Average 20RR) and trade expectancy (Average +0.091RR). Research purged wf-validation and monte-carlo returned positive expectancy yet edge decayed very quickly. Past performance doesn't guarantee future results.
- MoE system with underlying strategy using VQ-VAE to act as OOD filter. Still utilizes OFMR with each expert trained specifically for certain latent space. Use this as a method of trading only in markets we know.
- Optimize strategy with objective of both sortino and trade expectancy in mind. Use improved data source, restart OFMR research over strategy. Not for solving edge, but rather solving for managing non-stationarity problem.