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

# Testing with OOD and MoE system (1 Oct 2026 - 14 Oct 2026)
After 2 weeks of testing OOD system, we figured out that similarity regimes don't necessarily equate to edge.
- Running a naive strategy inside of a huge sample of brute-forced regimes returning high similarity scores (temporally and geometrically) along with TDMI failed to yield any meaningful result with edge expectancy.
    - Possible reasons of failure includes: Forced overfitting. Using GPLearn for regime separation and running the model might've meant the similarity meant nothing where the point of similarity being contrasted here is meaningless. For example, a strategy that profits from volatility means nothing if the current regime is 99% similar with high volatility regime in terms of price direction, orderflow, etc but not volatility prediction. In simple words, OOD only identified similarity in variables that yield no economical meaning with GPLearn.
    - Naive trading model. If we focus on simplicity and turned OOD into a pure binary classifier of unprofitable regime or profitable regime, paired with OFMR, the strategy might perform better in contrast to a true MoE, overfitted system.
- Overall, this segment still requires more research. Challenge these assumptions by:
    - Path-seeking algorithm with GPLearn to extract economically meaningful variables. How: Variables are thrown into pools of hash maps representing buckets of similarity. GPLearn discards variable that are strongly correlated with a bucket. From all uncorrelated features, preventing it from exploding combinatorically.
    - Using purged-WF and monte carlo after extracting all uncorrelated features to yield meaningful features. Extract the model with features that performed best by purged-wf, monte carlo and synth data generation. Test for stability.

# Implementation Finished, Figured out problem (10 Oct 2026 - 18 Oct 2026)
Trained on much higher quality data data, fixed parity between rust and python implementation. OOS sample now works more similarly to backtested result + major improvements in edge
- including more bid ask levels improved mutual information, edge expectancy increased from average 0.10-0.20rr to 0.30-0.40rr. 
- Proven that orderflow correlataes with short term price very well. Majoritively, orderflow feataure returned higher average MI over other features.

## Observed result
1. Purged WF across data returned very strong min cVar which suggests within chunked samples, the strategy consistently performed well in the market.
2. Live OOS trading with strategy. We tested edge decay and feature incompatibility by swapping models in between tickers. It showed a strong parity where for example, a model trained in BTC trading in BTC returned a 33.33% WR whereas the same model in 9 other assets averaged a 16% WR, which hasn't been observed in the previous version.
    - This suggests the model is very sensitive to changes in data temporality and geometric distance. Having an OOD might be useful here.
    - Next problem: Execution algorithm + expectancy.
3. In terms of memory and computational efficiency, each instance of OFMR takes over an average 1.2%-1.6% CPU usage and memory of 15MB vs previously which bottlenecked over 100MB in the older version and suffered lots of data staleness/slowdown problem due to DB overwriting.
4. Current model strongly underperforms the market if the market is trending downwards since its a long-only strategy.