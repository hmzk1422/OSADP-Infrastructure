# Volatility Decay Capture (VDCV)
```
OSADP version : v1.0.0
Author        : Himaska Lin
```
## Abstract
Strategy based on the volatility decay edge of TQQQ+SQQQ done in intraday hedging. Use machine learning, specifically short term orderflow prediction and risk management to quickly eliminate risk of persistent moves.

## Why this edge might work?
- Orderflow might not be able to predict price, but it could probably be leading signal for volatility and persistent moves in the short term. 
- Volatility decay is a proven phenomenon with track record on TQQQ+SQQQ. 
- Allows for funding fee profiting, giving positive expectancy.
- Scalable to all timeframes. Provided low timeframe is too competitive, we can always shift to higher timeframes to avoid execution problems.
- Edge is highly defensive since it uses math rather than competition. It is hard for algorithms, bots, to reverse engineer the edge (esp if its managed as large portfolio)

## Why this edge might fail?
- Persistent moves. Same problem as pure arbitrage where tail risk destroys the portfolio massively.
- Rebalancing difficulties and execution-heavy algorithm. 

## Solution
- Persistent moves: Major, hedging problem.
    - Orderflow short term correlation with future might be able to help us avoid persistent moves in time such that drawdown risk is small.
    - Metalabelling, create a trading system to take advantage of volatility decay's PnL curve. For example, it is known if d^2 price > 0, rsi(2) snapback is fast. Risk management optimization
    - Underlying asset. Statarb spreads might be more stationary and less persistent such that this strategy thrives better. some cryptos for example, large cap cryptos might be more mean reverting vs small cap cryptos.
    - Screening/regime detection. Running regime detection on underlying asset to see if its safe for this strategy or not to run.
- Rebalancing difficulties: Relies on good practices, fallbacks, defensive algos, large timeframes to increase tolerance.

## Methodology
- Parametric monte carlo simulation w uncorrelated synthetic data level 2
- Purged chronological walk forwards w live financial data (Differing groups)