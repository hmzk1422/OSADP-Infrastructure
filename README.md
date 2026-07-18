# Intraday-Momentum-Composite
Systematic framework for capturing short-term alpha in high-volatility crypto-asset markets. By constructing a synthetic, equal-weighted intraday index, we neutralize idiosyncratic risk while isolating systemic momentum. We hypothesize that the persistence of price action is functionally linked to the second derivative of price (MA Acceleration). By identifying regimes where MA acceleration remains $\text{acc} > 0$ (long) or $\text{acc} < 0$ (short), we optimize entry points for RSI-based mean reversion (snapback) strategies, effectively simulating the structural drift of traditional equity indices within a high-frequency digital asset context.

# Focus
1. Investigating how RSI snapback speed has a very strong correlation with MA acceleration.
2. Approaches to solving the 'momnentum decay' factor relating to MA acceleration
3. Creathing synthetic asset - Intraday based S&P500 model where instead we are taking advantage of high momentum crypto
4. Utilizing ML and PDE to solve for momentum decay issue