# Volatility Decay Capture (VDC)
```
OSADP version : v1.0.0
Author        : Himaska Lin
```
## Abstract
Interpolation of OFMR. In daily timeframes, leveraged ETFS (such as obserrved in TQQQ and SQQQ) exhibit a negative bias known as volatility decay.
- Volatility decay increases as rebalancing occurs in choppy regimes
- Volatility decay decreases as rebalancing occurs in sharp spikes.

## Why this edge might work
1. Mathematically validated. It doesn't participate in competition (e.g. information coefficient, latency competition, etc) and is mathematically deterministic, making for a durable edge.
2. High possible sortino. Economically rational edge.
3. Execution wise very stable, doesn't require complex, machine learning systems. Fixed rules could work for this system. Complexity comes in a different form
4. Using portfolio optimization theory, provided that there are many uncorrelated assets, variance shrinks and closes to the mean.
5. Very liquidity friendly. 

## Why this edge might not work
1. Sensitive execution, requires rebalancing. Commission matters significantly. Or maybe not so much.
    - Let A and A' be two inverses of each other following volatility decay.
        Interday volatility
        - st : A = 1.00, A' = 1.00 --> Net asset = 0
        - -2%: A = 0.98, A' = 1.02 --> Net asset = -0.04A
        - -1%: A = 0.9702, A' = 1.0302 --> Net asset = -0.06A
        - +3%: A = 0.999306, A' = 1.000 --> Net asset = -8.88 * 10^-4
        Intraday volatility
        - st : A = 1.00, A' = 1.00 --> Net asset = 0
        - -0.05%: A = 0.9995, A' = 1.0005 --> Net asset = -0.001A
        - -0.02%: A = 0.9993001, A' = 1.0007001 --> Net asset = -0.0014
        - +0.07%: A = 0.99999961007, A' = 0.99999960993 --> Net asset = -1.4 * 10^-10
    - (-8.88*10^-4) / 2 = 0.04%
    - Total hedging = ((0.04 + 0.02) /2) * 0.04% = 0.0012%
    - 7*10^-9%
    - Total hedging = ((0.001 + 0.0014) / 2) * 0.04% = 4.8*10^-7%
    - cost-wise, its not as expensive as we think. However, 
2. tail risk, possibly failing portfolio optimization theory as volatility decay crashes during an event of massive moves. Even if they are all uncorrelated, mean during this regime might be negative.

## Mitigation Plans
1. Reducing tail risk:
    - Volatility modelling. E.g., using garch, or metrics like VPIN, spoofing, etc to detect danger zones.
    - Statarb Pairs. If statarb pairs generally exhibit more mean-reverting properties, tail risk is reduced. It also reduces correlation.
2. Maintaining positive expectancy:
    - Model for funding fee as additional edge. Use the statarb method to hedge (Let each pair's edge = volatility_decay - funding fee/t). This also solves the previous funding fee problem
        - Spot + Futures hedging introduces margin problem + one direction only
        - Futures - futures introduces cross-exchange arbitrage problem.
    - Prioritize execution algorithm and reduce fees.

# Design Rational
1. Statarb system: Three very important metrics which are cross-correlation, net funding fee and edge expectancy.
    - Use BSH (binary search heap to find best pair)