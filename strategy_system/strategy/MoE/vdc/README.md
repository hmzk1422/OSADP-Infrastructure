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
1. Sensitive execution, requires rebalancing. Commission matters significantly.
2. tail risk, possibly failing portfolio optimization theory as volatility decay crashes during an event of massive moves. Even if they are all uncorrelated, mean during this regime might be negative.

## Mitigation Plans
1. Reducing tail risk:
    - Volatility modelling. E.g., using garch, or metrics like VPIN, spoofing, etc to detect danger zones.
    - Statarb Pairs. If statarb pairs generally exhibit more mean-reverting properties, tail risk is reduced. It also reduces correlation.
2. Maintaining positive expectancy:
    - Model for funding fee as additional edge. Use the statarb method to hedge (Let each pair's edge = volatility_decay - funding fee/t)
    - Prioritize execution algorithm and reduce fees.

