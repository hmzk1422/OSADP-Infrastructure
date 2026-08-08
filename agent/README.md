# What is Agent?
Side project under intraday-momentum-composite where we create deep learning RL agent or LLM that takes up strategy and is trained under the rules a quant trader should have such as game theory, probability, risk management, etc.

## Initialization Test
Let there be n price paths simulated using Geometric Brownian Motion (GBM) with drift Mu and Volatility to simulate market regime changes.
$$\text{Return}_t = \mu \cdot \Delta t + \sigma \cdot \sqrt{\Delta t} \cdot \epsilon_t \quad (\text{where } \epsilon_t \sim \mathcal{N}(0, 1))$$
Objective for Deep RL Agent: Maximize Sortino Ratio (higher is better = more return per unit of downside risk).
Secondary Objective: Maximize Trade Expectancy (average edge per period, e.g., portfolio 1.0 -> 1.4 in 20 periods = 0.02 edge/period).

However drift and MU are not constants. As such, we will utilize markov-switching component to simulate states of the market (volatility, drift, how long it last, etc)
Optionally, introduce poisson jump distribution to introduce the factor of unexpected spikes in the market and how the AI could combat against that.

## After Test
Add correlation/covariance matrix test between two assets. 