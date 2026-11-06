# Multi-Asset and Feature-Augmented Statistical Arbitrage with Static Beta Calibration and Structural Drift Diagnostics

## Abstract

Traditional statistical arbitrage strategies rely on the assumption that a stable, mean-reverting relationship exists between two or more financial assets. Commonly implemented through cointegration analysis, these strategies construct a synthetic spread using estimated hedge ratios and exploit deviations from their historical equilibrium.

However, conventional approaches face several practical limitations. High correlation does not necessarily imply cointegration, statistical stationarity may be temporary, and structural changes in the underlying economic relationship can invalidate previously estimated hedge ratios. In particular, when the true hedge ratio varies over time, a spread constructed using a fixed coefficient can accumulate tracking errors that are not captured by conventional spread calculations.

This research introduces a multi-asset, feature-augmented statistical arbitrage framework that combines pre-calibrated static hedge ratios, weighted linear combinations, structural explanatory variables, and out-of-sample stress testing. Rather than continuously recalibrating hedge ratios through rolling ordinary least squares (OLS) or Kalman filtering, the implementation maintains fixed coefficients derived from a historical training baseline. This design prioritizes computational efficiency, parameter stability, and the avoidance of noisy coefficient updates during execution.

The framework is extended to incorporate macroeconomic and physical friction variables in traditional commodity relationships, as well as market microstructure and protocol-specific characteristics in cryptocurrency markets. Its implementation consists of a static weighted-spread strategy (`strategy.py`) and a dedicated validation framework (`mmc_stress_test.py`).

Initial experiments reveal important limitations associated with fixed-window calibration, false stationarity, sparse trading frequency, and out-of-sample generalization. These findings motivate a more rigorous distinction between statistical stationarity, structural stability, and economically justified mean reversion.

The objective is not to assume that a fixed hedge ratio guarantees stationarity, but to evaluate whether a computationally efficient, pre-calibrated statistical relationship can remain sufficiently stable under changing market conditions.

---

## 1. Introduction

Statistical arbitrage has traditionally been applied across medium-to-higher timeframes, relying on the identification of statistically related financial instruments whose relative prices exhibit mean-reverting behavior.

A foundational approach is cointegration analysis, particularly the Engle-Granger methodology. Unlike ordinary correlation analysis, cointegration examines whether a linear combination of individually non-stationary time series produces a stationary residual process.

Consider two non-stationary asset price series, \(A_t\) and \(B_t\). A conventional statistical arbitrage framework constructs:

$$
S_t = A_t - \beta B_t
$$

where:

- \(S_t\) represents the synthetic spread.
- \(A_t\) represents the target asset.
- \(B_t\) represents the reference asset.
- \(\beta\) represents the estimated hedge ratio.

If the individual price series are integrated of order one, \(I(1)\), but the resulting spread is stationary, \(I(0)\), the two assets are cointegrated under the specified linear relationship.

However, conventional statistical arbitrage models suffer from several real-world vulnerabilities.

### 1.1 The Absence of Guaranteed Stationarity

High linear correlation between two assets does not establish cointegration.

Two assets can exhibit substantial correlation because they respond to common market factors, without possessing a stable long-term equilibrium relationship.

Furthermore, a successful Augmented Dickey-Fuller (ADF) test does not establish that a relationship is permanently stationary. It provides statistical evidence against a unit-root null hypothesis under the assumptions of the test and the specified sample.

Consequently, relationships identified as stationary over historical training windows may represent temporary market regimes rather than persistent structural relationships.

This creates a fundamental distinction between:

- **Historical statistical stationarity:** A spread exhibits mean-reverting behavior within a particular sample.
- **Structural stability:** The underlying economic relationship remains sufficiently consistent across different market regimes.
- **Out-of-sample mean reversion:** The relationship continues to exhibit statistically and economically meaningful reversion after calibration.

A robust statistical arbitrage framework must therefore evaluate all three dimensions rather than relying exclusively on historical ADF results.

### 1.2 Beta Drift and Hedge-Ratio Instability

The classical statistical arbitrage spread is:

$$
S_t = A_t - \beta_t B_t
$$

In conventional implementations, the hedge ratio is frequently estimated over a historical period and subsequently treated as constant.

However, the true economic relationship between two assets may change because of structural, macroeconomic, and market-specific factors.

To understand the resulting error, consider the differential of the spread.

Applying the product rule:

$$
dS = dA - \beta\,dB - B\,d\beta - d[\beta,B]
$$

where \(d[\beta,B]\) represents the quadratic covariation term under stochastic calculus.

For a differentiable, finite-variation hedge ratio, the quadratic covariation term vanishes, giving:

$$
dS = dA - \beta\,dB - B\,d\beta
$$

The additional component,

$$
-B\,d\beta
$$

represents the contribution of hedge-ratio variation to spread dynamics.

This term is absent when the model assumes a fixed hedge ratio.

Importantly, fixing \(\beta\) does not eliminate actual beta drift. It eliminates coefficient variation within the implemented model. If the underlying economic relationship continues to change, the difference between the true and calibrated hedge ratios becomes an unmodeled source of tracking error.

This distinction motivates the use of static beta calibration combined with explicit out-of-sample structural stability diagnostics.

---

## 2. Methodology: Multi-Asset and Feature-Augmented Spread Construction

### 2.1 Macroeconomic Spread Dynamics: Crude Oil versus Brent

Consider a traditional commodity statistical arbitrage relationship between crude oil and Brent crude.

The conventional spread is:

$$
S_t = P_{\text{Crude},t} - \beta P_{\text{Brent},t}
$$

Crude oil and Brent are closely related benchmarks, reflecting overlapping global supply-demand conditions. Their relative valuations can therefore exhibit long-term statistical relationships.

However, these relationships are not guaranteed to remain stationary.

A significant source of structural variation arises from the physical and logistical costs associated with transporting, storing, and insuring crude oil.

These include:

**Freight Rates**

Transportation expenses associated with redirecting tanker supply between regional delivery locations.

**Insurance**

Maritime insurance premiums and risk-related costs associated with transporting physical crude oil.

**Storage Costs**

Physical warehousing and floating-storage expenditures required to maintain or relocate crude inventories.

These components fluctuate according to geopolitical developments, changing supply-demand dynamics, and logistical constraints.

Consequently, the relative valuation of crude oil and Brent can experience structural deviations that are not adequately explained by a simple two-asset spread.

The proposed framework therefore investigates whether incorporating these explanatory variables improves the stability of the synthetic spread.

### 2.2 Feature-Augmented Multi-Factor Spread

The extended model takes the form:

$$
\begin{aligned}
S_t ={}& P_{\text{Crude},t}
-\beta_1P_{\text{Brent},t}\\
&-\beta_2P_{\text{DXY},t}\\
&-\beta_3C_{\text{Insurance},t}\\
&-\beta_4C_{\text{Storage},t}\\
&-\beta_5C_{\text{Freight},t}
\end{aligned}
$$

where:

- \(P_{\text{Crude}}\) represents the crude oil price.
- \(P_{\text{Brent}}\) represents the Brent benchmark price.
- \(P_{\text{DXY}}\) represents the US Dollar Index.
- \(C_{\text{Insurance}}\) represents maritime insurance costs.
- \(C_{\text{Storage}}\) represents storage expenditures.
- \(C_{\text{Freight}}\) represents transportation costs.
- \(\beta_1,\ldots,\beta_5\) represent calibrated coefficients.

The purpose of this extension is to investigate whether incorporating macroeconomic and physical friction variables provides a more representative approximation of the underlying economic relationship.

A critical econometric requirement is that the variables must be expressed in economically meaningful units and aligned to consistent observation frequencies.

Furthermore, the resulting linear combination must be independently evaluated for stationarity. Including additional explanatory variables does not automatically establish cointegration or mean reversion.

For example, if the price variables are \(I(1)\), a stationary residual requires an appropriate linear combination that cancels their non-stationary components. Stationary explanatory variables can help explain short-term spread fluctuations, but their inclusion alone does not guarantee that the resulting spread is stationary.

The coefficients must therefore be calibrated and validated as a complete vector rather than selected solely because individual variables exhibit favorable historical correlations.

### 2.3 Structural Stability and Beta-Drift Diagnostics

A central objective of the framework is to reduce sensitivity to unstable hedge-ratio estimation.

The conventional approach of continuously recalculating beta through rolling OLS or Kalman filtering can introduce additional model complexity and coefficient variability.

The proposed implementation instead evaluates whether a historically calibrated coefficient vector remains sufficiently robust across unseen market regimes.

For a time-varying reference relationship:

$$
S_t^* = A_t - \beta_t^* B_t
$$

and an implemented static-beta relationship:

$$
\widehat{S}_t = A_t - \widehat{\beta}B_t
$$

their difference is:

$$
\widehat{S}_t-S_t^*
=
(\beta_t^*-\widehat{\beta})B_t
$$

This represents the level error resulting from using a fixed coefficient instead of the hypothetical time-varying coefficient.

A useful diagnostic is the contribution of hedge-ratio instability relative to total spread variation.

For example:

$$
R_{\beta}
=
\frac{
\operatorname{Var}\left(
B_t\Delta\beta_t^*
\right)
}{
\operatorname{Var}(\Delta S_t^*)
}
$$

where \(\beta_t^*\) represents an independently estimated diagnostic reference rather than an execution parameter.

A lower value indicates that beta variation contributes less to the variance of the reference spread under the chosen diagnostic specification.

This ratio is a diagnostic, not proof of stationarity. It should not be minimized without considering covariance effects, estimation uncertainty, and out-of-sample performance.

The intended objective is to identify relationships for which a fixed calibrated hedge ratio remains sufficiently representative without requiring continuous coefficient recalibration.

---

## 3. Extension to Intraday Cryptocurrency Statistical Arbitrage

Although traditional statistical arbitrage is commonly implemented across medium-to-higher timeframes, highly liquid cryptocurrency markets provide an alternative environment for investigating intraday relative-value opportunities.

The framework extends beyond conventional price-based cointegration by incorporating granular market microstructure information and protocol-specific structural characteristics.

### 3.1 Market Microstructure Features

The proposed feature set includes:

**Open Interest**

Measures outstanding derivative contract exposure and provides information about the accumulation or reduction of leveraged positioning.

**Forced Liquidations**

Captures involuntary position closures that can generate abnormal short-term buying or selling pressure.

**Order Book Imbalances**

Measures asymmetry between available bid-side and ask-side liquidity, providing information about short-term supply-demand pressure.

**Funding Fees**

Captures the periodic payments exchanged between perpetual futures participants and provides information about derivative-market positioning and relative pricing.

These variables can help characterize short-term deviations between related cryptocurrency instruments.

However, they are not necessarily integrated price series. Their statistical treatment should reflect their individual properties, including whether they are stationary, non-stationary, bounded, or regime-dependent.

For example, order book imbalance and funding rates may be more appropriately incorporated as explanatory features than treated as additional non-stationary price components.

The framework therefore distinguishes between variables used to construct the long-term equilibrium relationship and variables used to explain or adjust short-term deviations.

### 3.2 Protocol-Specific Structural Factors

Beyond market microstructure, cryptocurrency assets possess underlying protocol characteristics that can influence their structural risk profiles and long-term valuations.

Examples include:

- Native mining algorithms, such as SHA-256 for Bitcoin and Scrypt for Litecoin.
- Cumulative Proof-of-Work (PoW) security.
- Differences in the computational resources and economic costs required to attack individual networks.
- Protocol-specific security and network characteristics.

These differences are relevant because distinct blockchain architectures can exhibit different security assumptions, attack costs, and long-term risk profiles.

For example, SHA-256 and Scrypt represent different mining algorithms, and their respective networks cannot be assumed to possess equivalent security characteristics.

Such differences may contribute to persistent valuation premiums or discounts between cryptocurrency assets, independently of transient order-flow imbalances.

Nevertheless, protocol characteristics should be treated as candidate structural explanatory variables rather than established price determinants. Their economic significance must be demonstrated empirically, particularly because some network-level variables change slowly or may be highly correlated with other structural factors.

The expanded framework therefore investigates both:

1. Short-term market microstructure effects.
2. Longer-term protocol-specific structural differences.

This creates a broader feature space for evaluating relative-value relationships between cryptocurrency assets.

---

## 4. Implementation Architecture

The framework is implemented through two primary components:

1. `strategy.py`: Static-beta spread construction and signal generation.
2. `mmc_stress_test.py`: Out-of-sample validation and structural stress testing.

The central architectural decision is to separate coefficient calibration from execution.

Rather than recalculating regression coefficients continuously, the strategy uses a pre-calibrated coefficient vector obtained from historical training data.

### 4.1 `strategy.py`: Static Beta and Weighted Sums

The strategy relies on a fixed hedge-ratio vector derived from a pre-computed cointegration analysis over a historical training baseline.

Instead of implementing rolling OLS or Kalman filtering within the execution loop, the script calculates the synthetic spread through a weighted linear combination of the selected variables.

The general formulation is:

$$
S_t=P_{\text{target},t}
-\sum_{i=1}^{n}\beta_iP_{i,t}
$$

where:

- \(P_{\text{target},t}\) represents the primary asset.
- \(P_{i,t}\) represents the reference assets or appropriately defined explanatory variables.
- \(\beta_i\) represents the corresponding pre-calibrated coefficient.
- \(n\) represents the number of reference variables.

For a fixed coefficient vector:

$$
\boldsymbol{\beta}
=
\begin{bmatrix}
\beta_1 & \beta_2 & \cdots & \beta_n
\end{bmatrix}^{T}
$$

the spread calculation becomes a straightforward weighted-sum operation.

**Implementation mechanics**

- The script initializes fixed hedge ratios derived from historical cointegration analysis.
- The coefficients remain unchanged during the execution period.
- The spread is calculated using a weighted linear combination of the primary asset and its reference variables.
- The execution loop avoids continuous regression recalculation.

This architecture reduces computational overhead and avoids introducing noisy coefficient updates into high-frequency execution.

It also ensures that the strategy evaluates deviations relative to a consistent historical calibration rather than continuously adapting its equilibrium estimate to recent observations.

An important qualification is that fixed coefficients do not guarantee that the underlying relationship remains stationary. The strategy deliberately accepts this model risk in exchange for computational simplicity and parameter consistency.

**Signal generation**

The strategy tracks the rolling mean and standard deviation of the static spread:

$$
\mu_t=\operatorname{RollingMean}(S_t)
$$

$$
\sigma_t=\operatorname{RollingStd}(S_t)
$$

The standardized spread is:

$$
Z_t=\frac{S_t-\mu_t}{\sigma_t}
$$

When the absolute z-score exceeds the predefined entry threshold:

$$
|Z_t|>Z_{\text{entry}}
$$

the strategy triggers a mean-reversion positioning signal based on the predefined weighted spread.

A positive z-score represents an upward deviation relative to the estimated rolling equilibrium, while a negative z-score represents a downward deviation.

The resulting positioning is determined by the direction of the deviation and the corresponding hedge structure.

The rolling statistics are used for signal normalization; they do not alter the calibrated hedge ratios.

### 4.2 `mmc_stress_test.py`: Validation of the Static Model

The second component evaluates whether the static-beta assumption remains sufficiently robust under changing market conditions.

Rather than evaluating the model exclusively within its historical training window, the stress-testing framework applies the calibrated coefficients to entirely unseen market regimes.

**Implementation mechanics**

**1. Out-of-Sample Stress Application**

The test script takes the fixed beta weights generated during training and locks them across unseen, potentially highly volatile market regimes.

This prevents the evaluation from benefiting from continuous coefficient recalibration and directly tests the generalization capability of the original calibration.

**2. Stationarity Verification**

The script performs batch Augmented Dickey-Fuller tests on the resulting static spread time series.

The purpose is to evaluate whether the spread retains evidence of mean-reverting behavior outside the original training period.

The ADF test evaluates the null hypothesis of a unit root against a stationary alternative under the specified test model.

Failure to reject the null does not necessarily establish that the spread is non-mean-reverting, particularly in short or noisy samples. Conversely, rejection does not guarantee that the relationship will remain stationary in future regimes.

For cointegrating residuals estimated through a first-stage regression, appropriate cointegration-test critical values must be used rather than automatically applying ordinary univariate ADF critical values.

The OOS evaluation therefore focuses on whether the original relationship remains statistically defensible rather than treating individual ADF outcomes as definitive proof.

**3. Friction Stress Testing**

The framework explicitly stresses the static model by introducing fluctuating cost vectors that simulate changes in:

- Storage expenses.
- Freight rates.
- Insurance margins.

The purpose is to measure how variations in structural friction components affect the stability and mean-reversion characteristics of the synthetic spread.

One relevant performance measure is the estimated half-life of mean reversion.

For an approximately stationary AR(1) spread process:

$$
S_t-\mu
=
\phi(S_{t-1}-\mu)+\epsilon_t
$$

where \(0<\phi<1\), the estimated half-life is:

$$
\tau_{1/2}
=
-\frac{\ln(2)}{\ln(\phi)}
$$

This represents the expected time required for a deviation from equilibrium to decay by approximately 50% under the assumed autoregressive dynamics.

A larger half-life indicates slower estimated mean reversion, while a smaller half-life indicates faster estimated reversion.

If \(\phi\leq 0\), \(\phi\geq1\), or the process does not adequately satisfy the assumed stationary AR(1) specification, this formula should not be interpreted as a conventional positive mean-reversion half-life.

The stress-testing framework uses these diagnostics to investigate whether the static relationship deteriorates under changing structural costs.

---

## 5. Results and Observations

### 5.1 Initial Approach and Screening Strategy

The initial implementation employed a fixed z-score threshold over a static historical training window, with an example calibration length of approximately 5,000 bars.

The original hypothesis was that incorporating liquidity and macroeconomic factors into the spread definition could improve its structural representation and increase the likelihood of identifying stationary relationships.

The screening architecture used weighted sums to evaluate whether individual feature contributions aligned favorably with expected trading outcomes.

Combinatorial beam searches were used to investigate candidate combinations of assets and explanatory variables.

The intended objective was to isolate viable synthetic spreads whose historical deviations exhibited mean-reverting behavior.

However, the subsequent experiments revealed that the presence of additional explanatory variables did not automatically produce robust out-of-sample stationarity.

### 5.2 Identified Failure Modes and Bottlenecks

**1. Overfitting and False Stationarity**

The fixed 5,000-bar window assumption proved brittle.

Many spreads identified through beam search were not genuinely robust stationary relationships. Instead, they captured temporary historical artifacts that did not generalize reliably to unseen market conditions.

The problem was not necessarily the weighted-sum methodology itself, but the possibility that the screening procedure selected combinations whose historical statistical properties were specific to the calibration period.

Tracking the frequency with which a spread appears in the screener can theoretically provide an additional stability diagnostic.

However, repeated appearances do not independently establish structural stability. A spread may appear frequently because of short-lived volatility spikes, correlated screening criteria, or repeated exposure to the same market regime.

Consequently, screener frequency should be interpreted as a supplementary selection metric rather than independent evidence of cointegration.

Out-of-sample generalization remained a significant bottleneck.

**2. Trade Frequency and Outlier Distortion**

Wide static z-score thresholds produced sparse trading frequency.

In several cases, the strategy became excessively selective, entering only after deviations had become unusually large.

This created the risk of entering lagging outlier moves that represented genuine structural breakdowns rather than temporary deviations from equilibrium.

A large z-score is not necessarily evidence of an attractive mean-reversion opportunity.

It may instead reflect:

- A structural change in the relationship.
- A persistent directional market regime.
- An abnormal volatility event.
- A breakdown in the original hedge ratio.

This distinction highlights the limitation of relying exclusively on static z-score thresholds without additional structural validation.

### 5.3 Engineering Takeaways and Structural Improvements

Despite the out-of-sample performance challenges associated with fixed parameters, the experimental process produced several important architectural insights.

**1. Visual Validation**

Visual inspection of spread plots revealed that some core relationships continued to exhibit recognizable structural behavior out-of-sample, even when strict ADF criteria occasionally failed because of transient noise.

This suggests that statistical test outcomes and visual behavior should be considered complementary diagnostic tools.

However, visual inspection cannot replace formal stationarity testing or establish statistical significance. It is particularly vulnerable to subjective interpretation and should be used to identify potential regime changes, anomalies, and structural patterns requiring further investigation.

**2. Computational Efficiency**

A brute-force combinatorial search across an entire asset universe can become computationally expensive.

If the search evaluates combinations of up to \(m\) variables from an available universe of \(N\) candidates, the number of possible subsets is:

$$
\sum_{k=1}^{m}\binom{N}{k}
$$

For ordered combinations with unrestricted repetition, the corresponding search space can instead scale as \(N^m\).

The implemented filtering mechanism is designed to reduce this search space before the more computationally expensive candidate evaluation stage.

By isolating specific candidate pairs and their relevant features beforehand, the algorithm can investigate structural relationships without exhaustively evaluating every possible combination across the entire asset universe.

This provides an important computational advantage while retaining the ability to investigate multi-factor spread constructions.

The effectiveness of this filtering mechanism, however, must be evaluated independently of its computational efficiency. A smaller search space does not automatically imply better statistical selection or improved out-of-sample performance.

---

## 6. Discussion and Research Implications

The experimental findings highlight a fundamental challenge in statistical arbitrage: identifying a historically stationary spread is substantially easier than establishing that the underlying relationship will remain economically meaningful across changing market regimes.

The proposed static-beta architecture addresses one specific dimension of this problem.

By separating historical coefficient calibration from execution, the strategy avoids continuously changing its estimated equilibrium relationship in response to recent market fluctuations.

This provides three principal engineering advantages:

1. Reduced computational overhead during execution.
2. Consistent hedge-ratio application throughout the trading period.
3. Reduced exposure to noisy coefficient updates generated by rolling regression procedures.

However, these advantages introduce a corresponding trade-off.

A fixed coefficient vector may become progressively less representative of the true underlying relationship when structural market conditions change.

The framework therefore treats beta stability as an empirical condition to be evaluated through out-of-sample testing, rather than an assumption guaranteed by the implementation.

The feature-augmented extension introduces another important research dimension.

Traditional cointegration models primarily focus on statistical relationships between asset prices. Incorporating physical friction costs, macroeconomic variables, market microstructure measurements, and protocol-specific characteristics creates an opportunity to investigate whether these variables provide additional explanatory information about relative asset valuations.

Nevertheless, explanatory power and cointegration are distinct concepts.

A variable can improve the explanation of short-term spread movements without contributing to a stationary long-term equilibrium. Likewise, a historically significant feature may lose explanatory relevance when market regimes change.

Future evaluation should therefore distinguish between:

- The statistical stationarity of the synthetic spread.
- The stability of the calibrated coefficient vector.
- The explanatory contribution of individual features.
- The persistence of mean-reverting behavior out-of-sample.
- The economic profitability of the resulting trading signals after transaction costs, funding, slippage, and execution constraints.

These distinctions are essential for determining whether an identified statistical relationship represents a genuine relative-value opportunity or merely a historical statistical artifact.

---

## 7. Conclusion

This research presents a multi-asset and feature-augmented statistical arbitrage framework centered on static-beta calibration, weighted spread construction, and out-of-sample structural stress testing.

The methodology extends conventional two-asset cointegration analysis by investigating the incorporation of macroeconomic and physical friction variables in commodity markets, alongside market microstructure and protocol-specific characteristics in cryptocurrency markets.

Its implementation deliberately separates historical parameter calibration from real-time execution.

In `strategy.py`, pre-calibrated hedge ratios are applied through computationally efficient weighted-sum operations, with rolling spread statistics used to generate standardized mean-reversion signals.

In `mmc_stress_test.py`, the resulting synthetic spreads are evaluated under unseen market regimes through out-of-sample stationarity testing and structural friction stress scenarios.

Initial experiments identified several limitations, particularly false historical stationarity, overfitting within fixed calibration windows, sparse trade generation, and the tendency of extreme z-score observations to represent structural breakdowns rather than genuine mean-reversion opportunities.

Despite these challenges, the research established several important engineering principles.

First, static coefficient calibration offers a computationally efficient alternative to continuously recalibrating regression models during execution, provided that the associated parameter instability risk is explicitly monitored.

Second, incorporating structural explanatory variables creates a broader framework for investigating the economic and market-specific forces that influence relative asset valuations.

Third, preliminary filtering and candidate isolation can substantially reduce the computational burden associated with combinatorial spread discovery.

Ultimately, the framework does not assume that static beta guarantees stationarity. Instead, it investigates whether a historically calibrated, feature-augmented linear relationship can maintain sufficient statistical and structural stability to support mean-reversion trading across unseen market regimes.

The central research contribution is therefore the integration of computationally efficient static-beta execution, multi-factor spread construction, and explicit structural validation into a unified statistical arbitrage architecture.