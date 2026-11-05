# A Rigorous Evaluation of Out-of-Sample Mean Reversion
## Variance Ratios, Structural Payoffs, and the Persistence Gate
**Status:** Second Implementation Review & Empirical Evaluation alternative 2

---

## Abstract

This module evaluates statistical-arbitrage spread discovery and execution dynamics using high-frequency cryptocurrency price series.

Traditional spread-selection pipelines rely heavily on in-sample (IS) Augmented Dickey-Fuller (ADF) tests. However, searching thousands of candidate spreads through beam-search optimization introduces selection bias: combinations that appear strongly stationary in-sample may exhibit substantially weaker mean-reversion characteristics out-of-sample (OOS).

This project investigates an important distinction:

> An OOS failure to reject the unit-root hypothesis does not necessarily imply that a spread has become non-stationary. It may instead reflect a substantial increase in its mean-reversion half-life.

To investigate this, we combine simulated Ornstein-Uhlenbeck (OU) processes with empirical cryptocurrency spread data and develop an ex-ante framework incorporating:

1. **Half-Life Ratio Prediction:** Ridge regression predicts forward half-life expansion using train-side structural statistics.
2. **Dynamic Structural Payoff:** Candidate z-score windows are evaluated through normalized trade payoff, allowing window selection to account for the expected mean-reversion horizon.
3. **OOS Persistence Calibration:** An isotonic-calibrated persistence model estimates the likelihood of positive structural payoff and supports an explicit trade-filtering rule.

The results suggest that predictive window selection and persistence-based filtering provide a more economically relevant evaluation framework than binary ADF screening alone.

---

# 1. Introduction: Diagnosing OOS Mean-Reversion Failure

A central problem in high-frequency statistical arbitrage is the discrepancy between in-sample screening performance and out-of-sample behavior.

### 1.1 Winner's Curse from Spread Selection

Beam-search optimization evaluates combinations across large feature sets and asset legs. Repeatedly selecting candidates with favorable in-sample statistics creates a winner's curse: the selected candidates tend to exhibit unusually favorable training results that do not fully persist OOS.

In the observed candidate universe:

- Approximately **93%** of train-set ADF tests achieve significance at \(p < 0.05\).
- OOS ADF significance falls to approximately **20%**.
- Empirical half-life expands substantially, with an illustrative increase from **102 bars IS to 373 bars OOS**.

The discrepancy indicates that the original screening statistic does not adequately describe the persistence of the selected spread in future data.

### 1.2 ADF Power Degradation for Slow Mean Reversion

The ADF test evaluates the unit-root null hypothesis against a stationary alternative.

A stationary process with a sufficiently long half-life can be difficult to distinguish from a unit-root process over a finite observation window. Consequently, failure to reject the unit-root null is not equivalent to proving that the process is non-stationary.

This distinction is particularly relevant when the OOS mean-reversion speed is substantially slower than the training estimate.

### 1.3 Statistical Persistence Versus Trading Relevance

ADF significance is a statistical property. Trading profitability is an economic property.

A spread can be stationary over a long horizon but still be unsuitable for a strategy whose holding period, stop-loss distance, and execution constraints operate over much shorter horizons.

The relevant questions are therefore:

- Does the spread retain mean-reverting behavior?
- How quickly does its deviation decay?
- Is the expected decay horizon compatible with the trading window?
- Does the resulting trade distribution retain positive structural payoff?

This motivates evaluating persistence, half-life expansion, and trade payoff jointly.

---

# 2. Theoretical Motivation: ADF Versus Variance-Ratio Power

## 2.1 Ornstein-Uhlenbeck Reference Process

We use the stationary Ornstein-Uhlenbeck process as a controlled reference model:

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t
$$

where:

- \(X_t\) is the spread deviation from equilibrium.
- \(\theta > 0\) is the mean-reversion speed.
- \(\sigma\) is the diffusion volatility.
- \(W_t\) is standard Brownian motion.

The theoretical half-life is:

$$
t_{1/2} = \frac{\ln(2)}{\theta}
$$

The conditional expectation follows:

$$
\mathbb{E}[X_{t+h}\mid X_t]
=
X_t e^{-\theta h}
$$

Therefore, the expected fraction of the initial deviation dissipated over horizon \(h\) is:

$$
D(h)
=
1-e^{-\theta h}
=
1-2^{-h/t_{1/2}}
$$

This quantity measures expected deviation decay. It is **not** the probability that the process crosses its equilibrium level within the specified horizon.

## 2.2 Simulated Statistical Power

We simulate stationary OU spreads using:

- Sample length: \(n=5{,}000\) bars.
- Simulation count: 150 per half-life configuration.
- Multiple theoretical half-life settings.
- ADF rejection threshold: \(p<0.05\).
- Variance-ratio reversion threshold: \(z<-1.645\).

The purpose is to examine how statistical detection changes as mean reversion becomes slower relative to the available sample.

### Table 1. Statistical Detection on Simulated OU Spreads

| Half-Life | ADF Rejection Rate | VR(60) Rejection Rate | Expected Deviation Decay in 60 Bars |
|---:|---:|---:|---:|
| 50 | 1.000 | 0.947 | 0.565 |
| 100 | 1.000 | 0.427 | 0.340 |
| 200 | 0.747 | 0.167 | 0.188 |
| 373 | 0.280 | 0.073 | 0.106 |
| 700 | 0.113 | 0.033 | 0.058 |
| 1500 | 0.067 | 0.087 | 0.027 |

The final column is calculated from:

$$
D(60)=1-2^{-60/t_{1/2}}
$$

The ADF and VR columns represent empirical rejection frequencies across the simulations, not theoretical probabilities.

### Key Finding

At a theoretical half-life of 373 bars, the ADF test rejects the unit-root null in only 28% of simulations.

Equivalently, approximately 72% of these simulated samples fail to reject the unit-root hypothesis despite being generated from a stationary process.

This demonstrates an important limitation of binary ADF screening:

**A failed ADF rejection can reflect insufficient statistical power against slow mean reversion rather than definitive evidence of a random walk.**

The variance-ratio results provide a complementary diagnostic, although the two tests evaluate different statistical properties and should not be interpreted as interchangeable tests of stationarity.

## 2.3 Variance-Ratio Statistic

The variance-ratio statistic is defined as:

$$
VR(q)
=
\frac{
\operatorname{Var}(X_t-X_{t-q})
}{
q\operatorname{Var}(X_t-X_{t-1})
}
$$

Under a random-walk reference model with uncorrelated increments:

$$
VR(q)\approx 1
$$

Mean-reverting behavior can produce:

$$
VR(q)<1
$$

The implementation uses a heteroskedasticity-robust variance-ratio diagnostic to evaluate deviations from the random-walk benchmark.

The reported rejection condition is:

$$
z_{VR(60)}<-1.645
$$

This is a lower-tail, one-sided threshold. Its interpretation depends on the precise variance estimator and standard-error construction used by the implementation.

---

# 3. Co-Movement and Predictive Ex-Ante Features

We evaluate candidate features across:

$$
N=2{,}539
$$

rolling-origin candidate spreads.

Feature relationships are measured using Spearman Information Coefficients (IC), with 20-block bootstrap confidence intervals.

The reported intervals use the 5th and 95th percentiles of the bootstrap distribution.

## 3.1 Ex-Post Forward Payoff Co-Movement

The first analysis identifies structural statistics associated with realized forward payoff.

### Table 2. Ex-Post Forward Correlations with Structural Payoff

| Metric | Information Coefficient | Low 5% CI | High 95% CI | N |
|---|---:|---:|---:|---:|
| `y_noise` | +0.366 | +0.221 | +0.495 | 2,539 |
| `y_phi` | -0.327 | -0.444 | -0.182 | 2,539 |
| `y_loghl` | -0.327 | -0.444 | -0.181 | 2,539 |
| `y_vrz_hold` | -0.317 | -0.406 | -0.186 | 2,539 |
| `y_vr120` | -0.269 | -0.378 | -0.128 | 2,539 |
| `y_hurst` | -0.261 | -0.364 | -0.115 | 2,539 |

The strongest observed relationships include:

- `y_noise`: positive association with structural payoff.
- `y_phi`: negative association with structural payoff.
- `y_loghl`: negative association with structural payoff.
- Variance-ratio and Hurst-related measures: negative associations.

These are ex-post relationships. They describe which realized structural characteristics co-move with payoff, but they do not establish that those characteristics are independently predictive or causal.

## 3.2 Ex-Ante Prediction of Half-Life Expansion

The next analysis evaluates whether train-side structural features contain information about future half-life expansion.

The prediction target is:

$$
y_{HL}
=
\ln\left(
\frac{
t_{1/2,\text{fwd}}
}{
t_{1/2,\text{train}}
}
\right)
$$

Interpretation:

- \(y_{HL}>0\): forward half-life expansion.
- \(y_{HL}<0\): forward half-life contraction.
- \(y_{HL}=0\): unchanged half-life.

### Table 3. Ex-Ante Train Statistics Versus Forward Half-Life Expansion

| Metric | Information Coefficient | Low 5% CI | High 95% CI |
|---|---:|---:|---:|
| `f_hold` / `f_phi` / `f_loghl` | -0.428 | -0.546 | -0.289 |
| `f_score` | -0.426 | -0.517 | -0.321 |
| `f_noise` | +0.415 | +0.281 | +0.531 |
| `f_adf_stat` | -0.410 | -0.524 | -0.272 |

The results indicate that train-side structural statistics contain useful information about subsequent half-life expansion.

In particular, the observed negative association between `f_hold` / `f_phi` / `f_loghl` and the expansion target suggests that stronger train-side persistence characteristics are associated with less severe forward half-life deterioration.

These correlations motivate a predictive model rather than a fixed assumption that train-side half-life remains constant.

---

# 4. Modeling and Validation

## 4.1 Half-Life Predictor

We compare a regularized Ridge regression model against two simple baselines.

The prediction target is the logarithmic half-life ratio:

$$
\widehat{y}_{HL}
=
\widehat{
\ln\left(
\frac{
t_{1/2,\text{fwd}}
}{
t_{1/2,\text{train}}
}
\right)
}
$$

The corresponding predicted forward half-life is:

$$
\widehat{t}_{1/2,\text{fwd}}
=
t_{1/2,\text{train}}
\exp(\widehat{y}_{HL})
$$

The logarithmic target makes expansion and contraction multiplicative rather than additive.

### Validation Design

We use five-fold purged time-blocked cross-validation.

The purge interval is designed to cover:

$$
\text{Purge Span}
=
\text{Training Span}
+
\text{Maximum Forward Horizon}
$$

This reduces contamination from overlapping observations and forward-label windows.

The Ridge model is compared against:

1. Training half-life used directly as the forecast.
2. A shrunk constant half-life ratio.
3. Ridge regression using train-side structural statistics.

### Table 4. Purged CV Half-Life Ratio Error

*Log-space error; lower is better.*

| Predictor Model | MAE (Log) | RMSE (Log) | Rank Correlation |
|---|---:|---:|---:|
| Train Half-Life (Naive) | 1.190 | 1.538 | NaN |
| Shrunk Constant Ratio | 0.844 | 1.134 | NaN |
| Ridge on Statistics | **0.827** | **1.099** | **0.309** |

Ridge regression improves on both baselines.

The MAE reduction relative to the naive model is:

$$
\Delta MAE
=
1-\frac{0.827}{1.190}
\approx 30.5\%
$$

This indicates that train-side structural statistics provide additional information about forward half-life behavior.

The improvement is predictive rather than causal: it demonstrates that the selected features help forecast the target under the validation design.

## 4.2 Dynamic Z-Score Window Selection

A spread's z-score depends on the rolling window used to estimate its mean and standard deviation.

For a spread \(S_t\), the rolling z-score is:

$$
z_t
=
\frac{
S_t-\mu_{t,W}
}{
\sigma_{t,W}
}
$$

where:

- \(\mu_{t,W}\) is the rolling mean over window \(W\).
- \(\sigma_{t,W}\) is the rolling standard deviation.
- \(W\) controls the statistical memory of the signal.

A fixed window may be too short for slow-moving spreads or too long for faster-moving regimes.

We therefore compare fixed windows with half-life-scaled policies:

$$
W=k\cdot t_{1/2}
$$

For predictive policies, the estimated half-life replaces the unavailable forward half-life.

## 4.3 Structural Trade Payoff

We define normalized structural trade payoff as:

$$
R
=
\frac{
\text{PnL}
}{
\text{Initial Distance to Stop-Loss}
}
$$

This normalizes realized PnL by the initial stop-loss distance, providing a common payoff scale across candidate trades.

It is a structural payoff measure, not automatically equivalent to a net return, Sharpe ratio, or portfolio-level risk-adjusted performance.

### Table 5. Realized Structural Payoff Across Window Policies

| Window Policy | Mean R | Low 10% CI | High 90% CI | Trade Count |
|---|---:|---:|---:|---:|
| Fixed 60 | **0.961** | 0.597 | 1.358 | 139,516 |
| Fixed 120 | 0.556 | 0.307 | 0.802 | 88,763 |
| 2 × True Half-Life (Oracle) | 0.348 | 0.240 | 0.496 | 37,974 |
| 2 × Train Half-Life | 0.345 | 0.291 | 0.403 | 70,209 |
| 4 × Predicted Half-Life | 0.274 | 0.102 | 0.478 | 19,227 |
| Fixed 240 | 0.256 | 0.212 | 0.306 | 54,104 |
| 3 × Predicted Half-Life | 0.199 | 0.095 | 0.326 | 23,404 |

### Interpretation

Several observations emerge:

1. Fixed 60 produces the highest reported mean structural payoff.
2. Longer fixed windows reduce trade frequency.
3. Half-life-scaled windows produce materially fewer trades than short static windows.
4. Predicted half-life windows provide a way to adapt signal memory to estimated persistence.

The results do not establish that dynamic windows maximize mean payoff. Instead, they show a trade-off between payoff, signal memory, and trading frequency.

The primary motivation for dynamic window selection is to reduce dependence on a single arbitrary lookback period and align the signal's statistical memory with the expected mean-reversion horizon.

---

# 5. The Persistence Gate: Out-of-Sample Evaluation

## 5.1 Objective

A candidate spread can exhibit favorable train-side statistics while deteriorating in the forward period.

The persistence gate attempts to identify candidates whose structural payoff is likely to remain positive.

The framework combines:

- Ridge-based structural prediction.
- Isotonic calibration.
- Empirical payoff confidence bounds.
- A score-based acceptance threshold.

The gate is designed to evaluate persistence before allocating capital, rather than relying exclusively on historical ADF significance.

## 5.2 Calibration and Decision Logic

Let the model produce a raw persistence score:

$$
s_i=f(X_i)
$$

where \(X_i\) contains train-side structural features.

Isotonic regression maps the raw score to a calibrated persistence estimate:

$$
\widehat{P}(\text{Persist}\mid s_i)
=
g(s_i)
$$

where \(g\) is a monotonic calibration function.

The economic acceptance condition is based on the estimated lower confidence bound of structural payoff:

$$
\text{Accept}_i
\iff
\operatorname{lo\_R}(s_i)>0
$$

The key distinction is that a high estimated probability of persistence is not itself sufficient. The corresponding payoff distribution must also retain a positive lower confidence bound.

## 5.3 Cumulative Confidence Cuts

### Table 6. Cumulative Confidence Cuts and Gate Performance

| Quantile | Raw Score Cut | Coverage | P(Persist) | Mean R | Low 10% CI (`lo_R`) | High 90% CI | Trades |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | -5.417 | 0.999 | 0.453 | 0.316 | 0.134 | 0.527 | 23,727 |
| 0.250 | -0.145 | 0.749 | 0.455 | 0.402 | 0.156 | 0.684 | 17,882 |
| 0.500 | 0.205 | 0.499 | 0.436 | 0.367 | 0.080 | 0.721 | 11,918 |
| 0.700 | 0.520 | 0.300 | 0.490 | 0.513 | 0.039 | 1.055 | 7,033 |
| 0.800 | 0.702 | 0.200 | 0.522 | 0.732 | 0.030 | 1.444 | 4,693 |
| 0.900 | 1.065 | 0.100 | 0.587 | 0.007 | -0.086 | 0.114 | 2,278 |
| 0.950 | 1.302 | 0.050 | 0.661 | 0.064 | -0.074 | 0.209 | 1,091 |

## 5.4 Interpreting the Gate

The table indicates that the lower 10% payoff bound remains positive through the 80th-percentile score cut:

$$
\operatorname{lo\_R}(0.800)=0.030>0
$$

At the 90th-percentile cut:

$$
\operatorname{lo\_R}(0.900)=-0.086<0
$$

Therefore, the reported results support positive lower-tail structural payoff through the 80th-percentile cut, but not at the 90th-percentile cut.

This provides an empirical basis for evaluating the operating threshold.

**Implementation distinction:** The reported raw score of `-5.417` is the baseline, zero-quantile score cut. It should not be described as the final selective operating threshold if the intended rule is to retain only candidates passing a positive-payoff confidence condition. Under the reported table, `0.702` is the 80th-percentile score cut associated with 20% coverage and a positive `lo_R` of `0.030`.

The final deployed threshold must be selected using calibration data and evaluated on a separate locked OOS sample. The table describes the observed calibration results; it does not independently establish the performance of a threshold selected after inspecting those same results.

---

# 6. Implementation Summary and Key Findings

## 6.1 Diagnosis: ADF Failure Is Not a Binary Structural Verdict

The OU simulation demonstrates that finite-sample ADF power can deteriorate substantially as half-life increases.

At a half-life of 373 bars, the simulated stationary process rejects the unit-root null only 28% of the time.

This supports a more careful interpretation of OOS ADF failure:

- ADF non-rejection does not prove non-stationarity.
- Slow mean reversion can be difficult to identify over finite samples.
- Structural breaks and genuine loss of stationarity remain alternative explanations.
- Half-life and complementary diagnostics provide additional information.

## 6.2 Ex-Ante Prediction Efficiency

The Ridge model improves log-space half-life forecast error:

$$
MAE_{\text{Naive}}=1.190
$$

$$
MAE_{\text{Ridge}}=0.827
$$

This corresponds to an approximately 30.5% reduction in MAE under the reported purged validation design.

Train-side structural statistics therefore contain useful predictive information about forward half-life expansion.

## 6.3 Dynamic Window Optimization

Half-life-scaled windows provide an adaptive alternative to fixed z-score lookbacks:

$$
W=3\widehat{t}_{1/2}
$$

This allows signal memory to respond to predicted persistence rather than assuming every candidate spread operates on the same timescale.

The reported results show lower trade counts for dynamic policies, although fixed 60 retains the highest mean structural payoff in the comparison.

The appropriate policy therefore depends on the trade-off between payoff, frequency, holding horizon, and execution constraints.

## 6.4 Execution Safeguard: Persistence-Based Filtering

The persistence gate adds an economic acceptance condition to the statistical screening process:

$$
\operatorname{lo\_R}>0
$$

This shifts the decision from asking only whether a spread appears stationary to asking whether its estimated forward structural payoff remains positive under the chosen confidence criterion.

The reported results show positive lower-tail payoff estimates through the 80th-percentile score cut.

The operational threshold should be calibrated and locked before final OOS evaluation.

---

# 7. Research Interpretation and Limitations

This module focuses on structural diagnostics and predictive validation. Several distinctions are important when interpreting its results.

### Statistical stationarity is not equivalent to profitability

A stationary spread may revert too slowly to be economically useful. Conversely, a spread may generate positive short-horizon trading opportunities without exhibiting strong evidence of long-run stationarity.

### Predictive association is not causation

The reported Information Coefficients identify relationships between structural features and future outcomes. They do not prove that any individual feature causes persistence or profitability.

### Structural payoff is not portfolio performance

The normalized payoff measure does not automatically account for portfolio-level exposure, correlated positions, capacity, or all execution costs. Net trading performance requires explicit modeling of fees, spread, slippage, funding, and market impact.

### Model selection must remain separated from final evaluation

Feature selection, Ridge tuning, isotonic calibration, and threshold selection should be performed without access to the final locked OOS evaluation period.

Where forward labels overlap, validation and uncertainty estimation should account for their dependence.

### Simulation results depend on the assumed process

The OU model provides a controlled stationary reference process. Real cryptocurrency spreads can exhibit volatility clustering, jumps, structural breaks, changing correlations, and non-linear dynamics that are not captured by a basic OU process.

---

# 8. Final Conclusion

The central result of this research module is that **binary ADF screening is an incomplete measure of spread quality**.

A candidate spread should not be evaluated solely on whether its in-sample residual rejects a unit-root hypothesis. Its estimated mean-reversion horizon, forward half-life stability, structural payoff, and persistence under OOS evaluation are also relevant.

The proposed framework combines:

1. OU simulation to demonstrate finite-sample limitations of ADF screening.
2. Variance-ratio diagnostics as a complementary measure of spread dynamics.
3. Ridge regression to forecast forward half-life expansion.
4. Half-life-scaled z-score windows to adapt signal memory.
5. Isotonic calibration and payoff confidence bounds to construct a persistence-based acceptance rule.

The reported experiments show that train-side structural statistics improve half-life forecasting over simple baselines, while persistence-based payoff analysis offers a more economically meaningful way to evaluate candidate spreads.

The broader engineering principle is straightforward:

> Do not optimize a spread merely because it looks stationary in-sample. Estimate how its persistence evolves, align the trading horizon with its expected dynamics, and require evidence that its forward structural payoff remains positive.

**The objective is not to find the spread with the smallest historical ADF p-value. It is to identify spreads whose statistical behavior remains useful when the model is asked to trade beyond the data that selected them.**