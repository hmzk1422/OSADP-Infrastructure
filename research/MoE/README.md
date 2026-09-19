
# System Architecture
Historical training
```
Raw market data
      │
      ▼
Feature engineering + normalization
      │
      ▼
Autoencoder
      │
      ▼
Latent representations z_t
      │
      ▼
Latent State Discovery
      │
      ▼
Verified states / prototypes
{z̄₁, z̄₂, ..., z̄K}
      │
      ▼
Historical observations are matched to states
      │
      ▼
Similarity scores Sₜ,ₖ
      │
      ▼
Split historical data by latent state
      │
      ├── State 1 → train Expert M₁
      ├── State 2 → train Expert M₂
      ├── State 3 → train Expert M₃
      └── ...
```
Live running
```
New market window x_t
       │
       ▼
Same deployed encoder
       │
       ▼
z_t
       │
       ▼
Compare against verified states
       │
       ▼
Similarity scoring Sₜ,ₖ
       │
       ├── Known State 1 ──┐
       ├── Known State 2 ──┤
       ├── Known State 7 ──┤
       └── No sufficient match
                           │
                           ▼
                       Latent Null
```

## Changes to original strategy
Change target focus from expectancy to both sortino and expectancy/trade. We should train the model for both. Or the easier way, train for lower RR model and focus fully on increasing expectancy/trade or in this case, WR.

## Latent State Identification
Assume d = 15, where d = dimensions representing features used to identify the markets. Vector Quantized-Variational AutoEncoder (VQ-VAE) is used to provide continuous similarity and reusable state. It provides $z_t$ for similarity metrics, while the quantized representation $q(Z_t) = e_k$ provides convenient identity/memory address for each states. However it introduces quantization error where two subtly different markets can still get assigned to $e_1$. Should this be the case, experiment for continuous AE + Clustering/prototype approach (GMM, HDBSCAN). 
(Baseline: GMM or PCA possibly, or no regime detection).

Once its created, in training data run VQ-VAE to identify latent state types, returning 

Codebook: The learned dictionary of discrete vectors used to convert continuous data representations into finite specific set of symbols or tokens.
$$C={e1​,e2​,e3​,…,eK​}$$
$$e_1​=[0.40,−0.80,0.20]$$
$$ e_2=[-0.70,0.20,0.60] $$
$$ e_3​=[0.10,0.90,−0.30].$$
The encoder produces the continuous $z_t$, then VQ selects the nearest codebook vector. It would return something like $z_t = [0.42, -0.81, 0.17]$ and its closest to $e_1$, hence $q(z_t) = 1$. This is repeated in a time series. Once latent states are identified, perform training window. Then do purged chronological walkforwards. In simple terms, aggregate all latent 1, latent 2, ... then train the model.  
>Very important: Since we're using AE, its very prone to overfitting and instability. Use L2 regularization to make latent representations capture broad, repeatable structure rather than highly specific historical configurations ($L=L_recon​+λ∥θ∥^2_2​+L_VQ​$) where L2 discourages unnecessarily large network weights  
>
> **Pertubation stability test (latent robustness)**: Small feature noise, tiny scaling changes, missing/pertubed observation then compare $S_(z_t, z^{'}_t)$ where $z_t$ is original state and $z^{'}_t$ is added-noise observation (Use this to evaluate how sensitive VQ-VAE is to change).  
>
> **Codebook/State stability with random seeding**: Compare whether the resulting partitions are similar using the Adjusted Rand Index (ARI) or by matching the resulting codebook vectors. We want independent training runs with different random seeds to discover broadly similar state structure, even if the state labels differ. If different seeds produce substantially different state partitions, the latent-state representation may be unstable and should be treated with caution. For example w1,2,3 -> latent 1; w4,5 -> latent 2. Rerun it. w1,2,3 -> latent 3; w4,5 -> latent 4. It can correctly identify w1,2,3 and w4,w5 as distinct latent spaces.
>
> **Purged Chronological Walkforward Stability** (Or temporal walk forward): Does the similar observations still map to similar latent regions? $S(f_a(x), f_b(x))$ should remain stable for common market conditions. Run the strategy through purged chronological walk forward to see if it returns acceptable edge and sortino, how close is it to backtested result (Don't do monte carlo as it might cause a loss of state persistency)
>
> **State robustness to future recurrence**: Take state discovered in training $e_k$ and ask how often it reappears later $z_t \approx e_k$. Check if similarity remains meaningful, state identity isn't constantly flipping and expert performance under that state remains reasonably stable/close to backtested result. 
>
> **Test persistency with dwell time**: Measure how many consecutive bars it stays sufficiently similar for. Or a better metric, how many trades went out-of-scope as in the regime switched when still holding trades.

```
Assume τ = 50%

Latent states (Training):
e1 -> train M1
e2 -> train M2
e3 -> train M3
...

Time (Live):
t1 -> state 1 (78% similarity) -> Activate expert 1
t2 -> state 1 (60% similarity) -> Activate expert 1
t3 -> state 3 (80% similarity) -> Activate expert 3
t4 -> state 3 (70% similarity) -> Activate expert 3
t5 -> state 2 (20% similarity) -> Null. Stop trading because S_t < τ.
```
Hence the MoE system. In simple implementation, each model trained based on the latent states are used as they are assumed to work the best. However, model trained on latent 1 isn't necessarily the strongest trader for latent 1. Hence there needs to be an comparison matrix layer, comparing the following:
- The same strategy trained under different latent states. Does it return higher expectancy compared to baseline (latent-trained model).
- New strategies (not necessarily variation of the same strategy). For ex, there's a certain latent state where OFMR performs terribly. We add a new strategy (Trend trading for example). 
- Very important to track average expectancy in purged chronological walkforward and cVar of edge to anticipate edge decay. 

## Similarity Score
| Metrics Used | Description | System Implementation |
|:---|:---|:---|
| **Cosine similarity / TS-SS** | Measures similarity between the current latent state and historical states. Cosine captures directional similarity but is invariant to magnitude; TS-SS additionally incorporates magnitude/separation information. Latent-space isotropy will be treated as an empirical property and evaluated after representation learning rather than assumed a priori. | **O(d)** per comparison, making both suitable for live candidate retrieval. |
| **Mahalanobis** | Measures how unusual the current latent point is relative to the covariance structure of a historical regime. More informative than Euclidean distance when different latent dimensions have different variances/correlations. Requires a well-conditioned covariance estimate and may require regularization in higher dimensions. | **O(d²)** per comparison when the inverse/regularized covariance matrix is precomputed. Suitable for live use when latent dimensionality is kept bounded. |
| **Wasserstein** | Measures how closely the recent latent-state window resembles the historical distribution of a regime. Captures distributional differences that point-based metrics may miss. | Simple 1D implementations are approximately **O(W log W)** due to sorting, but multivariate optimal transport can be substantially more expensive. Likely better used as a secondary confirmation metric rather than for all live comparisons. |

The overall similarity score $S_{t,k}$ will combine complementary geometric and distributional measures. Cosine similarity or TS-SS provides fast candidate retrieval, Mahalanobis distance evaluates point-level consistency with the regime distribution, and Wasserstein distance provides an optional window-level distributional check.

An ablation study will evaluate each metric independently and in combination, including under conditions where the assumptions of individual metrics are violated.

# NOTE:
MoE is its own thing. We use OFMR as baseline strategy for now to see if it works or not.

am i not doing that though? im basically using the autoencoder to identify if a regime exists within our books or not. if we've seen it before, we know how to trade it. if we never seen it before, its latent null so we use autoencoder still to reconstruct the market. then we use similarity score to score when the market aligns very closely to the autoencoder which signifies we entered a market that we've seen befofre

# Inspiration
Traditional backtesting assumes a stationary market, causing static strategies to fail via alpha decay. Instead of fitting models to historical return realizations, our framework explicitly models time-varying market dynamics and concept drift. There requires an OOD detection system for our strategy.