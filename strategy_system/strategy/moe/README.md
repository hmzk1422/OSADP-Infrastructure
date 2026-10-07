# Orderflow Mean Reversion (OFMR) Scalp
```
OSADP version : v1.0.0
Author        : Himaska Lin
```

## Structure
```text
moe/
├── strategy.py     # engine: settings -> data -> features -> OOD -> outcomes -> policies
├── features.py     # compute_features(): joins macro + micro feature groups
├── outcome/        # fixed/dynamic barriers, future returns, timeout regression
├── ood/            # raw temporal embeddings, recurring states, OOD/null detection
├── gp/              # bounded causal expression grammar and state-conditioned evolution
├── settings.yaml   # every tunable parameter (data / features / barriers / policy)
├── exchange/       # data access: load_L2, load_L1 (strategy_system/exchange/loader.py)
├── features/       # feature math: rsidprice.py, orderflow.py
└── policies/       # rules of the strategy: orderflow_prediction.py
```

## Data flow
```text
strategy.py::strategy(df)
  1. compute_features(df, **settings[features])        features.py
       ├── compute_macro_features -> features/rsidprice.py
       └── compute_micro_features -> features/orderflow.py
  2. OOD detector on raw df                           known/null latent states
  3. outcome builders                                 targets + labelled events
  4. optional GP search                              per-known-state features
  5. evaluate_orderflow_policy(feat_df, cols, policy)  policies/
  6. StrategyResult(features, regimes, events, outcomes, policy, gp, metalabel)
```

## Abstract
Orderflow provides strong correlation to short term future price. Use rsi(2) relationship with d^2 price and short term correlation orderflow to help

## Why this edge might work
- Orderflow might contain information that filters out rsi(2) trades.

## Outcome design

`outcome/triple_barrier.py` provides fixed percentage, volatility-scaled, and
predicted-MFE/MAE barrier builders. Set `outcome.triple_barrier.mode` to select
the strategy's builder; the separate functions remain available for side-by-
side studies. `horizon_bars` is a number of observations, while `horizon` is a
clock duration and `horizon_col` allows a per-event bar count or duration. Clock
horizons use the first observation at or after the requested time. Incomplete
vertical horizons are excluded by default. If OHLC highs/lows are supplied,
touches use them and a same-bar PT/SL tie is labeled as a stop. Optional bid/ask
columns use ask-entry/bid-exit for longs and bid-entry/ask-exit for shorts;
`fee_bps_per_side` is charged once at entry and once at exit. Event `return` is
net of those explicit fees; `gross_return` and `fee_return` are also retained.
Barrier `label` records which barrier was touched, while `meta_label` records
whether the executed net return was positive. MFE/MAE are signed side-adjusted
reference-price excursions (MFE >= 0, MAE <= 0), and `uniqueness_weight`
discounts overlapping event intervals.

`outcome/timeout.py` builds raw forward simple-return targets and optional
long/short MFE and MAE targets for configurable clock-time horizons (or integer
bar horizons). Excursions use observed highs/lows when configured, otherwise
the sampled price path; their target timestamps are retained for purging.
Clock targets select the first observation at or after the requested time;
`forward_returns.max_lag` limits how far a delayed observation may overshoot
that target. Targets are returned separately from the causal feature frame. Its multi-horizon linear regression
requires an explicit chronological `train_mask` and purges labels whose future
endpoint reaches the validation boundary. A separate `regime_mask` selects
rows for a state-specific fit without changing the chronological split. OLS
and ridge standardize features using training rows only. Timeout selection
ranks cost-adjusted expected return per square-root duration; this is a ranking
rule, not a profit guarantee. Dynamic barriers scale trailing per-observation
volatility to the requested bar or clock horizon. Predicted barriers are
available when causal out-of-sample MFE/MAE predictions have been produced by
an upstream model; the barrier builder itself does not train that model.

## Run
```bash
# from the repository root
python -m strategy_system.strategy.moe.strategy
# optional: point at another settings file
python -m strategy_system.strategy.moe.strategy path/to/settings.yaml
python -m strategy_system.strategy.moe.strategy example  # load L2 -> OOD -> GP -> linreg -> triple barrier
```

File structure:
moe
 |_ features --> everything will be as is. bid ask, their qty, the depth, etc. for now, this is probably empty
 |_ policies
 |_ gp (or genetic_programming) --> grammar.py, population.py, evolution.py, etc
 |_ outcome --> triple_barrier.py, timeout.py, etc
 |_ ood --> encoder.py, regimes.py, distance.py, detector.py, etc
 |_ regime_registry --> registry.py, models/
 |_ feature.py
 |_ strategy.py
 |_ settings.yaml
Task: Focus on making policies. GP, outcome, OOD.
- strategy.py will run compute_features first. OOD will run first, differentiating the given price data to many latent states. this is an MoE-like framework. for ex, there are 10 latent states it managed to identify (10 represents different 10 different market regimes, but recurring latent states should score a high similarity score or the same temporality pattern as their identified latent (e.g., latent 10 should reoccur, and latent 10 should have the same temporality pattern or similarity score). there should be latent null that represents: a regime that the MoE hasn't stored or observed before. This then should become a learning period. it doesnt execute trades. it learns, until eventually it identifies a reoccuring latent state again.
- After that, from those features (bid, ask, qty, etc), itll go through a function from gp (for example, gp.extrapolate_features which will return the columns that gp found to be useful). gp performs basic arithmetic (+,-,/,*), calculus (derivatives and partial differentiation specifically), lookback. Limit the branching height to 5 maybe or 10. so each latent state has a selection of features that works well (\(question here, how do we identify latent states if there are no features built yet? propose a better method to solve this. and how does it know the objective (the triple barrier method) if it hasnt been built yet? should we change our objective?\)).
- Those features should then be used for triple barrier sequence, where tree models takes the features that gp used and backtested. then a linear regression will be used to take advantage of the short term correlation advantage (where we believe orderflow has a edge with short term future price, linear regression should also use the gp). We set the timeout based on the time period where linear regression returned a positive return (for example, we found that at t+9, the return is +0.100%, so we know the edge is in our favor where if we win, or if we timeout, we still profit. the stop volatility size should be resized according to the timeout). so we should segregate them by their functions (for example, triple barrier sequence builder, either we have fixed triple barriers and the model determines if a trade will win or not, or we use the model to determine the MAE/MFE with the objective of maximizing average expectancy per trade \(Question here, should we use fixed or use dynamic?\))
Functions should be clearly segregated. for example, triple barrier.py contains multiple segregated functions which includes one for building the triple_barrier_fixed, one for triple_barrier_dynamic, one for the timeout prediction with linear regression model, etc
Solution to the answers:
1. OOP must be independent of GP. it should operate on a fixed set of raw, always-available observations and their temporal sequence (such as price/returns, bid/ask, spread, bid/ask qty, depth, orderflow, liquidation primitives). use temporal encoder on rolling sequence rather than single row, producing a latent embedding. Cluster/store recurring embeddings as latent states. use similarity/local-density distance against stored latent states to determine whether the current state belongs to known latent or null latent. In simple words, provide pre calculated features (primitive features we already know of for the OOP to identify.
2. Use future return regression as the target objective from 1s, 5s, 10s, 20s, 40s, 60s, 2m, 3m, 4m, 5m, etc. this should be changeable in settings.yaml. If a variable has predictive power on the future, we can say it has a high predictive power too for predicting tp hit, with the only downside being mfe hit. so the algorithm should be able to know how to use correlative features to predict for example, where/if it will mfe be hit? where/if it will mae be hit? for example, the features tell you that around 5s-10s, price will go donw -0.15%. stop hit. Correlation stays just as a metric.
3. we should use both fixed and dynamic
and remember this is state-based problem, not time based problem. so if the features fit a certain condition it will say "[entry zone]" or whatever (basically when the condition appears itll enter) rather than saying "enter after 3s pass" or smt like that.

## OOD state detection

`ood/` is independent of GP-evolved features and forward outcomes. Its four
modules separate the swappable `BaseRegimeEncoder`, swappable
`BaseRegimeSegmenter`, `RegimeValidator`, and top-level `OODSystem`. The default
encoder is a causal ordered-window baseline; `TCNRegimeEncoder`,
`TS2VecRegimeEncoder`, and `TransformerRegimeEncoder` implement the same encoder
contract. Hysteresis clustering and the sticky state model implement the same
segmenter contract. Inject a concrete encoder or segmenter to compare methods
without changing strategy integration.

Discovery consumes raw or explicitly selected primitive columns. The default
encoder combines causally normalized within-window patterns with slow
volatility/liquidity level channels, so persistent conditions remain visible
after the fast rolling baseline adapts. It returns one timestamp-aligned
embedding per row and does not read outcomes.
New prototypes remain `null` until they have both
`min_novel_observations` support and `min_novel_episodes` separated recurrence
episodes. This prevents one close embedding from defining a known regime. The
promotion row remains null. `min_dwell` causally debounces every label,
including `null`; while a transition is pending, the public label stays stable
and `trade_allowed` is false. Raw and filtered assignments are both returned
so validation can compare their episode durations and transition stability.

The validator reports embedding similarity, temporal-pattern similarity,
episode-duration similarity, and persistence separately. Forward expectancy
is calculated only when forward outcome columns are explicitly supplied to
`validate_regimes`; outcomes never enter encoder fitting, prototype creation,
or state assignment. Similarity and expectancy are diagnostics, not a combined
regime score. Supply benchmark labels as `ground_truth` to
`validate_regimes()` for ARI, NMI, recurrence purity, and assignment coverage.

The default `strategy()` call starts a fresh OOD system. For continuity across
chunks or runs, keep one system and pass a raw context prefix for encoder
warmup. Set `ood_process_from` to the first new row so context rows are not
processed twice:

```python
from strategy_system.strategy.moe.ood import OODSystem
from strategy_system.strategy.moe.strategy import load_settings, strategy

settings = load_settings()
detector = OODSystem(**settings.get("ood", {}))
# Keep a raw context prefix; only new rows update the learned registry.
result = strategy(
    raw_context_and_new_rows,
    settings,
    ood_detector=detector,
    ood_process_from=len(raw_context),
)
detector.save("ood-system.pkl")
# Later: detector = OODSystem.load("ood-system.pkl")
```

Keep raw primitive columns and their order fixed when reusing a saved system.
The encoder uses preceding observations as normalization history, so each new
chunk needs a raw warmup prefix; pass `process_from` at the first new row to
avoid processing context twice. With `ood.timestamp_col: null`, the system
auto-detects a `timestamp` column, then a DatetimeIndex, and otherwise
preserves row order. IDs are persistent prototype keys, not semantic names;
keep model names and fitted policies in a separate registry.

## Genetic programming

`gp/` is the offline feature discovery package. Its public modules separate
grammar, population/evolution, evaluation, and per-latent feature-library
persistence. Expressions are bounded by `gp.max_height`; arithmetic can
recurse while temporal and calculus operations are restricted. Window
operations are rolling and causal; block summaries update only from the current
fixed block's observed prefix.

`extrapolate_features()` runs one search per known latent and target. `null`
rows are excluded by default; `gp.include_null: true` allows an explicit
offline search after `gp.min_null_samples` are available. The search screens
generated expressions by target correlation, Ridge, and redundancy; fitness
uses model-family-balanced scores and a search-size-adjusted t-stat gate.
Purged expanding folds hold out recurring latent episodes, followed by joint
feature-set evaluation and a later chronological holdout veto. Report columns
keep model errors, directional accuracy, expectancy, recurrence consistency,
and stability separate.

The expression model is stored as data by latent ID. `result.gp.features` are
the selected columns, `result.gp.report` contains diagnostics, and
`result.gp.feature_library` contains the versioned library. Set
`gp.feature_library_path` to persist it. The GP switch is false by default and
must be explicitly enabled for offline retraining; no evolution is performed
for live observations. The ordinary strategy output and live OOD labels do not
trigger GP retraining.

# Summary Structure
All three are maintained to be as decoupled to each other as possible.
1. OOD -> Resamples market data to many kinds of latent data (AutoEncoder passes temporality similarity and similarity score between recurring regimes). Market data is chunked in this order. Either include precomputed, meaningful features or not (look at its effectiveness).
2. GP (Optional) -> **Main point is to use RankIC and Linreg (or tree if its non-linear) w features to correlate with t+1, t+2, ... t+n time.** Either use precomputed, meaningful features or use GP to automate feature development. Those precomputed, meaningful features should pass purged chronological walkforwards on latent states.
3. Triple Barrier -> Optimizer. Tree model evaluates MFE/MAE based on computed linreg prediction on future prices. Tree model can take the precomputed features or GP features to re-evalutae as metalabeller.
