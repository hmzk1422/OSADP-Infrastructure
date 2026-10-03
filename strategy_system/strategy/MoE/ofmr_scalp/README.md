# Orderflow Mean Reversion (OFMR) Scalp
```
OSADP version : v1.0.0
Author        : Himaska Lin
```

## Structure
```text
ofmr_scalp/
├── strategy.py     # engine: settings -> data -> features -> OOD -> outcomes -> policies
├── features.py     # compute_features(): joins macro + micro feature groups
├── outcome/        # fixed/dynamic barriers, future returns, timeout regression
├── ood/            # raw temporal embeddings, recurring states, OOD/null detection
├── gp/              # bounded causal expression grammar and state-conditioned evolution
├── settings.yaml   # every tunable parameter (data / features / barriers / policy)
├── exchange/       # data access: load_orderbook_state, load_ohlcv_state
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
  6. StrategyResult(features, regimes, events, outcomes, policy, gp)
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
python -m strategy_system.strategy.MoE.ofmr_scalp.strategy
# optional: point at another settings file
python -m strategy_system.strategy.MoE.ofmr_scalp.strategy path/to/settings.yaml
```

File structure:
ofmr_scalp
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

`ood/` is independent of GP-evolved features. It builds fixed observations
from raw price/return, spread, bid/ask queue and depth, trade-flow, activity,
and liquidation columns. Its embedding combines fast rolling-z temporal shape
with slow rolling volatility/liquidity levels normalized against expanding
past statistics. This preserves persistent high-volatility or wide-spread
conditions after the fast rolling baseline adapts. During either encoder
warmup, `latent_id` is `null` and the row is non-tradeable.

The online registry matches known prototypes using a per-state diagonal
Mahalanobis distance and an observed state-specific radius. Candidate centers
are promoted only after both `min_novel_observations` and
`min_novel_episodes` are met; observations closer together than
`candidate_episode_gap` count as one episode, so overlapping windows cannot
promote a single transient. Stale candidates expire, the oldest candidate is
evicted if its buffer is full, and inactive prototypes expire after
`state_expiry_observations`. Prototype centers are frozen by default to keep
state identities from drifting together. The promotion row remains `null`.

A promoted state then needs `mature_state_observations` post-promotion matches.
Execution also requires a fitted state model: configure its ID in
`ood.model_ready_states`, or call
`detector.set_model_ready_states(state_ids)` before detection. The
default empty list deliberately blocks execution when no fitted model has been
registered. `strategy()` attaches `latent_id`, `state_mature`, `model_ready`,
and `trade_allowed` to both all-row forward targets and triple-barrier events.
For a model fitted to one state, use a separate row filter such as
`result.outcomes['latent_id'].eq(state_id) & result.outcomes['trade_allowed']`;
keep that filter separate from the chronological train/validation mask.
Outcome labels remain available for research during null and maturing periods.

The default `strategy()` call starts a fresh registry. For continuity across
chunks or runs, keep one detector and pass a raw context prefix for feature and
encoder warmup. Set `ood_process_from` to the first new row so the context is
not processed twice:

```python
from strategy_system.strategy.MoE.ofmr_scalp.ood import OODRegimeDetector
from strategy_system.strategy.MoE.ofmr_scalp.strategy import load_settings, strategy

settings = load_settings()
detector = OODRegimeDetector(**settings.get("ood", {}))
# Keep a raw context prefix; only new rows update the learned registry.
result = strategy(
    raw_context_and_new_rows,
    settings,
    ood_detector=detector,
    ood_process_from=len(raw_context),
)
saved_registry = detector.state_dict()
# Later: detector = OODRegimeDetector.from_state_dict(saved_registry)
```

Keep raw primitive columns and their order fixed when reusing a registry. The
encoder uses preceding observations as normalization history, so a new chunk
needs a raw warmup prefix; pass `process_from` at the first new row so context
does not update the registry a second time. With `ood.timestamp_col: null`,
the detector auto-detects a `timestamp` column, then a DatetimeIndex, and
otherwise preserves input row order. IDs
are online labels, not semantic regime names; a separate model registry should
store the meaning and fitted model for each ID. Top-of-book queue measurements
are not duplicated as depth-level-1 measurements in auto primitive mode.

## Genetic programming

`gp/` evolves bounded expression trees from causal numeric columns in the
feature frame plus timestamp-aligned raw market primitives. The grammar includes protected `+`, `-`, `*`, `/`, signed
nonlinear transforms, lag/difference/finite-difference derivative, rolling
mean/std, and analytic partial derivatives with respect to a selected input.
Trees are capped by `gp.max_height`; operations are vectorized over the time
series and use trailing windows only.

`extrapolate_features()` requires an explicit chronological training prefix,
state labels, and separate targets. It runs one search per known latent state
and target, excludes `null`, and purges training labels whose target endpoints
reach the validation boundary. It scores candidates by chronological block IC
consistency, applies a tree-size penalty, and injects random trees to preserve
population diversity. Candidates must clear the train IC and block-sign gates;
a validation veto then rejects features with weak, collapsed, or reversed IC.
The validation set does not rank candidates. Selected features are pruned by
correlation rather than expression text alone, and their direction is saved in
`SelectedExpression` for downstream use. State-specific transformed columns
are masked outside their state. The serializable
`GeneticFeatureModel` stores expression trees as data, not executable source.

To run it through the strategy engine, set `gp.enabled: true` in
`settings.yaml`. The result exposes `result.gp.report` and
`result.gp.features`; the latter can be passed as inputs to the downstream
barrier classifier or timeout regression. The stored `result.gp.model` can
recreate those columns on the same causal feature/raw-primitive frame.

GP is opt-in (`gp.enabled: false` by default). When enabled in `strategy()`,
the chronological split purges training rows whose selected target timestamp
crosses into validation. Set `gp.target_columns` to the outcomes configured in
`outcome.forward_returns`; adjust population size, generations, state sample
minimums, and expression height to control search cost. GP selection does not
change OOD state definitions or the outcome labels.
