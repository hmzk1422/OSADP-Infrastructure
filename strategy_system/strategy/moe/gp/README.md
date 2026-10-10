# GP Feature-Search System: Design, Mechanics, Variables and Cost
OSADP-Core Version  : v1.0.0  
Author              : Himaska Lin  
version             : v1.0.0

> VERY IMPORTANT NOTE: As of now, we won't use custom-made GP as GPLearn fulfills all of our requirements.

Scope: `evolution.py`, `population.py`, `grammar.py`, `evaluation.py`, `feature_library.py`, plus the glue in `strategy.py` (`run_gp`).

How to read this document: the system is explained as numbered segments **in the order they execute**. Each segment says **What** it does (with the actual numbers from the code), **Why** it is built that way, and **Cost** (time and memory). Timings are **measured** on synthetic data unless marked *extrapolated* (see Section 7). Where a rationale is not stated in the code comments, it is my reading of the design and I say so.

---

## 0. One-paragraph summary

For every latent state that the OOD stage found (`latent_001`, `latent_002`, ...) and for every forward-return target (`future_return__30bar`, ...), the system runs a **separate genetic-programming (GP) search**. The search evolves small math expressions (trees such as `rolling_mean[8](x__queue_imbalance) - signed_sqrt(x__spread_bps)`) whose values correlate with the future return **only inside that state**. Candidates must clear a strict statistical gate on the training period, are ranked by walk-forward predictive performance across *recurring episodes* of the state, and must survive a held-out validation veto. The survivors (at most 2 per state-target pair) become new feature columns that are `NaN` outside their own state. The expressions are stored as data (JSON trees), never as executable code.

---

## 1. Big picture

### 1.1 Data flow

```
strategy.run_gp
   |  features (T x V), targets, latent_id Series, train_mask, train_end_time, purge_rows
   v
extrapolate_features (evolution.py)
 [2] validate arguments
 [3] build_gp_inputs      -> causal z-scored terminals x__*            (T x V')
 [4] validate train mask, choose targets, purge target overlap         -> train / validation masks
 [5] choose states         (top max_states by purged-train rows, null excluded)
 [6] search constants      (rng, multiple-testing t threshold)
 for each (state, target):                                             <- S x Q independent searches
 [7]   prune terminals     (|corr| top 64 -> ridge top 32)
 [8-9] build initial population (32 random trees)
 [10-11] generation loop (<= 8):
          screen -> pad with random trees -> score -> hall of fame -> breed next generation
 [12]  scoring a candidate: Spearman IC + block ICs + t-stat gate + purged walk-forward (4 models)
 [13]  selection: validation veto, de-duplication, keep <= 2
 [14] joint bundle check per (state, target)
 [15] GeneticFeatureModel + feature frame + report + feature library
 [16] feature_library.py: JSON persistence
```

### 1.2 File map

| File | Lines | Role |
|---|---|---|
| `grammar.py` | 400 | The expression language: `Node`, operators, safe interpreter (value-only and value-plus-derivative), random tree builder, path helpers, (de)serialisation |
| `population.py` | 166 | Population creation, screening filters, tournament selection, crossover, mutation |
| `evaluation.py` | 314 | Purged walk-forward evaluation, four numpy-only predictive models, feature pre-filters, fitness formula |
| `evolution.py` | 945 | The orchestrator `extrapolate_features`, the GP loop, input normaliser, purging, statistics, result classes |
| `feature_library.py` | 80 | Versioned JSON persistence of the selected features, grouped by latent ID |

`strategy.py` imports `GPEvolutionResult, extrapolate_features` from `.gp`. I was not given that module, so I assume it re-exports them from `evolution.py`.

### 1.3 Design principles (these explain most of the code)

1. **Causality everywhere.** Every operator looks backwards only (`shift`, trailing `rolling`). Targets are forward-looking, so training rows whose target window reaches into the validation period are removed (purging).
2. **Safe by construction.** No `eval()`/`exec`. Expressions are trees interpreted by a whitelist of operators with protected arithmetic.
3. **State-conditioned.** Each latent state gets its own search, its own terminal set, its own training rows, and its own fitness. A feature is only valid inside the state it was found in.
4. **Recurrence-aware validation.** The inner walk-forward test holds out whole *episodes* of a state (a run of consecutive rows with the same label), so it asks "does this work the next time the state comes back?".
5. **Validation never ranks.** The held-out block is only a veto. Ranking uses training data only.
6. **Bounded search space.** Tree height is capped (default 5), temporal and derivative operators cannot be nested, and the number of states is capped.

---

## 2. Notation and defaults

| Symbol | Meaning | Default / typical |
|---|---|---|
| T | rows in the frame | |
| V | numeric input columns in `features` | tens to hundreds |
| V' | terminals after `build_gp_inputs` (one per usable column) | about V |
| T_tr | rows in the training prefix (`train_end`) | 70% of T in `strategy.py` |
| n | training rows that belong to one state and have a finite target | |
| S | states searched | at most `max_states` = 10 |
| Q | target columns | all `future_return__*` unless `target_columns` is given |
| P | population size | 32 |
| G | max generations | 8 |
| H | max tree height | `max_height` = 5 |
| s | nodes in a tree | mean 4.8, median 3, max 27 at height 5 (measured on 20,000 random trees) |
| M | evaluation models | 4 (`ridge`, `ols`, `tree`, `neural_network`) |
| F | walk-forward folds | 4 (hard-coded in `test_feature_set` defaults) |
| Bk | blocks for block-wise IC | `block_count` = 5 |
| W | temporal window choices | (2, 4, 8, 16) |

---

## 3. Segments in execution order

### Segment 1. `strategy.py` → `run_gp()` (glue, not part of the GP package)

**What.** `run_gp` returns `None` unless `gp.enabled` is true. It then:
- builds the GP source with `build_gp_source(raw, feat_df, timestamp_col)` (causal features plus `raw__` primitives, with `future_*`, `latent`, `label` and similar excluded by name);
- computes `split_position = int(len(feat_df) * train_fraction)` with `train_fraction` default **0.70**, and `train_end_time = feat_df.index[split_position]` when the index is a `DatetimeIndex`;
- builds `train_mask = chronological_train_mask(feat_df, train_fraction, purge_rows)` with `purge_rows` default **0**;
- aligns the OOD regimes to the feature index and passes `latent_id` as the state labels;
- calls `extrapolate_features(source, targets, latent_id, train_mask, target_columns=..., train_end_time=..., purge_rows=..., **rest_of_gp_settings)`.

**Why.** Config handling lives here so `strategy()` stays a straight-line call sequence. The GP receives the whole targets frame but only uses `future_return__*` columns as objectives; its input selection forbids target-like names.

**Cost.** O(T) glue. Everything below is the real cost.

---

### Segment 2. `extrapolate_features`: argument and index validation (evolution.py)

**What.** About 30 explicit checks run first, for example:
- `features`, `targets` and `state_labels` must have **identical indexes**;
- a `DatetimeIndex` must have no `NaT` and be monotonically increasing;
- `max_height >= 2`, `population_size >= 4`, `generations >= 1`, `1 <= elite_count < population_size`;
- rates in [0, 1], `immigrant_fraction` in [0, 0.5), `block_count >= 2`, `1 <= min_valid_blocks <= block_count`, `multiple_testing_alpha` in (0, 1);
- model names are canonicalised (`"Neural Network"`, `"mlp"`, `"nn"` all become `neural_network`).

**Why.** A search can run for hours. Failing in the first millisecond on a bad parameter or a misaligned index is cheaper than discovering it afterwards. Index equality is also a correctness guard: a one-row misalignment between features and targets is a silent look-ahead bug.

**Cost.** O(T) for the index equality, `isna` and monotonic checks. Negligible.

---

### Segment 3. `build_gp_inputs`: causal terminals (evolution.py)

**What.** For every numeric column that passes the name filter:

1. **Name filter.** Excluded if the lowercase name contains any of `future_`, `target`, `label`, `barrier`, `latent`, `trade_allowed`, `embedding`, or contains `timestamp`, or ends in `_id`. If you pass explicit `input_columns`, forbidden names raise a `ValueError` instead of being dropped.
2. **Price levels become log returns.** `_is_price_level_name` returns true for names containing the tokens `price`, `close`, `mid` or `last`, or ending in `bid`/`ask`, *unless* the name also contains a token such as `return`, `qty`, `volume`, `depth`, `imbalance`, `flow`, `ratio`. Those columns become `log(value).diff()` and are named `x__log_return__<col>`; everything else is named `x__<col>`.
3. **Causal rolling z-score.**
   ```
   prior = values.shift(1)
   mean, std = rolling(500, min_periods=50) of prior   (ddof = 0)
   z = (value - mean) / std
   ```
   If the history is flat (`std <= 1e-12`): `z = 0` when the value equals the mean, `+8` or `-8` when it deviates. Result clipped to ±8 (`clip_z`).
4. Infinities become `NaN`; columns that are entirely `NaN` are dropped.

**Why.**
- Raw price levels are non-stationary; a rule discovered at one price level would not transfer. Log returns fix that.
- Z-scoring puts every terminal on a comparable scale, which matters because the grammar also contains absolute constants (±0.5, ±1, ±2) and operators like `log` and `power` that behave very differently at different magnitudes.
- `shift(1)` keeps the row out of its own baseline (no look-ahead). The 500/50/8 recipe is the same as the one used by the OOD encoder, so the whole pipeline normalises consistently.

**Cost.** Time O(T·V): two rolling operations per column, in a Python loop over columns. Memory about 3 to 6 transient T-length float64 arrays per column, and a final T×V' float64 frame (8·T·V' bytes). *Extrapolated:* T = 1.1M and V = 100 is about 0.9 GB for the result and perhaps 2 to 3 GB at peak. The frame is rebuilt a second time inside `GeneticFeatureModel.transform` (segment 15).

---

### Segment 4. Train mask, target choice and purging (evolution.py)

**What.**

1. `_validate_train_mask` requires a **chronological prefix**: once a `False` appears, no later `True` is allowed, and both train and validation rows must exist. `split` is the position of the first `False`.
2. `targets_to_use` is `target_columns` if given, otherwise every column starting with `future_return__`. Missing columns raise `KeyError`.
3. `split_boundary` is the training cut. If `train_end_time` is supplied with a `DatetimeIndex`, it is located with `searchsorted`.
4. `_purge_target_overlap` removes training rows whose **forward target reaches the validation period**. It chooses the first method that applies, per target:
   - **endpoint column** `future_timestamp__<h>` exists in `targets` → keep rows whose endpoint is before the cutoff time;
   - **bar horizon** label like `30bar` → keep rows where `position + 30 < split`;
   - **time horizon** label like `30s` → keep rows where `timestamp + horizon < cutoff`;
   - otherwise fall back to `purge_rows` (an embargo of that many rows before the split), or raise a `ValueError` if none can be derived.
   Rows at or after the cutoff are also dropped.
5. The result is three masks: `initial_train` (the raw prefix), `train` (the purged prefix, used for fitting) and `validation = ~initial_train`.

**Why.** A 30-bar forward return at row t uses prices up to t+30. If t is within 30 bars of the validation start, that training label already contains information about the validation period. Purging removes exactly those rows. Note that the **validation block is not purged at its start**; only the training side is. Purging only the training side is the usual convention.

**Cost.** O(T·Q) vectorised (datetime conversions dominate). Negligible next to the search.

---

### Segment 5. Choosing which states to search (evolution.py)

**What.** State labels are lower-cased; `NaN`, `"null"` and `""` are treated as unknown. Unknown is excluded unless `include_null=True` and enough null training rows exist. Each known state is counted by its **purged training rows**; the top `max_states` (10) by count (ties broken alphabetically) are searched. If none exist, a `ValueError` is raised.

**Why.** It bounds cost (S ≤ 10), searches the states with the most data first, and is deterministic. `null` is excluded because it is not a coherent regime: a feature "specific to null" would be specific to "whatever the OOD stage could not recognise".

**Cost.** The label normalisation is a Python-level loop over T, so O(T) with a larger constant; counting is O(T·S). Negligible compared with the search (but a pure-Python T-length loop at about 1 µs per row, so about 1 s at 1M rows).

---

### Segment 6. Search constants (evolution.py)

**What.**
- `windows` are validated positive integers (default (2, 4, 8, 16)).
- **One** `np.random.default_rng(seed=7)` is shared across *all* state-target pairs.
- The multiple-testing threshold is a Bonferroni-style normal-tail bound:
  ```
  search_tstat = sqrt( 2 * ln( 2 * tests / alpha ) ),   tests = population_size * generations
  ```
  With defaults, tests = 256, alpha = 0.05 gives **t = 4.30**. The effective gates are `max(min_train_tstat, 4.30)` and `max(min_validation_tstat, 4.30)`. The floors (2.0 and 1.96) therefore never bind at defaults.
- Target arrays are extracted once as float64: `target_values[column]`, Q arrays of length T.

**Why.** Each state-target search scores up to 256 candidates, so some will correlate by chance. A Bonferroni-style bound raises the bar in proportion to the number of tries. The shared RNG makes the whole run reproducible from one seed.

**Consequences.**
- Because the RNG is shared, adding or removing a state changes the random streams of every later pair.
- The family size is 256 per **pair**, not per run. With S·Q pairs, the run-level false-discovery rate is higher than the 5% suggests (see Section 8).
- What the gate means in practice (Spearman IC needed to reach t ≥ 4.30, assuming independent samples):

| n (state-train rows) | abs(IC) needed |
|---|---|
| 60 | 0.491 |
| 500 | 0.189 |
| 2,000 | 0.096 |
| 10,000 | 0.043 |

**Cost.** O(Q·T) memory for the target arrays. Everything else is O(1).

---

### Segment 7. Per-pair setup and terminal pruning (evolution.py + evaluation.py)

The loop `for state in states: for target in targets:` runs S·Q independent searches. For each:

**What.**
1. Masks: `train_mask_pair = train & state & isfinite(y)` and `validation_mask_pair = validation & state & isfinite(y)`. If the state has fewer than `required_samples` (60; `min_null_samples` for null) training rows, the pair is skipped.
2. `prefilter_features(primitive_frame, y, max_features=64)`: absolute Pearson correlation of each terminal with `y`, computed on this state's training rows only; keep the top 64.
3. `ridge_select_features(..., max_features=min(32, len(cheap)))`: median-impute and standardise, solve ridge with `alpha = 1` (`(X'X + alpha*I) beta = X'y`), rank terminals by `abs(beta)`, then walk down the ranking and skip any terminal whose correlation with an already-kept one is at least 0.98. The survivors (at most 32) become `search_variables`.
4. If nothing survives, the pair is skipped.

**Why.** Random trees pick terminals uniformly, so every useless terminal dilutes the search. Cutting V terminals to at most 32, using only this state's training rows (no leakage), raises the chance that a random tree contains something useful. The 0.98 rule prevents the set from being 32 near-copies of one signal.

**Trade-off.** A terminal that matters only through an interaction or non-linearly (low marginal correlation and low ridge weight) never reaches the GP. This is a real blind spot: the search can only discover what the linear pre-filters let through.

**Cost.** `prefilter_features` is O(V·n) (one `corrcoef` per column in a Python loop). `ridge_select_features` is O(n·V''² + V''³) with V'' ≤ 64, plus up to O(V''²·n) in the redundancy loop. Small compared with the search itself.

---

### Segment 8. The expression language (grammar.py)

This is the most reusable piece. It defines what a "feature" is.

#### 8.1 Data structure

```python
@dataclass(frozen=True)
class Node:
    op: str                       # "var", "const", an operator name
    children: tuple[Node, ...]    # () for terminals
    value: str | float | int | None
```
- `var` nodes store the terminal column name in `value`; `const` nodes store a float; temporal operators store their window (int) in `value`; `partial` stores the variable name it differentiates by.
- Nodes are **immutable**: crossover and mutation build new trees with `replace_at_path` and never modify the parent. This makes sharing sub-trees between individuals safe.
- `height` and `size` are recursive properties (recomputed on every access, O(size)); fine at at most 31 nodes.
- `str(node)` is a canonical readable string, e.g. `(a + b)`, `rolling_mean[8](x)`, `partial[x](...)`. **It is the identity key** used for de-duplication, the score cache and the hall of fame. Two trees with the same string are the same individual.
- `node_to_dict` / `node_from_dict` give a JSON-friendly form.

#### 8.2 Operators (20, plus the two terminal kinds `var` and `const`)

| Family | Operators | Arity | Notes |
|---|---|---|---|
| Arithmetic (binary) | `add`, `sub`, `mul`, `div`, `power` | 2 | `div` is protected: denominators with absolute value below 1e-8 become `NaN`. `power`: the exponent is **rounded and clipped to [-4, 4]**. |
| Pointwise (unary) | `neg`, `abs`, `log`, `sqrt`, `signed_log`, `signed_sqrt` | 1 | `log` uses `log(abs(x))` clipped at 1e-8; `sqrt` uses `sqrt(abs(x))`; the signed versions keep the sign. |
| Temporal | `lag`, `diff`, `derivative`, `time_derivative`, `rolling_mean`, `rolling_std`, `window`, `block` | 1 + window | All backward-looking. `window` is a rolling **sum**. `derivative` is `diff/lag`. `time_derivative` divides by elapsed seconds (needs a `DatetimeIndex`, else it assumes 1 per row). `rolling_std` uses `ddof=0`. Rolling operators need a **full window** (`min_periods = window`). |
| Calculus | `partial` | 1 + variable | `partial[v](child)` = the pointwise derivative of `child` with respect to terminal `v` (a "marginal effect" feature). |

Every result passes through `_bounded`: infinities become `NaN` and values are clipped to ±1e6.

`block[w]` is special: it takes the **expanding mean within non-overlapping blocks of w rows**, where blocks are defined by absolute row position (`arange(len(x)) // w`). That makes it **position-dependent** (see Section 8 on risks) and very slow (see 8.6).

#### 8.3 Structural rules (`validate_expression`)

- Tree height at most `max_depth`.
- Each node's child count must match its arity; `var` nodes need a name in the allowed terminal set; `const` values must be finite; temporal windows must be integers of at least 1.
- **Temporal and calculus operators cannot sit under another temporal or calculus operator.** So there is no `rolling_mean(lag(x))` and no `partial` inside a `rolling_std`. Arithmetic may sit above and below them (`rolling_mean[4](a*b) - lag[2](c)` is legal).
- `partial` cannot be nested inside another `partial` (`has_nested_partial`).

**Why.** These rules cap the search space and keep every operator count small and interpretable. They also prevent chains of transforms that would create enormous effective lookbacks (`rolling_mean[16](lag[16](rolling_std[16]...))`), which would silently increase warm-up and make features untestable.

#### 8.4 How a random tree is built (`random_tree`)

At each call with remaining depth d:
1. If `d <= 1`, or (grow mode and a 32% coin flip), return a **terminal**: 82% a variable chosen uniformly from the terminal set, 18% a constant from {-2, -1, -0.5, 0.5, 1, 2}.
2. Otherwise, with 42% probability: a **binary** operator chosen uniformly from five, with two recursive children at depth d-1.
3. Otherwise, if partials are allowed, with 12% probability: a `partial`. Its child is built with temporal operators and partials disallowed; the variable it differentiates must occur in the child, and the child must contain a non-linear operation (`mul`, `div`, `power`, `abs`, `log`, `sqrt`, signed variants, `rolling_std`), otherwise the plain child is returned (a derivative of a linear expression is a constant, which is useless).
4. Otherwise a **unary** operator is chosen uniformly from 14 (6 pointwise plus 8 temporal). If temporal/calculus, its child is built arithmetic-only (the restriction above) and a window is drawn uniformly from `temporal_windows`.

Measured over 20,000 random initial trees at height 5: mean size 4.8 nodes, median 3, max 27; **18% are a single terminal**; 48% contain a temporal operator; each temporal operator (including `block`) appears in about 7% of trees; 5.7% contain a `partial`; all were valid.

#### 8.5 Two interpreters

- `_evaluate_value`: fast, value-only, used by GP scoring. Recursion creates one pandas Series per node.
- `evaluate(node, inputs, wrt)`: returns **(value, tangent)**, i.e. **forward-mode automatic differentiation**. Used only for `partial` and `expression_partial`. Each pointwise and temporal operator has an analytic derivative rule (product rule for `mul`, quotient rule for `div`, exact population-variance derivative for `rolling_std`). A tree evaluated this way costs about 2 to 3 times a value-only pass.

#### 8.6 Cost per operator (measured)

Single-operator trees on 100,000 rows (best of three, pandas 3.0.2):

| Operator | Time | Operator | Time |
|---|---|---|---|
| `var` (convert only) | 0.7 ms | `lag[8]` | 0.9 ms |
| `add` / `sub` / `mul` / `div` | 2.4 to 2.7 ms | `diff[8]` | 1.9 ms |
| `power` | 9.2 ms | `derivative[8]` | 2.4 ms |
| `neg` / `abs` | 0.7 to 0.8 ms | `time_derivative[8]` | 6.7 ms |
| `log` / `sqrt` / signed variants | 1.7 to 2.5 ms | `rolling_mean[8]` / `window[8]` | 2.3 to 2.4 ms |
| `partial` of `mul(signed_log(a), b)` | 5.1 ms (vs 2.2 ms value-only child) | `rolling_std[8]` | 3.6 ms |
| **`block[16]`** | **892 ms** | **`block[8]`** | **1,999 ms** |
| **`block[2]`** | **7,573 ms** | | |

So an ordinary node costs about 20 ns per row. `block` costs **9 to 76 µs per row**, which is roughly 400 to 3,500 times more, because `groupby(...).transform(lambda ...)` makes one Python call per block (T/window calls at about 150 µs each). This is the single largest performance problem in the system (Section 6.2).

**Total cost of evaluating a tree** is O(s·T) for ordinary operators: a node costs about 20 ns per row, so a tree of 5 nodes on 700,000 rows is about 70 ms.

---

### Segment 9. Initial population (population.py → `initialize_population`)

**What.** Creates `population_size` (32) random trees. For individual `i`, the target depth is drawn uniformly from {2, ..., `max_height`} (when `max_height > 2`), and the tree is built in **grow** mode for even `i` and **full** mode for odd `i`. Terminal draws come only from `search_variables` (segment 7). `create_initial_population` and `create_random_expression` are thin wrappers that convert arguments to tuples.

**Why.** "Ramped half-and-half" initialisation is the classic GP recipe: mixing depths and mixing grow/full gives varied shape and size from generation zero. In grow mode a node may stop early (32% terminal chance); in full mode it never stops before the depth limit.

**Cost.** O(P·s). Negligible.

---

### Segment 10. Screening: cheap filters before expensive scoring (population.py + closure `screen_expression_population` in evolution.py)

This runs at the **start of every generation** (including generation 0), via the `population_filter` hook of `evolve_population`.

**What.** In order:

| Step | Function | What it removes | Evaluates trees? |
|---|---|---|---|
| a | `filter_invalid_features` | trees violating the grammar rules, terminals outside `search_variables`, height above `max_height` | no |
| b | `deduplicate_population` | repeated `str(node)` | no |
| c | `filter_constant_features` | trees with fewer than 2 distinct finite values on the state's training rows (also anything that raises an error) | **yes (1st)** |
| d | build `expression_frame` | (a table of every survivor's values on the state's training rows) | **yes (2nd)** |
| e | `prefilter_features` then `ridge_select_features` | keep top 64 by abs Pearson, then **at most 16** by ridge weight (alpha 1), dropping those at least 0.98 correlated with a better-ranked one | no |
| f | `filter_redundant_features` | trees at least 0.98 correlated (abs Pearson, at least 20 shared samples) with a kept tree | **yes (3rd)** |

Afterwards, in `evolve_population`, the population is **padded back to 32 with fresh random "grow" trees**. Those padding trees are scored immediately and bypass the screen that generation.

**Why.** Model fits (especially the neural network) are costly, so obviously useless candidates (constants, copies, nonsense) are removed before fitting. Everything is computed on the state's training rows only.

**What it implies (read this).**
- Step e is a hard truncation: **at most 16 of 32** survive the screen. Together with the padding, roughly **half of every generation can be new random trees** when 16 survive.
- Ranking in step e uses ridge coefficient magnitude, not IC and not fitness. It is a different selection pressure from the one in segment 12.
- Steps c, d and f each evaluate every tree on the **whole training prefix** (T_tr rows, not just the state's n rows), because rolling operators need contiguous history. So screening cost depends on T_tr, not on how large the state is. Nothing is cached between steps or between generations.

**Cost.** Per generation: 2 to 3 full-length evaluations per candidate (about 20 ns per row per node), plus O(P·n) correlations, ridge O(n·P² + P³), and redundancy O(P·16·n). In the profile, screening was the dominant segment (about 63 s of a 91 s profiled run), almost entirely because of `block` (Section 6.2).

---

### Segment 11. The generation loop (evolution.py → `evolve_population`, `evolve_generation`; population.py operators)

**What.** One generation of one state-target search:

1. Screen (segment 10), then pad to 32 with random trees.
2. Score every individual via `evaluator` (= `score_expression`, segment 12). Scores are cached by `str(node)`, so a repeated tree is not rescored.
3. Record every scored `(node, score)` in the **hall of fame** (a dict keyed by `str(node)`) and append the generation's best score to `history`.
4. Stop if this is the last generation or `stop_evolution(history, patience=3, min_improvement=1e-6)`.
5. Otherwise build the next generation with `evolve_generation`:
   - sort by score, copy the top `elite_count` (**2**) unchanged;
   - reserve `round(P × immigrant_fraction)` slots for immigrants. `extrapolate_features` passes `max(immigrant_fraction, 1/P)`, which at defaults is `max(0.10, 1/32)` = 0.10, so **3 immigrants**;
   - fill the remaining **27** slots by breeding: pick parent 1 by tournament; with probability **0.70** pick parent 2 by tournament and `crossover`, otherwise the child is parent 1 unchanged; then with probability **0.30**, independently, `mutate` the child;
   - append the 3 random immigrants.

**The genetic operators.**
- `tournament_select`: sample `tournament_size` (4) individuals **without replacement**, return the highest score (ties go to the first drawn).
- `crossover`: choose a **uniformly random node** in each parent (root, internal nodes and leaves all equally likely, so leaf-heavy), and replace the first parent's chosen subtree with the second's. The child is accepted only if its height is at most `max_depth`, it has no nested `partial`, and it passes `validate_expression`. Up to **12 attempts**; if all fail, the first parent is returned unchanged.
- `mutate`: shuffle all node paths; for each path with at least 1 level of depth left (`max_depth - len(path) >= 1`), replace that subtree with a new random "grow" tree no taller than the remaining depth; return the first child that validates; if none do, return the original unchanged. Mutation can propose illegal combinations (a temporal operator under a temporal parent); validation simply rejects them and the next path is tried (rejection sampling).
- `stop_evolution`: only active once `len(history) > patience` (so from the 4th generation). It stops if the **latest best score is not finite** (`-inf`), or if the best of the last 3 generations did not beat the best of earlier ones by more than 1e-6.

**Why.** This is a standard elitist GP: elites keep the best, tournaments apply selection pressure, crossover recombines, mutation explores, immigrants keep diversity. Rejection sampling keeps all operators simple while the validator remains the single source of truth for legality.

**What it implies.**
- A failed crossover returns a clone of parent 1, and a failed mutation returns the original. Clones are removed by the next screen and replaced by random trees, so failures cost diversity less than they might.
- If **no candidate passes the segment 12 gate** in the first four generations, every score is `-inf`, the latest best is non-finite, and the search stops at generation 4. A pure-noise state therefore costs about half of a full 8-generation search.

**Cost.** At most `G × P` = 256 scorings per pair (fewer because of the cache and early stop). The operators themselves are O(s) per call and negligible.

---

### Segment 12. Scoring one candidate (evolution.py → `score_expression`; evaluation.py)

This is where the system decides whether an expression is any good. Steps run in order and the expensive ones are gated behind the cheap ones.

**12.1 Cheap statistics (every candidate).**

1. Look up `str(node)` in `score_cache`.
2. Evaluate the tree on `train_inputs` (all T_tr rows). Any numerical exception turns the values into all `NaN`.
3. **Train IC**: Spearman rank correlation between the expression and the target on the state's training rows (`_spearman`, needs at least `min_state_samples` = 60 finite pairs). Ranking makes it robust to outliers and monotonic transformations.
4. **Block ICs** (`_blockwise_spearman`): split the training prefix into `block_count` (5) contiguous chronological blocks; in each block with at least `min_block_samples` (20) state rows, compute the Spearman IC. Result: an array of valid block ICs, from which `block_mean` and `block_std` are taken.
5. **Sign agreement**: all block ICs have the same sign as the first, and the same sign as the train IC.
6. **Gate** (`eligible`), all of:
   - train IC finite and its t-statistic is at least `required_train_tstat` (4.30 at defaults), where `t = |ic| * sqrt((n-2)/(1-ic²))`;
   - at least `min_valid_blocks` (3) valid blocks;
   - `abs(block_mean) >= min_train_ic` (0.03);
   - signs agree (when `require_block_sign_agreement` is true).

If not eligible, the fitness is `-inf` and the candidate stops here (this is the common case).

**12.2 Walk-forward over recurring episodes (eligible candidates only).** For each of the 4 models, `test_feature_set` calls `run_purged_walk_forward`, configured by the GP as follows.

- Input: a one-column feature frame (the candidate), a target masked to the state's training rows (`NaN` elsewhere), the state labels of the training prefix and the state id.
- A row is usable if the target and feature are finite and the label equals the state.
- **Episodes**: `episode_ids = cumsum(label changes)`, so each maximal run of the same label is an episode. Episodes are ordered in time.
- **Folds**: the test episodes are the **last `folds` (4) episodes after dropping the first**. Each one is a fold. Its training set is **all earlier episodes of that state**, with an extra embargo of `purge_rows` rows before the test start (`purge_rows` is the supplied value or inferred from the target label: `30bar` gives 30; a time label like `30s` is divided by the median row cadence; otherwise 1).
- A fold needs at least `min_samples = max(20, required_samples // 5)` = 20 training rows.
- Per fold: fit the model, predict the held-out episode, then compute
  - `prediction_error` = RMSE;
  - `train_mean` = mean target on the fold's training rows, `signal = sign(prediction - train_mean)`;
  - `directional_accuracy` = fraction of rows where `signal == sign(actual - train_mean)`;
  - `expectancy` = mean of `signal * (actual - train_mean)`;
  - `target_scale` = standard deviation of the target on the fold's training rows.
- Aggregate: mean error, accuracy and expectancy across folds;
  `stability = clip(1 - std(expectancy across folds) / mean(target_scale), 0, 1)`;
  `recurrence_consistency` = share of folds sharing the dominant expectancy sign.
- If the state has fewer than 2 usable episodes (or fewer than `min_samples` rows), all metrics are `NaN` and the fitness becomes `-inf`.

**Why.** It answers "if I had learned this relationship from earlier visits to the state, would it have worked on the next visit?", which is the question that matters for a regime strategy. Holding out whole episodes avoids the leakage you get from randomly splitting autocorrelated rows from the same visit.

**12.3 Fitness** (`calculate_fitness`), for one model's result:

```
fitness = (expectancy / scale) * stability  -  0.05 * (error / scale)  -  0.0005 * ln(1 + tree_size)
```
(`scale` = `target_scale`; non-finite inputs or a non-positive scale give `-inf`.)

Per-model scores are averaged **within a family** (`ridge` and `ols` form the family "linear"), then **across families** (linear, tree, neural_network). So the linear family counts once in total, not twice.

Magnitudes matter: an expectancy of 0.1 target-standard-deviations gives about 0.1; the error term is about -0.05 for any candidate whose RMSE is near the target scale (almost all of them); a 10-node tree is penalised only 0.0005·ln(11) = 0.0012. The complexity penalty therefore barely discriminates; tree size is controlled mainly by `max_height`.

**12.4 The four models** (`fit_predictive_model`, numpy only). All first drop non-finite rows, impute with the train median, standardise with train mean and standard deviation, and centre the target.

| Model | Details |
|---|---|
| `ridge` | closed form `(X'X + alpha*I) b = X'y`, `alpha` = 1 |
| `ols` | `lstsq` with an intercept column |
| `tree` | custom regression tree, `max_depth` 3, `min_samples_leaf` 8 (a node splits only if it has at least 16 rows); one candidate split per column, at the **median**; chooses the column with the lowest squared error |
| `neural_network` | 1 hidden layer, **16 tanh units**, 120 epochs of **full-batch** gradient descent, `learning_rate` 0.01, weight seed 0 (output weights start at 0) |

With a single input column (the GP case), ridge/OLS fit a line, the tree fits at most 8 steps, and the network fits a smooth curve. So the four models test whether the same feature is useful linearly versus non-linearly. Parameters can be overridden through `evaluation_model_params`.

**Cost.**
- Ungated (every candidate): one evaluation O(s·T_tr) plus 6 Spearman computations (1 train + 5 blocks), each O(n log n) because of ranking.
- Gated (eligible candidates only): `M × F` fits. Measured on one feature column:

| Training rows | ridge | ols | tree | neural_network | one Spearman |
|---|---|---|---|---|---|
| 5,000 | 1 ms | 1 ms | 2 ms | **147 ms** | 2 ms |
| 20,000 | 1 ms | 2 ms | 6 ms | **611 ms** | 5 ms |
| 100,000 | 5 ms | 7 ms | 28 ms | **3,301 ms** | 26 ms |

The neural network costs about **30 µs per training row per fit**, about 100 times ridge. A 4-fold walk-forward with the network on a 20,000-row state took 1.6 s. This is why the gate matters for cost: it keeps the network off most candidates.

---

### Segment 13. Selection: validation veto and de-duplication (evolution.py)

**What.** After evolution, the hall of fame (every candidate ever scored for this pair, `-inf` ones included) is sorted by fitness, best first. For each candidate until `max_features_per_state_target` (2) have been accepted:

1. Skip if its fitness is not finite or its string was already accepted.
2. Evaluate it on the **full** input frame (train and validation; this is safe because the expression is causal).
3. Compute the **validation IC** on the held-out rows of this state.
4. **Validation veto**: all of
   - finite, and t-statistic at least `required_validation_tstat` (4.30);
   - `abs(validation_ic) >= min_validation_ic` (0.02);
   - `abs(validation_ic) >= validation_ic_retention * abs(train_ic)` (at least 25% of the train IC survives);
   - same sign as the train IC.
5. **De-duplicate** against already accepted features of this pair: abs Pearson correlation on the training rows at least `correlation_threshold` (0.98) means skip.
6. Re-run the 4-model walk-forward on the training prefix to collect `model_metrics` (per model: prediction error, directional accuracy, expectancy, stability, episode expectancy standard deviation, recurrence consistency, episode count) and recompute the fitness with the same family averaging.
7. Name it `gp__{state}__{target without 'future_return__'}__{k:02d}`, with `direction = sign(train_ic)`.

**Why.** The held-out block is used only as a **veto**, never to rank, so it cannot be overfitted by search (apart from the multiple-looks effect discussed in Section 8). De-duplication keeps the two accepted features from being the same signal twice.

**Cost.** Up to |hall of fame| (at most 256) evaluations over T rows, but the loop usually ends after 2 acceptances. Each accepted candidate repeats the M×F model fits from segment 12 (the work is recomputed, not reused).

---

### Segment 14. Joint feature-set check (evolution.py)

**What.** For each (state, target) pair that has selected features, the selected expressions are evaluated on the training prefix, stacked into a small frame, and the 4-model walk-forward is run on **all columns together**. The fitness uses the **sum of node counts** as complexity. Every member of the pair then receives the **same** `fitness` (the set fitness) and the `feature_set_*` metrics are added to its `model_metrics` and its report row.

**Why.** Single-feature scores cannot reveal redundancy or interaction. A set that fits no better than its best member exposes itself here.

**Note.** This overwrites each member's individual fitness with the set fitness, so the `fitness` column in the report (and library) is a pair-level number, not a per-feature one.

**Cost.** M × F fits with at most 2 columns per pair; the NN again dominates.

---

### Segment 15. Building the outputs (evolution.py)

**What.**
- `GeneticFeatureModel(selected, input_columns, normalization_window, normalization_min_periods, clip_z)` is created. It stores the trees and the normalisation settings (everything needed to reproduce the features), and can be serialised with `to_dict` / `from_dict` (**no executable code**).
- `model.transform(features, state_labels)`:
  1. checks that the two indexes are identical;
  2. **rebuilds `build_gp_inputs`** on the frame (the T×V' normalisation, again);
  3. evaluates each selected expression over the whole frame;
  4. masks each result with `.where(state == item.state_id)`, so the column is `NaN` outside the state it was found in.
- The feature frame has columns `gp__<state>__<horizon>__<NN>`.
- The `report` frame has one row per selected feature (columns listed in Section 4), is sorted by state, target and descending fitness, and filtered by `select_validated_features(min_validation_ic)`.
- `build_feature_library(...)` (segment 16) is called with metadata, and written to disk if `feature_library_path` is set.
- Returns `GPEvolutionResult(features, report, model, feature_library)`.

**Why.** Masking by state makes each feature a **conditional** signal: downstream models see it only where it was validated, and a missing value carries the meaning "not in this state".

**Cost.** O(T·V') for the input rebuild plus O(s·T) per selected feature. Memory: the T×V' inputs plus `selected × T × 8` bytes for the output. In `strategy.py`, `gp_result.model` is also handed to `build_model_frame`; if that function calls `transform` (I have not seen its code), the inputs are rebuilt a third time.

---

### Segment 16. Persistence (feature_library.py)

**What.** A versioned JSON structure (schema 1) of the selected features grouped by latent ID:

```json
{
  "schema_version": 1,
  "metadata": {"source_input_columns": null, "encoded_terminal_columns": ["x__..."],
               "normalization_window": 500, "normalization_min_periods": 50, "clip_z": 8.0},
  "regimes": {
    "latent_001": [
      {"target_column": "future_return__30bar", "output_column": "gp__latent_001__30bar__00",
       "expression": {"op": "...", "value": null, "children": [...]},
       "train_ic": 0.23, "validation_ic": 0.23, "fitness": 0.14, "direction": 1,
       "n_train": 1234, "n_validation": 456, "model_metrics": {...}}
    ]
  }
}
```

| Function | Behaviour |
|---|---|
| `build_feature_library(selected, metadata=...)` | groups selected expressions by `str(state_id)` |
| `select_validated_features(report, min_fitness, min_validation_ic)` | filters the report: `fitness >= min_fitness` and `abs(validation_ic) >= min_validation_ic` |
| `save_feature_library(library, path)` | creates parent folders, writes indented JSON; a `Node` (a dataclass) is converted to a dict by `_json_default` |
| `load_feature_library(path)` | **raises** on a schema-version mismatch; converts each `expression` dict back to a `Node` |
| `get_features_for_regime(library, id)` | returns a copy of that state's records |
| `register_regime_features(library, id, features, replace=True)` | **copy-on-write**: returns a new library; `replace=False` appends |

**Why.** Trees are stored as data and interpreted by the whitelist, so loading a library cannot run arbitrary code, and the schema version allows safe evolution. Copy-on-write avoids accidental mutation of a shared library object.

**Important.** The library is keyed by **latent ID strings** such as `latent_001`. Those IDs are assigned in the order states are promoted by the OOD stage, so they only mean something together with the OOD prototype set (`state_dict()`) that produced them. A library loaded against a different OOD run would attach features to the wrong regimes without any error. `load_feature_library` does not re-validate the trees; an unknown operator would only fail at evaluation time.

**Cost.** O(total nodes) to save or load.

---

## 4. Data structures and naming conventions

### 4.1 Classes

| Class | Fields | Meaning |
|---|---|---|
| `Node` (frozen dataclass) | `op`, `children`, `value` | one node of an expression tree |
| `SelectedExpression` (frozen dataclass) | `state_id`, `target_column`, `output_column`, `expression`, `train_ic`, `validation_ic`, `fitness`, `n_train`, `n_validation`, `direction`, `model_metrics` | one accepted feature |
| `GeneticFeatureModel` | `selected`, `input_columns`, `normalization_window`, `normalization_min_periods`, `clip_z` | everything required to recompute the features on new data; has `transform`, `to_dict`, `from_dict` |
| `GPEvolutionResult` (frozen dataclass) | `features`, `report`, `model`, `feature_library` | what `extrapolate_features` returns |

### 4.2 Naming

| Pattern | Example | Meaning |
|---|---|---|
| `x__<col>` | `x__queue_imbalance` | z-scored terminal |
| `x__log_return__<col>` | `x__log_return__mid_price` | price-level column converted to a log return, then z-scored |
| `gp__<state>__<horizon>__<NN>` | `gp__latent_001__30bar__00` | output feature for a state and target; `NN` is the acceptance order |
| `future_return__<h>` | `future_return__30bar` | forward-return target; `<h>` is parsed as the horizon |
| `future_timestamp__<h>` | `future_timestamp__30bar` | optional endpoint column used for exact purging |

### 4.3 `report` columns

`state_id`, `target_column`, `output_column`, `expression`, `tree_height`, `tree_size`, `train_ic`, `validation_ic`, `fitness`, `block_ic_mean`, `block_ic_std`, `n_train_blocks`, `direction`, `n_train`, `n_validation`, `models_evaluated`, then per model `<model>_prediction_error`, `_directional_accuracy`, `_expectancy`, `_stability`, `_episode_expectancy_std`, `_recurrence_consistency`, `_episode_count`, and the pair-level `feature_set_fitness` and `feature_set_<model>_<metric>`.

---

## 5. Complexity summary

### 5.1 Overall

```
Total  =  O(T·V)                                              building inputs (once; again in transform)
        + Σ over S·Q pairs of:
              O(n·V) + O(n·V''^2)                             terminal pruning            (V'' <= 64)
            + Σ over <= G generations of:
                  P · s · T_tr · (3 to 4)                     tree evaluations (screen + score)
                + P · n · log n · (1 + Bk)                    Spearman, train + block ICs
                + P_eligible · M · F · fit(n)                 walk-forward model fits
            + |hall of fame|·(s·T + n log n)  +  M·F·fit(n)   selection and bundle check
        + O(T·V + selected·s·T)                               final transform
```
The number of tree evaluations is bounded by about `S × Q × G × P × 4` = roughly **10,000 × Q** full-length evaluations at defaults (S = 10, G = 8, P = 32). Real runs are smaller because of the score cache, early stopping and skipped pairs.

### 5.2 Per segment

| # | Segment | Time | Memory |
|---|---|---|---|
| 2 | Validation | O(T) | O(1) |
| 3 | `build_gp_inputs` | O(T·V) | about 3 to 6 transient T-arrays per column + 8·T·V' result |
| 4 | Purge | O(T·Q) | a few T-length masks |
| 5 | State choice | O(T·S), Python loop over T | O(T) |
| 6 | Constants | O(1) | 8·T·Q targets |
| 7 | Terminal pruning | O(n·V + n·V''²) per pair | O(n·V'') |
| 9 | Initial population | O(P·s) | O(P·s) |
| 10 | Screening | 3 evaluations per candidate per generation + O(P·n·16) | n × up to 32 table |
| 11 | Generation loop | operators O(s); at most G·P scorings | hall of fame: at most 256 nodes |
| 12 | Scoring | evaluation O(s·T_tr) + 6 × O(n log n); if eligible M·F fits (NN about 30 µs per training row per fit) | a few T_tr-length arrays |
| 13 | Selection | at most 256 evaluations over T (usually few) + M·F fits per accepted | small |
| 14 | Bundle | M·F fits, at most 2 columns | small |
| 15 | Outputs | O(T·V' + s·T per feature) | 8·T·V' + 8·T·selected |
| 16 | Library | O(nodes) | O(nodes) |

### 5.3 What the cost scales with (measured, no `block` operator)

| Change | Observed |
|---|---|
| T = 20,000 → 40,000 → 80,000 (V = 20, 3 states, 1 target) | 7.7 s → 11.9 s → 19.7 s (**about linear in T**, about 3.5 s fixed overhead) |
| V = 20 → 60 at T = 40,000 | 11.9 s → 10.8 s (**V barely matters**, because terminals are pruned to at most 32 per pair) |
| Peak RSS | 94 MB (T = 20k) → 114 MB → 152 MB (T = 80k); 169 MB at V = 60 |
| States | cost is roughly linear in the number of **pairs that pass the sample check**; a state with no gate-passing candidate is stopped by the code's rule after generation 4 (4 of 8 generations, so roughly half the cost; inferred from the stop rule, not separately timed) |

---

## 6. Performance hotspots and what controls them

### 6.1 Repeated, uncached tree evaluation

The same tree is evaluated on the same training rows 3 to 4 times per generation (constant filter, expression frame, redundancy filter, scoring), and again in selection, the bundle check, and `transform`. Only *scores* are cached. Caching evaluated Series by `str(node)` for the duration of a pair would remove most of this (my estimate: a 2 to 3 times reduction in evaluation time; not measured).

### 6.2 The `block` operator is the dominant cost when it appears

- Measured: 892 to 7,573 ms per evaluation at 100,000 rows, against about 2 ms for ordinary operators (Segment 8.6).
- In a `cProfile` run on 20,000 rows and 3 states (profiling overhead inflates absolute seconds), `groupby.transform` took **78 s of 91.5 s** (about 85%), with **226,625 lambda calls** across 93 `block` evaluations.
- End-to-end, runs where `block` appeared were **5.0 times** (T = 20k: 38.8 s versus 7.7 s) and **9.4 times** (T = 40k: 112.4 s versus 11.9 s) slower than the same runs with `block` removed. The effect varies with how often `block` is sampled, since each of the 8 temporal operators appears in about 7% of random trees.
- A vectorised replacement computes the same thing:
  ```python
  groups = np.arange(len(x)) // window
  cs  = x.fillna(0).groupby(groups).cumsum()
  cnt = x.notna().astype(float).groupby(groups).cumsum()
  result = _bounded(cs / cnt.where(cnt > 0))
  ```
  I tested it against the existing lambda version on 100,000 rows with NaNs: maximum absolute difference **at most 2.2e-16**, identical NaN pattern, and **111 to 505 times faster** (for windows 16 to 2). The same change would apply in `evaluate()` (the tangent path uses the same pattern for `dx`).

### 6.3 Neural-network fits

About 30 µs per training row per fit. They only run for candidates that pass the gate, but when many pass (a state with a strong signal) they dominate. If cost is a concern, lower `epochs` or `hidden_units` through `evaluation_model_params`, or drop the network from `evaluation_models` (note that this changes the fitness definition).

### 6.4 Memory

Dominated by the T×V' float64 inputs (8·T·V' bytes), rebuilt in `transform` while the earlier copy may still be alive. At T = 1.1M and V' = 100 that is about 0.9 GB per copy (*extrapolated*), plus per-evaluation Series of 8·T_tr bytes (about 6 MB at 770,000 rows), only a few alive at a time.

### 6.5 Minor

Python loops over columns (`build_gp_inputs`) and over T-length label arrays (state normalisation). Each is O(T) but with a Python-level constant.

---

## 7. Measured benchmarks

**Method.** The five uploaded files were run unmodified (except where stated) with synthetic data:
- `T` rows at one-second cadence on a `DatetimeIndex`; `V` terminals (one price-level column plus V-1 smoothed noise columns);
- 4 repeating state blocks of 800 rows (`latent_001`, `latent_002`, `latent_003`, `null`), so every state has many episodes;
- one target `future_return__30bar`, noise plus a planted signal (a rolling z-score of one feature) present only in `latent_001`;
- 70/30 chronological split, `purge_rows=30`, default search parameters (P = 32, G = 8, H = 5, 4 models, seed 7).

The search found the planted feature in `latent_001` (`signed_sqrt(x__feat_0)`, train IC 0.226, validation IC 0.230) and found nothing in the other two states.

**Caveats.** Synthetic data, one CPU, pandas 3.0.2, tqdm disabled. Trust the **scaling shapes** (linear in T, 5 to 9 times from `block`, NN about 100 times ridge) more than absolute seconds. Random-tree composition (and so the cost) varies with the seed.

**Results.**

| Experiment | Result |
|---|---|
| Full run, T = 20k, V = 20 | 38.8 s (with `block` allowed), 7.7 s (without) |
| Full run, T = 40k, V = 20 | 112.4 s (with), 11.9 s (without) |
| Full run, T = 80k, V = 20 | 19.7 s (without) |
| Full run, T = 40k, V = 60 | 10.8 s (without) |
| Per-operator cost | table in Segment 8.6 |
| Model fits and Spearman | table in segment 12 |
| Random tree sizes (20,000 trees, H = 5) | mean 4.8, median 3, max 27 nodes; 18% single terminals; 48% with a temporal operator |
| Vectorised `block` | 111 to 505 times faster, equal output |

**Illustrative extrapolation (not measured).** Without `block`, the three-pair runs scaled at about 0.2 ms per input row, roughly 65 µs per row per state-target pair. At T = 1.1M and 30 pairs (10 states × 3 targets) that is on the order of **35 to 40 minutes**, before any extra cost from neural-network fits in strong states. With `block` the same arithmetic would be several times larger. Treat this as an order of magnitude only.

---

## 8. Behaviours worth knowing

These are properties of the code as written, not accusations; several are deliberate trade-offs.

1. **`block` is position-dependent.** Block boundaries come from `arange(len(x)) // window`, i.e. from the row's position in whatever frame is evaluated. Evaluating the same tree on a slice that starts at a different row (a rolling live buffer, a chunk) gives different values, so a feature using `block` can differ between research and production. Rolling operators do not have this problem.
2. **The fitness landscape is a plateau with islands.** Every candidate that fails the gate scores `-inf`, so until a candidate passes, selection has nothing to climb: parent choice among `-inf` ties is arbitrary and the search is effectively random sampling. This is also why early stopping at generation 4 is common in states with no signal.
3. **Heavy truncation in screening.** At most 16 of 32 survive the ridge step, and padding with random trees follows (segment 10). Combined with elitism of 2 and 3 immigrants, a large share of each generation is new random material.
4. **The t-statistic assumes independent samples.** Forward returns over a horizon of h bars overlap, and many features are autocorrelated, so the effective sample size can be as low as roughly n/h. The reported t can then overstate significance by up to about `sqrt(h)`. For 30-bar targets, a nominal t of 4.3 could correspond to about 0.8 under fully overlapping samples (worst case). Purging protects the train/validation boundary; it does not fix this.
5. **Multiple-testing correction is per pair, not per run.** The Bonferroni family is 256 candidates for one state-target pair. A run with S·Q pairs makes S·Q such families, so the run-level false-positive expectation is larger than the nominal 5%.
6. **`stability` is trivially 1 with one fold.** The standard deviation of a single expectancy is 0. The number of folds is `min(4, usable_episodes - 1)`; states with just two episodes get one fold and a perfect stability score.
7. **The complexity penalty is weak** (0.0012 for 10 nodes against a signal term of about 0.1), so parsimony is enforced mainly by `max_height`.
8. **Run order affects results.** One RNG is shared across all pairs, so changing which states are present changes later random streams.
9. **Pre-filter blind spot.** Terminals must show marginal correlation or ridge weight (segment 7) to be offered to the GP at all; features that only matter through interaction can be dropped.
10. **Validation is consulted repeatedly.** The held-out block is checked for each hall-of-fame candidate in fitness order until 2 pass. It is only a veto and uses the same 4.30 t threshold, but it is still multiple looks at one holdout.
11. **State IDs are not portable.** See segment 16: features and libraries are only meaningful with the OOD prototypes that created the IDs.
12. **Warm-up NaNs.** A feature is `NaN` for roughly the first 50 rows (z-score minimum periods) plus the lookback of its widest window (up to 16 rows), and `NaN` outside its state by design.

---

## 9. Parameter reference: `extrapolate_features`

### 9.1 Data and split

| Parameter | Default | Meaning |
|---|---|---|
| `features` | | frame of candidate columns (T×V); passes through `build_gp_inputs` |
| `targets` | | frame with `future_return__*` columns (and optionally `future_timestamp__*`) |
| `state_labels` | | Series of latent IDs, same index as `features` |
| `train_mask` | | chronological-prefix boolean mask; the rest is validation |
| `target_columns` | all `future_return__*` | restrict which targets are searched (cost is linear in Q) |
| `input_columns` | auto | explicit terminal columns; forbidden names raise |
| `train_end_time` | None | exact split time; used for purging and the split boundary |
| `purge_rows` | None | embargo in rows; fallback when horizons cannot be inferred |

### 9.2 Normalisation

| Parameter | Default | Meaning |
|---|---|---|
| `normalization_window` | 500 | trailing window for the terminal z-score |
| `normalization_min_periods` | 50 | rows required before a z-score exists |
| `clip_z` | 8.0 | z-score clip |

### 9.3 Search shape

| Parameter | Default | Effect on behaviour and cost |
|---|---|---|
| `max_height` | 5 | maximum tree height; the main cap on expression size (at most 31 nodes) and on cost per evaluation |
| `population_size` | 32 | candidates per generation; cost linear in P; also sets the multiple-testing family |
| `generations` | 8 | upper bound; early stop usually triggers first in noise states |
| `tournament_size` | 4 | selection pressure |
| `elite_count` | 2 | copied unchanged each generation |
| `crossover_rate` / `mutation_rate` | 0.70 / 0.30 | independent probabilities |
| `immigrant_fraction` | 0.10 | random newcomers per generation (3 at P = 32) |
| `temporal_windows` | (2, 4, 8, 16) | windows available to temporal operators; longer windows lengthen warm-up |
| `max_states` | 10 | states searched, by training-row count |
| `max_features_per_state_target` | 2 | accepted features per pair |
| `include_null` / `min_null_samples` | False / 60 | also search the null state |
| `seed` | 7 | the single shared RNG |

### 9.4 Statistics and gating

| Parameter | Default | Meaning |
|---|---|---|
| `min_state_samples` | 60 | training rows a state needs; minimum pairs for a Spearman IC |
| `block_count` | 5 | chronological blocks in the train IC stability check |
| `min_block_samples` | 20 | rows a block needs to count |
| `min_valid_blocks` | 3 | valid blocks required |
| `require_block_sign_agreement` | True | all block ICs and the train IC must share a sign |
| `block_std_penalty` | 0.5 | validated but **not used** in the code shown (no reference in the scoring path) |
| `min_train_ic` | 0.03 | floor on abs(mean block IC) |
| `min_validation_ic` | 0.02 | floor on abs(validation IC) |
| `min_train_tstat` / `min_validation_tstat` | 2.0 / 1.96 | floors; the multiple-testing threshold (4.30 at defaults) normally dominates |
| `multiple_testing_alpha` | 0.05 | family-wise level for the Bonferroni-style bound |
| `validation_ic_retention` | 0.25 | share of the train IC the validation IC must keep |
| `correlation_threshold` | 0.98 | de-duplication and redundancy threshold |
| `complexity_penalty` | 0.0005 | coefficient in the fitness complexity term |

### 9.5 Evaluation models and output

| Parameter | Default | Meaning |
|---|---|---|
| `evaluation_models` | ridge, ols, tree, neural_network | models used in the walk-forward fitness; the network is the cost driver |
| `evaluation_model_params` | None | per-model overrides, e.g. `{"neural_network": {"epochs": 30, "hidden_units": 8}}` |
| `feature_library_path` | None | if set, the JSON library is written there |

### 9.6 Minimal call

```python
from strategy_system.strategy.moe.gp import extrapolate_features  # as imported by strategy.py

result = extrapolate_features(
    features,                # T x V DataFrame (DatetimeIndex, increasing)
    targets,                 # same index; columns like "future_return__30bar"
    latent_id,               # Series of state labels, same index
    train_mask,              # boolean chronological prefix
    target_columns=["future_return__30bar"],
    train_end_time=features.index[int(len(features) * 0.70)],
    purge_rows=30,
    feature_library_path="gp_library.json",
)
result.report          # one row per accepted feature
result.features        # gp__<state>__<horizon>__<NN> columns, NaN outside their state
result.model.transform(new_frame, new_state_labels)   # recompute on new data
```