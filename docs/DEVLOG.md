# Developer Log
Documenting architectural planning and engineering hurdles.

# Unified Backtest/Live Architecture System
Reference: feature/unified-trading-system
```
          +---------------------------------+
          |        Trading Strategy         |
          |       (Agnostic Core Logic)     |
          +---------------------------------+
                           │ model.txt
                           ▼
          +---------------------------------+
          |         Execution Layer         |
          |     (Shared Engine / Core)      |
          +---------------------------------+
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    +---------------+             +---------------+
    | Histbacktest  |             |Live Execution |
    | (Parquet/CSV) |             | (WebSockets)  |
    +---------------+             +---------------+
```
Two systems, one strategy. Using shared architecture prevents inconsistency between backtest and live executions.
- **Trading Strategy**: Purely for defining the strategy code. 
    - ml_state_features.py: Computing the features, completely separate to ensure with very high confidence no bleeding or lookahead bias happens. 
    - ml_state_optimized.py: Composing features together with market such as calculating triple barrier or model training.
- **Execution Layer**: ml_state_runtime.py connected with ml_state_optimized as abstraction for true execution, querying data, providing commands, etc. Connects to txt file obtained from trading strategy.
- **Histbacktest/Live Execution**: Execution layer connects with histbackteest and live execution to query for data. 

Note: Should latency matter, migrate execution layer and histbacktest/live execution into rust.

## Checklist of Items

### 1. `unified-execution/engine/src/lib.rs` — Feature vector completed
- **Line 700**: Added 4 missing orderflow features to the 15-element feature array:
  - `m.tick_autocorr`, `m.inst_ratio`, `m.delta_divergence_z`, `m.depth_imbalance_z`
  - Previously the array had only 11 of 15 initializers; Rust zero-filled the rest, giving the model garbage features

### 2. `unified-execution/engine/src/lib.rs` — Orderflow config wired
- **Module constant** `USE_ORDERFLOW_FEATURES: bool = true` (lines 67/81)
- **StrategyConfig field** `use_orderflow_features: bool` (line 129)
- **Config initialization** passes `use_orderflow_features: USE_ORDERFLOW_FEATURES` (line 157)
- **Feature gating** at line 1190: `if self.cfg.use_orderflow_features` computes all 8 orderflow accumulators (ofi_z, cvd_delta_z, cvd_slope_z, absorption_z, tick_autocorr, inst_ratio, delta_divergence_z, depth_imbalance_z); otherwise falls back to `(0.0, 0.0, ...)`

### 3. `unified-execution/engine/src/lib.rs` — Model gate + hysteresis already existed
- **Lines 763–774**: Hysteresis signal detector with state machine:
  - `signal_armed` toggles when `p_win < lower` (where `lower = confidence_threshold - hysteresis_buffer`)
  - Model gate: `if p_win + 1e-12 < confidence_threshold { return; }` — returns LOW_CONFIDENCE if below 0.5
  - Only fires entry when `P(WIN) ≥ 0.5` AND signal is armed
- This was the missing link: the gate was correct, but the model was never loaded so `p_win` was always the proxy (~0.42)

### 4. `unified-execution/live-trade/src/main.rs` — Model loading added
- **Lines 42–68**: New code after DB restore, before trade/ob tasks:
  - Finds `main/models/lgb_model.txt` relative to CWD
  - Reads and parses with `LightGbm::from_str()` (the zero-dep text parser, validated by `real_model_matches_lightgbm` test)
  - Calls `strategy.set_model(mdl)` to install the real 200-tree model
  - Logs model stats (trees, features) or fallback errors
- If the file is absent, the engine gracefully falls back to the proxy logit (original behavior)

### 5. `unified-execution/live-trade/src/main.rs` — Historical seeding added
- **Lines 67–100**: After model loading:
  - Attempts to read `build/btc_usdt_5m_last_150.csv` from the working directory
  - Parses CSV of `[timestamp, close]` rows
  - Calls `strategy.seed_macro_closes(&closes)` to warm `rsi_2`, `sma_14`, `ma_slope_5`, `ma_accel` from startup
  - Logs how many bars were seeded; if file not found, warns that macro features will warm on live data

# Hurdles of Historical Backtest Performance and Data Management
Reference: fix/historical-backtest
- Old historical backtest was a dual architectural system that ingested csv files that relied on querying through live websockets which introduced two problems. Model must be specialized to one asset and querying data is very slow. As such, download shared aggregated data from historical archives in parquet format.
- Due to the large file size of processing months long orderflow/websocket data, loading to main ram introduces performance issues relating to Out-Of-Memory (OOM) crashes. In considering hardware limitations, this was the solution.
    - Chunking parquet files to smaller segments to reduce load in main memory. However, certain known as recursive indicators requires continuous stream of data and chunking resets it. Even with SessionFeatureContext class set, test_chunking_loss showed a major +-5-10% error due to compounding error. Example of recursive feature, VPIN. Temporarily, recursive feature columns are selected out to process without chunking which may introduce peroformance issues.
    - Migration from Pandas to Numpy and Polars helps significantly improve performance where numpy helps process calculations for polars that are not included in pandas without slowing down process. Polar's multithreading and lazy scanning feature helps run algorithm under certain load.
    - Virtual Memory and Disk allocation. With polars, partition tasks to drive to reduce as much load in the main memory as possible at the cost of possible disk thrashing. However, when considering limited hardware resources, this method is advantageous.
    - Biggest process bottlenecks lies within sorting algorithm and model training, where sorting takes up O(nlog(n)) complexity which could be easily avoided provided endpoints are already sorted. Tree model training with O(n*m), where m represent features, is the main priority.
- Why is ml_state_optimized to ml_state_optimized_csv transition considered lossy? All features are recursive and affected slightly by chunking. For example, micro-cvd has a 1.2*10^-5 error margin from mean value 17. Very small error when compounded over time, becomes an error that could affect ML decision.

# Hurdles of Live Trading Execution
Reference: features/unified-trading-system

The importance of difference between live trading and historical backtest requires a very accurate infrastructure.
1. Feature Timing Issue: Initially 1s aggregation was done to reduce feature mismatch risk where for example, websocket during network spikes would cause feature distortion. Using 1s allows trades too be evaluated slowly but never truly solved the problem where for example, 1s interval features received is 0.200s late would still affect it either way.
2. Requires more comprehensive system to calculate/prioritize for limit order and market fills. Using TWAP hysteresis method might be too predictable and cause alpha to decay both from position size and alpha signal being revealed.

## Latency, Staleness and clock skew
These 3 variables are used to determined the viability of our strategy in live execution.
- Staleness (stal): Trades received from the market. If the market receives trades >1000ms it signifies this market isn't a viable option as data source as it might be an illiquid market with very few participants.
- Latency (lat): How fast our executor process information accepted.
- Skew (SK): Time it takes for algorithm to receive transmission from exchange