# Developer Log
Documenting architectural planning and engineering hurdles.

# Unified Backtest/Live Architecture System
Reference: feature/unified-trading-system
```
RESEARCH                    EXECUTION

[Feature Builder]
       ↓
[ML State Builder]
       ↓
[ML State Optimized]
       ↓
 ┌─────┴──────┐
 ▼            ▼
trade.txt   results.csv
 │            │
 │       [Meta-labeller]
 │            │
 └──────┬─────┘
        ▼
   [engine.rs]
        ↓
 [Meta-labeller]
        ↓
 ┌──────┴───────┐
 ▼              ▼
live-test    live-trade
```
Two systems, one strategy. Using shared architecture prevents inconsistency between backtest and live executions.
- **Trading Strategy**: Purely for defining the strategy code. 
    - ml_state_features.py: Computing the features, completely separate to ensure with very high confidence no bleeding or lookahead bias happens. 
    - ml_state_optimized.py: Composing features together with market such as calculating triple barrier or model training.
- **Execution Layer**: ml_state_runtime.py connected with ml_state_optimized as abstraction for true execution, querying data, providing commands, etc. Connects to txt file obtained from trading strategy.
- **Histbacktest/Live Execution**: Execution layer connects with histbackteest and live execution to query for data. 

Note: Should latency matter, migrate execution layer and histbacktest/live execution into rust.

## Checklist of Items

### 1. `uets_rs/engine/src/lib.rs` — Feature vector completed
- **Line 700**: Added 4 missing orderflow features to the 15-element feature array:
  - `m.tick_autocorr`, `m.inst_ratio`, `m.delta_divergence_z`, `m.depth_imbalance_z`
  - Previously the array had only 11 of 15 initializers; Rust zero-filled the rest, giving the model garbage features

### 2. `uets_rs/engine/src/lib.rs` — Orderflow config wired
- **Module constant** `USE_ORDERFLOW_FEATURES: bool = true` (lines 67/81)
- **StrategyConfig field** `use_orderflow_features: bool` (line 129)
- **Config initialization** passes `use_orderflow_features: USE_ORDERFLOW_FEATURES` (line 157)
- **Feature gating** at line 1190: `if self.cfg.use_orderflow_features` computes all 8 orderflow accumulators (ofi_z, cvd_delta_z, cvd_slope_z, absorption_z, tick_autocorr, inst_ratio, delta_divergence_z, depth_imbalance_z); otherwise falls back to `(0.0, 0.0, ...)`

### 3. `uets_rs/engine/src/lib.rs` — Model gate + hysteresis already existed
- **Lines 763–774**: Hysteresis signal detector with state machine:
  - `signal_armed` toggles when `p_win < lower` (where `lower = confidence_threshold - hysteresis_buffer`)
  - Model gate: `if p_win + 1e-12 < confidence_threshold { return; }` — returns LOW_CONFIDENCE if below 0.5
  - Only fires entry when `P(WIN) ≥ 0.5` AND signal is armed
- This was the missing link: the gate was correct, but the model was never loaded so `p_win` was always the proxy (~0.42)

### 4. `uets_rs/live-trade/src/main.rs` — Model loading added
- **Lines 42–68**: New code after DB restore, before trade/ob tasks:
  - Finds `main/models/lgb_model.txt` relative to CWD
  - Reads and parses with `LightGbm::from_str()` (the zero-dep text parser, validated by `real_model_matches_lightgbm` test)
  - Calls `strategy.set_model(mdl)` to install the real 200-tree model
  - Logs model stats (trees, features) or fallback errors
- If the file is absent, the engine gracefully falls back to the proxy logit (original behavior)

### 5. `uets_rs/live-trade/src/main.rs` — Historical seeding added
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

# live-test — Historical Parquet Replay Through the Live-Trade Loop
Reference: live-test replay runner

The goal: run the **exact** live-trading code path (same `run_execution_loop`,
same micro/macro bar aggregation, same entry/exit gates, same RocksDB + HTML
report) against historical data — so a replay of the same parquet that fed the
research backtest should reproduce the same trades. If the outputs diverge, the
strategy/engine implementation is the suspect.

- **Why not read parquet in Rust?** `Cargo.lock` ships no parquet/arrow crate and
  Rust only reads plain text today. Parquet is read in Python (pyarrow/polars,
  already the research stack), so the design is a hybrid:
  - `unified_runner.py live-test` reads `research/csv/{TICKER}_trades.parquet` +
    `{TICKER}_orderbook.parquet`, merges both streams, sorts by
    `(timestamp, ordinal)` (trades before book tops at equal ms — deterministic)
    and writes one event-per-line replay stream to
    `build/live_test_feed_{ticker}.replay`.
  - `uets_rs/live-trade/src/exchange/parquet.rs` (`ReplayExchange`)
    parses that stream into `MarketEvent`s and serves them back in
    `REPLAY_BATCH`-sized batches — the replay equivalent of a WebSocket frame.
  - `uets_rs/live-test/src/main.rs` builds a `LoopContext` with the
    replay exchange and calls the identical `live_trade::execution::run_execution_loop`.
- **Stream format**: `#TICKER btcusdt`, optional `#CLOSES a,b,c` (feature warmup),
  `T <ts> <price> <amount> <buy|sell> <id>`, `B <ts> <bid_px> <bid_amt> <ask_px> <ask_amt>`.
- **End-of-data**: when the feed is exhausted the exchange flips the shared `stop`
  AtomicBool, the loop drains, `finish()` writes `build/pnl_report_{ticker}.html`,
  then the process exits cleanly. `--loop` rewinds instead (continuous sim).
- **Window / scale**: BTCUSDT alone is 223M trades / ~155 days (~multi-GB), so
  `--start-ms / --end-ms / --limit` bound the replay. The default RocksDB store
  (`build/live_test_{ticker}.rocksdb`) is wiped per run so re-runs can't restore
  stale equity or duplicate trades (explicit `--db` is kept, live-trade semantics).
- Model parity: the loop loads `main/models/lgb_model.txt` exactly like
  live-trade, so `P(WIN)` comes from the real LightGBM when the research export
  is present (proxy logit otherwise).

Run it:
```
python unified_runner.py live-test --ticker btcusdt --limit 500000 --release
python unified_runner.py live-test --ticker btcusdt --start-ms 1709344800000 --end-ms 1709431200000
```

# Replay Parity Validation — Python research vs Rust live-test
Reference: `research/parity_diagnostics.py`, `config/strategy.py REPLAY_FEED`

The design goal above ("a replay of the same parquet should reproduce the same
trades") needs a MEASUREMENT, not a hope. Three moving parts form the parity
chain:

1. **Same feed.** `unified_runner.py live-test` materializes
   `build/live_test_feed_{ticker}.replay` from the research parquet/csv store.
   `config/strategy.py` exposes `REPLAY_FEED`: pointing it at that same file
   makes the research pipeline (`research/ofmr/ml_state_optimized.py`) build its
   state space from the identical event stream the Rust engine consumes.
2. **Same outputs, as txt.**
   - Python txt: `build/walk_forward_trade_log_v3.csv` (STEP 10 of
     `ml_state_optimized.py` — columns `datetime, ret, threshold, cum_pnl`,
     returns in R).
   - Rust txt: the `pnl` column family of the live-test RocksDB store
     (closed-trade JSON records; export with
     `python research/extract_rocksdb.py --db <store> --out-dir <dir>` →
     `pnl_trades.csv` / `pnl_snapshots.csv`). Columns: `entry_ms, exit_ms,
     side, entry_price, exit_price, reason, qty, net_usd, r_multiple,
     equity_after, duration_s, barrier_pct, target_price, stop_price`.
3. **Column-by-column validator.** `research/parity_diagnostics.py` loads both
   logs ASYNCHRONOUSLY (asyncio worker threads + gathered comparison tasks),
   aligns trades by order, and reports per-column similarity: the % of aligned
   pairs agreeing within tolerance (100% / 90% / FAIL verdicts), mean/max
   absolute difference, correlation, and trade-count agreement. Timestamp
   verdicts are offset-corrected (a constant clock shift is reported as the
   median offset instead of failing every pair).

Run it:
```
# after running BOTH paths on the same feed window:
python unified_runner.py live-test --ticker btcusdt --limit 500000 --release
#   (research, same feed — set in config/strategy.py)
#   REPLAY_FEED = "build/live_test_feed_btcusdt.replay"
#   python -m research.ofmr.ml_state_optimized

# then validate (Rust side read straight from the RocksDB store):
python -m research.parity_diagnostics --rust-db build/live_test_btcusdt.rocksdb

# or from an extract_rocksdb.py CSV export instead:
python -m research.parity_diagnostics --rust-csv build/pnl_trades.csv
```

Reading the report (`build/parity_diagnostic_report.md`):
- `return_r` (python `ret` vs rust `r_multiple`) is THE parity column: PASS
  here means both engines priced/exited identically.
- `exit_time` FAIL means the trade SEQUENCES themselves diverge — check entry
  gates (`--wf-dir` parity gates: `require_fold_coverage`,
  `one_candidate_per_bar`), fold-model coverage of the replay window, and the
  hunting-zone filter before suspecting pricing.
- `cum_return_r` compares python `cum_pnl` against the running sum of the
  Rust `r_multiple` — drift that grows over the session shows up here even
  when individual trades are near-identical.

# Live-Trade Hardening: Feeds, Report, Portfolio Supervisor
Reference: production-hardening pass

## Feed connectivity (the `11001` / silent-feed family)
Three stacked defects made live feeds fail or look dead; all are endpoint/data
issues, not strategy issues:

1. **Bybit v5 WS path** — `/v5/public` returns **404**: v5 public streams are
   category-specific (`/v5/public/linear|spot|inverse|option`). The feed now
   connects to `wss://stream.bybit.com/v5/public/linear` with
   `ws.bybit.com/v5/public/linear` as DNS-fallback host (some VPN resolvers -
   e.g. ProtonVPN - do not resolve `ws.bybit.com`; error surfaced historically
   as `os error 11001 No such host is known`). The subscribe ACK also changed:
   v5 answers `{"success":true,...,"op":"subscribe"}` at TOP level, not nested
   under `request.op` - both forms accepted.
2. **Bybit seed klines** - two serde bugs: envelope key is `retCode`
   (struct said `ret_code`, so EVERY fetch failed with "error decoding
   response body"), and rows are ARRAYS `[ts,o,h,l,c,vol,turnover]` with close
   at index 4 (struct expected objects). Interval fixed `1 -> 5` minutes to
   match the 300 s composite bar (see features.py "seeding MUST use the same
   bar size"). Regression test `historical_closes_decode_and_order` guards it.
3. **Binance futures data degraded on some networks** - measured on a VPN exit:
   `fstream.binance.com` served `@bookTicker`/depth fine but delivered ZERO
   frames for `@aggTrade`/klines across raw-path, combined-URL and SUBSCRIBE
   forms. The binance feed therefore uses **SPOT** endpoints
   (`stream.binance.com:9443` + `api.binance.com/api/v3`); spot BTCUSDT tracks
   the perp within bps. Spot bookTicker frames carry no `"e"` field - the
   parser dispatches on payload shape when `e` is absent.

## Per-second feature persistence (report indicator history)
New RocksDB column family `features`: one `FeatureSample {ts_ms, rsi_2,
win_prob}` per closed micro-bar, written from the execution loop's
`EngineEvent::BarClosed` arm. This is what lets the HTML report draw the FULL
RSI(2) history instead of only the hunting-zone moments. Note: samples only
exist for sessions run on this build - older tape has none.

## HTML report rewritten TradingView-style (`report_chart.js`)
Single canvas, three panes (price candles / RSI(2)+P(WIN) / equity) sharing a
crosshair and a data-window legend. Candles aggregate client-side from the
1 s tape (chip shows e.g. `20 x 1s`); zoom = wheel, pan = drag, dblclick =
reset to the recent default window. Trade list is sortable/filterable and
jumps the chart via `locateTrade()`. Summary cards use REAL profit factor
(grossProfit/grossLoss - was win/loss counts) plus max drawdown, expectancy,
avg win/loss, best/worst. JS lives in `live-trade/src/report_chart.js`
(`include_str!`), not inline Rust.

**Marker alignment bug**: aggregated candles were drawn in group-index space
while markers/ticks used base-index space - diverging by up to ~28 px once
`k > 1` (the "icons glitch when panning" report). Fixed by giving every group
its absolute base-index center (`bi`) and routing ALL drawing through one
`xForIdx(baseIdx)` mapping (verified 0.0000 px max error at k=20).

## Portfolio supervisor (`portfolio_supervisor.py`) + `db` command
Multi-asset without touching the engine: `unified_runner.py portfolio` reads a
master JSON table (`portfolio.example.json`) and spawns ONE `live-trade`
PROCESS per asset - own tokio runtime, own RocksDB
(`data/portfolio_<ticker>.rocksdb`), own auto-refreshing report. The
supervisor parses each child's console protocol (`[bar]`/`[entry]`/`[exit]`/
connect failures) into live state, prints a per-asset + TOTAL table every
`status_interval`, raises `*** ALERT *** <asset> PROCESS DOWN` on unexpected
exit with optional auto-restart (3 s cooldown), forwards Ctrl-C as
CTRL_BREAK/SIGINT for graceful flushes, and writes a tabbed dashboard
(`build/portfolio_dashboard.html`: overview tab + sidebar tab per asset,
linking each asset's full report). `unified_runner.py db --list /
--clean-ticker <sym> / --clean-all [--yes]` manages local stores. Console
status lines for `live-trade` moved from 0.5 s to `--status-interval`
(default 2 s) - ingestion cadence is unaffected.

# UETS/Research Validator
Though both of them use the same txt file, research and UETS must be double verified to
ensure they work in the same way. Implemented following methods to solve for it.
- executor_matching_diagnostics.py -> Runs ml_state_optimized back using txt file and executor.rs
asynchronously. At the end, run lossless test. If lossless factor >= 99%, uets and research
execution matches. Otherwise, requires double checking. Config files are altercated so both
tests match.

# Report session scoping + research replay/validation (2026-08-28)
- HTML PnL reports are now SESSION-SCOPED by default: every equity snapshot and closed
  trade is tagged with a `session_id` (unix-ms at process start) via a new RocksDB
  `sessions` column family. `live-trade --session latest|all` (default `latest`) selects
  whether the report shows only the current run (matching the terminal status table) or
  the whole store's history. Existing rows without the field (session_id=0) still read fine.
- `research/ofmr/ml_state_optimized.py` now writes per-feature OPEN / HIGH / LOW / CLOSE columns
  into `build/walk_forward_trade_log_v3.csv` for every walk-forward trade (entry value,
  min/max over the holding window, value at exit; `TRADE_FEATURE_WINDOWS` toggle, default
  on, observational only, never used for prediction/labeling).
- `config/strategy.py: REPLAY_FEED` lets the research pipeline build its state space from
  the same replay feed TEXT FILE the Rust live-test engine consumes
  (`build/live_test_feed_<sym>.replay`, `T`/`B` event lines). Implemented via
  `load_state_space_from_replay()` in `ml_state_builder.py`; feeds without orderbook lines
  get a degenerate bid==ask==price book so feature construction still runs.