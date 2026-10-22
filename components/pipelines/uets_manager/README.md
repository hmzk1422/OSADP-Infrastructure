# Unified Execution Trading System Manager
OSADP ver   : v1.0.0  
Author      : Himaska Lin  
version     : v1.0.0

uets_manager is the execution ownership and routing layer for the UETS (Unified Execution and Trading System) platform. It acts as the secure boundary and bridge between abstract trading strategy decisions and live or simulated exchange venues.

## Systems Design
The system is designed around a strict architectural separation: trading strategies never talk to an exchange directly. Instead, strategies emit intents and orders through the shared uets_trades backend trait, and uets_manager handles how those orders are validated, translated, and routed.
```
Strategy / Feature Loop (e.g., OFMR)
         │
         ▼ emits OrderRequest / TradeIntent
  ManagerEngine (ExecutionBackend)
         │
         ├──► SimEngine (Paper / Playback)[cite: 3, 19]
         │
         └──► ManagerRouter (Live Trading) ──► VenueConnectorAdapter ──► Exchange Venue[cite: 19, 21]
```
## Components
`lib.rs`
uets_manager exposes ManagerEngine, which encapsulates and delegates execution across three distinct modes through a unified interface:
- **LiveTest**: Paper trading mode that simulates order fills, fees, spreads, and slippage locally against live market data feeds.
- **LiveTrade**: Real-money or testnet trading mode that routes validated orders directly to exchange venues via ManagerRouter.
- **Playback**: Accelerated historical replay mode using the same simulated matching engine as paper trading for reproducible backtesting.

### Select a playback evaluation

Playback evaluation is configured independently of live/paper execution:

```yaml
mode: playback
acceleration: 100
playback:
  method: purged_walk_forward # default; or parametric_monte_carlo / cpcv
  max_abs_correlation: 0.5
  n_splits: 6
  n_test_splits: 2
  purge_bars: 0
  max_paths: 100000
```

`ManagerEngine::playback_settings()` exposes the validated settings. Pass them
to `PlaybackSettings::run` with either CryptoL2 `MarketEvent`s or caller-adapted
synthetic scenarios and a strategy callback factory. The evaluator runs the
paths unpaced and in parallel, then returns a final report with cumulative
return, average realized expectancy per trade, maximum drawdown, Sortino, and
CVaR05/CVaR95 summaries. The callback factory is invoked independently for each
fold/scenario; it must not share mutable strategy state across paths.

`router.rs`
The ManagerRouter acts as the exclusive gatekeeper for live orders. Every outbound order request must pass strict safety validations before hitting a venue:
- **Duplicate Protection**: Tracks active order IDs using a secure set (BTreeSet) to prevent accidental double-submissions.
- **Stale Market Data Guard**: Rejects orders if the latest market tick data exceeds the allowed age threshold (e.g., older than 2 seconds), protecting against execution on stale feeds.
- **Sanity Validation**: Verifies that quantities are positive finite numbers and that symbols, strategies, and asset IDs are correctly formatted.

`venue.rs`
The UetsExecutionConnectorAdapter translates abstract execution requests into native exchange connector calls:
- **Order Types**: Supports both aggressive Market orders (with immediate fill processing) and passive Limit (Maker) orders (allowing resting orders on the book).
- **Risk Flags**: Preserves critical execution flags such as reduce_only for position exits and custom client order IDs for tracking.