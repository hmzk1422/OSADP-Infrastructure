# uets_trades — Shared Trading & Execution Library

OSADP ver : v1.0.0  
Author    : Himaska Lin  
version   : v0.1.0  

Shared execution layer used by all three modes (`live_test`, `live_trade`, `playback`).
Strategies emit `TradeIntent`/`OrderRequest` through `ExecutionBackend`; the backend decides
whether fills are simulated (paper / playback) or routed to a venue (live). Accounting
(`Book`, `Capital`, triple-barrier exits) is identical across modes — simulated results
are reproducible in playback and live_test.

```text
market data → UETS features/decision → trade intent → uets_trades/ExecutionBackend
                                       ↓
                             live_test   live_trade       playback
                             paper sim   venue route      accelerated replay
                                       ↓                ↓
                             execution events   execution events   MarketEvent replay
                                       ↓                ↓                ↓
                             uets_trades/calc (PnL, equity, R)  ← shared accounting
```

## Modes

| Mode | Backend | What it does |
|------|---------|--------------|
| `live_test` | `SimEngine` | Paper trading against live market data. Simulated fills, fees, slippage, triple-barrier exits. No real orders. |
| `live_trade` | `uets_manager::ManagerRouter` | Real/testnet trading. Routes validated orders to exchange venues via `LiveRouter`. |
| `playback` | `PlaybackBackend` + `SimEngine` | Accelerated historical replay. Reads merged Parquet, replays through the **same** `SimEngine` as paper. Configurable speed multiplier. Never submits real orders. |

## Playback — backtesting historical data

**Playback does not call a live API.** It reads the CryptoL2 ingest merged Parquet table
(`market_data/ingest_parquet/<SYMBOL>/merged/<DATE>/<HH>.parquet`), decodes it into `MergedRow`s,
stable-sorts + deduplicates them deterministically, then feeds `MarketEvent`s to a
`PlaybackBackend` running the real UETS at an acceleration factor (e.g. 100×).

The replay driver is wall-clock paced by `acceleration`: an event 1 s after the first is
delivered `1 / acceleration` seconds after playback starts. No event is skipped to hit a
speed target — every event is processed in timestamp order.

### Backtest workflow

1. **Locate the merged Parquet file(s)** produced by `ingestion/CryptoL2/merge_events.py`.
2. **Decode** them with `crypto_l2::load_merged_parquet(path)` → `Vec<MergedRow>`.
3. **Prepare playback** with `crypto_l2::prepare_playback(rows, symbol)` → `Playback`
   (decodes, stable-sorts, deduplicates, carries last mid forward across bookless rows,
   skips rows before the first book).
4. **Replay** `playback.events` through `PlaybackBackend::replay_events(events, |backend, event| ...)`
   — inside the callback call your strategy's `on_market_update` + `long`/`short`/`exit`.
5. **Inspect** `backend.engine()` for capital, positions, orders, and `drain_events()` for fills.

### Example

```rust
use uets_trades::playback::crypto_l2::{load_merged_parquet, prepare_playback};
use uets_trades::playback::{MarketEvent, PlaybackBackend};
use std::path::Path;

let path = Path::new("market_data/ingest_parquet/BTCUSDT/merged/20260101/00.parquet");
let rows = load_merged_parquet(path).expect("decode merged parquet");
let playback = prepare_playback(rows, "BTCUSDT");

let mut backend = PlaybackBackend::with_capital("ofmr@btcusdt", 10_000.0, 100.0); // 100×
let count = backend.replay_events(playback.events.into_iter().map(|e| MarketEvent {
    timestamp: e.timestamp,
    marks: e.marks,
}), |backend, event| {
    // your strategy: read marks, emit long/short/exit via backend.long() / backend.short()
    Ok(())
}).expect("replay");

let capital = backend.engine().capital("ofmr@btcusdt").unwrap();
let events = backend.drain_events();
```

### Playback evaluation methods

`ExecutionSettings` selects the evaluation method under `playback.method`. If
omitted, playback defaults to single-asset purged chronological walk-forward.

```yaml
mode: playback
acceleration: 100
playback:
  method: cpcv # purged_walk_forward | parametric_monte_carlo | cpcv
  max_abs_correlation: 0.5
  n_splits: 6
  n_test_splits: 2
  purge_bars: 0
  max_paths: 100000
```

`PlaybackSettings::run` takes either ordered `MarketEvent`s (walk-forward or
CPCV) or independent scenario paths (Monte Carlo), and a factory that creates
fresh strategy state per path. Each path uses a new `PlaybackBackend` and
`SimEngine`; all paths execute unpaced on Rayon worker threads. The final
`PlaybackReport` includes mean cumulative return, per-trade expectancy, maximum
drawdown and Sortino, plus lower/upper five-percent conditional tail means
(`CVaR05`/`CVaR95`).

The Monte Carlo and CPCV correlation masks are computed before any strategy
callback is run. Monte Carlo keeps paths only when their aligned simple-return
correlations are defined and `|rho| <= max_abs_correlation`. CPCV builds
correlation compatibility across chronological chunks once, then enumerates
only compatible test-chunk combinations. Walk-forward rejects multi-asset
input and advances through one chronological asset in purged folds.

The Rust evaluator accepts `MarketEvent`s to keep data loading separate from
strategy execution. Feed `prepare_playback(...).events` from CryptoL2 for CPCV
or walk-forward. For Monte Carlo, adapt generated `price` series from the
synthetic-data generators into timestamped single-symbol `MarketEvent` paths.
Strategy-specific feature construction and decisions remain in the supplied
callback; the evaluator handles execution, metrics, correlation filtering,
and aggregation.

## Useful commands

All run from the workspace root (`intraday-momentum-composite`):

```bash
# Build the crate (uets_trades compiles; full workspace has pre-existing backend.rs errors unrelated to this crate)
cargo check -p uets_trades

# Format
cargo fmt -p uets_trades

# Run unit tests (requires the pre-existing backend.rs errors to be fixed first)
cargo test -p uets_trades

# Run from the crate directory (equivalent)
cargo check --manifest-path components/pipelines/uets_trades/Cargo.toml
cargo fmt --manifest-path components/pipelines/uets_trades/Cargo.toml
cargo test --manifest-path components/pipelines/uets_trades/Cargo.toml

# Full workspace check (uets feature integration)
cargo check
```

## Key files

- `src/lib.rs` — crate root, module declarations, public re-exports
- `src/backend.rs` — `ExecutionBackend` trait, `BackendError`, `ActionContext`, `LiveRouter`
- `src/sim.rs` — `SimEngine`, `SimConfig`, `Book`, `Capital` (shared by paper + playback)
- `src/playback/mod.rs` — `PlaybackBackend`, `replay_events()`, `MarketEvent`
- `src/playback/crypto_l2.rs` — Parquet decode, stable sort, deduplicate, `prepare_playback()`
- `src/types.rs` — `Order`, `OrderRequest`, `TradeIntent`, `ExecutionEvent`, `FillEvent`, etc.
- `src/calc/` — PnL, equity, capital, risk, performance metrics
- `src/triple_barrier.rs` — `TripleBarrier` entry/exit risk logic