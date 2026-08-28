//! live-trade — exchange abstraction, execution loop and local persistence.
//!
//! The strategy logic lives in the `engine` crate; this crate provides the
//! venue-facing half:
//!
//!   - [`exchange`]    — `Exchange` trait + Bybit / Bitfinex implementations
//!                       (direct WebSocket + REST, no SDKs)
//!   - [`execution`]   — the venue-agnostic strategy execution loop
//!   - [`db`]          — RocksDB persistence (trades, book, PnL)
//!   - [`report`]      — HTML PnL report generation
//!   - [`model`]       — research LightGBM model loading
//!   - [`config`]      — strategy-parameter loader (`build/live_test_config.json`)

pub mod config;
pub mod db;
pub mod exchange;
pub mod execution;
pub mod model;
pub mod report;

/// Re-exported so downstream crates (e.g. the PyO3 `engine-bindings` module,
/// whose own `#[pymodule] fn engine` would otherwise shadow the external
/// `engine` crate name) can name `StrategyConfig` as `live_trade::StrategyConfig`
/// without touching the `engine` crate directly.
pub use engine::StrategyConfig;
