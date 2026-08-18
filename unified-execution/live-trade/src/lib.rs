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

pub mod db;
pub mod exchange;
pub mod execution;
pub mod model;
pub mod report;
