//! `core` — the custom-made engine: everything this project owns.
//!
//! Nothing here talks to a broker. Buying, selling, marks, and reading the
//! account all live behind the connector traits in [`crate::exchange`]; `core`
//! consumes their results (`Fill`, `AccountSnapshot`) as *data*. What remains
//! is pure logic plus local file I/O:
//!
//! | module | role |
//! |--------|------|
//! | [`order`] | `Side` / `OrderType` / `OrderRequest` / `Order` / `Fill` — the vocabulary |
//! | [`position`] | `Holding` + `PositionBook`: net size per symbol, marked to market |
//! | [`pnl`] | `Trade` + `PnlLedger`: realized pnl, equity curve, drawdown |
//! | [`portfolio`] | `Portfolio`: cash, margin and equity over the book + ledger |
//! | [`risk`] | `RiskLimits` + pre-trade checks, applied to every venue alike |
//! | [`stats`] | `sortino`, `win_rate`, `cumulative_return`, `max_drawdown` |
//! | [`telemetry`] | counters + JSON writer, plus [`telemetry::latency`] and [`telemetry::skew`] diagnostics |
//! | [`gui`] | the file bridge a vim (or any) dashboard reads and writes |
//! | [`feature`] | the feature engine: ingest -> spec -> compute -> row |
//! | [`strategy`] | strategy primitives: `Decision`, `MarketState`, model choice |
//!
//! The split that matters: **`core` never knows a venue's wire format**, and
//! `connectors/<venue>` never knows about pnl or risk. `Portfolio` is the
//! system's own view of the account; `crate::exchange::AccountConnector::snapshot`
//! returns the venue's own figures, and the two are reconciled rather than
//! conflated. The boundary is enforced by `tests/architecture.rs`, which
//! scans these sources from the outside.

pub mod feature;
pub mod gui;
pub mod order;
pub mod pnl;
pub mod portfolio;
pub mod position;
pub mod risk;
pub mod stats;
pub mod strategy;
pub mod telemetry;
