//! `uets_trades` — shared trading and execution library.
//!
//! This crate is the trading layer shared by all three execution modes
//! (live_test, live_trade, playback). It is independent of exchange-specific
//! execution logic: the execution backend lives in `components/pipelines/uets_manager`.
//!
//! ```text
//! uets_trades/
//! ├── calc/        # shared PnL, equity, capital, risk, performance
//! ├── live_test/   # paper trading against live market data
//! ├── live_trade/  # real trading via uets_manager
//! └── playback/    # accelerated historical replay
//!
//! Execution flow:
//!   market data → UETS features/decision → trade intent → uets_manager
//!   → execution backend → execution events → uets_trades/calc
//!

pub mod backend;
pub mod calc;
pub mod live_test;
pub mod live_trade;
pub mod playback;
pub mod sim;
pub mod triple_barrier;
pub mod types;

pub use backend::ExecutionBackend;
pub use calc::metrics::PerformanceSnapshot;
pub use calc::reporter::Reporter;
pub use types::{
    Capital, DecisionIntent, ExecutionEvent, FillEvent, Order, OrderRequest, OrderSide,
    OrderStatus, TradeIntent,
};
pub use triple_barrier::{BarrierExit, BarrierReason, TripleBarrier};