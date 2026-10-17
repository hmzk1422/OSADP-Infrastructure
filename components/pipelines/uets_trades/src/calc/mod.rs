//! Shared calculation layer — one implementation used by all three modes.
//!
//! All metrics (PnL, equity, drawdown, Sortino, win rate, etc.) are computed
//! here with identical formulas regardless of execution mode.

pub mod metrics;
pub mod reporter;

pub use metrics::PerformanceSnapshot;
pub use reporter::Reporter;
