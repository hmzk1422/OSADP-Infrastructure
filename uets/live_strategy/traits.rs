//! Shared traits for per-strategy bundles.
//!
//! `ComputeFeatures`: YOUR computation (stream row in -> feature row out).
//! `StateBuilder`: YOUR rules (feature row -> Decision; static today, read
//! your own `model.onnx` here in Phase 2). Pure: no clock, no I/O.

use crate::core::feature::FeatureRow;
use crate::core::strategy::{Decision, MarketState};

pub const DEFAULT_WARMUP_LOOKBACK_SECS: u64 = 2 * 60 * 60;

/// YOUR computation: stream row in -> feature row out (stateful windows here).
pub trait ComputeFeatures: Send {
    fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, String>;
    fn output_names(&self) -> Vec<String>;

    /// Historical market data loaded before live events. Strategies with longer
    /// windows can override this; the feed uses the maximum across its workers.
    fn warmup_lookback_secs(&self) -> u64 {
        DEFAULT_WARMUP_LOOKBACK_SECS
    }

    /// Clear feature state before replaying a new history seed.
    fn reset(&mut self);
}

/// YOUR rules: feature row -> Decision. Pure: no clock, no I/O.
pub trait StateBuilder: Send + Sync {
    fn decide(&self, state: &MarketState) -> Decision;
}

/// One compiled strategy bundle.
pub struct LiveStrategy {
    pub name: String,
    pub computer: Box<dyn ComputeFeatures>,
    pub builder: Box<dyn StateBuilder>,
}
