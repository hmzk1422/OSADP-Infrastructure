//! Shared traits for per-strategy bundles.
//!
//! `ComputeFeatures`: YOUR computation (stream row in -> feature row out).
//! `StateBuilder`: YOUR rules (feature row -> Decision; static today, read
//! your own `model.onnx` here in Phase 2). Pure: no clock, no I/O.

use crate::core::feature::FeatureRow;
use crate::core::strategy::{Decision, MarketState};

/// YOUR computation: stream row in -> feature row out (stateful windows here).
pub trait ComputeFeatures: Send {
    fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, String>;
    fn output_names(&self) -> Vec<String>;
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
