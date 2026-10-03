//! strategy1 — YOUR strategy bundle.
//!
//! Wire YOUR files here. `compute_feature.rs` runs first on every stream row,
//! `state_builder.rs` runs second on the computed features.

pub mod compute_feature;
pub mod state_builder;

use super::{LiveStrategy, traits};

/// Fresh bundle with independent state (called per runner slot).
pub fn load() -> Result<LiveStrategy, String> {
    Ok(LiveStrategy {
        name: "strategy1".to_string(),
        computer: Box::new(compute_feature::Strategy1Features::new()),
        builder: Box::new(state_builder::Strategy1Rules::new()),
    })
}

/// Re-export the contracts so strategy code reads `super::ComputeFeatures`.
pub use traits::{ComputeFeatures, StateBuilder};
