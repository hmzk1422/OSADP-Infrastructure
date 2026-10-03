//! strategy2 — second independent bundle (same contract as strategy1).

pub mod compute_feature;
pub mod state_builder;

use super::{LiveStrategy, traits};

/// Fresh bundle with independent state.
pub fn load() -> Result<LiveStrategy, String> {
    Ok(LiveStrategy {
        name: "strategy2".to_string(),
        computer: Box::new(compute_feature::Strategy2Features::new()),
        builder: Box::new(state_builder::Strategy2Rules::new()),
    })
}

pub use traits::{ComputeFeatures, StateBuilder};
