//! Small reference bundle demonstrating the live strategy contracts.

pub mod compute_feature;
pub mod state_builder;

use super::{LiveStrategy, traits};

/// Fresh bundle with independent state.
pub fn load() -> Result<LiveStrategy, String> {
    Ok(LiveStrategy {
        name: "example_strategy".to_string(),
        computer: Box::new(compute_feature::ExampleFeatures::new()),
        builder: Box::new(state_builder::ExampleRules::new()),
    })
}

pub use traits::{ComputeFeatures, StateBuilder};
