//! OFMR — live strategy bundle.
//!
//! `features` runs first on every stream row: raw market tick -> 15-column
//! feature row (Column_0 .. Column_14, exact LightGBM training order).
//! `strategy` runs second: hand-rolled LightGBM txt inference over those 15
//! columns, mapping P(WIN) >= threshold to `EnterLong` (long-only gated).

pub mod features;
pub mod settings;
pub mod socket;
pub mod strategy;

use super::{LiveStrategy, traits};

/// Fresh bundle with independent state (called per runner slot).
pub fn load() -> Result<LiveStrategy, String> {
    let settings = settings::Settings::try_load()?;
    let computer = features::OfmrFeatures::with_settings(settings.clone());
    let builder = strategy::OfmrStrategy::try_with_settings(settings)
        .map_err(|error| format!("load OFMR model: {error}"))?;
    Ok(LiveStrategy {
        name: "ofmr".to_string(),
        computer: Box::new(computer),
        builder: Box::new(builder),
    })
}

/// Re-export the contracts so strategy code reads `super::ComputeFeatures`.
pub use traits::{ComputeFeatures, StateBuilder};
