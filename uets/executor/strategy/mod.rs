//! Strategy orchestration.  State construction, model choice, and decisions
//! are separate so a strategy can be replaced without touching feature code.

mod decision;
mod model;
mod reconstruct;
mod registry;
mod state;

pub use decision::{Decision, DecisionEngine, ThresholdDecision};
pub use model::{FixedModel, ModelId, ModelSelector, RotatingModel};
pub use reconstruct::{Bar, ReconstructError, Reconstructor};
pub use registry::{StrategyRegistry, select_model};
pub use state::{MarketState, Position};
