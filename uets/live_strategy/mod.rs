//! `live_strategy` — one folder per strategy, picked by `executor.rs`.
//!
//! ```text
//! uets/live_strategy/
//!   mod.rs        # this file: module wiring + loader
//!   traits.rs     # ComputeFeatures + StateBuilder contracts
//!   runner.rs     # StrategyRunner: per-(strategy x symbol) state
//!   ofmr/         # OFMR strategy: LightGBM inference over Column_0..Column_14
//!   strategy2/    # YOUR strategy: same shape, independent state
//! ```
//!
//! Flow per row: `compute_feature` first, `state_builder` second, back to
//! `executor.rs`. Unknown name is a hard error, never a silent hold.

pub mod live_db;
pub mod ofmr;
pub mod strategy2;
pub mod traits;

pub mod runner;

pub use live_db::LiveDb;
pub use runner::{StrategyRunner, discover_strategy_dirs};
pub use traits::{ComputeFeatures, LiveStrategy, StateBuilder};

/// Strategies compiled into this binary.
pub fn available_strategies() -> Vec<&'static str> {
    vec!["ofmr", "strategy2"]
}

/// Build a fresh bundle by name (independent state per call).
pub fn load_strategy(name: &str) -> Result<LiveStrategy, String> {
    match name {
        "ofmr" => ofmr::load(),
        "strategy2" => strategy2::load(),
        other => Err(format!(
            "unknown strategy '{other}' (available: {:?})",
            available_strategies()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::feature::FeatureRow;

    fn row(price: f64) -> FeatureRow {
        let mut r = FeatureRow::new();
        r.insert("price", price);
        r
    }

    #[test]
    fn known_strategies_load() {
        for name in available_strategies() {
            assert!(load_strategy(name).is_ok(), "loads {name}");
        }
    }

    #[test]
    fn unknown_strategy_is_an_error() {
        assert!(load_strategy("typo").is_err());
    }

    #[test]
    fn discovery_lists_strategy_folders() {
        let dirs = discover_strategy_dirs();
        assert!(dirs.contains(&"ofmr".to_string()), "got {dirs:?}");
        assert!(dirs.contains(&"strategy2".to_string()), "got {dirs:?}");
    }

    #[test]
    fn step_is_isolated_per_strategy_and_symbol() {
        let mut runner = StrategyRunner::new();
        runner.step("ofmr", "BTCUSDT", &row(100.0)).unwrap();
        runner.step("strategy2", "BTCUSDT", &row(100.0)).unwrap();
        assert_eq!(runner.slot_count(), 2);
        runner.step("ofmr", "BTCUSDT", &row(101.0)).unwrap();
        assert_eq!(runner.slot_count(), 2, "same slot reused");
        runner.step("ofmr", "ETHUSDT", &row(100.0)).unwrap();
        assert_eq!(runner.slot_count(), 3);
    }
}
