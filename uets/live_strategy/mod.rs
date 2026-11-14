//! `live_strategy` — one folder per strategy, picked by `executor.rs`.
//!
//! ```text
//! uets/live_strategy/
//!   mod.rs        # this file: module wiring + loader
//!   traits.rs     # ComputeFeatures + StateBuilder contracts
//!   runner.rs     # StrategyRunner: per-(strategy x symbol) state
//!   ofmr/         # OFMR strategy: LightGBM inference over Column_0..Column_14
//!   example_strategy/ # Minimal reference strategy for new implementations
//! ```
//!
//! Flow per row: `compute_feature` first, `state_builder` second, back to
//! `executor.rs`. Unknown name is a hard error, never a silent hold.

pub mod example_strategy;
pub mod live_db;
pub mod ofmr;
pub mod traits;
pub mod warmup;

pub mod runner;

pub use live_db::LiveDb;
pub use runner::{StrategyRunner, discover_strategy_dirs};
pub use traits::{ComputeFeatures, DEFAULT_WARMUP_LOOKBACK_SECS, LiveStrategy, StateBuilder};

/// Strategies compiled into this binary.
pub fn available_strategies() -> Vec<String> {
    vec!["ofmr".to_string(), "example_strategy".to_string()]
}

/// Build a fresh bundle by name (independent state per call).
pub fn load_strategy(name: &str) -> Result<LiveStrategy, String> {
    let base = name.split_once('@').map_or(name, |(strategy, _)| strategy);
    if !available_strategies()
        .iter()
        .any(|available| available == base)
    {
        return Err(format!(
            "unknown strategy folder '{base}' (discovered: {:?})",
            available_strategies()
        ));
    }

    match base {
        "ofmr" => ofmr::load(),
        "example_strategy" => example_strategy::load(),
        other => Err(format!(
            "strategy folder '{other}' has no compiled loader; add its module to live_strategy"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::feature::FeatureRow;

    fn row(price: f64, event_time_ms: f64) -> FeatureRow {
        let mut r = FeatureRow::new();
        r.insert("price", price);
        r.insert("event_time_ms", event_time_ms);
        r
    }

    #[test]
    fn known_strategies_load() {
        for name in available_strategies() {
            assert!(load_strategy(&name).is_ok(), "loads {name}");
        }
    }

    #[test]
    fn every_strategy_has_a_valid_history_warmup_requirement() {
        for name in available_strategies() {
            let bundle = load_strategy(&name).expect("loads strategy");
            assert!(
                bundle.computer.warmup_lookback_secs() > 0,
                "strategy {name} must request historical data before live input"
            );
        }
    }

    #[test]
    fn unknown_strategy_is_an_error() {
        assert!(load_strategy("typo").is_err());
    }

    #[test]
    fn discovery_lists_strategy_folders() {
        let dirs = discover_strategy_dirs();
        assert!(
            available_strategies().contains(&"ofmr".to_string()),
            "got {dirs:?}"
        );
        assert!(
            available_strategies().contains(&"example_strategy".to_string()),
            "got {dirs:?}"
        );
    }

    #[test]
    fn step_is_isolated_per_strategy_and_symbol() {
        let mut runner = StrategyRunner::new();
        runner
            .step("ofmr", "BTCUSDT", &row(100.0, 1_000.0))
            .unwrap();
        runner
            .step("example_strategy", "BTCUSDT", &row(100.0, 1_000.0))
            .unwrap();
        assert_eq!(runner.slot_count(), 2);
        runner
            .step("ofmr", "BTCUSDT", &row(101.0, 2_000.0))
            .unwrap();
        assert_eq!(runner.slot_count(), 2, "same slot reused");
        runner
            .step("ofmr", "ETHUSDT", &row(100.0, 1_000.0))
            .unwrap();
        assert_eq!(runner.slot_count(), 3);
    }

    #[test]
    fn runner_seeds_history_without_deciding_until_live_input() {
        let mut runner = StrategyRunner::new();
        let mut start = FeatureRow::new();
        start.insert("warmup_start", 1.0);
        assert_eq!(
            runner.step("example_strategy", "BTCUSDT", &start).unwrap(),
            (
                crate::core::strategy::Decision::Hold,
                crate::core::strategy::Position::Flat
            )
        );

        for event_time_ms in 1..=4 {
            let mut historical = row(10.0, event_time_ms as f64);
            historical.insert("warmup", 1.0);
            assert_eq!(
                runner
                    .step("example_strategy", "BTCUSDT", &historical)
                    .unwrap(),
                (
                    crate::core::strategy::Decision::Hold,
                    crate::core::strategy::Position::Flat
                )
            );
        }

        let mut end = FeatureRow::new();
        end.insert("warmup_end", 1.0);
        assert_eq!(
            runner.step("example_strategy", "BTCUSDT", &end).unwrap(),
            (
                crate::core::strategy::Decision::Hold,
                crate::core::strategy::Position::Flat
            )
        );

        assert_eq!(
            runner
                .step("example_strategy", "BTCUSDT", &row(20.0, 5.0))
                .unwrap(),
            (
                crate::core::strategy::Decision::EnterLong,
                crate::core::strategy::Position::Long
            )
        );
    }
}
