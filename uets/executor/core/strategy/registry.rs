//! Named strategy lookup.
//!
//! The registry is what lets a config pick `"momentum_v3"` at runtime instead of
//! the code hard-wiring one strategy. A name that is not registered is an
//! explicit error, never a silent no-op: a typo in a config that quietly trades
//! nothing is the worst possible failure mode for a live system.

use std::collections::BTreeMap;

use super::decision::{Decision, DecisionEngine};
use super::model::{ModelId, ModelSelector};
use super::state::{MarketState, Position};

/// One registered strategy: a model plus the rule that consumes it.
struct Entry {
    model: ModelId,
    engine: Box<dyn DecisionEngine>,
}

/// All strategies available to this process, keyed by name.
#[derive(Default)]
pub struct StrategyRegistry {
    entries: BTreeMap<String, Entry>,
}

impl StrategyRegistry {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Register `name`, replacing any previous entry with that name.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        model: ModelId,
        engine: Box<dyn DecisionEngine>,
    ) {
        self.entries.insert(name.into(), Entry { model, engine });
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The model a strategy is bound to.
    pub fn model_for(&self, name: &str) -> Option<&ModelId> {
        self.entries.get(name).map(|e| &e.model)
    }

    /// Run a registered strategy and apply the result to `state`.
    ///
    /// Returns the position reached. An unknown name is an error rather than
    /// `Hold`, so a misconfiguration surfaces on the first row instead of
    /// producing a capture that looks fine and trades nothing.
    pub fn apply(&self, name: &str, state: &mut MarketState) -> Result<Position, String> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("unknown strategy '{name}'"))?;
        let decision: Decision = entry.engine.decide(state);
        Ok(state.apply(decision))
    }

    /// Run a registered strategy without mutating state, for research.
    pub fn decide(&self, name: &str, state: &MarketState) -> Result<Decision, String> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("unknown strategy '{name}'"))?;
        Ok(entry.engine.decide(state))
    }
}

impl std::fmt::Debug for StrategyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StrategyRegistry")
            .field("names", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Picks a model per call and reports what it chose.
///
/// Exists so the "which model" concern is injectable in tests without the
/// registry depending on a concrete selector.
pub fn select_model<'a>(selector: &'a dyn ModelSelector, symbol: &str, sequence: u64) -> ModelId {
    selector.select(symbol, sequence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::feature::FeatureRow;
    use crate::core::strategy::decision::ThresholdDecision;

    fn state(value: f64) -> MarketState {
        let mut features = FeatureRow::new();
        features.insert("z", value);
        MarketState::new(features, 1)
    }

    fn registry() -> StrategyRegistry {
        let mut r = StrategyRegistry::new();
        r.register(
            "momentum",
            ModelId::new("m1"),
            Box::new(ThresholdDecision::new("z", 1.0, -1.0)),
        );
        r.register(
            "mean_revert",
            ModelId::new("m2"),
            Box::new(ThresholdDecision::new("z", 1.0, -1.0).exiting()),
        );
        r
    }

    #[test]
    fn registered_strategies_are_listed_and_looked_up() {
        let r = registry();
        assert_eq!(r.len(), 2);
        assert!(r.contains("momentum"));
        assert!(!r.contains("nope"));
        assert_eq!(
            r.names().collect::<Vec<_>>(),
            vec!["mean_revert", "momentum"]
        );
        assert_eq!(r.model_for("momentum"), Some(&ModelId::new("m1")));
    }

    #[test]
    fn applying_a_strategy_moves_the_position() {
        let r = registry();
        let mut s = state(2.0);
        assert_eq!(r.apply("momentum", &mut s).unwrap(), Position::Long);
    }

    #[test]
    fn unknown_strategy_is_an_error_not_a_silent_hold() {
        let r = registry();
        let mut s = state(2.0);
        assert!(r.apply("typo", &mut s).is_err());
        assert_eq!(s.position, Position::Flat, "nothing traded on the typo");
    }

    #[test]
    fn re_registering_replaces_the_previous_engine() {
        let mut r = registry();
        r.register(
            "momentum",
            ModelId::new("m9"),
            Box::new(ThresholdDecision::new("z", 100.0, -100.0)),
        );
        let mut s = state(2.0);
        assert_eq!(r.apply("momentum", &mut s).unwrap(), Position::Flat);
        assert_eq!(r.model_for("momentum"), Some(&ModelId::new("m9")));
    }

    #[test]
    fn exit_rule_flattens_an_open_position() {
        let r = registry();
        let mut s = state(2.0);
        r.apply("mean_revert", &mut s).unwrap();
        let mut s2 = state(0.0);
        s2.position = Position::Long;
        assert_eq!(r.apply("mean_revert", &mut s2).unwrap(), Position::Flat);
    }
}
