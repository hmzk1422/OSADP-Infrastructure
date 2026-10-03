//! Folder discovery for strategy bundles.
//!
//! `executor.rs` intersects [`discover_strategy_dirs`] with
//! [`available_strategies`]: a folder with no compiled module means "add
//! `pub mod` + `load_strategy` arm, rebuild".

use std::collections::BTreeMap;

use crate::core::feature::FeatureRow;
use crate::core::strategy::{Decision, MarketState, Position};
use crate::live_strategy::{LiveStrategy, load_strategy};

/// Folders under `uets/live_strategy/` holding a strategy bundle.
pub fn discover_strategy_dirs() -> Vec<String> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("live_strategy");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with('.') || name == "target" {
            continue;
        }
        if path.join("mod.rs").is_file() {
            out.push(name.to_string());
        }
    }
    out.sort();
    out
}

/// One slot: YOUR computer + YOUR builder + isolated position/sequence.
struct Slot {
    computer: Box<dyn crate::live_strategy::ComputeFeatures>,
    builder: Box<dyn crate::live_strategy::StateBuilder>,
    state: MarketState,
}

/// Per-(strategy x symbol) live state; windows/positions never leak.
pub struct StrategyRunner {
    slots: BTreeMap<(String, String), Slot>,
}

impl StrategyRunner {
    pub fn new() -> Self {
        Self {
            slots: BTreeMap::new(),
        }
    }

    fn ensure_slot(&mut self, strategy: &str, symbol: &str) -> Result<&mut Slot, String> {
        let key = (strategy.to_string(), symbol.to_string());
        if !self.slots.contains_key(&key) {
            let bundle: LiveStrategy = load_strategy(strategy)?;
            self.slots.insert(
                key.clone(),
                Slot {
                    computer: bundle.computer,
                    builder: bundle.builder,
                    state: MarketState::new(FeatureRow::new(), 0),
                },
            );
        }
        self.slots
            .get_mut(&key)
            .ok_or_else(|| format!("missing slot for '{strategy}/{symbol}'"))
    }

    /// YOUR pipeline: `compute_feature` first, `state_builder` second.
    pub fn step(
        &mut self,
        strategy: &str,
        symbol: &str,
        row: &FeatureRow,
    ) -> Result<(Decision, Position), String> {
        let slot = self.ensure_slot(strategy, symbol)?;
        let computed = slot.computer.compute(row)?;
        slot.state.features = computed;
        slot.state.sequence += 1;
        let decision = slot.builder.decide(&slot.state);
        let position = slot.state.apply(decision);
        Ok((decision, position))
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

impl Default for StrategyRunner {
    fn default() -> Self {
        Self::new()
    }
}
