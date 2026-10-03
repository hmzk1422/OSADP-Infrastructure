//! strategy1/state_builder.rs — YOUR rules.
//!
//! Runs SECOND: computed feature row (inside `MarketState`) in, `Decision`
//! out. Pure function: no clock, no I/O, no position writes (the runner
//! applies the decision to `MarketState` for you).
//!
//! Code a static rule here today. In Phase 2, read your own `model.onnx`
//! here: load the session once in `new()` (path relative to
//! `CARGO_MANIFEST_DIR`, e.g. `live_strategy/strategy1/model.onnx`), build
//! the input tensor from `state.features` in `output_names()` order inside
//! `decide()`, and map the score to a `Decision`. Hold on any `Missing`
//! input — never feed zero/NaN to the model.

use crate::core::strategy::{Decision, MarketState, Position};
use crate::live_strategy::StateBuilder;

/// Example: moving-average crossover band. Replace with your rules.
pub struct Strategy1Rules {
    pub feature: String,
    pub long_above: f64,
    pub short_below: f64,
}

impl Strategy1Rules {
    pub fn new() -> Self {
        Self {
            feature: "ma3".to_string(),
            long_above: 100.0,
            short_below: 90.0,
        }
    }

    // Phase 2 sketch (do not enable until you drop in your model):
    // pub fn with_onnx(model_path: &str) -> Result<Self, String> {
    //     let _session = ort::Session::builder()?.commit_from_file(model_path)
    //         .map_err(|e| e.to_string())?;
    //     // store session + input names (= computer.output_names())
    //     unimplemented!("wire your model.onnx here")
    // }
}

impl Default for Strategy1Rules {
    fn default() -> Self {
        Self::new()
    }
}

impl StateBuilder for Strategy1Rules {
    fn decide(&self, state: &MarketState) -> Decision {
        let Some(value) = state.features.get(&self.feature) else {
            return Decision::Hold;
        };
        let signal = if value > self.long_above {
            Decision::EnterLong
        } else if value < self.short_below {
            Decision::EnterShort
        } else {
            Decision::Hold
        };
        match (signal, state.position) {
            (Decision::EnterLong | Decision::EnterShort, Position::Flat) => signal,
            (Decision::EnterLong | Decision::EnterShort, _) => Decision::Hold,
            (Decision::Exit, Position::Flat) => Decision::Hold,
            (Decision::Exit, _) => Decision::Exit,
            (Decision::Hold, _) => Decision::Hold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::feature::FeatureRow;

    fn state(value: Option<f64>) -> MarketState {
        let mut features = FeatureRow::new();
        match value {
            Some(v) => features.insert("ma3", v),
            None => features.insert_missing("ma3"),
        }
        MarketState {
            features,
            position: Position::Flat,
            sequence: 1,
        }
    }

    #[test]
    fn long_above_band() {
        let r = Strategy1Rules::new();
        assert_eq!(r.decide(&state(Some(101.0))), Decision::EnterLong);
    }

    #[test]
    fn missing_holds() {
        let r = Strategy1Rules::new();
        assert_eq!(r.decide(&state(None)), Decision::Hold);
    }
}
