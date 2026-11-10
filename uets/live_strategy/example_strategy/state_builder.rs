//! Threshold rules for the reference strategy.
//!
//! Pure: computed features in, `Decision` out. Static rules today; read your
//! own `model.onnx` here in Phase 2 (load once in `new()`, score in
//! `decide()`, hold on `Missing`).

use crate::core::strategy::{Decision, MarketState, Position};
use crate::live_strategy::StateBuilder;

pub struct ExampleRules {
    pub feature: String,
    pub long_above: f64,
    pub short_below: f64,
}

impl ExampleRules {
    pub fn new() -> Self {
        Self {
            feature: "dz".to_string(),
            long_above: 1.0,
            short_below: -1.0,
        }
    }
}

impl Default for ExampleRules {
    fn default() -> Self {
        Self::new()
    }
}

impl StateBuilder for ExampleRules {
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
            Some(v) => features.insert("dz", v),
            None => features.insert_missing("dz"),
        }
        MarketState {
            features,
            position: Position::Flat,
            sequence: 1,
        }
    }

    #[test]
    fn signals_on_deviation() {
        let r = ExampleRules::new();
        assert_eq!(r.decide(&state(Some(2.0))), Decision::EnterLong);
        assert_eq!(r.decide(&state(Some(-2.0))), Decision::EnterShort);
        assert_eq!(r.decide(&state(Some(0.0))), Decision::Hold);
        assert_eq!(r.decide(&state(None)), Decision::Hold);
    }

    #[test]
    fn does_not_stack_an_entry_while_a_position_is_open() {
        assert_eq!(
            ExampleRules::new().decide(&MarketState {
                features: state(Some(2.0)).features,
                position: Position::Long,
                sequence: 2,
            }),
            Decision::Hold
        );
    }
}
