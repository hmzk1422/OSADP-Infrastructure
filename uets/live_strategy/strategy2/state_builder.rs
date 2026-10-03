//! strategy2/state_builder.rs — YOUR rules for strategy2.
//!
//! Pure: computed features in, `Decision` out. Static rules today; read your
//! own `model.onnx` here in Phase 2 (load once in `new()`, score in
//! `decide()`, hold on `Missing`).

use crate::core::strategy::{Decision, MarketState, Position};
use crate::live_strategy::StateBuilder;

/// Example: threshold on `dz`. Replace with your rules.
pub struct Strategy2Rules {
    pub feature: String,
    pub long_above: f64,
    pub short_below: f64,
}

impl Strategy2Rules {
    pub fn new() -> Self {
        Self {
            feature: "dz".to_string(),
            long_above: 1.0,
            short_below: -1.0,
        }
    }
}

impl Default for Strategy2Rules {
    fn default() -> Self {
        Self::new()
    }
}

impl StateBuilder for Strategy2Rules {
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
        let r = Strategy2Rules::new();
        assert_eq!(r.decide(&state(Some(2.0))), Decision::EnterLong);
        assert_eq!(r.decide(&state(Some(-2.0))), Decision::EnterShort);
        assert_eq!(r.decide(&state(Some(0.0))), Decision::Hold);
        assert_eq!(r.decide(&state(None)), Decision::Hold);
    }
}
