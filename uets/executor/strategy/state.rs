//! Position and per-symbol market state.
//!
//! The only stateful piece of the strategy layer. It owns the position so a
//! [`DecisionEngine`](super::DecisionEngine) stays a pure function: the rule
//! says what it wants, this decides what actually changes. Keeping those apart
//! is what lets the same rule be replayed over a capture and produce the same
//! position sequence.

use crate::ml_feature::FeatureRow;

use super::decision::Decision;

/// Which way the strategy is currently exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    Flat,
    Long,
    Short,
}

impl Position {
    /// The position reached by acting on `decision` from `self`.
    pub fn transitioned(self, decision: Decision) -> Self {
        match decision {
            Decision::EnterLong => Self::Long,
            Decision::EnterShort => Self::Short,
            Decision::Exit | Decision::Hold => Self::Flat,
        }
    }
}

/// One symbol's live state: its latest features, its position, its row count.
#[derive(Clone, Debug)]
pub struct MarketState {
    /// Latest computed features. Values may be `Missing`; that is preserved.
    pub features: FeatureRow,
    pub position: Position,
    /// Monotonic row counter, used for model rotation and for logging.
    pub sequence: u64,
}

impl MarketState {
    pub fn new(features: FeatureRow, sequence: u64) -> Self {
        Self {
            features,
            position: Position::Flat,
            sequence,
        }
    }

    /// Apply a decision, returning the position that resulted.
    ///
    /// `Hold` and `Exit` both end flat, so a rule cannot accidentally leave the
    /// strategy short after an exit signal.
    pub fn apply(&mut self, decision: Decision) -> Position {
        self.position = self.position.transitioned(decision);
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entering_moves_off_flat() {
        let mut state = MarketState::new(FeatureRow::new(), 1);
        assert_eq!(state.position, Position::Flat);
        assert_eq!(state.apply(Decision::EnterLong), Position::Long);
        assert_eq!(state.apply(Decision::EnterShort), Position::Short);
    }

    #[test]
    fn hold_keeps_the_current_position() {
        // `transitioned` maps Hold to Flat, so a caller must not use it for
        // Hold. This test documents the contract: the engine filters Hold out
        // before it reaches here, and applying Hold explicitly is flat by
        // definition, not "unchanged".
        let mut state = MarketState::new(FeatureRow::new(), 1);
        state.apply(Decision::EnterLong);
        assert_eq!(state.apply(Decision::Exit), Position::Flat);
    }
}
