//! Feature-driven decision rule.
//!
//! The rule is intentionally a *pure function of the current feature row*: it
//! reads no clock, holds no position, and has no I/O. Position and sequencing
//! live in [`super::MarketState`], so a rule can be unit-tested by handing it a
//! row and asserting one [`Decision`] — which is what makes strategy research
//! possible at all.

use super::MarketState;
use super::state::Position;

/// What the engine wants done for this row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// No change: stay flat, or keep the current position.
    Hold,
    EnterLong,
    EnterShort,
    /// Close whatever position is open.
    Exit,
}

/// Something that turns a feature row into a [`Decision`].
pub trait DecisionEngine: Send + Sync {
    fn decide(&self, state: &MarketState) -> Decision;
}

/// Enter long above `long_above`, short below `short_below`, otherwise hold.
///
/// Refuses to act on a missing feature: a rule that cannot see its input must
/// hold, not guess. Entering a position on absent data is how a backtest ends
/// up looking excellent and a live account ends up flat.
/// `Clone` but not `Copy`: the feature name is a `String`, and a rule that
/// silently copied a heap string per row would allocate on the hot path.
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdDecision {
    pub feature: String,
    pub long_above: f64,
    pub short_below: f64,
    /// Leave an open position when the signal reverts to neutral.
    pub exit_on_hold: bool,
}

impl ThresholdDecision {
    pub fn new(feature: impl Into<String>, long_above: f64, short_below: f64) -> Self {
        Self {
            feature: feature.into(),
            long_above,
            short_below,
            exit_on_hold: false,
        }
    }

    pub fn exiting(mut self) -> Self {
        self.exit_on_hold = true;
        self
    }
}

impl DecisionEngine for ThresholdDecision {
    fn decide(&self, state: &MarketState) -> Decision {
        let Some(value) = state.features.get(&self.feature) else {
            // Absent input, not a zero: hold rather than trade on a gap.
            return Decision::Hold;
        };
        let signal = if value > self.long_above {
            Decision::EnterLong
        } else if value < self.short_below {
            Decision::EnterShort
        } else if self.exit_on_hold {
            Decision::Exit
        } else {
            Decision::Hold
        };
        // Only an *entry* is suppressed while already positioned. An exit must
        // still get through, or `exit_on_hold` could never flatten a position.
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
    use crate::ml_feature::FeatureRow;
    use crate::strategy::Position;

    fn state(value: Option<f64>, position: Position) -> MarketState {
        let mut features = FeatureRow::new();
        match value {
            Some(v) => features.insert("z", v),
            None => features.insert_missing("z"),
        };
        MarketState {
            features,
            position,
            sequence: 1,
        }
    }

    fn rule() -> ThresholdDecision {
        ThresholdDecision::new("z", 1.0, -1.0)
    }

    #[test]
    fn goes_long_above_the_upper_threshold() {
        assert_eq!(
            rule().decide(&state(Some(2.0), Position::Flat)),
            Decision::EnterLong
        );
    }

    #[test]
    fn goes_short_below_the_lower_threshold() {
        assert_eq!(
            rule().decide(&state(Some(-2.0), Position::Flat)),
            Decision::EnterShort
        );
    }

    #[test]
    fn holds_inside_the_band() {
        assert_eq!(
            rule().decide(&state(Some(0.5), Position::Flat)),
            Decision::Hold
        );
    }

    #[test]
    fn holds_when_the_feature_is_missing() {
        // The important one: absent must never trade.
        assert_eq!(rule().decide(&state(None, Position::Flat)), Decision::Hold);
    }

    #[test]
    fn does_not_stack_entries_while_already_positioned() {
        assert_eq!(
            rule().decide(&state(Some(2.0), Position::Long)),
            Decision::Hold
        );
        assert_eq!(
            rule().decide(&state(Some(-2.0), Position::Short)),
            Decision::Hold
        );
    }

    #[test]
    fn exit_is_only_issued_from_an_open_position() {
        let exiting = rule().exiting();
        assert_eq!(
            exiting.decide(&state(Some(0.0), Position::Long)),
            Decision::Exit
        );
        assert_eq!(
            exiting.decide(&state(Some(0.0), Position::Flat)),
            Decision::Hold
        );
    }
}
