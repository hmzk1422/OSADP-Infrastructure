//! OFMR triple-barrier handling.
//!
//! Inspected from `strategy_system/strategy/ofmr/config.py` and
//! `strategy_system/strategy/moe/outcome/triple_barrier.py`:
//! - TP/SL widths are in bps (stop_loss_bps=5, take_profit_bps=5)
//! - Vertical barrier = timeframe_barrier_secs (300s)
//! - Barriers are used for BOTH training labels (Python) and live exits (Rust).
//!
//! The barrier is attached to an entry order and monitored during the trade.
//! On breach, an Exit intent is emitted with the exit reason.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Why a triple-barrier exit occurred.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarrierReason {
    TakeProfit,
    StopLoss,
    Vertical,
}

/// Result of a barrier check: whether to exit and why.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarrierExit {
    pub should_exit: bool,
    pub reason: Option<BarrierReason>,
    pub exit_price: f64,
}

/// Triple-barrier configuration attached to an entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TripleBarrier {
    /// Entry price.
    pub entry_price: f64,
    /// Entry timestamp.
    pub entry_time: DateTime<Utc>,
    /// Take-profit barrier as a fraction of entry (e.g. 0.0005 = 5 bps).
    pub tp_fraction: f64,
    /// Stop-loss barrier as a fraction of entry (e.g. 0.0005 = 5 bps).
    pub sl_fraction: f64,
    /// Maximum holding duration.
    pub max_hold_seconds: u64,
    /// Position side: true = long, false = short.
    pub is_long: bool,
    /// Actual TP price (entry_price * (1 + tp_fraction) for long).
    pub tp_price: f64,
    /// Actual SL price (entry_price * (1 - sl_fraction) for long).
    pub sl_price: f64,
    /// Vertical barrier timestamp (entry_time + max_hold_seconds).
    pub vertical_time: DateTime<Utc>,
}

impl TripleBarrier {
    /// Build barriers for a long position.
    pub fn long(
        entry_price: f64,
        entry_time: DateTime<Utc>,
        tp_bps: f64,
        sl_bps: f64,
        max_hold_seconds: u64,
    ) -> Self {
        let tp_fraction = tp_bps / 10_000.0;
        let sl_fraction = sl_bps / 10_000.0;
        Self {
            entry_price,
            entry_time,
            tp_fraction,
            sl_fraction,
            max_hold_seconds,
            is_long: true,
            tp_price: entry_price * (1.0 + tp_fraction),
            sl_price: entry_price * (1.0 - sl_fraction),
            vertical_time: entry_time + chrono::Duration::seconds(max_hold_seconds as i64),
        }
    }

    /// Build barriers for a short position.
    pub fn short(
        entry_price: f64,
        entry_time: DateTime<Utc>,
        tp_bps: f64,
        sl_bps: f64,
        max_hold_seconds: u64,
    ) -> Self {
        let tp_fraction = tp_bps / 10_000.0;
        let sl_fraction = sl_bps / 10_000.0;
        Self {
            entry_price,
            entry_time,
            tp_fraction,
            sl_fraction,
            max_hold_seconds,
            is_long: false,
            tp_price: entry_price * (1.0 - tp_fraction),
            sl_price: entry_price * (1.0 + sl_fraction),
            vertical_time: entry_time + chrono::Duration::seconds(max_hold_seconds as i64),
        }
    }

    /// Check the current mark price and time against all three barriers.
    /// Price barriers (TP/SL) take precedence over the vertical barrier.
    pub fn check(&self, mark_price: f64, now: DateTime<Utc>) -> BarrierExit {
        if self.is_long {
            if mark_price >= self.tp_price {
                return BarrierExit {
                    should_exit: true,
                    reason: Some(BarrierReason::TakeProfit),
                    exit_price: mark_price,
                };
            }
            if mark_price <= self.sl_price {
                return BarrierExit {
                    should_exit: true,
                    reason: Some(BarrierReason::StopLoss),
                    exit_price: mark_price,
                };
            }
        } else {
            if mark_price <= self.tp_price {
                return BarrierExit {
                    should_exit: true,
                    reason: Some(BarrierReason::TakeProfit),
                    exit_price: mark_price,
                };
            }
            if mark_price >= self.sl_price {
                return BarrierExit {
                    should_exit: true,
                    reason: Some(BarrierReason::StopLoss),
                    exit_price: mark_price,
                };
            }
        }

        // Vertical barrier (maximum holding time) is checked last.
        if now >= self.vertical_time {
            return BarrierExit {
                should_exit: true,
                reason: Some(BarrierReason::Vertical),
                exit_price: mark_price,
            };
        }

        BarrierExit {
            should_exit: false,
            reason: None,
            exit_price: mark_price,
        }
    }

    /// The risk distance from entry to stop, per unit.
    pub fn risk_per_unit(&self) -> f64 {
        (self.entry_price - self.sl_price).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn entry_time() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn long_tp_hit() {
        let tb = TripleBarrier::long(100.0, entry_time(), 5.0, 5.0, 300);
        let exit = tb.check(100.501, entry_time());
        assert!(exit.should_exit);
        assert_eq!(exit.reason, Some(BarrierReason::TakeProfit));
    }

    #[test]
    fn long_sl_hit() {
        let tb = TripleBarrier::long(100.0, entry_time(), 5.0, 5.0, 300);
        let exit = tb.check(99.499, entry_time());
        assert!(exit.should_exit);
        assert_eq!(exit.reason, Some(BarrierReason::StopLoss));
    }

    #[test]
    fn short_tp_hit() {
        let tb = TripleBarrier::short(100.0, entry_time(), 5.0, 5.0, 300);
        let exit = tb.check(99.499, entry_time());
        assert!(exit.should_exit);
        assert_eq!(exit.reason, Some(BarrierReason::TakeProfit));
    }

    #[test]
    fn short_sl_hit() {
        let tb = TripleBarrier::short(100.0, entry_time(), 5.0, 5.0, 300);
        let exit = tb.check(100.501, entry_time());
        assert!(exit.should_exit);
        assert_eq!(exit.reason, Some(BarrierReason::StopLoss));
    }

    #[test]
    fn vertical_barrier_hit() {
        let entry = entry_time();
        let tb = TripleBarrier::long(100.0, entry, 5.0, 5.0, 10);
        let late = entry + chrono::Duration::seconds(11);
        let exit = tb.check(100.0, late);
        assert!(exit.should_exit);
        assert_eq!(exit.reason, Some(BarrierReason::Vertical));
    }

    #[test]
    fn no_barrier_hit() {
        let tb = TripleBarrier::long(100.0, entry_time(), 5.0, 5.0, 300);
        let exit = tb.check(100.02, entry_time());
        assert!(!exit.should_exit);
        assert_eq!(exit.reason, None);
    }

    #[test]
    fn tp_wins_tie_over_vertical() {
        let entry = entry_time();
        let tb = TripleBarrier::long(100.0, entry, 0.1, 5.0, 1);
        let late = entry + chrono::Duration::seconds(2);
        let exit = tb.check(100.001, late);
        assert!(exit.should_exit);
        assert_eq!(exit.reason, Some(BarrierReason::TakeProfit));
    }
}
