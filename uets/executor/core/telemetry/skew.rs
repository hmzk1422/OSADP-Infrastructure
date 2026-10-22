//! Clock skew: our clock minus the venue's clock.
//!
//! Latency answers "how late is this data?"; skew answers a different
//! question — "do the two clocks even agree?". A feed can be fast and still
//! be wrong if our clock is off by two seconds, because every event time we
//! compute (session buckets, event ages, reorder windows) inherits that error.
//!
//! The offset is signed: positive means *our* clock runs ahead of the
//! venue's. `exceeds(limit)` is the alarm — feed it the tolerance you are
//! willing to trade with and it tells you when the run should stop.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// An immutable reading of a [`ClockSkew`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClockSkewSnapshot {
    pub count: u64,
    pub last_us: Option<i64>,
    pub mean_us: Option<f64>,
    /// Largest magnitude observed, sign discarded.
    pub max_abs_us: Option<i64>,
}

impl ClockSkewSnapshot {
    pub fn to_json(self) -> String {
        fn opt(value: Option<i64>) -> String {
            value.map_or_else(|| "null".to_string(), |v| v.to_string())
        }
        format!(
            "{{\"count\":{},\"last_us\":{},\"mean_us\":{},\"max_abs_us\":{}}}",
            self.count,
            opt(self.last_us),
            super::json_number(self.mean_us.unwrap_or(f64::NAN)),
            opt(self.max_abs_us),
        )
    }
}

/// Accumulates clock offsets. Safe to update from any thread.
pub struct ClockSkew {
    count: AtomicU64,
    last_us: AtomicI64,
    sum_us: AtomicI64,
    max_abs_us: AtomicU64,
}

impl ClockSkew {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            last_us: AtomicI64::new(0),
            sum_us: AtomicI64::new(0),
            max_abs_us: AtomicU64::new(0),
        }
    }

    /// Record one observation: `local - event`, in milliseconds.
    pub fn record(&self, event_time_ms: u64, local_ms: u64) {
        let delta_us = (local_ms as i64).saturating_sub(event_time_ms as i64);
        self.record_us(delta_us.saturating_mul(1_000));
    }

    /// Record an offset already measured in microseconds. May be negative.
    pub fn record_us(&self, offset_us: i64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.last_us.store(offset_us, Ordering::Relaxed);
        self.sum_us.fetch_add(offset_us, Ordering::Relaxed);
        self.max_abs_us
            .fetch_max(offset_us.unsigned_abs(), Ordering::Relaxed);
    }

    /// The most recent offset, or `None` before the first observation.
    pub fn last_us(&self) -> Option<i64> {
        (self.count.load(Ordering::Relaxed) > 0).then(|| self.last_us.load(Ordering::Relaxed))
    }

    /// Has the clock drifted further than `limit_us` (a non-negative
    /// tolerance) at any point in this run?
    pub fn exceeds(&self, limit_us: i64) -> bool {
        self.max_abs_us.load(Ordering::Relaxed) > limit_us.unsigned_abs()
    }

    pub fn snapshot(&self) -> ClockSkewSnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum_us.load(Ordering::Relaxed);
        let max_abs = self.max_abs_us.load(Ordering::Relaxed);
        let seen = count > 0;
        ClockSkewSnapshot {
            count,
            last_us: seen.then(|| self.last_us.load(Ordering::Relaxed)),
            mean_us: seen.then(|| sum as f64 / count as f64),
            max_abs_us: seen.then_some(max_abs as i64),
        }
    }
}

impl Default for ClockSkew {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_offset_is_signed_and_positive_means_we_run_ahead() {
        let s = ClockSkew::new();
        s.record(1_000, 1_005);
        assert_eq!(s.last_us(), Some(5_000), "+5 ms");
        s.record(2_000, 1_990);
        assert_eq!(s.last_us(), Some(-10_000), "-10 ms: the venue is ahead");
    }

    #[test]
    fn max_abs_keeps_the_worst_drift_seen() {
        let s = ClockSkew::new();
        s.record_us(5_000);
        s.record_us(-20_000);
        s.record_us(3_000);
        assert_eq!(s.snapshot().max_abs_us, Some(20_000), "magnitude, not sign");
    }

    #[test]
    fn exceeds_is_the_trading_tolerance_alarm() {
        let s = ClockSkew::new();
        assert!(!s.exceeds(1_000), "no observation cannot exceed anything");
        s.record_us(5_000);
        assert!(!s.exceeds(10_000), "5 ms is within a 10 ms tolerance");
        assert!(s.exceeds(1_000), "5 ms breaks a 1 ms tolerance");
    }

    #[test]
    fn empty_skew_reports_nothing_rather_than_zero() {
        let s = ClockSkew::new();
        let snap = s.snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.last_us, None);
        assert_eq!(snap.mean_us, None);
        assert_eq!(snap.max_abs_us, None);
        assert!(snap.to_json().contains("\"last_us\":null"));
    }

    #[test]
    fn mean_spreads_both_directions() {
        let s = ClockSkew::new();
        s.record_us(1_000);
        s.record_us(-1_000);
        assert_eq!(
            s.snapshot().mean_us,
            Some(0.0),
            "a steady clock averages out"
        );
    }
}
