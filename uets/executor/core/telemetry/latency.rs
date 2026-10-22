//! Feed latency: when an event claims it happened vs when we actually saw it.
//!
//! One number per frame — `received - event` — kept as a running distribution
//! (last / min / max / mean) so a degraded feed shows up as a climbing mean
//! instead of a vague feeling that the tape is sluggish. Negative values are
//! recorded, not clamped: an event stamped *after* we received it means the
//! two clocks disagree, and hiding that would be lying about the data.
//!
//! Thread-safe by atomics, like [`super::Telemetry`]: the socket thread
//! records, any reporter reads.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// An immutable reading of a [`LatencyTracker`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LatencySnapshot {
    pub count: u64,
    pub last_us: Option<i64>,
    pub min_us: Option<i64>,
    pub max_us: Option<i64>,
    pub mean_us: Option<f64>,
}

impl LatencySnapshot {
    pub fn to_json(self) -> String {
        fn opt(value: Option<i64>) -> String {
            value.map_or_else(|| "null".to_string(), |v| v.to_string())
        }
        format!(
            "{{\"count\":{},\"last_us\":{},\"min_us\":{},\"max_us\":{},\"mean_us\":{}}}",
            self.count,
            opt(self.last_us),
            opt(self.min_us),
            opt(self.max_us),
            super::json_number(self.mean_us.unwrap_or(f64::NAN)),
        )
    }
}

/// Accumulates frame latencies. Safe to update from any thread.
pub struct LatencyTracker {
    count: AtomicU64,
    last_us: AtomicI64,
    min_us: AtomicI64,
    max_us: AtomicI64,
    sum_us: AtomicI64,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            last_us: AtomicI64::new(0),
            min_us: AtomicI64::new(i64::MAX),
            max_us: AtomicI64::new(i64::MIN),
            sum_us: AtomicI64::new(0),
        }
    }

    /// Record one frame, given the event's timestamp and our receive time.
    pub fn record(&self, event_time_ms: u64, received_ms: u64) {
        let delta_us = (received_ms as i64).saturating_sub(event_time_ms as i64);
        self.record_us(delta_us.saturating_mul(1_000));
    }

    /// Record a latency already measured in microseconds. May be negative.
    pub fn record_us(&self, delta_us: i64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.last_us.store(delta_us, Ordering::Relaxed);
        self.sum_us.fetch_add(delta_us, Ordering::Relaxed);
        self.min_us.fetch_min(delta_us, Ordering::Relaxed);
        self.max_us.fetch_max(delta_us, Ordering::Relaxed);
    }

    pub fn last_us(&self) -> Option<i64> {
        (self.count.load(Ordering::Relaxed) > 0).then(|| self.last_us.load(Ordering::Relaxed))
    }

    pub fn snapshot(&self) -> LatencySnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let min = self.min_us.load(Ordering::Relaxed);
        let max = self.max_us.load(Ordering::Relaxed);
        let sum = self.sum_us.load(Ordering::Relaxed);
        let seen = count > 0;
        LatencySnapshot {
            count,
            last_us: seen.then(|| self.last_us.load(Ordering::Relaxed)),
            min_us: (seen && min != i64::MAX).then_some(min),
            max_us: (seen && max != i64::MIN).then_some(max),
            mean_us: seen.then(|| sum as f64 / count as f64),
        }
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_last_min_max_and_mean() {
        let t = LatencyTracker::new();
        t.record_us(1_000);
        t.record_us(3_000);
        t.record_us(2_000);
        let s = t.snapshot();
        assert_eq!(s.count, 3);
        assert_eq!(s.last_us, Some(2_000));
        assert_eq!(s.min_us, Some(1_000));
        assert_eq!(s.max_us, Some(3_000));
        assert_eq!(s.mean_us, Some(2_000.0));
    }

    #[test]
    fn an_empty_tracker_reports_nothing_rather_than_zero() {
        let t = LatencyTracker::new();
        let s = t.snapshot();
        assert_eq!(s.count, 0);
        assert_eq!(s.last_us, None, "no observation is None, not 0");
        assert_eq!(s.min_us, None);
        assert_eq!(s.max_us, None);
        assert_eq!(s.mean_us, None);
        let json = s.to_json();
        assert!(json.contains("\"last_us\":null"));
        assert!(json.contains("\"mean_us\":null"));
    }

    #[test]
    fn negative_latency_is_kept_visible_not_clamped() {
        let t = LatencyTracker::new();
        t.record_us(-500);
        let s = t.snapshot();
        assert_eq!(s.min_us, Some(-500), "clock trouble must stay visible");
        assert_eq!(s.last_us, Some(-500));
    }

    #[test]
    fn timestamps_convert_into_microseconds() {
        let t = LatencyTracker::new();
        t.record(1_000, 1_010);
        assert_eq!(t.last_us(), Some(10_000), "10 ms is 10_000 us");
    }
}
