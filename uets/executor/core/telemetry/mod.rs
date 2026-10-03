//! Telemetry: the counters, the std-only JSON writer they are logged with,
//! and the two diagnostic stores that answer "how late?" (latency) and
//! "do the clocks agree?" (skew).
//!
//! Three jobs, deliberately kept together so the JSON writer has exactly one
//! home:
//!
//! * [`Telemetry`] — atomic counters a live loop increments from any thread, and
//!   [`TelemetrySnapshot`] — the immutable reading of them a reporter takes.
//! * [`latency`] — [`latency::LatencyTracker`]: event time vs local arrival,
//!   as min/max/mean/last over the run.
//! * [`skew`] — [`skew::ClockSkew`]: our clock minus the venue's clock, signed,
//!   with a worst-observed magnitude and an `exceeds` alarm.
//! * [`write_json_atomic`] plus the escaping helpers, used here and by
//!   [`super::gui`] to publish state to a reader that only ever sees whole files.
//!
//! Why no `serde_json`: a dashboard should be able to read the live numbers with
//! nothing but `std`, the same std-only property the live merger holds. Hand-writing
//! the few objects below is cheaper than that dependency, and it keeps number
//! formatting under our control — which matters, because `NaN` and `inf` are not
//! JSON, and both are reachable from an equity curve.

pub mod latency;
pub mod skew;

pub use latency::{LatencySnapshot, LatencyTracker};
pub use skew::{ClockSkew, ClockSkewSnapshot};

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Wall-clock milliseconds since the Unix epoch.
///
/// `unwrap_or_default` rather than `unwrap`: a clock set before 1970 is absurd
/// but not worth crashing a live trading loop over.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A number as JSON, or `null` when it is not finite.
///
/// `inf` and `NaN` have no JSON spelling. Writing `null` (and letting the
/// dashboard show `--`) is honest; writing `0` would be a lie a viewer cannot
/// detect.
pub fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_string()
    }
}

/// A string as a JSON literal, quotes included.
pub fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Write `contents` to `path` so a concurrent reader never sees a half file.
///
/// Write to a sibling temp file, then move it into place. On Windows
/// `fs::rename` refuses to overwrite an existing destination, which is why the
/// old file is removed first — the reader may miss one frame in that window, but
/// it can never read a torn one.
pub fn write_json_atomic(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let tmp = path.with_file_name(format!("{}.tmp", file_name.to_string_lossy()));
    fs::write(&tmp, contents)?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(&tmp, path)
}
/// Live counters, safe to update from any thread.
///
/// Every field is an atomic because the loops that increment them (the socket
/// reader, the strategy, the order path) are separate threads, and telemetry
/// must never be the thing that serialises them.
#[derive(Debug)]
pub struct Telemetry {
    start_ms: u64,
    rows: AtomicU64,
    orders: AtomicU64,
    fills: AtomicU64,
    rejects: AtomicU64,
    errors: AtomicU64,
    equity_marks: AtomicU64,
    last_event_ms: AtomicU64,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            start_ms: now_ms(),
            rows: AtomicU64::new(0),
            orders: AtomicU64::new(0),
            fills: AtomicU64::new(0),
            rejects: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            equity_marks: AtomicU64::new(0),
            last_event_ms: AtomicU64::new(0),
        }
    }

    pub fn start_ms(&self) -> u64 {
        self.start_ms
    }

    /// One market-data row folded into the feature table.
    pub fn record_row(&self) {
        self.rows.fetch_add(1, Ordering::Relaxed);
    }

    /// One order handed to a venue, whether or not it filled.
    pub fn record_order(&self) {
        self.orders.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_fills(&self, count: u64) {
        self.fills.fetch_add(count, Ordering::Relaxed);
    }

    /// The venue said no. Distinct from [`Self::record_error`]: a reject is an
    /// answer, not a malfunction.
    pub fn record_reject(&self) {
        self.rejects.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_equity_mark(&self) {
        self.equity_marks.fetch_add(1, Ordering::Relaxed);
    }

    /// Note that the feed produced something at `at_ms`.
    ///
    /// Kept as a max rather than a store so two threads racing cannot make the
    /// feed look older than it is — a stale feed is what triggers a flatten.
    pub fn record_event(&self, at_ms: u64) {
        self.last_event_ms.fetch_max(at_ms, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        self.snapshot_at(now_ms())
    }

    /// Read every counter as of `at_ms`, so the derived numbers are consistent.
    pub fn snapshot_at(&self, at_ms: u64) -> TelemetrySnapshot {
        let last_event_ms = self.last_event_ms.load(Ordering::Relaxed);
        let uptime_ms = at_ms.saturating_sub(self.start_ms);
        let rows = self.rows.load(Ordering::Relaxed);
        TelemetrySnapshot {
            uptime_ms,
            rows,
            orders: self.orders.load(Ordering::Relaxed),
            fills: self.fills.load(Ordering::Relaxed),
            rejects: self.rejects.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            equity_marks: self.equity_marks.load(Ordering::Relaxed),
            last_event_ms: (last_event_ms != 0).then_some(last_event_ms),
            feed_age_ms: (last_event_ms != 0).then(|| at_ms.saturating_sub(last_event_ms)),
            rows_per_sec: if uptime_ms > 0 {
                rows as f64 / (uptime_ms as f64 / 1000.0)
            } else {
                0.0
            },
        }
    }

    /// The counters as one JSON object.
    pub fn to_json(&self) -> String {
        self.snapshot().to_json()
    }

    /// Publish the counters to `path` for the dashboard.
    pub fn write_json(&self, path: &Path) -> io::Result<()> {
        write_json_atomic(path, &self.to_json())
    }
}
/// An immutable reading of [`Telemetry`].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct TelemetrySnapshot {
    pub uptime_ms: u64,
    pub rows: u64,
    pub orders: u64,
    pub fills: u64,
    pub rejects: u64,
    pub errors: u64,
    pub equity_marks: u64,
    /// `None` until the first market event arrives.
    pub last_event_ms: Option<u64>,
    /// How long ago the last event arrived; `None` if there never was one.
    pub feed_age_ms: Option<u64>,
    pub rows_per_sec: f64,
}

impl TelemetrySnapshot {
    pub fn to_json(&self) -> String {
        let last = match self.last_event_ms {
            Some(ms) => ms.to_string(),
            None => "null".to_string(),
        };
        let age = match self.feed_age_ms {
            Some(ms) => ms.to_string(),
            None => "null".to_string(),
        };
        format!(
            concat!(
                "{{\"uptime_ms\":{},\"rows\":{},\"orders\":{},\"fills\":{},\"rejects\":{},",
                "\"errors\":{},\"equity_marks\":{},\"last_event_ms\":{},\"feed_age_ms\":{},",
                "\"rows_per_sec\":{}}}"
            ),
            self.uptime_ms,
            self.rows,
            self.orders,
            self.fills,
            self.rejects,
            self.errors,
            self.equity_marks,
            last,
            age,
            json_number(self.rows_per_sec),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    /// A unique scratch path per test, so a parallel run cannot collide.
    fn scratch(name: &str) -> std::path::PathBuf {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        std::env::temp_dir()
            .join(format!("uets-telemetry-{}", std::process::id()))
            .join(format!(
                "{}-{}",
                name,
                SEQ.fetch_add(1, AtomicOrdering::Relaxed)
            ))
    }

    #[test]
    fn counters_accumulate_independently() {
        let t = Telemetry::new();
        t.record_row();
        t.record_row();
        t.record_order();
        t.record_fills(3);
        t.record_reject();
        t.record_error();
        t.record_equity_mark();
        let s = t.snapshot();
        assert_eq!((s.rows, s.orders, s.fills), (2, 1, 3));
        assert_eq!((s.rejects, s.errors, s.equity_marks), (1, 1, 1));
    }

    #[test]
    fn a_feed_that_has_never_spoken_reports_no_age_at_all() {
        let t = Telemetry::new();
        let s = t.snapshot_at(t.start_ms() + 5_000);
        assert_eq!(s.last_event_ms, None);
        assert_eq!(s.feed_age_ms, None, "silence is not an age of zero");
        assert_eq!(s.uptime_ms, 5_000);
        assert_eq!(s.rows_per_sec, 0.0);
        assert!(s.to_json().contains("\"feed_age_ms\":null"));
    }

    #[test]
    fn the_newest_event_wins_whichever_thread_reports_it() {
        let t = Telemetry::new();
        t.record_event(1_000);
        t.record_event(900);
        let s = t.snapshot_at(1_500);
        assert_eq!(s.last_event_ms, Some(1_000), "an older stamp cannot win");
        assert_eq!(s.feed_age_ms, Some(500));
    }

    #[test]
    fn row_rate_is_rows_over_elapsed_seconds() {
        let t = Telemetry::new();
        for _ in 0..10 {
            t.record_row();
        }
        let s = t.snapshot_at(t.start_ms() + 2_000);
        assert!((s.rows_per_sec - 5.0).abs() < 1e-12);
    }

    #[test]
    fn non_finite_numbers_become_null_and_strings_are_escaped() {
        assert_eq!(json_number(f64::INFINITY), "null");
        assert_eq!(json_number(f64::NAN), "null");
        assert_eq!(json_number(1.5), "1.5");
        assert_eq!(json_string("BTCUSDT"), "\"BTCUSDT\"");
        assert_eq!(json_string("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(json_string("l1\nl2"), "\"l1\\nl2\"");
        assert_eq!(json_string("bell\u{7}"), "\"bell\\u0007\"");
    }

    #[test]
    fn writing_publishes_a_whole_file_and_leaves_no_temp_behind() {
        let path = scratch("telemetry.json");
        let t = Telemetry::new();
        t.record_row();
        t.write_json(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.starts_with('{') && text.ends_with('}'));
        assert!(text.contains("\"rows\":1"));
        assert!(!path.with_file_name("telemetry.json.tmp").exists());

        // A second write replaces the frame rather than appending to it.
        t.record_row();
        t.write_json(&path).unwrap();
        assert!(fs::read_to_string(&path).unwrap().contains("\"rows\":2"));
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
}
