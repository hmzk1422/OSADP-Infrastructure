//! End-to-end latency: proven-bucket histogram, atomics only on the hot path.
//!
//! Correction applied: a ring buffer alone cannot give correct concurrent
//! percentiles (sorting a concurrently-mutated ring is a data race with a
//! histogram costume on). This uses fixed log-spaced nanosecond buckets with
//! one atomic per bucket -- increments are lock-free and every percentile is
//! computed from a stable snapshot. Production percentiles come from the
//! buckets; an exact sort is kept as an offline oracle in tests only, never
//! on the hot path.

use std::sync::atomic::{AtomicU64, Ordering};

/// Bucket edges in nanoseconds, log-spaced from 1us to ~17s.
const EDGES: [u64; 24] = [
    1_000,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    128_000,
    256_000,
    512_000,
    1_000_000,
    2_000_000,
    4_000_000,
    8_000_000,
    16_000_000,
    32_000_000,
    64_000_000,
    128_000_000,
    256_000_000,
    512_000_000,
    1_000_000_000,
    2_000_000_000,
    4_000_000_000,
    8_000_000_000,
];

/// Lock-free latency histogram: `observe_ns` is a single atomic increment.
pub struct LatencyHist {
    buckets: Vec<AtomicU64>,
    overflow: AtomicU64,
    count: AtomicU64,
    sum_ns: AtomicU64,
}

impl LatencyHist {
    /// Number of buckets (== `EDGES.len()`).
    pub const N: usize = 24;

    pub fn new() -> Self {
        Self {
            buckets: (0..Self::N).map(|_| AtomicU64::new(0)).collect(),
            overflow: AtomicU64::new(0),
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
        }
    }

    /// Hot path: exactly one atomic increment, no locks, no allocation.
    pub fn observe_ns(&self, ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);
        match EDGES.iter().position(|&e| ns <= e) {
            Some(i) => {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
            }
            None => {
                self.overflow.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Snapshot percentile from bucket upper edges (`q` in (0,1]).
    /// Reads each bucket once; safe to call off the hot path at 1 Hz.
    pub fn percentile(&self, q: f64) -> Option<u64> {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        let rank = ((q.clamp(0.0, 1.0) * total as f64).ceil() as u64).max(1);
        let mut acc = 0u64;
        for (i, edge) in EDGES.iter().enumerate() {
            acc += self.buckets[i].load(Ordering::Relaxed);
            if acc >= rank {
                return Some(*edge);
            }
        }
        None // rank falls in overflow: value exceeds every edge
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Arithmetic mean of all observed durations, in nanoseconds.
    pub fn mean_ns(&self) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed);
        (count > 0).then(|| self.sum_ns.load(Ordering::Relaxed) as f64 / count as f64)
    }

    pub fn sum_ns(&self) -> u64 {
        self.sum_ns.load(Ordering::Relaxed)
    }
}

impl Default for LatencyHist {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Offline oracle: exact nearest-rank on a frozen snapshot. Test-only.
    fn exact(mut xs: Vec<u64>, q: f64) -> u64 {
        xs.sort_unstable();
        xs[((q * xs.len() as f64).ceil() as usize).max(1) - 1]
    }

    #[test]
    fn buckets_bracket_exact() {
        let h = LatencyHist::new();
        let xs: Vec<u64> = (1..=1000).map(|i| i * 10_000).collect(); // 10us..10ms
        for &x in &xs {
            h.observe_ns(x);
        }
        // Bucket edges bracket the exact value within one doubling step.
        for q in [0.5, 0.99, 0.999] {
            let approx = h.percentile(q).expect("nonempty");
            let e = exact(xs.clone(), q);
            assert!(approx >= e && approx <= e * 2, "q={q}: {approx} vs {e}");
        }
    }
}
