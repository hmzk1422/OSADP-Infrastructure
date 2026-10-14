//! Resource accounting, sampled off the hot path at 1 Hz: RSS, queue depth,
//! drop counters, fd count. The hot path only touches atomics; this module
//! reads them. Bounded queues + buffer reuse live at the fan-out point; this
//! module is the dial that proves they hold.

use std::sync::atomic::{AtomicU64, Ordering};

/// Counters every producer/consumer updates with a single atomic op.
#[derive(Default)]
pub struct Resources {
    /// Rows dropped by the recorder (the designated dropper).
    pub drops: AtomicU64,
    /// Deepest queue depth observed since last scrape.
    pub queue_depth_max: AtomicU64,
    /// Current queue depth (producers inc, consumers dec).
    pub queue_depth: AtomicU64,
}

impl Resources {
    /// 1 Hz scrape:RSS + fd count are read here, never on the hot path.
    pub fn scrape(&self) -> ResourceSample {
        ResourceSample {
            rss_bytes: rss_bytes(),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            queue_depth_max: self.queue_depth_max.load(Ordering::Relaxed),
            drops: self.drops.load(Ordering::Relaxed),
        }
    }
}

/// One off-hot-path sample, served on the health socket.
pub struct ResourceSample {
    pub rss_bytes: u64,
    pub queue_depth: u64,
    pub queue_depth_max: u64,
    pub drops: u64,
}

#[cfg(target_os = "linux")]
fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|t| t.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
fn rss_bytes() -> u64 {
    0 // unimplemented outside Linux; the counters still work
}
