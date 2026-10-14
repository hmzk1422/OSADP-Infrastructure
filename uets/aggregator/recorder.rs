//! Recorder: market-data capture. CSV today, RocksDB when the replay path needs it.
//!
//! Correction applied: drops are explicit. The recorder is the designated
//! dropper in the pipeline (bounded queue + `try_send`), and every drop is
//! counted AND marks the recording incomplete -- a capture that lost an event
//! is never presented as a complete replay dataset. Consumers check
//! [`RecorderState::complete`] before trusting a file.

use std::sync::atomic::{AtomicU64, Ordering};

/// Recorder lifecycle state, reported at shutdown and on the health socket.
pub struct RecorderState {
    rows: AtomicU64,
    drops: AtomicU64,
}

impl RecorderState {
    pub fn new() -> Self {
        Self {
            rows: AtomicU64::new(0),
            drops: AtomicU64::new(0),
        }
    }

    pub fn record_row(&self) {
        self.rows.fetch_add(1, Ordering::Relaxed);
    }

    /// Explicit drop: counted, and the recording is incomplete from here on.
    pub fn record_drop(&self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }

    /// A recording with any drop is NOT a complete replay dataset.
    pub fn complete(&self) -> bool {
        self.drops.load(Ordering::Relaxed) == 0
    }

    pub fn rows(&self) -> u64 {
        self.rows.load(Ordering::Relaxed)
    }

    pub fn drops(&self) -> u64 {
        self.drops.load(Ordering::Relaxed)
    }
}

impl Default for RecorderState {
    fn default() -> Self {
        Self::new()
    }
}
