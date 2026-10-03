//! Causal bar reconstruction.
//!
//! Aggregates a stream of rows into fixed-size bars, emitting a bar only once
//! it is **complete**. This is the Rust counterpart of
//! `live.feature.reconstruct_rows` and of the compiled `reconstruct` op.
//!
//! Causality is the whole point: a bar is stamped with the timestamp of its
//! *last* constituent row, so it is only emitted after that row has been seen.
//!
//! # Why this lives in `strategy` and not `core::feature`
//!
//! `core::feature` computes a value for every row. Reconstruction changes the
//! *shape* of the data â€” many rows in, one bar out â€” which is a decision about
//! how a strategy consumes data, not about a feature's value. Keeping it
//! separate means a strategy can be fed either tick-level or bar-level data
//! without either side knowing about the other.

use crate::core::feature::{Aggregation, aggregate};

/// One reconstructed bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    /// Timestamp of the **last** row in the bar, i.e. the bar close time.
    pub timestamp_ms: i64,
    /// Number of rows that made up the bar.
    pub row_count: usize,
    /// The aggregated value.
    pub value: f64,
}

/// Why reconstruction could not be set up or driven.
#[derive(Clone, Debug, PartialEq)]
pub enum ReconstructError {
    /// A bar size of zero would never complete.
    ZeroBarSize,
    /// Timestamps went backwards, so bar boundaries would be meaningless.
    NonMonotonicTimestamp { previous: i64, current: i64 },
}

impl std::fmt::Display for ReconstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBarSize => write!(f, "bar size must be positive"),
            Self::NonMonotonicTimestamp { previous, current } => write!(
                f,
                "timestamps must be non-decreasing: {current} follows {previous}"
            ),
        }
    }
}

impl std::error::Error for ReconstructError {}

/// Streaming bar builder.
///
/// Feed rows with [`Reconstructor::push`]; it returns a `Bar` whenever one
/// completes and `None` while the current bar is still filling. The trailing
/// partial bar is never emitted, matching Python â€” it could still change, so
/// treating it as closed would be a look-ahead bug.
#[derive(Clone, Debug, PartialEq)]
pub struct Reconstructor {
    size: usize,
    aggregation: Aggregation,
    buffer: Vec<f64>,
    last_timestamp: Option<i64>,
}

impl Reconstructor {
    /// Build a reconstructor for bars of `size` rows.
    pub fn new(size: usize, aggregation: Aggregation) -> Result<Self, ReconstructError> {
        if size == 0 {
            return Err(ReconstructError::ZeroBarSize);
        }
        Ok(Self {
            size,
            aggregation,
            buffer: Vec::with_capacity(size),
            last_timestamp: None,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Rows accumulated into the bar currently being built.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }

    /// Push one row, returning a `Bar` when this row completes one.
    ///
    /// Timestamps must be non-decreasing; a decrease means the stream is out of
    /// order and every later bar would be wrong, so it is rejected outright
    /// rather than silently producing misaligned bars.
    pub fn push(&mut self, timestamp_ms: i64, value: f64) -> Result<Option<Bar>, ReconstructError> {
        if let Some(previous) = self.last_timestamp {
            if timestamp_ms < previous {
                return Err(ReconstructError::NonMonotonicTimestamp {
                    previous,
                    current: timestamp_ms,
                });
            }
        }
        self.last_timestamp = Some(timestamp_ms);

        self.buffer.push(value);
        if self.buffer.len() < self.size {
            return Ok(None);
        }
        // The bar closes on *this* row, so the timestamp is this row's.
        let bar = Bar {
            timestamp_ms,
            row_count: self.size,
            value: aggregate(&self.buffer, self.aggregation),
        };
        self.buffer.clear();
        Ok(Some(bar))
    }

    /// Flush an incomplete trailing bar.
    ///
    /// A live stream that ends mid-bar has no close time, so this is only for
    /// end-of-session / backtest teardown. It is explicit rather than implicit
    /// in `Drop`, because emitting a partial bar by accident is exactly the bug
    /// this module exists to prevent.
    pub fn flush(&mut self) -> Option<Bar> {
        if self.buffer.is_empty() {
            return None;
        }
        let bar = Bar {
            timestamp_ms: self.last_timestamp.unwrap_or_default(),
            row_count: self.buffer.len(),
            value: aggregate(&self.buffer, self.aggregation),
        };
        self.buffer.clear();
        Some(bar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(size: usize, aggregation: Aggregation, values: &[(i64, f64)]) -> Vec<Bar> {
        let mut r = Reconstructor::new(size, aggregation).unwrap();
        let mut bars = Vec::new();
        for (ts, v) in values {
            if let Some(bar) = r.push(*ts, *v).unwrap() {
                bars.push(bar);
            }
        }
        bars
    }

    #[test]
    fn emits_one_bar_per_complete_group() {
        let bars = drive(
            2,
            Aggregation::Mean,
            &[(0, 1.0), (1, 3.0), (2, 5.0), (3, 7.0)],
        );
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].value, 2.0);
        assert_eq!(bars[1].value, 6.0);
    }

    #[test]
    fn bar_is_timestamped_at_its_close() {
        let bars = drive(3, Aggregation::Mean, &[(10, 1.0), (20, 1.0), (30, 1.0)]);
        assert_eq!(bars[0].timestamp_ms, 30, "bar must close on its last row");
    }

    #[test]
    fn trailing_partial_bar_is_not_emitted() {
        // 7 rows at size 4 completes one bar and leaves 3 rows pending.
        let mut r = Reconstructor::new(4, Aggregation::Mean).unwrap();
        let mut emitted = 0;
        for i in 0..7 {
            if r.push(i, 1.0).unwrap().is_some() {
                emitted += 1;
            }
        }
        assert_eq!(emitted, 1, "only the complete group should have closed");
        assert_eq!(r.pending(), 3, "three rows remain in an unclosed bar");
    }

    #[test]
    fn flush_emits_the_partial_bar_on_demand() {
        let mut r = Reconstructor::new(4, Aggregation::Last).unwrap();
        for i in 0..6 {
            r.push(i * 10, i as f64).unwrap();
        }
        let bar = r.flush().expect("partial bar flushed");
        assert_eq!(bar.row_count, 2);
        assert_eq!(bar.value, 5.0);
        assert_eq!(bar.timestamp_ms, 50);
        assert!(r.flush().is_none(), "buffer is empty");
    }

    #[test]
    fn zero_size_is_rejected() {
        assert_eq!(
            Reconstructor::new(0, Aggregation::Mean),
            Err(ReconstructError::ZeroBarSize)
        );
    }

    #[test]
    fn out_of_order_timestamps_are_rejected() {
        let mut r = Reconstructor::new(2, Aggregation::Mean).unwrap();
        r.push(10, 1.0).unwrap();
        assert_eq!(
            r.push(5, 1.0),
            Err(ReconstructError::NonMonotonicTimestamp {
                previous: 10,
                current: 5
            })
        );
    }
}
