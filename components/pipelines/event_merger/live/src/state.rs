//! `state.rs` — the live state carried between events.
//!
//! Holds the full-depth book, the sequence bookkeeping used to detect diff
//! gaps, and the validity flags that make a compromised book *visible* rather
//! than silently wrong.

use crate::book::{LobBook, PriceTick};
use crate::event::{DiffFrame, Source, Trade};

/// Why the book cannot currently be trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookHealth {
    /// Book reflects every diff event received so far.
    Valid,
    /// A sequence gap was detected; the book is stale until a re-seed lands.
    Stale,
    /// No snapshot has been applied yet.
    Unseeded,
}

impl BookHealth {
    pub fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }
}

/// Counters for operational visibility. Cheap to update, always surfaced.
#[derive(Debug, Clone, Copy, Default)]
pub struct Counters {
    pub events_applied: u64,
    pub diff_frames: u64,
    pub trades: u64,
    /// Frames rejected because they arrived after the watermark.
    pub late_dropped: u64,
    /// Sequence gaps detected on the diff stream.
    pub gaps: u64,
    /// REST snapshot seeds applied (initial + re-seeds).
    pub seeds_applied: u64,
    /// Updates dropped because their price could not be represented as a tick.
    pub bad_price_dropped: u64,
}

/// Everything the merger needs to emit one canonical row.
pub struct State {
    pub book: LobBook,
    /// Last `final_update_id` applied to the book, if the book is seeded.
    pub last_update_id: Option<u64>,
    /// As-of LOB identity for the next emitted row.
    pub lob_event_time_ms: Option<i64>,
    pub lob_final_update_id: Option<i64>,
    /// Monotonic emission counter.
    pub event_sequence: u64,
    pub health: BookHealth,
    pub counters: Counters,
}

impl Default for State {
    fn default() -> Self {
        Self {
            book: LobBook::new(),
            last_update_id: None,
            lob_event_time_ms: None,
            lob_final_update_id: None,
            event_sequence: 0,
            health: BookHealth::Unseeded,
            counters: Counters::default(),
        }
    }
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of price levels the book considers valid to emit.
    pub fn book_valid(&self) -> bool {
        self.health.is_valid() && !self.book.is_empty()
    }

    /// `depth__*` / `depth_state__*` have no live source (Binance's bookDepth
    /// is a downloaded dataset, not a stream), so they are always absent.
    /// Consumers must treat these as *missing*, never as zero.
    pub fn depth_valid(&self) -> bool {
        false
    }

    /// Mark the book stale. Called on a detected sequence gap; the caller is
    /// responsible for triggering a re-seed.
    pub fn mark_gap(&mut self) {
        self.health = BookHealth::Stale;
        self.counters.gaps += 1;
    }

    /// Apply a trusted full snapshot, clearing any stale marking.
    pub fn apply_seed(
        &mut self,
        bids: &[(PriceTick, f64)],
        asks: &[(PriceTick, f64)],
        last_update_id: u64,
    ) {
        self.book.apply_snapshot(bids, asks);
        self.last_update_id = Some(last_update_id);
        self.health = if self.book.is_empty() {
            BookHealth::Unseeded
        } else {
            BookHealth::Valid
        };
        self.counters.seeds_applied += 1;
    }

    /// Apply one diff frame, detecting sequence gaps.
    ///
    /// Binance's diff stream is ordered, so continuity is
    /// `first_update_id <= last_applied + 1`. Anything else is a gap: the book
    /// is marked stale and the frame is *still applied* (best effort) while the
    /// caller re-seeds. Historical replay treats the same condition as
    /// `first_update_id > last_final_update_id + 1`.
    pub fn apply_diff(&mut self, frame: &DiffFrame) -> bool {
        self.counters.diff_frames += 1;

        if let Some(last) = self.last_update_id {
            if frame.first_update_id > last.saturating_add(1) {
                self.mark_gap();
            }
        }

        for update in &frame.updates {
            let Some(tick) = crate::book::price_to_tick(update.price) else {
                self.counters.bad_price_dropped += 1;
                continue;
            };
            self.book.apply_update(update.is_bid, tick, update.quantity);
        }

        self.last_update_id = Some(frame.final_update_id);
        self.lob_event_time_ms = Some(frame.event_time_ms);
        self.lob_final_update_id = Some(frame.final_update_id as i64);
        self.counters.events_applied += 1;
        true
    }

    /// Record a trade. The book is untouched: the as-of snapshot is read at
    /// emit time, not captured here.
    pub fn apply_trade(&mut self, _trade: &Trade) {
        self.counters.trades += 1;
        self.counters.events_applied += 1;
    }

    /// Source priority of the last applied event, for row labelling.
    pub fn last_source(&self) -> Source {
        if self.lob_event_time_ms.is_some() {
            Source::Lob
        } else {
            Source::Trade
        }
    }
}
