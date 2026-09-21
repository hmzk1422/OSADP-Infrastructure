use crate::book::{LobBook, PriceTick, tick_to_price};

/// Storage-budget policy default: exactly one full-book snapshot is written
/// per UTC day — at the day's hour-00 frame — instead of every snapshot
/// event. `should_checkpoint` encodes that rule; `parquet.rs::ingest_parquet`
/// drives it to write one `<YYYY-MM-DD>.parquet` file per day.
///
/// [`CheckpointPolicy::EverySnapshot`] lifts that restriction: every valid
/// event's book image is stored. `parquet.rs::process_matched` drives this
/// policy to write one full-book row per event.
///
/// This module only decides *whether* the book is trustworthy enough to
/// checkpoint and *which* events are checkpoint-eligible under a policy; it
/// never decides when to write a file — that belongs to `parquet.rs`.
pub const CHECKPOINT_HOUR_UTC: u32 = 0;

/// True when `event_time` (epoch ms) falls in the checkpoint hour (00 UTC).
pub fn should_checkpoint(event_time_ms: i64) -> bool {
    if event_time_ms < 0 {
        return false;
    }
    const DAY_MS: i64 = 86_400_000;
    const HOUR_MS: i64 = 3_600_000;
    let ms_of_day = event_time_ms % DAY_MS;
    let hour = (ms_of_day / HOUR_MS) as u32;
    hour == CHECKPOINT_HOUR_UTC
}

/// Which snapshot events materialize a stored full-book image.
///
/// * `DayStart`       — the storage-budget default: exactly one image per
///   UTC day, taken at the day's hour-00 frame (`should_checkpoint`).
/// * `EverySnapshot`  — no hour gate and no once-per-day guard: `checkpoint`
///   materializes an image for every call it is given while the engine is
///   valid. It is the caller's choice (see `parquet.rs::process_matched`)
///   whether "every call" means every event or only snapshot-type ones —
///   this policy itself does not filter by event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPolicy {
    DayStart,
    EverySnapshot,
}

impl CheckpointPolicy {
    /// CLI/config name of the policy.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DayStart => "day_start",
            Self::EverySnapshot => "every_snapshot",
        }
    }

    /// Parse a policy name ("day_start" | "every_snapshot").
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "day_start" => Some(Self::DayStart),
            "every_snapshot" => Some(Self::EverySnapshot),
            _ => None,
        }
    }
}

/// UTC day index of an epoch-ms timestamp (floor division, so pre-1970
/// timestamps stay aligned to the same day boundaries).
pub fn day_index(event_time_ms: i64) -> i64 {
    event_time_ms.div_euclid(86_400_000)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    Snapshot,
    Update,
}

#[derive(Debug, Clone, Copy)]
pub struct LobRow {
    pub event_time: i64,
    pub event_type: EventType,
    pub is_bid: bool,
    pub price: f64,
    pub quantity: f64,
    pub first_update_id: Option<u64>,
    pub final_update_id: Option<u64>,
    pub prev_final_update_id: Option<u64>,
}

/// One stored full-book image.
///
/// Under [`CheckpointPolicy::DayStart`] emitted at most once per UTC day
/// (hour 00); under [`CheckpointPolicy::EverySnapshot`] emitted for every
/// checkpoint-eligible event. Carries integer-tick levels (bids desc, asks
/// asc) plus float mirrors so `parquet.rs` can write prices straight to disk
/// without re-scaling.
#[derive(Debug, Clone)]
pub struct BookCheckpoint {
    pub event_time: i64,
    pub final_update_id: u64,
    pub bids: Vec<(PriceTick, f64)>,
    pub asks: Vec<(PriceTick, f64)>,
    pub bid_prices: Vec<f64>,
    pub ask_prices: Vec<f64>,
}

impl BookCheckpoint {
    pub fn bid_levels(&self) -> usize {
        self.bids.len()
    }

    pub fn ask_levels(&self) -> usize {
        self.asks.len()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReplayStats {
    /// Snapshot events that reseeded the book (every snapshot still resets
    /// state — the daily checkpoint policy only limits *writes*, not reseeds).
    pub snapshot_events: u64,
    pub update_events: u64,
    pub gaps: u64,
    pub malformed_events: u64,
    /// Crossed states seen at emission time (or refused at checkpoint time):
    /// `best_bid >= best_ask` is unverifiable, so it is never published.
    pub crossed_books: u64,
    pub rows_before_initial_snapshot: u64,
    /// Checkpoints actually materialized (at most one per UTC day, hour 00,
    /// under [`CheckpointPolicy::DayStart`]).
    pub snapshots: u64,
    /// Day-start boundaries where no image could be stored (state unseeded,
    /// invalid, one-sided or crossed) plus non-00-hour `checkpoint()` calls
    /// rejected by the storage policy.
    pub skipped_checkpoints: u64,
    /// `event_time` batches replayed (one diff frame each).
    pub batches: u64,
    /// Book-state rows actually emitted (one per published batch).
    pub emitted_rows: u64,
    /// Batches dropped because their state could not be published (unseeded,
    /// invalid, a side empty, or crossed).
    pub skipped_state_rows: u64,
    /// Batches applied on top of a diff-only (unseeded) or just-rebased book —
    /// i.e. batches whose starting state came from a warm start rather than
    /// from a trusted snapshot. Zero for a fully snapshot-seeded replay.
    pub warm_start_batches: u64,
}

#[derive(Debug)]
pub enum ReplayError {
    InvalidTickSize,
    InvalidPrice(f64),
    NegativeQuantity(f64),
    NonMonotonicEventTime,
    SnapshotMissingFinalId,
    UpdateMissingIds,
    InconsistentBatchIds,
    PartialSnapshot,
    /// Columnar input columns do not all carry the same number of rows.
    RaggedColumns,
    PriceNotAligned {
        price: f64,
        tick_size: f64,
    },
}

impl std::fmt::Display for ReplayError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::InvalidTickSize =>
                write!(f, "tick_size must be > 0"),

            Self::InvalidPrice(p) =>
                write!(f, "invalid price: {p}"),

            Self::NegativeQuantity(q) =>
                write!(f, "negative quantity: {q}"),

            Self::NonMonotonicEventTime =>
                write!(f, "event_time is not monotonic"),

            Self::SnapshotMissingFinalId =>
                write!(f, "snapshot missing final_update_id"),

            Self::UpdateMissingIds =>
                write!(f, "update missing required update IDs"),

            Self::InconsistentBatchIds =>
                write!(f, "rows inside one event have inconsistent IDs"),

            Self::PartialSnapshot =>
                write!(f, "snapshot does not contain both bid and ask sides"),

            Self::RaggedColumns =>
                write!(f, "columnar input has mismatched column lengths"),

            Self::PriceNotAligned {
                price,
                tick_size,
            } =>
                write!(
                    f,
                    "price {price} is not aligned to tick size {tick_size}"
                ),
        }
    }
}

impl std::error::Error for ReplayError {}

pub struct ReplayEngine {
    pub book: LobBook,

    tick_size: f64,

    /// `true` = accept diff-only streams (no leading snapshot frame at all):
    /// the first diff batch itself seeds the book, and a gap rebases from
    /// diffs alone instead of latching invalid. `false` = strict: nothing is
    /// applied before a trusted snapshot, and a gap latches the state
    /// invalid until the next snapshot.
    warm_start: bool,

    /// True only after a complete trusted snapshot (or the first diff batch in
    /// warm-start mode).
    seeded: bool,

    /// False after a detected sequence gap (strict mode only).
    valid: bool,

    last_final_update_id: Option<u64>,
    last_event_time: Option<i64>,

    /// UTC day of the last materialized checkpoint, so a day is written once
    /// even when the stream re-enters its hour-00 window (e.g. a resumed run).
    last_checkpoint_day: Option<i64>,

    /// Which snapshot events materialize stored images (see
    /// [`CheckpointPolicy`]). Default `DayStart` — the storage-budget rule.
    checkpoint_policy: CheckpointPolicy,

    pub stats: ReplayStats,
}

impl ReplayEngine {
    /// Strict engine: a trusted snapshot must seed the book before any diff is
    /// applied, and a gap keeps the state invalid until the next snapshot.
    pub fn new(
        tick_size: f64,
    ) -> Result<Self, ReplayError> {
        Self::with_options(tick_size, false)
    }

    /// `warm_start = true` enables the data-driven fallback: the first diff
    /// batch itself seeds the book, and a gap rebases (clear + continue)
    /// instead of latching the state invalid. Both counters (`gaps`,
    /// `warm_start_batches`) still record what happened.
    pub fn with_options(
        tick_size: f64,
        warm_start: bool,
    ) -> Result<Self, ReplayError> {
        if !tick_size.is_finite() || tick_size <= 0.0 {
            return Err(ReplayError::InvalidTickSize);
        }

        Ok(Self {
            book: LobBook::new(),
            tick_size,
            warm_start,
            seeded: false,
            valid: false,
            last_final_update_id: None,
            last_event_time: None,
            last_checkpoint_day: None,
            checkpoint_policy: CheckpointPolicy::DayStart,
            stats: ReplayStats::default(),
        })
    }

    pub fn is_valid(&self) -> bool {
        self.seeded && self.valid
    }

    pub fn warm_start(&self) -> bool {
        self.warm_start
    }

    /// Which snapshot events materialize stored images.
    pub fn checkpoint_policy(&self) -> CheckpointPolicy {
        self.checkpoint_policy
    }

    /// Select the snapshot storage policy (see [`CheckpointPolicy`]).
    /// Must be set before replaying; switching mid-stream would make the
    /// stored-image history inconsistent.
    pub fn set_checkpoint_policy(&mut self, policy: CheckpointPolicy) {
        self.checkpoint_policy = policy;
    }

    /// The book under replay (emission/checkpoint reads only).
    pub fn book(&self) -> &LobBook {
        &self.book
    }

    /// Seed the engine with a stored day-start checkpoint so a later run can
    /// resume from it instead of replaying history (the storage-side reason
    /// those snapshots exist). The id chain continues from
    /// `final_update_id`, so the first batch after the seed must have
    /// `prev_final_update_id == final_update_id` to look gap-free.
    pub fn seed_snapshot(
        &mut self,
        event_time: i64,
        final_update_id: u64,
        bids: &[(crate::book::PriceTick, f64)],
        asks: &[(crate::book::PriceTick, f64)],
    ) -> Result<(), ReplayError> {
        if bids.is_empty() || asks.is_empty() {
            return Err(ReplayError::PartialSnapshot);
        }

        self.book.apply_snapshot(bids, asks);

        self.last_final_update_id = Some(final_update_id);
        self.last_event_time = Some(event_time);
        self.last_checkpoint_day = Some(day_index(event_time));
        self.seeded = true;
        self.valid = true;

        Ok(())
    }

    fn invalidate(&mut self) {
        self.book.clear();
        self.seeded = false;
        self.valid = false;
        self.last_final_update_id = None;
    }

    fn price_to_tick(
        &self,
        price: f64,
    ) -> Result<PriceTick, ReplayError> {
        if !price.is_finite() || price <= 0.0 {
            return Err(ReplayError::InvalidPrice(price));
        }

        let scaled = price / self.tick_size;
        let rounded = scaled.round();

        let tolerance =
            1e-9 * scaled.abs().max(1.0);

        if (scaled - rounded).abs() > tolerance {
            return Err(
                ReplayError::PriceNotAligned {
                    price,
                    tick_size: self.tick_size,
                }
            );
        }

        crate::book::price_to_tick(price).ok_or(ReplayError::InvalidPrice(price))
    }

    /// Process one complete exchange event.
    ///
    /// `rows` contains all L2 rows belonging to the same
    /// snapshot/update event.
    ///
    /// IMPORTANT:
    /// This function does NOT materialize top-N output.
    ///
    /// Hot path is:
    ///     validate -> HashMap mutations
    ///
    /// Daily checkpointing happens separately.
    pub fn process_event(
        &mut self,
        rows: &[LobRow],
    ) -> Result<(), ReplayError> {
        if rows.is_empty() {
            return Ok(());
        }

        let event_time = rows[0].event_time;

        if let Some(previous) = self.last_event_time {
            if event_time < previous {
                return Err(
                    ReplayError::NonMonotonicEventTime
                );
            }
        }

        self.last_event_time = Some(event_time);

        let event_type = rows[0].event_type;
        let first_id = rows[0].first_update_id;
        let final_id = rows[0].final_update_id;
        let prev_id = rows[0].prev_final_update_id;

        let consistent = rows.iter().all(|row| {
            row.event_time == event_time
                && row.event_type == event_type
                && row.first_update_id == first_id
                && row.final_update_id == final_id
                && row.prev_final_update_id == prev_id
        });

        if !consistent {
            self.stats.malformed_events += 1;
            self.invalidate();

            return Err(
                ReplayError::InconsistentBatchIds
            );
        }

        match event_type {
            EventType::Snapshot => {
                self.process_snapshot(
                    rows,
                    final_id,
                )
            }

            EventType::Update => {
                self.process_update(
                    rows,
                    first_id,
                    final_id,
                    prev_id,
                )
            }
        }
    }

    fn process_snapshot(
        &mut self,
        rows: &[LobRow],
        final_id: Option<u64>,
    ) -> Result<(), ReplayError> {
        self.stats.snapshot_events += 1;

        let final_id = final_id.ok_or(
            ReplayError::SnapshotMissingFinalId
        )?;

        let has_bid =
            rows.iter().any(|r| r.is_bid);

        let has_ask =
            rows.iter().any(|r| !r.is_bid);

        if !has_bid || !has_ask {
            self.invalidate();

            return Err(
                ReplayError::PartialSnapshot
            );
        }

        let mut bids =
            Vec::with_capacity(rows.len());

        let mut asks =
            Vec::with_capacity(rows.len());

        for row in rows {
            if row.quantity < 0.0 {
                return Err(
                    ReplayError::NegativeQuantity(
                        row.quantity
                    )
                );
            }

            let price =
                self.price_to_tick(row.price)?;

            if row.quantity == 0.0 {
                continue;
            }

            if row.is_bid {
                bids.push((price, row.quantity));
            } else {
                asks.push((price, row.quantity));
            }
        }

        self.book.apply_snapshot(
            &bids,
            &asks,
        );

        self.last_final_update_id =
            Some(final_id);

        self.seeded = true;
        self.valid = true;

        Ok(())
    }

    fn process_update(
        &mut self,
        rows: &[LobRow],
        first_id: Option<u64>,
        final_id: Option<u64>,
        prev_id: Option<u64>,
    ) -> Result<(), ReplayError> {
        self.stats.update_events += 1;

        // Missing continuity ids: the batch cannot be verified.
        //   * strict mode  -> refuse it (caller sees UpdateMissingIds)
        //   * warm start   -> treat it as diff-only data and apply it, counted
        //                     as a warm-start batch (diffs are absolute
        //                     upserts, so applying them is still meaningful)
        let ids = match (first_id, final_id, prev_id) {
            (Some(f), Some(l), Some(p)) => Some((f, l, p)),
            _ => {
                if !self.warm_start {
                    return Err(ReplayError::UpdateMissingIds);
                }

                // Applying it is still meaningful (diffs are absolute
                // upserts); the warm-start counter is bumped where the state
                // provenance is decided, not here.
                None
            }
        };

        // Ignore updates before the initial trusted snapshot — unless the
        // stream is diff-only (warm start), in which case the first diff
        // batch becomes the base of a warm-started book.
        if !self.seeded {
            if !self.warm_start {
                self.stats.rows_before_initial_snapshot +=
                    rows.len() as u64;

                return Ok(());
            }

            self.seeded = true;
            self.valid = true;
            self.stats.warm_start_batches += 1;
        }

        // Once invalid, remain invalid until another trusted snapshot.
        if !self.valid {
            return Ok(());
        }

        // Binance-style continuity check.
        //
        // U <= previous_final + 1 <= u
        // and pu == previous_final.
        if let (Some((first_id, final_id, prev_id)), Some(last_final)) =
            (ids, self.last_final_update_id)
        {
            let expected = last_final
                .checked_add(1)
                .ok_or(
                    ReplayError::UpdateMissingIds
                )?;

            if prev_id != last_final
                || first_id > expected
                || final_id < expected
            {
                self.stats.gaps += 1;

                if self.warm_start {
                    // Diff-only recovery: drop the (now unverifiable) book and
                    // rebase on this batch. Deep levels are gone until they
                    // are touched again — the documented cost of a
                    // snapshot-less stream, counted so callers can see it.
                    self.book.clear();
                    self.last_final_update_id = None;
                    self.stats.warm_start_batches += 1;
                } else {
                    self.invalidate();

                    return Ok(());
                }
            }
        }

        // Apply every changed level.
        //
        // This is the hot path:
        // HashMap lookup/insert/remove = O(1) expected.
        for row in rows {
            if row.quantity < 0.0 {
                return Err(
                    ReplayError::NegativeQuantity(
                        row.quantity
                    )
                );
            }

            let price =
                self.price_to_tick(row.price)?;

            self.book.apply_update(
                row.is_bid,
                price,
                row.quantity,
            );
        }

        if let Some((_, final_id, _)) = ids {
            self.last_final_update_id =
                Some(final_id);
        }

        Ok(())
    }

    /// Materialize a full-book checkpoint.
    ///
    /// STORAGE POLICY (`self.checkpoint_policy`):
    ///   * `DayStart` — only snapshot events whose `event_time` falls in
    ///     hour 00 UTC (`should_checkpoint`), at most ONCE per UTC day —
    ///     every snapshot still reseeds the book via `process_event`, but
    ///     only the day-start image is kept on disk. Callers that pass a
    ///     non-00-hour event get `None` and a `skipped_checkpoints` bump so
    ///     the daily print can report generated-vs-skipped counts.
    ///   * `EverySnapshot` — no hour gate and no once-per-day guard: every
    ///     call (one per snapshot event, gated by the columnar driver) stores
    ///     that instant's book image.
    pub fn checkpoint(
        &mut self,
        event_time: i64,
    ) -> Option<BookCheckpoint> {
        if let CheckpointPolicy::DayStart = self.checkpoint_policy {
            if !should_checkpoint(event_time) {
                self.stats.skipped_checkpoints += 1;
                return None;
            }

            if self.last_checkpoint_day == Some(day_index(event_time)) {
                // That UTC day is already on disk: keep the FIRST image of the
                // day (a resumed run re-enters the same hour-00 window).
                return None;
            }
        }

        if !self.is_valid() {
            return None;
        }

        if self.book.is_crossed() {
            self.stats.crossed_books += 1;
            return None;
        }

        let final_update_id =
            self.last_final_update_id?;

        let (bids, asks) =
            self.book.full_snapshot();

        if bids.is_empty() || asks.is_empty() {
            return None;
        }

        let bid_prices: Vec<f64> =
            bids.iter().map(|(t, _)| tick_to_price(*t)).collect();
        let ask_prices: Vec<f64> =
            asks.iter().map(|(t, _)| tick_to_price(*t)).collect();

        self.stats.snapshots += 1;
        self.last_checkpoint_day = Some(day_index(event_time));

        Some(BookCheckpoint {
            event_time,
            final_update_id,
            bids,
            asks,
            bid_prices,
            ask_prices,
        })
    }
}