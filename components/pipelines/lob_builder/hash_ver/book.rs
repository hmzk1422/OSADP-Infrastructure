use crate::book::{LobBook, PriceTick};

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

#[derive(Debug)]
pub struct BookCheckpoint {
    pub event_time: i64,
    pub final_update_id: u64,
    pub bids: Vec<(PriceTick, f64)>,
    pub asks: Vec<(PriceTick, f64)>,
}

#[derive(Debug, Default)]
pub struct ReplayStats {
    pub snapshot_events: u64,
    pub update_events: u64,
    pub gaps: u64,
    pub malformed_events: u64,
    pub crossed_books: u64,
    pub rows_before_initial_snapshot: u64,
    pub checkpoints: u64,
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

    /// True only after a complete trusted snapshot.
    seeded: bool,

    /// False after a detected sequence gap.
    valid: bool,

    last_final_update_id: Option<u64>,
    last_event_time: Option<i64>,

    pub stats: ReplayStats,
}

impl ReplayEngine {
    pub fn new(
        tick_size: f64,
    ) -> Result<Self, ReplayError> {
        if !tick_size.is_finite() || tick_size <= 0.0 {
            return Err(ReplayError::InvalidTickSize);
        }

        Ok(Self {
            book: LobBook::new(),
            tick_size,
            seeded: false,
            valid: false,
            last_final_update_id: None,
            last_event_time: None,
            stats: ReplayStats::default(),
        })
    }

    pub fn is_valid(&self) -> bool {
        self.seeded && self.valid
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

        Ok(rounded as PriceTick)
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

        let first_id = first_id.ok_or(
            ReplayError::UpdateMissingIds
        )?;

        let final_id = final_id.ok_or(
            ReplayError::UpdateMissingIds
        )?;

        let prev_id = prev_id.ok_or(
            ReplayError::UpdateMissingIds
        )?;

        // Ignore updates before the initial trusted snapshot.
        if !self.seeded {
            self.stats.rows_before_initial_snapshot +=
                rows.len() as u64;

            return Ok(());
        }

        // Once invalid, remain invalid until another trusted snapshot.
        if !self.valid {
            return Ok(());
        }

        // Binance-style continuity check.
        //
        // U <= previous_final + 1 <= u
        // and pu == previous_final.
        if let Some(last_final) =
            self.last_final_update_id
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

                self.invalidate();

                return Ok(());
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

        self.last_final_update_id =
            Some(final_id);

        Ok(())
    }

    /// Materialize a full-book checkpoint.
    ///
    /// Call this ONCE per UTC day, not once per event.
    pub fn checkpoint(
        &mut self,
        event_time: i64,
    ) -> Option<BookCheckpoint> {
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

        self.stats.checkpoints += 1;

        Some(BookCheckpoint {
            event_time,
            final_update_id,
            bids,
            asks,
        })
    }
}