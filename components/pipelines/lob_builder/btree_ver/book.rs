//! `book.rs` — what the book is.
//!
//! This module owns the L2 order-book data structure and nothing else: given
//! book updates, it maintains the current bids/asks. It knows nothing about
//! parquet, the CLI, ingest mode, or matched mode — those live in
//! `parquet.rs` and `main.rs`. Whether a given update stream is *trustworthy*
//! (gaps, sequencing, warm start) is `replay.rs`'s job, not this one.

use std::collections::BTreeMap;

/// Fixed-point resolution used to key book levels as integers.
///
/// Every price the book sees — regardless of the instrument's own exchange
/// tick size — is rounded to the nearest `1 / TICK_SCALE` and stored as that
/// integer. This keeps `BTreeMap<PriceTick, Level>` lookups exact (no
/// floating-point key comparisons) while resolving prices far finer than any
/// real tick size.
///
/// Example: price `42100.10` at `TICK_SCALE = 1e8` -> tick `4_210_010_000_000`.
pub const TICK_SCALE: f64 = 1e8;

/// Price represented as an integer tick (see [`TICK_SCALE`]).
pub type PriceTick = i64;

/// Convert a price into its integer tick representation.
///
/// Returns `None` for non-finite, negative, or out-of-range prices.
pub fn price_to_tick(price: f64) -> Option<PriceTick> {
    if !price.is_finite() || price < 0.0 {
        return None;
    }

    let scaled = price * TICK_SCALE;

    if scaled > i64::MAX as f64 {
        return None;
    }

    Some(scaled.round() as PriceTick)
}

/// Convert an integer tick back into a price.
pub fn tick_to_price(tick: PriceTick) -> f64 {
    tick as f64 / TICK_SCALE
}

#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub quantity: f64,
}

#[derive(Debug, Default)]
pub struct LobBook {
    // Sorted ascending.
    // Best bid = largest key.
    bids: BTreeMap<PriceTick, Level>,

    // Sorted ascending.
    // Best ask = smallest key.
    asks: BTreeMap<PriceTick, Level>,
}

impl LobBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    pub fn bid_levels(&self) -> usize {
        self.bids.len()
    }

    pub fn ask_levels(&self) -> usize {
        self.asks.len()
    }

    /// Replace the entire book with a trusted full snapshot.
    ///
    /// IMPORTANT:
    /// We retain the FULL book internally.
    /// `levels` is only applied when emitting top-N output.
    pub fn apply_snapshot(
        &mut self,
        bids: &[(PriceTick, f64)],
        asks: &[(PriceTick, f64)],
    ) {
        self.clear();

        for &(price, quantity) in bids {
            if quantity > 0.0 {
                self.bids.insert(
                    price,
                    Level { quantity },
                );
            }
        }

        for &(price, quantity) in asks {
            if quantity > 0.0 {
                self.asks.insert(
                    price,
                    Level { quantity },
                );
            }
        }
    }

    /// Apply one update.
    ///
    /// qty > 0:
    ///     absolute new quantity at this price.
    ///
    /// qty == 0:
    ///     remove this price level.
    ///
    /// Returns whether the book changed.
    pub fn apply_update(
        &mut self,
        is_bid: bool,
        price: PriceTick,
        quantity: f64,
    ) -> bool {
        if quantity < 0.0 {
            return false;
        }

        if quantity == 0.0 {
            // IMPORTANT:
            // Delete ONLY from the specified side.
            //
            // Do not remove the opposite side here.
            return if is_bid {
                self.bids.remove(&price).is_some()
            } else {
                self.asks.remove(&price).is_some()
            };
        }

        // Positive quantity:
        // enforce the single-side invariant.
        if is_bid {
            self.asks.remove(&price);

            let old = self.bids.insert(
                price,
                Level { quantity },
            );

            old.map_or(true, |x| x.quantity != quantity)
        } else {
            self.bids.remove(&price);

            let old = self.asks.insert(
                price,
                Level { quantity },
            );

            old.map_or(true, |x| x.quantity != quantity)
        }
    }

    /// Best bid.
    pub fn best_bid(&self) -> Option<(PriceTick, f64)> {
        self.bids
            .iter()
            .next_back()
            .map(|(&price, level)| (price, level.quantity))
    }

    /// Best ask.
    pub fn best_ask(&self) -> Option<(PriceTick, f64)> {
        self.asks
            .iter()
            .next()
            .map(|(&price, level)| (price, level.quantity))
    }

    /// Returns true if best bid >= best ask.
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => bid >= ask,
            _ => false,
        }
    }

    /// Extract top-N levels.
    ///
    /// Bids: highest -> lowest
    /// Asks: lowest -> highest
    pub fn top_n(
        &self,
        levels: usize,
    ) -> (
        Vec<(PriceTick, f64)>,
        Vec<(PriceTick, f64)>,
    ) {
        let bids = self
            .bids
            .iter()
            .rev()
            .take(levels)
            .map(|(&price, level)| {
                (price, level.quantity)
            })
            .collect();

        let asks = self
            .asks
            .iter()
            .take(levels)
            .map(|(&price, level)| {
                (price, level.quantity)
            })
            .collect();

        (bids, asks)
    }

    /// Every resting level on both sides (no `levels` cap).
    ///
    /// Bids: highest -> lowest
    /// Asks: lowest -> highest
    ///
    /// This is what `replay.rs` uses to materialize a full-book checkpoint —
    /// `top_n` is for bounded display/output, `full_snapshot` is for storage.
    pub fn full_snapshot(
        &self,
    ) -> (
        Vec<(PriceTick, f64)>,
        Vec<(PriceTick, f64)>,
    ) {
        let bids = self
            .bids
            .iter()
            .rev()
            .map(|(&price, level)| (price, level.quantity))
            .collect();

        let asks = self
            .asks
            .iter()
            .map(|(&price, level)| (price, level.quantity))
            .collect();

        (bids, asks)
    }
}