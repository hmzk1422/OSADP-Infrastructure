//! `book.rs` â€” the live full-depth order book.
//!
//! Owned entirely by this crate. It is a deliberate, self-contained
//! reimplementation of the data structure and semantics used by the historical
//! engine (`rust/lob_builder/btree_ver/book.rs`), so live and offline agree on
//! what a book is â€” without importing or depending on that file, so the two
//! stay independently buildable.
//!
//! Why full depth rather than the top-20 partial snapshot:
//! `depth20@100ms` pushes a *complete top-20 snapshot* whose `lastUpdateId` is
//! an id for that snapshot object. The historical `lob__final_update_id` is
//! the id of the last diff event applied to a *full-depth* book. Those are
//! different objects, so the partial snapshot cannot stand in for it. The live
//! engine therefore consumes the diff stream (`@depth@100ms`) plus a REST
//! snapshot seed, which makes `lob__final_update_id` mean the same thing in
//! both engines by construction.
//!
//! Complexity: `BTreeMap` gives `O(log n)` per mutation and best-price lookup,
//! and keeps both sides sorted so emitting top-N is a bounded prefix walk
//! rather than a sort per row.

use std::collections::BTreeMap;

/// Fixed-point resolution used to key book levels as integers.
///
/// Prices are rounded to the nearest `1 / TICK_SCALE` and stored as `i64`, so
/// map keys compare exactly (no floating-point key comparisons) while still
/// resolving far finer than any real tick size.
pub const TICK_SCALE: f64 = 1e8;

/// Price represented as an integer tick.
pub type PriceTick = i64;

/// Convert a price into its integer tick representation.
///
/// Returns `None` for non-finite or negative prices.
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

/// Convert an integer tick back to a price.
pub fn tick_to_price(tick: PriceTick) -> f64 {
    tick as f64 / TICK_SCALE
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Level {
    pub quantity: f64,
}

/// Full-depth L2 book.
#[derive(Debug, Default)]
pub struct LobBook {
    /// Sorted ascending; best bid is the largest key.
    bids: BTreeMap<PriceTick, Level>,
    /// Sorted ascending; best ask is the smallest key.
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

    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Replace the whole book with a trusted full snapshot.
    ///
    /// Non-positive quantities are dropped.
    pub fn apply_snapshot(&mut self, bids: &[(PriceTick, f64)], asks: &[(PriceTick, f64)]) {
        self.clear();
        for &(price, quantity) in bids {
            if quantity > 0.0 {
                self.bids.insert(price, Level { quantity });
            }
        }
        for &(price, quantity) in asks {
            if quantity > 0.0 {
                self.asks.insert(price, Level { quantity });
            }
        }
    }

    /// Apply one update.
    ///
    /// `quantity > 0` sets the absolute new quantity at that price;
    /// `quantity == 0` removes the level. Returns whether the book changed.
    ///
    /// The single-side invariant is enforced: a price may not rest on both
    /// sides at once.
    pub fn apply_update(&mut self, is_bid: bool, price: PriceTick, quantity: f64) -> bool {
        if quantity < 0.0 {
            return false;
        }
        if quantity == 0.0 {
            return if is_bid {
                self.bids.remove(&price).is_some()
            } else {
                self.asks.remove(&price).is_some()
            };
        }
        if is_bid {
            self.asks.remove(&price);
            self.bids
                .insert(price, Level { quantity })
                .is_none_or(|old| old.quantity != quantity)
        } else {
            self.bids.remove(&price);
            self.asks
                .insert(price, Level { quantity })
                .is_none_or(|old| old.quantity != quantity)
        }
    }

    pub fn best_bid(&self) -> Option<(PriceTick, f64)> {
        self.bids
            .iter()
            .next_back()
            .map(|(&price, level)| (price, level.quantity))
    }

    pub fn best_ask(&self) -> Option<(PriceTick, f64)> {
        self.asks
            .iter()
            .next()
            .map(|(&price, level)| (price, level.quantity))
    }

    /// True when best bid >= best ask.
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => bid >= ask,
            _ => false,
        }
    }

    /// Borrowing bid-side top-N: highest -> lowest, no allocation.
    pub fn top_n_bids(&self, levels: usize) -> impl Iterator<Item = (PriceTick, f64)> + '_ {
        self.bids
            .iter()
            .rev()
            .take(levels)
            .map(|(&price, level)| (price, level.quantity))
    }

    /// Borrowing ask-side top-N: lowest -> highest, no allocation.
    pub fn top_n_asks(&self, levels: usize) -> impl Iterator<Item = (PriceTick, f64)> + '_ {
        self.asks
            .iter()
            .take(levels)
            .map(|(&price, level)| (price, level.quantity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (tick, quantity) pair, as `apply_snapshot` expects.
    fn p(price: f64) -> (PriceTick, f64) {
        (price_to_tick(price).expect("finite price"), 1.0)
    }

    fn tick(price: f64) -> PriceTick {
        price_to_tick(price).expect("finite price")
    }

    #[test]
    fn snapshot_then_best_prices() {
        let mut book = LobBook::new();
        book.apply_snapshot(&[p(100.0), p(99.0)], &[p(101.0), p(102.0)]);
        assert_eq!(book.best_bid().map(|(t, _)| tick_to_price(t)), Some(100.0));
        assert_eq!(book.best_ask().map(|(t, _)| tick_to_price(t)), Some(101.0));
        assert!(!book.is_crossed());
    }

    #[test]
    fn zero_quantity_removes_level() {
        let mut book = LobBook::new();
        book.apply_snapshot(&[p(100.0)], &[p(101.0)]);
        assert!(book.apply_update(true, tick(100.0), 0.0));
        assert_eq!(book.best_bid(), None);
    }

    #[test]
    fn top_n_is_best_first_and_bounded() {
        let mut book = LobBook::new();
        let bids: Vec<_> = (1..=30).map(|i| p(100.0 - i as f64)).collect();
        book.apply_snapshot(&bids, &[]);
        let top: Vec<f64> = book.top_n_bids(5).map(|(t, _)| tick_to_price(t)).collect();
        assert_eq!(top, vec![99.0, 98.0, 97.0, 96.0, 95.0]);
    }

    #[test]
    fn update_enforces_single_side_invariant() {
        // apply_snapshot loads both sides verbatim, so a price CAN appear on
        // both. The invariant is enforced on update: re-quoting one side
        // evicts the other at the same price.
        let mut book = LobBook::new();
        book.apply_snapshot(&[p(100.0)], &[p(100.0)]);
        assert_eq!(book.bid_levels(), 1);
        assert_eq!(book.ask_levels(), 1);

        book.apply_update(true, tick(100.0), 2.0);
        assert_eq!(book.bid_levels(), 1, "bid remains");
        assert_eq!(book.ask_levels(), 0, "ask evicted at the same price");
    }

    #[test]
    fn tick_roundtrip_is_stable() {
        let price = 108663.2;
        let tick = price_to_tick(price).unwrap();
        assert!((tick_to_price(tick) - price).abs() < 1e-6);
    }

    #[test]
    fn non_finite_and_negative_prices_are_rejected() {
        assert!(price_to_tick(f64::NAN).is_none());
        assert!(price_to_tick(f64::INFINITY).is_none());
        assert!(price_to_tick(-1.0).is_none());
    }

    #[test]
    fn snapshot_drops_non_positive_quantities() {
        let mut book = LobBook::new();
        book.apply_snapshot(&[p(100.0), (tick(99.0), 0.0)], &[]);
        assert_eq!(book.bid_levels(), 1);
    }
}
