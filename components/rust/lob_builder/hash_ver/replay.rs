use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Price represented as an integer tick.
///
/// Example:
/// BTCUSDT tick size = 0.10
/// price 42100.10 -> 421001
pub type PriceTick = i64;

#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub quantity: f64,
}

#[derive(Debug, Default)]
pub struct LobBook {
    /// Canonical full-depth state.
    ///
    /// HashMap gives O(1) expected lookup / insert / remove.
    bids: HashMap<PriceTick, Level>,
    asks: HashMap<PriceTick, Level>,

    /// Candidate best prices.
    ///
    /// These are maintained lazily:
    /// - stale prices may remain in the heap
    /// - stale entries are discarded only when queried
    ///
    /// This avoids scanning the entire HashMap when the best
    /// bid/ask changes.
    bid_heap: BinaryHeap<PriceTick>,
    ask_heap: BinaryHeap<Reverse<PriceTick>>,
}

impl LobBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.bid_heap.clear();
        self.ask_heap.clear();
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
    /// Output depth is decided later.
    pub fn apply_snapshot(
        &mut self,
        bids: &[(PriceTick, f64)],
        asks: &[(PriceTick, f64)],
    ) {
        self.clear();

        for &(price, quantity) in bids {
            if quantity > 0.0 {
                self.bids.insert(price, Level { quantity });
                self.bid_heap.push(price);
            }
        }

        for &(price, quantity) in asks {
            if quantity > 0.0 {
                self.asks.insert(price, Level { quantity });
                self.ask_heap.push(Reverse(price));
            }
        }
    }

    /// Apply one absolute-quantity update.
    ///
    /// quantity > 0:
    ///     set absolute quantity at this price.
    ///
    /// quantity == 0:
    ///     remove this price level from this side.
    ///
    /// Expected complexity:
    ///     O(1) expected for HashMap mutation.
    ///
    /// Heap insertion only occurs when a previously absent price
    /// becomes active.
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
            return if is_bid {
                self.bids.remove(&price).is_some()
            } else {
                self.asks.remove(&price).is_some()
            };
        }

        if is_bid {
            // Enforce single-side invariant.
            self.asks.remove(&price);

            let existed = self.bids.contains_key(&price);

            self.bids.insert(
                price,
                Level { quantity },
            );

            if !existed {
                self.bid_heap.push(price);
            }

            true
        } else {
            // Enforce single-side invariant.
            self.bids.remove(&price);

            let existed = self.asks.contains_key(&price);

            self.asks.insert(
                price,
                Level { quantity },
            );

            if !existed {
                self.ask_heap.push(Reverse(price));
            }

            true
        }
    }

    /// Remove stale bid heap entries until the top points to
    /// an actually active bid.
    fn clean_bid_heap(&mut self) {
        while let Some(&price) = self.bid_heap.peek() {
            if self.bids.contains_key(&price) {
                break;
            }

            self.bid_heap.pop();
        }
    }

    /// Remove stale ask heap entries until the top points to
    /// an actually active ask.
    fn clean_ask_heap(&mut self) {
        while let Some(&Reverse(price)) = self.ask_heap.peek() {
            if self.asks.contains_key(&price) {
                break;
            }

            self.ask_heap.pop();
        }
    }

    /// Best bid.
    pub fn best_bid(&mut self) -> Option<(PriceTick, f64)> {
        self.clean_bid_heap();

        self.bid_heap
            .peek()
            .and_then(|&price| {
                self.bids
                    .get(&price)
                    .map(|level| (price, level.quantity))
            })
    }

    /// Best ask.
    pub fn best_ask(&mut self) -> Option<(PriceTick, f64)> {
        self.clean_ask_heap();

        self.ask_heap
            .peek()
            .and_then(|&Reverse(price)| {
                self.asks
                    .get(&price)
                    .map(|level| (price, level.quantity))
            })
    }

    /// Returns true if best bid >= best ask.
    pub fn is_crossed(&mut self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => bid >= ask,
            _ => false,
        }
    }

    /// Extract top-N levels.
    ///
    /// This is intentionally NOT used on every event.
    /// It is for checkpointing / occasional strategy output.
    ///
    /// Complexity:
    ///     O(r log N)
    ///
    /// where r = number of active price levels.
    pub fn top_n(
        &self,
        levels: usize,
    ) -> (
        Vec<(PriceTick, f64)>,
        Vec<(PriceTick, f64)>,
    ) {
        if levels == 0 {
            return (Vec::new(), Vec::new());
        }

        // Small bounded heaps.
        // This avoids sorting the entire book just to obtain top N.
        let mut bid_candidates: BinaryHeap<PriceTick> =
            BinaryHeap::with_capacity(levels);

        let mut ask_candidates: BinaryHeap<Reverse<PriceTick>> =
            BinaryHeap::with_capacity(levels);

        for &price in self.bids.keys() {
            bid_candidates.push(price);

            if bid_candidates.len() > levels {
                bid_candidates.pop();
            }
        }

        for &price in self.asks.keys() {
            ask_candidates.push(Reverse(price));

            if ask_candidates.len() > levels {
                ask_candidates.pop();
            }
        }

        let mut bids = bid_candidates
            .into_iter()
            .filter_map(|price| {
                self.bids
                    .get(&price)
                    .map(|level| (price, level.quantity))
            })
            .collect::<Vec<_>>();

        let mut asks = ask_candidates
            .into_iter()
            .filter_map(|Reverse(price)| {
                self.asks
                    .get(&price)
                    .map(|level| (price, level.quantity))
            })
            .collect::<Vec<_>>();

        bids.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        asks.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        (bids, asks)
    }

    /// Extract the FULL current book for a daily checkpoint.
    ///
    /// This is intentionally called only at the checkpoint boundary.
    ///
    /// Complexity:
    ///     O(r log r)
    ///
    /// where r = active levels on both sides.
    pub fn full_snapshot(
        &self,
    ) -> (
        Vec<(PriceTick, f64)>,
        Vec<(PriceTick, f64)>,
    ) {
        let mut bids = self
            .bids
            .iter()
            .map(|(&price, level)| (price, level.quantity))
            .collect::<Vec<_>>();

        let mut asks = self
            .asks
            .iter()
            .map(|(&price, level)| (price, level.quantity))
            .collect::<Vec<_>>();

        // Preserve canonical book ordering for serialized checkpoints.
        bids.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        asks.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        (bids, asks)
    }
}