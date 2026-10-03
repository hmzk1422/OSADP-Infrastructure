//! `merger.rs` — the state machine: one event in, one canonical row out.
//!
//! This is the point of the crate. It is deliberately free of I/O, of
//! Binance's wire format, and of Arrow/Parquet, so the same code path serves a
//! live capture today and a recorded-feed backtest later.

use crate::book::{price_to_tick, tick_to_price, PriceTick};
use crate::event::{BookSnapshot, MarketEvent, Source, Trade};
use crate::row::FieldWriter;
use crate::state::{BookHealth, Counters, State};

/// One canonical row.
///
/// `book_state__*` is written from live state at emit time — the as-of carry
/// the historical finalize stage reconstructs with a forward fill is *exact*
/// here, because the book is maintained rather than reconstructed.
pub struct CanonicalRow {
    /// Epoch **nanoseconds**, matching the historical `timestamp` column.
    pub timestamp_ns: i64,
    pub source: Source,
    pub event_sequence: u64,
    /// Id of the last diff applied to the book; null on non-LOB rows.
    pub lob_final_update_id: Option<i64>,
}

/// What the caller must do after `merge_event`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    /// A sequence gap was hit: the book is stale and must be re-seeded from
    /// REST before it can be trusted again.
    pub needs_reseed: bool,
    /// False when a diff arrived while the book was already stale and was
    /// therefore not applied. The row is still emitted, still flagged.
    pub applied: bool,
}

/// The stateful canonical-row engine.
pub struct Merger {
    state: State,
    levels: usize,
    /// Trade payload of the row being emitted.
    last_trade: Option<Trade>,
    /// Reused price/quantity scratch so row emission allocates nothing.
    bids: Vec<f64>,
    bid_qty: Vec<f64>,
    asks: Vec<f64>,
    ask_qty: Vec<f64>,
}

impl Merger {
    pub fn new(levels: usize) -> Self {
        Self {
            state: State::new(),
            levels,
            last_trade: None,
            bids: Vec::with_capacity(levels),
            bid_qty: Vec::with_capacity(levels),
            asks: Vec::with_capacity(levels),
            ask_qty: Vec::with_capacity(levels),
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn counters(&self) -> Counters {
        self.state.counters
    }

    /// True while the book is unusable and a REST re-seed is required.
    pub fn needs_reseed(&self) -> bool {
        self.state.health == BookHealth::Stale
    }

    /// Apply a REST snapshot, clearing any stale marking.
    pub fn apply_seed(&mut self, snapshot: &BookSnapshot) {
        let bids = to_ticks(&snapshot.bids);
        let asks = to_ticks(&snapshot.asks);
        self.state.apply_seed(&bids, &asks, snapshot.last_update_id);
    }

    /// The core: apply one event to state and produce its canonical row.
    pub fn merge_event(&mut self, event: MarketEvent) -> (CanonicalRow, Outcome) {
        let mut needs_reseed = false;
        let mut applied = true;

        let source = event.source();
        let timestamp_ms = event.event_time_ms();
        self.last_trade = None;

        match &event {
            MarketEvent::Snapshot(snapshot) => self.apply_seed(snapshot),
            MarketEvent::Diff(frame) => {
                // While the book is already stale, do not pile further diffs
                // onto a book we know is wrong: freeze it and wait for the
                // re-seed. The row is still emitted, flagged invalid.
                if self.state.health == BookHealth::Stale {
                    applied = false;
                    needs_reseed = true;
                } else {
                    let gaps_before = self.state.counters.gaps;
                    self.state.apply_diff(frame);
                    needs_reseed = self.state.counters.gaps > gaps_before;
                }
            }
            MarketEvent::Trade(trade) => {
                self.state.apply_trade(trade);
                self.last_trade = Some(*trade);
            }
        }

        self.state.event_sequence += 1;

        let row = CanonicalRow {
            timestamp_ns: timestamp_ms.saturating_mul(1_000_000),
            source,
            event_sequence: self.state.event_sequence,
            lob_final_update_id: if source == Source::Lob {
                self.state.lob_final_update_id
            } else {
                None
            },
        };

        (
            row,
            Outcome {
                needs_reseed,
                applied,
            },
        )
    }

    /// Encode a row as one CSV line into the caller's reused buffer.
    ///
    /// Returns `None` if the field count does not match the schema, which
    /// would silently misalign every downstream column.
    pub fn write_csv(&mut self, row: &CanonicalRow, out: &mut FieldWriter) -> Option<()> {
        self.refresh_book();

        let ts_ms = row.timestamp_ns / 1_000_000;
        out.begin_row(row.timestamp_ns, row.source.name(), row.event_sequence);

        // LOB identity: present only on LOB rows.
        if row.source == Source::Lob {
            out.push_opt_i64(Some(ts_ms));
            out.push_opt_i64(row.lob_final_update_id);
        } else {
            out.push_null();
            out.push_null();
        }

        // Book: best 4, then levels 2..=20.
        out.push_opt_f64(self.bids.first().copied());
        out.push_opt_f64(self.bid_qty.first().copied());
        out.push_opt_f64(self.asks.first().copied());
        out.push_opt_f64(self.ask_qty.first().copied());
        for level in 1..self.levels {
            out.push_opt_f64(self.bids.get(level).copied());
            out.push_opt_f64(self.bid_qty.get(level).copied());
            out.push_opt_f64(self.asks.get(level).copied());
            out.push_opt_f64(self.ask_qty.get(level).copied());
        }

        self.write_trade_block(out, true);
        // depth__*, depth_state__* and their notional columns have NO live
        // source (Binance's bookDepth is a downloaded dataset, not a stream).
        // Always empty: never 0.0, which would read as "no liquidity".
        for _ in 0..(4 + 10 + 10) {
            out.push_null();
        }
        self.write_trade_block(out, false);
        for _ in 0..4 {
            out.push_null();
        }

        out.finish_row().map(|_| ())
    }

    /// Emit `agg__*` (8 cols) or `source_agg__*` (7 cols). Both blocks carry
    /// the same values; only the verbatim-source column set differs.
    fn write_trade_block(&self, out: &mut FieldWriter, with_qty_alias: bool) {
        let width = if with_qty_alias { 8 } else { 7 };
        let Some(trade) = self.last_trade else {
            for _ in 0..width {
                out.push_null();
            }
            return;
        };
        out.push_opt_u64(Some(trade.agg_trade_id));
        out.push_opt_f64(Some(trade.price));
        out.push_opt_f64(Some(trade.quantity));
        if with_qty_alias {
            // agg__qty is the raw Binance `q`; quantity is carried in
            // agg__quantity. The live feed gives one value, so the alias stays
            // empty rather than duplicating a value under two names.
            out.push_null();
        }
        out.push_opt_u64(Some(trade.first_trade_id));
        out.push_opt_u64(Some(trade.last_trade_id));
        out.push_opt_i64(Some(trade.transact_time_ms));
        out.push_opt_bool(Some(trade.is_buyer_maker));
    }

    /// Copy the top-N book into the reused scratch vectors. O(levels), and no
    /// allocation once the buffers have grown.
    fn refresh_book(&mut self) {
        self.bids.clear();
        self.bid_qty.clear();
        self.asks.clear();
        self.ask_qty.clear();
        for (tick, qty) in self.state.book.top_n_bids(self.levels) {
            self.bids.push(tick_to_price(tick));
            self.bid_qty.push(qty);
        }
        for (tick, qty) in self.state.book.top_n_asks(self.levels) {
            self.asks.push(tick_to_price(tick));
            self.ask_qty.push(qty);
        }
    }
}

fn to_ticks(raw: &[(f64, f64)]) -> Vec<(PriceTick, f64)> {
    raw.iter()
        .filter_map(|&(price, quantity)| {
            if quantity <= 0.0 || !quantity.is_finite() {
                return None;
            }
            price_to_tick(price).map(|tick| (tick, quantity))
        })
        .collect()
}
