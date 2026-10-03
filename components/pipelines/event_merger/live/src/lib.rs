//! `event_merger_live` — the live, stateful canonical-row engine.
//!
//! # Boundary
//!
//! ```text
//! Binance wire format  ──(uets)──►  MarketEvent  ──(this crate)──►  CanonicalRow
//! ```
//!
//! This crate is **dependency-free on purpose**: no `serde_json` (the caller
//! parses the wire format), no Arrow/Parquet (that belongs to the historical
//! `merger/`), and no networking (that belongs to the caller). What remains is
//! a pure state machine, so it is allocation-light, deterministic, and
//! testable with no I/O — and the same code can later replay a recorded feed.
//!
//! # Parity with the historical pipeline
//!
//! The emitted schema is byte-compatible with
//! `event_merger/merger/writer.rs`, and the ordering key matches
//! `merger/merge.rs`. Two things are deliberately *not* claimed:
//!
//! * `lob__final_update_id` is the id of the last diff applied to a
//!   **full-depth** book. Binance's `depth20@100ms` `lastUpdateId` is an id for
//!   a *partial top-20 snapshot* — a different object — so the live engine
//!   consumes the diff stream plus a REST seed instead of aliasing it.
//! * `depth__*` / `depth_state__*` have no live source (Binance's `bookDepth`
//!   is a downloaded dataset). They are emitted as **empty fields, never
//!   `0.0`**, so a consumer cannot mistake "absent" for "no liquidity".

pub mod book;
pub mod csv;
pub mod event;
pub mod merger;
pub mod reorder;
pub mod row;
pub mod state;

pub use book::{price_to_tick, tick_to_price, LobBook, PriceTick};
pub use csv::{header_line, CsvError, CsvRecorder};
pub use event::{BookSnapshot, BookUpdate, DiffFrame, MarketEvent, Source, Trade};
pub use merger::{CanonicalRow, Merger, Outcome};
pub use reorder::ReorderBuffer;
pub use row::{schema, FieldWriter};
pub use state::{BookHealth, Counters, State};

/// Default book depth emitted per side, matching the historical default.
pub const DEFAULT_LEVELS: usize = 20;

/// Default reorder window in milliseconds. Bounds how far the two sockets may
/// interleave before events are dropped rather than emitted mis-ordered.
pub const DEFAULT_LATENESS_MS: i64 = 200;

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(ts: i64, id: u64, price: f64, qty: f64) -> MarketEvent {
        MarketEvent::Trade(Trade {
            event_time_ms: ts,
            transact_time_ms: ts,
            agg_trade_id: id,
            first_trade_id: id,
            last_trade_id: id,
            price,
            quantity: qty,
            is_buyer_maker: false,
        })
    }

    fn diff(ts: i64, first: u64, last: u64, updates: Vec<BookUpdate>) -> MarketEvent {
        MarketEvent::Diff(DiffFrame {
            event_time_ms: ts,
            first_update_id: first,
            final_update_id: last,
            updates,
        })
    }

    fn snapshot(last: u64, bids: &[(f64, f64)], asks: &[(f64, f64)]) -> MarketEvent {
        MarketEvent::Snapshot(BookSnapshot {
            last_update_id: last,
            bids: bids.to_vec(),
            asks: asks.to_vec(),
        })
    }

    #[test]
    fn schema_is_128_columns_and_ordered() {
        let names = schema();
        assert_eq!(names.len(), 128);
        assert_eq!(names[0], "timestamp");
        assert_eq!(names[2], "event_sequence");
        assert_eq!(names[5], "book_state__best_bid_price");
        assert_eq!(names[9], "book_state__bid_price_2");
        // 5 common + 4 best + 76 levels == 85, then agg__agg_trade_id.
        assert_eq!(names[85], "agg__agg_trade_id");
        assert_eq!(names[93], "depth__timestamp");
        assert_eq!(names[97], "depth_state__-5");
        assert_eq!(names[107], "depth_state__notional_-5");
        assert_eq!(names[127], "source_depth__notional");
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(unique.len(), 128);
    }

    #[test]
    fn unseeded_book_is_flagged_invalid() {
        let mut merger = Merger::new(DEFAULT_LEVELS);
        let (row, outcome) = merger.merge_event(trade(1000, 1, 100.0, 1.0));
        assert!(!merger.state().book_valid());
        assert!(!outcome.needs_reseed);
        assert_eq!(row.event_sequence, 1);
    }

    #[test]
    fn seed_then_diff_gives_valid_book_and_update_id() {
        let mut merger = Merger::new(DEFAULT_LEVELS);
        merger.merge_event(snapshot(100, &[(99.0, 1.0)], &[(101.0, 2.0)]));
        let (row, _) = merger.merge_event(diff(
            1000,
            101,
            105,
            vec![BookUpdate {
                is_bid: true,
                price: 100.0,
                quantity: 5.0,
            }],
        ));
        assert!(merger.state().book_valid());
        // The id is the last *diff* applied, not the snapshot id.
        assert_eq!(row.lob_final_update_id, Some(105));
        assert_eq!(row.source, Source::Lob);
    }

    #[test]
    fn sequence_gap_requests_reseed_and_freezes_book() {
        let mut merger = Merger::new(DEFAULT_LEVELS);
        merger.merge_event(snapshot(100, &[(99.0, 1.0)], &[(101.0, 2.0)]));
        merger.merge_event(diff(1000, 101, 105, vec![]));

        let (_, outcome) = merger.merge_event(diff(2000, 200, 205, vec![]));
        assert!(outcome.needs_reseed);
        assert!(merger.needs_reseed());
        assert!(!merger.state().book_valid());
        assert_eq!(merger.counters().gaps, 1);

        // While stale, further diffs are not applied.
        let (_, outcome2) = merger.merge_event(diff(3000, 206, 210, vec![]));
        assert!(!outcome2.applied);
    }

    #[test]
    fn reseed_restores_validity() {
        let mut merger = Merger::new(DEFAULT_LEVELS);
        merger.merge_event(snapshot(100, &[(99.0, 1.0)], &[(101.0, 2.0)]));
        merger.merge_event(diff(1000, 500, 505, vec![]));
        assert!(merger.needs_reseed());
        merger.merge_event(snapshot(600, &[(98.0, 1.0)], &[(102.0, 1.0)]));
        assert!(!merger.needs_reseed());
        assert!(merger.state().book_valid());
    }

    #[test]
    fn trade_rows_omit_lob_identity_but_carry_the_book() {
        let mut merger = Merger::new(DEFAULT_LEVELS);
        merger.merge_event(snapshot(100, &[(99.0, 1.0)], &[(101.0, 2.0)]));
        let (row, _) = merger.merge_event(trade(1000, 55, 100.5, 0.25));
        assert_eq!(row.source, Source::Trade);
        assert_eq!(row.lob_final_update_id, None);
        assert!(merger.state().book_valid());
    }

    #[test]
    fn emitted_row_has_128_fields() {
        let mut merger = Merger::new(DEFAULT_LEVELS);
        let mut writer = FieldWriter::new(schema().len());
        merger.merge_event(snapshot(100, &[(99.0, 1.0)], &[(101.0, 2.0)]));
        let (row, _) = merger.merge_event(diff(
            1000,
            101,
            105,
            vec![BookUpdate {
                is_bid: true,
                price: 100.0,
                quantity: 5.0,
            }],
        ));
        assert!(merger.write_csv(&row, &mut writer).is_some());
    }

    #[test]
    fn depth_columns_are_reported_absent() {
        assert!(!Merger::new(DEFAULT_LEVELS).state().depth_valid());
    }

    #[test]
    fn timestamp_is_nanoseconds() {
        let mut merger = Merger::new(DEFAULT_LEVELS);
        let (row, _) = merger.merge_event(trade(1_751_245_200_083, 1, 100.0, 1.0));
        assert_eq!(row.timestamp_ns, 1_751_245_200_083_000_000);
    }
}
