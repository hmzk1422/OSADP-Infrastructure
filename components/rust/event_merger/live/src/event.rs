//! `event.rs` — the typed events this crate accepts.
//!
//! This is the crate's entire input surface. It is deliberately free of any
//! Binance wire-format knowledge: whoever feeds the engine (`uets` today)
//! parses JSON/websocket frames and hands over a `MarketEvent`. That keeps
//! this crate allocation-free (no `String`, no `serde_json::Value`) and
//! independently testable.

/// Which stream an event came from.
///
/// The discriminants deliberately match
/// `event_merger/merger/model.rs::Source` (`Lob = 0, Trade = 1, Depth = 2`)
/// so the deterministic tie-break below produces the *same ordering
/// semantics* as the historical merger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    Lob = 0,
    Trade = 1,
    Depth = 2,
}

impl Source {
    /// The literal written to the `event_type` column.
    pub fn name(self) -> &'static str {
        match self {
            Self::Lob => "LOB",
            Self::Trade => "TRADE",
            Self::Depth => "BOOKDEPTH",
        }
    }
}

/// One order-book mutation from the diff stream.
#[derive(Debug, Clone, Copy)]
pub struct BookUpdate {
    pub is_bid: bool,
    /// Price as received (caller converts to a tick via `book::price_to_tick`).
    pub price: f64,
    /// Absolute new quantity; `0.0` removes the level.
    pub quantity: f64,
}

/// A full-depth REST snapshot used to seed (or re-seed) the book.
#[derive(Debug, Clone)]
pub struct BookSnapshot {
    pub last_update_id: u64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

/// A diff-frame envelope: one websocket `depth` push carries many updates that
/// share an event time and an update-id range.
#[derive(Debug, Clone)]
pub struct DiffFrame {
    /// Exchange event time, epoch **milliseconds**.
    pub event_time_ms: i64,
    pub first_update_id: u64,
    /// Id of the last update applied by this frame.
    pub final_update_id: u64,
    pub updates: Vec<BookUpdate>,
}

/// One aggregated trade.
#[derive(Debug, Clone, Copy)]
pub struct Trade {
    pub event_time_ms: i64,
    /// Exchange trade time, epoch ms (`T` on the wire).
    pub transact_time_ms: i64,
    pub agg_trade_id: u64,
    pub first_trade_id: u64,
    pub last_trade_id: u64,
    pub price: f64,
    pub quantity: f64,
    /// `true` => the buyer was the maker, i.e. the taker sold.
    pub is_buyer_maker: bool,
}

/// Anything the engine can accept.
#[derive(Debug, Clone)]
pub enum MarketEvent {
    /// A diff frame from the depth stream.
    Diff(DiffFrame),
    /// A snapshot seed / re-seed.
    Snapshot(BookSnapshot),
    /// An aggregated trade.
    Trade(Trade),
}

impl MarketEvent {
    /// Source priority used as the secondary sort key.
    pub fn source(&self) -> Source {
        match self {
            Self::Diff(_) | Self::Snapshot(_) => Source::Lob,
            Self::Trade(_) => Source::Trade,
        }
    }

    /// Exchange event time in epoch ms, used as the primary sort key.
    pub fn event_time_ms(&self) -> i64 {
        match self {
            Self::Diff(frame) => frame.event_time_ms,
            Self::Snapshot(snapshot) => snapshot.last_update_id as i64,
            Self::Trade(trade) => trade.event_time_ms,
        }
    }

    /// Monotonic per-source counter, used as the tertiary sort key so that two
    /// events sharing a timestamp and a source still have a total order.
    pub fn source_sequence(&self) -> u64 {
        match self {
            Self::Diff(frame) => frame.final_update_id,
            Self::Snapshot(snapshot) => snapshot.last_update_id,
            Self::Trade(trade) => trade.agg_trade_id,
        }
    }
}
