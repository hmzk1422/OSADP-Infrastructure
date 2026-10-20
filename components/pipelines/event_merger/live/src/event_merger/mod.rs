//! `event_merger` — the canonical-row state machine (the live half of the
//! historical `event_merger/` pipeline).
//!
//! Typed [`MarketEvent`]s in, one [`CanonicalRow`] per event out, with the
//! order book maintained incrementally. Nothing here knows about the wire
//! format (that belongs to `uets`), files, or networking — see
//! [`crate::matched`] for the timeframe reconstruction that sits on top.

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
