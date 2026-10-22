//! Deterministic CryptoL2 merged-Parquet playback.
//!
//! # What this reads
//!
//! The **ingest** merged table written by
//! `components/pipelines/ingestion/CryptoL2/merge_events.py`, i.e.
//! `market_data/ingest_parquet/<SYMBOL>/merged/<DATE>/<HH>.parquet`. That is
//! a *different table* from the event-merger matched Parquet
//! (`market_data/matched/...`), and the two decoders are deliberately kept
//! apart:
//!
//! | | CryptoL2 ingest merged | event-merger matched |
//! |---|---|---|
//! | producer | `ingestion/CryptoL2/merge_events.py` | `event_merger/{merger,live}` |
//! | book columns | `best_bid_price`, `bid_price_2..20` | `book_state__best_bid_price` |
//! | trade columns | `trade_price`, `agg_trade_id` | `agg__price`, `agg__agg_trade_id` |
//! | depth columns | `bd_depth_m5..p5` | `depth_state__-5..5` |
//! | event types | `LOB`, `TRADE`, `DEPTH` | `LOB`, `TRADE`, `BOOKDEPTH` |
//! | `timestamp` unit | epoch **milliseconds** | epoch **nanoseconds** |
//! | identity columns | none | `event_sequence`, `lob__final_update_id` |
//!
//! The matched table's decoder lives in `stream_handler` and is not reused
//! here: the two producers write different schemas, so sharing a reader would
//! only make it easier to feed one file to the wrong decoder.
//!
//! # Determinism
//!
//! Playback order is a **pure function of the decoded row contents**, never of
//! how Parquet happened to lay the file out:
//!
//! 1. **Decode** every row, rejecting malformed rows with a per-row error
//!    rather than coercing bad values into plausible ones.
//! 2. **Sort** by `(timestamp, source rank, payload)`. The payload comparison
//!    makes the order total, so dictionary encoding, row-group boundaries and
//!    row iteration order cannot change the result.
//! 3. **Deduplicate** consecutive rows that compare equal under that same
//!    order — which, because the order is total on decoded content, means
//!    exactly "byte-identical decoded row".
//!
//! The ingest merged schema carries no `event_sequence` and no
//! `lob__final_update_id`, so two book updates sharing a millisecond are
//! ordered by their book contents. That is deterministic, but it is not the
//! exchange's causal order — see `MergedRow::cmp_for_playback`.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use uets::core::feature::{
    CANON_AGG_IS_BUYER_MAKER, CANON_AGG_PRICE, CANON_AGG_QUANTITY, CANON_BEST_ASK_PRICE,
    CANON_BEST_ASK_QTY, CANON_BEST_BID_PRICE, CANON_BEST_BID_QTY, CanonicalIntegers,
    CanonicalValues, EVENT_TYPE_LOB, FeatureRow, ask_price_column, ask_qty_column,
    book_price_column, book_qty_column, row_from_canonical,
};

use crate::playback::MarketEvent;

// ---------------------------------------------------------------------------
// Flat column names of the CryptoL2 ingest merged table.
//
// These intentionally mirror `ingestion/CryptoL2/merge_events.py` and NOT the
// canonical `book_state__*` / `agg__*` names — those are produced by
// `MergedRow::to_canonical`, which is the single place the rename happens.
// ---------------------------------------------------------------------------

const COL_TIMESTAMP: &str = "timestamp";
const COL_EVENT_TYPE: &str = "event_type";
const COL_BEST_BID_PRICE: &str = "best_bid_price";
const COL_BEST_BID_QTY: &str = "best_bid_qty";
const COL_BEST_ASK_PRICE: &str = "best_ask_price";
const COL_BEST_ASK_QTY: &str = "best_ask_qty";
const COL_AGG_TRADE_ID: &str = "agg_trade_id";
const COL_TRADE_PRICE: &str = "trade_price";
const COL_TRADE_QTY: &str = "trade_qty";
const COL_FIRST_TRADE_ID: &str = "first_trade_id";
const COL_LAST_TRADE_ID: &str = "last_trade_id";
const COL_IS_BUYER_MAKER: &str = "is_buyer_maker";

/// Number of book levels per side: the best level plus levels 2..=20.
const TOTAL_LEVELS: usize = 20;

/// First level index that gets its own `bid_price_{n}` columns. Level 1 is
/// carried by the `best_*` columns.
const FIRST_INDEXED_LEVEL: usize = 2;

/// The ten book-depth bands, in emission order, and the `m`/`p` infix their
/// merged-table columns use. Band 0 is excluded, matching both producers.
const BAND_LABELS: [&str; 10] = ["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5"];
const BAND_INFIXES: [&str; 10] = ["m5", "m4", "m3", "m2", "m1", "p1", "p2", "p3", "p4", "p5"];

/// Canonical names that `uets::core::feature::ingest` does not export as
/// constants. Kept here as literals so a rename is caught by the schema test
/// at the bottom of this file rather than silently dropping a column.
const CANON_AGG_FIRST_TRADE_ID: &str = "agg__first_trade_id";
const CANON_AGG_LAST_TRADE_ID: &str = "agg__last_trade_id";
const CANON_DEPTH_STATE: &str = "depth_state__";
const CANON_DEPTH_STATE_NOTIONAL: &str = "depth_state__notional_";

/// Which producer stream a row came from.
///
/// The discriminants match `event_merger/merger/model.rs::Source`
/// (`Lob = 0, Trade = 1, Depth = 2`), so the tie-break here has the same
/// shape as the historical merger's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    Lob = 0,
    Trade = 1,
    Depth = 2,
}

/// `event_type` as written by the CryptoL2 ingest merger.
///
/// Note `DEPTH`, not the matched table's `BOOKDEPTH`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventType {
    Lob,
    Trade,
    Depth,
}

impl EventType {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "LOB" => Some(Self::Lob),
            "TRADE" => Some(Self::Trade),
            "DEPTH" => Some(Self::Depth),
            _ => None,
        }
    }

    /// The literal the ingest merger wrote to `event_type`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Lob => EVENT_TYPE_LOB,
            Self::Trade => "TRADE",
            Self::Depth => "DEPTH",
        }
    }

    /// Tie-break rank, mirroring the historical `Source` discriminants.
    pub fn source(self) -> Source {
        match self {
            Self::Lob => Source::Lob,
            Self::Trade => Source::Trade,
            Self::Depth => Source::Depth,
        }
    }
}

/// One depth band of a `DEPTH` row.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Band {
    pub depth: Option<f64>,
    pub notional: Option<f64>,
}

/// One book level above the best, kept in level order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BookLevel {
    pub bid_price: Option<f64>,
    pub bid_qty: Option<f64>,
    pub ask_price: Option<f64>,
    pub ask_qty: Option<f64>,
}

/// One decoded row of the CryptoL2 ingest merged table.
///
/// `None` means *absent*, never zero: a `TRADE` row has no book, and reading
/// its absent bid as `0.0` would fabricate a catastrophic mid price.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergedRow {
    /// Exchange event time, epoch **milliseconds** (the merged table's unit).
    pub timestamp_ms: i64,
    pub event_type: Option<EventType>,
    /// The raw `event_type` literal, kept for diagnostics when unparsed.
    pub event_type_raw: String,

    pub best_bid_price: Option<f64>,
    pub best_bid_qty: Option<f64>,
    pub best_ask_price: Option<f64>,
    pub best_ask_qty: Option<f64>,
    /// Levels 2..=20, so `levels[i]` is level `i + FIRST_INDEXED_LEVEL`.
    pub levels: Vec<BookLevel>,

    pub agg_trade_id: Option<i64>,
    pub trade_price: Option<f64>,
    pub trade_qty: Option<f64>,
    pub first_trade_id: Option<i64>,
    pub last_trade_id: Option<i64>,
    pub is_buyer_maker: Option<bool>,

    /// The ten bands in `BAND_LABELS` order.
    pub bands: Vec<Band>,
}

impl MergedRow {
    /// Top-of-book mid price, or `None` while either side is absent.
    ///
    /// `TRADE` and `DEPTH` rows in this table carry no book at all, so they
    /// report `None` here and `prepare_playback` carries the last known mid
    /// forward onto their marks.
    pub fn mid(&self) -> Option<f64> {
        match (self.best_bid_price, self.best_ask_price) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    /// Total, content-based playback order.
    ///
    /// Keys, in order:
    ///
    /// 1. `timestamp` (the merged table's millisecond event time),
    /// 2. source rank (`LOB` < `TRADE` < `DEPTH`),
    /// 3. the payload itself, compared field by field.
    ///
    /// Step 3 is what makes replay independent of Parquet layout: without it,
    /// two book updates inside one millisecond would keep whatever order the
    /// file listed them in. It is deterministic but *not* causal — the ingest
    /// table records no update id, so the exchange's real order inside a
    /// millisecond is not recoverable from this file.
    pub fn cmp_for_playback(&self, other: &Self) -> Ordering {
        self.timestamp_ms
            .cmp(&other.timestamp_ms)
            .then_with(|| rank(self.event_type).cmp(&rank(other.event_type)))
            .then_with(|| self.event_type_raw.cmp(&other.event_type_raw))
            .then_with(|| cmp_opt_f64(self.best_bid_price, other.best_bid_price))
            .then_with(|| cmp_opt_f64(self.best_bid_qty, other.best_bid_qty))
            .then_with(|| cmp_opt_f64(self.best_ask_price, other.best_ask_price))
            .then_with(|| cmp_opt_f64(self.best_ask_qty, other.best_ask_qty))
            .then_with(|| cmp_levels(&self.levels, &other.levels))
            .then_with(|| cmp_opt_i64(self.agg_trade_id, other.agg_trade_id))
            .then_with(|| cmp_opt_f64(self.trade_price, other.trade_price))
            .then_with(|| cmp_opt_f64(self.trade_qty, other.trade_qty))
            .then_with(|| cmp_opt_i64(self.first_trade_id, other.first_trade_id))
            .then_with(|| cmp_opt_i64(self.last_trade_id, other.last_trade_id))
            .then_with(|| cmp_opt_bool(self.is_buyer_maker, other.is_buyer_maker))
            .then_with(|| cmp_bands(&self.bands, &other.bands))
    }

    /// Map this flat ingest row onto the canonical merge-table columns.
    ///
    /// `event_sequence` is *assigned by playback*, because the ingest merged
    /// table has no sequence column: the caller passes the row's 0-based
    /// position in the deduplicated stream, which is stable across runs.
    ///
    /// Absence is preserved end to end: a column the row did not carry is
    /// inserted as `None`, which `row_from_canonical` records as `Missing`.
    /// The single deliberate normalization is the trade's transact time — the
    /// merged table records one event time per row, so it is reused rather
    /// than fabricated from wall-clock time.
    pub fn to_canonical(&self, event_sequence: u64) -> CanonicalRow {
        let mut values = CanonicalValues::new();
        let mut set = |name: String, value: Option<f64>| {
            values.insert(name, value);
        };

        for (name, value) in [
            (CANON_BEST_BID_PRICE, self.best_bid_price),
            (CANON_BEST_BID_QTY, self.best_bid_qty),
            (CANON_BEST_ASK_PRICE, self.best_ask_price),
            (CANON_BEST_ASK_QTY, self.best_ask_qty),
        ] {
            set(name.to_string(), value);
        }

        for (offset, level) in self.levels.iter().enumerate() {
            let n = offset + FIRST_INDEXED_LEVEL;
            set(book_price_column(n), level.bid_price);
            set(book_qty_column(n), level.bid_qty);
            set(ask_price_column(n), level.ask_price);
            set(ask_qty_column(n), level.ask_qty);
        }

        for (label, band) in BAND_LABELS.iter().zip(self.bands.iter()) {
            set(format!("{CANON_DEPTH_STATE}{label}"), band.depth);
            set(
                format!("{CANON_DEPTH_STATE_NOTIONAL}{label}"),
                band.notional,
            );
        }

        set(CANON_AGG_PRICE.to_string(), self.trade_price);
        set(CANON_AGG_QUANTITY.to_string(), self.trade_qty);
        set(
            CANON_AGG_FIRST_TRADE_ID.to_string(),
            self.first_trade_id.map(|v| v as f64),
        );
        set(
            CANON_AGG_LAST_TRADE_ID.to_string(),
            self.last_trade_id.map(|v| v as f64),
        );
        set(
            CANON_AGG_IS_BUYER_MAKER.to_string(),
            self.is_buyer_maker.map(|flag| if flag { 1.0 } else { 0.0 }),
        );

        let is_lob = self.event_type == Some(EventType::Lob);
        let is_trade = self.event_type == Some(EventType::Trade);
        let integers = CanonicalIntegers {
            timestamp_ns: self.timestamp_ms.saturating_mul(1_000_000),
            event_sequence,
            // `lob__event_time` is the LOB row's own time; a trade has no LOB
            // identity, so it stays absent rather than reading as "time 0".
            event_time_ms: is_lob.then_some(self.timestamp_ms),
            lob_final_update_id: None,
            agg_trade_id: self
                .agg_trade_id
                .and_then(|id| u64::try_from(id).ok())
                .filter(|_| is_trade),
            agg_transact_time_ms: is_trade.then_some(self.timestamp_ms),
        };

        CanonicalRow {
            event_type: self.event_type.map(EventType::name),
            feature_row: row_from_canonical(&values, &integers),
        }
    }
}

/// One canonical row produced from a [`MergedRow`].
///
/// `event_type` rides alongside the [`FeatureRow`] rather than inside it
/// because the canonical row encodes values as `f64`, and a string column
/// cannot survive that trip.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalRow {
    /// `"LOB"`, `"TRADE"` or `"DEPTH"`; `None` on an unparsed row.
    pub event_type: Option<&'static str>,
    pub feature_row: FeatureRow,
}

/// Sort, deduplicate and carry marks forward through a decoded stream.
///
/// Returns the deduplicated rows in playback order together with the
/// [`MarketEvent`]s to feed a [`crate::playback::PlaybackBackend`].
pub fn prepare_playback(rows: Vec<MergedRow>, symbol: &str) -> Playback {
    let mut rows = rows;
    let before_dedup = rows.len();
    rows.sort_unstable_by(MergedRow::cmp_for_playback);
    rows.dedup_by(|a, b| a.cmp_for_playback(b) == Ordering::Equal);
    let duplicates_removed = before_dedup - rows.len();

    // Carry the last known mid forward: TRADE and DEPTH rows carry no book,
    // and a mark that flickered to "no mark" every trade would make barrier
    // checks time out on a perfectly tradable tape.
    let mut last_mid: Option<f64> = None;
    let mut events = Vec::with_capacity(rows.len());
    for row in &rows {
        if let Some(mid) = row.mid() {
            last_mid = Some(mid);
        }
        // Nothing has established a book yet: skip rather than invent a mark
        // of 0.0 for the opening rows.
        let Some(mid) = last_mid else { continue };
        let Some(timestamp) = ms_to_datetime(row.timestamp_ms) else {
            continue;
        };
        let mut marks = std::collections::BTreeMap::new();
        marks.insert(symbol.to_string(), mid);
        events.push(MarketEvent { timestamp, marks });
    }

    Playback {
        rows,
        events,
        duplicates_removed,
    }
}

/// Decoded rows in playback order, plus the accelerated-replay events.
#[derive(Clone, Debug, Default)]
pub struct Playback {
    /// The deduplicated stream, sorted by `MergedRow::cmp_for_playback`.
    pub rows: Vec<MergedRow>,
    /// One mark event per row that had a book available (own or carried).
    pub events: Vec<MarketEvent>,
    /// Rows dropped by deduplication.
    pub duplicates_removed: usize,
}

impl Playback {
    /// Canonical feature rows in playback order, one per deduplicated row.
    ///
    /// `event_sequence` is the row's 0-based position, which is stable across
    /// runs because the stream itself is deterministically ordered.
    pub fn canonical_rows(&self) -> Vec<CanonicalRow> {
        self.rows
            .iter()
            .enumerate()
            .map(|(index, row)| row.to_canonical(index as u64))
            .collect()
    }
}

/// Why one merged row could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowError {
    MissingColumn(&'static str),
    /// A present column whose type cannot represent this schema.
    WrongType(String),
    NullTimestamp,
    NegativeTimestamp(i64),
    UnknownEventType(String),
}

impl std::fmt::Display for RowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingColumn(name) => write!(f, "required column `{name}` is absent"),
            Self::WrongType(name) => write!(f, "column `{name}` has an unexpected type"),
            Self::NullTimestamp => write!(f, "`timestamp` is null"),
            Self::NegativeTimestamp(v) => write!(f, "`timestamp` is negative ({v})"),
            Self::UnknownEventType(raw) => write!(f, "unknown `event_type` {raw:?}"),
        }
    }
}

impl std::error::Error for RowError {}

/// A file-level playback failure.
#[derive(Debug)]
pub enum PlaybackError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parquet {
        path: PathBuf,
        source: parquet::errors::ParquetError,
    },
    Row {
        path: PathBuf,
        row: RowError,
    },
}

impl std::fmt::Display for PlaybackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "failed to open {}: {source}", path.display())
            }
            Self::Parquet { path, source } => {
                write!(f, "failed to read {}: {source}", path.display())
            }
            Self::Row { path, row } => {
                write!(f, "malformed row in {}: {row}", path.display())
            }
        }
    }
}

impl std::error::Error for PlaybackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parquet { source, .. } => Some(source),
            Self::Row { row, .. } => Some(row),
        }
    }
}

fn rank(event_type: Option<EventType>) -> u8 {
    match event_type {
        Some(EventType::Lob) => 0,
        Some(EventType::Trade) => 1,
        Some(EventType::Depth) => 2,
        None => 3,
    }
}

fn cmp_opt_f64(a: Option<f64>, b: Option<f64>) -> Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
    }
}

fn cmp_opt_i64(a: Option<i64>, b: Option<i64>) -> Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.cmp(&y),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
    }
}

fn cmp_opt_bool(a: Option<bool>, b: Option<bool>) -> Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.cmp(&y),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
    }
}

fn cmp_levels(a: &[BookLevel], b: &[BookLevel]) -> Ordering {
    let ord = a
        .iter()
        .zip(b.iter())
        .fold(Ordering::Equal, |acc, (la, lb)| {
            acc.then_with(|| cmp_opt_f64(la.bid_price, lb.bid_price))
                .then_with(|| cmp_opt_f64(la.bid_qty, lb.bid_qty))
                .then_with(|| cmp_opt_f64(la.ask_price, lb.ask_price))
                .then_with(|| cmp_opt_f64(la.ask_qty, lb.ask_qty))
        });
    ord.then_with(|| a.len().cmp(&b.len()))
}

fn cmp_bands(a: &[Band], b: &[Band]) -> Ordering {
    let ord = a
        .iter()
        .zip(b.iter())
        .fold(Ordering::Equal, |acc, (ba, bb)| {
            acc.then_with(|| cmp_opt_f64(ba.depth, bb.depth))
                .then_with(|| cmp_opt_f64(ba.notional, bb.notional))
        });
    ord.then_with(|| a.len().cmp(&b.len()))
}

/// Milliseconds since epoch -> UTC DateTime.
fn ms_to_datetime(ms: i64) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::from_timestamp_millis(ms)
}

/// Decode one `RecordBatch` into `MergedRow`s, failing on the first bad row.
#[allow(clippy::type_complexity)]
fn decode_batch(batch: &RecordBatch) -> Result<Vec<MergedRow>, RowError> {
    let num_rows = batch.num_rows();

    let timestamp_col = batch
        .column_by_name(COL_TIMESTAMP)
        .ok_or(RowError::MissingColumn(COL_TIMESTAMP))?;
    let event_type_col = batch
        .column_by_name(COL_EVENT_TYPE)
        .ok_or(RowError::MissingColumn(COL_EVENT_TYPE))?;

    let best_bid_price_col = optional_float_column(batch, COL_BEST_BID_PRICE)?;
    let best_bid_qty_col = optional_float_column(batch, COL_BEST_BID_QTY)?;
    let best_ask_price_col = optional_float_column(batch, COL_BEST_ASK_PRICE)?;
    let best_ask_qty_col = optional_float_column(batch, COL_BEST_ASK_QTY)?;

    let level_cols: Vec<(
        Option<&Float64Array>,
        Option<&Float64Array>,
        Option<&Float64Array>,
        Option<&Float64Array>,
    )> = (FIRST_INDEXED_LEVEL..=TOTAL_LEVELS)
        .map(|n| {
            Ok((
                optional_float_column(batch, &book_price_column(n))?,
                optional_float_column(batch, &book_qty_column(n))?,
                optional_float_column(batch, &ask_price_column(n))?,
                optional_float_column(batch, &ask_qty_column(n))?,
            ))
        })
        .collect::<Result<_, RowError>>()?;

    let agg_trade_id_col = optional_int_column(batch, COL_AGG_TRADE_ID)?;
    let trade_price_col = optional_float_column(batch, COL_TRADE_PRICE)?;
    let trade_qty_col = optional_float_column(batch, COL_TRADE_QTY)?;
    let first_trade_id_col = optional_int_column(batch, COL_FIRST_TRADE_ID)?;
    let last_trade_id_col = optional_int_column(batch, COL_LAST_TRADE_ID)?;
    let is_buyer_maker_col = optional_bool_column(batch, COL_IS_BUYER_MAKER)?;

    let depth_cols: Vec<Option<&Float64Array>> = BAND_INFIXES
        .iter()
        .map(|infix| optional_float_column(batch, &format!("bd_depth_{infix}")))
        .collect::<Result<_, RowError>>()?;
    let notional_cols: Vec<Option<&Float64Array>> = BAND_INFIXES
        .iter()
        .map(|infix| optional_float_column(batch, &format!("bd_notional_{infix}")))
        .collect::<Result<_, RowError>>()?;

    let timestamp_array = timestamp_col
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| RowError::WrongType(COL_TIMESTAMP.to_string()))?;
    let event_type_array = event_type_col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| RowError::WrongType(COL_EVENT_TYPE.to_string()))?;

    let mut rows = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        if timestamp_array.is_null(i) {
            return Err(RowError::NullTimestamp);
        }
        let ts = timestamp_array.value(i);
        if ts < 0 {
            return Err(RowError::NegativeTimestamp(ts));
        }

        let raw = event_type_array.value(i).to_string();
        let event_type =
            EventType::parse(&raw).ok_or_else(|| RowError::UnknownEventType(raw.clone()))?;

        rows.push(MergedRow {
            timestamp_ms: ts,
            event_type: Some(event_type),
            event_type_raw: raw,
            best_bid_price: opt_f64(best_bid_price_col, i),
            best_bid_qty: opt_f64(best_bid_qty_col, i),
            best_ask_price: opt_f64(best_ask_price_col, i),
            best_ask_qty: opt_f64(best_ask_qty_col, i),
            levels: level_cols
                .iter()
                .map(|(bp, bq, ap, aq)| BookLevel {
                    bid_price: opt_f64(*bp, i),
                    bid_qty: opt_f64(*bq, i),
                    ask_price: opt_f64(*ap, i),
                    ask_qty: opt_f64(*aq, i),
                })
                .collect(),
            agg_trade_id: opt_i64(agg_trade_id_col, i),
            trade_price: opt_f64(trade_price_col, i),
            trade_qty: opt_f64(trade_qty_col, i),
            first_trade_id: opt_i64(first_trade_id_col, i),
            last_trade_id: opt_i64(last_trade_id_col, i),
            is_buyer_maker: opt_bool(is_buyer_maker_col, i),
            bands: depth_cols
                .iter()
                .zip(notional_cols.iter())
                .map(|(d, n)| Band {
                    depth: opt_f64(*d, i),
                    notional: opt_f64(*n, i),
                })
                .collect(),
        });
    }

    Ok(rows)
}

fn optional_float_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a Float64Array>, RowError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    col.as_any()
        .downcast_ref::<Float64Array>()
        .map(Some)
        .ok_or_else(|| RowError::WrongType(name.to_string()))
}

fn optional_int_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a Int64Array>, RowError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    col.as_any()
        .downcast_ref::<Int64Array>()
        .map(Some)
        .ok_or_else(|| RowError::WrongType(name.to_string()))
}

fn optional_bool_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a BooleanArray>, RowError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    col.as_any()
        .downcast_ref::<BooleanArray>()
        .map(Some)
        .ok_or_else(|| RowError::WrongType(name.to_string()))
}

fn opt_f64(arr: Option<&Float64Array>, i: usize) -> Option<f64> {
    arr.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
}

fn opt_i64(arr: Option<&Int64Array>, i: usize) -> Option<i64> {
    arr.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
}

fn opt_bool(arr: Option<&BooleanArray>, i: usize) -> Option<bool> {
    arr.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
}

/// Read one merged Parquet file and decode every row.
pub fn load_merged_parquet(path: &Path, symbol: &str) -> Result<Vec<MergedRow>, PlaybackError> {
    let _ = symbol; // reserved for future error-context use
    let file = std::fs::File::open(path).map_err(|source| PlaybackError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| {
        PlaybackError::Parquet {
            path: path.to_path_buf(),
            source,
        }
    })?;

    let reader = builder.build().map_err(|source| PlaybackError::Parquet {
        path: path.to_path_buf(),
        source,
    })?;
    let mut all_rows = Vec::new();
    for result in reader {
        let batch = result.map_err(|source| PlaybackError::Parquet {
            path: path.to_path_buf(),
            source: source.into(),
        })?;
        let rows = decode_batch(&batch).map_err(|row| PlaybackError::Row {
            path: path.to_path_buf(),
            row,
        })?;
        all_rows.extend(rows);
    }

    Ok(all_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(timestamp_ms: i64, event_type: EventType) -> MergedRow {
        MergedRow {
            timestamp_ms,
            event_type: Some(event_type),
            event_type_raw: event_type.name().to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn sort_orders_by_timestamp_then_source_then_payload() {
        let mut rows = [make_row(1000, EventType::Trade),
            make_row(1000, EventType::Lob),
            make_row(999, EventType::Depth)];
        rows.sort_unstable_by(MergedRow::cmp_for_playback);
        assert_eq!(rows[0].timestamp_ms, 999);
        assert_eq!(rows[0].event_type, Some(EventType::Depth));
        assert_eq!(rows[1].timestamp_ms, 1000);
        assert_eq!(rows[1].event_type, Some(EventType::Lob));
        assert_eq!(rows[2].timestamp_ms, 1000);
        assert_eq!(rows[2].event_type, Some(EventType::Trade));
    }

    #[test]
    fn dedup_removes_content_duplicates() {
        let mut rows = vec![
            make_row(1000, EventType::Lob),
            make_row(1000, EventType::Lob),
            make_row(1000, EventType::Trade),
        ];
        rows.sort_unstable_by(MergedRow::cmp_for_playback);
        let before = rows.len();
        rows.dedup_by(|a, b| a.cmp_for_playback(b) == Ordering::Equal);
        assert_eq!(rows.len(), before - 1);
        assert_eq!(
            rows.iter()
                .filter(|r| r.event_type == Some(EventType::Lob))
                .count(),
            1
        );
    }

    #[test]
    fn mid_is_none_without_book() {
        let row = make_row(1000, EventType::Trade);
        assert_eq!(row.mid(), None);
    }

    #[test]
    fn mid_is_average_of_best_bid_ask() {
        let mut row = make_row(1000, EventType::Lob);
        row.best_bid_price = Some(99.5);
        row.best_ask_price = Some(100.5);
        assert_eq!(row.mid(), Some(100.0));
    }

    #[test]
    fn to_canonical_assigns_event_sequence() {
        let row = make_row(1000, EventType::Lob);
        let canonical = row.to_canonical(42);
        assert_eq!(canonical.event_type, Some("LOB"));
        assert_eq!(
            canonical.feature_row.get("timestamp"),
            Some(1_000_000_000.0)
        );
        assert_eq!(canonical.feature_row.get("event_sequence"), Some(42.0));
    }

    #[test]
    fn prepare_playback_carries_mid_forward() {
        let mut rows = vec![
            make_row(1000, EventType::Trade),
            MergedRow {
                timestamp_ms: 1001,
                event_type: Some(EventType::Lob),
                event_type_raw: "LOB".to_string(),
                best_bid_price: Some(99.0),
                best_bid_qty: Some(1.0),
                best_ask_price: Some(101.0),
                best_ask_qty: Some(1.0),
                ..Default::default()
            },
        ];
        rows.sort_unstable_by(MergedRow::cmp_for_playback);
        let playback = prepare_playback(rows, "BTCUSDT");
        assert_eq!(playback.events.len(), 1);
        assert_eq!(playback.events[0].marks.get("BTCUSDT"), Some(&100.0));
    }

    #[test]
    fn prepare_playback_skips_before_first_book() {
        let rows = vec![
            make_row(1000, EventType::Trade),
            make_row(1001, EventType::Trade),
        ];
        let playback = prepare_playback(rows, "BTCUSDT");
        assert_eq!(playback.events.len(), 0);
        assert_eq!(playback.duplicates_removed, 0);
    }

    #[test]
    fn row_error_display() {
        let err = RowError::MissingColumn("timestamp");
        assert_eq!(err.to_string(), "required column `timestamp` is absent");
    }

    #[test]
    fn schema_constants_match_merged_table() {
        assert_eq!(COL_TIMESTAMP, "timestamp");
        assert_eq!(COL_EVENT_TYPE, "event_type");
        assert_eq!(COL_BEST_BID_PRICE, "best_bid_price");
        assert_eq!(COL_AGG_TRADE_ID, "agg_trade_id");
        assert_eq!(COL_TRADE_PRICE, "trade_price");
        assert_eq!(COL_IS_BUYER_MAKER, "is_buyer_maker");
        assert_eq!(format!("bd_depth_{}", BAND_INFIXES[0]), "bd_depth_m5");
        assert_eq!(format!("bd_depth_{}", BAND_INFIXES[9]), "bd_depth_p5");
        assert_eq!(format!("bd_notional_{}", BAND_INFIXES[0]), "bd_notional_m5");
    }
}
