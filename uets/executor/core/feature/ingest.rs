//! Turn a raw market-data event into a [`FeatureRow`].
//!
//! This is the "query" half of `core::feature`, kept strictly separate from
//! `compute`: it knows about *where numbers come from*, never about *what is
//! computed from them*. That split is what lets the same feature definitions
//! run over a live websocket, a recorded file, or the parity harness.
//!
//! Binance sends prices and sizes as JSON **strings**, so they are parsed
//! explicitly here rather than being coerced by a generic deserializer.

use super::row::FeatureRow;

/// Canonical column names produced by [`row_from_frame`].
///
/// A feature spec refers to these, not to Binance's wire keys, so renaming a
/// wire field is a one-line change here instead of a spec rewrite.
pub const COL_EVENT_TIME: &str = "event_time_ms";
pub const COL_AGG_TRADE_ID: &str = "agg_trade_id";
pub const COL_PRICE: &str = "price";
pub const COL_QUANTITY: &str = "quantity";
pub const COL_BEST_BID_PRICE: &str = "best_bid_price";
pub const COL_BEST_BID_QTY: &str = "best_bid_qty";
pub const COL_BEST_ASK_PRICE: &str = "best_ask_price";
pub const COL_BEST_ASK_QTY: &str = "best_ask_qty";

/// Canonical column names taken from the merged table.
///
/// These are the `schema()` names of the live/historical merger, so a feature
/// spec written against a recorded capture names exactly the same columns the
/// live path produces. That shared naming is what lets the Python research
/// spec and the Rust executor agree without a remapping table.
pub const CANON_TIMESTAMP_NS: &str = "timestamp";
pub const CANON_EVENT_TYPE: &str = "event_type";
pub const CANON_EVENT_SEQUENCE: &str = "event_sequence";
pub const CANON_LOB_EVENT_TIME: &str = "lob__event_time";
pub const CANON_LOB_FINAL_UPDATE_ID: &str = "lob__final_update_id";
pub const CANON_AGG_TRADE_ID: &str = "agg__agg_trade_id";
pub const CANON_AGG_PRICE: &str = "agg__price";
pub const CANON_AGG_QUANTITY: &str = "agg__quantity";
pub const CANON_AGG_TRANSACT_TIME: &str = "agg__transact_time";
pub const CANON_AGG_IS_BUYER_MAKER: &str = "agg__is_buyer_maker";

/// `event_type` value on a row that carried a book update.
pub const EVENT_TYPE_LOB: &str = "LOB";
/// `event_type` value on a row that carried a trade.
pub const EVENT_TYPE_TRADE: &str = "TRADE";

/// `book_state__*` column names for a specific depth level.
pub fn book_price_column(level: usize) -> String {
    format!("book_state__bid_price_{level}")
}

pub fn book_qty_column(level: usize) -> String {
    format!("book_state__bid_qty_{level}")
}

pub fn ask_price_column(level: usize) -> String {
    format!("book_state__ask_price_{level}")
}

pub fn ask_qty_column(level: usize) -> String {
    format!("book_state__ask_qty_{level}")
}

pub const CANON_BEST_BID_PRICE: &str = "book_state__best_bid_price";
pub const CANON_BEST_BID_QTY: &str = "book_state__best_bid_qty";
pub const CANON_BEST_ASK_PRICE: &str = "book_state__best_ask_price";
pub const CANON_BEST_ASK_QTY: &str = "book_state__best_ask_qty";

/// One canonical row, encoded as the merge table's column values.
///
/// `None` means *absent*, which is not the same as `Some(0.0)`.
pub type CanonicalValues = std::collections::BTreeMap<String, Option<f64>>;

/// Identity fields a canonical row carries alongside the float map.
///
/// Kept separate because a timestamp or sequence number is an identity, not a
/// measurement: feeding `event_sequence` into a `mean()` is meaningless in a
/// way that feeding a price is not.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CanonicalIntegers {
    pub timestamp_ns: i64,
    pub event_sequence: u64,
    pub event_time_ms: Option<i64>,
    pub lob_final_update_id: Option<i64>,
    pub agg_trade_id: Option<u64>,
    pub agg_transact_time_ms: Option<i64>,
}
/// Build a [`FeatureRow`] from one canonical row.
///
/// Every column the row does not carry is recorded as `Missing`, never omitted
/// and never zero. A TRADE row genuinely has no `lob__*` identity; writing
/// `0.0` there would let a feature read "last update at time 0" and compute
/// garbage instead of failing loudly.
pub fn row_from_canonical(values: &CanonicalValues, integers: &CanonicalIntegers) -> FeatureRow {
    let mut row = FeatureRow::new();
    for (name, value) in values {
        match value {
            Some(v) => row.insert(name.clone(), *v),
            None => row.insert_missing(name.clone()),
        }
    }

    // Identity columns are always present, so features over them are total
    // functions of the stream rather than of whatever the row happened to hold.
    set(
        &mut row,
        CANON_TIMESTAMP_NS,
        Some(integers.timestamp_ns as f64),
    );
    set(
        &mut row,
        CANON_EVENT_SEQUENCE,
        Some(integers.event_sequence as f64),
    );
    set(&mut row, CANON_EVENT_TYPE, None);
    set(
        &mut row,
        CANON_LOB_EVENT_TIME,
        integers.event_time_ms.map(|v| v as f64),
    );
    set(
        &mut row,
        CANON_LOB_FINAL_UPDATE_ID,
        integers.lob_final_update_id.map(|v| v as f64),
    );
    set(
        &mut row,
        CANON_AGG_TRADE_ID,
        integers.agg_trade_id.map(|v| v as f64),
    );
    set(
        &mut row,
        CANON_AGG_TRANSACT_TIME,
        integers.agg_transact_time_ms.map(|v| v as f64),
    );
    row
}

/// The top of book, read from a canonical row while preserving absence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CanonicalTop {
    pub best_bid_price: Option<f64>,
    pub best_bid_qty: Option<f64>,
    pub best_ask_price: Option<f64>,
    pub best_ask_qty: Option<f64>,
}

impl CanonicalTop {
    pub fn from_values(values: &CanonicalValues) -> Self {
        let get = |name: &str| values.get(name).copied().flatten();
        Self {
            best_bid_price: get(CANON_BEST_BID_PRICE),
            best_bid_qty: get(CANON_BEST_BID_QTY),
            best_ask_price: get(CANON_BEST_ASK_PRICE),
            best_ask_qty: get(CANON_BEST_ASK_QTY),
        }
    }

    /// Mid price, or `None` while either side is unknown.
    pub fn mid(&self) -> Option<f64> {
        match (self.best_bid_price, self.best_ask_price) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    /// Top-of-book spread, or `None` while either side is unknown.
    ///
    /// Returning `None` rather than a clamped zero matters: an inverted book
    /// is a data fault, and reporting it as a real number would hide that
    /// fault inside a downstream feature.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid_price, self.best_ask_price) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    /// Order-book imbalance in [-1, 1], or `None` without both quantities.
    pub fn imbalance(&self) -> Option<f64> {
        let (b, a) = (self.best_bid_qty?, self.best_ask_qty?);
        let total = b + a;
        if total <= 0.0 {
            return None;
        }
        Some((b - a) / total)
    }
}

/// Top-of-book carried across events.
///
/// Kept separate from `FeatureRow` so a row holds only what a single frame
/// actually observed, and carried levels stay clearly derived.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BookTop {
    pub best_bid_price: Option<f64>,
    pub best_bid_qty: Option<f64>,
    pub best_ask_price: Option<f64>,
    pub best_ask_qty: Option<f64>,
}

impl BookTop {
    /// Update the top of book from one partial-book frame.
    ///
    /// `bids` arrives price-descending and `asks` price-ascending, so the
    /// first entry on each side is the best.
    pub fn apply_partial(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        if let Some((price, qty)) = bids.first() {
            self.best_bid_price = Some(*price);
            self.best_bid_qty = Some(*qty);
        }
        if let Some((price, qty)) = asks.first() {
            self.best_ask_price = Some(*price);
            self.best_ask_qty = Some(*qty);
        }
    }
}

/// Build one row from a parsed websocket frame plus the current top of book.
///
/// A value the frame does not carry is recorded as `Missing` rather than
/// omitted, so a feature needing it fails loudly instead of reading a
/// silently-absent column as zero.
pub fn row_from_frame(frame: &serde_json::Value, book: &BookTop) -> FeatureRow {
    let mut row = FeatureRow::new();

    let event_type = frame
        .get("e")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();

    match event_type {
        "aggTrade" => {
            set(&mut row, COL_EVENT_TIME, num(frame.get("E")));
            set(&mut row, CANON_AGG_TRADE_ID, num(frame.get("a")));
            set(&mut row, COL_PRICE, num(frame.get("p")));
            set(&mut row, COL_QUANTITY, num(frame.get("q")));
        }
        "depthUpdate" | "bookTicker" => {
            set(&mut row, COL_EVENT_TIME, num(frame.get("E")));
        }
        _ => {}
    }

    // The top of book is carried onto every row: a trade row still has a book.
    for (name, value) in [
        (CANON_BEST_BID_PRICE, book.best_bid_price),
        (CANON_BEST_BID_QTY, book.best_bid_qty),
        (CANON_BEST_ASK_PRICE, book.best_ask_price),
        (CANON_BEST_ASK_QTY, book.best_ask_qty),
    ] {
        match value {
            Some(v) => row.insert(name, v),
            None => row.insert_missing(name),
        }
    }
    row
}

fn set(row: &mut FeatureRow, name: &str, value: Option<f64>) {
    match value {
        Some(v) => row.insert(name, v),
        None => row.insert_missing(name),
    }
}

/// Parse a Binance numeric field, which may arrive as a string or a number.
pub fn num(value: Option<&serde_json::Value>) -> Option<f64> {
    match value? {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Parse a `"[["price","qty"], ...]` level array.
pub fn parse_levels(value: Option<&serde_json::Value>) -> Vec<(f64, f64)> {
    let Some(serde_json::Value::Array(levels)) = value else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(levels.len());
    for entry in levels {
        let serde_json::Value::Array(pair) = entry else {
            continue;
        };
        if pair.len() < 2 {
            continue;
        }
        let Some(price) = num(Some(&pair[0])) else {
            continue;
        };
        let Some(qty) = num(Some(&pair[1])) else {
            continue;
        };
        out.push((price, qty));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agg_trade_string_fields_are_parsed() {
        let frame: serde_json::Value = serde_json::from_str(
            r#"{"e":"aggTrade","E":1751245200354,"a":2773899591,"p":"108665.6","q":"0.002"}"#,
        )
        .unwrap();
        let row = row_from_frame(&frame, &BookTop::default());
        assert_eq!(row.get(COL_PRICE), Some(108665.6));
        assert_eq!(row.get(COL_QUANTITY), Some(0.002));
        assert_eq!(row.get(CANON_AGG_TRADE_ID), Some(2773899591.0));
        // No book supplied: recorded as missing, not zero.
        assert_eq!(row.get(CANON_BEST_BID_PRICE), None);
        assert!(row.get_value(CANON_BEST_BID_PRICE).unwrap().is_missing());
    }

    #[test]
    fn book_top_is_carried_onto_a_trade_row() {
        let book = BookTop {
            best_bid_price: Some(100.0),
            best_bid_qty: Some(1.0),
            best_ask_price: Some(101.0),
            best_ask_qty: Some(2.0),
        };
        let frame: serde_json::Value =
            serde_json::from_str(r#"{"e":"aggTrade","E":1,"a":2,"p":"100.5","q":"1"}"#).unwrap();
        let row = row_from_frame(&frame, &book);
        assert_eq!(row.get(CANON_BEST_BID_PRICE), Some(100.0));
        assert_eq!(row.get(CANON_BEST_ASK_PRICE), Some(101.0));
    }

    #[test]
    fn parses_string_level_arrays() {
        let value: serde_json::Value =
            serde_json::from_str(r#"[["108665.6","0.002"],["108664.0","1.5"]]"#).unwrap();
        assert_eq!(
            parse_levels(Some(&value)),
            vec![(108665.6, 0.002), (108664.0, 1.5)]
        );
    }

    #[test]
    fn partial_book_takes_the_first_level_per_side() {
        let mut book = BookTop::default();
        book.apply_partial(&[(99.0, 1.0), (98.0, 2.0)], &[(101.0, 3.0), (102.0, 4.0)]);
        assert_eq!(book.best_bid_price, Some(99.0));
        assert_eq!(book.best_ask_price, Some(101.0));
    }
}
