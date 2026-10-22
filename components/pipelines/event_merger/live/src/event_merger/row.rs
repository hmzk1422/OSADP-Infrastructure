//! `row.rs` â€” the canonical 128-column row and its sparse CSV encoding.
//!
//! Column order and names are byte-compatible with the historical output
//! (`event_merger/merger/writer.rs::state_fields`), so a live capture and a
//! historical merge can be read by the same feature code.
//!
//! Sparsity is the point: 100 of the 128 columns are unchanged between book
//! updates, but a fixed-width format still has to emit them. The win therefore
//! comes from *not allocating* them â€” one buffer is reused for every row, and
//! a null costs nothing but a separator.

use std::fmt::Write as _;
use std::sync::OnceLock;

/// Band labels in emission order: -5..-1 then 1..5 (0 is excluded).
const BAND_LABELS: [&str; 10] = ["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5"];

/// Number of columns per row. Guards against silently misaligned output.
pub const FIELD_COUNT: usize = 128;

/// The 128 column names, in emission order. Built once; not a hot path.
pub fn schema() -> &'static [&'static str] {
    static LEAKED: OnceLock<&'static [&'static str]> = OnceLock::new();
    LEAKED.get_or_init(|| Box::leak(build_schema().into_boxed_slice()))
}

fn intern(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn build_schema() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::with_capacity(FIELD_COUNT);
    names.extend([
        "timestamp",
        "event_type",
        "event_sequence",
        "lob__event_time",
        "lob__final_update_id",
        "book_state__best_bid_price",
        "book_state__best_bid_qty",
        "book_state__best_ask_price",
        "book_state__best_ask_qty",
    ]);
    // Levels 2..=20, four columns each = 76.
    for level in 2..=20 {
        names.push(intern(format!("book_state__bid_price_{level}")));
        names.push(intern(format!("book_state__bid_qty_{level}")));
        names.push(intern(format!("book_state__ask_price_{level}")));
        names.push(intern(format!("book_state__ask_qty_{level}")));
    }
    names.extend([
        "agg__agg_trade_id",
        "agg__price",
        "agg__quantity",
        "agg__qty",
        "agg__first_trade_id",
        "agg__last_trade_id",
        "agg__transact_time",
        "agg__is_buyer_maker",
        "depth__timestamp",
        "depth__percentage",
        "depth__depth",
        "depth__notional",
    ]);
    for band in BAND_LABELS {
        names.push(intern(format!("depth_state__{band}")));
    }
    for band in BAND_LABELS {
        names.push(intern(format!("depth_state__notional_{band}")));
    }
    names.extend([
        "source_agg__agg_trade_id",
        "source_agg__price",
        "source_agg__quantity",
        "source_agg__first_trade_id",
        "source_agg__last_trade_id",
        "source_agg__transact_time",
        "source_agg__is_buyer_maker",
        "source_depth__timestamp",
        "source_depth__percentage",
        "source_depth__depth",
        "source_depth__notional",
    ]);
    names
}

/// Reusable, allocation-free-per-row CSV field writer.
///
/// The buffer is owned here and cleared per row, so steady-state emission does
/// no allocator work: a null field costs one separator, a populated field is
/// formatted in place.
pub struct FieldWriter {
    buf: String,
    fields: usize,
    expected: usize,
}

impl FieldWriter {
    pub fn new(expected_fields: usize) -> Self {
        Self {
            // One merged row is ~2 KB; reserving it once avoids regrowth.
            buf: String::with_capacity(4096),
            fields: 0,
            expected: expected_fields,
        }
    }

    /// Start a new row and write the header line.
    pub fn begin_header(&mut self) {
        self.buf.clear();
        self.fields = 0;
        for name in schema() {
            self.push_str(name);
        }
        self.newline();
    }

    /// Begin a row, writing the common leading fields.
    ///
    /// `timestamp` is epoch **nanoseconds**, matching the historical column.
    pub fn begin_row(&mut self, timestamp_ns: i64, event_type: &str, sequence: u64) {
        self.buf.clear();
        self.fields = 0;
        self.push_i64(timestamp_ns);
        self.push_str(event_type);
        self.push_u64(sequence);
    }

    /// Emit an explicitly empty field.
    pub fn push_null(&mut self) {
        self.fields += 1;
        if self.fields > 1 {
            self.buf.push(',');
        }
    }

    pub fn push_opt_i64(&mut self, value: Option<i64>) {
        match value {
            Some(v) => self.push_i64(v),
            None => self.push_null(),
        }
    }

    pub fn push_opt_u64(&mut self, value: Option<u64>) {
        match value {
            Some(v) => self.push_u64(v),
            None => self.push_null(),
        }
    }

    pub fn push_opt_f64(&mut self, value: Option<f64>) {
        match value {
            // Non-finite would serialize as `NaN`/`inf`, which is not valid
            // CSV for the downstream reader. Treat it as missing.
            Some(v) if v.is_finite() => self.push_f64(v),
            _ => self.push_null(),
        }
    }

    pub fn push_opt_bool(&mut self, value: Option<bool>) {
        match value {
            Some(v) => self.push_str(if v { "True" } else { "False" }),
            None => self.push_null(),
        }
    }

    /// Number of fields written to the row currently in the buffer.
    pub fn field_count(&self) -> usize {
        self.fields
    }

    /// Take the line currently buffered (header or row) and clear the buffer.
    ///
    /// Lets a caller assemble several encoded lines before writing them out,
    /// which keeps the syscall count low without exposing the buffer.
    pub fn take_row(&mut self) -> String {
        self.fields = 0;
        std::mem::take(&mut self.buf)
    }

    /// Finish the row and return the encoded line (newline included).
    ///
    /// Returns `None` when the field count does not match the schema, which
    /// would silently shift every downstream column.
    pub fn finish_row(&mut self) -> Option<&str> {
        if self.fields != self.expected {
            return None;
        }
        self.newline();
        Some(&self.buf)
    }

    fn push_i64(&mut self, value: i64) {
        self.fields += 1;
        if self.fields > 1 {
            self.buf.push(',');
        }
        let _ = write!(self.buf, "{value}");
    }

    fn push_u64(&mut self, value: u64) {
        self.fields += 1;
        if self.fields > 1 {
            self.buf.push(',');
        }
        let _ = write!(self.buf, "{value}");
    }

    fn push_f64(&mut self, value: f64) {
        self.fields += 1;
        if self.fields > 1 {
            self.buf.push(',');
        }
        let _ = write!(self.buf, "{value}");
    }

    fn push_str(&mut self, value: &str) {
        self.fields += 1;
        if self.fields > 1 {
            self.buf.push(',');
        }
        self.buf.push_str(value);
    }

    fn newline(&mut self) {
        self.buf.push('\n');
        self.fields = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_128_unique_columns_in_order() {
        let names = schema();
        assert_eq!(names.len(), FIELD_COUNT);
        assert_eq!(names[0], "timestamp");
        assert_eq!(names[1], "event_type");
        assert_eq!(names[2], "event_sequence");
        assert_eq!(names[3], "lob__event_time");
        assert_eq!(names[4], "lob__final_update_id");
        assert_eq!(names[5], "book_state__best_bid_price");
        assert_eq!(names[9], "book_state__bid_price_2");
        // 5 common + 4 best-of-book + 19 levels x 4 == 85.
        assert_eq!(names[85], "agg__agg_trade_id");
        assert_eq!(names[93], "depth__timestamp");
        assert_eq!(names[97], "depth_state__-5");
        assert_eq!(names[107], "depth_state__notional_-5");
        assert_eq!(names[117], "source_agg__agg_trade_id");
        assert_eq!(names[127], "source_depth__notional");
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(unique.len(), FIELD_COUNT);
    }

    #[test]
    fn writer_emits_one_field_per_column() {
        let mut w = FieldWriter::new(FIELD_COUNT);
        w.begin_row(1_751_245_200_083_000_000, "LOB", 1);
        while w.field_count() < FIELD_COUNT {
            w.push_null();
        }
        let line = w.finish_row().expect("field count matches schema");
        assert_eq!(line.matches(',').count(), FIELD_COUNT - 1);
    }

    #[test]
    fn wrong_field_count_is_rejected() {
        let mut w = FieldWriter::new(FIELD_COUNT);
        w.begin_row(0, "LOB", 0);
        w.push_null();
        assert!(w.finish_row().is_none(), "short row must not be emitted");
    }

    #[test]
    fn non_finite_numbers_become_empty_fields() {
        // begin_row emits 3 fields (timestamp, event_type, sequence).
        let mut w = FieldWriter::new(6);
        w.begin_row(0, "LOB", 0);
        w.push_opt_f64(Some(f64::NAN));
        w.push_opt_f64(Some(1.5));
        w.push_opt_f64(Some(f64::INFINITY));
        let line = w.finish_row().expect("field count matches");
        assert_eq!(line, "0,LOB,0,,1.5,\n");
    }

    #[test]
    fn nulls_are_empty_not_zero() {
        // begin_row emits 3 fields; 3 more nulls makes 6.
        let mut w = FieldWriter::new(6);
        w.begin_row(0, "TRADE", 0);
        w.push_null();
        w.push_null();
        w.push_null();
        let line = w.finish_row().expect("ok");
        // The guard that keeps a fabricated 0.0 out of absent liquidity.
        assert_eq!(line, "0,TRADE,0,,,\n");
    }

    #[test]
    fn header_matches_schema() {
        let mut w = FieldWriter::new(FIELD_COUNT);
        w.begin_header();
        assert_eq!(w.field_count(), 0, "header resets the row counter");
    }
}
