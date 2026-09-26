//! `csv.rs` — the live canonical-row CSV sink.
//!
//! The live capture is written as CSV because that is the interchange format
//! the feature layer and the parity harness already speak, and because a row
//! here is *sparse*: a TRADE row has no `lob__*` and a LOB row has no
//! `agg__*`, so most fields are empty.
//!
//! # Byte-for-byte schema compatibility
//!
//! The header is [`crate::row::schema`] and the field order is produced by
//! [`Merger::write_row`], the same emitter the historical pipeline uses. That
//! is the point: a live capture and a historical merge are readable by one
//! feature spec with no column remapping. Every row is checked against the
//! 128-column count before it is written, so a column can never silently
//! shift.
//!
//! # Missing vs zero
//!
//! Absent values are written as **empty fields**, never `0.0`. There is no
//! live source for `depth__*` / `depth_state__*` (Binance's `bookDepth` is a
//! downloaded dataset, not a stream), and a fabricated zero there would read
//! downstream as "no liquidity" rather than "not observed".
//!
//! # Durability
//!
//! Writes are line-buffered and flushed on [`CsvRecorder::flush`], on `Drop`,
//! and every [`CsvRecorder::flush_every`] rows, so a capture killed
//! mid-session still leaves a readable file rather than a zero-byte one.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::merger::{CanonicalRow, Merger};
use crate::row::{schema, FieldWriter, FIELD_COUNT};

/// Why a row could not be written.
#[derive(Debug)]
pub enum CsvError {
    Io(std::io::Error),
    /// The row did not carry exactly one field per column. Writing it anyway
    /// would shift every downstream column, so it is refused instead.
    FieldCountMismatch {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for CsvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "csv io error: {e}"),
            Self::FieldCountMismatch { expected, got } => write!(
                f,
                "row has {got} fields but the schema has {expected}; \
                 refusing to write a row that would shift columns"
            ),
        }
    }
}

impl std::error::Error for CsvError {}

impl From<std::io::Error> for CsvError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// True when the file exists and holds at least one byte.
fn file_is_non_empty(path: &Path) -> Result<bool, CsvError> {
    Ok(std::fs::metadata(path)
        .map(|m| m.len() > 0)
        .unwrap_or(false))
}

/// Append-only CSV writer for the canonical 128-column row.
///
/// Owns the [`FieldWriter`], so steady-state encoding allocates nothing: the
/// buffer is reused and a null costs one separator.
pub struct CsvRecorder {
    out: BufWriter<File>,
    writer: FieldWriter,
    path: PathBuf,
    /// Guards against emitting a second header when reopening for append.
    header_written: bool,
    rows_written: u64,
    rows_since_flush: u64,
    flush_every: u64,
    /// True when the file already existed and rows were appended, which is
    /// legitimate for a resumed capture but worth reporting.
    appended: bool,
}

impl CsvRecorder {
    /// Create a capture file, truncating any previous one.
    ///
    /// The parent directory is created when missing, so a fresh
    /// `market_data/live_db` needs no manual `mkdir`.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, CsvError> {
        Self::open(path, false)
    }

    /// Open a capture for appending, keeping the existing header.
    pub fn append(path: impl AsRef<Path>) -> Result<Self, CsvError> {
        Self::open(path, true)
    }

    fn open(path: impl AsRef<Path>, append: bool) -> Result<Self, CsvError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let existed = path.exists();
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&path)?;
        // An appended file already carries a header, so suppress writing a
        // second one. A truncated (fresh) file never does.
        let header_written = append && file_is_non_empty(&path)?;
        Ok(Self {
            out: BufWriter::new(file),
            writer: FieldWriter::new(FIELD_COUNT),
            path,
            header_written,
            rows_written: 0,
            rows_since_flush: 0,
            flush_every: 256,
            appended: append && existed,
        })
    }

    /// Flush to the OS every `n` rows. Lower means less data at risk on a
    /// crash; higher means fewer syscalls.
    pub fn flush_every(mut self, n: u64) -> Self {
        self.flush_every = n.max(1);
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rows_written(&self) -> u64 {
        self.rows_written
    }

    /// True when this capture appends to a pre-existing file.
    pub fn is_appending(&self) -> bool {
        self.appended
    }

    /// Write the header when it has not been written yet.
    pub fn write_header(&mut self) -> Result<(), CsvError> {
        if self.header_written {
            return Ok(());
        }
        self.writer.begin_header();
        let line = self.writer.take_row();
        self.out.write_all(line.as_bytes())?;
        self.header_written = true;
        Ok(())
    }

    /// Encode one canonical row through the merger and append it.
    ///
    /// Returns the encoded line length in bytes. A row the merger could not
    /// encode at the full field width is refused rather than written, because
    /// a short row would shift every downstream column.
    pub fn write_row(
        &mut self,
        merger: &mut Merger,
        row: &CanonicalRow,
    ) -> Result<usize, CsvError> {
        self.write_header()?;
        // `Merger::write_csv` finishes the row itself (it calls
        // FieldWriter::finish_row, which appends the newline and resets the
        // field counter). So the line is taken afterwards, never re-finished,
        // and the field count can no longer be read here -- the None return
        // from write_csv is what reports a width mismatch.
        if merger.write_csv(row, &mut self.writer).is_none() {
            return Err(CsvError::FieldCountMismatch {
                expected: FIELD_COUNT,
                got: 0,
            });
        }
        let line = self.writer.take_row();
        self.out.write_all(line.as_bytes())?;
        self.rows_written += 1;
        self.rows_since_flush += 1;
        if self.rows_since_flush >= self.flush_every {
            self.out.flush()?;
            self.rows_since_flush = 0;
        }
        Ok(line.len())
    }

    /// Push buffered rows to the OS.
    pub fn flush(&mut self) -> Result<(), CsvError> {
        self.out.flush()?;
        self.rows_since_flush = 0;
        Ok(())
    }
}

impl Drop for CsvRecorder {
    fn drop(&mut self) {
        // Best effort: a recorder dropped during a panic should still leave
        // every completed row on disk.
        let _ = self.out.flush();
    }
}

/// The 128-column header line, newline included.
pub fn header_line() -> String {
    let mut line = String::with_capacity(4096);
    for name in schema() {
        line.push_str(name);
        line.push(',');
    }
    line.pop();
    line.push('\n');
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{BookSnapshot, BookUpdate, DiffFrame, MarketEvent, Trade};
    use crate::DEFAULT_LEVELS;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("eml_csv_test_{name}"));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn seed() -> MarketEvent {
        MarketEvent::Snapshot(BookSnapshot {
            last_update_id: 100,
            bids: vec![(99.0, 1.0), (98.0, 2.0)],
            asks: vec![(101.0, 1.5), (102.0, 2.5)],
        })
    }

    fn trade(ts: i64, id: u64) -> MarketEvent {
        MarketEvent::Trade(Trade {
            event_time_ms: ts,
            transact_time_ms: ts,
            agg_trade_id: id,
            first_trade_id: id,
            last_trade_id: id,
            price: 100.0,
            quantity: 0.5,
            is_buyer_maker: false,
        })
    }

    fn diff(ts: i64, first: u64, last: u64) -> MarketEvent {
        MarketEvent::Diff(DiffFrame {
            event_time_ms: ts,
            first_update_id: first,
            final_update_id: last,
            updates: vec![BookUpdate {
                is_bid: true,
                price: 99.5,
                quantity: 3.0,
            }],
        })
    }

    #[test]
    fn header_has_128_columns_in_schema_order() {
        let line = header_line();
        let names: Vec<&str> = line.trim_end().split(',').collect();
        assert_eq!(names.len(), FIELD_COUNT);
        assert_eq!(names[0], "timestamp");
        assert_eq!(names[1], "event_type");
        assert_eq!(names[2], "event_sequence");
        assert_eq!(names[85], "agg__agg_trade_id");
        assert_eq!(names[127], "source_depth__notional");
    }

    #[test]
    fn every_written_row_has_the_full_field_width() {
        let path = temp_path("width");
        let mut rec = CsvRecorder::create(&path).expect("create");
        let mut merger = Merger::new(DEFAULT_LEVELS);

        let (seed_row, _) = merger.merge_event(seed());
        rec.write_row(&mut merger, &seed_row).expect("seed row");
        let (lob_row, _) = merger.merge_event(diff(1000, 101, 105));
        rec.write_row(&mut merger, &lob_row).expect("lob row");
        let (trade_row, _) = merger.merge_event(trade(2000, 55));
        rec.write_row(&mut merger, &trade_row).expect("trade row");
        rec.flush().expect("flush");

        let text = std::fs::read_to_string(&path).expect("read back");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "header + 3 rows");
        for line in &lines {
            assert_eq!(
                line.matches(',').count(),
                FIELD_COUNT - 1,
                "row must carry {FIELD_COUNT} fields"
            );
        }
        assert_eq!(rec.rows_written(), 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn trade_row_leaves_lob_identity_empty_not_zero() {
        let path = temp_path("sparse");
        let mut rec = CsvRecorder::create(&path).unwrap();
        let mut merger = Merger::new(DEFAULT_LEVELS);
        merger.merge_event(seed());
        let (row, _) = merger.merge_event(trade(1000, 7));
        rec.write_row(&mut merger, &row).unwrap();
        rec.flush().unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let fields: Vec<&str> = text.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(fields[1], "TRADE");
        // A fabricated 0.0 here would read as "LOB update at time 0".
        assert_eq!(fields[3], "", "lob__event_time must be absent");
        assert_eq!(fields[4], "", "lob__final_update_id must be absent");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn depth_block_is_always_empty() {
        let path = temp_path("depth");
        let mut rec = CsvRecorder::create(&path).unwrap();
        let mut merger = Merger::new(DEFAULT_LEVELS);
        let (row, _) = merger.merge_event(seed());
        rec.write_row(&mut merger, &row).unwrap();
        rec.flush().unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let fields: Vec<&str> = text.lines().nth(1).unwrap().split(',').collect();
        // depth__timestamp .. depth_state__notional_5 == fields 93..=116.
        for (offset, value) in fields[93..=116].iter().enumerate() {
            assert_eq!(*value, "", "depth field {offset} must be absent");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn appending_keeps_exactly_one_header() {
        let path = temp_path("append");
        {
            let mut rec = CsvRecorder::create(&path).unwrap();
            let mut merger = Merger::new(DEFAULT_LEVELS);
            let (row, _) = merger.merge_event(seed());
            rec.write_row(&mut merger, &row).unwrap();
            rec.flush().unwrap();
        }
        {
            let mut rec = CsvRecorder::append(&path).unwrap();
            assert!(rec.is_appending(), "reopening an existing file appends");
            let mut merger = Merger::new(DEFAULT_LEVELS);
            let (row, _) = merger.merge_event(trade(1000, 9));
            rec.write_row(&mut merger, &row).unwrap();
            rec.flush().unwrap();
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "one header + two rows");
        assert!(lines[0].starts_with("timestamp,"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn creates_missing_parent_directories() {
        let root = std::env::temp_dir().join("eml_csv_nested_test");
        let _ = std::fs::remove_dir_all(&root);
        let path = root.join("inner").join("capture.csv");
        let mut rec = CsvRecorder::create(&path).expect("create nested");
        rec.write_header().expect("header");
        rec.flush().expect("flush");
        assert!(path.exists(), "parent directories are created on demand");
        let _ = std::fs::remove_dir_all(&root);
    }
}
