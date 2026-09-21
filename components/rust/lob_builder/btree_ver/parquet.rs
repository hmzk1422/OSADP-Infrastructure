//! `parquet.rs` — when to save the book.
//!
//! Both workflows below share exactly the same `LobBook` (via `book.rs`) and
//! `ReplayEngine` (via `replay.rs`). The only thing that differs between them
//! is *which* [`CheckpointPolicy`] the engine is run under, and what this
//! module does with the resulting checkpoints:
//!
//! ```text
//! ingest_parquet:
//!     snapshot = first valid event during UTC 00:00 hour
//!     -> one file per UTC day: <output_dir>/<YYYY-MM-DD>.parquet
//!
//! process_matched:
//!     snapshot = every valid event
//!     -> one file, one full-book image per event: <output>
//! ```
//!
//! Neither function decides *whether* a given event is trustworthy — that is
//! `ReplayEngine`'s job. This module only reads rows in, replays them, and
//! decides when a resulting checkpoint gets written to disk.

use crate::replay::{
    BookCheckpoint, CheckpointPolicy, EventType, LobRow, ReplayEngine, ReplayError, ReplayStats,
};

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
// Aliased to `parquet_rs` in Cargo.toml — see the note there. This is the
// arrow-rs `parquet` crate, unrelated to this file's own module name.
use parquet_rs::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet_rs::arrow::ArrowWriter;
use parquet_rs::file::properties::WriterProperties;

type BoxError = Box<dyn std::error::Error>;

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Expected input schema (any extra columns are ignored):
///
/// | column                  | type    | nullable |
/// |--------------------------|--------|----------|
/// | event_time                | int64  | no       |
/// | is_bid                     | bool   | no       |
/// | price                       | float64| no       |
/// | quantity                   | float64| no       |
/// | first_update_id             | int64  | yes      |
/// | final_update_id             | int64  | yes      |
/// | prev_final_update_id        | int64  | yes      |
/// | is_snapshot                 | bool   | no       |
fn read_lob_rows(path: &Path) -> Result<Vec<LobRow>, BoxError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut rows = Vec::new();

    for batch in reader {
        let batch = batch?;
        append_rows(&batch, &mut rows)?;
    }

    Ok(rows)
}

fn append_rows(batch: &RecordBatch, rows: &mut Vec<LobRow>) -> Result<(), BoxError> {
    let event_time = get_i64(batch, "event_time")?;
    let is_bid = get_bool(batch, "is_bid")?;
    let price = get_f64(batch, "price")?;
    let quantity = get_f64(batch, "quantity")?;
    let first_update_id = get_i64(batch, "first_update_id")?;
    let final_update_id = get_i64(batch, "final_update_id")?;
    let prev_final_update_id = get_i64(batch, "prev_final_update_id")?;
    let is_snapshot = get_bool(batch, "is_snapshot")?;

    for i in 0..batch.num_rows() {
        let event_type = if is_snapshot.value(i) {
            EventType::Snapshot
        } else {
            EventType::Update
        };

        rows.push(LobRow {
            event_time: event_time.value(i),
            event_type,
            is_bid: is_bid.value(i),
            price: price.value(i),
            quantity: quantity.value(i),
            first_update_id: id_or_none(first_update_id, i),
            final_update_id: id_or_none(final_update_id, i),
            prev_final_update_id: id_or_none(prev_final_update_id, i),
        });
    }

    Ok(())
}

/// A null slot, or the `-1` sentinel some sources use on the wire, both mean
/// "no id".
fn id_or_none(col: &Int64Array, i: usize) -> Option<u64> {
    if col.is_null(i) {
        return None;
    }

    let v = col.value(i);

    if v < 0 {
        None
    } else {
        Some(v as u64)
    }
}

fn get_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, BoxError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column `{name}`"))?;

    col.as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("column `{name}` is not int64").into())
}

fn get_f64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array, BoxError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column `{name}`"))?;

    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("column `{name}` is not float64").into())
}

fn get_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, BoxError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column `{name}`"))?;

    col.as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| format!("column `{name}` is not bool").into())
}

// ---------------------------------------------------------------------------
// Shared replay loop
// ---------------------------------------------------------------------------

/// Group `rows` by `event_time` (rows must already be sorted by it), replay
/// each batch through `engine`, and collect every checkpoint the engine's
/// current [`CheckpointPolicy`] materializes.
///
/// Recoverable per-batch errors (a malformed frame) are skipped so replay
/// continues at the next event; everything else propagates to the caller.
fn replay_and_checkpoint(
    rows: &[LobRow],
    engine: &mut ReplayEngine,
) -> Result<Vec<BookCheckpoint>, ReplayError> {
    let mut checkpoints = Vec::new();

    let n = rows.len();
    let mut i = 0usize;

    while i < n {
        let event_time = rows[i].event_time;

        let mut j = i + 1;
        while j < n && rows[j].event_time == event_time {
            j += 1;
        }

        let batch = &rows[i..j];

        if let Err(err) = engine.process_event(batch) {
            match err {
                ReplayError::PartialSnapshot
                | ReplayError::InconsistentBatchIds
                | ReplayError::NonMonotonicEventTime => {
                    // The engine already invalidated its state; this batch
                    // emits nothing, and the next trusted event reseeds or
                    // rebases it. Everything else is a schema-level bug and
                    // must reach the caller.
                    i = j;
                    continue;
                }
                other => return Err(other),
            }
        }

        if let Some(ckpt) = engine.checkpoint(event_time) {
            checkpoints.push(ckpt);
        }

        i = j;
    }

    Ok(checkpoints)
}

// ---------------------------------------------------------------------------
// Mode 1: ingest-parquet
// ---------------------------------------------------------------------------

/// Read `input`, replay it start to finish, and write exactly one full-book
/// snapshot per UTC day — taken at that day's first valid hour-00 event — to
/// `<output_dir>/<YYYY-MM-DD>.parquet`.
pub fn ingest_parquet(
    input: &Path,
    output_dir: &Path,
    tick_size: f64,
    warm_start: bool,
) -> Result<ReplayStats, BoxError> {
    let mut rows = read_lob_rows(input)?;
    rows.sort_by_key(|r| r.event_time);

    let mut engine = ReplayEngine::with_options(tick_size, warm_start)?;
    engine.set_checkpoint_policy(CheckpointPolicy::DayStart);

    let checkpoints = replay_and_checkpoint(&rows, &mut engine)?;

    std::fs::create_dir_all(output_dir)?;

    for ckpt in &checkpoints {
        let path = output_dir.join(day_filename(ckpt.event_time));
        write_checkpoints_parquet(&path, std::slice::from_ref(ckpt))?;
    }

    Ok(engine.stats)
}

// ---------------------------------------------------------------------------
// Mode 2: matched
// ---------------------------------------------------------------------------

/// Read `input`, replay it start to finish, and write a full-book snapshot
/// for EVERY valid event into a single file at `output`.
pub fn process_matched(
    input: &Path,
    output: &Path,
    tick_size: f64,
    warm_start: bool,
) -> Result<ReplayStats, BoxError> {
    let mut rows = read_lob_rows(input)?;
    rows.sort_by_key(|r| r.event_time);

    let mut engine = ReplayEngine::with_options(tick_size, warm_start)?;
    engine.set_checkpoint_policy(CheckpointPolicy::EverySnapshot);

    let checkpoints = replay_and_checkpoint(&rows, &mut engine)?;

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    write_checkpoints_parquet(output, &checkpoints)?;

    Ok(engine.stats)
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write full-book checkpoints as a level-per-row parquet table:
/// `event_time, final_update_id, is_bid, price, quantity`. A checkpoint's
/// full image is reconstructed by grouping rows on `(event_time, is_bid)`
/// and sorting by `price` (bids descending, asks ascending).
fn write_checkpoints_parquet(path: &Path, checkpoints: &[BookCheckpoint]) -> Result<(), BoxError> {
    let mut event_time = Vec::new();
    let mut final_update_id = Vec::new();
    let mut is_bid = Vec::new();
    let mut price = Vec::new();
    let mut quantity = Vec::new();

    for ckpt in checkpoints {
        for (idx, &(_, qty)) in ckpt.bids.iter().enumerate() {
            event_time.push(ckpt.event_time);
            final_update_id.push(ckpt.final_update_id);
            is_bid.push(true);
            price.push(ckpt.bid_prices[idx]);
            quantity.push(qty);
        }

        for (idx, &(_, qty)) in ckpt.asks.iter().enumerate() {
            event_time.push(ckpt.event_time);
            final_update_id.push(ckpt.final_update_id);
            is_bid.push(false);
            price.push(ckpt.ask_prices[idx]);
            quantity.push(qty);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("event_time", DataType::Int64, false),
        Field::new("final_update_id", DataType::UInt64, false),
        Field::new("is_bid", DataType::Boolean, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(event_time)) as ArrayRef,
            Arc::new(UInt64Array::from(final_update_id)) as ArrayRef,
            Arc::new(BooleanArray::from(is_bid)) as ArrayRef,
            Arc::new(Float64Array::from(price)) as ArrayRef,
            Arc::new(Float64Array::from(quantity)) as ArrayRef,
        ],
    )?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// `<YYYY-MM-DD>.parquet` for the UTC day containing `event_time_ms`.
fn day_filename(event_time_ms: i64) -> String {
    let day = crate::replay::day_index(event_time_ms);
    let (y, m, d) = civil_from_days(day);
    format!("{y:04}-{m:02}-{d:02}.parquet")
}

/// Proleptic-Gregorian (year, month, day) from a day count since
/// 1970-01-01, per Howard Hinnant's `civil_from_days` algorithm. Kept
/// dependency-free rather than pulling in a date/time crate for one
/// filename.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };

    (year, m, d)
}