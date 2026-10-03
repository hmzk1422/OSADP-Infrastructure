//! `columnar.rs` — columnar replay + emission.
//!
//! `parquet.rs` converts a file into `Vec<LobRow>` up front and only ever
//! writes full-book checkpoint *images* (`ingest_parquet`, `process_matched`).
//! Neither mode emits the thing `live.py::fetch_source_data` actually joins
//! trades against: one top-N-per-side book-state row per `event_time` batch.
//! This module is that missing piece, and it stays columnar the whole way:
//! the file is read as flat `Vec<i64>`/`Vec<f64>`/`Vec<bool>` columns (SoA),
//! and only a small per-batch slice (bounded by that event's row count, not
//! the file's) is ever turned into `LobRow`s to drive `ReplayEngine`.
//!
//! Two invocations (wired in `main.rs` as `columnar --snapshot-policy ...`):
//!
//!   * `--snapshot-policy day_start`      — one full-book image per UTC day
//!     (hour 00), written wide-schema per day: `<dir>/<YYYY-MM-DD>.parquet`.
//!     Same file layout `live.py::load_day_snapshot` already reads/writes,
//!     so a Python-seeded day and a Rust-seeded day are interchangeable.
//!   * `--snapshot-policy every_snapshot` — one full-book image per VALID
//!     EVENT (literally per row, not just exchange snapshot frames) —
//!     storage-heavy, meant to run only when building the `matched` dataset,
//!     not for routine day-start ingestion.
//!
//! Both invocations always emit the top-N book-state file; the policy only
//! controls which checkpoint images additionally get written to disk.

use crate::book::{price_to_tick, tick_to_price, PriceTick};
use crate::replay::{
    day_index, BookCheckpoint, CheckpointPolicy, EventType, LobRow, ReplayEngine, ReplayError,
    ReplayStats,
};

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet_rs::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet_rs::arrow::ArrowWriter;
use parquet_rs::file::properties::WriterProperties;

type BoxError = Box<dyn std::error::Error>;

/// Sentinel for "no id" in the columnar id columns (mirrors `live.py::NULL_ID`
/// and `parquet.rs::id_or_none`'s wire convention).
pub const NULL_ID: i64 = -1;

// ===========================================================
// COLUMNAR INPUT
// ===========================================================

/// Flat columnar view of the LOB depth-diff stream — the same 8 columns
/// `live.py::_LOB_REPLAY_SELECT` sends across the boundary — held as
/// contiguous `Vec`s instead of `Vec<LobRow>`.
pub struct LobColumns {
    pub event_time: Vec<i64>,
    pub is_bid: Vec<bool>,
    pub is_snapshot: Vec<bool>,
    pub price: Vec<f64>,
    pub quantity: Vec<f64>,
    pub first_update_id: Vec<i64>,
    pub final_update_id: Vec<i64>,
    pub prev_final_update_id: Vec<i64>,
}

impl LobColumns {
    pub fn len(&self) -> usize {
        self.event_time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.event_time.is_empty()
    }
}

pub fn read_lob_columns(path: &Path) -> Result<LobColumns, BoxError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut out = LobColumns {
        event_time: Vec::new(),
        is_bid: Vec::new(),
        is_snapshot: Vec::new(),
        price: Vec::new(),
        quantity: Vec::new(),
        first_update_id: Vec::new(),
        final_update_id: Vec::new(),
        prev_final_update_id: Vec::new(),
    };

    for batch in reader {
        append_columns(&batch?, &mut out)?;
    }

    Ok(out)
}

fn append_columns(batch: &RecordBatch, out: &mut LobColumns) -> Result<(), BoxError> {
    let event_time = get_i64(batch, "event_time")?;
    let is_bid = get_bool(batch, "is_bid")?;
    let is_snapshot = get_bool(batch, "is_snapshot")?;
    let price = get_f64(batch, "price")?;
    let quantity = get_f64(batch, "quantity")?;
    let first_update_id = get_i64(batch, "first_update_id")?;
    let final_update_id = get_i64(batch, "final_update_id")?;
    let prev_final_update_id = get_i64(batch, "prev_final_update_id")?;

    let n = batch.num_rows();
    for v in [
        &mut out.event_time.capacity(),
    ] {
        let _ = v; // no-op: silence unused warning if reserve calls are trimmed below
    }
    out.event_time.reserve(n);
    out.is_bid.reserve(n);
    out.is_snapshot.reserve(n);
    out.price.reserve(n);
    out.quantity.reserve(n);
    out.first_update_id.reserve(n);
    out.final_update_id.reserve(n);
    out.prev_final_update_id.reserve(n);

    for i in 0..n {
        out.event_time.push(event_time.value(i));
        out.is_bid.push(is_bid.value(i));
        out.is_snapshot.push(is_snapshot.value(i));
        out.price.push(price.value(i));
        out.quantity.push(quantity.value(i));
        out.first_update_id.push(id_or_sentinel(first_update_id, i));
        out.final_update_id.push(id_or_sentinel(final_update_id, i));
        out.prev_final_update_id
            .push(id_or_sentinel(prev_final_update_id, i));
    }

    Ok(())
}

fn id_or_sentinel(col: &Int64Array, i: usize) -> i64 {
    if col.is_null(i) {
        return NULL_ID;
    }
    let v = col.value(i);
    if v < 0 {
        NULL_ID
    } else {
        v
    }
}

fn opt_id(v: i64) -> Option<u64> {
    if v < 0 {
        None
    } else {
        Some(v as u64)
    }
}

fn get_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, BoxError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column `{name}`"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("column `{name}` is not int64").into())
}

fn get_f64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array, BoxError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column `{name}`"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("column `{name}` is not float64").into())
}

fn get_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array, BoxError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column `{name}`"))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| format!("column `{name}` is not uint64").into())
}

fn get_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, BoxError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column `{name}`"))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| format!("column `{name}` is not bool").into())
}

/// Start index of every maximal `event_time` run (mirrors `live.py`'s
/// `_batch_bounds`).
fn batch_starts(event_time: &[i64]) -> Vec<usize> {
    let mut starts = Vec::new();
    if event_time.is_empty() {
        return starts;
    }
    starts.push(0);
    for i in 1..event_time.len() {
        if event_time[i] != event_time[i - 1] {
            starts.push(i);
        }
    }
    starts
}

// ===========================================================
// COLUMNAR REPLAY
// ===========================================================

/// Top-N-per-side book state, one row per published batch, flattened
/// `[bid_px, bid_qty, ask_px, ask_qty]` per level (NaN = untouched level) —
/// the layout `live.py::_book_frame` used to build in Python.
pub struct BookStateRows {
    pub event_time: Vec<i64>,
    pub levels: Vec<f64>, // len == event_time.len() * 4 * lob_levels
    pub lob_levels: usize,
}

pub struct ColumnarOutput {
    pub rows: BookStateRows,
    pub checkpoints: Vec<BookCheckpoint>,
    pub stats: ReplayStats,
}

/// Replay `cols` end to end. `skip_until`: drop rows at/before this
/// `event_time` (a resumed day-start seed already covers them — re-feeding
/// them would look like an update-ID gap).
pub fn replay_columnar(
    engine: &mut ReplayEngine,
    cols: &LobColumns,
    lob_levels: usize,
    skip_until: i64,
) -> Result<ColumnarOutput, ReplayError> {
    let starts = batch_starts(&cols.event_time);
    let n = cols.event_time.len();
    let stride = 4 * lob_levels;

    let mut out_times = Vec::new();
    let mut out_levels = Vec::new();
    let mut checkpoints = Vec::new();

    for (bi, &start) in starts.iter().enumerate() {
        let end = starts.get(bi + 1).copied().unwrap_or(n);
        let event_time = cols.event_time[start];

        if event_time <= skip_until {
            continue;
        }

        engine.stats.batches += 1;

        let rows: Vec<LobRow> = (start..end)
            .map(|i| LobRow {
                event_time: cols.event_time[i],
                event_type: if cols.is_snapshot[i] {
                    EventType::Snapshot
                } else {
                    EventType::Update
                },
                is_bid: cols.is_bid[i],
                price: cols.price[i],
                quantity: cols.quantity[i],
                first_update_id: opt_id(cols.first_update_id[i]),
                final_update_id: opt_id(cols.final_update_id[i]),
                prev_final_update_id: opt_id(cols.prev_final_update_id[i]),
            })
            .collect();

        match engine.process_event(&rows) {
            Ok(()) => {}
            Err(ReplayError::PartialSnapshot) => {
                // `process_snapshot` already invalidated; count it (Rust's
                // own path doesn't, live.py's mirror does — keep parity).
                engine.stats.malformed_events += 1;
            }
            Err(ReplayError::InconsistentBatchIds) => {
                // Already counted + invalidated inside `process_event`.
            }
            Err(ReplayError::NonMonotonicEventTime) => {
                engine.stats.malformed_events += 1;
            }
            Err(other) => return Err(other),
        }

        // Checkpoint attempt happens every batch under either policy; the
        // engine's own `checkpoint()` applies the DayStart hour/day gate
        // internally, and applies no gate at all under EverySnapshot —
        // "full reconstruction snapshots per row" is literal.
        if let Some(ckpt) = engine.checkpoint(event_time) {
            checkpoints.push(ckpt);
        }

        if !engine.is_valid() {
            engine.stats.skipped_state_rows += 1;
            continue;
        }

        let (bids, asks) = engine.book().top_n(lob_levels);

        if bids.is_empty() || asks.is_empty() {
            engine.stats.skipped_state_rows += 1;
            continue;
        }

        if bids[0].0 >= asks[0].0 {
            engine.stats.crossed_books += 1;
            engine.stats.skipped_state_rows += 1;
            continue;
        }

        let mut row = vec![f64::NAN; stride];
        for (idx, &(tick, qty)) in bids.iter().enumerate() {
            row[4 * idx] = tick_to_price(tick);
            row[4 * idx + 1] = qty;
        }
        for (idx, &(tick, qty)) in asks.iter().enumerate() {
            row[4 * idx + 2] = tick_to_price(tick);
            row[4 * idx + 3] = qty;
        }

        out_times.push(event_time);
        out_levels.extend_from_slice(&row);
        engine.stats.emitted_rows += 1;
    }

    Ok(ColumnarOutput {
        rows: BookStateRows {
            event_time: out_times,
            levels: out_levels,
            lob_levels,
        },
        checkpoints,
        stats: engine.stats,
    })
}

// ===========================================================
// OUTPUT: top-N book-state parquet
// ===========================================================

fn book_state_level_names(lob_levels: usize) -> Vec<String> {
    let mut names = vec![
        "best_bid_price".to_string(),
        "best_bid_qty".to_string(),
        "best_ask_price".to_string(),
        "best_ask_qty".to_string(),
    ];
    for level in 2..=lob_levels {
        names.push(format!("bid_price_{level}"));
        names.push(format!("bid_qty_{level}"));
        names.push(format!("ask_price_{level}"));
        names.push(format!("ask_qty_{level}"));
    }
    names
}

fn write_book_state_parquet(path: &Path, rows: &BookStateRows) -> Result<(), BoxError> {
    let stride = 4 * rows.lob_levels;
    let names = book_state_level_names(rows.lob_levels);
    let n = rows.event_time.len();

    let mut fields = vec![Field::new("event_time", DataType::Int64, false)];
    for name in &names {
        fields.push(Field::new(name, DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));

    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(rows.event_time.clone()))];
    for c in 0..stride {
        let col: Vec<Option<f64>> = (0..n)
            .map(|r| {
                let v = rows.levels[r * stride + c];
                if v.is_nan() {
                    None
                } else {
                    Some(v)
                }
            })
            .collect();
        arrays.push(Arc::new(Float64Array::from(col)));
    }

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let tmp = path.with_extension("tmp.parquet");
    {
        let file = File::create(&tmp)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
    }
    std::fs::rename(&tmp, path)?;

    Ok(())
}

// ===========================================================
// OUTPUT: wide per-day checkpoint images
// ===========================================================
//
// Same schema `live.py::_write_day_snapshot` / `load_day_snapshot` already
// use: event_time, final_update_id, n_bid_levels, n_ask_levels, then
// bid_price_1..M / bid_qty_1..M / ask_price_1..M / ask_qty_1..M.
//
// DayStart: one checkpoint per day -> one row, M = that day's own level
// count (no cap). EverySnapshot: many checkpoints per day -> one row per
// event, M = `--snapshot-levels` (default 50), shorter books null-padded.

fn write_wide_snapshot_file(
    path: &Path,
    checkpoints: &[BookCheckpoint],
    cap_levels: Option<usize>,
) -> Result<(), BoxError> {
    let m = cap_levels.unwrap_or_else(|| {
        checkpoints
            .iter()
            .map(|c| c.bid_levels().max(c.ask_levels()))
            .max()
            .unwrap_or(0)
    });

    let mut fields = vec![
        Field::new("event_time", DataType::Int64, false),
        Field::new("final_update_id", DataType::UInt64, false),
        Field::new("n_bid_levels", DataType::UInt32, false),
        Field::new("n_ask_levels", DataType::UInt32, false),
    ];
    for level in 1..=m {
        fields.push(Field::new(format!("bid_price_{level}"), DataType::Float64, true));
        fields.push(Field::new(format!("bid_qty_{level}"), DataType::Float64, true));
    }
    for level in 1..=m {
        fields.push(Field::new(format!("ask_price_{level}"), DataType::Float64, true));
        fields.push(Field::new(format!("ask_qty_{level}"), DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));

    let event_time: Vec<i64> = checkpoints.iter().map(|c| c.event_time).collect();
    let final_update_id: Vec<u64> = checkpoints.iter().map(|c| c.final_update_id).collect();
    let n_bid: Vec<u32> = checkpoints.iter().map(|c| c.bid_levels() as u32).collect();
    let n_ask: Vec<u32> = checkpoints.iter().map(|c| c.ask_levels() as u32).collect();

    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(event_time)),
        Arc::new(UInt64Array::from(final_update_id)),
        Arc::new(UInt32Array::from(n_bid)),
        Arc::new(UInt32Array::from(n_ask)),
    ];

    for level in 1..=m {
        let px: Vec<Option<f64>> = checkpoints
            .iter()
            .map(|c| c.bid_prices.get(level - 1).copied())
            .collect();
        let qty: Vec<Option<f64>> = checkpoints
            .iter()
            .map(|c| c.bids.get(level - 1).map(|&(_, q)| q))
            .collect();
        arrays.push(Arc::new(Float64Array::from(px)));
        arrays.push(Arc::new(Float64Array::from(qty)));
    }
    for level in 1..=m {
        let px: Vec<Option<f64>> = checkpoints
            .iter()
            .map(|c| c.ask_prices.get(level - 1).copied())
            .collect();
        let qty: Vec<Option<f64>> = checkpoints
            .iter()
            .map(|c| c.asks.get(level - 1).map(|&(_, q)| q))
            .collect();
        arrays.push(Arc::new(Float64Array::from(px)));
        arrays.push(Arc::new(Float64Array::from(qty)));
    }

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let tmp = path.with_extension("tmp.parquet");
    {
        let file = File::create(&tmp)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
    }
    std::fs::rename(&tmp, path)?;

    Ok(())
}

fn write_day_start_snapshots(checkpoints: &[BookCheckpoint], dir: &Path) -> Result<(), BoxError> {
    for ckpt in checkpoints {
        let path = dir.join(format!("{}.parquet", day_string(day_index(ckpt.event_time))));
        write_wide_snapshot_file(&path, std::slice::from_ref(ckpt), None)?;
    }
    Ok(())
}

fn write_every_snapshot_images(
    checkpoints: &[BookCheckpoint],
    dir: &Path,
    cap: usize,
) -> Result<(), BoxError> {
    use std::collections::BTreeMap;

    let mut by_day: BTreeMap<i64, Vec<BookCheckpoint>> = BTreeMap::new();
    for c in checkpoints {
        by_day
            .entry(day_index(c.event_time))
            .or_default()
            .push(c.clone());
    }

    for (day, group) in by_day {
        let path = dir.join(format!("{}.parquet", day_string(day)));
        write_wide_snapshot_file(&path, &group, Some(cap))?;
    }

    Ok(())
}

/// `YYYY-MM-DD` for a UTC day index (see `replay::day_index`). Duplicated
/// from `parquet.rs::civil_from_days` (Howard Hinnant's algorithm) so this
/// file has no cross-module dependency beyond `book`/`replay`.
fn day_string(day: i64) -> String {
    let z = day + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}-{m:02}-{d:02}")
}

// ===========================================================
// SEED (resume from a stored day-start image)
// ===========================================================

fn read_seed_snapshot(
    path: &Path,
) -> Result<(i64, u64, Vec<(PriceTick, f64)>, Vec<(PriceTick, f64)>), BoxError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let mut reader = builder.build()?;
    let batch = reader
        .next()
        .ok_or("seed snapshot file is empty")??;

    let event_time = get_i64(&batch, "event_time")?.value(0);
    let final_update_id = get_u64(&batch, "final_update_id")?.value(0);

    let mut bid_levels: Vec<usize> = schema
        .fields()
        .iter()
        .filter_map(|f| f.name().strip_prefix("bid_price_").and_then(|s| s.parse().ok()))
        .collect();
    bid_levels.sort_unstable();

    let mut ask_levels: Vec<usize> = schema
        .fields()
        .iter()
        .filter_map(|f| f.name().strip_prefix("ask_price_").and_then(|s| s.parse().ok()))
        .collect();
    ask_levels.sort_unstable();

    let mut bids = Vec::new();
    for lvl in &bid_levels {
        let px = get_f64(&batch, &format!("bid_price_{lvl}"))?;
        if px.is_null(0) {
            continue;
        }
        let qty = get_f64(&batch, &format!("bid_qty_{lvl}"))?;
        let tick = price_to_tick(px.value(0)).ok_or("invalid seed bid price")?;
        bids.push((tick, qty.value(0)));
    }

    let mut asks = Vec::new();
    for lvl in &ask_levels {
        let px = get_f64(&batch, &format!("ask_price_{lvl}"))?;
        if px.is_null(0) {
            continue;
        }
        let qty = get_f64(&batch, &format!("ask_qty_{lvl}"))?;
        let tick = price_to_tick(px.value(0)).ok_or("invalid seed ask price")?;
        asks.push((tick, qty.value(0)));
    }

    Ok((event_time, final_update_id, bids, asks))
}

// ===========================================================
// CLI
// ===========================================================

struct ColumnarArgs {
    input: PathBuf,
    output: PathBuf,
    tick_size: f64,
    warm_start: bool,
    levels: usize,
    policy: CheckpointPolicy,
    snapshot_dir: Option<PathBuf>,
    snapshot_levels: usize,
    seed_snapshot: Option<PathBuf>,
    skip_until: i64,
}

fn next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, BoxError> {
    let value = args
        .get(*i + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?;
    *i += 2;
    Ok(value.as_str())
}

fn parse_args(args: &[String]) -> Result<ColumnarArgs, BoxError> {
    let mut input = None;
    let mut output = None;
    let mut tick_size = None;
    let mut warm_start = false;
    let mut levels = 20usize;
    let mut policy = None;
    let mut snapshot_dir = None;
    let mut snapshot_levels = 50usize;
    let mut seed_snapshot = None;
    let mut skip_until = 0i64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => input = Some(PathBuf::from(next_value(args, &mut i, "--input")?)),
            "--output" => output = Some(PathBuf::from(next_value(args, &mut i, "--output")?)),
            "--tick-size" => {
                tick_size = Some(next_value(args, &mut i, "--tick-size")?.parse::<f64>()?)
            }
            "--warm-start" => {
                warm_start = true;
                i += 1;
            }
            "--levels" => levels = next_value(args, &mut i, "--levels")?.parse()?,
            "--snapshot-policy" => {
                let name = next_value(args, &mut i, "--snapshot-policy")?;
                policy = Some(
                    CheckpointPolicy::parse(name)
                        .ok_or_else(|| format!("unknown snapshot policy `{name}`"))?,
                );
            }
            "--snapshot-dir" => {
                snapshot_dir = Some(PathBuf::from(next_value(args, &mut i, "--snapshot-dir")?))
            }
            "--snapshot-levels" => {
                snapshot_levels = next_value(args, &mut i, "--snapshot-levels")?.parse()?
            }
            "--seed-snapshot" => {
                seed_snapshot = Some(PathBuf::from(next_value(args, &mut i, "--seed-snapshot")?))
            }
            "--skip-until" => skip_until = next_value(args, &mut i, "--skip-until")?.parse()?,
            other => return Err(format!("unknown flag `{other}`").into()),
        }
    }

    Ok(ColumnarArgs {
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
        tick_size: tick_size.ok_or("--tick-size is required")?,
        warm_start,
        levels,
        policy: policy.ok_or("--snapshot-policy is required (day_start | every_snapshot)")?,
        snapshot_dir,
        snapshot_levels,
        seed_snapshot,
        skip_until,
    })
}

fn print_stats(stats: &ReplayStats) {
    println!(
        "STATS snapshot_events={} update_events={} gaps={} malformed_events={} \
crossed_books={} rows_before_initial_snapshot={} snapshots={} skipped_checkpoints={} \
batches={} emitted_rows={} skipped_state_rows={} warm_start_batches={}",
        stats.snapshot_events,
        stats.update_events,
        stats.gaps,
        stats.malformed_events,
        stats.crossed_books,
        stats.rows_before_initial_snapshot,
        stats.snapshots,
        stats.skipped_checkpoints,
        stats.batches,
        stats.emitted_rows,
        stats.skipped_state_rows,
        stats.warm_start_batches,
    );
}

/// `lob_replay columnar --input <FILE> --output <FILE> --tick-size <F>
/// --levels <N> --snapshot-policy <day_start|every_snapshot>
/// [--snapshot-dir <DIR>] [--snapshot-levels <N>] [--warm-start]
/// [--seed-snapshot <FILE>] [--skip-until <MS>]`
pub fn run(args: &[String]) -> Result<(), BoxError> {
    let parsed = parse_args(args)?;

    let mut engine = ReplayEngine::with_options(parsed.tick_size, parsed.warm_start)?;
    engine.set_checkpoint_policy(parsed.policy);

    let mut skip_until = parsed.skip_until;

    if let Some(seed_path) = &parsed.seed_snapshot {
        let (event_time, final_update_id, bids, asks) = read_seed_snapshot(seed_path)?;
        engine.seed_snapshot(event_time, final_update_id, &bids, &asks)?;
        skip_until = skip_until.max(event_time);
    }

    let cols = read_lob_columns(&parsed.input)?;
    let output = replay_columnar(&mut engine, &cols, parsed.levels, skip_until)?;

    write_book_state_parquet(&parsed.output, &output.rows)?;

    if let Some(dir) = &parsed.snapshot_dir {
        std::fs::create_dir_all(dir)?;
        match parsed.policy {
            CheckpointPolicy::DayStart => write_day_start_snapshots(&output.checkpoints, dir)?,
            CheckpointPolicy::EverySnapshot => {
                write_every_snapshot_images(&output.checkpoints, dir, parsed.snapshot_levels)?
            }
        }
    }

    print_stats(&output.stats);

    Ok(())
}