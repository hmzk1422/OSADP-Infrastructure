//! parquet.rs
//!
//! Raw input:
//!   received_time
//!   event_time
//!   transaction_time
//!   symbol
//!   event_type
//!   first_update_id
//!   final_update_id
//!   prev_final_update_id
//!   last_update_id
//!   side
//!   price
//!   quantity
//!
//! `is_bid` is derived from `side`.
//! Snapshot/update classification is derived from `event_type`.

use crate::book::tick_to_price;
use crate::replay::{
    BookCheckpoint,
    CheckpointPolicy,
    EventType,
    LobRow,
    ReplayEngine,
    ReplayError,
    ReplayStats,
};

use arrow::array::{
    Array,
    ArrayRef,
    BooleanArray,
    Float64Array,
    Int64Array,
    StringArray,
    UInt64Array,
};
use arrow::datatypes::{
    DataType,
    Field,
    Schema,
};
use arrow::record_batch::RecordBatch;

use parquet_rs::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet_rs::arrow::ArrowWriter;
use parquet_rs::file::properties::WriterProperties;

use std::fs::File;
use std::path::{
    Path,
    PathBuf,
};
use std::sync::Arc;

type BoxError = Box<dyn std::error::Error>;


// ============================================================================
// INPUT FILE DISCOVERY
// ============================================================================

fn discover_input_files(
    input: &Path,
) -> Result<Vec<PathBuf>, BoxError> {
    if input.is_file() {
        if input.extension()
            .and_then(|x| x.to_str())
            .map(|x| x.eq_ignore_ascii_case("parquet"))
            != Some(true)
        {
            return Err(
                format!(
                    "input file is not parquet: {}",
                    input.display()
                )
                .into()
            );
        }

        return Ok(vec![input.to_path_buf()]);
    }

    if !input.is_dir() {
        return Err(
            format!(
                "input path does not exist: {}",
                input.display()
            )
            .into()
        );
    }

    fn visit(
        dir: &Path,
        files: &mut Vec<PathBuf>,
    ) -> Result<(), BoxError> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();

            if path.is_dir() {
                visit(&path, files)?;
                continue;
            }

            if path.is_file()
                && path
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.eq_ignore_ascii_case("parquet"))
                    .unwrap_or(false)
            {
                files.push(path);
            }
        }

        Ok(())
    }

    let mut files = Vec::new();

    visit(input, &mut files)?;

    files.sort();

    if files.is_empty() {
        return Err(
            format!(
                "no parquet files found under {}",
                input.display()
            )
            .into()
        );
    }

    Ok(files)
}


// ============================================================================
// ARROW COLUMN HELPERS
// ============================================================================

fn get_i64<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Int64Array, BoxError> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| {
            format!("missing column `{name}`")
        })?;

    column
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            format!(
                "column `{name}` is not int64"
            )
            .into()
        })
}


fn get_string<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a StringArray, BoxError> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| {
            format!("missing column `{name}`")
        })?;

    column
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            format!(
                "column `{name}` is not string/utf8"
            )
            .into()
        })
}


fn optional_update_id(
    column: &Int64Array,
    row: usize,
) -> Option<u64> {
    if column.is_null(row) {
        return None;
    }

    let value = column.value(row);

    if value < 0 {
        return None;
    }

    Some(value as u64)
}


// ============================================================================
// RAW PARQUET -> LobRow
// ============================================================================

fn append_rows(
    batch: &RecordBatch,
    rows: &mut Vec<LobRow>,
) -> Result<(), BoxError> {
    let event_time =
        get_i64(batch, "event_time")?;

    let event_type =
        get_string(batch, "event_type")?;

    let side =
        get_string(batch, "side")?;

    let price =
        get_string(batch, "price")?;

    let quantity =
        get_string(batch, "quantity")?;

    let first_update_id =
        get_i64(batch, "first_update_id")?;

    let final_update_id =
        get_i64(batch, "final_update_id")?;

    let prev_final_update_id =
        get_i64(batch, "prev_final_update_id")?;

    for row in 0..batch.num_rows() {
        let event_type =
            match event_type
                .value(row)
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "snapshot" => EventType::Snapshot,

                "update" => EventType::Update,

                value if value.contains("snapshot") =>
                    EventType::Snapshot,

                value
                    if value.contains("update")
                        || value.contains("delta")
                        || value.contains("diff") =>
                {
                    EventType::Update
                }

                value => {
                    return Err(
                        format!(
                            "unknown event_type `{value}` at row {row}"
                        )
                        .into()
                    );
                }
            };

        let is_bid =
            match side
                .value(row)
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "bid" | "b" | "buy" => true,

                "ask" | "a" | "sell" => false,

                value => {
                    return Err(
                        format!(
                            "unknown side `{value}` at row {row}"
                        )
                        .into()
                    );
                }
            };

        let price =
            price
                .value(row)
                .trim()
                .parse::<f64>()
                .map_err(|err| {
                    format!(
                        "invalid price `{}` at row {row}: {err}",
                        price.value(row)
                    )
                })?;

        let quantity =
            quantity
                .value(row)
                .trim()
                .parse::<f64>()
                .map_err(|err| {
                    format!(
                        "invalid quantity `{}` at row {row}: {err}",
                        quantity.value(row)
                    )
                })?;

        rows.push(LobRow {
            event_time: event_time.value(row),
            event_type,
            is_bid,
            price,
            quantity,
            first_update_id:
                optional_update_id(
                    first_update_id,
                    row,
                ),
            final_update_id:
                optional_update_id(
                    final_update_id,
                    row,
                ),
            prev_final_update_id:
                optional_update_id(
                    prev_final_update_id,
                    row,
                ),
        });
    }

    Ok(())
}


// ============================================================================
// EXCHANGE EVENT IDENTITY
// ============================================================================

fn same_event(
    a: &LobRow,
    b: &LobRow,
) -> bool {
    a.event_time
        == b.event_time
        && a.event_type
            == b.event_type
        && a.first_update_id
            == b.first_update_id
        && a.final_update_id
            == b.final_update_id
        && a.prev_final_update_id
            == b.prev_final_update_id
}


// ============================================================================
// OUTPUT CHECKPOINT WRITER
// ============================================================================

struct CheckpointWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
}


impl CheckpointWriter {
    fn new(
        path: &Path,
    ) -> Result<Self, BoxError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let schema =
            Arc::new(Schema::new(vec![
                Field::new(
                    "event_time",
                    DataType::Int64,
                    false,
                ),
                Field::new(
                    "final_update_id",
                    DataType::UInt64,
                    false,
                ),
                Field::new(
                    "is_bid",
                    DataType::Boolean,
                    false,
                ),
                Field::new(
                    "price",
                    DataType::Float64,
                    false,
                ),
                Field::new(
                    "quantity",
                    DataType::Float64,
                    false,
                ),
            ]));

        let file =
            File::create(path)?;

        let properties =
            WriterProperties::builder()
                .build();

        let writer =
            ArrowWriter::try_new(
                file,
                schema.clone(),
                Some(properties),
            )?;

        Ok(Self {
            writer,
            schema,
        })
    }


    fn write_checkpoint(
        &mut self,
        checkpoint: &BookCheckpoint,
    ) -> Result<(), BoxError> {
        let mut event_time = Vec::new();
        let mut final_update_id = Vec::new();
        let mut is_bid = Vec::new();
        let mut price = Vec::new();
        let mut quantity = Vec::new();

        for &(tick, qty) in &checkpoint.bids {
            event_time.push(
                checkpoint.event_time
            );

            final_update_id.push(
                checkpoint.final_update_id
            );

            is_bid.push(true);

            price.push(
                tick_to_price(tick)
            );

            quantity.push(qty);
        }

        for &(tick, qty) in &checkpoint.asks {
            event_time.push(
                checkpoint.event_time
            );

            final_update_id.push(
                checkpoint.final_update_id
            );

            is_bid.push(false);

            price.push(
                tick_to_price(tick)
            );

            quantity.push(qty);
        }

        let batch =
            RecordBatch::try_new(
                self.schema.clone(),
                vec![
                    Arc::new(
                        Int64Array::from(
                            event_time
                        )
                    ) as ArrayRef,

                    Arc::new(
                        UInt64Array::from(
                            final_update_id
                        )
                    ) as ArrayRef,

                    Arc::new(
                        BooleanArray::from(
                            is_bid
                        )
                    ) as ArrayRef,

                    Arc::new(
                        Float64Array::from(
                            price
                        )
                    ) as ArrayRef,

                    Arc::new(
                        Float64Array::from(
                            quantity
                        )
                    ) as ArrayRef,
                ],
            )?;

        self.writer.write(&batch)?;

        Ok(())
    }


    fn finish(
        self,
    ) -> Result<(), BoxError> {
        self.writer.close()?;
        Ok(())
    }
}


// ============================================================================
// PROCESS ONE COMPLETE EXCHANGE EVENT
// ============================================================================

fn process_one_event(
    event: &[LobRow],
    engine: &mut ReplayEngine,
) -> Result<Option<BookCheckpoint>, BoxError> {
    if event.is_empty() {
        return Ok(None);
    }

    engine.stats.batches += 1;

    match engine.process_event(event) {
        Ok(()) => {}

        Err(
            ReplayError::PartialSnapshot
            | ReplayError::InconsistentBatchIds
            | ReplayError::NonMonotonicEventTime,
        ) => {
            return Ok(None);
        }

        Err(other) => {
            return Err(other.into());
        }
    }

    Ok(
        engine.checkpoint(
            event[0].event_time
        )
    )
}


// ============================================================================
// UTC DAY
// ============================================================================

fn utc_day(
    event_time_ms: i64,
) -> i64 {
    crate::replay::day_index(
        event_time_ms
    )
}


// ============================================================================
// MAIN INGEST API
// ============================================================================
pub fn process_matched(
    input: &Path,
    output: &Path,
    tick_size: f64,
    warm_start: bool,
) -> Result<ReplayStats, BoxError> {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let input_files = discover_input_files(input)?;

    let mut engine =
        ReplayEngine::with_options(
            tick_size,
            warm_start,
        )?;

    engine.set_checkpoint_policy(
        CheckpointPolicy::EverySnapshot
    );

    let mut writer =
        CheckpointWriter::new(output)?;

    let mut pending_event: Vec<LobRow> =
        Vec::new();

    for input_file in input_files {
        let file = File::open(&input_file)?;

        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file)?;

        let reader = builder.build()?;

        for batch_result in reader {
            let batch = batch_result?;

            let mut rows =
                Vec::with_capacity(batch.num_rows());

            append_rows(
                &batch,
                &mut rows,
            )?;

            for row in rows {
                if pending_event.is_empty() {
                    pending_event.push(row);
                    continue;
                }

                if same_event(
                    &pending_event[0],
                    &row,
                ) {
                    pending_event.push(row);
                    continue;
                }

                if let Some(checkpoint) =
                    process_one_event(
                        &pending_event,
                        &mut engine,
                    )?
                {
                    writer.write_checkpoint(
                        &checkpoint
                    )?;
                }

                pending_event.clear();
                pending_event.push(row);
            }
        }
    }

    // Flush the final event.
    if !pending_event.is_empty() {
        if let Some(checkpoint) =
            process_one_event(
                &pending_event,
                &mut engine,
            )?
        {
            writer.write_checkpoint(
                &checkpoint
            )?;
        }
    }

    writer.finish()?;

    Ok(engine.stats)
}

pub fn ingest_parquet(
    input: &Path,
    output_dir: &Path,
    tick_size: f64,
    warm_start: bool,
) -> Result<ReplayStats, BoxError> {
    ingest_parquet_with_policy(
        input,
        output_dir,
        tick_size,
        warm_start,
        CheckpointPolicy::DayStart,
    )
}
pub fn ingest_parquet_with_policy(
    input: &Path,
    output_dir: &Path,
    tick_size: f64,
    warm_start: bool,
    policy: CheckpointPolicy,
) -> Result<ReplayStats, BoxError> {
    // Do not allow output to live inside the input tree.
    // Otherwise a rerun can discover its own previously-generated parquet files.
    if input.is_dir() && output_dir.starts_with(input) {
        return Err(
            format!(
                "output directory must not be inside input directory:\n  input: {}\n  output: {}",
                input.display(),
                output_dir.display()
            )
            .into()
        );
    }

    // For a single input file, do not write output into the same directory
    // as that input file. This prevents accidentally opening the source
    // parquet for writing and avoids Windows sharing/access-denied errors.
    if input.is_file() {
        if let Some(parent) = input.parent() {
            if output_dir == parent {
                return Err(
                    format!(
                        "output directory must be different from the input file's directory:\n  input: {}\n  output: {}",
                        input.display(),
                        output_dir.display()
                    )
                    .into()
                );
            }
        }
    }

    // Discover inputs BEFORE creating output files.
    let input_files = discover_input_files(input)?;

    std::fs::create_dir_all(output_dir)?;

    let mut engine =
        ReplayEngine::with_options(
            tick_size,
            warm_start,
        )?;

    // Replay continuously through gaps in every-event mode.
    //
    // DayStart is implemented as an output filter below:
    // first eligible checkpoint of each UTC day.
    engine.set_checkpoint_policy(
        CheckpointPolicy::EverySnapshot
    );

    let mut written_days =
        std::collections::BTreeSet::<i64>::new();

    for input_file in input_files {
        // ------------------------------------------------------------------
        // OUTPUT PATH
        //
        // Never use the same basename as the input parquet.
        //
        // Example:
        //   input/day1/book.parquet
        // becomes:
        //   output/day1/book_every_event.parquet
        // ------------------------------------------------------------------

        let mut current_writer: Option<CheckpointWriter> = None;

        if matches!(
            policy,
            CheckpointPolicy::EverySnapshot
        ) {
            let relative_path = if input.is_dir() {
                input_file
                    .strip_prefix(input)
                    .map_err(|err| {
                        format!(
                            "failed to determine relative path for {} under {}: {err}",
                            input_file.display(),
                            input.display()
                        )
                    })?
                    .to_path_buf()
            } else {
                input_file
                    .file_name()
                    .ok_or_else(|| {
                        format!(
                            "invalid input filename: {}",
                            input_file.display()
                        )
                    })?
                    .into()
            };

            let stem = relative_path
                .file_stem()
                .and_then(|x| x.to_str())
                .ok_or_else(|| {
                    format!(
                        "invalid parquet filename: {}",
                        relative_path.display()
                    )
                })?;

            let output_filename =
                format!("{stem}_every_event.parquet");

            let output_relative =
                relative_path
                    .parent()
                    .unwrap_or_else(|| Path::new(""))
                    .join(output_filename);

            let out_path =
                output_dir.join(output_relative);

            // Make the mirrored subdirectory if necessary.
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Final safety check: never write to the source file.
            if out_path == input_file {
                return Err(
                    format!(
                        "refusing to overwrite input parquet: {}",
                        input_file.display()
                    )
                    .into()
                );
            }

            current_writer =
                Some(
                    CheckpointWriter::new(
                        &out_path
                    )?
                );
        }

        // ------------------------------------------------------------------
        // OPEN INPUT ONCE
        // ------------------------------------------------------------------

        let file =
            File::open(&input_file)?;

        let builder =
            ParquetRecordBatchReaderBuilder::try_new(
                file
            )?;

        let reader =
            builder.build()?;

        // Event grouping is per input parquet file.
        let mut pending_event =
            Vec::<LobRow>::new();

        // ------------------------------------------------------------------
        // STREAM RECORD BATCHES
        // ------------------------------------------------------------------

        for batch_result in reader {
            let batch =
                batch_result?;

            let mut rows =
                Vec::with_capacity(
                    batch.num_rows()
                );

            append_rows(
                &batch,
                &mut rows
            )?;

            for row in rows {
                if pending_event.is_empty() {
                    pending_event.push(row);
                    continue;
                }

                if same_event(
                    &pending_event[0],
                    &row,
                ) {
                    pending_event.push(row);
                    continue;
                }

                // The previous grouped exchange event is complete.
                if let Some(checkpoint) =
                    process_one_event(
                        &pending_event,
                        &mut engine,
                    )?
                {
                    match policy {
                        CheckpointPolicy::EverySnapshot => {
                            current_writer
                                .as_mut()
                                .expect(
                                    "every-event writer missing"
                                )
                                .write_checkpoint(
                                    &checkpoint
                                )?;
                        }

                        CheckpointPolicy::DayStart => {
                            let day =
                                utc_day(
                                    checkpoint.event_time
                                );

                            if written_days.insert(day) {
                                let path =
                                    output_dir.join(
                                        format!(
                                            "day_start_{day}.parquet"
                                        )
                                    );

                                let mut writer =
                                    CheckpointWriter::new(
                                        &path
                                    )?;

                                writer.write_checkpoint(
                                    &checkpoint
                                )?;

                                writer.finish()?;
                            }
                        }
                    }
                }

                pending_event.clear();
                pending_event.push(row);
            }
        }

        // ------------------------------------------------------------------
        // FLUSH FINAL EVENT FROM THIS INPUT FILE
        // ------------------------------------------------------------------

        if !pending_event.is_empty() {
            if let Some(checkpoint) =
                process_one_event(
                    &pending_event,
                    &mut engine,
                )?
            {
                match policy {
                    CheckpointPolicy::EverySnapshot => {
                        current_writer
                            .as_mut()
                            .expect(
                                "every-event writer missing"
                            )
                            .write_checkpoint(
                                &checkpoint
                            )?;
                    }

                    CheckpointPolicy::DayStart => {
                        let day =
                            utc_day(
                                checkpoint.event_time
                            );

                        if written_days.insert(day) {
                            let path =
                                output_dir.join(
                                    format!(
                                        "day_start_{day}.parquet"
                                    )
                                );

                            let mut writer =
                                CheckpointWriter::new(
                                    &path
                                )?;

                            writer.write_checkpoint(
                                &checkpoint
                            )?;

                            writer.finish()?;
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // CLOSE OUTPUT WRITER FOR THIS INPUT FILE
        // ------------------------------------------------------------------

        if let Some(writer) =
            current_writer
        {
            writer.finish()?;
        }

        // `reader` and its underlying input File are dropped here
        // before moving to the next input parquet.
    }

    Ok(engine.stats)
}