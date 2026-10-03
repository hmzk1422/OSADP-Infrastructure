//! Bounded Parquet writer for merged events.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    new_null_array, Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray,
    UInt64Array,
};
use arrow::compute::concat;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet_rs::arrow::ArrowWriter;
use parquet_rs::file::properties::WriterProperties;

use crate::args::BoxError;
use crate::model::{Event, Source};

#[derive(Debug, Clone)]
pub struct SourceSchema {
    pub source: Source,
    pub schema: Arc<Schema>,
}

#[derive(Debug, Clone)]
pub struct SourceOutput {
    source: Source,
    source_name: String,
    data_type: DataType,
}

pub fn schema(levels: usize) -> Arc<Schema> {
    Arc::new(Schema::new(state_fields(levels)))
}

fn state_fields(levels: usize) -> Vec<Field> {
    let mut fields = vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("event_sequence", DataType::UInt64, false),
        Field::new("lob__event_time", DataType::Int64, true),
        Field::new("lob__final_update_id", DataType::Int64, true),
    ];
    for name in [
        "book_state__best_bid_price",
        "book_state__best_bid_qty",
        "book_state__best_ask_price",
        "book_state__best_ask_qty",
    ] {
        fields.push(Field::new(name, DataType::Float64, true));
    }
    for level in 2..=levels {
        for name in [
            format!("book_state__bid_price_{level}"),
            format!("book_state__bid_qty_{level}"),
            format!("book_state__ask_price_{level}"),
            format!("book_state__ask_qty_{level}"),
        ] {
            fields.push(Field::new(name, DataType::Float64, true));
        }
    }
    for name in [
        "agg__agg_trade_id",
        "agg__price",
        "agg__quantity",
        "agg__qty",
        "agg__first_trade_id",
        "agg__last_trade_id",
        "agg__transact_time",
        "agg__is_buyer_maker",
    ] {
        let kind = if name == "agg__price" || name == "agg__quantity" || name == "agg__qty" {
            DataType::Float64
        } else if name == "agg__is_buyer_maker" {
            DataType::Boolean
        } else {
            DataType::Int64
        };
        fields.push(Field::new(name, kind, true));
    }
    for name in [
        "depth__timestamp",
        "depth__percentage",
        "depth__depth",
        "depth__notional",
    ] {
        let kind = if name == "depth__depth" || name == "depth__notional" {
            DataType::Float64
        } else {
            DataType::Int64
        };
        fields.push(Field::new(name, kind, true));
    }
    for band in -5..=5 {
        if band != 0 {
            fields.push(Field::new(
                format!("depth_state__{band}"),
                DataType::Float64,
                true,
            ));
        }
    }
    for band in -5..=5 {
        if band != 0 {
            fields.push(Field::new(
                format!("depth_state__notional_{band}"),
                DataType::Float64,
                true,
            ));
        }
    }
    fields
}

pub fn source_schema(
    levels: usize,
    sources: &[SourceSchema],
) -> Result<(Arc<Schema>, Vec<SourceOutput>), BoxError> {
    let mut fields = state_fields(levels);
    let mut outputs = Vec::new();
    let mut seen: HashMap<(Source, String), DataType> = HashMap::new();
    for source in sources {
        // LOB levels are emitted only through the aggregated canonical
        // book-state fields. Keeping a second verbatim `source_lob__*` copy
        // would duplicate the same resting levels, so skip the LOB source
        // schema here.
        if source.source == Source::Lob {
            continue;
        }
        for field in source.schema.fields() {
            let output_name = format!("source_{}__{}", source_prefix(source.source), field.name());
            let key = (source.source, field.name().to_string());
            if let Some(existing) = seen.get(&key) {
                if existing != field.data_type() {
                    return Err(format!(
                        "source field {}::{} has incompatible types {} and {}",
                        source_prefix(source.source), field.name(), existing, field.data_type()
                    )
                    .into());
                }
                continue;
            }
            seen.insert(key, field.data_type().clone());
            fields.push(Field::new(&output_name, field.data_type().clone(), true));
            outputs.push(SourceOutput {
                source: source.source,
                source_name: field.name().to_string(),
                data_type: field.data_type().clone(),
            });
        }
    }
    Ok((Arc::new(Schema::new(fields)), outputs))
}

fn source_prefix(source: Source) -> &'static str {
    match source { Source::Lob => "lob", Source::Trade => "agg", Source::Depth => "depth" }
}

fn i64s(values: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}
fn f64s(values: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}
fn bools(values: &[Option<bool>]) -> ArrayRef {
    Arc::new(BooleanArray::from(values.to_vec()))
}

fn source_arrays(
    events: &[Event],
    source_outputs: &[SourceOutput],
) -> Result<Vec<ArrayRef>, BoxError> {
    let mut output = Vec::with_capacity(source_outputs.len());
    for source in source_outputs {
        let mut pieces = Vec::with_capacity(events.len());
        for event in events {
            if event.source == source.source {
                let (fields, values) = match event.source {
                    Source::Lob => {
                        return Err("LOB source output is disabled; LOB levels use book_state__*".into())
                    }
                    Source::Trade => event.trade.as_ref().map(|x| (x.source_fields.as_ref(), x.source_values.as_slice())),
                    Source::Depth => event.depth.as_ref().map(|x| (x.source_fields.as_ref(), x.source_values.as_slice())),
                }
                .ok_or_else(|| "source state missing from event".to_string())?;
                let Some(position) = fields
                    .iter()
                    .position(|field| field.name() == source.source_name.as_str())
                else {
                    pieces.push(new_null_array(&source.data_type, 1));
                    continue;
                };
                pieces.push(values[position].slice(0, 1));
            } else {
                pieces.push(new_null_array(&source.data_type, 1));
            }
        }
        let refs: Vec<&dyn Array> = pieces.iter().map(|array| array.as_ref()).collect();
        output.push(concat(&refs)?);
    }
    Ok(output)
}

pub struct HourWriter {
    path: std::path::PathBuf,
    temp: std::path::PathBuf,
    writer: Option<ArrowWriter<File>>,
    buffer: Vec<Event>,
    rows: usize,
    levels: usize,
    schema: Arc<Schema>,
    source_outputs: Vec<SourceOutput>,
}

impl HourWriter {
    pub fn new(
        path: &Path,
        levels: usize,
        schema: Arc<Schema>,
        source_outputs: Vec<SourceOutput>,
    ) -> Result<Self, BoxError> {
        let temp = path.with_extension("tmp.parquet");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let writer = ArrowWriter::try_new(
            File::create(&temp)?,
            schema.clone(),
            Some(WriterProperties::builder().build()),
        )?;
        Ok(Self {
            path: path.to_path_buf(),
            temp,
            writer: Some(writer),
            schema: schema.clone(),
            source_outputs,
            buffer: Vec::new(),
            rows: 0,
            levels,
        })
    }

    pub fn push(&mut self, event: Event) -> Result<(), BoxError> {
        self.buffer.push(event);
        self.rows += 1;
        if self.buffer.len() >= 8192 {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), BoxError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let batch = build_batch(
            &self.buffer,
            self.levels,
            self.schema.clone(),
            &self.source_outputs,
        )?;
        self.writer.as_mut().expect("writer open").write(&batch)?;
        self.buffer.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), BoxError> {
        self.flush()?;
        self.writer.take().expect("writer open").close()?;
        if self.rows == 0 {
            std::fs::remove_file(&self.temp)?;
            if self.path.exists() {
                std::fs::remove_file(&self.path)?;
            }
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(self.temp, &self.path)?;
        Ok(())
    }
}

fn build_batch(
    events: &[Event],
    levels: usize,
    output_schema: Arc<Schema>,
    source_outputs: &[SourceOutput],
) -> Result<RecordBatch, BoxError> {
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            events.iter().map(|e| e.timestamp).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            events.iter().map(|e| e.source.name()).collect::<Vec<_>>(),
        )),
        Arc::new(UInt64Array::from(
            events.iter().map(|e| e.sequence).collect::<Vec<_>>(),
        )),
    ];
    macro_rules! li {
        ($field:ident) => {{
            let v: Vec<Option<i64>> = events
                .iter()
                .map(|e| e.lob.as_ref().and_then(|x| x.$field))
                .collect();
            arrays.push(i64s(&v));
        }};
    }
    li!(event_time);
    li!(final_update_id);
    for c in 0..4 * levels {
        let v: Vec<Option<f64>> = events
            .iter()
            .map(|e| {
                e.lob
                    .as_ref()
                    .and_then(|x| x.levels.get(c).copied().flatten())
            })
            .collect();
        arrays.push(f64s(&v));
    }
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.agg_trade_id))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.price))
        .collect();
    arrays.push(f64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.quantity))
        .collect();
    arrays.push(f64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.source_qty))
        .collect();
    arrays.push(f64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.first_trade_id))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.last_trade_id))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.transact_time))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<bool>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.is_buyer_maker))
        .collect();
    arrays.push(bools(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.timestamp))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.percentage))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.depth))
        .collect();
    arrays.push(f64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.notional))
        .collect();
    arrays.push(f64s(&v));
    for band in -5..=5 {
        if band == 0 {
            continue;
        }
        let v: Vec<Option<f64>> = events
            .iter()
            .map(|e| {
                e.depth_state
                    .as_ref()
                    .and_then(|s| s.get(&band).and_then(|x| x.0))
            })
            .collect();
        arrays.push(f64s(&v));
    }
    for band in -5..=5 {
        if band == 0 {
            continue;
        }
        let v: Vec<Option<f64>> = events
            .iter()
            .map(|e| {
                e.depth_state
                    .as_ref()
                    .and_then(|s| s.get(&band).and_then(|x| x.1))
            })
            .collect();
        arrays.push(f64s(&v));
    }
    let base_len = arrays.len();
    let source_arrays = source_arrays(events, source_outputs)?;
    debug_assert_eq!(base_len + source_arrays.len(), output_schema.fields().len());
    arrays.extend(source_arrays);
    Ok(RecordBatch::try_new(output_schema, arrays)?)
}

pub fn write(path: &Path, events: &[Event], levels: usize) -> Result<(), BoxError> {
    let schema = schema(levels);
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            events.iter().map(|e| e.timestamp).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            events.iter().map(|e| e.source.name()).collect::<Vec<_>>(),
        )),
        Arc::new(UInt64Array::from(
            events.iter().map(|e| e.sequence).collect::<Vec<_>>(),
        )),
    ];
    macro_rules! li {
        ($field:ident) => {{
            let v: Vec<Option<i64>> = events
                .iter()
                .map(|e| e.lob.as_ref().and_then(|x| x.$field))
                .collect();
            arrays.push(i64s(&v));
        }};
    }
    li!(event_time);
    li!(final_update_id);
    for c in 0..4 * levels {
        let v: Vec<Option<f64>> = events
            .iter()
            .map(|e| {
                e.lob
                    .as_ref()
                    .and_then(|x| x.levels.get(c).copied().flatten())
            })
            .collect();
        arrays.push(f64s(&v));
    }
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.agg_trade_id))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.price))
        .collect();
    arrays.push(f64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.quantity))
        .collect();
    arrays.push(f64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.first_trade_id))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.last_trade_id))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.transact_time))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<bool>> = events
        .iter()
        .map(|e| e.trade.as_ref().and_then(|x| x.is_buyer_maker))
        .collect();
    arrays.push(bools(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.timestamp))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<i64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.percentage))
        .collect();
    arrays.push(i64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.depth))
        .collect();
    arrays.push(f64s(&v));
    let v: Vec<Option<f64>> = events
        .iter()
        .map(|e| e.depth.as_ref().and_then(|x| x.notional))
        .collect();
    arrays.push(f64s(&v));
    for band in -5..=5 {
        if band == 0 {
            continue;
        }
        let v: Vec<Option<f64>> = events
            .iter()
            .map(|e| {
                e.depth_state
                    .as_ref()
                    .and_then(|s| s.get(&band).and_then(|x| x.0))
            })
            .collect();
        arrays.push(f64s(&v));
    }
    for band in -5..=5 {
        if band == 0 {
            continue;
        }
        let v: Vec<Option<f64>> = events
            .iter()
            .map(|e| {
                e.depth_state
                    .as_ref()
                    .and_then(|s| s.get(&band).and_then(|x| x.1))
            })
            .collect();
        arrays.push(f64s(&v));
    }
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp.parquet");
    let file = File::create(&tmp)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))?;
    writer.write(&batch)?;
    writer.close()?;
    std::fs::rename(tmp, path)?;
    Ok(())
}
