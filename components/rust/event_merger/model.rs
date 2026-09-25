//! Shared event model for the streaming event merger.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatchReader,
    StringArray, UInt64Array,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet_rs::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

pub type BoxError = Box<dyn std::error::Error>;

pub fn timestamp_ns(value: i64) -> i64 {
    let magnitude = value.unsigned_abs();
    if magnitude >= 100_000_000_000_000_000 {
        value
    } else if magnitude >= 100_000_000_000_000 {
        value.saturating_mul(1_000)
    } else if magnitude >= 100_000_000_000 {
        value.saturating_mul(1_000_000)
    } else {
        value.saturating_mul(1_000_000_000)
    }
}

fn index(batch: &RecordBatch, name: &str) -> Option<usize> {
    batch.schema().index_of(name).ok()
}

pub fn i64_at(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<i64>, BoxError> {
    let Some(i) = index(batch, name) else {
        return Ok(None);
    };
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("{name} is not int64"))?;
    Ok((!a.is_null(row)).then(|| a.value(row)))
}

pub fn u64_at(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<u64>, BoxError> {
    let Some(i) = index(batch, name) else {
        return Ok(None);
    };
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| format!("{name} is not uint64"))?;
    Ok((!a.is_null(row)).then(|| a.value(row)))
}

pub fn integer_at(batch: &RecordBatch, name: &str, i: usize) -> Result<Option<i64>, BoxError> {
    if let Ok(value) = i64_at(batch, name, i) {
        return Ok(value);
    }
    if let Ok(value) = u64_at(batch, name, i) {
        return Ok(value.and_then(|v| i64::try_from(v).ok()));
    }
    Ok(None)
}

pub fn f64_at(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<f64>, BoxError> {
    let Some(i) = index(batch, name) else {
        return Ok(None);
    };
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("{name} is not float64"))?;
    Ok((!a.is_null(row)).then(|| a.value(row)))
}

pub fn bool_at(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<bool>, BoxError> {
    let Some(i) = index(batch, name) else {
        return Ok(None);
    };
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| format!("{name} is not bool"))?;
    Ok((!a.is_null(row)).then(|| a.value(row)))
}

pub fn string_at(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<String>, BoxError> {
    let Some(i) = index(batch, name) else {
        return Ok(None);
    };
    let column = batch.column(i);
    if let Some(a) = column.as_any().downcast_ref::<StringArray>() {
        return Ok((!a.is_null(row)).then(|| a.value(row).to_string()));
    }
    if let Some(a) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok((!a.is_null(row)).then(|| a.value(row).to_string()));
    }
    Err(format!("{name} is not string or large_string").into())
}

pub struct Cursor {
    pub reader: ParquetRecordBatchReader,
    pub batch: Option<RecordBatch>,
    pub row: usize,
    pub path: PathBuf,
}

impl Cursor {
    pub fn open(path: &Path) -> Result<Self, BoxError> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
        Ok(Self {
            reader,
            batch: None,
            row: 0,
            path: path.to_path_buf(),
        })
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.reader.schema()
    }

    pub fn next_row(&mut self) -> Result<bool, BoxError> {
        loop {
            if self.batch.as_ref().is_some_and(|b| self.row < b.num_rows()) {
                return Ok(true);
            }
            match self.reader.next() {
                Some(Ok(batch)) => {
                    self.batch = Some(batch);
                    self.row = 0;
                }
                Some(Err(e)) => return Err(e.into()),
                None => {
                    self.batch = None;
                    return Ok(false);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    Lob = 0,
    Trade = 1,
    Depth = 2,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Self::Lob => "LOB",
            Self::Trade => "TRADE",
            Self::Depth => "BOOKDEPTH",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LobState {
    pub levels: Vec<Option<f64>>,
    pub source_fields: Arc<Vec<Field>>,
    pub source_values: Vec<ArrayRef>,
    pub received_time: Option<i64>,
    pub event_time: Option<i64>,
    pub transaction_time: Option<i64>,
    pub symbol: Option<String>,
    pub source_event_type: Option<String>,
    pub side: Option<String>,
    pub is_bid: Option<bool>,
    pub is_snapshot: Option<bool>,
    pub source_price: Option<String>,
    pub source_quantity: Option<String>,
    pub first_update_id: Option<i64>,
    pub final_update_id: Option<i64>,
    pub prev_final_update_id: Option<i64>,
    pub last_update_id: Option<i64>,
}

#[derive(Debug, Clone, Default)]
pub struct TradeState {
    pub agg_trade_id: Option<i64>,
    pub price: Option<f64>,
    pub quantity: Option<f64>,
    pub source_qty: Option<f64>,
    pub source_quantity: Option<f64>,
    pub source_fields: Arc<Vec<Field>>,
    pub source_values: Vec<ArrayRef>,
    pub first_trade_id: Option<i64>,
    pub last_trade_id: Option<i64>,
    pub transact_time: Option<i64>,
    pub is_buyer_maker: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct DepthState {
    pub timestamp: Option<i64>,
    pub percentage: Option<i64>,
    pub depth: Option<f64>,
    pub notional: Option<f64>,
    pub source_fields: Arc<Vec<Field>>,
    pub source_values: Vec<ArrayRef>,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub timestamp: i64,
    pub source: Source,
    pub sequence: u64,
    pub lob: Option<LobState>,
    pub trade: Option<TradeState>,
    pub depth: Option<DepthState>,
    pub depth_state: Option<BTreeMap<i64, (Option<f64>, Option<f64>)>>,
}
