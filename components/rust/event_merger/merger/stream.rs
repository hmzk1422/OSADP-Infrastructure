//! Streaming readers for the three source streams.
//!
//! LOB inputs are long-format LOB-builder checkpoints:
//! one physical row per resting level with `(event_time, final_update_id,
//! is_bid, price, quantity)`. `next_lob` groups every physical row sharing the
//! same `(event_time, final_update_id)` into one aggregated book-state event
//! and emits `bid_level_1..N` / `ask_level_1..N` in the canonical order.

use std::sync::Arc;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use crate::args::BoxError;
use crate::model::{
    bool_at, f64_at, i64_at, integer_at, timestamp_ns, Cursor, DepthState, Event, LobState, Source,
    TradeState,
};

fn source_values(batch: &RecordBatch, row: usize) -> Vec<arrow::array::ArrayRef> {
    (0..batch.num_columns())
        .map(|index| batch.column(index).slice(row, 1))
        .collect()
}

fn source_fields(batch: &RecordBatch) -> Arc<Vec<arrow::datatypes::Field>> {
    Arc::new(batch.schema().fields().iter().map(|field| field.as_ref().clone()).collect())
}

pub fn next_lob(
    cursor: &mut Cursor,
    levels: usize,
    sequence: u64,
) -> Result<Option<Event>, BoxError> {
    if !cursor.next_row()? {
        return Ok(None);
    }
    let batch = cursor.batch.clone().expect("cursor row available");
    let row = cursor.row;
    cursor.consume_row();
    let event_time = i64_at(&batch, "event_time", row)?.ok_or("LOB event_time is null")?;
    let final_update_id = integer_at(&batch, "final_update_id", row)?;
    let first_is_bid = bool_at(&batch, "is_bid", row)?.ok_or("LOB is_bid is null")?;
    let first_price = f64_at(&batch, "price", row)?.ok_or("LOB price is null")?;
    let first_quantity = f64_at(&batch, "quantity", row)?.ok_or("LOB quantity is null")?;

    // Aggregate every physical level row belonging to this exchange event.
    let mut bids: Vec<(f64, f64)> = Vec::new();
    let mut asks: Vec<(f64, f64)> = Vec::new();
    if first_is_bid {
        bids.push((first_price, first_quantity));
    } else {
        asks.push((first_price, first_quantity));
    }
    loop {
        if !cursor.next_row()? {
            break;
        }
        let batch = cursor.batch.clone().expect("cursor row available");
        let row = cursor.row;
        let row_time = i64_at(&batch, "event_time", row)?.ok_or("LOB event_time is null")?;
        let row_final = integer_at(&batch, "final_update_id", row)?;
        if row_time != event_time || row_final != final_update_id {
            cursor.push_back();
            break;
        }
        let is_bid = bool_at(&batch, "is_bid", row)?.ok_or("LOB is_bid is null")?;
        let price = f64_at(&batch, "price", row)?.ok_or("LOB price is null")?;
        let quantity = f64_at(&batch, "quantity", row)?.ok_or("LOB quantity is null")?;
        if is_bid {
            bids.push((price, quantity));
        } else {
            asks.push((price, quantity));
        }
        cursor.consume_row();
    }

    // Canonical book order: bids high to low, asks low to high.
    bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut values = vec![None; 4 * levels];
    if let Some((price, quantity)) = bids.first() {
        values[0] = Some(*price);
        values[1] = Some(*quantity);
    }
    if let Some((price, quantity)) = asks.first() {
        values[2] = Some(*price);
        values[3] = Some(*quantity);
    }
    for level in 2..=levels {
        let base = 4 * (level - 1);
        if let Some((price, quantity)) = bids.get(level - 1) {
            values[base] = Some(*price);
            values[base + 1] = Some(*quantity);
        }
        if let Some((price, quantity)) = asks.get(level - 1) {
            values[base + 2] = Some(*price);
            values[base + 3] = Some(*quantity);
        }
    }

    Ok(Some(Event {
        timestamp: timestamp_ns(event_time),
        source: Source::Lob,
        sequence,
        lob: Some(LobState {
            levels: values,
            event_time: Some(event_time),
            final_update_id,
        }),
        trade: None,
        depth: None,
        depth_state: None,
    }))
}

pub fn next_trade(cursor: &mut Cursor, sequence: u64) -> Result<Option<Event>, BoxError> {
    if !cursor.next_row()? {
        return Ok(None);
    }
    let batch = cursor.batch.clone().expect("cursor row available");
    let row = cursor.row;
    cursor.consume_row();
    let transact_time =
        i64_at(&batch, "transact_time", row)?.ok_or("aggTrades transact_time is null")?;
    let source_qty = f64_at(&batch, "qty", row)?;
    let source_quantity = f64_at(&batch, "quantity", row)?;
    let quantity = source_quantity.or(source_qty);
    Ok(Some(Event {
        timestamp: timestamp_ns(transact_time),
        source: Source::Trade,
        sequence,
        lob: None,
        trade: Some(TradeState {
            agg_trade_id: i64_at(&batch, "agg_trade_id", row)?,
            price: f64_at(&batch, "price", row)?,
            quantity,
            source_qty,
            source_quantity,
            first_trade_id: i64_at(&batch, "first_trade_id", row)?,
            last_trade_id: i64_at(&batch, "last_trade_id", row)?,
            transact_time: Some(transact_time),
            is_buyer_maker: bool_at(&batch, "is_buyer_maker", row)?,
            source_fields: source_fields(&batch),
            source_values: source_values(&batch, row),
        }),
        depth: None,
        depth_state: None,
    }))
}

pub fn next_depth(cursor: &mut Cursor, sequence: u64) -> Result<Option<Event>, BoxError> {
    if !cursor.next_row()? {
        return Ok(None);
    }
    let batch = cursor.batch.clone().expect("cursor row available");
    let row = cursor.row;
    cursor.consume_row();
    let timestamp = i64_at(&batch, "timestamp", row)?.ok_or("bookDepth timestamp is null")?;
    let percentage = i64_at(&batch, "percentage", row)?;
    let depth = f64_at(&batch, "depth", row)?;
    let notional = f64_at(&batch, "notional", row)?;
    Ok(Some(Event {
        timestamp: timestamp_ns(timestamp),
        source: Source::Depth,
        sequence,
        lob: None,
        trade: None,
        depth: Some(DepthState {
            timestamp: Some(timestamp),
            percentage,
            depth,
            notional,
            source_fields: source_fields(&batch),
            source_values: source_values(&batch, row),
        }),
        depth_state: None,
    }))
}
