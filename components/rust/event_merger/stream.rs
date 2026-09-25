//! Streaming readers for the three source streams.

use std::sync::Arc;

use arrow::array::Array;
use arrow::record_batch::RecordBatch;

use crate::args::BoxError;
use crate::model::{
    bool_at, f64_at, i64_at, integer_at, string_at, timestamp_ns, Cursor, DepthState, Event,
    LobState, Source, TradeState,
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
    let batch = cursor.batch.as_ref().expect("cursor row available");
    let row = cursor.row;
    cursor.row += 1;
    let event_time = i64_at(batch, "event_time", row)?.ok_or("LOB event_time is null")?;
    let received_time = i64_at(batch, "received_time", row)?;
    let values = {
        let mut values = vec![None; 4 * levels];
        values[0] = f64_at(batch, "best_bid_price", row)?
            .or(f64_at(batch, "bid_price_1", row)?);
        values[1] = f64_at(batch, "best_bid_qty", row)?
            .or(f64_at(batch, "bid_qty_1", row)?);
        values[2] = f64_at(batch, "best_ask_price", row)?
            .or(f64_at(batch, "ask_price_1", row)?);
        values[3] = f64_at(batch, "best_ask_qty", row)?
            .or(f64_at(batch, "ask_qty_1", row)?);
        for level in 2..=levels {
            let base = 4 * (level - 1);
            values[base] = f64_at(batch, &format!("bid_price_{level}"), row)?;
            values[base + 1] = f64_at(batch, &format!("bid_qty_{level}"), row)?;
            values[base + 2] = f64_at(batch, &format!("ask_price_{level}"), row)?;
            values[base + 3] = f64_at(batch, &format!("ask_qty_{level}"), row)?;
        }
        values
    };
    let final_update_id = integer_at(batch, "final_update_id", row)?;
    let event_type = string_at(batch, "source_event_type", row)?
        .or(string_at(batch, "event_type", row)?);
    let side = string_at(batch, "side", row)?;
    let is_bid = bool_at(batch, "is_bid", row)?.or_else(|| {
        side.as_deref().map(|value| {
            value.eq_ignore_ascii_case("bid") || value.eq_ignore_ascii_case("b")
        })
    });
    let is_snapshot = bool_at(batch, "is_snapshot", row)?.or_else(|| {
        event_type
            .as_deref()
            .map(|value| value.eq_ignore_ascii_case("snapshot"))
    });
    Ok(Some(Event {
        timestamp: timestamp_ns(received_time.unwrap_or(event_time)),
        source: Source::Lob,
        sequence,
        lob: Some(LobState {
            levels: values,
            received_time,
            event_time: Some(event_time),
            transaction_time: i64_at(batch, "transaction_time", row)?,
            symbol: string_at(batch, "symbol", row)?,
            source_event_type: event_type,
            side,
            is_bid,
            is_snapshot,
            source_price: string_at(batch, "source_price", row)?
                .or(string_at(batch, "price", row)?),
            source_quantity: string_at(batch, "source_quantity", row)?
                .or(string_at(batch, "quantity", row)?),
            first_update_id: integer_at(batch, "first_update_id", row)?,
            final_update_id,
            prev_final_update_id: integer_at(batch, "prev_final_update_id", row)?,
            last_update_id: integer_at(batch, "last_update_id", row)?,
            source_fields: source_fields(batch),
            source_values: source_values(batch, row),
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
    let batch = cursor.batch.as_ref().expect("cursor row available");
    let row = cursor.row;
    cursor.row += 1;
    let transact_time =
        i64_at(batch, "transact_time", row)?.ok_or("aggTrades transact_time is null")?;
    let source_qty = f64_at(batch, "qty", row)?;
    let source_quantity = f64_at(batch, "quantity", row)?;
    let quantity = source_quantity.or(source_qty);
    Ok(Some(Event {
        timestamp: timestamp_ns(transact_time),
        source: Source::Trade,
        sequence,
        lob: None,
        trade: Some(TradeState {
            agg_trade_id: i64_at(batch, "agg_trade_id", row)?,
            price: f64_at(batch, "price", row)?,
            quantity,
            source_qty,
            source_quantity,
            first_trade_id: i64_at(batch, "first_trade_id", row)?,
            last_trade_id: i64_at(batch, "last_trade_id", row)?,
            transact_time: Some(transact_time),
            is_buyer_maker: bool_at(batch, "is_buyer_maker", row)?,
            source_fields: source_fields(batch),
            source_values: source_values(batch, row),
        }),
        depth: None,
        depth_state: None,
    }))
}

pub fn next_depth(cursor: &mut Cursor, sequence: u64) -> Result<Option<Event>, BoxError> {
    if !cursor.next_row()? {
        return Ok(None);
    }
    let batch = cursor.batch.as_ref().expect("cursor row available");
    let row = cursor.row;
    cursor.row += 1;
    let timestamp = i64_at(batch, "timestamp", row)?.ok_or("bookDepth timestamp is null")?;
    let percentage = i64_at(batch, "percentage", row)?;
    let depth = f64_at(batch, "depth", row)?;
    let notional = f64_at(batch, "notional", row)?;
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
            source_fields: source_fields(batch),
            source_values: source_values(batch, row),
        }),
        depth_state: None,
    }))
}
