use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::core::feature::{FeatureRow, FeatureValue};
use crate::live_strategy::ComputeFeatures;

const BINANCE_FUTURES_KLINES: &str = "https://fapi.binance.com/fapi/v1/klines";
const BINANCE_FUTURES_AGG_TRADES: &str = "https://fapi.binance.com/fapi/v1/aggTrades";
const CANDLE_INTERVAL: &str = "1m";
const CANDLE_INTERVAL_MS: i64 = 60_000;
const API_PAGE_SIZE: usize = 1_000;

#[derive(Default)]
pub struct HistorySeedGate {
    warmed: bool,
    discard_seed: bool,
}

impl HistorySeedGate {
    /// Consume history-control and history-data frames; return true for live
    /// frames that should proceed to telemetry, decisions, and execution.
    pub fn consume(
        &mut self,
        row: &FeatureRow,
        computer: &mut dyn ComputeFeatures,
    ) -> Result<bool, String> {
        if row.get("warmup_start") == Some(1.0) {
            self.discard_seed = self.warmed;
            if !self.discard_seed {
                computer.reset();
            }
            return Ok(false);
        }
        if row.get("warmup_end") == Some(1.0) {
            if !self.discard_seed {
                self.warmed = true;
            }
            self.discard_seed = false;
            return Ok(false);
        }
        if row.get("warmup") == Some(1.0) {
            if !self.discard_seed {
                computer.compute(row)?;
            }
            return Ok(false);
        }
        Ok(true)
    }
}

/// Retrieve closed one-minute bars and the aggregate trades in the same window.
pub fn fetch_market_history(symbol: &str, lookback_secs: u64) -> Result<Vec<FeatureRow>, String> {
    if lookback_secs == 0 {
        return Err("strategy warm-up lookback must be greater than zero".into());
    }
    let symbol = symbol.to_ascii_uppercase();
    if symbol.is_empty() || !symbol.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
        return Err(format!("invalid Binance warm-up symbol '{symbol}'"));
    }

    let now_ms: i64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before the Unix epoch: {error}"))?
        .as_millis()
        .try_into()
        .map_err(|_| "system time is outside the Binance timestamp range".to_string())?;
    let start_ms = now_ms
        .checked_sub(
            i64::try_from(lookback_secs)
                .ok()
                .and_then(|seconds| seconds.checked_mul(1_000))
                .ok_or_else(|| "strategy warm-up lookback is too large".to_string())?,
        )
        .ok_or_else(|| "strategy warm-up start time is out of range".to_string())?;
    let closed_candle_end_ms = now_ms.div_euclid(CANDLE_INTERVAL_MS) * CANDLE_INTERVAL_MS - 1;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| format!("build Binance history client: {error}"))?;

    let candles = fetch_candles(&client, &symbol, start_ms, closed_candle_end_ms)?;
    let trades = fetch_aggregate_trades(&client, &symbol, start_ms, now_ms)?;
    if candles.is_empty() {
        return Err(format!(
            "Binance returned no closed {CANDLE_INTERVAL} candles for {symbol} warm-up"
        ));
    }
    if trades.is_empty() {
        return Err(format!(
            "Binance returned no aggregate trades for {symbol} warm-up"
        ));
    }

    let candle_count = candles.len();
    let trade_count = trades.len();
    let mut rows = candles.into_iter().chain(trades).collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        let left_time = left.get("event_time_ms").unwrap_or_default();
        let right_time = right.get("event_time_ms").unwrap_or_default();
        left_time.total_cmp(&right_time).then_with(|| {
            let left_is_candle = left.contains("bar_close_price");
            let right_is_candle = right.contains("bar_close_price");
            left_is_candle.cmp(&right_is_candle)
        })
    });
    eprintln!(
        "[warmup] fetched {candle_count} closed {CANDLE_INTERVAL} bars and {trade_count} aggregate trades for {symbol} over {lookback_secs}s"
    );
    Ok(rows)
}

fn fetch_candles(
    client: &reqwest::blocking::Client,
    symbol: &str,
    start_ms: i64,
    end_ms: i64,
) -> Result<Vec<FeatureRow>, String> {
    let mut rows = Vec::new();
    let mut cursor_ms = start_ms;
    loop {
        let response: Value = client
            .get(BINANCE_FUTURES_KLINES)
            .query(&[
                ("symbol", symbol.to_string()),
                ("interval", CANDLE_INTERVAL.to_string()),
                ("startTime", cursor_ms.to_string()),
                ("endTime", end_ms.to_string()),
                ("limit", API_PAGE_SIZE.to_string()),
            ])
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|error| format!("Binance {CANDLE_INTERVAL} candle request failed: {error}"))?
            .json()
            .map_err(|error| format!("decode Binance {CANDLE_INTERVAL} candles: {error}"))?;
        let candles = response
            .as_array()
            .ok_or_else(|| "Binance klines response is not an array".to_string())?;
        if candles.is_empty() {
            break;
        }
        let page = candles
            .iter()
            .map(parse_candle)
            .collect::<Result<Vec<_>, _>>()?;
        let last_open_ms = page
            .last()
            .and_then(|row| row.get("bar_open_time_ms"))
            .ok_or_else(|| "Binance kline page has no open time".to_string())?
            as i64;
        rows.extend(page);
        if candles.len() < API_PAGE_SIZE {
            break;
        }
        let next_cursor_ms = last_open_ms
            .checked_add(CANDLE_INTERVAL_MS)
            .ok_or_else(|| "Binance kline timestamp overflow".to_string())?;
        if next_cursor_ms <= cursor_ms {
            return Err("Binance kline pagination did not advance".into());
        }
        cursor_ms = next_cursor_ms;
    }
    Ok(rows)
}

fn parse_candle(value: &Value) -> Result<FeatureRow, String> {
    let candle = value
        .as_array()
        .ok_or_else(|| "Binance kline entry is not an array".to_string())?;
    let number = |index: usize, name: &str| -> Result<f64, String> {
        candle
            .get(index)
            .and_then(parse_number)
            .ok_or_else(|| format!("Binance kline has invalid {name}"))
    };
    let close_time_ms = number(6, "close time")? as i64;
    let mut row = FeatureRow::new();
    row.insert("event_time_ms", close_time_ms as f64);
    row.insert("bar_open_time_ms", number(0, "open time")?);
    row.insert("bar_open_price", number(1, "open price")?);
    row.insert("bar_high_price", number(2, "high price")?);
    row.insert("bar_low_price", number(3, "low price")?);
    row.insert("bar_close_price", number(4, "close price")?);
    row.insert("bar_volume", number(5, "volume")?);
    row.insert("bar_close_time_ms", close_time_ms as f64);
    row.insert("bar_timeframe_secs", CANDLE_INTERVAL_MS as f64 / 1_000.0);
    Ok(row)
}

fn fetch_aggregate_trades(
    client: &reqwest::blocking::Client,
    symbol: &str,
    start_ms: i64,
    end_ms: i64,
) -> Result<Vec<FeatureRow>, String> {
    let mut rows = Vec::new();
    let mut next_id = None::<u64>;
    loop {
        let mut query = vec![
            ("symbol", symbol.to_string()),
            ("limit", API_PAGE_SIZE.to_string()),
        ];
        if let Some(id) = next_id {
            query.push(("fromId", id.to_string()));
        } else {
            query.push(("startTime", start_ms.to_string()));
        }
        let response: Value = client
            .get(BINANCE_FUTURES_AGG_TRADES)
            .query(&query)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|error| format!("Binance aggregate-trade request failed: {error}"))?
            .json()
            .map_err(|error| format!("decode Binance aggregate trades: {error}"))?;
        let page = response
            .as_array()
            .ok_or_else(|| "Binance aggregate-trades response is not an array".to_string())?;
        if page.is_empty() {
            break;
        }

        let mut reached_end = false;
        let mut last_id = None;
        for trade in page {
            let event_ms = value_i64(trade.get("T")).ok_or_else(|| {
                "Binance aggregate trade has invalid transaction time".to_string()
            })?;
            let trade_id = value_u64(trade.get("a"))
                .ok_or_else(|| "Binance aggregate trade has invalid id".to_string())?;
            last_id = Some(trade_id);
            if event_ms < start_ms {
                continue;
            }
            if event_ms > end_ms {
                reached_end = true;
                break;
            }
            rows.push(parse_trade(trade, event_ms, trade_id)?);
        }
        if reached_end || page.len() < API_PAGE_SIZE {
            break;
        }
        let Some(last_id) = last_id else {
            return Err("Binance aggregate-trade page made no progress".into());
        };
        next_id = Some(
            last_id
                .checked_add(1)
                .ok_or_else(|| "Binance aggregate-trade id overflow".to_string())?,
        );
    }
    Ok(rows)
}

fn parse_trade(value: &Value, event_ms: i64, trade_id: u64) -> Result<FeatureRow, String> {
    let number = |key: &str| {
        value
            .get(key)
            .and_then(parse_number)
            .ok_or_else(|| format!("Binance aggregate trade has invalid {key}"))
    };
    let mut row = FeatureRow::new();
    row.insert("event_time_ms", event_ms as f64);
    row.insert("price", number("p")?);
    row.insert("amount", number("q")?);
    row.insert(
        "is_buyer_maker",
        if value.get("m").and_then(Value::as_bool).unwrap_or(false) {
            1.0
        } else {
            0.0
        },
    );
    row.insert("aggregate_trade_id", trade_id as f64);
    Ok(row)
}

fn parse_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(number) => number.parse().ok(),
        _ => None,
    }
}

fn value_i64(value: Option<&Value>) -> Option<i64> {
    match value? {
        Value::Number(number) => number.as_i64(),
        Value::String(number) => number.parse().ok(),
        _ => None,
    }
}

fn value_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(number) => number.as_u64(),
        Value::String(number) => number.parse().ok(),
        _ => None,
    }
}

pub fn encode_history_frame(row: Option<&FeatureRow>, marker: &str) -> Result<Vec<u8>, String> {
    let payload = if let Some(row) = row {
        let mut payload = String::new();
        for (name, value) in row.iter() {
            if let FeatureValue::Number(value) = value
                && value.is_finite()
            {
                payload.push_str(&format!("{name}:{value}\n"));
            }
        }
        payload.push_str("warmup:1\n");
        payload
    } else {
        format!("{marker}:1\n")
    };
    let payload = payload.into_bytes();
    let frame_len = u32::try_from(payload.len())
        .map_err(|_| "historical warm-up frame is too large".to_string())?;
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&frame_len.to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_strategy::example_strategy::compute_feature::ExampleFeatures;

    #[test]
    fn parses_historical_candles_and_aggregate_trades() {
        let candle = serde_json::json!([60_000, "10.0", "11.0", "9.0", "10.5", "25.0", 119_999]);
        let row = parse_candle(&candle).unwrap();
        assert_eq!(row.get("event_time_ms"), Some(119_999.0));
        assert_eq!(row.get("bar_close_price"), Some(10.5));
        assert_eq!(row.get("bar_volume"), Some(25.0));

        let trade = serde_json::json!({
            "a": 123,
            "p": "10.4",
            "q": "0.5",
            "T": 119_500,
            "m": true
        });
        let row = parse_trade(&trade, 119_500, 123).unwrap();
        assert_eq!(row.get("event_time_ms"), Some(119_500.0));
        assert_eq!(row.get("price"), Some(10.4));
        assert_eq!(row.get("amount"), Some(0.5));
        assert_eq!(row.get("is_buyer_maker"), Some(1.0));
    }

    #[test]
    fn warmup_frames_seed_generic_feature_state_once() {
        let mut features = ExampleFeatures::new();
        let mut gate = HistorySeedGate::default();
        let start = crate::live_strategy::ofmr::socket::decode_frame(b"warmup_start:1\n").unwrap();
        assert!(!gate.consume(&start, &mut features).unwrap());
        for price in 1..=5 {
            let mut row = FeatureRow::new();
            row.insert("event_time_ms", price as f64);
            row.insert("price", price as f64);
            row.insert("warmup", 1.0);
            assert!(!gate.consume(&row, &mut features).unwrap());
        }
        let end = crate::live_strategy::ofmr::socket::decode_frame(b"warmup_end:1\n").unwrap();
        assert!(!gate.consume(&end, &mut features).unwrap());

        let mut replay_start = FeatureRow::new();
        replay_start.insert("warmup_start", 1.0);
        gate.consume(&replay_start, &mut features).unwrap();
        let mut replay_row = FeatureRow::new();
        replay_row.insert("event_time_ms", 100.0);
        replay_row.insert("price", 100.0);
        replay_row.insert("warmup", 1.0);
        gate.consume(&replay_row, &mut features).unwrap();
        let mut replay_end = FeatureRow::new();
        replay_end.insert("warmup_end", 1.0);
        gate.consume(&replay_end, &mut features).unwrap();

        let mut live = FeatureRow::new();
        live.insert("price", 6.0);
        let computed = features.compute(&live).unwrap();
        assert_eq!(computed.get("dz"), Some(2.0));
    }
}
