//! Bitfinex v2 — direct WebSocket + REST client (no SDK).
//!
//! References:
//!   - Public WS (trades):    https://docs.bitfinex.com/reference/ws-public-trades
//!   - Auth (REST signing):   https://docs.bitfinex.com/docs/rest-auth
//!   - Submit order:          https://docs.bitfinex.com/reference/rest-auth-submit-order
//!   - Wallets:               https://docs.bitfinex.com/reference/rest-auth-wallets
//!
//! Authentication (v2 REST): sign the string
//!   `/api/v2/{endpoint}{nonce}{json_body}`
//! with HMAC-SHA384 and send `bfx-nonce`, `bfx-apikey`, `bfx-signature`.
//! The signed body string MUST be byte-identical to the request body bytes.
//!
//! Public market data is streamed over the trading-pair trades channel
//! (`wss://api-pub.bitfinex.com/ws/2`); trade frames are
//! `[chanId, "te"|"tu", [ID, MTS, AMOUNT, PRICE]]` where a positive AMOUNT is a
//! buy and a negative AMOUNT is a sell.

use super::{
    Balance, Broker, ExchangeError, ExchangeKind, Feed, MarketBookTop, MarketEvent, MarketTrade,
    OrderRequest, OrderResponse, OrderType, TradeSide, header_map, hmac_sha384, now_ms, symbols,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::{Interval, interval};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};

const PUBLIC_WS_URL: &str = "wss://api-pub.bitfinex.com/ws/2";
const PUBLIC_REST_URL: &str = "https://api-pub.bitfinex.com/v2";
const AUTH_REST_URL: &str = "https://api.bitfinex.com/v2";
const PING_INTERVAL_SECS: u64 = 30;
const MAX_BARS: u32 = 200;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Bitfinex v2 exchange implementation.
pub struct BitfinexExchange {
    api_key: Option<String>,
    api_secret: Option<String>,
    symbol: String,
    ws: Option<WsStream>,
    chan_id: Option<i64>,         // trades channel
    book_chan_id: Option<i64>,    // book channel (new)
    last_bid: Option<(f64, f64)>, // (price, amount) — new
    last_ask: Option<(f64, f64)>, // (price, amount) — new
    ping: Interval,
}

impl BitfinexExchange {
    pub fn new(api_key: Option<String>, api_secret: Option<String>) -> Self {
        Self {
            api_key,
            api_secret,
            symbol: String::new(),
            ws: None,
            chan_id: None,
            book_chan_id: None,
            last_bid: None,
            last_ask: None,
            ping: interval(Duration::from_secs(PING_INTERVAL_SECS)),
        }
    }
    // credentials() unchanged
    fn credentials(&self) -> Result<(&str, &str), ExchangeError> {
        let key = self
            .api_key
            .as_deref()
            .ok_or_else(|| ExchangeError::Auth("BFX_API_KEY is not set".into()))?;
        let secret = self
            .api_secret
            .as_deref()
            .ok_or_else(|| ExchangeError::Auth("BFX_API_SECRET is not set".into()))?;
        Ok((key, secret))
    }
}

#[async_trait::async_trait]
impl Feed for BitfinexExchange {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Bitfinex
    }

    async fn connect(&mut self, ticker: &str) -> Result<(), ExchangeError> {
        self.symbol = symbols::bitfinex(ticker);

        let (mut ws, _resp) = connect_async(PUBLIC_WS_URL).await?;

        let sub_trades = json!({
            "event": "subscribe",
            "channel": "trades",
            "symbol": self.symbol,
            "hbt": true,
        });
        ws.send(Message::Text(sub_trades.to_string())).await?;

        // Top-of-book only: prec=P0 (raw), freq=F0 (realtime), len="1"
        // (single level per side — best bid / best ask only).
        let sub_book = json!({
            "event": "subscribe",
            "channel": "book",
            "symbol": self.symbol,
            "prec": "P0",
            "freq": "F0",
            "len": "1",
            "hbt": true,
        });
        ws.send(Message::Text(sub_book.to_string())).await?;

        // Wait for both subscription acks before handing the stream back.
        let mut trades_acked = false;
        let mut book_acked = false;
        while !trades_acked || !book_acked {
            let msg = ws.next().await.ok_or(ExchangeError::ConnectionClosed)??;
            if let Message::Text(text) = &msg {
                let v: serde_json::Value = serde_json::from_str(text)?;
                match v.get("event").and_then(|e| e.as_str()) {
                    Some("subscribed") => match v.get("channel").and_then(|c| c.as_str()) {
                        Some("trades") => {
                            self.chan_id = v.get("chanId").and_then(|c| c.as_i64());
                            trades_acked = true;
                        }
                        Some("book") => {
                            self.book_chan_id = v.get("chanId").and_then(|c| c.as_i64());
                            book_acked = true;
                        }
                        _ => {}
                    },
                    Some("error") => return Err(ExchangeError::SubscriptionRejected(text.clone())),
                    _ => {} // info / conf / pong frames
                }
            }
        }

        self.ws = Some(ws);
        Ok(())
    }

    async fn next_events(&mut self) -> Result<Vec<MarketEvent>, ExchangeError> {
        let ws = self.ws.as_mut().expect("connect() must be called first");
        let trades_chan = self.chan_id;
        let book_chan = self.book_chan_id;
        loop {
            tokio::select! {
                _ = self.ping.tick() => {
                    ws.send(Message::Text(r#"{"event":"ping"}"#.into())).await?;
                }
                msg = ws.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(events) = parse_trade_msg(&text, trades_chan)? {
                                return Ok(events);
                            }
                            if let Some(events) = parse_book_msg(
                                &text, book_chan, &mut self.last_bid, &mut self.last_ask,
                            )? {
                                return Ok(events);
                            }
                        }
                        Some(Ok(Message::Ping(p))) => {
                            ws.send(Message::Pong(p)).await?;
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            return Err(ExchangeError::ConnectionClosed);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    async fn historical_closes(&self, ticker: &str, bars: u32) -> Result<Vec<f64>, ExchangeError> {
        let symbol = symbols::bitfinex(ticker);
        let limit = bars.min(MAX_BARS).max(1);
        // Candle rows are [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME].
        //
        // CAUTION: Bitfinex `/hist` PAGES through history in the direction of
        // `sort`. With `sort=1` and no explicit `start`, "ascending" means it
        // returns the OLDEST candles on record (2019, BTC ≈ $3,870) — seeding
        // the feature rings with years-stale prices and pinning Wilder RSI(2)
        // near 100 (the seam diff into today's price is a colossal fake gain),
        // which permanently blocks the rsi_2 < 30 hunting zone. `sort=-1`
        // pages newest-first; reverse locally into the oldest-first order
        // `seed_macro_closes` expects.
        let url = format!("{PUBLIC_REST_URL}/candles/trade:5m:{symbol}/hist?limit={limit}&sort=-1");
        let rows: Vec<Vec<serde_json::Value>> = reqwest::get(&url).await?.json().await?;
        let mut closes: Vec<f64> = rows
            .iter()
            .filter_map(|r| r.get(2).and_then(|c| c.as_f64()))
            .collect();
        // Newest-first → oldest-first for seed warming.
        closes.reverse();
        Ok(closes)
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        self.ws = None;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Broker for BitfinexExchange {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Bitfinex
    }

    async fn place_order(&self, req: &OrderRequest) -> Result<OrderResponse, ExchangeError> {
        let (key, secret) = self.credentials()?;
        let endpoint = "auth/w/order/submit";

        let mut body = json!({
            "type": if req.order_type == OrderType::Market {
                "EXCHANGE MARKET"
            } else {
                "EXCHANGE LIMIT"
            },
            "symbol": req.symbol,
            "amount": req.amount.to_string(),
        });
        if req.order_type == OrderType::Limit {
            body["price"] = json!(req.price.to_string());
        }
        let body_str = serde_json::to_string(&body)?;
        let headers = self.signed_headers(endpoint, key, secret, &body_str);

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{AUTH_REST_URL}/{endpoint}"))
            .header("Content-Type", "application/json")
            .headers(header_map(&headers))
            .body(body_str)
            .send()
            .await?;
        let text = resp.text().await?;

        // Response: [MTS, "on-req", msg_id, null, [[order]], code, status, text]
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let status = v
            .get(6)
            .and_then(|s| s.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        if status != "SUCCESS" {
            return Err(ExchangeError::OrderRejected(text));
        }
        let order = v
            .get(4)
            .and_then(|d| d.as_array())
            .and_then(|d| d.first())
            .and_then(|o| o.as_array());
        let exchange_order_id = order
            .and_then(|o| o.first())
            .and_then(|x| x.as_i64())
            .map(|id| id.to_string())
            .unwrap_or_default();
        let order_status = order
            .and_then(|o| o.get(13))
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let price = order
            .and_then(|o| o.get(16))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        Ok(OrderResponse {
            exchange_order_id,
            status: order_status,
            avg_price: price,
            filled_amount: 0.0,
        })
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExchangeError> {
        let (key, secret) = self.credentials()?;
        let endpoint = "auth/w/order/cancel";
        let order_id_int: i64 = order_id
            .parse()
            .map_err(|_| ExchangeError::OrderRejected(format!("invalid order id {order_id:?}")))?;
        let body = json!({ "id": order_id_int });
        let body_str = serde_json::to_string(&body)?;
        let headers = self.signed_headers(endpoint, key, secret, &body_str);

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{AUTH_REST_URL}/{endpoint}"))
            .header("Content-Type", "application/json")
            .headers(header_map(&headers))
            .body(body_str)
            .send()
            .await?;
        let text = resp.text().await?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let status = v
            .get(6)
            .and_then(|s| s.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        if status != "SUCCESS" {
            return Err(ExchangeError::OrderRejected(text));
        }
        Ok(())
    }

    async fn balances(&self) -> Result<Vec<Balance>, ExchangeError> {
        let (key, secret) = self.credentials()?;
        let endpoint = "auth/r/wallets";
        let body_str = "{}";
        let headers = self.signed_headers(endpoint, key, secret, body_str);

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{AUTH_REST_URL}/{endpoint}"))
            .header("Content-Type", "application/json")
            .headers(header_map(&headers))
            .body(body_str)
            .send()
            .await?;
        let text = resp.text().await?;
        // Response rows: [type, currency, balance, unsettled_interest, available, ...]
        let rows: Vec<Vec<serde_json::Value>> = serde_json::from_str(&text)?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                if row.len() < 5 {
                    return None;
                }
                let currency = row[1].as_str()?.to_string();
                Some(Balance {
                    currency,
                    total: row[2].as_f64().unwrap_or(0.0),
                    available: row[4].as_f64().unwrap_or(0.0),
                })
            })
            .collect())
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        self.ws = None;
        Ok(())
    }
}

impl BitfinexExchange {
    /// Compute the v2 REST auth header triple. `body` MUST be the exact bytes
    /// that will be sent as the request body.
    fn signed_headers(
        &self,
        endpoint: &str,
        api_key: &str,
        api_secret: &str,
        body: &str,
    ) -> Vec<(&'static str, String)> {
        let nonce = now_ms().to_string();
        let message = format!("/api/v2/{endpoint}{nonce}{body}");
        let signature = hmac_sha384(api_secret, &message);
        vec![
            ("bfx-nonce", nonce),
            ("bfx-apikey", api_key.to_string()),
            ("bfx-signature", signature),
        ]
    }
}

/// Parse a raw Bitfinex WS frame into normalized trade events (or `None` for
/// control/heartbeat frames). Trade frames:
///   [chanId, [ [id,mts,amt,px], ... ]]  initial snapshot
///   [chanId, "te", [id,mts,amt,px]]     trade executed (provisional id)
///   [chanId, "tu", [id,mts,amt,px]]     trade updated (confirmed id)
fn parse_trade_msg(
    text: &str,
    expected_chan: Option<i64>,
) -> Result<Option<Vec<MarketEvent>>, ExchangeError> {
    let v: serde_json::Value = serde_json::from_str(text)?;

    // Control frames are JSON objects.
    if let Some(obj) = v.as_object() {
        if let Some(event) = obj.get("event").and_then(|e| e.as_str()) {
            return match event {
                "error" => Err(ExchangeError::SubscriptionRejected(text.to_string())),
                "info" => {
                    // 20051/20060 = maintenance window / reconnect required.
                    if let Some(code) = obj.get("code").and_then(|c| c.as_i64()) {
                        if code == 20051 || code == 20060 {
                            return Err(ExchangeError::ConnectionClosed);
                        }
                    }
                    Ok(None)
                }
                _ => Ok(None), // "subscribed", "conf", "pong", ...
            };
        }
        return Ok(None);
    }

    let arr = v
        .as_array()
        .ok_or_else(|| ExchangeError::Parse("expected JSON array".into()))?;
    if arr.len() < 2 {
        return Ok(None);
    }
    let chan = arr[0].as_i64().unwrap_or(-1);
    if let Some(expected) = expected_chan {
        if chan != expected {
            return Ok(None); // a different channel's frame
        }
    }

    match &arr[1] {
        serde_json::Value::String(tag) if tag == "hb" => Ok(None),
        serde_json::Value::String(tag) if tag == "te" || tag == "tu" => {
            let row = arr
                .get(2)
                .ok_or_else(|| ExchangeError::Parse("trade row missing".into()))?;
            Ok(Some(vec![MarketEvent::Trade(parse_trade(row)?)]))
        }
        serde_json::Value::Array(rows) => {
            let mut out = Vec::new();
            for row in rows {
                if let Ok(trade) = parse_trade(row) {
                    out.push(MarketEvent::Trade(trade));
                }
            }
            Ok(Some(out))
        }
        _ => Ok(None),
    }
}

/// Parse one `[id, mts, amount, price]` trade row.
fn parse_trade(row: &serde_json::Value) -> Result<MarketTrade, ExchangeError> {
    let a = row
        .as_array()
        .ok_or_else(|| ExchangeError::Parse("trade row is not an array".into()))?;
    if a.len() < 4 {
        return Err(ExchangeError::Parse("trade row too short".into()));
    }
    let id = a[0].as_i64().unwrap_or(0).to_string();
    let timestamp = a[1].as_i64().unwrap_or(0);
    let signed_amount = a[2].as_f64().unwrap_or(0.0);
    let price = a[3].as_f64().unwrap_or(0.0);
    Ok(MarketTrade {
        timestamp,
        price,
        amount: signed_amount.abs(),
        side: if signed_amount > 0.0 {
            TradeSide::Buy
        } else {
            TradeSide::Sell
        },
        id,
    })
}
/// Parse a raw Bitfinex WS book-channel frame into a top-of-book event (or
/// `None` for control/heartbeat frames or frames on a channel we don't own).
/// With `len="1"`, the book carries exactly one level per side:
///   snapshot: [chanId, [[price, count, amount], [price, count, amount]]]
///   update:   [chanId, [price, count, amount]]
/// `amount > 0` = bid level, `amount < 0` = ask level (magnitude = size);
/// `count == 0` deletes that side's current level.
///
/// Bitfinex book rows carry no timestamp, unlike trades — the emitted
/// event's `timestamp` is therefore local receive time (`now_ms()`), not an
/// exchange-stamped time. Downstream clock-skew diagnostics computed from a
/// book-top event will read ~0 by construction; only trade-derived skew is
/// meaningful for that purpose.
fn parse_book_msg(
    text: &str,
    expected_chan: Option<i64>,
    last_bid: &mut Option<(f64, f64)>,
    last_ask: &mut Option<(f64, f64)>,
) -> Result<Option<Vec<MarketEvent>>, ExchangeError> {
    let v: serde_json::Value = serde_json::from_str(text)?;

    // Control frames are handled during connect()'s ack loop; ignore here.
    if v.as_object().is_some() {
        return Ok(None);
    }

    let arr = v
        .as_array()
        .ok_or_else(|| ExchangeError::Parse("expected JSON array".into()))?;
    if arr.len() < 2 {
        return Ok(None);
    }
    let chan = arr[0].as_i64().unwrap_or(-1);
    let Some(expected) = expected_chan else {
        return Ok(None);
    };
    if chan != expected {
        return Ok(None); // trades channel frame, or not yet subscribed
    }

    match &arr[1] {
        serde_json::Value::String(tag) if tag == "hb" => Ok(None),
        // Snapshot: array of [price, count, amount] rows.
        serde_json::Value::Array(rows) if rows.first().map_or(false, |r| r.is_array()) => {
            for row in rows {
                apply_book_row(row, last_bid, last_ask);
            }
            Ok(emit_book_top(last_bid, last_ask))
        }
        // Update: a single [price, count, amount] row.
        serde_json::Value::Array(row) if row.len() == 3 => {
            apply_book_row(&serde_json::Value::Array(row.clone()), last_bid, last_ask);
            Ok(emit_book_top(last_bid, last_ask))
        }
        _ => Ok(None),
    }
}

/// Apply one `[price, count, amount]` row to the running best-bid/best-ask
/// state. `count == 0` deletes that side's current level.
fn apply_book_row(
    row: &serde_json::Value,
    last_bid: &mut Option<(f64, f64)>,
    last_ask: &mut Option<(f64, f64)>,
) {
    let Some(a) = row.as_array() else { return };
    if a.len() < 3 {
        return;
    }
    let price = a[0].as_f64().unwrap_or(0.0);
    let count = a[1].as_i64().unwrap_or(0);
    let amount = a[2].as_f64().unwrap_or(0.0);

    if amount > 0.0 {
        *last_bid = if count == 0 {
            None
        } else {
            Some((price, amount))
        };
    } else if amount < 0.0 {
        *last_ask = if count == 0 {
            None
        } else {
            Some((price, amount.abs()))
        };
    }
}

/// Build a MarketEvent::BookTop once both sides are known.
fn emit_book_top(
    last_bid: &Option<(f64, f64)>,
    last_ask: &Option<(f64, f64)>,
) -> Option<Vec<MarketEvent>> {
    let (bid_price, bid_amount) = (*last_bid)?;
    let (ask_price, ask_amount) = (*last_ask)?;
    Some(vec![MarketEvent::BookTop(MarketBookTop {
        timestamp: now_ms(),
        bid_price,
        bid_amount,
        ask_price,
        ask_amount,
    })])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_mapping() {
        assert_eq!(symbols::bitfinex("btcusdt"), "tBTCUST");
        assert_eq!(symbols::bitfinex("btcusd"), "tBTCUSD");
        assert_eq!(symbols::bitfinex("ethusdt"), "tETHUST");
        assert_eq!(symbols::bitfinex("SOLUSDT"), "tSOLUST");
    }

    #[test]
    fn parses_te_frame() {
        let frame = r#"[12345,"te",[401597395,1574694478808,0.005,7245.3]]"#;
        let events = parse_trade_msg(frame, Some(12345)).unwrap().unwrap();
        match &events[0] {
            MarketEvent::Trade(t) => {
                assert_eq!(t.id, "401597395");
                assert_eq!(t.timestamp, 1574694478808);
                assert_eq!(t.amount, 0.005);
                assert_eq!(t.price, 7245.3);
                assert_eq!(t.side, TradeSide::Buy);
            }
            other => panic!("expected trade, got {other:?}"),
        }
    }

    #[test]
    fn negative_amount_is_sell() {
        let frame = r#"[1,"te",[9,1000,-0.25,100.0]]"#;
        let events = parse_trade_msg(frame, Some(1)).unwrap().unwrap();
        match &events[0] {
            MarketEvent::Trade(t) => {
                assert_eq!(t.amount, 0.25);
                assert_eq!(t.side, TradeSide::Sell);
            }
            other => panic!("expected trade, got {other:?}"),
        }
    }

    #[test]
    fn parses_snapshot() {
        let frame = r#"[7,[[1,1000,0.1,100.0],[2,1001,-0.2,99.5]]]"#;
        let events = parse_trade_msg(frame, Some(7)).unwrap().unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn ignores_heartbeats_and_foreign_channels() {
        assert!(parse_trade_msg(r#"[7,"hb"]"#, Some(7)).unwrap().is_none());
        assert!(
            parse_trade_msg(r#"[8,"te",[1,1000,0.1,100.0]]"#, Some(7))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn control_frames_are_ignored_or_raise() {
        assert!(
            parse_trade_msg(
                r#"{"event":"subscribed","channel":"trades","chanId":7}"#,
                Some(7)
            )
            .unwrap()
            .is_none()
        );
        assert!(parse_trade_msg(r#"{"event":"error","msg":"bad"}"#, Some(7)).is_err());
    }

    #[test]
    fn auth_message_layout() {
        // Message = /api/v2/{endpoint}{nonce}{json_body}; HMAC-SHA384 → 96 hex chars.
        let msg = "/api/v2/auth/w/order/submit1617817390826{\"type\":\"EXCHANGE LIMIT\"}";
        let sig = hmac_sha384("s3cr3t", msg);
        assert_eq!(sig.len(), 96);
    }
}
