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
    header_map, hmac_sha384, now_ms, symbols, Balance, Exchange, ExchangeError, ExchangeKind,
        MarketEvent, MarketTrade, OrderRequest, OrderResponse, OrderType, TradeSide,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::{interval, Interval};
use tokio_tungstenite::{
    connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream,
};

const PUBLIC_WS_URL: &str = "wss://api-pub.bitfinex.com/ws/2";
const PUBLIC_REST_URL: &str = "https://api-pub.bitfinex.com/v2";
const AUTH_REST_URL: &str = "https://api.bitfinex.com/v2";
const PING_INTERVAL_SECS: u64 = 30;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Bitfinex v2 exchange implementation.
pub struct BitfinexExchange {
    api_key: Option<String>,
    api_secret: Option<String>,
    symbol: String,       // venue-native, e.g. "tBTCUST"
    ws: Option<WsStream>, // public trades channel
    chan_id: Option<i64>, // channel id of the trades subscription
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
            ping: interval(Duration::from_secs(PING_INTERVAL_SECS)),
        }
    }

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
impl Exchange for BitfinexExchange {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Bitfinex
    }

    async fn connect(&mut self, ticker: &str) -> Result<(), ExchangeError> {
        self.symbol = symbols::bitfinex(ticker);

        let (mut ws, _resp) = connect_async(PUBLIC_WS_URL).await?;

        // Subscribe to the public trades channel. `hbt: true` asks the server
        // to emit `[chanId, "hb"]` heartbeats so we can detect a dead pipe.
        let sub = json!({
            "event": "subscribe",
            "channel": "trades",
            "symbol": self.symbol,
            "hbt": true,
        });
        ws.send(Message::Text(sub.to_string())).await?;

        // Wait for the subscription ack before handing the stream back.
        loop {
            let msg = ws.next().await.ok_or(ExchangeError::ConnectionClosed)??;
            if let Message::Text(text) = &msg {
                let v: serde_json::Value = serde_json::from_str(text)?;
                match v.get("event").and_then(|e| e.as_str()) {
                    Some("subscribed") => {
                        self.chan_id = v.get("chanId").and_then(|c| c.as_i64());
                        break;
                    }
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
        let expected_chan = self.chan_id;
        loop {
            tokio::select! {
                _ = self.ping.tick() => {
                    ws.send(Message::Text(r#"{"event":"ping"}"#.into())).await?;
                }
                msg = ws.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(events) = parse_trade_msg(&text, expected_chan)? {
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
        // Candle rows are [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]; sort=1 = oldest first.
        let url = format!(
            "{PUBLIC_REST_URL}/candles/trade:5m:{symbol}/hist?limit={bars}&sort=1"
        );
        let rows: Vec<Vec<serde_json::Value>> = reqwest::get(&url).await?.json().await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get(2).and_then(|c| c.as_f64()))
            .collect())
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
        let order_id_int: i64 = order_id.parse().map_err(|_| {
            ExchangeError::OrderRejected(format!("invalid order id {order_id:?}"))
        })?;
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
        side: if signed_amount > 0.0 { TradeSide::Buy } else { TradeSide::Sell },
        id,
    })
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
        assert!(parse_trade_msg(r#"[8,"te",[1,1000,0.1,100.0]]"#, Some(7))
            .unwrap()
            .is_none());
    }

    #[test]
    fn control_frames_are_ignored_or_raise() {
        assert!(parse_trade_msg(
            r#"{"event":"subscribed","channel":"trades","chanId":7}"#,
            Some(7)
        )
        .unwrap()
        .is_none());
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

