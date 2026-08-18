//! Minimal Bybit v5 public WebSocket client for trades and top-of-book (L1) data.
//!
//! Docs referenced:
//!   - Connect / ping-pong:      https://bybit-exchange.github.io/docs/v5/ws/connect
//!   - Public trade stream:      https://bybit-exchange.github.io/docs/v5/websocket/public/trade
//!   - Public orderbook stream:  https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook
//!
//! Notes:
//!   - Uses the `linear` (USDT perpetual) public stream. Swap `PUBLIC_WS_URL` for
//!     "wss://stream.bybit.com/v5/public/spot" if you want spot instead.
//!   - Subscribes to `orderbook.1.{symbol}` (depth = 1). At depth 1, Bybit only
//!     ever sends "snapshot" messages (a fresh one is re-pushed every 3s even if
//!     nothing changed), so there's no local book to reconstruct from deltas --
//!     each message already *is* the current best bid/ask.
//!   - A ping is sent every 20s per Bybit's recommendation to keep the connection
//!     alive; incoming server Pings are answered with Pongs automatically.

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use tokio::net::TcpStream;
use tokio::time::{interval, Duration, Interval};
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};

const PUBLIC_WS_URL: &str = "wss://stream.bybit.com/v5/public/linear";
const REST_BASE_URL: &str = "https://api.bybit.com";
const PING_INTERVAL_SECS: u64 = 20;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;


#[derive(Debug, thiserror::Error)]
pub enum BybitError {
    #[error("websocket transport error: {0}")]
    Transport(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("failed to parse websocket payload: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("subscription rejected: {0}")]
    SubscriptionRejected(String),
    #[error("connection closed by server")]
    ConnectionClosed,
    #[error("http request error: {0}")]
    Http(#[from] reqwest::Error),   // <-- new
}

/// A single public trade print.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitTrade {
    pub timestamp: i64,
    pub price: f64,
    pub amount: f64,
    pub side: String,
    pub id: String,
}

impl fmt::Display for BybitTrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{},{},{},{},{}",
            self.timestamp, self.price, self.amount, self.side, self.id
        )
    }
}

/// Best bid / ask snapshot (level-1 order book).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitOrderBookTop {
    pub timestamp: i64,
    pub bid_price: f64,
    pub bid_amount: f64,
    pub ask_price: f64,
    pub ask_amount: f64,
}

impl fmt::Display for BybitOrderBookTop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{},{},{},{},{}",
            self.timestamp, self.bid_price, self.bid_amount, self.ask_price, self.ask_amount
        )
    }
}

// ---- raw wire formats -------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawEnvelope<'a> {
    topic: Option<&'a str>,
    success: Option<bool>,
    #[serde(rename = "ret_msg")]
    ret_msg: Option<&'a str>,
    op: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct RawTradeMsg {
    data: Vec<RawTradeEntry>,
}

#[derive(Debug, Deserialize)]
struct RawTradeEntry {
    #[serde(rename = "T")]
    t: i64,
    #[serde(rename = "S")]
    side: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "i")]
    id: String,
}

#[derive(Debug, Deserialize)]
struct RawBookMsg {
    ts: i64,
    data: RawBookData,
}

#[derive(Debug, Deserialize)]
struct RawBookData {
    b: Vec<[String; 2]>,
    a: Vec<[String; 2]>,
}

// ---- connection helpers ------------------------------------------------------

async fn connect_and_subscribe(topic: &str) -> Result<WsStream, BybitError> {
    let (mut ws, _resp) = connect_async(PUBLIC_WS_URL).await?;

    let sub = json!({ "op": "subscribe", "args": [topic] });
    ws.send(Message::Text(sub.to_string())).await?;

    // Wait for the subscription ack before handing the stream back.
    loop {
        let msg = ws.next().await.ok_or(BybitError::ConnectionClosed)??;
        if let Message::Text(text) = &msg {
            if let Ok(env) = serde_json::from_str::<RawEnvelope>(text) {
                if env.op == Some("subscribe") {
                    if env.success == Some(true) {
                        break;
                    }
                    return Err(BybitError::SubscriptionRejected(
                        env.ret_msg.unwrap_or("unknown error").to_string(),
                    ));
                }
            }
        }
    }

    Ok(ws)
}

fn make_ping_interval() -> Interval {
    interval(Duration::from_secs(PING_INTERVAL_SECS))
}

async fn send_ping(ws: &mut WsStream) -> Result<(), BybitError> {
    ws.send(Message::Text(json!({ "op": "ping" }).to_string()))
        .await?;
    Ok(())
}

// ---- trade stream --------------------------------------------------------------

pub struct TradeWatcher {
    ws: WsStream,
    ping_timer: Interval,
}

impl TradeWatcher {
    pub async fn connect(symbol: &str) -> Result<Self, BybitError> {
        let topic = format!("publicTrade.{symbol}");
        let ws = connect_and_subscribe(&topic).await?;
        Ok(Self {
            ws,
            ping_timer: make_ping_interval(),
        })
    }

    /// Waits for the next websocket message and returns the batch of trades it
    /// contained (Bybit may push more than one print per message). Loops past
    /// non-trade frames (acks, pongs) internally.
    pub async fn next_batch(&mut self) -> Result<Vec<BybitTrade>, BybitError> {
        loop {
            tokio::select! {
                _ = self.ping_timer.tick() => {
                    send_ping(&mut self.ws).await?;
                }
                msg = self.ws.next() => {
                    let msg = msg.ok_or(BybitError::ConnectionClosed)??;
                    match msg {
                        Message::Text(text) => {
                            if let Some(trades) = parse_trade_msg(&text)? {
                                return Ok(trades);
                            }
                        }
                        Message::Ping(payload) => {
                            self.ws.send(Message::Pong(payload)).await?;
                        }
                        Message::Close(_) => return Err(BybitError::ConnectionClosed),
                        _ => {}
                    }
                }
            }
        }
    }
}

fn parse_trade_msg(text: &str) -> Result<Option<Vec<BybitTrade>>, BybitError> {
    let env: RawEnvelope = serde_json::from_str(text)?;
    if env.topic.map(|t| t.starts_with("publicTrade")) != Some(true) {
        return Ok(None);
    }
    let raw: RawTradeMsg = serde_json::from_str(text)?;
    let trades = raw
        .data
        .into_iter()
        .map(|d| BybitTrade {
            timestamp: d.t,
            price: d.price.parse().unwrap_or(0.0),
            amount: d.volume.parse().unwrap_or(0.0),
            side: d.side,
            id: d.id,
        })
        .collect();
    Ok(Some(trades))
}

// ---- order book (level 1) stream ------------------------------------------------

pub struct OrderBookWatcher {
    ws: WsStream,
    ping_timer: Interval,
}

impl OrderBookWatcher {
    pub async fn connect(symbol: &str) -> Result<Self, BybitError> {
        let topic = format!("orderbook.1.{symbol}");
        let ws = connect_and_subscribe(&topic).await?;
        Ok(Self {
            ws,
            ping_timer: make_ping_interval(),
        })
    }

    /// Waits for the next top-of-book update. Returns `Ok(None)` for frames that
    /// don't carry a usable book so the caller can just loop and try again.
    pub async fn next_top(&mut self) -> Result<Option<BybitOrderBookTop>, BybitError> {
        loop {
            tokio::select! {
                _ = self.ping_timer.tick() => {
                    send_ping(&mut self.ws).await?;
                }
                msg = self.ws.next() => {
                    let msg = msg.ok_or(BybitError::ConnectionClosed)??;
                    match msg {
                        Message::Text(text) => {
                            if let Some(top) = parse_book_msg(&text)? {
                                return Ok(Some(top));
                            }
                        }
                        Message::Ping(payload) => {
                            self.ws.send(Message::Pong(payload)).await?;
                        }
                        Message::Close(_) => return Err(BybitError::ConnectionClosed),
                        _ => {}
                    }
                }
            }
        }
    }
}

fn parse_book_msg(text: &str) -> Result<Option<BybitOrderBookTop>, BybitError> {
    let env: RawEnvelope = serde_json::from_str(text)?;
    if env.topic.map(|t| t.starts_with("orderbook")) != Some(true) {
        return Ok(None);
    }
    let raw: RawBookMsg = serde_json::from_str(text)?;

    let (Some(best_bid), Some(best_ask)) = (raw.data.b.first(), raw.data.a.first()) else {
        return Ok(None);
    };

    Ok(Some(BybitOrderBookTop {
        timestamp: raw.ts,
        bid_price: best_bid[0].parse().unwrap_or(0.0),
        bid_amount: best_bid[1].parse().unwrap_or(0.0),
        ask_price: best_ask[0].parse().unwrap_or(0.0),
        ask_amount: best_ask[1].parse().unwrap_or(0.0),
    }))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct KlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: KlineResult,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<[String; 7]>,
}

/// Fetch historical klines via Bybit's public REST endpoint, used to seed the
/// macro/regime feature rings on startup so they're warm from the first live
/// bar (mirrors executor.py's `seed_historical_data`).
///
/// `interval_secs` is the composite bar size in seconds (300 -> Bybit's "5" =
/// 5-minute interval); `limit` is the number of most-recent bars to fetch
/// (Bybit max 1000). Returns bars oldest-first.
pub async fn get_historical_klines(
    symbol: &str,
    interval_secs: i64,
    limit: u32,
) -> Result<Vec<Kline>, BybitError> {
    let interval_min = (interval_secs / 60).max(1);
    let url = format!(
        "{REST_BASE_URL}/v5/market/kline?category=linear&symbol={symbol}&interval={interval_min}&limit={limit}"
    );

    let resp: KlineResponse = reqwest::get(&url).await?.json().await?;
    if resp.ret_code != 0 {
        return Err(BybitError::SubscriptionRejected(resp.ret_msg));
    }

    // Bybit returns newest-first; flip to chronological (oldest-first).
    let mut klines: Vec<Kline> = resp
        .result
        .list
        .iter()
        .map(|row| Kline {
            open_time: row[0].parse().unwrap_or(0),
            open: row[1].parse().unwrap_or(0.0),
            high: row[2].parse().unwrap_or(0.0),
            low: row[3].parse().unwrap_or(0.0),
            close: row[4].parse().unwrap_or(0.0),
            volume: row[5].parse().unwrap_or(0.0),
        })
        .collect();
    klines.reverse();
    Ok(klines)
}