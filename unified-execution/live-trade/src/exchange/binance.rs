//! Binance USD-M futures — public market-data feed (feed-only venue).
//!
//! Wraps the **public** WebSocket + REST endpoints directly (no SDK):
//!
//!   * [`aggTrade`](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Stream)
//!     trade prints -> [`MarketEvent::Trade`].
//!   * [`bookTicker`](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Individual-Symbol-Book-Ticker-Streams)
//!     best bid/ask -> [`MarketEvent::BookTop`].
//!   * REST [`/fapi/v1/klines`](https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data)
//!     to warm feature rings (`historical_closes`).
//!
//! Aggressor-side convention (matches CCXT): for an `aggTrade`, when `m`
//! (isBuyerMaker) is `true` the buyer was maker and the **seller** was the
//! aggressor -> `Sell`; when `false` the buyer was the aggressor -> `Buy`.

use super::{
    ExchangeError, ExchangeKind, Feed, MarketBookTop, MarketEvent, MarketTrade, TradeSide, now_ms,
    symbols,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::{Interval, interval};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};

const PUBLIC_WS_URL: &str = "wss://fstream.binance.com/stream";
const REST_URL: &str = "https://fapi.binance.com/fapi/v1";
const PING_INTERVAL_SECS: u64 = 30;
const MAX_BARS: u32 = 200;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Binance USD-M futures feed (market data only).
pub struct BinanceFeed {
    symbol: String,
    ws: Option<WsStream>,
    ping: Interval,
}

impl BinanceFeed {
    pub fn new() -> Self {
        Self {
            symbol: String::new(),
            ws: None,
            ping: interval(Duration::from_secs(PING_INTERVAL_SECS)),
        }
    }

    /// Parse a JSON value that may be a numeric string or a number.
    fn parse_f64(v: Option<&serde_json::Value>) -> f64 {
        v.and_then(|x| x.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or_else(|| v.and_then(|x| x.as_f64()).unwrap_or(0.0))
    }
}

impl Default for BinanceFeed {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Feed for BinanceFeed {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Binance
    }

    async fn connect(&mut self, ticker: &str) -> Result<(), ExchangeError> {
        self.symbol = symbols::binance(ticker);
        let (mut ws, _resp) = connect_async(PUBLIC_WS_URL).await?;

        let low = self.symbol.to_ascii_lowercase();
        ws.send(Message::Text(
            json!({
                "method": "SUBSCRIBE",
                "params": [
                    format!("{low}@aggTrade"),
                    format!("{low}@bookTicker"),
                ],
                "id": 1,
            })
            .to_string(),
        ))
        .await?;

        // Wait for the subscription ack before handing the stream back.
        loop {
            let msg = ws.next().await.ok_or(ExchangeError::ConnectionClosed)??;
            if let Message::Text(text) = &msg {
                let v: serde_json::Value = serde_json::from_str(text)?;
                if v.get("id").and_then(|i| i.as_i64()) == Some(1) {
                    // `result` of `null` = success; otherwise it's an error object.
                    if let Some(err) = v.get("error") {
                        return Err(ExchangeError::SubscriptionRejected(err.to_string()));
                    }
                    break;
                }
            }
        }

        self.ws = Some(ws);
        Ok(())
    }

    async fn next_events(&mut self) -> Result<Vec<MarketEvent>, ExchangeError> {
        let ws = self.ws.as_mut().expect("connect() must be called first");
        loop {
            tokio::select! {
                _ = self.ping.tick() => {
                    ws.send(Message::Ping(vec![0xff; 4])).await?;
                }
                msg = ws.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(events) = parse_stream_frame(&text)? {
                                return Ok(events);
                            }
                        }
                        Some(Ok(Message::Ping(p))) => {
                            ws.send(Message::Pong(p)).await?;
                        }
                        Some(Ok(Message::Pong(_))) => {}
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
        let symbol = symbols::binance(ticker);
        let limit = bars.min(MAX_BARS).max(1);
        let url = format!("{REST_URL}/klines?symbol={symbol}&interval=1m&limit={limit}");

        let text = reqwest::get(&url).await?.text().await?;
        // Each row: [openTime, open, high, low, close, volume, ...]
        let rows: Vec<Vec<serde_json::Value>> = serde_json::from_str(&text)?;
        let mut closes: Vec<f64> = rows
            .iter()
            .filter_map(|row| row.get(4).and_then(|c| c.as_str())?.parse().ok())
            .collect();
        closes.truncate(bars as usize);
        Ok(closes)
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        self.ws = None;
        Ok(())
    }
}
/// Parse a combined-stream frame. Messages arrive as `{"stream": "...",
/// "data": {...}}`; we dispatch on the inner `e` event type and return a
/// normalized event (aggTrade -> Trade, bookTicker -> BookTop) or `None` for
/// anything not actionable.
fn parse_stream_frame(text: &str) -> Result<Option<Vec<MarketEvent>>, ExchangeError> {
    let v: serde_json::Value = serde_json::from_str(text)?;
    let data = match v.get("data") {
        Some(d) => d,
        None => return Ok(None),
    };
    match data.get("e").and_then(|e| e.as_str()) {
        Some("aggTrade") => {
            let timestamp = data.get("E").and_then(|x| x.as_i64()).unwrap_or_else(now_ms);
            let price = BinanceFeed::parse_f64(data.get("p"));
            let amount = BinanceFeed::parse_f64(data.get("q"));
            if price <= 0.0 {
                return Ok(None);
            }
            let is_buyer_maker = data.get("m").and_then(|m| m.as_bool()).unwrap_or(false);
            let side = if is_buyer_maker {
                TradeSide::Sell // buyer maker => seller was aggressor
            } else {
                TradeSide::Buy
            };
            let id = data.get("a").map(|a| a.to_string()).unwrap_or_default();
            Ok(Some(vec![MarketEvent::Trade(MarketTrade {
                timestamp,
                price,
                amount,
                side,
                id,
            })]))
        }
        Some("bookTicker") => {
            let bid_price = BinanceFeed::parse_f64(data.get("b"));
            let bid_amount = BinanceFeed::parse_f64(data.get("B"));
            let ask_price = BinanceFeed::parse_f64(data.get("a"));
            let ask_amount = BinanceFeed::parse_f64(data.get("A"));
            if bid_price <= 0.0 || ask_price <= 0.0 {
                return Ok(None);
            }
            Ok(Some(vec![MarketEvent::BookTop(MarketBookTop {
                timestamp: now_ms(),
                bid_price,
                bid_amount,
                ask_price,
                ask_amount,
            })]))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agg_trade_buyer_maker_true_is_sell() {
        let frame = r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1631654789521,"s":"BTCUSDT","a":1020,"p":"7218.00","q":"0.004","f":1,"l":1,"T":1631654789521,"m":true}}"#;
        let events = parse_stream_frame(frame).unwrap().unwrap();
        match &events[0] {
            MarketEvent::Trade(t) => {
                assert_eq!(t.timestamp, 1631654789521);
                assert_eq!(t.price, 7218.0);
                assert_eq!(t.amount, 0.004);
                assert_eq!(t.side, TradeSide::Sell);
                assert_eq!(t.id, "1020");
            }
            other => panic!("expected trade, got {other:?}"),
        }
    }

    #[test]
    fn agg_trade_buyer_maker_false_is_buy() {
        let frame = r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1639444789521,"s":"BTCUSDT","a":1021,"p":"7220.50","q":"0.100","f":1,"l":1,"T":1631654789521,"m":false}}"#;
        let events = parse_stream_frame(frame).unwrap().unwrap();
        match &events[0] {
            MarketEvent::Trade(t) => {
                assert_eq!(t.side, TradeSide::Buy);
                assert_eq!(t.price, 7220.5);
            }
            other => panic!("expected trade, got {other:?}"),
        }
    }

    #[test]
    fn parses_book_ticker() {
        let frame = r#"{"stream":"btcusdt@bookTicker","data":{"u":400900217,"e":"bookTicker","s":"BTCUSDT","b":"27135.20","B":"0.500","a":"27135.50","A":"0.300"}}"#;
        let events = parse_stream_frame(frame).unwrap().unwrap();
        match &events[0] {
            MarketEvent::BookTop(b) => {
                assert_eq!(b.bid_price, 27135.2);
                assert_eq!(b.bid_amount, 0.5);
                assert_eq!(b.ask_price, 27135.5);
                assert_eq!(b.ask_amount, 0.3);
            }
            other => panic!("expected booktop, got {other:?}"),
        }
    }

    #[test]
    fn ignores_control_and_empty_payloads() {
        assert!(parse_stream_frame("{\"stream\":\"btcusdt@aggTrade\",\"data\":{}}")
            .unwrap()
            .is_none());
        assert!(parse_stream_frame(r#"{"id":null,"result":null}"#).unwrap().is_none());
    }
}