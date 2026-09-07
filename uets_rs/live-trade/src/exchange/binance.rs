//! Binance SPOT — public market-data feed (feed-only venue).
//!
//! Wraps the **public** WebSocket + REST endpoints directly (no SDK):
//!
//!   * [`aggTrade`](https://developers.binance.com/docs/binance-spot-api-docs/websocket-streams#aggregate-trade-streams)
//!     trade prints -> [`MarketEvent::Trade`].
//!   * [`bookTicker`](https://developers.binance.com/docs/binance-spot-api-docs/websocket-streams#individual-symbol-book-ticker-streams)
//!     best bid/ask -> [`MarketEvent::BookTop`].
//!   * REST `/api/v3/klines` to warm feature rings (`historical_closes`).
//!
//! NOTE on venue choice: some networks / VPN exits receive silently-degraded
//! **futures** (`fstream.binance.com`) market data — `@aggTrade` and kline
//! streams deliver nothing while `@bookTicker`/depth keep flowing. The SPOT
//! endpoints (`stream.binance.com` + `api.binance.com`) serve every stream
//! reliably from those same networks, so this feed uses SPOT. Spot BTCUSDT
//! tracks the perp within a few bps, which is immaterial for feature input.
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

const PUBLIC_WS_URL: &str = "wss://stream.binance.com:9443/stream";
const REST_URL: &str = "https://api.binance.com/api/v3";
const PING_INTERVAL_SECS: u64 = 30;
const MAX_BARS: u32 = 200;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Binance USD-M futures feed (market data only).
pub struct BinanceFeed {
    symbol: String,
    ws: Option<WsStream>,
    ping: Interval,
    pong_at: Option<std::time::Instant>, // instant the last Ping was sent (RTT timing)
    last_rtt_ms: Option<f64>,            // last ping→pong RTT (monotonic, skew-free)
}

impl BinanceFeed {
    pub fn new() -> Self {
        Self {
            symbol: String::new(),
            ws: None,
            ping: interval(Duration::from_secs(PING_INTERVAL_SECS)),
            pong_at: None,
            last_rtt_ms: None,
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
        eprintln!("[feed] {}: subscribing to aggTrade + bookTicker", self.symbol);
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

        // Wait for the subscription ack with timeout. Some networks/VPNs silently
        // drop WS frames after the initial handshake, so we time out instead of hanging.
        let ack_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if std::time::Instant::now() > ack_deadline {
                return Err(ExchangeError::SubscriptionRejected(
                    "timeout waiting for subscription ACK (VPN may be blocking WS traffic)".into()
                ));
            }
            let msg = ws.next().await.ok_or(ExchangeError::ConnectionClosed)??;
            if let Message::Text(text) = &msg {
                let v: serde_json::Value = serde_json::from_str(text)?;
                if v.get("id").and_then(|i| i.as_i64()) == Some(1) {
                    // `result` of `null` = success; otherwise it's an error object.
                    if let Some(err) = v.get("error") {
                        return Err(ExchangeError::SubscriptionRejected(err.to_string()));
                    }
                    eprintln!("[feed] {}: subscription ACK received", self.symbol);
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
                    self.pong_at = Some(std::time::Instant::now());
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
                        Some(Ok(Message::Pong(_))) => {
                            if let Some(t) = self.pong_at.take() {
                                let ms = t.elapsed().as_secs_f64() * 1000.0;
                                // Discard samples contaminated by loop stalls
                                // (pacing/catch-up): a pong read seconds after
                                // it arrived measures our backlog, not the
                                // network. Healthy WS RTT is < 3 s.
                                if ms < 3000.0 {
                                    self.last_rtt_ms = Some(ms);
                                }
                            }
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
        let symbol = symbols::binance(ticker);
        let limit = bars.min(MAX_BARS).max(1);
        // Seed with 5-minute closes so the composite (TIMEFRAME_BARRIER=300 s)
        // RSI/MA feature rings are warmed on the SAME bar size the engine
        // evaluates on — feeding 1-minute closes into a 5-minute ring pins the
        // seeded rsi_2 near 0 (see features.py "Seeding MUST use the same bar
        // size"). Matches the Bitfinex `trade:5m` path.
        let url = format!("{REST_URL}/klines?symbol={symbol}&interval=5m&limit={limit}");
        kline_closes_from_url(&url, bars).await
    }

    fn ws_rtt_ms(&self) -> Option<f64> {
        self.last_rtt_ms
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        self.ws = None;
        Ok(())
    }
}

/// Fetch + decode a Binance `/api/v3/klines` response body from `url`.
///
/// Split out of [`Feed::historical_closes`] so tests can point it at a local
/// mock server (Binance geo-blocks shared CI runner IPs, so live-network
/// tests are `#[ignore]`d). Each row is
/// `[openTime, open, high, low, close, volume, ...]` — close is index 4 as a
/// numeric string; closes are returned oldest-first.
async fn kline_closes_from_url(url: &str, bars: u32) -> Result<Vec<f64>, ExchangeError> {
    let resp = tokio::time::timeout(Duration::from_secs(10), reqwest::get(url))
        .await
        .map_err(|_| ExchangeError::Http("seed fetch timed out after 10s".into()))??;
    let text = resp.error_for_status()?.text().await?;
    let rows: Vec<Vec<serde_json::Value>> = serde_json::from_str(&text)?;
    let mut closes: Vec<f64> = Vec::with_capacity(rows.len());
    for row in &rows {
        // Rows are [openTime, open, high, low, close, ...], oldest-first.
        let open_ms = row.get(0).and_then(|v| v.as_i64()).unwrap_or(i64::MAX);
        let close = match row.get(4).and_then(|c| c.as_str()) {
            Some(s) => s.parse::<f64>().ok(),
            None => row.get(4).and_then(|c| c.as_f64()),
        };
        let Some(close) = close else { continue };
        // Skip the CURRENTLY-FORMING 5-min bar: its close changes every trade
        // and two processes fetching a moment apart would seed DIFFERENT
        // closes, permanently diverging the recursive RSI state. Only closed
        // bars (open + 300s <= now) are used for seeding — same as research.
        let now_ms = now_ms();
        if open_ms + 300_000 <= now_ms {
            closes.push(close);
        }
    }
    closes.truncate(bars as usize);
    Ok(closes)
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
    // Spot bookTicker frames carry no `"e"` event field at all
    // (`{"u":..,"s":"BTCUSDT","b":..,"B":..,"a":..,"A":..}`), so dispatch on
    // the payload shape when `e` is absent.
    let event = data.get("e").and_then(|e| e.as_str());
    if event.is_none() && data.get("b").is_some() && data.get("a").is_some() {
        let bid_price = BinanceFeed::parse_f64(data.get("b"));
        let bid_amount = BinanceFeed::parse_f64(data.get("B"));
        let ask_price = BinanceFeed::parse_f64(data.get("a"));
        let ask_amount = BinanceFeed::parse_f64(data.get("A"));
        if bid_price <= 0.0 || ask_price <= 0.0 {
            return Ok(None);
        }
        return Ok(Some(vec![MarketEvent::BookTop(MarketBookTop {
            timestamp: now_ms(),
            bid_price,
            bid_amount,
            ask_price,
            ask_amount,
        })]));
    }
    match event {
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

    /// Spot bookTicker frames carry no `"e"` field — they must still parse
    /// into a BookTop (regression for the futures-only dispatcher).
    #[test]
    fn spot_book_ticker_without_event_field_parses() {
        let frame = r#"{"stream":"btcusdt@bookTicker","data":{"u":400900217,"s":"BTCUSDT","b":"27135.20","B":"0.500","a":"27135.50","A":"0.300"}}"#;
        let events = parse_stream_frame(frame).unwrap().unwrap();
        match &events[0] {
            MarketEvent::BookTop(b) => {
                assert_eq!(b.bid_price, 27135.2);
                assert_eq!(b.ask_price, 27135.5);
            }
            other => panic!("expected booktop, got {other:?}"),
        }
    }

    use crate::exchange::testing::spawn_mock_json_http;

    /// CI-safe kline decode test: a local mock HTTP server serves the exact
    /// live /api/v3/klines shape (array of arrays, close = index 4 as a
    /// numeric string) and the fetch must decode + order oldest-first.
    /// (Binance geo-blocks CI runner IPs with a JSON error *object*, which
    /// used to fail this test with "invalid type: map, expected a sequence".)
    #[tokio::test]
    async fn historical_closes_decode_and_order() {
        let body = r#"[
            [1700000000000,"42000.0","42100.0","41900.0","42050.5","10",1700000299999,"10",10,"0","0","0"],
            [1700000300000,"42050.5","42300.0","42000.0","42250.0","12",1700000599999,"12",10,"0","0","0"],
            [1700000600000,"42250.0","42400.0","42100.0","42350.0","9",1700000899999,"9",10,"0","0","0"]
        ]"#;
        let base = spawn_mock_json_http(body);
        let url = format!("{base}/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=20");
        let closes = kline_closes_from_url(&url, 20)
            .await
            .expect("binance kline fetch must decode");
        assert_eq!(closes, vec![42050.5, 42250.0, 42350.0], "oldest-first closes");
    }

    /// Live-network smoke test — NOT run in CI (`--ignored`): Binance
    /// geo-blocks shared CI runner IPs. Run manually from a permitted
    /// network: `cargo test -p live-trade --lib -- --ignored`
    #[tokio::test]
    #[ignore = "live network; Binance geo-blocks CI runners"]
    async fn historical_closes_live_smoke() {
        let feed = BinanceFeed::new();
        let closes = Feed::historical_closes(&feed, "btcusdt", 20)
            .await
            .expect("binance kline fetch must decode");
        assert!(closes.len() >= 10, "expected ~20 closes, got {}", closes.len());
        assert!(closes.iter().all(|c| *c > 1000.0), "BTCUSDT prices expected");
        let spread = (closes.last().unwrap() - closes.first().unwrap()).abs()
            / closes.first().unwrap();
        assert!(spread < 0.10, "20x5m closes should be within 10% of spot, {spread}");
    }

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