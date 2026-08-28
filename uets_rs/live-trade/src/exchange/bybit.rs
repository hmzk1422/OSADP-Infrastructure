//! Bybit v5 — direct WebSocket + REST client (no SDK).
//!
//! Public market data flows over a single TLS socket
//! (`wss://ws.bybit.com/v5/public`): both `publicTrade.{symbol}` and
//! `orderbook.1.{symbol}` (best bid/ask) are subscribed on one connection.
//! Private order routing (create / cancel) and wallet balances go through the
//! v5 REST API, signed with HMAC-SHA256 (`X-BAPI-SIGN`). Only tokio,
//! tokio-tungstenite and reqwest are used — no exchange SDK.
//!
//! Signing rule (BAPi v5): message = `${timestamp}${recvWindow}${params}` where
//! `params` is the GET query string (no leading `?`) or the raw JSON body of a
//! POST. Reference: https://bybit-exchange.github.io/docs/v5/

use super::{
    Balance, Broker, ExchangeError, ExchangeKind, Feed, MarketBookTop, MarketEvent, MarketTrade,
    OrderRequest, OrderResponse, OrderType, TradeSide, header_map, hmac_sha256, now_ms, symbols,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::{Interval, interval};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};

// Bybit v5 public WS endpoints are category-specific: a bare `/v5/public`
// path returns 404. BTCUSDT-style linear perpetuals stream on the `linear`
// endpoint. `stream.bybit.com` resolves reliably; `ws.bybit.com` (the old
// default) is not resolved by some networks (e.g. ProtonVPN) and is kept only
// as a secondary if a network resolves it.
const PUBLIC_WS_URL: &str = "wss://stream.bybit.com/v5/public/linear";
const PUBLIC_WS_FALLBACK: &str = "wss://ws.bybit.com/v5/public/linear";
const REST_URL: &str = "https://api.bybit.com/v5";
const PING_INTERVAL_SECS: u64 = 30;
const MAX_BARS: u32 = 200;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Bybit v5 exchange implementation.
pub struct BybitExchange {
    api_key: Option<String>,
    api_secret: Option<String>,
    symbol: String, // venue-native, e.g. "BTCUSDT"
    ws: Option<WsStream>,
    ping: Interval,
}

impl BybitExchange {
    pub fn new(api_key: Option<String>, api_secret: Option<String>) -> Self {
        Self {
            api_key,
            api_secret,
            symbol: String::new(),
            ws: None,
            ping: interval(Duration::from_secs(PING_INTERVAL_SECS)),
        }
    }

    fn credentials(&self) -> Result<(&str, &str), ExchangeError> {
        let key = self
            .api_key
            .as_deref()
            .ok_or_else(|| ExchangeError::Auth("BYBIT_API_KEY is not set".into()))?;
        let secret = self
            .api_secret
            .as_deref()
            .ok_or_else(|| ExchangeError::Auth("BYBIT_API_SECRET is not set".into()))?;
        Ok((key, secret))
    }

    /// BAPi v5 auth headers. `params` is the GET query string (no leading `?`)
    /// or the raw JSON body for a POST.
    fn signed_headers(
        &self,
        api_key: &str,
        api_secret: &str,
        params: &str,
    ) -> Vec<(&'static str, String)> {
        let recv_window = 5000u64;
        let ts = now_ms();
        let msg = format!("{ts}{recv_window}{params}");
        let sig = hmac_sha256(api_secret, &msg);
        vec![
            ("X-BAPI-API-KEY", api_key.to_string()),
            ("X-BAPI-SIGN", sig),
            ("X-BAPI-SIGN-TYPE", "2".to_string()),
            ("X-BAPI-TIMESTAMP", ts.to_string()),
            ("X-BAPI-RECV-WINDOW", recv_window.to_string()),
        ]
    }

    /// Subscribe to `publicTrade` + `orderbook.1` on a connected socket and wait
    /// for the ack before handing the stream back to the caller.
    async fn after_connect(&mut self, mut ws: WsStream) -> Result<(), ExchangeError> {
        let sub = json!({
            "op": "subscribe",
            "args": [
                format!("publicTrade.{}", self.symbol),
                format!("orderbook.1.{}", self.symbol),
            ],
        });
        ws.send(Message::Text(sub.to_string())).await?;

        // Wait for the subscription ack before handing the stream back. Bybit
        // v5 acknowledges with a top-level `"operate":"subscribe"` field
        // (`{"success":true,"ret_msg":"","conn_id":...,"op":"subscribe"}`) —
        // NOT a nested `request.op`. Accept both forms so older/newer Bybit
        // responses are handled.
        loop {
            let msg = ws.next().await.ok_or(ExchangeError::ConnectionClosed)??;
            if let Message::Text(text) = &msg {
                let v: serde_json::Value = serde_json::from_str(text)?;
                let is_sub = v.get("op").and_then(|o| o.as_str()) == Some("subscribe")
                    || v.get("request")
                        .and_then(|r| r.get("op"))
                        .and_then(|o| o.as_str())
                        == Some("subscribe");
                let is_ack = v.get("success").and_then(|x| x.as_bool()).unwrap_or(false)
                    && is_sub;
                if is_ack {
                    break;
                }
                if v.get("success").and_then(|x| x.as_bool()) == Some(false) {
                    let m = v.get("ret_msg").and_then(|r| r.as_str()).unwrap_or("");
                    if !m.is_empty() {
                        return Err(ExchangeError::SubscriptionRejected(m.to_string()));
                    }
                }
            }
        }

        self.ws = Some(ws);
        Ok(())
    }
}

#[async_trait::async_trait]
impl Feed for BybitExchange {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Bybit
    }

    async fn connect(&mut self, ticker: &str) -> Result<(), ExchangeError> {
        self.symbol = symbols::bybit(ticker);
        // Try the primary public endpoint first, then the fallback: a network's
        // DNS can fail to resolve one Bybit host but answer the other, so this
        // keeps the feed alive across VPN nodes / resolver changes.
        let mut last_err: Option<ExchangeError> = None;
        for url in [PUBLIC_WS_URL, PUBLIC_WS_FALLBACK] {
            match connect_async(url).await {
                Ok((ws, _resp)) => return self.after_connect(ws).await,
                Err(e) => last_err = Some(e.into()),
            }
        }
        Err(last_err.unwrap_or(ExchangeError::ConnectionClosed))
    }

    async fn next_events(&mut self) -> Result<Vec<MarketEvent>, ExchangeError> {
        let ws = self.ws.as_mut().expect("connect() must be called first");
        loop {
            tokio::select! {
                _ = self.ping.tick() => {
                    ws.send(Message::Ping(Vec::new())).await?;
                }
                msg = ws.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(events) = parse_frame(&text)? {
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
        let symbol = symbols::bybit(ticker);
        let limit = bars.min(MAX_BARS).max(1);
        let url = format!(
            "{REST_URL}/market/kline?category=linear&symbol={symbol}&interval=5&limit={limit}"
        );
        kline_closes_from_url(&url).await
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        self.ws = None;
        Ok(())
    }
}

/// Fetch + decode a Bybit v5 `/market/kline` response from `url`.
///
/// Split out of [`Feed::historical_closes`] so tests can point it at a local
/// mock server (Bybit serves rate-limit/challenge pages to shared CI runner
/// IPs, so live-network tests are `#[ignore]`d). Wire shape:
/// `{"retCode":0,"result":{"list":[[startTime,open,high,low,close,...],..]}}`
/// — close is index 4 (numeric string) and `list` is NEWEST-first, so the
/// closes are reversed to oldest-first for seed warming.
async fn kline_closes_from_url(url: &str) -> Result<Vec<f64>, ExchangeError> {
    #[derive(Deserialize)]
    struct KlineResult {
        list: Vec<Vec<serde_json::Value>>,
    }
    #[derive(Deserialize)]
    struct KlineEnvelope {
        // Bybit v5 returns camelCase `retCode`; without this rename serde
        // looks for a literal `ret_code` key and every fetch fails with
        // "missing field `ret_code`" -> "error decoding response body".
        #[serde(rename = "retCode")]
        ret_code: i64,
        result: Option<KlineResult>,
    }

    let env: KlineEnvelope = reqwest::get(url)
        .await?
        .error_for_status()?
        .json()
        .await?;
    let mut closes: Vec<f64> = Vec::new();
    if env.ret_code == 0 {
        if let Some(res) = env.result {
            // Each row is an array:
            // [startTime, open, high, low, close, volume, turnover]
            // — close is index 4 (a numeric string).
            for row in res.list {
                if let Some(c) = row.get(4).and_then(|v| v.as_str()) {
                    if let Ok(c) = c.parse::<f64>() {
                        closes.push(c);
                    }
                }
            }
        }
    }
    // Bybit returns newest-first; return oldest-first for seed warming.
    closes.reverse();
    Ok(closes)
}

#[async_trait::async_trait]
impl Broker for BybitExchange {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Bybit
    }

    async fn place_order(&self, req: &OrderRequest) -> Result<OrderResponse, ExchangeError> {
        let (key, secret) = self.credentials()?;
        let side = match req.side {
            TradeSide::Buy => "Buy",
            TradeSide::Sell => "Sell",
        };
        let order_type = match req.order_type {
            OrderType::Market => "Market",
            OrderType::Limit => "Limit",
        };
        let mut body = json!({
            "category": "linear",
            "symbol": req.symbol,
            "side": side,
            "orderType": order_type,
            "qty": format!("{:.8}", req.amount),
        });
        if req.order_type == OrderType::Limit {
            body["price"] = json!(format!("{:.8}", req.price));
        }
        let body_str = serde_json::to_string(&body)?;
        let headers = self.signed_headers(key, secret, &body_str);

        let resp: serde_json::Value = reqwest::Client::new()
            .post(format!("{REST_URL}/order/create"))
            .header("Content-Type", "application/json")
            .headers(header_map(&headers))
            .body(body_str)
            .send()
            .await?
            .json()
            .await?;
        let ret = resp.get("retCode").and_then(|r| r.as_i64()).unwrap_or(1);
        if ret != 0 {
            return Err(ExchangeError::OrderRejected(
                resp.get("retMsg")
                    .and_then(|r| r.as_str())
                    .unwrap_or("")
                    .to_string(),
            ));
        }
        let order_id = resp
            .get("result")
            .and_then(|r| r.get("orderId"))
            .and_then(|i| i.as_str())
            .unwrap_or("")
            .to_string();
        Ok(OrderResponse {
            exchange_order_id: order_id,
            status: "new".to_string(),
            avg_price: 0.0,
            filled_amount: 0.0,
        })
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExchangeError> {
        let (key, secret) = self.credentials()?;
        let body = json!({
            "category": "linear",
            "orderId": order_id,
            "symbol": self.symbol,
        });
        let body_str = serde_json::to_string(&body)?;
        let headers = self.signed_headers(key, secret, &body_str);

        let resp: serde_json::Value = reqwest::Client::new()
            .post(format!("{REST_URL}/order/cancel"))
            .header("Content-Type", "application/json")
            .headers(header_map(&headers))
            .body(body_str)
            .send()
            .await?
            .json()
            .await?;
        let ret = resp.get("retCode").and_then(|r| r.as_i64()).unwrap_or(1);
        if ret != 0 {
            return Err(ExchangeError::OrderRejected(
                resp.get("retMsg")
                    .and_then(|r| r.as_str())
                    .unwrap_or("")
                    .to_string(),
            ));
        }
        Ok(())
    }

    async fn balances(&self) -> Result<Vec<Balance>, ExchangeError> {
        let (key, secret) = self.credentials()?;
        let qs = "accountType=UNIFIED";
        let headers = self.signed_headers(key, secret, qs);

        let resp: serde_json::Value = reqwest::Client::new()
            .get(format!("{REST_URL}/account/wallet-balance?{qs}"))
            .headers(header_map(&headers))
            .send()
            .await?
            .json()
            .await?;
        let ret = resp.get("retCode").and_then(|r| r.as_i64()).unwrap_or(1);
        if ret != 0 {
            return Err(ExchangeError::Http(
                resp.get("retMsg")
                    .and_then(|r| r.as_str())
                    .unwrap_or("bybit balance error")
                    .to_string(),
            ));
        }

        let mut out = Vec::new();
        if let Some(coins) = resp
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .and_then(|list| list.first())
            .and_then(|acct| acct.get("coin"))
            .and_then(|c| c.as_array())
        {
            for c in coins {
                let currency = c
                    .get("coin")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let total = parse_f64(c.get("total"));
                let available = parse_f64(c.get("walletBalance"));
                if !currency.is_empty() {
                    out.push(Balance {
                        currency,
                        total,
                        available,
                    });
                }
            }
        }
        Ok(out)
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        self.ws = None;
        Ok(())
    }
}

/// Parse a JSON value that may be a numeric string or a number.
fn parse_f64(v: Option<&serde_json::Value>) -> f64 {
    v.and_then(|x| x.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or_else(|| v.and_then(|x| x.as_f64()).unwrap_or(0.0))
}

// ── WebSocket frame parsing ──────────────────────────────────────────
//
// Bybit v5 public frames are JSON objects with a `topic` ("publicTrade.*" /
// "orderbook.1.*") and a `data` payload. Acks / control frames carry no
// `topic` and are ignored.

fn parse_frame(text: &str) -> Result<Option<Vec<MarketEvent>>, ExchangeError> {
    let v: serde_json::Value = serde_json::from_str(text)?;
    let topic = match v.get("topic").and_then(|t| t.as_str()) {
        Some(t) => t.to_string(),
        None => return Ok(None),
    };
    let data = v.get("data");

    if topic.starts_with("publicTrade") {
        let mut out = Vec::new();
        match data {
            Some(serde_json::Value::Array(list)) => {
                for item in list {
                    if let Some(t) = parse_trade(item)? {
                        out.push(MarketEvent::Trade(t));
                    }
                }
            }
            Some(item) => {
                if let Some(t) = parse_trade(item)? {
                    out.push(MarketEvent::Trade(t));
                }
            }
            None => {}
        }
        Ok(if out.is_empty() { None } else { Some(out) })
    } else if topic.starts_with("orderbook.1") {
        let ts = v.get("ts").and_then(|x| x.as_i64()).unwrap_or_else(now_ms);
        match parse_book_top(data, ts) {
            Ok(Some(b)) => Ok(Some(vec![MarketEvent::BookTop(b)])),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    } else {
        Ok(None)
    }
}

/// Parse one Bybit trade object (`publicTrade.data`).
/// Field reference: `p` price, `s` size, `S` side ("Buy"/"Sell"), `T` ts, `i`/`id` id.
fn parse_trade(item: &serde_json::Value) -> Result<Option<MarketTrade>, ExchangeError> {
    let o = match item.as_object() {
        Some(o) => o,
        None => return Ok(None),
    };
    let timestamp = o
        .get("T")
        .and_then(|x| x.as_i64())
        .or_else(|| o.get("t").and_then(|x| x.as_i64()))
        .or_else(|| o.get("ts").and_then(|x| x.as_i64()))
        .unwrap_or_else(now_ms);
    let price = parse_f64(o.get("p")).max(0.0);
    let amount = parse_f64(o.get("s")).max(0.0);
    let id = o
        .get("i")
        .and_then(|x| x.as_str())
        .or_else(|| o.get("id").and_then(|x| x.as_str()))
        .unwrap_or("")
        .to_string();
    let side = match o.get("S").and_then(|x| x.as_str()) {
        Some("Buy") | Some("buy") => TradeSide::Buy,
        _ => TradeSide::Sell,
    };
    if price == 0.0 {
        return Ok(None);
    }
    Ok(Some(MarketTrade {
        timestamp,
        price,
        amount,
        side,
        id,
    }))
}

/// Best bid / ask from a Level-1 `orderbook.1` snapshot or delta frame.
fn parse_book_top(
    data: Option<&serde_json::Value>,
    ts: i64,
) -> Result<Option<MarketBookTop>, ExchangeError> {
    let o = match data.and_then(|d| d.as_object()) {
        Some(o) => o,
        None => return Ok(None),
    };
    let bid = o
        .get("b")
        .and_then(|a| a.get(0))
        .and_then(|x| x.as_object());
    let ask = o
        .get("a")
        .and_then(|a| a.get(0))
        .and_then(|x| x.as_object());
    if let (Some(b), Some(a)) = (bid, ask) {
        let bp = parse_f64(b.get("p")).max(0.0);
        let ap = parse_f64(a.get("p")).max(0.0);
        if bp > 0.0 && ap > 0.0 {
            return Ok(Some(MarketBookTop {
                timestamp: ts,
                bid_price: bp,
                bid_amount: parse_f64(b.get("s")),
                ask_price: ap,
                ask_amount: parse_f64(a.get("s")),
            }));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::exchange::testing::spawn_mock_json_http;

    /// CI-safe kline decode test: a local mock HTTP server serves the exact
    /// live v5 wire shape — `retCode` envelope, `result.list` as an array of
    /// arrays (close = index 4, numeric string), NEWEST-first. Guards the
    /// envelope deserializer + oldest-first reversal without touching the
    /// network (Bybit challenge-pages CI runner IPs, which used to fail this
    /// test with "error decoding response body" and left rsi_2 at its 100.0
    /// placeholder).
    #[tokio::test]
    async fn historical_closes_decode_and_order() {
        let body = r#"{"retCode":0,"retMsg":"OK","result":{"category":"linear","symbol":"BTCUSDT","list":[
            ["1700000600000","42250.0","42400.0","42100.0","42350.0","9","19"],
            ["1700000300000","42050.5","42300.0","42000.0","42250.0","12","25"],
            ["1700000000000","42000.0","42100.0","41900.0","42050.5","10","21"]
        ]}}"#;
        let base = spawn_mock_json_http(body);
        let url =
            format!("{base}/v5/market/kline?category=linear&symbol=BTCUSDT&interval=5&limit=20");
        let closes = kline_closes_from_url(&url)
            .await
            .expect("bybit kline fetch must decode");
        assert_eq!(
            closes,
            vec![42050.5, 42250.0, 42350.0],
            "newest-first list must decode and reverse to oldest-first"
        );
    }

    /// Live-network smoke test — NOT run in CI (`--ignored`): Bybit serves
    /// rate-limit/challenge pages to shared CI runner IPs. Run manually:
    /// `cargo test -p live-trade --lib -- --ignored`
    #[tokio::test]
    #[ignore = "live network; Bybit blocks CI runners"]
    async fn historical_closes_live_smoke() {
        let ex = BybitExchange::new(None, None);
        let closes = Feed::historical_closes(&ex, "btcusdt", 20)
            .await
            .expect("bybit kline fetch must decode");
        assert!(closes.len() >= 10, "expected ~20 closes, got {}", closes.len());
        assert!(closes.iter().all(|c| *c > 1000.0), "BTCUSDT prices expected");
        // Oldest-first ordering: the last close must track current spot.
        let live = closes.last().copied().unwrap();
        assert!(
            (live - closes[closes.len() - 2]).abs() < closes[closes.len() - 2] * 0.05,
            "consecutive 5m closes should be within 5% of each other, got {live}"
        );
    }

    #[test]
    fn symbol_mapping() {
        assert_eq!(symbols::bybit("btcusdt"), "BTCUSDT");
        assert_eq!(symbols::bybit("SOLUSDT"), "SOLUSDT");
    }

    #[test]
    fn parses_trade_frame() {
        let frame = r#"{"topic":"publicTrade.BTCUSDT","type":"snapshot","ts":1631654789521,"data":{"i":"2468814995","T":1568274289947,"p":"7218.00","s":0.004,"S":"Buy"}}"#;
        let events = parse_frame(frame).unwrap().unwrap();
        match &events[0] {
            MarketEvent::Trade(t) => {
                assert_eq!(t.timestamp, 1568274289947);
                assert_eq!(t.price, 7218.0);
                assert_eq!(t.amount, 0.004);
                assert_eq!(t.side, TradeSide::Buy);
                assert_eq!(t.id, "2468814995");
            }
            other => panic!("expected trade, got {other:?}"),
        }
    }

    #[test]
    fn ignore_control_frames() {
        let ack = r#"{"success":true,"ret_msg":"","conn_id":"abc","request":{"op":"subscribe","args":["publicTrade.BTCUSDT"]}}"#;
        assert!(parse_frame(ack).unwrap().is_none());
        assert!(
            parse_frame(r#"{"success":true,"ret_msg":"pong"}"#)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn parses_book_top() {
        let frame = r#"{"topic":"orderbook.1.BTCUSDT","type":"snapshot","ts":1631654000000,"data":{"b":[{"p":"27135.2","s":"0.5"}],"a":[{"p":"27135.5","s":"0.3"}]}}"#;
        let events = parse_frame(frame).unwrap().unwrap();
        match &events[0] {
            MarketEvent::BookTop(b) => {
                assert_eq!(b.timestamp, 1631654000000);
                assert_eq!(b.bid_price, 27135.2);
                assert_eq!(b.ask_price, 27135.5);
                assert_eq!(b.bid_amount, 0.5);
                assert_eq!(b.ask_amount, 0.3);
            }
            other => panic!("expected booktop, got {other:?}"),
        }
    }
}
