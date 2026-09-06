//! Exchange abstraction layer.
//!
//! The strategy engine (crate `engine`) is 100% venue-agnostic. This module
//! defines the [`Exchange`] trait plus the normalized market-data / order
//! routing types every venue implementation (`bybit`, `bitfinex`, ...)
//! satisfies — so the *same* execution loop drives any venue with zero
//! strategy changes.
//!
//! Implementations talk to the venues' **direct WebSocket / HTTP APIs** —
//! no exchange SDK dependencies (only tokio, tokio-tungstenite, reqwest).

pub mod binance;
pub mod bitfinex;
pub mod bybit;
pub mod parquet;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

// ─────────────────────────────────────────────────────────────────────────────
// Venue identity
// ─────────────────────────────────────────────────────────────────────────────

/// The set of trading venues the execution layer can drive.
///
/// A venue may be used as a **feed** (public market data), a **broker**
/// (order routing), or both. For example the strategy can stream orderflow
/// from `Binance` (feed only) while routing orders to `Bitfinex` (broker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeKind {
    Bybit,
    Bitfinex,
    /// Binance USD-M futures (currently feed-only: no order routing is
    /// implemented in this layer).
    Binance,
    /// Historical replay of a pre-converted parquet feed (`live-test`).
    /// Not a real venue: no orders are ever routed in this mode.
    Parquet,
}

impl ExchangeKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bybit" => Some(Self::Bybit),
            "bitfinex" => Some(Self::Bitfinex),
            "binance" => Some(Self::Binance),
            "parquet" | "replay" => Some(Self::Parquet),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bybit => "bybit",
            Self::Bitfinex => "bitfinex",
            Self::Binance => "binance",
            Self::Parquet => "parquet",
        }
    }
}

impl fmt::Display for ExchangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalized market-data types
// ─────────────────────────────────────────────────────────────────────────────

/// Aggressor side of a trade print.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// One market trade print, normalized across venues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTrade {
    /// Milliseconds since the Unix epoch.
    pub timestamp: i64,
    pub price: f64,
    /// Base-currency quantity (always positive; direction is `side`).
    pub amount: f64,
    pub side: TradeSide,
    /// Venue-native trade id (used for de-duplication).
    pub id: String,
}

/// Best bid / ask (level-1) snapshot, normalized across venues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketBookTop {
    /// Milliseconds since the Unix epoch.
    pub timestamp: i64,
    pub bid_price: f64,
    pub bid_amount: f64,
    pub ask_price: f64,
    pub ask_amount: f64,
}

/// A normalized market-data event emitted by an [`Exchange`] stream.
#[derive(Debug, Clone)]
pub enum MarketEvent {
    Trade(MarketTrade),
    BookTop(MarketBookTop),
}

// ─────────────────────────────────────────────────────────────────────────────
// Order-routing types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
}

/// An order-routing request (venue-native symbol).
#[derive(Debug, Clone)]
pub struct OrderRequest {
    /// Venue-native symbol, e.g. "BTCUSDT" (bybit) or "tBTCUST" (bitfinex).
    pub symbol: String,
    pub side: TradeSide,
    pub order_type: OrderType,
    /// Limit price (ignored for Market orders).
    pub price: f64,
    /// Base-currency quantity; always positive — direction comes from `side`.
    pub amount: f64,
}

/// Result of submitting an order.
#[derive(Debug, Clone)]
pub struct OrderResponse {
    pub exchange_order_id: String,
    pub status: String,
    pub avg_price: f64,
    pub filled_amount: f64,
}

/// One wallet balance line.
#[derive(Debug, Clone)]
pub struct Balance {
    pub currency: String,
    pub total: f64,
    pub available: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ExchangeError {
    #[error("websocket transport error: {0}")]
    Transport(String),
    #[error("http request error: {0}")]
    Http(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("subscription rejected: {0}")]
    SubscriptionRejected(String),
    #[error("order rejected: {0}")]
    OrderRejected(String),
    #[error("authentication error: {0}")]
    Auth(String),
    #[error("connection closed by server")]
    ConnectionClosed,
    #[error("{0}")]
    Other(String),
}

impl From<reqwest::Error> for ExchangeError {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(e.to_string())
    }
}

impl From<tokio_tungstenite::tungstenite::Error> for ExchangeError {
    fn from(e: tokio_tungstenite::tungstenite::Error) -> Self {
        Self::Transport(e.to_string())
    }
}

impl From<serde_json::Error> for ExchangeError {
    fn from(e: serde_json::Error) -> Self {
        Self::Parse(e.to_string())
    }
}

// Test-only helpers for exchange-decode tests. Binance geo-blocks and Bybit
// challenge-pages shared CI runner IPs, so live-network tests are `#[ignore]`d
// and decode tests serve canned wire payloads from a local mock server.
#[cfg(test)]
pub(crate) mod testing {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Serve `body` as an HTTP/1.1 200 JSON response on a random 127.0.0.1
    /// port (one thread; one canned response per connection). Returns the
    /// base URL (`http://127.0.0.1:<port>`) to build endpoint paths against.
    pub(crate) fn spawn_mock_json_http(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("mock http bind");
        let addr = listener.local_addr().expect("mock http addr");
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                // Consume the request head (best effort — we always answer
                // with the canned body regardless of the request).
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Market-data feed trait
// ─────────────────────────────────────────────────────────────────────────────

/// A market-data feed: public trade prints + top-of-book depth, normalized into
/// [`MarketEvent`]s.
///
/// Feeds are decoupled from order routing so the strategy can stream orderflow
/// from a venue that is *not* where its capital lives (e.g. read Binance / Bybit
/// tape while executing on Bitfinex). Implementors keep venue wire formats
/// internal and expose only the normalized types.
#[async_trait::async_trait]
pub trait Feed: Send + Sync {
    /// The venue this feed's data is sourced from.
    fn kind(&self) -> ExchangeKind;

    /// Connect the public market-data streams for `ticker` (lowercase, e.g.
    /// "btcusdt"). Safe to call again after a dropped stream (reconnect).
    async fn connect(&mut self, ticker: &str) -> Result<(), ExchangeError>;

    /// Next batch of normalized market events (trades and/or top-of-book).
    async fn next_events(&mut self) -> Result<Vec<MarketEvent>, ExchangeError>;

    /// Historical composite-bar closes (oldest first) used to warm the
    /// strategy's feature rings at startup.
    async fn historical_closes(&self, ticker: &str, bars: u32) -> Result<Vec<f64>, ExchangeError>;

    /// Last WebSocket ping→pong round-trip time in ms, measured with the
    /// MONOTONIC clock on the same socket the market data flows on —
    /// clock-skew-free by construction. `None` until the first pong arrives.
    /// This is the actionable network-latency number; raw clock skew mixes
    /// NTP offset with transit delay and is only a clock-health indicator.
    fn ws_rtt_ms(&self) -> Option<f64> {
        None
    }

    /// Tear down connections (drop is enough in practice).
    async fn close(&mut self) -> Result<(), ExchangeError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Order-routing broker trait
// ─────────────────────────────────────────────────────────────────────────────

/// Order-routing + wallet balance broker. Paper / dry-run mode never touches a
/// broker (fills are simulated inside the engine); only `Live` mode routes real
/// orders here.
#[async_trait::async_trait]
pub trait Broker: Send + Sync {
    /// The venue orders are routed to.
    fn kind(&self) -> ExchangeKind;

    /// Place an order.
    async fn place_order(&self, req: &OrderRequest) -> Result<OrderResponse, ExchangeError>;

    /// Cancel an order by venue-native id.
    async fn cancel_order(&self, order_id: &str) -> Result<(), ExchangeError>;

    /// Current wallet balances.
    async fn balances(&self) -> Result<Vec<Balance>, ExchangeError>;

    /// Tear down connections (drop is enough in practice).
    async fn close(&mut self) -> Result<(), ExchangeError>;
}

/// A no-op [`Broker`] used by the paper / replay paths so the loop always has a
/// routing handle. Any routing attempt errors so a live-trade cannot silently
/// fall through the dry-run broker.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopBroker;

#[async_trait::async_trait]
impl Broker for NoopBroker {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Parquet
    }

    async fn place_order(&self, req: &OrderRequest) -> Result<OrderResponse, ExchangeError> {
        Err(ExchangeError::Other(format!(
            "no-op broker: order routing is disabled in paper/replay mode ({})",
            req.symbol
        )))
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExchangeError> {
        Err(ExchangeError::Other(format!(
            "no-op broker: cannot cancel order {order_id} in paper/replay mode"
        )))
    }

    async fn balances(&self) -> Result<Vec<Balance>, ExchangeError> {
        Ok(Vec::new())
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory + symbol helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Construct a market-data feed. API credentials are read from the environment
/// (`BYBIT_API_KEY`/`BYBIT_API_SECRET`, `BFX_API_KEY`/`BFX_API_SECRET`) when the
/// venue is also used for routing; the public streams work without them.
pub fn create_feed(kind: ExchangeKind) -> Box<dyn Feed> {
    match kind {
        ExchangeKind::Bybit => Box::new(bybit::BybitExchange::new(
            std::env::var("BYBIT_API_KEY").ok(),
            std::env::var("BYBIT_API_SECRET").ok(),
        )),
        ExchangeKind::Bitfinex => Box::new(bitfinex::BitfinexExchange::new(
            std::env::var("BFX_API_KEY").ok(),
            std::env::var("BFX_API_SECRET").ok(),
        )),
        ExchangeKind::Binance => Box::new(binance::BinanceFeed::new()),
        // Only reachable if a caller drives a replay via `--exchange parquet`;
        // `live-test` builds its own feed path directly. Falls back to the
        // `LIVE_TEST_FEED` env var (connect will fail loudly if it is unset).
        ExchangeKind::Parquet => Box::new(parquet::ReplayExchange::new(
            &std::env::var("LIVE_TEST_FEED").unwrap_or_default(),
            None,
            None,
            None,
            false,
            None,
        )),
    }
}

/// Construct a broker (order-routing venue). Binance / parquet have no order
/// routing from this layer, so they fall back to a [`NoopBroker`].
pub fn create_broker(kind: ExchangeKind) -> Box<dyn Broker> {
    match kind {
        ExchangeKind::Bybit => Box::new(bybit::BybitExchange::new(
            std::env::var("BYBIT_API_KEY").ok(),
            std::env::var("BYBIT_API_SECRET").ok(),
        )),
        ExchangeKind::Bitfinex => Box::new(bitfinex::BitfinexExchange::new(
            std::env::var("BFX_API_KEY").ok(),
            std::env::var("BFX_API_SECRET").ok(),
        )),
        ExchangeKind::Binance | ExchangeKind::Parquet => Box::<NoopBroker>::default(),
    }
}

/// Construct a historical replay feed that reads a pre-converted parquet feed
/// (see `unified_runner.py live-test`). `start_ms` / `end_ms` / `limit` bound
/// the replayed window (useful for the multi-GB research parquet files); when
/// `stop` is provided and the feed is exhausted, the exchange flips it so
/// `run_execution_loop` drains and emits the final report — exactly like
/// Ctrl-C on a live session.
pub fn create_replay_feed(
    feed_path: &str,
    start_ms: Option<i64>,
    end_ms: Option<i64>,
    limit: Option<usize>,
    loop_feed: bool,
    stop: Option<Arc<AtomicBool>>,
) -> Box<dyn Feed> {
    Box::new(parquet::ReplayExchange::new(
        feed_path, start_ms, end_ms, limit, loop_feed, stop,
    ))
}

/// Ticker → venue-native symbol mapping helpers.
pub mod symbols {
    /// Bybit linear-perpetual symbol: "btcusdt" -> "BTCUSDT".
    pub fn bybit(ticker: &str) -> String {
        ticker.trim().to_ascii_uppercase()
    }

    /// Binance USD-M futures symbol: "btcusdt" -> "BTCUSDT".
    pub fn binance(ticker: &str) -> String {
        ticker.trim().to_ascii_uppercase()
    }

    /// Bitfinex trading-pair symbol: "btcusdt" -> "tBTCUST", "btcusd" -> "tBTCUSD".
    pub fn bitfinex(ticker: &str) -> String {
        let t = ticker.trim().to_ascii_uppercase();
        let (base, quote) = if let Some(b) = t.strip_suffix("USDT") {
            (b, "UST")
        } else if let Some(b) = t.strip_suffix("USD") {
            (b, "USD")
        } else {
            return format!("t{t}");
        };
        format!("t{base}{quote}")
    }
}

/// Milliseconds since the Unix epoch (used for nonces + timestamps).
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// HMAC-SHA256 hex digest — used by Bybit v5 request signing.
pub fn hmac_sha256(secret: &str, msg: &str) -> String {
    use hmac::{Hmac, Mac};
    type HmacSha256 = Hmac<sha2::Sha256>;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("hmac key");
    mac.update(msg.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// HMAC-SHA384 hex digest — used by Bitfinex v2 request signing.
pub fn hmac_sha384(secret: &str, msg: &str) -> String {
    use hmac::{Hmac, Mac};
    type HmacSha384 = Hmac<sha2::Sha384>;
    let mut mac = HmacSha384::new_from_slice(secret.as_bytes()).expect("hmac key");
    mac.update(msg.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Build a `reqwest` header map from string pairs (invalid values are skipped).
pub fn header_map(pairs: &[(&str, String)]) -> reqwest::header::HeaderMap {
    let mut map = reqwest::header::HeaderMap::new();
    for (key, value) in pairs {
        if let (Ok(name), Ok(val)) = (
            reqwest::header::HeaderName::from_bytes(key.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            map.insert(name, val);
        }
    }
    map
}
