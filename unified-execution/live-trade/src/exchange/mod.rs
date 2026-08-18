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

pub mod bitfinex;
pub mod bybit;

use serde::{Deserialize, Serialize};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Venue identity
// ─────────────────────────────────────────────────────────────────────────────

/// The set of trading venues the execution layer can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeKind {
    Bybit,
    Bitfinex,
}

impl ExchangeKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bybit" => Some(Self::Bybit),
            "bitfinex" => Some(Self::Bitfinex),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bybit => "bybit",
            Self::Bitfinex => "bitfinex",
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

// ─────────────────────────────────────────────────────────────────────────────
// The trait
// ─────────────────────────────────────────────────────────────────────────────

/// A trading venue: public market-data streams + private order routing.
///
/// Implementors keep all venue-specific wire formats internal and expose only
/// the normalized [`MarketEvent`] / [`OrderRequest`] types above.
#[async_trait::async_trait]
pub trait Exchange: Send + Sync {
    fn kind(&self) -> ExchangeKind;

    /// Connect the public market-data streams for `ticker` (lowercase, e.g.
    /// "btcusdt"). Safe to call again after a dropped stream (reconnect).
    async fn connect(&mut self, ticker: &str) -> Result<(), ExchangeError>;

    /// Next batch of normalized market events (trades and/or top-of-book).
    async fn next_events(&mut self) -> Result<Vec<MarketEvent>, ExchangeError>;

    /// Historical composite-bar closes (oldest first) used to warm the
    /// strategy's feature rings at startup.
    async fn historical_closes(&self, ticker: &str, bars: u32) -> Result<Vec<f64>, ExchangeError>;

    /// Place an order.
    async fn place_order(&self, req: &OrderRequest) -> Result<OrderResponse, ExchangeError>;

    /// Cancel an order by venue-native id.
    async fn cancel_order(&self, order_id: &str) -> Result<(), ExchangeError>;

    /// Current wallet balances.
    async fn balances(&self) -> Result<Vec<Balance>, ExchangeError>;

    /// Tear down connections (drop is enough in practice).
    async fn close(&mut self) -> Result<(), ExchangeError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory + symbol helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Construct an exchange implementation. API credentials are read from the
/// environment (`BYBIT_API_KEY`/`BYBIT_API_SECRET`, `BFX_API_KEY`/`BFX_API_SECRET`);
/// the public data streams work without any credentials.
pub fn create_exchange(kind: ExchangeKind) -> Box<dyn Exchange> {
    match kind {
        ExchangeKind::Bybit => Box::new(bybit::BybitExchange::new(
            std::env::var("BYBIT_API_KEY").ok(),
            std::env::var("BYBIT_API_SECRET").ok(),
        )),
        ExchangeKind::Bitfinex => Box::new(bitfinex::BitfinexExchange::new(
            std::env::var("BFX_API_KEY").ok(),
            std::env::var("BFX_API_SECRET").ok(),
        )),
    }
}

/// Ticker → venue-native symbol mapping helpers.
pub mod symbols {
    /// Bybit linear-perpetual symbol: "btcusdt" -> "BTCUSDT".
    pub fn bybit(ticker: &str) -> String {
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

