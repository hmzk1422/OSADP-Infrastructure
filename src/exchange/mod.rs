/// Client for the Python CCXT sidecar (tools/ccxt_sidecar.py).
///
/// The sidecar owns all exchange interaction (CCXT is Python-only). We
/// connect to it over a local TCP socket and receive newline-delimited
/// JSON: `hello`, `seed`, `seed_done`, `trade`, `orderbook`.
use anyhow::Context;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;

#[derive(Debug, Clone, Deserialize)]
pub struct HelloMsg {
    pub symbol: String,
    pub exchange: String,
    #[serde(default)]
    pub has_pro: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SeedMsg {
    pub bars: usize,
    pub closes: Vec<f32>,
    #[serde(default)]
    pub timestamps_ms: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TradeMsg {
    #[serde(rename = "ts_ms")]
    pub ts_ms: i64,
    pub price: f32,
    pub amount: f32,
    #[serde(default)]
    pub side: String, // "buy" | "sell"
}

#[derive(Debug, Clone, Deserialize)]
pub struct OrderbookMsg {
    #[serde(rename = "ts_ms")]
    pub ts_ms: i64,
    #[serde(default)]
    pub bids: Vec<(f32, f32)>,
    #[serde(default)]
    pub asks: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum FeedMsg {
    #[serde(rename = "hello")]
    Hello(HelloMsg),
    #[serde(rename = "seed")]
    Seed(SeedMsg),
    #[serde(rename = "seed_done")]
    SeedDone { bars: usize },
    #[serde(rename = "trade")]
    Trade(TradeMsg),
    #[serde(rename = "orderbook")]
    Orderbook(OrderbookMsg),
}

/// Read messages from the sidecar, dispatching each to `on_message`.
pub async fn run_feed_stream(
    host: &str,
    port: u16,
    mut on_message: impl FnMut(FeedMsg),
) -> anyhow::Result<()> {
    let addr = format!("{host}:{port}");
    let stream = TcpStream::connect(&addr)
        .await
        .with_context(|| format!("cannot connect to sidecar at {addr}"))?;
    log::info!("connected to sidecar at {addr}");

    let mut lines = BufReader::new(stream).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        let msg: FeedMsg =
            serde_json::from_str(&line).with_context(|| format!("bad sidecar JSON: {line}"))?;
        on_message(msg);
    }
    anyhow::bail!("sidecar closed the connection");
}