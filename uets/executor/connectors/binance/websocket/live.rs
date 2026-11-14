//! `live.rs` — the live capture pipeline: websocket -> merge -> CSV.
//!
//! This is the module that makes `executor.rs` produce the merged table. It is
//! the only place that knows about Binance's wire format *and* about writing
//! files; `stream_handler` stays free of both, and `core::feature` stays free
//! of both, so each piece remains independently testable.
//!
//! ```text
//!   wss aggTrade   --\
//!                     >-- parse --> ReorderBuffer --> Merger --> CsvRecorder
//!   wss depth@100ms--/                                |
//!                                                      +--> FeatureRow (core::feature)
//! ```
//!
//! # Why a REST seed is mandatory
//!
//! The diff-depth stream carries only *changed* levels, so without a REST
//! snapshot the book would be built from partial information and every
//! `book_state__*` column would be wrong in a way that is invisible in the
//! output. The book is therefore seeded from REST before the diffs are
//! applied, and re-seeded whenever a sequence gap is detected.
//!
//! # Two sources of truth for `lob__final_update_id`
//!
//! This uses `@depth@100ms` (diffs) plus a REST seed, so the id is the last
//! *diff* applied to a full-depth book -- the same meaning it has in the
//! historical pipeline. The partial `@depth20@100ms` stream is deliberately
//! not used: its `lastUpdateId` identifies a different object.

use std::collections::BTreeMap;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tungstenite::{Message, WebSocket, connect, stream::MaybeTlsStream};

use stream_handler::{
    BookSnapshot, CsvRecorder, DEFAULT_LATENESS_MS, DEFAULT_LEVELS, DiffFrame, MarketEvent, Merger,
    ReorderBuffer, Trade,
};

/// How long a capture may sit with zero rows before it is treated as broken.
///
/// A live feed that is working produces rows continuously, so a long silence
/// means the socket never connected rather than a quiet market.
const STALL_TIMEOUT: Duration = Duration::from_secs(30);
const READ_IDLE_TIMEOUT: Duration = Duration::from_secs(10);
const RESEED_RETRY: Duration = Duration::from_secs(3);
const SOURCE_FAILOVER_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MarketSource {
    Binance,
    Bybit,
}

impl MarketSource {
    fn index(self) -> usize {
        match self {
            Self::Binance => 0,
            Self::Bybit => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Binance => "binance",
            Self::Bybit => "bybit",
        }
    }
}

/// How the capture is configured.
#[derive(Clone, Debug)]
pub struct CaptureConfig {
    pub symbol: String,
    /// Where the CSV is written. Parent directories are created on demand.
    pub output: std::path::PathBuf,
    pub levels: usize,
    /// Reorder window for the two independently-arriving sockets.
    pub lateness_ms: i64,
    /// Stop after this many canonical rows. `None` runs until interrupted.
    pub max_rows: Option<u64>,
    /// Append to an existing capture instead of truncating it.
    pub append: bool,
    /// Strategy instances that should receive framed feature-input rows.
    pub fanout_instances: Vec<String>,
}

impl CaptureConfig {
    /// A capture for `symbol` writing into `output_dir`.
    ///
    /// The default filename is stamped with the UTC start time so repeated
    /// captures never silently overwrite each other.
    pub fn new(symbol: &str, output_dir: impl Into<std::path::PathBuf>) -> Self {
        let symbol = symbol.to_uppercase();
        let output = output_dir
            .into()
            .join(format!("{symbol}_{}.csv", now_utc_compact()));
        Self {
            symbol,
            output,
            levels: DEFAULT_LEVELS,
            lateness_ms: DEFAULT_LATENESS_MS,
            max_rows: None,
            append: false,
            fanout_instances: Vec::new(),
        }
    }
}

/// Shared, mutex-guarded capture state.
///
/// Both stream threads push into this, so emission order is decided in exactly
/// one place: the [`ReorderBuffer`].
struct Capture {
    merger: Merger,
    reorder: ReorderBuffer,
    csv: CsvRecorder,
    rows_written: u64,
    reseeds_pending: bool,
    fanout_skipped_invalid_book: u64,
    fanout: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
    fanout_failed: Arc<AtomicBool>,
    fanout_queue_depth: Arc<AtomicUsize>,
    received_times: BTreeMap<(i64, u8, u64), u64>,
    last_persistence_write_ms: f64,
    persistence_errors: u64,
    active_source: MarketSource,
    last_source_event: [Option<Instant>; 2],
    source_reseed_failed: [bool; 2],
    last_source_evaluation: Instant,
}

impl Capture {
    /// Merge one event and write whatever rows become emittable.
    #[allow(dead_code)]
    fn offer(&mut self, event: MarketEvent) -> Result<u64, String> {
        let received_ms = wall_time_ms();
        self.offer_received(event, received_ms)
    }

    fn offer_received(&mut self, event: MarketEvent, received_ms: u64) -> Result<u64, String> {
        self.offer_from_source(event, received_ms, MarketSource::Binance)
    }

    fn offer_from_source(
        &mut self,
        event: MarketEvent,
        received_ms: u64,
        source: MarketSource,
    ) -> Result<u64, String> {
        if matches!(event, MarketEvent::Diff(_)) {
            let now = Instant::now();
            self.last_source_event[source.index()] = Some(now);
            if source != self.active_source {
                if self.last_source_evaluation.elapsed() < Duration::from_secs(1) {
                    return Ok(0);
                }
                self.last_source_evaluation = now;
                let active_is_stale = self.source_reseed_failed[self.active_source.index()]
                    || self.last_source_event[self.active_source.index()]
                        .is_none_or(|last| last.elapsed() >= SOURCE_FAILOVER_TIMEOUT);
                if !active_is_stale {
                    return Ok(0);
                }
                eprintln!(
                    "[capture] market source failover: {} -> {} after {:?} without active book updates",
                    self.active_source.name(),
                    source.name(),
                    SOURCE_FAILOVER_TIMEOUT
                );
                self.active_source = source;
                self.source_reseed_failed[source.index()] = false;
                self.reorder.clear();
                self.received_times.clear();
                self.merger.mark_stale();
                self.reseeds_pending = true;
            }
        } else if source != self.active_source {
            return Ok(0);
        }

        let event_key = market_event_key(&event);
        self.received_times.insert(event_key, received_ms);
        if self.received_times.len() > 65_536 {
            self.received_times.pop_first();
        }
        let mut written = 0;
        for ready in self.reorder.push(event) {
            let received_ms = self
                .received_times
                .remove(&market_event_key(&ready))
                .unwrap_or(received_ms);
            let trade = match &ready {
                MarketEvent::Trade(trade) => Some(*trade),
                _ => None,
            };
            if let MarketEvent::Diff(frame) = &ready
                && let Some(last_update_id) = self.merger.state().last_update_id
                && frame.final_update_id <= last_update_id as u64
            {
                continue;
            }
            let (row, outcome) = self.merger.merge_event(ready);
            if outcome.needs_reseed {
                // Visible, not silent: the book is now wrong, so the next
                // row's book_state__* block cannot be trusted.
                self.reseeds_pending = true;
            }
            let write_started = std::time::Instant::now();
            if let Err(error) = self.csv.write_row(&mut self.merger, &row) {
                self.persistence_errors = self.persistence_errors.saturating_add(1);
                return Err(error.to_string());
            }
            self.last_persistence_write_ms = write_started.elapsed().as_secs_f64() * 1_000.0;
            if let Some(fanout) = &self.fanout {
                if !self.merger.state().book_valid() {
                    self.fanout_skipped_invalid_book =
                        self.fanout_skipped_invalid_book.saturating_add(1);
                } else {
                    let frame = match strategy_frame(
                        &self.merger,
                        trade,
                        row.timestamp_ns / 1_000_000,
                        received_ms,
                        self.last_persistence_write_ms,
                        self.persistence_errors,
                        self.fanout_queue_depth.load(Ordering::Relaxed),
                    ) {
                        Ok(frame) => frame,
                        Err(error) => {
                            self.fanout_failed.store(true, Ordering::Release);
                            return Err(error);
                        }
                    };
                    self.fanout_queue_depth.fetch_add(1, Ordering::Relaxed);
                    if let Err(error) = fanout.try_send(frame) {
                        self.fanout_queue_depth.fetch_sub(1, Ordering::Relaxed);
                        self.fanout_failed.store(true, Ordering::Release);
                        return Err(match error {
                            std::sync::mpsc::TrySendError::Full(_) => {
                                "strategy socket queue saturated; stopping rather than dropping market data".to_string()
                            }
                            std::sync::mpsc::TrySendError::Disconnected(_) => {
                                "strategy socket sender disconnected".to_string()
                            }
                        });
                    }
                }
            }
            written += 1;
        }
        self.rows_written += written;
        Ok(written)
    }

    /// Emit everything still buffered, in total order.
    fn flush_pending(&mut self) -> Result<u64, String> {
        let mut written = 0;
        for ready in self.reorder.flush() {
            let (row, _) = self.merger.merge_event(ready);
            self.csv
                .write_row(&mut self.merger, &row)
                .map_err(|e| e.to_string())?;
            written += 1;
        }
        self.rows_written += written;
        Ok(written)
    }
}

fn strategy_frame(
    merger: &Merger,
    trade: Option<Trade>,
    event_time_ms: i64,
    received_time_ms: u64,
    persistence_write_ms: f64,
    persistence_errors: u64,
    persistence_queue: usize,
) -> Result<Vec<u8>, String> {
    let state = merger.state();
    if !state.book_valid() {
        return Err("refusing to fan out a strategy row with an invalid order book".into());
    }
    let (bid_tick, bid_qty) = state
        .book
        .best_bid()
        .ok_or_else(|| "best bid unavailable".to_string())?;
    let (ask_tick, ask_qty) = state
        .book
        .best_ask()
        .ok_or_else(|| "best ask unavailable".to_string())?;
    let mut payload = format!(
        "event_time_ms:{event_time_ms}\nreceived_time_ms:{received_time_ms}\npersistence_write_ms:{persistence_write_ms}\npersistence_errors:{persistence_errors}\npersistence_queue:{persistence_queue}\nbid_price:{}\nbid_qty:{bid_qty}\nask_price:{}\nask_qty:{ask_qty}\n",
        stream_handler::tick_to_price(bid_tick),
        stream_handler::tick_to_price(ask_tick),
    );
    if let Some(trade) = trade {
        payload.push_str(&format!(
            "price:{}\namount:{}\nis_buyer_maker:{}\n",
            trade.price,
            trade.quantity,
            u8::from(trade.is_buyer_maker),
        ));
    } else {
        payload.push_str("amount:0\nis_buyer_maker:0\n");
    }
    let payload = payload.into_bytes();
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

fn market_event_key(event: &MarketEvent) -> (i64, u8, u64) {
    (
        event.event_time_ms(),
        event.source() as u8,
        event.source_sequence(),
    )
}

fn wall_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(unix)]
fn start_strategy_fanout(
    instances: &[String],
    symbol: &str,
    failed: Arc<AtomicBool>,
    queue_depth: Arc<AtomicUsize>,
) -> Result<std::sync::mpsc::SyncSender<Vec<u8>>, String> {
    use std::io::Write;
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;

    let mut paths = Vec::new();
    for instance in instances {
        let Some((strategy, asset)) = instance.split_once('@') else {
            return Err(format!(
                "invalid strategy instance '{instance}'; expected strategy@asset"
            ));
        };
        if !asset.eq_ignore_ascii_case(symbol) {
            return Err(format!(
                "feed for {symbol} cannot route strategy instance '{instance}'"
            ));
        }
        crate::live_strategy::load_strategy(strategy)?;
        let path = PathBuf::from(format!(
            "sockets/strategies/{}_{}.sock",
            strategy.to_ascii_lowercase(),
            symbol
        ));
        if !paths.contains(&path) {
            paths.push(path);
        }
    }
    if paths.is_empty() {
        return Err("strategy fan-out requires at least one configured instance".into());
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel::<Vec<u8>>(4096);
    let queue_depth_worker = Arc::clone(&queue_depth);
    std::thread::Builder::new()
        .name(format!("uds-fanout-{symbol}"))
        .spawn(move || {
            let mut streams: Vec<Option<UnixStream>> = paths.iter().map(|_| None).collect();
            while let Ok(frame) = receiver.recv() {
                queue_depth_worker.fetch_sub(1, Ordering::Relaxed);
                for (path, stream) in paths.iter().zip(streams.iter_mut()) {
                    if stream.is_none() {
                        match UnixStream::connect(path) {
                            Ok(connection) => *stream = Some(connection),
                            Err(error) => {
                                eprintln!("[fanout] connect {} failed: {error}; will retry on next frame", path.display());
                                *stream = None;
                                continue;
                            }
                        }
                    }
                    if let Some(connection) = stream {
                        if let Err(error) = connection.write_all(&frame) {
                            eprintln!("[fanout] write {} failed: {error}", path.display());
                            *stream = None; continue;
                        }
                    }
                }
            }
        })
        .map_err(|error| format!("start strategy fan-out: {error}"))?;
    Ok(sender)
}

fn warmup_lookback_for_instances(instances: &[String], symbol: &str) -> Result<u64, String> {
    let mut lookback = 0;
    for instance in instances {
        let Some((strategy, asset)) = instance.split_once('@') else {
            return Err(format!(
                "invalid strategy instance '{instance}'; expected strategy@asset"
            ));
        };
        if !asset.eq_ignore_ascii_case(symbol) {
            return Err(format!(
                "feed for {symbol} cannot warm strategy instance '{instance}'"
            ));
        }
        let bundle = crate::live_strategy::load_strategy(strategy)?;
        lookback = lookback.max(bundle.computer.warmup_lookback_secs());
    }
    if lookback == 0 && !instances.is_empty() {
        return Err("strategy warm-up lookback must be greater than zero".into());
    }
    Ok(lookback)
}

fn send_history_frame(
    sender: &std::sync::mpsc::SyncSender<Vec<u8>>,
    frame: Vec<u8>,
    queue_depth: &AtomicUsize,
    failed: &AtomicBool,
) -> Result<(), String> {
    queue_depth.fetch_add(1, Ordering::Relaxed);
    if let Err(error) = sender.send(frame) {
        queue_depth.fetch_sub(1, Ordering::Relaxed);
        failed.store(true, Ordering::Release);
        return Err(format!("send strategy warm-up frame: {error}"));
    }
    Ok(())
}

fn seed_strategies_from_rest(
    capture: &Arc<Mutex<Capture>>,
    instances: &[String],
    symbol: &str,
) -> Result<(), String> {
    if instances.is_empty() {
        return Ok(());
    }
    let lookback_secs = warmup_lookback_for_instances(instances, symbol)?;
    let history = crate::live_strategy::warmup::fetch_market_history(symbol, lookback_secs)?;
    let (sender, queue_depth, failed) = {
        let guard = capture
            .lock()
            .map_err(|error| format!("lock capture for strategy warm-up: {error}"))?;
        (
            guard
                .fanout
                .as_ref()
                .ok_or_else(|| "strategy warm-up has no fan-out sender".to_string())?
                .clone(),
            Arc::clone(&guard.fanout_queue_depth),
            Arc::clone(&guard.fanout_failed),
        )
    };

    send_history_frame(
        &sender,
        crate::live_strategy::warmup::encode_history_frame(None, "warmup_start")?,
        &queue_depth,
        &failed,
    )?;
    for row in &history {
        send_history_frame(
            &sender,
            crate::live_strategy::warmup::encode_history_frame(Some(row), "")?,
            &queue_depth,
            &failed,
        )?;
    }
    send_history_frame(
        &sender,
        crate::live_strategy::warmup::encode_history_frame(None, "warmup_end")?,
        &queue_depth,
        &failed,
    )?;
    eprintln!(
        "[warmup] seeded {} strategy instance(s) before live streams started",
        instances.len()
    );
    Ok(())
}

#[cfg(not(unix))]
fn start_strategy_fanout(
    _instances: &[String],
    _symbol: &str,
    _failed: Arc<AtomicBool>,
    _queue_depth: Arc<AtomicUsize>,
) -> Result<std::sync::mpsc::SyncSender<Vec<u8>>, String> {
    Err("live strategy fan-out requires Unix-domain sockets on a Unix host".into())
}

/// UTC timestamp as `YYYYMMDD_HHMMSS`, from the Unix epoch.
fn now_utc_compact() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}{m:02}{d:02}_{:02}{:02}{:02}",
        rem / 3_600,
        (rem % 3_600) / 60,
        rem % 60
    )
}

/// Fetch a REST depth snapshot to seed the book.
///
/// The `binance` crate is the only HTTP client already in the dependency
/// graph, so the seed uses it rather than adding a new dependency.
pub fn seed_from_rest(symbol: &str) -> Result<BookSnapshot, String> {
    use binance::api::*;
    use binance::futures::market::*;
    use binance::futures::model::*;

    // `Binance` is a trait, not a client. The concrete futures client is
    // `FuturesMarket`, which matches the `fstream` websocket used below.
    let client = <FuturesMarket as Binance>::new(None, None);
    let book: OrderBook = client
        .get_depth(symbol)
        .map_err(|e| format!("REST depth for {symbol} failed: {e}"))?;
    Ok(BookSnapshot {
        last_update_id: book.last_update_id,
        bids: book.bids.iter().map(|l| (l.price, l.qty)).collect(),
        asks: book.asks.iter().map(|l| (l.price, l.qty)).collect(),
    })
}

/// Fetch a Bybit V5 linear order-book snapshot for a source switch or reseed.
pub fn seed_bybit_from_rest(symbol: &str) -> Result<BookSnapshot, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|error| format!("build Bybit REST client: {error}"))?;
    let response: Value = client
        .get("https://api.bybit.com/v5/market/orderbook")
        .query(&[
            ("category", "linear"),
            ("symbol", &symbol.to_ascii_uppercase()),
            ("limit", "200"),
        ])
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("Bybit REST orderbook request failed: {error}"))?
        .json()
        .map_err(|error| format!("decode Bybit REST orderbook response: {error}"))?;
    if u64_of(response.get("retCode")) != Some(0) {
        return Err(format!(
            "Bybit REST orderbook rejected request: retCode={:?} retMsg={:?}",
            response.get("retCode"),
            response.get("retMsg")
        ));
    }
    let result = response
        .get("result")
        .ok_or_else(|| "Bybit REST orderbook response has no result".to_string())?;
    let parse_side = |key: &str| -> Result<Vec<(f64, f64)>, String> {
        let levels = result
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Bybit REST orderbook response has no {key} levels"))?;
        levels
            .iter()
            .map(|level| {
                let pair = level
                    .as_array()
                    .ok_or_else(|| format!("Bybit REST {key} level is not an array"))?;
                let price = num(pair.first())
                    .ok_or_else(|| format!("Bybit REST {key} level has invalid price"))?;
                let quantity = num(pair.get(1))
                    .ok_or_else(|| format!("Bybit REST {key} level has invalid quantity"))?;
                Ok((price, quantity))
            })
            .collect()
    };
    Ok(BookSnapshot {
        last_update_id: u64_of(result.get("u"))
            .ok_or_else(|| "Bybit REST orderbook response has no update ID".to_string())?,
        bids: parse_side("b")?,
        asks: parse_side("a")?,
    })
}

fn set_read_timeout(
    socket: &mut WebSocket<MaybeTlsStream<TcpStream>>,
    timeout: Duration,
) -> std::io::Result<()> {
    match socket.get_mut() {
        MaybeTlsStream::Plain(stream) => stream.set_read_timeout(Some(timeout)),
        MaybeTlsStream::NativeTls(stream) => stream.get_mut().set_read_timeout(Some(timeout)),
        _ => Err(std::io::Error::other(
            "unsupported TLS stream for websocket read timeout",
        )),
    }
}

fn is_idle_read_timeout(error: &tungstenite::Error) -> bool {
    matches!(
        error,
        tungstenite::Error::Io(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            )
    )
}

fn mark_market_stream_disconnected(capture: &mut Capture, source: MarketSource) {
    if capture.active_source == source {
        capture.merger.mark_stale();
        capture.reseeds_pending = true;
    }
}

/// Reconnect a stream after close, error, or an idle read timeout.
fn spawn_stream(
    url: String,
    subscribe: Option<String>,
    mut on_msg: impl FnMut(&str) + Send + 'static,
    on_disconnect: impl Fn() + Send + 'static,
    keep_running: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        while keep_running.load(Ordering::Acquire) {
            match connect(url.as_str()) {
                Ok((mut ws, _)) => {
                    eprintln!("[capture] connected: {url}");
                    if let Some(message) = subscribe.as_ref()
                        && let Err(error) = ws.send(Message::Text(message.clone()))
                    {
                        eprintln!("[capture] subscription failed: {error} ({url})");
                        on_disconnect();
                        std::thread::sleep(Duration::from_secs(3));
                        continue;
                    }
                    match set_read_timeout(&mut ws, READ_IDLE_TIMEOUT) {
                        Ok(()) => loop {
                            match ws.read() {
                                Ok(Message::Text(text)) => on_msg(&text),
                                Ok(Message::Ping(payload)) => {
                                    if let Err(error) = ws.send(Message::Pong(payload)) {
                                        eprintln!(
                                            "[capture] websocket pong failed: {error} ({url})"
                                        );
                                        break;
                                    }
                                }
                                Ok(Message::Pong(_))
                                | Ok(Message::Binary(_))
                                | Ok(Message::Frame(_)) => (),
                                Ok(Message::Close(frame)) => {
                                    eprintln!(
                                        "[capture] closed by server: {:?} ({url})",
                                        frame.map(|f| f.code)
                                    );
                                    break;
                                }
                                Err(error) if is_idle_read_timeout(&error) => (),
                                Err(error) => {
                                    eprintln!("[capture] socket error: {error} ({url})");
                                    break;
                                }
                            }
                            if !keep_running.load(Ordering::Acquire) {
                                break;
                            }
                        },
                        Err(error) => {
                            eprintln!(
                                "[capture] set websocket read timeout failed: {error} ({url})"
                            );
                        }
                    }
                }
                Err(error) => eprintln!("[capture] connect failed: {error} ({url})"),
            }
            if !keep_running.load(Ordering::Acquire) {
                break;
            }
            on_disconnect();
            eprintln!("[capture] reconnecting in 3s...");
            std::thread::sleep(Duration::from_secs(3));
        }
    });
}

/// Binance sends numbers as JSON strings; parse either form.
fn num(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn u64_of(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Unwrap a combined-stream envelope: `{"stream":..,"data":{..}}` -> data.
fn payload(text: &str) -> Option<Value> {
    let value: Value = serde_json::from_str(text).ok()?;
    Some(match value.get("data") {
        Some(inner) => inner.clone(),
        None => value,
    })
}

/// Parse one `aggTrade` push into a [`Trade`].
///
/// `m = true` means the buyer was the maker, i.e. the taker sold.
pub fn parse_trade(text: &str) -> Option<MarketEvent> {
    let v = payload(text)?;
    if v.get("e").and_then(Value::as_str) != Some("aggTrade") {
        return None;
    }
    Some(MarketEvent::Trade(Trade {
        event_time_ms: num(v.get("E"))? as i64,
        transact_time_ms: num(v.get("T")).unwrap_or_else(|| num(v.get("E")).unwrap_or(0.0)) as i64,
        agg_trade_id: u64_of(v.get("a"))?,
        first_trade_id: u64_of(v.get("f")).unwrap_or(0),
        last_trade_id: u64_of(v.get("l")).unwrap_or(0),
        price: num(v.get("p"))?,
        quantity: num(v.get("q"))?,
        is_buyer_maker: v.get("m").and_then(Value::as_bool).unwrap_or(false),
    }))
}

/// Parse one `depthUpdate` push into a [`DiffFrame`].
///
/// A quantity of `0.0` is meaningful here: it is how Binance removes a level,
/// so it is passed through rather than filtered out.
pub fn parse_diff(text: &str) -> Option<MarketEvent> {
    let v = payload(text)?;
    if v.get("e").and_then(Value::as_str) != Some("depthUpdate") {
        return None;
    }
    let mut updates = Vec::new();
    for (key, is_bid) in [("b", true), ("a", false)] {
        let Some(Value::Array(levels)) = v.get(key) else {
            continue;
        };
        for level in levels {
            let Value::Array(pair) = level else { continue };
            if pair.len() < 2 {
                continue;
            }
            let (Some(price), Some(quantity)) = (num(pair.first()), num(pair.get(1))) else {
                continue;
            };
            updates.push(stream_handler::BookUpdate {
                is_bid,
                price,
                quantity,
            });
        }
    }
    Some(MarketEvent::Diff(DiffFrame {
        event_time_ms: num(v.get("E"))? as i64,
        first_update_id: u64_of(v.get("U"))?,
        final_update_id: u64_of(v.get("u"))?,
        previous_update_id: Some(u64_of(v.get("pu"))?),
        updates,
    }))
}

/// Parse a Bybit V5 linear order-book delta into the shared merger event.
pub fn parse_bybit_diff(text: &str) -> Option<MarketEvent> {
    let envelope: Value = serde_json::from_str(text).ok()?;
    if envelope.get("type").and_then(Value::as_str) != Some("delta")
        || !envelope
            .get("topic")
            .and_then(Value::as_str)
            .is_some_and(|topic| topic.starts_with("orderbook."))
    {
        return None;
    }
    let data = envelope.get("data")?;
    let update_id = u64_of(data.get("u"))?;
    let mut updates = Vec::new();
    for (key, is_bid) in [("b", true), ("a", false)] {
        let Some(Value::Array(levels)) = data.get(key) else {
            continue;
        };
        for level in levels {
            let Value::Array(pair) = level else { continue };
            if pair.len() < 2 {
                continue;
            }
            let (Some(price), Some(quantity)) = (num(pair.first()), num(pair.get(1))) else {
                continue;
            };
            updates.push(stream_handler::BookUpdate {
                is_bid,
                price,
                quantity,
            });
        }
    }
    Some(MarketEvent::Diff(DiffFrame {
        event_time_ms: i64::try_from(u64_of(envelope.get("ts"))?).ok()?,
        first_update_id: update_id,
        final_update_id: update_id,
        previous_update_id: None,
        updates,
    }))
}

/// Parse the possibly batched Bybit V5 public-trade message.
pub fn parse_bybit_trades(text: &str) -> Vec<MarketEvent> {
    let Ok(envelope) = serde_json::from_str::<Value>(text) else {
        return Vec::new();
    };
    if !envelope
        .get("topic")
        .and_then(Value::as_str)
        .is_some_and(|topic| topic.starts_with("publicTrade."))
    {
        return Vec::new();
    }
    let Some(Value::Array(items)) = envelope.get("data") else {
        return Vec::new();
    };
    let envelope_time = u64_of(envelope.get("ts")).unwrap_or_default();
    items
        .iter()
        .filter_map(|item| {
            let trade_id = item.get("i").and_then(Value::as_str)?;
            let transact_time_ms = u64_of(item.get("T")).unwrap_or(envelope_time);
            let event_time_ms = i64::try_from(transact_time_ms).ok()?;
            Some(MarketEvent::Trade(Trade {
                event_time_ms,
                transact_time_ms: event_time_ms,
                agg_trade_id: stable_trade_id(trade_id),
                first_trade_id: 0,
                last_trade_id: 0,
                price: num(item.get("p"))?,
                quantity: num(item.get("v"))?,
                is_buyer_maker: item.get("S").and_then(Value::as_str) == Some("Sell"),
            }))
        })
        .collect()
}

fn stable_trade_id(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

/// Run a live capture until interrupted, or until `max_rows` is reached.
pub fn run_capture(config: CaptureConfig) -> Result<(), String> {
    let symbol = config.symbol.to_lowercase();
    let keep_running = Arc::new(AtomicBool::new(true));
    let rows_seen = Arc::new(AtomicU64::new(0));
    let fanout_failed = Arc::new(AtomicBool::new(false));
    let fanout_queue_depth = Arc::new(AtomicUsize::new(0));
    let fanout = if config.fanout_instances.is_empty() {
        None
    } else {
        Some(start_strategy_fanout(
            &config.fanout_instances,
            &config.symbol.to_ascii_lowercase(),
            Arc::clone(&fanout_failed),
            Arc::clone(&fanout_queue_depth),
        )?)
    };

    let csv = if config.append {
        CsvRecorder::append(&config.output)
    } else {
        CsvRecorder::create(&config.output)
    }
    .map_err(|e| e.to_string())?
    .flush_every(64);

    eprintln!("[capture] output   {}", config.output.display());
    eprintln!("[capture] symbol   {}", config.symbol);
    eprintln!("[capture] columns  {}", stream_handler::schema().len());

    let capture = Arc::new(Mutex::new(Capture {
        merger: Merger::new(config.levels),
        reorder: ReorderBuffer::new(config.lateness_ms, 65_536),
        csv,
        rows_written: 0,
        reseeds_pending: false,
        fanout,
        fanout_failed: Arc::clone(&fanout_failed),
        fanout_skipped_invalid_book: 0,
        fanout_queue_depth: Arc::clone(&fanout_queue_depth),
        received_times: BTreeMap::new(),
        last_persistence_write_ms: 0.0,
        persistence_errors: 0,
        active_source: MarketSource::Binance,
        last_source_event: [None, None],
        source_reseed_failed: [false, false],
        last_source_evaluation: Instant::now(),
    }));

    // Write the header immediately, before any data. Deferring it to the first
    // row leaves a zero-byte file whenever a capture connects but never
    // receives data, and a headerless CSV is readable by nothing downstream.
    {
        let mut guard = capture.lock().expect("capture mutex poisoned");
        guard.csv.write_header().map_err(|e| e.to_string())?;
        guard.csv.flush().map_err(|e| e.to_string())?;
    }

    // Seed first. The diff stream alone cannot build a correct book.
    match seed_from_rest(&config.symbol) {
        Ok(snapshot) => {
            let mut guard = capture.lock().expect("capture mutex poisoned");
            // Applied straight to the merger, never through the reorder
            // buffer: a snapshot carries no exchange event time, so ordering
            // it against live events would corrupt the watermark.
            guard.merger.apply_seed(&snapshot);
            eprintln!("[capture] book seeded from REST");
        }
        Err(e) => {
            eprintln!("[capture] WARNING: REST seed failed: {e}");
            let mut guard = capture.lock().map_err(|error| {
                format!("capture mutex poisoned after initial REST seed failure: {error}")
            })?;
            mark_market_stream_disconnected(&mut guard, MarketSource::Binance);
            guard.source_reseed_failed[MarketSource::Binance.index()] = true;
        }
    }

    seed_strategies_from_rest(&capture, &config.fanout_instances, &config.symbol)?;

    let max_rows = config.max_rows;
    spawn_stream(
        format!("wss://fstream.binance.com/stream?streams={symbol}@aggTrade"),
        None,
        {
            let shared = Arc::clone(&capture);
            let seen = Arc::clone(&rows_seen);
            let keep = Arc::clone(&keep_running);
            move |text| {
                let received_ms = wall_time_ms();
                if let Some(event) = parse_trade(text) {
                    push(
                        &shared,
                        &seen,
                        &keep,
                        max_rows,
                        event,
                        received_ms,
                        MarketSource::Binance,
                    );
                }
            }
        },
        || {},
        Arc::clone(&keep_running),
    );
    let disconnected_capture = Arc::clone(&capture);
    let disconnected_keep_running = Arc::clone(&keep_running);
    spawn_stream(
        format!("wss://fstream.binance.com/stream?streams={symbol}@depth@100ms"),
        None,
        {
            let shared = Arc::clone(&capture);
            let seen = Arc::clone(&rows_seen);
            let keep = Arc::clone(&keep_running);
            move |text| {
                let received_ms = wall_time_ms();
                if let Some(event) = parse_diff(text) {
                    push(
                        &shared,
                        &seen,
                        &keep,
                        max_rows,
                        event,
                        received_ms,
                        MarketSource::Binance,
                    );
                }
            }
        },
        move || match disconnected_capture.lock() {
            Ok(mut guard) => mark_market_stream_disconnected(&mut guard, MarketSource::Binance),
            Err(error) => {
                eprintln!("[capture] cannot mark order book stale after disconnect: {error}");
                disconnected_keep_running.store(false, Ordering::Release);
            }
        },
        Arc::clone(&keep_running),
    );

    let bybit_symbol = config.symbol.to_ascii_uppercase();
    let bybit_trade_subscription = serde_json::json!({
        "op": "subscribe",
        "args": [format!("publicTrade.{bybit_symbol}")]
    })
    .to_string();
    spawn_stream(
        "wss://stream.bybit.com/v5/public/linear".to_string(),
        Some(bybit_trade_subscription),
        {
            let shared = Arc::clone(&capture);
            let seen = Arc::clone(&rows_seen);
            let keep = Arc::clone(&keep_running);
            move |text| {
                let received_ms = wall_time_ms();
                for event in parse_bybit_trades(text) {
                    push(
                        &shared,
                        &seen,
                        &keep,
                        max_rows,
                        event,
                        received_ms,
                        MarketSource::Bybit,
                    );
                }
            }
        },
        || {},
        Arc::clone(&keep_running),
    );

    let bybit_book_subscription = serde_json::json!({
        "op": "subscribe",
        "args": [format!("orderbook.50.{bybit_symbol}")]
    })
    .to_string();
    let disconnected_capture = Arc::clone(&capture);
    let disconnected_keep_running = Arc::clone(&keep_running);
    spawn_stream(
        "wss://stream.bybit.com/v5/public/linear".to_string(),
        Some(bybit_book_subscription),
        {
            let shared = Arc::clone(&capture);
            let seen = Arc::clone(&rows_seen);
            let keep = Arc::clone(&keep_running);
            move |text| {
                let received_ms = wall_time_ms();
                if let Some(event) = parse_bybit_diff(text) {
                    push(
                        &shared,
                        &seen,
                        &keep,
                        max_rows,
                        event,
                        received_ms,
                        MarketSource::Bybit,
                    );
                }
            }
        },
        move || match disconnected_capture.lock() {
            Ok(mut guard) => mark_market_stream_disconnected(&mut guard, MarketSource::Bybit),
            Err(error) => {
                eprintln!("[capture] cannot mark Bybit book stale after disconnect: {error}");
                disconnected_keep_running.store(false, Ordering::Release);
            }
        },
        Arc::clone(&keep_running),
    );

    // Recover the book from a detected sequence gap without a restart.
    {
        let shared = Arc::clone(&capture);
        let sym = config.symbol.clone();
        let keep = Arc::clone(&keep_running);
        std::thread::spawn(move || {
            while keep.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                let stale = match shared.lock() {
                    Ok(guard) => guard.reseeds_pending,
                    Err(error) => {
                        eprintln!("[capture] cannot inspect reseed state: {error}");
                        keep.store(false, Ordering::Release);
                        return;
                    }
                };
                if !stale {
                    continue;
                }
                let source = match shared.lock() {
                    Ok(guard) => guard.active_source,
                    Err(error) => {
                        eprintln!("[capture] cannot inspect active feed source: {error}");
                        keep.store(false, Ordering::Release);
                        return;
                    }
                };
                let reseed = match source {
                    MarketSource::Binance => seed_from_rest(&sym),
                    MarketSource::Bybit => seed_bybit_from_rest(&sym),
                };
                match reseed {
                    Ok(snapshot) => match shared.lock() {
                        Ok(mut guard) => {
                            if guard.active_source != source {
                                continue;
                            }
                            guard.merger.apply_seed(&snapshot);
                            guard.reseeds_pending = false;
                            guard.source_reseed_failed[source.index()] = false;
                            eprintln!(
                                "[capture] {} book re-seeded after a stream interruption or sequence gap",
                                source.name()
                            );
                        }
                        Err(error) => {
                            eprintln!("[capture] cannot apply REST reseed: {error}");
                            keep.store(false, Ordering::Release);
                            return;
                        }
                    },
                    Err(error) => {
                        match shared.lock() {
                            Ok(mut guard) if guard.active_source == source => {
                                guard.source_reseed_failed[source.index()] = true;
                            }
                            Ok(_) => {}
                            Err(lock_error) => {
                                eprintln!(
                                    "[capture] cannot record failed REST reseed: {lock_error}"
                                );
                                keep.store(false, Ordering::Release);
                                return;
                            }
                        }
                        eprintln!(
                            "[capture] re-seed failed: {error}; retrying in {RESEED_RETRY:?}"
                        );
                        std::thread::sleep(RESEED_RETRY);
                    }
                }
            }
        });
    }

    // Stall guard. Without it a capture that cannot reach the exchange (no
    // network, blocked host, wrong endpoint) spins forever writing an empty
    // file, which looks like "the market is quiet" rather than "nothing is
    // connected". Refuse to keep a capture alive that never saw a row.
    let started = std::time::Instant::now();
    let mut last_beat: u64 = 0;
    loop {
        std::thread::sleep(Duration::from_millis(200));
        let total = rows_seen.load(Ordering::Relaxed);
        if let Some(limit) = max_rows
            && total >= limit
        {
            break;
        }
        if fanout_failed.load(Ordering::Acquire) {
            keep_running.store(false, Ordering::Release);
            return Err("strategy socket disconnected; feed process stopping".into());
        }
        if !keep_running.load(Ordering::Relaxed) {
            break;
        }
        let (both_reseeds_failed, active_reseed_failed, fallback_stale) = {
            let guard = capture.lock().expect("capture mutex poisoned");
            let active_index = guard.active_source.index();
            let fallback_index = 1 - active_index;
            (
                guard.source_reseed_failed.iter().all(|failed| *failed),
                guard.source_reseed_failed[active_index],
                guard.last_source_event[fallback_index]
                    .is_none_or(|last| last.elapsed() >= STALL_TIMEOUT),
            )
        };
        if both_reseeds_failed {
            keep_running.store(false, Ordering::Release);
            return Err(
                "Binance and Bybit REST order-book reseeds both failed; refusing to run the strategy"
                    .into(),
            );
        }
        if active_reseed_failed && fallback_stale && started.elapsed() >= STALL_TIMEOUT {
            keep_running.store(false, Ordering::Release);
            return Err(
                "active market source could not reseed and the alternate feed is unavailable; refusing to run the strategy"
                    .into(),
            );
        }
        if total > 0 {
            let no_healthy_source = {
                let guard = capture.lock().expect("capture mutex poisoned");
                guard
                    .last_source_event
                    .iter()
                    .all(|last| last.is_none_or(|last| last.elapsed() >= STALL_TIMEOUT))
            };
            if no_healthy_source {
                keep_running.store(false, Ordering::Release);
                return Err(format!(
                    "Binance and Bybit order books have both been stale for {:?}; refusing to run the strategy on jumped data",
                    STALL_TIMEOUT
                ));
            }
        }
        if total == 0 && started.elapsed() > STALL_TIMEOUT {
            eprintln!(
                "[capture] no rows after {:?}: cannot reach the exchange. \
                 Check the network, then retry.",
                STALL_TIMEOUT
            );
            keep_running.store(false, Ordering::Relaxed);
            return Err(format!(
                "no market data received within {:?}",
                STALL_TIMEOUT
            ));
        }

        // Heartbeat. Without it a healthy capture is indistinguishable from a
        // hung one: both print nothing. Once a second, show the row count and
        // whether the book is trustworthy, so a silent file growing on disk is
        // backed by evidence the feed is actually live.
        if started.elapsed().as_secs() > last_beat {
            last_beat = started.elapsed().as_secs();
            let (book_ok, gaps, late, seeds, skipped) = {
                let guard = capture.lock().expect("capture mutex poisoned");
                let c = guard.merger.counters();
                (
                    guard.merger.state().book_valid(),
                    c.gaps,
                    c.late_dropped,
                    c.seeds_applied,
                    guard.fanout_skipped_invalid_book,
                )
            };
            eprintln!(
                "[capture] rows={total} book={} gaps={gaps} late={late} seeds={seeds} strategy_skipped_invalid_book={skipped}",
                if book_ok { "valid" } else { "UNSEEDED/STALE" }
            );
        }
    }
    keep_running.store(false, Ordering::Relaxed);

    let mut guard = capture.lock().expect("capture mutex poisoned");
    let trailing = guard.flush_pending()?;
    guard.csv.flush().map_err(|e| e.to_string())?;
    let counters = guard.merger.counters();
    eprintln!(
        "[capture] wrote {} rows ({} trailing)",
        guard.csv.rows_written(),
        trailing
    );
    eprintln!(
        "[capture] applied={} diffs={} trades={} late_dropped={} gaps={} seeds={}",
        counters.events_applied,
        counters.diff_frames,
        counters.trades,
        counters.late_dropped,
        counters.gaps,
        counters.seeds_applied
    );
    Ok(())
}

/// Merge one event and count the row against the cap.
fn push(
    shared: &Arc<Mutex<Capture>>,
    rows_seen: &Arc<AtomicU64>,
    keep: &Arc<AtomicBool>,
    max_rows: Option<u64>,
    event: MarketEvent,
    received_ms: u64,
    source: MarketSource,
) {
    let Ok(mut guard) = shared.lock() else { return };
    match guard.offer_from_source(event, received_ms, source) {
        Ok(written) => {
            let total = rows_seen.fetch_add(written, Ordering::Relaxed) + written;
            if max_rows.is_some_and(|limit| total >= limit) {
                keep.store(false, Ordering::Relaxed);
            }
        }
        Err(e) => {
            eprintln!("[capture] write failed: {e}");
            keep.store(false, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_lookback_is_selected_from_any_loaded_strategy() {
        let instances = vec!["example_strategy@btcusdt".to_string()];
        assert_eq!(
            warmup_lookback_for_instances(&instances, "BTCUSDT").unwrap(),
            crate::live_strategy::DEFAULT_WARMUP_LOOKBACK_SECS
        );
    }

    #[test]
    fn websocket_idle_read_timeout_does_not_disconnect_stream() {
        assert!(is_idle_read_timeout(&tungstenite::Error::Io(
            std::io::Error::from(std::io::ErrorKind::WouldBlock)
        )));
        assert!(is_idle_read_timeout(&tungstenite::Error::Io(
            std::io::Error::from(std::io::ErrorKind::TimedOut)
        )));
        assert!(!is_idle_read_timeout(&tungstenite::Error::Io(
            std::io::Error::from(std::io::ErrorKind::ConnectionReset)
        )));
    }

    #[test]
    fn parses_a_combined_stream_agg_trade() {
        let text = r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1751245200354,"a":2773899591,"p":"108665.6","q":"0.002","f":6438488841,"l":6438488841,"T":1751245200354,"m":true}}"#;
        match parse_trade(text).expect("parsed") {
            MarketEvent::Trade(t) => {
                assert_eq!(t.agg_trade_id, 2_773_899_591);
                assert_eq!(t.price, 108665.6);
                assert_eq!(t.quantity, 0.002);
                assert_eq!(t.event_time_ms, 1_751_245_200_354);
                assert!(t.is_buyer_maker, "m=true means the buyer was the maker");
            }
            other => panic!("expected a trade, got {other:?}"),
        }
    }

    #[test]
    fn parses_a_bare_agg_trade_frame() {
        let text = r#"{"e":"aggTrade","E":10,"a":5,"p":"1.5","q":"2.0","m":false}"#;
        assert!(parse_trade(text).is_some());
    }

    #[test]
    fn rejects_a_non_trade_frame() {
        assert!(parse_trade(r#"{"e":"depthUpdate","E":1}"#).is_none());
        assert!(parse_trade("not json").is_none());
    }

    #[test]
    fn parses_a_diff_frame_with_both_sides() {
        let text = r#"{"e":"depthUpdate","E":1751245200406,"U":1,"u":10,"pu":0,"b":[["108665.5","9.141"]],"a":[["108665.6","12.556"]]}"#;
        match parse_diff(text).expect("parsed") {
            MarketEvent::Diff(f) => {
                assert_eq!(f.first_update_id, 1);
                assert_eq!(f.final_update_id, 10);
                assert_eq!(f.previous_update_id, Some(0));
                assert_eq!(f.updates.len(), 2);
                assert!(f.updates[0].is_bid, "bids come first");
                assert_eq!(f.updates[0].price, 108665.5);
                assert!(!f.updates[1].is_bid);
            }
            other => panic!("expected a diff, got {other:?}"),
        }
    }

    #[test]
    fn diff_keeps_zero_quantity_so_levels_can_be_removed() {
        // A zero qty removes a level; dropping it would leave a phantom level.
        let text = r#"{"e":"depthUpdate","E":1,"U":1,"u":2,"pu":0,"b":[["100.0","0"]],"a":[]}"#;
        match parse_diff(text).expect("parsed") {
            MarketEvent::Diff(f) => {
                assert_eq!(f.updates.len(), 1);
                assert_eq!(f.updates[0].quantity, 0.0);
            }
            other => panic!("expected a diff, got {other:?}"),
        }
    }

    #[test]
    fn malformed_levels_are_skipped_not_fatal() {
        let text = r#"{"e":"depthUpdate","E":1,"U":1,"u":2,"pu":0,"b":[["bad","1"],["2.0"],"3",["4.0","5.0"]],"a":[]}"#;
        match parse_diff(text).expect("parsed") {
            MarketEvent::Diff(f) => assert_eq!(f.updates.len(), 1),
            other => panic!("expected a diff, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_frame_missing_its_final_update_id() {
        // Without `u` the sequence cannot advance, so the frame is refused
        // rather than silently resetting continuity.
        let text = r#"{"e":"depthUpdate","E":1,"U":1,"u":2,"b":[],"a":[]}"#;
        assert!(parse_diff(text).is_none());
    }

    #[test]
    fn parses_bybit_v5_orderbook_delta_into_the_shared_event_schema() {
        let text = r#"{"topic":"orderbook.50.BTCUSDT","type":"delta","ts":1751245200406,"data":{"s":"BTCUSDT","u":10,"seq":20,"b":[["108665.5","9.141"]],"a":[["108665.6","0"]]}}"#;
        match parse_bybit_diff(text).expect("parsed") {
            MarketEvent::Diff(frame) => {
                assert_eq!(frame.event_time_ms, 1_751_245_200_406);
                assert_eq!(frame.first_update_id, 10);
                assert_eq!(frame.final_update_id, 10);
                assert_eq!(frame.updates.len(), 2);
                assert!(frame.updates[0].is_bid);
                assert_eq!(frame.updates[0].price, 108_665.5);
                assert_eq!(frame.updates[1].quantity, 0.0);
            }
            other => panic!("expected a diff, got {other:?}"),
        }
    }

    #[test]
    fn parses_batched_bybit_trades_and_maps_taker_side() {
        let text = r#"{"topic":"publicTrade.BTCUSDT","ts":1751245200406,"data":[{"T":1751245200405,"s":"BTCUSDT","S":"Sell","v":"0.02","p":"100.5","i":"trade-a"},{"T":1751245200406,"s":"BTCUSDT","S":"Buy","v":"0.03","p":"100.6","i":"trade-b"}]}"#;
        let events = parse_bybit_trades(text);
        assert_eq!(events.len(), 2);
        match &events[0] {
            MarketEvent::Trade(trade) => {
                assert_eq!(trade.price, 100.5);
                assert_eq!(trade.quantity, 0.02);
                assert!(trade.is_buyer_maker, "Sell is the taker side");
            }
            other => panic!("expected a trade, got {other:?}"),
        }
        match &events[1] {
            MarketEvent::Trade(trade) => {
                assert_eq!(trade.price, 100.6);
                assert!(!trade.is_buyer_maker, "Buy is the taker side");
            }
            other => panic!("expected a trade, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod end_to_end {
    use super::*;
    use crate::core::feature::{
        CanonicalIntegers, CanonicalTop, CanonicalValues, row_from_canonical,
    };
    use stream_handler::schema;

    /// Drive real wire frames through the whole pipeline, offline.
    ///
    /// This is the test that proves the claim the recorder makes: what the
    /// capture writes is the merged table, column for column. It uses the
    /// actual `parse_*` functions and the actual `Merger`/`CsvRecorder`, so a
    /// schema change anywhere in the chain fails here.
    fn run_offline(path: &std::path::Path) -> Vec<Vec<String>> {
        let mut capture = Capture {
            merger: Merger::new(DEFAULT_LEVELS),
            reorder: ReorderBuffer::new(1_000, 1024),
            csv: CsvRecorder::create(path).expect("create"),
            rows_written: 0,
            reseeds_pending: false,
            fanout: None,
            fanout_failed: Arc::new(AtomicBool::new(false)),
            fanout_skipped_invalid_book: 0,
            fanout_queue_depth: Arc::new(AtomicUsize::new(0)),
            received_times: BTreeMap::new(),
            last_persistence_write_ms: 0.0,
            persistence_errors: 0,
            active_source: MarketSource::Binance,
            last_source_event: [None, None],
            source_reseed_failed: [false, false],
            last_source_evaluation: Instant::now(),
        };

        // A REST seed, applied DIRECTLY to the merger. It must not go
        // through the reorder buffer: a snapshot has no exchange event time
        // (`BookSnapshot::event_time_ms` reports the update id), so it would
        // set the watermark far into the future and make every real event
        // look late, dropping all of them.
        capture.merger.apply_seed(&BookSnapshot {
            last_update_id: 1,
            bids: vec![(108_663.2, 0.095), (108_662.9, 0.553), (108_653.8, 0.025)],
            asks: vec![(108_666.0, 0.004), (108_668.8, 0.002), (108_668.9, 0.562)],
        });

        for text in [
            r#"{"e":"depthUpdate","E":1751245200083,"U":2,"u":3,"pu":1,"b":[["108663.2","9.141"]],"a":[["108665.6","12.556"]]}"#,
            r#"{"e":"aggTrade","E":1751245200354,"a":2773899591,"p":"108665.6","q":"0.002","f":6438488841,"l":6438488841,"T":1751245200354,"m":true}"#,
            r#"{"e":"depthUpdate","E":1751245200406,"U":4,"u":5,"pu":3,"b":[["108653.8","0.025"]],"a":[["108666.0","0.004"]]}"#,
            r#"{"e":"aggTrade","E":1751245200441,"a":2773899592,"p":"108665.5","q":"0.303","f":6438488842,"l":6438488842,"T":1751245200441,"m":false}"#,
        ] {
            let event = parse_diff(text)
                .or_else(|| parse_trade(text))
                .expect("parsed frame");
            capture.offer(event).expect("offer");
        }
        capture.flush_pending().expect("flush");
        capture.csv.flush().expect("csv flush");

        let text = std::fs::read_to_string(path).expect("read back");
        text.lines()
            .map(|line| line.split(',').map(str::to_string).collect())
            .collect()
    }

    fn temp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("uets_live_e2e_{name}.csv"));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn written_header_matches_the_merger_schema_exactly() {
        let path = temp("header");
        let rows = run_offline(&path);
        let header = &rows[0];
        let schema = schema();
        assert_eq!(header.len(), 128, "128 columns");
        for (i, name) in schema.iter().enumerate() {
            assert_eq!(&header[i], name, "column {i} must match schema()");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_data_row_has_128_fields_including_the_header_row() {
        let path = temp("width");
        let rows = run_offline(&path);
        assert!(rows.len() >= 3, "header plus several rows");
        for row in &rows {
            assert_eq!(row.len(), 128, "no row may shift a column");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_trade_row_carries_the_book_and_empty_lob_identity() {
        let path = temp("trade");
        let rows = run_offline(&path);
        let schema = schema().to_vec();
        let index = |name: &str| schema.iter().position(|c| *c == name).expect(name);
        let trade_row = rows
            .iter()
            .find(|r| r[index("event_type")] == "TRADE")
            .expect("a TRADE row exists");

        // The book rides along on a trade row, as in the historical table.
        assert_ne!(trade_row[index("book_state__best_bid_price")], "");
        assert_ne!(trade_row[index("book_state__best_ask_price")], "");
        // But a trade is not a book update, so LOB identity stays absent.
        assert_eq!(trade_row[index("lob__event_time")], "", "absent, not 0.0");
        assert_eq!(trade_row[index("lob__final_update_id")], "");
        // And the trade payload is present.
        assert_eq!(trade_row[index("agg__price")], "108665.6");
        assert_eq!(trade_row[index("agg__is_buyer_maker")], "True");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_lob_row_carries_lob_identity_and_empty_trade_block() {
        let path = temp("lob");
        let rows = run_offline(&path);
        let schema = schema().to_vec();
        let index = |name: &str| schema.iter().position(|c| *c == name).expect(name);
        let lob_row = rows
            .iter()
            .find(|r| {
                r[index("event_type")] == "LOB" && !r[index("lob__final_update_id")].is_empty()
            })
            .expect("a LOB row with an update id exists");
        assert_ne!(lob_row[index("lob__event_time")], "");
        assert_eq!(lob_row[index("agg__price")], "", "no trade on a LOB row");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn depth_block_is_absent_never_zero() {
        let path = temp("depth");
        let rows = run_offline(&path);
        let schema = schema().to_vec();
        let first = schema
            .iter()
            .position(|c| *c == "depth__timestamp")
            .unwrap();
        let last = schema
            .iter()
            .position(|c| *c == "depth_state__notional_5")
            .unwrap();
        for row in &rows[1..] {
            for field in &row[first..=last] {
                assert_eq!(field, "", "depth columns must be absent, never 0.0");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_canonical_row_feeds_the_feature_layer() {
        // The bridge the feature layer relies on: the same row, as named
        // columns, with absence preserved.
        let mut values = CanonicalValues::new();
        values.insert(
            crate::core::feature::CANON_BEST_BID_PRICE.to_string(),
            Some(108_663.2),
        );
        values.insert(
            crate::core::feature::CANON_BEST_ASK_PRICE.to_string(),
            Some(108_666.0),
        );
        values.insert(crate::core::feature::CANON_AGG_PRICE.to_string(), None);

        let integers = CanonicalIntegers {
            timestamp_ns: 1_751_245_200_083_000_000,
            event_sequence: 1,
            event_time_ms: Some(1_751_245_200_083),
            ..Default::default()
        };
        let row = row_from_canonical(&values, &integers);
        assert_eq!(
            row.get(crate::core::feature::CANON_BEST_BID_PRICE),
            Some(108_663.2)
        );
        // Absent stays absent, so a feature over it fails loudly.
        assert_eq!(row.get(crate::core::feature::CANON_AGG_PRICE), None);
        assert!(
            row.get_value(crate::core::feature::CANON_AGG_PRICE)
                .expect("present as Missing")
                .is_missing()
        );

        let top = CanonicalTop::from_values(&values);
        assert_eq!(top.mid(), Some((108_663.2 + 108_666.0) / 2.0));
        assert_eq!(top.spread(), Some(2.8000000000029104));
    }
}

#[cfg(test)]
mod strategy_fanout_tests {
    use super::*;

    #[test]
    fn websocket_stream_reaches_every_registered_strategy_without_errors() {
        use crate::core::strategy::{Decision, MarketState, Position};
        use crate::live_strategy::ofmr::{
            features::OfmrFeatures, settings::Settings, socket::decode_frame,
            strategy::OfmrStrategy,
        };
        use crate::live_strategy::{LiveStrategy, available_strategies, load_strategy};
        use std::net::TcpListener;
        use std::path::Path;
        use tungstenite::{Message, accept};

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local websocket test server");
        let address = listener.local_addr().expect("get local websocket address");
        let (shutdown_server, server_shutdown) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept websocket client");
            let mut socket = accept(stream).expect("perform websocket handshake");
            let first_event_time = 1_700_000_000_000_i64;
            for bar in 0..50_u64 {
                let price = 100.0 + bar as f64 * 0.1;
                let event_time = first_event_time + bar as i64 * 300_000;
                let depth = serde_json::json!({
                    "stream": "btcusdt@depth@100ms",
                    "data": {
                        "e": "depthUpdate",
                        "E": event_time,
                        "U": bar + 2,
                        "u": bar + 2,
                        "pu": bar + 1,
                        "b": [[format!("{:.1}", price - 0.1), "2.0"]],
                        "a": [[format!("{:.1}", price + 0.1), "3.0"]]
                    }
                });
                let trade = serde_json::json!({
                    "stream": "btcusdt@aggTrade",
                    "data": {
                        "e": "aggTrade",
                        "E": event_time + 1,
                        "T": event_time + 1,
                        "a": bar + 10,
                        "f": bar + 10,
                        "l": bar + 10,
                        "p": format!("{price:.1}"),
                        "q": format!("{:.3}", 0.1 + (bar % 5) as f64 * 0.01),
                        "m": bar % 2 == 0
                    }
                });
                socket
                    .send(Message::Text(depth.to_string()))
                    .expect("send depth event");
                socket
                    .send(Message::Text(trade.to_string()))
                    .expect("send trade event");
            }
            server_shutdown
                .recv_timeout(Duration::from_secs(5))
                .expect("test should finish reading strategy frames");
            socket.close(None).expect("close websocket cleanly");
        });

        let directory = tempfile::tempdir().expect("create test output directory");
        let output = directory.path().join("capture.csv");
        let (fanout, strategy_frames) = std::sync::mpsc::sync_channel(256);
        let mut merger = Merger::new(DEFAULT_LEVELS);
        merger.apply_seed(&BookSnapshot {
            last_update_id: 1,
            bids: vec![(99.0, 2.0)],
            asks: vec![(101.0, 3.0)],
        });
        let capture = Arc::new(Mutex::new(Capture {
            merger,
            reorder: ReorderBuffer::new(0, 8),
            csv: CsvRecorder::create(&output).expect("create capture CSV"),
            rows_written: 0,
            reseeds_pending: false,
            fanout_skipped_invalid_book: 0,
            fanout: Some(fanout),
            fanout_failed: Arc::new(AtomicBool::new(false)),
            fanout_queue_depth: Arc::new(AtomicUsize::new(0)),
            received_times: BTreeMap::new(),
            last_persistence_write_ms: 0.0,
            persistence_errors: 0,
            active_source: MarketSource::Binance,
            last_source_event: [None, None],
            source_reseed_failed: [false, false],
            last_source_evaluation: Instant::now(),
        }));
        let keep_running = Arc::new(AtomicBool::new(true));
        let rows_seen = Arc::new(AtomicU64::new(0));
        spawn_stream(
            format!("ws://{address}"),
            None,
            {
                let capture = Arc::clone(&capture);
                let keep_running = Arc::clone(&keep_running);
                let rows_seen = Arc::clone(&rows_seen);
                move |text| {
                    if let Some(event) = parse_diff(text).or_else(|| parse_trade(text)) {
                        push(
                            &capture,
                            &rows_seen,
                            &keep_running,
                            None,
                            event,
                            wall_time_ms(),
                            MarketSource::Binance,
                        );
                    }
                }
            },
            || {},
            Arc::clone(&keep_running),
        );

        let strategy_names = available_strategies();
        assert!(
            !strategy_names.is_empty(),
            "at least one live strategy must be registered"
        );
        let mut strategies: Vec<LiveStrategy> = strategy_names
            .iter()
            .map(|name| {
                if name == "ofmr" {
                    let mut settings = Settings::default();
                    settings.model.path = Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("models/ofmr/trader.txt")
                        .display()
                        .to_string();
                    settings.model.threshold = 0.25;
                    let builder = OfmrStrategy::try_with_settings(settings.clone())
                        .expect("the OFMR model should load");
                    LiveStrategy {
                        name: name.clone(),
                        computer: Box::new(OfmrFeatures::with_settings(settings)),
                        builder: Box::new(builder),
                    }
                } else {
                    load_strategy(name)
                        .unwrap_or_else(|error| panic!("load registered strategy {name}: {error}"))
                }
            })
            .collect();
        let feature_names: Vec<Vec<String>> = strategies
            .iter()
            .map(|strategy| strategy.computer.output_names())
            .collect();
        for (strategy, names) in strategies.iter().zip(&feature_names) {
            assert!(
                !names.is_empty(),
                "registered strategy {} must declare output features",
                strategy.name
            );
        }
        let mut ready_inputs = vec![0; strategies.len()];
        let mut price_rows_seen = vec![0; strategies.len()];
        let mut decisions_made = vec![0; strategies.len()];
        let mut ofmr_entries = 0;
        for event_index in 0..100 {
            let frame = strategy_frames
                .recv_timeout(Duration::from_secs(5))
                .expect("websocket event should be forwarded to every strategy");
            let payload_length =
                u32::from_le_bytes(frame[..4].try_into().expect("read frame length")) as usize;
            assert_eq!(payload_length, frame.len() - 4);
            let row = decode_frame(&frame[4..]).expect("strategies should decode the market frame");
            assert!(row.get("event_time_ms").is_some());
            assert!(row.get("bid_price").is_some());
            assert!(row.get("ask_price").is_some());
            assert_eq!(
                row.get("price").is_some(),
                event_index % 2 == 1,
                "depth and aggregate-trade rows should retain their event type"
            );
            for (index, strategy) in strategies.iter_mut().enumerate() {
                let computed = strategy.computer.compute(&row).unwrap_or_else(|error| {
                    panic!(
                        "strategy {} feature computation failed: {error}",
                        strategy.name
                    )
                });
                for name in &feature_names[index] {
                    assert!(
                        computed.contains(name),
                        "strategy {} omitted declared feature {name}",
                        strategy.name
                    );
                }
                if row.get("price").is_some() {
                    price_rows_seen[index] += 1;
                }
                if feature_names[index]
                    .iter()
                    .all(|name| computed.get(name).is_some())
                {
                    ready_inputs[index] += 1;
                }
                let decision = strategy.builder.decide(&MarketState {
                    features: computed,
                    position: Position::Flat,
                    sequence: event_index,
                });
                assert!(
                    matches!(
                        decision,
                        Decision::Hold
                            | Decision::EnterLong
                            | Decision::EnterShort
                            | Decision::Exit
                    ),
                    "strategy {} must return a valid decision",
                    strategy.name
                );
                if strategy.name == "ofmr"
                    && matches!(decision, Decision::EnterLong | Decision::EnterShort)
                {
                    ofmr_entries += 1;
                }
                decisions_made[index] += 1;
            }
        }

        keep_running.store(false, Ordering::Release);
        shutdown_server
            .send(())
            .expect("signal the websocket server to close");
        server.join().expect("local websocket server should finish");
        assert_eq!(rows_seen.load(Ordering::Acquire), 100);
        for (index, name) in strategy_names.iter().enumerate() {
            assert_eq!(
                price_rows_seen[index], 50,
                "strategy {name} must receive every aggregate-trade price row"
            );
            assert_eq!(
                decisions_made[index], 100,
                "strategy {name} must produce a decision for every websocket event"
            );
            assert!(
                ready_inputs[index] > 0,
                "websocket events should populate all declared inputs for strategy {name}"
            );
        }
        assert!(
            ofmr_entries > 0,
            "the loaded OFMR model should emit at least one entry on the deterministic, warmed market sequence"
        );
        assert!(
            !capture
                .lock()
                .expect("capture mutex")
                .fanout_failed
                .load(Ordering::Acquire)
        );
    }

    #[test]
    fn depth_disconnect_marks_book_stale_until_rest_reseed() {
        let path =
            std::env::temp_dir().join(format!("uets_disconnect_reseed_{}.csv", std::process::id()));
        let mut capture = Capture {
            merger: Merger::new(DEFAULT_LEVELS),
            reorder: ReorderBuffer::new(0, 8),
            csv: CsvRecorder::create(&path).unwrap(),
            rows_written: 0,
            reseeds_pending: false,
            fanout_skipped_invalid_book: 0,
            fanout: None,
            fanout_failed: Arc::new(AtomicBool::new(false)),
            fanout_queue_depth: Arc::new(AtomicUsize::new(0)),
            received_times: BTreeMap::new(),
            last_persistence_write_ms: 0.0,
            persistence_errors: 0,
            active_source: MarketSource::Binance,
            last_source_event: [None, None],
            source_reseed_failed: [false, false],
            last_source_evaluation: Instant::now(),
        };
        let snapshot = BookSnapshot {
            last_update_id: 1,
            bids: vec![(99.0, 2.0)],
            asks: vec![(101.0, 3.0)],
        };
        capture.merger.apply_seed(&snapshot);
        assert!(capture.merger.state().book_valid());

        mark_market_stream_disconnected(&mut capture, MarketSource::Binance);
        assert!(capture.reseeds_pending);
        assert!(!capture.merger.state().book_valid());
        assert!(strategy_frame(&capture.merger, None, 1, 1, 0.0, 0, 0).is_err());

        capture.merger.apply_seed(&snapshot);
        capture.reseeds_pending = false;
        assert!(capture.merger.state().book_valid());
        assert!(!capture.reseeds_pending);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn strategy_payload_is_length_framed_and_contains_canonical_market_inputs() {
        let mut merger = Merger::new(20);
        merger.apply_seed(&BookSnapshot {
            last_update_id: 1,
            bids: vec![(99.0, 2.0)],
            asks: vec![(101.0, 3.0)],
        });
        let trade = Trade {
            event_time_ms: 1_700_000_000_000,
            transact_time_ms: 1_700_000_000_000,
            agg_trade_id: 10,
            first_trade_id: 10,
            last_trade_id: 10,
            price: 100.0,
            quantity: 0.25,
            is_buyer_maker: true,
        };
        merger.merge_event(MarketEvent::Trade(trade));

        let frame = strategy_frame(
            &merger,
            Some(trade),
            trade.event_time_ms,
            1_700_000_000_000,
            0.2,
            0,
            0,
        )
        .unwrap();
        let payload_length = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        assert_eq!(payload_length, frame.len() - 4);
        let payload = std::str::from_utf8(&frame[4..]).unwrap();
        assert!(payload.contains("bid_price:99"));
        assert!(payload.contains("ask_price:101"));
        assert!(payload.contains("price:100"));
        assert!(payload.contains("amount:0.25"));
        assert!(payload.contains("is_buyer_maker:1"));
    }

    #[test]
    fn strategy_payload_refuses_unseeded_or_stale_books() {
        let merger = Merger::new(20);
        assert!(strategy_frame(&merger, None, 1, 1, 0.0, 0, 0).is_err());
    }

    #[test]
    fn invalid_book_temporarily_pauses_fanout_without_failing_capture() {
        let path = std::env::temp_dir().join(format!(
            "uets_invalid_book_fanout_{}.csv",
            std::process::id()
        ));
        let (sender, receiver) = std::sync::mpsc::sync_channel(8);
        let failed = Arc::new(AtomicBool::new(false));
        let mut capture = Capture {
            merger: Merger::new(DEFAULT_LEVELS),
            reorder: ReorderBuffer::new(0, 8),
            csv: CsvRecorder::create(&path).unwrap(),
            rows_written: 0,
            reseeds_pending: false,
            fanout_skipped_invalid_book: 0,
            fanout: Some(sender),
            fanout_failed: Arc::clone(&failed),
            fanout_queue_depth: Arc::new(AtomicUsize::new(0)),
            received_times: BTreeMap::new(),
            last_persistence_write_ms: 0.0,
            persistence_errors: 0,
            active_source: MarketSource::Binance,
            last_source_event: [None, None],
            source_reseed_failed: [false, false],
            last_source_evaluation: Instant::now(),
        };
        capture.csv.write_header().unwrap();
        let trade = Trade {
            event_time_ms: 1_700_000_000_000,
            transact_time_ms: 1_700_000_000_000,
            agg_trade_id: 1,
            first_trade_id: 1,
            last_trade_id: 1,
            price: 100.0,
            quantity: 0.1,
            is_buyer_maker: false,
        };

        capture
            .offer(MarketEvent::Trade(trade))
            .expect("invalid books should pause strategy fanout, not the feed");

        assert_eq!(capture.fanout_skipped_invalid_book, 1);
        assert!(!failed.load(Ordering::Acquire));
        assert!(receiver.try_recv().is_err());

        capture.merger.apply_seed(&BookSnapshot {
            last_update_id: 2,
            bids: vec![(99.0, 1.0)],
            asks: vec![(101.0, 1.0)],
        });
        let resumed_trade = Trade {
            event_time_ms: trade.event_time_ms + 1,
            transact_time_ms: trade.transact_time_ms + 1,
            agg_trade_id: 2,
            first_trade_id: 2,
            last_trade_id: 2,
            ..trade
        };
        capture
            .offer(MarketEvent::Trade(resumed_trade))
            .expect("strategy fanout should resume after a valid book is reseeded");
        assert!(receiver.try_recv().is_ok());
        drop(capture);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bybit_takes_over_only_after_the_active_book_feed_times_out() {
        let path =
            std::env::temp_dir().join(format!("uets_source_failover_{}.csv", std::process::id()));
        let mut capture = Capture {
            merger: Merger::new(DEFAULT_LEVELS),
            reorder: ReorderBuffer::new(10_000, 8),
            csv: CsvRecorder::create(&path).unwrap(),
            rows_written: 0,
            reseeds_pending: false,
            fanout_skipped_invalid_book: 0,
            fanout: None,
            fanout_failed: Arc::new(AtomicBool::new(false)),
            fanout_queue_depth: Arc::new(AtomicUsize::new(0)),
            received_times: BTreeMap::new(),
            last_persistence_write_ms: 0.0,
            persistence_errors: 0,
            active_source: MarketSource::Binance,
            last_source_event: [Some(Instant::now()), None],
            source_reseed_failed: [false, false],
            last_source_evaluation: Instant::now() - Duration::from_secs(2),
        };
        capture.merger.apply_seed(&BookSnapshot {
            last_update_id: 1,
            bids: vec![(99.0, 1.0)],
            asks: vec![(101.0, 1.0)],
        });
        capture.reorder.push(MarketEvent::Trade(Trade {
            event_time_ms: 100,
            transact_time_ms: 100,
            agg_trade_id: 1,
            first_trade_id: 1,
            last_trade_id: 1,
            price: 100.0,
            quantity: 0.1,
            is_buyer_maker: false,
        }));
        capture.last_source_event[MarketSource::Binance.index()] =
            Some(Instant::now() - SOURCE_FAILOVER_TIMEOUT - Duration::from_secs(1));

        capture
            .offer_from_source(
                MarketEvent::Diff(DiffFrame {
                    event_time_ms: 200,
                    first_update_id: 1,
                    final_update_id: 2,
                    previous_update_id: None,
                    updates: vec![],
                }),
                200,
                MarketSource::Bybit,
            )
            .unwrap();

        assert_eq!(capture.active_source, MarketSource::Bybit);
        assert!(capture.reseeds_pending);
        assert!(!capture.merger.state().book_valid());
        assert_eq!(
            capture.reorder.len(),
            1,
            "old pending events are cleared and only the triggering Bybit event remains"
        );
        drop(capture);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bybit_takes_over_when_primary_rest_reseed_fails() {
        let path = std::env::temp_dir().join(format!(
            "uets_source_reseed_failover_{}.csv",
            std::process::id()
        ));
        let mut capture = Capture {
            merger: Merger::new(DEFAULT_LEVELS),
            reorder: ReorderBuffer::new(0, 8),
            csv: CsvRecorder::create(&path).unwrap(),
            rows_written: 0,
            reseeds_pending: true,
            fanout_skipped_invalid_book: 0,
            fanout: None,
            fanout_failed: Arc::new(AtomicBool::new(false)),
            fanout_queue_depth: Arc::new(AtomicUsize::new(0)),
            received_times: BTreeMap::new(),
            last_persistence_write_ms: 0.0,
            persistence_errors: 0,
            active_source: MarketSource::Binance,
            last_source_event: [Some(Instant::now()), None],
            source_reseed_failed: [true, false],
            last_source_evaluation: Instant::now() - Duration::from_secs(2),
        };

        capture
            .offer_from_source(
                MarketEvent::Diff(DiffFrame {
                    event_time_ms: 200,
                    first_update_id: 1,
                    final_update_id: 2,
                    previous_update_id: None,
                    updates: vec![],
                }),
                200,
                MarketSource::Bybit,
            )
            .unwrap();

        assert_eq!(capture.active_source, MarketSource::Bybit);
        assert!(!capture.source_reseed_failed[MarketSource::Bybit.index()]);
        assert!(capture.reseeds_pending);
        assert!(!capture.merger.state().book_valid());
        drop(capture);
        let _ = std::fs::remove_file(path);
    }
}

