//! `live.rs` — the live capture pipeline: websocket -> merge -> CSV.
//!
//! This is the module that makes `executor.rs` produce the merged table. It is
//! the only place that knows about Binance's wire format *and* about writing
//! files; `event_merger_live` stays free of both, and `core::feature` stays free
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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tungstenite::{Message, connect};

use event_merger_live::{
    BookSnapshot, CsvRecorder, DEFAULT_LATENESS_MS, DEFAULT_LEVELS, DiffFrame, MarketEvent, Merger,
    ReorderBuffer, Trade,
};

/// How long a capture may sit with zero rows before it is treated as broken.
///
/// A live feed that is working produces rows continuously, so a long silence
/// means the socket never connected rather than a quiet market.
const STALL_TIMEOUT: Duration = Duration::from_secs(30);

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
}

impl Capture {
    /// Merge one event and write whatever rows become emittable.
    fn offer(&mut self, event: MarketEvent) -> Result<u64, String> {
        let mut written = 0;
        for ready in self.reorder.push(event) {
            let (row, outcome) = self.merger.merge_event(ready);
            if outcome.needs_reseed {
                // Visible, not silent: the book is now wrong, so the next
                // row's book_state__* block cannot be trusted.
                self.reseeds_pending = true;
            }
            self.csv
                .write_row(&mut self.merger, &row)
                .map_err(|e| e.to_string())?;
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

/// Fire-and-forget reconnecting stream thread.
fn spawn_stream(url: String, on_msg: impl FnMut(&str) + Send + 'static) {
    std::thread::spawn(move || {
        let mut on_msg = on_msg;
        loop {
            match connect(url.as_str()) {
                Ok((mut ws, _)) => loop {
                    match ws.read() {
                        Ok(Message::Text(text)) => on_msg(&text),
                        Ok(Message::Ping(_))
                        | Ok(Message::Pong(_))
                        | Ok(Message::Binary(_))
                        | Ok(Message::Frame(_)) => (),
                        Ok(Message::Close(frame)) => {
                            eprintln!("[capture] closed by server: {:?}", frame.map(|f| f.code));
                            break;
                        }
                        Err(e) => {
                            eprintln!("[capture] socket error: {e}");
                            break;
                        }
                    }
                },
                Err(e) => eprintln!("[capture] connect failed: {e}"),
            }
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
            updates.push(event_merger_live::BookUpdate {
                is_bid,
                price,
                quantity,
            });
        }
    }
    Some(MarketEvent::Diff(DiffFrame {
        event_time_ms: num(v.get("E"))? as i64,
        first_update_id: u64_of(v.get("U")).unwrap_or(0),
        final_update_id: u64_of(v.get("u"))?,
        updates,
    }))
}

/// Run a live capture until interrupted, or until `max_rows` is reached.
pub fn run_capture(config: CaptureConfig) -> Result<(), String> {
    let symbol = config.symbol.to_lowercase();
    let keep_running = Arc::new(AtomicBool::new(true));
    let rows_seen = Arc::new(AtomicU64::new(0));

    let csv = if config.append {
        CsvRecorder::append(&config.output)
    } else {
        CsvRecorder::create(&config.output)
    }
    .map_err(|e| e.to_string())?
    .flush_every(64);

    eprintln!("[capture] output   {}", config.output.display());
    eprintln!("[capture] symbol   {}", config.symbol);
    eprintln!("[capture] columns  {}", event_merger_live::schema().len());

    let capture = Arc::new(Mutex::new(Capture {
        merger: Merger::new(config.levels),
        reorder: ReorderBuffer::new(config.lateness_ms, 65_536),
        csv,
        rows_written: 0,
        reseeds_pending: false,
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
        Err(e) => eprintln!("[capture] WARNING: REST seed failed: {e}"),
    }

    let max_rows = config.max_rows;
    spawn_stream(
        format!("wss://fstream.binance.com/stream?streams={symbol}@aggTrade"),
        {
            let shared = Arc::clone(&capture);
            let seen = Arc::clone(&rows_seen);
            let keep = Arc::clone(&keep_running);
            move |text| {
                if let Some(event) = parse_trade(text) {
                    push(&shared, &seen, &keep, max_rows, event);
                }
            }
        },
    );
    spawn_stream(
        format!("wss://fstream.binance.com/stream?streams={symbol}@depth@100ms"),
        {
            let shared = Arc::clone(&capture);
            let seen = Arc::clone(&rows_seen);
            let keep = Arc::clone(&keep_running);
            move |text| {
                if let Some(event) = parse_diff(text) {
                    push(&shared, &seen, &keep, max_rows, event);
                }
            }
        },
    );

    // Recover the book from a detected sequence gap without a restart.
    {
        let shared = Arc::clone(&capture);
        let sym = config.symbol.clone();
        let keep = Arc::clone(&keep_running);
        std::thread::spawn(move || {
            while keep.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                let stale = shared.lock().map(|g| g.reseeds_pending).unwrap_or(false);
                if !stale {
                    continue;
                }
                match seed_from_rest(&sym) {
                    Ok(snapshot) => {
                        if let Ok(mut guard) = shared.lock() {
                            guard.merger.apply_seed(&snapshot);
                            guard.reseeds_pending = false;
                            eprintln!("[capture] book re-seeded after a sequence gap");
                        }
                    }
                    Err(e) => eprintln!("[capture] re-seed failed: {e}"),
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
        if let Some(limit) = max_rows {
            if total >= limit {
                break;
            }
        }
        if !keep_running.load(Ordering::Relaxed) {
            break;
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
            let (book_ok, gaps, late, seeds) = {
                let guard = capture.lock().expect("capture mutex poisoned");
                let c = guard.merger.counters();
                (
                    guard.merger.state().book_valid(),
                    c.gaps,
                    c.late_dropped,
                    c.seeds_applied,
                )
            };
            eprintln!(
                "[capture] rows={total} book={} gaps={gaps} late={late} seeds={seeds}",
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
) {
    let Ok(mut guard) = shared.lock() else { return };
    match guard.offer(event) {
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
        let text = r#"{"e":"depthUpdate","E":1751245200406,"U":1,"u":10,"b":[["108665.5","9.141"]],"a":[["108665.6","12.556"]]}"#;
        match parse_diff(text).expect("parsed") {
            MarketEvent::Diff(f) => {
                assert_eq!(f.first_update_id, 1);
                assert_eq!(f.final_update_id, 10);
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
        let text = r#"{"e":"depthUpdate","E":1,"U":1,"u":2,"b":[["100.0","0"]],"a":[]}"#;
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
        let text = r#"{"e":"depthUpdate","E":1,"U":1,"u":2,"b":[["bad","1"],["2.0"],"3",["4.0","5.0"]],"a":[]}"#;
        match parse_diff(text).expect("parsed") {
            MarketEvent::Diff(f) => assert_eq!(f.updates.len(), 1),
            other => panic!("expected a diff, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_frame_missing_its_final_update_id() {
        // Without `u` the sequence cannot advance, so the frame is refused
        // rather than silently resetting continuity.
        let text = r#"{"e":"depthUpdate","E":1,"U":1,"b":[],"a":[]}"#;
        assert!(parse_diff(text).is_none());
    }
}

#[cfg(test)]
mod end_to_end {
    use super::*;
    use crate::core::feature::{
        CanonicalIntegers, CanonicalTop, CanonicalValues, row_from_canonical,
    };
    use event_merger_live::schema;

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
        };

        // A REST seed, applied DIRECTLY to the merger. It must not go
        // through the reorder buffer: a snapshot has no exchange event time
        // (`BookSnapshot::event_time_ms` reports the update id), so it would
        // set the watermark far into the future and make every real event
        // look late, dropping all of them.
        capture.merger.apply_seed(&BookSnapshot {
            last_update_id: 7_911_048_639_390,
            bids: vec![(108_663.2, 0.095), (108_662.9, 0.553), (108_653.8, 0.025)],
            asks: vec![(108_666.0, 0.004), (108_668.8, 0.002), (108_668.9, 0.562)],
        });

        for text in [
            r#"{"e":"depthUpdate","E":1751245200083,"U":1,"u":2,"b":[["108663.2","9.141"]],"a":[["108665.6","12.556"]]}"#,
            r#"{"e":"aggTrade","E":1751245200354,"a":2773899591,"p":"108665.6","q":"0.002","f":6438488841,"l":6438488841,"T":1751245200354,"m":true}"#,
            r#"{"e":"depthUpdate","E":1751245200406,"U":3,"u":4,"b":[["108653.8","0.025"]],"a":[["108666.0","0.004"]]}"#,
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
