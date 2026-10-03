//! `diagnostic.rs` — one row per websocket event, never merged.
//!
//! The canonical capture in [`live`] merges aggTrades and LOB updates into
//! one wide row (absent blocks left empty) so the table matches history.
//! This module deliberately does the *opposite*, because a diagnostic wants
//! to see each frame exactly as it arrived:
//!
//! ```text
//! timestamp,update_type,agg__price,...,lob__bid_p1,...,lob__ask_q5
//! 1751245200083,aggTrades,108665.6,...,NaN,...,NaN
//! 1751245200094,orderbook,NaN,...,108663.2,...,108666.0
//! ```
//!
//! | column group | filled when | otherwise |
//! |--------------|-------------|-----------|
//! | `agg__*` | `update_type = aggTrades` | `NaN` |
//! | `lob__*` | `update_type = orderbook` | `NaN` |
//!
//! `update_type` is therefore the discriminator: a row belongs to exactly one
//! stream, and the other stream's columns are `NaN` — the CSV spelling a
//! reader like pandas already treats as missing. "Latest info available"
//! means a row carries that stream's *latest* frame and nothing back-filled
//! from the other one.
//!
//! ```text
//! cargo run --bin dev_test -- diag btcusdt
//! ```
//!
//! runs both streams (`@aggTrade` + `@depth20@100ms`), prints each row, and
//! appends them to `out/live/<symbol>_diag_<epoch>.csv`.

use std::io::{self, Write};
use std::sync::mpsc::{self, Sender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tungstenite::{Message, connect};

/// Book levels carried per orderbook row. Beyond this the frame's levels are
/// dropped: a diagnostic row is for eyeballing, not for reconstructing depth.
pub const BOOK_LEVELS: usize = 5;

/// Which stream a row came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateType {
    /// The aggregated-trade stream.
    AggTrades,
    /// The order-book stream (partial or diff).
    Orderbook,
}

impl UpdateType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AggTrades => "aggTrades",
            Self::Orderbook => "orderbook",
        }
    }
}

/// The trade half of a row.
#[derive(Clone, Debug, PartialEq)]
pub struct AggTradeData {
    pub price: f64,
    pub quantity: f64,
    /// Aggressor side: `is_buyer_maker` in, taker side out (`m = true` means
    /// the buyer rested, so the aggressor sold).
    pub side: &'static str,
    pub trade_id: Option<u64>,
    pub event_time_ms: Option<u64>,
    pub transact_time_ms: Option<u64>,
}

/// The book half of a row: whatever levels the frame carried, as arrived.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderbookData {
    pub update_id: Option<u64>,
    pub event_time_ms: Option<u64>,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

/// One diagnostic row: exactly one stream's latest frame.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticRow {
    /// Local receive time, milliseconds since the Unix epoch.
    pub timestamp: u64,
    pub update_type: UpdateType,
    pub agg: Option<AggTradeData>,
    pub lob: Option<OrderbookData>,
}

/// A numeric field that may arrive as a JSON string or a number.
fn num(value: &Value, key: &str) -> Option<f64> {
    let raw = value.get(key)?;
    raw.as_str()
        .and_then(|s| s.parse().ok())
        .or_else(|| raw.as_f64())
}

/// An integer field that may arrive as a JSON string or a number.
fn uint(value: &Value, key: &str) -> Option<u64> {
    let raw = value.get(key)?;
    raw.as_u64()
        .or_else(|| raw.as_str().and_then(|s| s.parse().ok()))
}

/// Price/qty levels from a `[[p, q], ...]` array, capped at [`BOOK_LEVELS`].
fn levels(value: Option<&Value>) -> Vec<(f64, f64)> {
    value
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .take(BOOK_LEVELS)
                .filter_map(|entry| {
                    let pair = entry.as_array()?;
                    let price = pair.first()?.as_str()?.parse().ok()?;
                    let qty = pair.get(1)?.as_str()?.parse().ok()?;
                    Some((price, qty))
                })
                .collect()
        })
        .unwrap_or_default()
}

impl DiagnosticRow {
    /// Parse one websocket text frame. Returns `None` for anything that is
    /// neither an aggTrade nor an order-book update, so unrelated streams
    /// (klines, tickers, heartbeats) cannot pollute the diagnostic table.
    pub fn from_frame(text: &str, received_ms: u64) -> Option<Self> {
        let mut value: Value = serde_json::from_str(text).ok()?;
        // Combined-stream envelopes wrap the payload in `data`.
        if value.get("stream").is_some() {
            value = value.get("data")?.clone();
        }
        let event = value.get("e").and_then(Value::as_str);

        if event == Some("aggTrade") {
            let maker = value.get("m").and_then(Value::as_bool).unwrap_or(false);
            return Some(Self {
                timestamp: received_ms,
                update_type: UpdateType::AggTrades,
                agg: Some(AggTradeData {
                    price: num(&value, "p")?,
                    quantity: num(&value, "q")?,
                    side: if maker { "sell" } else { "buy" },
                    trade_id: uint(&value, "a"),
                    event_time_ms: uint(&value, "E"),
                    transact_time_ms: uint(&value, "T"),
                }),
                lob: None,
            });
        }

        // Partial book: `{lastUpdateId, bids, asks}` (full snapshots).
        // Diff book: `{e:"depthUpdate", U, u, b, a}` (changed levels only).
        let partial = value.get("lastUpdateId").is_some() && value.get("bids").is_some();
        let diff = (value.get("b").is_some() || value.get("a").is_some())
            && (event == Some("depthUpdate") || value.get("u").is_some());
        if partial || diff {
            return Some(Self {
                timestamp: received_ms,
                update_type: UpdateType::Orderbook,
                agg: None,
                lob: Some(OrderbookData {
                    update_id: if partial {
                        uint(&value, "lastUpdateId")
                    } else {
                        uint(&value, "u")
                    },
                    event_time_ms: uint(&value, "E"),
                    bids: levels(value.get(if partial { "bids" } else { "b" })),
                    asks: levels(value.get(if partial { "asks" } else { "a" })),
                }),
            });
        }
        None
    }
}

/// The literal written into a cell whose data does not exist on this row —
/// the CSV spelling readers (pandas included) already treat as missing.
const NAN: &str = "NaN";

/// The full CSV header, in column order.
pub fn csv_header() -> String {
    let mut columns = vec![
        "timestamp".to_string(),
        "update_type".to_string(),
        "agg__price".to_string(),
        "agg__quantity".to_string(),
        "agg__side".to_string(),
        "agg__trade_id".to_string(),
        "agg__event_time_ms".to_string(),
        "agg__transact_time_ms".to_string(),
        "lob__update_id".to_string(),
        "lob__event_time_ms".to_string(),
    ];
    for level in 1..=BOOK_LEVELS {
        columns.push(format!("lob__bid_p{level}"));
        columns.push(format!("lob__bid_q{level}"));
        columns.push(format!("lob__ask_p{level}"));
        columns.push(format!("lob__ask_q{level}"));
    }
    columns.join(",")
}

fn cell(value: Option<u64>) -> String {
    value.map_or_else(|| NAN.to_string(), |v| v.to_string())
}

impl DiagnosticRow {
    /// One CSV line: this row's own stream filled, the other stream `NaN`.
    pub fn to_csv(&self) -> String {
        let mut cells = vec![
            self.timestamp.to_string(),
            self.update_type.as_str().to_string(),
        ];
        match &self.agg {
            Some(agg) => {
                cells.push(format!("{}", agg.price));
                cells.push(format!("{}", agg.quantity));
                cells.push(agg.side.to_string());
                cells.push(cell(agg.trade_id));
                cells.push(cell(agg.event_time_ms));
                cells.push(cell(agg.transact_time_ms));
            }
            None => cells.extend(vec![NAN.to_string(); 6]),
        }
        match &self.lob {
            Some(lob) => {
                cells.push(cell(lob.update_id));
                cells.push(cell(lob.event_time_ms));
                for level in 0..BOOK_LEVELS {
                    for side in [&lob.bids, &lob.asks] {
                        match side.get(level) {
                            Some((price, qty)) => {
                                cells.push(format!("{price}"));
                                cells.push(format!("{qty}"));
                            }
                            None => {
                                cells.push(NAN.to_string());
                                cells.push(NAN.to_string());
                            }
                        }
                    }
                }
            }
            None => cells.extend(vec![NAN.to_string(); 2 + BOOK_LEVELS * 4]),
        }
        cells.join(",")
    }
}

/// Appends header + rows to a writer, flushing on demand. Generic over
/// [`Write`], so the tests drive it from a `Vec<u8>` and `run` from a file.
pub struct DiagnosticCsv<W: Write> {
    inner: W,
}

impl<W: Write> DiagnosticCsv<W> {
    /// Wrap `inner` and write the header line immediately.
    pub fn new(mut inner: W) -> io::Result<Self> {
        writeln!(inner, "{}", csv_header())?;
        Ok(Self { inner })
    }

    pub fn write_row(&mut self, row: &DiagnosticRow) -> io::Result<()> {
        writeln!(self.inner, "{}", row.to_csv())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Reconnect a stream forever, forwarding every text frame to `tx`.
fn spawn_stream(url: String, tx: Sender<String>) {
    std::thread::spawn(move || {
        loop {
            match connect(&url) {
                Ok((socket, _)) => {
                    let mut socket = socket;
                    loop {
                        match socket.read() {
                            Ok(Message::Text(text)) => {
                                if tx.send(text.to_string()).is_err() {
                                    return; // the reader is gone; nothing left to do
                                }
                            }
                            Ok(
                                Message::Ping(_)
                                | Message::Pong(_)
                                | Message::Binary(_)
                                | Message::Frame(_),
                            ) => {}
                            Ok(Message::Close(frame)) => {
                                eprintln!("[{url}] closed by server: {frame:?}");
                                break;
                            }
                            Err(e) => {
                                eprintln!("[{url}] error: {e}");
                                break;
                            }
                        }
                    }
                }
                Err(e) => eprintln!("[{url}] connect failed: {e}"),
            }
            eprintln!("[{url}] reconnecting in 3s...");
            std::thread::sleep(Duration::from_secs(3));
        }
    });
}

/// Stream both feeds and print/write one diagnostic row per frame until the
/// process is interrupted.
///
/// The file is created (header included) before the sockets open, so a feed
/// that never connects still leaves a well-formed, empty diagnostic table
/// rather than nothing at all.
pub fn run(
    symbol: &str,
    output_dir: impl Into<std::path::PathBuf>,
) -> io::Result<std::path::PathBuf> {
    let symbol = symbol.to_uppercase();
    let low = symbol.to_lowercase();
    let dir = output_dir.into();
    std::fs::create_dir_all(&dir)?;
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = dir.join(format!("{symbol}_diag_{epoch}.csv"));
    let file = std::fs::File::create(&path)?;
    let mut csv = DiagnosticCsv::new(io::BufWriter::new(file))?;

    let (tx, rx) = mpsc::channel::<String>();
    spawn_stream(
        format!("wss://stream.binance.com:9443/ws/{low}@aggTrade"),
        tx.clone(),
    );
    spawn_stream(
        format!("wss://stream.binance.com:9443/ws/{low}@depth20@100ms"),
        tx,
    );

    println!("diag:    {symbol}  (aggTrades + depth20@100ms)");
    println!("columns: {}", csv_header());
    println!("file:    {}", path.display());
    println!("stop:    Ctrl+C");

    while let Ok(text) = rx.recv() {
        let Some(row) = DiagnosticRow::from_frame(&text, now_ms()) else {
            continue; // not a trade and not a book: not a diagnostic row
        };
        let line = row.to_csv();
        println!("{line}");
        csv.write_row(&row)?;
        csv.flush()?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGG_FRAME: &str = r#"{"e":"aggTrade","E":1751245200083,"a":9,"p":"108665.600","q":"0.340","f":1,"l":5,"T":1751245200083,"m":false,"M":true}"#;
    const BOOK_FRAME: &str = r#"{"lastUpdateId":1017,"bids":[["108663.20","1.500"],["108662.90","0.400"]],"asks":[["108666.00","2.000"],["108667.10","0.250"]]}"#;
    const DIFF_FRAME: &str = r#"{"e":"depthUpdate","E":1751245200094,"s":"BTCUSDT","U":57,"u":58,"b":[["108660.00","0.600"]],"a":[["108668.00","1.100"]]}"#;

    fn cells(row: &DiagnosticRow) -> Vec<String> {
        row.to_csv().split(',').map(str::to_string).collect()
    }

    fn index_of(column: &str) -> usize {
        csv_header()
            .split(',')
            .position(|c| c == column)
            .unwrap_or_else(|| panic!("no column {column}"))
    }

    fn columns_with(prefix: &str) -> Vec<String> {
        csv_header()
            .split(',')
            .filter(|c| c.starts_with(prefix))
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn an_aggtrade_row_fills_the_trade_columns_and_nans_the_book() {
        let row = DiagnosticRow::from_frame(AGG_FRAME, 1_751_245_200_500).unwrap();
        assert_eq!(row.update_type, UpdateType::AggTrades);
        let agg = row.agg.as_ref().unwrap();
        assert_eq!(agg.price, 108665.6);
        assert_eq!(agg.quantity, 0.34);
        assert_eq!(agg.side, "buy", "m=false: the taker bought");
        assert_eq!(agg.trade_id, Some(9));
        assert_eq!(agg.event_time_ms, Some(1_751_245_200_083));
        assert!(row.lob.is_none(), "a trade carries no book");

        let row_cells = cells(&row);
        assert_eq!(row_cells[index_of("timestamp")], "1751245200500");
        assert_eq!(row_cells[index_of("update_type")], "aggTrades");
        assert_eq!(row_cells[index_of("agg__price")], "108665.6");
        assert_eq!(row_cells[index_of("agg__side")], "buy");
        for column in columns_with("lob__") {
            assert_eq!(
                row_cells[index_of(&column)],
                "NaN",
                "{column} on an aggTrades row"
            );
        }
    }

    #[test]
    fn an_orderbook_row_fills_the_book_columns_and_nans_the_trade() {
        let row = DiagnosticRow::from_frame(BOOK_FRAME, 1_751_245_200_600).unwrap();
        assert_eq!(row.update_type, UpdateType::Orderbook);
        let lob = row.lob.as_ref().unwrap();
        assert_eq!(lob.update_id, Some(1017));
        assert_eq!(lob.bids.len(), 2);
        assert_eq!(lob.bids[0], (108663.2, 1.5));
        assert!(row.agg.is_none(), "a book update carries no trade");

        let row_cells = cells(&row);
        assert_eq!(row_cells[index_of("update_type")], "orderbook");
        assert_eq!(row_cells[index_of("lob__update_id")], "1017");
        assert_eq!(row_cells[index_of("lob__bid_p1")], "108663.2");
        assert_eq!(row_cells[index_of("lob__ask_q1")], "2");
        assert_eq!(
            row_cells[index_of("lob__bid_p5")],
            "NaN",
            "levels the frame did not carry are NaN, never 0"
        );
        for column in columns_with("agg__") {
            assert_eq!(
                row_cells[index_of(&column)],
                "NaN",
                "{column} on an orderbook row"
            );
        }
    }

    #[test]
    fn trades_and_book_updates_are_separate_rows_never_merged() {
        let rows = [
            DiagnosticRow::from_frame(AGG_FRAME, 1_000).unwrap(),
            DiagnosticRow::from_frame(BOOK_FRAME, 1_100).unwrap(),
        ];
        assert_eq!(rows.len(), 2, "one event, one row");
        assert_ne!(
            rows[0].update_type, rows[1].update_type,
            "the two streams stay distinguishable"
        );
        for row in &rows {
            assert_ne!(
                row.agg.is_some(),
                row.lob.is_some(),
                "a row belongs to exactly one stream"
            );
            let row_cells = cells(row);
            let agg_absent = row_cells[index_of("agg__price")] == "NaN";
            let lob_absent = row_cells[index_of("lob__bid_p1")] == "NaN";
            assert_ne!(
                agg_absent, lob_absent,
                "exactly one stream's columns are NaN"
            );
        }
    }

    #[test]
    fn the_header_lists_every_column_once_and_rows_match_its_width() {
        let header = csv_header();
        let columns: Vec<&str> = header.split(',').collect();
        assert_eq!(columns[0], "timestamp");
        assert_eq!(columns[1], "update_type");
        assert_eq!(columns.iter().filter(|c| c.starts_with("agg__")).count(), 6);
        assert_eq!(
            columns.iter().filter(|c| c.starts_with("lob__")).count(),
            2 + BOOK_LEVELS * 4
        );
        let mut dedup = columns.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), columns.len(), "no duplicate column names");
        for frame in [AGG_FRAME, BOOK_FRAME, DIFF_FRAME] {
            let row = DiagnosticRow::from_frame(frame, 1).unwrap();
            assert_eq!(row.to_csv().split(',').count(), columns.len());
        }
    }

    #[test]
    fn diff_depth_frames_are_orderbook_rows_too() {
        let row = DiagnosticRow::from_frame(DIFF_FRAME, 2_000).unwrap();
        assert_eq!(row.update_type, UpdateType::Orderbook);
        let lob = row.lob.as_ref().unwrap();
        assert_eq!(
            lob.update_id,
            Some(58),
            "the final update id identifies the book"
        );
        assert_eq!(lob.event_time_ms, Some(1_751_245_200_094));
        assert_eq!(lob.bids, vec![(108660.0, 0.6)]);
        assert_eq!(lob.asks, vec![(108668.0, 1.1)]);
    }

    #[test]
    fn unrelated_and_wrapped_frames_are_handled() {
        // A kline belongs to neither stream, so it produces no row.
        let kline = r#"{"e":"kline","E":1,"k":{"t":1,"s":"BTCUSDT","c":"1.0"}}"#;
        assert!(DiagnosticRow::from_frame(kline, 1).is_none());
        // Garbage is ignored, never a panic.
        assert!(DiagnosticRow::from_frame("not json", 1).is_none());
        assert!(DiagnosticRow::from_frame("", 1).is_none());
        // A combined-stream envelope unwraps to its payload.
        let wrapped = format!(r#"{{"stream":"btcusdt@aggTrade","data":{AGG_FRAME}}}"#);
        let row = DiagnosticRow::from_frame(&wrapped, 1).unwrap();
        assert_eq!(row.update_type, UpdateType::AggTrades);
        assert_eq!(row.agg.as_ref().unwrap().price, 108665.6);
    }

    #[test]
    fn the_csv_writer_emits_header_then_one_row_per_event() {
        let mut sink: Vec<u8> = Vec::new();
        {
            let mut csv = DiagnosticCsv::new(&mut sink).unwrap();
            let trade = DiagnosticRow::from_frame(AGG_FRAME, 1_000).unwrap();
            let book = DiagnosticRow::from_frame(BOOK_FRAME, 1_100).unwrap();
            csv.write_row(&trade).unwrap();
            csv.write_row(&book).unwrap();
            csv.flush().unwrap();
        }
        let text = String::from_utf8(sink).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "header + two rows, never merged");
        assert_eq!(lines[0], csv_header());
        assert!(lines[1].contains(",aggTrades,"));
        assert!(lines[2].contains(",orderbook,"));
        assert!(
            lines[1].contains(",NaN"),
            "book columns NaN on the trade row"
        );
        assert!(
            lines[2].contains(",NaN"),
            "trade columns NaN on the book row"
        );
    }
}
