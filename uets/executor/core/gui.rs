//! The GUI bridge: a directory of files, not a protocol.
//!
//! The dashboard is a vim window (or anything else that can `cat` a file), so
//! the interface is deliberately the dumbest one that works:
//!
//! ```text
//!   out/live/state.json     <- whole-frame rewrite, every publish  (rust -> gui)
//!   out/live/trades.jsonl   <- append-only, one trade per line     (rust -> gui)
//!   out/live/command.json   <- dashboard drops a line here         (gui -> rust)
//! ```
//!
//! Three properties this buys, and why each is worth a file:
//!
//! * **Crash-visibility.** State is on disk, so a process that dies mid-session
//!   leaves a readable last frame instead of an empty terminal.
//! * **No runtime coupling.** The dashboard can attach late, detach, or restart
//!   without the executor noticing, and neither side links the other.
//! * **One-directional risk.** Only [`DashboardCommand`] travels back, and the
//!   loop decides what to do with it — the dashboard can ask, never act.
//!
//! Writes go through `telemetry::write_json_atomic`, so a reader never sees a
//! half-written frame (the same guarantee `monitor.rs` gives `display.py`).

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::order::Side;
use super::portfolio::Portfolio;
use super::stats::PerformanceStats;
use super::telemetry::{TelemetrySnapshot, json_number, json_string, now_ms, write_json_atomic};

pub const STATE_FILE: &str = "state.json";
pub const TRADES_FILE: &str = "trades.jsonl";
pub const COMMAND_FILE: &str = "command.json";

/// What the dashboard draws for one open symbol.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenPosition {
    pub symbol: String,
    pub side: Side,
    pub size: f64,
    pub avg_entry: f64,
    /// The price this row was valued at, so the dashboard can show whether the
    /// mark is fresh or just the entry carried forward.
    pub mark: f64,
    pub unrealized_pnl: f64,
}

/// One complete frame of what the dashboard shows.
///
/// Built by [`DashboardState::capture`] from live objects, so nothing on screen
/// comes from a different instant than anything else on screen.
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardState {
    pub active_strategy: String,
    pub venue: &'static str,
    pub symbol: String,
    pub starting_equity: f64,
    pub equity: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub gross_exposure: f64,
    pub positions: Vec<OpenPosition>,
    pub stats: PerformanceStats,
    pub telemetry: TelemetrySnapshot,
    pub updated_ms: u64,
}

impl DashboardState {
    /// Read the strategy, the book and the counters into one frame.
    ///
    /// `symbol` is the instrument being traded (the dashboard is per-symbol);
    /// every symbol in the book is still listed, because a hedge on a second
    /// instrument must not be invisible.
    ///
    /// Call `Portfolio::mark(&marks)` before this: `PerformanceStats` reads the
    /// equity curve, so a portfolio that is never marked reports a cumulative
    /// return of zero however much it has actually made.
    pub fn capture(
        active_strategy: impl Into<String>,
        venue: &'static str,
        symbol: impl Into<String>,
        portfolio: &Portfolio,
        marks: &BTreeMap<String, f64>,
        telemetry: &TelemetrySnapshot,
    ) -> Self {
        let positions = portfolio
            .holdings()
            .open_holdings()
            .map(|(symbol, holding)| {
                // Same rule as the book: an unmarked symbol contributes its
                // entry price rather than a fabricated mark.
                let mark = marks.get(symbol).copied().unwrap_or(holding.avg_entry);
                OpenPosition {
                    symbol: symbol.clone(),
                    side: holding.side().unwrap_or(Side::Buy),
                    size: holding.size,
                    avg_entry: holding.avg_entry,
                    mark,
                    unrealized_pnl: holding.unrealized_pnl(mark),
                }
            })
            .collect();
        Self {
            active_strategy: active_strategy.into(),
            venue,
            symbol: symbol.into(),
            starting_equity: portfolio.starting_cash(),
            equity: portfolio.equity(marks),
            cash: portfolio.cash(),
            realized_pnl: portfolio.ledger().realized_pnl(),
            unrealized_pnl: portfolio.unrealized_pnl(marks),
            gross_exposure: portfolio.gross_exposure(marks),
            positions,
            // Un-annualised: an intraday session is not a yearly sample, and
            // multiplying it by sqrt(252) would flatter it on screen.
            stats: PerformanceStats::from_ledger(portfolio.ledger(), 1.0),
            telemetry: *telemetry,
            updated_ms: now_ms(),
        }
    }
}
impl DashboardState {
    /// The frame as one JSON object. This is the file a vim dashboard reads.
    pub fn to_json(&self) -> String {
        let positions = self
            .positions
            .iter()
            .map(|p| {
                format!(
                    "{{\"symbol\":{},\"side\":{},\"size\":{},\"avg_entry\":{},\"mark\":{},\"unrealized_pnl\":{}}}",
                    json_string(&p.symbol),
                    json_string(if p.side.is_buy() { "long" } else { "short" }),
                    json_number(p.size),
                    json_number(p.avg_entry),
                    json_number(p.mark),
                    json_number(p.unrealized_pnl),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"updated_ms\":{},\"active_strategy\":{},\"venue\":{},\"symbol\":{},",
                "\"starting_equity\":{},\"equity\":{},\"cash\":{},\"realized_pnl\":{},",
                "\"unrealized_pnl\":{},\"gross_exposure\":{},\"positions\":[{}],",
                "\"sortino\":{},\"win_rate\":{},\"cumulative_return\":{},\"max_drawdown\":{},",
                "\"trades\":{},\"wins\":{},\"losses\":{},\"telemetry\":{}}}"
            ),
            self.updated_ms,
            json_string(&self.active_strategy),
            json_string(self.venue),
            json_string(&self.symbol),
            json_number(self.starting_equity),
            json_number(self.equity),
            json_number(self.cash),
            json_number(self.realized_pnl),
            json_number(self.unrealized_pnl),
            json_number(self.gross_exposure),
            positions,
            json_number(self.stats.sortino),
            json_number(self.stats.win_rate),
            json_number(self.stats.cumulative_return),
            json_number(self.stats.max_drawdown),
            self.stats.trades,
            self.stats.wins,
            self.stats.losses,
            self.telemetry.to_json(),
        )
    }
}

/// What the dashboard is allowed to ask for.
///
/// A deliberately closed set. Every variant is something the *loop* may decide
/// to ignore — the dashboard has no way to reach a connector directly, which is
/// what keeps a display bug from becoming a trading bug.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DashboardCommand {
    /// Close everything and stop opening new risk.
    Flatten,
    /// Stop opening new risk; leave what is open alone.
    Pause,
    Resume,
    /// Switch the active strategy by registry name.
    SetStrategy(String),
    /// Forget the trade log (a fresh session's numbers, not a fresh account).
    ClearTrades,
}

impl DashboardCommand {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Flatten => "flatten",
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::SetStrategy(_) => "set_strategy",
            Self::ClearTrades => "clear_trades",
        }
    }

    /// Parse one command line: `{"action":"pause"}` or
    /// `{"action":"set_strategy","value":"strategy2"}`.
    ///
    /// Hand-parsed for the same reason the writer is hand-rolled: the dashboard
    /// side of this contract is a text editor, and one `find` is a smaller
    /// dependency than a deserialiser. Unknown fields are ignored, so a richer
    /// dashboard stays compatible with an older loop.
    pub fn parse(line: &str) -> Result<Self, CommandError> {
        let line = line.trim();
        if line.is_empty() {
            return Err(CommandError::Empty);
        }
        let action = field(line, "action").ok_or(CommandError::MissingAction)?;
        match action {
            "flatten" => Ok(Self::Flatten),
            "pause" => Ok(Self::Pause),
            "resume" => Ok(Self::Resume),
            "clear_trades" => Ok(Self::ClearTrades),
            "set_strategy" => field(line, "value")
                .map(|value| Self::SetStrategy(value.to_string()))
                .ok_or(CommandError::MissingValue("set_strategy")),
            other => Err(CommandError::UnknownAction(other.to_string())),
        }
    }
}

/// Why a command file could not be honoured.
#[derive(Clone, Debug, PartialEq)]
pub enum CommandError {
    Empty,
    MissingAction,
    MissingValue(&'static str),
    UnknownAction(String),
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "command file was empty"),
            Self::MissingAction => write!(f, "command has no \"action\" field"),
            Self::MissingValue(action) => write!(f, "{action} needs a \"value\" field"),
            Self::UnknownAction(action) => write!(f, "unknown action {action:?}"),
        }
    }
}

impl std::error::Error for CommandError {}

/// The value of `"key": "..."` in a flat JSON object, if present.
///
/// Escaped quotes are skipped rather than unescaped: the actions and strategy
/// names this is used for are plain identifiers, and inventing an unescaper that
/// is never exercised would be untested code on the hot path.
fn field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let after_key = &json[json.find(&needle)? + needle.len()..];
    let after_colon = after_key.get(after_key.find(':')? + 1..)?.trim_start();
    let quoted = after_colon.strip_prefix('"')?;
    let bytes = quoted.as_bytes();
    let mut end = 0;
    while end < bytes.len() {
        match bytes[end] {
            // `\` and `"` are ASCII, so a UTF-8 continuation byte can never be
            // mistaken for either and these offsets always land on char bounds.
            b'\\' => end += 2,
            b'"' => return Some(&quoted[..end]),
            _ => end += 1,
        }
    }
    None
}
/// The directory the dashboard reads and writes.
#[derive(Clone, Debug)]
pub struct GuiBridge {
    dir: PathBuf,
}

impl GuiBridge {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The same `out/live` directory `monitor.rs` publishes frames into, so a
    /// dashboard has one place to watch instead of two.
    pub fn in_out_dir() -> Self {
        Self::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("out")
                .join("live"),
        )
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn state_path(&self) -> PathBuf {
        self.dir.join(STATE_FILE)
    }

    pub fn trades_path(&self) -> PathBuf {
        self.dir.join(TRADES_FILE)
    }

    pub fn command_path(&self) -> PathBuf {
        self.dir.join(COMMAND_FILE)
    }

    /// Publish the frame and return where it landed, for the log line.
    pub fn publish(&self, state: &DashboardState) -> io::Result<PathBuf> {
        let path = self.state_path();
        write_json_atomic(&path, &state.to_json())?;
        Ok(path)
    }

    /// Append one closed trade as a line.
    ///
    /// Append-only and never rewritten: it is the session's audit trail, and a
    /// log a crash can truncate mid-line is still readable up to the last
    /// complete entry. A malformed tail is the reader's problem, not the
    /// writer's, which is why this is not JSON-arrayed into `state.json`.
    pub fn append_trade(&self, trade: &super::pnl::Trade) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.trades_path())?;
        writeln!(file, "{}", trade_json(trade))
    }

    /// Empty the trade log. Truncates rather than deletes so a `tail -f` on the
    /// dashboard keeps its file handle.
    pub fn clear_trades(&self) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        fs::write(self.trades_path(), b"")
    }

    /// Read a command without consuming it.
    ///
    /// `Ok(None)` covers both "no command file" and "command file is blank" —
    /// the dashboard clears the file after acting, and neither state is an
    /// error.
    pub fn read_command(&self) -> io::Result<Option<Result<DashboardCommand, CommandError>>> {
        match fs::read_to_string(self.command_path()) {
            Ok(text) => {
                let text = text.trim();
                if text.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(DashboardCommand::parse(text)))
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Read the command and delete the file, so one keystroke is one action.
    ///
    /// The file is removed even when the parse failed: a dashboard that writes
    /// garbage would otherwise wedge every later command behind it, and the
    /// error is returned either way.
    pub fn take_command(&self) -> io::Result<Option<Result<DashboardCommand, CommandError>>> {
        let path = self.command_path();
        let taken = match self.read_command() {
            Ok(taken) => taken,
            Err(e) => return Err(e),
        };
        match (taken, fs::remove_file(&path)) {
            (Some(command), Ok(())) => Ok(Some(command)),
            (Some(command), Err(e)) if e.kind() == io::ErrorKind::NotFound => Ok(Some(command)),
            (Some(_), Err(e)) => Err(e),
            (None, _) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::order::{Fill, OrderId};
    use crate::core::pnl::Trade;
    use crate::core::telemetry::Telemetry;
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    fn scratch(name: &str) -> PathBuf {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        std::env::temp_dir()
            .join(format!("uets-gui-{}", std::process::id()))
            .join(format!(
                "{}-{}",
                name,
                SEQ.fetch_add(1, AtomicOrdering::Relaxed)
            ))
    }

    fn marks(price: f64) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert("BTCUSDT".to_string(), price);
        m
    }

    /// A portfolio holding one long, one banked winner, marked at 150.
    fn booked() -> (Portfolio, BTreeMap<String, f64>) {
        let mut portfolio = Portfolio::new(1_000.0, 10.0).unwrap();
        portfolio.apply_fill(&Fill::new(
            OrderId::new("a"),
            "BTCUSDT",
            Side::Buy,
            1.0,
            100.0,
            1.0,
        ));
        portfolio.record_trade(Trade::new(
            "BTCUSDT",
            Side::Buy,
            100.0,
            120.0,
            1.0,
            1.0,
            1,
            2,
        ));
        let marks = marks(150.0);
        portfolio.mark(&marks);
        (portfolio, marks)
    }

    #[test]
    fn capture_reads_one_consistent_frame_from_the_live_objects() {
        let (portfolio, marks) = booked();
        let telemetry = Telemetry::new();
        telemetry.record_row();
        telemetry.record_event(telemetry.start_ms());
        let state = DashboardState::capture(
            "strategy1",
            "binance",
            "BTCUSDT",
            &portfolio,
            &marks,
            &telemetry.snapshot(),
        );

        assert_eq!(state.active_strategy, "strategy1");
        assert_eq!(state.starting_equity, 1_000.0);
        // Cash follows the fills: only the open fill's fee lands here, while
        // the banked trade's own fee belongs to the ledger's net pnl.
        assert_eq!(state.cash, 999.0);
        assert_eq!(state.unrealized_pnl, 50.0);
        assert_eq!(state.equity, 1_049.0, "cash plus the open 50");
        assert_eq!(state.gross_exposure, 150.0);
        assert_eq!(state.realized_pnl, 19.0, "+20 gross less 2.0 of fees");
        assert_eq!(state.telemetry.rows, 1);

        assert_eq!(state.positions.len(), 1);
        let position = &state.positions[0];
        assert_eq!(position.symbol, "BTCUSDT");
        assert_eq!(position.side, Side::Buy);
        assert_eq!(position.mark, 150.0);
        assert_eq!(position.unrealized_pnl, 50.0);

        // `mark()` above recorded 1_049 into the curve, which is what makes the
        // dashboard's cumulative return the live figure rather than a zero.
        assert!((state.stats.cumulative_return - 0.049).abs() < 1e-12);
        assert_eq!(state.stats.wins, 1);
        assert_eq!(state.stats.losses, 0);
    }

    #[test]
    fn an_open_position_without_a_mark_reports_its_entry_rather_than_a_number() {
        let mut portfolio = Portfolio::new(1_000.0, 10.0).unwrap();
        portfolio.apply_fill(&Fill::new(
            OrderId::new("a"),
            "ETHUSDT",
            Side::Sell,
            2.0,
            10.0,
            0.0,
        ));
        let state = DashboardState::capture(
            "s",
            "stub",
            "ETHUSDT",
            &portfolio,
            &BTreeMap::new(),
            &TelemetrySnapshot::default(),
        );
        let position = &state.positions[0];
        assert_eq!(position.side, Side::Sell);
        assert_eq!(position.mark, position.avg_entry);
        assert_eq!(position.unrealized_pnl, 0.0, "no mark, no invented pnl");
    }

    #[test]
    fn the_frame_is_json_a_text_editor_can_read() {
        let (portfolio, marks) = booked();
        let state = DashboardState::capture(
            "strategy1",
            "binance",
            "BTCUSDT",
            &portfolio,
            &marks,
            &TelemetrySnapshot::default(),
        );
        let json = state.to_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"active_strategy\":\"strategy1\""));
        assert!(json.contains("\"cumulative_return\":0.049"));
        assert!(json.contains("\"win_rate\":1"));
        assert!(json.contains("\"positions\":[{\"symbol\":\"BTCUSDT\",\"side\":\"long\""));
        assert!(
            json.contains("\"sortino\":"),
            "the headline ratio is present"
        );
    }

    #[test]
    fn the_trade_log_appends_a_line_per_trade_and_can_be_cleared() {
        let dir = scratch("trades");
        let bridge = GuiBridge::new(&dir);
        let trade = Trade::new("BTCUSDT", Side::Buy, 100.0, 120.0, 1.0, 1.0, 4, 9);
        bridge.append_trade(&trade).unwrap();
        bridge.append_trade(&trade).unwrap();
        let lines: Vec<String> = fs::read_to_string(bridge.trades_path())
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(lines.len(), 2, "append, never rewrite");
        assert!(lines[0].starts_with("{\"symbol\":\"BTCUSDT\""));
        assert!(lines[0].contains("\"net_pnl\":19"));
        assert!(lines[0].contains("\"win\":true"));
        assert!(lines[0].contains("\"exit_seq\":9"));

        bridge.clear_trades().unwrap();
        assert_eq!(fs::read_to_string(bridge.trades_path()).unwrap(), "");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn publishing_creates_the_directory_and_rewrites_the_frame() {
        let dir = scratch("publish");
        let bridge = GuiBridge::new(dir.join("nested"));
        let (portfolio, marks) = booked();
        let state = DashboardState::capture(
            "strategy1",
            "binance",
            "BTCUSDT",
            &portfolio,
            &marks,
            &TelemetrySnapshot::default(),
        );
        let path = bridge.publish(&state).unwrap();
        assert!(path.exists());
        assert!(
            fs::read_to_string(&path)
                .unwrap()
                .contains("\"equity\":1049")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_command_parses_from_the_line_a_dashboard_would_write() {
        assert_eq!(
            DashboardCommand::parse("{\"action\":\"pause\"}").unwrap(),
            DashboardCommand::Pause
        );
        assert_eq!(
            DashboardCommand::parse(" {\"action\":\"flatten\"} \n").unwrap(),
            DashboardCommand::Flatten
        );
        assert_eq!(
            DashboardCommand::parse(r#"{"action":"set_strategy","value":"strategy2"}"#).unwrap(),
            DashboardCommand::SetStrategy("strategy2".to_string())
        );
        // Unknown fields are ignored, so a newer dashboard still works here.
        assert_eq!(
            DashboardCommand::parse("{\"ts\":12,\"action\":\"resume\"}").unwrap(),
            DashboardCommand::Resume
        );
        // Escapes are passed through raw: identifiers need no unescaper.
        assert_eq!(
            DashboardCommand::parse("{\"action\":\"set_strategy\",\"value\":\"a\\\"b\"}").unwrap(),
            DashboardCommand::SetStrategy("a\\\"b".to_string())
        );
        for command in [
            DashboardCommand::Flatten,
            DashboardCommand::Pause,
            DashboardCommand::Resume,
            DashboardCommand::ClearTrades,
            DashboardCommand::SetStrategy("s".to_string()),
        ] {
            let line = format!("{{\"action\":\"{}\"}}", command.as_str());
            if let DashboardCommand::SetStrategy(name) = &command {
                assert_eq!(
                    DashboardCommand::parse(&line),
                    Err(CommandError::MissingValue("set_strategy")),
                    "a strategy switch without a name ({name}) must not silently pass"
                );
            } else {
                assert_eq!(DashboardCommand::parse(&line).unwrap(), command);
            }
        }
    }

    #[test]
    fn a_bad_command_says_which_kind_of_bad_it_is() {
        assert_eq!(DashboardCommand::parse("   "), Err(CommandError::Empty));
        assert_eq!(
            DashboardCommand::parse("{\"note\":\"hello\"}"),
            Err(CommandError::MissingAction)
        );
        assert_eq!(
            DashboardCommand::parse("{\"action\":\"explode\"}"),
            Err(CommandError::UnknownAction("explode".to_string()))
        );
        assert_eq!(
            DashboardCommand::parse("{\"action\":\"set_strategy\"}")
                .unwrap_err()
                .to_string(),
            "set_strategy needs a \"value\" field"
        );
    }

    #[test]
    fn take_command_consumes_the_file_exactly_once_even_when_it_is_garbage() {
        let dir = scratch("command");
        let bridge = GuiBridge::new(&dir);
        assert_eq!(
            bridge.take_command().unwrap(),
            None,
            "no file is not an error"
        );

        fs::create_dir_all(&dir).unwrap();
        fs::write(bridge.command_path(), b"{\"action\":\"flatten\"}\n").unwrap();
        assert_eq!(
            bridge.read_command().unwrap().unwrap().unwrap(),
            DashboardCommand::Flatten
        );
        assert!(bridge.command_path().exists(), "reading leaves it in place");
        assert_eq!(
            bridge.take_command().unwrap().unwrap().unwrap(),
            DashboardCommand::Flatten
        );
        assert!(!bridge.command_path().exists(), "taking removes it");
        assert_eq!(bridge.take_command().unwrap(), None);

        // Garbage is consumed *and* reported: a poison file must not wedge the
        // loop behind it.
        fs::write(bridge.command_path(), b"{ not json").unwrap();
        assert_eq!(
            bridge.take_command().unwrap().unwrap(),
            Err(CommandError::MissingAction)
        );
        assert!(!bridge.command_path().exists());

        fs::write(bridge.command_path(), b"   \n").unwrap();
        assert_eq!(
            bridge.take_command().unwrap(),
            None,
            "a blank file is no command"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

/// One trade as a JSON line.
fn trade_json(trade: &super::pnl::Trade) -> String {
    format!(
        concat!(
            "{{\"symbol\":{},\"side\":{},\"entry_price\":{},\"exit_price\":{},\"qty\":{},",
            "\"fee\":{},\"gross_pnl\":{},\"net_pnl\":{},\"win\":{},\"entry_seq\":{},",
            "\"exit_seq\":{}}}"
        ),
        json_string(&trade.symbol),
        json_string(if trade.side.is_buy() { "long" } else { "short" }),
        json_number(trade.entry_price),
        json_number(trade.exit_price),
        json_number(trade.qty),
        json_number(trade.fee),
        json_number(trade.gross_pnl()),
        json_number(trade.net_pnl()),
        trade.is_win(),
        trade.entry_sequence,
        trade.exit_sequence,
    )
}
