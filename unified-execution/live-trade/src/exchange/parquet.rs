//! Historical replay "exchange" — feeds a pre-converted parquet replay stream
//! into the SAME execution loop live-trade uses, instead of a live WebSocket.
//!
//! `unified_runner.py live-test` reads the raw research parquet files
//! (`research/csv/{TICKER}_trades.parquet` + `{TICKER}_orderbook.parquet`)
//! via pyarrow/polars and merges them into one timestamp-ordered, one-event-
//! per-line replay stream (default `build/live_test_feed_{ticker}.replay`).
//! This exchange parses that stream *lazily*: the feed file is read through a
//! buffered line reader in `REPLAY_BATCH`-sized chunks, never materialized in
//! full, so a multi-GB replay (BTCUSDT ~223M trades) stays RAM-bounded -- the
//! same chunked-feed idea `ml_state_optimized` uses for its parquet, but
//! applied to the *feed* here rather than feature recompute. Exactly one
//! continuous event stream enters the SAME execution loop live-trade uses;
//! because nothing is re-initialized at chunk boundaries, every indicator
//! stays warm across the whole replay (no warm-up buffer required). Everything
//! downstream (1 s micro-bars, 300 s composite features, entry/exit gates,
//! RocksDB persistence, HTML report) is the *identical* code path as live
//! trading.
//!
//! When the stream is exhausted, `next_events()` sets the shared `stop` flag
//! (if one was given) so `run_execution_loop` drains and emits the final
//! report — the replay equivalent of a Ctrl-C on a live session. With
//! `loop_feed` the exchange instead rewinds and plays forever (a live-style
//! continuous sim that keeps running until Ctrl-C).
//!
//! Replay stream format (one event per line, `#` header lines ignored):
//! ```text
//! #TICKER btcusdt
//! #CLOSES 64520.1,64510.0,...
//! T <ts_ms> <price> <amount> <buy|sell> <id>
//! B <ts_ms> <bid_price> <bid_amount> <ask_price> <ask_amount>
//! ```

use super::{ExchangeError, ExchangeKind, Feed, MarketBookTop, MarketEvent, MarketTrade, TradeSide};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// How many events to hand to the loop per `next_events()` call. The engine
/// does per-event work (micro-bar aggregation), so a 2k batch behaves like a
/// busy WebSocket frame without piling the whole file into the loop at once.
const REPLAY_BATCH: usize = 2048;

/// Millisecond epoch timestamp of any market event (used by the window filter).
fn event_ts(ev: &MarketEvent) -> i64 {
    match ev {
        MarketEvent::Trade(t) => t.timestamp,
        MarketEvent::BookTop(b) => b.timestamp,
    }
}

/// A replay "venue": serves pre-recorded historical events as a live feed.
///
/// The feed file is read **lazily** (a buffered line reader), not slurped into
/// memory, so a multi-GB replay (BTCUSDT is 223M trades) stays bounded. Headers
/// are consumed once on connect; events are parsed only as the loop asks for
/// them, `REPLAY_BATCH` at a time. Parallel to how research chunks the parquet:
/// chunked feeding into ONE continuous engine instance keeps every indicator
/// warm (timestamps stay globally ordered, nothing is duplicated), so no
/// recursive-feature reset can happen — unlike research's per-chunk feature
/// recompute, which needs warm-up buffers.
pub struct ReplayExchange {
    feed_path: String,
    ticker: String,
    /// Composite closes from the optional `#CLOSES` header (feature warm-up).
    closes: Vec<f64>,
    /// Lazily-opened feed reader (None until `connect`).
    reader: Option<BufReader<File>>,
    /// Events parsed but not yet handed to the loop (max REPLAY_BATCH_SIZE).
    pending: VecDeque<MarketEvent>,
    /// An already-read-but-not-parsed first event line (put back by header scan).
    pushback: Option<String>,
    /// Keep only events with `ts >= start_ms` (optional).
    pub start_ms: Option<i64>,
    /// Keep only events with `ts <= end_ms` (optional).
    pub end_ms: Option<i64>,
    /// Stop after `limit` kept events (optional; use to cap a huge replay).
    pub limit: Option<usize>,
    /// Rewind to the start once exhausted (continuous sim) instead of stopping.
    loop_feed: bool,
    /// Set when the (non-looping) feed is exhausted so the loop drains.
    stop: Option<Arc<AtomicBool>>,
    /// Pending one-shot rewind for the next `next_events()` call (loop mode).
    needs_rewind: bool,
    /// Total kept events emitted so far (drives `limit` + rewind bookkeeping).
    emitted: usize,
}
impl ReplayExchange {
    pub fn new(
        feed_path: &str,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        limit: Option<usize>,
        loop_feed: bool,
        stop: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            feed_path: feed_path.to_string(),
            ticker: String::new(),
            closes: Vec::new(),
            reader: None,
            pending: VecDeque::new(),
            pushback: None,
            start_ms,
            end_ms,
            limit,
            loop_feed,
            stop,
            needs_rewind: false,
            emitted: 0,
        }
    }

    /// Consume the `#` header lines at the top of the feed (updating `ticker`
    /// / `closes`). The first content line, if any, is stashed in `pushback`
    /// so it isn't lost.
    fn scan_headers(&mut self) -> std::io::Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = {
                let reader = self.reader.as_mut().ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotConnected, "not connected")
                })?;
                reader.read_line(&mut line)?
            };
            if n == 0 {
                return Ok(()); // header-only file; next_events will report EOF
            }
            if !line.trim_start().starts_with('#') {
                self.pushback = Some(line);
                return Ok(());
            }
            self.parse_line(&line); // reader borrow was scoped above; safe to touch self
        }
    }

    /// Rewind to the start of the feed (loop mode) — headers re-read, counters
    /// reset, buffered events dropped (they have already been delivered).
    fn rewind(&mut self) -> Result<(), ExchangeError> {
        {
            let reader = self
                .reader
                .as_mut()
                .ok_or_else(|| ExchangeError::Other("replay exchange not connected".into()))?;
            reader
                .seek(SeekFrom::Start(0))
                .map_err(|e| ExchangeError::Other(format!("replay rewind failed: {e}")))?;
        }
        self.pending.clear();
        self.pushback = None;
        self.emitted = 0;
        self.scan_headers()
            .map_err(|e| ExchangeError::Other(format!("replay header re-parse failed: {e}")))?;
        println!("[replay] feed exhausted — looping back to start");
        Ok(())
    }

    /// One line of the replay stream. Header lines (`#...`) update `self`
    /// state, event lines produce a [`MarketEvent`].
    fn parse_line(&mut self, line: &str) -> Option<MarketEvent> {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            return None;
        }

        if let Some(rest) = line.strip_prefix('#') {
            let (key, value) = rest.split_once(' ').unwrap_or((rest, ""));
            match key {
                "TICKER" => self.ticker = value.trim().to_ascii_lowercase(),
                "CLOSES" => {
                    for c in value.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                        if let Ok(v) = c.parse::<f64>() {
                            self.closes.push(v);
                        }
                    }
                }
                _ => {}
            }
            return None;
        }

        let mut it = line.splitn(6, ' ');
        let kind = it.next()?;
        let ts: i64 = it.next()?.parse().ok()?;
        match kind {
            "T" => {
                let price: f64 = it.next()?.parse().ok()?;
                let amount: f64 = it.next()?.parse().ok()?;
                let side = it.next()?;
                let id = it.next()?.to_string();
                Some(MarketEvent::Trade(MarketTrade {
                    timestamp: ts,
                    price,
                    amount,
                    side: if side == "buy" {
                        TradeSide::Buy
                    } else {
                        TradeSide::Sell
                    },
                    id,
                }))
            }
            "B" => {
                let bid_price: f64 = it.next()?.parse().ok()?;
                let bid_amount: f64 = it.next()?.parse().ok()?;
                let ask_price: f64 = it.next()?.parse().ok()?;
                let ask_amount: f64 = it.next()?.parse().ok()?;
                Some(MarketEvent::BookTop(MarketBookTop {
                    timestamp: ts,
                    bid_price,
                    bid_amount,
                    ask_price,
                    ask_amount,
                }))
            }
            _ => None,
        }
    }
}
#[async_trait::async_trait]
impl Feed for ReplayExchange {
    fn kind(&self) -> ExchangeKind {
        ExchangeKind::Parquet
    }

    async fn connect(&mut self, ticker: &str) -> Result<(), ExchangeError> {
        self.ticker = ticker.to_ascii_lowercase();
        // Open the feed file but do NOT slurp it — headers are read once here,
        // events are parsed lazily by `next_events`.
        let file = File::open(&self.feed_path).map_err(|e| {
            ExchangeError::Other(format!("replay feed {} unreadable: {e}", self.feed_path))
        })?;
        self.reader = Some(BufReader::new(file));
        self.scan_headers().map_err(|e| {
            ExchangeError::Other(format!(
                "replay feed {} header parse failed: {e}",
                self.feed_path
            ))
        })?;
        println!(
            "[replay] connected ticker={} closes={} (events streamed in {}-batch chunks; RSS bounded)",
            self.ticker,
            self.closes.len(),
            REPLAY_BATCH
        );
        Ok(())
    }

    async fn next_events(&mut self) -> Result<Vec<MarketEvent>, ExchangeError> {
        // If the previous call hit EOF while in loop mode, restart now.
        if self.needs_rewind {
            self.rewind()?;
            self.needs_rewind = false;
        }

        // Deliver any events parsed but not yet handed over first.
        if !self.pending.is_empty() {
            let n = self.pending.len().min(REPLAY_BATCH);
            return Ok(self.pending.drain(..n).collect());
        }

        let mut line_buf = String::new();
        let mut eof = false;
        let mut stop_now = false;

        // Parse only enough lines to refill up to REPLAY_BATCH events. The
        // reader borrow is scoped to a block so it never overlaps the mutable
        // `self.parse_line` / `self.pending` calls below.
        while self.pending.len() < REPLAY_BATCH {
            let line = if let Some(pb) = self.pushback.take() {
                pb
            } else {
                line_buf.clear();
                let n = {
                    let reader = self.reader.as_mut().ok_or_else(|| {
                        ExchangeError::Other("replay exchange not connected".into())
                    })?;
                    reader
                        .read_line(&mut line_buf)
                        .map_err(|e| ExchangeError::Other(format!("replay read error: {e}")))?
                };
                if n == 0 {
                    eof = true;
                    break;
                }
                line_buf.clone()
            };

            if let Some(ev) = self.parse_line(&line) {
                let ts = event_ts(&ev);
                if self.start_ms.map_or(false, |s| ts < s) {
                    continue;
                }
                if self.end_ms.map_or(false, |e| ts > e) {
                    continue;
                }
                self.pending.push_back(ev);
                self.emitted += 1;
                if self.limit.map_or(false, |l| self.emitted >= l) {
                    stop_now = true;
                    break;
                }
            }
        }

        // Signal the loop to finalize (or schedule a rewind for loop mode).
        if eof || stop_now {
            if self.loop_feed {
                self.needs_rewind = true;
            } else if let Some(stop) = &self.stop {
                stop.store(true, Ordering::SeqCst);
            }
        }

        let n = self.pending.len().min(REPLAY_BATCH);
        Ok(self.pending.drain(..n).collect())
    }

    /// Historical composite closes for feature-ring warm-up come from the
    /// `#CLOSES` header (optional; omitted => features warm live from the feed).
    async fn historical_closes(&self, _ticker: &str, bars: u32) -> Result<Vec<f64>, ExchangeError> {
        let from = self.closes.len().saturating_sub(bars as usize);
        Ok(self.closes[from..].to_vec())
    }

    async fn close(&mut self) -> Result<(), ExchangeError> {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn temp_replay(tag: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "imc_replay_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("feed.replay");
        std::fs::write(&path, body).expect("write replay feed");
        path
    }

    fn cleanup(feed: &std::path::Path) {
        if let Some(dir) = feed.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn parses_and_replays_merged_feed() {
        let feed = temp_replay(
            "basic",
            "#TICKER btcusdt\n\
             #CLOSES 100.0,100.5,101.0\n\
             T 100 100.0 0.5 buy 1\n\
             B 101 99.9 1.0 100.1 1.0\n\
             T 102 100.2 0.2 sell 2\n",
        );
        let stop = Arc::new(AtomicBool::new(false));
        let mut ex = ReplayExchange::new(
            feed.to_str().expect("utf8 path"),
            None,
            None,
            None,
            false,
            Some(stop.clone()),
        );
        ex.connect("btcusdt").await.expect("connect");
        assert_eq!(ex.ticker, "btcusdt");
        assert_eq!(ex.closes, vec![100.0, 100.5, 101.0]);

        let batch = ex.next_events().await.expect("next events");
        assert_eq!(batch.len(), 3, "all events returned in one batch");
        match &batch[0] {
            MarketEvent::Trade(t) => {
                assert_eq!(t.timestamp, 100);
                assert_eq!(t.side, TradeSide::Buy);
                assert_eq!(t.id, "1");
                assert_eq!(t.amount, 0.5);
            }
            _ => panic!("first event must be a trade"),
        }
        match &batch[1] {
            MarketEvent::Trade(_) => panic!("second event must be the book top"),
            MarketEvent::BookTop(b) => {
                assert_eq!(b.timestamp, 101);
                assert_eq!(b.bid_price, 99.9);
                assert_eq!(b.ask_price, 100.1);
            }
        }
        match &batch[2] {
            MarketEvent::Trade(t) => {
                assert_eq!(t.timestamp, 102);
                assert_eq!(t.side, TradeSide::Sell);
            }
            _ => panic!("third event must be a trade"),
        }

        let eof = ex.next_events().await.expect("eof returns Ok");
        assert!(eof.is_empty(), "feed exhausted => empty batch");
        assert!(stop.load(Ordering::SeqCst), "EOF must flip the stop flag");
        cleanup(&feed);
    }

    #[tokio::test]
    async fn window_and_limit_filter_events() {
        let body = "T 10 1 0.1 buy 1\nT 20 2 0.1 sell 2\nT 30 3 0.1 buy 3\n";
        let feed = temp_replay("window", body);

        let mut ex = ReplayExchange::new(
            feed.to_str().expect("utf8 path"),
            Some(15),
            Some(25),
            None,
            false,
            None,
        );
        ex.connect("btcusdt").await.expect("connect");
        let batch = ex.next_events().await.expect("next events");
        assert_eq!(batch.len(), 1, "window [15,25] keeps only ts=20");
        match &batch[0] {
            MarketEvent::Trade(t) => assert_eq!(t.timestamp, 20),
            _ => panic!("trade expected"),
        }

        let mut capped =
            ReplayExchange::new(feed.to_str().unwrap(), None, None, Some(2), false, None);
        capped.connect("btcusdt").await.expect("connect");
        let batch = capped.next_events().await.expect("next events");
        assert_eq!(batch.len(), 2, "limit=2 caps the replay");
        cleanup(&feed);
    }

    #[tokio::test]
    async fn historical_closes_are_returned() {
        let feed = temp_replay(
            "closes",
            "#CLOSES 500.0,501.0,502.0,503.0\nT 1 502.0 0.1 buy 1\n",
        );
        let mut ex = ReplayExchange::new(feed.to_str().unwrap(), None, None, None, false, None);
        ex.connect("btcusdt").await.expect("connect loads feed");
        let closes = ex.historical_closes("btcusdt", 2).await.expect("closes");
        assert_eq!(closes, vec![502.0, 503.0], "trailing `bars` closes");
        cleanup(&feed);
    }
}
