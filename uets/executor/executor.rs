//! Dispatcher for the Binance websocket readers in `connectors/binance/websocket/reader.rs`.
//!
//! Run any reader directly through this binary:
//!
//!     cargo run --bin dev_test -- market btcusdt aggTrade
//!     cargo run --bin dev_test -- help
//!
//! The Python extension module (lib.rs) is a separate build target -- build
//! it with `maturin develop` (maturin enables the extension-module feature
//! automatically).

use std::path::PathBuf;
use std::process::ExitCode;

use uets::connectors::binance::websocket::{diagnostic, live, monitor, reader};
use uets::live_strategy::{self, ComputeFeatures, LiveStrategy, StateBuilder};

/// Default capture directory, `market_data/live_db` at the repo root.
///
/// Resolved from the manifest directory rather than the process CWD, because
/// this binary is normally run through `cargo run` from anywhere and a capture
/// landing in a random CWD is worse than one landing in a predictable place.
fn default_live_db_dir() -> String {
    // CARGO_MANIFEST_DIR is <repo>/uets, so exactly one parent reaches the
    // repo root. Two would land beside the repo.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("market_data").join("live_db"))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "market_data/live_db".to_string())
}

/// Default diagnostic directory: `out/live` under the package (beside the
/// monitor's frames), resolved from the manifest rather than the CWD.
fn default_out_dir() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("out")
        .join("live")
        .display()
        .to_string()
}

/// The executor's live strategy selection — test 2's subject.
///
/// `executor.rs` is the only place that decides *which* bundle runs. A
/// selection swaps the whole [`LiveStrategy`] — state builder
/// (`state_builder.rs`) and feature builder (`compute_feature.rs`) travel
/// together — and a failed switch leaves the previous selection untouched,
/// so a typo in a config can never half-configure the process.
struct StrategySwitch {
    active: Option<String>,
    bundle: Option<LiveStrategy>,
}

impl StrategySwitch {
    fn new() -> Self {
        Self {
            active: None,
            bundle: None,
        }
    }

    /// Strategies compiled into this binary.
    fn available(&self) -> Vec<&'static str> {
        live_strategy::available_strategies()
    }

    /// Strategy folders on disk (`uets/live_strategy/*`).
    fn discovered(&self) -> Vec<String> {
        live_strategy::discover_strategy_dirs()
    }

    fn active(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// Switch the selection to `name`. Unknown names change nothing.
    fn select(&mut self, name: &str) -> Result<(), String> {
        let bundle = live_strategy::load_strategy(name)?;
        self.active = Some(name.to_string());
        self.bundle = Some(bundle);
        Ok(())
    }

    /// The selected bundle's state builder (`state_builder.rs`).
    fn state_builder(&self) -> Result<&dyn StateBuilder, String> {
        self.bundle
            .as_ref()
            .map(|bundle| bundle.builder.as_ref())
            .ok_or_else(|| "no strategy selected".to_string())
    }

    /// The selected bundle's feature builder (`compute_feature.rs`).
    fn feature_builder_mut(&mut self) -> Result<&mut (dyn ComputeFeatures + 'static), String> {
        self.bundle
            .as_mut()
            .map(|bundle| bundle.computer.as_mut())
            .ok_or_else(|| "no strategy selected".to_string())
    }
}

impl Default for StrategySwitch {
    fn default() -> Self {
        Self::new()
    }
}

fn print_help() {
    println!(
        "usage: dev_test [command] [args...]\n\
         \n\
         commands:\n\
         \x20 market [SYMBOL] [TYPE]  public market stream (default: btcusdt aggTrade)\n\
         \x20 depth [SYMBOL]          diff-depth stream @100ms (default: btcusdt)\n\
         \x20 kline [SYMBOL]          1-minute kline stream (default: btcusdt)\n\
         \x20 ticker [SYMBOL]         24h rolling ticker (default: btcusdt)\n\
         \x20 all-tickers             all-symbols mini-ticker stream (!miniTicker@arr)\n\
         \x20 multi [SYMBOLS...]      multiple diff-depth streams on one connection\n\
         \x20 record [SYMBOL] [DIR]    live capture -> merged CSV under DIR\n\
         \x20                         (default DIR: market_data/live_db)\n\
         \x20                         flags: --rows N   stop after N rows\n\
         \x20                                --append  keep the existing file\n\
         \x20 monitor [SYMBOL]        live tape + 20-level orderbook monitor;\n\
         \x20                         view with:  python uets/display.py [SYMBOL]\n\
         \x20 diag [SYMBOL] [DIR]      diagnostic CSV: one row per event with\n\
         \x20                         update_type=aggTrades|orderbook and the\n\
         \x20                         other stream's columns as NaN\n\
         \x20                         (default DIR: uets/out/live)\n\
         \x20 strategies               list compiled vs on-disk strategy bundles\n\
         \x20 user-stream             REST listen-key lifecycle (needs API key)\n\
         \x20 user-stream-ws          account/order update stream (needs API key)\n\
         \x20 help                    show this message\n\
         \n\
         examples:\n\
         \x20 dev_test market btcusdt aggTrade\n\
         \x20 dev_test depth btcusdt\n\
         \x20 dev_test multi ethbtc bnbusdt\n\
         \x20 dev_test record btcusdt            # -> market_data/live_db\n\
         \n\
         All streams run until interrupted (Ctrl+C)."
    );
}

/// Live capture orchestration: websockets -> merge -> the 128-column CSV.
///
/// This mirrors `uets::connectors::binance::websocket::live::run_capture` but fixes two defects
/// that could not be fixed there (`live.rs` is frozen; only this file may
/// change -- see `docs/system_design/uets.md` for the network history):
///
/// * **No aggTrades in the output.** On some networks `fstream.binance.com`
///   serves the depth stream but delivers *zero* `@aggTrade` frames (the
///   exact symptom documented under "Binance futures data degraded on some
///   networks"). This capture keeps the futures trade feed but also
///   subscribes to the proven spot feed (`stream.binance.com:9443`); when
///   the futures feed stays silent for [`TRADE_FALLBACK`] the spot feed is
///   added, and single-venue ownership guarantees at most one trade venue
///   contributes rows at a time.
/// * **UNSEEDED/STALE book.** The old flow seeded *before* connecting and
///   gave up after one attempt. Here the sockets connect first, incoming
///   events are staged until a REST snapshot lands (so the snapshot covers
///   every buffered diff -- no phantom startup gap), and a background thread
///   retries the seed forever, doubling as the gap re-seed.
mod record {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use event_merger_live::{BookSnapshot, CsvRecorder, MarketEvent, Merger, ReorderBuffer};
    use tungstenite::{Message, connect};

    use super::live;

    /// No frames at all for this long (sockets connected but silent) -> give up.
    const STALL_TIMEOUT: Duration = Duration::from_secs(30);
    /// No *seed* for this long -> give up (REST unreachable despite retries).
    const SEED_TIMEOUT: Duration = Duration::from_secs(60);
    /// Poll interval of the seed / re-seed thread while a seed is needed.
    const RESEED_POLL: Duration = Duration::from_millis(500);
    /// Backoff after a failed seed attempt (bounds DNS-failure log spam).
    const RESEED_BACKOFF: Duration = Duration::from_secs(3);
    /// Heartbeat cadence.
    const HEARTBEAT: Duration = Duration::from_secs(1);
    /// Futures trade feed silent this long -> also subscribe the spot feed.
    const TRADE_FALLBACK: Duration = Duration::from_secs(10);
    /// Current trade venue silent this long -> ownership may move to the other feed.
    const VENUE_TAKEOVER: Duration = Duration::from_secs(30);
    /// Cap on events buffered before the first successful seed.
    const STAGE_CAP: usize = 50_000;
    /// Reorder buffer capacity (matches `live::run_capture`).
    const REORDER_CAP: usize = 65_536;

    pub(crate) const VENUE_NONE: u8 = 0;
    pub(crate) const VENUE_FUTURES: u8 = 1;
    pub(crate) const VENUE_SPOT: u8 = 2;

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    fn venue_name(venue: u8) -> &'static str {
        match venue {
            VENUE_FUTURES => "futures",
            VENUE_SPOT => "spot",
            _ => "none",
        }
    }

    /// Shared, mutex-guarded capture state.
    ///
    /// Both websocket threads push into this; emission order is decided in
    /// exactly one place: the [`ReorderBuffer`].
    pub(crate) struct Capture {
        pub(crate) merger: Merger,
        pub(crate) reorder: ReorderBuffer,
        csv: CsvRecorder,
        /// A gap was detected; the seed thread re-seeds while this is set.
        pub(crate) reseeds_pending: bool,
        /// False until the first REST snapshot has been applied.
        pub(crate) seeded: bool,
        /// Events received before the first seed, held until it lands.
        staged: Vec<MarketEvent>,
        /// Events dropped because the pre-seed buffer was full.
        staged_dropped: u64,
        /// Diffs dropped because the snapshot already covers them (the
        /// Binance "discard events with `u < lastUpdateId`" rule).
        covered_dropped: u64,
        /// Trade rows refused because the other venue owns the trade stream.
        foreign_dropped: u64,
        /// Every event that entered [`Capture::offer`], seeded or not.
        pub(crate) events_seen: u64,
        /// Venue currently allowed to emit trade rows.
        pub(crate) trade_owner: u8,
        /// When the owner last emitted a trade (drives takeover/fallback).
        pub(crate) last_owner_trade_ms: u64,
    }

    impl Capture {
        /// Open (or append to) the capture file and write the header
        /// immediately, so a capture that never connects still leaves a
        /// readable CSV.
        pub(crate) fn new(
            path: &std::path::Path,
            append: bool,
            levels: usize,
            lateness_ms: i64,
        ) -> Result<Self, String> {
            let csv = if append {
                CsvRecorder::append(path)
            } else {
                CsvRecorder::create(path)
            }
            // One row, one syscall.
            //
            // This is the live capture path, and its output is read by a human
            // or a trader watching the file *while it is being written*. The
            // previous `.flush_every(64)` batched 64 rows -- roughly a minute
            // of market time -- into the OS at once, so the file sat flat and
            // then jumped. Measured: 104 of 116 samples had zero new rows and
            // 5 had >=20, which is exactly that flat-then-burst profile.
            //
            // At ~10 rows/s the syscall cost is irrelevant; what matters is
            // that a row lands on disk when it is produced, not a minute later.
            // The reorder buffer is what actually decides *when* a row becomes
            // emittable, and that is a separate problem (see the watermark
            // note on `Capture::emit`); this only removes the sink-side batching.
            .map_err(|e| e.to_string())?
            .flush_every(1);
            let mut capture = Self {
                merger: Merger::new(levels),
                reorder: ReorderBuffer::new(lateness_ms, REORDER_CAP)
                    // The watermark is exchange time, so it only advances when
                    // an event arrives. On a quiet book that holds rows hostage
                    // until the next flurry moves it, and the flurry then
                    // releases the whole backlog at once: measured on live
                    // BTCUSDT, 41% of rows shared an identical timestamp with
                    // their predecessor, which is a batch release rather than a
                    // 100 ms stream. Age bounds the wait in wall-clock terms so
                    // a row is never held longer than this regardless of
                    // whether the exchange is talking.
                    //
                    // One cadence tick, not the 200ms lateness window: the
                    // guard should only fire on a genuine stall, never during
                    // ordinary traffic, so the ordering guarantee is untouched
                    // in the normal case.
                    .max_hold(Duration::from_millis(120)),
                csv,
                reseeds_pending: false,
                seeded: false,
                staged: Vec::new(),
                staged_dropped: 0,
                covered_dropped: 0,
                foreign_dropped: 0,
                events_seen: 0,
                trade_owner: VENUE_NONE,
                last_owner_trade_ms: now_ms(),
            };
            capture.csv.write_header().map_err(|e| e.to_string())?;
            capture.csv.flush().map_err(|e| e.to_string())?;
            Ok(capture)
        }

        /// Single-venue ownership for trade rows.
        ///
        /// The first venue to deliver a trade claims the stream. Ownership
        /// moves to the other feed only after the owner has been silent for
        /// [`VENUE_TAKEOVER`], so a healthy feed is never overruled and a
        /// dead one never blocks recovery. Returns false when the event must
        /// be dropped.
        fn claim_trade(&mut self, venue: u8) -> bool {
            if venue == VENUE_NONE {
                return true;
            }
            let now = now_ms();
            if self.trade_owner == venue {
                self.last_owner_trade_ms = now;
                return true;
            }
            if self.trade_owner == VENUE_NONE {
                self.trade_owner = venue;
                self.last_owner_trade_ms = now;
                eprintln!("[capture] trade source: {}", venue_name(venue));
                return true;
            }
            if now.saturating_sub(self.last_owner_trade_ms) >= VENUE_TAKEOVER.as_millis() as u64 {
                eprintln!(
                    "[capture] trade source: {} -> {} (old source silent {:?})",
                    venue_name(self.trade_owner),
                    venue_name(venue),
                    VENUE_TAKEOVER
                );
                self.trade_owner = venue;
                self.last_owner_trade_ms = now;
                return true;
            }
            self.foreign_dropped += 1;
            false
        }

        /// Accept one event: gate trades by venue, stage until seeded, then emit.
        pub(crate) fn offer(&mut self, event: MarketEvent, venue: u8) -> Result<u64, String> {
            self.events_seen += 1;
            if matches!(event, MarketEvent::Trade(_)) && !self.claim_trade(venue) {
                return Ok(0);
            }
            if !self.seeded {
                if self.staged.len() < STAGE_CAP {
                    self.staged.push(event);
                } else {
                    self.staged_dropped += 1;
                }
                return Ok(0);
            }
            self.emit(event)
        }

        /// Merge one already-reordered event and write its row.
        ///
        /// Shared by the live path ([`Capture::emit`]) and the shutdown path
        /// ([`Capture::flush_pending`]), so both apply the covered-diff rule
        /// and both flag re-seeds identically.
        fn write_ready(&mut self, event: MarketEvent) -> Result<u64, String> {
            // Binance sync rule: drop diffs the snapshot already covers.
            // Without this, applying an old diff after a fresh seed would
            // rewind book levels the seed has superseded. A re-seed lands
            // directly on the merger, so diffs waiting in the reorder buffer
            // can become covered at any moment -- checked here for both paths.
            if let MarketEvent::Diff(frame) = &event {
                if let Some(last) = self.merger.state().last_update_id {
                    if frame.final_update_id <= last {
                        self.covered_dropped += 1;
                        return Ok(0);
                    }
                }
            }
            let (row, outcome) = self.merger.merge_event(event);
            if outcome.needs_reseed {
                // Visible, not silent: the book is wrong until the seed
                // thread lands a fresh snapshot.
                self.reseeds_pending = true;
            }
            self.csv
                .write_row(&mut self.merger, &row)
                .map_err(|e| e.to_string())?;
            Ok(1)
        }

        /// Merge one event and write whatever rows become emittable.
        fn emit(&mut self, event: MarketEvent) -> Result<u64, String> {
            let ready = self.reorder.push(event);
            let mut written = 0;
            for ready_event in ready {
                written += self.write_ready(ready_event)?;
            }
            Ok(written)
        }

        /// Apply a REST snapshot, then flush everything staged before it.
        ///
        /// Staged diffs covered by the snapshot vanish under the same rule as
        /// in [`Capture::emit`], so the first applied diff joins the snapshot
        /// without a phantom gap.
        pub(crate) fn apply_seed(&mut self, snapshot: &BookSnapshot) -> Result<u64, String> {
            self.merger.apply_seed(snapshot);
            self.seeded = true;
            self.reseeds_pending = false;
            let staged = std::mem::take(&mut self.staged);
            let mut written = 0;
            for event in staged {
                written += self.emit(event)?;
            }
            Ok(written)
        }

        /// Emit everything still held by the reorder buffer (shutdown path).
        ///
        /// Drained events go straight to the merger -- pushing them back
        /// through [`Capture::emit`] would only re-buffer them behind the
        /// lateness window and lose them on exit.
        pub(crate) fn flush_pending(&mut self) -> Result<u64, String> {
            let mut written = 0;
            for event in self.reorder.flush() {
                written += self.write_ready(event)?;
            }
            Ok(written)
        }
    }

    /// Merge one event and count it against the row cap.
    fn push(
        shared: &Arc<Mutex<Capture>>,
        rows_seen: &Arc<AtomicU64>,
        keep: &Arc<AtomicBool>,
        max_rows: Option<u64>,
        event: MarketEvent,
        venue: u8,
    ) {
        let Ok(mut guard) = shared.lock() else { return };
        match guard.offer(event, venue) {
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

    /// Fire-and-forget reconnecting stream thread (same shape as the private
    /// helper in `live.rs`, plus the venue tag that gates trade rows).
    fn spawn_stream(
        url: String,
        venue: u8,
        parser: fn(&str) -> Option<MarketEvent>,
        shared: Arc<Mutex<Capture>>,
        rows_seen: Arc<AtomicU64>,
        keep: Arc<AtomicBool>,
        max_rows: Option<u64>,
    ) {
        std::thread::spawn(move || {
            loop {
                match connect(url.as_str()) {
                    Ok((mut ws, _)) => {
                        eprintln!("[capture] connected: {url}");
                        loop {
                            match ws.read() {
                                Ok(Message::Text(text)) => {
                                    if let Some(event) = parser(&text) {
                                        push(&shared, &rows_seen, &keep, max_rows, event, venue);
                                    }
                                }
                                Ok(Message::Ping(_))
                                | Ok(Message::Pong(_))
                                | Ok(Message::Binary(_))
                                | Ok(Message::Frame(_)) => (),
                                Ok(Message::Close(frame)) => {
                                    eprintln!(
                                        "[capture] closed by server: {:?} ({url})",
                                        frame.map(|f| f.code)
                                    );
                                    break;
                                }
                                Err(e) => {
                                    eprintln!("[capture] socket error: {e} ({url})");
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("[capture] connect failed: {e} ({url})"),
                }
                if !keep.load(Ordering::Relaxed) {
                    break;
                }
                eprintln!("[capture] reconnecting in 3s... ({url})");
                std::thread::sleep(Duration::from_secs(3));
            }
        });
    }

    /// Background seed thread: applies the first REST snapshot (retrying
    /// until it lands) and keeps re-seeding after sequence gaps.
    ///
    /// The old flow attempted the seed exactly once, *before* connecting, and
    /// never again -- one dropped DNS answer left the book UNSEEDED forever.
    /// This thread makes the seed self-healing: it runs while `!seeded` or
    /// while `reseeds_pending`, and backs off between failures so a dead
    /// resolver produces a line every few seconds instead of a firehose.
    fn spawn_seed_thread(
        symbol: String,
        capture: Arc<Mutex<Capture>>,
        rows_seen: Arc<AtomicU64>,
        keep: Arc<AtomicBool>,
        max_rows: Option<u64>,
    ) {
        std::thread::spawn(move || {
            let mut attempts = 0u32;
            while keep.load(Ordering::Relaxed) {
                let needed = capture
                    .lock()
                    .map(|g| !g.seeded || g.reseeds_pending)
                    .unwrap_or(false);
                if !needed {
                    std::thread::sleep(RESEED_POLL);
                    continue;
                }
                attempts += 1;
                match live::seed_from_rest(&symbol) {
                    Ok(snapshot) => {
                        let outcome = {
                            let mut guard = match capture.lock() {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                            let seeded = guard.seeded;
                            guard.apply_seed(&snapshot).map(|written| (seeded, written))
                        };
                        match outcome {
                            Ok((was_seeded, written)) => {
                                let total =
                                    rows_seen.fetch_add(written, Ordering::Relaxed) + written;
                                if max_rows.is_some_and(|limit| total >= limit) {
                                    keep.store(false, Ordering::Relaxed);
                                }
                                if was_seeded {
                                    eprintln!("[capture] book re-seeded after a sequence gap");
                                } else {
                                    eprintln!(
                                        "[capture] book seeded from REST (attempt {attempts})"
                                    );
                                }
                            }
                            Err(e) => {
                                eprintln!("[capture] write failed after seed: {e}");
                                keep.store(false, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[capture] REST seed attempt {attempts} failed: {e}");
                        std::thread::sleep(RESEED_BACKOFF);
                    }
                }
            }
        });
    }

    /// Run a live capture until interrupted, or until `max_rows` is reached.
    ///
    /// Ordering matters: sockets connect *first*, events are staged, and the
    /// seed thread lands the REST snapshot whenever the network allows. That
    /// is the reverse of the old flow and eliminates both the "UNSEEDED
    /// forever" state (the seed is retried until it succeeds) and the
    /// phantom startup gap (the snapshot covers every buffered diff).
    pub(crate) fn run(config: super::live::CaptureConfig) -> Result<(), String> {
        let symbol = config.symbol.to_lowercase();
        let keep_running = Arc::new(AtomicBool::new(true));
        let rows_seen = Arc::new(AtomicU64::new(0));

        let capture = Arc::new(Mutex::new(Capture::new(
            &config.output,
            config.append,
            config.levels,
            config.lateness_ms,
        )?));

        eprintln!("[capture] output   {}", config.output.display());
        eprintln!("[capture] symbol   {}", config.symbol);
        eprintln!("[capture] columns  {}", event_merger_live::schema().len());

        let max_rows = config.max_rows;

        // 1. Sockets first; events stage until the seed lands.
        spawn_stream(
            format!("wss://fstream.binance.com/stream?streams={symbol}@depth@100ms"),
            VENUE_NONE,
            live::parse_diff,
            Arc::clone(&capture),
            Arc::clone(&rows_seen),
            Arc::clone(&keep_running),
            max_rows,
        );
        spawn_stream(
            format!("wss://fstream.binance.com/stream?streams={symbol}@aggTrade"),
            VENUE_FUTURES,
            live::parse_trade,
            Arc::clone(&capture),
            Arc::clone(&rows_seen),
            Arc::clone(&keep_running),
            max_rows,
        );

        // 2. Seed thread: first snapshot + every gap re-seed, retrying forever.
        spawn_seed_thread(
            config.symbol.clone(),
            Arc::clone(&capture),
            Arc::clone(&rows_seen),
            Arc::clone(&keep_running),
            max_rows,
        );

        // 3. Fallback trade feed (spot). Spawned lazily from the heartbeat
        //    once the futures feed proves silent, so a healthy network keeps
        //    historical futures parity and a broken one still gets trades.
        let mut spot_spawned = false;

        let started = Instant::now();
        let mut last_beat = Duration::ZERO;

        loop {
            std::thread::sleep(Duration::from_millis(200));
            let total = rows_seen.load(Ordering::Relaxed);
            if max_rows.is_some_and(|limit| total >= limit) {
                break;
            }
            if !keep_running.load(Ordering::Relaxed) {
                break;
            }

            let (seeded, events_seen, trade_owner, last_owner_trade_ms) = {
                let guard = capture.lock().expect("capture mutex poisoned");
                (
                    guard.seeded,
                    guard.events_seen,
                    guard.trade_owner,
                    guard.last_owner_trade_ms,
                )
            };

            // Stall guards: distinguish "no network at all" from "REST seed
            // unreachable" -- both otherwise look like a hung capture.
            let elapsed = started.elapsed();
            if events_seen == 0 && elapsed > STALL_TIMEOUT {
                keep_running.store(false, Ordering::Relaxed);
                return Err(format!(
                    "no market data received within {:?}: cannot reach the websockets. \
                     Check the network, then retry.",
                    STALL_TIMEOUT
                ));
            }
            if !seeded && elapsed > SEED_TIMEOUT {
                keep_running.store(false, Ordering::Relaxed);
                return Err(format!(
                    "REST depth seed failed for {:?}: the book cannot be trusted. \
                     Check DNS/VPN for fapi.binance.com, then retry.",
                    SEED_TIMEOUT
                ));
            }

            // Lazy spot fallback: no trade venue has produced anything for
            // TRADE_FALLBACK while the capture is demonstrably alive, so the
            // documented futures-aggTrade-silent network loses no data.
            if seeded && !spot_spawned && elapsed >= TRADE_FALLBACK {
                let silent_for = now_ms().saturating_sub(last_owner_trade_ms);
                if silent_for >= TRADE_FALLBACK.as_millis() as u64 {
                    spot_spawned = true;
                    eprintln!(
                        "[capture] no aggTrade for {TRADE_FALLBACK:?} -- also subscribing \
                         the spot feed (stream.binance.com:9443)"
                    );
                    spawn_stream(
                        format!("wss://stream.binance.com:9443/ws/{symbol}@aggTrade"),
                        VENUE_SPOT,
                        live::parse_trade,
                        Arc::clone(&capture),
                        Arc::clone(&rows_seen),
                        Arc::clone(&keep_running),
                        max_rows,
                    );
                }
            }

            // Heartbeat: the evidence a capture is alive and the book is sound.
            if elapsed - last_beat >= HEARTBEAT {
                last_beat = elapsed;
                let (book_ok, gaps, seeds, staged_left, trades) = {
                    let guard = capture.lock().expect("capture mutex poisoned");
                    let c = guard.merger.counters();
                    (
                        guard.merger.state().book_valid(),
                        c.gaps,
                        c.seeds_applied,
                        guard.staged.len(),
                        c.trades,
                    )
                };
                eprintln!(
                    "[capture] rows={total} book={} trades={trades} venue={} \
                     staged={staged_left} gaps={gaps} seeds={seeds}",
                    if book_ok { "valid" } else { "UNSEEDED/STALE" },
                    venue_name(trade_owner),
                );
            }
        }
        keep_running.store(false, Ordering::Relaxed);

        let mut guard = capture.lock().expect("capture mutex poisoned");
        let trailing = guard.flush_pending()?;
        guard.csv.flush().map_err(|e| e.to_string())?;
        let counters = guard.merger.counters();
        eprintln!(
            "[capture] wrote {} rows ({trailing} trailing)",
            guard.csv.rows_written()
        );
        eprintln!(
            "[capture] applied={} diffs={} trades={} gaps={} seeds={} \
             covered_dropped={} foreign_dropped={} staged_dropped={} \
             age_released={} late_dropped={} venue={}",
            counters.events_applied,
            counters.diff_frames,
            counters.trades,
            counters.gaps,
            counters.seeds_applied,
            guard.covered_dropped,
            guard.foreign_dropped,
            guard.staged_dropped,
            // Rows released by the wall-clock age guard rather than by the
            // watermark. Non-zero is not an error, but it IS an ordering
            // compromise, so it belongs in the same line as the other loss
            // counters instead of being invisible.
            guard.reorder.age_released(),
            guard.reorder.late_dropped(),
            venue_name(guard.trade_owner),
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use event_merger_live::schema;

        fn temp(name: &str) -> std::path::PathBuf {
            let mut p = std::env::temp_dir();
            p.push(format!("uets_record_{name}.csv"));
            let _ = std::fs::remove_file(&p);
            p
        }

        fn seed(id: u64) -> BookSnapshot {
            BookSnapshot {
                last_update_id: id,
                bids: vec![(108_663.2, 0.095), (108_662.9, 0.553)],
                asks: vec![(108_666.0, 0.004), (108_668.8, 0.002)],
            }
        }

        fn diff(text: &str) -> MarketEvent {
            live::parse_diff(text).expect("diff frame")
        }

        fn trade(text: &str) -> MarketEvent {
            live::parse_trade(text).expect("trade frame")
        }

        const DIFF_OLD: &str =
            r#"{"e":"depthUpdate","E":1751245200083,"U":1,"u":900,"b":[],"a":[]}"#;
        const DIFF_SPANS: &str = r#"{"e":"depthUpdate","E":1751245200406,"U":901,"u":1100,"b":[["108663.2","9.141"]],"a":[["108665.6","12.556"]]}"#;
        const TRADE_WIRE: &str = r#"{"e":"aggTrade","E":1751245200354,"a":2773899591,"p":"108665.6","q":"0.002","f":6438488841,"l":6438488841,"T":1751245200354,"m":true}"#;

        #[test]
        fn events_stage_until_the_seed_then_flush_without_a_phantom_gap() {
            let path = temp("staged");
            let mut cap = Capture::new(&path, false, 20, 1_000).expect("open");

            // Before the seed nothing is written: an unseeded book would put
            // wrong book_state__* values on disk.
            assert_eq!(cap.offer(diff(DIFF_OLD), VENUE_NONE).unwrap(), 0);
            assert_eq!(cap.offer(trade(TRADE_WIRE), VENUE_FUTURES).unwrap(), 0);
            assert_eq!(cap.offer(diff(DIFF_SPANS), VENUE_NONE).unwrap(), 0);
            assert_eq!(cap.events_seen, 3);

            // Seed at id 1000: DIFF_OLD (u=900) is covered and must vanish;
            // DIFF_SPANS (u=1100 > 1000) joins without flagging a gap.
            // The reorder buffer still holds both survivors behind its
            // lateness window, so nothing reaches the file yet.
            let written = cap.apply_seed(&seed(1_000)).expect("seed");
            assert_eq!(written, 0, "rows wait in the reorder buffer");
            assert!(!cap.reseeds_pending, "no phantom gap from covered diffs");
            assert!(cap.merger.state().book_valid());

            // A covered diff arriving after the seed is also dropped.
            assert_eq!(cap.offer(diff(DIFF_OLD), VENUE_NONE).unwrap(), 0);

            let trailing = cap.flush_pending().expect("flush");
            assert_eq!(trailing, 2, "trade + spanning diff become rows");
            cap.csv.flush().expect("csv flush");

            let text = std::fs::read_to_string(&path).expect("read back");
            let rows: Vec<&str> = text.lines().collect();
            assert_eq!(rows.len(), 3, "header + two rows");
            let width = rows[0].split(',').count();
            assert_eq!(width, schema().len(), "header matches schema");
            for row in &rows {
                assert_eq!(row.split(',').count(), width, "no column shift");
            }
            assert!(
                rows[1].contains("TRADE") || rows[2].contains("TRADE"),
                "a TRADE row"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn trade_venue_ownership_is_exclusive_and_self_healing() {
            let path = temp("venue");
            let mut cap = Capture::new(&path, false, 20, 1_000).expect("open");
            cap.apply_seed(&seed(1_000)).expect("seed");

            // First claim wins; the other venue is refused while the owner is alive.
            assert_eq!(cap.offer(trade(TRADE_WIRE), VENUE_FUTURES).unwrap(), 0);
            assert_eq!(cap.trade_owner, VENUE_FUTURES);
            assert_eq!(cap.offer(trade(TRADE_WIRE), VENUE_SPOT).unwrap(), 0);
            assert_eq!(cap.trade_owner, VENUE_FUTURES, "no takeover while fresh");
            assert_eq!(cap.foreign_dropped, 1);

            // Owner silent past VENUE_TAKEOVER -> ownership moves.
            cap.last_owner_trade_ms =
                now_ms().saturating_sub(VENUE_TAKEOVER.as_millis() as u64 + 1_000);
            assert_eq!(cap.offer(trade(TRADE_WIRE), VENUE_SPOT).unwrap(), 0);
            assert_eq!(cap.trade_owner, VENUE_SPOT, "takeover after silence");
            assert_eq!(cap.offer(trade(TRADE_WIRE), VENUE_FUTURES).unwrap(), 0);
            assert_eq!(cap.foreign_dropped, 2, "old owner now refused");

            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn a_fresh_seed_supersedes_older_diffs() {
            let path = temp("reseed");
            let mut cap = Capture::new(&path, false, 20, 1_000).expect("open");
            cap.apply_seed(&seed(1_000)).expect("first seed");

            // Gap -> re-seed at a much higher id; diffs below it are stale.
            cap.reseeds_pending = true;
            cap.apply_seed(&seed(5_000)).expect("second seed");
            assert!(!cap.reseeds_pending, "seed clears the re-seed flag");
            assert_eq!(cap.offer(diff(DIFF_SPANS), VENUE_NONE).unwrap(), 0);
            assert_eq!(cap.flush_pending().unwrap(), 0, "u=1100 <= 5000 dropped");

            let _ = std::fs::remove_file(&path);
        }
    }
}

#[cfg(test)]
mod strategy_tests {
    //! Test 2: the executor knows strategy1 and strategy2 are present and can
    //! switch their state builders / feature builders on demand.

    use super::*;
    use uets::core::feature::FeatureRow;
    use uets::core::strategy::{Decision, MarketState, Position};

    fn row(price: f64) -> FeatureRow {
        let mut row = FeatureRow::new();
        row.insert("price", price);
        row
    }

    fn state(feature: &str, value: f64) -> MarketState {
        let mut features = FeatureRow::new();
        features.insert(feature, value);
        MarketState {
            features,
            position: Position::Flat,
            sequence: 1,
        }
    }

    #[test]
    fn the_executor_knows_both_strategies_on_disk_and_in_the_binary() {
        let switch = StrategySwitch::new();
        let available = switch.available();
        assert!(available.contains(&"strategy1"));
        assert!(available.contains(&"strategy2"));
        let discovered = switch.discovered();
        assert!(discovered.contains(&"strategy1".to_string()));
        assert!(discovered.contains(&"strategy2".to_string()));
        for dir in &discovered {
            assert!(
                available.contains(&dir.as_str()),
                "folder {dir} has no compiled module"
            );
        }
    }

    #[test]
    fn selecting_strategy1_returns_strategy1s_builders() {
        let mut switch = StrategySwitch::new();
        assert!(
            switch.state_builder().is_err(),
            "nothing selected yet is an error, not a default"
        );

        switch.select("strategy1").unwrap();
        assert_eq!(switch.active(), Some("strategy1"));

        // Its feature builder computes ma3...
        let outputs = {
            let features = switch.feature_builder_mut().unwrap();
            let computed = features.compute(&row(100.0)).unwrap();
            assert_eq!(computed.get("ma3"), Some(100.0));
            features.output_names()
        };
        assert_eq!(outputs, vec!["ma3".to_string()]);

        // ...and its state builder keys on the ma3 band.
        let builder = switch.state_builder().unwrap();
        assert_eq!(builder.decide(&state("ma3", 101.0)), Decision::EnterLong);
        // strategy2's rule would hold on this state: it reads `dz`.
        let other = live_strategy::load_strategy("strategy2").unwrap();
        assert_eq!(other.builder.decide(&state("ma3", 101.0)), Decision::Hold);
    }

    #[test]
    fn selecting_strategy2_returns_strategy2s_builders() {
        let mut switch = StrategySwitch::new();
        switch.select("strategy2").unwrap();
        assert_eq!(switch.active(), Some("strategy2"));

        let outputs = {
            let features = switch.feature_builder_mut().unwrap();
            let computed = features.compute(&row(100.0)).unwrap();
            assert_eq!(computed.get("dz"), Some(0.0), "first row: price - mean");
            features.output_names()
        };
        assert_eq!(outputs, vec!["dz".to_string()]);

        let builder = switch.state_builder().unwrap();
        assert_eq!(builder.decide(&state("dz", 2.0)), Decision::EnterLong);
        // strategy1's rule would hold: it reads `ma3`.
        let other = live_strategy::load_strategy("strategy1").unwrap();
        assert_eq!(other.builder.decide(&state("dz", 2.0)), Decision::Hold);
    }

    #[test]
    fn switching_back_and_forth_returns_the_right_builders_each_time() {
        let mut switch = StrategySwitch::new();
        for (name, feature) in [
            ("strategy1", "ma3"),
            ("strategy2", "dz"),
            ("strategy1", "ma3"),
        ] {
            switch.select(name).unwrap();
            assert_eq!(switch.active(), Some(name));
            let outputs = switch.feature_builder_mut().unwrap().output_names();
            assert_eq!(outputs, vec![feature.to_string()], "after selecting {name}");
        }
    }

    #[test]
    fn an_unknown_strategy_is_rejected_and_keeps_the_current_selection() {
        let mut switch = StrategySwitch::new();
        switch.select("strategy1").unwrap();
        let error = switch.select("no_such_strategy").unwrap_err();
        assert!(error.contains("no_such_strategy"), "got: {error}");
        assert_eq!(
            switch.active(),
            Some("strategy1"),
            "a failed switch changes nothing"
        );
        let outputs = switch.feature_builder_mut().unwrap().output_names();
        assert_eq!(
            outputs,
            vec!["ma3".to_string()],
            "still strategy1's builder"
        );
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let rest: Vec<String> = args.iter().skip(1).cloned().collect();
    let first = || rest.first().cloned().unwrap_or_default();

    match cmd {
        "market" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            let datatype = rest.get(1).cloned().unwrap_or_else(|| "aggTrade".into());
            reader::market_websocket(&symbol, &datatype);
        }
        "depth" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            reader::market_websocket(&symbol, "depth@100ms");
        }
        "kline" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            reader::kline_websocket(&symbol);
        }
        "ticker" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            reader::ticker_websocket(&symbol);
        }
        "all-tickers" => reader::all_trades_websocket(),
        "multi" => {
            let symbols: Vec<&str> = if rest.is_empty() {
                vec!["ethbtc", "bnbusdt"]
            } else {
                rest.iter().map(String::as_str).collect()
            };
            reader::multiple_streams(&symbols);
        }
        "record" => {
            // Live capture -> the merged 128-column CSV, byte-compatible with
            // the historical table. Streams until Ctrl+C unless a row cap is
            // given as `--rows N` in the output-dir position's sibling flag.
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            // A flag is not a positional. Without this, `record btcusdt
            // --rows 10` would take "--rows" as the output directory and write
            // the capture into a directory literally named `--rows`.
            let dir = rest
                .iter()
                .skip(1)
                .find(|a| !a.starts_with("--") && !a.chars().all(|c| c.is_ascii_digit()))
                .cloned()
                .unwrap_or_else(default_live_db_dir);
            let mut config = live::CaptureConfig::new(&symbol, dir);
            // `--rows N` bounds the capture, which is what makes it testable
            // and scriptable instead of only interactive.
            if let Some(index) = rest.iter().position(|a| a == "--rows") {
                if let Some(value) = rest.get(index + 1) {
                    match value.parse::<u64>() {
                        Ok(n) => config.max_rows = Some(n),
                        Err(_) => {
                            eprintln!("--rows expects a number, got: {value}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
            if rest.iter().any(|a| a == "--append") {
                config.append = true;
            }
            if let Err(e) = record::run(config) {
                eprintln!("capture failed: {e}");
                return ExitCode::FAILURE;
            }
        }
        "monitor" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            monitor::monitor_websocket(&symbol);
        }
        "diag" => {
            // One row per event: timestamp, update_type (aggTrades|orderbook),
            // that stream's data — the other stream's columns stay NaN.
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            let dir = rest
                .iter()
                .skip(1)
                .find(|a| !a.starts_with("--"))
                .cloned()
                .unwrap_or_else(default_out_dir);
            if let Err(e) = diagnostic::run(&symbol, dir) {
                eprintln!("diag failed: {e}");
                return ExitCode::FAILURE;
            }
        }
        "strategies" => {
            // The executor's own view of what it can run: compiled-in
            // bundles vs folders on disk, and (with a name) the switch.
            let mut switch = StrategySwitch::new();
            println!("compiled:   {:?}", switch.available());
            println!("discovered: {:?}", switch.discovered());
            let name = first();
            if name.is_empty() {
                println!("selected:   (none); try: dev_test strategies strategy1");
            } else {
                if let Err(e) = switch.select(&name) {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
                println!("selected:   {}", switch.active().unwrap_or("?"));
                println!("state_builder present:  {}", switch.state_builder().is_ok());
                println!(
                    "feature_builder present: {}",
                    switch.feature_builder_mut().is_ok()
                );
            }
        }
        "user-stream" => {
            println!("note: requires a real API key (edit reader::user_stream)");
            reader::user_stream();
        }
        "user-stream-ws" => {
            println!("note: requires a real API key (edit reader::user_stream_websocket)");
            reader::user_stream_websocket();
        }
        "help" | "--help" | "-h" => print_help(),
        other => {
            eprintln!("unknown command: {other}");
            print_help();
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
