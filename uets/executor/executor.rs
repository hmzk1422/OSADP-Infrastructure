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
use uets_db::{QueryFilter, UetsDb};

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
         \x20 record SYM1 [SYM2 ...] [DIR] live capture -> one merged CSV per symbol under DIR\n\
         \x20 feed SYMBOL INSTANCE...  capture live data and fan it out over Unix sockets\n\
         \x20 run INSTANCE...         start the VIM monitor and isolated strategy/feed processes\n\
         \x20                         prefix an instance with disabled: to omit its worker\n\
         \x20                         (default DIR: market_data/live_db)\n\
         \x20                         flags: --dir DIR  explicit output dir (removes ambiguity)\n\
         \x20                                --rows N   stop after N rows *per symbol*\n\
         \x20                                --append  keep the existing file\n\
         \x20 monitor [SYMBOL]        live tape + 20-level orderbook monitor;\n\
         \x20                         view with:  python uets/display.py [SYMBOL]\n\
         \x20 diag [SYMBOL] [DIR]      diagnostic CSV: one row per event with\n\
         \x20                         update_type=aggTrades|orderbook and the\n\
         \x20                         other stream's columns as NaN\n\
         \x20                         (default DIR: uets/out/live)\n\
         \x20 strategies               list compiled vs on-disk strategy bundles\n\
         \x20 strategy <strategy>@<asset> run one strategy socket consumer\n\
         \x20 db [SESSION [STRAT@ASSET]]  inspect live_db contents (trades + portfolio)\n\
         \x20 user-stream             REST listen-key lifecycle (needs API key)\n\
         \x20 user-stream-ws          account/order update stream (needs API key)\n\
         \x20 help                    show this message\n\
         \n\
         examples:\n\
         \x20 dev_test market btcusdt aggTrade\n\
         \x20 dev_test depth btcusdt\n\
         \x20 dev_test multi ethbtc bnbusdt\n\
         \x20 dev_test strategy ofmr@btcusdt\n\
         \x20 dev_test feed btcusdt ofmr@btcusdt\n\
         \x20 dev_test run ofmr@btcusdt ofmr@ethusdt\n\
         \x20 dev_test record btcusdt            # -> market_data/live_db\n\
         \n\
         All streams run until interrupted (Ctrl+C)."
    );
}

fn parse_strategy_instance(value: &str) -> Result<(String, String), String> {
    let Some((strategy, asset)) = value.split_once('@') else {
        return Err(format!(
            "invalid strategy instance '{value}'; expected strategy@asset"
        ));
    };
    if strategy.is_empty()
        || asset.is_empty()
        || asset.contains('@')
        || !strategy
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        || !asset
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "invalid strategy instance '{value}'; expected strategy@asset"
        ));
    }
    Ok((strategy.to_ascii_lowercase(), asset.to_ascii_lowercase()))
}

#[cfg(unix)]
struct LiveProcesses(Vec<std::process::Child>);

#[cfg(unix)]
impl Drop for LiveProcesses {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
        }
        for child in &mut self.0 {
            let _ = child.wait();
        }
    }
}

#[cfg(unix)]
fn run_live_system(requested: &[String]) -> Result<(), String> {
    use uets_gui::{Monitor, SocketSnapshotSource};

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "cannot resolve repository root".to_string())?
        .to_path_buf();
    std::env::set_current_dir(&root).map_err(|error| error.to_string())?;
    let gui_socket = root.join("sockets").join("gui.sock");
    let snapshots = SocketSnapshotSource::bind(&gui_socket)
        .map_err(|error| format!("bind UETS GUI socket: {error}"))?;
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let log_dir = root.join("logs").join("uets");
    std::fs::create_dir_all(&log_dir).map_err(|error| error.to_string())?;
    let mut dashboard = LiveDashboard {
        snapshots,
        gui_socket,
        root,
        executable,
        log_dir,
        instances: std::collections::BTreeMap::new(),
    };

    for argument in requested {
        let disabled = argument.starts_with("disabled:");
        let identity = argument.strip_prefix("disabled:").unwrap_or(argument);
        let (strategy, asset) = parse_strategy_instance(identity)?;
        let canonical_identity = format!("{strategy}@{asset}");
        dashboard.add(canonical_identity.clone())?;
        if !disabled {
            dashboard.enable(&canonical_identity)?;
        }
    }

    eprintln!("UETS monitor ready: j/k select, a add, e enable/disable, x delete, q quit");
    Monitor::new(dashboard)
        .run()
        .map_err(|error| format!("UETS monitor failed: {error}"))
}

#[cfg(unix)]
struct ManagedInstance {
    strategy: String,
    asset: String,
    worker: Option<std::process::Child>,
    feed: Option<std::process::Child>,
    enabled: bool,
}

#[cfg(unix)]
struct LiveDashboard {
    snapshots: uets_gui::SocketSnapshotSource,
    gui_socket: PathBuf,
    root: PathBuf,
    executable: PathBuf,
    log_dir: PathBuf,
    instances: std::collections::BTreeMap<String, ManagedInstance>,
}

#[cfg(unix)]
impl LiveDashboard {
    fn process_log(log_dir: &std::path::Path, file: &str) -> Result<std::fs::File, String> {
        std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log_dir.join(file))
            .map_err(|error| format!("open logs/uets/{file}: {error}"))
    }

    fn add(&mut self, value: String) -> Result<(), String> {
        use uets_gui::{InstanceSnapshot, Statistics, publish_instance};
        let (strategy, asset) = parse_strategy_instance(&value)?;
        if strategy != "ofmr" {
            return Err(format!("'{strategy}' has no live UDS consumer yet"));
        }
        let identity = format!("{strategy}@{asset}");
        if self.instances.contains_key(&identity) {
            return Err(format!("{identity} already exists"));
        }
        if self
            .instances
            .values()
            .any(|instance| instance.asset == asset)
        {
            return Err(format!(
                "an instance for {asset} already exists; this feed adapter allows one strategy per asset"
            ));
        }
        publish_instance(
            &self.gui_socket,
            &InstanceSnapshot {
                strategy_id: strategy.clone(),
                asset_id: asset.clone(),
                statistics: Statistics {
                    status: Some("disabled".to_string()),
                    ..Statistics::default()
                },
                equity_history: Vec::new(),
            },
        )
        .map_err(|error| format!("publish disabled status for {identity}: {error}"))?;
        self.instances.insert(
            identity,
            ManagedInstance {
                strategy,
                asset,
                worker: None,
                feed: None,
                enabled: false,
            },
        );
        Ok(())
    }

    fn enable(&mut self, identity: &str) -> Result<(), String> {
        use std::time::{Duration, Instant};
        use uets_gui::SnapshotSource;
        use uets_gui::publish_instance;
        let Some(instance) = self.instances.get_mut(identity) else {
            return Err(format!("unknown strategy {identity}"));
        };
        if instance.enabled {
            return Err(format!("{identity} is already enabled"));
        }
        if instance
            .worker
            .as_mut()
            .is_some_and(|child| child.try_wait().ok().flatten().is_some())
        {
            instance.worker = None;
        }
        let worker_was_running = instance.worker.is_some();
        if instance.worker.is_none() {
            let file = format!("strategy_{}_{}.log", instance.strategy, instance.asset);
            let log = Self::process_log(&self.log_dir, &file)?;
            let mut command = std::process::Command::new(&self.executable);
            command
                .current_dir(&self.root)
                .env("UETS_GUI_SOCKET", &self.gui_socket)
                .arg("strategy")
                .arg(identity)
                .stdout(std::process::Stdio::from(
                    log.try_clone().map_err(|error| error.to_string())?,
                ))
                .stderr(std::process::Stdio::from(log));
            instance.worker = Some(
                command
                    .spawn()
                    .map_err(|error| format!("start {identity}: {error}"))?,
            );
        }
        if !worker_was_running {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let snapshot = self.snapshots.snapshot();
                if snapshot.instances.iter().any(|item| {
                    item.identity() == identity
                        && item.statistics.status.as_deref() == Some("enabled")
                }) {
                    break;
                }
                if instance
                    .worker
                    .as_mut()
                    .and_then(|child| child.try_wait().ok().flatten())
                    .is_some()
                {
                    return Err(format!(
                        "worker exited during startup; inspect logs/uets/strategy_{}_{}.log",
                        instance.strategy, instance.asset
                    ));
                }
                if Instant::now() >= deadline {
                    return Err(format!(
                        "worker telemetry timed out; inspect logs/uets/strategy_{}_{}.log",
                        instance.strategy, instance.asset
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        let file = format!("feed_{}.log", instance.asset);
        let log = Self::process_log(&self.log_dir, &file)?;
        let mut command = std::process::Command::new(&self.executable);
        command
            .current_dir(&self.root)
            .env("UETS_GUI_SOCKET", &self.gui_socket)
            .arg("feed")
            .arg(&instance.asset)
            .arg(identity)
            .stdout(std::process::Stdio::from(
                log.try_clone().map_err(|error| error.to_string())?,
            ))
            .stderr(std::process::Stdio::from(log));
        instance.feed = Some(
            command
                .spawn()
                .map_err(|error| format!("start feed for {identity}: {error}"))?,
        );
        instance.enabled = true;
        let mut status_snapshot = self
            .snapshots
            .snapshot()
            .instances
            .into_iter()
            .find(|snapshot| snapshot.identity() == identity)
            .unwrap_or_default();
        status_snapshot.strategy_id = instance.strategy.clone();
        status_snapshot.asset_id = instance.asset.clone();
        status_snapshot.statistics.status = Some("enabled".into());
        let _ = publish_instance(&self.gui_socket, &status_snapshot);
        Ok(())
    }

    fn disable(&mut self, identity: &str) -> Result<(), String> {
        use uets_gui::SnapshotSource;
        use uets_gui::publish_instance;
        let mut snapshot = self
            .snapshots
            .snapshot()
            .instances
            .into_iter()
            .find(|snapshot| snapshot.identity() == identity)
            .unwrap_or_default();
        let Some(instance) = self.instances.get_mut(identity) else {
            return Err(format!("unknown strategy {identity}"));
        };
        if !instance.enabled {
            return Err(format!("{identity} is already disabled"));
        }
        if let Some(mut feed) = instance.feed.take() {
            let _ = feed.kill();
            let _ = feed.wait();
        }
        instance.enabled = false;
        snapshot.strategy_id = instance.strategy.clone();
        snapshot.asset_id = instance.asset.clone();
        snapshot.statistics.status = Some("disabled".into());
        publish_instance(&self.gui_socket, &snapshot)
            .map_err(|error| format!("publish disabled status: {error}"))
    }

    fn remove(&mut self, identity: &str) -> Result<(), String> {
        let Some(instance) = self.instances.get(identity) else {
            return Err(format!("unknown strategy {identity}"));
        };
        if instance.enabled {
            return Err("disable the strategy before deleting it".to_string());
        }
        if let Some(mut instance) = self.instances.remove(identity) {
            if let Some(mut worker) = instance.worker.take() {
                let _ = worker.kill();
                let _ = worker.wait();
            }
        }
        self.snapshots.forget_instance(identity);
        Ok(())
    }
}

#[cfg(unix)]
impl uets_gui::SnapshotSource for LiveDashboard {
    fn snapshot(&mut self) -> uets_gui::DashboardSnapshot {
        let mut snapshot = self.snapshots.snapshot();
        for instance in &mut snapshot.instances {
            let identity = instance.identity();
            if let Some(managed) = self.instances.get_mut(&identity) {
                if let Some(feed) = managed.feed.as_mut() {
                    if let Ok(Some(status)) = feed.try_wait() {
                        if !status.success() {
                            instance.statistics.status = Some("error".into());
                            instance.statistics.last_error = Some(format!(
                                "feed exited ({status}); see logs/uets/feed_{}.log",
                                managed.asset
                            ));
                        }
                    }
                }
                if let Some(worker) = managed.worker.as_mut() {
                    if let Ok(Some(status)) = worker.try_wait() {
                        instance.statistics.status = Some("error".into());
                        instance.statistics.last_error = Some(format!(
                            "strategy worker exited ({status}); see logs/uets/strategy_{}_{}.log",
                            managed.strategy, managed.asset
                        ));
                    }
                }
            }
        }
        snapshot
    }

    fn control(&mut self, command: uets_gui::StrategyCommand) -> Result<(), String> {
        match command {
            uets_gui::StrategyCommand::Add(identity) => self.add(identity),
            uets_gui::StrategyCommand::Toggle(identity) => {
                if self
                    .instances
                    .get(&identity)
                    .is_some_and(|instance| instance.enabled)
                {
                    self.disable(&identity)
                } else {
                    self.enable(&identity)
                }
            }
            uets_gui::StrategyCommand::Remove(identity) => self.remove(&identity),
        }
    }
}

#[cfg(unix)]
impl Drop for LiveDashboard {
    fn drop(&mut self) {
        for instance in self.instances.values_mut() {
            if let Some(mut child) = instance.feed.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            if let Some(mut child) = instance.worker.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[cfg(not(unix))]
fn run_live_system(_requested: &[String]) -> Result<(), String> {
    Err("the live UETS runner requires Unix-domain sockets; run it on Linux or in Docker".into())
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

    use stream_handler::{BookSnapshot, CsvRecorder, MarketEvent, Merger, ReorderBuffer};
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
                reorder: ReorderBuffer::new(lateness_ms, REORDER_CAP),
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
        eprintln!("[capture] columns  {}", stream_handler::schema().len());

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
             covered_dropped={} foreign_dropped={} staged_dropped={} venue={}",
            counters.events_applied,
            counters.diff_frames,
            counters.trades,
            counters.gaps,
            counters.seeds_applied,
            guard.covered_dropped,
            guard.foreign_dropped,
            guard.staged_dropped,
            venue_name(guard.trade_owner),
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use stream_handler::schema;

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
        assert!(available.contains(&"ofmr"));
        assert!(available.contains(&"strategy2"));
        let discovered = switch.discovered();
        assert!(discovered.contains(&"ofmr".to_string()));
        assert!(discovered.contains(&"strategy2".to_string()));
        for dir in &discovered {
            assert!(
                available.contains(&dir.as_str()),
                "folder {dir} has no compiled module"
            );
        }
    }

    #[test]
    fn selecting_ofmr_returns_ofmrs_builders() {
        let mut switch = StrategySwitch::new();
        assert!(
            switch.state_builder().is_err(),
            "nothing selected yet is an error, not a default"
        );

        switch.select("ofmr").unwrap();
        assert_eq!(switch.active(), Some("ofmr"));

        // Its feature builder computes Column_0..Column_14...
        let outputs = {
            let features = switch.feature_builder_mut().unwrap();
            let computed = features.compute(&row(100.0)).unwrap();
            for i in 0..15 {
                assert!(
                    computed.contains(&format!("Column_{i}")),
                    "missing Column_{i}"
                );
            }
            features.output_names()
        };
        assert_eq!(outputs.len(), 15);
        assert_eq!(outputs[0], "Column_0");
        assert_eq!(outputs[14], "Column_14");

        // ...and its state builder holds when features are missing.
        let builder = switch.state_builder().unwrap();
        let empty = FeatureRow::new();
        let st = MarketState {
            features: empty,
            position: Position::Flat,
            sequence: 1,
        };
        assert_eq!(builder.decide(&st), Decision::Hold);
        // strategy2's rule would hold on this state: it reads `dz`.
        let other = live_strategy::load_strategy("strategy2").unwrap();
        assert_eq!(other.builder.decide(&state("dz", 2.0)), Decision::EnterLong);
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
        // ofmr's rule would hold: it needs all 15 columns and reads nothing but them.
        let other = live_strategy::load_strategy("ofmr").unwrap();
        assert_eq!(other.builder.decide(&state("dz", 2.0)), Decision::Hold);
    }

    #[test]
    fn switching_back_and_forth_returns_the_right_builders_each_time() {
        let mut switch = StrategySwitch::new();
        for (name, feature) in [
            ("ofmr", "Column_0"),
            ("strategy2", "dz"),
            ("ofmr", "Column_0"),
        ] {
            switch.select(name).unwrap();
            assert_eq!(switch.active(), Some(name));
            let outputs = switch.feature_builder_mut().unwrap().output_names();
            assert_eq!(
                outputs.first(),
                Some(&feature.to_string()),
                "after selecting {name}"
            );
        }
    }

    #[test]
    fn an_unknown_strategy_is_rejected_and_keeps_the_current_selection() {
        let mut switch = StrategySwitch::new();
        switch.select("ofmr").unwrap();
        let error = switch.select("no_such_strategy").unwrap_err();
        assert!(error.contains("no_such_strategy"), "got: {error}");
        assert_eq!(
            switch.active(),
            Some("ofmr"),
            "a failed switch changes nothing"
        );
        let outputs = switch.feature_builder_mut().unwrap().output_names();
        assert_eq!(
            outputs.first(),
            Some(&"Column_0".to_string()),
            "still ofmr's builder"
        );
    }
}

/// `record` argument parsing: `record SYM1 [SYM2 ...] [DIR] [--dir DIR] [--rows N] [--append]`.
///
/// Every positional that is not path-like is a symbol; a trailing path-like
/// positional (contains `/`, `\` or `.`, or is an existing directory) is the
/// output dir, which keeps `record btcusdt <dir>` working. `--dir` always
/// wins and removes all ambiguity when a bare directory name would otherwise
/// read as a symbol:
///
/// ```text
/// record btcusdt                     # one symbol, default dir
/// record btcusdt ethusdt bnbusdt     # three symbols, default dir
/// record btcusdt ethusdt ./captures  # two symbols, explicit dir
/// record btcusdt out --dir out       # bare dirname needs --dir
/// ```
fn parse_record_args(rest: &[String]) -> Result<(Vec<String>, String, Option<u64>, bool), String> {
    let mut dir_flag: Option<String> = None;
    let mut max_rows: Option<u64> = None;
    let mut append = false;
    let mut positionals: Vec<String> = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--rows" => {
                let value = rest.get(i + 1).ok_or("--rows expects a number")?;
                max_rows = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| format!("--rows expects a number, got: {value}"))?,
                );
                i += 2;
            }
            "--dir" => {
                dir_flag = Some(
                    rest.get(i + 1)
                        .cloned()
                        .ok_or("--dir expects a directory")?,
                );
                i += 2;
            }
            "--append" => {
                append = true;
                i += 1;
            }
            other if other.starts_with("--") => {
                i += 1; // unknown flag: skip rather than misread as a symbol
            }
            other => {
                positionals.push(other.to_string());
                i += 1;
            }
        }
    }
    let mut symbols = positionals;
    let dir = match dir_flag {
        Some(d) => d,
        None => {
            // Backward compatible: `record btcusdt <dir>` still resolves the
            // second positional as the dir -- but only when it is path-like,
            // so `record btcusdt ethusdt` reads as two symbols.
            if symbols.len() >= 2 && looks_like_dir(symbols.last().expect("len checked")) {
                symbols.pop().expect("len checked")
            } else {
                default_live_db_dir()
            }
        }
    };
    if symbols.is_empty() {
        symbols.push("btcusdt".to_string());
    }
    Ok((symbols, dir, max_rows, append))
}

/// Symbols are bare alphanumerics (`btcusdt`); anything with a separator, a
/// dot, or an existing directory on disk reads as an output path.
fn looks_like_dir(s: &str) -> bool {
    s.contains('/') || s.contains('\\') || s.contains('.') || std::path::Path::new(s).is_dir()
}

/// Run `record`: one capture thread per symbol, one CSV per symbol.
///
/// Each thread owns its websocket pair, seed thread, merger, and CSV writer,
/// so symbols never block each other -- nineteen quiet books and one frantic
/// book still write at their own cadences. `--rows N` applies per symbol.
/// Ctrl+C stops the process; per-row flushing (fix A) means at most one row
/// per symbol is lost on a hard kill.
fn run_record(rest: &[String]) -> ExitCode {
    let (symbols, dir, max_rows, append) = match parse_record_args(rest) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("record: {e}");
            return ExitCode::FAILURE;
        }
    };
    if symbols.len() == 1 {
        let mut config = live::CaptureConfig::new(&symbols[0], dir);
        config.max_rows = max_rows;
        config.append = append;
        return match record::run(config) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("capture failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    eprintln!("[record] {} symbols -> {}", symbols.len(), dir);
    let handles: Vec<std::thread::JoinHandle<(String, Result<(), String>)>> = symbols
        .into_iter()
        .map(|symbol| {
            let dir = dir.clone();
            std::thread::spawn(move || {
                let mut config = live::CaptureConfig::new(&symbol, &dir);
                config.max_rows = max_rows;
                config.append = append;
                let outcome = record::run(config);
                (symbol, outcome)
            })
        })
        .collect();
    let mut failed = false;
    for handle in handles {
        match handle.join() {
            Ok((symbol, Ok(()))) => eprintln!("[record] {symbol}: done"),
            Ok((symbol, Err(e))) => {
                eprintln!("[record] {symbol}: failed: {e}");
                failed = true;
            }
            Err(_) => {
                eprintln!("[record] worker panicked");
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod record_args_tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn empty_defaults_to_btcusdt() {
        let (syms, _dir, rows, append) = parse_record_args(&[]).expect("parses");
        assert_eq!(syms, vec!["btcusdt"]);
        assert!(rows.is_none() && !append);
    }

    #[test]
    fn two_bare_words_are_two_symbols() {
        let (syms, _dir, _, _) = parse_record_args(&args(&["btcusdt", "ethusdt"])).expect("parses");
        assert_eq!(syms, vec!["btcusdt", "ethusdt"]);
    }

    #[test]
    fn trailing_path_is_dir_not_symbol() {
        let (syms, dir, _, _) =
            parse_record_args(&args(&["btcusdt", "ethusdt", "./captures"])).expect("parses");
        assert_eq!(syms, vec!["btcusdt", "ethusdt"]);
        assert_eq!(dir, "./captures");
    }

    #[test]
    fn dir_flag_wins_over_heuristic() {
        let (syms, dir, _, _) =
            parse_record_args(&args(&["btcusdt", "out", "--dir", "out"])).expect("parses");
        assert_eq!(syms, vec!["btcusdt", "out"]);
        assert_eq!(dir, "out");
    }

    #[test]
    fn single_symbol_with_legacy_dir() {
        let (syms, dir, _, _) =
            parse_record_args(&args(&["btcusdt", "market_data/live_db"])).expect("parses");
        assert_eq!(syms, vec!["btcusdt"]);
        assert_eq!(dir, "market_data/live_db");
    }

    #[test]
    fn rows_and_append_parse() {
        let (_, _, rows, append) =
            parse_record_args(&args(&["btcusdt", "--rows", "50", "--append"])).expect("parses");
        assert_eq!(rows, Some(50));
        assert!(append);
    }
}

/// `db` inspector: verify live_db contents.
///
/// Physical layout: `live_db/trades/<session>/` and
/// `live_db/portfolio/<session>/` are independent RocksDB instances.
///
/// Usage:
///   dev_test db                      — list all sessions and record counts
///   dev_test db <SESSION>            — list trades + portfolio for a session
///   dev_test db <SESSION> <STRAT@ASSET>  — list trades + portfolio for one instance
fn cmd_db(args: &[String]) -> Result<(), String> {
    let root: PathBuf = PathBuf::from(default_live_db_dir());

    let session = args.first().map(String::as_str).filter(|s| !s.is_empty());
    let instance = args.get(1).map(String::as_str).filter(|s| !s.is_empty());

    if let Some((strategy, asset)) = instance.and_then(|i| i.split_once('@')) {
        // Specific session + instance.
        let Some(sid) = session else {
            return Err("usage: dev_test db <SESSION> <STRATEGY@ASSET>".into());
        };
        let filter = QueryFilter::new()
            .session(sid)
            .strategy(strategy)
            .asset(asset);
        let trades_db =
            UetsDb::open(&root.join("trades").join(sid)).map_err(|e| format!("open trades: {e}"))?;
        let portfolio_db = UetsDb::open(&root.join("portfolio").join(sid))
            .map_err(|e| format!("open portfolio: {e}"))?;
        print_instance(&trades_db, &portfolio_db, sid, &filter)?;
    } else if let Some(sid) = session {
        // Whole session: both strategy instances.
        let filter = QueryFilter::new().session(sid);
        let trades_db =
            UetsDb::open(&root.join("trades").join(sid)).map_err(|e| format!("open trades: {e}"))?;
        let portfolio_db = UetsDb::open(&root.join("portfolio").join(sid))
            .map_err(|e| format!("open portfolio: {e}"))?;
        print_session(&trades_db, &portfolio_db, sid, &filter)?;
    } else {
        // No args: list all sessions from both stage directories.
        print_all_sessions(&root)?;
    }

    Ok(())
}

fn list_session_dirs(root: &std::path::Path, stage: &str) -> Vec<String> {
    let stage_dir = root.join(stage);
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&stage_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

fn print_all_sessions(root: &std::path::Path) -> Result<(), String> {
    let trades_sessions = list_session_dirs(root, "trades");
    let portfolio_sessions = list_session_dirs(root, "portfolio");

    let mut all: std::collections::BTreeSet<String> =
        trades_sessions.into_iter().collect();
    all.extend(portfolio_sessions);

    if all.is_empty() {
        println!("no sessions found");
        return Ok(());
    }
    println!("sessions:");
    for sid in &all {
        let trades_db =
            UetsDb::open(&root.join("trades").join(sid)).map_err(|e| format!("open trades: {e}"))?;
        let portfolio_db = UetsDb::open(&root.join("portfolio").join(sid))
            .map_err(|e| format!("open portfolio: {e}"))?;
        let trades = trades_db
            .query_trades(&QueryFilter::new().session(sid))
            .map_err(|e| e.to_string())?
            .len();
        let portfolios = portfolio_db
            .query_portfolio(&QueryFilter::new().session(sid))
            .map_err(|e| e.to_string())?
            .len();
        println!("  {}  trades={} portfolio={}", sid, trades, portfolios);
    }
    Ok(())
}

fn print_session(
    trades_db: &UetsDb,
    portfolio_db: &UetsDb,
    sid: &str,
    filter: &QueryFilter,
) -> Result<(), String> {
    let trades = trades_db.query_trades(filter).map_err(|e| e.to_string())?;
    let portfolios = portfolio_db.query_portfolio(filter).map_err(|e| e.to_string())?;

    println!("session: {sid}");
    println!("  trades: {}   portfolio: {}", trades.len(), portfolios.len());

    if !trades.is_empty() {
        println!("  trades:");
        for t in &trades {
            println!(
                "    t={}  sym={}  strat={}  asset={}  decision={:?}  price={:?}",
                t.event_time.timestamp_millis(),
                t.symbol,
                t.instance.strategy_id,
                t.instance.asset_id,
                t.decision,
                t.entry_price
            );
        }
    }

    if !portfolios.is_empty() {
        println!("  portfolio:");
        for p in &portfolios {
            println!(
                "    t={}  cash={:.2}  equity={:.2}  realized={:.2}  unrealized={:.2}  dd={:.4}  ret={:.4}",
                p.timestamp.timestamp_millis(),
                p.cash,
                p.equity,
                p.realized_pnl,
                p.unrealized_pnl,
                p.max_drawdown,
                p.cumulative_return
            );
        }
    }

    Ok(())
}

fn print_instance(
    trades_db: &UetsDb,
    portfolio_db: &UetsDb,
    sid: &str,
    filter: &QueryFilter,
) -> Result<(), String> {
    print_session(trades_db, portfolio_db, sid, filter)
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
            return run_record(&rest);
        }
        "run" => {
            return match run_live_system(&rest) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("UETS runtime failed: {error}");
                    ExitCode::FAILURE
                }
            };
        }
        "feed" => {
            if rest.len() < 2 {
                eprintln!("usage: dev_test feed <symbol> <strategy>@<asset> [...]");
                return ExitCode::FAILURE;
            }
            let mut config = live::CaptureConfig::new(&rest[0], default_live_db_dir());
            config.fanout_instances = rest.iter().skip(1).cloned().collect();
            return match live::run_capture(config) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("live feed failed: {error}");
                    ExitCode::FAILURE
                }
            };
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
                println!("selected:   (none); try: dev_test strategies ofmr");
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
        "strategy" => {
            // Run a strategy's Unix-domain socket consumer. Aggregator
            // (aggregator.py) binds the exchange side at
            // sockets/exchanges/<venue>.sock and fans out to
            // sockets/strategies/<name>_<symbol>.sock so multiple instances
            // can run side by side (ofmr->btc, ofmr->eth, ofmr->sol,
            // ... 100 assets).
            let identity = first();
            if identity.is_empty() {
                eprintln!("usage: dev_test strategy <strategy>@<asset> (or <strategy> <asset>)");
                print_help();
                return ExitCode::FAILURE;
            }
            let (name, symbol) = if identity.contains('@') {
                match parse_strategy_instance(&identity) {
                    Ok(instance) => instance,
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                let symbol = rest.get(1).cloned().unwrap_or_else(|| "btcusdt".into());
                match parse_strategy_instance(&format!("{identity}@{symbol}")) {
                    Ok(instance) => instance,
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::FAILURE;
                    }
                }
            };
            if name != "ofmr" {
                eprintln!("strategy socket consumer is currently implemented only for 'ofmr'");
                return ExitCode::FAILURE;
            }
            let socket_path = format!("sockets/strategies/{name}_{symbol}.sock");
            println!("instance:   {name}@{symbol}");
            println!("socket:     {socket_path}");
            if let Err(e) = uets::live_strategy::ofmr::socket::run(&socket_path, &symbol) {
                eprintln!("strategy socket failed: {e}");
                return ExitCode::FAILURE;
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
        "db" => {
            return match cmd_db(&rest) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("db failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
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

#[cfg(test)]
mod strategy_instance_tests {
    use super::parse_strategy_instance;

    #[test]
    fn parses_canonical_identity_case_insensitively() {
        assert_eq!(
            parse_strategy_instance("OFMR@BTCUSDT").unwrap(),
            ("ofmr".to_string(), "btcusdt".to_string())
        );
    }

    #[test]
    fn rejects_incomplete_or_unsafe_identity() {
        assert!(parse_strategy_instance("ofmr").is_err());
        assert!(parse_strategy_instance("@btcusdt").is_err());
        assert!(parse_strategy_instance("ofmr@../../tmp").is_err());
    }
}
