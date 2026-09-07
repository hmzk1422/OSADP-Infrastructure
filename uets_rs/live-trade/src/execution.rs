//! Venue-agnostic strategy execution loop.
//!
//! Connects an [`Exchange`] implementation, feeds normalized market events into
//! the `engine` strategy, persists state to RocksDB and -- in `Live` mode --
//! routes the engine's entry/exit events to the exchange as real orders.
//!
//! The loop is fully async and owns a single `MomentumStrategy` instance; it
//! is designed to be driven from:
//!   - the `live-trade` CLI binary, or
//!   - the PyO3 `Strategy` binding (one loop per ticker on its own tokio
//!     runtime thread), so Python can run 50+ assets concurrently.

use crate::db::{ClosedTrade, DbWriter, PnLSnapshot, SessionMeta};
use crate::exchange::{
    Broker, ExchangeKind, Feed, MarketEvent, OrderRequest, OrderType, TradeSide, symbols,
};
use crate::model;
use engine::{CONFIDENCE_THRESHOLD, EngineEvent, MomentumStrategy, StrategyConfig};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// DEVLOG staleness standard: once the newest market event ages past this,
/// the feed counts as stale (network outage or illiquid market). Consecutive
/// ticks past the threshold are ONE outage episode (`RunStats::n_outages`).
const NET_OUTAGE_STALENESS_MS: f64 = 1000.0;
/// Cadence of the periodic `[net]` console line (parsed by the portfolio
/// supervisor into the dashboard; also handy in a bare terminal).
const NET_PRINT_SECS: u64 = 5;
/// Cadence of the per-second `[tick]` live-market telemetry line (see the
/// print block in the main loop). This is the MICRO channel: the supervisor /
/// VIM dashboard refreshes price, gate state and the open position's
/// mark-to-market PnL from it once per second, independent of the ~5-minute
/// `[bar]` macro cadence.
const TICK_PRINT_SECS: u64 = 1;

/// Rolling minimum of `raw_skew` over a trailing window. Since `raw_skew =
/// now - exchange_ts` mixes clock offset and network/exchange transit delay
/// into one number that can't be split without an independent time source,
/// this tracks the best (lowest) value seen recently as a live baseline.
/// `raw_skew - floor` then isolates *relative* staleness — how much worse
/// than normal the newest event is — without claiming to know the absolute
/// clock offset or transit time.
struct SkewFloor {
    window: VecDeque<(std::time::Instant, f64)>,
    max_age: std::time::Duration,
}

impl SkewFloor {
    fn new(max_age_secs: u64) -> Self {
        Self {
            window: VecDeque::new(),
            max_age: std::time::Duration::from_secs(max_age_secs),
        }
    }

    /// Feed a new raw_skew sample (ms), evict stale entries, return the
    /// current floor. Returns `raw_skew` itself if the window is empty
    /// (first sample — nothing to compare against yet).
    fn push(&mut self, raw_skew: f64) -> f64 {
        let now = std::time::Instant::now();
        self.window.push_back((now, raw_skew));
        while self
            .window
            .front()
            .map_or(false, |(t, _)| now.duration_since(*t) > self.max_age)
        {
            self.window.pop_front();
        }
        self.window
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::INFINITY, f64::min)
    }
}

/// Execution mode: simulated fills (paper) vs. real order routing (live).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// The engine simulates fills internally (dry-run); no real orders.
    Paper,
    /// Engine events are routed to the exchange as real orders.
    Live,
}

/// WebSocket connection state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WsState {
    #[default]
    Disconnected,
    Connected,
    Reconnecting,
}

impl WsState {
    /// String representation for terminal display.
    pub fn as_str(self) -> &'static str {
        match self {
            WsState::Connected => "CONNECTED",
            WsState::Disconnected => "DISCONNECTED",
            WsState::Reconnecting => "RECONNECTING",
        }
    }
}

impl std::fmt::Display for WsState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Meta-labeller gate switch for one run: `Auto` follows the manifest's
/// `gate_deployed` flag; `On`/`Off` override it in-memory for THIS run only
/// (the manifest is never rewritten), so a test can A/B the filter without
/// touching `build/meta_labeller_manifest.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetaGate {
    #[default]
    Auto,
    On,
    Off,
}

/// Static configuration for one execution loop.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Ticker in lowercase, e.g. "btcusdt".
    pub ticker: String,
    /// Venue that supplies the market-data feed (trades + top-of-book).
    pub feed: ExchangeKind,
    /// Venue orders are routed to (used in `Live` mode only).
    pub broker: ExchangeKind,
    pub mode: ExecutionMode,
    pub db_path: String,
    /// Number of historical composite bars to warm feature rings with.
    pub seed_bars: u32,
    /// Persist market data + PnL to RocksDB.
    pub persist: bool,
    /// Optional strategy-parameter override (single source of truth: the
    /// `build/strategy_config.json` exported by `unified_runner.py` from
    /// `main/config.py`). When `None` the engine falls back to its built-in
    /// `Default`, which mirrors `main/config.py`.
    pub strategy_config: Option<StrategyConfig>,
    /// Optional walk-forward fold-model dir (`build/walkforward_models`).
    /// When set, each 1 s row is scored with the fold model whose OOS test
    /// window covers it (live-legal equivalent of periodic retrain+deploy).
    pub walkforward_dir: Option<String>,
    /// Portfolio meta-labeller gate switch (see [`MetaGate`]).
    pub meta_gate: MetaGate,
    /// Which persisted runs the HTML report shows (defaults to `latest`,
    /// matching the terminal session counters).
    pub report_scope: crate::report::ReportScope,
    /// Optional external-control file (JSON, e.g. `build/control_<ticker>.json`).
    /// Polled by the loop every ~100 ms; a `{"cmd":"flatten|pause|resume"}`
    /// object is consumed (file deleted) after being acted on. Lets a parent
    /// supervisor (portfolio dashboard / VIM GUI) pause an asset and close its
    /// position without killing the process.
    pub control_path: Option<String>,
    /// When true, do NOT restore equity from the last persisted RocksDB snapshot
    /// (used by the supervisor when the operator edits an asset's capital in
    /// portfolio.json — so the restart reports the NEW capital, not the old equity).
    pub fresh_equity: bool,
    /// Optional per-asset report tag (the portfolio supervisor passes each
    /// asset's unique `id`). When set, the HTML report is written to
    /// `build/pnl_report_{tag}.html` instead of `build/pnl_report_{ticker}.html`
    /// so portfolio twins (same ticker, different id) never clobber each other's
    /// report file. Data lookups inside the report still use `ticker` (each twin
    /// has its own RocksDB store).
    pub report_tag: Option<String>,
}

/// HTML report file name for a run: `build/pnl_report_{tag}.html` when a
/// per-asset `--report-tag` is configured, else `build/pnl_report_{ticker}.html`.
fn report_out_path(cfg: &ExecutionConfig) -> String {
    match cfg.report_tag.as_deref() {
        Some(tag) => format!("build/pnl_report_{tag}.html"),
        None => format!("build/pnl_report_{}.html", cfg.ticker),
    }
}

/// Cheap shared snapshot of loop state (read by CLI / PyO3 `status()`).
///
/// The latency fields are the substrate for arbitrage work:
///   * `last_event_ts_ms` / `last_event_instant` -- exchange timestamp and the
///     monotonic instant the newest WS frame landed in the loop (`feed_recv`).
///     `last_event_instant` is set per delivered data frame, NOT per loop
///     iteration, so `feed_staleness_ms` (age since it; computed live in
///     `status()`) grows on a silent feed instead of sawing to the 2s cadence.
///   * `last_event_latency_ms` -- MONOTONIC local code latency: WS dequeue
///     (`feed_recv`) -> fully processed (parse + strategy + db write).
///     Clock-independent; NOT venue RTT.
///   * `clock_skew_ms` -- wall-clock delta (local - exchange) on the newest
///     event. ~0 when NTP-synced; a large value (e.g. -64s) is a CLOCK/NTP
///     problem, surfaced separately, never as `last_event_latency_ms`.
///   * `last_feed_wakeup` -- monotonic instant of the last loop wake (real data
///     OR the 2s read timeout). Drives `loop_liveness_ms`; decoupled from
///     `feed_staleness_ms`.
///   * `loop_cadence_ms` -- wall time of one full feed iteration (wait + work).
///   * `batch_process_ms` -- pure processing time of the last event batch.
#[derive(Debug, Clone, Default)]

pub struct RunStats {
    pub running: bool,
    pub paused: bool,
    pub stop_requested: bool,
    pub capital_usd: f64,
    pub symbol: String,
    pub equity: f64,
    pub n_trades: u64,
    pub n_wins: u64,
    pub n_losses: u64,
    pub total_r: f64,
    pub mid: f64,
    /// Gate telemetry from the newest closed 1-second bar: RSI(2) (hunting
    /// zone) and model P(WIN) (confidence gate). Surfaced in `status()` /
    /// the CLI so operators can watch WHY entries do or don't fire live.
    pub rsi_2: f64,
    pub win_prob: f64,
    /// Which entry gate rejected the newest bar-close evaluation ("" = none).
    /// See `engine::MomentumStrategy::entry_gate`.
    pub gate: String,
    pub last_event: String,
    pub last_error: Option<String>,
    /// Total market events (trades + book-top ticks) ingested by the loop.
    pub events_seen: u64,
    /// Exchange-side timestamp (ms since epoch) of the newest ingested event.
    pub last_event_ts_ms: Option<i64>,
    /// Monotonic instant the latest WS frame was delivered to the loop (right
    /// after `next_events()` returned). Staleness + code-latency anchor.
    pub last_event_instant: Option<std::time::Instant>,
    /// Local code latency: `feed_recv` -> fully processed (ms). Monotonic,
    /// not venue RTT; relative changes are reliable.
    pub last_event_latency_ms: f64,
    /// Wall-clock delta (local - exchange) on the newest event; the NTP skew,
    /// kept separate from `last_event_latency_ms` on purpose.
    pub clock_skew_ms: f64,
    /// Monotonic instant of the last feed wakeup (next_events() returned OR the
    /// 2s read timeout fired). Liveness/stall detector.
    pub last_feed_wakeup: Option<std::time::Instant>,
    /// Wall time of the last full loop iteration, wait included (ms).
    pub loop_cadence_ms: f64,
    /// Pure processing time of the last event batch (ms).
    pub batch_process_ms: f64,
    /// Wall-clock delta (local - exchange) on the newest event; the raw NTP
    /// skew, unchanged — mixes clock offset and transit delay, kept for
    /// backward compatibility and as the input to skew_floor_ms.
    /// Rolling minimum of clock_skew_ms over the trailing window (see
    /// SkewFloor). A live best-case baseline; NOT an absolute clock offset.
    pub skew_floor_ms: f64,
    /// clock_skew_ms - skew_floor_ms: how much staler than recent-normal the
    /// newest event is. This is the number to act on (widen stops, pause
    /// entries) — it isolates transient queuing/network jitter from whatever
    /// constant baseline (clock error + typical transit time) is present.
    pub relative_staleness_ms: f64,
    /// Network-outage telemetry (see NET_OUTAGE_STALENESS_MS): number of
    /// distinct episodes where the newest market event aged past the outage
    /// threshold, the worst staleness seen this session, and whether such an
    /// episode is active right now. Surfaced in `status()` and the periodic
    /// `[net]` console line so the portfolio supervisor / dashboard can count
    /// outages and judge speed standards without a separate diagnostic run.
    pub n_outages: u64,
    pub outage_active: bool,
    pub max_staleness_ms: f64,
    /// Number of trades voided by the regime gate (rejected before entry).
    /// Order round-trip latency: T1 (local timestamp right before the order
    /// is pushed to the socket) → T2 (local timestamp the instant the venue
    /// acknowledgement is read back). Measured ONLY on the local monotonic
    /// clock — no exchange timestamps, no wall clock, no skew. This is the
    /// true "us → exchange → us" trading latency.
    pub last_order_rtt_ms: f64,
    /// Session max of `last_order_rtt_ms`.
    pub max_order_rtt_ms: f64,
    /// Surfaced in the periodic `[net]` console line so the dashboard can
    /// display how many potential trades the regime filter is blocking.
    pub n_voided: u64,

    // ── Phase A: Timing breakdown (ms, monotonic) ─────────────────────────
    /// Time blocked awaiting next_events().await (high = quiet feed OR slow network).
    pub feed_wait_ms: f64,
    /// Time processing the returned batch (feature + model + execution).
    pub feed_process_ms: f64,
    /// Time computing features (finish_micro).
    pub feature_calc_ms: f64,
    /// Time in LightGBM model.predict_proba().
    pub model_infer_ms: f64,
    /// Time in meta-labeller.predict_proba().
    pub labeller_infer_ms: f64,
    /// Time in try_enter / handle_engine_events.
    pub execution_ms: f64,
    /// Time in record_trade / record_orderbook_top (RocksDB writes).
    pub db_write_ms: f64,
    /// Time in generate_html_report.
    pub report_ms: f64,

    // ── Phase A: Catch-up specific (THE central hypothesis) ────────────────
    /// Bars finalized in last catch-up.
    pub catchup_bars: u64,
    /// Worst catch-up this session.
    pub max_catchup_bars: u64,
    /// Time spent in catch-up loop (ms).
    pub catchup_time_ms: f64,
    /// Total catch-up events this session.
    pub n_catchups: u64,

    // ── Phase A: Strategy lag (see header comment for usage notes) ──────
    /// Newest exchange timestamp the feed has delivered.
    pub feed_newest_market_ts: Option<i64>,
    /// Newest exchange timestamp fully processed.
    pub processed_newest_market_ts: Option<i64>,
    /// feed_newest_market_ts - processed_newest_market_ts.
    /// NOTE: In single-threaded architecture this is usually ~0.
    /// Use last_event_latency_ms for per-batch processing time instead.
    pub strategy_lag_ms: f64,
    /// When the last feed batch was received (for freshness measurement).
    pub last_feed_receive_instant: Option<std::time::Instant>,
    /// When the last batch finished processing.
    pub last_processed_instant: Option<std::time::Instant>,

    // ── Phase A: Connection state machine ──────────────────────────────────
    pub ws_state: WsState,
    pub ws_state_changed_at: Option<std::time::Instant>,

    // ── Phase A: Distribution tracking ─────────────────────────────────────
    /// Session max loop time (ms).
    pub max_loop_ms: f64,
}

/// Shared handles the loop and its controlling caller both hold.
pub struct LoopContext {
    /// Market-data source (trades + top-of-book + historical closes warm-up).
    pub feed: Box<dyn Feed>,
    /// Order-routing venue; a no-op broker in paper/replay modes.
    pub broker: Box<dyn Broker>,
    pub stats: Arc<Mutex<RunStats>>,
    pub paused: Arc<AtomicBool>,
    pub stop: Arc<AtomicBool>,
    /// Allocated capital in USD cents (read for position sizing).
    pub capital_usd_cents: Arc<AtomicU64>,
}
/// Run the strategy loop for one ticker until `ctx.stop` is set.
///
/// Reconnects with backoff on stream errors, warms feature rings from
/// historical closes, loads the research model if present, and emits a final
/// HTML report on shutdown.
pub async fn run_execution_loop(cfg: ExecutionConfig, mut ctx: LoopContext) {
    let ticker = cfg.ticker.to_ascii_lowercase();
    // Unique id for THIS process run (unix-ms wall clock at loop start). Stored
    // on every equity snapshot / closed trade so the HTML report can scope
    // itself to the current session, exactly like the terminal counters do.
    let run_session_id: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(1);
    let native_symbol = match cfg.broker {
        ExchangeKind::Bybit => symbols::bybit(&ticker),
        ExchangeKind::Bitfinex => symbols::bitfinex(&ticker),
        ExchangeKind::Binance => symbols::binance(&ticker),
        ExchangeKind::Parquet => ticker.to_ascii_uppercase(),
    };

    // -- shared-state housekeeping -----------------------------------------
    {
        let mut s = ctx.stats.lock().unwrap();
        s.symbol = ticker.clone();
        s.running = true;
        s.capital_usd = ctx.capital_usd_cents.load(Ordering::Relaxed) as f64 / 100.0;
    }

    // -- connect (with retry + backoff) ------------------------------------
    if !connect_with_retry(&mut *ctx.feed, &ticker, &ctx.stop, &ctx.stats).await {
        finish(&cfg, ctx, &None).await;
        return;
    }

    let capital_usd = ctx.capital_usd_cents.load(Ordering::Relaxed) as f64 / 100.0;
    // Start from the file-driven strategy config (single source of truth), then
    // pin position sizing/equity to THIS loop's actual capital.
    let mut strategy_cfg = cfg.strategy_config.clone().unwrap_or_default();
    strategy_cfg.max_position_usd = capital_usd;
    strategy_cfg.starting_equity = capital_usd;
    let mut strategy = MomentumStrategy::new(&ticker, strategy_cfg);

    // Walk-forward parity mode (--wf-dir): score each second with the fold
    // model whose OOS test window covers it, mirroring research's walk-forward
    // evaluation. Retrain-and-deploy is what a live desk actually does, so
    // this remains fully implementable in production.
    if let Some(wf_dir) = &cfg.walkforward_dir {
        match crate::config::load_fold_catalog(std::path::Path::new(wf_dir)) {
            Ok(models) => {
                println!("[wf] loaded {} fold model(s) from {}", models.len(), wf_dir);
                strategy.set_fold_models(models);
            }
            Err(e) => println!("[wf] WARNING: fold catalog unavailable: {e}"),
        }
    }

    // -- persistence --------------------------------------------------------
    let db = if cfg.persist {
        match DbWriter::connect(&cfg.db_path).await {
            Ok(db) => Some(Arc::new(db)),
            Err(e) => {
                eprintln!(
                    "[db] failed to open {}: {e} ??? running without persistence",
                    cfg.db_path
                );
                None
            }
        }
    } else {
        None
    };

    // Register this run in the store's session log so the HTML report can
    // scope itself to the current session even before its first trade closes.
    if let Some(db) = &db {
        db.record_session_start(
            &ticker,
            &SessionMeta {
                session_id: run_session_id,
                start_ms: run_session_id as i64,
            },
        );
    }

    // -- research model ------------------------------------------------------
    if let Some(mdl) = model::load_model() {
        strategy.set_model(mdl);
    } else {
        println!("[model] no research model found ??? using proxy logit (P(WIN) gate only)");
    }

    // -- REGIME-DETECTION GATE (checked FIRST inside the engine — the model
    //    that sits between the main executor and execution). Loaded per
    //    symbol; missing export = gate OPEN (pure addition, never blocks). --
    match cfg.meta_gate {
        MetaGate::Off => println!("[regime] {ticker}: regime gate OFF (--meta-gate off)"),
        _ => match model::load_regime_gate(&ticker) {
            Some(g) => strategy.set_regime_gate(Some(g)),
            None => println!("[regime] {ticker}: no regime_gate export — regime gate OPEN"),
        },
    }

    // -- portfolio meta-labeller gate (PURE ADDITION over the executor:
    //    a missing or un-deployed labeller leaves the gate OPEN and the
    //    strategy trades exactly as before) -------------------------------
    match cfg.meta_gate {
        MetaGate::Off => {
            println!("[meta] {ticker}: labeller gate OFF (--meta-gate off)")
        }
        MetaGate::On => match model::load_labeller(&ticker, true) {
            Some(l) => strategy.set_labeller(Some(l.model), l.threshold),
            None => println!(
                "[meta] {ticker}: --meta-gate on but no labeller export — gate OPEN"
            ),
        },
        MetaGate::Auto => match model::load_labeller(&ticker, false) {
            Some(l) => strategy.set_labeller(Some(l.model), l.threshold),
            None => println!("[meta] {ticker}: labeller gate OPEN (no deployed labeller)"),
        },
    }

    // -- warm feature rings from history -------------------------------------
    match ctx.feed.historical_closes(&ticker, cfg.seed_bars).await {
        Ok(closes) if !closes.is_empty() => {
            strategy.seed_macro_closes(&closes);
            println!(
                "[seed] warmed {} macro closes from {}",
                closes.len(),
                cfg.feed
            );
        }
        Ok(_) => eprintln!("[seed] no historical closes available ??? features will warm live"),
        Err(e) => eprintln!("[seed] historical fetch failed ({e}) ??? features will warm live"),
    }

    // -- restore equity from the last persisted snapshot ---------------------
    let last_exit_ms = AtomicI64::new(i64::MIN);
    let mut skew_floor = SkewFloor::new(120); // 2-minute trailing window
    let mut bar_print = (f64::NAN, f64::NAN); // [bar] telemetry throttle state

    // Emit the effective strategy config so the operator can verify that the
    // JSON file was parsed correctly (especially risk_pct_per_trade, which
    // controls position sizing). This is the single most common source of
    // "why is my position size wrong" bugs.
    println!(
        "[config] {} risk_pct={:.4} max_leverage={:.1} max_pos_usd={:.0} barrier_mult={:.2} rr={:.0}",
        cfg.ticker,
        strategy.cfg.risk_pct_per_trade,
        strategy.cfg.max_leverage,
        strategy.cfg.max_position_usd,
        strategy.cfg.barrier_multiplier,
        strategy.cfg.rr_ratio,
    );

    if !cfg.fresh_equity {

        if let Some(db) = &db {
        if let Ok(snaps) = db.iter_pnl_snapshots(&ticker) {
            if let Some(last) = snaps.last() {
                strategy.restore(last.equity, last.last_exit_ms);
                last_exit_ms.store(last.last_exit_ms, Ordering::SeqCst);
                println!(
                    "[restore] equity={:.2} n_trades={} n_wins={} n_losses={} total_r={:+.2}",
                    last.equity, last.n_trades, last.n_wins, last.n_losses, last.total_r
                );
            }
        }
    }

        // Restore any open position so an in-flight trade survives a restart/crash.
        if let Some(db) = &db {
            if let Ok(Some(pos)) = db.restore_open_position(&ticker) {
                let side = match pos.side.as_str() {
                    "long" => engine::Side::Long,
                    "short" => engine::Side::Short,
                    _ => {
                        eprintln!("[restore] open position has invalid side '{}' — ignoring", pos.side);
                        engine::Side::Long // fallback, will be caught by the check below
                    }
                };
                strategy.restore_position(
                    side,
                    pos.entry_ms,
                    pos.entry_mid,
                    pos.entry_price,
                    pos.qty,
                    pos.notional,
                    pos.barrier_pct,
                    pos.stop,
                    pos.target,
                    pos.hi,
                    pos.lo,
                );
                println!(
                    "[restore] open position: {} {} @ {:.2} stop={:.2} target={:.2}",
                    pos.side, pos.qty, pos.entry_price, pos.stop, pos.target
                );
            }
        }
    }

    println!(
        "=== {} loop [{}] mode={} symbol={} ===",
        cfg.feed,
        ticker,
        match cfg.mode {
            ExecutionMode::Paper => "paper",
            ExecutionMode::Live => "live",
        },
        native_symbol
    );

    // -- main feed loop -------------------------------------------------------
    let mut last_report_wake = std::time::Instant::now();
    // Periodic [net] telemetry cadence (see the print block inside the loop).
    let mut last_net_print = std::time::Instant::now();
// Per-second [tick] live-market telemetry cadence (see the print block).
    let mut last_tick_print = std::time::Instant::now();
    // External-control-file poll cadence (flatten/pause/resume from the parent).
    let mut last_control_poll = std::time::Instant::now() - std::time::Duration::from_millis(100);
    loop {
        if ctx.stop.load(Ordering::SeqCst) {
            break;
        }
        if last_control_poll.elapsed() >= std::time::Duration::from_millis(100) {
            poll_control(
                &cfg, &ctx, &mut strategy, &db, &last_exit_ms,
                &native_symbol, &mut bar_print, run_session_id,
            ).await;
            last_control_poll = std::time::Instant::now();
        }
        let iter_started = std::time::Instant::now();

        // ── Phase A: Track feed wait time separately ─────────────────────
        let feed_wait_start = std::time::Instant::now();
        let events = match ctx.feed.next_events().await {
            Ok(events) => {
                // Mark connection as healthy when we successfully receive events
                if let Ok(mut s) = ctx.stats.lock() {
                    if s.ws_state != WsState::Connected {
                        s.ws_state = WsState::Connected;
                        s.ws_state_changed_at = Some(std::time::Instant::now());
                    }
                }
                events
            }
            Err(e) => {
                eprintln!("[{ticker}] stream error: {e} ??? reconnecting...");
                if let Ok(mut s) = ctx.stats.lock() {
                    s.ws_state = WsState::Disconnected;
                    s.ws_state_changed_at = Some(std::time::Instant::now());
                    s.last_error = Some(e.to_string());
                    s.last_feed_wakeup = Some(std::time::Instant::now());
                }
                // Set RECONNECTING before attempting retry (not after success)
                if let Ok(mut s) = ctx.stats.lock() {
                    s.ws_state = WsState::Reconnecting;
                    s.ws_state_changed_at = Some(std::time::Instant::now());
                }
                if !connect_with_retry(&mut *ctx.feed, &ticker, &ctx.stop, &ctx.stats).await {
                    break;
                }
                continue;
            }
        };
        let feed_wait_ms = feed_wait_start.elapsed().as_secs_f64() * 1000.0;

        // Monotonic instant the WS frame was delivered to the loop, captured
        // right after next_events() returned. Drives last_event_instant (feed
        // freshness) and last_event_latency_ms (code latency) clock-skew-free.
        let feed_recv = std::time::Instant::now();
        if let Ok(mut s) = ctx.stats.lock() {
            s.last_feed_wakeup = Some(feed_recv);
        }

        // Per-batch diagnostics. No per-event lock: tally locally and publish
        // once on the stats refresh below.
        let mut batch_events: u64 = 0;
        let mut batch_last_ts: Option<i64> = None; // any event — drives stal
        let mut batch_last_trade_ts: Option<i64> = None; // trades only — drives sk
        let batch_started = feed_recv;

        // ── Phase A: Accumulators for timing ────────────────────────────
        let mut batch_db_ms: f64 = 0.0;
        let mut batch_exec_ms: f64 = 0.0;

        // ── Phase A: Track newest timestamp from feed for strategy_lag ──
        let mut batch_feed_newest_ts: Option<i64> = None;
        let mut batch_processed_newest_ts: Option<i64> = None;
        for ev in &events {
            let ts = match ev {
                MarketEvent::Trade(t) => t.timestamp,
                MarketEvent::BookTop(b) => b.timestamp,
            };
            batch_feed_newest_ts = Some(batch_feed_newest_ts.map_or(ts, |v| v.max(ts)));
        }

        for event in events {
            // A stop may be raised by the exchange while it returns its final
            // batch (EOF/limit), or by Ctrl-C while a batch is in flight. The
            // batch has already been delivered, so drain every event before
            // the outer loop observes stop and finishes.
            if ctx.paused.load(Ordering::SeqCst) {
                // Paused: keep persisting raw market data but stop feeding the
                // strategy so no new entries/exits are generated.
                if let Some(db) = &db {
                    persist_event(db, &ticker, &event);
                }
                continue;
            }

            match event {
                MarketEvent::Trade(t) => {
                    // ── Phase A: DB write timing ────────────────────────
                    let db_start = std::time::Instant::now();
                    if let Some(db) = &db {
                        db.record_trade(&ticker, t.clone());
                    }
                    batch_db_ms += db_start.elapsed().as_secs_f64() * 1000.0;

                    let is_buy = t.side == TradeSide::Buy;
                    let engine_events = strategy.on_trade(t.timestamp, t.price, t.amount, is_buy);

                    // ── Phase A: Engine execution timing ───────────────
                    let exec_start = std::time::Instant::now();
                    handle_engine_events(
                        &cfg,
                        &ctx,
                        &mut strategy,
                        &db,
                        &last_exit_ms,
                        &native_symbol,
                        &mut bar_print,
                        run_session_id,
                        engine_events,
                    )
                    .await;
                    batch_exec_ms += exec_start.elapsed().as_secs_f64() * 1000.0;

                    batch_events += 1;
                    batch_last_ts = Some(t.timestamp);
                    batch_last_trade_ts = Some(t.timestamp);
                    batch_processed_newest_ts = Some(t.timestamp);
                }
                MarketEvent::BookTop(b) => {
                    let db_start = std::time::Instant::now();
                    if let Some(db) = &db {
                        db.record_orderbook_top(&ticker, b.clone());
                    }
                    batch_db_ms += db_start.elapsed().as_secs_f64() * 1000.0;

                    let engine_events = strategy.on_orderbook(
                        b.timestamp,
                        b.bid_price,
                        b.bid_amount,
                        b.ask_price,
                        b.ask_amount,
                    );
                    let exec_start = std::time::Instant::now();
                    handle_engine_events(
                        &cfg,
                        &ctx,
                        &mut strategy,
                        &db,
                        &last_exit_ms,
                        &native_symbol,
                        &mut bar_print,
                        run_session_id,
                        engine_events,
                    )
                    .await;
                    batch_exec_ms += exec_start.elapsed().as_secs_f64() * 1000.0;

                    batch_events += 1;
                    batch_last_ts = Some(b.timestamp);
                    batch_processed_newest_ts = Some(b.timestamp);
                }
            }
        }

        // Refresh the shared stats snapshot.
        if let Ok(mut s) = ctx.stats.lock() {
            let snap = strategy.snapshot();
            s.equity = snap.equity;
            s.n_trades = snap.n_trades;
            s.n_wins = snap.n_wins;
            s.n_losses = snap.n_losses;
            s.n_voided = snap.n_voided;
            s.total_r = snap.total_r;
            s.mid = snap.mid;
            s.paused = ctx.paused.load(Ordering::SeqCst);
            s.running = !ctx.stop.load(Ordering::SeqCst);

            s.events_seen += batch_events;
            if let Some(ts) = batch_last_ts {
                s.last_event_ts_ms = Some(ts);
                // ── Phase A: Track feed receive time (arrival) ─────────
                s.last_feed_receive_instant = Some(feed_recv);
                // Anchor feed freshness AFTER the batch is fully processed.
                // With real-time pacing a burst batch can take several seconds
                // to replay; anchoring at `feed_recv` (arrival) would count the
                // deliberate replay as an outage (fake [OUTAGE] flicker) even
                // though data is flowing. Post-processing anchor = freshness
                // of the feed HEAD, and a genuine new stall is still measured
                // from the moment the loop becomes idle again.
                s.last_event_instant = Some(std::time::Instant::now());
                s.last_event_latency_ms = feed_recv.elapsed().as_secs_f64() * 1000.0;
                // ── Phase A: Track processing completion time ─────────
                s.last_processed_instant = Some(std::time::Instant::now());
            }
            // Local loop timing is measured on EVERY iteration (pure code
            // latency, meaningful in live AND replay) — outside the net gate.
            s.loop_cadence_ms = iter_started.elapsed().as_secs_f64() * 1000.0;
            s.batch_process_ms = batch_started.elapsed().as_secs_f64() * 1000.0;

            // ── Phase A: Feed wait timing ─────────────────────────────
            s.feed_wait_ms = feed_wait_ms;
            s.feed_process_ms = batch_started.elapsed().as_secs_f64() * 1000.0;

            // ── Phase A: DB and execution timing (accumulated) ────────
            s.db_write_ms = batch_db_ms;
            s.execution_ms = batch_exec_ms;

            // ── Phase A: Engine timing (from strategy snapshot) ───────
            s.feature_calc_ms = snap.feature_calc_ms;
            s.model_infer_ms = snap.model_infer_ms;
            s.labeller_infer_ms = snap.labeller_infer_ms;
            s.catchup_bars = snap.catchup_bars;

            // ── Phase A: Track catch-up session stats ──────────────────
            if snap.catchup_bars > 0 {
                s.n_catchups += 1;
                if snap.catchup_bars > s.max_catchup_bars {
                    s.max_catchup_bars = snap.catchup_bars;
                }
            }

            // ── Phase A: Strategy lag (see note in header) ────────────
            // NOTE: This is NOT yet a perfect backlog measurement in the
            // single-threaded architecture. It measures the difference between
            // the newest event in this batch and the last processed event,
            // which is usually the same. For true backlog measurement across
            // batches, we need Phase C (separate feed reader + channel).
            // For now, use last_event_latency_ms as the better indicator
            // of per-batch processing time.
            if let Some(feed_ts) = batch_feed_newest_ts {
                s.feed_newest_market_ts = Some(feed_ts);
            }
            if let Some(proc_ts) = batch_processed_newest_ts {
                s.processed_newest_market_ts = Some(proc_ts);
            }
            if let (Some(feed_ts), Some(proc_ts)) = (s.feed_newest_market_ts, s.processed_newest_market_ts) {
                s.strategy_lag_ms = (feed_ts - proc_ts) as f64;
            }

            // ── Phase A: Session max loop time ────────────────────────
            if s.loop_cadence_ms > s.max_loop_ms {
                s.max_loop_ms = s.loop_cadence_ms;
            }

            // Network diagnostics are LIVE-TRADE ONLY. A replay (live-test)
            // has no network: staleness/latency/skew against historical
            // timestamps are meaningless, and the per-tick diag RocksDB write
            // would dominate the hot loop. A replay records trades ONLY and
            // runs as fast as possible.
            if cfg.feed != ExchangeKind::Parquet {
                if let Some(trade_ts) = batch_last_trade_ts {
                    let raw_skew = clock_skew_ms(trade_ts);
                    let floor = skew_floor.push(raw_skew);
                    s.clock_skew_ms = raw_skew;
                    s.skew_floor_ms = floor;
                    s.relative_staleness_ms = raw_skew - floor;
                }

                // Feed-health / outage tracking. The DEVLOG standard marks a
                // feed stale (network outage or illiquid market) once the
                // newest event ages past 1000 ms; consecutive ticks past the
                // threshold are ONE episode, so `n_outages` counts distinct
                // events, not ticks.
                let feed_staleness_ms = s
                    .last_event_instant
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                if feed_staleness_ms > NET_OUTAGE_STALENESS_MS {
                    if !s.outage_active {
                        s.outage_active = true;
                        s.n_outages += 1;
                    }
                } else {
                    s.outage_active = false;
                }
                if feed_staleness_ms > s.max_staleness_ms {
                    s.max_staleness_ms = feed_staleness_ms;
                }

                // Persist a latency / feed-health sample to RocksDB every
                // tick so the HTML report's diagnostics pane can show latency
                // / skew / staleness aligned with the price tape.
                if let Some(dbw) = db.as_ref() {
                    let ts_ms = s.last_event_ts_ms.unwrap_or_else(|| {
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as i64)
                            .unwrap_or(0)
                    });
                    dbw.record_diag(&crate::db::DiagSample {
                        ts_ms,
                        latency_ms: s.last_event_latency_ms,
                        clock_skew_ms: s.clock_skew_ms,
                        feed_staleness_ms,
                        relative_staleness_ms: s.relative_staleness_ms,
                    });
                }
            }
        }

        // ── Phase D: Move report generation off hot path ──────────────
        // Report generation is non-critical work. Spawn it as a detached task
        // so it cannot block the trading loop.
        if cfg.persist && last_report_wake.elapsed().as_secs() >= crate::report::REPORT_REFRESH_SECS {
            last_report_wake = std::time::Instant::now();
            if let Some(db) = &db {
                let db_clone = db.clone();
                let ticker = cfg.ticker.clone();
                let scope = cfg.report_scope;
                let rr = cfg
                    .strategy_config
                    .as_ref()
                    .map(|c| c.rr_ratio)
                    .unwrap_or(engine::RR_RATIO);
                let tag = cfg.report_tag.clone();

                // spawn_blocking because generate_html_report is CPU/IO-bound
                // We create a mini-runtime for the blocking task to run async code
                tokio::task::spawn_blocking(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("create mini-runtime for report");
                    rt.block_on(async {
                        crate::report::generate_html_report(
                            &db_clone,
                            &ticker,
                            scope,
                            rr,
                            tag.as_deref(),
                        )
                        .await;
                    });
                });
            }
        }

        // Periodic network-health telemetry. The portfolio supervisor runs each
        // asset as a separate OS process and only sees this console stream, so
        // this line is what carries staleness / latency / clock skew / outage
        // counts into the dashboard (the same numbers latency_diag.py samples
        // via PyO3 status(), and the same ones persisted to RocksDB above).
        if cfg.feed != ExchangeKind::Parquet
            && last_net_print.elapsed().as_secs() >= NET_PRINT_SECS
        {
            last_net_print = std::time::Instant::now();
            if let Ok(s) = ctx.stats.lock() {
                let staleness_ms = s
                    .last_event_instant
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(-1.0);
                // Monotonic ping→pong RTT on the data socket — the actionable
                // network number (clock-skew-free). '-' until the first pong.
                let rtt = match ctx.feed.ws_rtt_ms() {
                    Some(r) => format!("{:.1}ms", r),
                    None => "-".to_string(),
                };
                // Order round-trip T1→T2 (local send → local ack) from the
                // Live order router: `ord=last/max`. '-' while paper mode or
                // before the first order.
                let _ord = if s.last_order_rtt_ms > 0.0 {
                    format!("{:.1}/{:.1}ms", s.last_order_rtt_ms, s.max_order_rtt_ms)
                } else {
                    "-".to_string()
                };

                // ── Phase A: Enhanced [net] line with timing breakdown ──
                // NOTE: strategy_lag_ms is ~0 in single-threaded architecture.
                // Use last_event_latency_ms (proc time) for per-batch cost.
                println!(
                    "[net] {} ws={} loop={:.1}ms feed_wait={:.0}ms proc={:.1}ms feat={:.1}ms model={:.1}ms label={:.1}ms db={:.1}ms catchup={}bars n_catch={} lat_ms={:.0}ms max_loop={:.0}ms stal={:.0}ms rtt={} sk={:+.0}ms outages={} voided={}{}",
                    cfg.ticker,
                    s.ws_state,
                    s.loop_cadence_ms,
                    s.feed_wait_ms,
                    s.last_event_latency_ms,
                    s.feature_calc_ms,
                    s.model_infer_ms,
                    s.labeller_infer_ms,
                    s.db_write_ms,
                    s.catchup_bars,
                    s.n_catchups,
                    s.strategy_lag_ms,
                    s.max_loop_ms,
                    staleness_ms,
                    rtt,
                    s.clock_skew_ms,
                    s.n_outages,
                    s.n_voided,
                    if s.outage_active { " [OUTAGE]" } else { "" },
                );
            }
        }
// Per-second live-market telemetry (PHASE 1 — OBSERVABILITY ONLY):
        // `[tick]` gives the supervisor / VIM dashboard a MICRO price/position
        // channel independent of the ~5-minute `[bar]` macro cadence. mid /
        // rsi / p_win refresh every tick; an open position's mark-to-market
        // R & USD are recomputed each tick; `armed` is the STATIC threshold
        // observation `p_win >= confidence_threshold` — exactly the check the
        // `[bar]` line's `[armed]` tag makes — NOT a state machine. Nothing
        // here is persisted and nothing changes strategy behavior (a
        // debounced `signal_armed` hysteresis flag is Phase 2 and stays OUT
        // of this line).
        if cfg.feed != ExchangeKind::Parquet
            && last_tick_print.elapsed().as_secs() >= TICK_PRINT_SECS
        {
            last_tick_print = std::time::Instant::now();
            if let Ok(s) = ctx.stats.lock() {
                let staleness_ms = s
                    .last_event_instant
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(-1.0);
                let conf_thr = cfg
                    .strategy_config
                    .as_ref()
                    .map(|c| c.confidence_threshold)
                    .unwrap_or(CONFIDENCE_THRESHOLD);
                let armed = if s.win_prob + 1e-12 >= conf_thr { 1 } else { 0 };
                let ts_ms = s.last_event_ts_ms.unwrap_or_else(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0)
                });
                // Open position block (absent when flat): side/qty/entry/stop/
                // target plus live mark-to-market R and USD at the current mid.
                let pos_txt = match strategy.open_position() {
                    Some(p) => {
                        let side_str = match p.side {
                            engine::Side::Long => "long",
                            engine::Side::Short => "short",
                        };
                        let (rr, usd) =
                            mark_to_market(p.side, s.mid, p.entry_price, p.qty, p.barrier_pct);
                        let notional = p.qty * p.entry_price;
                        format!(
                            " side={} qty={:.4} notional={:.2} entry={:.2} stop={:.2} target={:.2} bp={:.5} rr={:+.2} usd={:+.4}",
                            side_str,
                            p.qty,
                            notional,
                            p.entry_price,
                            p.stop,
                            p.target,
                            p.barrier_pct,
                            rr,
                            usd,
                        )
                    },
                    None => " flat".to_string(),
                };
                println!(
                    "[tick] {} {} mid={:.2} rsi2={:.1} p_win={:.2} armed={} gate={}{} stal={:.0}ms",
                    cfg.ticker,
                    format_utc(ts_ms),
                    s.mid,
                    s.rsi_2,
                    s.win_prob,
                    armed,
                    s.gate,
                    pos_txt,
                    staleness_ms,
                );
            }
        }
    }

    finish(&cfg, ctx, &db).await;
}
/// Persist a market event (used while paused so the raw feed is never lost).
fn persist_event(db: &DbWriter, ticker: &str, event: &MarketEvent) {
    match event {
        MarketEvent::Trade(t) => db.record_trade(ticker, t.clone()),
        MarketEvent::BookTop(b) => db.record_orderbook_top(ticker, b.clone()),
    }
}

/// Wall-clock delta between this machine and the venue server, in ms, for the
/// newest exchange-reported event timestamp: `now - exchange_ts`.
///
/// ~0 when the box is NTP-synced to the venue; a large value (e.g. -64s in
/// this sandbox) is a CLOCK/NTP problem, not a code or network problem. It is
/// surfaced separately from `last_event_latency_ms` on purpose and never
/// returned as ingest latency.
fn clock_skew_ms(exchange_ts_ms: i64) -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    (now_ms - exchange_ts_ms) as f64
}

/// UTC calendar date from days since the Unix epoch (Howard Hinnant's
/// civil_from_days). Only used for human-readable console timestamps — no
/// chrono dependency needed for that.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Human-readable UTC timestamp for console telemetry: `2026-08-25 18:02:45Z`.
fn format_utc(ts_ms: i64) -> String {
    let secs = ts_ms.div_euclid(1000);
    let (days, sod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, mo, d) = civil_from_days(days);
    format!(
        "{y:04}-{mo:02}-{d:02} {:02}:{:02}:{:02}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// Connect with retry + backoff until `stop` is set. Returns false if the
/// caller should abort.
async fn connect_with_retry(
    feed: &mut dyn Feed,
    ticker: &str,
    stop: &AtomicBool,
    stats: &Arc<Mutex<RunStats>>,
) -> bool {
    loop {
        if stop.load(Ordering::SeqCst) {
            return false;
        }
        match feed.connect(ticker).await {
            Ok(()) => return true,
            Err(e) => {
                eprintln!("[{ticker}] connect failed: {e} ??? retrying in 2s");
                if let Ok(mut s) = stats.lock() {
                    s.last_error = Some(e.to_string());
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    }
}

/// React to strategy events: log, persist PnL snapshots/trade records, and in
/// `Live` mode route entries/exits to the exchange as real orders.
async fn handle_engine_events(
    cfg: &ExecutionConfig,
    ctx: &LoopContext,
    strategy: &mut MomentumStrategy,
    db: &Option<Arc<DbWriter>>,
    last_exit_ms: &AtomicI64,
    native_symbol: &str,
    bar_print: &mut (f64, f64), // (rsi_2, win_prob) of last printed [bar] line
    run_session_id: u64,
    events: Vec<EngineEvent>,
) {
    for event in events {
        match event {
            EngineEvent::BarClosed(f) => {
                if let Ok(mut s) = ctx.stats.lock() {
                    s.last_event = format!(
                        "bar ts={} close={:.2} rsi2={:.1} win={:.2}",
                        f.ts, f.close, f.rsi_2, f.win_prob
                    );
                    s.rsi_2 = f.rsi_2;
                    s.win_prob = f.win_prob;
                    s.gate = strategy.entry_gate().to_string();
                }
                // Persist the feature sample so the HTML report can draw the
                // full RSI / P(WIN) history in its indicator panes.
                if let Some(db) = db {
                    db.record_feature(
                        &cfg.ticker,
                        &crate::db::FeatureSample {
                            ts_ms: f.ts,
                            rsi_2: f.rsi_2,
                            win_prob: f.win_prob,
                        },
                    );
                }
                // Console gate telemetry. RSI(2)/P(WIN) are macro features
                // recomputed once per composite-bar boundary, so printing only
                // on a change yields one line per ~5 minutes instead of one
                // per second — enough to watch the hunting zone / confidence
                // gates approach an entry without drowning the console.
                let conf_thr = cfg
                    .strategy_config
                    .as_ref()
                    .map(|c| c.confidence_threshold)
                    .unwrap_or(CONFIDENCE_THRESHOLD);
                if (*bar_print).0 != f.rsi_2 || (*bar_print).1 != f.win_prob {
                    *bar_print = (f.rsi_2, f.win_prob);
                    println!(
                        "[bar] {} {} close={:.2} rsi2={:.1} p_win={:.2}{}{}",
                        cfg.ticker,
                        format_utc(f.ts),
                        f.close,
                        f.rsi_2,
                        f.win_prob,
                        if f.rsi_2 < 30.0 { " [in-zone]" } else { "" },
                        if f.win_prob + 1e-12 >= conf_thr { " [armed]" } else { "" },
                    );
                }
            }
            EngineEvent::Entered(r) => {
                println!(
                    "[entry] {} {} qty={:.4} @ {:.2} stop={:.2} target={:.2} bp={:.5} notional={:.2} (rsi2={:.1}, win={:.2})",
                    cfg.ticker,
                    r.side.as_str(),
                    r.qty,
                    r.entry_price,
                    r.stop,
                    r.target,
                    r.barrier_pct,
                    r.notional,
                    r.rsi_2,
                    r.win_prob
                );
                if cfg.mode == ExecutionMode::Live {
                    route_entry(ctx.broker.as_ref(), native_symbol, &r, &ctx.stats).await;
                }
                if let Some(db) = db {
                    let snap = strategy.snapshot();
                    let pnl = PnLSnapshot {
                        ts_ms: r.ts,
                        equity: snap.equity,
                        n_trades: snap.n_trades,
                        n_wins: snap.n_wins,
                        n_losses: snap.n_losses,
                        total_r: snap.total_r,
                        mid: snap.mid,
                        last_exit_ms: last_exit_ms.load(Ordering::SeqCst),
                        session_id: run_session_id,
                    };
                    db.record_pnl_snapshot(&snap.symbol, &pnl);
                    // Persist the open position so it survives a restart/crash.
                    if let Some(p) = strategy.open_position() {
                        db.record_open_position(
                            &snap.symbol,
                            &crate::db::PersistPosition {
                                side: p.side.as_str().to_string(),
                                entry_ms: p.entry_ms,
                                entry_mid: p.entry_mid,
                                entry_price: p.entry_price,
                                qty: p.qty,
                                notional: p.notional,
                                barrier_pct: p.barrier_pct,
                                stop: p.stop,
                                target: p.target,
                                hi: p.hi,
                                lo: p.lo,
                            },
                        );
                    }
                }
                if let Ok(mut s) = ctx.stats.lock() {
                    s.last_event = format!("entry {} @ {:.2}", r.side.as_str(), r.entry_price);
                }
            }
            EngineEvent::Exited(rec) => {
                last_exit_ms.store(rec.exit_ms, Ordering::SeqCst);
                println!(
                    "[exit] {} reason={} @ {:.2} r={:+.2} net={:+.2} equity={:.2} ({:.0}s)",
                    cfg.ticker,
                    rec.reason,
                    rec.exit_price,
                    rec.r_multiple,
                    rec.net_usd,
                    rec.equity_after,
                    rec.duration_s
                );
                if cfg.mode == ExecutionMode::Live {
                    route_exit(ctx.broker.as_ref(), native_symbol, &rec, &ctx.stats).await;
                }
                if let Some(db) = db {
                    let snap = strategy.snapshot();
                    let pnl = PnLSnapshot {
                        ts_ms: rec.exit_ms,
                        equity: snap.equity,
                        n_trades: snap.n_trades,
                        n_wins: snap.n_wins,
                        n_losses: snap.n_losses,
                        total_r: snap.total_r,
                        mid: snap.mid,
                        last_exit_ms: rec.exit_ms,
                        session_id: run_session_id,
                    };
                    db.record_pnl_snapshot(&snap.symbol, &pnl);

                    let rr = cfg
                        .strategy_config
                        .as_ref()
                        .map(|c| c.rr_ratio)
                        .unwrap_or(10.0);
                    let (target_price, stop_price) = if rec.barrier_pct > 0.0 {
                        match rec.side {
                            engine::Side::Long => (
                                rec.entry_price * (1.0 + rec.barrier_pct * rr),
                                rec.entry_price * (1.0 - rec.barrier_pct),
                            ),
                            engine::Side::Short => (
                                rec.entry_price * (1.0 - rec.barrier_pct * rr),
                                rec.entry_price * (1.0 + rec.barrier_pct),
                            ),
                        }
                    } else {
                        (0.0, 0.0)
                    };

                    let ct = ClosedTrade {
                        entry_ms: rec.entry_ms,
                        exit_ms: rec.exit_ms,
                        side: rec.side.as_str().to_string(),
                        entry_price: rec.entry_price,
                        exit_price: rec.exit_price,
                        reason: rec.reason.to_string(),
                        qty: rec.qty,
                        net_usd: rec.net_usd,
                        r_multiple: rec.r_multiple,
                        r_raw: Some(rec.r_raw),
                        equity_after: rec.equity_after,
                        duration_s: rec.duration_s,
                        session_id: run_session_id,
                        barrier_pct: rec.barrier_pct,
                        target_price,
                        stop_price,
                    };
                    db.record_closed_trade(&snap.symbol, &ct);
                    // Clear the persisted open position — the trade is closed.
                    db.clear_open_position(&snap.symbol);
                }
                if let Ok(mut s) = ctx.stats.lock() {
                    s.last_event = format!("exit {} r={:+.2}", rec.reason, rec.r_multiple);
                }
            }
            // Voided (regime/labeller-rejected) candidate closed its shadow
            // window. NEVER touches equity/stats — it is appended to the
            // gate log so the regime research never loses the data.
            EngineEvent::VoidClosed(vt) => {
                println!(
                    "[meta] VOIDED {} {} r={:+.2} pwin={:.3} ({}s, {})",
                    cfg.ticker,
                    vt.side.as_str(),
                    vt.r_multiple,
                    vt.meta_pwin,
                    vt.duration_s,
                    vt.reason
                );
                if let Ok(mut s) = ctx.stats.lock() {
                    s.last_event = format!("voided r={:+.2}", vt.r_multiple);
                }
                if cfg.persist {
                    let tag = cfg.report_tag.as_deref().unwrap_or(&cfg.ticker);
                    let path = std::path::Path::new("build").join(format!("gate_log_{tag}.jsonl"));
                    let record = serde_json::json!({
                        "gate": "voided",
                        "ticker": cfg.ticker,
                        "entry_ms": vt.entry_ms,
                        "exit_ms": vt.exit_ms,
                        "side": vt.side.as_str(),
                        "entry_price": vt.entry_price,
                        "exit_price": vt.exit_price,
                        "reason": vt.reason,
                        "qty": vt.qty,
                        "notional": vt.notional,
                        "barrier_pct": vt.barrier_pct,
                        "r_multiple": vt.r_multiple,
                        "r_raw": vt.r_raw,
                        "duration_s": vt.duration_s,
                        "meta_pwin": vt.meta_pwin,
                        "features": vt.feature_snapshot,
                        "session_id": run_session_id,
                    });
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        use std::io::Write;
                        let _ = writeln!(f, "{record}");
                    }
                }
            }
        }
    }
}

/// Graceful teardown: final stats, HTML report, close the exchange.
async fn finish(cfg: &ExecutionConfig, mut ctx: LoopContext, db: &Option<Arc<DbWriter>>) {
    if let Ok(mut s) = ctx.stats.lock() {
        s.running = false;
        s.stop_requested = true;
        s.paused = false;
    }
    if cfg.persist {
        if let Some(db) = db {
            let rr = cfg
                .strategy_config
                .as_ref()
                .map(|c| c.rr_ratio)
                .unwrap_or(engine::RR_RATIO);
            crate::report::generate_html_report(
                db,
                &cfg.ticker,
                cfg.report_scope,
                rr,
                cfg.report_tag.as_deref(),
            )
            .await;
        }
    }
    // The session report is transient by design: it exists only WHILE the
    // strategy runs (regenerated on a timer for the live browser view) and is
    // removed at session end — every trade / equity snapshot / diag sample it
    // rendered is already persisted in RocksDB, so nothing is lost and build/
    // does not accumulate stale per-session HTML files.
    let report_path = report_out_path(cfg);
    match std::fs::remove_file(&report_path) {
        Ok(_) => println!(
            "[report] session ended - removed {report_path} (data persists in RocksDB)"
        ),
        Err(_) => {} // never written (no-db) or already gone
    }
    if let Err(e) = ctx.feed.close().await {
        eprintln!("[feed close] {e}");
    }
    if let Err(e) = ctx.broker.close().await {
        eprintln!("[broker close] {e}");
    }
}

/// Route an engine entry to the exchange (Live mode): market order sized to
/// the engine receipt quantity. Measures the order round-trip T1→T2 (local
/// send → local ack) on the monotonic clock and records it in RunStats.
async fn route_entry(
    broker: &dyn Broker,
    symbol: &str,
    r: &engine::EntryReceipt,
    stats: &Arc<Mutex<RunStats>>,
) {
    let side = if r.side == engine::Side::Long {
        TradeSide::Buy
    } else {
        TradeSide::Sell
    };
    let req = OrderRequest {
        symbol: symbol.to_string(),
        side,
        order_type: OrderType::Market,
        price: 0.0,
        amount: r.qty,
    };
    let t1 = std::time::Instant::now(); // T1: right before hitting the socket
    let result = broker.place_order(&req).await;
    let rtt_ms = t1.elapsed().as_secs_f64() * 1000.0; // T2: ack read back
    record_order_rtt(stats, rtt_ms);
    match result {
        Ok(resp) => println!(
            "[order] submitted {} side={} status={} rtt={rtt_ms:.1}ms",
            resp.exchange_order_id,
            if side == TradeSide::Buy {
                "buy"
            } else {
                "sell"
            },
            resp.status
        ),
        Err(e) => eprintln!(
            "[order] FAILED to place entry order rtt={rtt_ms:.1}ms: {e}"
        ),
    }
}

/// Route an engine exit to the exchange (Live mode): close the position by
/// trading back the recorded quantity on the opposite side. Same T1→T2
/// round-trip measurement as `route_entry`.
async fn route_exit(
    broker: &dyn Broker,
    symbol: &str,
    rec: &engine::TradeRecord,
    stats: &Arc<Mutex<RunStats>>,
) {
    let side = if rec.side == engine::Side::Long {
        TradeSide::Sell
    } else {
        TradeSide::Buy
    };
    let req = OrderRequest {
        symbol: symbol.to_string(),
        side,
        order_type: OrderType::Market,
        price: 0.0,
        amount: rec.qty,
    };
    let t1 = std::time::Instant::now(); // T1: right before hitting the socket
    let result = broker.place_order(&req).await;
    let rtt_ms = t1.elapsed().as_secs_f64() * 1000.0; // T2: ack read back
    record_order_rtt(stats, rtt_ms);
    match result {
        Ok(resp) => println!(
            "[order] closed {} status={} rtt={rtt_ms:.1}ms",
            resp.exchange_order_id, resp.status
        ),
        Err(e) => eprintln!(
            "[order] FAILED to place exit order rtt={rtt_ms:.1}ms: {e}"
        ),
    }
}

/// Store one order round-trip sample (last + session max). Never blocks the
/// trading loop on a poisoned lock.
fn record_order_rtt(stats: &Arc<Mutex<RunStats>>, rtt_ms: f64) {
    if let Ok(mut s) = stats.lock() {
        s.last_order_rtt_ms = rtt_ms;
        if rtt_ms > s.max_order_rtt_ms {
            s.max_order_rtt_ms = rtt_ms;
        }
    }
}

/// Poll the external control file (`--control <path>`) and act on its command.
/// Supported payloads: `{"cmd":"flatten"}` (pause + close any open position
/// at the current mid), `{"cmd":"pause"}`, `{"cmd":"resume"}`. The file is
/// consumed (deleted) after being acted on, so a restart never re-triggers
/// a stale command and the supervisor stays idempotent.

async fn poll_control(
    cfg: &ExecutionConfig,
    ctx:&LoopContext,
    strategy:&mut MomentumStrategy,
    db:&Option<Arc<DbWriter>>,
    last_exit_ms:&AtomicI64,
    native_symbol:&str,
    bar_print:&mut (f64,f64),
    run_session_id: u64,
) {
    let Some(path) = cfg.control_path.as_deref() else { return; };
    let Ok(text) = std::fs::read_to_string(path) else { return; };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        // Malformed control payload: drop it so it can't wedge the loop.

        let _ = std::fs::remove_file(path);
        return;
    };
    let cmd = value
        .get("cmd")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let _ = std::fs::remove_file(path);
    match cmd.as_str() {
        "" => {}
        "pause" => {
            ctx.paused.store(true, Ordering::SeqCst);
            println!("[paused] {} (external control) - entries paused", cfg.ticker);
        }
        "resume" => {
            ctx.paused.store(false, Ordering::SeqCst);
            println!("[resumed] {} (external control) - entries active", cfg.ticker);
        }
        "flatten" => {
            ctx.paused.store(true, Ordering::SeqCst);
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            match strategy.force_close(now_ms, "flatten", strategy.mid()) {
                Some(rec) => {
                    println!(
                        "[flatten] {} closed @ {:.2} r={:+.2} (entries paused)",
                        cfg.ticker, rec.exit_price, rec.r_multiple
                    );
                    handle_engine_events(
                        cfg, ctx, strategy, db, last_exit_ms, native_symbol,
                        bar_print, run_session_id,
                        vec![EngineEvent::Exited(rec)],
                    ).await;
                }
                None => println!(
                    "[paused] {} flatten - flat (no open position), entries paused",
                    cfg.ticker
                ),
            }
        }
        other => eprintln!("[control] unknown command {other:?}"),
    }
}
/// Phase-1 mark-to-market telemetry for the `[tick]` console line.
/// Mirrors `close_position`'s timeout-R basis WITHOUT slip/fees: R is the raw
/// side-adjusted move from entry divided by the entry barrier (0.0 when no
/// barrier exists yet); USD is the raw side-adjusted mark on the open
/// quantity. Observational only — the strategy's real PnL accounting and the
/// persisted `net_usd`/`r_multiple` record are untouched by this helper.
fn mark_to_market(
    side: engine::Side,
    mid: f64,
    entry_price: f64,
    qty: f64,
    barrier_pct: f64,
) -> (f64, f64) {
    let side_sign = match side {
        engine::Side::Long => 1.0,
        engine::Side::Short => -1.0,
    };
    let rr = if barrier_pct > 1e-9 {
        (mid - entry_price) / entry_price * side_sign / barrier_pct
    } else {
        0.0 // no vol-derived barrier yet -> no R, never divide by zero
    };
    (rr, (mid - entry_price) * qty * side_sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase-1 mark-to-market: LONG above entry -> positive R & USD; SHORT
    /// below entry -> positive; degenerate zero barrier -> R 0 (no div-by-zero)
    /// while USD is still measured.
    #[test]
    fn mark_to_market_sign_and_scale() {
        let (rr_long, usd_long) =
            mark_to_market(engine::Side::Long, 101.0, 100.0, 2.0, 0.01);
        assert!((rr_long - 1.0).abs() < 1e-9);   // (101-100)/100 / 0.01 = 1.0
        assert!((usd_long - 2.0).abs() < 1e-9);  // (101-100) * 2.0
        assert!(rr_long > 0.0 && usd_long > 0.0);

        let (rr_short, usd_short) =
            mark_to_market(engine::Side::Short, 99.99, 101.0, 1.0, 0.01);
        // (101-99.99)/101 = 1% drop from entry; short gains → ~+1R.
        assert!((rr_short - 1.0).abs() < 1e-6);
        assert!(usd_short > 0.0);

        let (rr_zero, usd_zero) =
            mark_to_market(engine::Side::Long, 101.0, 100.0, 2.0, 0.0);
        assert!(rr_zero == 0.0);                 // zero barrier -> R 0, no NaN/Inf
        assert!(usd_zero > 0.0);                 // USD mark is barrier-independent
    }
}
