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
    /// `build/live_test_config.json` exported by `unified_runner.py` from
    /// `main/config.py`). When `None` the engine falls back to its built-in
    /// `Default`, which mirrors `main/config.py`.
    pub strategy_config: Option<StrategyConfig>,
    /// Optional walk-forward fold-model dir (`build/walkforward_models`).
    /// When set, each 1 s row is scored with the fold model whose OOS test
    /// window covers it (live-legal equivalent of periodic retrain+deploy).
    pub walkforward_dir: Option<String>,
    /// Which persisted runs the HTML report shows (defaults to `latest`,
    /// matching the terminal session counters).
    pub report_scope: crate::report::ReportScope,
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
    if let Some(db) = &db {
        if let Ok(snaps) = db.iter_pnl_snapshots(&ticker) {
            if let Some(last) = snaps.last() {
                strategy.restore(last.equity, last.last_exit_ms);
                last_exit_ms.store(last.last_exit_ms, Ordering::SeqCst);
                println!(
                    "[restore] equity={:.2} USD n_trades={}",
                    last.equity, last.n_trades
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
    loop {
        if ctx.stop.load(Ordering::SeqCst) {
            break;
        }
        let iter_started = std::time::Instant::now();

        let events = match ctx.feed.next_events().await {
            Ok(events) => events,
            Err(e) => {
                eprintln!("[{ticker}] stream error: {e} ??? reconnecting...");
                if let Ok(mut s) = ctx.stats.lock() {
                    s.last_error = Some(e.to_string());
                    s.last_feed_wakeup = Some(std::time::Instant::now());
                }
                if !connect_with_retry(&mut *ctx.feed, &ticker, &ctx.stop, &ctx.stats).await {
                    break;
                }
                continue;
            }
        };

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
                    if let Some(db) = &db {
                        db.record_trade(&ticker, t.clone());
                    }
                    let is_buy = t.side == TradeSide::Buy;
                    let engine_events = strategy.on_trade(t.timestamp, t.price, t.amount, is_buy);
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
                    batch_events += 1;
                    batch_last_ts = Some(t.timestamp);
                    batch_last_trade_ts = Some(t.timestamp); // new
                }
                MarketEvent::BookTop(b) => {
                    if let Some(db) = &db {
                        db.record_orderbook_top(&ticker, b.clone());
                    }
                    let engine_events = strategy.on_orderbook(
                        b.timestamp,
                        b.bid_price,
                        b.bid_amount,
                        b.ask_price,
                        b.ask_amount,
                    );
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
                    batch_events += 1;
                    batch_last_ts = Some(b.timestamp);
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
            s.total_r = snap.total_r;
            s.mid = snap.mid;
            s.paused = ctx.paused.load(Ordering::SeqCst);
            s.running = !ctx.stop.load(Ordering::SeqCst);

            s.events_seen += batch_events;
            if let Some(ts) = batch_last_ts {
                s.last_event_ts_ms = Some(ts);
                s.last_event_instant = Some(feed_recv);
                s.last_event_latency_ms = feed_recv.elapsed().as_secs_f64() * 1000.0;
            }
            if let Some(trade_ts) = batch_last_trade_ts {
                let raw_skew = clock_skew_ms(trade_ts);
                let floor = skew_floor.push(raw_skew);
                s.clock_skew_ms = raw_skew;
                s.skew_floor_ms = floor;
                s.relative_staleness_ms = raw_skew - floor;
            }
            s.loop_cadence_ms = iter_started.elapsed().as_secs_f64() * 1000.0;
            s.batch_process_ms = batch_started.elapsed().as_secs_f64() * 1000.0;

            // Persist a latency / feed-health sample to RocksDB every tick so
            // the HTML report's diagnostics pane can show latency / skew /
            // staleness aligned with the price tape (previously only surfaced by
            // a Python poller writing CSVs into build/).
            if let Some(dbw) = db.as_ref() {
                let feed_staleness_ms = s
                    .last_event_instant
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                let ts_ms = s.last_event_ts_ms.unwrap_or_else(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0)
                });
                dbw.record_diag(
                    &cfg.ticker,
                    &crate::db::DiagSample {
                        ts_ms,
                        latency_ms: s.last_event_latency_ms,
                        clock_skew_ms: s.clock_skew_ms,
                        feed_staleness_ms,
                        relative_staleness_ms: s.relative_staleness_ms,
                    },
                );
            }
        }

        // Live report: republish the HTML snapshot on a timer so an open
        // browser tab (the page meta-refreshes) shows entries/exits and the
        // growing price tape as they happen — not only on shutdown.
        if cfg.persist && last_report_wake.elapsed().as_secs() >= crate::report::REPORT_REFRESH_SECS
        {
            last_report_wake = std::time::Instant::now();
            if let Some(db) = &db {
                let rr = cfg
                    .strategy_config
                    .as_ref()
                    .map(|c| c.rr_ratio)
                    .unwrap_or(engine::RR_RATIO);
                crate::report::generate_html_report(db, &cfg.ticker, cfg.report_scope, rr).await;
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
                    "[entry] {} {} qty={:.4} @ {:.2} stop={:.2} target={:.2} (rsi2={:.1}, win={:.2})",
                    cfg.ticker,
                    r.side.as_str(),
                    r.qty,
                    r.entry_price,
                    r.stop,
                    r.target,
                    r.rsi_2,
                    r.win_prob
                );
                if cfg.mode == ExecutionMode::Live {
                    route_entry(ctx.broker.as_ref(), native_symbol, &r).await;
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
                    route_exit(ctx.broker.as_ref(), native_symbol, &rec).await;
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
                }
                if let Ok(mut s) = ctx.stats.lock() {
                    s.last_event = format!("exit {} r={:+.2}", rec.reason, rec.r_multiple);
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
            crate::report::generate_html_report(db, &cfg.ticker, cfg.report_scope, rr).await;
        }
    }
    if let Err(e) = ctx.feed.close().await {
        eprintln!("[feed close] {e}");
    }
    if let Err(e) = ctx.broker.close().await {
        eprintln!("[broker close] {e}");
    }
}

/// Route an engine entry to the exchange (Live mode): market order sized to
/// the engine receipt quantity.
async fn route_entry(broker: &dyn Broker, symbol: &str, r: &engine::EntryReceipt) {
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
    match broker.place_order(&req).await {
        Ok(resp) => println!(
            "[order] submitted {} side={} status={}",
            resp.exchange_order_id,
            if side == TradeSide::Buy {
                "buy"
            } else {
                "sell"
            },
            resp.status
        ),
        Err(e) => eprintln!("[order] FAILED to place entry order: {e}"),
    }
}

/// Route an engine exit to the exchange (Live mode): close the position by
/// trading back the recorded quantity on the opposite side.
async fn route_exit(broker: &dyn Broker, symbol: &str, rec: &engine::TradeRecord) {
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
    match broker.place_order(&req).await {
        Ok(resp) => println!(
            "[order] closed {} status={}",
            resp.exchange_order_id, resp.status
        ),
        Err(e) => eprintln!("[order] FAILED to place exit order: {e}"),
    }
}
