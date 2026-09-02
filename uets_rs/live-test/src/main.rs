//! live-test CLI binary — historical replay through the live-trade loop.
//!
//! Instead of a live WebSocket feed, the exchange is a [`ReplayExchange`]
//! that replays a pre-converted parquet feed. `unified_runner.py live-test`
//! reads the research parquet files (`research/csv/{TICKER}_trades.parquet`,
//! `{TICKER}_orderbook.parquet`) via pyarrow/pandas and merges them into a
//! timestamp-ordered replay stream (`build/live_test_feed_{ticker}.replay`).
//! Everything downstream — 1 s micro-bars, 300 s composite features,
//! entry/exit gates, RocksDB persistence, HTML report — is the identical
//! code path as `live-trade`, so a replay of the same data that produced the
//! research backtest should reproduce the same trades.
//!
//!     cargo run -p live-test -- btcusdt --feed build/live_test_feed_btcusdt.replay
//!     cargo run -p live-test -- ethusdt --feed build/live_test_feed_ethusdt.replay \
//!         --capital 2000 --seed 150

use live_trade::exchange::{ExchangeKind, create_broker, create_replay_feed};
use live_trade::execution::{
    ExecutionConfig, ExecutionMode, LoopContext, MetaGate, RunStats, run_execution_loop,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn usage<T>() -> T {
    eprintln!(
        "usage: live-test [ticker] [--feed PATH] [--capital USD] [--db PATH] [--no-db] [--seed BARS] [--loop] [--start-ms MS] [--end-ms MS] [--limit N] [--config PATH] [--parity] [--wf-dir DIR] [--meta-gate auto|on|off]"
    );
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    let mut ticker = "btcusdt".to_string();
    let mut feed: Option<String> = None;
    let mut capital = 10_000.0f64;
    let mut db_path: Option<String> = None;
    let mut persist = true;
    let mut seed_bars = 0u32;
    let mut loop_feed = false;
    let mut start_ms: Option<i64> = None;
    let mut end_ms: Option<i64> = None;
    let mut limit: Option<usize> = None;
    let mut config_path: Option<String> = None;
    let mut parity = false;
    let mut wf_dir: Option<String> = None;
    let mut meta_gate = MetaGate::Auto;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--feed" => feed = Some(args.next().unwrap_or_else(usage)),
            "--capital" => {
                capital = args
                    .next()
                    .unwrap_or_else(usage)
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--db" => db_path = Some(args.next().unwrap_or_else(usage)),
            "--no-db" => persist = false,
            "--seed" => {
                seed_bars = args
                    .next()
                    .unwrap_or_else(usage)
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--config" => config_path = Some(args.next().unwrap_or_else(usage)),
            "--parity" => parity = true,
            "--wf-dir" => wf_dir = Some(args.next().unwrap_or_else(usage)),
            "--loop" => loop_feed = true,
            "--start-ms" => {
                start_ms = Some(
                    args.next()
                        .unwrap_or_else(usage)
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "--end-ms" => {
                end_ms = Some(
                    args.next()
                        .unwrap_or_else(usage)
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "--limit" => {
                limit = Some(
                    args.next()
                        .unwrap_or_else(usage)
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "--meta-gate" => {
                meta_gate = match args.next().unwrap_or_else(usage).as_str() {
                    "auto" => MetaGate::Auto,
                    "on" => MetaGate::On,
                    "off" => MetaGate::Off,
                    _ => usage(),
                }
            }
            "--help" | "-h" => usage(),
            other if other.starts_with("--") => usage(),
            other => ticker = other.to_ascii_lowercase(),
        }
    }

    let ticker_lower = ticker.to_ascii_lowercase();
    let feed = feed.unwrap_or_else(|| format!("build/live_test_feed_{ticker_lower}.replay"));

    // A replay run must start from a clean slate. The default DB location is
    // wiped every run so a re-run can't restore stale equity or duplicate
    // trades; an explicitly provided `--db` is left alone (user's choice).
    let db_path = db_path.clone().unwrap_or_else(|| {
        let p = format!("build/live_test_{ticker_lower}.rocksdb");
        if persist && std::path::Path::new(&p).exists() {
            let _ = std::fs::remove_dir_all(&p);
        }
        p
    });

    let stats = Arc::new(Mutex::new(RunStats {
        symbol: ticker_lower.clone(),
        capital_usd: capital,
        ..Default::default()
    }));
    let paused = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let capital_cents = Arc::new(AtomicU64::new((capital * 100.0) as u64));

    // Strategy parameters exported by unified_runner.py from main/config.py —
    // the single source of truth for live-test / research parity.
        let strategy_config = config_path
        .as_deref()
        .map(std::path::Path::new)
        .and_then(|p| live_trade::config::load_strategy_config(p, &ticker_lower));
    // Walk-forward parity mode (--wf-dir implies --parity): flip the two
    // research-universe gates so live-test scores each second with the fold
    // model covering it and takes one volume>0 candidate per composite bar —
    // exactly the research trade universe.
    let mut strategy_config = strategy_config;
    if parity {
        if let Some(c) = strategy_config.as_mut() {
            c.require_fold_coverage = true;
            c.one_candidate_per_bar = true;
        }
    }
    if let Some(path) = &config_path {
        match &strategy_config {
            Some(cfg) => println!(
                "[config] loaded {path} (zone_gen={} conf={} lockout={} rr={} dir_long={})",
                cfg.enable_rsi_accel_filter,
                cfg.confidence_threshold,
                cfg.lockout_seconds,
                cfg.rr_ratio,
                cfg.trade_direction_is_long
            ),
            None => println!("[config] WARNING: could not parse {path} — using engine defaults"),
        }
    }

    println!(
        "[live-test] ticker={} feed={} db={} persist={} seed_bars={} loop={} window={:?}..{:?} limit={:?} wf_dir={:?}",
        ticker_lower, feed, db_path, persist, seed_bars, loop_feed, start_ms, end_ms, limit, wf_dir
    );

    let cfg = ExecutionConfig {
        ticker: ticker_lower.clone(),
        feed: ExchangeKind::Parquet,
        broker: ExchangeKind::Parquet, // paper replay: no real order routing
        mode: ExecutionMode::Paper, // the engine simulates fills; no real orders
        db_path: db_path.clone(),
        seed_bars,
        persist,
        strategy_config,
        walkforward_dir: wf_dir,
        meta_gate,
        report_scope: live_trade::report::ReportScope::default(),
    };
    let ctx = LoopContext {
        feed: create_replay_feed(
            &feed,
            start_ms,
            end_ms,
            limit,
            loop_feed,
            Some(Arc::clone(&stop)),
        ),
        broker: create_broker(ExchangeKind::Parquet),
        stats: Arc::clone(&stats),
        paused,
        stop: Arc::clone(&stop),
        capital_usd_cents: capital_cents,
    };

    // Run the identical loop as live-trade. The ReplayExchange flips `stop`
    // when the feed is exhausted, so the loop drains and `finish()` writes the
    // HTML report — unless Ctrl-C arrives first, which we honour by signalling
    // stop and letting the loop flush RocksDB cleanly.
    let run = run_execution_loop(cfg, ctx);
    tokio::pin!(run);
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl-C received — draining replay loop and closing RocksDB...");
            stop.store(true, Ordering::SeqCst);
            (&mut run).await;
        }
        _ = &mut run => {}
    }

    let s = stats.lock().unwrap();
    println!(
        "=== final: equity={:.2} trades={} (W:{}/L:{}) total_r={:.2} ===",
        s.equity, s.n_trades, s.n_wins, s.n_losses, s.total_r
    );
    if let Some(err) = &s.last_error {
        println!("last error: {err}");
    }
    if persist {
        println!(
            "[live-test] report -> build/pnl_report_{}.html",
            ticker_lower
        );
    }
}
