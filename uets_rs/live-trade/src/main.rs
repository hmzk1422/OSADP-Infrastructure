//! live-trade CLI binary.
//!
//! Thin wrapper over [`live_trade::execution`]: parses ticker / exchange /
//! mode / capital and runs the strategy loop until Ctrl+C.
//!
//!     cargo run -p live-trade -- btcusdt --exchange bitfinex --mode paper
//!     cargo run -p live-trade -- ethusdt --exchange bybit --mode paper --capital 2000

use live_trade::exchange::{ExchangeKind, create_broker, create_feed};
use live_trade::execution::{
    ExecutionConfig, ExecutionMode, LoopContext, RunStats, run_execution_loop,
};
use live_trade::report::ReportScope;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn usage<T>() -> T {
    eprintln!(
        "usage: live-trade [ticker] [--exchange bybit|bitfinex] [--feed bybit|bitfinex|binance] [--mode paper|live] [--capital USD] [--db PATH] [--no-db] [--seed N] [--config PATH] [--control PATH] [--session latest|all] [--meta-gate auto|on|off] [--report-tag ID] [--fresh-equity]"
    );
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    let mut ticker = "btcusdt".to_string();
    let mut exchange = ExchangeKind::Bitfinex;
    // Market-data venue; defaults to the execution venue unless overridden. Use
    // `--feed bybit|binance` to stream orderflow from one venue while routing
    // orders to another (`--exchange`) that holds your capital.
    let mut feed = ExchangeKind::Bitfinex;
    let mut mode = ExecutionMode::Paper;
    let mut capital = 10_000.0f64;
    let mut db_path = "./data/marketdata.rocksdb".to_string();
    let mut persist = true;
    let mut seed_bars = 150u32;
    let mut config_path: Option<String> = None;
    let mut control_path: Option<String> = None;
    let mut report_scope = ReportScope::Latest;
    let mut fresh_equity = false;               // --fresh-equity: skip RocksDB equity restore
    let mut meta_gate = live_trade::execution::MetaGate::Auto;
    // --report-tag <asset id>: the portfolio supervisor passes each asset's
    // unique id so two twins with the SAME ticker (e.g. eth-rec / eth-exec)
    // never clobber each other's HTML report file.
    let mut report_tag: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--exchange" => {
                let v = args.next().unwrap_or_else(usage);
                exchange = ExchangeKind::parse(&v).unwrap_or_else(usage);
            }
            "--feed" => {
                let v = args.next().unwrap_or_else(usage);
                feed = ExchangeKind::parse(&v).unwrap_or_else(usage);
            }
            "--mode" => {
                let v = args.next().unwrap_or_else(usage);
                mode = match v.to_ascii_lowercase().as_str() {
                    "paper" | "dry-run" => ExecutionMode::Paper,
                    "live" => ExecutionMode::Live,
                    _ => usage(),
                };
            }
            "--capital" => {
                capital = args
                    .next()
                    .unwrap_or_else(usage)
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--db" => db_path = args.next().unwrap_or_else(usage),
            "--no-db" => persist = false,
            "--seed" => {
                seed_bars = args
                    .next()
                    .unwrap_or_else(usage)
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--config" => config_path = Some(args.next().unwrap_or_else(usage)),
            "--control" => control_path = Some(args.next().unwrap_or_else(usage)),
            "--fresh-equity" => fresh_equity = true,
            "--meta-gate" => {
                let v = args.next().unwrap_or_else(usage).to_ascii_lowercase();
                meta_gate = match v.as_str() {
                    "auto" => live_trade::execution::MetaGate::Auto,
                    "on" => live_trade::execution::MetaGate::On,
                    "off" => live_trade::execution::MetaGate::Off,
                    _ => usage(),
                };
            }
            "--report-tag" => report_tag = Some(args.next().unwrap_or_else(usage)),
            "--session" => {
                let v = args.next().unwrap_or_else(usage).to_ascii_lowercase();
                report_scope = match v.as_str() {
                    "all" | "all-time" | "alltime" => ReportScope::All,
                    "latest" | "session" | "current" => ReportScope::Latest,
                    _ => usage(),
                };
            }
            "--help" | "-h" => usage(),
            other if other.starts_with("--") => usage(),
            other => ticker = other.to_ascii_lowercase(),
        }
    }

    let stats = Arc::new(Mutex::new(RunStats {
        symbol: ticker.clone(),
        capital_usd: capital,
        ..Default::default()
    }));
    let paused = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
        let capital_cents = Arc::new(AtomicU64::new((capital * 100.0) as u64));

    let strategy_config = config_path
        .as_deref()
        .map(std::path::Path::new)
        .and_then(|p| live_trade::config::load_strategy_config(p, &ticker));

    let cfg = ExecutionConfig {
        ticker: ticker.clone(),
        feed,
        broker: exchange,
        mode,
        db_path,
        seed_bars,
        persist,
        strategy_config,
        walkforward_dir: None,
        meta_gate,
        report_scope,
        control_path,
        fresh_equity,
        report_tag,
    };
    let ctx = LoopContext {
        feed: create_feed(feed),
        broker: create_broker(exchange),
        stats: Arc::clone(&stats),
        paused,
        stop: Arc::clone(&stop),
        capital_usd_cents: capital_cents,
    };

    // Run the loop until Ctrl-C, then let it drain fully so the RocksDB store
    // is flushed and closed before the process exits (a hard kill here is what
    // corrupts CURRENT/MANIFEST).
    let run = run_execution_loop(cfg, ctx);
    tokio::pin!(run);
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl-C received — draining loop and closing RocksDB...");
            stop.store(true, Ordering::SeqCst);
            (&mut run).await;
        }
        _ = &mut run => {}
    }

    let s = stats.lock().unwrap();
    println!(
        "=== final: equity={:.2} trades={} (W:{}/L:{}) total_r={:.2} outages={} max_stal={:.0}ms ===",
        s.equity, s.n_trades, s.n_wins, s.n_losses, s.total_r, s.n_outages, s.max_staleness_ms
    );
    if let Some(err) = &s.last_error {
        println!("last error: {err}");
    }
}
