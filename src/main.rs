//! Production Rust executor binary.
//!
//! Spawns the Python CCXT sidecar (`tools/ccxt_sidecar.py`) as a subprocess,
//! connects to its local TCP feed, warms up the feature engine with the
//! historical seed, then — every wall second — computes the 15-feature vector,
//! runs ONNX inference for P(WIN), and applies the research entry/exit gates.
//!
//! Run:  cargo run --bin executor -- config.toml

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Context;
use chrono::Utc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::time::MissedTickBehavior;

use intraday_momentum_composite::exchange::FeedMsg;
use intraday_momentum_composite::{
    Action, Config, FeatureEngine, OnnxModel, StrategyExecutor,
};

/// Cross-task state: the feature engine (mutated by the feed reader) and the
/// strategy/gatekeeper (driven by the 1-second clock). The mutex serializes
/// bar updates between the two so the clock never trips on a half-built bar.
struct SharedState {
    engine: FeatureEngine,
    strategy: StrategyExecutor,
    /// Flipped once `seed_done` is received — the clock stays idle until then.
    seeded: bool,
}

impl SharedState {
    /// Bridge the disjoint borrow (`&engine` + `&mut strategy`) so the borrow
    /// checker is happy through a `MutexGuard`.
    fn clock_tick(&mut self, now: chrono::DateTime<Utc>) -> Action {
        self.strategy.on_clock_tick(&self.engine, now)
    }
}

/// Kills the sidecar subprocess when dropped (i.e. on program exit).
struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn_sidecar(cfg: &Config) -> anyhow::Result<ChildGuard> {
    let child = std::process::Command::new(&cfg.sidecar_python)
        .arg("tools/ccxt_sidecar.py")
        .arg("--symbol")
        .arg(&cfg.symbol)
        .arg("--host")
        .arg(&cfg.sidecar_host)
        .arg("--port")
        .arg(cfg.sidecar_port.to_string())
        .arg("--seed-bars")
        .arg(cfg.seed_bars.to_string())
        .arg("--seed-interval")
        .arg(&cfg.seed_interval)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .with_context(|| {
            format!(
                "failed to spawn {} (is ccxt installed?)",
                cfg.sidecar_python
            )
        })?;
    log::info!("spawned ccxt sidecar pid={}", child.id());
    Ok(ChildGuard(child))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    let config_path =
        std::env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    let cfg = Config::from_file(&config_path)
        .with_context(|| format!("reading config {config_path}"))?;
    log::info!(
        "config loaded - symbol={} direction={} conf>={}",
        cfg.symbol,
        cfg.trade_direction,
        cfg.confidence_threshold
    );

    let _sidecar = spawn_sidecar(&cfg)?;

    let addr = format!("{}:{}", cfg.sidecar_host, cfg.sidecar_port);
    log::info!("connecting to sidecar feed at {addr} ...");
    let stream = retry_connect(&addr).await?;
    log::info!("connected; awaiting seed...");
    let mut lines = BufReader::new(stream).lines();

    let engine = FeatureEngine::new(cfg.clone());
    let model = OnnxModel::load(&cfg.model_path, &cfg.model_manifest_path)
        .context("loading ONNX model + feature manifest")?;
    log::info!("model ready: {} input features", model.feature_names().len());
    let strategy = StrategyExecutor::new(cfg.clone(), model);

    let state = std::sync::Arc::new(Mutex::new(SharedState {
        engine,
        strategy,
        seeded: false,
    }));

    // ---- 1-second decision clock ----------------------------------------
    let state_clock = state.clone();
    let clock_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let now = Utc::now();
            let mut g = state_clock.lock().await;
            if !g.seeded {
                continue;
            }
            let action = g.clock_tick(now);
            match action {
                Action::Wait | Action::Debounced | Action::RsiFiltered
                | Action::InPosition | Action::LowConfidence => {}
                other => log::info!("clock tick: {other}"),
            }
            // ---- Feed loop (trades + orderbook) -----------------------------------
    let running = std::sync::Arc::new(AtomicBool::new(true));
    let running_feed = running.clone();
    let state_feed = state.clone();
    let feed_handle = tokio::spawn(async move {
        while running_feed.load(Ordering::Relaxed) {
            let line = match lines.next_line().await {
                Ok(Some(l)) if !l.trim().is_empty() => l,
                Ok(Some(_)) => continue,
                Ok(None) => break,
                Err(e) => {
                    log::error!("feed read error: {e}");
                    break;
                }
            };
            let msg: FeedMsg = match serde_json::from_str(&line) {
                Ok(m) => m,
                Err(e) => {
                    let preview = &line[..line.len().min(120)];
                    log::warn!("bad feed JSON: {e}; raw={preview}");
                    continue;
                }
            };
            let mut g = state_feed.lock().await;
            match msg {
                FeedMsg::Hello(h) => log::info!(
                    "sidecar hello: symbol={} exchange={} pro={}",
                    h.symbol,
                    h.exchange,
                    h.has_pro
                ),
                FeedMsg::Seed(s) => {
                    g.engine.seed_bar_series(&s.closes);
                    log::info!("seed: {} historical closes", s.bars);
                }
                FeedMsg::SeedDone { bars } => {
                    g.seeded = true;
                    log::info!("seeding complete: {bars} bars - live trading");
                }
                FeedMsg::Trade(t) => {
                    g.engine.on_trade(t.ts_ms, t.price, t.amount, &t.side);
                    // Position exit check on every trade tick (research on_trade).
                    if let Some(action) = g.strategy.on_trade(t.price, Utc::now()) {
                        log::info!("position exit: {action}");
                    }
                }
                FeedMsg::Orderbook(o) => {
                    if let (Some((bp, ba)), Some((ap, aa))) =
                        (o.bids.first(), o.asks.first())
                    {
                        g.engine.on_orderbook(o.ts_ms, *bp, *ba, *ap, *aa);
                    }
                }
            }
        }
        log::warn!("feed loop ended");
    });

        // Keep the process alive until ctrl-c; either task exiting is fatal.
    // We borrow the handles (`&mut`) in select! so we still own them to abort
    // and join cleanly on shutdown.
    tokio::select! {
        r = &mut clock_handle => {
            if r.is_ok() { log::error!("clock task exited unexpectedly"); }
        }
        r = &mut feed_handle => {
            if r.is_ok() { log::error!("feed task exited unexpectedly"); }
        }
        _ = tokio::signal::ctrl_c() => {
            log::info!("ctrl-c received, shutting down");
        }
    }
    running.store(false, Ordering::Relaxed);
    // Abort both tasks and wait for them to wind down (max 2s). Dropping
    // `state` and `ChildGuard` finalizes the engine + kills the sidecar.
    clock_handle.abort();
    feed_handle.abort();
    let _ = tokio::time::timeout(
        Duration::from_secs(2),
        async { let _ = tokio::join!(clock_handle, feed_handle); },
    )
    .await;
    Ok(())
}


