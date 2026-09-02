//! PyO3 bindings — the Python-visible `engine` module.
//!
//! Exposes a `Strategy` class that wraps the Rust execution stack:
//!
//!   * `engine` crate      — research-faithful strategy logic
//!   * `live-trade` crate  — exchange abstraction, execution loop, RocksDB
//!
//! Python is only the control plane (`reallocate` / `run` / `pause` /
//! `resume` / `stop` / `status`); every tick of the strategy runs on its own
//! tokio runtime thread inside the Rust process, so a Python `Portfolio` can
//! drive 50+ assets concurrently without Python ever touching market data.
//!
//! Build with `maturin develop --release` (enables the `python` feature).

#![cfg(feature = "python")]

use live_trade::config as strategy_config_loader;
use live_trade::exchange::{ExchangeKind, create_broker, create_feed};
use live_trade::execution::{
    ExecutionConfig, ExecutionMode, LoopContext, RunStats, run_execution_loop,
};
use live_trade::StrategyConfig;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn parse_mode(s: &str) -> PyResult<ExecutionMode> {
    match s.to_ascii_lowercase().as_str() {
        "paper" | "dry-run" | "dryrun" => Ok(ExecutionMode::Paper),
        "live" => Ok(ExecutionMode::Live),
        other => Err(PyValueError::new_err(format!(
            "unknown mode {other:?} (expected 'paper' or 'live')"
        ))),
    }
}

/// Control-plane handle to one Rust strategy runner.
///
/// ```python
/// s = engine.Strategy("btcusdt", exchange="bitfinex", mode="paper", capital=1000.0)
/// s.run()              # starts the Rust loop on its own thread
/// s.reallocate(5000.0) # resize position sizing
/// s.pause()            # stop generating new trades (feed still persists)
/// s.resume()
/// s.status()           # -> JSON string
/// s.stop()             # graceful shutdown + HTML report
///
/// # Read orderflow from one venue, execute on another:
/// s2 = engine.Strategy("btcusdt", exchange="bitfinex", feed="binance",
///                      mode="paper", capital=1000.0)
/// ```
/// The `feed` parameter (bybit | bitfinex | binance) supplies market data; it
/// defaults to `exchange`. Only `exchange` receives real orders (`Live` mode).
#[pyclass(module = "engine", name = "Strategy")]
pub struct Strategy {
    ticker: String,
    /// Order-routing venue (the `exchange` the strategy executes on).
    exchange: ExchangeKind,
    /// Market-data feed venue (can differ from `exchange` to read orderflow
    /// where the capital does not live).
    feed: ExchangeKind,
    mode: ExecutionMode,
    db_path: String,
    stats: Arc<Mutex<RunStats>>,
    paused: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    capital_usd_cents: Arc<AtomicU64>,
    started: Arc<AtomicBool>,
    handle: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    strategy_config: Option<StrategyConfig>,
}

#[pymethods]
impl Strategy {
    #[new]
        #[pyo3(signature = (ticker, exchange="bitfinex", mode="paper", capital=10_000.0, db_path=None, config_path=None, feed=None))]
    fn new(
        ticker: String,
        exchange: &str,
        mode: &str,
        capital: f64,
        db_path: Option<String>,
        config_path: Option<String>,
        feed: Option<String>,
    ) -> PyResult<Self> {
        let kind = ExchangeKind::parse(exchange).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown exchange {exchange:?} (expected 'bybit', 'bitfinex' or 'binance')"
            ))
        })?;
        // Feed venue defaults to the execution venue unless explicitly given,
        // so existing callers keep streaming from where they execute.
        let feed_kind = match feed {
            Some(f) => ExchangeKind::parse(&f).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown feed {f:?} (expected 'bybit', 'bitfinex' or 'binance')"
                ))
            })?,
            None => kind,
        };
        let mode = parse_mode(mode)?;
        if capital <= 0.0 {
            return Err(PyValueError::new_err("capital must be > 0"));
        }
        let ticker = ticker.to_ascii_lowercase();
        let db_path = db_path.unwrap_or_else(|| "./data/marketdata.rocksdb".to_string());
    let strategy_config = config_path
        .as_deref()
        .map(std::path::Path::new)
        .and_then(|p| strategy_config_loader::load_strategy_config(p, &ticker));
        Ok(Self {
            stats: Arc::new(Mutex::new(RunStats {
                capital_usd: capital,
                symbol: ticker.clone(),
                equity: capital,
                ..Default::default()
            })),
            ticker,
            exchange: kind,
            feed: feed_kind,
            mode,
            db_path,
            paused: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
            capital_usd_cents: Arc::new(AtomicU64::new((capital * 100.0) as u64)),
            started: Arc::new(AtomicBool::new(false)),
            handle: Arc::new(Mutex::new(None)),
        strategy_config,
        })
    }

    /// Set the capital (USD) allocated to this strategy for position sizing.
    /// While running, the loop re-reads it on every new entry, so an in-flight
    /// reallocation applies to the next position immediately.
    fn reallocate(&self, capital_usd: f64) -> PyResult<()> {
        if capital_usd <= 0.0 {
            return Err(PyValueError::new_err("capital must be > 0"));
        }
        self.capital_usd_cents
            .store((capital_usd * 100.0) as u64, Ordering::Relaxed);
        if let Ok(mut s) = self.stats.lock() {
            s.capital_usd = capital_usd;
        }
        Ok(())
    }

    /// Start the Rust execution loop on a dedicated thread (returns
    /// immediately). The loop owns the strategy + exchange and runs until
    /// `stop()` or a fatal stream error.
    fn run(&self) -> PyResult<()> {
        if self.started.swap(true, Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err("strategy already running"));
        }
        self.stop.store(false, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);

        let ticker = self.ticker.clone();
        let exchange_kind = self.exchange;
        let feed_kind = self.feed;
        let mode = self.mode;
        let db_path = self.db_path.clone();
        let stats = Arc::clone(&self.stats);
        let paused = Arc::clone(&self.paused);
        let stop = Arc::clone(&self.stop);
        let capital = Arc::clone(&self.capital_usd_cents);
        let strategy_config = self.strategy_config.clone();

        let handle = std::thread::Builder::new()
            .name(format!("strategy-{ticker}"))
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        if let Ok(mut s) = stats.lock() {
                            s.last_error = Some(format!("tokio runtime: {e}"));
                        }
                        return;
                    }
                };
                // Build the exchange (and its tokio::time resources) INSIDE the
                // runtime context: `BitfinexExchange::new` / bybit construct a
                // WS ping `interval`, which requires a current Tokio reactor.
                // Evaluating `create_exchange` as a plain arg to `rt.block_on`
                // runs it before the reactor is entered and panics with
                // "there is no reactor running".
                rt.block_on(async {
                    run_execution_loop(
                        ExecutionConfig {
                            ticker: ticker.clone(),
                            feed: feed_kind,
                            broker: exchange_kind,
                            mode,
                            db_path,
                            seed_bars: 150,
                            persist: true,
                            strategy_config,
                            walkforward_dir: None,
                            meta_gate: live_trade::execution::MetaGate::Auto,
                            report_scope: live_trade::report::ReportScope::default(),
                        },
                        LoopContext {
                            feed: create_feed(feed_kind),
                            broker: create_broker(exchange_kind),
                            stats,
                            paused,
                            stop,
                            capital_usd_cents: capital,
                        },
                    )
                    .await
                });
            })
            .map_err(|e| PyRuntimeError::new_err(format!("failed to spawn runner thread: {e}")))?;

        *self.handle.lock().unwrap() = Some(handle);
        Ok(())
    }

    /// Gracefully pause signal consumption: streams stay connected and raw
    /// market data keeps persisting, but no new entries/exits are generated.
    fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
        if let Ok(mut s) = self.stats.lock() {
            s.paused = true;
        }
    }

    /// Resume after `pause()`.
    fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
        if let Ok(mut s) = self.stats.lock() {
            s.paused = false;
        }
    }

    /// Hard stop: request loop shutdown, wait briefly for it to drain, and
    /// detach the thread if it needs longer (it exits on its next tick).
    fn stop(&self) -> PyResult<()> {
        self.stop.store(true, Ordering::SeqCst);
        self.started.store(false, Ordering::SeqCst);
        if let Ok(mut s) = self.stats.lock() {
            s.stop_requested = true;
        }

        let mut guard = self
            .handle
            .lock()
            .map_err(|_| PyRuntimeError::new_err("handle lock poisoned"))?;
        if let Some(handle) = guard.take() {
            // Give the loop up to ~5s to drain, then detach.
            for _ in 0..50 {
                if handle.is_finished() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
        Ok(())
    }

    /// JSON string snapshot of runtime state (equity, trades, capital, ...).
    fn status(&self) -> PyResult<String> {
        let s = self
            .stats
            .lock()
            .map_err(|_| PyRuntimeError::new_err("stats lock poisoned"))?;
        let json = serde_json::json!({
        "symbol": s.symbol,
        "exchange": self.exchange.as_str(),
        "mode": match self.mode {
            ExecutionMode::Paper => "paper",
            ExecutionMode::Live => "live",
        },
        "running": s.running,
        "paused": s.paused,
        "stop_requested": s.stop_requested,
        "capital_usd": s.capital_usd,
        "equity": s.equity,
        "n_trades": s.n_trades,
        "n_wins": s.n_wins,
        "n_losses": s.n_losses,
        "total_r": s.total_r,
        "mid": s.mid,
        "last_event": s.last_event,
        "last_error": s.last_error,
        "events_seen": s.events_seen,
        "rsi_2": s.rsi_2,
        "win_prob": s.win_prob,
        "gate": s.gate,
        "last_event_ts_ms": s.last_event_ts_ms,
        "last_event_latency_ms": s.last_event_latency_ms,
        "clock_skew_ms": s.clock_skew_ms,
        "n_outages": s.n_outages,
        "outage_active": s.outage_active,
        "max_staleness_ms": s.max_staleness_ms,
        "feed_staleness_ms": s.last_event_instant
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0),
        "loop_liveness_ms": s.last_feed_wakeup
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0),
        "loop_cadence_ms": s.loop_cadence_ms,
        "batch_process_ms": s.batch_process_ms,
            });
        Ok(json.to_string())
    }

    /// Whether the strategy loop is currently running (`started && !stop`).
    #[getter]
    fn is_running(&self) -> bool {
        self.started.load(Ordering::SeqCst) && !self.stop.load(Ordering::SeqCst)
    }
}

/// The Python-visible module: `import engine`.
#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Strategy>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
