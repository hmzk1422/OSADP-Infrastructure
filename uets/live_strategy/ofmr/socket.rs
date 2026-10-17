//! OFMR socket consumer — IPC Unix socket (`sockets/strategies/ofmr.sock`).
//!
//! Framing: `u32` LE length prefix + payload (max 1 MB). Each frame carries a
//! per-symbol tick row; we decode it into a `FeatureRow`, run `compute` +
//! `decide`, then submit resulting orders through the USD-M Futures connector.
//!
//! On win32 (`cfg(not(unix))`) this is a compile-time stub so the crate
//! builds everywhere; the live loop only runs on Unix.

#[cfg(any(unix, test))]
use std::path::Path;

#[cfg(any(unix, test))]
use crate::core::feature::FeatureRow;
#[cfg(unix)]
use crate::core::strategy::{Decision, Position};
#[cfg(unix)]
use crate::connectors::binance::BinanceFuturesTrade;
#[cfg(unix)]
use crate::core::order::{OrderRequest, Side};
#[cfg(unix)]
use crate::exchange::ExecutionConnector;
#[cfg(unix)]
use crate::live_strategy::ofmr::features::OfmrFeatures;
#[cfg(unix)]
use crate::live_strategy::ofmr::{ComputeFeatures, StateBuilder};
#[cfg(unix)]
use crate::live_strategy::ofmr::settings::Settings;
#[cfg(unix)]
use crate::live_strategy::ofmr::strategy::OfmrStrategy;
#[cfg(unix)]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
const MAX_FRAME: usize = 1_048_576;

/// Decode one IPC frame into a `FeatureRow`. Frame body is expected to be a
/// simple `name:value` line protocol, one feature per line.
#[cfg(any(unix, test))]
fn decode_frame(payload: &[u8]) -> Result<FeatureRow, String> {
    let text =
        std::str::from_utf8(payload).map_err(|e| format!("frame payload is not UTF-8: {e}"))?;
    let mut row = FeatureRow::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((name, val)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        let val = val.trim();
        if let Ok(v) = val.parse::<f64>() {
            row.insert(name, v);
        } else {
            row.insert_missing(name);
        }
    }
    Ok(row)
}

#[cfg(unix)]
struct OrderExecutor {
    connector: Option<BinanceFuturesTrade>,
    mode: String,
    max_position_usd: f64,
    min_order_size: f64,
    quantity: f64,
    sequence: u64,
}

#[cfg(any(unix, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionMode {
    Paper,
    Testnet,
    Live,
}

#[cfg(any(unix, test))]
struct ExecutionConfig {
    mode: ExecutionMode,
    allow_live_trading: bool,
}

#[cfg(any(unix, test))]
impl ExecutionConfig {
    fn load() -> Result<Self, String> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/execution.yaml");
        let content = std::fs::read_to_string(&path)
            .map_err(|error| format!("read execution config {}: {error}", path.display()))?;
        Self::parse(&content)
    }

    fn parse(content: &str) -> Result<Self, String> {
        let mut mode = None;
        let mut allow_live_trading = None;
        let mut in_execution = false;
        for (line_number, raw_line) in content.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            if line.ends_with(':')
                && raw_line
                    .chars()
                    .next()
                    .is_some_and(|character| !character.is_whitespace())
            {
                in_execution = line == "execution:";
                continue;
            }
            if !in_execution {
                continue;
            }
            let Some((key, value)) = line.split_once(':') else {
                return Err(format!("invalid execution config at line {}", line_number + 1));
            };
            let value = value.trim().trim_matches(['\"', '\'']);
            match key.trim() {
                "mode" => {
                    mode = Some(match value {
                        "paper" => ExecutionMode::Paper,
                        "testnet" => ExecutionMode::Testnet,
                        "live" => ExecutionMode::Live,
                        _ => return Err(format!("invalid execution mode {value:?}")),
                    });
                }
                "allow_live_trading" => {
                    allow_live_trading = Some(value.parse::<bool>().map_err(|_| {
                        format!("allow_live_trading must be true or false, got {value:?}")
                    })?);
                }
                other => return Err(format!("unknown execution config key {other:?}")),
            }
        }
        Ok(Self {
            mode: mode.ok_or_else(|| "execution.mode is required".to_string())?,
            allow_live_trading: allow_live_trading.unwrap_or(false),
        })
    }
}

#[cfg(unix)]
impl OrderExecutor {
    fn from_env(settings: &Settings) -> Result<Self, String> {
        let config = ExecutionConfig::load()?;
        let api_key = std::env::var("BINANCE_API_KEY").ok();
        let secret_key = std::env::var("BINANCE_API_SECRET").ok();
        let mode = match config.mode {
            ExecutionMode::Paper => "paper",
            ExecutionMode::Testnet => "testnet",
            ExecutionMode::Live => "live",
        }
        .to_string();
        let connector = match config.mode {
            ExecutionMode::Paper => None,
            ExecutionMode::Testnet => {
                let (Some(api_key), Some(secret_key)) = (api_key, secret_key) else {
                    return Err("testnet execution requires BINANCE_API_KEY and BINANCE_API_SECRET".into());
                };
                Some(BinanceFuturesTrade::new(Some(api_key), Some(secret_key)))
            }
            ExecutionMode::Live => {
                if !config.allow_live_trading {
                    return Err("live execution requires allow_live_trading: true in uets/config/execution.yaml".into());
                }
                let (Some(api_key), Some(secret_key)) = (api_key, secret_key) else {
                    return Err("live execution requires BINANCE_API_KEY and BINANCE_API_SECRET".into());
                };
                Some(BinanceFuturesTrade::production(Some(api_key), Some(secret_key)))
            }
        };
        if !settings.trade.max_position_usd.is_finite()
            || settings.trade.max_position_usd <= 0.0
            || !settings.trade.min_order_size.is_finite()
            || settings.trade.min_order_size <= 0.0
        {
            return Err("OFMR max_position_usd and min_order_size must be positive finite values".into());
        }
        Ok(Self {
            connector,
            mode,
            max_position_usd: settings.trade.max_position_usd,
            min_order_size: settings.trade.min_order_size,
            quantity: 0.0,
            sequence: 0,
        })
    }

    fn act(&mut self, symbol: &str, decision: Decision, position: &mut Position, price: f64) -> Result<(), String> {
        if decision == Decision::Hold {
            return Ok(());
        }
        self.sequence = self.sequence.saturating_add(1);
        let client_id = format!("uets-{}-{}", std::process::id(), self.sequence);
        let request = match decision {
            Decision::EnterLong | Decision::EnterShort => {
                if *position != Position::Flat {
                    return Ok(());
                }
                let qty = quantity_for_capital(self.max_position_usd, price, self.min_order_size)
                    .ok_or_else(|| format!("cannot size {symbol} order within ${} capital and minimum quantity {}", self.max_position_usd, self.min_order_size))?;
                let side = if decision == Decision::EnterLong { Side::Buy } else { Side::Sell };
                OrderRequest::market(symbol, side, qty).with_client_order_id(client_id)
            }
            Decision::Exit => {
                if *position == Position::Flat || self.quantity <= 0.0 {
                    return Ok(());
                }
                let side = if *position == Position::Long { Side::Sell } else { Side::Buy };
                OrderRequest::market(symbol, side, self.quantity)
                    .reducing()
                    .with_client_order_id(client_id)
            }
            Decision::Hold => return Ok(()),
        };
        if let Some(connector) = self.connector.as_mut() {
            let fills = connector.submit(&request).map_err(|error| format!("{} order {} failed: {error}", self.mode, request.client_order_id.as_deref().unwrap_or("unknown")))?;
            let filled_qty: f64 = fills.iter().map(|fill| fill.qty).sum();
            if filled_qty <= 0.0 || !filled_qty.is_finite() {
                return Err(format!("order {} returned no valid fills", request.client_order_id.as_deref().unwrap_or("unknown")));
            }
            println!("OFMR routed mode={} symbol={} side={:?} qty={} fills={} fees={}", self.mode, symbol, request.side, filled_qty, fills.len(), fills.iter().map(|fill| fill.fee).sum::<f64>());
            if matches!(decision, Decision::EnterLong | Decision::EnterShort) {
                self.quantity = filled_qty;
                *position = if decision == Decision::EnterLong { Position::Long } else { Position::Short };
            } else {
                self.quantity = 0.0;
                *position = Position::Flat;
            }
        } else {
            println!("OFMR simulated mode=paper symbol={symbol} decision={decision:?} qty={}", request.qty);
            if matches!(decision, Decision::EnterLong | Decision::EnterShort) {
                self.quantity = request.qty;
                *position = if decision == Decision::EnterLong { Position::Long } else { Position::Short };
            } else {
                self.quantity = 0.0;
                *position = Position::Flat;
            }
        }
        Ok(())
    }
}

#[cfg(any(unix, test))]
fn quantity_for_capital(max_position_usd: f64, price: f64, minimum: f64) -> Option<f64> {
    if !price.is_finite() || price <= 0.0 || !minimum.is_finite() || minimum <= 0.0 {
        return None;
    }
    let decimals = minimum.to_string().split('.').nth(1).map_or(0, str::len).min(12);
    let scale = 10_f64.powi(decimals as i32);
    let qty = (max_position_usd / price * scale).floor() / scale;
    (qty >= minimum && qty.is_finite()).then_some(qty)
}

/// Run the OFMR socket consumer loop (Unix only).
#[cfg(unix)]
pub fn run(socket_path: &str, symbol: &str) -> Result<(), String> {
    use std::io::Read;
    use std::os::unix::net::UnixListener;
    use std::time::Instant;

    if let Some(parent) = Path::new(socket_path).parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create socket directory {}: {e}", parent.display()))?;
    }

    // Claim the socket path (reuse aggregator-style stale-socket cleanup).
    let claim = crate::aggregator::cleanup::claim(Path::new(socket_path))
        .map_err(|e| format!("claim socket {socket_path}: {e}"))?;
    match claim {
        crate::aggregator::cleanup::Claim::OwnedByLive(_pid) => {
            return Err(format!("socket {socket_path} is owned by a live process"));
        }
        crate::aggregator::cleanup::Claim::Free => {}
    }

    let listener =
        UnixListener::bind(socket_path).map_err(|e| format!("bind {socket_path}: {e}"))?;

    let mut features = OfmrFeatures::new();
    let mut strategy = OfmrStrategy::new();
    let settings = Settings::load();
    let mut executor = OrderExecutor::from_env(&settings)?;
    let mut telemetry = StrategyTelemetry::new(symbol);

    let mut position = Position::Flat;

    for stream in listener.incoming() {
        let mut stream = stream.map_err(|e| format!("accept: {e}"))?;
        let mut len_buf = [0u8; 4];
        loop {
            // Read 4-byte LE length prefix.
            let mut got = 0usize;
            while got < 4 {
                let n = stream
                    .read(&mut len_buf[got..])
                    .map_err(|e| format!("read length: {e}"))?;
                if n == 0 {
                    break; // peer closed
                }
                got += n;
            }
            if got < 4 {
                break;
            }
            let frame_len = u32::from_le_bytes(len_buf) as usize;
            if frame_len > MAX_FRAME {
                return Err(format!("frame too large: {frame_len} > {MAX_FRAME}"));
            }

            // Read payload.
            let mut payload = vec![0u8; frame_len];
            let mut got = 0usize;
            while got < frame_len {
                let n = stream
                    .read(&mut payload[got..])
                    .map_err(|e| format!("read payload: {e}"))?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            if got < frame_len {
                break;
            }

            // Decode -> compute -> decide -> execute.
            let started = Instant::now();
            let row = decode_frame(&payload)?;
            let computed = features
                .compute(&row)
                .map_err(|e| format!("compute: {e}"))?;
            let state = crate::core::strategy::MarketState {
                features: computed,
                position,
                sequence: 0,
            };
            let decision = strategy.decide(&state);
            let price = row.get("price").or_else(|| row.get("bid_price").zip(row.get("ask_price")).map(|(bid, ask)| (bid + ask) / 2.0)).unwrap_or_default();
            executor.act(symbol, decision, &mut position, price)?;
            let event_time_ms = row.get("event_time_ms").map(|value| value as u64);
            let processing_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            telemetry.publish(event_time_ms, format!("{decision:?}"), Some(processing_ns));
        }
    }

    Ok(())
}

#[cfg(unix)]
struct StrategyTelemetry {
    socket: Option<std::os::unix::net::UnixDatagram>,
    symbol: String,
    latency: crate::aggregator::latency::LatencyHist,
    clock_skew: crate::core::telemetry::ClockSkew,
}

#[cfg(unix)]
impl StrategyTelemetry {
    fn new(symbol: &str) -> Self {
        use std::os::unix::net::UnixDatagram;

        let socket_path =
            std::env::var("UETS_GUI_SOCKET").unwrap_or_else(|_| "sockets/gui.sock".to_string());
        let socket = UnixDatagram::unbound().and_then(|socket| {
            socket.set_nonblocking(true)?;
            socket.connect(&socket_path)?;
            Ok(socket)
        });
        let socket = match socket {
            Ok(socket) => Some(socket),
            Err(error) => {
                eprintln!("[telemetry] GUI unavailable at {socket_path}: {error}");
                None
            }
        };
        let mut publisher = Self {
            socket,
            symbol: symbol.to_ascii_lowercase(),
            latency: crate::aggregator::latency::LatencyHist::new(),
            clock_skew: crate::core::telemetry::ClockSkew::new(),
        };
        publisher.publish(None, "enabled".to_string(), None);
        publisher
    }

    fn publish(
        &mut self,
        event_time_ms: Option<u64>,
        decision: String,
        processing_ns: Option<u64>,
    ) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or_default();
        if let Some(event_time_ms) = event_time_ms {
            self.clock_skew.record(event_time_ms, now_ms);
        }
        if let Some(processing_ns) = processing_ns {
            self.latency.observe_ns(processing_ns);
        }
        let skew = self.clock_skew.snapshot();
        let memory_mb = process_memory_mb();
        let statistics = serde_json::json!({
            "status": "enabled",
            "latest_market_data": event_time_ms.map(|time| time.to_string()),
            "latest_decision": decision,
            "processing_latency_ms": processing_ns.map(|value| value as f64 / 1_000_000.0),
            "exchange_latency_ms": event_time_ms.map(|event_time| (now_ms as i64 - event_time as i64) as f64),
            "latency_p05_ms": self.latency.percentile(0.05).map(|value| value as f64 / 1_000_000.0),
            "latency_p95_ms": self.latency.percentile(0.95).map(|value| value as f64 / 1_000_000.0),
            "latency_p99_ms": self.latency.percentile(0.99).map(|value| value as f64 / 1_000_000.0),
            "clock_skew_mean_ms": skew.mean_us.map(|value| value / 1_000.0),
            "clock_skew_max_ms": skew.max_abs_us.map(|value| value as f64 / 1_000.0),
            "memory_mb": memory_mb,
        });
        let snapshot = serde_json::json!({
            "strategy_id": "ofmr",
            "asset_id": self.symbol,
            "statistics": statistics,
            "equity_history": [],
        });
        if let (Some(socket), Ok(payload)) = (&self.socket, serde_json::to_vec(&snapshot)) {
            if let Err(error) = socket.send(&payload) {
                if error.kind() != std::io::ErrorKind::WouldBlock {
                    eprintln!("[telemetry] GUI update failed: {error}");
                }
            }
        }
    }
}

#[cfg(all(unix, target_os = "linux"))]
fn process_memory_mb() -> Option<f64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(resident_pages as f64 * 4096.0 / 1_000_000.0)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn process_memory_mb() -> Option<f64> {
    None
}

/// Stub on non-Unix platforms: nothing to run.
#[cfg(not(unix))]
pub fn run(_socket_path: &str, _symbol: &str) -> Result<(), String> {
    Err("OFMR socket consumer requires a Unix platform".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_frame_parses_name_value_lines() {
        let payload = b"price:100.5\namount:1.25\nbid_price:100.49\n";
        let row = decode_frame(payload).unwrap();
        assert_eq!(row.get("price"), Some(100.5));
        assert_eq!(row.get("amount"), Some(1.25));
        assert_eq!(row.get("bid_price"), Some(100.49));
    }

    #[test]
    fn decode_frame_missing_on_bad_value() {
        let payload = b"price:not_a_number\n";
        let row = decode_frame(payload).unwrap();
        assert!(row.get("price").is_none());
        assert!(row.get_value("price").unwrap().is_missing());
    }

    #[test]
    fn order_quantity_never_exceeds_allocated_notional() {
        let quantity = quantity_for_capital(100.0, 30_000.0, 0.001).unwrap();
        assert_eq!(quantity, 0.003);
        assert!(quantity * 30_000.0 <= 100.0);
        assert!(quantity_for_capital(100.0, 200_000.0, 0.001).is_none());
    }

    #[test]
    fn execution_config_requires_mode_and_defaults_live_guard_to_false() {
        let config = ExecutionConfig::parse("execution:\n  mode: paper\n").unwrap();
        assert_eq!(config.mode, ExecutionMode::Paper);
        assert!(!config.allow_live_trading);
        assert!(ExecutionConfig::parse("execution:\n  mode: live\n").is_ok());
        assert!(ExecutionConfig::parse("execution:\n  mode: unsafe\n").is_err());
    }
}
