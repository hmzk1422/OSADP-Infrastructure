//! uets live strategy OFMR socket consumer
//!
//! This module handles the IPC Unix socket communication for the OFMR strategy.
//! It supports playback mode (ofmr@pwf, ofmr@mc, ofmr@cpcv) where tick data
//! is received via Unix domain socket from parquet data instead of connecting
//! to the live Binance exchange.
//!,
//! Framing: `u32` LE length prefix + payload (max 1 MB). Each frame carries a
//! per-symbol tick row; we decode it into a `FeatureRow`, run `compute` +
//! `decide`, then submit resulting orders through the USD-M Futures connector.
//!,
//! On win32 (`cfg(not(unix))`) this is a compile-time stub so the crate
//! builds everywhere; the live loop only runs on Unix.
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

#[cfg(unix)]
use crate::connectors::binance::BinanceFuturesTrade;
#[cfg(any(unix, test))]
use crate::core::feature::FeatureRow;
#[cfg(unix)]
use crate::core::order::OrderRequest;
#[cfg(any(unix, test))]
use crate::core::order::{Fill, Side};
#[cfg(any(unix, test))]
use crate::core::pnl::Trade;
#[cfg(any(unix, test))]
use crate::core::portfolio::Portfolio;
#[cfg(any(unix, test))]
use crate::core::stats::PerformanceStats;
#[cfg(unix)]
use crate::core::strategy::{Decision, Position};
#[cfg(unix)]
use crate::exchange::ExecutionConnector;
#[cfg(unix)]
use crate::live_strategy::ofmr::features::OfmrFeatures;
#[cfg(unix)]
use crate::live_strategy::ofmr::settings::Settings;
#[cfg(unix)]
use crate::live_strategy::ofmr::strategy::OfmrStrategy;
#[cfg(unix)]
use crate::live_strategy::ofmr::{ComputeFeatures, StateBuilder};
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
    last_decision_to_order_ms: Option<f64>,
    last_order_to_fill_ms: Option<f64>,
}

#[cfg(any(unix, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionMode {
    LiveTest,
    Testnet,
    LiveTrade,
}

#[cfg(any(unix, test))]
struct ExecutionConfig {
    mode: ExecutionMode,
    allow_live_trading: bool,
}

#[cfg(any(unix, test))]
impl ExecutionConfig {
    #[allow(dead_code)]
    fn load() -> Result<Self, String> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/execution.yaml");
        let content = std::fs::read_to_string(&path)
            .map_err(|error| format!("read execution config {}: {error}", path.display()))?;
        Self::parse(&content)
    }

    fn parse(content: &str) -> Result<Self, String> {
        let mut mode = None;
        let mut allow_live_trading = None;
        let mut section = "";
        for (line_number, raw_line) in content.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let indented = raw_line.chars().next().is_some_and(char::is_whitespace);
            if !indented && line.ends_with(':') {
                section = match line {
                    "execution:" => "execution",
                    "playback:" => "settings",
                    _ => "",
                };
                continue;
            }
            if section == "settings" {
                continue;
            }
            if indented && section.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once(':') else {
                return Err(format!(
                    "invalid execution config at line {}",
                    line_number + 1
                ));
            };
            let value = value.trim().trim_matches(['\"', '\'']);
            match key.trim() {
                "mode" => {
                    mode = Some(match value {
                        "live-test" => ExecutionMode::LiveTest,
                        "testnet" => ExecutionMode::Testnet,
                        "live-trade" => ExecutionMode::LiveTrade,
                        _ => return Err(format!("invalid execution mode {value:?}")),
                    });
                }
                "allow_live_trading" => {
                    allow_live_trading = Some(value.parse::<bool>().map_err(|_| {
                        format!("allow_live_trading must be true or false, got {value:?}")
                    })?);
                }
                other if section == "execution" => {
                    return Err(format!("unknown execution config key {other:?}"));
                }
                _ => {}
            }
        }
        Ok(Self {
            mode: mode.ok_or_else(|| "mode is required".to_string())?,
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

        // Detect playback strategies: ofmr@pwf, ofmr@mc, ofmr@cpcv
        let strategy_name = std::env::var("UETS_STRATEGY_NAME").unwrap_or_default();
        let is_playback = strategy_name.starts_with("ofmr@")
            && matches!(
                strategy_name.split_once('@').map(|(_, selector)| selector),
                Some("pwf") | Some("mc") | Some("cpcv")
            );

        let mode = if is_playback {
            "playback".to_string()
        } else {
            match config.mode {
                ExecutionMode::LiveTest => "live-test",
                ExecutionMode::Testnet => "testnet",
                ExecutionMode::LiveTrade => "live-trade",
            }
            .to_string()
        };
        let connector = if is_playback {
            // Playback mode: no Binance connector needed.
            // Data flows through Unix domain socket from local parquet files.
            None
        } else {
            match config.mode {
                ExecutionMode::LiveTest => None,
                ExecutionMode::Testnet => {
                    let (Some(api_key), Some(secret_key)) = (api_key, secret_key) else {
                        return Err(
                            "testnet execution requires BINANCE_API_KEY and BINANCE_API_SECRET"
                                .into(),
                        );
                    };
                    Some(BinanceFuturesTrade::new(Some(api_key), Some(secret_key)))
                }
                ExecutionMode::LiveTrade => {
                    if !config.allow_live_trading {
                        return Err("live-trade execution requires allow_live_trading: true in uets/config/execution.yaml".into());
                    }
                    let (Some(api_key), Some(secret_key)) = (api_key, secret_key) else {
                        return Err(
                            "live execution requires BINANCE_API_KEY and BINANCE_API_SECRET".into(),
                        );
                    };
                    Some(BinanceFuturesTrade::production(
                        Some(api_key),
                        Some(secret_key),
                    ))
                }
            }
        };
        if !settings.trade.max_position_usd.is_finite()
            || settings.trade.max_position_usd <= 0.0
            || !settings.trade.min_order_size.is_finite()
            || settings.trade.min_order_size <= 0.0
        {
            return Err(
                "OFMR max_position_usd and min_order_size must be positive finite values".into(),
            );
        }
        Ok(Self {
            connector,
            mode,
            max_position_usd: settings.trade.max_position_usd,
            min_order_size: settings.trade.min_order_size,
            quantity: 0.0,
            sequence: 0,
            last_decision_to_order_ms: None,
            last_order_to_fill_ms: None,
        })
    }

    fn act(
        &mut self,
        symbol: &str,
        decision: Decision,
        position: &mut Position,
        price: f64,
        bid_price: Option<f64>,
        ask_price: Option<f64>,
    ) -> Result<Vec<Fill>, String> {
        self.last_decision_to_order_ms = None;
        self.last_order_to_fill_ms = None;
        if decision == Decision::Hold {
            return Ok(Vec::new());
        }
        let decision_at = std::time::Instant::now();
        self.sequence = self.sequence.saturating_add(1);
        let client_id = format!("uets-{}-{}", std::process::id(), self.sequence);

        let request = match decision {
            Decision::EnterLong | Decision::EnterShort => {
                if *position != Position::Flat {
                    return Ok(Vec::new());
                }
                let qty = quantity_for_capital(self.max_position_usd, price, self.min_order_size)
                    .ok_or_else(|| {
                    format!(
                        "cannot size {symbol} order within ${} capital and minimum quantity {}",
                        self.max_position_usd, self.min_order_size
                    )
                })?;
                let side = if decision == Decision::EnterLong {
                    Side::Buy
                } else {
                    Side::Sell
                };

                // Toggle between Market and Limit (Maker) orders here
                let use_limit_orders = false; // Change to true if you want limit/maker orders

                if use_limit_orders {
                    let limit_px = if side == Side::Buy {
                        bid_price.unwrap_or(price)
                    } else {
                        ask_price.unwrap_or(price)
                    };
                    OrderRequest::limit(symbol, side, qty, limit_px, 0.01) // Added tick_size: 0.01
                        .with_client_order_id(client_id)
                } else {
                    OrderRequest::market(symbol, side, qty).with_client_order_id(client_id)
                }
            }
            Decision::Exit => {
                if *position == Position::Flat || self.quantity <= 0.0 {
                    return Ok(Vec::new());
                }
                let side = if *position == Position::Long {
                    Side::Sell
                } else {
                    Side::Buy
                };
                OrderRequest::market(symbol, side, self.quantity)
                    .reducing()
                    .with_client_order_id(client_id)
            }
            Decision::Hold => return Ok(Vec::new()),
        };

        self.last_decision_to_order_ms = Some(decision_at.elapsed().as_secs_f64() * 1_000.0);
        if let Some(connector) = self.connector.as_mut() {
            let order_submitted_at = std::time::Instant::now();
            let fills = connector.submit(&request).map_err(|error| {
                format!(
                    "{} order {} failed: {error}",
                    self.mode,
                    request.client_order_id.as_deref().unwrap_or("unknown")
                )
            })?;
            self.last_order_to_fill_ms = Some(order_submitted_at.elapsed().as_secs_f64() * 1_000.0);
            let filled_qty: f64 = fills.iter().map(|fill| fill.qty).sum();
            if filled_qty <= 0.0 || !filled_qty.is_finite() {
                return Err(format!(
                    "order {} returned no valid fills",
                    request.client_order_id.as_deref().unwrap_or("unknown")
                ));
            }
            println!(
                "OFMR routed mode={} symbol={} side={:?} qty={} fills={} fees={}",
                self.mode,
                symbol,
                request.side,
                filled_qty,
                fills.len(),
                fills.iter().map(|fill| fill.fee).sum::<f64>()
            );
            if matches!(decision, Decision::EnterLong | Decision::EnterShort) {
                self.quantity = filled_qty;
                *position = if decision == Decision::EnterLong {
                    Position::Long
                } else {
                    Position::Short
                };
            } else {
                self.quantity = 0.0;
                *position = Position::Flat;
            }
            Ok(fills)
        } else {
            println!(
                "OFMR simulated mode={} symbol={symbol} decision={decision:?} qty={}",
                self.mode, request.qty
            );
            let fill = Fill::new(
                request.client_order_id.clone().unwrap_or_default(),
                symbol,
                request.side,
                request.qty,
                price,
                0.0,
            );
            self.last_order_to_fill_ms = Some(0.0);
            if matches!(decision, Decision::EnterLong | Decision::EnterShort) {
                self.quantity = request.qty;
                *position = if decision == Decision::EnterLong {
                    Position::Long
                } else {
                    Position::Short
                };
            } else {
                self.quantity = 0.0;
                *position = Position::Flat;
            }
            Ok(vec![fill])
        }
    }
}

#[cfg(any(unix, test))]
struct StrategyAccount {
    portfolio: Portfolio,
    entry_fees: f64,
    sequence: u64,
    entry_risk_per_unit: Option<f64>,
    /// Absolute stop-loss price of the open position (if any), used for the
    /// GUI stop-loss % readout.
    #[allow(dead_code)]
    stop_price: Option<f64>,
    /// Absolute take-profit price of the open position (if any), used for the
    /// GUI take-profit % readout.
    #[allow(dead_code)]
    target_price: Option<f64>,
    r_multiple_sum: f64,
    r_trades: usize,
    last_mark: f64,
    last_mark_at: Option<std::time::Instant>,
}

#[cfg(any(unix, test))]
impl StrategyAccount {
    fn new(starting_equity: f64) -> Result<Self, String> {
        let portfolio = Portfolio::new(starting_equity, 1.0)
            .map_err(|error| format!("invalid OFMR starting equity: {error}"))?;
        Ok(Self {
            portfolio,
            entry_fees: 0.0,
            sequence: 0,
            entry_risk_per_unit: None,
            stop_price: None,
            target_price: None,
            r_multiple_sum: 0.0,
            r_trades: 0,
            last_mark: 0.0,
            last_mark_at: None,
        })
    }

    fn apply_fills_and_mark(&mut self, fills: &[Fill], symbol: &str, price: f64) {
        let old = self.portfolio.holdings().holding(symbol);
        let mut exit_notional = 0.0;
        let mut exit_qty = 0.0;
        let mut exit_fees = 0.0;
        let mut new_entry_fees = 0.0;
        for fill in fills {
            if !old.is_flat() && fill.side != old.side().unwrap_or(fill.side) {
                let closed = fill.qty.min((old.size.abs() - exit_qty).max(0.0));
                exit_notional += fill.price * closed;
                exit_qty += closed;
                exit_fees += if fill.qty > 0.0 {
                    fill.fee * closed / fill.qty
                } else {
                    0.0
                };
            } else {
                new_entry_fees += fill.fee;
            }
            self.portfolio.apply_fill(fill);
        }
        let new = self.portfolio.holdings().holding(symbol);
        if old.is_flat() && !new.is_flat() {
            self.entry_fees = new_entry_fees;
        } else if !old.is_flat() && exit_qty > 0.0 {
            let closed_fraction = (exit_qty / old.size.abs()).clamp(0.0, 1.0);
            let trade = Trade::new(
                symbol,
                old.side().unwrap_or(Side::Buy),
                old.avg_entry,
                exit_notional / exit_qty,
                exit_qty,
                self.entry_fees * closed_fraction + exit_fees,
                self.sequence,
                self.sequence.saturating_add(1),
            );
            self.portfolio.record_trade(trade.clone());
            let risk_per_unit = self.entry_risk_per_unit.unwrap_or(old.avg_entry * 0.0003);
            if let Some(r_multiple) = trade.r_multiple(risk_per_unit) {
                self.r_multiple_sum += r_multiple;
                self.r_trades += 1;
            }
            self.entry_fees = if new.is_flat() {
                0.0
            } else {
                self.entry_fees * (1.0 - closed_fraction) + new_entry_fees
            };
            if new.is_flat() {
                self.entry_risk_per_unit = None;
            }
        }
        if !fills.is_empty() {
            self.sequence = self.sequence.saturating_add(1);
        }
        if price.is_finite() && price > 0.0 {
            self.last_mark = price;
            let now = std::time::Instant::now();
            if !fills.is_empty()
                || self
                    .last_mark_at
                    .is_none_or(|last| now.duration_since(last).as_millis() >= 100)
            {
                let mut marks = std::collections::BTreeMap::new();
                marks.insert(symbol.to_string(), price);
                self.portfolio.mark(&marks);
                self.last_mark_at = Some(now);
            }
        }
    }

    #[allow(dead_code)]
    fn equity_history(&self) -> &[f64] {
        self.portfolio.ledger().equity_curve()
    }
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug)]
struct ActiveBarriers {
    stop_price: f64,
    target_price: f64,
    is_long: bool,
}

#[cfg(unix)]
impl ActiveBarriers {
    fn new(entry_price: f64, position: Position, stop_fraction: f64, target_fraction: f64) -> Self {
        let is_long = position == Position::Long;
        Self {
            stop_price: if is_long {
                entry_price * (1.0 - stop_fraction)
            } else {
                entry_price * (1.0 + stop_fraction)
            },
            target_price: if is_long {
                entry_price * (1.0 + target_fraction)
            } else {
                entry_price * (1.0 - target_fraction)
            },
            is_long,
        }
    }

    fn hit(self, price: f64) -> bool {
        if self.is_long {
            price <= self.stop_price || price >= self.target_price
        } else {
            price >= self.stop_price || price <= self.target_price
        }
    }
}

#[cfg(any(unix, test))]
fn quantity_for_capital(max_position_usd: f64, price: f64, minimum: f64) -> Option<f64> {
    if !price.is_finite() || price <= 0.0 || !minimum.is_finite() || minimum <= 0.0 {
        return None;
    }
    let decimals = minimum
        .to_string()
        .split('.')
        .nth(1)
        .map_or(0, str::len)
        .min(12);
    let scale = 10_f64.powi(decimals as i32);
    let qty = (max_position_usd / price * scale).floor() / scale;
    (qty >= minimum && qty.is_finite()).then_some(qty)
}

/// Run the OFMR socket consumer loop (Unix only).
#[cfg(unix)]
pub fn run(socket_path: &str, symbol: &str) -> Result<(), String> {
    use std::io::Read;
    use std::os::unix::net::UnixListener;
    use std::time::{Duration, Instant};

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
    let starting_equity = std::env::var("UETS_STARTING_EQUITY")
        .ok()
        .map(|value| value.parse::<f64>())
        .transpose()
        .map_err(|error| format!("UETS_STARTING_EQUITY must be a positive number: {error}"))?
        .unwrap_or(10_000.0);
    let mut account = StrategyAccount::new(starting_equity)?;
    let mut telemetry = StrategyTelemetry::new(symbol, &account);
    // Live RocksDB link: live_db/<session>/ofmr@<asset>/ with trades/ +
    // portfolio/ keys. Session comes from the monitor env so every worker
    // of one `run` shares it; standalone workers mint their own.
    let session_id = std::env::var("UETS_SESSION_ID")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(crate::live_strategy::live_db::new_session_id);
    let live_db_root = std::env::var("UETS_LIVE_DB")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| crate::live_strategy::live_db::default_live_db_root());
    let live_db =
        crate::live_strategy::live_db::LiveDb::open(&live_db_root, &session_id, "ofmr", symbol);
    eprintln!(
        "[strategy] live_db session={session_id} persistence={}",
        if live_db.enabled() {
            live_db_root.display().to_string()
        } else {
            "disabled".to_string()
        }
    );

    let mut position = Position::Flat;
    let mut active_barriers: Option<ActiveBarriers> = None;

    for stream in listener.incoming() {
        let mut stream = stream.map_err(|e| format!("accept: {e}"))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .map_err(|error| format!("set strategy socket read timeout: {error}"))?;
        let mut len_buf = [0u8; 4];
        loop {
            // Read 4-byte LE length prefix.
            let mut got = 0usize;
            while got < 4 {
                let n = match stream.read(&mut len_buf[got..]) {
                    Ok(n) => n,
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                    {
                        telemetry.heartbeat(&account);
                        continue;
                    }
                    Err(error) => return Err(format!("read length: {error}")),
                };
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
                let n = match stream.read(&mut payload[got..]) {
                    Ok(n) => n,
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                    {
                        telemetry.heartbeat(&account);
                        continue;
                    }
                    Err(error) => return Err(format!("read payload: {error}")),
                };
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
            telemetry.observe_row(&row);
            let computed = features
                .compute(&row)
                .map_err(|e| format!("compute: {e}"))?;
            let volatility_1m = computed.get("bar_volatility_1m");
            let state = crate::core::strategy::MarketState {
                features: computed,
                position,
                sequence: 0,
            };
            let model_decision = strategy.decide(&state);
            let price = row
                .get("price")
                .or_else(|| {
                    row.get("bid_price")
                        .zip(row.get("ask_price"))
                        .map(|(bid, ask)| (bid + ask) / 2.0)
                })
                .unwrap_or_default();
            let (stop_fraction, target_fraction) =
                crate::live_strategy::ofmr::strategy::barrier_fractions(volatility_1m);
            let decision = if active_barriers.is_some_and(|barriers| barriers.hit(price)) {
                Decision::Exit
            } else {
                model_decision
            };
            if decision != Decision::Hold && (!price.is_finite() || price <= 0.0) {
                return Err(format!(
                    "refusing {decision:?} for {symbol}: invalid market price {price}"
                ));
            }
            let previous_position = position;
            let bid_price = row.get("bid_price");
            let ask_price = row.get("ask_price");
            let fills =
                executor.act(symbol, decision, &mut position, price, bid_price, ask_price)?;
            account.apply_fills_and_mark(&fills, symbol, price);
            // live_db/trades: what the strategy saw (tick + features +
            // decision + fills), keyed by ticker/strategy/session. Fail-open.
            let event_ms = row.get("event_time_ms").map(|v| v as u64);
            let recv_ms = row.get("received_time_ms").map(|v| v as u64);
            let fills_json = fills_to_json(&fills);
            let tick_seq = live_db.next_seq();
            live_db.record_tick(
                &row,
                &state.features,
                event_ms,
                recv_ms,
                &format!("{decision:?}"),
                &format!("{position:?}"),
                &fills_json,
                account.portfolio.ledger().current_equity(),
                account.portfolio.cash(),
            );
            // live_db/portfolio: what the GUI shows. Same seq as the tick.
            live_db.record_portfolio(portfolio_snapshot(
                &session_id,
                symbol,
                tick_seq,
                event_ms,
                price,
                &account,
            ));
            telemetry.persistence_write_ms = Some(live_db.last_write_ms());
            telemetry.persistence_errors = Some(live_db.error_count());
            if previous_position == Position::Flat && position != Position::Flat {
                let filled_qty = fills.iter().map(|fill| fill.qty).sum::<f64>();
                let entry_price = if filled_qty > 0.0 {
                    fills.iter().map(|fill| fill.price * fill.qty).sum::<f64>() / filled_qty
                } else {
                    price
                };
                active_barriers = Some(ActiveBarriers::new(
                    entry_price,
                    position,
                    stop_fraction,
                    target_fraction,
                ));
                account.entry_risk_per_unit = Some(entry_price * stop_fraction);
                let barriers = active_barriers.as_ref().unwrap();
                account.stop_price = Some(barriers.stop_price);
                account.target_price = Some(barriers.target_price);
            } else if position == Position::Flat {
                active_barriers = None;
                account.stop_price = None;
                account.target_price = None;
            }
            let processing_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            telemetry.publish(
                Some(format!("{decision:?}")),
                Some(processing_ns),
                executor.last_decision_to_order_ms,
                executor.last_order_to_fill_ms,
                &account,
            );
        }
    }

    Ok(())
}

#[cfg(unix)]
fn fills_to_json(fills: &[Fill]) -> String {
    let items: Vec<String> = fills
        .iter()
        .map(|f| {
            format!(
                "{{\"order\":\"{}\",\"symbol\":\"{}\",\"side\":\"{}\",\"qty\":{},\"price\":{},\"fee\":{}}}",
                f.order_id, f.symbol, f.side.as_str(), f.qty, f.price, f.fee
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

#[cfg(unix)]
fn portfolio_snapshot(
    session_id: &str,
    symbol: &str,
    seq: u64,
    event_time_ms: Option<u64>,
    price: f64,
    account: &StrategyAccount,
) -> crate::live_strategy::live_db::LivePortfolioSnapshot {
    use crate::live_strategy::live_db::LivePortfolioSnapshot;
    let stats = PerformanceStats::from_ledger(account.portfolio.ledger(), 1.0);
    let asset = symbol.to_ascii_lowercase();
    // Same marks rule as the GUI telemetry: last mark, else entry price
    // (which contributes zero unrealized — a stale mark is worse).
    let marks = std::collections::BTreeMap::from([(
        asset.clone(),
        if account.last_mark > 0.0 {
            account.last_mark
        } else {
            price
        },
    )]);
    let portfolio = &account.portfolio;
    LivePortfolioSnapshot {
        session_id: session_id.to_string(),
        strategy_id: "ofmr".to_string(),
        asset_id: asset.clone(),
        symbol: symbol.to_ascii_uppercase(),
        seq,
        event_time_ms,
        cash: portfolio.cash(),
        equity: portfolio.ledger().current_equity(),
        realized_pnl: portfolio.holdings().realized_pnl(),
        unrealized_pnl: portfolio.unrealized_pnl(&marks),
        fees_paid: portfolio.holdings().fees_paid(),
        gross_exposure: portfolio.gross_exposure(&marks),
        net_exposure: portfolio.net_exposure(&marks),
        margin_used: portfolio.margin_used(&marks),
        available_margin: portfolio.available_margin(&marks),
        effective_leverage: portfolio.effective_leverage(&marks),
        r_multiple_sum: account.r_multiple_sum,
        r_trades: account.r_trades as u64,
        sortino: stats.sortino,
        win_rate: if stats.trades > 0 {
            stats.wins as f64 / stats.trades as f64
        } else {
            0.0
        },
        cumulative_return: stats.cumulative_return,
        max_drawdown: stats.max_drawdown,
        trades: stats.trades,
        wins: stats.wins,
        losses: stats.losses,
        expectancy_rr: if account.r_trades > 0 {
            account.r_multiple_sum / account.r_trades as f64
        } else {
            0.0
        },
        mdd: stats.max_drawdown,
    }
}

#[cfg(unix)]
struct StrategyTelemetry {
    socket: Option<std::os::unix::net::UnixDatagram>,
    symbol: String,
    latency: crate::aggregator::latency::LatencyHist,
    clock_skew: crate::core::telemetry::ClockSkew,
    last_publish: Option<std::time::Instant>,
    last_event_time_ms: Option<u64>,
    last_decision: Option<String>,
    last_processing_ms: Option<f64>,
    last_decision_to_order_ms: Option<f64>,
    last_order_to_fill_ms: Option<f64>,
    /// One-way exchange->local feed latency measured at socket receive
    /// (`received_time_ms - event_time_ms`), before any queueing here.
    exchange_latency_ms: Option<f64>,
    /// RocksDB durability write time reported by the feed producer.
    persistence_write_ms: Option<f64>,
    persistence_errors: Option<u64>,
    persistence_queue: Option<usize>,
    cpu: CpuSampler,
}
#[cfg(unix)]
impl StrategyTelemetry {
    fn new(symbol: &str, account: &StrategyAccount) -> Self {
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
            last_publish: None,
            last_event_time_ms: None,
            last_decision: None,
            last_processing_ms: None,
            last_decision_to_order_ms: Some(0.0),
            last_order_to_fill_ms: Some(0.0),
            exchange_latency_ms: None,
            persistence_write_ms: Some(0.0),
            persistence_errors: Some(0),
            persistence_queue: Some(0),
            cpu: CpuSampler::new(),
        };
        publisher.publish(
            Some("waiting for market data".to_string()),
            None,
            None,
            None,
            account,
        );
        publisher
    }

    fn heartbeat(&mut self, account: &StrategyAccount) {
        self.publish(None, None, None, None, account);
    }

    /// Fold one decoded feed row into the rolling telemetry state. Called for
    /// every frame *before* the publish throttle so feed latency, clock skew,
    /// and persistence observations are never dropped.
    fn observe_row(&mut self, row: &FeatureRow) {
        let event_time_ms = row.get("event_time_ms").map(|value| value as u64);
        let received_time_ms = row.get("received_time_ms").map(|value| value as u64);
        if let Some(event_time_ms) = event_time_ms {
            if let Some(received_time_ms) = received_time_ms.filter(|ms| *ms > 0) {
                // Venue stamp vs local receipt: skew + one-way transit,
                // sampled before this process queues the frame.
                self.clock_skew.record(event_time_ms, received_time_ms);
                self.exchange_latency_ms =
                    Some((received_time_ms as i64 - event_time_ms as i64) as f64);
            }
            self.last_event_time_ms = Some(event_time_ms);
        }
        if let Some(write_ms) = row.get("persistence_write_ms") {
            self.persistence_write_ms = Some(write_ms);
        }
        if let Some(errors) = row.get("persistence_errors") {
            self.persistence_errors = Some(errors as u64);
        }
        if let Some(queue) = row.get("persistence_queue") {
            self.persistence_queue = Some(queue as usize);
        }
    }

    fn publish(
        &mut self,
        decision: Option<String>,
        processing_ns: Option<u64>,
        decision_to_order_ms: Option<f64>,
        order_to_fill_ms: Option<f64>,
        account: &StrategyAccount,
    ) {
        // Record every observation *before* the send throttle: the executor
        // clears its per-frame values on the next tick, so dropping them here
        // pinned Decision->order / Order->fill at n/a forever.
        if let Some(decision) = decision {
            self.last_decision = Some(decision);
        }
        if let Some(processing_ns) = processing_ns {
            self.latency.observe_ns(processing_ns);
            self.last_processing_ms = Some(processing_ns as f64 / 1_000_000.0);
        }
        if decision_to_order_ms.is_some() {
            self.last_decision_to_order_ms = decision_to_order_ms;
        }
        if order_to_fill_ms.is_some() {
            self.last_order_to_fill_ms = order_to_fill_ms;
        }
        let publish_started = std::time::Instant::now();
        if self
            .last_publish
            .is_some_and(|last| publish_started.duration_since(last).as_millis() < 100)
        {
            return;
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or_default();
        let skew = self.clock_skew.snapshot();
        // `or(Some(0.0))`: first sample primes the /proc delta, show 0.000 not n/a.
        let cpu_percent = self.cpu.sample().or(Some(0.0));
        let memory_mb = process_memory_mb();
        // Tick-to-trade loop: feed one-way + compute + decision->order + order->fill.
        // This is how you prove orders arrive in time: compare the full loop
        // against the edge horizon (5m barriers). Loop << horizon = fast enough.
        let tick_to_trade_ms: f64 = self
            .exchange_latency_ms
            .iter()
            .chain(self.last_processing_ms.iter())
            .chain(self.last_decision_to_order_ms.iter())
            .chain(self.last_order_to_fill_ms.iter())
            .copied()
            .sum();
        let full_history = account.equity_history();
        let history_start = full_history.len().saturating_sub(3_000);
        let history = &full_history[history_start..];
        let current_equity = history
            .last()
            .copied()
            .unwrap_or(account.portfolio.starting_cash());
        let stats = PerformanceStats::from_ledger(account.portfolio.ledger(), 1.0);
        let holding = account.portfolio.holdings().holding(&self.symbol);
        let entry = holding.avg_entry;
        let position_open = !holding.is_flat() && entry > 0.0;
        let mark = if account.last_mark > 0.0 {
            account.last_mark
        } else {
            entry
        };
        let (stop_loss_pct, take_profit_pct, price_vs_entry_pct) = if position_open {
            (
                account
                    .stop_price
                    .map(|stop| (entry - stop).abs() / entry * 100.0),
                account
                    .target_price
                    .map(|target| (target - entry).abs() / entry * 100.0),
                Some((mark - entry) / entry * 100.0),
            )
        } else {
            (None, None, None)
        };
        let marks = std::collections::BTreeMap::from([(
            self.symbol.clone(),
            if account.last_mark > 0.0 {
                account.last_mark
            } else {
                holding.avg_entry
            },
        )]);
        let cash = account.portfolio.cash();
        let unrealized_pnl = account.portfolio.unrealized_pnl(&marks);
        let realized_pnl =
            account.portfolio.holdings().realized_pnl() - account.portfolio.holdings().fees_paid();
        let sortino = if stats.sortino.is_finite() {
            stats.sortino
        } else {
            0.0
        };
        let statistics = serde_json::json!({
            "status": "enabled",
            "starting_equity": account.portfolio.starting_cash(),
            "equity": current_equity,
            "cash": cash,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "exposure": account.portfolio.gross_exposure(&marks),
            "open_positions": if holding.is_flat() { 0 } else { 1 },
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "price_vs_entry_pct": price_vs_entry_pct,
            "trades": stats.trades,
            "win_rate": stats.win_rate,
            "sortino": sortino,
            "expectancy_rr": if account.r_trades > 0 { account.r_multiple_sum / account.r_trades as f64 } else { 0.0 },
            "expectancy_trades": account.r_trades,
            "max_drawdown": stats.max_drawdown,
            "cumulative_return": stats.cumulative_return,
            "absolute_return": current_equity - account.portfolio.starting_cash(),
            "latest_market_data": self.last_event_time_ms.map(|time| time.to_string()),
            "latest_decision": self.last_decision,
            "processing_latency_ms": self.last_processing_ms,
            "processing_latency_mean_ms": self.latency.mean_ns().map(|value| value / 1_000_000.0),
            "latency_sum_ms": self.latency.sum_ns() as f64 / 1_000_000.0,
            "latency_observations": self.latency.count(),
            "decision_to_order_ms": self.last_decision_to_order_ms,
            "order_to_fill_ms": self.last_order_to_fill_ms,
            "tick_to_trade_ms": tick_to_trade_ms,
            "exchange_latency_ms": self.exchange_latency_ms.or_else(|| self.last_event_time_ms.map(|event_time| (now_ms as i64 - event_time as i64) as f64)),
            "latency_p05_ms": self.latency.percentile(0.05).map(|value| value as f64 / 1_000_000.0),
            "latency_p95_ms": self.latency.percentile(0.95).map(|value| value as f64 / 1_000_000.0),
            "latency_p99_ms": self.latency.percentile(0.99).map(|value| value as f64 / 1_000_000.0),
            "clock_skew_mean_ms": skew.mean_us.map(|value| value / 1_000.0),
            "clock_skew_max_ms": skew.max_abs_us.map(|value| value as f64 / 1_000.0),
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "persistence_queue": self.persistence_queue,
            "persistence_write_ms": self.persistence_write_ms,
            "persistence_errors": self.persistence_errors,
        });
        let snapshot = serde_json::json!({
            "strategy_id": "ofmr",
            "asset_id": self.symbol,
            "statistics": statistics,
            "equity_history": history,
        });
        if let (Some(socket), Ok(payload)) = (&self.socket, serde_json::to_vec(&snapshot)) {
            if let Err(error) = socket.send(&payload) {
                if error.kind() != std::io::ErrorKind::WouldBlock {
                    eprintln!("[telemetry] GUI update failed: {error}");
                }
            }
        }
        self.last_publish = Some(publish_started);
    }
}

/// Process CPU utilization sampled from `/proc/self/stat` (`utime + stime`
/// deltas over wall time between samples). Returns None until the second
/// sample lands.
#[cfg(all(unix, target_os = "linux"))]
struct CpuSampler {
    last_ticks: Option<u64>,
    last_wall: Option<std::time::Instant>,
    percent: Option<f64>,
}

#[cfg(all(unix, target_os = "linux"))]
impl CpuSampler {
    fn new() -> Self {
        Self {
            last_ticks: None,
            last_wall: None,
            percent: None,
        }
    }

    fn sample(&mut self) -> Option<f64> {
        let ticks = process_cpu_ticks()?;
        let now = std::time::Instant::now();
        if let (Some(prev_ticks), Some(prev_wall)) = (self.last_ticks, self.last_wall) {
            let wall_s = now.duration_since(prev_wall).as_secs_f64();
            if wall_s > 0.0 {
                let cpu_s = ticks.saturating_sub(prev_ticks) as f64 / process_clock_ticks();
                self.percent = Some((cpu_s / wall_s * 100.0).clamp(0.0, 100.0));
            }
        }
        self.last_ticks = Some(ticks);
        self.last_wall = Some(now);
        self.percent
    }
}

/// `utime + stime` in clock ticks from `/proc/self/stat`. The `comm` field
/// can contain spaces and parentheses, so parsing starts after the final ')'.
#[cfg(all(unix, target_os = "linux"))]
fn process_cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // After comm: state(3) ppid(4) ... utime is field 14, stime field 15.
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime.saturating_add(stime))
}

#[cfg(all(unix, target_os = "linux"))]
fn process_clock_ticks() -> f64 {
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks > 0 { ticks as f64 } else { 100.0 }
}

/// No procfs on this Unix flavor: CPU% stays n/a.
#[cfg(all(unix, not(target_os = "linux")))]
struct CpuSampler;

#[cfg(all(unix, not(target_os = "linux")))]
impl CpuSampler {
    fn new() -> Self {
        Self
    }

    fn sample(&mut self) -> Option<f64> {
        None
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
    fn strategy_account_marks_equity_and_records_closed_trade_metrics() {
        let mut account = StrategyAccount::new(1_000.0).unwrap();
        let entry = Fill::new("entry", "btcusdt", Side::Buy, 1.0, 100.0, 0.4);
        account.apply_fills_and_mark(&[entry], "btcusdt", 100.0);
        let exit = Fill::new("exit", "btcusdt", Side::Sell, 1.0, 120.0, 0.4);
        account.apply_fills_and_mark(&[exit], "btcusdt", 120.0);

        let stats = PerformanceStats::from_ledger(account.portfolio.ledger(), 1.0);
        assert_eq!(
            account.portfolio.ledger().equity_curve().last(),
            Some(&1_019.2)
        );
        assert_eq!(account.portfolio.cash(), 1_019.2);
        assert_eq!(stats.trades, 1);
        assert_eq!(stats.wins, 1);
        assert_eq!(account.r_trades, 1);
        // Fallback risk = 100 * 0.0003 (min-vol 0.001 x 0.3 barrier) = 0.03,
        // so R = 19.2 / 0.03 = 640.
        assert!((account.r_multiple_sum - 640.0).abs() < 1e-9);
    }

    #[test]
    fn execution_config_requires_mode_and_defaults_live_guard_to_false() {
        let config = ExecutionConfig::parse("execution:\n  mode: live-test\n").unwrap();
        assert_eq!(config.mode, ExecutionMode::LiveTest);
        assert!(!config.allow_live_trading);
        assert!(ExecutionConfig::parse("execution:\n  mode: live-trade\n").is_ok());
        assert!(ExecutionConfig::parse("execution:\n  mode: unsafe\n").is_err());
    }

    #[test]
    fn research_settings_do_not_define_an_execution_mode() {
        let config = ExecutionConfig::parse(
            "mode: live-test\nallow_live_trading: false\nplayback:\n  method: cpcv\n",
        )
        .unwrap();
        assert_eq!(config.mode, ExecutionMode::LiveTest);
        assert!(!config.allow_live_trading);
    }
}
