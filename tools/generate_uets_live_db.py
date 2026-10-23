from pathlib import Path

path = Path(__file__).resolve().parents[1] / "uets" / "live_strategy" / "live_db.rs"

part1 = r'''//! live_db: live engine -> RocksDB via `uets_db`.
//! Physical layout:
//!   live_db/trades/<session>/     -> independent RocksDB instance
//!   live_db/portfolio/<session>/  -> independent RocksDB instance
//! Every record carries session_id, strategy_id, asset_id as columns/fields
//! inside the RocksDB records (visible only when reading via QueryFilter).
//! Fail-open.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uets_db::{QueryFilter, SchemaVersion, StrategyInstance, TradeIntent, TradeRecord};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveTradeTick {
    pub session_id: String,
    pub strategy_id: String,
    pub asset_id: String,
    pub symbol: String,
    pub seq: u64,
    pub event_time_ms: Option<u64>,
    pub received_time_ms: Option<u64>,
    pub price: Option<f64>,
    pub amount: Option<f64>,
    pub is_buyer_maker: Option<bool>,
    pub bid_price: Option<f64>,
    pub bid_qty: Option<f64>,
    pub ask_price: Option<f64>,
    pub ask_qty: Option<f64>,
    pub raw_inputs_json: String,
    pub features_json: String,
    pub decision: String,
    pub position: String,
    pub fills_json: String,
    pub equity: f64,
    pub cash: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LivePortfolioSnapshot {
    pub session_id: String,
    pub strategy_id: String,
    pub asset_id: String,
    pub symbol: String,
    pub seq: u64,
    pub event_time_ms: Option<u64>,
    pub cash: f64,
    pub equity: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub fees_paid: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub margin_used: f64,
    pub available_margin: f64,
    pub effective_leverage: f64,
    pub r_multiple_sum: f64,
    pub r_trades: u64,
    pub sortino: f64,
    pub win_rate: f64,
    pub cumulative_return: f64,
    pub max_drawdown: f64,
    pub trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub expectancy_rr: f64,
    pub mdd: f64,
}

pub struct LiveDb {
    trades_db: Option<Arc<uets_db::UetsDb>>,
    portfolio_db: Option<Arc<uets_db::UetsDb>>,
    session_id: String,
    strategy_id: String,
    asset_id: String,
    symbol: String,
    seq: AtomicU64,
    last_write_ms: Arc<std::sync::Mutex<f64>>,
    errors: Arc<AtomicU64>,
}

impl LiveDb {
    pub fn open(
        live_db_root: impl Into<PathBuf>,
        session_id: &str,
        strategy_id: &str,
        asset_id: &str,
    ) -> Self {
        let asset_id = asset_id.to_ascii_lowercase();
        let strategy_id = strategy_id.to_ascii_lowercase();
        let symbol = asset_id.to_ascii_uppercase();
        let root: PathBuf = live_db_root.into();
        let errors = Arc::new(AtomicU64::new(0));
        let last_write_ms = Arc::new(std::sync::Mutex::new(0.0));

        let trades_path = root.join("trades").join(session_id);
        let portfolio_path = root.join("portfolio").join(session_id);

        let trades_db = match uets_db::UetsDb::open(&trades_path) {
            Ok(db) => Some(Arc::new(db)),
            Err(error) => {
                eprintln!("[live_db] trades open failed: {error}; trades persistence disabled");
                errors.fetch_add(1, Ordering::Relaxed);
                None
            }
        };
        let portfolio_db = match uets_db::UetsDb::open(&portfolio_path) {
            Ok(db) => Some(Arc::new(db)),
            Err(error) => {
                eprintln!("[live_db] portfolio open failed: {error}; portfolio persistence disabled");
                errors.fetch_add(1, Ordering::Relaxed);
                None
            }
        };

        Self {
            trades_db,
            portfolio_db,
            session_id: session_id.to_string(),
            strategy_id: strategy_id.to_string(),
            asset_id: asset_id.to_string(),
            symbol,
            seq: AtomicU64::new(0),
            last_write_ms,
            errors,
        }
    }

    pub fn disabled() -> Self {
        Self {
            trades_db: None,
            portfolio_db: None,
            session_id: String::new(),
            strategy_id: String::new(),
            asset_id: String::new(),
            symbol: String::new(),
            seq: AtomicU64::new(0),
            last_write_ms: Arc::new(std::sync::Mutex::new(0.0)),
            errors: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn enabled(&self) -> bool {
        self.trades_db.is_some() || self.portfolio_db.is_some()
    }

    pub fn last_write_ms(&self) -> f64 {
        *self.last_write_ms.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn error_count(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    pub fn next_seq(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }
'''

with open(path, 'w', encoding="utf-8") as f:
    f.write(part1)

part2 = r'''
    pub fn flush(&self) {
        if let Some(db) = self.trades_db.as_ref() {
            if let Err(error) = db.flush() {
                self.errors.fetch_add(1, Ordering::Relaxed);
                eprintln!("[live_db] trades flush failed (counted): {error}");
            }
        }
        if let Some(db) = self.portfolio_db.as_ref() {
            if let Err(error) = db.flush() {
                self.errors.fetch_add(1, Ordering::Relaxed);
                eprintln!("[live_db] portfolio flush failed (counted): {error}");
            }
        }
    }

    pub fn read_all(&self) -> Result<(Vec<LiveTradeTick>, Vec<LivePortfolioSnapshot>), String> {
        let filter = QueryFilter::new()
            .session(&self.session_id)
            .strategy(&self.strategy_id)
            .asset(&self.asset_id);

        let mut trades = Vec::new();
        if let Some(db) = self.trades_db.as_ref() {
            for record in db.query_trades(&filter).map_err(|e| e.to_string())? {
                trades.push(trade_record_to_tick(&record));
            }
        }
        trades.sort_by_key(|t: &LiveTradeTick| t.seq);

        let mut portfolios = Vec::new();
        if let Some(db) = self.portfolio_db.as_ref() {
            for record in db.query_portfolio(&filter).map_err(|e| e.to_string())? {
                portfolios.push(portfolio_record_to_snapshot(&record));
            }
        }
        portfolios.sort_by_key(|p: &LivePortfolioSnapshot| p.seq);

        Ok((trades, portfolios))
    }

    pub fn record_tick(
        &self,
        raw: &crate::core::feature::FeatureRow,
        computed: &crate::core::feature::FeatureRow,
        event_time_ms: Option<u64>,
        received_time_ms: Option<u64>,
        decision: &str,
        position: &str,
        fills_json: &str,
        equity: f64,
        cash: f64,
    ) {
        let Some(db) = self.trades_db.as_ref() else { return };
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let started = std::time::Instant::now();
        let out = (|| -> Result<(), String> {
            let event_time = event_time_ms
                .and_then(|ms| DateTime::from_timestamp_millis(ms as i64))
                .unwrap_or_else(Utc::now);
            let intent = match decision {
                s if s.contains("Short") => TradeIntent::Short,
                _ => TradeIntent::Long,
            };
            let record = TradeRecord {
                schema: SchemaVersion::default(),
                session_id: self.session_id.clone(),
                instance: StrategyInstance::new(&self.strategy_id, &self.asset_id),
                symbol: self.symbol.clone(),
                event_time,
                feature_inputs: raw_inputs_json(raw),
                decision: intent.clone(),
                decision_metadata: serde_json::json!({"decision": decision, "position": position}).to_string(),
                intent,
                order_events: Vec::new(),
                entry_price: None,
                exit_price: None,
                tp_price: None,
                sl_price: None,
                vertical_time: None,
                fill_qty: 0.0,
                fees: 0.0,
                realized_pnl: 0.0,
                model_version: self.strategy_id.clone(),
                config_id: self.strategy_id.clone(),
            };
            db.write_trade(&record).map_err(|e| e.to_string())
        })();
        *self.last_write_ms.lock().unwrap_or_else(|e| e.into_inner()) =
            started.elapsed().as_secs_f64() * 1_000.0;
        if let Err(error) = out {
            self.errors.fetch_add(1, Ordering::Relaxed);
            eprintln!("[live_db] tick write failed (counted): {error}");
        }
    }
'''

with open(path, 'a', encoding="utf-8") as f:
    f.write(part2)

part3 = r'''
    pub fn record_portfolio(&self, snapshot: LivePortfolioSnapshot) {
        let Some(db) = self.portfolio_db.as_ref() else { return };
        let started = std::time::Instant::now();
        let out = (|| -> Result<(), String> {
            let event_time = snapshot
                .event_time_ms
                .and_then(|ms| DateTime::from_timestamp_millis(ms as i64))
                .unwrap_or_else(Utc::now);
            let record = uets_db::PortfolioRecord {
                schema: SchemaVersion::default(),
                session_id: self.session_id.clone(),
                instance: StrategyInstance::new(&self.strategy_id, &self.asset_id),
                asset_id: self.asset_id.clone(),
                timestamp: event_time,
                allocated_capital: snapshot.equity,
                cash: snapshot.cash,
                equity: snapshot.equity,
                realized_pnl: snapshot.realized_pnl,
                unrealized_pnl: snapshot.unrealized_pnl,
                cumulative_return: snapshot.cumulative_return,
                sortino: snapshot.sortino,
                drawdown: snapshot.mdd,
                max_drawdown: snapshot.mdd,
                gross_exposure: snapshot.gross_exposure,
                net_exposure: snapshot.net_exposure,
                margin_used: snapshot.margin_used,
                available_margin: snapshot.available_margin,
                effective_leverage: snapshot.effective_leverage,
                fees_paid: snapshot.fees_paid,
                trade_count: snapshot.trades,
                win_count: snapshot.wins,
                loss_count: snapshot.losses,
                win_rate: snapshot.win_rate,
            };
            db.write_portfolio(&record).map_err(|e| e.to_string())
        })();
        *self.last_write_ms.lock().unwrap_or_else(|e| e.into_inner()) =
            started.elapsed().as_secs_f64() * 1_000.0;
        if let Err(error) = out {
            self.errors.fetch_add(1, Ordering::Relaxed);
            eprintln!("[live_db] portfolio write failed (counted): {error}");
        }
    }
}
'''

with open(path, 'a', encoding="utf-8") as f:
    f.write(part3)

part4 = r'''
fn trade_record_to_tick(record: &TradeRecord) -> LiveTradeTick {
    LiveTradeTick {
        session_id: record.session_id.clone(),
        strategy_id: record.instance.strategy_id.clone(),
        asset_id: record.instance.asset_id.clone(),
        symbol: record.symbol.clone(),
        seq: record
            .event_time
            .timestamp_nanos_opt()
            .map(|n| (n / 1_000_000) as u64)
            .unwrap_or(0),
        event_time_ms: Some(record.event_time.timestamp_millis() as u64),
        received_time_ms: None,
        price: None,
        amount: None,
        is_buyer_maker: None,
        bid_price: None,
        bid_qty: None,
        ask_price: None,
        ask_qty: None,
        raw_inputs_json: record.feature_inputs.clone(),
        features_json: record.feature_inputs.clone(),
        decision: format!("{:?}", record.decision),
        position: String::new(),
        fills_json: String::new(),
        equity: record.entry_price.unwrap_or(0.0),
        cash: 0.0,
    }
}

fn portfolio_record_to_snapshot(record: &uets_db::PortfolioRecord) -> LivePortfolioSnapshot {
    LivePortfolioSnapshot {
        session_id: record.session_id.clone(),
        strategy_id: record.instance.strategy_id.clone(),
        asset_id: record.instance.asset_id.clone(),
        symbol: record.instance.strategy_id.clone(),
        seq: record.timestamp.timestamp_nanos_opt().map(|n| (n / 1_000_000) as u64).unwrap_or(0),
        event_time_ms: Some(record.timestamp.timestamp_millis() as u64),
        cash: record.cash,
        equity: record.equity,
        realized_pnl: record.realized_pnl,
        unrealized_pnl: record.unrealized_pnl,
        fees_paid: record.fees_paid,
        gross_exposure: record.gross_exposure,
        net_exposure: record.net_exposure,
        margin_used: record.margin_used,
        available_margin: record.available_margin,
        effective_leverage: record.effective_leverage,
        r_multiple_sum: 0.0,
        r_trades: 0,
        sortino: record.sortino,
        win_rate: record.win_rate,
        cumulative_return: record.cumulative_return,
        max_drawdown: record.max_drawdown,
        trades: record.trade_count,
        wins: record.win_count,
        losses: record.loss_count,
        expectancy_rr: 0.0,
        mdd: record.max_drawdown,
    }
}
'''

with open(path, 'a', encoding="utf-8") as f:
    f.write(part4)

part5 = r'''
fn raw_inputs_json(row: &crate::core::feature::FeatureRow) -> String {
    let mut map = serde_json::Map::new();
    for (name, value) in row.iter() {
        match value {
            crate::core::feature::FeatureValue::Number(v) => {
                map.insert(name.to_string(), serde_json::json!(v));
            }
            crate::core::feature::FeatureValue::Missing => {
                map.insert(name.to_string(), serde_json::Value::Null);
            }
        }
    }
    serde_json::Value::Object(map).to_string()
}

pub fn default_live_db_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("market_data").join("live_db"))
        .unwrap_or_else(|| PathBuf::from("market_data/live_db"))
}

pub fn new_session_id() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    format!("live_{secs}_{}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_db_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let db = LiveDb::open(dir.path(), "sess_1", "ofmr", "btcusdt");
        assert!(db.enabled());

        let disabled = LiveDb::disabled();
        assert!(!disabled.enabled());
        assert_eq!(disabled.next_seq(), 0);
        disabled.record_tick(
            &crate::core::feature::FeatureRow::default(),
            &crate::core::feature::FeatureRow::default(),
            None,
            None,
            "Hold",
            "Flat",
            "{}",
            0.0,
            0.0,
        );
        disabled.record_portfolio(crate::live_strategy::live_db::LivePortfolioSnapshot {
            session_id: String::new(),
            strategy_id: String::new(),
            asset_id: String::new(),
            symbol: String::new(),
            seq: 0,
            event_time_ms: None,
            cash: 0.0,
            equity: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            fees_paid: 0.0,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            margin_used: 0.0,
            available_margin: 0.0,
            effective_leverage: 0.0,
            r_multiple_sum: 0.0,
            r_trades: 0,
            sortino: 0.0,
            win_rate: 0.0,
            cumulative_return: 0.0,
            max_drawdown: 0.0,
            trades: 0,
            wins: 0,
            losses: 0,
            expectancy_rr: 0.0,
            mdd: 0.0,
        });
        assert_eq!(disabled.error_count(), 0);
    }
}
'''

with open(path, 'a', encoding="utf-8") as f:
    f.write(part5)

print('part5 ok')