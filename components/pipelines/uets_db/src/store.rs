//! Database store — wraps the embedded key-value store with session-aware
//! keys, indexes, batched writes, and async persistence.

use crate::schema::{
    IntegrityReport, PortfolioRecord, QueryFilter, SessionMeta, SessionSummary, TradeRecord,
};
use chrono::Utc;
use rocksdb::{Direction, IteratorMode, WriteBatch, DB};
use std::sync::Arc;
use std::path::Path;
use thiserror::Error;

use crate::schema::SchemaVersion;

/// Database errors.
#[derive(Debug, Error)]
pub enum DbError {
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("record not found: {key}")]
    NotFound { key: String },
    #[error("corrupt record: {key}")]
    Corrupt { key: String },
    #[error("incompatible schema version: found {found}, supported {supported}")]
    IncompatibleSchema { found: String, supported: String },
    #[error("session already exists: {id}")]
    SessionExists { id: String },
    #[error("session not found: {id}")]
    SessionNotFound { id: String },
    /// Test-only injected failure (never produced outside tests).
    #[cfg(test)]
    #[error("injected write failure")]
    Injected,
}

/// The database store. Wraps a single RocksDB instance with logical namespaces.
pub struct UetsDb {
    db: Arc<DB>,
    trades: DbTree,
    portfolio: DbTree,
    metadata: DbTree,
    /// Test-only: simulate storage failures for failure-handling tests.
    #[cfg(test)]
    fail_writes: std::sync::atomic::AtomicBool,
}

#[derive(Clone)]
struct DbTree {
    db: Arc<DB>,
    namespace: &'static str,
}

impl DbTree {
    fn key(&self, key: impl AsRef<[u8]>) -> Vec<u8> {
        let mut result = self.namespace.as_bytes().to_vec();
        result.push(b'/');
        result.extend_from_slice(key.as_ref());
        result
    }

    fn get(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>, rocksdb::Error> {
        self.db.get(self.key(key))
    }

    fn insert(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<(), rocksdb::Error> {
        self.db.put(self.key(key), value)
    }

    fn apply_batch(&self, batch: DbBatch) -> Result<(), rocksdb::Error> {
        let mut writes = WriteBatch::default();
        for (key, value) in batch.0 {
            writes.put(self.key(key), value);
        }
        self.db.write(writes)
    }

    fn iter(&self) -> impl Iterator<Item = Result<(Vec<u8>, Vec<u8>), rocksdb::Error>> + '_ {
        self.range(Vec::<u8>::new()..)
    }

    fn range<K: AsRef<[u8]>>(
        &self,
        start: std::ops::RangeFrom<K>,
    ) -> impl Iterator<Item = Result<(Vec<u8>, Vec<u8>), rocksdb::Error>> + '_ {
        let start = self.key(start.start);
        let prefix = format!("{}/", self.namespace).into_bytes();
        let prefix_len = prefix.len();
        let range_prefix = prefix.clone();
        self.db
            .iterator(IteratorMode::From(&start, Direction::Forward))
            .take_while(move |entry| match entry {
                Ok((key, _)) => key.starts_with(&range_prefix),
                Err(_) => true,
            })
            .map(move |entry| {
                entry.map(|(key, value)| (key[prefix_len..].to_vec(), value.to_vec()))
            })
    }
}

#[derive(Default)]
struct DbBatch(Vec<(Vec<u8>, Vec<u8>)>);

impl DbBatch {
    fn insert(&mut self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) {
        self.0.push((key.as_ref().to_vec(), value.as_ref().to_vec()));
    }
}

impl UetsDb {
    /// Open or create the database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let db = Arc::new(DB::open_default(path)?);
        let trades = DbTree { db: db.clone(), namespace: "trades" };
        let portfolio = DbTree { db: db.clone(), namespace: "portfolio" };
        let metadata = DbTree { db: db.clone(), namespace: "metadata" };

        // Write schema version if not present; validate it if present.
        if let Some(raw) = metadata.get(b"schema_version")? {
            let stored: SchemaVersion = bincode::deserialize(raw.as_ref())
                .map_err(|e| DbError::Serialization(e.to_string()))?;
            let current = SchemaVersion::default();
            if stored.major != current.major {
                return Err(DbError::IncompatibleSchema {
                    found: format!("{}.{}", stored.major, stored.minor),
                    supported: format!("{}.{}", current.major, current.minor),
                });
            }
        } else {
            let version = SchemaVersion::default();
            let encoded = bincode::serialize(&version)
                .map_err(|e| DbError::Serialization(e.to_string()))?;
            metadata.insert(b"schema_version", encoded)?;
        }

        Ok(Self {
            db,
            trades,
            portfolio,
            metadata,
            #[cfg(test)]
            fail_writes: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Flush all writes to disk.
    pub fn flush(&self) -> Result<(), DbError> {
        self.db.flush()?;
        Ok(())
    }

    /// Shut down the database cleanly.
    pub fn shutdown(self) -> Result<(), DbError> {
        // Flush before dropping the database handle.
        self.db.flush()?;
        Ok(())
    }

    // ------- Trades -------

    /// Insert a trade record (record + index atomically, same tree).
    pub fn write_trade(&self, record: &TradeRecord) -> Result<(), DbError> {
        self.ensure_writable()?;
        let key = Self::trade_key(record);
        let encoded = bincode::serialize(record)
            .map_err(|e| DbError::Serialization(e.to_string()))?;
        let ikey = Self::trade_index_key(record);

        let mut batch = DbBatch::default();
        batch.insert(key.clone(), encoded);
        batch.insert(ikey, key);
        self.trades.apply_batch(batch)?;
        Ok(())
    }

    /// Query trade records matching the filter.
    ///
    /// Uses the session/instance index prefix when a session is specified;
    /// falls back to a full scan otherwise.
    pub fn query_trades(&self, filter: &QueryFilter) -> Result<Vec<TradeRecord>, DbError> {
        let mut results = Vec::new();

        if let Some(prefix) = Self::trade_index_prefix(filter) {
            let prefix_bytes = prefix.into_bytes();
            for result in self.trades.range(prefix_bytes.clone()..) {
                let (key, value) = result?;
                if !key.starts_with(&prefix_bytes) {
                    break;
                }
                // Index value holds the primary record key.
                if let Some(raw) = self.trades.get(&value)? {
                    let record: TradeRecord = bincode::deserialize(raw.as_ref())
                        .map_err(|e| DbError::Serialization(e.to_string()))?;
                    if Self::matches_trade_filter(&record, filter) {
                        results.push(record);
                    }
                }
            }
        } else {
            for result in self.trades.iter() {
                let (key, value) = result?;
                if Self::is_index_key(&key) {
                    continue;
                }
                let record: TradeRecord = bincode::deserialize(value.as_ref())
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                if Self::matches_trade_filter(&record, filter) {
                    results.push(record);
                }
            }
        }

        Ok(results)
    }

    // ------- Portfolio -------

    /// Insert a portfolio record (record + index atomically, same tree).
    pub fn write_portfolio(&self, record: &PortfolioRecord) -> Result<(), DbError> {
        self.ensure_writable()?;
        let key = Self::portfolio_key(record);
        let encoded = bincode::serialize(record)
            .map_err(|e| DbError::Serialization(e.to_string()))?;
        let ikey = Self::portfolio_index_key(record);

        let mut batch = DbBatch::default();
        batch.insert(key.clone(), encoded);
        batch.insert(ikey, key);
        self.portfolio.apply_batch(batch)?;
        Ok(())
    }

    /// Query portfolio records matching the filter (indexed when possible).
    pub fn query_portfolio(&self, filter: &QueryFilter) -> Result<Vec<PortfolioRecord>, DbError> {
        let mut results = Vec::new();

        if let Some(prefix) = Self::portfolio_index_prefix(filter) {
            let prefix_bytes = prefix.into_bytes();
            for result in self.portfolio.range(prefix_bytes.clone()..) {
                let (key, value) = result?;
                if !key.starts_with(&prefix_bytes) {
                    break;
                }
                if let Some(raw) = self.portfolio.get(&value)? {
                    let record: PortfolioRecord = bincode::deserialize(raw.as_ref())
                        .map_err(|e| DbError::Serialization(e.to_string()))?;
                    if Self::matches_portfolio_filter(&record, filter) {
                        results.push(record);
                    }
                }
            }
        } else {
            for result in self.portfolio.iter() {
                let (key, value) = result?;
                if Self::is_index_key(&key) {
                    continue;
                }
                let record: PortfolioRecord = bincode::deserialize(value.as_ref())
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                if Self::matches_portfolio_filter(&record, filter) {
                    results.push(record);
                }
            }
        }

        Ok(results)
    }

    // ------- Batched writes -------

    /// Atomically write multiple trade and portfolio records in one batch.
    ///
    /// Records and their index entries are applied atomically per domain
    /// (trades and portfolio are separate logical trees).
    pub fn write_batch(
        &self,
        trades: &[TradeRecord],
        portfolios: &[PortfolioRecord],
    ) -> Result<(), DbError> {
        self.ensure_writable()?;
        if trades.is_empty() && portfolios.is_empty() {
            return Ok(());
        }

        if !trades.is_empty() {
            let mut batch = DbBatch::default();
            for record in trades {
                let key = Self::trade_key(record);
                let encoded = bincode::serialize(record)
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                let ikey = Self::trade_index_key(record);
                batch.insert(key.clone(), encoded);
                batch.insert(ikey, key);
            }
            self.trades.apply_batch(batch)?;
        }

        if !portfolios.is_empty() {
            let mut batch = DbBatch::default();
            for record in portfolios {
                let key = Self::portfolio_key(record);
                let encoded = bincode::serialize(record)
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                let ikey = Self::portfolio_index_key(record);
                batch.insert(key.clone(), encoded);
                batch.insert(ikey, key);
            }
            self.portfolio.apply_batch(batch)?;
        }

        Ok(())
    }

    // ------- Sessions -------

    fn session_key(id: &str) -> Vec<u8> {
        format!("session|{}", id).into_bytes()
    }

    fn summary_key(id: &str) -> Vec<u8> {
        format!("summary|{}", id).into_bytes()
    }

    /// Create a new session. Fails if the id already exists.
    pub fn create_session(&self, id: impl Into<String>) -> Result<SessionMeta, DbError> {
        let id = id.into();
        let key = Self::session_key(&id);
        if self.metadata.get(&key)?.is_some() {
            return Err(DbError::SessionExists { id });
        }
        let meta = SessionMeta {
            id,
            started_at: Utc::now(),
            ended_at: None,
            active: true,
        };
        let encoded = bincode::serialize(&meta)
            .map_err(|e| DbError::Serialization(e.to_string()))?;
        self.metadata.insert(key, encoded)?;
        Ok(meta)
    }

    /// Look up one session by id.
    pub fn get_session(&self, id: &str) -> Result<Option<SessionMeta>, DbError> {
        match self.metadata.get(Self::session_key(id))? {
            Some(raw) => {
                let meta: SessionMeta = bincode::deserialize(raw.as_ref())
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                Ok(Some(meta))
            }
            None => Ok(None),
        }
    }

    /// All sessions, ordered by id.
    pub fn list_sessions(&self) -> Result<Vec<SessionMeta>, DbError> {
        let prefix = b"session|";
        let mut sessions = Vec::new();
        for result in self.metadata.range(prefix.as_slice()..) {
            let (key, value) = result?;
            if !key.starts_with(prefix) {
                break;
            }
            let meta: SessionMeta = bincode::deserialize(value.as_ref())
                .map_err(|e| DbError::Serialization(e.to_string()))?;
            sessions.push(meta);
        }
        Ok(sessions)
    }

    /// Finalize a session: mark it inactive and persist the session summary.
    /// Metadata + summary are written atomically (metadata tree batch).
    pub fn finalize_session(&self, summary: &SessionSummary) -> Result<SessionMeta, DbError> {
        self.ensure_writable()?;
        let id = &summary.session_id;
        let raw = self
            .metadata
            .get(Self::session_key(id))?
            .ok_or_else(|| DbError::SessionNotFound { id: id.clone() })?;
        let mut meta: SessionMeta = bincode::deserialize(raw.as_ref())
            .map_err(|e| DbError::Serialization(e.to_string()))?;
        meta.active = false;
        meta.ended_at = Some(summary.finalized_at);

        let mut batch = DbBatch::default();
        let meta_encoded = bincode::serialize(&meta)
            .map_err(|e| DbError::Serialization(e.to_string()))?;
        let sum_encoded = bincode::serialize(summary)
            .map_err(|e| DbError::Serialization(e.to_string()))?;
        batch.insert(Self::session_key(id), meta_encoded);
        batch.insert(Self::summary_key(id), sum_encoded);
        self.metadata.apply_batch(batch)?;
        Ok(meta)
    }

    /// The final summary of a session, if it was finalized.
    pub fn session_summary(&self, id: &str) -> Result<Option<SessionSummary>, DbError> {
        match self.metadata.get(Self::summary_key(id))? {
            Some(raw) => {
                let summary: SessionSummary = bincode::deserialize(raw.as_ref())
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                Ok(Some(summary))
            }
            None => Ok(None),
        }
    }

    // ------- Integrity -------

    /// Scan all records for decodability. Corruption is reported explicitly,
    /// never hidden: `IntegrityReport.corrupt_keys` lists every bad key.
    pub fn integrity_check(&self) -> Result<IntegrityReport, DbError> {
        let mut report = IntegrityReport::default();

        for result in self.trades.iter() {
            let (key, value) = result?;
            if Self::is_index_key(&key) {
                continue;
            }
            match bincode::deserialize::<TradeRecord>(value.as_ref()) {
                Ok(_) => report.trade_records += 1,
                Err(_) => report.corrupt_keys.push(format!(
                    "trades:{}",
                    String::from_utf8_lossy(&key)
                )),
            }
        }

        for result in self.portfolio.iter() {
            let (key, value) = result?;
            if Self::is_index_key(&key) {
                continue;
            }
            match bincode::deserialize::<PortfolioRecord>(value.as_ref()) {
                Ok(_) => report.portfolio_records += 1,
                Err(_) => report.corrupt_keys.push(format!(
                    "portfolio:{}",
                    String::from_utf8_lossy(&key)
                )),
            }
        }

        report.active_sessions = self
            .list_sessions()?
            .iter()
            .filter(|s| s.active)
            .count();
        Ok(report)
    }

    // ------- Private helpers -------

    /// Test-only: toggle injected storage failures (all record writes fail).
    #[cfg(test)]
    pub fn inject_write_failure(&self, on: bool) {
        self.fail_writes
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    fn ensure_writable(&self) -> Result<(), DbError> {
        #[cfg(test)]
        if self
            .fail_writes
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err(DbError::Injected);
        }
        #[cfg(not(test))]
        {
            Ok(())
        }
        #[cfg(test)]
        {
            Ok(())
        }
    }

    fn trade_key(record: &TradeRecord) -> Vec<u8> {
        let ts = record.event_time.timestamp_nanos_opt().unwrap_or(0);
        format!(
            "trades|{}|{}|{}|{:020}",
            record.session_id,
            record.instance.composite(),
            record.symbol,
            ts
        )
        .into_bytes()
    }

    fn portfolio_key(record: &PortfolioRecord) -> Vec<u8> {
        let ts = record.timestamp.timestamp_nanos_opt().unwrap_or(0);
        format!(
            "portfolio|{}|{}|{}|{:020}",
            record.session_id,
            record.instance.composite(),
            record.asset_id,
            ts
        )
        .into_bytes()
    }

    /// In-tree index key for a trade record: `ix|{session}|{composite}|{ts}`.
    /// Value = primary record key. Lives in the same tree so record + index
    /// writes are atomic.
    fn trade_index_key(record: &TradeRecord) -> Vec<u8> {
        let ts = record.event_time.timestamp_nanos_opt().unwrap_or(0);
        format!(
            "ix|{}|{}|{:020}",
            record.session_id,
            record.instance.composite(),
            ts
        )
        .into_bytes()
    }

    fn portfolio_index_key(record: &PortfolioRecord) -> Vec<u8> {
        let ts = record.timestamp.timestamp_nanos_opt().unwrap_or(0);
        format!(
            "ix|{}|{}|{:020}",
            record.session_id,
            record.instance.composite(),
            ts
        )
        .into_bytes()
    }

    fn is_index_key(key: &[u8]) -> bool {
        key.starts_with(b"ix|")
    }

    /// Index prefix for narrowing trade candidates from a filter.
    fn trade_index_prefix(filter: &QueryFilter) -> Option<String> {
        match (
            filter.session_id.as_ref(),
            filter.strategy_id.as_ref(),
            filter.asset_id.as_ref(),
        ) {
            (Some(s), Some(st), Some(a)) => Some(format!("ix|{}|{}@{}|", s, st, a)),
            (Some(s), Some(st), None) => Some(format!("ix|{}|{}@", s, st)),
            (Some(s), None, _) => Some(format!("ix|{}|", s)),
            _ => None,
        }
    }

    fn portfolio_index_prefix(filter: &QueryFilter) -> Option<String> {
        Self::trade_index_prefix(filter)
    }

    fn matches_trade_filter(record: &TradeRecord, filter: &QueryFilter) -> bool {
        if let Some(ref sid) = filter.session_id {
            if record.session_id != *sid {
                return false;
            }
        }
        if let Some(ref strat) = filter.strategy_id {
            if record.instance.strategy_id != *strat {
                return false;
            }
        }
        if let Some(ref asset) = filter.asset_id {
            if record.instance.asset_id != *asset && record.symbol != *asset {
                return false;
            }
        }
        if let Some(start) = filter.start_time {
            if record.event_time < start {
                return false;
            }
        }
        if let Some(end) = filter.end_time {
            if record.event_time > end {
                return false;
            }
        }
        true
    }

    fn matches_portfolio_filter(record: &PortfolioRecord, filter: &QueryFilter) -> bool {
        if let Some(ref sid) = filter.session_id {
            if record.session_id != *sid {
                return false;
            }
        }
        if let Some(ref strat) = filter.strategy_id {
            if record.instance.strategy_id != *strat {
                return false;
            }
        }
        if let Some(ref asset) = filter.asset_id {
            if record.asset_id != *asset {
                return false;
            }
        }
        if let Some(start) = filter.start_time {
            if record.timestamp < start {
                return false;
            }
        }
        if let Some(end) = filter.end_time {
            if record.timestamp > end {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{PortfolioRecord, SchemaVersion, SessionSummary, StrategyInstance};
    use crate::schema::TradeIntent;
    use chrono::{TimeZone, Utc};
    use std::env;
    use std::fs;

    fn temp_db_path() -> std::path::PathBuf {
        // Unique per call so parallel tests never share (or delete) each other's DB.
        env::temp_dir().join(format!("uets_db_test_{}_{:x}", std::process::id(), uuid::Uuid::new_v4().as_u128()))
    }

    fn sample_trade() -> TradeRecord {
        TradeRecord {
            schema: SchemaVersion::default(),
            session_id: "session_1".to_string(),
            instance: StrategyInstance::new("ofmr", "btcusdt"),
            symbol: "BTCUSDT".to_string(),
            event_time: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            feature_inputs: serde_json::json!({"price": 100.0}).to_string(),
            decision: TradeIntent::Long,
            decision_metadata: serde_json::json!({}).to_string(),
            intent: TradeIntent::Long,
            order_events: Vec::new(),
            entry_price: Some(100.0),
            exit_price: None,
            tp_price: Some(100.5),
            sl_price: Some(99.5),
            vertical_time: Some(
                Utc.with_ymd_and_hms(2026, 1, 1, 0, 5, 0).unwrap(),
            ),
            fill_qty: 1.0,
            fees: 0.1,
            realized_pnl: 0.0,
            model_version: "v1.0".to_string(),
            config_id: "config_1".to_string(),
        }
    }

    #[test]
    fn open_and_write_trade() {
        let path = temp_db_path();
        let db = UetsDb::open(&path).unwrap();
        let trade = sample_trade();
        db.write_trade(&trade).unwrap();
        db.flush().unwrap();

        let filter = QueryFilter::new().session("session_1");
        let results = db.query_trades(&filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].symbol, "BTCUSDT");
        assert_eq!(results[0].entry_price, Some(100.0));

        let _ = db.shutdown();
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn write_and_query_portfolio() {
        let path = temp_db_path();
        let db = UetsDb::open(&path).unwrap();
        let record = PortfolioRecord {
            schema: SchemaVersion::default(),
            session_id: "session_1".to_string(),
            instance: StrategyInstance::new("ofmr", "btcusdt"),
            asset_id: "btcusdt".to_string(),
            timestamp: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            allocated_capital: 10_000.0,
            cash: 10_000.0,
            equity: 10_500.0,
            realized_pnl: 0.0,
            unrealized_pnl: 500.0,
            cumulative_return: 0.05,
            sortino: 1.5,
            drawdown: 0.02,
            max_drawdown: 0.02,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            margin_used: 0.0,
            available_margin: 10_500.0,
            effective_leverage: 0.0,
            fees_paid: 0.0,
            trade_count: 0,
            win_count: 0,
            loss_count: 0,
            win_rate: 0.0,
        };
        db.write_portfolio(&record).unwrap();
        db.flush().unwrap();

        let filter = QueryFilter::new().session("session_1");
        let results = db.query_portfolio(&filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].equity, 10_500.0);

        let _ = db.shutdown();
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn batch_write() {
        let path = temp_db_path();
        let db = UetsDb::open(&path).unwrap();
        let trade = sample_trade();
        db.write_batch(&[trade], &[]).unwrap();
        db.flush().unwrap();

        let filter = QueryFilter::new().session("session_1");
        let results = db.query_trades(&filter).unwrap();
        assert_eq!(results.len(), 1);

        let _ = db.shutdown();
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn session_lifecycle_create_finalize_query() {
        let path = temp_db_path();
        let db = UetsDb::open(&path).unwrap();

        let meta = db.create_session("session_1").unwrap();
        assert!(meta.active);
        assert!(db.get_session("session_1").unwrap().is_some());

        // Duplicate id rejected — sessions are never overwritten silently.
        assert!(matches!(
            db.create_session("session_1"),
            Err(DbError::SessionExists { .. })
        ));

        let summary = SessionSummary {
            schema: SchemaVersion::default(),
            session_id: "session_1".to_string(),
            finalized_at: Utc::now(),
            trade_count: 3,
            portfolio_snapshots: 10,
            final_equity: 10_500.0,
            cumulative_return: 0.05,
            sortino: 1.2,
            max_drawdown: 0.02,
        };
        let finalized = db.finalize_session(&summary).unwrap();
        assert!(!finalized.active);
        assert!(finalized.ended_at.is_some());

        let stored = db.session_summary("session_1").unwrap().unwrap();
        assert_eq!(stored.trade_count, 3);
        assert!(!db.get_session("session_1").unwrap().unwrap().active);
        assert_eq!(db.list_sessions().unwrap().len(), 1);

        // Finalizing an unknown session is an explicit error, not a no-op.
        assert!(matches!(
            db.finalize_session(&SessionSummary {
                session_id: "missing".to_string(),
                ..summary
            }),
            Err(DbError::SessionNotFound { .. })
        ));

        let _ = db.shutdown();
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn combo_filters_return_correct_records() {
        let path = temp_db_path();
        let db = UetsDb::open(&path).unwrap();

        let base = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let mut trades = Vec::new();
        for (sess, strat, asset, sym, minute) in [
            ("session_1", "ofmr", "btcusdt", "BTCUSDT", 0),
            ("session_1", "ofmr", "ethusdt", "ETHUSDT", 1),
            ("session_1", "moe", "btcusdt", "BTCUSDT", 2),
            ("session_2", "ofmr", "btcusdt", "BTCUSDT", 0),
        ] {
            let mut t = sample_trade();
            t.session_id = sess.to_string();
            t.instance = StrategyInstance::new(strat, asset);
            t.symbol = sym.to_string();
            t.event_time = base + chrono::Duration::minutes(minute);
            trades.push(t);
        }
        db.write_batch(&trades, &[]).unwrap();

        // Session only (indexed prefix scan).
        let r = db
            .query_trades(&QueryFilter::new().session("session_1"))
            .unwrap();
        assert_eq!(r.len(), 3);

        // Session + strategy.
        let r = db
            .query_trades(&QueryFilter::new().session("session_1").strategy("ofmr"))
            .unwrap();
        assert_eq!(r.len(), 2);

        // Session + strategy + asset.
        let r = db
            .query_trades(
                &QueryFilter::new()
                    .session("session_1")
                    .strategy("ofmr")
                    .asset("btcusdt"),
            )
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].symbol, "BTCUSDT");

        // Asset only (no session -> full scan path).
        let r = db.query_trades(&QueryFilter::new().asset("btcusdt")).unwrap();
        assert_eq!(r.len(), 3);

        // Strategy only (crosses sessions).
        let r = db
            .query_trades(&QueryFilter::new().strategy("ofmr"))
            .unwrap();
        assert_eq!(r.len(), 3);

        // Time range within a session.
        let r = db
            .query_trades(
                &QueryFilter::new().session("session_1").time_range(
                    base + chrono::Duration::minutes(1),
                    base + chrono::Duration::minutes(2),
                ),
            )
            .unwrap();
        assert_eq!(r.len(), 2);

        let _ = db.shutdown();
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn restart_recovers_written_records() {
        let path = temp_db_path();
        {
            let db = UetsDb::open(&path).unwrap();
            db.write_trade(&sample_trade()).unwrap();
            db.create_session("session_1").unwrap();
            db.flush().unwrap();
            db.shutdown().unwrap();
        }

        // Reopen: records, indexes and sessions survive.
        let db = UetsDb::open(&path).unwrap();
        let results = db
            .query_trades(&QueryFilter::new().session("session_1"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(db.get_session("session_1").unwrap().unwrap().active);

        let report = db.integrity_check().unwrap();
        assert!(report.is_clean());
        assert_eq!(report.trade_records, 1);
        assert_eq!(report.active_sessions, 1);

        let _ = db.shutdown();
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn integrity_reports_corrupt_records() {
        let path = temp_db_path();
        let db = UetsDb::open(&path).unwrap();
        db.write_trade(&sample_trade()).unwrap();

        // Write garbage directly (bypassing the API) to simulate corruption.
        db.trades
            .insert(
                &b"trades|session_1|ofmr@btcusdt|BTCUSDT|99"[..],
                &b"not-a-valid-record"[..],
            )
            .unwrap();

        let report = db.integrity_check().unwrap();
        assert_eq!(report.trade_records, 1);
        assert_eq!(report.corrupt_keys.len(), 1, "corruption must be reported");
        assert!(!report.is_clean());

        let _ = db.shutdown();
        let _ = fs::remove_dir_all(&path);
    }
}
