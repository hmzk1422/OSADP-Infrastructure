//! Local, embedded storage for Bybit trade prints and L1 order book snapshots,
//! backed by RocksDB. Replaces the previous TimescaleDB/Postgres writer --
//! there's no external database server to run, everything lives on disk next
//! to the process.
//!
//! Layout: three column families.
//!   - "trades":         one row per trade print
//!   - "orderbook_top":  one row per L1 (best bid/ask) snapshot
//!   - "pnl":            equity snapshots + closed-trade records for the
//!                       dry-run strategy
//!
//! Keys are `symbol \0 be_bytes(timestamp) \0 be_bytes(seq) [\0 trade_id]`.
//! Because RocksDB keeps keys in lexicographic (byte) order, and big-endian
//! encoding of a non-negative i64 sorts the same as the integer itself, this
//! means all rows for a given symbol come back in chronological order for
//! free -- handy for range scans / backtesting later. `seq` is a
//! process-local monotonic counter used purely as a tie-breaker so two writes
//! landing on the same millisecond can never collide and silently overwrite
//! each other.

use crate::bybit_ws::{BybitOrderBookTop, BybitTrade};
use rocksdb::{ColumnFamilyDescriptor, Direction, IteratorMode, Options, DB};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const CF_TRADES: &str = "trades";
const CF_ORDERBOOK: &str = "orderbook_top";
const CF_PNL: &str = "pnl";

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("rocksdb error: {0}")]
    RocksDb(#[from] rocksdb::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("column family '{0}' missing (this should never happen)")]
    MissingColumnFamily(&'static str),
}

/// Equity snapshot for the equity curve. Written to RocksDB after every
/// entry/exit so the curve survives process restarts.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PnLSnapshot {
    pub ts_ms: i64,
    pub equity: f64,
    pub n_trades: u64,
    pub n_wins: u64,
    pub n_losses: u64,
    pub total_r: f64,
    pub mid: f64,
    pub last_exit_ms: i64,
}

/// A closed-trade record persisted for the HTML trade log.
/// Mirrors `engine::TradeRecord` but with owned strings for serde.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClosedTrade {
    pub entry_ms: i64,
    pub exit_ms: i64,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub reason: String,
    pub qty: f64,
    pub net_usd: f64,
    pub r_multiple: f64,
    pub equity_after: f64,
    pub duration_s: f64,
}

pub struct DbWriter {
    db: DB,
    seq: AtomicU64,
}

impl DbWriter {
    /// Opens (or creates) a RocksDB database at `path`. The actual open runs
    /// on a blocking thread since it does blocking file I/O.
    pub async fn connect(path: &str) -> Result<Self, DbError> {
        let owned_path = path.to_string();
        let db = tokio::task::spawn_blocking(move || Self::open(&owned_path))
            .await
            .expect("rocksdb open task panicked")?;
        Ok(Self {
            db,
            seq: AtomicU64::new(0),
        })
    }

    fn open(path: &str) -> Result<DB, DbError> {
        if let Some(parent) = Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent); // best effort; DB::open surfaces real errors
        }

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

                let cfs = vec![
            ColumnFamilyDescriptor::new(CF_TRADES, Options::default()),
            ColumnFamilyDescriptor::new(CF_ORDERBOOK, Options::default()),
            ColumnFamilyDescriptor::new(CF_PNL, Options::default()),
        ];

        Ok(DB::open_cf_descriptors(&db_opts, path, cfs)?)
    }

    /// Persists a single trade print. Write failures are logged and swallowed
    /// rather than propagated -- a transient local I/O error shouldn't take
    /// down a live market data feed.
    pub fn record_trade(&self, symbol: &str, trade: BybitTrade) {
        if let Err(e) = self.try_record_trade(symbol, &trade) {
            eprintln!(
                "[db] failed to write trade {symbol} @ {}: {e}",
                trade.timestamp
            );
        }
    }

    /// Persists a single L1 order book snapshot. Same fire-and-forget error
    /// handling as `record_trade`.
    pub fn record_orderbook_top(&self, symbol: &str, top: BybitOrderBookTop) {
        if let Err(e) = self.try_record_orderbook_top(symbol, &top) {
            eprintln!(
                "[db] failed to write orderbook top {symbol} @ {}: {e}",
                top.timestamp
            );
        }
    }

    /// Returns all stored trades for `symbol`, oldest first.
    pub fn iter_trades(&self, symbol: &str) -> Result<Vec<BybitTrade>, DbError> {
        let cf = self.cf(CF_TRADES)?;
        let prefix = Self::symbol_prefix(symbol);
        let mut out = Vec::new();
        for item in self
            .db
            .iterator_cf(cf, IteratorMode::From(&prefix, Direction::Forward))
        {
            let (key, value) = item?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            out.push(serde_json::from_slice(&value)?);
        }
        Ok(out)
    }

    /// Returns all stored L1 snapshots for `symbol`, oldest first.
    pub fn iter_orderbook(&self, symbol: &str) -> Result<Vec<BybitOrderBookTop>, DbError> {
        let cf = self.cf(CF_ORDERBOOK)?;
        let prefix = Self::symbol_prefix(symbol);
        let mut out = Vec::new();
        for item in self
            .db
            .iterator_cf(cf, IteratorMode::From(&prefix, Direction::Forward))
        {
            let (key, value) = item?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            out.push(serde_json::from_slice(&value)?);
        }
        Ok(out)
    }

    // ---- internal helpers -------------------------------------------------

    fn try_record_trade(&self, symbol: &str, trade: &BybitTrade) -> Result<(), DbError> {
        let cf = self.cf(CF_TRADES)?;
        let key = Self::trade_key(symbol, trade.timestamp, &trade.id, self.next_seq());
        let value = serde_json::to_vec(trade)?;
        self.db.put_cf(cf, key, value)?;
        Ok(())
    }

    fn try_record_orderbook_top(
        &self,
        symbol: &str,
        top: &BybitOrderBookTop,
    ) -> Result<(), DbError> {
        let cf = self.cf(CF_ORDERBOOK)?;
        let key = Self::orderbook_key(symbol, top.timestamp, self.next_seq());
        let value = serde_json::to_vec(top)?;
        self.db.put_cf(cf, key, value)?;
        Ok(())
    }

    fn cf(&self, name: &'static str) -> Result<&rocksdb::ColumnFamily, DbError> {
        self.db
            .cf_handle(name)
            .ok_or(DbError::MissingColumnFamily(name))
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    fn symbol_prefix(symbol: &str) -> Vec<u8> {
        let mut p = symbol.as_bytes().to_vec();
        p.push(0x00);
        p
    }

    fn trade_key(symbol: &str, ts: i64, id: &str, seq: u64) -> Vec<u8> {
        let mut key = Self::symbol_prefix(symbol);
        key.extend_from_slice(&ts.to_be_bytes());
        key.push(0x00);
        key.extend_from_slice(&seq.to_be_bytes());
        key.push(0x00);
        key.extend_from_slice(id.as_bytes());
        key
    }

                fn orderbook_key(symbol: &str, ts: i64, seq: u64) -> Vec<u8> {
        let mut key = Self::symbol_prefix(symbol);
        key.extend_from_slice(&ts.to_be_bytes());
        key.push(0x00);
        key.extend_from_slice(&seq.to_be_bytes());
        key
    }

    // ---- PnL tracking -------------------------------------------------------

    /// Persist an equity snapshot. Fire-and-forget (errors logged, not propagated).
    pub fn record_pnl_snapshot(&self, symbol: &str, snap: &PnLSnapshot) {
        if let Err(e) = self.try_record_pnl_snapshot(symbol, snap) {
            eprintln!("[db] failed to write pnl snapshot {symbol} @ {}: {e}", snap.ts_ms);
        }
    }

    /// Persist a closed-trade record. Fire-and-forget.
    pub fn record_closed_trade(&self, symbol: &str, trade: &ClosedTrade) {
        if let Err(e) = self.try_record_closed_trade(symbol, trade) {
            eprintln!("[db] failed to write closed trade {symbol} @ {}: {e}", trade.exit_ms);
        }
    }

    /// Read all equity snapshots for `symbol`, oldest first.
    pub fn iter_pnl_snapshots(&self, symbol: &str) -> Result<Vec<PnLSnapshot>, DbError> {
        let cf = self.cf(CF_PNL)?;
        let prefix = Self::pnl_snapshot_prefix(symbol);
        let mut out = Vec::new();
        for item in self
            .db
            .iterator_cf(cf, IteratorMode::From(&prefix, Direction::Forward))
        {
            let (key, value) = item?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            out.push(serde_json::from_slice(&value)?);
        }
        Ok(out)
    }

    /// Read all closed-trade records for `symbol`, oldest first.
    pub fn iter_closed_trades(&self, symbol: &str) -> Result<Vec<ClosedTrade>, DbError> {
        let cf = self.cf(CF_PNL)?;
        let prefix = Self::pnl_trade_prefix(symbol);
        let mut out = Vec::new();
        for item in self
            .db
            .iterator_cf(cf, IteratorMode::From(&prefix, Direction::Forward))
        {
            let (key, value) = item?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            out.push(serde_json::from_slice(&value)?);
        }
        Ok(out)
    }

    fn try_record_pnl_snapshot(&self, symbol: &str, snap: &PnLSnapshot) -> Result<(), DbError> {
        let cf = self.cf(CF_PNL)?;
        let key = Self::pnl_snapshot_key(symbol, snap.ts_ms, self.next_seq());
        let value = serde_json::to_vec(snap)?;
        self.db.put_cf(cf, key, value)?;
        Ok(())
    }

    fn try_record_closed_trade(&self, symbol: &str, trade: &ClosedTrade) -> Result<(), DbError> {
        let cf = self.cf(CF_PNL)?;
        let key = Self::pnl_trade_key(symbol, trade.exit_ms, self.next_seq());
        let value = serde_json::to_vec(trade)?;
        self.db.put_cf(cf, key, value)?;
        Ok(())
    }

    fn pnl_snapshot_prefix(symbol: &str) -> Vec<u8> {
        let mut p = Self::symbol_prefix(symbol);
        p.extend_from_slice(b"snap\x00");
        p
    }

    fn pnl_trade_prefix(symbol: &str) -> Vec<u8> {
        let mut p = Self::symbol_prefix(symbol);
        p.extend_from_slice(b"trade\x00");
        p
    }

    fn pnl_snapshot_key(symbol: &str, ts: i64, seq: u64) -> Vec<u8> {
        let mut key = Self::pnl_snapshot_prefix(symbol);
        key.extend_from_slice(&ts.to_be_bytes());
        key.push(0x00);
        key.extend_from_slice(&seq.to_be_bytes());
        key
    }

    fn pnl_trade_key(symbol: &str, ts: i64, seq: u64) -> Vec<u8> {
        let mut key = Self::pnl_trade_prefix(symbol);
        key.extend_from_slice(&ts.to_be_bytes());
        key.push(0x00);
        key.extend_from_slice(&seq.to_be_bytes());
        key
    }
}