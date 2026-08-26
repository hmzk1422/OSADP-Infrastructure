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

use crate::exchange::{MarketBookTop, MarketTrade};
use rocksdb::{ColumnFamilyDescriptor, DB, Direction, IteratorMode, Options};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const CF_TRADES: &str = "trades";
const CF_ORDERBOOK: &str = "orderbook_top";
const CF_PNL: &str = "pnl";
const CF_FEATURES: &str = "features";

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

/// One closed micro-bar of strategy features, persisted for the HTML report's
/// indicator pane. Written once per second while the feed is live.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureSample {
    pub ts_ms: i64,
    pub rsi_2: f64,
    pub win_prob: f64,
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

    fn column_family_descriptors() -> Vec<ColumnFamilyDescriptor> {
        vec![
            ColumnFamilyDescriptor::new(CF_TRADES, Options::default()),
            ColumnFamilyDescriptor::new(CF_ORDERBOOK, Options::default()),
            ColumnFamilyDescriptor::new(CF_PNL, Options::default()),
            ColumnFamilyDescriptor::new(CF_FEATURES, Options::default()),
        ]
    }

    fn open(path: &str) -> Result<DB, DbError> {
        if let Some(parent) = Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent); // best effort; DB::open surfaces real errors
        }

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        match DB::open_cf_descriptors(&db_opts, path, Self::column_family_descriptors()) {
            Ok(db) => Ok(db),
            Err(open_err) => Self::recover_and_reopen(&db_opts, path, open_err),
        }
    }

    /// Attempt to recover a store whose metadata got corrupted by an unclean
    /// quit (e.g. "Corruption: CURRENT file corrupted"). Recovery is staged to
    /// lose as little data as possible:
    ///
    ///   1. Point CURRENT back at the newest MANIFEST in the directory (LF-only,
    ///      no stray CR). This is the common, cheap recovery — no data loss.
    ///   2. Only if that still won't open, run RocksDB's `repair`, which
    ///      rebuilds the metadata from what it can verify (may move stale files
    ///      into a `lost/` subdirectory, but restores the store).
    ///
    fn recover_and_reopen(
        db_opts: &Options,
        path: &str,
        first_err: rocksdb::Error,
    ) -> Result<DB, DbError> {
        if Self::try_fix_current(path) {
            eprintln!("[db] repaired CURRENT pointer at {path} — reopening");
            if let Ok(db) =
                DB::open_cf_descriptors(db_opts, path, Self::column_family_descriptors())
            {
                return Ok(db);
            }
        }

        eprintln!("[db] open failed ({first_err}) — attempting one-shot RocksDB repair at {path}");
        match DB::repair(db_opts, path) {
            Ok(()) => {
                eprintln!("[db] repair OK — reopening {path}");
                DB::open_cf_descriptors(db_opts, path, Self::column_family_descriptors())
                    .map_err(DbError::from)
            }
            Err(repair_err) => {
                eprintln!("[db] repair failed: {repair_err}");
                Err(DbError::RocksDb(first_err))
            }
        }
    }

    /// Best-effort rewrite of the `CURRENT` file so it points (LF-only) at the
    /// highest-numbered MANIFEST present in the store directory.
    ///
    /// RocksDB resolves its current manifest by reading this one line; a stray
    /// `\r` (CRLF, seen after Windows quits) breaks that lookup and is reported
    /// as "Corruption: CURRENT file corrupted" even though the MANIFEST + every
    /// data file is intact. Returns true if the file was rewritten or requires
    /// a reopen.
    fn try_fix_current(path: &str) -> bool {
        let root = Path::new(path);
        let current_path = root.join("CURRENT");

        // Find the highest-numbered MANIFEST-<seq> present.
        let mut seqs: Vec<u64> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(stripped) = name.strip_prefix("MANIFEST-") {
                    if let Ok(seq) = stripped.parse::<u64>() {
                        seqs.push(seq);
                    }
                }
            }
        }
        let Some(&max_seq) = seqs.iter().max() else {
            return false; // no manifest at all — leave it to `::repair`
        };

        let line = format!("MANIFEST-{max_seq:06}");

        // If CURRENT is already a clean LF-only line naming that manifest,
        // there is nothing to fix.
        if let Ok(cur) = std::fs::read(&current_path) {
            if let Ok(text) = std::str::from_utf8(&cur) {
                let trimmed = text.trim_end_matches(|c| c == '\r' || c == '\n');
                let clean = !cur.ends_with(b"\r\n");
                if clean && trimmed == line {
                    return false;
                }
            }
        }

        let _ = std::fs::write(&current_path, format!("{line}\n").as_bytes());
        true
    }

    /// Flush all pending writes to disk and sync the write-ahead log. Called
    /// during graceful shutdown so the SST / MANIFEST / CURRENT files are
    /// durably updated before the handle is dropped — a hard kill between
    /// writes is exactly what corrupts `CURRENT`.
    pub fn flush(&self) -> Result<(), DbError> {
        // `DB::flush()` (default FlushOptions, wait = true) writes every
        // column family's memtable out to an SST file.
        self.db.flush().map_err(DbError::from)?;
        // Force the WAL to disk so committed writes are durable even if the
        // process dies before the memtable is flushed again.
        self.db.flush_wal(true).map_err(DbError::from)?;
        Ok(())
    }

    /// Persists a single trade print. Write failures are logged and swallowed
    /// rather than propagated -- a transient local I/O error shouldn't take
    /// down a live market data feed.
    pub fn record_trade(&self, symbol: &str, trade: MarketTrade) {
        if let Err(e) = self.try_record_trade(symbol, &trade) {
            eprintln!(
                "[db] failed to write trade {symbol} @ {}: {e}",
                trade.timestamp
            );
        }
    }

    /// Persists a single L1 order book snapshot. Same fire-and-forget error
    /// handling as `record_trade`.
    pub fn record_orderbook_top(&self, symbol: &str, top: MarketBookTop) {
        if let Err(e) = self.try_record_orderbook_top(symbol, &top) {
            eprintln!(
                "[db] failed to write orderbook top {symbol} @ {}: {e}",
                top.timestamp
            );
        }
    }

    /// Returns all stored trades for `symbol`, oldest first.
    pub fn iter_trades(&self, symbol: &str) -> Result<Vec<MarketTrade>, DbError> {
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

    /// Downsampled per-second last-price series for charting: walks the trades
    /// column family in key order (chronological) and keeps the LAST print of
    /// each 1-second bucket — i.e. the bucket close. Memory-bounded at O(buckets)
    /// instead of O(trade prints), so a multi-million-print session still yields
    /// a compact series. Returns `(bucket_start_ms, close)` pairs, oldest first.
    pub fn price_series_1s(&self, symbol: &str) -> Result<Vec<(i64, f64)>, DbError> {
        let cf = self.cf(CF_TRADES)?;
        let prefix = Self::symbol_prefix(symbol);
        let mut out: Vec<(i64, f64)> = Vec::new();
        for item in self
            .db
            .iterator_cf(cf, IteratorMode::From(&prefix, Direction::Forward))
        {
            let (key, value) = item?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            let trade: MarketTrade = serde_json::from_slice(&value)?;
            if trade.price <= 0.0 {
                continue; // ignore malformed prints rather than corrupting the chart
            }
            let bucket = trade.timestamp - trade.timestamp.rem_euclid(1000);
            match out.last_mut() {
                Some((ts, px)) if *ts == bucket => *px = trade.price,
                _ => out.push((bucket, trade.price)),
            }
        }
        Ok(out)
    }

    /// Returns all stored L1 snapshots for `symbol`, oldest first.
    pub fn iter_orderbook(&self, symbol: &str) -> Result<Vec<MarketBookTop>, DbError> {
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

    fn try_record_trade(&self, symbol: &str, trade: &MarketTrade) -> Result<(), DbError> {
        let cf = self.cf(CF_TRADES)?;
        let key = Self::trade_key(symbol, trade.timestamp, &trade.id, self.next_seq());
        let value = serde_json::to_vec(trade)?;
        self.db.put_cf(cf, key, value)?;
        Ok(())
    }

    fn try_record_orderbook_top(&self, symbol: &str, top: &MarketBookTop) -> Result<(), DbError> {
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
            eprintln!(
                "[db] failed to write pnl snapshot {symbol} @ {}: {e}",
                snap.ts_ms
            );
        }
    }

    /// Persist a closed-trade record. Fire-and-forget.
    pub fn record_closed_trade(&self, symbol: &str, trade: &ClosedTrade) {
        if let Err(e) = self.try_record_closed_trade(symbol, trade) {
            eprintln!(
                "[db] failed to write closed trade {symbol} @ {}: {e}",
                trade.exit_ms
            );
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

    // ---- strategy feature history (report indicator panes) ------------------

    /// Persist one closed micro-bar of strategy features. Fire-and-forget.
    pub fn record_feature(&self, symbol: &str, sample: &FeatureSample) {
        if let Err(e) = self.try_record_feature(symbol, sample) {
            eprintln!(
                "[db] failed to write feature sample {symbol} @ {}: {e}",
                sample.ts_ms
            );
        }
    }

    fn try_record_feature(&self, symbol: &str, sample: &FeatureSample) -> Result<(), DbError> {
        let cf = self.cf(CF_FEATURES)?;
        let key = Self::feature_key(symbol, sample.ts_ms, self.next_seq());
        let value = serde_json::to_vec(sample)?;
        self.db.put_cf(cf, key, value)?;
        Ok(())
    }

    /// Read all feature samples for `symbol`, oldest first (one per second).
    pub fn iter_features(&self, symbol: &str) -> Result<Vec<FeatureSample>, DbError> {
        let cf = self.cf(CF_FEATURES)?;
        let prefix = Self::feature_prefix(symbol);
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

    fn feature_prefix(symbol: &str) -> Vec<u8> {
        let mut p = Self::symbol_prefix(symbol);
        p.extend_from_slice(b"feat\x00");
        p
    }

    fn feature_key(symbol: &str, ts: i64, seq: u64) -> Vec<u8> {
        let mut key = Self::feature_prefix(symbol);
        key.extend_from_slice(&ts.to_be_bytes());
        key.push(0x00);
        key.extend_from_slice(&seq.to_be_bytes());
        key
    }
}

/// Last-chance flush before the RocksDB handle/threads are torn down.
/// This is the safety net that makes abrupt drops (strategy dropped without an
/// explicit `stop()`, early `return`s, etc.) still land their data durably on
/// disk before the DB is closed. Errors cannot be propagated from `drop`, so
/// they are logged at the interactive level.
impl Drop for DbWriter {
    fn drop(&mut self) {
        if let Err(e) = self.db.flush() {
            eprintln!("[db] final flush during close failed: {e}");
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn test_opts() -> Options {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts
    }

    fn temp_db_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "imc_rocksdb_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Build a healthy store with one row, flushed + closed cleanly.
    fn seed_healthy_store(path: &str) {
        let db = DB::open_cf_descriptors(&test_opts(), path, DbWriter::column_family_descriptors())
            .expect("initial open");
        let cf = db.cf_handle(CF_TRADES).expect("trades cf exists");
        db.put_cf(cf, b"k1", b"v1").expect("put");
        db.flush().expect("flush");
        db.flush_wal(true).expect("flush_wal");
        // db dropped -> RocksDB closes and rewrites CURRENT/MANIFEST
    }

    /// Regression test for "Corruption: CURRENT file corrupted" caused by a
    /// stray CRLF line ending in `CURRENT` (seen after unclean Windows quits).
    /// `DbWriter::open` must point CURRENT back at the real MANIFEST and
    /// preserve all data — no `repair`, no data loss.
    #[test]
    fn open_fixes_crlf_current() {
        let dir = temp_db_dir("crlf");
        let path = dir.to_str().expect("temp path is utf8");
        seed_healthy_store(path);

        // Corrupt CURRENT with CRLF, exactly as observed in the wild.
        let good = std::fs::read(dir.join("CURRENT")).expect("current exists");
        let text = std::str::from_utf8(&good)
            .expect("current utf8")
            .trim_end_matches(|c| c == '\r' || c == '\n');
        std::fs::write(dir.join("CURRENT"), format!("{text}\r\n").as_bytes())
            .expect("corrupt CURRENT");

        let db = DbWriter::open(path).expect("open after CURRENT fix");
        let cf = db.cf_handle(CF_TRADES).expect("trades cf survived");
        let got = db.get_cf(cf, b"k1").expect("read after recovery");
        assert_eq!(
            got.as_deref(),
            Some(&b"v1"[..]),
            "pre-corruption data must survive"
        );

        drop(db);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Fallback path: if the MANIFEST itself is gone (no CURRENT fix possible),
    /// `DB::repair` must still produce an openable store rather than leaving
    /// the caller with "running without persistence".
    #[test]
    fn open_falls_back_to_repair_when_manifest_missing() {
        let dir = temp_db_dir("repair");
        let path = dir.to_str().expect("temp path is utf8");
        seed_healthy_store(path);

        // Delete every MANIFEST so CURRENT cannot be repaired by pointing at one.
        for entry in std::fs::read_dir(&dir).expect("read dir") {
            let entry = entry.expect("entry");
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("MANIFEST-") {
                std::fs::remove_file(entry.path()).expect("remove manifest");
            }
        }
        // CURRENT now points at a manifest that no longer exists.
        std::fs::write(dir.join("CURRENT"), b"MANIFEST-999999\n").expect("write CURRENT");

        let db = DbWriter::open(path).expect("open after repair fallback");
        drop(db);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
