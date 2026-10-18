//! Asynchronous persistence — bounded queue, explicit backpressure, and
//! guaranteed handling of critical records.
//!
//! Design:
//! - One dedicated writer thread performs all disk I/O, so neither the
//!   strategy loop nor the reporting calculation path ever blocks on disk
//!   (except critical-item backpressure when the queue is saturated).
//! - The queue is bounded (`PersistenceConfig::queue_capacity`).
//! - Critical items (trade records, session summaries) are **never silently
//!   dropped**: on a full queue they apply backpressure (blocking send); on
//!   storage failure they are retried, then quarantined with a loud error
//!   log and a stats counter, retried again on later cycles, and finally
//!   reported as an error from [`PersistenceWorker::shutdown`].
//! - Non-critical items (portfolio snapshots) are dropped under saturation,
//!   but every drop is counted in stats and logged (rate-limited), so
//!   nothing is discarded silently.
//!
//! ```text
//! producers ──try_send──▶ bounded queue ──writer thread──▶ UetsDb
//!    (strategy/reporter)    (backpressure)    (batched writes, retries)
//! ```

use crate::schema::{PortfolioRecord, SessionSummary, TradeRecord};
use crate::store::{DbError, UetsDb};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Retry cadence after a failed write attempt.
const RETRY_DELAY: Duration = Duration::from_millis(25);
/// Attempts before an item is quarantined (≈ 1 s with [`RETRY_DELAY`]).
const MAX_ATTEMPTS: u32 = 40;

/// One item to persist.
#[derive(Clone, Debug)]
pub enum PersistItem {
    /// Critical: a trade/order record. Must never be lost.
    Trade(TradeRecord),
    /// Non-critical: a portfolio snapshot at reporting cadence.
    /// Droppable under saturation (but counted, never silent).
    Portfolio(PortfolioRecord),
    /// Critical: a finalized session summary. Must never be lost.
    SessionSummary(SessionSummary),
}

impl PersistItem {
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            PersistItem::Trade(_) | PersistItem::SessionSummary(_)
        )
    }

    fn describe(&self) -> String {
        match self {
            PersistItem::Trade(t) => format!("trade {}/{}", t.session_id, t.instance.composite()),
            PersistItem::Portfolio(p) => {
                format!("portfolio {}/{}", p.session_id, p.instance.composite())
            }
            PersistItem::SessionSummary(s) => format!("session summary {}", s.session_id),
        }
    }
}

/// Outcome of a submit attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// Queued without waiting.
    Accepted,
    /// Critical item queued only after waiting for space (backpressure).
    AcceptedBlocked,
    /// Non-critical item dropped because the queue was full — counted in
    /// [`PersistStats::dropped_snapshots`] and logged.
    Dropped,
}

/// Errors returned to producers.
#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("persistence is closed; {0} item(s) not accepted")]
    Closed(usize),
    #[error("shutdown finished with {critical} critical and {total} total item(s) unwritten")]
    Unwritten { critical: usize, total: usize },
}

/// Point-in-time persistence statistics (for monitoring/GUI).
#[derive(Clone, Debug, Default)]
pub struct PersistStats {
    /// Items currently waiting in the queue.
    pub queue_depth: usize,
    /// Items accepted into the queue (all kinds).
    pub enqueued: u64,
    /// Items durably written.
    pub written: u64,
    /// Failed write attempts (storage errors surfaced, not hidden).
    pub write_errors: u64,
    /// Portfolio snapshots dropped due to queue saturation.
    pub dropped_snapshots: u64,
    /// Items currently in quarantine (failed after all retries).
    pub quarantined: usize,
    /// Items that ever entered quarantine (some may have recovered).
    pub quarantined_total: u64,
    /// Latency of the most recent successful write.
    pub last_write_micros: u64,
    /// Total time spent writing (successful writes).
    pub total_write_micros: u64,
    /// Slowest successful write.
    pub max_write_micros: u64,
}

/// Configuration for the persistence writer.
#[derive(Clone, Debug)]
pub struct PersistenceConfig {
    /// Bounded queue capacity.
    pub queue_capacity: usize,
    /// Maximum items coalesced into one write cycle.
    pub batch_max: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 4096,
            batch_max: 128,
        }
    }
}


#[derive(Default)]
struct StatsAtomics {
    enqueued: AtomicU64,
    written: AtomicU64,
    write_errors: AtomicU64,
    dropped_snapshots: AtomicU64,
    quarantined: AtomicUsize,
    quarantined_total: AtomicU64,
    last_write_micros: AtomicU64,
    total_write_micros: AtomicU64,
    max_write_micros: AtomicU64,
}

struct Shared {
    /// `None` once persistence is closed (sender dropped).
    tx: Mutex<Option<async_channel::Sender<PersistItem>>>,
    rx: async_channel::Receiver<PersistItem>,
    quarantine: Mutex<VecDeque<PersistItem>>,
    stats: StatsAtomics,
}

impl Shared {
    fn push_quarantine(&self, items: impl Iterator<Item = PersistItem>) {
        let mut q = self.quarantine.lock().unwrap();
        for item in items {
            tracing::error!(item = %item.describe(), "persistence: write failed after all retries; quarantined (retained, will be retried)");
            q.push_back(item);
            self.stats.quarantined.fetch_add(1, Ordering::Relaxed);
            self.stats.quarantined_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_write(&self, items: usize, elapsed: Duration) {
        let micros = elapsed.as_micros() as u64;
        self.stats.written.fetch_add(items as u64, Ordering::Relaxed);
        self.stats.last_write_micros.store(micros, Ordering::Relaxed);
        self.stats
            .total_write_micros
            .fetch_add(micros, Ordering::Relaxed);
        self.stats.max_write_micros.fetch_max(micros, Ordering::Relaxed);
    }
}

/// Producer-side handle: enqueue items and read stats.
pub struct PersistenceHandle {
    shared: Arc<Shared>,
}

/// Read-only view of persistence stats, safe to hand to a GUI consumer.
/// Dropping it never affects persistence (GUI-close independence).
#[derive(Clone)]
pub struct StatsHandle {
    shared: Arc<Shared>,
}

impl StatsHandle {
    pub fn stats(&self) -> PersistStats {
        snapshot(&self.shared)
    }
}

fn snapshot(shared: &Shared) -> PersistStats {
    let s = &shared.stats;
    PersistStats {
        queue_depth: shared.rx.len(),
        enqueued: s.enqueued.load(Ordering::Relaxed),
        written: s.written.load(Ordering::Relaxed),
        write_errors: s.write_errors.load(Ordering::Relaxed),
        dropped_snapshots: s.dropped_snapshots.load(Ordering::Relaxed),
        quarantined: shared.quarantine.lock().unwrap().len(),
        quarantined_total: s.quarantined_total.load(Ordering::Relaxed),
        last_write_micros: s.last_write_micros.load(Ordering::Relaxed),
        total_write_micros: s.total_write_micros.load(Ordering::Relaxed),
        max_write_micros: s.max_write_micros.load(Ordering::Relaxed),
    }
}

impl PersistenceHandle {
    /// Current persistence statistics (queue depth, write latency, errors).
    pub fn stats(&self) -> PersistStats {
        snapshot(&self.shared)
    }

    /// A read-only stats view for external consumers (e.g. `uets_gui`).
    /// Dropping the returned handle does not affect the writer.
    pub fn stats_handle(&self) -> StatsHandle {
        StatsHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Submit an item for asynchronous persistence.
    ///
    /// - Critical items: queued; if the queue is full, blocks until space
    ///   becomes available (backpressure) instead of dropping. Returns
    ///   [`PersistError::Closed`] only after persistence has shut down —
    ///   the caller then retains the item itself.
    /// - Portfolio snapshots: never block; dropped when full, but the drop
    ///   is counted and logged (never silent).
    pub fn submit(&self, item: PersistItem) -> Result<SubmitOutcome, PersistError> {
        let critical = item.is_critical();
        let tx = self
            .shared
            .tx
            .lock()
            .unwrap()
            .clone()
            .ok_or(PersistError::Closed(1))?;

        match tx.try_send(item) {
            Ok(()) => {
                self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                Ok(SubmitOutcome::Accepted)
            }
            Err(async_channel::TrySendError::Full(item)) => {
                if critical {
                    match tx.send_blocking(item) {
                        Ok(()) => {
                            self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                            Ok(SubmitOutcome::AcceptedBlocked)
                        }
                        Err(_) => Err(PersistError::Closed(1)),
                    }
                } else {
                    let dropped = self
                        .shared
                        .stats
                        .dropped_snapshots
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    // Rate-limited logging: powers of two, never silent.
                    if dropped.is_power_of_two() {
                        tracing::warn!(
                            dropped_total = dropped,
                            "persistence queue full: portfolio snapshot dropped (counted in stats)"
                        );
                    }
                    Ok(SubmitOutcome::Dropped)
                }
            }
            Err(async_channel::TrySendError::Closed(_)) => Err(PersistError::Closed(1)),
        }
    }
}

/// Writer-thread handle.
pub struct PersistenceWorker {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl PersistenceWorker {
    /// Close the queue, drain it, flush quarantined items, and join the
    /// writer thread. Returns [`PersistError::Unwritten`] if any item could
    /// not be written (surfaced loudly — never silent).
    pub fn shutdown(mut self) -> Result<(), PersistError> {
        // Close the channel: drop the sender so the writer drains and exits.
        *self.shared.tx.lock().unwrap() = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let q = self.shared.quarantine.lock().unwrap();
        let total = q.len();
        let critical = q.iter().filter(|i| i.is_critical()).count();
        drop(q);
        if total > 0 {
            tracing::error!(
                critical,
                total,
                "persistence shutdown: items remain unwritten"
            );
            return Err(PersistError::Unwritten { critical, total });
        }
        Ok(())
    }
}

/// Spawn the asynchronous persistence writer over `db`.
///
/// Returns a producer handle and a worker. The worker must eventually be
/// [`PersistenceWorker::shutdown`] to drain and flush everything.
pub fn spawn(db: Arc<UetsDb>, config: PersistenceConfig) -> (PersistenceHandle, PersistenceWorker) {
    let (tx, rx) = async_channel::bounded(config.queue_capacity.max(1));
    let shared = Arc::new(Shared {
        tx: Mutex::new(Some(tx)),
        rx,
        quarantine: Mutex::new(VecDeque::new()),
        stats: StatsAtomics::default(),
    });

    let thread_shared = Arc::clone(&shared);
    let thread = std::thread::Builder::new()
        .name("uets-persistence".into())
        .spawn(move || writer_loop(db, thread_shared, config.batch_max.max(1)))
        .expect("failed to spawn persistence writer thread");

    (
        PersistenceHandle {
            shared: Arc::clone(&shared),
        },
        PersistenceWorker {
            shared,
            thread: Some(thread),
        },
    )
}

/// Attempt to write a batch of items; returns how many were written.
fn write_items(db: &UetsDb, items: &[PersistItem], shared: &Shared) -> Result<usize, DbError> {
    let started = Instant::now();

    let mut trades_owned: Vec<TradeRecord> = Vec::new();
    let mut portfolios_owned: Vec<PortfolioRecord> = Vec::new();
    let mut summaries_owned: Vec<SessionSummary> = Vec::new();
    for item in items {
        match item {
            PersistItem::Trade(t) => trades_owned.push(t.clone()),
            PersistItem::Portfolio(p) => portfolios_owned.push(p.clone()),
            PersistItem::SessionSummary(s) => summaries_owned.push(s.clone()),
        }
    }

    let mut written = 0usize;
    match db.write_batch(&trades_owned, &portfolios_owned) {
        Ok(()) => written += trades_owned.len() + portfolios_owned.len(),
        Err(DbError::Serialization(msg)) => {
            // Isolate poison records so one bad record never blocks the rest.
            tracing::error!("batch serialization failed ({msg}); isolating records individually");
            for t in trades_owned {
                match db.write_trade(&t) {
                    Ok(()) => written += 1,
                    Err(DbError::Serialization(m)) => {
                        tracing::error!(record = %t.instance.composite(), "poison trade record ({m}) quarantined");
                        shared.push_quarantine(std::iter::once(PersistItem::Trade(t)));
                    }
                    Err(other) => return Err(other),
                }
            }
            for p in portfolios_owned {
                match db.write_portfolio(&p) {
                    Ok(()) => written += 1,
                    Err(DbError::Serialization(m)) => {
                        tracing::error!(record = %p.instance.composite(), "poison portfolio record ({m}) quarantined");
                        shared.push_quarantine(std::iter::once(PersistItem::Portfolio(p)));
                    }
                    Err(other) => return Err(other),
                }
            }
        }
        Err(other) => return Err(other),
    }

    for s in summaries_owned {
        db.finalize_session(&s)?;
        written += 1;
    }

    shared.record_write(written, started.elapsed());
    Ok(written)
}

fn writer_loop(db: Arc<UetsDb>, shared: Arc<Shared>, batch_max: usize) {
    let mut pending: VecDeque<PersistItem> = VecDeque::new();
    let mut attempts: u32 = 0;
    let mut channel_closed = false;

    loop {
        // Once shutdown closes the channel, stop retry loops and let the
        // bounded final flush below take over (shutdown must terminate).
        let closed = shared.tx.lock().unwrap().is_none();

        // Recover one quarantined item first (storage may have recovered).
        // After close, the final flush below owns quarantine.
        if pending.is_empty() && attempts == 0 && !closed {
            let recovered = shared.quarantine.lock().unwrap().pop_front();
            if let Some(item) = recovered {
                shared.stats.quarantined.fetch_sub(1, Ordering::Relaxed);
                pending.push_back(item);
            }
        }

        // Fill the batch from the queue.
        while pending.len() < batch_max {
            if channel_closed {
                match shared.rx.try_recv() {
                    Ok(item) => pending.push_back(item),
                    Err(_) => break,
                }
            } else if pending.is_empty() {
                match shared.rx.recv_blocking() {
                    Ok(item) => pending.push_back(item),
                    Err(_) => {
                        channel_closed = true;
                        break;
                    }
                }
            } else {
                match shared.rx.try_recv() {
                    Ok(item) => pending.push_back(item),
                    Err(async_channel::TryRecvError::Closed) => {
                        channel_closed = true;
                        break;
                    }
                    Err(async_channel::TryRecvError::Empty) => break,
                }
            }
        }

        if pending.is_empty() && channel_closed {
            break; // Queue drained and closed.
        }
        if pending.is_empty() {
            continue;
        }

        let batch: Vec<PersistItem> = pending.drain(..).collect();
        match write_items(&db, &batch, &shared) {
            Ok(_n) => {
                attempts = 0;
            }
            Err(err) => {
                shared.stats.write_errors.fetch_add(1, Ordering::Relaxed);
                attempts += 1;
                let closed_now = shared.tx.lock().unwrap().is_none();
                if attempts >= MAX_ATTEMPTS || closed_now {
                    tracing::error!(
                        error = %err,
                        attempts,
                        closed_now,
                        "persistence write failing; quarantining batch (retained for bounded final flush)"
                    );
                    shared.push_quarantine(batch.into_iter());
                    attempts = 0;
                } else {
                    std::thread::sleep(RETRY_DELAY);
                    // Retry the same batch next cycle (keep its head position).
                    let mut retried = VecDeque::new();
                    retried.extend(batch);
                    retried.append(&mut pending);
                    pending = retried;
                }
            }
        }
    }

    // Queue is closed and drained: final flush of quarantined items.
    loop {
        let next = shared.quarantine.lock().unwrap().pop_front();
        let Some(item) = next else { break };
        shared.stats.quarantined.fetch_sub(1, Ordering::Relaxed);
        let mut flushed = false;
        for _ in 0..MAX_ATTEMPTS {
            if write_items(&db, std::slice::from_ref(&item), &shared).is_ok() {
                flushed = true;
                break;
            }
            shared.stats.write_errors.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(RETRY_DELAY);
        }
        if !flushed {
            // Put it back so shutdown() reports it as unwritten (never silent).
            shared.stats.quarantined.fetch_add(1, Ordering::Relaxed);
            shared.quarantine.lock().unwrap().push_front(item);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        PortfolioRecord, QueryFilter, SchemaVersion, StrategyInstance, TradeRecord,
    };
    use chrono::{TimeZone, Utc};
    use std::sync::atomic::AtomicBool;
    use crate::schema::TradeIntent;

    fn sample_trade(n: i64) -> TradeRecord {
        TradeRecord {
            schema: SchemaVersion::default(),
            session_id: "session_1".into(),
            instance: StrategyInstance::new("ofmr", "btcusdt"),
            symbol: "BTCUSDT".into(),
            event_time: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
                + chrono::Duration::nanoseconds(n),
            feature_inputs: serde_json::json!({"price": 100.0}).to_string(),
            decision: TradeIntent::Long,
            decision_metadata: serde_json::json!({}).to_string(),
            intent: TradeIntent::Long,
            order_events: Vec::new(),
            entry_price: Some(100.0),
            exit_price: None,
            tp_price: Some(100.5),
            sl_price: Some(99.5),
            vertical_time: None,
            fill_qty: 1.0,
            fees: 0.1,
            realized_pnl: 0.0,
            model_version: "v1.0".into(),
            config_id: "cfg_1".into(),
        }
    }

    fn sample_portfolio() -> PortfolioRecord {
        PortfolioRecord {
            schema: SchemaVersion::default(),
            session_id: "session_1".into(),
            instance: StrategyInstance::new("ofmr", "btcusdt"),
            asset_id: "btcusdt".into(),
            timestamp: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            allocated_capital: 10_000.0,
            cash: 10_000.0,
            equity: 10_000.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            cumulative_return: 0.0,
            sortino: 0.0,
            drawdown: 0.0,
            max_drawdown: 0.0,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            margin_used: 0.0,
            available_margin: 10_000.0,
            effective_leverage: 0.0,
            fees_paid: 0.0,
            trade_count: 0,
            win_count: 0,
            loss_count: 0,
            win_rate: 0.0,
        }
    }

    fn temp_db_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("uets_db_persist_{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn saturation_never_drops_critical_items() {
        let path = temp_db_path();
        let db = Arc::new(UetsDb::open(&path).unwrap());
        let (handle, worker) = spawn(
            Arc::clone(&db),
            PersistenceConfig {
                queue_capacity: 8,
                batch_max: 4,
            },
        );

        for i in 0..100i64 {
            let outcome = handle
                .submit(PersistItem::Trade(sample_trade(i)))
                .unwrap();
            assert!(
                matches!(
                    outcome,
                    SubmitOutcome::Accepted | SubmitOutcome::AcceptedBlocked
                ),
                "critical trade must always be queued (backpressure if full)"
            );
            // Snapshots may be accepted or dropped; drops must be counted.
            let snap = handle.submit(PersistItem::Portfolio(sample_portfolio()));
            assert!(snap.is_ok());
        }

        worker.shutdown().unwrap();

        let records = db
            .query_trades(&QueryFilter::new().session("session_1"))
            .unwrap();
        assert_eq!(records.len(), 100, "every critical record must persist");
        let stats = handle.stats();
        assert_eq!(stats.queue_depth, 0);
        assert!(stats.written >= 100);
        assert_eq!(stats.write_errors, 0);

        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn full_queue_applies_backpressure_and_counts_drops() {
        let (tx, rx) = async_channel::bounded(2);
        let shared = Arc::new(Shared {
            tx: Mutex::new(Some(tx)),
            rx: rx.clone(),
            quarantine: Mutex::new(VecDeque::new()),
            stats: StatsAtomics::default(),
        });
        let handle_a = PersistenceHandle {
            shared: Arc::clone(&shared),
        };
        let handle_b = PersistenceHandle {
            shared: Arc::clone(&shared),
        };

        // Fill the bounded queue with non-critical snapshots.
        assert_eq!(
            handle_a
                .submit(PersistItem::Portfolio(sample_portfolio()))
                .unwrap(),
            SubmitOutcome::Accepted
        );
        assert_eq!(
            handle_a
                .submit(PersistItem::Portfolio(sample_portfolio()))
                .unwrap(),
            SubmitOutcome::Accepted
        );
        // Third snapshot: dropped, but counted — never silent.
        assert_eq!(
            handle_a
                .submit(PersistItem::Portfolio(sample_portfolio()))
                .unwrap(),
            SubmitOutcome::Dropped
        );
        assert_eq!(handle_a.stats().dropped_snapshots, 1);

        // Critical item on a full queue blocks (backpressure), not dropped.
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let thread = std::thread::spawn(move || {
            let outcome = handle_b
                .submit(PersistItem::Trade(sample_trade(0)))
                .unwrap();
            assert_eq!(outcome, SubmitOutcome::AcceptedBlocked);
            flag.store(true, Ordering::SeqCst);
        });

        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !done.load(Ordering::SeqCst),
            "critical submit must block while the queue is full"
        );

        // Draining frees space; the blocked submit completes with its item intact.
        let first = rx.recv_blocking().unwrap();
        assert!(matches!(first, PersistItem::Portfolio(_)));
        thread.join().unwrap();
        assert!(done.load(Ordering::SeqCst));
        let _second = rx.recv_blocking().unwrap();
        let third = rx.recv_blocking().unwrap();
        assert!(
            matches!(third, PersistItem::Trade(_)),
            "critical item must arrive after backpressure releases"
        );
        // Enqueued counts accepted + blocked, not dropped.
        assert_eq!(handle_a.stats().enqueued, 3);
    }

    #[test]
    fn storage_failure_quarantines_but_never_drops_critical() {
        let path = temp_db_path();
        let db = Arc::new(UetsDb::open(&path).unwrap());
        let (handle, worker) = spawn(Arc::clone(&db), PersistenceConfig::default());

        db.inject_write_failure(true);
        handle.submit(PersistItem::Trade(sample_trade(0))).unwrap();

        // Wait for the retry cycle to exhaust attempts and quarantine (≈1 s).
        let deadline = Instant::now() + Duration::from_secs(5);
        while handle.stats().quarantined_total == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let stats = handle.stats();
        assert!(stats.write_errors > 0, "storage errors must be surfaced");
        assert!(
            stats.quarantined_total >= 1,
            "critical item must be quarantined, not lost"
        );

        // Storage recovers; a wakeup item triggers reprocessing.
        db.inject_write_failure(false);
        handle
            .submit(PersistItem::Portfolio(sample_portfolio()))
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let records = db
                .query_trades(&QueryFilter::new().session("session_1"))
                .unwrap();
            if records.len() == 1 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "quarantined critical item must be rewritten after recovery"
            );
            std::thread::sleep(Duration::from_millis(20));
        }

        worker.shutdown().unwrap();
        assert_eq!(handle.stats().quarantined, 0);

        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn shutdown_reports_unwritten_critical_items() {
        let path = temp_db_path();
        let db = Arc::new(UetsDb::open(&path).unwrap());
        let (handle, worker) = spawn(Arc::clone(&db), PersistenceConfig::default());

        db.inject_write_failure(true);
        handle.submit(PersistItem::Trade(sample_trade(0))).unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        while handle.stats().quarantined_total == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(handle.stats().quarantined_total >= 1);

        // Storage stays down: shutdown must report the loss loudly, not swallow it.
        let result = worker.shutdown();
        assert!(
            matches!(result, Err(PersistError::Unwritten { critical, .. }) if critical >= 1),
            "shutdown must report unwritten critical items, got {result:?}"
        );
        assert!(handle.stats().quarantined >= 1);

        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn dropping_stats_consumer_does_not_stop_persistence() {
        let path = temp_db_path();
        let db = Arc::new(UetsDb::open(&path).unwrap());
        let (handle, worker) = spawn(Arc::clone(&db), PersistenceConfig::default());

        // A GUI-like consumer subscribes, then closes.
        {
            let stats_view = handle.stats_handle();
            let s = stats_view.stats();
            assert_eq!(s.written, 0);
        } // stats_view dropped — GUI closed

        handle.submit(PersistItem::Trade(sample_trade(0))).unwrap();
        worker.shutdown().unwrap();

        let records = db
            .query_trades(&QueryFilter::new().session("session_1"))
            .unwrap();
        assert_eq!(
            records.len(),
            1,
            "persistence must continue after the GUI consumer closes"
        );

        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }
}








