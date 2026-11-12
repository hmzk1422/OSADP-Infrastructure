//! Daily UTC RocksDB partitions.
//!
//! On-disk layout (one directory per UTC day, one RocksDB per kind/session):
//!
//! ```text
//! <live_db>/
//! ├── 2026-11-12/
//! │   ├── telemetry/<session>/
//! │   ├── portfolio/<session>/
//! │   ├── trades/<session>/
//! │   └── .sealed                 <- written by `db seal-days`, never by a writer
//! ├── 2026-11-13/                 <- today (live; never archived)
//! └── current_session             <- TUI session marker, unchanged
//! ```
//!
//! This is *daily partitioning of separate RocksDB instances*, not in-process
//! MVCC snapshots. A [`RollingDb`] is owned by exactly one writer thread. All
//! rollover work (pre-opening tomorrow's DB, carrying recovery state forward,
//! flushing and closing yesterday's DB) happens on that thread, so the strategy
//! or TUI thread only ever touches its bounded queue and is never blocked by a
//! database swap.
//!
//! The module is deliberately std-only and generic over [`PartitionStore`] so
//! the rollover logic can be tested without RocksDB.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Marker file written into a day directory once no process holds its DBs.
pub const SEALED_MARKER: &str = ".sealed";
/// How long before UTC midnight the next day's DB is opened (off the hot path).
pub const PREOPEN_SECONDS: i64 = 60;
/// Back-off after a failed rollover/pre-open before trying again.
pub const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(5);

const SECONDS_PER_DAY: i64 = 86_400;

// ---------------------------------------------------------------------------
// Kinds, paths and UTC day arithmetic
// ---------------------------------------------------------------------------

/// The three databases kept inside every daily partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DbKind {
    Telemetry,
    Portfolio,
    Trades,
}

impl DbKind {
    pub const ALL: [DbKind; 3] = [DbKind::Telemetry, DbKind::Portfolio, DbKind::Trades];

    pub fn dir_name(self) -> &'static str {
        match self {
            DbKind::Telemetry => "telemetry",
            DbKind::Portfolio => "portfolio",
            DbKind::Trades => "trades",
        }
    }

    pub fn from_dir_name(name: &str) -> Option<DbKind> {
        DbKind::ALL.into_iter().find(|kind| kind.dir_name() == name)
    }
}

/// Seconds since the Unix epoch (0 if the clock is before 1970).
pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0)
}

/// `YYYY-MM-DD` of the UTC day containing `unix_secs`.
pub fn utc_day(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(SECONDS_PER_DAY);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}")
}

pub fn utc_today() -> String {
    utc_day(unix_now())
}

/// The UTC day after the one containing `unix_secs`.
pub fn next_utc_day(unix_secs: i64) -> String {
    utc_day(unix_secs.div_euclid(SECONDS_PER_DAY) * SECONDS_PER_DAY + SECONDS_PER_DAY)
}

/// Seconds from `unix_secs` until the next 00:00:00 UTC (always 1..=86400).
pub fn seconds_until_next_utc_midnight(unix_secs: i64) -> i64 {
    SECONDS_PER_DAY - unix_secs.rem_euclid(SECONDS_PER_DAY)
}

/// True for a strictly formatted `YYYY-MM-DD` day directory name.
pub fn is_day_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let digits = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    if !(digits(0..4) && digits(5..7) && digits(8..10)) {
        return false;
    }
    let month: u32 = name[5..7].parse().unwrap_or(0);
    let day: u32 = name[8..10].parse().unwrap_or(0);
    (1..=12).contains(&month) && (1..=31).contains(&day)
}

fn valid_session(session: &str) -> bool {
    !session.is_empty()
        && session != "."
        && session != ".."
        && session
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// `<root>/<day>/<kind>/<session>`
pub fn db_path(root: &Path, day: &str, kind: DbKind, session: &str) -> PathBuf {
    root.join(day).join(kind.dir_name()).join(session)
}

/// All day directories under `root`, oldest first.
pub fn list_days(root: &Path) -> io::Result<Vec<String>> {
    let mut days = Vec::new();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(days),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            if is_day_name(name) {
                days.push(name.to_string());
            }
        }
    }
    days.sort();
    Ok(days)
}

/// Every `<day>/<kind>/<session>` database directory inside one day.
pub fn db_dirs_in_day(root: &Path, day: &str) -> io::Result<Vec<(DbKind, String, PathBuf)>> {
    let mut dirs = Vec::new();
    for kind in DbKind::ALL {
        let kind_dir = root.join(day).join(kind.dir_name());
        let entries = match fs::read_dir(&kind_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if let Some(session) = entry.file_name().to_str() {
                dirs.push((kind, session.to_string(), entry.path()));
            }
        }
    }
    dirs.sort_by(|a, b| a.2.cmp(&b.2));
    Ok(dirs)
}

pub fn is_sealed(root: &Path, day: &str) -> bool {
    root.join(day).join(SEALED_MARKER).is_file()
}

/// Atomically create the `.sealed` marker.
pub fn write_sealed(root: &Path, day: &str) -> io::Result<()> {
    let dir = root.join(day);
    let tmp = dir.join(format!("{SEALED_MARKER}.{}.tmp", std::process::id()));
    fs::write(&tmp, b"sealed\n")?;
    fs::rename(&tmp, dir.join(SEALED_MARKER))
}

/// Seal every completed day (`day < today`) whose databases are all closed.
///
/// `locked` reports whether a RocksDB directory is currently held open by some
/// process (for example via an `fcntl` probe of its `LOCK` file). A day with any
/// locked database is left unsealed, so a writer that is still draining a late
/// batch can never have its data checkpointed or deleted. Returns the days
/// sealed by this call. Days without any database directory are ignored.
pub fn seal_days(
    root: &Path,
    today: &str,
    locked: impl Fn(&Path) -> bool,
) -> io::Result<Vec<String>> {
    if !is_day_name(today) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid today date: {today}"),
        ));
    }
    let mut sealed = Vec::new();
    for day in list_days(root)? {
        if day.as_str() >= today || is_sealed(root, &day) {
            continue;
        }
        let dirs = db_dirs_in_day(root, &day)?;
        if dirs.is_empty() || dirs.iter().any(|(_, _, path)| locked(path)) {
            continue;
        }
        write_sealed(root, &day)?;
        sealed.push(day);
    }
    Ok(sealed)
}

// ---------------------------------------------------------------------------
// RollingDb
// ---------------------------------------------------------------------------

/// What a database must provide to be partitioned by day.
pub trait PartitionStore: Sized {
    type Error: fmt::Display;
    /// Open or create the database at `path`.
    fn open_at(path: &Path) -> Result<Self, Self::Error>;
    /// Copy recovery state (runtime state, session metadata, newest records)
    /// from the previous day's database so the new day is self-contained.
    fn carry_forward_from(&self, previous: &Self) -> Result<(), Self::Error>;
    /// Flush memtables to disk.
    fn flush_all(&self) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub enum PartitionError {
    Io(io::Error),
    Store(String),
    InvalidSession(String),
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PartitionError::Io(error) => write!(f, "i/o error: {error}"),
            PartitionError::Store(message) => write!(f, "{message}"),
            PartitionError::InvalidSession(session) => write!(f, "invalid session id: {session:?}"),
        }
    }
}

impl std::error::Error for PartitionError {}

impl From<io::Error> for PartitionError {
    fn from(error: io::Error) -> Self {
        PartitionError::Io(error)
    }
}

fn store_error<E: fmt::Display>(context: &str, error: E) -> PartitionError {
    PartitionError::Store(format!("{context}: {error}"))
}

/// Result of a completed day switch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rollover {
    pub from: String,
    pub to: String,
    /// True when no other holder kept the old handle alive, i.e. the old DB
    /// was actually closed (and its `LOCK` released) by the swap.
    pub previous_handle_closed: bool,
    /// A flush failure on the old DB (non-fatal: closing flushes again).
    pub flush_error: Option<String>,
}

/// One kind/session database that follows the UTC day.
///
/// Not `Sync`: it is owned and ticked by a single writer thread.
pub struct RollingDb<S: PartitionStore> {
    root: PathBuf,
    kind: DbKind,
    session: String,
    day: String,
    current: Arc<S>,
    next: Option<(String, Arc<S>)>,
    retry_after: Duration,
    retry_not_before: Option<Instant>,
}

impl<S: PartitionStore> RollingDb<S> {
    /// Open today's partition. If it does not exist yet and an earlier day for
    /// the same kind/session does, recovery state is carried forward from the
    /// newest earlier day (fail closed if that cannot be read).
    pub fn open(
        root: &Path,
        kind: DbKind,
        session: &str,
        now_unix: i64,
    ) -> Result<Self, PartitionError> {
        if !valid_session(session) {
            return Err(PartitionError::InvalidSession(session.to_string()));
        }
        let day = utc_day(now_unix);
        let path = db_path(root, &day, kind, session);
        let existed = path.is_dir();
        fs::create_dir_all(&path)?;
        let current = Arc::new(S::open_at(&path).map_err(|e| store_error("open daily database", e))?);

        if !existed {
            if let Some(previous_path) = latest_previous_path(root, &day, kind, session)? {
                let seeded = S::open_at(&previous_path)
                    .map_err(|e| store_error("open previous day for carry-forward", e))
                    .and_then(|previous| {
                        current
                            .carry_forward_from(&previous)
                            .map_err(|e| store_error("carry state forward from previous day", e))
                    });
                if let Err(error) = seeded {
                    // Never leave a half-seeded day that a restart would mistake for complete.
                    drop(current);
                    let _ = fs::remove_dir_all(&path);
                    return Err(error);
                }
            }
        }

        Ok(Self {
            root: root.to_path_buf(),
            kind,
            session: session.to_string(),
            day,
            current,
            next: None,
            retry_after: DEFAULT_RETRY_AFTER,
            retry_not_before: None,
        })
    }

    /// Override the retry back-off after a failed rollover (tests).
    pub fn with_retry_after(mut self, retry_after: Duration) -> Self {
        self.retry_after = retry_after;
        self
    }

    pub fn current(&self) -> &Arc<S> {
        &self.current
    }

    pub fn current_day(&self) -> &str {
        &self.day
    }

    fn backoff_active(&self) -> bool {
        self.retry_not_before
            .is_some_and(|not_before| Instant::now() < not_before)
    }

    /// Advance the partition for `now_unix`. Call between write batches on the
    /// writer thread. On error the previous day's DB stays current, so writes
    /// are never dropped and never blocked; the roll is retried after a back-off.
    pub fn tick(&mut self, now_unix: i64) -> Result<Option<Rollover>, PartitionError> {
        let today = utc_day(now_unix);
        if today.as_str() < self.day.as_str() {
            // Clock stepped backwards: never roll backwards, keep the current day.
            return Ok(None);
        }
        if today != self.day {
            if self.backoff_active() {
                return Ok(None);
            }
            return match self.roll_to(&today) {
                Ok(rollover) => {
                    self.retry_not_before = None;
                    Ok(Some(rollover))
                }
                Err(error) => {
                    self.retry_not_before = Some(Instant::now() + self.retry_after);
                    Err(error)
                }
            };
        }

        if self.next.is_none()
            && seconds_until_next_utc_midnight(now_unix) <= PREOPEN_SECONDS
            && !self.backoff_active()
        {
            let next_day = next_utc_day(now_unix);
            let opened = self.open_day(&next_day);
            match opened {
                Ok(db) => self.next = Some((next_day, db)),
                Err(error) => {
                    self.retry_not_before = Some(Instant::now() + self.retry_after);
                    return Err(error);
                }
            }
        }
        Ok(None)
    }

    fn open_day(&self, day: &str) -> Result<Arc<S>, PartitionError> {
        let path = db_path(&self.root, day, self.kind, &self.session);
        fs::create_dir_all(&path)?;
        let db = S::open_at(&path).map_err(|e| store_error("open daily database", e))?;
        Ok(Arc::new(db))
    }

    fn roll_to(&mut self, today: &str) -> Result<Rollover, PartitionError> {
        let new_db = match self.next.take() {
            Some((day, db)) if day == today => db,
            // Stale pre-open (clock jumped more than a day) or none: open now.
            _ => self.open_day(today)?,
        };

        // Carry state at swap time, after the last write to the old day.
        if let Err(error) = new_db.carry_forward_from(&self.current) {
            self.next = Some((today.to_string(), new_db));
            return Err(store_error("carry state forward at rollover", error));
        }
        let flush_error = self.current.flush_all().err().map(|e| e.to_string());

        let old = std::mem::replace(&mut self.current, new_db);
        let from = std::mem::replace(&mut self.day, today.to_string());
        let previous_handle_closed = Arc::strong_count(&old) == 1;
        drop(old);
        Ok(Rollover {
            from,
            to: today.to_string(),
            previous_handle_closed,
            flush_error,
        })
    }
}

fn latest_previous_path(
    root: &Path,
    today: &str,
    kind: DbKind,
    session: &str,
) -> Result<Option<PathBuf>, PartitionError> {
    let mut days = list_days(root)?;
    days.retain(|day| day.as_str() < today);
    for day in days.into_iter().rev() {
        let path = db_path(root, &day, kind, session);
        if path.is_dir() {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    // 2026-11-12 23:59:30 UTC and 2026-11-13 00:00:00 UTC.
    const BEFORE_MIDNIGHT: i64 = 1_794_527_970;
    const MIDNIGHT: i64 = 1_794_528_000;

    static OPEN_PATHS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    static OPEN_COUNT: AtomicU64 = AtomicU64::new(0);
    static OPEN_DELAY_MS: AtomicU64 = AtomicU64::new(0);
    static FAIL_OPEN_SUBSTR: Mutex<String> = Mutex::new(String::new());
    // Serialise tests that touch the process-wide fake-store globals.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn open_paths() -> &'static Mutex<HashSet<PathBuf>> {
        OPEN_PATHS.get_or_init(|| Mutex::new(HashSet::new()))
    }

    /// Fake DB: opening takes a process-wide "lock" per path (like RocksDB's
    /// LOCK file); dropping releases it. State lives in files so carry-forward
    /// is observable on disk.
    struct FakeStore {
        path: PathBuf,
    }

    impl FakeStore {
        fn put(&self, name: &str, value: &str) {
            fs::write(self.path.join(name), value).unwrap();
        }
        fn append(&self, line: &str) {
            use std::io::Write;
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.path.join("data"))
                .unwrap();
            writeln!(file, "{line}").unwrap();
        }
    }

    impl PartitionStore for FakeStore {
        type Error = String;

        fn open_at(path: &Path) -> Result<Self, String> {
            let delay = OPEN_DELAY_MS.load(Ordering::SeqCst);
            if delay > 0 {
                std::thread::sleep(Duration::from_millis(delay));
            }
            let fail = FAIL_OPEN_SUBSTR.lock().unwrap().clone();
            if !fail.is_empty() && path.to_string_lossy().contains(&fail) {
                return Err("simulated open failure".into());
            }
            if !open_paths().lock().unwrap().insert(path.to_path_buf()) {
                return Err(format!("LOCK held: {}", path.display()));
            }
            fs::create_dir_all(path).map_err(|e| e.to_string())?;
            OPEN_COUNT.fetch_add(1, Ordering::SeqCst);
            Ok(Self {
                path: path.to_path_buf(),
            })
        }

        fn carry_forward_from(&self, previous: &Self) -> Result<(), String> {
            if let Ok(state) = fs::read_to_string(previous.path.join("runtime_state")) {
                fs::write(self.path.join("runtime_state"), state).map_err(|e| e.to_string())?;
            }
            Ok(())
        }

        fn flush_all(&self) -> Result<(), String> {
            Ok(())
        }
    }

    impl Drop for FakeStore {
        fn drop(&mut self) {
            open_paths().lock().unwrap().remove(&self.path);
        }
    }

    fn lock_held(path: &Path) -> bool {
        open_paths().lock().unwrap().contains(path)
    }

    fn reset_globals() {
        OPEN_DELAY_MS.store(0, Ordering::SeqCst);
        FAIL_OPEN_SUBSTR.lock().unwrap().clear();
        open_paths().lock().unwrap().clear();
        OPEN_COUNT.store(0, Ordering::SeqCst);
    }

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "uets_partition_{tag}_{}_{}",
            std::process::id(),
            unix_now()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn utc_day_and_midnight_arithmetic() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(utc_day(0), "1970-01-01");
        assert_eq!(utc_day(1_700_000_000), "2023-11-14");
        assert_eq!(utc_day(1_709_208_000), "2024-02-29"); // leap day
        assert_eq!(utc_day(BEFORE_MIDNIGHT), "2026-11-12");
        assert_eq!(utc_day(MIDNIGHT), "2026-11-13");
        assert_eq!(utc_day(1_798_761_599), "2026-12-31");
        assert_eq!(utc_day(1_798_761_600), "2027-01-01"); // year roll
        assert_eq!(next_utc_day(BEFORE_MIDNIGHT), "2026-11-13");
        assert_eq!(seconds_until_next_utc_midnight(BEFORE_MIDNIGHT), 30);
        assert_eq!(seconds_until_next_utc_midnight(MIDNIGHT), 86_400);
        assert!(is_day_name("2026-11-12"));
        assert!(!is_day_name("2026-13-12"));
        assert!(!is_day_name("session-1"));
        assert!(!is_day_name("2026-11-1"));
    }

    #[test]
    fn rollover_cleanly_separates_days_and_keeps_all_three_folders() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();
        let root = temp_root("separate");
        let mut dbs: Vec<RollingDb<FakeStore>> = DbKind::ALL
            .into_iter()
            .map(|kind| RollingDb::open(&root, kind, "run1", BEFORE_MIDNIGHT - 120).unwrap())
            .collect();

        for db in &dbs {
            db.current().append("day-12");
        }
        // Approach midnight: pre-open only, still writing to the 12th.
        for db in &mut dbs {
            assert_eq!(db.tick(BEFORE_MIDNIGHT).unwrap(), None);
            assert_eq!(db.current_day(), "2026-11-12");
        }
        // Midnight: every kind rolls; old handles are really closed.
        for db in &mut dbs {
            let rollover = db.tick(MIDNIGHT).unwrap().expect("must roll at 00:00 UTC");
            assert_eq!(rollover.from, "2026-11-12");
            assert_eq!(rollover.to, "2026-11-13");
            assert!(rollover.previous_handle_closed);
            db.current().append("day-13");
        }
        drop(dbs);

        for (day, expected) in [("2026-11-12", "day-12\n"), ("2026-11-13", "day-13\n")] {
            for kind in DbKind::ALL {
                let data = fs::read_to_string(db_path(&root, day, kind, "run1").join("data"))
                    .unwrap_or_else(|_| panic!("{day}/{} missing", kind.dir_name()));
                assert_eq!(data, expected, "{day}/{} must hold only its own day", kind.dir_name());
            }
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn next_day_is_opened_ahead_so_the_swap_does_not_open_a_database() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();
        let root = temp_root("preopen");
        let mut db =
            RollingDb::<FakeStore>::open(&root, DbKind::Trades, "run1", BEFORE_MIDNIGHT - 300).unwrap();

        db.tick(BEFORE_MIDNIGHT - 200).unwrap();
        assert!(
            !db_path(&root, "2026-11-13", DbKind::Trades, "run1").exists(),
            "must not pre-open earlier than the pre-open window"
        );
        db.tick(BEFORE_MIDNIGHT).unwrap();
        assert!(db_path(&root, "2026-11-13", DbKind::Trades, "run1").is_dir());
        assert_eq!(db.current_day(), "2026-11-12");

        let opens_before_swap = OPEN_COUNT.load(Ordering::SeqCst);
        db.tick(MIDNIGHT).unwrap().expect("roll");
        assert_eq!(
            OPEN_COUNT.load(Ordering::SeqCst),
            opens_before_swap,
            "swap at 00:00 must reuse the pre-opened database"
        );
        drop(db);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recovery_state_is_carried_into_the_new_day_at_rollover_and_on_restart() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();
        let root = temp_root("carry");
        let mut db =
            RollingDb::<FakeStore>::open(&root, DbKind::Portfolio, "run1", BEFORE_MIDNIGHT - 600).unwrap();
        db.current().put("runtime_state", "open long 1.5 BTC");
        db.tick(MIDNIGHT).unwrap().expect("roll");
        assert_eq!(
            fs::read_to_string(db_path(&root, "2026-11-13", DbKind::Portfolio, "run1").join("runtime_state"))
                .unwrap(),
            "open long 1.5 BTC",
            "OFMR recovery state must not stay behind in an archivable day"
        );
        drop(db);

        // Restart two days later: today's DB does not exist, newest earlier day seeds it.
        let later = MIDNIGHT + 2 * SECONDS_PER_DAY;
        let restarted = RollingDb::<FakeStore>::open(&root, DbKind::Portfolio, "run1", later).unwrap();
        assert_eq!(restarted.current_day(), "2026-11-15");
        assert_eq!(
            fs::read_to_string(db_path(&root, "2026-11-15", DbKind::Portfolio, "run1").join("runtime_state"))
                .unwrap(),
            "open long 1.5 BTC"
        );
        drop(restarted);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn seeding_failure_fails_closed_and_leaves_no_half_seeded_day() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();
        let root = temp_root("seedfail");
        let holder =
            RollingDb::<FakeStore>::open(&root, DbKind::Trades, "run1", BEFORE_MIDNIGHT - 600).unwrap();
        // The previous day is still held by another process: cannot seed.
        let error = RollingDb::<FakeStore>::open(&root, DbKind::Trades, "run1", MIDNIGHT + 5);
        assert!(error.is_err(), "must fail closed rather than start without state");
        assert!(
            !db_path(&root, "2026-11-13", DbKind::Trades, "run1").exists(),
            "a failed seed must not leave a day directory that looks complete"
        );
        drop(holder);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_rollover_keeps_writing_to_the_old_day_then_recovers() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();
        let root = temp_root("rollfail");
        let mut db = RollingDb::<FakeStore>::open(&root, DbKind::Telemetry, "run1", BEFORE_MIDNIGHT - 600)
            .unwrap()
            .with_retry_after(Duration::ZERO);
        *FAIL_OPEN_SUBSTR.lock().unwrap() = "2026-11-13".into();

        assert!(db.tick(MIDNIGHT).is_err());
        assert_eq!(db.current_day(), "2026-11-12", "old DB stays current on failure");
        db.current().append("still-writing");

        FAIL_OPEN_SUBSTR.lock().unwrap().clear();
        let rollover = db.tick(MIDNIGHT + 1).unwrap().expect("retry succeeds");
        assert_eq!(rollover.to, "2026-11-13");
        drop(db);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn seal_days_never_seals_today_or_a_day_with_a_held_lock() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();
        let root = temp_root("seal");
        for day in ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"] {
            for kind in DbKind::ALL {
                fs::create_dir_all(db_path(&root, day, kind, "run1")).unwrap();
            }
        }
        fs::create_dir_all(root.join("2026-11-08")).unwrap(); // empty day: ignored
        // 11-10 still has a live writer (lock held).
        let held = FakeStore::open_at(&db_path(&root, "2026-11-10", DbKind::Trades, "run1")).unwrap();

        let sealed = seal_days(&root, "2026-11-12", lock_held).unwrap();
        assert_eq!(sealed, vec!["2026-11-09".to_string(), "2026-11-11".to_string()]);
        assert!(!is_sealed(&root, "2026-11-10"), "locked day must stay unsealed");
        assert!(!is_sealed(&root, "2026-11-12"), "today must never be sealed");
        assert!(!is_sealed(&root, "2026-11-08"), "empty day is not sealed");

        // Idempotent, and picks up the day once its writer closes.
        assert!(seal_days(&root, "2026-11-12", lock_held).unwrap().is_empty());
        drop(held);
        assert_eq!(
            seal_days(&root, "2026-11-12", lock_held).unwrap(),
            vec!["2026-11-10".to_string()]
        );
        assert!(seal_days(&root, "not-a-date", lock_held).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    /// The architectural property: a slow database open during the rollover
    /// window runs on the writer thread, so the producer's `try_send` latency is
    /// unaffected, nothing is lost or reordered, and records land in the correct day.
    #[test]
    fn producer_is_never_blocked_by_rollover_and_no_record_is_lost() {
        use std::sync::atomic::AtomicI64;
        use std::sync::mpsc::sync_channel;

        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset_globals();
        let root = temp_root("detached");
        let clock = Arc::new(AtomicI64::new(BEFORE_MIDNIGHT - 20)); // 23:59:10
        let mut rolling =
            RollingDb::<FakeStore>::open(&root, DbKind::Trades, "run1", clock.load(Ordering::SeqCst)).unwrap();
        OPEN_DELAY_MS.store(250, Ordering::SeqCst); // pre-open of the 13th is slow

        let (tx, rx) = sync_channel::<(i64, u32)>(4096);
        let writer_clock = Arc::clone(&clock);
        let writer = std::thread::spawn(move || {
            loop {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok((virtual_time, sequence)) => {
                        let _ = rolling.tick(virtual_time);
                        rolling.current().append(&format!("{virtual_time}:{sequence}"));
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        let _ = rolling.tick(writer_clock.load(Ordering::SeqCst));
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });

        let total = 600u32;
        let mut max_send = Duration::ZERO;
        for sequence in 0..total {
            // 600 sends sweep virtual time from 23:59:10 to 00:00:10 (crossing midnight).
            let virtual_time = BEFORE_MIDNIGHT - 20 + (sequence as i64 * 70 / total as i64);
            clock.store(virtual_time, Ordering::SeqCst);
            let started = Instant::now();
            tx.try_send((virtual_time, sequence)).expect("queue must never be full here");
            max_send = max_send.max(started.elapsed());
            std::thread::sleep(Duration::from_millis(2));
        }
        drop(tx);
        writer.join().unwrap();

        assert!(
            max_send < Duration::from_millis(50),
            "producer try_send stalled for {max_send:?} while the writer opened a 250 ms database"
        );

        let mut seen = 0u32;
        for (day, expect_before_midnight) in [("2026-11-12", true), ("2026-11-13", false)] {
            let data = fs::read_to_string(db_path(&root, day, DbKind::Trades, "run1").join("data")).unwrap();
            let mut last = -1i64;
            for line in data.lines() {
                let (time, sequence) = line.split_once(':').unwrap();
                let time: i64 = time.parse().unwrap();
                let sequence: i64 = sequence.parse().unwrap();
                assert!(sequence > last, "records must stay in order across the swap");
                last = sequence;
                seen += 1;
                assert_eq!(time < MIDNIGHT, expect_before_midnight, "record in wrong day: {line}");
            }
        }
        assert_eq!(seen, total, "no record may be lost across the rollover");
        let _ = fs::remove_dir_all(&root);
    }
}