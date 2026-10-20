//! The live capture contract: cadence, schema parity, and explicit loss.
//!
//! Three guarantees, each of which has broken before and each of which fails
//! *silently* when it does:
//!
//! 1. **Cadence.** The book stream is `@depth@100ms`, so consecutive LOB rows
//!    must be ~100 ms apart in *exchange* time. The regression this guards is
//!    the one where the feed degraded and rows trickled in seconds apart while
//!    every row still looked individually valid.
//! 2. **Schema parity.** A live row must be byte-compatible with the historical
//!    `market_data/matched` table, or the feature engine reads live columns by
//!    position and silently consumes the wrong field.
//! 3. **Loss is explicit.** A capture that drops events must say so. A
//!    silently-truncated recording is worse than no recording: it replays as a
//!    complete dataset.
//!
//! # Why the live test is `#[ignore]`d
//!
//! It drives the real `dev_test` binary against Binance, so it needs network
//! and is not deterministic in wall-clock terms. Everything offline (schema,
//! structure, absence semantics) runs in the default `cargo test`. For the
//! live run:
//!
//! ```text
//! set UETS_LIVE_TEST=1
//! cargo test --test live_capture -- --ignored --nocapture
//! ```
//!
//! # Percentiles here are exact, not histogrammed
//!
//! These tests sort a bounded sample and index it. That is the correct oracle
//! *off* the hot path. The production latency path must not do this -- sorting
//! shared state is exactly the lock that stalls the feed under contention, and
//! a ring buffer alone does not make concurrent percentiles correct. The test's
//! job is to be right; the runtime's job is to be fast.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How many rows to capture for the live run.
///
/// Small enough to finish in seconds, large enough for stable percentiles
/// (~100 LOB rows at 10/s) and for the trade fallback to have mattered.
const LIVE_ROWS: &str = "500";

/// Kill the capture if it has not finished by then.
const LIVE_TIMEOUT: Duration = Duration::from_secs(120);

/// Nominal book push interval of the subscribed `@depth@100ms` stream.
const NOMINAL_MS: u64 = 100;

/// The package root (`uets/`), the anchor for both the binary and the data.
fn uets_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The repo root, one level above `uets/`.
fn repo_root() -> PathBuf {
    uets_dir()
        .parent()
        .expect("uets lives inside the repo")
        .to_path_buf()
}

/// Any historical matched CSV, or `None` when the data is not checked out.
///
/// `*.csv` is gitignored, so a fresh clone legitimately has none. Absence is
/// a skip, never a failure: the parity test must not fail purely because the
/// dataset was not downloaded.
fn matched_csv() -> Option<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "csv") {
                out.push(path);
            }
        }
    }
    let root = repo_root().join("market_data").join("matched");
    let mut found = Vec::new();
    walk(&root, &mut found);
    found.sort();
    found.into_iter().next()
}

/// A parsed CSV: the header and the data rows as raw string fields.
struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

/// Read a CSV as raw string fields. No numeric coercion -- this test judges
/// *shape and text*, and a formatting change is exactly what it must see.
fn read_csv(path: &Path) -> std::io::Result<Table> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .map(|l| l.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let rows = lines
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect();
    Ok(Table { header, rows })
}

/// Column index of `name`, or a panic naming what *is* there.
fn column(table: &Table, name: &str) -> usize {
    table
        .header
        .iter()
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("no column '{name}' in header: {:?}", table.header))
}

/// Exact nearest-rank quantile over a sorted slice: `q` in `0.0..=1.0`.
///
/// Sorting is fine here -- see the module note.
fn percentile(sorted: &[u64], q: f64) -> u64 {
    assert!(!sorted.is_empty(), "percentile of an empty sample");
    let rank = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Gap statistics for one event type, all in milliseconds.
#[derive(Debug)]
struct Cadence {
    count: usize,
    p50: u64,
    p90: u64,
    p99: u64,
    max: u64,
    /// Fraction of gaps landing within the nominal band (60..=140 ms).
    ///
    /// The most robust single measure of "is this feed still 100 ms": it
    /// tolerates a handful of network hiccups while still collapsing to zero
    /// if the whole distribution drifts up, which is what a sustained slowdown
    /// looks like in the data.
    in_band: f64,
}

/// Consecutive-timestamp gaps for the rows of one `event_type`.
///
/// Measured on the exchange event time the writer stamps into `timestamp`
/// (nanoseconds), never on local arrival time. Local time would fold in
/// scheduler noise and answer the wrong question: what matters is whether the
/// *exchange* is pushing at 100 ms, not whether this process was awake.
fn cadence(table: &Table, event_type: &str) -> Option<Cadence> {
    let ts = column(table, "timestamp");
    let et = column(table, "event_type");
    let mut stamps: Vec<u64> = table
        .rows
        .iter()
        .filter(|r| r.get(et).is_some_and(|v| v == event_type))
        .filter_map(|r| r.get(ts).and_then(|v| v.parse::<u64>().ok()))
        .collect();
    if stamps.len() < 2 {
        return None;
    }
    // The reorder buffer is supposed to hand these over in total order. If it
    // is not, the gaps are meaningless -- so sort first rather than reporting a
    // beautiful p50 over a shuffled sequence.
    stamps.sort_unstable();
    let mut gaps: Vec<u64> = stamps
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]) / 1_000_000)
        .filter(|g| *g > 0)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable();
    let in_band =
        gaps.iter().filter(|&&g| (60..=140).contains(&g)).count() as f64 / gaps.len() as f64;
    Some(Cadence {
        count: stamps.len(),
        p50: percentile(&gaps, 0.50),
        p90: percentile(&gaps, 0.90),
        p99: percentile(&gaps, 0.99),
        max: *gaps.last().expect("non-empty"),
        in_band,
    })
}

// =============================================================================
// Offline: schema and structure. These run in the default `cargo test`.
// =============================================================================

#[test]
fn live_schema_is_128_unique_columns_starting_with_timestamp() {
    let schema = stream_handler::schema();
    assert_eq!(schema.len(), 128, "the merged table is 128 columns wide");
    assert_eq!(schema[0], "timestamp");
    assert_eq!(schema[1], "event_type");
    let unique: std::collections::HashSet<&&str> = schema.iter().collect();
    assert_eq!(
        unique.len(),
        schema.len(),
        "duplicate column name in schema"
    );
}

#[test]
fn live_schema_matches_the_historical_matched_header() {
    let Some(path) = matched_csv() else {
        eprintln!("skipping: no matched CSV under market_data/matched (data is gitignored)");
        return;
    };
    let table = read_csv(&path).expect("read matched csv");

    // The historical writer is pandas, which emits a leading unnamed index
    // column. It is a dataframe artifact, not part of the schema.
    assert!(
        table.header[0].is_empty(),
        "expected a leading pandas index column, got {:?}",
        table.header[0]
    );
    let historical: Vec<&str> = table.header[1..].iter().map(String::as_str).collect();
    let live: &[&str] = stream_handler::schema();

    assert_eq!(
        live.len(),
        historical.len(),
        "column count drift: live {} vs historical {}",
        live.len(),
        historical.len()
    );
    // Order matters as much as membership: the feature layer reads by index.
    for (i, (a, b)) in live.iter().zip(historical.iter()).enumerate() {
        assert_eq!(a, b, "column {i} drifted: live '{a}' vs historical '{b}'");
    }
}

#[test]
fn matched_rows_are_129_wide_with_correct_block_sparsity() {
    let Some(path) = matched_csv() else {
        eprintln!("skipping: no matched CSV under market_data/matched");
        return;
    };
    let table = read_csv(&path).expect("read matched csv");
    assert!(!table.rows.is_empty(), "matched csv has no data rows");

    // +1 for the pandas index column.
    for (i, row) in table.rows.iter().enumerate() {
        assert_eq!(
            row.len(),
            129,
            "row {i} is {} wide, expected 129 (index + 128)",
            row.len()
        );
    }

    // Two different kinds of block, and conflating them is the trap.
    //
    // `depth__*` is an *event payload*: populated only on the 1200 BOOKDEPTH
    // rows that own it, empty everywhere else. That mirrors `agg__price`, which
    // is empty on a LOB row.
    //
    // `depth_state__*` is a *state* block, exactly like `book_state__*`: it is
    // forward-filled on every row, because the as-of depth is defined at every
    // instant even when no depth event just arrived. So "always populated" is
    // correct for it, and asserting absence there would be asserting a lie.
    let et = column(&table, "event_type");
    let depth_ts = column(&table, "depth__timestamp");
    let notional = column(&table, "depth__notional");
    let state_notional = column(&table, "depth_state__notional_-5");
    for (i, row) in table.rows.iter().enumerate() {
        let owns_depth = row[et] == "BOOKDEPTH";
        for col in [depth_ts, notional] {
            let value = &row[col];
            if owns_depth {
                assert!(
                    !value.is_empty(),
                    "row {i} col {}: a BOOKDEPTH row must carry the depth payload",
                    table.header[col]
                );
            } else {
                assert!(
                    value.is_empty(),
                    "row {i} col {}: absent must be empty, not '{value}'",
                    table.header[col]
                );
            }
        }
        // The state block is forward-filled, so it is never absent.
        assert!(
            !row[state_notional].is_empty(),
            "row {i} col {}: depth_state__* is a state block, it must be filled",
            table.header[state_notional]
        );
    }
}

#[test]
fn every_matched_row_is_a_known_event_type_with_matching_blocks() {
    let Some(path) = matched_csv() else {
        eprintln!("skipping: no matched CSV under market_data/matched");
        return;
    };
    let table = read_csv(&path).expect("read matched csv");
    let et = column(&table, "event_type");
    let lob_id = column(&table, "lob__final_update_id");
    let agg_price = column(&table, "agg__price");
    let depth_ts = column(&table, "depth__timestamp");

    // Sparse blocks, and this is the whole point of them: a row carries the
    // payload of exactly one source, and every other source's identity stays
    // empty rather than 0.0.
    for (i, row) in table.rows.iter().enumerate() {
        match row[et].as_str() {
            // A LOB row carries book identity and no trade payload.
            "LOB" => {
                assert!(
                    !row[lob_id].is_empty(),
                    "row {i}: LOB row without an update id"
                );
                assert_eq!(row[agg_price], "", "row {i}: LOB row carries a price");
                assert_eq!(
                    row[depth_ts], "",
                    "row {i}: LOB row carries a depth timestamp"
                );
            }
            // A TRADE row carries the trade payload and no book identity.
            "TRADE" => {
                assert!(
                    !row[agg_price].is_empty(),
                    "row {i}: TRADE row with no price"
                );
                assert_eq!(row[lob_id], "", "row {i}: TRADE row carries an update id");
                assert_eq!(
                    row[depth_ts], "",
                    "row {i}: TRADE row carries a depth timestamp"
                );
            }
            // BOOKDEPTH comes from the downloaded `bookDepth` dataset and has
            // no live websocket source, so it never appears in a live capture.
            // Its presence here is why `depth__*` must stay optional.
            "BOOKDEPTH" => {
                assert!(
                    !row[depth_ts].is_empty(),
                    "row {i}: BOOKDEPTH row without a depth timestamp"
                );
                assert_eq!(row[agg_price], "", "row {i}: BOOKDEPTH carries a price");
                assert_eq!(row[lob_id], "", "row {i}: BOOKDEPTH carries an update id");
            }
            other => panic!("row {i}: unknown event_type '{other}'"),
        }
    }
}

// =============================================================================
// Online: the real binary against the real feed.
// =============================================================================

/// The loss counters `record` prints on shutdown.
///
/// Parsed from stderr rather than read from a struct: the contract is with the
/// operator reading the log, so that is what must be asserted.
#[derive(Debug, Default)]
struct LossReport {
    covered: u64,
    foreign: u64,
    staged: u64,
    gaps: u64,
}

/// Pull the final `[capture] applied=...` line's counters out of stderr.
///
/// Returns `None` when the summary line never appeared, which is itself a
/// failure: a capture that ends without accounting cannot be trusted.
fn parse_loss(stderr: &str) -> Option<LossReport> {
    let summary = stderr
        .lines()
        .rfind(|l| l.contains("applied=") && l.contains("staged_dropped="))?;
    let grab = |key: &str| -> Option<u64> {
        summary
            .split_whitespace()
            .find_map(|token| token.strip_prefix(key))
            .and_then(|v| v.parse().ok())
    };
    Some(LossReport {
        covered: grab("covered_dropped=").unwrap_or(0),
        foreign: grab("foreign_dropped=").unwrap_or(0),
        staged: grab("staged_dropped=").unwrap_or(0),
        gaps: grab("gaps=").unwrap_or(0),
    })
}

#[test]
#[ignore = "needs network + the dev_test binary; run with UETS_LIVE_TEST=1"]
fn live_capture_holds_100ms_cadence_and_reports_its_losses() {
    if std::env::var_os("UETS_LIVE_TEST").is_none() {
        eprintln!("skipping live capture: set UETS_LIVE_TEST=1 to run it (it hits Binance)");
        return;
    }

    let binary = PathBuf::from(env!("CARGO_BIN_EXE_dev_test"));

    // A dedicated directory per run, so a stale capture can never be read as
    // this run's output.
    let dir = std::env::temp_dir().join(format!("uets_live_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp capture dir");

    let started = Instant::now();
    let mut child = Command::new(&binary)
        .current_dir(uets_dir())
        .args([
            "record",
            "btcusdt",
            dir.to_str().expect("utf8 temp path"),
            "--rows",
            LIVE_ROWS,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dev_test record");

    // Bounded wait. A hung capture must fail the test, not hang the suite.
    let deadline = Instant::now() + LIVE_TIMEOUT;
    loop {
        match child.try_wait().expect("try_wait failed") {
            Some(_) => break,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("capture did not finish within {LIVE_TIMEOUT:?}");
            }
            None => std::thread::sleep(Duration::from_millis(200)),
        }
    }
    let wall = started.elapsed();
    let result = child.wait_with_output().expect("collect output");
    let stderr = String::from_utf8_lossy(&result.stderr);

    let csv = std::fs::read_dir(&dir)
        .expect("list capture dir")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "csv"))
        .unwrap_or_else(|| panic!("no CSV written into {}\nstderr:\n{stderr}", dir.display()));
    let table = read_csv(&csv).expect("read capture csv");

    // -- Report: what a human needs to see with --nocapture ----------------

    println!("\n=== live capture ===");
    println!("file      {}", csv.display());
    println!("rows      {}", table.rows.len());
    println!(
        "wall      {:.1?} ({:.1} rows/s)",
        wall,
        table.rows.len() as f64 / wall.as_secs_f64()
    );
    println!("columns   {}", table.header.len());
    for kind in ["LOB", "TRADE"] {
        if let Some(c) = cadence(&table, kind) {
            println!(
                "{kind:<5} n={:<5} p50={}ms p90={}ms p99={}ms max={}ms in_band={:.0}%",
                c.count,
                c.p50,
                c.p90,
                c.p99,
                c.max,
                c.in_band * 100.0
            );
        }
    }
    println!("loss      {:?}", parse_loss(&stderr));
    if !stderr.is_empty() {
        println!("--- stderr ---\n{stderr}");
    }

    // -- 1. Schema parity with the live engine ----------------------------
    assert_eq!(
        table.header.len(),
        128,
        "capture emitted {} columns, schema has {}",
        table.header.len(),
        stream_handler::schema().len()
    );
    for (i, (got, want)) in table
        .header
        .iter()
        .zip(stream_handler::schema().iter())
        .enumerate()
    {
        assert_eq!(
            got.as_str(),
            *want,
            "column {i} drifted in the live capture"
        );
    }
    for (i, row) in table.rows.iter().enumerate() {
        assert_eq!(row.len(), 128, "row {i} is {} wide", row.len());
    }

    // -- 2. Both halves of the feed actually arrived ----------------------
    //
    // The historical failure: the futures aggTrade feed served depth but zero
    // trades. Every row was individually well-formed and the capture "worked".
    let lob = cadence(&table, "LOB").expect("no LOB rows: the book stream never produced");
    let trades = cadence(&table, "TRADE").expect("no TRADE rows: the aggTrade feed was silent");
    assert!(trades.count > 0, "capture produced no trade rows at all");
    assert!(
        lob.count >= 100,
        "only {} LOB rows: too few to judge cadence",
        lob.count
    );

    // -- 3. Cadence: the book is pushed every 100 ms ----------------------
    //
    // What "it slowed down incredibly" looked like in the data was a *sustained*
    // shift: the whole distribution moving up, so p50 left 100 ms and stayed
    // there. That is what these bounds catch.
    //
    // They deliberately do NOT police the tail. A p99/max of several hundred ms
    // is ordinary internet variance and, more to the point, is already accounted
    // for elsewhere: a frame the reorder window cannot place is dropped and
    // counted, and a book that went stale is counted in `gaps`. Failing the test
    // on a tail spike would make every run a coin flip and train the reader to
    // ignore the failure.
    assert!(
        lob.p50 >= 60 && lob.p50 <= 140,
        "LOB p50 gap is {}ms, expected ~{NOMINAL_MS}ms: the feed has slowed down \
         rather than hiccupped",
        lob.p50
    );
    assert!(
        lob.in_band >= 0.60,
        "only {:.0}% of LOB gaps land in 60..=140ms (expected ~{NOMINAL_MS}ms): \
         the distribution has drifted, not just its tail",
        lob.in_band * 100.0
    );
    // A loose sanity floor. If this ever trips the cause is a dead feed, not
    // jitter, and `gaps` in the loss report will say so too.
    assert!(
        lob.max <= 30_000,
        "LOB max gap is {}ms: the feed was dead for half a minute",
        lob.max
    );

    // -- 4. Loss is explicit, never silent --------------------------------
    //
    // A capture that drops events is still useful; one that drops them without
    // saying so is a trap. Parse it here so the report above stays a borrow.
    let loss = match parse_loss(&stderr) {
        Some(report) => report,
        None => panic!("capture ended without a loss summary:\n{stderr}"),
    };
    let dropped = loss.covered + loss.foreign + loss.staged;
    if dropped > 0 || loss.gaps > 0 {
        eprintln!(
            "\n*** INCOMPLETE CAPTURE ***\n\
             {dropped} events dropped (covered={} foreign={} staged={}), \
             {} sequence gaps seen.\n\
             This is NOT a complete replay dataset: treat book_state__* after \
             a gap as untrustworthy.",
            loss.covered, loss.foreign, loss.staged, loss.gaps
        );
    }
    // Events dropped before the seed landed mean the *head* of the capture is
    // missing. That is not a rewind mid-stream, it is a hole in the record, so
    // it fails rather than warns.
    assert_eq!(
        loss.staged, 0,
        "{} events were dropped before the seed landed: the head of this capture \
         is missing and must not be replayed as complete",
        loss.staged
    );

    let _ = std::fs::remove_dir_all(&dir);
}
