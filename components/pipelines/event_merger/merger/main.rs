//! `event_merger` — unified event timeline over LOB book-state + aggTrades + bookDepth.
//!
//! LOB replay is deliberately outside this binary. The merger reads materialized
//! LOB rows, keeps every aggTrades/bookDepth source row, and writes a sparse,
//! hourly chronological stream. Source-specific fields stay null on other event
//! types; Python may later forward-fill persistent state explicitly.
//!
//! Timeline uses received_time when present, otherwise event_time, for LOB rows;
//! aggTrades use transact_time and bookDepth uses timestamp. Source-specific
//! state stays null until its source is triggered; this binary never mutates
//! or deletes `ingest_parquet` or existing matched samples.

pub mod args;
pub mod layout;
pub mod merge;
pub mod model;
pub mod stream;
pub mod writer;

use std::process::ExitCode;

fn print_usage() {
    eprintln!(
        "event_merger --root <market_data> --symbol <SYM> \
         --source <ingest_parquet|matched> --date <YYYY-MM-DD> \
         [--hour <0..23|all>] [--lob-levels <N>]"
    );
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return ExitCode::SUCCESS;
    }
    match args::parse_args(&raw).and_then(|a| merge::run_merge(&a)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            print_usage();
            ExitCode::FAILURE
        }
    }
}
