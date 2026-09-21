//! `main.rs` — which operation to run.
//!
//! The only entry point. Parses CLI arguments, picks a mode, and hands off
//! to `parquet.rs`. Owns none of the replay/book logic itself.
//!
//! ```bash
//! cargo run -- ingest-parquet --input day.parquet --output-dir snapshots/ --tick-size 0.1
//! cargo run -- matched --input trades.parquet --output book_states.parquet --tick-size 0.1
//! ```

mod book;
mod parquet;
mod replay;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return ExitCode::FAILURE;
    }

    let mode = args[1].as_str();
    let rest = &args[2..];

    let result = match mode {
        "ingest-parquet" => run_ingest(rest),
        "matched" => run_matched(rest),
        "-h" | "--help" => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown mode `{other}`").into()),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "\
lob_replay — L2 order-book replay CLI

USAGE:
    lob_replay ingest-parquet --input <FILE> --output-dir <DIR> --tick-size <F> [--warm-start]
    lob_replay matched        --input <FILE> --output <FILE>    --tick-size <F> [--warm-start]

MODES:
    ingest-parquet   write ONE full-book snapshot per UTC day
                     (taken at that day's first valid hour-00 event)
    matched          write ONE full-book snapshot for EVERY valid event

OPTIONS:
    --input <FILE>        input parquet file (required)
    --output <FILE>        output parquet file (matched, required)
    --output-dir <DIR>       output directory (ingest-parquet, required)
    --tick-size <F>          instrument tick size, e.g. 0.1 (required)
    --warm-start             accept diff-only streams with no leading snapshot
"
    );
}

type BoxError = Box<dyn std::error::Error>;

struct IngestArgs {
    input: PathBuf,
    output_dir: PathBuf,
    tick_size: f64,
    warm_start: bool,
}

struct MatchedArgs {
    input: PathBuf,
    output: PathBuf,
    tick_size: f64,
    warm_start: bool,
}

fn run_ingest(args: &[String]) -> Result<(), BoxError> {
    let parsed = parse_ingest_args(args)?;

    let stats = parquet::ingest_parquet(
        &parsed.input,
        &parsed.output_dir,
        parsed.tick_size,
        parsed.warm_start,
    )?;

    println!("{stats:#?}");

    Ok(())
}

fn run_matched(args: &[String]) -> Result<(), BoxError> {
    let parsed = parse_matched_args(args)?;

    let stats = parquet::process_matched(
        &parsed.input,
        &parsed.output,
        parsed.tick_size,
        parsed.warm_start,
    )?;

    println!("{stats:#?}");

    Ok(())
}

fn parse_ingest_args(args: &[String]) -> Result<IngestArgs, BoxError> {
    let mut input = None;
    let mut output_dir = None;
    let mut tick_size = None;
    let mut warm_start = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(PathBuf::from(next_value(args, &mut i, "--input")?));
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_value(args, &mut i, "--output-dir")?));
            }
            "--tick-size" => {
                tick_size = Some(next_value(args, &mut i, "--tick-size")?.parse::<f64>()?);
            }
            "--warm-start" => {
                warm_start = true;
                i += 1;
            }
            other => return Err(format!("unknown flag `{other}`").into()),
        }
    }

    Ok(IngestArgs {
        input: input.ok_or("--input is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
        tick_size: tick_size.ok_or("--tick-size is required")?,
        warm_start,
    })
}

fn parse_matched_args(args: &[String]) -> Result<MatchedArgs, BoxError> {
    let mut input = None;
    let mut output = None;
    let mut tick_size = None;
    let mut warm_start = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(PathBuf::from(next_value(args, &mut i, "--input")?));
            }
            "--output" => {
                output = Some(PathBuf::from(next_value(args, &mut i, "--output")?));
            }
            "--tick-size" => {
                tick_size = Some(next_value(args, &mut i, "--tick-size")?.parse::<f64>()?);
            }
            "--warm-start" => {
                warm_start = true;
                i += 1;
            }
            other => return Err(format!("unknown flag `{other}`").into()),
        }
    }

    Ok(MatchedArgs {
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
        tick_size: tick_size.ok_or("--tick-size is required")?,
        warm_start,
    })
}

/// Consume `--flag <value>`, advancing `i` past both tokens.
fn next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, BoxError> {
    let value = args
        .get(*i + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?;
    *i += 2;
    Ok(value.as_str())
}