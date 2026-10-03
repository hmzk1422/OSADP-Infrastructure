
//! `main.rs` — CLI entry point for the LOB replay component.
//!
//! The entry point parses CLI arguments and dispatches to:
//!
//!     columnar
//!     ingest-parquet
//!     matched
//!
//! Replay/book mechanics remain in `replay.rs` and `book.rs`.

mod book;
mod columnar;
mod parquet;
mod replay;

use std::path::PathBuf;
use std::process::ExitCode;

type BoxError = Box<dyn std::error::Error>;

fn main() -> ExitCode {
    let args: Vec<String> =
        std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return ExitCode::FAILURE;
    }

    let mode = args[1].as_str();
    let rest = &args[2..];

    let result = match mode {
        "columnar" => columnar::run(rest),

        "ingest-parquet" =>
            run_ingest(rest),

        "matched" =>
            run_matched(rest),

        "-h" | "--help" | "help" => {
            print_usage();
            return ExitCode::SUCCESS;
        }

        other => Err(
            format!("unknown mode `{other}`").into()
        ),
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

    lob_replay ingest-parquet \
        --input <FILE|DIR> \
        --output-dir <DIR> \
        --tick-size <F> \
        [--snapshot-policy <day_start|every_event>] \
        [--warm-start]

    lob_replay matched \
        --input <FILE> \
        --output <FILE> \
        --tick-size <F> \
        [--warm-start]

    lob_replay columnar \
        --input <FILE> \
        --output <FILE> \
        --tick-size <F> \
        --levels <N> \
        --snapshot-policy <day_start|every_event> \
        [--snapshot-dir <DIR>] \
        [--snapshot-levels <N>] \
        [--seed-snapshot <FILE>] \
        [--skip-until <MS>] \
        [--warm-start]


MODES:

    ingest-parquet
        Replay one parquet file OR every parquet file in a directory.

        day_start:
            write one full-book checkpoint per UTC day,
            using the first valid exchange event during UTC hour 00.

        every_event:
            write one full-book checkpoint after every valid
            exchange event.

        Directory mode uses one shared ReplayEngine across
        all parquet files.

    matched
        Replay one parquet file and write a full-book checkpoint
        for every valid exchange event.

    columnar
        Replay one parquet file while preserving every original
        source row and attaching the reconstructed top-N book state.


OPTIONS:

    --input <FILE|DIR>
        Input parquet file or directory.

    --output <FILE>
        Output parquet file.

    --output-dir <DIR>
        Output directory.

    --tick-size <F>
        Instrument tick size, e.g. 0.01.

    --snapshot-policy <day_start|every_event>
        Checkpoint policy for ingest/columnar.

        `every_snapshot` is accepted as a legacy alias
        for `every_event`.

    --levels <N>
        Number of book levels emitted by columnar mode.

    --snapshot-dir <DIR>
        Optional directory for additional checkpoint images
        in columnar mode.

    --snapshot-levels <N>
        Maximum levels retained in columnar checkpoint images.

    --seed-snapshot <FILE>
        Full-book seed snapshot for columnar replay.

    --skip-until <MS>
        Skip source events at or before this market timestamp.

    --warm-start
        Accept diff-only input and rebase after gaps.
"
    );
}

// ============================================================================
// INGEST ARGUMENTS
// ============================================================================

struct IngestArgs {
    input: PathBuf,
    output_dir: PathBuf,
    tick_size: f64,
    warm_start: bool,
    policy: replay::CheckpointPolicy,
}

// ============================================================================
// MATCHED ARGUMENTS
// ============================================================================

struct MatchedArgs {
    input: PathBuf,
    output: PathBuf,
    tick_size: f64,
    warm_start: bool,
}

// ============================================================================
// INGEST
// ============================================================================

fn run_ingest(args: &[String]) -> Result<(), BoxError> {
    let parsed = parse_ingest_args(args)?;

    let stats =
        parquet::ingest_parquet_with_policy(
            &parsed.input,
            &parsed.output_dir,
            parsed.tick_size,
            parsed.warm_start,
            parsed.policy,
        )?;

    println!("{stats:#?}");

    Ok(())
}

// ============================================================================
// MATCHED
// ============================================================================

fn run_matched(
    args: &[String],
) -> Result<(), BoxError> {
    let parsed =
        parse_matched_args(args)?;

    let stats =
        parquet::process_matched(
            &parsed.input,
            &parsed.output,
            parsed.tick_size,
            parsed.warm_start,
        )?;

    println!("{stats:#?}");

    Ok(())
}

// ============================================================================
// PARSE INGEST ARGUMENTS
// ============================================================================

fn parse_ingest_args(
    args: &[String],
) -> Result<IngestArgs, BoxError> {
    let mut input = None;
    let mut output_dir = None;
    let mut tick_size = None;
    let mut warm_start = false;

    let mut policy =
        replay::CheckpointPolicy::DayStart;

    let mut i = 0usize;

    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    next_value(
                        args,
                        &mut i,
                        "--input",
                    )?,
                ));
            }

            "--output-dir" => {
                output_dir = Some(
                    PathBuf::from(
                        next_value(
                            args,
                            &mut i,
                            "--output-dir",
                        )?,
                    ),
                );
            }

            "--tick-size" => {
                let value =
                    next_value(
                        args,
                        &mut i,
                        "--tick-size",
                    )?;

                tick_size =
                    Some(value.parse::<f64>()?);
            }

            "--warm-start" => {
                warm_start = true;
                i += 1;
            }

            "--snapshot-policy" => {
                let value =
                    next_value(
                        args,
                        &mut i,
                        "--snapshot-policy",
                    )?;

                policy =
                    replay::CheckpointPolicy::parse(
                        value,
                    )
                    .ok_or_else(|| {
                        format!(
                            "unknown snapshot policy `{value}`; \
                             expected `day_start` or `every_event`"
                        )
                    })?;
            }

            other => {
                return Err(
                    format!(
                        "unknown flag `{other}`"
                    )
                    .into()
                );
            }
        }
    }

    Ok(IngestArgs {
        input: input.ok_or(
            "--input is required"
        )?,

        output_dir: output_dir.ok_or(
            "--output-dir is required"
        )?,

        tick_size: tick_size.ok_or(
            "--tick-size is required"
        )?,

        warm_start,
        policy,
    })
}

// ============================================================================
// PARSE MATCHED ARGUMENTS
// ============================================================================

fn parse_matched_args(
    args: &[String],
) -> Result<MatchedArgs, BoxError> {
    let mut input = None;
    let mut output = None;
    let mut tick_size = None;
    let mut warm_start = false;

    let mut i = 0usize;

    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    next_value(
                        args,
                        &mut i,
                        "--input",
                    )?,
                ));
            }

            "--output" => {
                output = Some(PathBuf::from(
                    next_value(
                        args,
                        &mut i,
                        "--output",
                    )?,
                ));
            }

            "--tick-size" => {
                let value =
                    next_value(
                        args,
                        &mut i,
                        "--tick-size",
                    )?;

                tick_size =
                    Some(value.parse::<f64>()?);
            }

            "--warm-start" => {
                warm_start = true;
                i += 1;
            }

            other => {
                return Err(
                    format!(
                        "unknown flag `{other}`"
                    )
                    .into()
                );
            }
        }
    }

    Ok(MatchedArgs {
        input: input.ok_or(
            "--input is required"
        )?,

        output: output.ok_or(
            "--output is required"
        )?,

        tick_size: tick_size.ok_or(
            "--tick-size is required"
        )?,

        warm_start,
    })
}

// ============================================================================
// COMMON CLI HELPER
// ============================================================================

/// Consume `--flag <value>` and advance `i` past both tokens.
fn next_value<'a>(
    args: &'a [String],
    i: &mut usize,
    flag: &str,
) -> Result<&'a str, BoxError> {
    let value =
        args.get(*i + 1)
            .ok_or_else(|| {
                format!(
                    "{flag} requires a value"
                )
            })?;

    *i += 2;

    Ok(value.as_str())
}