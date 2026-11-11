use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use uets::core::feature::FeatureRow;
use uets::core::strategy::{Decision, MarketState, Position};
use uets::live_strategy::load_strategy;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("example strategy parity executable failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let data_path = required_env_path("PARITY_DATA_PATH")?;
    let output_path = required_env_path("PARITY_RUST_OUT")?;
    let symbol = env::var("PARITY_SYMBOL").unwrap_or_else(|_| "example".into());
    let input = File::open(&data_path)
        .map_err(|error| format!("open input {}: {error}", data_path.display()))?;
    let mut lines = BufReader::new(input).lines();
    let header = lines
        .next()
        .transpose()
        .map_err(|error| format!("read input header: {error}"))?
        .ok_or_else(|| format!("input {} is empty", data_path.display()))?;
    if header.trim() != "timestamp_ms,price" {
        return Err(format!(
            "expected CSV header 'timestamp_ms,price', got {header:?}"
        ));
    }

    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create output directory {}: {error}", parent.display()))?;
    }
    let output = File::create(&output_path)
        .map_err(|error| format!("create output {}: {error}", output_path.display()))?;
    let mut writer = BufWriter::new(output);
    writeln!(writer, "timestamp_ms,dz,decision,position")
        .map_err(|error| format!("write output header: {error}"))?;

    let mut strategy = load_strategy("example_strategy")?;
    let mut state = MarketState::new(FeatureRow::new(), 0);
    for (line_index, line) in lines.enumerate() {
        let line = line.map_err(|error| format!("read input row {}: {error}", line_index + 2))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() != 2 {
            return Err(format!(
                "input row {} must contain exactly two fields",
                line_index + 2
            ));
        }
        let timestamp_ms = fields[0]
            .parse::<u64>()
            .map_err(|error| format!("invalid timestamp on row {}: {error}", line_index + 2))?;
        let price = fields[1]
            .parse::<f64>()
            .map_err(|error| format!("invalid price on row {}: {error}", line_index + 2))?;
        if !price.is_finite() {
            return Err(format!("non-finite price on row {}", line_index + 2));
        }

        let mut row = FeatureRow::new();
        row.insert("price", price);
        state.features = strategy.computer.compute(&row)?;
        state.sequence = state.sequence.saturating_add(1);
        let decision = strategy.builder.decide(&state);
        let position = state.apply(decision);
        let dz = state
            .features
            .get("dz")
            .ok_or_else(|| format!("missing dz output on row {}", line_index + 2))?;
        writeln!(
            writer,
            "{timestamp_ms},{dz},{},{}",
            decision_code(decision),
            position_code(position)
        )
        .map_err(|error| format!("write output row {}: {error}", line_index + 2))?;
    }
    writer
        .flush()
        .map_err(|error| format!("flush output {}: {error}", output_path.display()))?;
    println!(
        "Rust example strategy processed {} for {symbol}",
        data_path.display()
    );
    Ok(())
}

fn required_env_path(name: &str) -> Result<PathBuf, String> {
    env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{name} must be set"))
}

fn decision_code(decision: Decision) -> i8 {
    match decision {
        Decision::Hold => 0,
        Decision::EnterLong => 1,
        Decision::EnterShort => -1,
        Decision::Exit => 2,
    }
}

fn position_code(position: Position) -> i8 {
    match position {
        Position::Flat => 0,
        Position::Long => 1,
        Position::Short => -1,
    }
}
