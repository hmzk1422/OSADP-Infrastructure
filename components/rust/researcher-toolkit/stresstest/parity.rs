//! Parity harness: evaluate a Python-compiled spec over mock rows.
//!
//! Reads `spec.json` (from the Python `FeaturePipeline.compile()`) and
//! `rows.csv` (from `mock_research.py`), runs the real executor feature code,
//! and writes `rust_out.csv`. `test_lossless.py` then compares this against
//! the Python output cell for cell.
//!
//! Usage: `parity <spec.json> <rows.csv> <out.csv> [feature_name ...]`
//!
//! When feature names are given, only those are emitted; otherwise every
//! compiled feature is. Columns are `row,feature,value`, and a value that the
//! engine reports as missing is written as `nan` so the Python side can
//! distinguish "not enough history" from a real measurement.

use std::io::Write;
use std::process::ExitCode;

use parity::ml_feature::{parse_spec, Aggregation, FeatureComputer, FeatureRow, FeatureValue};
use parity::strategy::Reconstructor;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: parity <spec.json> <rows.csv> <out.csv> [feature ...]");
        eprintln!("       parity --reconstruct <rows.csv> [bar_size] [aggregation]");
        return ExitCode::FAILURE;
    }

    // Second mode: exercise `strategy::Reconstructor` directly and print the
    // bars as JSON, so Python can diff them against `reconstruct_rows`.
    if args[0] == "--reconstruct" {
        return run_reconstruct(&args);
    }
    if args.len() < 3 {
        eprintln!("usage: parity <spec.json> <rows.csv> <out.csv> [feature ...]");
        return ExitCode::FAILURE;
    }
    let spec_path = &args[0];
    let rows_path = &args[1];
    let out_path = &args[2];
    let wanted: Vec<String> = args[3..].to_vec();

    let spec_json = match std::fs::read_to_string(spec_path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("cannot read spec {spec_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let specs = match parse_spec(&spec_json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot compile spec: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut computer = match FeatureComputer::new(specs) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cannot build computer: {e}");
            return ExitCode::FAILURE;
        }
    };

    let rows_csv = match std::fs::read_to_string(rows_path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("cannot read rows {rows_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let rows = match parse_rows(&rows_csv) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot parse rows: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut out = String::from("row,feature,value\n");
    for (index, row) in rows.iter().enumerate() {
        let computed = match computer.compute(row) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("row {index}: {e}");
                return ExitCode::FAILURE;
            }
        };
        for (name, value) in computed.iter() {
            if !wanted.is_empty() && !wanted.iter().any(|w| w == name) {
                continue;
            }
            let rendered = match value {
                FeatureValue::Number(v) => v.to_string(),
                FeatureValue::Missing => "nan".to_string(),
            };
            out.push_str(&format!("{index},{name},{rendered}\n"));
        }
    }

    match std::fs::File::create(out_path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(out.as_bytes()) {
                eprintln!("cannot write {out_path}: {e}");
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("cannot create {out_path}: {e}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

/// Drive `strategy::Reconstructor` over a rows CSV and print the bars as JSON.
///
/// Mirrors the Python `reconstruct_rows` contract exactly: one entry per
/// completed bar, no trailing partial bar, each stamped with its close time.
fn run_reconstruct(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: parity --reconstruct <rows.csv> [bar_size] [aggregation]");
        return ExitCode::FAILURE;
    }
    let size: usize = match args.get(2).map(|s| s.parse()) {
        Some(Ok(v)) => v,
        None => 4,
        Some(Err(e)) => {
            eprintln!("bad bar size: {e}");
            return ExitCode::FAILURE;
        }
    };
    let aggregation = match args.get(3).map(String::as_str).unwrap_or("mean") {
        name => match Aggregation::parse(name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad aggregation: {e}");
                return ExitCode::FAILURE;
            }
        },
    };

    let text = match std::fs::read_to_string(&args[1]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let rows = match parse_rows(&text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot parse rows: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut reconstructor = match Reconstructor::new(size, aggregation) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot build reconstructor: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut bars = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let timestamp = row.get("event_time_ms").unwrap_or(index as f64) as i64;
        let value = match row.get("price") {
            Some(v) => v,
            None => {
                eprintln!("row {index} has no 'price' column");
                return ExitCode::FAILURE;
            }
        };
        match reconstructor.push(timestamp, value) {
            Ok(Some(bar)) => bars.push(bar),
            Ok(None) => {}
            Err(e) => {
                eprintln!("row {index}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let payload: Vec<serde_json::Value> = bars
        .iter()
        .map(|bar| {
            serde_json::json!({
                "timestamp_ms": bar.timestamp_ms,
                "row_count": bar.row_count,
                "value": bar.value,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string(&payload).unwrap_or_else(|e| format!("[] /* {e} */"))
    );
    ExitCode::SUCCESS
}

/// Minimal RFC4180 reader for the flat `rows.csv` the mock writes.
///
/// Hand-rolled rather than pulling in a CSV crate: the harness is the only
/// consumer, and a dependency-free build keeps `cargo` cheap here.
fn parse_rows(text: &str) -> Result<Vec<FeatureRow>, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| "rows.csv is empty".to_string())?;
    let columns: Vec<String> = split_csv(header)
        .into_iter()
        .map(|c| c.trim().to_string())
        .collect();
    if columns.first().map(String::as_str) != Some("row") {
        return Err("rows.csv must start with a 'row' column".to_string());
    }

    let mut rows = Vec::new();
    for line in lines {
        let fields = split_csv(line);
        if fields.len() != columns.len() {
            return Err(format!(
                "row has {} fields but header has {}",
                fields.len(),
                columns.len()
            ));
        }
        let mut row = FeatureRow::new();
        for (name, field) in columns.iter().zip(fields.iter()) {
            if name == "row" {
                continue;
            }
            match field.trim().parse::<f64>() {
                Ok(v) => row.insert(name.clone(), v),
                // An empty cell is genuinely absent, not zero.
                Err(_) => row.insert_missing(name.clone()),
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

fn split_csv(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                out.push(std::mem::take(&mut field));
            }
            _ => field.push(ch),
        }
    }
    out.push(field);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_header_and_rows() {
        let csv = "row,bid_price,ask_price\n0,10.0,10.5\n1,10.2,10.7\n";
        let rows = parse_rows(csv).expect("parses");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("bid_price"), Some(10.0));
        assert_eq!(rows[1].get("ask_price"), Some(10.7));
    }

    #[test]
    fn empty_cell_is_missing_not_zero() {
        let csv = "row,bid_price\n0,\n";
        let rows = parse_rows(csv).expect("parses");
        assert_eq!(rows[0].get("bid_price"), None);
    }
}
