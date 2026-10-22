use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use uets_trades::playback::crypto_l2::{load_merged_parquet, prepare_playback};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut current_symbol: Option<String> = None;
    let mut files: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--symbol" => {
                let symbol = args.next().ok_or("--symbol requires a symbol")?;
                current_symbol = Some(symbol.to_ascii_uppercase());
            }
            "--file" => {
                let path = args.next().ok_or("--file requires a Parquet path")?;
                let symbol = current_symbol
                    .as_ref()
                    .ok_or("--symbol must precede each --file group")?;
                files
                    .entry(symbol.clone())
                    .or_default()
                    .push(PathBuf::from(path));
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if files.is_empty() {
        return Err("usage: playback_data --symbol SYMBOL --file PATH [--file PATH ...]".into());
    }

    println!("symbol,timestamp_ms,price");
    for (symbol, paths) in files {
        let mut rows = Vec::new();
        for path in paths {
            rows.extend(load_merged_parquet(&path, &symbol)?);
        }
        let playback = prepare_playback(rows, &symbol);
        for event in playback.events {
            if let Some(price) = event.marks.get(&symbol) {
                println!("{symbol},{},{price}", event.timestamp.timestamp_millis());
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("CryptoL2 playback data export failed: {error}");
            ExitCode::FAILURE
        }
    }
}
