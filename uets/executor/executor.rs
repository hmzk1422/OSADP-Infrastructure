//! Dispatcher for the Binance websocket readers in `websocket/reader.rs`.
//!
//! Run any reader directly through this binary:
//!
//!     cargo run --bin dev_test -- market btcusdt aggTrade
//!     cargo run --bin dev_test -- help
//!
//! The Python extension module (lib.rs) is a separate build target -- build
//! it with `maturin develop` (maturin enables the extension-module feature
//! automatically).

use std::path::PathBuf;
use std::process::ExitCode;

use uets::websocket::{live, monitor, reader};

/// Default capture directory, `market_data/live_db` at the repo root.
///
/// Resolved from the manifest directory rather than the process CWD, because
/// this binary is normally run through `cargo run` from anywhere and a capture
/// landing in a random CWD is worse than one landing in a predictable place.
fn default_live_db_dir() -> String {
    // CARGO_MANIFEST_DIR is <repo>/uets, so exactly one parent reaches the
    // repo root. Two would land beside the repo.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("market_data").join("live_db"))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "market_data/live_db".to_string())
}

fn print_help() {
    println!(
        "usage: dev_test [command] [args...]\n\
         \n\
         commands:\n\
         \x20 market [SYMBOL] [TYPE]  public market stream (default: btcusdt aggTrade)\n\
         \x20 depth [SYMBOL]          diff-depth stream @100ms (default: btcusdt)\n\
         \x20 kline [SYMBOL]          1-minute kline stream (default: btcusdt)\n\
         \x20 ticker [SYMBOL]         24h rolling ticker (default: btcusdt)\n\
         \x20 all-tickers             all-symbols mini-ticker stream (!miniTicker@arr)\n\
         \x20 multi [SYMBOLS...]      multiple diff-depth streams on one connection\n\
         \x20 record [SYMBOL] [DIR]    live capture -> merged CSV under DIR\n\
         \x20                         (default DIR: market_data/live_db)\n\
         \x20                         flags: --rows N   stop after N rows\n\
         \x20                                --append  keep the existing file\n\
         \x20 monitor [SYMBOL]        live tape + 20-level orderbook monitor;\n\
         \x20                         view with:  python uets/display.py [SYMBOL]\n\
         \x20 user-stream             REST listen-key lifecycle (needs API key)\n\
         \x20 user-stream-ws          account/order update stream (needs API key)\n\
         \x20 help                    show this message\n\
         \n\
         examples:\n\
         \x20 dev_test market btcusdt aggTrade\n\
         \x20 dev_test depth btcusdt\n\
         \x20 dev_test multi ethbtc bnbusdt\n\
         \x20 dev_test record btcusdt            # -> market_data/live_db\n\
         \n\
         All streams run until interrupted (Ctrl+C)."
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let rest: Vec<String> = args.iter().skip(1).cloned().collect();
    let first = || rest.first().cloned().unwrap_or_default();

    match cmd {
        "market" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            let datatype = rest.get(1).cloned().unwrap_or_else(|| "aggTrade".into());
            reader::market_websocket(&symbol, &datatype);
        }
        "depth" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            reader::market_websocket(&symbol, "depth@100ms");
        }
        "kline" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            reader::kline_websocket(&symbol);
        }
        "ticker" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            reader::ticker_websocket(&symbol);
        }
        "all-tickers" => reader::all_trades_websocket(),
        "multi" => {
            let symbols: Vec<&str> = if rest.is_empty() {
                vec!["ethbtc", "bnbusdt"]
            } else {
                rest.iter().map(String::as_str).collect()
            };
            reader::multiple_streams(&symbols);
        }
        "record" => {
            // Live capture -> the merged 128-column CSV, byte-compatible with
            // the historical table. Streams until Ctrl+C unless a row cap is
            // given as `--rows N` in the output-dir position's sibling flag.
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            // A flag is not a positional. Without this, `record btcusdt
            // --rows 10` would take "--rows" as the output directory and write
            // the capture into a directory literally named `--rows`.
            let dir = rest
                .iter()
                .skip(1)
                .find(|a| !a.starts_with("--") && !a.chars().all(|c| c.is_ascii_digit()))
                .cloned()
                .unwrap_or_else(default_live_db_dir);
            let mut config = live::CaptureConfig::new(&symbol, dir);
            // `--rows N` bounds the capture, which is what makes it testable
            // and scriptable instead of only interactive.
            if let Some(index) = rest.iter().position(|a| a == "--rows") {
                if let Some(value) = rest.get(index + 1) {
                    match value.parse::<u64>() {
                        Ok(n) => config.max_rows = Some(n),
                        Err(_) => {
                            eprintln!("--rows expects a number, got: {value}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
            if rest.iter().any(|a| a == "--append") {
                config.append = true;
            }
            if let Err(e) = live::run_capture(config) {
                eprintln!("capture failed: {e}");
                return ExitCode::FAILURE;
            }
        }
        "monitor" => {
            let symbol = if first().is_empty() {
                "btcusdt".into()
            } else {
                first()
            };
            monitor::monitor_websocket(&symbol);
        }
        "user-stream" => {
            println!("note: requires a real API key (edit reader::user_stream)");
            reader::user_stream();
        }
        "user-stream-ws" => {
            println!("note: requires a real API key (edit reader::user_stream_websocket)");
            reader::user_stream_websocket();
        }
        "help" | "--help" | "-h" => print_help(),
        other => {
            eprintln!("unknown command: {other}");
            print_help();
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
