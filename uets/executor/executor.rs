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

use std::process::ExitCode;

use uets::websocket::{monitor, reader};

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
            let symbol = if first().is_empty() { "btcusdt".into() } else { first() };
            let datatype = rest.get(1).cloned().unwrap_or_else(|| "aggTrade".into());
            reader::market_websocket(&symbol, &datatype);
        }
        "depth" => {
            let symbol = if first().is_empty() { "btcusdt".into() } else { first() };
            reader::market_websocket(&symbol, "depth@100ms");
        }
        "kline" => {
            let symbol = if first().is_empty() { "btcusdt".into() } else { first() };
            reader::kline_websocket(&symbol);
        }
        "ticker" => {
            let symbol = if first().is_empty() { "btcusdt".into() } else { first() };
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
        "monitor" => {
            let symbol = if first().is_empty() { "btcusdt".into() } else { first() };
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