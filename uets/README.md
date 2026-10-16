# Commands:
cargo check            : compiles the crate
cargo test             : runs all tests
cargo run --bin dev_test -- help         : prints help
cargo run --bin dev_test -- market [symbol] [datatype]     : symbol, datatype (aggTrade|depth@100ms|...)
cargo run --bin dev_test -- depth [symbol]     : symbol
cargo run --bin dev_test -- kline [symbol]     : symbol
cargo run --bin dev_test -- ticker [symbol]     : symbol
cargo run --bin dev_test -- all-tickers     : none
cargo run --bin dev_test -- multi [symbols...]     : one or more symbols
cargo run --bin dev_test -- record [args...]     : record args
cargo run --bin dev_test -- monitor [symbol]     : symbol
cargo run --bin dev_test -- diag [symbol] [dir]     : symbol, output dir
cargo run --bin dev_test -- strategies [name]     : strategy name (optional)
cargo run --bin dev_test -- strategy <name> <symbol>     : strategy name, symbol (ofmr btc, ofmr eth, ...)
cargo run --bin dev_test -- user-stream     : none (needs API key)
cargo run --bin dev_test -- user-stream-ws     : none (needs API key)
