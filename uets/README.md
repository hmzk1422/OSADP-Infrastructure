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
cargo run --release --bin dev_test -- run [<strategy>@<asset> ...] : launch the TUI and isolated workers
cargo run --bin dev_test -- playback          : run YAML-selected offline playback
cargo run --release --bin dev_test -- run <strategy>@pwf|mc|cpcv ... : run offline research jobs
cargo run --bin dev_test -- user-stream     : none (needs API key)
cargo run --bin dev_test -- user-stream-ws     : none (needs API key)
docker compose -f docker-compose.dev.yml --profile live run --rm uets-live

From the repository root, start multiple live market-data/OFMR monitor
instances with:

```sh
cargo run --release --manifest-path uets/Cargo.toml --bin dev_test -- run ofmr@btcusdt ofmr@ethusdt
```

The terminal monitor owns strategy lifecycle and uses Unix-domain sockets.
Press `a` and enter an instance such as `ofmr@solusdt` to add it disabled;
select it with `j`/`k`, then press `e` to enable or disable it. Press `x` to
delete a disabled instance only, `Enter` to inspect it, `Esc` for aggregate,
and `q` to stop all workers and quit. You can also start with no instances:

```sh
cargo run --release --manifest-path uets/Cargo.toml --bin dev_test -- run
```

Worker and feed stderr/stdout are separated into `logs/uets/strategy_<strategy>_<asset>.log`
and `logs/uets/feed_<asset>.log`; each process start overwrites its own file.
The GUI displays stale telemetry and process failures with a pointer to the
relevant log. On Windows, use the Linux Docker profile documented in
`docs/system_design/uets.md`.
# Live order modes

The OFMR socket worker supports three execution modes in
`uets/config/execution.yaml`: `live-test` uses real-time public market data but
simulates all orders locally; `testnet` sends orders to the exchange test
network; `live-trade` sends real orders. Testnet/live-trade require
`BINANCE_API_KEY` and `BINANCE_API_SECRET`. Live-trade also requires
`allow_live_trading: true`. Provide credentials only through environment
variables or Docker Compose secrets.

The worker limits entry notional using `trade.max_position_usd` and exits with
`reduce_only`. Order/transport failures terminate that worker rather than
silently converting to paper mode. Live routing is not yet production-ready:
fill/account reconciliation, authoritative portfolio persistence, and the
RocksDB session writer are not yet connected to this process.

# Offline playback

Offline research is independent of the execution mode and never connects to
an exchange. Run `run ofmr@pwf`, `run ofmr@cpcv`, or `run ofmr@mc` to select
purged walk-forward, CPCV, or Monte Carlo. Available strategies are discovered
from folders under `uets/live_strategy/`. Results, including the equity-path
CSV and terminal path chart, are written under
`market_data/live_db/playback/`. The Python runner can also be invoked directly
with `python -m uets.playback`. Configure inputs and method-specific settings
in `uets/config/execution.yaml`, `execution_cpcv.yaml`, and `execution_mc.yaml`.
CryptoL2 inputs are projected and aggregated to the configured bar interval
one Parquet file at a time before collection, keeping raw event rows out of the
backtest's in-memory working set. If live `strategy@asset` instances and
research selectors are passed to the same `run` command, research completes
before the live dashboard starts.

Run `python tests/test_playback_runner.py` to execute the research methods plus
OFMR playback against state-space and CryptoL2 fixtures; success prints
`SUCCESSFUL`.
