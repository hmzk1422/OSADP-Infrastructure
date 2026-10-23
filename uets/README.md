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

The OFMR socket worker now submits USD-M Futures market orders after a strategy
decision. Choose `paper`, `testnet`, or `live` in
`uets/config/execution.yaml`; the checked-in default is `playback`, so live
workers reject startup until you choose an order-routing mode. OFMR model,
feature, and trade parameters are in `uets/config/ofmr.yaml`. Testnet/live
require `BINANCE_API_KEY` and `BINANCE_API_SECRET`. Live also requires
`allow_live_trading: true` in that config file. Provide only the credentials
through environment variables or Docker Compose secrets.

The worker limits entry notional using `trade.max_position_usd` and exits with
`reduce_only`. Order/transport failures terminate that worker rather than
silently converting to paper mode. Live routing is not yet production-ready:
fill/account reconciliation, authoritative portfolio persistence, and the
RocksDB session writer are not yet connected to this process.

# Offline playback

Edit `uets/config/execution.yaml`, set `mode: playback`, and choose
`playback.method` (`parametric_monte_carlo`, `purged_walk_forward`, or `cpcv`).
Run `python -m uets.playback`; on Windows this command works from the repository
root. The checked-in playback config uses the research state-space Parquet and
the exported fold models to run Rust OFMR on the exact 5-minute feature rows
used by purged chronological walk-forward. It verifies each fold's start/end,
test-row count, and purged training-row count against the model catalog before
reporting results. To use raw CryptoL2 instead, set `playback.source: cryptol2`
and configure its merged Parquet paths; OFMR requires a single symbol and
catalogued models whose test dates match that tape. Monte Carlo reads the
configured Synth1/2/3 generator, while CPCV reads configured CryptoL2 merged
Parquet files/symbols. Correlation masks are prepared before strategy
simulation and paths run on a thread pool. The report includes CVaR05/CVaR95
and standard metrics; OFMR expectancy is trade-count-weighted net R, and
Sortino is annualized from daily equity returns.

Run `python tests/test_playback_runner.py` to execute all three methods plus
OFMR playback against state-space and CryptoL2 fixtures; success prints
`SUCCESSFUL`.
