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

The live capture uses Binance as the primary market-data source and Bybit V5
linear as its single fallback. Bybit order-book and trade messages are
normalized into the same event-merger schema. If the active order book stops
updating for 10 seconds, UETS switches sources, clears buffered events from the
old feed, and reseeds from the selected exchange's REST order book before
strategy rows can resume. If both feeds stop producing updates, capture exits
with an error rather than continuing on stale data. This is fixed
Binance-to-Bybit failover; it does not rank multiple alternatives by measured
latency.

For a Linux VPS, use `bash restart_uets.sh ofmr@btcusdt` from the extracted
bundle root to restart after unexpected top-level exits. It records stderr and
restart decisions in `logs/uets/watchdog.log`; the UETS UI remains attached to
the terminal. The watchdog passes `--reuse-rocksdb` on each launch, which
reuses the persisted session ID and its per-strategy trade/portfolio RocksDB
directories. OFMR workers restore their persisted account, open position,
order quantity, barrier levels, and latest market mark from those databases;
cash and unrealized PnL therefore survive an unexpected process restart. If
recovery state is missing while old portfolio/trade records exist, the worker
fails closed rather than silently resetting the account. Older sessions
created before account recovery was added need an intentional fresh session;
their records do not contain enough information to safely reconstruct an open
position. If the `q` command exits normally, the watchdog stops. Five
consecutive unexpected exits stop the watchdog with an error instead of
repeating a crash loop indefinitely. A new ordinary `uets run` (without that
flag) creates a fresh session.

The watchdog prevents duplicate supervisors with a lock, records the child PID,
and reports nonzero exits including likely OOM kills (and cgroup OOM counters
when available). It stops after five consecutive failures by default so an
unrecoverable startup fault does not loop forever; configure
`UETS_MAX_RESTARTS`, `UETS_RESTART_DELAY_SECONDS`, and `UETS_STABLE_SECONDS`
to tune that policy.

For telemetry backup, set `AWS_ARCHIVE_URI`, `AWS_REGION`, and
`UETS_LIVE_DB`, then run `./cloud.sh [SESSION_ID]`. The binary creates a
consistent checkpoint through the RocksDB handle already open in UETS, verifies
that it reopens, uploads a compressed copy to S3 Glacier Instant Retrieval,
downloads and checksums it, then verifies a separate restore. The live database
is never removed. See [VPS_SERVER_SETUP.md](../VPS_SERVER_SETUP.md) for the
daily systemd timer and IAM requirements. The Linux Docker OOM supervisor
fixture is `unit-tests/production_check/crash_recovery_test.sh`.

## Runtime cost and scaling

The aggregate monitor reports summed RSS and CPU across workers with current
samples, plus an arithmetic 100-worker projection from the sampled per-worker
mean. This projection assumes independent workers with the same workload; it is
a capacity estimate, not a measurement of 100 workers. RSS is resident memory
per worker and may count shared pages more than once. CPU is sampled process
CPU as a percentage of one core, so summed CPU can exceed 100% on a multicore
host. The worker's `Strategy ms` is its latest compute observation; use latency
p95/p99 over the rolling histogram to describe typical and tail strategy time.
For a defensible capacity claim, report the host/container limits, build mode,
strategy and feed workload, sample duration, worker count, and observed totals
alongside the projection.

## Linux VPS executable

The Linux CI job publishes a release-mode x86_64 bundle built on Amazon Linux
2023 named `uets-linux-x86_64-amazonlinux2023`. Download
`uets-linux-x86_64.tar.gz` from the successful workflow run's artifacts and
extract it on the VPS. The archive contains
`bin/uets`, editable defaults under `uets/config/`, and the OFMR model file.
Run it from the extracted directory so runtime data, logs, sockets, and the
live database are stored there:

```sh
tar -xzf uets-linux-x86_64.tar.gz
cd uets-linux-x86_64
./bin/uets run ofmr@btcusdt
```

Edit `uets/config/execution.yaml` and `uets/config/ofmr.yaml` on the VPS; the
binary reads them at startup, so restart the relevant process after changing a
file. Set `UETS_HOME` to the deployment directory when launching from another
working directory. `UETS_EXECUTION_CONFIG` and `OFMR_SETTINGS_PATH` can point
to alternate YAML files. The bundle changes the default execution mode to
`live-test` (real-time public data, simulated orders); set `testnet` or
`live-trade` explicitly to route orders. Keep API credentials in environment
variables, not in the bundle.

The artifact targets Linux x86_64 and Amazon Linux 2023's glibc baseline. A
native executable avoids installing Rust/Cargo and Cargo's build-time memory
use on the VPS. It does not remove UETS runtime processes: the live dashboard
still starts feed and strategy workers, so measure RSS under the intended
workload before assuming a 1 GB host is sufficient.

Each `run` also records a one-second telemetry snapshot in its own RocksDB at
`market_data/live_db/telemetry/<run-id>/`. A snapshot contains total RSS and
CPU across running strategy worker processes (feed-process overhead is not
included), per-connection readings, and cumulative
strategy-compute latency sum, observation count, arithmetic mean, p95, and p99
since each worker started. It also stores Binance Futures public API ping RTT
and order-acknowledgement RTT summaries per connection. Public ping RTT is
sampled every 10 seconds without submitting an order; order-ack RTT measures
the REST order request/response and is only available in `testnet` or
`live-trade` mode (it is unavailable in `live-test`, where orders are simulated).
The RTT means and p95s use each worker's most recent 300 successful samples.
RAM and CPU are point-in-time samples; strategy-latency stats cover all
observations accumulated by each worker. Inspect
the latest samples with:

```sh
cargo run --manifest-path uets/Cargo.toml --bin dev_test -- db <run-id>
```

OFMR keeps rolling feature history per strategy/asset. History is capped at
2,048 bars; an extreme time jump resets that history instead of allocating
every missing bar. This prevents per-ticker feature state from growing without
bound, while preserving the configured rolling features after warm-up.

The portfolio equity curve retains the latest 3,001 samples per worker. Lifetime
Sortino and maximum drawdown are maintained as running statistics, so frequent
market marks do not grow per-worker memory without bound.

Trade and portfolio records remain in their existing
stores; workers use run-prefixed per-instance session IDs so their RocksDB
writers do not contend for the same database lock.
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
fill/account reconciliation and authoritative portfolio persistence are not
connected. Trade, portfolio, and telemetry RocksDB records are best-effort
development records, not a reconciled account ledger.

# Offline playback

Offline research is independent of the execution mode and never connects to
an exchange. Run `run ofmr@pwf`, `run ofmr@cpcv`, or `run ofmr@mc` to select
purged walk-forward, CPCV, or Monte Carlo. Available strategies are compiled
into the executable from modules under `uets/live_strategy/`, including the
minimal `example_strategy` reference bundle. Results,
including the equity-path CSV and terminal path chart, are written under
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
