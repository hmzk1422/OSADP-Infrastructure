# OSADP-Core / UETS

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hybrid Python/Rust research and trading repository. On Windows, start Docker
Desktop for UETS commands that need Linux Unix-domain sockets; market data is
bind-mounted and remains on the host. Run the Python commands from an activated
repository virtual environment with `python -m pip install -r requirements.txt`.

## Run UETS

Run from the repository root. The playback command runs directly in Python;
Docker commands require the Docker Desktop Linux engine to be running:

Master Command: `docker compose -f docker-compose.dev.yml --profile live run --rm uets-live run strategy@ticker`
- uets/live_strategy/config/execution.yaml to change the mode.
- Testnet/live-test/live-trade: strategy@ticker
- Playback: strategy@method (with master command)

```bash
python -m uets.playback # Run the configured offline research analysis
docker compose -f docker-compose.dev.yml config --quiet # Validate Docker Compose configuration without starting containers
docker compose -f docker-compose.dev.yml build uets-dev # Build the Linux Rust/Python development image
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo check --workspace --all-targets # Compile UETS and all workspace components in Linux
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo test --workspace # Run all UETS and component tests in Linux
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo clippy --workspace --all-targets -- -D clippy::correctness -D clippy::suspicious # Run Rust static analysis
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo run --manifest-path uets/Cargo.toml --bin dev_test -- help # Show UETS CLI help
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo run --manifest-path uets/Cargo.toml --bin dev_test -- playback # Run selected playback method from the YAML
docker compose -f docker-compose.dev.yml run --rm uets-dev python tests/test_playback_runner.py # Verify all playback modes; prints SUCCESSFUL
# Master command to open the TUI and start trading:
#   - Docker (live profile):       docker compose -f docker-compose.dev.yml --profile live run --rm uets-live run ofmr@btcusdt
#   - Docker (multiple workers):    docker compose -f docker-compose.dev.yml --profile live run --rm uets-live run ofmr@btcusdt ofmr@ethusdt
#   - Local Rust:                   cargo run --release --manifest-path uets/Cargo.toml --bin dev_test -- run ofmr@btcusdt ofmr@ethusdt
docker compose -f docker-compose.dev.yml --profile live run --rm uets-live run ofmr@btcusdt ofmr@ethusdt # Start multiple OFMR workers; configure execution.yaml first
docker compose -f docker-compose.dev.yml --profile live run --rm uets-live run ofmr@pwf ofmr@cpcv # Run two offline research jobs; prints SUCCESS 2/2
```

For an interactive development container:

```bash
docker compose -f docker-compose.dev.yml up -d uets-dev # Start the reusable dev container
docker compose -f docker-compose.dev.yml exec uets-dev cargo run --manifest-path uets/Cargo.toml --bin dev_test -- strategies ofmr # List OFMR strategies
docker compose -f docker-compose.dev.yml down # Stop the dev container
```

Local Rust equivalents (Linux/macOS):

```bash
cargo run --release --manifest-path uets/Cargo.toml --bin dev_test -- run ofmr@btcusdt ofmr@ethusdt # Launch UETS TUI and strategy workers
cargo check --workspace --all-targets # Compile UETS and all workspace components
cargo test --workspace # Run all UETS and component tests
cargo clippy --workspace --all-targets -- -D clippy::correctness -D clippy::suspicious # Run Rust static analysis
cargo run --manifest-path uets/Cargo.toml --bin dev_test -- playback # Run YAML-selected playback
python -m uets.playback # Run configured playback directly and print its report
python tests/test_playback_runner.py # Run all three playback smoke tests; prints SUCCESSFUL
```

The Bybit order-acceptance smoke test is optional and submits real orders to
testnet; run it only after setting all `UETS_BYBIT_TESTNET_*` variables and
confirming the configured symbol and order size:
`python tests/bybit_testnet_order_smoke.py`.

Before starting an OFMR worker, set `mode: live-test` in
[execution.yaml](./uets/config/execution.yaml) to use exchange market data with
simulated orders and no exchange order connection. `testnet` routes orders to
the exchange test network. `live-trade` routes real orders and requires API
credentials plus `allow_live_trading: true`. Workers run until stopped with
`q` or Ctrl+C.

## Pre-loaded tickers

Set `tickers: [` ofmr@btcusdt, ofmr@ethusdt `]` in
[execution.yaml](./uets/config/execution.yaml) to automatically start the TUI
with those strategy@asset instances already loaded. If no tickers are configured
and none are passed on the command line, the TUI will start with a prompt to
configure tickers.

The TUI supports adding/disabling workers; press `q` to stop. Logs are written
under `logs/uets/`. See [uets/README.md](./uets/README.md) for CLI commands and
TUI controls.

## Configuration and playback status

UETS execution mode and offline research settings are in
[uets/config/execution.yaml](./uets/config/execution.yaml); OFMR model, feature,
and trade settings are separate in [uets/config/ofmr.yaml](./uets/config/ofmr.yaml).
Choose `playback.method`: `parametric_monte_carlo` (synthetic paths),
`purged_walk_forward` (single asset), or `cpcv` (CryptoL2 Parquet; multiple
symbols supported). Analysis selectors such as `run ofmr@pwf ofmr@cpcv`
discover strategy folders under `uets/live_strategy/`, run offline without
connecting to an exchange, and save reports/equity paths in
`market_data/live_db/playback/`. The path chart is rendered in the terminal.
Configure synthetic generation, data paths/symbols,
correlation threshold, purging, strategy, and workers in that file. Playback
analysis is independent of the selected order mode and cannot route live orders.
CryptoL2 scans project only the required price/signal columns and aggregate each
Parquet file to `bar_seconds` before collecting it, rather than keeping raw event
rows from the full dataset in memory. When live asset instances and research
selectors are supplied together, research runs first and the live dashboard
starts afterward so the two workloads do not overlap.

Research is runnable via `dev_test playback` or `run <strategy>@pwf|mc|cpcv`
and prints a report
with mean cumulative return, per-trade expectancy, MDD, Sortino, and CVaR05/95.
Its current built-in strategy is a vectorized momentum baseline, **not the
trained OFMR model**; use results accordingly. The Python end-to-end smoke test
executes all three methods and verifies their reports. It uses generated
temporary CryptoL2-format Parquet fixtures for CPCV, not exchange history.
Research and portfolio configuration now lives under `strategy_system/config/`.
See [uets/trading docs](./uets/README.md) and
[playback details](./components/pipelines/uets_trades/README.md).
