# OSADP-Core / UETS

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hybrid Python/Rust research and trading repository. On Windows, use Docker for
UETS commands that need Linux Unix-domain sockets; market data is bind-mounted
and remains on the host.

## Run UETS

Run from the repository root:

```bash
docker compose -f docker-compose.dev.yml build uets-dev # Build the Linux Rust/Python development image
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo check --workspace --all-targets # Compile UETS and all workspace components in Linux
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo test --workspace # Run all UETS and component tests in Linux
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo clippy --workspace --all-targets -- -D clippy::correctness -D clippy::suspicious # Run Rust static analysis
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo run --manifest-path uets/Cargo.toml --bin dev_test -- help # Show UETS CLI help
docker compose -f docker-compose.dev.yml run --rm uets-dev cargo run --manifest-path uets/Cargo.toml --bin dev_test -- playback # Run selected playback method from the YAML
docker compose -f docker-compose.dev.yml run --rm uets-dev python tests/test_playback_runner.py # Verify all playback modes; prints SUCCESSFUL
docker compose -f docker-compose.dev.yml --profile live run --rm uets-live run ofmr@btcusdt # Start the OFMR worker through Docker (paper by default)
docker compose -f docker-compose.dev.yml --profile live run --rm uets-live run ofmr@btcusdt ofmr@ethusdt # Start multiple OFMR workers
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
python tests/test_playback_runner.py # Run all three playback smoke tests; prints SUCCESSFUL
python tests/bybit_testnet_order_smoke.py # Verify Bybit testnet market/limit/stop long/short calls; testnet env vars required
```

The TUI supports adding/disabling workers; press `q` to stop. Logs are written
under `logs/uets/`. See [uets/README.md](./uets/README.md) for CLI commands and
TUI controls.

## Configuration and playback status

All UETS mode, playback, and OFMR parameters are now in
[uets/config/execution.yaml](./uets/config/execution.yaml). Set `mode: playback`
and choose `playback.method`: `parametric_monte_carlo` (synthetic paths),
`purged_walk_forward` (single asset), or `cpcv` (CryptoL2 Parquet; multiple
symbols supported). Configure synthetic generation, data paths/symbols,
correlation threshold, purging, strategy, and workers in that file. Playback
is offline and cannot route live orders. For testnet/live, use their modes,
credentials from environment variables, and require `allow_live_trading: true`
for production live orders.

Playback is runnable via the `dev_test playback` command and prints a report
with mean cumulative return, per-trade expectancy, MDD, Sortino, and CVaR05/95.
Its current built-in strategy is a vectorized momentum baseline, **not the
trained OFMR model**; use results accordingly. The Python end-to-end smoke test
executes all three methods and verifies their reports. It uses generated
temporary CryptoL2-format Parquet fixtures for CPCV, not exchange history.
See [uets/trading docs](./uets/README.md) and
[playback details](./components/pipelines/uets_trades/README.md).
