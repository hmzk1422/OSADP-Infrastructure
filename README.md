# OSADP-Core (OrderFlow Systematic Alpha Discovery Pipeline)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.97.1-orange.svg)](https://www.rust-lang.org/)
[![Status](https://img.shields.io/badge/status-production-green.svg)]()

**OSADP-Core** is a hybrid Python/Rust systematic-trading framework for crypto
orderflow and price action. **Python** is the control plane and research lab
(feature engineering, walk-forward LightGBM training); **Rust** is the hot
path — market-data ingestion, 1-second micro-bar aggregation, 15-feature
scoring, dry-run fill simulation and order routing — running on its own tokio
runtime thread per asset.

---

## Architecture

```
research/ml_state_optimized.py   research backtest (features -> walk-forward -> model)
main/                            live Python packages (config, executor, dashboard)
unified-execution/engine/        Rust strategy engine (venue-agnostic)
unified-execution/live-trade/    real execution: WS/REST venues + RocksDB + HTML report
unified-execution/live-test/     historical replay (paper) runner
unified-execution/bindings/      PyO3 bindings -> the `engine` Python module
scripts/ingestion/               Binance caching / CCXT conversion
build/                           reports, configs, walk-forward models
```

* **Storage:** embedded RocksDB - trade tape, L1 book, equity snapshots,
  closed trades and per-second strategy features. No external DB server.
* **Feeds & venues:** `bybit` / `binance` (spot market data) / `bitfinex`
  (market data + order routing). Read orderflow from one venue while
  executing on another (`--feed` != `--exchange`).
* **Report:** every session writes `build/pnl_report_<symbol>.html` - a
  TradingView-style candle chart with the full RSI(2)/P(WIN) history, equity
  curve, trade markers and a sortable trade list. It regenerates every 30 s
  while a session runs; an open tab auto-refreshes every 15 s.

## Timeframes (how they relate)

| Layer           | Size    | Purpose                                            |
|-----------------|---------|----------------------------------------------------|
| Micro bars      | 1 s     | feature evaluation + fill simulation every second  |
| Macro composite | 300 s   | regime features: Wilder RSI(2), MA slope/accel, .. |
| Report candles  | 1 s x N | client-side aggregation; chip shows grouping       |

The RSI plotted in the report is the exact macro value the strategy gates on,
forward-filled to every second.

---

## Getting Started

### Prerequisites
- Python **3.11**
- Rust **1.97+** (rustup) and **LIBCLANG** (required by RocksDB bindgen):

  ```powershell
  [System.Environment]::SetEnvironmentVariable('LIBCLANG_PATH','C:\LLVM\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin','User')
  Test-Path "$env:LIBCLANG_PATH\libclang.dll"   # must print True
  ```

### Setup

```bash
python -m venv venv
venv\Scripts\activate              # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
cargo build --workspace            # build the Rust workspace
maturin develop --release          # build + install the `engine` Python module
```

---

## Terminal Instructions

All orchestration goes through one entry point:

```text
python unified_runner.py <command> [options]
```

| Command      | What it does                                                     |
|--------------|------------------------------------------------------------------|
| `live-trade` | run the live/paper strategy loop (requires the `engine` module)  |
| `live-test`  | replay recorded parquet/replay tapes through the engine          |
| `backtest`   | full research backtest + walk-forward evaluation                 |
| `doctor`     | validate environment, repo layout, native module                 |

Add `--help` to any command for the full option list.

### Paper-trade with a Bybit feed (recommended first run)

```bash
python unified_runner.py live-trade --exchange bitfinex --feed bybit --mode paper
```

- `--exchange bitfinex` - where orders would route in `--mode live`
- `--feed bybit`        - where market data comes from (bybit / binance / bitfinex)
- `--mode paper`        - simulates fills; `--mode live` routes real orders
- `--capital 10000`     - USD allocated to position sizing
- `--dashboard`         - serve a browser page of executed trades

While it runs, open `build/pnl_report_btcusdt.html` in a browser for the live
TradingView-style report.

### More examples

```bash
# Binance spot feed, paper trading, $5k capital:
python unified_runner.py live-trade --exchange bitfinex --feed binance --mode paper --capital 5000

# Replay a recorded tape against the engine (paper):
python unified_runner.py live-test --ticker btcusdt --release
python unified_runner.py live-test --ticker btcusdt --start-ms 1709344800000 --end-ms 1709431200000

# Environment sanity check:
python unified_runner.py doctor --require-native
```

### Environment variables

| Variable           | Used by   | Purpose                                  |
|--------------------|-----------|------------------------------------------|
| `BYBIT_API_KEY`    | bybit v5  | only needed for `--mode live`            |
| `BYBIT_API_SECRET` | bybit v5  |                                          |
| `BFX_API_KEY`      | bitfinex  | order routing in `--mode live`           |
| `BFX_API_SECRET`   | bitfinex  |                                          |

Market-data feeds are public and need no keys.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No such host is known (os error 11001)` | Your network/VPN DNS cannot resolve an exchange host. The Bybit feed already falls back between `stream.bybit.com` and `ws.bybit.com`; if both fail, switch VPN node or use `--feed binance`. |
| `[seed] historical fetch failed` | Seed klines failed; features warm live and RSI reads its placeholder until the next 300 s boundary. |
| RSI line missing for old sessions | Per-second feature history is persisted since this build; older tape has none. Run a session to fill it. |
| Binance futures streams silent | Some VPN exits receive degraded `fstream` data (`@aggTrade` empty). The binance feed uses **spot** endpoints, which are unaffected. |
| `engine module is not built` | Run `maturin develop --release`, then retry. |

---

## Docker
```
# Pull your newly built image
docker pull ghcr.io/yourusername/osadp-core:latest

# Run a backtest interactively
docker run --rm -it ghcr.io/yourusername/osadp-core:latest backtest

# Run the live trader in the background
docker run -d --name live-bot ghcr.io/yourusername/osadp-core:latest live-trade

docker run -d \
  --name live-trader \
  --restart unless-stopped \
  -v $(pwd)/config.json:/app/config.json \
  ghcr.io/yourusername/osadp-core:latest live-trade

docker run --rm -it \
  -v $(pwd)/config.json:/app/config.json \
  ghcr.io/yourusername/osadp-core:latest backtest

# Stream live execution logs in real-time
docker logs -f live-trader

# Check resource utilization (CPU/Memory usage on your Oracle ARM instance)
docker stats live-trader

# Stop or restart the bot
docker stop live-trader
docker start live-trader
```
## Edge: Android (Termux / proot-distro)
```javascript
pkg install proot-distro && proot-distro install ubuntu   # glibc only!
proot-distro login ubuntu
apt install python3 python3-pip
pip install <linux-aarch64 wheel>                          # self-contained:
                                                           # rustls TLS + bundled RocksDB
python3 unified_runner.py doctor --require-native
python3 unified_runner.py live-trade --tickers btcusd --exchange bitfinex --mode paper
```
