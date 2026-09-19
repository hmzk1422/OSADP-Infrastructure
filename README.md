# OSADP-Core
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#)

A hybrid Python/Rust production grade framework specifically for alpha research and multi-strategy live trading. One JSON file (`config/portfolio.json`) drives everything — strategy parameters, which tickers to trade, how much capital each gets, and which exchange/feed to use. Edit that file, save it, and the running system picks up the changes live. No Rust rebuild, no generated config files.

---

## 1. Install

```bash
# 1. clone
git clone https://github.com/hmzk1422/intraday-momentum-composite.git
cd intraday-momentum-composite

# 2. Python deps (3.11 recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Rust engine (one-time native build)
pip install maturin
maturin develop --release

# 4. verify
python unified_runner.py doctor --require-native
```

---

## 2. Configure

Everything lives in **`config/portfolio.json`** — there is no other config file.

```json
{
  "exchange": "bitfinex",
  "feed": "bybit",
  "mode": "paper",
  "capital": 10000,
  "restart": true,
  "status_interval": 5.0,
  "_strategy_defaults": {
    "rr_ratio": 10,
    "confidence_threshold": 0.2,
    "risk_pct_per_trade": 0.002,
    "max_position_usd": 100,
    "lookahead_seconds": 6000,
    "lockout_seconds": 1000,
    "rsi_filter_threshold": 30,
    "trade_direction_is_long": true,
    "use_orderflow_features": true
  },
  "assets": [
    { "ticker": "btcusdt", "feed": "bybit",   "exchange": "bitfinex", "capital": 5000 },
    { "ticker": "ethusdt", "feed": "binance", "exchange": "bitfinex", "capital": 3000,
      "strategy": { "rr_ratio": 5, "confidence_threshold": 0.35 }
    }
  ]
}
```

**Rules:**
- `_strategy_defaults` — strategy parameters shared by every ticker. The Rust engine reads these directly; edit them live (see below).
- `assets` — one entry per ticker. Each inherits `_strategy_defaults` and can override any parameter with its own `strategy` block (only the keys you want to change).
- `mode` — `"paper"` (fake orders, real tape) or `"live"` (real orders on the exchange).
- `feed` — where the market data comes from (`bybit`, `binance`, `bitfinex`).
- `exchange` — where orders are routed (can differ from feed).

---

## 3. Run — Headless (no GUI)

`unified_runner.py` is the only entry point. Every command works headless — the VIM dashboard is optional.

```bash
# single ticker, paper
python unified_runner.py live-trade --ticker btcusdt --exchange bitfinex --feed bybit --mode paper --capital 10000

# single ticker, LIVE ORDERS
python unified_runner.py live-trade --ticker btcusdt --exchange bitfinex --feed bybit --mode live --capital 10000

# multi-asset portfolio (one OS process per ticker, all from config/portfolio.json)
python unified_runner.py portfolio --config config/portfolio.json --restart

# replay historical data through the Rust engine
python unified_runner.py live-test --ticker btcusdt --release

# research walk-forward backtest
python unified_runner.py backtest

# manage RocksDB stores
python unified_runner.py db --list
python unified_runner.py db --clean-ticker btcusdt --yes

# environment / setup self-check
python unified_runner.py doctor --require-native
```

---

## 4. Run — VIM Dashboard

```bash
python unified_runner.py portfolio --config config/portfolio.json --restart
```

This launches a curses terminal dashboard (the supervisor keeps running on a background thread). Three views, toggled with **1 / 2 / 3** or **Tab**:

| View      | Key | What you see |
|-----------|-----|--------------|
| Dashboard | 1   | PnL, equity, trades, win rate, RR, Sortino, MDD-R, RSI, P(WIN) — one row per asset |
| Trades    | 2   | The event stream (entries/exits) for the **selected** asset, plus LAT/STAL/SK stats |
| Config    | 3   | The raw `config/portfolio.json` with a vim-style cursor |

**Navigation:**

| Key | Action |
|-----|--------|
| `j` / `k` | move down / up (assets in dashboard, lines in config) |
| `g` / `G` | jump to top / bottom |
| `Ctrl-d` / `Ctrl-u` | page down / up |
| `Tab` / `h` / `l` | cycle views |
| `Enter` or `i` | edit the config line under the cursor |
| `R` / `F5` | restart the selected asset with the latest config |
| `q` | quit |

**Editing config live:**
1. Switch to Config view (`3`).
2. Move to any line, press **Enter** or **i**.
3. For a parameter line like `"rr_ratio": 10,` the editor prefills just the value (`10`). Type the new value, **Enter** to save, **Esc** to cancel.
4. The save is validated — invalid JSON or a broken config structure is rejected (disk untouched). A valid save writes the file, auto-reloads, and restarts only the assets whose effective config changed.

**:commands** (type `:` in any view, then the command + Enter):

| Command | Action |
|---------|--------|
| `:q` | quit |
| `:w` | reload config from disk |
| `:view dashboard` / `:view trades` / `:view config` | switch view |
| `:restart [ticker]` | restart an asset (no arg = selected asset) |
| `:stop [ticker]` | stop an asset |
| `:start [ticker]` | start a stopped asset |
| `:flatten [ticker]` | close any open position + pause entries |
| `:pause [ticker]` | pause entries (keep monitoring) |
| `:resume [ticker]` | resume a paused asset |

All commands work with no argument (acts on the selected asset) or with a ticker prefix (`:flatten eth` matches `ethusdt`).

**Status indicators:**

| Status   | Meaning |
|----------|---------|
| running  | connected to feed, receiving data, trading |
| outage   | feed staleness > 1000ms (network problem or illiquid market) |
| paused   | entries paused (via `:pause` or `:flatten`) |
| DOWN     | the child process exited (auto-restarts if `restart: true`) |
---

## 5. Outputs

- `build/pnl_report_<ticker>.html` — per-ticker report (equity curve, R-multiple histogram, latency/skew charts, trade table with LAT/STAL/SK columns).
- `build/portfolio_dashboard.html` — multi-asset tabbed dashboard (open in browser; auto-refreshes). Same data as the terminal dashboard plus sparkline charts for latency/staleness/skew.

---

## 6. Termux / Docker (legacy)

See `docs/TERMUX.md` for the Android/Termux setup (runs the same stack inside a Docker container on a phone/tablet).

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `doctor --require-native` fails | Run `maturin develop --release` |
| `ModuleNotFoundError: uets_rs` | Same — native binding not built |
| Dashboard shows all DOWN | Check exchange/feed names; check network |
| Config edit rejected | Check JSON syntax; the validator tells you what's wrong |
| `live-trade` crashes on start | Check `config/portfolio.json` has an `assets` list with valid tickers |

# Important System Commands
**historical backtest ingestion**: keep concurrency deliberately low to avoid saturating disk I/O and network bandwidth. Run **two Binance download workers** for the smaller daily `aggTrades`/`bookDepth` files, while limiting **CryptoHFTData L2 to one worker** because its hourly order-book files are significantly larger and more disk-intensive. This prioritizes stable throughput and reliability over maximum parallelism.

```bash
python ingestion.py \
  --symbol YOUR_SYMBOL \
  --start-date START_DATE \ # 2025-06-28 furthest date
  --end-date END_DATE \ # 2026-09-01 example (ISO date format)
  --workers NUM_OF_WORKERS \
  --overwrite # Use 
  --delete-raw
```