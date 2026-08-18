# Preparation & Setup — OSADP-Core

## 1. Prerequisites

| Component | Minimum Version | Installation |
|-----------|----------------|--------------|
| **Rust** | 1.75+ | `choco install rustup` or `rustup install stable` |
| **Python** | 3.9+ | `py -m venv .venv && .venv\Scripts\activate` |
| **Maturin** | 1.14+ | `pip install maturin` |
| **Git** | 2.30+ | Included with VS Code / Windows Terminal |

## 2. Build the Native Extension

From the repository root:

```powershell
# Compile the Rust engine into a native Python wheel
maturin develop --release
```

This builds the `engine` PyO3 module (wrapping the Rust strategy crate) and installs it in the current venv. The module exposes the `Strategy` class used by `unified_runner.py`.

## 3. Verify the Environment

Run the built-in diagnostics:

```powershell
python unified_runner.py doctor
```

Expected output:

```
[OK] cargo available
[OK] maturin available
[OK] research backtest script
[info] native module 'engine' present
OK.
```

If `[OK] native module 'engine' present` is missing, re-run `maturin develop --release`.

## 4. Running the Pipeline

| Command | Description |
|---------|-------------|
| `python unified_runner.py backtest` | Runs research backtest (`research/ml_state_optimized.py`) |
| `python unified_runner.py live-trade --tickers btcusdt ethusdt --exchange bitfinex --mode paper --capital 5000` | Launches in-process Rust strategies for tickers (paper mode) |
| `python unified_runner.py live-test --ticker btcusdt --release` | Replays historical data through the Rust engine (paper trading) |
| `python unified_runner.py portfolio` | (see `unified_runner.py` help) — manage 50+ assets via `Portfolio` class |

## 5. Library & Import Overview

### Rust (`unified-execution/`)

| Crate | Used By | Status |
|-------|---------|--------|
| `engine` | `bindings`, `live-trade` | ✅ Core strategy engine |
| `live-trade` | `bindings`, CLI `main.rs` | ✅ Execution loop + exchange abstraction |
| `live-test` | (test target) | ✅ Historical replay |
| `bindings` | `unified_runner.py` | ✅ PyO3 bridge to Python |
| `engine-bindings` (within `bindings/`) | Python `engine` module | ✅ PyO3 `#[pymodule]` |

**Unused in this release:** None. All declared dependencies (`tokio`, `serde_json`, `pyo3`) are actively referenced in the crate source files. If you add new Rust crates, remember to add `#[cfg(test)]` modules or unused imports will trigger `cargo warn`.

### Python (`unified_runner.py`, `research/ml_state_optimized.py`, etc.)

| Module | Purpose | Status |
|--------|---------|--------|
| `unified_runner.Strategy` | One-Rust-loop-per-ticker control plane | ✅ Used |
| `unified_runner.Portfolio` | Coordinates 50+ `Strategy` instances | ✅ Used |
| `research.ml_state_optimized` | Feature engineering + model training | ✅ Used by `backtest` command |
| `main.execution` | Position tracking (reference implementation) | ⚠️ Exists for reference; not imported at runtime |
| `scripts.ingestion.*` | Binance downloader / CCXT converter | ✅ Used for data onboarding |
| `scripts.utils.*` | Global helper functions | ✅ Used across scripts |

**Note:** The `main/` package at repo root contains standalone Python applications (`execution.py`, `config.py`). These are **not** imported by the core runner — they are provided as reference implementations and can be deleted or replaced when you have your own counterparty setup.

## 6. CI / CD

The `.github/workflows/ci.yml` performs the following on every push / PR:

1. **`rust`** — `cargo check --workspace --all-targets` + `cargo test --workspace` on Linux (ubuntu-24.04) and Windows (windows-2025)
2. **`wheels`** — `maturin-action` builds native wheels for `linux-x86_64`, `linux-aarch64`, `windows-x86_64` (Android proot / Oracle Cloud compatible)
3. **`smoke`** — Installs the Linux wheel, validates `unified_runner.py --help` and `doctor`
4. **`release`** — On version tags (`v*`), publishes wheels to PyPI with trusted publishing

## 7. Tier-1 Grade Checklist

- [x] Rust workspace with `Cargo.toml` at root, members per subsystem
- [x] Maturin-compiled Python extension (`engine` module)
- [x] Exchange-agnostic `Exchange` trait (`bybit`, `bitfinex` implementations, direct WS + REST, no SDK)
- [x] `Strategy` class with `run()`/`pause()`/`resume()`/`stop()`/`reallocate()`/`status()` — callable from Python
- [x] `Portfolio` class for concurrent 50+ asset management
- [x] Multi-platform CI (Linux + Windows matrix)
- [x] Docs: `PREPARATION.md`, `DEVLOG.md`, `trade_journal.md`
- [x] `.gitignore` ignoring `target/`, `venv/`, caches, DB snapshots
- [x] Output artifacts directory: `build/` (HTML reports, CSV trade logs)
- [x] RocksDB persistence at `data/marketdata.rocksdb/`