# Preparation & Setup

This guide gets the system installed and verified. For **how to configure and
run trades**, see the [README](../README.md) — it's the only guide you need for
day-to-day use.

## 1. What you need

| Component | Version | Why |
|---|---|---|
| Python | 3.11 | runs the supervisor + runner (`unified_runner.py`) |
| Rust | 1.75+ (`rustup`) | builds the trading engine (`uets_rs`) |
| LIBCLANG | — | RocksDB (compiled into the engine) needs it |

Windows: `choco install rustup llvm` · macOS: `brew install rustup llvm` ·
Linux: your package manager (`libclang-dev`, `clang`).

## 2. Install

```bash
python -m venv venv
venv\Scripts\activate              # macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

Build the Rust engine and the Python native module:

```bash
cargo build --workspace
maturin develop --release
```

## 3. Verify

```bash
python unified_runner.py doctor
```

Expected:

```
[OK] cargo available
[OK] maturin available
[OK] research backtest script
[OK] native module 'engine' present
OK.
```

If `native module 'engine'` is missing, re-run `maturin develop --release`.

## 4. Next step

Copy the example config and edit it to your liking:

```bash
copy config\portfolio.example.json config\portfolio.json
```

Then follow  **Configure → Run** in the README:

```bash
python unified_runner.py portfolio
```

## Troubleshooting installs

| Problem | Fix |
|---|---|
| `pip install` fails on RocksDB/`windows-curses` | install LIBCLANG, then `pip install -r requirements.txt` again |
| `cargo build` errors about `libclang` | `choco install llvm` (Win) / `brew install llvm` (mac) / `apt install libclang-dev clang` (Linux), restart the shell |
| `maturin` not found | `pip install maturin` |
| `doctor` shows old engine | rebuild: `cargo build --workspace && maturin develop --release` |

## Try it risk-free first

Run in **paper mode** (the default) before trading live:

```bash
python unified_runner.py portfolio
```

Runs every asset in `assets` with simulated fills — no real orders, no API keys.