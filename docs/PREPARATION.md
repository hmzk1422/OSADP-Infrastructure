# OSADP-Core Documentation
<div style="display: flex;">
  <div style="flex: 1; padding-right: 10px;">
    ### Left Column
    Your text for the left side goes here. 
  </div>
  <div style="flex: 1; padding-left: 10px;">

# Preparation & Setup

## 1. Prerequisites

| Component | Minimum Version | Installation |
|-----------|----------------|--------------|
| **Rust** | 1.75+ | `choco install rustup` or `rustup install stable` |
| **Python** | 3.9 - 3.11| `py -m venv .venv && .venv\Scripts\activate` |
| **Maturin** | 1.14+ | `pip install maturin` |
| **Git** | 2.30+ | Included with VS Code / Windows Terminal |
## 2. Quick Setup
After cloning this repository, run the following command.
```powershell
# Setting up python env
python -m venv venv
python -m pip install -r requirements.txt

# Setting up rust env
maturin develop --release
```
This builds the `engine` PyO3 module (wrapping the Rust strategy crate) and installs it in the current venv. The module exposes the `Strategy` class used by `unified_runner.py`.
Running the python build creates all the necessary components needed to build executor and research scripts.

## 3. Verify the Environment

Run the built-in diagnostics:

```powershell
# For more help
python unified_runner.py -h
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

# About Repository
## Libraries
| Libraries | Description | State |
|-----------|----------------|--------------|
| `scripts/ingestion/*` |  Download historical data for backtesting | Stable |
| `scripts/utils/*` |  Functions used for research | Outdated, in use |
| `research/feature_builder` | Feature building + LLM pipeline for alpha generation automation. | Stable |
| `research/ml_state_builder` | Feature abstraction (Recursive/Non-recursive) | Stable |

## CI/CD

The `.github/workflows/ci.yml` performs the following on every push / PR:

1. **`rust`** — `cargo check --workspace --all-targets` + `cargo test --workspace` on Linux (ubuntu-24.04) and Windows (windows-2025)
2. **`wheels`** — `maturin-action` builds native wheels for `linux-x86_64`, `linux-aarch64`, `windows-x86_64` (Android proot / Oracle Cloud compatible)
3. **`smoke`** — Installs the Linux wheel, validates `unified_runner.py --help` and `doctor`
4. **`release`** — On version tags (`v*`), publishes wheels to PyPI with trusted publishing

# Documentation/How-To
## Diagnostics/Hypothesis
```bash
# Checking latency: Run python file directly ending with diag or diagnostics.
python latency_diag.py --duration 30 --interval 0.05 --warmup 3 --no-persist
python latency_diag.py --exchange bybit --ticker ethusdt --duration 60
python latency_diag.py --help

# Hypothesis testing: Run python file directly ending with hypothesis.
python -m research.xxx_hypothesis
```
## Executor Algorithm
live-trade
live-test
## Research Backtesting
ml_state_optimized
  </div>
</div>