# OSADP-Core (OrderFlow Systematic Alpha Discovery Pipeline)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.97.1-orange.svg)](https://www.rust-lang.org/)
[![Status](https://img.shields.io/badge/status-research%20%2F%20production-green.svg)]()

**OSADP-Core** is a high-performance, hybrid research and production framework engineered for orderflow and price action across a broad range of asset classes. Beyond standard microstructure features, it supports niche data streams including **crypto liquidation cascades** and **logistic freight costs (Crude/Brent oil spreads)**.

The framework utilizes a dual-language architecture: **Python** handles machine learning research, feature labs, and model retraining, while **Rust** manages low-level abstractions, data ingestion, and live market execution.

---

## Core Architecture & Tech Stack

* **Data Storage:** Embedded **RocksDB** for low-latency state storage and **Parquet** for analytical datasets.
* **Research & ML:** Python-driven feature engineering lab with automated polling loops for periodic AI model retraining.
* **Execution & Ingestion:** Rust-powered core for handling high-frequency tick data, order books, and live exchange interaction.

---

## Getting Started

### Prerequisites
* **Python:** Version **3.11** (Recommended for maximum library compatibility)
* **Rust:** Cargo & Rustc **1.97.1**

### Quick Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
cargo build --
```
## Running 24/7
### Local / Edge Use (Termux / Proot-Distro on Android)
Note: Maintained for personal/legacy use cases to replace Raspberry Pi usage.

1. Android System Settings (MIUI / Battery Optimization):

2. Auto-start: Open your device's Security app or go to Settings > Apps > Permissions > Auto-start, and enable Termux.

3. Battery Saver: Go to Settings > Apps > Manage apps > Termux, tap Battery saver, and set it to "No restrictions".

4. Lock in Recents: Open the Recent Apps screen, long-press Termux, and tap the Padlock icon to prevent the OS from killing the background process.
```
termux-wake-lock
pkg update && pkg upgrade -y
pkg install python git clang make -y
termux-setup-storage

git clone <repo-url>
cd OSADP-Core

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run background execution loops
nohup python bot.py > bot.log 2>&1 &
nohup python -m main.execution > execution.log 2>&1 &
```
### Cargo Setup
```
# Set permanently for your user account (persists across restarts)
[System.Environment]::SetEnvironmentVariable('LIBCLANG_PATH', 'C:\LLVM\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin', 'User')

# Set for your current open session immediately
$env:LIBCLANG_PATH = "C:\LLVM\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin"

# Tessting if its present or not
Test-Path "$env:LIBCLANG_PATH\libclang.dll"
```

### Cloud Setup (Oracle VPS)
Coming soon
# Documentation/Research
Each of the following documentation referred here.
- [Documentation on System Architecture](./DEVLOG.md)
- [Research on Edge](./arxiv_link)

