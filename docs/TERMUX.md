# docs/TERMUX.md — Running OSADP-Core on Android (Termux + proot-distro)

The Rust engine ships as an **abi3 manylinux aarch64 wheel** with **rustls**
(pure-Rust TLS) and a statically-linked RocksDB. Nothing needs to be compiled
on-device and there are no OpenSSL/system C dependencies — the wheel is fully
self-contained. This makes a glibc proot distro a first-class target.

> **Docker does NOT work on Android.** Containers require kernel privileges
> (namespaces/cgroups) that Android denies to apps. The GHCR images produced by
> CI target Oracle Cloud / x86 edge servers. On-device, use proot-distro +
> `pip install <wheel>` directly, as described below.

## 1. Install a glibc proot distro (Termux)

```bash
pkg update && pkg install proot-distro
proot-distro install ubuntu      # or debian — must be glibc-based
proot-distro login ubuntu
```

**Do NOT use Alpine/musl** — manylinux wheels are glibc-linked and will fail
to load (`Error loading shared library ld-linux-aarch64.so.1`).
**Do NOT use Termux's native python** — it targets Android bionic, not glibc.

## 2. Install Python + the wheel (inside the proot distro)

```bash
apt update && apt install -y python3 python3-pip curl
python3 -m pip install --upgrade pip

# Option A — from a GitHub Actions run (Actions → CI → wheels-linux-aarch64):
pip install unified_execution-*.whl

# Option B — from PyPI once a version is tagged:
pip install unified-execution
```

## 3. Run the runner

```bash
git clone https://github.com/hmzak1422/intraday-momentum-composite.git
cd intraday-momentum-composite

# environment check (validates native module end-to-end):
python3 unified_runner.py doctor --require-native
# → cargo / maturin lines print as informational `[info]` on wheel-only
#   installs (they're build-time tools, not needed to run the wheel).
#   `OK.` + exit code 0 means the native module and repo are healthy.

# paper-trade over Bitfinex (Bybit feed, Bitfinex execution):
python3 unified_runner.py live-trade --exchange bitfinex --feed bybit --mode paper

# or supervise a whole portfolio of assets from one master table:
python3 unified_runner.py portfolio --config portfolio.json --restart
```

Market data persists to `./data/marketdata.rocksdb` (relative paths work
anywhere). Stop with Ctrl-C — the runner drains and flushes RocksDB cleanly.

## Notes & limitations

- **Memory**: installing the prebuilt wheel needs ~200 MB; never `pip install maturin`
  + compile on-device (Rust builds will OOM on phones).
- **Python version**: any ≥3.9 works (abi3 wheel).
- **Backtest**: `unified_runner.py backtest` shells out to `research/ml_state_optimized.py`,
  which needs the heavy scientific stack (`pandas`, `polars`, …). Install
  `requirements.txt` inside the proot distro if you need it — but live-trade,
  live-test and doctor run from the wheel alone.
- **CI guarantee**: the `smoke-arm64` job validates this exact wheel on
  emulated `linux/arm64` (glibc) on every push.
