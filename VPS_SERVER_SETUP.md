# UETS VPS Server Setup and Storage Operations

This guide covers running the compiled UETS binary with an interactive tmux
TUI, a crash-recovery watchdog, and verified telemetry archives to S3 Glacier
Instant Retrieval.


## Install the compiled UETS package

The v4.0.0 GitHub Release asset is a ZIP containing the `.tar.gz` package. On
the VPS, install `curl`, `unzip`, and `tar` if they are not already available:

```bash
sudo apt update
sudo apt install -y curl unzip tmux zstd util-linux
sudo useradd --system --create-home --shell /bin/bash uets
sudo install -d -o uets -g uets /opt/uets /var/lib/uets/live_db /var/tmp/uets-archive
```

Download the `uets-linux-x86_64-amazonlinux2023` artifact from the successful
GitHub Actions run for the commit containing these changes. The previously
released v4.0.0 binary does not implement the checkpoint commands required by
`cloud.sh`. Extract the downloaded tarball:

```bash
cd "$HOME"
sudo tar -xzf "$HOME/Downloads/uets-linux-x86_64.tar.gz" -C /opt/uets
sudo chown -R uets:uets /opt/uets/uets-linux-x86_64
cd /opt/uets/uets-linux-x86_64
ldd ./bin/uets
./bin/uets help
```

Check `ldd` output for `not found` libraries. The package is built on Amazon
Linux 2023 for Linux x86_64; verify dependencies on the target host.

Configure `uets/config/execution.yaml` and `uets/config/ofmr.yaml` before
starting a worker. `live-test` uses live public market data with simulated
orders. `testnet` submits to the exchange test environment; `live-trade` can
submit real orders and requires credentials and the explicit live-trading
guard. Never put API secrets in these YAML files or in this guide.

Set `uets/config/execution.yaml` to `live-test` initially. Start the interactive
TUI in tmux at boot using the included systemd unit:

```bash
sudo install -D -m 0644 deploy/systemd/uets-tmux.service /etc/systemd/system/uets-tmux.service
sudo systemctl daemon-reload
sudo systemctl enable --now uets-tmux
sudo -u uets tmux attach -t uets
```

The TUI is displayed inside tmux; detach with `Ctrl+B`, then `D`, and reattach
with `sudo -u uets tmux attach -t uets`. The watchdog restarts unexpected
failures. Pressing `q` intentionally exits UETS. Provide AWS access through an
EC2 IAM instance profile rather than hard-coding AWS keys.

## Where UETS stores data

Unless `UETS_LIVE_DB` overrides it, the extracted deployment directory is the
runtime home. Live RocksDB data is written below:

```text
market_data/live_db/
├── trades/<session-id>/
├── portfolio/<session-id>/
└── telemetry/<session-id>/
```

The dashboard writes one system/per-worker resource and strategy-latency
telemetry snapshot per second. RTT-enabled builds also include public Binance
ping and exchange order-ack RTT summaries. The v4.0.0 binary predates the RTT
telemetry changes; use a release built with those changes to see those fields.

The live strategy's RocksDB records contain the tick/features passed to the
strategy, decisions, fills, and portfolio snapshots. They do **not** store the
top-20 bid/ask book arrays. The input may include best bid/ask fields; that is
not the same as persisting 20 levels. Other market-monitor commands have
separate output paths and are not the live strategy RocksDB writer.

Set a different database root before launching UETS if needed:

```bash
export UETS_LIVE_DB="$HOME/uets-data/live_db"
```

The active session ID is stored in
`$UETS_LIVE_DB/current_session`; the monitor and archive script read it
automatically. The UETS startup output also prints
`telemetry session=<session-id>`.

## `telemetry.sh`: live host monitoring without a log file

This monitor prints host memory, aggregate UETS-process RSS, CPU, and disk
utilization. It keeps only a bounded sample window in process memory, prints
median/p05/p95/p99 summaries, and does not create a telemetry log or database.
The existing UETS dashboard separately records its own telemetry in RocksDB.

Save the following as `telemetry.sh` in the deployment directory:

```bash
#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${1:-1}"
WINDOW_SAMPLES="${2:-300}"
DB_ROOT="${UETS_LIVE_DB:-$PWD/market_data/live_db}"
DISK_THRESHOLD_PERCENT="${DISK_THRESHOLD_PERCENT:-90}"
SESSION_ID="${UETS_SESSION_ID:-}"
CLOUD_SCRIPT="${CLOUD_SCRIPT:-$PWD/cloud.sh}"

exec python3 - "$INTERVAL_SECONDS" "$WINDOW_SAMPLES" "$DB_ROOT" \
  "$DISK_THRESHOLD_PERCENT" "$SESSION_ID" "$CLOUD_SCRIPT" <<'PY'
import math
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

interval = float(sys.argv[1])
window_size = int(sys.argv[2])
db_root = Path(sys.argv[3])
disk_threshold = float(sys.argv[4])
session_id = sys.argv[5]
cloud_script = Path(sys.argv[6])

if interval <= 0 or window_size < 1:
    raise SystemExit("interval must be positive and window must be at least 1")
if not 0 < disk_threshold <= 100:
    raise SystemExit("disk threshold must be in (0, 100]")

samples = deque(maxlen=window_size)
cloud_triggered = False

def cpu_counters():
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle

def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def uets_rss_mib():
    result = subprocess.run(
        ["ps", "-C", "uets", "-o", "rss="],
        capture_output=True, text=True, check=False
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"ps failed: {result.stderr.strip()}")
    return sum(int(line) for line in result.stdout.splitlines() if line.strip()) / 1024

previous_total, previous_idle = cpu_counters()
print("Monitoring in memory only; Ctrl+C stops it. Redirecting output will create a file.")

try:
    while True:
        time.sleep(interval)
        total, idle = cpu_counters()
        delta_total = total - previous_total
        delta_idle = idle - previous_idle
        previous_total, previous_idle = total, idle
        cpu_percent = (
            100.0 * (delta_total - delta_idle) / delta_total
            if delta_total > 0 else 0.0
        )

        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = int(value.strip().split()[0])
        total_mib = meminfo["MemTotal"] / 1024
        used_mib = (meminfo["MemTotal"] - meminfo["MemAvailable"]) / 1024
        ram_percent = 100.0 * used_mib / total_mib
        process_rss_mib = uets_rss_mib()

        disk_path = db_root if db_root.exists() else Path("/")
        disk = shutil.disk_usage(disk_path)
        disk_percent = 100.0 * (disk.total - disk.free) / disk.total
        sample = {
            "ram_mib": used_mib,
            "ram_percent": ram_percent,
            "cpu_percent": cpu_percent,
            "disk_percent": disk_percent,
        }
        samples.append(sample)

        print(
            f"RAM={used_mib:.1f}/{total_mib:.1f} MiB ({ram_percent:.1f}%) "
            f"UETS_RSS={process_rss_mib:.1f} MiB CPU={cpu_percent:.1f}% "
            f"disk={disk_percent:.1f}% window={len(samples)}/{window_size}",
            flush=True,
        )
        for name in ("ram_mib", "cpu_percent", "disk_percent"):
            values = [item[name] for item in samples]
            print(
                f"  {name}: median={quantile(values, 0.50):.2f} "
                f"p05={quantile(values, 0.05):.2f} "
                f"p95={quantile(values, 0.95):.2f} "
                f"p99={quantile(values, 0.99):.2f}",
                flush=True,
            )

        if disk_percent >= disk_threshold and not cloud_triggered:
            cloud_triggered = True
            if not session_id:
                marker = db_root / "current_session"
                if marker.is_file():
                    session_id = marker.read_text(encoding="utf-8").strip()
            if session_id and os.environ.get("AWS_ARCHIVE_URI"):
                print(
                    f"Disk is {disk_percent:.1f}% full; starting S3 checkpoint archive.",
                    flush=True,
                )
                subprocess.Popen([str(cloud_script), session_id])
            else:
                print(
                    "Disk threshold reached, but S3 archival is not configured; "
                    "no data was moved.",
                    flush=True,
                )
except KeyboardInterrupt:
    print("\nTelemetry monitor stopped.", flush=True)
PY
```

Run it without writing output to disk:

```bash
chmod +x telemetry.sh
chmod +x cloud.sh
./telemetry.sh                 # sample every second; retain at most 300 samples in RAM
./telemetry.sh 2 900           # sample every 2 seconds; retain at most 900 samples
```

Do not append `> telemetry.log` if the intent is to avoid writing telemetry
data to disk. The bounded window is only for the running monitor process and is
lost when it exits. Its quantiles describe host resource samples, not the
RocksDB telemetry table.

### Querying saved RocksDB telemetry

The current UETS inspector reads the session's telemetry store and prints the
record count and up to the latest five snapshots:

```bash
./bin/uets db "$UETS_SESSION_ID"
```

The current inspector does **not** calculate median/p05/p95/p99 across the
entire RocksDB table. Add a dedicated UETS export/summary command before
relying on a database-backed percentile report. Its intended interface should
be documented here once implemented, for example:

```text
./bin/uets telemetry summary --session <session-id> \
  --metrics total_ram_mb,total_cpu_percent,public_ping_rtt_ms,order_ack_rtt_ms \
  --quantiles 0.05,0.50,0.95,0.99
```

That is a **proposed interface, not a command available in the current
release**. Keep the existing one-second RocksDB snapshots if historical
quantiles are required; the no-storage monitor above cannot answer historical
queries after it exits.

## `cloud.sh`: S3 checkpoint archive

The package includes `cloud.sh`; it does not use a third-party RocksDB backup
program. When the disk threshold is met, it sends `q` to the tmux TUI and waits for UETS
to stop cleanly. It checkpoints telemetry RocksDB while closed, uploads a
`.tar.zst` to S3 using `GLACIER_IR`, downloads and checksums the object, and
opens the restored database. **Only after all checks pass**, it removes the
local telemetry RocksDB and restarts UETS in tmux with the same session ID.
Trade and portfolio databases are not removed, so OFMR account recovery remains
available. The TUI has a brief interruption during this archive/cleanup cycle.
If upload or verification fails, the local database is kept and UETS is
restarted.

Install AWS CLI v2 and configure AWS through the EC2 instance profile. Then
create the archive service environment:

```bash
sudo install -d -o root -g root -m 0755 /etc/uets
sudo tee /etc/uets/archive.env >/dev/null <<'EOF'
AWS_ARCHIVE_URI=s3://YOUR_BUCKET/cold-sst
AWS_REGION=ap-southeast-2
DISK_THRESHOLD_PERCENT=90
ARCHIVE_TMP_DIR=/var/tmp/uets-archive
UETS_HOME=/var/lib/uets
UETS_LIVE_DB=/var/lib/uets/live_db
UETS_BIN=/opt/uets/uets-linux-x86_64/bin/uets
UETS_TMUX_SESSION=uets
EOF
sudo chmod 0600 /etc/uets/archive.env
```

The IAM role needs `s3:PutObject` and `s3:GetObject` on the chosen prefix
(`HeadObject` is authorized by `s3:GetObject`). `cloud.sh` reads
`current_session` if no session ID is passed.
It exits without archiving when usage is below threshold, and fails explicitly
if stopping UETS, checkpointing, upload, download, checksum, or restore
verification fails.
It needs free temporary space for the compressed archive; configure
`ARCHIVE_TMP_DIR` accordingly.

The archive service must run as the same Linux user as the tmux session (`uets`
in the example); set its `User` and `Group` accordingly. Do not enable
`PrivateTmp`, since tmux uses a socket under `/tmp`. The included
[systemd service](./deploy/systemd/uets-cloud-archive.service)
and [timer](./deploy/systemd/uets-cloud-archive.timer) run that threshold
check daily at 00:00 UTC. Create `/etc/uets/archive.env` with the environment
above before enabling it:

```bash
sudo install -D -m 0644 deploy/systemd/uets-cloud-archive.service /etc/systemd/system/uets-cloud-archive.service
sudo install -D -m 0644 deploy/systemd/uets-cloud-archive.timer /etc/systemd/system/uets-cloud-archive.timer
sudo systemctl daemon-reload
sudo systemctl enable --now uets-cloud-archive.timer
```

Confirm `/etc/uets/archive.env` is populated before enabling the timer. Glacier
Instant Retrieval has storage and retrieval charges. Local **telemetry** is
deleted only after the S3 object is downloaded and restore-verified; portfolio
and trade databases remain local and are not covered by this archive.

### OOM restart harness

Run the Docker harness on a Linux host to confirm the watchdog detects an OOM
exit, passes `--reuse-rocksdb`, and recovers the prior session marker:

```bash
bash unit-tests/production_check/crash_recovery_test.sh
```

This is a deterministic supervisor/flag recovery test using a fixture; it is
not a multi-day AWS run and does not claim to validate exchange connectivity.
OFMR now persists its account state after fills and its latest market mark so
the restarted worker restores cash, open quantity, realized/unrealized PnL,
and active barriers. Missing recovery state alongside old portfolio/trade
records causes a fail-closed startup. Pre-recovery sessions lack open-position
details and cannot be safely migrated; do not reuse one if preserving its
account is required. Pressing `q` exits cleanly and retires the current-session
marker, so a later watchdog launch starts a fresh session; the old RocksDB
directories remain on disk. The live feed reseed path is separately covered by
Rust tests. Run a real binary crash/reseed smoke only in `live-test` mode,
never with real trading enabled.
