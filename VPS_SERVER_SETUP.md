# UETS VPS setup and operations

This guide covers a fresh Ubuntu/Debian x86_64 VPS using the published UETS
release. The installer configures the `uets` service account, runtime
directories, required packages, systemd, tmux, crash recovery, S3 archival
(when requested), and automatic release updates. You do not need to clone the
repository or manually download/extract the UETS binary.

> **Release format:** the installer downloads
> `uets-linux-x86_64-amazonlinux2023.1.zip`. The current v4.1.0 release already
> contains this ZIP, with `uets-linux-x86_64.tar.gz` and `install_vps.sh`
> inside. It has no checksum sidecar, so the installer warns and validates ZIP
> integrity; the updated release workflow adds a SHA-256 sidecar to new releases.

## 1. Create and connect to the VPS

Create an Ubuntu or Debian x86_64 instance with a public IP. For S3 archival,
attach an EC2 IAM instance profile that allows `s3:PutObject` and
`s3:GetObject` on the archive bucket/prefix. Do not put AWS access keys in
commands or configuration files.

Connect from your computer:

```bash
ssh -i /path/to/your-key.pem ubuntu@YOUR_VPS_IP
```

Use the SSH username provided by your image if it is not `ubuntu`.

## 2. Install UETS

On the VPS, install the small prerequisites needed to download the installer,
then fetch and run it:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
curl -fsSL https://raw.githubusercontent.com/hmzk1422/osadp-core/main/deploy/install_vps.sh -o install_vps.sh
sudo bash install_vps.sh \
  --strategy ofmr@btcusdt \
  --archive-uri s3://YOUR_BUCKET/cold-sst \
  --region ap-southeast-2
```

Replace `YOUR_BUCKET/cold-sst` with your S3 bucket and prefix. To install
without S3 archival, omit the `--archive-uri` and `--region` options:

```bash
sudo bash install_vps.sh --strategy ofmr@btcusdt
```

The installer creates the `uets` Linux account itself; you do not need to run
`useradd` manually. It also installs the remaining packages and services,
downloads the latest release ZIP, extracts the `.tar.gz` bundle inside it, and
starts UETS. It verifies the SHA-256 checksum when the release provides a
sidecar and validates the ZIP and tar archives. UETS starts automatically at
boot. Runtime data is stored separately in `/var/lib/uets`.

UETS starts in `live-test` mode by default (live public market data, simulated
orders). Check the configured mode before enabling `testnet` or `live-trade`.
Real orders require credentials and `allow_live_trading: true`.

## Configure execution mode

Edit the execution configuration and restart UETS to apply changes:

```bash
sudo nano /opt/uets/uets-linux-x86_64/uets/config/execution.yaml
sudo systemctl restart uets-tmux
```

The installer preserves local `execution.yaml` and `ofmr.yaml` files when
installing an update.

## 3. Check status and logs

```bash
sudo systemctl status uets-tmux --no-pager
sudo journalctl -u uets-tmux -f
```

Press `Ctrl+C` to stop following the logs; this does not stop UETS.

## 4. Attach to or detach from tmux

Attach to the running UETS terminal:

```bash
sudo -u uets tmux attach -t uets
```

Detach and leave UETS running by pressing `Ctrl+B`, releasing both keys, then
pressing `D`.

Press `q` inside UETS to intentionally quit it. An intentional quit is not
automatically restarted. To start it again:

```bash
sudo systemctl start uets-tmux
```

## 5. Stop, start, or restart UETS

```bash
sudo systemctl stop uets-tmux
sudo systemctl start uets-tmux
sudo systemctl restart uets-tmux
```

Unexpected UETS failures are retried by the watchdog with `--reuse-rocksdb` to
recover the existing session. If the watchdog or tmux supervisor itself fails,
systemd restarts it. Watchdog logs are under `/var/lib/uets/logs/uets/`.

## 6. Update UETS

The installed system checks for new **published tagged releases** daily and
installs them automatically. Updates preserve `/var/lib/uets` and local
`execution.yaml`/`ofmr.yaml` configuration.

To check for and install an update immediately:

```bash
sudo /opt/uets/uets-linux-x86_64/deploy/install_vps.sh --update
```

Check the installed version and update timer:

```bash
cat /opt/uets/uets-linux-x86_64/VERSION
sudo systemctl status uets-update.timer --no-pager
sudo journalctl -u uets-update.service -n 100 --no-pager
```

Updates are published when a new `v*` tag passes the release workflow; they
are not installed directly from ordinary commits.

## 7. Configure S3 archival after installation

If S3 was omitted during installation, add it later by creating the archive
environment file:

```bash
sudo install -d -o root -g root -m 0755 /etc/uets
sudo tee /etc/uets/archive.env >/dev/null <<'EOF'
AWS_ARCHIVE_URI=s3://YOUR_BUCKET/cold-sst
AWS_REGION=ap-southeast-2
DISK_THRESHOLD_PERCENT=90
ARCHIVE_TMP_DIR=/var/tmp/uets-archive-tmp
UETS_HOME=/var/lib/uets
UETS_LIVE_DB=/var/lib/uets/live_db
UETS_BIN=/opt/uets/uets-linux-x86_64/bin/uets
UETS_TMUX_SESSION=uets
EOF
sudo chmod 0600 /etc/uets/archive.env
sudo systemctl enable --now uets-cloud-archive.timer
```

The timer checks disk usage daily. A sealed local day is removed only after its
S3 upload, download, checksum, restore, and verification succeed. The IAM role
needs `s3:PutObject` and `s3:GetObject` on the selected prefix.

Check or run the archive job manually:

```bash
sudo systemctl status uets-cloud-archive.timer --no-pager
sudo journalctl -u uets-cloud-archive.service -n 100 --no-pager
sudo systemctl start uets-cloud-archive.service
```
