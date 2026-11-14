# UETS VPS setup

The installer configures a supported Linux VPS (Ubuntu/Debian or Amazon Linux
2023): packages, the `uets` service account, persistent data directories,
systemd, tmux, and the UETS watchdog. It expects the Linux x86_64 bundle
published by this repository's tagged GitHub releases.

The first compatible release must be published from this updated workflow;
the existing v4.1.0 release does not contain the new installer bundle assets.
After this change is merged, push a new `v*` tag and wait for its release job
to finish before provisioning a VPS.

On a new VPS, run:

```bash
curl -fsSL https://raw.githubusercontent.com/hmzk1422/osadp-core/main/deploy/install_vps.sh -o install_vps.sh
sudo bash install_vps.sh \
  --strategy ofmr@btcusdt \
  --archive-uri s3://YOUR_BUCKET/cold-sst \
  --region ap-southeast-2
```

The script downloads the latest release bundle and SHA-256 file, verifies the
download, and installs it. No separate binary download or repository checkout
is needed. Use an EC2 IAM instance profile; do not put AWS keys in the command.
To update manually:

```bash
sudo /opt/uets/uets-linux-x86_64/deploy/install_vps.sh --update
```

The installer also checks for updates daily and applies new tagged releases.
Release publishing is triggered by pushing a `v*` tag after the workflow
passes. Do not reuse or move published release tags.

The installer starts UETS in `live-test` mode by default. Review
`/opt/uets/uets-linux-x86_64/uets/config/execution.yaml` before selecting
`testnet` or `live-trade`; real orders require API credentials and the explicit
`allow_live_trading: true` guard.

## Operations

```bash
sudo -u uets tmux attach -t uets       # Attach; Ctrl+B then D detaches
sudo systemctl status uets-tmux
sudo journalctl -u uets-tmux -f
sudo systemctl restart uets-tmux
```

Unexpected UETS exits are retried by `restart_uets.sh` with
`--reuse-rocksdb`, preserving the current session on disk. If the watchdog or
tmux supervisor itself fails (including an OOM kill), systemd restarts it.
Intentional `q` exits are not restarted. Retries are bounded to avoid an
uncontrolled crash loop. Runtime databases and watchdog logs live under
`/var/lib/uets` and are separate from the application bundle.

When configured, the `uets-cloud-archive.timer` checks the disk threshold daily.
Archives use RocksDB checkpoints and are removed locally only after upload,
download, checksum, restore, and verification succeed. The IAM role needs
`s3:PutObject` and `s3:GetObject` on the selected prefix. Set
`DISK_THRESHOLD_PERCENT` and `ARCHIVE_TMP_DIR` in
`/etc/uets/archive.env` to change the defaults; then run
`sudo systemctl restart uets-cloud-archive.timer`.
