"""Portfolio supervisor - run many live-trade processes from one master table.

Every asset runs as its OWN `live-trade` OS process (own tokio runtime, own
RocksDB store, own auto-refreshing HTML report) - exactly like running it
manually - while this module plays air-traffic control:

* spawns one process per entry of the master config (JSON),
* parses each child's console output for live equity / RSI / trades / errors,
* prints a per-asset + PORTFOLIO-TOTAL table on a fixed cadence,
* raises an ALERT the moment a child exits unexpectedly (optional restart),
* regenerates `build/portfolio_dashboard.html` - a tabbed dashboard with a
  sidebar (overview + one tab per asset) linking each asset's full report,
* forwards Ctrl-C to every child for a graceful RocksDB flush.

Master config example (see `portfolio.example.json`):

    {
      "mode": "paper",
      "restart": true,
      "status_interval": 5.0,
      "assets": [
        {"ticker": "btcusdt", "exchange": "bitfinex", "feed": "bybit",   "capital": 5000},
        {"ticker": "ethusdt", "exchange": "bitfinex", "feed": "binance", "capital": 3000}
      ]
    }

Per-asset overrides: `exchange`, `feed`, `capital`, `config` (strategy JSON),
`db` (explicit store path). Everything else inherits from the top level.
"""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("portfolio")

REPO_ROOT = Path(__file__).resolve().parent  # module lives at the repo root
BUILD_DIR = REPO_ROOT / "build"
DATA_DIR = REPO_ROOT / "data"
DASHBOARD_PATH = BUILD_DIR / "portfolio_dashboard.html"
DASHBOARD_TEMPLATE = (REPO_ROOT / "templates" / "portfolio_dashboard.html").read_text(
    encoding="utf-8"
)

CREATE_NEW_PROCESS_GROUP = 0x200  # Windows-only process-group flag


class AssetHandle:
    """One supervised `live-trade` child process and its parsed live state."""

    def __init__(self, spec: dict[str, Any], defaults: dict[str, Any], bin_path: str):
        self.ticker = str(spec["ticker"]).lower()
        self.exchange = spec.get("exchange", defaults.get("exchange", "bitfinex"))
        self.feed = spec.get("feed", defaults.get("feed", self.exchange))
        self.mode = spec.get("mode", defaults.get("mode", "paper"))
        self.capital = float(spec.get("capital", defaults.get("capital", 10_000.0)))
        self.config = spec.get("config", str(BUILD_DIR / "live_trade_config.json"))
        self.db = spec.get("db", str(DATA_DIR / f"portfolio_{self.ticker}.rocksdb"))

        self.cmd = [
            bin_path,
            self.ticker,
            "--exchange", self.exchange,
            "--feed", self.feed,
            "--mode", self.mode,
            "--capital", str(self.capital),
            "--db", self.db,
            "--config", self.config,
        ]
        self.proc: Optional[subprocess.Popen] = None
        self.exit_code: Optional[int] = None
        self.restarts = 0
        # live state parsed from stdout
        self.equity: float = self.capital
        self.n_trades = 0
        self.total_r = 0.0
        self.mid: Optional[float] = None
        self.rsi: Optional[float] = None
        self.pwin: Optional[float] = None
        self.status = "starting"
        self.last_event = ""
        self.alerts: list[str] = []
        self.started_at = datetime.now()

    # ---- lifecycle ----------------------------------------------------------

    def spawn(self) -> None:
        kwargs: Dict[str, Any] = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(self.cmd, **kwargs)
        self.status = "running"
        self.started_at = datetime.now()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Parse the child's console stream into live state (reader thread)."""
        assert self.proc and self.proc.stdout
        try:
            for line in self.proc.stdout:
                self._parse_line(line.rstrip())
        except Exception as exc:  # noqa: BLE001 - reader must never crash the app
            logger.debug("[%s] reader stopped: %s", self.ticker, exc)

    def _parse_line(self, line: str) -> None:
        """Extract live state from the live-trade console protocol."""
        self.last_event = line[:160]
        if line.startswith("[bar] "):
            for tok in line.split():
                if tok.startswith("close="):
                    try:
                        self.mid = float(tok[6:])
                    except ValueError:
                        pass
                elif tok.startswith("rsi2="):
                    try:
                        self.rsi = float(tok[5:])
                    except ValueError:
                        pass
                elif tok.startswith("p_win="):
                    try:
                        self.pwin = float(tok[6:])
                    except ValueError:
                        pass
        elif line.startswith("[exit] "):
            # [exit] btcusdt reason=.. @ px r=+x.xx net=+x.xx equity=X (Ns)
            self.n_trades += 1
            for tok in line.split():
                if tok.startswith("r="):
                    try:
                        self.total_r += float(tok[2:])
                    except ValueError:
                        pass
                elif tok.startswith("equity="):
                    try:
                        self.equity = float(tok[7:])
                    except ValueError:
                        pass
        elif line.startswith("[seed] warmed"):
            self.last_event = "features seeded"
        elif "connect failed" in line or "stream error" in line:
            self._alert(line.strip())

    def _alert(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.alerts.append(f"{stamp} {msg}")
        del self.alerts[:-20]  # keep only the last 20

    def mark_down(self, exit_code: int) -> None:
        self.status = "DOWN"
        self.exit_code = exit_code
        msg = f"[{self.ticker}] PROCESS DOWN (exit code {exit_code})"
        logger.warning("*** ALERT *** %s", msg)
        self._alert(msg)

    def graceful_stop(self, grace_s: float = 12.0) -> None:
        """Ask the child to flush RocksDB and exit; kill only as last resort."""
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
        except (OSError, ProcessLookupError):
            return
        try:
            self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            logger.warning("[%s] did not exit in %.0fs - killing", self.ticker, grace_s)
            self.proc.kill()

class PortfolioSupervisor:
    """Spawn, watch, aggregate and report over a set of AssetHandles."""

    def __init__(
        self,
        assets: list[AssetHandle],
        *,
        restart: bool = False,
        status_interval: float = 5.0,
        dashboard: bool = True,
    ):
        self.assets = assets
        self.restart_enabled = restart
        self.status_interval = max(1.0, float(status_interval))
        self.dashboard_enabled = dashboard
        self.stop_requested = False

    # ---- supervision ----------------------------------------------------------

    def start_all(self) -> None:
        for a in self.assets:
            a.spawn()
            logger.info(
                "started %s (%s/%s, $%.0f) pid=%s db=%s",
                a.ticker, a.feed, a.exchange, a.capital,
                a.proc.pid, Path(a.db).name,
            )

    def monitor(self) -> None:
        while not self.stop_requested:
            for a in self.assets:
                if a.proc is None or a.status == "DOWN":
                    continue
                code = a.proc.poll()
                if code is not None:
                    a.mark_down(code)
                    if self.restart_enabled:
                        self._restart(a)
            self._print_table()
            if self.dashboard_enabled:
                self.write_dashboard()
            waited = 0.0
            while waited < self.status_interval and not self.stop_requested:
                time.sleep(0.25)
                waited += 0.25

    def _restart(self, a: AssetHandle) -> None:
        if self.stop_requested:
            return
        a.restarts += 1
        logger.info("[%s] auto-restart #%d in 3s (same config)", a.ticker, a.restarts)
        time.sleep(3.0)
        if self.stop_requested:
            return
        a.spawn()

    def stop_all(self) -> None:
        self.stop_requested = True
        logger.info("stopping %d asset(s)...", len(self.assets))
        threads = []
        for a in self.assets:
            t = threading.Thread(target=a.graceful_stop)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=15.0)

    # ---- aggregation ----------------------------------------------------------

    def totals(self) -> Dict[str, float]:
        cap = sum(a.capital for a in self.assets)
        eq = sum(a.equity for a in self.assets)
        return {
            "capital": cap,
            "equity": eq,
            "net": eq - cap,
            "trades": sum(a.n_trades for a in self.assets),
            "r": sum(a.total_r for a in self.assets),
            "up": sum(1 for a in self.assets if a.status == "running"),
        }

    def _print_table(self) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n=== PORTFOLIO @ {stamp} ===")
        print(
            f"  {'asset':<10}{'status':<10}{'eq($)':>11}{'net($)':>10}"
            f"{'tr':>4}{'totR':>8}{'mid':>11}{'rsi':>6}{'pWin':>6}  last event"
        )
        for a in self.assets:
            mid = f"{a.mid:.2f}" if a.mid else "-"
            rsi = f"{a.rsi:.1f}" if a.rsi is not None else "-"
            pw = f"{a.pwin:.2f}" if a.pwin is not None else "-"
            print(
                f"  {a.ticker:<10}{a.status:<10}{a.equity:>11.2f}"
                f"{a.equity - a.capital:>10.2f}{a.n_trades:>4}{a.total_r:>8.2f}"
                f"{mid:>11}{rsi:>6}{pw:>6}  {a.last_event}"
            )
        tt = self.totals()
        print(
            f"  {'TOTAL':<10}{'':<10}{tt['equity']:>11.2f}{tt['net']:>10.2f}"
            f"{tt['trades']:>4}{tt['r']:>8.2f}   "
            f"up {tt['up']}/{len(self.assets)} | capital ${tt['capital']:.0f}"
        )

    # ---- dashboard --------------------------------------------------------------

    def write_dashboard(self) -> None:
        data = {
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "totals": self.totals(),
            "assets": [
                {
                    "ticker": a.ticker,
                    "feed": a.feed,
                    "exchange": a.exchange,
                    "mode": a.mode,
                    "status": a.status,
                    "capital": a.capital,
                    "equity": a.equity,
                    "net": a.equity - a.capital,
                    "trades": a.n_trades,
                    "total_r": a.total_r,
                    "mid": a.mid,
                    "rsi": a.rsi,
                    "pwin": a.pwin,
                    "restarts": a.restarts,
                    "lastEvent": a.last_event,
                    "alerts": a.alerts[-5:],
                    "report": f"pnl_report_{a.ticker}.html",
                }
                for a in self.assets
            ],
        }
        payload = json.dumps(data, indent=1)
        html = DASHBOARD_TEMPLATE.replace("__PAYLOAD__", payload)
        tmp = DASHBOARD_PATH.with_suffix(".tmp")
        try:
            tmp.write_text(html, encoding="utf-8")
            tmp.replace(DASHBOARD_PATH)
        except OSError as exc:
            logger.debug("dashboard write failed: %s", exc)


def find_live_trade_bin(prefer_release: bool = False) -> Optional[str]:
    """Locate a compiled `live-trade` binary (release preferred on prod boxes)."""
    exe = ".exe" if sys.platform == "win32" else ""
    candidates = []
    if prefer_release:
        candidates.append(REPO_ROOT / "target" / "release" / f"live-trade{exe}")
    candidates += [
        REPO_ROOT / "target" / "debug" / f"live-trade{exe}",
        REPO_ROOT / "target" / "release" / f"live-trade{exe}",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

def run_portfolio(
    config_path: str,
    *,
    release: bool = False,
    restart: bool = False,
    status_interval: float = 5.0,
    no_dashboard: bool = False,
) -> int:
    """Entry point behind `unified_runner.py portfolio --config ...`."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"ERROR: portfolio config not found: {cfg_file}", file=sys.stderr)
        return 1
    with open(cfg_file, "r", encoding="utf-8-sig") as fh:
        # utf-8-sig tolerates a leading BOM (common when the table is edited
        # in Notepad / PowerShell), while reading plain UTF-8 unchanged.
        cfg = json.load(fh)

    assets_spec: list[dict[str, Any]] = cfg.get("assets") or []
    if not assets_spec:
        print("ERROR: portfolio config has no 'assets' list", file=sys.stderr)
        return 1

    bin_path = find_live_trade_bin(prefer_release=release)
    if bin_path is None:
        print(
            "ERROR: live-trade binary not found under target/. "
            "Build it first: cargo build -p live-trade (--release)",
            file=sys.stderr,
        )
        return 1

    defaults = {k: cfg[k] for k in ("exchange", "feed", "mode", "capital") if k in cfg}
    assets = [AssetHandle(spec, defaults, bin_path) for spec in assets_spec]

    sup = PortfolioSupervisor(
        assets,
        restart=restart or bool(cfg.get("restart", False)),
        status_interval=float(cfg.get("status_interval", status_interval)),
        dashboard=not no_dashboard,
    )

    state = {"stopped": False}

    def _shutdown(signum, _frame):
        if not state["stopped"]:
            state["stopped"] = True
            logger.info(
                "%s received - shutting down portfolio...", signal.Signals(signum).name
            )

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, OSError):
        pass

    sup.start_all()
    logger.info(
        "supervising %d asset(s) - Ctrl-C to stop all | dashboard: %s",
        len(assets), DASHBOARD_PATH.name,
    )
    try:
        sup.monitor()
    except KeyboardInterrupt:
        pass
    finally:
        sup.stop_all()
    logger.info("portfolio stopped cleanly")
    return 0


