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

Master config example (see `config/portfolio.example.json`):

    {
      "mode": "paper",
      "restart": true,
      "status_interval": 5.0,
      "_strategy_defaults": { ...engine keys... },
      "assets": [
        {"ticker": "btcusdt", "exchange": "bitfinex", "feed": "bybit",   "capital": 5000},
        {"ticker": "ethusdt", "feed": "binance", "capital": 3000,
         "strategy": {"rr_ratio": 5}}
      ]
    }

Every child receives `--config config/portfolio.json` DIRECTLY — the Rust
engine resolves its own ticker (`_strategy_defaults` + per-asset `strategy`
overrides, deep-merged in-memory). No per-ticker config file is generated.
Per-asset overrides: `exchange`, `feed`, `capital`, `db` (explicit store
path). Everything else inherits from the top level.
"""

from __future__ import annotations

import copy
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("portfolio")

REPO_ROOT = Path(__file__).resolve().parent  # module lives at the repo root
BUILD_DIR = REPO_ROOT / "build"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "portfolio.json"
DASHBOARD_PATH = BUILD_DIR / "portfolio_dashboard.html"
DASHBOARD_TEMPLATE = (REPO_ROOT / "templates" / "portfolio_dashboard.html").read_text(
    encoding="utf-8"
)

CREATE_NEW_PROCESS_GROUP = 0x200  # Windows-only process-group flag

# Per-asset network-metric history kept for the HTML dashboard's LAT/SK/STAL
# charts. One sample per monitor tick (status_interval, default 5s), so 1440
# points spans two hours — downsampling to this window keeps the payload lean.
NET_HIST_MAX = 1440


# ---- unified config parsing (JSON -> JSON; nothing is ever written) --------
def _strip_comments(obj: Any) -> Any:
    """Recursively remove `_comment` documentation keys."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_portfolio_config(path: str | Path) -> dict:
    """Parse `config/portfolio.json` into a plain dict (no files written).

    Returns `{"mode", "exchange", "feed", "capital", "enable_regime_detection", "restart",
    "status_interval", "strategy", "assets": [{"id", "ticker", "exchange",
    "feed", "mode", "capital", "db", "enable_regime_detection", "strategy_merged"}, ...]}`.

    Each asset carries a unique `id` (defaults to its ticker) so the SAME
    ticker can appear multiple times with different configs — e.g. a gate-off
    recorder twin (id "eth-rec") and a gate-on execution twin (id "eth-exec").
    `enable_regime_detection` (strict true|false) is the per-asset regime
    meta-labeller gate and is inherited from the top level when omitted.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"portfolio config not found: {p}")
    raw = _strip_comments(json.loads(p.read_text(encoding="utf-8-sig")))

    defaults = raw.get("_strategy_defaults", {}) or {}
    pc_defaults = {
        "exchange": raw.get("exchange", "bitfinex"),
        "feed": raw.get("feed", "bitfinex"),
        "mode": raw.get("mode", "paper"),
        "capital": float(raw.get("capital", 10_000.0)),
        # Regime meta-labeller gate: strict true|false. Read from the
        # `_regime_gate` block (where it is documented in portfolio.json);
        # a top-level `enable_regime_detection` is also honoured as fallback.
        "enable_regime_detection": bool(
            raw.get("_regime_gate", {}).get(
                "enable_regime_detection", raw.get("enable_regime_detection", False)
            )
        ),
    }

    assets_raw = raw.get("assets") or []
    if not assets_raw:
        raise ValueError("portfolio config has no 'assets' list")

    seen: set[str] = set()
    assets = []
    for spec in assets_raw:
        ticker = str(spec.get("ticker", "")).lower()
        if not ticker:
            raise ValueError("an asset entry is missing 'ticker'")
        asset_id = str(spec.get("id") or ticker).strip().lower() or ticker
        if asset_id in seen:
            raise ValueError(f"duplicate asset id in assets: {asset_id}")
        seen.add(asset_id)
        overrides = spec.get("strategy") or {}
        assets.append({
            "id": asset_id,
            "ticker": ticker,
            "exchange": spec.get("exchange", pc_defaults["exchange"]),
            "feed": spec.get("feed", pc_defaults["feed"]),
            "mode": spec.get("mode", pc_defaults["mode"]),
            "capital": float(spec.get("capital", pc_defaults["capital"])),
            "db": spec.get("db"),
            "enable_regime_detection": bool(
                spec.get("enable_regime_detection", pc_defaults["enable_regime_detection"])
            ),
            "strategy_merged": deep_merge(defaults, overrides),
        })

    return {
        "mode": pc_defaults["mode"],
        "exchange": pc_defaults["exchange"],
        "feed": pc_defaults["feed"],
        "capital": pc_defaults["capital"],
        "enable_regime_detection": pc_defaults["enable_regime_detection"],
        "restart": bool(raw.get("restart", True)),
        "status_interval": float(raw.get("status_interval", 5.0)),
        "strategy": defaults,
        "assets": assets,
    }


class AssetHandle:
    """One supervised `live-trade` child process and its parsed live state."""

    def __init__(self, spec: dict[str, Any], defaults: dict[str, Any], bin_path: str):
        self.ticker = str(spec["ticker"]).lower()
        # Unique per-asset id (defaults to the ticker). Allows the same ticker
        # to run multiple times with different configs (twins), each with its
        # own RocksDB store, control file and HTML report.
        self.asset_id = str(spec.get("id") or self.ticker).strip().lower() or self.ticker
        self.exchange = spec.get("exchange", defaults.get("exchange", "bitfinex"))
        self.feed = spec.get("feed", defaults.get("feed", self.exchange))
        self.mode = spec.get("mode", defaults.get("mode", "paper"))
        self.capital = float(spec.get("capital", defaults.get("capital", 10_000.0)))
        self.config = spec.get("config") or str(DEFAULT_CONFIG_PATH)
        self.db = spec.get("db") or str(DATA_DIR / f"portfolio_{self.asset_id}.rocksdb")
        self.control_path = spec.get("control", str(BUILD_DIR / f"control_{self.asset_id}.json"))
        # Regime meta-labeller gate (strict true|false) forwarded to the child as
        # `--meta-gate on|off`. false = executor trades unfiltered (twin A/B).
        self.enable_regime_detection = bool(
            spec.get("enable_regime_detection", defaults.get("enable_regime_detection", False))
        )
        # Effective strategy parameters (defaults + this asset's overrides),
        # used by the dashboard to detect config changes on live reload.
        self.strategy_merged: Optional[dict] = spec.get("strategy_merged")

        self.cmd = [
            bin_path,
            self.ticker,
            "--exchange", self.exchange,
            "--feed", self.feed,
            "--mode", self.mode,
            "--capital", str(self.capital),
            "--db", self.db,
            "--config", self.config,
            "--control", self.control_path,
            "--meta-gate", "on" if self.enable_regime_detection else "off",
            "--report-tag", self.asset_id,
            "--fresh-equity",
        ]
        self.proc: Optional[subprocess.Popen] = None
        self.exit_code: Optional[int] = None
        self.restarts = 0
        # live state parsed from stdout. ALL fields are written by the _pump()
        # reader thread and read concurrently by the VIM dashboard thread —
        # every read/write goes through self._lock to prevent torn reads.
        self._lock = threading.Lock()
        self.equity: float = self.capital
        self.n_trades = 0
        self.n_wins = 0
        self.n_losses = 0
        self.total_r = 0.0
        self.mid: Optional[float] = None
        self.rsi: Optional[float] = None
        self.pwin: Optional[float] = None
        self.status = "starting"
        self.paused: bool = False
        self.last_event = ""
        self.alerts: list[str] = []
        # Ring buffer of economically meaningful child lines ([exit]/[bar]/
        # alerts) for the trades view of the terminal dashboard.
        self.event_lines: deque = deque(maxlen=400)
        # FULL child log (every stdout line, timestamped) — the trades view of
        # the VIM dashboard shows exactly what the algo would print with no
        # dashboard attached ([bar]/[net]/[order]/[entry]/[exit]/[seed]/...).
        self.log_lines: deque = deque(maxlen=2000)
        # Network-health state parsed from the child's periodic `[net]` line
        # (staleness / latency / clock skew / outage episodes). See
        # execution.rs: NET_OUTAGE_STALENESS_MS / NET_PRINT_SECS for definitions.
        self.stal_ms: Optional[float] = None
        self.lat_ms: Optional[float] = None
        self.skew_ms: Optional[float] = None
        self.rtt_ms: Optional[float] = None  # ws ping→pong RTT (monotonic, skew-free)
        self.ord_rtt_ms: Optional[float] = None  # order send→ack RTT (monotonic)
        self.outages: int = 0
        self.max_stal_ms: Optional[float] = None
        self.outage_active: bool = False
        self.n_voided: int = 0  # trades voided by regime gate
        # R-multiple of every closed trade, in close order (parsed from the
        # `r=` token of `[exit]` lines). This is the input series for the
        # TRADE-level Sortino/Sharpe — not UI samples.
        self.trade_rs: deque = deque(maxlen=20_000)
        # Trades closed since the supervisor last drained (see _record_metrics).
        self._pending_trade_rs: list[tuple[int, float]] = []
        self.last_bar_time: float = 0.0
        self.last_net_time: float = 0.0  # epoch of the last [net] line seen
        # Per-second live-market telemetry parsed from the child's `[tick]`
        # line (execution.rs TICK_PRINT_SECS) — the MICRO channel: mid / gate
        # / armed and the open position's mark-to-market PnL refresh here
        # ~1s, independent of the ~5-minute `[bar]` macro cadence.
        self.last_tick_time: float = 0.0  # epoch of the last [tick] line seen
        self.gate: str = ""   # which entry gate rejected the newest bar eval ("")
        # Phase-1 `armed` = STATIC threshold observation (p_win >=
        # confidence_threshold at tick time — exactly the [bar] [armed] tag).
        # A debounced signal_armed state machine lands in Phase 2 and replaces
        # this interpretation — NOT the field set — to keep the UI contract
        # stable across both phases.
        self.armed: bool = False
        # Open-position snapshot for mark-to-market PnL (all None when flat).
        self.open_side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.stop: Optional[float] = None
        self.target: Optional[float] = None
        self.qty: Optional[float] = None
        self.notional: Optional[float] = None    # qty * entry (position size in USD)
        self.barrier_pct: Optional[float] = None
        self.rr_mark: Optional[float] = None     # mark-to-market R of open position
        self.usd_mark: Optional[float] = None    # mark-to-market USD of open position
        self.tick_stal_ms: Optional[float] = None  # feed staleness at last [tick]
        # monotonic() epoch of the last child output line of ANY kind — the
        # honest liveness signal. A child that never connects (no [bar]) still
        # flips to "stalled" once this goes quiet, instead of lying "running".
        self.last_output_time: float = time.monotonic()
        self.started_at = datetime.now()  # wall-clock epoch when the asset was spawned
        self.stop_requested: bool = False  # True once we've asked the child to stop
        self.stopped_by_user: bool = False  # user asked to stop → monitor must NOT auto-restart it

    # ---- lifecycle ----------------------------------------------------------

    def spawn(self) -> None:
        kwargs: dict[str, Any] = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
        # Drop any stale control command so a restart never re-triggers it.
        try:
            Path(self.control_path).unlink()
        except OSError:
            pass
        self.proc = subprocess.Popen(self.cmd, **kwargs)
        self.stopped_by_user = False
        self.status = "running"
        self.paused = False
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
        """Extract live state from the live-trade console protocol.

        Writes go through self._lock so the VIM dashboard thread always sees
        a coherent snapshot (never a torn read where e.g. mid is from bar N
        but rsi is still from bar N-1).
        """
        stripped = line.strip()

        with self._lock:
            if stripped:
                self.log_lines.append(
                    f"{datetime.now():%H:%M:%S} {stripped}"
                )
                self.last_output_time = time.monotonic()

            if not (line.startswith("[net] ") or line.startswith("[tick] ")):
                # [net] lines arrive every few seconds — never let them drown the
                # last economically meaningful event in the dashboard/console.
                # Prefix with HH:MM:SS so the dashboard can show staleness —
                # without a timestamp you can't tell if an event was 2s or 2h ago.
                self.last_event = f"{datetime.now():%H:%M:%S} {line[:160]}"
            # NOTE: [restore] lines are intentionally IGNORED here.
            # The engine restores cumulative equity/n_trades/total_r from RocksDB
            # for its own internal consistency (avoiding duplicate positions etc),
            # but the live session must start FRESH every time. We do NOT restore
            # any stats — equity starts at capital, n_trades/total_r at zero.
            # The DB keeps its cumulative data for history; the live display shows
            # only THIS session's trades.
            if line.startswith("[bar] "):
                self.last_bar_time = time.monotonic()
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
                # Also record bar lines in the event log (for trades view).
                self.event_lines.append(f"{datetime.now():%H:%M:%S} {line.strip()}")
            elif line.startswith("[tick] "):
                # [tick] btcusdt 2026-.. mid=61820.55 rsi2=12.3 p_win=0.62
                #   armed=1 gate=confidence side=long qty=.. entry=.. stop=..
                #   target=.. bp=.. rr=+0.01 usd=+0.00 stal=12ms  (or "flat")
                # Per-second live-market channel (execution.rs TICK_PRINT_SECS).
                # This is what makes the dashboard's MID / gate / armed /
                # open-position mark-to-market PnL refresh ~1s instead of
                # waiting on a ~5-minute [bar]. [tick] lines are NOT mirrored
                # into event_lines/log_lines and do not touch last_event —
                # otherwise they'd outrank every meaningful [exit]/[entry].
                toks: dict[str, str] = {}
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        toks[k] = v

                def _fv(key: str) -> Optional[float]:
                    v = toks.get(key)
                    if v is None:
                        return None
                    try:
                        return float(v.rstrip("ms"))
                    except ValueError:
                        return None

                if toks.get("mid") is not None:
                    try:
                        self.mid = float(toks["mid"])
                    except ValueError:
                        pass
                self.gate = toks.get("gate", self.gate)
                self.armed = toks.get("armed") == "1"
                self.tick_stal_ms = _fv("stal")
                if toks.get("rsi2") is not None:
                    try:
                        self.rsi = float(toks["rsi2"])
                    except ValueError:
                        pass
                if toks.get("p_win") is not None:
                    try:
                        self.pwin = float(toks["p_win"])
                    except ValueError:
                        pass
                if toks.get("side") in ("long", "short"):
                    self.open_side = toks["side"]
                    self.qty = _fv("qty")
                    self.notional = _fv("notional")
                    self.entry_price = _fv("entry")
                    self.stop = _fv("stop")
                    self.target = _fv("target")
                    self.barrier_pct = _fv("bp")
                    self.rr_mark = _fv("rr")
                    self.usd_mark = _fv("usd")
                else:
                    # Flat (or malformed) — the position block is all-or-nothing,
                    # so clear every open-position field atomically.
                    self.open_side = None
                    self.qty = self.entry_price = self.stop = self.target = None
                    self.notional = self.barrier_pct = None
                    self.rr_mark = self.usd_mark = None
                self.last_tick_time = time.time()
            elif line.startswith("[exit] "):
                # [exit] btcusdt reason=.. @ px r=+x.xx net=+x.xx equity=X (Ns)
                self.n_trades += 1
                r_value: Optional[float] = None
                for tok in line.split():
                    if tok.startswith("r="):
                        try:
                            r_value = float(tok[2:])
                            self.total_r += r_value
                        except ValueError:
                            pass
                    elif tok.startswith("equity="):
                        try:
                            self.equity = float(tok[7:])
                        except ValueError:
                            pass
                if r_value is not None:
                    if r_value >= 0.0:
                        self.n_wins += 1
                    else:
                        self.n_losses += 1
                    # Record the per-trade R for trade-level Sortino/Sharpe and
                    # the closed-trade MDD-R curve (drained by the supervisor).
                    self.trade_rs.append(r_value)
                    self._pending_trade_rs.append((time.monotonic_ns(), r_value))
                self.event_lines.append(f"{datetime.now():%H:%M:%S} {line.strip()}")
            elif line.startswith("[entry] "):
                self.event_lines.append(f"{datetime.now():%H:%M:%S} {line.strip()}")
            elif line.startswith("[paused] "):
                self.paused = True
                if self.status != "DOWN":
                    self.status = "paused"
                self.event_lines.append(f"{datetime.now():%H:%M:%S} {line.strip()}")
            elif line.startswith("[resumed] "):
                self.paused = False
                if self.status != "DOWN":
                    self.status = "running"
                self.event_lines.append(f"{datetime.now():%H:%M:%S} {line.strip()}")
            elif line.startswith("[flatten] "):
                self.event_lines.append(f"{datetime.now():%H:%M:%S} {line.strip()}")
            elif line.startswith("[net] "):
                # [net] btcusdt stal=12ms lat=0.45ms sk=+2ms outages=1
                #       max_stal=1500ms [OUTAGE]
                # Emitted by the Rust loop every NET_PRINT_SECS; carries the same
                # staleness / latency / clock-skew numbers latency_diag.py samples,
                # plus the running count of distinct network-outage episodes.
                toks: dict[str, str] = {}
                for tok in line.split()[2:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        toks[k] = v

                def _ms(key: str) -> Optional[float]:
                    v = toks.get(key)
                    if v is None:
                        return None
                    try:
                        return float(v.rstrip("ms"))
                    except ValueError:
                        return None

                self.stal_ms = _ms("stal")
                self.lat_ms = _ms("lat")
                self.skew_ms = _ms("sk")
                self.rtt_ms = _ms("rtt")
                self.ord_rtt_ms = _ms("ord")
                self.max_stal_ms = _ms("max_stal")
                try:
                    n = int(toks.get("outages", ""))
                except ValueError:
                    n = self.outages
                try:
                    self.n_voided = int(toks.get("voided", "0"))
                except ValueError:
                    pass
                if n > self.outages:
                    # A NEW outage episode started since the last [net] line.
                    self._alert(
                        f"network outage #{n} - feed staleness exceeded 1000ms"
                    )
                self.outages = n
                self.outage_active = "[OUTAGE]" in line
                self.last_net_time = time.time()
                if self.outage_active:
                    self.status = "outage"
                elif self.status in ("outage", "stalled"):
                    # First healthy [net] telemetry after an outage/stall
                    # episode — the feed is back. [net] is the authority on
                    # feed health, so recovery lives HERE, not on a generic
                    # else-branch that would swallow [seed]/error lines.
                    self.status = "running"
            elif line.startswith("[seed] warmed"):
                self.last_event = "features seeded"
            elif "connect failed" in line or "stream error" in line:
                self._alert(line.strip())

    def snapshot(self) -> dict:
        """Return a coherent point-in-time copy of all live fields.

        The VIM dashboard thread calls this every ~50ms to render the table.
        Without the lock, concurrent writes from _pump() could produce a torn
        read (e.g. mid from bar N but rsi still from bar N-1).
        """
        with self._lock:
            return {
                "asset_id": self.asset_id,
                "ticker": self.ticker,
                "exchange": self.exchange,
                "feed": self.feed,
                "mode": self.mode,
                "status": self.status,
                "paused": self.paused,
                "equity": self.equity,
                "capital": self.capital,
                "n_trades": self.n_trades,
                "n_wins": self.n_wins,
                "n_losses": self.n_losses,
                "total_r": self.total_r,
                "mid": self.mid,
                "rsi": self.rsi,
                "pwin": self.pwin,
                "last_event": self.last_event,
                "log_lines": list(self.log_lines),
                "stal_ms": self.stal_ms,
                "lat_ms": self.lat_ms,
                "skew_ms": self.skew_ms,
                "rtt_ms": self.rtt_ms,
                "ord_rtt_ms": self.ord_rtt_ms,
                "outages": self.outages,
                "max_stal_ms": self.max_stal_ms,
                "outage_active": self.outage_active,
                "n_voided": self.n_voided,
                "last_net_time": self.last_net_time,
                "armed": self.armed,
                "gate": self.gate,
                "open_side": self.open_side,
                "entry_price": self.entry_price,
                "stop": self.stop,
                "target": self.target,
                "qty": self.qty,
                "notional": self.notional,
                "barrier_pct": self.barrier_pct,
                "rr_mark": self.rr_mark,
                "usd_mark": self.usd_mark,
                "tick_stal_ms": self.tick_stal_ms,
                "last_tick_time": self.last_tick_time,
                "last_bar_time": self.last_bar_time,
                "last_output_time": self.last_output_time,
            }

    def _alert(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.alerts.append(f"{stamp} {msg}")
        del self.alerts[:-20]  # keep only the last 20

    def mark_down(self, exit_code: int) -> None:
        self.exit_code = exit_code
        if self.stopped_by_user:
            # User asked to stop (:stop / :restart) — dim, no alert, no auto-restart.
            self.status = "STOPPED"
            logger.info("[%s] STOPPED by user (exit code %d)", self.ticker, exit_code)
        else:
            # Crash or signal — red alert + auto-restart (if enabled).
            self.status = "DOWN"
            msg = f"[{self.ticker}] PROCESS DOWN (exit code {exit_code})"
            logger.warning("*** ALERT *** %s", msg)
            self._alert(msg)

    def _flatten_and_wait(self) -> None:
        """Send flatten command and wait for the close to complete.

        The child polls the control file every 100ms and flattens at mid,
        printing ``[flatten] ... closed`` when done. We wait up to 1.2s
        for that to happen so no open trade is ever abandoned.
        """
        try:
            Path(self.control_path).write_text(
                json.dumps({"cmd": "flatten"}), encoding="utf-8"
            )
            logger.info("[control] %s -> flatten", self.ticker)
        except OSError as exc:
            logger.warning("[control] write failed for %s: %s", self.ticker, exc)
            return
        deadline = time.time() + 1.2
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                return
            time.sleep(0.1)

    def graceful_stop(self, grace_s: float = 12.0) -> None:
        """Ask the child to flush RocksDB and exit; kill only as last resort.

        Flattens any open position FIRST so no trade is ever abandoned
        (crucial for live). Then sends SIGINT / CTRL_BREAK_EVENT.
        """
        if not self.proc or self.proc.poll() is not None:
            return
        self._flatten_and_wait()
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
        quiet: bool = False,
    ):
        self.assets = assets
        self.restart_enabled = restart
        self.status_interval = max(1.0, float(status_interval))
        self.dashboard_enabled = dashboard
        # When a full-screen TUI owns the terminal the supervisor must not
        # print its plain status table (it would corrupt the curses screen).
        self.quiet = quiet
        self.stop_requested = False
        # MDD tracking: cumulative R (and equity) samples so the dashboards can
        # show max drawdown in RR terms (resetting peak -> trough of the curve).
        self.r_hist: deque = deque(maxlen=20_000)
        self.eq_hist: deque = deque(maxlen=20_000)
        # Trade-level portfolio R series (every closed trade across all assets,
        # in close order). Input for trade-level Sortino/Sharpe. Drained from
        # each AssetHandle._pending_trade_rs in _record_metrics.
        self.trade_rs: deque = deque(maxlen=100_000)
        self.trade_rs_by_asset: Dict[str, deque] = {
            a.asset_id: deque(maxlen=20_000)
            for a in self.assets
        }

        self._portfolio_mdd_r = 0.0
        self._asset_mdd_r: Dict[str, float] = {
            a.asset_id: 0.0
            for a in self.assets
        }

        self._portfolio_trade_r_cumulative = 0.0
        self._asset_trade_r_cumulative: Dict[str, float] = {
            a.asset_id: 0.0
            for a in self.assets
        }
        # Last seen n_trades per asset id — a decrease means the child was
        # restarted (counters reset to 0), so the cumulative-R curve must be
        # SPLIT there (None marker) instead of recording a phantom drawdown.
        self._prev_trade_counts: Dict[str, int] = {}
        # Clock-offset sanity guard: |sk| samples above this are clock-step
        # artifacts (unsynced wall clock), not feed behavior. Counted, not
        # silently dropped, so the dashboard can show what was filtered.
        self.SK_GUARD_MS = 60_000.0
        self.sk_discarded_by_asset: Dict[str, int] = {}
        # Keyed by per-asset `id` (not ticker) so duplicate-ticker twins each get
        # their own curve.
        self.r_hist_by_asset: Dict[str, deque] = {
            a.asset_id: deque(maxlen=20_000) for a in self.assets
        }
        # Per-asset network-metric history (lat / stal / clock-skew, one sample
        # per monitor tick) so the HTML dashboard can chart LAT/SK/STAL over
        # the session and show their avg/max/min.
        self.net_hist_by_asset: Dict[str, Dict[str, deque]] = {
            a.asset_id: {
                "lat": deque(maxlen=NET_HIST_MAX),
                "stal": deque(maxlen=NET_HIST_MAX),
                "sk": deque(maxlen=NET_HIST_MAX),
            }
            for a in self.assets
        }
        self._control_server = None

    # ---- external control (flatten / pause / resume) ------------------------------

    def _write_control(self, a: AssetHandle, cmd: str) -> None:
        try:
            Path(a.control_path).write_text(json.dumps({"cmd": cmd}), encoding="utf-8")
            logger.info("[control] %s -> %s", a.ticker, cmd)
        except OSError as exc:
            logger.warning("[control] write failed for %s: %s", a.ticker, exc)

    def send_control(self, ticker: str, cmd: str) -> None:
        """Drop a control command for one asset or all (flatten / pause / resume)."""
        if cmd not in ("flatten", "pause", "resume"):
            logger.warning("[control] unknown command: %s", cmd)
            return
        if ticker.strip().lower() == "all":
            for a in self.assets:
                self._write_control(a, cmd)
        else:
            target = ticker.strip().lower()
            for a in self.assets:
                if a.asset_id == target or a.ticker == target:
                    self._write_control(a, cmd)
                    return
            logger.warning("[control] unknown asset: %s", ticker)

    # ---- MDD (in RR terms) ------------------------------------------------------

    @staticmethod
    def _mdd_from_series(values) -> float:
        """Max peak-to-trough drawdown of a cumulative series (positive value).

        A None entry is a SESSION BREAK (child restart / counter reset): the
        curve is split there so the artificial collapse to zero is never
        counted as a real drawdown.
        """
        peak: Optional[float] = None
        mdd = 0.0
        for v in values:
            if v is None:  # restart boundary — start a fresh segment
                peak = None
                continue
            if peak is None or v > peak:
                peak = v
            dd = (peak - v) if peak is not None else 0.0
            if dd > mdd:
                mdd = dd
        return mdd

    def mdd_r(self) -> float:
        return self._portfolio_mdd_r

    def mdd_pct(self) -> float:
        """Portfolio-level max drawdown as a fraction of peak equity.

        None entries are restart session-breaks — each segment is evaluated
        independently (equity resets to capital on restart).
        """
        peak: Optional[float] = None
        mdd = 0.0
        for v in self.eq_hist:
            if v is None:
                peak = None
                continue
            if peak is None or v > peak:
                peak = v
            if peak is not None and peak > 0.0:
                dd = (peak - v) / peak
                if dd > mdd:
                    mdd = dd
        return mdd

    def asset_mdd_r(self, asset_id: str) -> float:
        return self._asset_mdd_r.get(asset_id, 0.0)
    
    def _record_metrics(self) -> None:
        # ------------------------------------------------------------
        # 1. Drain closed-trade events from every asset.
        #    Each event carries a monotonic observation timestamp.
        # ------------------------------------------------------------
        events: list[tuple[int, str, float]] = []

        for a in self.assets:
            with a._lock:
                pending = a._pending_trade_rs
                a._pending_trade_rs = []

            for ts_ns, r_value in pending:
                events.append((ts_ns, a.asset_id, r_value))

        # Merge all assets into actual observed close order.
        events.sort(key=lambda x: x[0])

        for _ts_ns, asset_id, r_value in events:
            self.trade_rs.append(r_value)

            asset_hist = self.trade_rs_by_asset.setdefault(
                asset_id,
                deque(maxlen=20_000),
            )
            asset_hist.append(r_value)

            # Portfolio cumulative closed-trade R.
            self._portfolio_trade_r_cumulative += r_value

            # Asset cumulative closed-trade R.
            self._asset_trade_r_cumulative.setdefault(asset_id, 0.0)
            self._asset_trade_r_cumulative[asset_id] += r_value

            # Update portfolio MDD-R.
            if not hasattr(self, "_portfolio_trade_r_peak"):
                self._portfolio_trade_r_peak = self._portfolio_trade_r_cumulative
            else:
                self._portfolio_trade_r_peak = max(
                    self._portfolio_trade_r_peak,
                    self._portfolio_trade_r_cumulative,
                )

            self._portfolio_mdd_r = max(
                self._portfolio_mdd_r,
                self._portfolio_trade_r_peak
                - self._portfolio_trade_r_cumulative,
            )

            # Update asset MDD-R.
            if not hasattr(self, "_asset_trade_r_peak"):
                self._asset_trade_r_peak = {}

            peak = self._asset_trade_r_peak.get(
                asset_id,
                self._asset_trade_r_cumulative[asset_id],
            )
            peak = max(
                peak,
                self._asset_trade_r_cumulative[asset_id],
            )
            self._asset_trade_r_peak[asset_id] = peak

            self._asset_mdd_r[asset_id] = max(
                self._asset_mdd_r.get(asset_id, 0.0),
                peak - self._asset_trade_r_cumulative[asset_id],
            )

        # ------------------------------------------------------------
        # 2. Equity MDD remains sampled at monitor cadence.
        # ------------------------------------------------------------
        tt = self.totals()
        self.eq_hist.append(tt["equity"])

        # ------------------------------------------------------------
        # 3. Network history.
        # ------------------------------------------------------------
        for a in self.assets:
            h = self.net_hist_by_asset.setdefault(
                a.asset_id,
                {
                    "lat": deque(maxlen=NET_HIST_MAX),
                    "stal": deque(maxlen=NET_HIST_MAX),
                    "sk": deque(maxlen=NET_HIST_MAX),
                },
            )

            if a.lat_ms is not None:
                h["lat"].append(a.lat_ms)

            if a.stal_ms is not None:
                h["stal"].append(a.stal_ms)

            if a.skew_ms is not None:
                if abs(a.skew_ms) <= self.SK_GUARD_MS:
                    h["sk"].append(a.skew_ms)
                else:
                    self.sk_discarded_by_asset[a.asset_id] = (
                        self.sk_discarded_by_asset.get(a.asset_id, 0) + 1
                    )
    @staticmethod
    def _stats_from(values) -> dict:
        vals = list(values)
        if not vals:
            return None
        return {"avg": sum(vals) / len(vals), "max": max(vals), "min": min(vals)}

    def asset_net_stats(self, asset_id: str) -> dict:
        """Session avg/max/min for LAT / STAL / SK of one asset (None if no data)."""
        h = self.net_hist_by_asset.get(asset_id, {})
        return {
            "lat": self._stats_from(h.get("lat", ())),
            "stal": self._stats_from(h.get("stal", ())),
            "sk": self._stats_from(h.get("sk", ())),
        }

    @staticmethod
    def _downsample(values, target: int = 180) -> list:
        """Downsample a deque to ~`target` points for the HTML chart payload."""
        vals = list(values)
        if len(vals) <= target:
            return vals
        step = len(vals) / target
        return [vals[int(i * step)] for i in range(target)]

    # ---- local HTTP control endpoint (used by the HTML dashboard buttons) ----------

    def _start_control_server(self, port: int = 8899) -> None:
        """Serve POST /api/control/<ticker|all>/<flatten|pause|resume> locally.


        The HTML dashboard is a static auto-refreshing file; a browser button
        cannot write to disk, so this tiny loopback endpoint bridges them.
"""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: str = "ok") -> None:
                payload = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_OPTIONS(self) -> None:  # CORS preflight
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                parts = self.path.strip("/").split("/")
                if (
                    len(parts) == 4
                    and parts[0] == "api"
                    and parts[1] == "control"
                    and parts[3] in ("flatten", "pause", "resume")
                ):
                    try:
                        self.server.owner.send_control(parts[2], parts[3])
                        self._send(200, "ok")
                    except Exception as exc:  # noqa: BLE001
                        self._send(500, f"error: {exc}")
                else:
                    self._send(400, "bad request")

            def log_message(self, *args) -> None:  # silence access logs
                pass

        try:
            srv = HTTPServer(("127.0.0.1", port), _Handler)
            srv.owner = self
            self._control_server = srv
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            logger.info("control endpoint: http://127.0.0.1:%d/api/control/<ticker|all>/<flatten|pause|resume>", port)
        except OSError as exc:
            logger.warning("control endpoint unavailable (port %d busy?): %s", port, exc)

    def _stop_control_server(self) -> None:
        if self._control_server is not None:
            try:
                self._control_server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._control_server = None

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
            now = time.time()
            for a in self.assets:
                if a.proc is None or a.status == "DOWN":
                    continue
                code = a.proc.poll()
                if code is not None:
                    a.mark_down(code)
                    # A user-driven :stop must NOT auto-restart; only crashes do.
                    if self.restart_enabled and not a.stopped_by_user:
                        self._restart(a)
                    continue
                # Child process alive but silent: [net] telemetry arrives every
                # 5s, so >10s of silence means the child is hung OR never
                # connected. We key off last_output_time (any line — [bar]/
                # [net]/[seed]/[feed]/error), so a child that fails to connect
                # and produces NOTHING still honestly flips to "stalled"
                # instead of lying "running" forever.
                if (
                    a.status in ("running", "outage")
                    and time.monotonic() - a.last_output_time > 10
                ):
                    a.status = "stalled"
            try:
                self._print_table()
                self._record_metrics()
            except Exception:  # noqa: BLE001 — a metrics/table bug must never freeze
                # the monitor (and therefore the whole dashboard). Log and keep
                # ticking so statuses stay live.
                logger.exception("[monitor] metrics/table step failed")
            if self.dashboard_enabled:
                try:
                    self.write_dashboard()
                except Exception:  # noqa: BLE001 — same guard; dashboard write is
                    # best-effort and must never kill the monitor loop.
                    logger.exception("[monitor] dashboard write failed")
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

    def _signal_children(self, signum) -> None:
        """Forward a signal to every running child immediately.
        Called from the SIGINT/SIGTERM handler so each live-trade process
        triggers its own graceful shutdown (RocksDB flush + close) in parallel,
        rather than waiting for the monitor loop to notice a flag.
        """
        for a in self.assets:
            if a.proc and a.proc.poll() is None and not a.stop_requested:
                try:
                    if sys.platform == "win32":
                        a.proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        a.proc.send_signal(signal.SIGINT)
                except (OSError, ProcessLookupError):
                    pass

    def stop_all(self) -> None:
        self.stop_requested = True
        self._stop_control_server()
        logger.info("stopping %d asset(s)...", len(self.assets))
        threads = []
        for a in self.assets:
            t = threading.Thread(target=a.graceful_stop)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=15.0)

    def totals(self) -> dict[str, float]:
        cap = sum(a.capital for a in self.assets)
        eq = sum(a.equity for a in self.assets)
        return {
            "capital": cap,
            "equity": eq,
            "net": eq - cap,
            "trades": sum(a.n_trades for a in self.assets),
            "voided": sum(a.n_voided for a in self.assets),
            "r": sum(a.total_r for a in self.assets),
            "up": sum(1 for a in self.assets if a.status == "running"),
            "outages": sum(a.outages for a in self.assets),
        }

    def network(self) -> dict:
        """ONE singular network-health measurement for the whole portfolio.

        Staleness / latency / clock skew measure the algorithm's general
        communication speed, not a per-asset property — so the dashboard
        shows a single aggregate (worst across the children) instead of one
        track per asset. Outages are summed (each child counts episodes on
        its own feed).
        """
        stals = [a.stal_ms for a in self.assets if a.stal_ms is not None]
        lats = [a.lat_ms for a in self.assets if a.lat_ms is not None]
        skews = [ a.skew_ms for a in self.assets if a.skew_ms is not None and abs(a.skew_ms) <= self.SK_GUARD_MS]
        rtts = [a.rtt_ms for a in self.assets if a.rtt_ms is not None]
        ords = [a.ord_rtt_ms for a in self.assets if a.ord_rtt_ms is not None]
        maxs = [a.max_stal_ms for a in self.assets if a.max_stal_ms is not None]
        return {
            "stal_ms": max(stals) if stals else None,
            "lat_ms": max(lats) if lats else None,
            "skew_ms": max(skews, key=abs) if skews else None,
            "rtt_ms": max(rtts) if rtts else None,
            "ord_ms": max(ords) if ords else None,
            "outages": sum(a.outages for a in self.assets),
            "max_stal_ms": max(maxs) if maxs else None,
            "outage_active": any(a.outage_active for a in self.assets),
        }

    def _print_table(self) -> None:
        if self.quiet:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n=== PORTFOLIO @ {stamp} ===")
        print(
            f"  {'asset':<10}{'status':<10}{'eq($)':>11}{'net($)':>10}"
            f"{'tr':>4}{'void':>5}{'totR':>8}{'mid':>11}{'rsi':>6}{'pWin':>6}  last event"
        )
        for a in self.assets:
            mid = f"{a.mid:.2f}" if a.mid else "-"
            rsi = f"{a.rsi:.1f}" if a.rsi is not None else "-"
            pw = f"{a.pwin:.2f}" if a.pwin is not None else "-"
            voided = f"{a.n_voided}" if a.n_voided > 0 else "-"
            print(
                f"  {a.asset_id:<10}{a.status:<10}{a.equity:>11.2f}"
                f"{a.equity - a.capital:>10.2f}{a.n_trades:>4}{voided:>5}{a.total_r:>8.2f}"
                f"{mid:>11}{rsi:>6}{pw:>6}  {a.last_event}"
            )
        tt = self.totals()
        print(
            f"  {'TOTAL':<10}{'':<10}{tt['equity']:>11.2f}{tt['net']:>10.2f}"
            f"{tt['trades']:>4}{tt['r']:>8.2f}   "
            f"up {tt['up']}/{len(self.assets)} | capital ${tt['capital']:.0f}"
        )
        # ONE network line — the singular general communication-speed
        # measurement (worst across the asset children), not per-asset.
        n = self.network()
        # stal_ms = -1.0 means NO data has ever been received (feed down)
        feed_down = n["stal_ms"] is not None and n["stal_ms"] < 0
        stal = "FEED DOWN" if feed_down else f"{n['stal_ms']:.0f}ms" if n["stal_ms"] is not None else "-"
        lat = f"{n['lat_ms']:.2f}ms" if n["lat_ms"] is not None else "-"
        rtt = f"{n['rtt_ms']:.1f}ms" if n["rtt_ms"] is not None else "-"
        ord_s = f"{n['ord_ms']:.1f}ms" if n["ord_ms"] is not None else "-"
        sk = f"{n['skew_ms']:+.0f}ms" if n["skew_ms"] is not None else "-"
        mx = f"{n['max_stal_ms']:.0f}ms" if n["max_stal_ms"] is not None else "-"
        total_voided = sum(a.n_voided for a in self.assets)
        void_str = f" voided={total_voided}" if total_voided > 0 else ""
        print(
            f"  network: stal={stal} lat={lat} rtt={rtt} ord={ord_s} sk={sk} "
            f"outages={n['outages']} max_stal={mx}{void_str}"
            f"{' [OUTAGE]' if n['outage_active'] else ''}  (worst across assets)"
        )

    # ---- dashboard --------------------------------------------------------------

    def write_dashboard(self) -> None:
        tt = self.totals()
        data = {
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "totals": {
                **tt,
                "mdd_r": self.mdd_r(),
                "mdd_pct": self.mdd_pct(),
            },
            # ONE singular network-health measurement (not per-asset)
            "network": self.network(),
            "assets": [
                {
                    "id": a.asset_id,
                    "ticker": a.ticker,
                    "enable_regime_detection": a.enable_regime_detection,
                    "feed": a.feed,
                    "exchange": a.exchange,
                    "mode": a.mode,
                    "status": a.status,
                    "paused": a.paused,
                    "capital": a.capital,
                    "equity": a.equity,
                    "net": a.equity - a.capital,
                    "trades": a.n_trades,
                    "voided": a.n_voided,
                    "total_r": a.total_r,
                    "mid": a.mid,
                    "rsi": a.rsi,
                    "pwin": a.pwin,
                    # per-asset network diagnostics (the [net] telemetry the
                    # supervisor parses from each child) + session
                    # avg/max/min + a downsampled series for the charts
                    "net": {
                        "lat_ms": a.lat_ms,
                        "stal_ms": a.stal_ms,
                        "skew_ms": (
                            a.skew_ms
                            if a.skew_ms is not None
                            and abs(a.skew_ms) <= self.SK_GUARD_MS
                            else None
                        ),
                        "outages": a.outages,
                        "outage_active": a.outage_active,
                        "stats": self.asset_net_stats(a.asset_id),
                        "series": {
                            k: self._downsample(
                                self.net_hist_by_asset.get(a.asset_id, {}).get(k, ())
                            )
                            for k in ("lat", "stal", "sk")
                        },
                    },
                    # per-asset max drawdown in R-multiples
                    "mdd_r": self.asset_mdd_r(a.asset_id),
                    # network diagnostics are ONE global measurement in
                    # data["network"] (worst across assets) — not per-asset.
                    # the session report is deleted at session end (data lives
                    # in RocksDB), so only link it while it actually exists
                    "has_report": (BUILD_DIR / f"pnl_report_{a.asset_id}.html").exists(),
                    "restarts": a.restarts,
                    "lastEvent": a.last_event,
                    "alerts": a.alerts[-5:],
                    "report": f"pnl_report_{a.asset_id}.html",
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
    vim: bool = True,
) -> int:
    """Entry point behind `unified_runner.py portfolio --config ...`.

    The unified JSON config (config/portfolio.json) is the single source of
    truth for the portfolio layout AND the strategy parameters. Every child
    receives `--config <that file>` DIRECTLY — the Rust engine resolves its
    own ticker (`_strategy_defaults` + per-asset `strategy` overrides,
    deep-merged in-memory). No per-ticker config file is ever written.
    """
    try:
        pc = load_portfolio_config(config_path)
    except Exception as exc:  # noqa: BLE001 — any parse/validation failure is fatal here
        print(f"ERROR: cannot load portfolio config {config_path}: {exc}", file=sys.stderr)
        return 1

    try:
        bin_path = find_live_trade_bin(prefer_release=release)
    except Exception as exc:  # noqa: BLE001
        bin_path = None
        print(f"ERROR: locating live-trade binary: {exc}", file=sys.stderr)
    if bin_path is None:
        print(
            "ERROR: live-trade binary not found under target/. "
            "Build it first: cargo build -p live-trade (--release)",
            file=sys.stderr,
        )
        return 1

    unified_cfg = str(Path(config_path).resolve())
    assets = [
        AssetHandle(
            {
                "id": a["id"],
                "ticker": a["ticker"],
                "exchange": a["exchange"],
                "feed": a["feed"],
                "mode": a["mode"],
                "capital": a["capital"],
                "db": a["db"],
                "enable_regime_detection": a["enable_regime_detection"],
                "config": unified_cfg,
                "strategy_merged": a["strategy_merged"],
            },
            {"exchange": pc["exchange"], "feed": pc["feed"], "mode": pc["mode"],
             "capital": pc["capital"]},
            bin_path,
        )
        for a in pc["assets"]
    ]

    sup = PortfolioSupervisor(
        assets,
        restart=restart or pc["restart"],
        status_interval=float(pc["status_interval"] or status_interval),
        dashboard=not no_dashboard,
        quiet=vim,  # the TUI owns the terminal — no plain table prints
    )

    state = {"stopped": False}

    def _shutdown(signum, _frame):
        if not state["stopped"]:
            state["stopped"] = True
            logger.info(
                "%s received - shutting down portfolio - signaling children...",
                signal.Signals(signum).name,
            )
            # A custom SIGINT handler suppresses Python's default KeyboardInterrupt,
            # so the monitor() loop only ever exits because *this flag* goes False.
            # Setting it here turns the loop, letting the finally below call
            # stop_all() (graceful RocksDB flush + kill-if-needed on every child).
            sup.stop_requested = True
            # Forward the signal to every child immediately so each live-trade
            # process gets its own graceful shutdown (RocksDB flush + close).
            sup._signal_children(signum)

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
        if vim:
            # Full-screen VIM-style terminal dashboard. The supervision loop
            # keeps running on a daemon thread so crash detection, restarts
            # and the HTML dashboard keep working while the TUI owns stdin.
            from vim_dashboard import run_vim_dashboard

            sup_thread = threading.Thread(target=sup.monitor, daemon=True)
            sup_thread.start()
            run_vim_dashboard(sup, config_path)
        else:
            sup.monitor()
    except KeyboardInterrupt:
        pass
    finally:
        sup.stop_all()
    logger.info("portfolio stopped cleanly")
    return 0


