"""VIM-style full-screen terminal dashboard for the portfolio supervisor.

Launched by `unified_runner.py portfolio` (default; `--no-vim` falls back to
the legacy plain console table). The terminal becomes a modal, VIM-like UI:

Views: [1] dashboard (totals + per-asset rows), [2] trades (event stream),
[3] config (the portfolio.json you edit on disk).

Keys: 1/2/3 or Tab switch views, j/k cursor, g/G top/bottom, Ctrl-d/u
half-page, r force-reload config, R restart highlighted asset, ':' command
mode (:q :w :view <v> :restart <t> :stop <t> :start <t>), q quit.

Live config editing: save config/portfolio.json in any editor — the dashboard
watches its mtime, and gracefully restarts every asset whose effective
config (defaults + that asset's strategy overrides) changed. Children re-read
the same JSON file directly; no per-ticker config file is ever written.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from portfolio_supervisor import load_portfolio_config

try:
    import curses
except ImportError as _exc:  # pragma: no cover — platform specific
    if os.name == "nt":
        raise RuntimeError(
            "the VIM dashboard needs 'windows-curses' on Windows: "
            "pip install windows-curses (or run with --no-vim)"
        ) from _exc
    raise

REFRESH_SECS = 0.5
HISTORY_MAX = 1200
VIEWS = ("dashboard", "trades", "config")


class VimDashboard:
    """Modal curses UI over a running PortfolioSupervisor."""

    def __init__(self, sup, config_path: str, ticker_configs: Optional[Dict[str, str]] = None):
        self.sup = sup
        self.config_path = Path(config_path)
        self.view = "dashboard"
        self.cursor = 0
        self.scroll = 0
        self.command = ""
        self.command_mode = False
        self.message = "ready - edit portfolio.json live; :q quits"
        self.message_until = 0.0
        self.cfg_mtime = self._mtime()
        self.asset_hist = {a.ticker: deque(maxlen=HISTORY_MAX) for a in sup.assets}
        self.port_hist: deque = deque(maxlen=HISTORY_MAX)
        self._last_totals = None

    # ------------------------------------------------------------------ utils
    def _mtime(self):
        try:
            return self.config_path.stat().st_mtime
        except OSError:
            return None

    def _flash(self, msg: str) -> None:
        self.message = msg
        self.message_until = time.monotonic() + 6.0

    def _selected(self):
        assets = self.sup.assets
        if not assets:
            return None
        return assets[min(self.cursor, len(assets) - 1)]

    # ------------------------------------------------------- live config edit
    def check_config_reload(self) -> None:
        mtime = self._mtime()
        if mtime is None or mtime == self.cfg_mtime:
            return
        self.cfg_mtime = mtime
        self.reload_config()

    def reload_config(self) -> None:
        """Re-parse the JSON and restart every asset whose effective config
        (defaults + that asset's strategy overrides) changed. Nothing is
        written to disk — children re-read the same file themselves."""
        try:
            pc = load_portfolio_config(self.config_path)
        except Exception as exc:  # noqa: BLE001 — survive a bad save
            self._flash(f"CONFIG ERROR (keeping last good): {exc}")
            return

        handles = {a.ticker: a for a in self.sup.assets}
        restarted = []
        for spec in pc["assets"]:
            a = handles.get(spec["ticker"])
            if a is None:
                self._spawn_new_asset(spec, pc)
                restarted.append(spec["ticker"] + "(new)")
                continue
            cfg_changed = json.dumps(spec["strategy_merged"], sort_keys=True) != \
                json.dumps(getattr(a, "strategy_merged", None), sort_keys=True)
            venue_changed = (
                a.exchange != spec["exchange"]
                or a.feed != spec["feed"]
                or a.mode != spec["mode"]
                or abs(a.capital - spec["capital"]) > 1e-9
            )
            if cfg_changed or venue_changed:
                a.strategy_merged = spec["strategy_merged"]
                self._restart_asset(a)
                restarted.append(spec["ticker"])

        self._flash(
            "reloaded - restarted: " + ", ".join(restarted)
            if restarted else "config reloaded - no effective changes"
        )

    def _spawn_new_asset(self, spec, pc) -> None:
        try:
            from portfolio_supervisor import AssetHandle

            handle = AssetHandle(
                {"ticker": spec["ticker"], "exchange": spec["exchange"],
                 "feed": spec["feed"], "mode": spec["mode"],
                 "capital": spec["capital"], "db": spec["db"],
                 "config": str(Path(self.config_path).resolve())},
                {"exchange": pc["exchange"], "feed": pc["feed"],
                 "mode": pc["mode"], "capital": pc["capital"]},
                self.sup.assets[0].cmd[0] if self.sup.assets else "live-trade",
            )
            handle.strategy_merged = spec["strategy_merged"]
            handle.spawn()
            self.sup.assets.append(handle)
            self.asset_hist[handle.ticker] = deque(maxlen=HISTORY_MAX)
        except Exception as exc:  # noqa: BLE001
            self._flash(f"failed to start {spec['ticker']}: {exc}")

    def _restart_asset(self, a) -> None:
        def _worker():
            try:
                a.graceful_stop()
            except Exception:  # noqa: BLE001
                pass
            if not self.sup.stop_requested:
                a.status = "starting"
                a.spawn()
        threading.Thread(target=_worker, daemon=True).start()

    def _stop_asset(self, a) -> None:
        threading.Thread(target=a.graceful_stop, daemon=True).start()

    # ------------------------------------------------------------- metrics
    def _sample(self) -> None:
        for a in self.sup.assets:
            hist = self.asset_hist.setdefault(a.ticker, deque(maxlen=HISTORY_MAX))
            hist.append((a.equity / a.capital - 1.0) if a.capital else 0.0)
        totals = self.sup.totals()
        if self._last_totals is not None and totals["capital"] > 0:
            self.port_hist.append(
                (totals["equity"] - self._last_totals["equity"]) / totals["capital"]
            )
        self._last_totals = totals

    def _summary(self) -> dict:
        totals = self.sup.totals()
        wins = sum(a.n_wins for a in self.sup.assets)
        losses = sum(a.n_losses for a in self.sup.assets)
        n = totals["trades"]
        net = totals["net"]
        return {
            "capital": totals["capital"], "equity": totals["equity"], "pnl": net,
            "pnl_pct": (net / totals["capital"] * 100.0) if totals["capital"] else 0.0,
            "trades": n, "wins": wins, "losses": losses,
            "win_rate": (wins / (wins + losses) * 100.0) if (wins + losses) else 0.0,
            "total_r": totals["r"], "avg_r": (totals["r"] / n) if n else 0.0,
            "sortino": _compute_sortino(list(self.port_hist)),
            "sharpe": _compute_sharpe(list(self.port_hist)),
            "up": totals["up"], "n_assets": len(self.sup.assets),
            "outages": totals["outages"],
        }

    # ------------------------------------------------------------ rendering
    def _fmt_pnl(self, v: float) -> str:
        return f"{v:+.2f}"

    def _draw(self, stdscr) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)   # header
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)

        self._draw_header(stdscr, w)
        body_top, body_bot = 2, h - 2
        if self.view == "dashboard":
            self._draw_dashboard(stdscr, body_top, body_bot, w)
        elif self.view == "trades":
            self._draw_trades(stdscr, body_top, body_bot, w)
        else:
            self._draw_config(stdscr, body_top, body_bot, w)
        self._draw_footer(stdscr, h, w)
        stdscr.noutrefresh()
        curses.doupdate()

    def _clip(self, s: str, w: int) -> str:
        return s[: max(0, w - 1)]

    def _draw_header(self, stdscr, w: int) -> None:
        tabs = "  ".join(
            (f"[{i+1}] {v.upper()}" if v == self.view else f" {i+1}  {v} ")
            for i, v in enumerate(VIEWS)
        )
        mode = self.sup.assets[0].mode if self.sup.assets else "?"
        stamp = datetime.now().strftime("%H:%M:%S")
        stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        stdscr.addnstr(0, 0, f" OSADP PORTFOLIO | {tabs} ".ljust(w), w - 1)
        stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
        stdscr.addnstr(1, 0, f" {mode.upper()} | {stamp} ".ljust(w), w - 1)

    def _draw_footer(self, stdscr, h: int, w: int) -> None:
        if self.command_mode:
            stdscr.addnstr(h - 2, 0, ":" + self.command, w - 1, curses.A_BOLD)
        else:
            msg = self.message if time.monotonic() < self.message_until else ""
            stdscr.addnstr(h - 2, 0, f" {msg}".ljust(w), w - 1)
        keys = (" j/k move | 1/2/3 view | r reload cfg | R restart | : cmd | q quit "
                if self.command_mode else
                " 1:dashtab 2:trades 3:config | j/k | g/G | r reload | R restart | : cmds | q quit")
        stdscr.addnstr(h - 1, 0, keys[:w - 1], w - 1, curses.A_DIM)

    def _draw_dashboard(self, stdscr, top: int, bot: int, w: int) -> None:
        s = self._summary()

        def row(line: int, label: str, value: str, pair: int = 0, bold: bool = False):
            attr = (curses.color_pair(pair) | (curses.A_BOLD if bold else 0))
            stdscr.addnstr(line, 0, f" {label}{value}", w - 1, attr)

        pnl_pair = 2 if s["pnl"] >= 0 else 3
        row(top, "PnL         $", f"{s['pnl']:.2f}  ({s['pnl_pct']:+.2f}%)", pnl_pair, True)
        row(top + 1, "Equity      $", f"{s['equity']:.2f}  / capital ${s['capital']:.0f}")
        row(top + 2, "Trades      ", f"{s['trades']}  (W {s['wins']} / L {s['losses']})")
        row(top + 3, "Win rate    ", f"{s['win_rate']:.1f}%")
        row(top + 4, "RR accum    ", f"{s['total_r']:+.2f}R   avg {s['avg_r']:+.3f}R/trade")
        row(top + 5, "Sortino     ", f"{s['sortino']:.3f}   (sharpe {s['sharpe']:.3f})")
        row(top + 6, "Assets up   ",
            f"{s['up']}/{s['n_assets']}   feed outages: {s['outages']}",
            4 if s["up"] < s["n_assets"] else 0)

        hdr_y = top + 8
        stdscr.addnstr(hdr_y, 0,
                       "  ASSET     STATUS    EQ($)      NET($)  TR  TOT_R    "
                       "MID        RSI   PWIN  LAST EVENT", w - 1,
                       curses.A_BOLD | curses.A_UNDERLINE)
        assets = self.sup.assets
        self.cursor = max(0, min(self.cursor, len(assets) - 1))
        win_h = bot - hdr_y - 1
        start = max(0, min(self.scroll, self.cursor - win_h + 1))
        start = min(start, max(0, len(assets) - win_h))
        self.scroll = start
        for i in range(start, min(len(assets), start + max(1, win_h))):
            a = assets[i]
            y = hdr_y + 1 + (i - start)
            sel = (i == self.cursor)
            attr = curses.A_REVERSE if sel else 0
            net = a.equity - a.capital
            mid = f"{a.mid:.2f}" if a.mid is not None else "-"
            rsi = f"{a.rsi:.1f}" if a.rsi is not None else "-"
            pw = f"{a.pwin:.2f}" if a.pwin is not None else "-"
            txt = (f"{('>' if sel else ' ')} {a.ticker:<9}{a.status:<9}"
                   f"{a.equity:>10.2f}{net:>+10.2f}{a.n_trades:>4}"
                   f"{a.total_r:>+8.2f}  {mid:>10} {rsi:>5} {pw:>5}  "
                   f"{a.last_event[:40]}")
            if not sel:
                if a.status == "DOWN":
                    attr = curses.color_pair(3)
                elif a.status == "outage":
                    attr = curses.color_pair(4)
                elif a.status == "running":
                    attr = curses.color_pair(2)
            stdscr.addnstr(y, 0, txt[:w - 1], w - 1, attr)

    def _draw_trades(self, stdscr, top: int, bot: int, w: int) -> None:
        a = self._selected()
        if a is None:
            stdscr.addnstr(top, 0, " no assets", w - 1)
            return
        stdscr.addnstr(top, 0,
                       f" {a.ticker} trades: {a.n_trades}  (W {a.n_wins} / "
                       f"L {a.n_losses}, {a.total_r:+.2f}R)  - newest last",
                       w - 1, curses.A_BOLD)
        lines = list(a.event_lines)
        win_h = bot - top - 2
        shown = lines[-max(1, win_h):]
        for i, ln in enumerate(shown):
            attr = 0
            if "[exit]" in ln:
                attr = curses.color_pair(2) if "r=+" in ln else curses.color_pair(3)
            elif "[entry]" in ln:
                attr = curses.color_pair(6)
            stdscr.addnstr(top + 2 + i, 0, ln[:w - 1], w - 1, attr)
        if not shown:
            stdscr.addnstr(top + 2, 0, " (no events yet)", w - 1, curses.A_DIM)

    def _draw_config(self, stdscr, top: int, bot: int, w: int) -> None:
        try:
            raw = self.config_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            stdscr.addnstr(top, 0, f" cannot read config: {exc}", w - 1)
            return
        lines = raw.splitlines()
        self.cursor = max(0, min(self.cursor, len(lines) - 1))
        win_h = bot - top + 1
        start = max(0, min(self.scroll, self.cursor - win_h + 1))
        start = min(start, max(0, len(lines) - win_h))
        self.scroll = start
        for i in range(start, min(len(lines), start + win_h)):
            y = top + (i - start)
            attr = curses.A_REVERSE if i == self.cursor else curses.A_DIM
            stdscr.addnstr(y, 0, f" {lines[i]}"[:w - 1], w - 1, attr)
        stdscr.addnstr(
            bot + 1, 0,
            f" {self.config_path}  line {self.cursor + 1}/{len(lines)}"
            "  (edit on disk -> auto-reload)".ljust(w - 1),
            w - 1, curses.color_pair(1))

    # --------------------------------------------------------------- input
    def _exec_command(self) -> None:
        cmd = self.command.strip()
        self.command, self.command_mode = "", False
        if not cmd:
            return
        parts = cmd.split()
        op, args = parts[0].lower(), parts[1:]
        if op in ("q", "quit", "q!"):
            raise SystemExit(0)
        if op in ("w", "write"):
            self.reload_config()
        elif op == "view":
            if args and args[0] in VIEWS:
                self.view, self.cursor, self.scroll = args[0], 0, 0
            else:
                self._flash(f":view expects one of {', '.join(VIEWS)}")
        elif op in ("restart", "stop", "start") and args:
            handles = {a.ticker: a for a in self.sup.assets}
            a = handles.get(args[0].lower())
            if a is None:
                self._flash(f"unknown asset: {args[0]}")
            elif op == "restart":
                self._restart_asset(a)
                self._flash(f"restarting {a.ticker} with its latest config...")
            elif op == "stop":
                self._stop_asset(a)
                self._flash(f"stopping {a.ticker}...")
            elif a.proc and a.proc.poll() is None:
                self._flash(f"{a.ticker} already running")
            else:
                a.spawn()
                self._flash(f"started {a.ticker}")
        else:
            self._flash("unknown command (try :q :w :view :restart :stop :start)")

    def handle_key(self, key: int) -> None:
        if self.command_mode:
            if key in (10, 13, curses.KEY_ENTER):
                self._exec_command()
            elif key == 27:
                self.command, self.command_mode = "", False
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                self.command = self.command[:-1]
            elif 32 <= key < 127:
                self.command += chr(key)
            return

        n_assets = max(1, len(self.sup.assets))
        if key == ord("1"):
            self.view, self.cursor, self.scroll = "dashboard", 0, 0
        elif key == ord("2"):
            self.view, self.cursor, self.scroll = "trades", 0, 0
        elif key == ord("3"):
            self.view, self.cursor, self.scroll = "config", 0, 0
        elif key in (9, curses.KEY_RIGHT, ord("l")):
            nxt = (VIEWS.index(self.view) + 1) % len(VIEWS)
            self.view, self.cursor, self.scroll = VIEWS[nxt], 0, 0
        elif key in (curses.KEY_LEFT, ord("h")):
            prv = (VIEWS.index(self.view) - 1) % len(VIEWS)
            self.view, self.cursor, self.scroll = VIEWS[prv], 0, 0
        elif key in (ord("j"), curses.KEY_DOWN):
            self.cursor = min(self.cursor + 1, n_assets - 1)
        elif key in (ord("k"), curses.KEY_UP):
            self.cursor = max(self.cursor - 1, 0)
        elif key == ord("g"):
            self.cursor = self.scroll = 0
        elif key == ord("G"):
            self.cursor = n_assets - 1
        elif key == 4:                                   # Ctrl-d
            self.cursor = min(self.cursor + 10, n_assets - 1)
        elif key == 21:                                  # Ctrl-u
            self.cursor = max(self.cursor - 10, 0)
        elif key == ord("r"):
            self.reload_config()
        elif key in (ord("R"), curses.KEY_F5):
            a = self._selected()
            if a:
                self._restart_asset(a)
                self._flash(f"restarting {a.ticker} with its latest config...")
        elif key == ord(":"):
            self.command, self.command_mode = "", True
        elif key in (ord("q"), ord("Q")):
            raise SystemExit(0)

    # ---------------------------------------------------------------- main
    def run(self, stdscr) -> None:
        stdscr.timeout(int(REFRESH_SECS * 1000))
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        last_sample = 0.0
        while True:
            self.check_config_reload()
            now = time.monotonic()
            if now - last_sample >= REFRESH_SECS:
                self._sample()
                last_sample = now
            self._draw(stdscr)
            key = stdscr.getch()
            if key != -1:
                self.handle_key(key)


# Dashboard return samples are taken every REFRESH_SECS; annualization uses
# the corresponding number of sampling periods per year (crypto trades 24/7).
_PERIODS_PER_YEAR = 365.0 * 86400.0 / REFRESH_SECS


def _compute_sortino(returns, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio from the sampled per-period portfolio returns."""
    n = len(returns)
    if n < 2:
        return 0.0
    excess = [r - risk_free_rate for r in returns]
    mean = sum(excess) / n
    downside_mean = sum(min(0.0, e) for e in excess) / n
    dd_var = sum((min(0.0, e) - downside_mean) ** 2 for e in excess) / (n - 1)
    dd = dd_var ** 0.5
    if dd <= 0.0:
        return 0.0
    return mean / dd * _PERIODS_PER_YEAR ** 0.5


def _compute_sharpe(returns, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from the sampled per-period portfolio returns."""
    n = len(returns)
    if n < 2:
        return 0.0
    excess = [r - risk_free_rate for r in returns]
    mean = sum(excess) / n
    var = sum((r - mean) ** 2 for r in excess) / (n - 1)
    sd = var ** 0.5
    if sd <= 0.0:
        return 0.0
    return mean / sd * _PERIODS_PER_YEAR ** 0.5


def run_vim_dashboard(sup, config_path: str, ticker_configs: Optional[Dict[str, str]] = None) -> None:
    """Blocking curses entry point (the supervisor keeps running in a thread)."""
    curses.wrapper(VimDashboard(sup, config_path).run)
# __PART2__
