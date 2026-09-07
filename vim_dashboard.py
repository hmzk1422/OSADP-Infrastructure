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
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from portfolio_supervisor import DATA_DIR, load_portfolio_config

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
        # Inline config editing state (Enter/i on a config line).
        self.editing = False
        self.edit_buf = ""
        self.edit_line = 0
        self.edit_token = False  # preloaded only the value token
        self.message = "ready - edit portfolio.json live; :q quits"
        self.message_until = 0.0
        self.cfg_mtime = self._mtime()
        # Keyed by asset_id (not ticker) so twins with the same ticker
        # (e.g. ethusdt + eth-raw) get independent buffers.
        self.asset_hist = {a.asset_id: deque(maxlen=HISTORY_MAX) for a in sup.assets}
        self.port_hist: deque = deque(maxlen=HISTORY_MAX)
        # Clock-offset samples discarded by the |sk| > SK_GUARD_MS guard,
        # per asset id — shown on the dashboard so filtering is visible.
        self.sk_discarded: Dict[str, int] = {}
        # Per-asset rolling network metrics (LAT / STAL / SK) for avg/max/min
        self.asset_metrics: Dict[str, dict] = {
            a.asset_id: {"lat": deque(maxlen=HISTORY_MAX),
                         "stal": deque(maxlen=HISTORY_MAX),
                         "sk": deque(maxlen=HISTORY_MAX)}
            for a in sup.assets
        }
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

    def _selected_snapshot(self):
        """Thread-safe version of _selected() -- returns a coherent dict snapshot
        instead of a live AssetHandle reference that may be mid-update."""
        assets = self.sup.assets
        if not assets:
            return None
        idx = max(0, min(self.cursor, len(assets) - 1))
        return assets[idx].snapshot()


    def _keep_cursor_visible(self, win_h: int) -> None:
        """Adjust config scroll so the cursor is always visible."""
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        elif self.cursor >= self.scroll + win_h:
            self.scroll = self.cursor - win_h + 1

        self.scroll = max(0, self.scroll)
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

        handles = {a.asset_id: a for a in self.sup.assets}
        restarted = []
        for spec in pc["assets"]:
            asset_id = str(
                spec.get("id") or spec["ticker"]
            ).strip().lower()

            a = handles.get(asset_id)
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
                # Apply the NEW venue/capital/config to the handle FIRST so the
                # restart spawns the child with the updated values (and the equity
                # display doesn't stay stuck on the old capital).
                self._apply_spec_to_handle(a, spec)
                self._restart_asset(a)
                restarted.append(spec["ticker"])

        self._flash(
            "reloaded - restarted: " + ", ".join(restarted)
            if restarted else "config reloaded - no effective changes"
        )

    def _apply_spec_to_handle(self, a, spec) -> None:
        old_cap = a.capital
        a.exchange = spec.get("exchange", a.exchange)
        a.feed = spec.get("feed", a.feed)
        a.mode = spec.get("mode", a.mode)
        a.capital = float(spec.get("capital", a.capital))
        a.enable_regime_detection = bool(
            spec.get("enable_regime_detection", a.enable_regime_detection)
        )
        a.config = spec.get("config") or str(self.config_path.resolve())
        a.db = spec.get("db") or str(DATA_DIR / f"portfolio_{a.asset_id}.rocksdb")
        a.strategy_merged = spec.get(
            "strategy_merged", getattr(a, "strategy_merged", None)
        )

        a.cmd = [
            a.cmd[0],
            a.ticker,
            "--exchange", a.exchange,
            "--feed", a.feed,
            "--mode", a.mode,
            "--capital", str(a.capital),
            "--db", a.db,
            "--config", a.config,
            "--control", a.control_path,
            "--meta-gate", "on" if a.enable_regime_detection else "off",
            "--report-tag", a.asset_id,
        ]

        a.equity = a.capital

        if abs(old_cap - a.capital) > 1e-9:
            a.cmd.append("--fresh-equity")

    def _spawn_new_asset(self, spec, pc) -> None:
        try:
            from portfolio_supervisor import AssetHandle

            handle = AssetHandle(
                {
                    "id": spec.get("id", spec["ticker"]),
                    "ticker": spec["ticker"],
                    "exchange": spec["exchange"],
                    "feed": spec["feed"],
                    "mode": spec["mode"],
                    "capital": spec["capital"],
                    "db": spec["db"],
                    "enable_regime_detection": spec["enable_regime_detection"],
                    "config": str(Path(self.config_path).resolve()),
                },
                {"exchange": pc["exchange"], "feed": pc["feed"],
                 "mode": pc["mode"], "capital": pc["capital"]},
                self.sup.assets[0].cmd[0] if self.sup.assets else "live-trade",
            )
            handle.strategy_merged = spec["strategy_merged"]
            handle.spawn()
            self.sup.assets.append(handle)
            # Key history by asset_id (NOT ticker): twin assets can share a
            # ticker (e.g. ethusdt + eth-raw both trading ethusdt) and must
            # never share one history deque.
            self.asset_hist[handle.asset_id] = deque(maxlen=HISTORY_MAX)
            self.asset_metrics.setdefault(
                handle.asset_id,
                {"lat": deque(maxlen=HISTORY_MAX),
                 "stal": deque(maxlen=HISTORY_MAX),
                 "sk": deque(maxlen=HISTORY_MAX)})
        except Exception as exc:  # noqa: BLE001
            self._flash(f"failed to start {spec['ticker']}: {exc}")

    def _restart_asset(self, a) -> None:
        def _worker():
            a.stopped_by_user = True   # monitor must NOT auto-restart mid-restart
            try:
                a.graceful_stop()
            except Exception:  # noqa: BLE001
                pass
            if not self.sup.stop_requested:
                a.status = "starting"
                a.spawn()
                a.stopped_by_user = False
        threading.Thread(target=_worker, daemon=True).start()

    def _force_restart(self, a) -> None:
        """Re-read the latest portfolio.json spec for this asset, apply it to the
        handle (capital/venue/config) and restart — so `R` / `:restart`
        actually pick up config edits, not just re-spawn with stale values."""
        try:
            pc = load_portfolio_config(self.config_path)
            spec = next(
                (
                    s for s in pc["assets"]
                    if str(s.get("id") or s["ticker"]).strip().lower()
                    == a.asset_id
                ),
                None,
            )
            if spec is not None:
                self._apply_spec_to_handle(a, spec)
        except Exception as exc:  # noqa: BLE001
            self._flash(f"restart: could not re-read config: {exc}")
        self._restart_asset(a)

    def _stop_asset(self, a) -> None:
        a.stopped_by_user = True   # user asked to stop → monitor won't auto-restart it
        threading.Thread(target=a.graceful_stop, daemon=True).start()

        # ------------------------------------------------------------- metrics
    def _sample(self) -> None:
        snapshots = [a.snapshot() for a in self.sup.assets]

        for snap in snapshots:
            hist = self.asset_hist.setdefault(
                snap["asset_id"], deque(maxlen=HISTORY_MAX)
            )
            hist.append(
                (snap["equity"] / snap["capital"] - 1.0)
                if snap["capital"] else 0.0
            )

            m = self.asset_metrics.setdefault(
                snap["asset_id"],
                {
                    "lat": deque(maxlen=HISTORY_MAX),
                    "stal": deque(maxlen=HISTORY_MAX),
                    "sk": deque(maxlen=HISTORY_MAX),
                },
            )

            if snap["lat_ms"] is not None:
                m["lat"].append(snap["lat_ms"])
            if snap["stal_ms"] is not None:
                m["stal"].append(snap["stal_ms"])

            if snap["skew_ms"] is not None:
                if abs(snap["skew_ms"]) <= self.sup.SK_GUARD_MS:
                    m["sk"].append(snap["skew_ms"])
                else:
                    self.sk_discarded[snap["asset_id"]] = (
                        self.sk_discarded.get(snap["asset_id"], 0) + 1
                    )

        totals = self.sup.totals()
        if self._last_totals is not None and totals["capital"] > 0:
            self.port_hist.append(
                (totals["equity"] - self._last_totals["equity"])
                / totals["capital"]
            )
        self._last_totals = totals

    def _summary(self) -> dict:
        """Read portfolio totals directly from live AssetState (via snapshot),
        NOT from sampled history -- so the dashboard reflects the current state
        instantly, not 5s-lagged."""
        snaps = [a.snapshot() for a in self.sup.assets]
        capital = sum(s["capital"] for s in snaps)
        equity = sum(s["equity"] for s in snaps)
        wins = sum(s["n_wins"] for s in snaps)
        losses = sum(s["n_losses"] for s in snaps)
        n = sum(s["n_trades"] for s in snaps)
        net = equity - capital
        total_r = sum(s["total_r"] for s in snaps)
        up = sum(1 for s in snaps if s["status"] == "running")
        outages = sum(s["outages"] for s in snaps)
        # Sortino/Sharpe are TRADE-LEVEL and NON-ANNUALIZED: computed from the
        # closed-trade R-multiple sequence (the strategy trades discretely, so
        # per-trade R is the honest return series — sampling the UI cannot
        # create new market observations). None => statistically undefined.
        return {
            "capital": capital, "equity": equity, "pnl": net,
            "pnl_pct": (net / capital * 100.0) if capital else 0.0,
            "trades": n, "wins": wins, "losses": losses,
            "win_rate": (wins / (wins + losses) * 100.0) if (wins + losses) else 0.0,
            "total_r": total_r, "avg_r": (total_r / n) if n else 0.0,
            "sortino": _compute_sortino(list(self.sup.trade_rs)),
            "sharpe": _compute_sharpe(list(self.sup.trade_rs)),
            "up": up, "n_assets": len(snaps),
            "outages": outages,
        }

    def _metric_stats(self, asset_id: str, key: str) -> dict:
        """avg/max/min over the session for one metric ('lat'|'stal'|'sk')."""
        vals = list(self.asset_metrics.get(asset_id, {}).get(key, ()))
        if not vals:
            return {"avg": None, "max": None, "min": None}
        return {"avg": sum(vals) / len(vals), "max": max(vals), "min": min(vals)}

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
        # The wall-clock stamp updates EVERY redraw (~20Hz) — this is your
        # proof the screen is alive even when LAST EVENT is frozen (feed stall).
        stdscr.addnstr(1, 0,
                       f" {mode.upper()} | refresh {stamp} ".ljust(w), w - 1)

    def _draw_footer(self, stdscr, h: int, w: int) -> None:
        if self.editing:
            stdscr.addnstr(h - 2, 0,
                           f" EDIT line {self.edit_line + 1}: {self.edit_buf}"
                           f"  [Enter=save  Esc=cancel]",
                           w - 1, curses.A_BOLD | curses.color_pair(4))
        elif self.command_mode:
            stdscr.addnstr(h - 2, 0, ":" + self.command, w - 1, curses.A_BOLD)
        else:
            msg = self.message if time.monotonic() < self.message_until else ""
            stdscr.addnstr(h - 2, 0, f" {msg}".ljust(w), w - 1)
        keys = (" j/k move | 1/2/3 view | r reload cfg | R restart | : cmd | q quit "
                if self.command_mode else
                " 1:dashboard 2:trades 3:config | j/k g/G | r reload | R restart"
                " | o open HTML | :q/:w/:html/:restart | q quit")
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
        # Trade-level, non-annualized. N/A = statistically undefined (too few
        # closed trades, or no downside observations yet) — NOT zero.
        sortino_s = f"{s['sortino']:.3f}" if s["sortino"] is not None else "N/A"
        sharpe_s = f"{s['sharpe']:.3f}" if s["sharpe"] is not None else "N/A"
        row(top + 5, "Sortino(trd)", f"{sortino_s}   (sharpe(trd) {sharpe_s})")
        row(top + 6, "Assets up   ",
            f"{s['up']}/{s['n_assets']}   feed outages: {s['outages']}",
            4 if s["up"] < s["n_assets"] else 0)
        # Portfolio-wide network metrics (worst across assets) with session
        # avg/max/min, plus max drawdown in RR terms.
        lat_all = [self._metric_stats(a.asset_id, "lat") for a in self.sup.assets]
        sk_all = [self._metric_stats(a.asset_id, "sk") for a in self.sup.assets]
        lats = [m["max"] for m in lat_all if m["max"] is not None]
        lat_avgs = [m["avg"] for m in lat_all if m["avg"] is not None]
        sks = [m["max"] for m in sk_all if m["max"] is not None]
        sk_avgs = [m["avg"] for m in sk_all if m["avg"] is not None]
        lat_s = (f"avg {sum(lat_avgs)/len(lat_avgs):.1f} worst-max {max(lats):.0f}ms"
                 if lat_avgs else "no data yet")
        # Clock offset is a DIAGNOSTIC (local wall clock vs exchange wall
        # clock) — it is NOT latency. Latency truth = rtt/ord/stal (monotonic).
        # If any child currently reports an offset beyond the guard, the local
        # clock is unsynced — say so instead of displaying garbage numbers.
        sk_discarded_total = sum(self.sk_discarded.values())
        unsynced = [a for a in self.sup.assets
                    if a.skew_ms is not None and abs(a.skew_ms) > self.sup.SK_GUARD_MS]
        if unsynced:
            worst = max((abs(a.skew_ms) for a in unsynced), default=0.0)
            sk_s = (f"CLOCK UNSYNCED (offset up to {worst/60000.0:.0f}min)"
                    " - run 'w32tm /resync'")
        else:
            sk_s = (f"avg {sum(sk_avgs)/len(sk_avgs):+.0f} worst-max {max(sks, key=abs):+.0f}ms"
                    if sk_avgs else "no data yet")
            if sk_discarded_total:
                sk_s += f"  (discarded {sk_discarded_total} clock-step samples)"
        row(top + 7, "Latency     ", lat_s)
        row(top + 8, "Clock offset", sk_s + "  [diag]")
        # MDD-R is the CLOSED-TRADE drawdown (cumulative trade R); MDD-eq is
        # the account-level equity drawdown (includes restarts as breaks).
        row(top + 9, "MDD         ",
            f"{self.sup.mdd_r():+.2f}R (closed-trade)   {self.sup.mdd_pct()*100.0:.2f}% (equity)")

        hdr_y = top + 11
        stdscr.addnstr(hdr_y, 0,
                       "  ASSET     STATUS    EQ($)      NET($)  TR  VOID TOT_R   "
                       "MID        RSI   PWIN  LAST EVENT", w - 1,
                       curses.A_BOLD | curses.A_UNDERLINE)
        snapshots = [a.snapshot() for a in self.sup.assets]
        self.cursor = max(0, min(self.cursor, len(snapshots) - 1))
        win_h = bot - hdr_y - 1
        start = max(0, min(self.scroll, self.cursor - win_h + 1))
        start = min(start, max(0, len(snapshots) - win_h))
        self.scroll = start
        for i in range(start, min(len(snapshots), start + max(1, win_h))):
            snap = snapshots[i]
            y = hdr_y + 1 + (i - start)
            sel = (i == self.cursor)
            attr = curses.A_REVERSE if sel else 0
            net = snap["equity"] - snap["capital"]
            mid = f"{snap['mid']:.2f}" if snap["mid"] is not None else "-"
            rsi = f"{snap['rsi']:.1f}" if snap["rsi"] is not None else "-"
            pw = f"{snap['pwin']:.2f}" if snap["pwin"] is not None else "-"
            voided = f"{snap['n_voided']}" if snap["n_voided"] > 0 else "-"
            asset_id = snap["asset_id"]
            status = snap["status"]
            last_event = snap["last_event"]
            last_net_time = snap["last_net_time"]
            if last_net_time > 0:
                net_age = int(time.time() - last_net_time)
                net_s = f"net{net_age}s"
            else:
                net_s = "net?"
            txt = (f"{('>' if sel else ' ')} {asset_id:<9}{status:<9}"
                   f"{snap['equity']:>10.2f}{net:>+10.2f}{snap['n_trades']:>4}{voided:>5}"
                   f"{snap['total_r']:>+8.2f}  {mid:>10} {rsi:>5} {pw:>5}  "
                   f"{last_event[:34]} {net_s}")
            if not sel:
                if status == "DOWN":
                    attr = curses.color_pair(3)  # red alert
                elif status == "STOPPED":
                    attr = curses.A_DIM  # user asked to stop -- dim, no alarm
                elif status in ("outage", "stalled"):
                    attr = curses.color_pair(4)  # yellow
                elif status == "running":
                    attr = curses.color_pair(2)  # green
            stdscr.addnstr(y, 0, txt[:w - 1], w - 1, attr)

    def _draw_trades(self, stdscr, top: int, bot: int, w: int) -> None:
        snap = self._selected_snapshot()
        if snap is None:
            stdscr.addnstr(top, 0, " no assets", w - 1)
            return
        stdscr.addnstr(top, 0,
                       f" {snap['asset_id']} (ticker {snap['ticker']}) trades: {snap['n_trades']}  "
                       f"(W {snap['n_wins']} / "
                       f"L {snap['n_losses']}, {snap['total_r']:+.2f}R)   "
                       f"MDD-R (closed-trade) {self.sup.asset_mdd_r(snap['asset_id']):.2f}",
                       w - 1, curses.A_BOLD)
        # Network metrics with session avg/max/min (LAT & SK are the important
        # ones; STAL completes the picture). Sampled every REFRESH_SECS tick.
        aid = snap["asset_id"]
        lat = self._metric_stats(aid, "lat")
        sk = self._metric_stats(aid, "sk")
        stal = self._metric_stats(aid, "stal")

        def _fmt3(avg, mx, mn, spec: str) -> str:
            if avg is None:
                return "-/-/-"
            return (f"{format(avg, spec)}/{format(mx, spec)}/{format(mn, spec)}")

        stdscr.addnstr(top + 1, 0,
                       f" LAT {_fmt3(lat['avg'], lat['max'], lat['min'], '.2f')}ms"
                       f"   SK(offset,diag) {_fmt3(sk['avg'], sk['max'], sk['min'], '+.0f')}ms"
                       f"   STAL {_fmt3(stal['avg'], stal['max'], stal['min'], '.0f')}ms"
                       + ("  [OUTAGE now]" if snap["outage_active"] else ""),
                       w - 1, curses.color_pair(1))
        # View 2 is a FAITHFUL MIRROR of the child's stdout: every line the
        # algo would print running bare in a terminal ([bar]/[net]/[order]/
        # [entry]/[exit]/[seed]/[regime]/...), timestamped, newest last.
        lines = snap["log_lines"]
        win_h = bot - top - 4
        shown = lines[-max(1, win_h):]
        for i, ln in enumerate(shown):
            attr = 0
            if "[exit]" in ln:
                attr = curses.color_pair(2) if "r=+" in ln else curses.color_pair(3)
            elif "[entry]" in ln or "[order]" in ln:
                attr = curses.color_pair(6)
            elif "[net]" in ln:
                attr = curses.color_pair(4) if "[OUTAGE]" in ln else curses.A_DIM
            stdscr.addnstr(top + 3 + i, 0, ln[:w - 1], w - 1, attr)
        if not shown:
            stdscr.addnstr(top + 3, 0, " (no output yet)", w - 1, curses.A_DIM)


    def _draw_config(self, stdscr, top: int, bot: int, w: int) -> None:
        try:
            raw = self.config_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            stdscr.addnstr(top, 0, f" cannot read config: {exc}", w - 1)
            return
        lines = raw.splitlines()
        self._config_nlines = len(lines)  # cursor limit for j/k in config view
        self.cursor = max(0, min(self.cursor, len(lines) - 1))
        win_h = bot - top + 1
        self._config_win_h = win_h

        self.cursor = max(0, min(self.cursor, len(lines) - 1))

        self._keep_cursor_visible(win_h)

        max_scroll = max(0, len(lines) - win_h)
        self.scroll = min(self.scroll, max_scroll)

        start = self.scroll
        for i in range(start, min(len(lines), start + win_h)):
            y = top + (i - start)
            attr = curses.A_REVERSE if i == self.cursor else curses.A_DIM
            stdscr.addnstr(y, 0, f" {lines[i]}"[:w - 1], w - 1, attr)
        stdscr.addnstr(
            bot + 1, 0,
            f" {self.config_path}  line {self.cursor + 1}/{len(lines)}"
            "  [Enter/i=edit line  saves only if valid JSON]".ljust(w - 1),
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
        elif op == "html":
            _open_html_dashboard(self)
        elif op == "view":
            if args and args[0] in VIEWS:
                self.view, self.scroll = args[0], 0
            else:
                self._flash(f":view expects one of {', '.join(VIEWS)}")
        elif op in ("restart", "stop", "start"):
            if args:
                target = args[0].lower()
                a = next(
                    (a for a in self.sup.assets
                     if a.ticker == target or a.ticker.startswith(target)),
                    None)
            else:
                a = self._selected()
            if a is None:
                self._flash(f"unknown asset: {' '.join(args) or '(none selected)'}")
            elif op == "restart":
                self._force_restart(a)
                self._flash(f"restarting {a.ticker} with its latest config...")
            elif op == "stop":
                self._stop_asset(a)
                self._flash(f"stopping {a.ticker}...")
            elif a.proc and a.proc.poll() is None:
                self._flash(f"{a.ticker} already running")
            else:
                a.stopped_by_user = False
                a.spawn()
                self._flash(f"started {a.ticker}")
        elif op in ("pause", "resume", "flatten"):
            if args:
                target = args[0].lower()
                tickers = [a.ticker for a in self.sup.assets
                           if a.ticker == target or a.ticker.startswith(target)]
            else:
                a = self._selected()
                tickers = [a.ticker] if a else []
            if not tickers:
                self._flash(f"unknown asset: {' '.join(args)}")
                return
            if hasattr(self.sup, "send_control"):
                for t in tickers:
                    self.sup.send_control(t, op)
            self._flash(f"{op} -> {', '.join(tickers)}")
        else:
            self._flash(f"unknown command :{cmd} (try :q :w :html :view :stop <ticker>)")

    def handle_key(self, key: int) -> None:
        # ---- inline config editing mode -----------------------------------
        if self.editing:
            if key in (10, 13, curses.KEY_ENTER):
                self._commit_edit()
            elif key == 27:                                   # Esc cancels
                self.editing = False
                self._flash("edit cancelled")
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                self.edit_buf = self.edit_buf[:-1]
            elif 32 <= key < 127:
                self.edit_buf += chr(key)
            return

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
        # The cursor means different things per view: an asset row in
        # dashboard/trades, a FILE LINE in config. Clamping it to the asset
        # count in config view is what stopped the cursor at ~"mode": "paper".
        if self.view == "config":
            cursor_max = max(0, getattr(self, "_config_nlines", 1) - 1)
        else:
            cursor_max = n_assets - 1
        if key == ord("1"):
            self.view, self.scroll = "dashboard", 0
            self.cursor = min(self.cursor, max(0, len(self.sup.assets) - 1))
        elif key == ord("2"):
            self.view, self.scroll = "trades", 0
            self.cursor = min(self.cursor, max(0, len(self.sup.assets) - 1))
        elif key == ord("3"):
            self.view = "config"
            self.cursor = 0
            self.scroll = 0
        elif key in (9, curses.KEY_RIGHT, ord("l")):
            nxt = (VIEWS.index(self.view) + 1) % len(VIEWS)
            self.view, self.scroll = VIEWS[nxt], 0
        elif key in (curses.KEY_LEFT, ord("h")):
            prv = (VIEWS.index(self.view) - 1) % len(VIEWS)
            self.view, self.scroll = VIEWS[prv], 0
        elif key in (ord("j"), curses.KEY_DOWN):
            self.cursor = min(self.cursor + 1, cursor_max)
        elif key in (ord("k"), curses.KEY_UP):
            self.cursor = max(self.cursor - 1, 0)
        elif key == ord("g"):
            self.cursor = self.scroll = 0
        elif key == ord("G"):
            self.cursor = cursor_max
        elif key == 4:  # Ctrl-d
            step = max(1, getattr(self, "_config_win_h", 20) // 2)
            self.cursor = min(self.cursor + step, cursor_max)

        elif key == 21:  # Ctrl-u
            step = max(1, getattr(self, "_config_win_h", 20) // 2)
            self.cursor = max(self.cursor - step, 0)
        elif key == ord("r"):
            self.reload_config()
        elif key in (10, 13, curses.KEY_ENTER, ord("i")) and self.view == "config":
            self._begin_edit()
        elif key in (ord("R"), curses.KEY_F5):
            a = self._selected()
            if a:
                self._force_restart(a)
                self._flash(f"restarting {a.ticker} with its latest config...")
        elif key == ord("o"):                             # open HTML dashboard
            _open_html_dashboard(self)
        elif key == ord(":"):
            self.command, self.command_mode = "", True
        elif key in (ord("q"), ord("Q")):
            raise SystemExit(0)

    # ------------------------------------------------- inline config editing
    _LINE_RE = re.compile(r'^(\s*"[^"]+"\s*:\s*)(.*?)(,?\s*)$')

    def _begin_edit(self) -> None:
        try:
            lines = self.config_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            return
        idx = min(self.cursor, len(lines) - 1)
        if idx < 0:
            return
        line = lines[idx]
        # For `"key": value,` prefill ONLY the value token so the user edits
        # the parameter, not the JSON syntax. Other lines edit whole.
        m = self._LINE_RE.match(line)
        if m and m.group(2):
            self.edit_buf, self.edit_token = m.group(2), True
        else:
            self.edit_buf, self.edit_token = line, False
        self.edit_line = idx
        self.editing = True

    def _commit_edit(self) -> None:
        self.editing = False
        try:
            lines = self.config_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            self._flash(f"cannot read config: {exc}")
            return
        idx = self.edit_line
        if idx >= len(lines):
            self._flash("config changed on disk - edit cancelled")
            return
        orig = lines[idx]
        if self.edit_token:
            m = self._LINE_RE.match(orig)
            if m:
                val = self.edit_buf.strip() or m.group(2)
                new = m.group(1) + val + m.group(3)
            else:
                new = self.edit_buf
        else:
            new = self.edit_buf
        if new == orig:
            self._flash("no change")
            return
        lines[idx] = new
        text = "\n".join(lines) + "\n"
        # Validate BEFORE touching the real file: valid JSON AND a structurally
        # loadable portfolio config. Bad edits never reach the engine.
        tmp = self.config_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            load_portfolio_config(tmp)
        except Exception as exc:  # noqa: BLE001
            try:
                tmp.unlink()
            except OSError:
                pass
            self._flash(f"invalid config - NOT saved: {exc}")
            return
        tmp.replace(self.config_path)
        self._flash(f"saved line {idx + 1} - reloading config...")
        self.reload_config()

    # ---------------------------------------------------------------- main
    def run(self, stdscr) -> None:
        stdscr.nodelay(True)  # NEVER block on input — the TUI must redraw on
        # its own cadence even on Windows consoles where timeout()+getch() can
        # block indefinitely waiting for a key.
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        last_sample = 0.0
        while True:
            try:
                self.check_config_reload()
                now = time.monotonic()
                if now - last_sample >= REFRESH_SECS:
                    self._sample()
                    last_sample = now
                self._draw(stdscr)
            except curses.error:
                # Terminal was resized/misbehaved mid-draw — never let a
                # rendering error freeze or kill the dashboard.
                stdscr.clear()
                stdscr.refresh()
            except Exception:  # noqa: BLE001 - the TUI must never die silently
                stdscr.clear()
                stdscr.refresh()
            key = stdscr.getch()
            if key != -1:
                self.handle_key(key)
            time.sleep(0.05)  # ~20Hz redraw; getch() above never blocks


def _open_html_dashboard(dash) -> None:
    """Open the live HTML portfolio dashboard in the default browser."""
    import webbrowser

    path = getattr(dash.sup, "dashboard_path", None)
    if path is None:
        from portfolio_supervisor import DASHBOARD_PATH as path
    p = Path(path)
    if not p.exists():
        dash._flash(f"{p.name} not written yet (wait for the first monitor tick)")
        return
    webbrowser.open(p.resolve().as_uri())
    dash._flash(f"opened {p.name} in your browser")


def _compute_sortino(trade_rs) -> Optional[float]:
    """Trade-level Sortino ratio (non-annualized) from closed-trade R multiples.

    Returns None (displayed as N/A) when statistically undefined: fewer than
    2 closed trades, or no downside observations yet (dd = 0 is undefined,
    NOT zero performance).
    """
    rs = list(trade_rs)
    n = len(rs)
    if n < 2:
        return None
    mean = sum(rs) / n
    downside_mean = sum(min(0.0, r) for r in rs) / n
    dd_var = sum((min(0.0, r) - downside_mean) ** 2 for r in rs) / (n - 1)
    dd = dd_var ** 0.5
    if dd <= 0.0:
        return None
    return mean / dd


def _compute_sharpe(trade_rs) -> Optional[float]:
    """Trade-level Sharpe ratio (non-annualized) from closed-trade R multiples.

    Returns None (displayed as N/A) when statistically undefined.
    """
    rs = list(trade_rs)
    n = len(rs)
    if n < 2:
        return None
    mean = sum(rs) / n
    var = sum((r - mean) ** 2 for r in rs) / (n - 1)
    sd = var ** 0.5
    if sd <= 0.0:
        return None
    return mean / sd


def run_vim_dashboard(sup, config_path: str, ticker_configs: Optional[Dict[str, str]] = None) -> None:
    """Blocking curses entry point (the supervisor keeps running in a thread)."""
    try:
        import curses
    except ImportError as exc:
        if os.name == "nt":
            raise RuntimeError(
                "the VIM dashboard needs 'windows-curses' on Windows: "
                "pip install windows-curses (or run with --no-vim)"
            ) from exc
        raise
    curses.wrapper(VimDashboard(sup, config_path).run)
