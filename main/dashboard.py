"""
Live Dashboard Server
======================
Serves a TradingView-style web UI and streams live price action, model
decisions, and simulated trade executions from the QuantSimulator over a
WebSocket connection.

This module is purely observational. It never touches order placement or
the exchange connection — it only broadcasts events that executor.py /
execution.py push into it. Safe to add to a running dry-run bot.

Embedded usage (inside executor.py's asyncio loop):

    from main.dashboard import Dashboard

    self.dashboard = Dashboard(symbol=SYMBOL)      # in __init__
    await self.dashboard.start()                   # in initialize()

    self.dashboard.push_tick(price, timestamp_ms)
    self.dashboard.push_trade_enter(receipt)        # dict from enter_short/enter_long
    self.dashboard.push_trade_exit(result)          # dict from check_exit/_exit
    self.dashboard.push_status(status_dict)         # dict from get_status() (+ extras)
    self.dashboard.push_decision(decision_dict)     # per-second model decision

Standalone preview (no bot, no exchange, synthetic data):

    python -m main.dashboard_demo
    → open http://localhost:8000
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Deque, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

logger = logging.getLogger("dashboard")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Dashboard:
    """
    Owns a small FastAPI app (one HTML page + one WebSocket endpoint) and a
    replay buffer so any browser tab that connects — including one that
    connects mid-session — gets caught up immediately, then streams live.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        symbol: str = "BTC/USDT:USDT",
        candle_seconds: int = 5,
        candle_history: int = 3000,
        marker_history: int = 500,
        decision_history: int = 3000,
        equity_history: int = 3000,
    ):
        self.host = host
        self.port = port
        self.symbol = symbol
        self.candle_seconds = max(1, candle_seconds)

        self._clients: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None

        # Replay buffers
        self._candles: Deque[dict] = deque(maxlen=candle_history)
        self._current_candle: Optional[dict] = None
        self._markers: Deque[dict] = deque(maxlen=marker_history)
        self._decisions: Deque[dict] = deque(maxlen=decision_history)
        self._equity: Deque[dict] = deque(maxlen=equity_history)
        self._last_status: dict = {}
        # Static-ish strategy config (timeframe_barrier_seconds, rsi_filter_threshold,
        # etc.) pushed once from executor.py so the UI can label the RSI/vol panel
        # correctly instead of hardcoding numbers that drift from config.py.
        self._config: dict = {}

        self.app = self._build_app()

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------
    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/")
        async def index():
            path = os.path.join(STATIC_DIR, "index.html")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
            except FileNotFoundError:
                return HTMLResponse(
                    "<h1>dashboard static file missing</h1>"
                    f"<p>Expected {path}</p>",
                    status_code=500,
                )
            html = html.replace("__SYMBOL__", self.symbol)
            return HTMLResponse(html)

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)
            logger.info(f"Dashboard client connected ({len(self._clients)} total)")
            try:
                await websocket.send_text(json.dumps(self._snapshot(), default=str))
                while True:
                    # No client -> server protocol yet; just keep the socket
                    # open so we notice disconnects promptly.
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.debug(f"Dashboard client loop ended: {e}")
            finally:
                self._clients.discard(websocket)
                logger.info(f"Dashboard client disconnected ({len(self._clients)} total)")

        return app

    def _snapshot(self) -> dict:
        candles = list(self._candles)
        if self._current_candle is not None:
            candles = candles + [self._current_candle]
        return {
            "type": "snapshot",
            "symbol": self.symbol,
            "candle_seconds": self.candle_seconds,
            "candles": candles,
            "markers": list(self._markers),
            "decisions": list(self._decisions),
            "equity": list(self._equity),
            "status": self._last_status,
            "config": self._config,
        }

    # ------------------------------------------------------------------
    # Lifecycle — run inside the caller's asyncio loop
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        # give uvicorn a beat to bind the socket before we log the URL
        await asyncio.sleep(0.1)
        display_host = "localhost" if self.host in ("0.0.0.0", "127.0.0.1") else self.host
        logger.info(f"Dashboard live -> http://{display_host}:{self.port}")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            try:
                await self._server_task
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------
    def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        data = json.dumps(payload, default=str)
        for ws in list(self._clients):
            asyncio.ensure_future(self._safe_send(ws, data))

    async def _safe_send(self, ws: WebSocket, data: str) -> None:
        try:
            await ws.send_text(data)
        except Exception:
            self._clients.discard(ws)

    def push_threadsafe(self, fn_name: str, *args, **kwargs) -> None:
        """
        Call a push_* method from a non-asyncio thread. Not needed for the
        executor.py integration (everything there already runs on the
        dashboard's event loop) — provided for standalone scripts that
        generate data off-thread.
        """
        if self._loop is None:
            return
        fn = getattr(self, fn_name)
        self._loop.call_soon_threadsafe(lambda: fn(*args, **kwargs))

    # ------------------------------------------------------------------
    # Event ingestion — call these from executor.py / execution.py
    # ------------------------------------------------------------------
    def push_tick(self, price: float, ts_ms: Optional[int] = None) -> None:
        if not price or price <= 0:
            return
        ts_ms = ts_ms or int(time.time() * 1000)
        bucket = (ts_ms // 1000 // self.candle_seconds) * self.candle_seconds

        c = self._current_candle
        if c is None or c["t"] != bucket:
            if c is not None:
                self._candles.append(c)
            c = {"t": bucket, "o": price, "h": price, "l": price, "c": price}
            self._current_candle = c
        else:
            c["h"] = max(c["h"], price)
            c["l"] = min(c["l"], price)
            c["c"] = price

        self._broadcast({"type": "tick", "price": price, "t": ts_ms, "candle": dict(c)})

    def _chart_time(self) -> int:
        """
        Epoch-seconds anchor for placing a marker on the chart. Derived from
        the in-progress candle bucket (unambiguous, timezone-free) rather
        than parsing a naive local ISO timestamp in the browser — that would
        drift if the server and the viewing browser are in different
        timezones.
        """
        if self._current_candle is not None:
            return int(self._current_candle["t"])
        return int(time.time() // self.candle_seconds * self.candle_seconds)

    def push_trade_enter(self, receipt: Dict) -> None:
        """receipt: the dict returned by PositionTracker.enter_short/enter_long."""
        marker = {
            "kind": "enter",
            "side": receipt.get("side"),
            "price": receipt.get("fill_price"),
            "mid_price": receipt.get("mid_price"),
            "stop": receipt.get("stop_barrier"),
            "tp": receipt.get("profit_barrier"),
            "qty": receipt.get("quantity"),
            "notional": receipt.get("notional_usd"),
            "barrier_pct": receipt.get("barrier_pct"),
            "lookahead_seconds": receipt.get("lookahead_seconds"),
            "rsi_2": receipt.get("rsi_2"),
            "p_win": receipt.get("p_win"),
            "time": receipt.get("timestamp"),
            "chart_time": self._chart_time(),
        }
        self._markers.append(marker)
        self._broadcast({"type": "trade_enter", "marker": marker})

    def push_trade_exit(self, result: Dict) -> None:
        """result: the dict returned by PositionTracker.check_exit/_exit."""
        marker = {
            "kind": "exit",
            "side": result.get("side"),
            "price": result.get("exit_price"),
            "reason": result.get("reason"),
            "net_pnl_usd": result.get("net_pnl_usd"),
            "gross_pnl_usd": result.get("gross_pnl_usd"),
            "r_multiple": result.get("r_multiple"),
            "duration_seconds": result.get("duration_seconds"),
            "time": result.get("exit_time"),
            "chart_time": self._chart_time(),
        }
        self._markers.append(marker)

        eq_point = {
            "n": len(self._equity) + 1,
            "net_r": result.get("r_multiple", 0.0),
            "gross_r": (result.get("gross_pnl_pct", 0.0) / result["barrier_pct"]
                        if result.get("barrier_pct") else 0.0),
        }
        self._equity.append(eq_point)

        self._broadcast({"type": "trade_exit", "marker": marker, "equity": eq_point})

    def push_config(self, config: Dict) -> None:
        """Static strategy config the UI needs to label things correctly, e.g.
        {timeframe_barrier_seconds, rsi_filter_threshold, confidence_threshold,
        lookahead_seconds, trade_direction, candle_seconds}. Call once at
        startup (or again if it ever changes) — cheap, retained for snapshot."""
        self._config.update(config)
        self._broadcast({"type": "config", "config": self._config})

    def push_status(self, status: Dict) -> None:
        """status: get_status() from QuantSimulator, optionally extended with
        side / stop_barrier / profit_barrier / unrealized_pnl_usd / mode / etc."""
        self._last_status = status
        self._broadcast({"type": "status", "status": status})

    def push_decision(self, decision: Dict) -> None:
        """decision: {time, sec, close, rsi_2, p_win, action, bar_vol_std?}.
        Stamped with chart_time (same aggregated-candle bucket used for trade
        markers) so the RSI/vol panel can share the price chart's time axis."""
        decision.setdefault("chart_time", self._chart_time())
        self._decisions.append(decision)
        self._broadcast({"type": "decision", "decision": decision})