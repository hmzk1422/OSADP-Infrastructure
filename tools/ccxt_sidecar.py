"""
CCXT Data Sidecar for the Rust Executor
========================================
Streams live trades + orderbook from a CCXT exchange to the Rust executor
over a local TCP socket (newline-delimited JSON), and seeds historical
klines for indicator warm-up before live trading begins.

Role split:
    - This Python process owns ALL exchange interaction (CCXT is Python-
      only and does not exist in Rust).
    - Rust owns feature computation, ONNX model inference, and ALL trading
      decisions.

Protocol (one JSON object per line, UTF-8):

    {"type":"seed_done","bars":N}                        -> historical warm-up complete
    {"type":"trade","ts_ms":1723700000123,"price":61234.5,"amount":0.0021,"side":"buy"}
    {"type":"orderbook","ts_ms":1723700000500,"bids":[[p,s],...],"asks":[[p,s],...]}

The Rust side connects to this process as a TCP *client*; this server
listens the way a data feed would.  Set SIDECAR_HOST/SIDECAR_PORT to
match Rust's config.  The exchange stays entirely CCXT-unified.

Usage:
    python tools/ccxt_sidecar.py [--symbol BTC/USDT:USDT] [--port 9100]
"""

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp
import ccxt.async_support as ccxt  # async CCXT (ccxt.pro is a superset)

try:
    import ccxt.pro as ccxtpro
    HAS_PRO = True
except Exception:  # pragma: no cover
    ccxtpro = None
    HAS_PRO = False

# ---------------------------------------------------------------------------
# Config (overridable via CLI / env)
# ---------------------------------------------------------------------------
DEFAULT_HOST = os.environ.get("SIDECAR_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SIDECAR_PORT", "9100"))


# ===========================================================================
# Minimal orderbook store: keeps top-N levels in a dict keyed by price.
# (ccxt.pro's watchOrderBook returns a full snapshot; we forward it as-is.)
# ===========================================================================
class OrderBookStore:
    def __init__(self, max_levels: int = 25):
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._max = max_levels
        self._ts_ms: int = 0

    def _top(self, levels, reverse: bool) -> list:
        levels.sort(key=lambda kv: kv[0], reverse=reverse)
        return levels[: self._max]

    def update(self, bids, asks):
        for price, size in bids or []:
            p, s = float(price), float(size)
            if s <= 0:
                self._bids.pop(p, None)
            else:
                self._bids[p] = s
        for price, size in asks or []:
            p, s = float(price), float(size)
            if s <= 0:
                self._asks.pop(p, None)
            else:
                self._asks[p] = s

    def snapshot(self) -> dict:
        return {
            "bids": self._top(list(self._bids.items()), reverse=True),
            "asks": self._top(list(self._asks.items()), reverse=False),
        }
# ===========================================================================
# Sidecar server
# ===========================================================================
class CcxtSidecar:
    def __init__(self, symbol: str, host: str, port: int, api_key: str = "",
                 api_secret: str = "", testnet: bool = False,
                 max_orderbook_levels: int = 25, seed_bars: int = 500,
                 seed_interval: str = "5m"):
        self.symbol = symbol
        self.host = host
        self.port = port
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.max_levels = max_orderbook_levels
        self.seed_bars = seed_bars
        self.seed_interval = seed_interval

        self._book = OrderBookStore(max_levels)
        self._exchange: ccxt.Exchange | None = None
        self._pro_exchange = None
        self._subscribers: list[asyncio.StreamWriter] = []
        self._server: asyncio.Server | None = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------
    async def _make_exchange(self):
        options = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "adjustForTimeDifference": True,
            },
        }
        if self.api_key and "..." not in self.api_key:
            options["apiKey"] = self.api_key
            options["secret"] = self.api_secret

        if HAS_PRO:
            self._pro_exchange = ccxtpro.bybit(options)
            connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
            self._pro_exchange.session = aiohttp.ClientSession(connector=connector)
            self._exchange = self._pro_exchange
        else:
            self._exchange = ccxt.bybit(options)

        if self.testnet:
            self._exchange.set_sandbox_mode(True)

    async def _close_exchange(self):
        for w in self._subscribers:
            try:
                w.close()
            except Exception:
                pass
        self._subscribers.clear()
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Broadcast / socket management
    # ------------------------------------------------------------------
    async def _handle_client(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter):
        self._subscribers.append(writer)
        peer = writer.get_extra_info("peername")
        print(f"[sidecar] Rust client connected: {peer}")
        try:
            # Consume anything the client sends (vestigial) so the socket
            # isn't stuck waiting — client is a single push direction.
            async for _ in reader:
                pass
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            if writer in self._subscribers:
                self._subscribers.remove(writer)
            try:
                writer.close()
            except Exception:
                pass
            print(f"[sidecar] Rust client disconnected: {peer}")

    def _broadcast(self, msg: dict):
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        dead = []
        for w in self._subscribers:
            try:
                w.write(line)
            except Exception:
                dead.append(w)
        for w in dead:
            if w in self._subscribers:
                self._subscribers.remove(w)

    # ------------------------------------------------------------------
    # Historical seed
    # ------------------------------------------------------------------
    async def _seed_historical(self):
        """Fetch OHLCV bars and forward them to warm up rolling indicators.

        The Rust side replays these closes through its bar-close series
        exactly like LiveFeatureEngine.seed_historical_data() does, so RSI
        / SMA / MA-slope buffers are warm before the first live tick.
        """
        print(f"[sidecar] Seeding {self.seed_bars} x {self.seed_interval} bars "
              f"for {self.symbol} ...")
        try:
            ohlcv = await self._exchange.fetch_ohlcv(self.symbol,
                                                     timeframe=self.seed_interval,
                                                     limit=self.seed_bars)
        except Exception as e:
            print(f"[sidecar] seed fetch failed: {e} — continuing without warm-up")
            self._broadcast({"type": "seed_done", "bars": 0})
            return

        closes = [float(row[4]) for row in ohlcv]
        self._broadcast({
            "type": "seed",
            "bars": len(closes),
            "closes": closes,
            "timestamps_ms": [int(row[0]) for row in ohlcv],
        })
        print(f"[sidecar] Seeded {len(closes)} closes. "
              f"Latest close: {closes[-1] if closes else 'n/a'}")
        self._broadcast({"type": "seed_done", "bars": len(closes)})
# ------------------------------------------------------------------
    # Live streams
    # ------------------------------------------------------------------
    async def _stream_trades(self, exchange):
        while True:
            try:
                trades = await exchange.watch_trades(self.symbol)
                for t in trades:
                    self._broadcast({
                        "type": "trade",
                        "ts_ms": int(t.get("timestamp") or 0),
                        "price": float(t["price"]),
                        "amount": float(t["amount"]),
                        "side": t.get("side", "buy"),
                    })
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[sidecar] watch_trades error: {e} — reconnecting ...")
                await asyncio.sleep(2)

    async def _stream_orderbook(self, exchange):
        while True:
            try:
                ob = await exchange.watch_order_book(self.symbol)
                self._book.update(ob.get("bids", []), ob.get("asks", []))
                snap = self._book.snapshot()
                self._broadcast({
                    "type": "orderbook",
                    "ts_ms": int(ob.get("timestamp") or 0),
                    "bids": snap["bids"],
                    "asks": snap["asks"],
                })
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[sidecar] watch_order_book error: {e} — reconnecting ...")
                await asyncio.sleep(2)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    async def run(self):
        await self._make_exchange()

        # 1. TCP server first so Rust client can attach while we warm up
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        print(f"[sidecar] Listening on tcp://{self.host}:{self.port}")

        # 2. Load markets once
        try:
            await self._exchange.load_markets()
        except Exception as e:
            print(f"[sidecar] load_markets failed: {e}")
        self._exchange.required_credentials_check = False  # public data only

        # 3. Send an initial identity banner so Rust can validate symbol match
        self._broadcast({
            "type": "hello",
            "symbol": self.symbol,
            "exchange": self._exchange.name,
            "has_pro": HAS_PRO,
        })

        # 4. Historical warm-up BEFORE live streams
        await self._seed_historical()

        # 5. Start live trade + orderbook streams concurrently
        tasks = [
            asyncio.create_task(self._stream_trades(self._exchange)),
            asyncio.create_task(self._stream_orderbook(self._exchange)),
        ]

        try:
            # Keep server alive; watch for client drops to re-send "hello"
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._close_exchange()
            if self._server:
                self._server.close()
                await self._server.wait_closed()


def main():
    ap = argparse.ArgumentParser(description="CCXT data sidecar for Rust executor")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-secret", default="")
    ap.add_argument("--testnet", action="store_true")
    ap.add_argument("--seed-bars", type=int, default=500)
    ap.add_argument("--seed-interval", default="5m",
                    help="CCXT timeframe used for warm-up (matches TIMEFRAME_BARRIER)")
    args = ap.parse_args()

    if sys.platform == "win32":
        # ccxt.pro / aiodns need the selector loop policy on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sidecar = CcxtSidecar(
        symbol=args.symbol,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        api_secret=args.api_secret,
        testnet=args.testnet,
        seed_bars=args.seed_bars,
        seed_interval=args.seed_interval,
    )

    try:
        asyncio.run(sidecar.run())
    except KeyboardInterrupt:
        print("[sidecar] interrupted.")
    finally:
        time.sleep(0.1)


if __name__ == "__main__":
    main()