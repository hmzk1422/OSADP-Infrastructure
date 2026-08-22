#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSADP-Core — Unified Runner
===========================

Single-point orchestration for the hybrid Python/Rust quant stack.

The Rust core (``unified-execution/``) is compiled into a native Python
extension with **maturin**. The compiled module is imported as ``engine`` and
exposes the ``engine.Strategy`` class. Each **asset** (ticker + venue) is
represented by one :class:`Strategy` instance, and a :class:`Portfolio` holds
many of them so you can run 50+ markets concurrently from a single process.

Concurrency model
-----------------
Every :meth:`Strategy.run` spawns a dedicated tokio runtime on its own OS
thread **inside the Rust process**. Market-data ingestion, 1-second micro-bar
aggregation, 15-feature LightGBM scoring and dry-run fill simulation all happen
in Rust at full speed — Python is only the control plane and never touches
market data. The Python `Strategy` simply forwards ``run``/``pause``/
``resume``/``stop``/``reallocate``/``status`` to the native object.

Stack map
---------
    research/ml_state_optimized.py   research backtest (feature engineering,
                                     walk-forward evaluation, model training)
    main/                            Python live / dry-run application packages
    unified-execution/engine/        Rust strategy engine
    unified-execution/live-trade/    real-market execution (Bybit/Bitfinex WS +
                                     RocksDB persistence)
    unified-execution/live-test/     historical replay (paper-trading) runner
    unified-execution/bindings/      PyO3 bindings -> the ``engine`` module
    scripts/ingestion/               Binance caching, download, CCXT conversion
    build/                           HTML reports, CSV trade logs
    docs/                            dev log + trade journal

CLI
---
    python unified_runner.py backtest
    python unified_runner.py live-trade --tickers btcusdt ethusdt solusdt \
        --exchange bitfinex --mode paper --capital 5000
    python unified_runner.py live-test --ticker btcusdt --release
    python unified_runner.py live-test --ticker btcusdt --limit 200000 --no-db
    python unified_runner.py live-test --ticker btcusdt --start-ms 1709344800000 \
        --end-ms 1709431200000 --release
    python unified_runner.py doctor [--require-native]
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent

# Must match `[tool.maturin] module-name` in pyproject.toml.
NATIVE_MODULE = "engine"

# Research backtest entry point (pure-Python research pipeline).
RESEARCH_BACKTEST = REPO_ROOT / "research" / "ml_state_optimized.py"

# Cargo binary/crate names (one per workspace manifest `name`).
LIVE_TEST_BIN = "live-test"

DEFAULT_TICKERS = ("btcusdt",)
DEFAULT_EXCHANGE = "bitfinex"
DEFAULT_CAPITAL_USD = 10_000.0
SUPPORTED_EXCHANGES = ("bybit", "bitfinex")
SUPPORTED_MODES = ("paper", "live")

logger = logging.getLogger("unified_runner")


# ─────────────────────────────────────────────────────────────────────────────
# Native (maturin) module
# ─────────────────────────────────────────────────────────────────────────────
def load_native_module() -> Optional[Any]:
    """Return the maturin-compiled Rust extension (``engine``), or ``None``.

    The ``engine`` module is produced by ``maturin develop --release`` (see
    ``pyproject.toml``). When it is not importable the caller should inform the
    user to build it before attempting live trading.
    """
    try:
        return importlib.import_module(NATIVE_MODULE)
    except ImportError as exc:
        logger.warning("native module %r not importable: %s", NATIVE_MODULE, exc)
        logger.warning("build it with `maturin develop --release` (see pyproject.toml)")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy — one Rust execution loop per asset
# ─────────────────────────────────────────────────────────────────────────────
class Strategy:
    """Control-plane handle to one Rust strategy runner (one ticker/venue).

    Wraps the native ``engine.Strategy``. A dedicated tokio thread inside the
    Rust process runs market-data ingestion, 1-second micro-bar aggregation,
    15-feature scoring and dry-run fill simulation; this object only issues
    commands::

        s = Strategy("btcusdt", exchange="bitfinex", mode="paper", capital=5_000)
        s.run()              # boots the Rust loop on its own thread
        s.reallocate(10_000) # resize position sizing (live: applies next entry)
        s.pause()            # stop generating entries/exits
        s.resume()           # resume after pause()
        s.status()           # -> dict snapshot (equity, trades, ...)
        s.is_running         # bool property
        s.stop()             # graceful shutdown
    """

    def __init__(
        self,
        ticker: str,
        *,
        exchange: str = DEFAULT_EXCHANGE,
        mode: str = "paper",
        capital: float = DEFAULT_CAPITAL_USD,
        db_path: Optional[str] = None,
    ) -> None:
        native = load_native_module()
        if native is None:
            raise RuntimeError(
                "native 'engine' module is not built — run "
                "`maturin develop --release` from the repo root first"
            )
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(f"unknown exchange {exchange!r} (expected {SUPPORTED_EXCHANGES})")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"unknown mode {mode!r} (expected {SUPPORTED_MODES})")
        if capital <= 0:
            raise ValueError("capital must be > 0")
        self.ticker = ticker.lower()
        self._exchange = exchange
        self._mode = mode
        self._rust = native.Strategy(
            ticker, exchange=exchange, mode=mode, capital=capital, db_path=db_path
        )

    @property
    def exchange(self) -> str:
        return self._exchange

    @property
    def mode(self) -> str:
        return self._mode

    def run(self) -> None:
        """Boot the Rust execution loop (idempotent; ignores if already running)."""
        if self.is_running:
            logger.info("strategy %s already running", self.ticker)
            return
        self._rust.run()

    def pause(self) -> None:
        """Pause signal consumption: raw data keeps persisting, no new trades."""
        self._rust.pause()

    def resume(self) -> None:
        """Resume after :meth:`pause`."""
        self._rust.resume()

    def stop(self) -> None:
        """Request a graceful shutdown.

        Blocks until the Rust runner thread has fully drained and its RocksDB
        store has been flushed + closed, so the DB is never left in a
        corrupted state when the process exits.
        """
        self._rust.stop()

    def reallocate(self, capital_usd: float) -> None:
        """Resize the capital allocated to this strategy (applies to the next entry)."""
        if capital_usd <= 0:
            raise ValueError("capital must be > 0")
        self._rust.reallocate(capital_usd)

    @property
    def is_running(self) -> bool:
        return self._rust.is_running

    def status(self) -> Dict[str, Any]:
        """Latest runtime snapshot (equity, trades, wins/losses, R-multiple, ...)."""
        return json.loads(self._rust.status())


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio — orchestrate many Strategy instances (50+ assets)
# ─────────────────────────────────────────────────────────────────────────────
class Portfolio:
    """A collection of :class:`Strategy` instances (one per ticker/venue).

    Each strategy owns its own tokio runtime thread in the Rust process, so
    Python only coordinates ``run``/``pause``/``resume``/``stop``/
    ``reallocate``/``status`` across arbitrarily many markets::

        pf = Portfolio()
        pf.add("btcusdt", capital=5_000)
        pf.add("ethusdt", capital=3_000)
        pf.run()
        pf.reallocate("btcusdt", 7_500)
        print(pf.status())
        pf.stop()
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Strategy] = {}

    def __contains__(self, ticker: str) -> bool:
        return ticker.lower() in self._strategies

    def __getitem__(self, ticker: str) -> Strategy:
        return self._strategies[ticker.lower()]

    def __iter__(self) -> Iterator[Strategy]:
        return iter(self._strategies.values())

    def __len__(self) -> int:
        return len(self._strategies)

    def add(
        self,
        ticker: str,
        *,
        exchange: str = DEFAULT_EXCHANGE,
        mode: str = "paper",
        capital: float = DEFAULT_CAPITAL_USD,
        db_path: Optional[str] = None,
    ) -> Strategy:
        key = ticker.lower()
        if key in self._strategies:
            raise ValueError(f"strategy for {key!r} already added")
        s = Strategy(
            ticker, exchange=exchange, mode=mode, capital=capital, db_path=db_path
        )
        self._strategies[key] = s
        return s

    def run(self) -> None:
        for s in self._strategies.values():
            s.run()

    def pause(self) -> None:
        for s in self._strategies.values():
            s.pause()

    def resume(self) -> None:
        for s in self._strategies.values():
            s.resume()

    def stop(self) -> None:
        for s in self._strategies.values():
            s.stop()

    def reallocate(self, ticker: str, capital_usd: float) -> None:
        """Reallocate capital for a single ticker."""
        self[ticker].reallocate(capital_usd)

    def reallocate_all(self, capital_usd: float) -> None:
        """Reallocate the same capital to every strategy in the portfolio."""
        for s in self._strategies.values():
            s.reallocate(capital_usd)

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate status snapshot keyed by ticker."""
        return {ticker: s.status() for ticker, s in self._strategies.items()}

# ─────────────────────────────────────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────────────────────────────────────
def cmd_backtest(args: argparse.Namespace) -> int:
    """Run the research backtest (research/ml_state_optimized.py)."""
    if not RESEARCH_BACKTEST.exists():
        print(f"research backtest not found: {RESEARCH_BACKTEST}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(RESEARCH_BACKTEST)])


def cmd_live_trade(args: argparse.Namespace) -> int:
    """Run one in-process Rust Strategy per ticker via a Portfolio."""
    if load_native_module() is None:
        print("ERROR: native 'engine' module is not built.", file=sys.stderr)
        print("       Run `maturin develop --release` from the repo root, then retry.", file=sys.stderr)
        return 1

    pf = Portfolio()
    state = {"stopped": False}

    def _shutdown(signum, _frame):
        sig = signal.Signals(signum).name
        if not state["stopped"]:
            state["stopped"] = True
            logger.info(
                "%s received — stopping all strategies (flush + close RocksDB)", sig
            )
        else:
            logger.warning(
                "%s received again — already stopping (this may take a few seconds)", sig
            )

    # Safety net: whichever path the process exits on (signal, KeyboardInterrupt,
    # exception, sys.exit), make sure every native runner thread is joined and
    # each RocksDB store is flushed + closed before the interpreter tears down.
    # Idempotent and safe to call even when nothing is running, so register it
    # before any strategy is spawned. This is what prevents the "CURRENT file
    # corrupted" failure after an unclean quit.
    def _stop_all() -> None:
        try:
            pf.stop()
        except Exception as exc:  # noqa: BLE001 — never mask the original error
            logger.warning("error while stopping strategies: %s", exc)

    atexit.register(_stop_all)

    for ticker in args.tickers:
        try:
            pf.add(ticker, exchange=args.exchange, mode=args.mode, capital=args.capital)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, OSError):
        pass  # SIGTERM not available on this platform (e.g. Windows)

    pf.run()
    logger.info(
        "running %d strategy/strategies on %s (%s) — Ctrl-C to stop",
        len(pf), args.exchange, args.mode,
    )

    try:
        while not state["stopped"] and any(s.is_running for s in pf):
            # High-frequency control-plane sampling: the Rust loop ingests every
            # websocket frame in real time — the sleep below only sets how often
            # Python prints a status line, NOT how fresh the data is. Use a
            # sub-second cadence and surface the millisecond latency counters so
            # feed staleness is visible end-to-end (arbitrage diagnostics).
            time.sleep(0.5)
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            for ticker, st in pf.status().items():
                mid = f'{st["mid"]:.2f}' if st["mid"] else "—"
                stal = st.get("feed_staleness_ms")
                lat = st.get("last_event_latency_ms")
                sk = st.get("clock_skew_ms")
                logger.info(
                    "  %-10s %s  eq=%9.2f tr=%d r=%+.2f mid=%s stal=%7.1fms lat=%6.2fms sk=%+8.1fms",
                    ticker, stamp,
                    st["equity"], st["n_trades"], st["total_r"], mid,
                    stal if isinstance(stal, (int, float)) else -1.0,
                    lat if isinstance(lat, (int, float)) else -1.0,
                    sk if isinstance(sk, (int, float)) else 0.0,
                )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("stopping strategies ...")
        _stop_all()
    logger.info("done — database flushed and closed cleanly")
    return 0

def _build_live_test_feed(
    ticker: str,
    *,
    feed_dir: Path,
    out_path: Path,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: Optional[int] = None,
) -> Optional[str]:
    """Convert the research parquet/csv trades + orderbook into the
    timestamp-ordered replay stream consumed by the Rust `live-test` binary.

    The trades and orderbook are loaded (parquet preferred, csv fallback),
    merged into one event table, sorted by (timestamp, ordinal) — so at equal
    milliseconds trades come before book tops, deterministically — and written
    one event per line::

        #TICKER btcusdt
        T <ts_ms> <price> <amount> <buy|sell> <id>
        B <ts_ms> <bid_px> <bid_amt> <ask_px> <ask_amt>

    Returns ``None`` on success, otherwise an error message string.
    """
        # This builder MUST stay memory-bounded: BTCUSDT's research parquet is 223M
    # trades, and `pl.read_parquet` on the whole frame is exactly what OOM-crashed
    # at ~3.5 GB. Instead each source is streamed in column-projected chunks
    # (parquet via pyarrow `iter_batches`, CSV via pandas `chunksize`) and the two
    # timestamp-ordered streams are merged on the fly — the same chunked-feed idea
    # `ml_state_optimized` uses, but the merge is into ONE continuous stream with
    # no per-chunk re-initialisation. That means the live engine's indicators stay
    # warm across every chunk boundary: nothing is duplicated, ordering is global,
    # and unlike research's per-chunk *feature recompute* no warm-up buffer (or
    # recursive-feature drift such as VPIN) is introduced.
    try:
        import pyarrow.parquet as pq  # streaming, column-projected parquet read
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        return f"pyarrow + pandas are required to build the replay feed: {exc}"

    sym = ticker.strip().upper()
    tq, tc = feed_dir / f"{sym}_trades.parquet", feed_dir / f"{sym}_trades.csv"
    bq, bc = feed_dir / f"{sym}_orderbook.parquet", feed_dir / f"{sym}_orderbook.csv"
    trades_src = tq if tq.exists() else (tc if tc.exists() else None)
    book_src = bq if bq.exists() else (bc if bc.exists() else None)
    if trades_src is None:
        return f"trades feed not found for {sym} (tried {tq} and {tc})"
    if book_src is None:
        logger.warning("orderbook feed missing for %s — replaying trades only", sym)

    CHUNK_ROWS = 500_000
    FLUSH_EVERY = 100_000
    BOOK_IN_MEM_LIMIT = 5_000_000
    TRADE_COLS = ["timestamp", "price", "amount", "side", "id"]
    BOOK_COLS = ["timestamp", "bid_price", "bid_amount", "ask_price", "ask_amount"]

    def _is_parquet(src):
        return src.suffix == ".parquet"

    def _iter_chunks(src, cols):
        """Yield `cols` (projected) as timestamp-ordered pandas chunks, windowed
        to [start_ms, end_ms]. Each chunk holds ~CHUNK_ROWS rows, so peak memory
        is a single row group / CSV chunk -- never the whole file."""
        if _is_parquet(src):
            for batch in pq.ParquetFile(src).iter_batches(
                batch_size=CHUNK_ROWS, columns=cols
            ):
                df = batch.to_pandas()
                if start_ms is not None:
                    df = df[df["timestamp"] >= start_ms]
                if end_ms is not None:
                    df = df[df["timestamp"] <= end_ms]
                if not df.empty:
                    yield df
        else:
            for chunk in pd.read_csv(src, chunksize=CHUNK_ROWS):
                if any(c not in chunk.columns for c in cols):
                    return
                chunk = chunk[cols].copy()
                chunk["timestamp"] = chunk["timestamp"].astype("int64")
                if start_ms is not None:
                    chunk = chunk[chunk["timestamp"] >= start_ms]
                if end_ms is not None:
                    chunk = chunk[chunk["timestamp"] <= end_ms]
                if not chunk.empty:
                    yield chunk

    def _format_lines(df, kind):
        """Vectorized one-event-per-line strings (each with trailing "\n"), so
        the merge writes straight through with `"".join` -- no per-row Python."""
        ts = df["timestamp"].astype("int64").astype(str)
        if kind == "T":
            s = ("T " + ts + " " + df["price"].astype(str) + " "
                 + df["amount"].astype(str) + " " + df["side"].astype(str) + " "
                 + df["id"].astype(str) + "\n")
        else:
            s = ("B " + ts + " " + df["bid_price"].astype(str) + " "
                 + df["bid_amount"].astype(str) + " " + df["ask_price"].astype(str) + " "
                 + df["ask_amount"].astype(str) + "\n")
        return s.tolist()
        # The orderbook is sparse (~1/s) vs trades (thousands/s), so buffer the
    # orderbook once and stream trades -- bounding RAM to the orderbook side
    # plus one trades chunk. Same chunked-feed idea as `ml_state_optimized`,
    # but a single continuous merge means the engine's indicators never reset
    # (no warm-up buffer, no recursive-feature drift like VPIN).
    book = []  # (ts_ms, line), ascending
    n_ob = 0
    if book_src is not None:
        if _is_parquet(book_src):
            try:
                meta = pq.ParquetFile(book_src).metadata
                n_bk = getattr(meta, "num_rows", None)
                if n_bk is None:  # older pyarrow exposes row_groups only
                    n_bk = sum(rg.num_rows for rg in meta.row_groups)
            except Exception:
                n_bk = None
            if n_bk is not None and n_bk > BOOK_IN_MEM_LIMIT:
                return (
                    f"orderbook feed for {sym} ({n_bk} rows) is too large to hold "
                    "in memory for the merge; pass --start-ms/--end-ms/--limit to bound it"
                )
        for df in _iter_chunks(book_src, BOOK_COLS):
            ts_list = df["timestamp"].astype("int64").tolist()
            for t, ln in zip(ts_list, _format_lines(df, "B")):
                book.append((t, ln))
        book.sort(key=lambda x: x[0])  # cheap defensive sort (book is sparse)
        n_ob = len(book)
        # book_src == None is already warned about in the detection block above

    # --- streaming two-pointer merge (trades win ties at equal ms) ----------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    n_tr = 0
    bi = 0
    buf = []
    last_tr_ts = None  # monotonicity guard for the streamed trades feed

    def _drain():
        if len(buf) >= FLUSH_EVERY:
            out.write("".join(buf))
            buf.clear()

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"#TICKER {ticker.strip().lower()}\n")
        stop = False
        for df in _iter_chunks(trades_src, TRADE_COLS):
            ts_list = df["timestamp"].astype("int64").tolist()
            t_lines = _format_lines(df, "T")
            for t, ln in zip(ts_list, t_lines):
                if last_tr_ts is not None and t < last_tr_ts:
                    return (
                        f"trades feed for {sym} is not sorted by timestamp "
                        f"(saw ts={t} after ts={last_tr_ts}); sort the feed or "
                        "restrict with --start-ms/--end-ms"
                    )
                last_tr_ts = t
                # Strict `<` => trades win ties at equal ms (matches _ord order).
                while bi < len(book) and book[bi][0] < t:
                    buf.append(book[bi][1]); written += 1; bi += 1
                    _drain()
                    if limit is not None and written >= limit:
                        stop = True; break
                if stop:
                    break
                buf.append(ln); written += 1; n_tr += 1
                _drain()
                if limit is not None and written >= limit:
                    stop = True; break
            _drain()
            if stop:
                break
        # Remaining book tops (all >= the last trade's timestamp).
        if not stop:
            while bi < len(book):
                buf.append(book[bi][1]); written += 1; bi += 1
                _drain()
                if limit is not None and written >= limit:
                    break
        if buf:
            out.write("".join(buf))
            buf.clear()

    logger.info("replay feed written: %s (%d events, %d trades, %d book tops)",
                out_path, written, n_tr, n_ob)
    return None


def cmd_live_test(args: argparse.Namespace) -> int:
    """Build the parquet replay feed (unless --feed is given) and replay it
    through the Rust `live-test` binary — the same execution loop as live
    trading, fed from historical data instead of a live WebSocket."""
    if shutil.which("cargo") is None:
        print("cargo not found on PATH", file=sys.stderr)
        return 1

    # 1. Produce (or reuse) the replay stream from the research parquet files.
    if args.feed:
        feed_path = Path(args.feed)
        if not feed_path.exists():
            print(f"live-test feed not found: {feed_path}", file=sys.stderr)
            return 1
    else:
        feed_dir = REPO_ROOT / args.feed_dir
        feed_path = REPO_ROOT / "build" / f"live_test_feed_{args.ticker.lower()}.replay"
        err = _build_live_test_feed(
            args.ticker,
            feed_dir=feed_dir,
            out_path=feed_path,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            limit=args.limit,
        )
        if err is not None:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1

    # 2. Kick off the replay through the identical live-trade execution loop.
    cmd = ["cargo", "run", "-p", LIVE_TEST_BIN]
    if args.release:
        cmd.append("--release")
    cmd += ["--", args.ticker.lower(), "--feed", str(feed_path)]
    if args.capital is not None:
        cmd += ["--capital", str(args.capital)]
    if args.no_db:
        cmd += ["--no-db"]
    if args.db:
        cmd += ["--db", args.db]
    if args.seed:
        cmd += ["--seed", str(args.seed)]
    if args.loop:
        cmd += ["--loop"]
    if args.start_ms is not None:
        cmd += ["--start-ms", str(args.start_ms)]
    if args.end_ms is not None:
        cmd += ["--end-ms", str(args.end_ms)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    logger.info("running historical replay: %s", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_doctor(args: argparse.Namespace) -> int:
    """Validate the environment, repo layout and native binding."""
    ok = True

    def _check(label: str, cond: bool) -> bool:
        nonlocal ok
        ok = ok and cond
        print(f"[{'OK' if cond else 'FAIL'}] {label}")
        return cond

    _check("cargo available", shutil.which("cargo") is not None)
    _check("maturin available", shutil.which("maturin") is not None)
    _check("research backtest script", RESEARCH_BACKTEST.exists())

    native = load_native_module()
    if args.require_native:
        _check(f"native module '{NATIVE_MODULE}'", native is not None)
        if native is not None:
            _check("native class 'Strategy'", hasattr(native, "Strategy"))
            try:
                s = native.Strategy("btcusdt", exchange="bitfinex", mode="paper", capital=1.0)
                _check("Strategy() instantiation", s is not None)
                _check("Strategy.is_running", isinstance(s.is_running, bool))
                _check("Strategy.status() returns JSON str", isinstance(s.status(), str))
                st = json.loads(s.status())
                _check(
                    "status() carries equity/n_trades",
                    "equity" in st and "n_trades" in st,
                )
            except Exception as exc:  # noqa: BLE001
                _check("Strategy smoke test", False)
                print(f"       {exc}")
    else:
        present = native is not None
        print(
            f"[info] native module '{NATIVE_MODULE}' "
            f"{'present' if present else 'not built — run `maturin develop --release`'}"
        )

    print("OK." if ok else "FAILURES PRESENT.")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified_runner",
        description="OSADP-Core unified runner — backtest, paper-replay and live-trade "
        "orchestration for the hybrid Python/Rust stack.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser(
        "backtest", help="run the research backtest (research/ml_state_optimized.py)"
    )
    p_backtest.set_defaults(func=cmd_backtest)

    p_trade = sub.add_parser(
        "live-trade", help="run strategies for one or more tickers (in-process Rust)"
    )
    p_trade.add_argument(
        "--tickers", nargs="+", default=list(DEFAULT_TICKERS),
        help="symbols to run concurrently, e.g. btcusdt ethusdt solusdt",
    )
    p_trade.add_argument("--exchange", default=DEFAULT_EXCHANGE, choices=SUPPORTED_EXCHANGES)
    p_trade.add_argument(
        "--mode", default="paper", choices=SUPPORTED_MODES,
        help="execution mode: paper (dry-run) or live (real orders)",
    )
    p_trade.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL_USD,
        help="capital (USD) allocated to EACH strategy",
    )
    p_trade.set_defaults(func=cmd_live_trade)

    p_test = sub.add_parser(
        "live-test",
        help="replay historical parquet data through the Rust engine (paper trading)",
    )
    p_test.add_argument(
        "--ticker", default=DEFAULT_TICKERS[0], help="symbol to replay, e.g. btcusdt"
    )
    p_test.add_argument(
        "--feed-dir", default="research/csv",
        help="directory with {TICKER}_trades / _{TICKER}_orderbook parquet (or csv) feeds",
    )
    p_test.add_argument(
        "--feed", default=None,
        help="use an existing replay stream file instead of building one from parquet",
    )
    p_test.add_argument(
        "--capital", type=float, default=None,
        help="capital (USD) for position sizing (default: 10000)",
    )
    p_test.add_argument(
        "--no-db", action="store_true",
        help="do not persist to RocksDB (skips the HTML report; default DB wiped per run)",
    )
    p_test.add_argument(
        "--db", default=None,
        help="explicit RocksDB path (default: build/live_test_{ticker}.rocksdb, wiped per run)",
    )
    p_test.add_argument(
        "--seed", type=int, default=0,
        help="composite closes used to warm feature rings (0 = warm live from the feed)",
    )
    p_test.add_argument(
        "--start-ms", type=int, default=None,
        help="only replay events with timestamp >= MS (uses for huge multi-day feeds)",
    )
    p_test.add_argument(
        "--end-ms", type=int, default=None,
        help="only replay events with timestamp <= MS",
    )
    p_test.add_argument(
        "--limit", type=int, default=None,
        help="cap the replay to the first N events (in timestamp order)",
    )
    p_test.add_argument(
        "--loop", action="store_true",
        help="loop the feed forever (continuous sim) instead of stopping at end-of-data",
    )
    p_test.add_argument(
        "--release", action="store_true", help="build the replay binary in release mode"
    )
    p_test.set_defaults(func=cmd_live_test)

    p_doc = sub.add_parser(
        "doctor", help="validate the environment, repo layout and native binding"
    )
    p_doc.add_argument(
        "--require-native", action="store_true",
        help="fail if the maturin module is not importable",
    )
    p_doc.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        logger.info("interrupted — exiting")
        return 130


if __name__ == "__main__":
    sys.exit(main())