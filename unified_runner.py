#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSADP-Core — Unified Runner
===========================

Single-point orchestration for the hybrid Python/Rust quant stack.

The Rust core (``uets_rs/``) is compiled into a native Python
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
    uets_rs/engine/        Rust strategy engine
    uets_rs/live-trade/    real-market execution (Bybit/Bitfinex WS +
                                     RocksDB persistence)
    uets_rs/live-test/     historical replay (paper-trading) runner
    uets_rs/bindings/      PyO3 bindings -> the ``engine`` module
    scripts/ingestion/               Binance caching, download, CCXT conversion
    build/                           HTML reports, CSV trade logs
    docs/                            dev log + trade journal

CLI
---
    python unified_runner.py backtest
    python unified_runner.py live-trade --exchange bitfinex --feed bybit \
        --mode paper --capital 10000
    # Multi-asset portfolio: one live-trade PROCESS per asset from a master
    # table (see portfolio.example.json), with total PnL tracking, per-asset
    # RocksDB stores, a tabbed dashboard (build/portfolio_dashboard.html) and
    # crash alerts:
    python unified_runner.py portfolio --config portfolio.json --restart
    # Inspect / delete RocksDB stores:
    python unified_runner.py db --list
    python unified_runner.py db --clean-ticker btcusdt --yes
    # Stream Binance/Bitfinex/Bybit orderflow while executing on Bitfinex:
    python unified_runner.py live-trade --exchange bitfinex --feed binance --mode live
    # Serve a live browser page of trades executed (+equity/wins/losses):
    python unified_runner.py live-trade --exchange bitfinex --feed bybit --dashboard
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

# Canonical strategy config: the Python source of truth (repo-root level,
# outside the legacy `main/` package) that gets exported to JSON for the
# Rust engine on every run.
STRATEGY_CONFIG_PY = REPO_ROOT / "config" / "strategy.py"

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
SUPPORTED_FEEDS = ("bybit", "bitfinex", "binance")
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
        feed: Optional[str] = None,
        config_path: Optional[str] = None,
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
        if feed is not None and feed not in SUPPORTED_FEEDS:
            raise ValueError(f"unknown feed {feed!r} (expected {SUPPORTED_FEEDS})")
        if capital <= 0:
            raise ValueError("capital must be > 0")
        self.ticker = ticker.lower()
        self._exchange = exchange
        # Feed venue defaults to the execution venue unless explicitly given.
        self._feed = feed or exchange
        self._mode = mode
        self._rust = native.Strategy(
            ticker,
            exchange=exchange,
            mode=mode,
            capital=capital,
            db_path=db_path,
            config_path=config_path,
            feed=self._feed,
        )

    @property
    def exchange(self) -> str:
        return self._exchange

    @property
    def feed(self) -> str:
        return self._feed

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
        feed: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Strategy:
        key = ticker.lower()
        if key in self._strategies:
            raise ValueError(f"strategy for {key!r} already added")
        s = Strategy(
            ticker,
            exchange=exchange,
            mode=mode,
            capital=capital,
            db_path=db_path,
            feed=feed,
            config_path=config_path,
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

    # Materialize the strategy config (single source of truth: config/strategy.py)
    # and hand it to every strategy, so live-trade runs the exact parameters
    # research/live-test use — e.g. a CONFIDENCE_THRESHOLD edit in
    # config/strategy.py applies on the next launch WITHOUT a Rust rebuild.
    trade_config_path = _export_strategy_config()
    if trade_config_path is None:
        print(f"ERROR: {STRATEGY_CONFIG_PY} not found — required for config export", file=sys.stderr)
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
            pf.add(
                ticker,
                exchange=args.exchange,
                feed=args.feed,
                mode=args.mode,
                capital=args.capital,
                config_path=trade_config_path,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    # Optional live trade-count dashboard (idea: a browser page showing how
    # many trades the executor has executed in real time). Zero overhead when
    # disabled — no server thread is spawned.
    dashboard = None
    if getattr(args, "dashboard", False):
        try:
            from uets_legacy.live_dashboard import LiveDashboard

            dashboard = LiveDashboard()
            dashboard.start()
        except Exception as exc:  # noqa: BLE001 — dashboard is optional
            logger.warning("live dashboard failed to start: %s", exc)
            dashboard = None

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
            # The Rust loop ingests every websocket frame in real time - this
            # sleep only sets how often Python prints a status line. Default
            # 2 s keeps the console readable; use --status-interval to tune.
            time.sleep(args.status_interval)
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            for ticker, st in pf.status().items():
                mid = f'{st["mid"]:.2f}' if st["mid"] else "—"
                stal = st.get("feed_staleness_ms")
                lat = st.get("last_event_latency_ms")
                sk = st.get("clock_skew_ms")
                # Gate telemetry from the newest closed 1-second bar:
                # rsi_2 drives the hunting zone (<30 for long-only), win_prob
                # is the model P(WIN) vs the confidence gate (>= 0.5).
                rsi = st.get("rsi_2")
                pwin = st.get("win_prob")
                gate = st.get("gate") or ""
                logger.info(
                    "  %-10s %s  eq=%9.2f tr=%d r=%+.2f mid=%s "
                    "rsi=%5.1f p_win=%.2f gate=%s stal=%7.1fms lat=%6.2fms sk=%+8.1fms",
                    ticker, stamp,
                    st["equity"], st["n_trades"], st["total_r"], mid,
                    rsi if isinstance(rsi, (int, float)) else -1.0,
                    pwin if isinstance(pwin, (int, float)) else 0.0,
                    gate,
                    stal if isinstance(stal, (int, float)) else -1.0,
                    lat if isinstance(lat, (int, float)) else -1.0,
                    sk if isinstance(sk, (int, float)) else 0.0,
                )
                if dashboard is not None:
                    # Push the latest snapshot so the browser page updates live.
                    dashboard.update(ticker, st)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("stopping strategies ...")
        _stop_all()
        if dashboard is not None:
            dashboard.stop()
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

    # 2. Materialize strategy config (single source: config/strategy.py) unless
    # the caller passed an explicit --config, then hand it to the engine.
    config_path = args.config or str(REPO_ROOT / "build" / "live_test_config.json")
    if not STRATEGY_CONFIG_PY.exists():
        raise RuntimeError(f"{STRATEGY_CONFIG_PY} not found — required for --config export")

    if args.config:
        config_path = args.config
        logger.info("using explicit strategy config -> %s", config_path)
    else:
        config_path = _export_strategy_config()
        if config_path is None:
            raise RuntimeError(
                f"{STRATEGY_CONFIG_PY} not found — required for config export"
            )
    logger.info("exported strategy config -> %s", config_path)

    # Walk-forward parity mode: when --wf-dir is given, flip the two research-
    # universe gates in the exported config so live-test (a) scores each second
    # with the fold model whose OOS test window covers it and (b) restricts
    # entries to one volume>0 candidate per composite bar -- exactly the
    # research trade universe.
    if getattr(args, "wf_dir", None):
        import json as _json

        with open(config_path, "r", encoding="utf-8") as fh:
            _cfg_json = _json.load(fh)
        _cfg_json["require_fold_coverage"] = True
        _cfg_json["one_candidate_per_bar"] = True
        with open(config_path, "w", encoding="utf-8") as fh:
            _json.dump(_cfg_json, fh, indent=2)
        logger.info(
            "walk-forward parity gates ON (fold models from %s)", args.wf_dir
        )

    # 3. Kick off the replay through the identical live-trade execution loop.
    cmd = ["cargo", "run", "-p", LIVE_TEST_BIN]
    if args.release:
        cmd.append("--release")
    cmd += ["--", args.ticker.lower(), "--feed", str(feed_path)]
    cmd += ["--config", config_path]
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
    if getattr(args, "wf_dir", None):
        cmd += ["--wf-dir", str(args.wf_dir)]
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

    # cargo/maturin are BUILD-TIME tools — needed to compile the wheel, not to
    # run/install it. They are informational here so `doctor` stays green on
    # wheel-only deployments (CI smoke, Termux proot, Oracle/Linux edge, RPi).
    if shutil.which("cargo") is not None:
        print("[OK] cargo available")
    else:
        print("[info] cargo not available (build-time only; not needed to run the wheel)")
    if shutil.which("maturin") is not None:
        print("[OK] maturin available")
    else:
        print("[info] maturin not available (build-time only; not needed to run the wheel)")
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


def _export_strategy_config() -> Optional[str]:
    """Materialize config/strategy.py -> build/live_trade_config.json.

    Also verify that the generated JSON actually contains the fields Rust
    needs for position sizing. Fail immediately rather than letting Rust
    silently fall back to defaults.
    """
    if not STRATEGY_CONFIG_PY.exists():
        return None

    import importlib.util as _ilu
    import json as _json

    spec = _ilu.spec_from_file_location("_strategy_config", STRATEGY_CONFIG_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strategy config: {STRATEGY_CONFIG_PY}")

    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    trade_config_path = REPO_ROOT / "build" / "live_trade_config.json"

    # ALWAYS regenerate. Never trust an existing/stale JSON file.
    mod.export_strategy_config(str(trade_config_path))

    # Read back what was actually written.
    with open(trade_config_path, "r", encoding="utf-8") as fh:
        generated = _json.load(fh)

    # These are critical runtime sizing/execution parameters.
    required = {
        "risk_pct_per_trade": getattr(mod, "RISK_PCT_PER_TRADE"),
        "max_leverage": getattr(mod, "MAX_LEVERAGE"),
        "rr_ratio": getattr(mod, "RR_RATIO"),
        "fee_bps": getattr(mod, "FEE_BPS"),
        "max_position_usd": getattr(mod, "MAX_POSITION_USD"),
    }

    missing = [k for k in required if k not in generated]
    if missing:
        raise RuntimeError(
            "Generated strategy config is missing required fields: "
            + ", ".join(missing)
        )

    mismatches = []
    for key, expected in required.items():
        actual = generated[key]
        if actual != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, generated {actual!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Generated strategy config does not match config/strategy.py:\n  "
            + "\n  ".join(mismatches)
        )

    logger.info(
        "strategy config -> %s | risk_pct_per_trade=%s max_leverage=%s "
        "rr_ratio=%s fee_bps=%s max_position_usd=%s",
        trade_config_path,
        generated["risk_pct_per_trade"],
        generated["max_leverage"],
        generated["rr_ratio"],
        generated["fee_bps"],
        generated["max_position_usd"],
    )

    return str(trade_config_path)


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Supervise one `live-trade` process per asset from a master JSON table."""
    from portfolio_supervisor import run_portfolio

    cfg = _export_strategy_config()
    if cfg is None:
        print(f"ERROR: {STRATEGY_CONFIG_PY} not found — required for config export", file=sys.stderr)
        return 1
    return run_portfolio(
        args.config,
        release=args.release,
        restart=args.restart,
        status_interval=args.status_interval,
        no_dashboard=args.no_dashboard,
    )


def _rocksdb_stores() -> list[Path]:
    """All local RocksDB stores managed by this repo (data/ + live-test tapes)."""
    stores: list[Path] = []
    data = REPO_ROOT / "data"
    if data.exists():
        for entry in sorted(data.glob("*.rocksdb*")):
            stores.append(entry)
    build = REPO_ROOT / "build"
    if build.exists():
        for entry in sorted(build.glob("live_test_*.rocksdb*")):
            stores.append(entry)
    return stores


def _dir_size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def cmd_db(args: argparse.Namespace) -> int:
    """List or delete local RocksDB stores (data/*.rocksdb, build/live_test_*)."""
    stores = _rocksdb_stores()
    if args.list or not (args.clean_ticker or args.clean_all):
        if not stores:
            print("no RocksDB stores found")
            return 0
        print(f"{'store':<55}{'size':>10}")
        for s in stores:
            print(f"{str(s.relative_to(REPO_ROOT)):<55}{_dir_size_mb(s):>8.1f}MB")
        print(f"\ntotal: {len(stores)} store(s)")
        print("delete with: python unified_runner.py db --clean-ticker <sym> | --all [--yes]")
        return 0

    targets: list[Path] = []
    sym = (args.clean_ticker or "").lower()
    for s in stores:
        name = s.name.lower()
        if args.clean_all or (sym and sym in name):
            targets.append(s)
    if not targets:
        print(f"no stores match '{args.clean_ticker}'")
        return 1

    print("about to DELETE:")
    total_mb = 0.0
    for t in targets:
        mb = _dir_size_mb(t)
        total_mb += mb
        print(f"  - {t.relative_to(REPO_ROOT)}  ({mb:.1f} MB)")
    print(f"total: {total_mb:.1f} MB across {len(targets)} store(s)")

    if not args.yes:
        answer = input("type 'yes' to confirm deletion: ").strip().lower()
        if answer != "yes":
            print("aborted.")
            return 1

    import shutil as _shutil

    for t in targets:
        try:
            if t.is_dir():
                _shutil.rmtree(t)
            else:
                t.unlink()
            print(f"deleted {t.relative_to(REPO_ROOT)}")
        except OSError as exc:
            # Most common cause: a session is still running with the DB open.
            print(f"ERROR deleting {t}: {exc}", file=sys.stderr)
            return 1
    print("done.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified_runner",
        description=(
            "OSADP-Core unified runner - orchestration for the hybrid Python/Rust "
            "quant stack (backtest, replay, live paper/live execution, HTML reports).\n"
            "\n"
            "Every command below auto-exports the strategy parameters from "
            "config/strategy.py into a JSON the Rust engine reads, so editing "
            "parameters does NOT need a Rust rebuild.\n"
            "\n"
            "COMMANDS (run '<command> -h' for full options of one):\n"
            "  doctor      check the environment / repo / native Rust binding\n"
            "  backtest    run the research walk-forward backtest (Python pipeline)\n"
            "  live-test   replay historical parquet/CSV through the Rust engine\n"
            "  live-trade  run one or more tickers live (paper or real orders)\n"
            "  portfolio   supervise many live-trade child PROCESSES from a JSON table\n"
            "  db          list / delete the RocksDB stores that back the reports"
        ),
        epilog=(
            "EXAMPLES\n"
            "  # environment health check (fail if the native module is missing):\n"
            "  python unified_runner.py doctor --require-native\n"
            "  # paper-trade several tickers on Bybit tape, executing on Bitfinex:\n"
            "  python unified_runner.py live-trade --tickers btcusdt ethusdt \\\n"
            "      --exchange bitfinex --feed bybit --mode paper --capital 10000\n"
            "  # real orders on Bitfinex while streaming Binance orderflow:\n"
            "  python unified_runner.py live-trade --exchange bitfinex \\\n"
            "      --feed binance --mode live\n"
            "  # multi-asset portfolio (one OS process per asset) + tabbed dashboard:\n"
            "  python unified_runner.py portfolio --config portfolio.json --restart\n"
            "  # replay history deterministically (to reproduce the backtest):\n"
            "  python unified_runner.py live-test --ticker btcusdt --release\n"
            "  # manage the RocksDB stores:\n"
            "  python unified_runner.py db --list\n"
            "  python unified_runner.py db --clean-ticker btcusdt --yes\n"
            "\n"
            "OUTPUTS\n"
            "  build/pnl_report_<symbol>.html   per-symbol TradingView-style report\n"
            "  build/portfolio_dashboard.html   multi-asset tabbed dashboard\n"
            "  build/live_trade_config.json     strategy params the Rust engine uses\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="enable debug-level logging (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser(
        "backtest",
        help="run the research walk-forward backtest",
        description=(
            "Run the pure-Python research backtest (research/ml_state_optimized.py): "
            "feature engineering, walk-forward evaluation and the model export used "
            "to train the LightGBM model that live-trade / live-test score with.\n"
            "This is the research side of the stack - it does not touch a live "
            "exchange and writes its reports under build/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_backtest.set_defaults(func=cmd_backtest)

    p_trade = sub.add_parser(
        "live-trade",
        help="run one or more tickers (paper or live orders), in-process Rust",
        description=(
            "Run strategies for one or more tickers concurrently. Each ticker runs "
            "in its own tokio runtime inside the Rust process; Python is only the "
            "control plane. By default this simulates fills (--mode paper); "
            "--mode live routes real orders.\n\n"
            "Feeds vs venues:\n"
            "  --feed       where market data streams from (orderflow source)\n"
            "  --exchange   where orders are routed (execution venue)\n"
            "  They can differ, e.g. --exchange bitfinex --feed binance streams "
            "Binance tape while executing on Bitfinex.\n\n"
            "While it runs the session report (build/pnl_report_<symbol>.html) "
            "auto-regenerates - open it in a browser to watch trades live."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_trade.add_argument(
        "--tickers", nargs="+", default=list(DEFAULT_TICKERS),
        help="symbols to run concurrently, e.g. btcusdt ethusdt solusdt",
    )
    p_trade.add_argument(
        "--exchange", default=DEFAULT_EXCHANGE, choices=SUPPORTED_EXCHANGES,
        help="venue orders are routed to (execution). Choices: "
             + ", ".join(SUPPORTED_EXCHANGES),
    )
    p_trade.add_argument(
        "--feed", default=None, choices=SUPPORTED_FEEDS,
        help="market-data feed venue (orderflow source); defaults to --exchange. "
             "e.g. --exchange bitfinex --feed binance streams Binance tape while "
             "executing on Bitfinex. Choices: " + ", ".join(SUPPORTED_FEEDS),
    )
    p_trade.add_argument(
        "--dashboard", action="store_true",
        help="serve a live browser page (http://localhost:8001) showing trades "
             "executed, equity, wins/losses and feed freshness per ticker",
    )
    p_trade.add_argument(
        "--mode", default="paper", choices=SUPPORTED_MODES,
        help="execution mode: paper (simulated fills, no orders) or live (real "
             "orders). Default: paper. Choices: " + ", ".join(SUPPORTED_MODES),
    )
    p_trade.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL_USD,
        help="capital (USD) allocated to EACH ticker "
             "(default: %d)" % DEFAULT_CAPITAL_USD,
    )
    p_trade.add_argument(
        "--status-interval", type=float, default=2.0,
        help="seconds between console status lines (default: 2.0). Only affects "
             "the printed status, never data ingestion (which is always real-time)",
    )
    p_trade.set_defaults(func=cmd_live_trade)

    p_port = sub.add_parser(
        "portfolio",
        help="supervise one live-trade PROCESS per asset from a master JSON table",
        description=(
            "Run a whole book of assets from one master JSON table (see "
            "portfolio.example.json). Each asset gets its OWN live-trade OS process "
            "(own tokio runtime, own RocksDB store data/portfolio_<ticker>.rocksdb, "
            "own auto-refreshing HTML report) while this supervisor:\n"
            "  - parses each child's console output into live state,\n"
            "  - prints a per-asset + PORTFOLIO-TOTAL table on a fixed cadence,\n"
            "  - raises an ALERT when a child exits unexpectedly (optional restart),\n"
            "  - writes build/portfolio_dashboard.html (tabbed overview + per asset),\n"
            "  - forwards Ctrl-C to every child for a graceful RocksDB flush."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_port.add_argument(
        "--config", default="portfolio.json",
        help="master portfolio table (JSON); see portfolio.example.json "
             "(default: portfolio.json)",
    )
    p_port.add_argument(
        "--release", action="store_true",
        help="prefer the release build of the live-trade binary "
             "(target/release/live-trade) over the debug build",
    )
    p_port.add_argument(
        "--restart", action="store_true",
        help="auto-restart a child process that exits unexpectedly (3s cooldown)",
    )
    p_port.add_argument(
        "--status-interval", type=float, default=None,
        help="seconds between portfolio summary tables (default: from config, else 5)",
    )
    p_port.add_argument(
        "--no-dashboard", action="store_true",
        help="skip writing build/portfolio_dashboard.html",
    )
    p_port.set_defaults(func=cmd_portfolio)

    p_db = sub.add_parser(
        "db",
        help="list or delete the local RocksDB stores (report data)",
        description=(
            "Manage the RocksDB stores under data/ that back each session's HTML "
            "report. These store market tape, features, PnL snapshots and closed "
            "trades. Cleaning a store discards that asset's history - a fresh "
            "session starts from a clean report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_db.add_argument(
        "--list", action="store_true",
        help="list all known stores (this is the default action)",
    )
    p_db.add_argument(
        "--clean-ticker", metavar="SYMBOL",
        help="delete stores whose name contains SYMBOL, e.g. --clean-ticker btcusdt",
    )
    p_db.add_argument(
        "--clean-all", action="store_true",
        help="delete ALL stores under data/ (dangerous - prompts unless --yes)",
    )
    p_db.add_argument(
        "--yes", action="store_true",
        help="skip the interactive confirmation prompt",
    )
    p_db.set_defaults(func=cmd_db)

    p_test = sub.add_parser(
        "live-test",
        help="replay historical parquet/CSV through the Rust engine (paper)",
        description=(
            "Replay historical orderbook + trade data (research/csv, parquet or a "
            "prebuilt .replay stream) through the exact same Rust execution loop as "
            "live-trade. Everything downstream - 1s micro-bars, 300s composite "
            "features, entry/exit gates, RocksDB persistence, HTML report - is the "
            "identical code path, so replaying the same data that produced the "
            "research backtest should reproduce the same trades.\n"
            "Use --wf-dir to reproduce the walk-forward backtest exactly (fold "
            "models + research trade-universe gates)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_test.add_argument(
        "--ticker", default=DEFAULT_TICKERS[0],
        help="symbol to replay, e.g. btcusdt (default: %s)" % DEFAULT_TICKERS[0],
    )
    p_test.add_argument(
        "--feed-dir", default="research/csv",
        help="directory with {TICKER}_trades / _{TICKER}_orderbook parquet (or csv) "
             "feeds (default: research/csv)",
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
        help="do not persist to RocksDB (skips the HTML report; default DB is wiped "
             "per run)",
    )
    p_test.add_argument(
        "--db", default=None,
        help="explicit RocksDB path (default: build/live_test_{ticker}.rocksdb, "
             "wiped per run)",
    )
    p_test.add_argument(
        "--seed", type=int, default=0,
        help="composite closes used to warm feature rings (0 = warm live from the "
             "feed)",
    )
    p_test.add_argument(
        "--start-ms", type=int, default=None,
        help="only replay events with timestamp >= MS (for huge multi-day feeds)",
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
        help="loop the feed forever (continuous sim) instead of stopping at "
             "end-of-data",
    )
    p_test.add_argument(
        "--release", action="store_true",
        help="build/use the release replay binary instead of debug",
    )
    p_test.add_argument(
        "--config", default=None,
        help="path to a StrategyConfig JSON (default: build/live_test_config.json "
             "exported from config/strategy.py)",
    )
    p_test.add_argument(
        "--wf-dir", default=None,
        help="walk-forward fold-model dir (build/walkforward_models). When set, "
             "live-test scores each second with the fold model whose OOS test "
             "window covers it and restricts entries to research's trade universe "
             "(one volume>0 candidate per composite bar) so the replay reproduces "
             "the walk-forward backtest.",
    )
    p_test.set_defaults(func=cmd_live_test)

    p_doc = sub.add_parser(
        "doctor",
        help="validate the environment, repo layout and native binding",
        description=(
            "Run a self-check of the environment: Python/cargo availability, repo "
            "layout, the RocksDB / native 'engine' binding, and the strategy config "
            "source. Use --require-native to fail hard if the maturin-compiled "
            "'engine' module is not importable - useful in CI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_doc.add_argument(
        "--require-native", action="store_true",
        help="fail if the maturin-compiled 'engine' module is not importable",
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