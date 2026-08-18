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
    python unified_runner.py doctor [--require-native]
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
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
        """Request graceful shutdown and detach the runner thread."""
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
    for ticker in args.tickers:
        try:
            pf.add(ticker, exchange=args.exchange, mode=args.mode, capital=args.capital)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    pf.run()
    logger.info(
        "running %d strategy/strategies on %s (%s) — Ctrl-C to stop",
        len(pf), args.exchange, args.mode,
    )

    state = {"stopped": False}

    def _shutdown(signum, _frame):
        state["stopped"] = True
        sig = signal.Signals(signum).name
        logger.info("%s received — stopping all strategies", sig)
        pf.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, OSError):
        pass  # SIGTERM not available on this platform (e.g. Windows)

    try:
        while not state["stopped"] and any(s.is_running for s in pf):
            time.sleep(2)
            for ticker, st in pf.status().items():
                mid = f'{st["mid"]:.2f}' if st["mid"] else "—"
                logger.info(
                    "  %-10s equity=%9.2f  trades=%d  r=%+.2f  mid=%s",
                    ticker, st["equity"], st["n_trades"], st["total_r"], mid,
                )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("stopping strategies ...")
        pf.stop()
    logger.info("done.")
    return 0

def cmd_live_test(args: argparse.Namespace) -> int:
    """Replay historical data through the Rust `live-test` binary (paper trading)."""
    if shutil.which("cargo") is None:
        print("cargo not found on PATH", file=sys.stderr)
        return 1
    cmd = ["cargo", "run", "-p", LIVE_TEST_BIN]
    if args.release:
        cmd.append("--release")
    cmd += ["--", args.ticker]
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
        "live-test", help="replay historical data through the Rust engine (paper trading)"
    )
    p_test.add_argument(
        "--ticker", default=DEFAULT_TICKERS[0], help="symbol to replay, e.g. btcusdt"
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