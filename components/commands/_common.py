"""
_common.py
==========
Shared helpers for the ``components/commands`` chain.

The commands layer is the only place that knows how the whole pipeline fits
together: it resolves the market-data root, walks the date range, clones the
canonical ``ingest_parquet`` files into the matched work tree, and sequences the
Rust-facing orchestrators under ``components/rust/<library>/orchestrator``.

Dependency direction is strictly one way — commands imports orchestrators,
never the other way round — so the orchestrators stay usable standalone.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make `components.*` importable when a command is run as a plain script
# (`python components/commands/01_ingest.py` puts only this folder on sys.path).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import components.python.data_pipeline.CryptoL2.datapaths as paths

DEFAULT_SYMBOL = "BTCUSDT"

__all__ = [
    "REPO_ROOT",
    "DEFAULT_SYMBOL",
    "logger",
    "configure_logging",
    "market_root",
    "iter_dates",
    "available_dates",
    "clone_sources",
    "pipeline_work",
    "orderflow_dir",
    "stage",
    "run_stage",
]


logger = logging.getLogger("commands")


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent logging setup for script entry points."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ============================================================
# PATHS
# ============================================================

def market_root(root: str | Path | None = None) -> Path:
    """
    Resolve the market_data root.

    Precedence matches ``datapaths``: explicit argument, then the
    ``MARKET_DATA_ROOT`` environment variable, then ``<repo>/market_data``.

    NOTE: ``datapaths.MARKET_DATA_ROOT`` is deliberately *not* used as the
    default here — its ``PROJECT_ROOT`` resolves to ``components/`` rather than
    the repo root, which would point at the non-existent ``components/market_data``.
    """
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("MARKET_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "market_data"


def pipeline_work(root: str | Path, symbol: str) -> Path:
    """``<root>/matched/<TICKER>/_pipeline`` — disposable work tree."""
    return Path(root) / "matched" / symbol.upper() / "_pipeline"


def orderflow_dir(root: str | Path, symbol: str, date: str) -> Path:
    """``<root>/matched/<TICKER>/_pipeline/orderflow/<DATE>`` — every-event LOB replay output."""
    return pipeline_work(root, symbol) / "orderflow" / date


# ============================================================
# DATES
# ============================================================

def iter_dates(start: str, end: str) -> list[str]:
    """Inclusive ``YYYY-MM-DD`` sequence from start to end."""
    if end < start:
        raise ValueError(f"end date {end} is before start date {start}")
    first = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")
    days = (last - first).days
    return [
        (first + timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(days + 1)
    ]


def available_dates(root: str | Path, symbol: str, start: str, end: str) -> list[str]:
    """
    Dates inside ``[start, end]`` that have all three canonical inputs.

    A date is only usable when aggTrades, bookDepth **and** orderbook exist —
    the merger rejects the run otherwise. Dates inside the window that have
    *some* data but not all three are reported so gaps are never silent.
    """
    root_p = Path(root)
    agg = set(paths.binance_parquet_days(root_p, symbol, "aggTrades"))
    depth = set(paths.binance_parquet_days(root_p, symbol, "bookDepth"))
    orderbook = set(paths.orderbook_parquet_days(root_p, symbol))

    in_window = sorted(day for day in (agg | depth | orderbook) if start <= day <= end)
    usable = [day for day in in_window if day in agg and day in depth and day in orderbook]

    skipped = [day for day in in_window if day not in set(usable)]
    if skipped:
        logger.warning(
            "skipping %d date(s) with incomplete inputs: %s",
            len(skipped),
            ", ".join(skipped),
        )
        for day in skipped:
            missing = [
                name
                for name, present in (
                    ("aggTrades", day in agg),
                    ("bookDepth", day in depth),
                    ("orderbook", day in orderbook),
                )
                if not present
            ]
            logger.warning("  %s missing %s", day, ", ".join(missing))

    return usable


# ============================================================
# CLONE (ingest_parquet -> matched/_pipeline/source)
# ============================================================

def clone_sources(root: str | Path, symbol: str, date: str) -> list[Path]:
    """
    Materialize aggTrades, bookDepth and orderbook for one UTC date into the
    matched work tree::

        matched/<TICKER>/_pipeline/source/aggTrades/<DATE>.parquet      (1)
        matched/<TICKER>/_pipeline/source/bookDepth/<DATE>.parquet      (1)
        matched/<TICKER>/_pipeline/source/orderbook/<DATE>/<HH>.parquet (24)

    Canonical ``ingest_parquet`` files are only ever read. Returns the cloned
    paths — 26 per complete day.
    """
    # Imported lazily so `--help` does not drag in config/ingestion.
    from components.python.data_pipeline.matched.parquet import fetch_source_data

    try:
        cloned = fetch_source_data(root=str(root), symbol=symbol, start=date, end=date)
    except FileNotFoundError:
        logger.warning("[%s] no ingest files available to clone", date)
        return []
    logger.info("[%s] cloned %d file(s) into the matched work tree", date, len(cloned))
    return cloned


# ============================================================
# STAGE LOGGING / RUNNER
# ============================================================

@contextmanager
def stage(title: str):
    """Log a pipeline stage banner with elapsed time."""
    logger.info("=== %s ===", title)
    started = monotonic()
    try:
        yield
    finally:
        logger.info("=== %s finished in %.1fs ===", title, monotonic() - started)


def run_stage(
    dates: Iterable[str],
    action: Callable[[str], object],
    *,
    workers: int = 1,
    label: str = "stage",
) -> dict[str, object]:
    """
    Run ``action(date)`` for every date, sequentially or in parallel.

    Dates are independent — each carries its own work dirs and state — so
    parallelism is safe across dates (never across hours *within* a date, which
    stays sequential to preserve the cross-hour carry).
    """
    date_list = list(dates)
    results: dict[str, object] = {}

    if workers <= 1:
        for date in date_list:
            results[date] = action(date)
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(action, date): date for date in date_list}
        for future in as_completed(futures):
            date = futures[future]
            try:
                results[date] = future.result()
            except Exception as exc:  # noqa: BLE001 - re-raised with date context
                logger.error("[%s] %s failed: %s", date, label, exc)
                raise
    return results

