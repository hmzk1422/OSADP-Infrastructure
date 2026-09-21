#!/usr/bin/env python3
"""
datapaths.py
============

Single source of truth for the on-disk ingestion layout.

Every path the ingestion layer reads or writes is derived here, so a layout
change is a one-file edit. The tree is organized by **data type**, not by
provider:

    market_data/                                  <- MARKET_DATA_ROOT
      ingest_source/                              <- raw, provider-native, immutable
        manifests/<TICKER>_manifest.json
        <TICKER>/
          aggTrades/<YYYY-MM-DD>/<TICKER>-aggTrades-<YYYY-MM-DD>.zip    (Binance)
          bookDepth/<YYYY-MM-DD>/<TICKER>-bookDepth-<YYYY-MM-DD>.zip    (Binance)
          orderbook/<YYYY-MM-DD>/<HH>/<TICKER>_orderbook.parquet        (CryptoHFTData)
      ingest_parquet/                             <- normalized, uniform Parquet
        <TICKER>/
          aggTrades/<YYYY-MM-DD>.parquet
          bookDepth/<YYYY-MM-DD>.parquet
          orderbook/<YYYY-MM-DD>/<HH>.parquet
      matched/                                    <- aggTrades x L2 join output
        <TICKER>/<TICKER>_<start>_<end>_<HH>-<HH>.parquet

The provider (binance / cryptohftdata) is deliberately NOT part of the path:
the dataset name already identifies its provider (aggTrades/bookDepth are
Binance, orderbook is CryptoHFTData L2) and every consumer selects by data
type. `PROVIDER_OF` keeps that mapping in code for manifest bookkeeping.

Root resolution order (highest wins):

    1. explicit `root=` argument / `--root` CLI flag
    2. $MARKET_DATA_ROOT environment variable
    3. <repo>/market_data

`--root` / $MARKET_DATA_ROOT always name the **market_data** directory itself,
never the ingest_source/ingest_parquet subdirectory.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "PROJECT_ROOT",
    "MARKET_DATA_ROOT",
    "DATASETS",
    "BINANCE_DATASETS",
    "CHD_DATASETS",
    "PROVIDER_OF",
    "DEFAULT_SYMBOL",
    "day_str",
    "source_root",
    "parquet_root",
    "matched_root",
    "manifest_dir",
    "manifest_file",
    "source_dir",
    "parquet_dir",
    "matched_dir",
    "source_binance_file",
    "source_chd_file",
    "parquet_binance_file",
    "parquet_chd_file",
    "source_file_for",
    "binance_parquet_days",
    "orderbook_parquet_days",
    "available_range",
    "ensure",
    "describe",
]


# `components/python/ingestion_L2/datapaths.py` -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MARKET_DATA_ROOT = Path(
    os.environ.get("MARKET_DATA_ROOT") or (PROJECT_ROOT / "market_data")
).expanduser().resolve()

SOURCE_DIRNAME = "ingest_source"
PARQUET_DIRNAME = "ingest_parquet"
MATCHED_DIRNAME = "matched"
MANIFESTS_DIRNAME = "manifests"

# Dataset -> provider. The only place the provider mapping is recorded.
DATASETS = ("aggTrades", "bookDepth", "orderbook")
BINANCE_DATASETS = ("aggTrades", "bookDepth")
CHD_DATASETS = ("orderbook",)
PROVIDER_OF = {
    "aggTrades": "binance",
    "bookDepth": "binance",
    "orderbook": "cryptohftdata",
}

DEFAULT_SYMBOL = "BTCUSDT"

Day = Union[str, date]

# ===========================================================
# PRIMITIVES
# ===========================================================

def day_str(day: Day) -> str:
    """Normalize a date / ISO string to the `YYYY-MM-DD` path token."""
    return day.isoformat() if isinstance(day, date) else str(day)


def _market_root(root: Optional[Union[Path, str]] = None) -> Path:
    return Path(root).expanduser().resolve() if root else MARKET_DATA_ROOT


def ensure(path: Union[Path, str]) -> Path:
    """mkdir -p the given path and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===========================================================
# TREES
# ===========================================================

def source_root(root: Optional[Union[Path, str]] = None) -> Path:
    """Raw provider-native files (Binance zips, CryptoHFTData parquet)."""
    return _market_root(root) / SOURCE_DIRNAME


def parquet_root(root: Optional[Union[Path, str]] = None) -> Path:
    """Normalized Parquet tree written by convert_to_parquet.py."""
    return _market_root(root) / PARQUET_DIRNAME


def matched_root(root: Optional[Union[Path, str]] = None) -> Path:
    """aggTrades x L2 matched tree (research read/write surface)."""
    return _market_root(root) / MATCHED_DIRNAME


def manifest_dir(root: Optional[Union[Path, str]] = None) -> Path:
    return source_root(root) / MANIFESTS_DIRNAME


def manifest_file(
    root: Optional[Union[Path, str]] = None, symbol: str = DEFAULT_SYMBOL
) -> Path:
    return manifest_dir(root) / f"{symbol.upper()}_manifest.json"


# ===========================================================
# PER-SYMBOL / PER-DATASET DIRECTORIES
# ===========================================================

def source_dir(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = "aggTrades",
) -> Path:
    """<root>/ingest_source/<TICKER>/<dataset>"""
    return source_root(root) / symbol.upper() / dataset


def parquet_dir(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = "aggTrades",
) -> Path:
    """<root>/ingest_parquet/<TICKER>/<dataset>"""
    return parquet_root(root) / symbol.upper() / dataset


def matched_dir(
    root: Optional[Union[Path, str]] = None, symbol: str = DEFAULT_SYMBOL
) -> Path:
    """<root>/matched/<TICKER>"""
    return matched_root(root) / symbol.upper()


# ===========================================================
# FILES
# ===========================================================

def source_binance_file(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = "aggTrades",
    day: Day = "",
) -> Path:
    """<root>/ingest_source/<TICKER>/<dataset>/<DATE>/<TICKER>-<dataset>-<DATE>.zip"""
    d = day_str(day)
    sym = symbol.upper()
    return source_dir(root, sym, dataset) / d / f"{sym}-{dataset}-{d}.zip"


def source_chd_file(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    day: Day = "",
    hour: int = 0,
) -> Path:
    """<root>/ingest_source/<TICKER>/orderbook/<DATE>/<HH>/<TICKER>_orderbook.parquet"""
    d = day_str(day)
    sym = symbol.upper()
    return (
        source_dir(root, sym, "orderbook")
        / d
        / f"{int(hour):02d}"
        / f"{sym}_orderbook.parquet"
    )


def parquet_binance_file(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = "aggTrades",
    day: Day = "",
) -> Path:
    """<root>/ingest_parquet/<TICKER>/<dataset>/<DATE>.parquet"""
    return parquet_dir(root, symbol, dataset) / f"{day_str(day)}.parquet"


def parquet_chd_file(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    day: Day = "",
    hour: int = 0,
) -> Path:
    """<root>/ingest_parquet/<TICKER>/orderbook/<DATE>/<HH>.parquet"""
    return (
        parquet_dir(root, symbol, "orderbook")
        / day_str(day)
        / f"{int(hour):02d}.parquet"
    )


def source_file_for(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    provider: str = "binance",
    dataset: str = "aggTrades",
    day: Day = "",
    hour: Optional[int] = None,
) -> Path:
    """Canonical raw path for a (provider, dataset, date, hour) manifest entry.

    Used by `main.py migrate-layout` to re-point manifest entries recorded
    against an older tree (e.g. `<provider>/<TICKER>/<dataset>/...` or
    `research/data_v2/...`) onto the current layout.
    """
    if dataset in CHD_DATASETS or provider == "cryptohftdata":
        return source_chd_file(root, symbol, day, 0 if hour is None else hour)
    return source_binance_file(root, symbol, dataset, day)


# ===========================================================
# INVENTORY HELPERS (metadata only, no data reads)
# ===========================================================

def binance_parquet_days(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = "aggTrades",
) -> list[str]:
    """Sorted `YYYY-MM-DD` days with a `<DATE>.parquet` in the parquet tree."""
    d = parquet_dir(root, symbol, dataset)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.parquet") if len(p.stem) == 10)


def orderbook_parquet_days(
    root: Optional[Union[Path, str]] = None, symbol: str = DEFAULT_SYMBOL
) -> list[str]:
    """Sorted `YYYY-MM-DD` day directories in the parquet orderbook tree."""
    d = parquet_dir(root, symbol, "orderbook")
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and len(p.name) == 10)


def available_range(
    root: Optional[Union[Path, str]] = None,
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = "aggTrades",
) -> Optional[tuple[str, str]]:
    """(first_day, last_day) present in the parquet tree, or None when empty."""
    days = binance_parquet_days(root, symbol, dataset)
    return (days[0], days[-1]) if days else None


def describe(
    root: Optional[Union[Path, str]] = None, symbol: str = DEFAULT_SYMBOL
) -> str:
    """Human-readable inventory of the ingestion trees (metadata only)."""
    root = _market_root(root)
    sym = symbol.upper()
    lines = [
        f"market_data root : {root}",
        f"symbol           : {sym}",
        "",
        "ingest_source (raw, provider-native)",
    ]

    src_root = source_root(root)
    if src_root.is_dir():
        legacy = sorted(
            p.name
            for p in src_root.iterdir()
            if p.is_dir() and p.name not in (MANIFESTS_DIRNAME, sym)
        )
        if legacy:
            lines.append(
                f"  ! legacy provider-first dirs present: {legacy} "
                "(run `main.py migrate-layout --apply`)"
            )
    else:
        lines.append("  (absent)")

    for dataset in DATASETS:
        d = source_dir(root, sym, dataset)
        files = sum(1 for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0
        lines.append(f"  {dataset:<10} {files:>7,} raw file(s)  {d}")

    mf = manifest_file(root, sym)
    lines.append(f"  manifest   {'present' if mf.exists() else 'absent':<8} {mf}")

    lines.append("")
    lines.append("ingest_parquet (normalized)")
    for dataset in DATASETS:
        if dataset == "orderbook":
            days = orderbook_parquet_days(root, sym)
            files = sum(
                len(list((parquet_dir(root, sym, dataset) / day).glob("*.parquet")))
                for day in days
            )
        else:
            days = binance_parquet_days(root, sym, dataset)
            files = len(days)
        rng = f"{days[0]} .. {days[-1]}" if days else "-"
        lines.append(
            f"  {dataset:<10} {len(days):>4} day(s) / {files:>7,} file(s)  [{rng}]"
        )

    lines.append("")
    lines.append("matched")
    md = matched_dir(root, sym)
    matched_files = sorted(md.glob("*.parquet")) if md.is_dir() else []
    lines.append(f"  {len(matched_files):>7,} file(s)  {md}")

    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - quick manual inspection
    print(describe())
