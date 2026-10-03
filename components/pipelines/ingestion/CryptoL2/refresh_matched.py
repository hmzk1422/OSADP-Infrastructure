#!/usr/bin/env python3
"""
refresh_matched.py
==================

Rebuild the matched aggTrades x L2 parquet files in
`<market_data>/matched/<TICKER>/` from the newest ingestion parquet tree.

The matching *implementation* stays with the strategy layer
(`.../MoE/data/live.py`) because it is research logic. This module is the
ingestion-side wrapper: it locates that data layer, supplies the current
`market_data` layout paths, forwards the CLI flags, and reports clearly when
the strategy side is not usable.

The data layer is DISCOVERED by walking the repo for `MoE/data/live.py`
rather than hard-coded, so renaming the top-level package (research ->
strategies -> strategy_system -> ...) does not break ingestion. `dataconfig`
is optional; when it is missing, this module supplies its own defaults from
the ingestion layout.

Usage
-----
    python refresh_matched.py --start 2026-02-01 --end 2026-02-02
    python main.py refresh-matched --chunk-size 1
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import components.pipelines.ingestion.CryptoL2.datapaths as paths

# Directories skipped when searching for the data layer (no source in them).
_SEARCH_SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "market_data",
    "docs",
    "build",
    "dist",
}


def _discover_data_layer_package() -> Optional[str]:
    """Return the dotted package of a `.../MoE/data/` holding `live.py`.

    E.g. `<repo>/strategy_system/strategy/MoE/data/live.py` ->
    `strategy_system.strategy.MoE.data`. Returns None when not found.
    """
    root = paths.PROJECT_ROOT
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in _SEARCH_SKIP and not d.startswith(".")
        ]
        if Path(dirpath).name != "data" or "live.py" not in filenames:
            continue
        package_dir = Path(dirpath)
        rel = package_dir.relative_to(root)
        if all(part.isidentifier() for part in rel.parts):
            return ".".join(rel.parts)
    return None


def _resolve_data_layer():
    """Return the (cfg, live) modules of the MoE data layer.

    `cfg` is None when the data layer ships no `dataconfig` module.

    Raises ImportError explaining exactly what was found and what failed.
    """
    package = _discover_data_layer_package()
    if package is None:
        raise ImportError(
            "Could not find a MoE data layer (a directory named 'data' "
            "containing live.py, e.g. <repo>/*/strategy/MoE/data/live.py).\n"
            "Matched data cannot be rebuilt without it."
        )

    try:
        live = importlib.import_module(f"{package}.live")
    except ImportError as exc:
        raise ImportError(
            f"Found the MoE data layer '{package}' but importing it failed:\n"
            f"  {exc}\n\n"
            "This is a strategy-layer import problem (its own module imports), "
            "not an ingestion path problem. The ingestion trees are intact."
        ) from exc

    try:
        cfg = importlib.import_module(f"{package}.dataconfig")
    except ImportError:
        cfg = None

    return cfg, live


def main(argv: Optional[list[str]] = None) -> int:
    # Running this file directly puts only ingestion_L2 on sys.path; the
    # discovered data-layer package is imported relative to the repo root.
    if str(paths.PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(paths.PROJECT_ROOT))

    parser = argparse.ArgumentParser(
        description="Rebuild matched aggTrades x LOB parquet files from the newest parquet tree."
    )
    parser.add_argument(
        "--root",
        default=str(paths.MARKET_DATA_ROOT),
        help="market_data root (default: %(default)s)",
    )
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument(
        "--start",
        default=None,
        help="Start date (default: first aggTrades day present in the parquet tree)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (default: last aggTrades day present in the parquet tree)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Hours processed per pass; 1 keeps peak memory at ~one hourly LOB "
        "file (default: %(default)s)",
    )
    parser.add_argument(
        "--lob-levels",
        type=int,
        default=20,
        help="Number of LOB price levels per side to reconstruct and attach to "
        "each trade (default: %(default)s; 1 = top-of-book only)",
    )
    parser.add_argument(
        "--include-bookdepth",
        action="store_true",
        help="Also asof-join Binance bookDepth cumulative band depth "
        "(+-1..5%% of mid) as bd_* columns (optional, off by default)",
    )
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()

    # Paths always come from the ingestion layout, so a refreshed build follows
    # the current tree even if the strategy config still holds older paths.
    agg_path = paths.parquet_dir(args.root, symbol, "aggTrades")
    lob_path = paths.parquet_dir(args.root, symbol, "orderbook")
    bookdepth_path = paths.parquet_dir(args.root, symbol, "bookDepth")

    start, end = args.start, args.end
    if start is None or end is None:
        available = paths.available_range(args.root, symbol)
        if available is None:
            print(f"No aggTrades parquet files under {agg_path}; nothing to match.")
            return 1
        start = start or available[0]
        end = end or available[1]

    try:
        cfg, live = _resolve_data_layer()
    except ImportError as exc:
        print(str(exc))
        return 2

    # The matched output directory is owned by the strategy-side config; when
    # it is absent or unset, fall back to the ingestion layout's matched tree.
    matched_out = (
        getattr(cfg, "MATCHED_PATH", None) if cfg is not None else None
    ) or paths.matched_dir(args.root, symbol)

    print(f"Refreshing matched data: {symbol} {start} -> {end} (chunk_size={args.chunk_size})")
    print(f"aggTrades: {agg_path}")
    print(f"LOB:       {lob_path}")
    if args.include_bookdepth:
        print(f"bookDepth: {bookdepth_path} (joined; target comes from the data layer)")
    else:
        print("bookDepth: not joined")
    print(f"LOB levels per side: {args.lob_levels}")
    print(f"Output:    {matched_out}")
    if cfg is None:
        print(
            "Note: the data layer has no `dataconfig`; its own defaults are "
            "used for anything not passed here."
        )

    live.fetch_source_data(
        agg_path=agg_path,
        lob_path=lob_path,
        start=start,
        end=end,
        chunk_size=args.chunk_size,
        lob_levels=args.lob_levels,
        include_bookdepth=args.include_bookdepth,
    )

    print("Refresh complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
refresh_matched.py
==================

Rebuild the matched aggTrades x top-of-book LOB parquet files in
research/data_v2/matched/<SYMBOL>/ from the *newest* source data.

Thin wrapper around research.MoE.data.live.fetch_source_data so the refresh
can be run as a single (backgroundable) command instead of ad-hoc snippets.

Usage
-----
    python scripts/refresh_matched.py                     # dataconfig defaults
    python scripts/refresh_matched.py --chunk-size 1      # one hour per pass (low memory)
    python scripts/refresh_matched.py --start 2026-02-01 --end 2026-02-02


import argparse
import sys
from pathlib import Path

# Make `research.*` importable when run as a plain script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.MoE.data import dataconfig as cfg  # noqa: E402
from research.strategy.MoE.data import live  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild matched aggTrades x LOB parquet files from the newest source data."
    )
    parser.add_argument("--start", default=cfg.START_DATE, help="Start date (default: dataconfig.START_DATE)")
    parser.add_argument("--end", default=cfg.END_DATE, help="End date (default: dataconfig.END_DATE)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Hours processed per pass; 1 keeps peak memory at ~one hourly LOB file (default: 1)",
    )
    parser.add_argument(
        "--lob-levels",
        type=int,
        default=cfg.LOB_LEVELS,
        help="Number of LOB price levels per side to reconstruct and attach to "
        "each trade (default: dataconfig.LOB_LEVELS = 20; 1 = top-of-book only)",
    )
    parser.add_argument(
        "--include-bookdepth",
        action="store_true",
        help="Also asof-join Binance bookDepth cumulative band depth "
        "(+-1..5%% of mid) as bd_* columns (optional, off by default)",
    )
    args = parser.parse_args()

    print(f"Refreshing matched data: {cfg.TICKER} {args.start} -> {args.end} (chunk_size={args.chunk_size})")
    print(f"aggTrades: {cfg.AGG_TRADES_PATH}")
    print(f"LOB:       {cfg.LOB_PATH}")
    print(f"LOB levels per side: {args.lob_levels}")
    print(f"bookDepth: {'INCLUDED' if args.include_bookdepth else 'off (optional)'}")
    print(f"Output:    {cfg.MATCHED_PATH}")

    live.fetch_source_data(
        start=args.start,
        end=args.end,
        chunk_size=args.chunk_size,
        lob_levels=args.lob_levels,
        include_bookdepth=args.include_bookdepth,
    )

    print("Refresh complete.")


if __name__ == "__main__":
    main()
"""