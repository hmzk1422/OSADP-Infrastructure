#!/usr/bin/env python3
"""
main.py
=======

The single entry point for the L2 ingestion component. Everything in
`components/python/ingestion_L2/` is reachable from here, and every module
stays runnable on its own (each exposes a `main(argv)` function).

    python main.py <command> [flags...]

Commands
--------
pipeline          download -> convert -> optional compress/refresh (full run)
download          raw acquisition only            (ofp_data_downloader.py)
convert           raw -> ingest_parquet tree      (convert_to_parquet.py)
compress          .parquet -> .parquet.zst        (zstandard_fix.py)
decompress        .parquet.zst -> .parquet        (zstandard_fix.py)
repair            fix mislabeled zstd .parquet    (zstandard_fix.py)
refresh-matched   rebuild matched/ from parquet   (refresh_matched.py)
scan-snapshots    census of snapshot hours        (scan_snapshots.py)
validate-replay   LOB replay invariant checks     (validate_replay.py)
migrate-layout    move a legacy provider-first ingest_source tree into the
                  ticker-first (data-type) layout and re-point the manifest
layout            print the resolved ingestion layout + inventory

Stage-specific flags are forwarded verbatim, so `main.py <command> --help`
prints that stage's own help.

Examples
--------
    python main.py layout
    python main.py pipeline --symbol BTCUSDT --start-date 2025-06-28 \\
        --end-date 2026-09-01 --workers 6
    python main.py pipeline --symbol BTCUSDT --start-date 2025-06-28 \\
        --end-date 2026-09-01 --datasets bookDepth,orderbook --overwrite
    python main.py download --symbol BTCUSDT --start-date 2026-09-02 \\
        --end-date 2026-09-19 --datasets aggTrades
    python main.py convert --symbol BTCUSDT
    python main.py migrate-layout --apply
    python main.py scan-snapshots --start 2025-06-28 --end 2025-07-01
    python main.py validate-replay --start 2026-02-01 --end 2026-02-02

Root selection: `--root` (or $MARKET_DATA_ROOT) names the `market_data`
directory that holds `ingest_source/`, `ingest_parquet/` and `matched/`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# BOOTSTRAP: make the component self-contained.
# This file may be executed as a script, run via `python -m`, or imported by
# another tool, so make sure its own directory (sibling stage modules) and the
# repo root (nested strategy namespace packages) are importable.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

for _path in (str(_REPO_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import components.python.data_pipeline.CryptoL2.datapaths as paths  # noqa: E402


# ===========================================================
# STAGE REGISTRY
# ===========================================================

# command -> (module, description)
STAGES = {
    "download": ("ofp_data_downloader", "ingestion: download raw files"),
    "convert": ("convert_to_parquet", "ingestion: convert raw -> parquet"),
    "refresh-matched": ("refresh_matched", "ingestion: refresh matched data"),
    "scan-snapshots": ("scan_snapshots", "ingestion: scan for snapshot hours"),
    "validate-replay": ("validate_replay", "ingestion: validate matched replay"),
}

# zstandard_fix.py is mode-driven, so it gets one wrapper command per mode.
ZSTD_COMMANDS = ("compress", "decompress", "repair")


def run_stage(label: str, module_name: str, argv: list[str]) -> int:
    """Import a stage module and run its `main(argv)`.

    Stage modules report failure either by returning a non-zero int or by
    raising SystemExit (argparse errors, `parser.error`, `sys.exit(1)`).
    Both are converted into a non-zero return code so `pipeline` can stop.
    """
    module = importlib.import_module(module_name)
    entry = getattr(module, "main")

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    try:
        rc = entry(argv)
    except SystemExit as exc:  # argparse errors and explicit sys.exit()
        code = exc.code
        rc = 0 if code in (0, None) else (code if isinstance(code, int) else 1)

    rc = int(rc or 0)
    if rc:
        print(f"{label} failed (exit {rc}).")
    return rc

# ===========================================================
# COMMAND DESCRIPTIONS (single source for the help text)
# ===========================================================

COMMANDS = {
    "pipeline": "download -> convert -> optional compress/refresh (full run)",
    "download": "download raw Binance + CryptoHFTData files",
    "convert": "normalize raw files into the ingest_parquet tree",
    "compress": ".parquet -> .parquet.zst (archive orderbook files)",
    "decompress": ".parquet.zst -> .parquet (restore orderbook files)",
    "repair": "decompress mislabeled zstd streams sitting at *.parquet",
    "refresh-matched": "rebuild matched/ from the ingest_parquet tree",
    "scan-snapshots": "scan orderbook hours for snapshot events",
    "validate-replay": "validate replayed LOB invariants in matched/",
    "migrate-layout": "move a legacy provider-first ingest_source tree to the "
                      "ticker-first layout (+ re-point the manifest)",
    "layout": "print the resolved ingestion layout and inventory",
}


def help_text() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = [
        "L2 ingestion executor.",
        "",
        "usage: python main.py <command> [flags...]",
        "",
        "commands:",
    ]
    for cmd, desc in COMMANDS.items():
        lines.append(f"  {cmd:<{width}}  {desc}")
    lines += [
        "",
        "Stage flags are forwarded verbatim: `main.py <command> --help` prints",
        "that stage's own options. Root selection: --root / $MARKET_DATA_ROOT",
        "names the market_data directory holding ingest_source/, ingest_parquet/",
        "and matched/.",
    ]
    return "\n".join(lines)


# ===========================================================
# migrate-layout helpers
# ===========================================================

def _move_file(src: Path, dst: Path) -> str:
    """Move a file, tolerating a pre-existing identical destination.

    Returns "moved", "exists" (destination already holds the same number of
    bytes) or "conflict" (destination exists with a different size, so nothing
    is touched).
    """
    if dst.exists():
        return "exists" if dst.stat().st_size == src.stat().st_size else "conflict"

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.replace(dst)
    except OSError:
        # Cross-volume fallback: copy then remove.
        import shutil

        shutil.copy2(src, dst)
        src.unlink(missing_ok=True)
    return "moved"


def _prune_empty_dirs(start: Path, stop: Path) -> None:
    """Remove every empty directory at or below `start`, never `stop` itself.

    Deepest-first, because emptying a leaf directory is what makes its parent
    removable. The old top-down walk only looked at `start` itself, so a
    provider dir holding empty `<dataset>/<DATE>/<HH>` shells survived.
    """
    stop = stop.resolve()
    start_res = start.resolve()

    dirs = [
        p.resolve()
        for p in start.rglob("*")
        if p.is_dir()
    ]
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        if d == stop or stop not in d.parents:
            continue
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            continue

    if start_res != stop and start_res.is_dir() and not any(start_res.iterdir()):
        try:
            start_res.rmdir()
        except OSError:
            pass

def migrate_layout(root: str, symbol: str, apply: bool) -> int:
    """Move a legacy provider-first raw tree to `<TICKER>/<dataset>/...`.

    Legacy:   ingest_source/<provider>/<TICKER>/<dataset>/<DATE>/...
    Current:  ingest_source/<TICKER>/<dataset>/<DATE>/...
    """
    symbol = symbol.upper()
    src_root = paths.source_root(root)
    if not src_root.is_dir():
        print(f"No ingest_source tree at {src_root}; nothing to migrate.")
        return 0

    keep = {paths.MANIFESTS_DIRNAME, symbol}
    legacy_providers = sorted(
        p for p in src_root.iterdir() if p.is_dir() and p.name not in keep
    )
    if not legacy_providers:
        print(f"Layout already ticker-first; no provider dirs under {src_root}.")
        _repoint_manifest(root, symbol, apply)
        return 0

    planned: list[tuple[Path, Path]] = []
    for provider in legacy_providers:
        for sym_dir in sorted(p for p in provider.iterdir() if p.is_dir()):
            for ds_dir in sorted(p for p in sym_dir.iterdir() if p.is_dir()):
                target = paths.source_dir(root, sym_dir.name, ds_dir.name)
                for f in sorted(p for p in ds_dir.rglob("*") if p.is_file()):
                    planned.append((f, target / f.relative_to(ds_dir)))

    print(f"Root: {src_root}")
    print(f"Legacy provider dirs: {[p.name for p in legacy_providers]}")
    print(f"Files to re-home: {len(planned):,}")
    for src, dst in planned[:10]:
        print(f"  {src}  ->  {dst}")
    if len(planned) > 10:
        print(f"  ... and {len(planned) - 10:,} more")

    if not apply:
        print("\nDry run. Re-run with --apply to perform the move.")
        return 0

    moved = exists = conflicts = 0
    for src, dst in planned:
        outcome = _move_file(src, dst)
        if outcome == "moved":
            moved += 1
        elif outcome == "exists":
            exists += 1
            src.unlink(missing_ok=True)
        else:
            conflicts += 1
            print(f"  CONFLICT (size mismatch, kept both): {src} vs {dst}")

    for provider in legacy_providers:
        _prune_empty_dirs(provider, stop=src_root)

    print(f"\nMoved={moved:,} already_present={exists:,} conflicts={conflicts:,}")
    remaining = sorted(
        p.name
        for p in src_root.iterdir()
        if p.is_dir() and p.name not in (paths.MANIFESTS_DIRNAME, symbol)
    )
    if remaining:
        print(f"Remaining provider dirs (not empty, inspect manually): {remaining}")

    _repoint_manifest(root, symbol, apply)
    return 0


def _repoint_manifest(root: str, symbol: str, apply: bool) -> None:
    """Rewrite manifest entry paths onto the current ticker-first layout.

    Entries recorded against the legacy `<provider>/<TICKER>/<dataset>/...`
    (or `research/data_v2/...`) tree are re-pointed to the canonical path for
    their (provider, dataset, date, hour). The original is preserved as
    `legacy_path` so recorded provenance survives.
    """
    mf = paths.manifest_file(root, symbol)
    if not mf.exists():
        print(f"No manifest at {mf}; nothing to re-point.")
        return

    try:
        payload = json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Manifest unreadable ({exc!r}); skipping.")
        return

    entries = payload.get("results") or []
    changed = 0
    for entry in entries:
        try:
            canonical = paths.source_file_for(
                root=root,
                symbol=symbol,
                provider=entry.get("provider") or "binance",
                dataset=entry.get("dataset") or "",
                day=entry.get("date") or "",
                hour=entry.get("hour"),
            )
        except (TypeError, ValueError):
            continue
        old = entry.get("path")
        new = str(canonical)
        if old != new:
            entry["legacy_path"] = old
            entry["path"] = new
            changed += 1

    payload["layout_version"] = 2
    payload["layout"] = "ingest_source/<TICKER>/<dataset>/..."
    payload["symbol"] = symbol

    print(f"Manifest entries: {len(entries):,}; re-pointed: {changed:,}")

    if not apply:
        print("Dry run. Re-run with --apply to write the manifest.")
        return

    tmp = mf.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, mf)
    print(f"Manifest -> {mf}")

# ===========================================================
# layout / migrate-layout CLI
# ===========================================================

def cmd_layout(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py layout",
        description="Print the resolved ingestion layout and its inventory.",
    )
    parser.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    args = parser.parse_args(argv)

    print(paths.describe(args.root, args.symbol))
    return 0


def cmd_migrate_layout(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py migrate-layout",
        description="Move a legacy provider-first ingest_source tree into the "
                    "ticker-first (data-type) layout and re-point the manifest.",
    )
    parser.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the move (default: dry run, printing the plan)",
    )
    args = parser.parse_args(argv)
    return migrate_layout(args.root, args.symbol, args.apply)


# ===========================================================
# pipeline
# ===========================================================

def cmd_pipeline(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py pipeline",
        description="Full ingestion run: download -> convert -> optional post-steps.",
    )
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD UTC inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD UTC inclusive")
    parser.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    parser.add_argument("--market", choices=["um", "cm"], default="um")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--datasets",
        default=",".join(paths.DATASETS),
        help="Comma-separated subset processed by both stages "
             f"(default: {','.join(paths.DATASETS)})",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild parquet files that already exist")
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete raw sources after a verified conversion "
             "(default: keep them, i.e. convert passes --keep-raw)",
    )
    parser.add_argument(
        "--recompress-chd",
        action="store_true",
        help="Rewrite CryptoHFTData files with zstd during conversion "
             "(default: move through unchanged)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="After conversion, recompress orderbook parquet to .parquet.zst "
             "(reclaims disk space; run `decompress` before reading them)",
    )
    parser.add_argument("--compress-level", type=int, default=15)
    parser.add_argument(
        "--refresh-matched",
        action="store_true",
        help="After conversion, rebuild matched/ for the same window",
    )
    parser.add_argument("--lob-levels", type=int, default=20)
    parser.add_argument("--include-bookdepth", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    root = args.root

    print(paths.describe(root, symbol))
    print(f"\nWindow: {args.start_date} -> {args.end_date}   datasets: {args.datasets}")

    # ---- Stage 1: raw acquisition --------------------------------------
    download_argv = [
        "--symbol", symbol,
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--root", root,
        "--market", args.market,
        "--workers", str(args.workers),
        "--datasets", args.datasets,
    ]
    if not args.verbose:
        download_argv.append("--quiet")
    rc = run_stage("STAGE 1 - RAW DATA INGESTION", STAGES["download"][0], download_argv)
    if rc:
        return rc

    # ---- Stage 2: normalize into the parquet tree -----------------------
    convert_argv = [
        "--root", root,
        "--symbol", symbol,
        "--datasets", args.datasets,
    ]
    if args.overwrite:
        convert_argv.append("--overwrite")
    if args.recompress_chd:
        convert_argv.append("--recompress")
    if not args.delete_raw:
        convert_argv.append("--keep-raw")
    if not args.verbose:
        convert_argv.append("--quiet")
    rc = run_stage("STAGE 2 - PARQUET CONVERSION", STAGES["convert"][0], convert_argv)
    if rc:
        return rc

# ---- Stage 3 (optional): archival compression -----------------------
    if args.compress:
        orderbook_parquet = paths.parquet_dir(root, symbol, "orderbook")
        rc = run_stage(
            "STAGE 3 - ZSTD ARCHIVAL COMPRESSION",
            "zstandard_fix",
            [
                "compress",
                "--level", str(args.compress_level),
                "--root", str(orderbook_parquet),
            ],
        )
        if rc:
            return rc
        print(
            "\nNote: orderbook parquet files are now .parquet.zst archives. "
            "Run `main.py decompress` before reading them."
        )

    # ---- Stage 4 (optional): matched data -------------------------------
    if args.refresh_matched:
        refresh_argv = [
            "--root", root,
            "--symbol", symbol,
            "--start", args.start_date,
            "--end", args.end_date,
            "--chunk-size", str(args.chunk_size),
            "--lob-levels", str(args.lob_levels),
        ]
        if args.include_bookdepth:
            refresh_argv.append("--include-bookdepth")
        rc = run_stage(
            "STAGE 4 - MATCHED DATA REFRESH",
            STAGES["refresh-matched"][0],
            refresh_argv,
        )
        if rc:
            return rc

    print(f"\nIngestion completed successfully. Tree: {paths.parquet_root(root)}")
    return 0


# ===========================================================
# DISPATCH
# ===========================================================

def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(help_text())
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "layout":
        return cmd_layout(rest)
    if cmd == "pipeline":
        return cmd_pipeline(rest)
    if cmd == "migrate-layout":
        return cmd_migrate_layout(rest)
    if cmd in ZSTD_COMMANDS:
        return run_stage(f"ZSTANDARD {cmd.upper()}", "zstandard_fix", [cmd, *rest])
    if cmd in STAGES:
        module, description = STAGES[cmd]
        return run_stage(description.upper(), module, rest)

    print(f"Unknown command: {cmd!r}\n")
    print(help_text())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())