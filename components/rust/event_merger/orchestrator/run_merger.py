"""
run_merger.py
=============
Parallel multi-day merge orchestrator with finalize and safe input deletion.

Role in Pipeline:
    Runs the Rust `event_merger` binary across a date range, then finalizes
    each date: the sparse merge is forward-filled so TRADE/BOOKDEPTH rows
    inherit the last LOB book state, producing merged_<DATE>_<HH>.parquet.
    Inputs are deleted ONLY after that succeeds for every hour of the date. 
    ThreadPoolExecutor. The as-of carry is written to
    _pipeline/state/<DATE>/<HH>.json so later hours keep the cross-hour state. 
    are deleted ONLY if the Rust binary and the finalize stage both succeed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# Script entry: put the repo root on sys.path before importing components.*.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components.python.data_pipeline.matched import merge as legacy_merge

STATE_PREFIXES = legacy_merge.STATE_PREFIXES


def resolve_binary() -> Path:
    """
    Locate the compiled ``event_merger`` executable.

    Raises:
        FileNotFoundError: if no binary exists, with the build command to run.
    """
    ext = ".exe" if sys.platform.startswith("win") else ""
    components_root = Path(__file__).resolve().parents[3]
    candidates = [
        components_root / "target" / "release" / f"event_merger{ext}",
        components_root / "target" / "debug" / f"event_merger{ext}",
        components_root / "target" / "release" / "event_merger",
        components_root / "target" / "debug" / "event_merger",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "event_merger binary not found; run:\n"
        "  cargo build --manifest-path components/Cargo.toml --bin event_merger --release"
    )


def orderflow_dir(root: str | Path, symbol: str, date: str) -> Path:
    """``<root>/matched/<TICKER>/_pipeline/orderflow/<DATE>`` — every-event replay output."""
    return Path(root) / "matched" / symbol.upper() / "_pipeline" / "orderflow" / date


def sparse_dir(root: str | Path, symbol: str, date: str) -> Path:
    """Directory holding the merger's per-hour sparse output for one date."""
    return Path(root) / "matched" / symbol.upper() / "sparse" / date


def build_command(
    *,
    root: str | Path,
    symbol: str,
    date: str,
    hour: int | None = None,
    lob_levels: int = 20,
    source: str = "matched",
    matched_lob_root: str | Path | None = None,
) -> list[str]:
    """
    Assemble an ``event_merger`` argument vector.

    ``matched_lob_root`` points the merger at the every-event LOB replay output.
    Without it the merger falls back to its candidate list and can pick up the
    *raw* orderbook (string ``side``/``price``, no ``is_bid``), which fails with
    ``LOB is_bid is null``.
    """
    command = [
        str(resolve_binary()),
        "--root", str(root),
        "--symbol", symbol.upper(),
        "--source", source,
        "--date", date,
        "--hour", "all" if hour is None else str(hour),
        "--lob-levels", str(lob_levels),
    ]
    if matched_lob_root is not None:
        command.extend(["--matched-lob-root", str(matched_lob_root)])
    return command


def run_merge(
    *,
    root: str | Path,
    symbol: str,
    date: str,
    hour: int | None = None,
    lob_levels: int = 20,
    source: str = "matched",
    matched_lob_root: str | Path | None = None,
) -> list[Path]:
    """Run ``event_merger`` for one date and return the sparse files it wrote."""
    root_p = Path(root)
    symbol = symbol.upper()
    if matched_lob_root is None:
        default_lob_root = orderflow_dir(root_p, symbol, date)
        if default_lob_root.is_dir():
            matched_lob_root = default_lob_root

    command = build_command(
        root=root_p,
        symbol=symbol,
        date=date,
        hour=hour,
        lob_levels=lob_levels,
        source=source,
        matched_lob_root=matched_lob_root,
    )
    print(f"[MERGE] {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        raise RuntimeError(
            f"event_merger failed for {date} (exit {result.returncode})\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    found = find_sparse_hours(root=root_p, symbol=symbol, date=date)
    if hour is not None:
        found = {h: p for h, p in found.items() if h == hour}
    if not found:
        raise RuntimeError(
            f"event_merger reported success for {date} but wrote no sparse output "
            f"under {sparse_dir(root_p, symbol, date)}"
        )
    return [found[h] for h in sorted(found)]


def find_sparse_hours(*, root: str | Path, symbol: str, date: str) -> dict[int, Path]:
    """
    Map each hour to its sparse merge file for ``date``, tolerating both
    on-disk layouts.

    The compiled ``event_merger`` writes ``sparse/<DATE>/<HH>.parquet`` while the
    current ``merge.rs`` source would write ``sparse/<DATE>_<HH>.parquet``. The
    binary can lag its source, so both are probed rather than assumed — otherwise
    a rebuild silently breaks finalize. When both exist for an hour the newer
    file wins.
    """
    base = Path(root) / "matched" / symbol.upper() / "sparse"
    candidates: dict[int, list[tuple[Path, float]]] = {}

    def record(hour: int, path: Path) -> None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        candidates.setdefault(hour, []).append((path, mtime))

    nested = base / date
    if nested.is_dir():
        for path in nested.glob("*.parquet"):
            if len(path.stem) == 2 and path.stem.isdigit():
                record(int(path.stem), path)

    prefix = f"{date}_"
    if base.is_dir():
        for path in base.glob(f"{date}_*.parquet"):
            suffix = path.stem[len(prefix):]
            if len(suffix) == 2 and suffix.isdigit():
                record(int(suffix), path)

    return {
        hour: max(entries, key=lambda item: item[1])[0]
        for hour, entries in candidates.items()
    }


def _has_lob_book_state(lob_root: str | Path) -> bool:
    """
    True when ``lob_root`` holds at least one non-empty every-event book-state
    parquet.

    Used as a pre-flight guard: pointing the merger at an empty or missing
    directory makes it fall back to the raw orderbook, which lacks ``is_bid`` and
    aborts with a confusing ``LOB is_bid is null``.
    """
    root = Path(lob_root)
    if root.is_file():
        return root.stat().st_size > 0
    if not root.is_dir():
        return False
    for path in root.rglob("*.parquet"):
        try:
            if path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _carry_state(*, root: Path, symbol: str, date: str, hour: int) -> dict:
    """
    State to seed hour ``hour`` with: the previous hour's saved JSON when
    present, otherwise the last row of the previous hour's merged output.
    """
    state_path = legacy_merge._state_path(root, symbol, date, hour)
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return legacy_merge._previous_state(root, symbol, date, hour)


def finalize_date(
    *,
    root: str | Path,
    symbol: str,
    date: str,
    hours: list[int] | None = None,
) -> list[Path]:
    """
    Turn the merger's sparse output into the final hourly deliverables.

    For each hour in ascending order: forward-fill ``book_state__*`` and
    ``depth_state__*`` so TRADE / BOOKDEPTH rows inherit the most recent LOB
    state, write ``merged_<DATE>_<HH>.parquet``, and persist the carry to
    ``_pipeline/state/<DATE>/<HH>.json`` for the next hour.

    Hours are processed sequentially on purpose — the cross-hour carry depends
    on it.
    """
    root_p = Path(root)
    symbol = symbol.upper()
    found = find_sparse_hours(root=root_p, symbol=symbol, date=date)
    if not found:
        raise FileNotFoundError(
            f"no sparse merge output for {date} under {sparse_dir(root_p, symbol, date)}"
        )

    targets = sorted(found) if hours is None else sorted(hours)
    missing = [h for h in targets if h not in found]
    if missing:
        raise FileNotFoundError(f"no sparse merge output for {date} hours {missing}")

    outputs: list[Path] = []
    for hour in targets:
        sparse_path = found[hour]
        frame = pl.read_parquet(sparse_path)
        if frame.height == 0:
            raise ValueError(f"empty sparse merge: {sparse_path}")
        if frame["timestamp"].null_count():
            raise ValueError(f"null timestamp in sparse merge: {sparse_path}")

        state_columns = [c for c in frame.columns if c.startswith(STATE_PREFIXES)]
        if not state_columns:
            raise ValueError(
                f"no book_state__/depth_state__ columns in {sparse_path}; "
                "the merge ran without a usable LOB input"
            )

        carry = _carry_state(root=root_p, symbol=symbol, date=date, hour=hour)
        filled, carry = legacy_merge._ffill_hour(frame, carry)

        nulls = sum(int(filled[c].null_count()) for c in state_columns)
        if nulls:
            lob_rows = (
                int((frame["event_type"] == "LOB").sum())
                if "event_type" in frame.columns
                else 0
            )
            if lob_rows == 0:
                print(
                    f"  [FINALIZE] {date} {hour:02d}: WARNING no LOB book state "
                    f"({nulls} null(s) across TRADE/BOOKDEPTH rows)"
                )
            else:
                print(f"  [FINALIZE] {date} {hour:02d}: {nulls} state null(s) after ffill")

        output = legacy_merge.final_path(root=root_p, symbol=symbol, date=date, hour=hour)
        temp = output.with_suffix(".tmp.parquet")
        filled.write_parquet(temp)
        temp.replace(output)

        state_path = legacy_merge._state_path(root_p, symbol, date, hour)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(carry, sort_keys=True), encoding="utf-8")
        outputs.append(output)
        print(f"  [FINALIZE] {date} {hour:02d} -> {output.name} ({frame.height} rows)")
    return outputs


def cleanup_date(
    *,
    root: str | Path,
    symbol: str,
    date: str,
) -> None:
    """
    Remove the per-hour inputs behind a completed date, leaving only the
    ``merged_<DATE>_<HH>.parquet`` deliverables.

    Drops the cloned ``_pipeline/source`` files, the every-event ``orderflow``
    replay output, the sparse intermediates (both layouts) and the carry state.
    Canonical ``ingest_parquet`` is never touched.
    """
    root_p = Path(root)
    symbol = symbol.upper()
    work = root_p / "matched" / symbol / "_pipeline"

    targets: list[Path] = [
        work / "source" / "orderbook" / date,
        work / "source" / "aggTrades" / f"{date}.parquet",
        work / "source" / "bookDepth" / f"{date}.parquet",
        work / "orderflow" / date,
        work / "state" / date,
        work / "sparse" / date,
        root_p / "matched" / symbol / "sparse" / date,
    ]
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    # Flat sparse layout written by newer event_merger builds: <DATE>_<HH>.parquet
    flat = root_p / "matched" / symbol / "sparse"
    if flat.is_dir():
        for path in flat.glob(f"{date}_*.parquet"):
            path.unlink(missing_ok=True)

    stop = work if work.exists() else root_p / "matched" / symbol
    for path in (
        work / "source" / "orderbook" / date,
        work / "orderflow" / date,
        work / "state" / date,
        work / "sparse" / date,
        root_p / "matched" / symbol / "sparse" / date,
    ):
        legacy_merge._prune_empty(path, stop)
    print(f"  [CLEANUP] {date}: per-hour work inputs removed")


def merge_and_finalize(
    *,
    root: str | Path,
    symbol: str,
    date: str,
    hour: int | None = None,
    lob_levels: int = 20,
    source: str = "matched",
    matched_lob_root: str | Path | None = None,
    delete_inputs: bool = True,
) -> list[Path]:
    """
    Full stage 4 for one date: run ``event_merger``, forward-fill the state, and
    — unless ``delete_inputs`` is False — remove the per-hour inputs behind it.

    Deletion happens only after finalize has succeeded for every merged hour.
    """
    if matched_lob_root is None:
        default_lob_root = orderflow_dir(root, symbol, date)
        if default_lob_root.is_dir():
            matched_lob_root = default_lob_root
    if matched_lob_root is None or not _has_lob_book_state(matched_lob_root):
        raise FileNotFoundError(
            f"no every-event LOB book state for {date} under "
            f"{orderflow_dir(root, symbol, date)}; run the LOB replay stage first "
            "(components/commands/01_ingest.py, or pipeline.py). Without it the "
            "merger falls back to the raw orderbook and fails with "
            "'LOB is_bid is null'."
        )

    run_merge(
        root=root,
        symbol=symbol,
        date=date,
        hour=hour,
        lob_levels=lob_levels,
        source=source,
        matched_lob_root=matched_lob_root,
    )
    outputs = finalize_date(root=root, symbol=symbol, date=date)
    if delete_inputs:
        cleanup_date(root=root, symbol=symbol, date=date)
    return outputs


def generate_dates(start_date: str, end_date: str):
    """
    Generates an inclusive sequence of YYYY-MM-DD date strings between start and end.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    curr = start
    while curr <= end:
        yield curr.strftime("%Y-%m-%d")
        curr += timedelta(days=1)


def _run_date(args: argparse.Namespace, root_p: Path, date_str: str) -> list[Path]:
    return merge_and_finalize(
        root=root_p,
        symbol=args.symbol,
        date=date_str,
        hour=args.hour,
        lob_levels=args.lob_levels,
        source=args.source,
        matched_lob_root=args.matched_lob_root,
        delete_inputs=not args.keep_work_inputs,
    )


def main(argv: list[str] | None = None) -> int:
    """Merge (and finalize) an inclusive date range."""
    parser = argparse.ArgumentParser(
        description="Run the Rust event_merger over a date range and finalize hourly state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="market_data root folder.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g. BTCUSDT).")
    parser.add_argument("--start", required=True, help="First UTC date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--end", required=True, help="Last UTC date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--hour", type=int, default=None, help="0..23; omit for all hours.")
    parser.add_argument("--lob-levels", type=int, default=20, help="Book levels per side.")
    parser.add_argument(
        "--source",
        choices=["ingest_parquet", "matched"],
        default="matched",
        help="Where the three input streams are resolved from.",
    )
    parser.add_argument(
        "--matched-lob-root",
        default=None,
        help="Every-event LOB replay dir. Defaults to _pipeline/orderflow/<DATE> when present.",
    )
    parser.add_argument(
        "--keep-work-inputs",
        action="store_true",
        help="Keep cloned inputs, orderflow, sparse and state after a successful finalize.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Dates merged in parallel.")
    args = parser.parse_args(argv)

    if args.hour is not None and not 0 <= args.hour <= 23:
        parser.error("--hour must be 0..23")

    root_p = Path(args.root)
    try:
        resolve_binary()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    dates = list(generate_dates(args.start, args.end))
    if not dates:
        print("[ERROR] empty date range")
        return 1

    print(f"[MERGE] {len(dates)} date(s), {args.workers} worker(s), finalize + cleanup")
    failures: list[tuple[str, Exception]] = []

    if args.workers <= 1:
        for date_str in dates:
            try:
                _run_date(args, root_p, date_str)
            except Exception as exc:  # noqa: BLE001 - reported per date
                print(f"[FAILED] {date_str}: {exc}")
                failures.append((date_str, exc))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_date, args, root_p, d): d for d in dates}
            for future in futures:
                date_str = futures[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - reported per date
                    print(f"[FAILED] {date_str}: {exc}")
                    failures.append((date_str, exc))

    if failures:
        print(f"[ERROR] {len(failures)}/{len(dates)} date(s) failed; inputs preserved")
        return 1
    print(f"[SUCCESS] merged + finalized {len(dates)} date(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())