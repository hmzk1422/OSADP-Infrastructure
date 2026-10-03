"""Run Rust event merger and finalize hourly state with cross-hour carry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

import polars as pl

import components.pipelines.ingestion.CryptoL2.datapaths as paths
from components.pipelines.event_merger.matched.parquet import pipeline_work_dir

EVENT_MERGER_BIN = Path("components/target/debug/event_merger.exe")
STATE_PREFIXES = ("book_state__", "depth_state__")


def _resolve_bin() -> Path:
    for candidate in (
        Path("components/target/release/event_merger.exe"),
        EVENT_MERGER_BIN,
        Path("components/target/release/event_merger"),
        Path("components/target/debug/event_merger"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("event_merger binary not found; run cargo build --bin event_merger")


def sparse_path(*, root: Path, symbol: str, date: str, hour: int) -> Path:
    return root / "matched" / symbol.upper() / "sparse" / date / f"{hour:02d}.parquet"


def final_path(*, root: Path, symbol: str, date: str, hour: int) -> Path:
    return root / "matched" / symbol.upper() / f"merged_{date}_{hour:02d}.parquet"


def _load_state(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _state_path(root: Path, symbol: str, date: str, hour: int) -> Path:
    return pipeline_work_dir(root, symbol) / "state" / date / f"{hour:02d}.json"


def _previous_state(root: Path, symbol: str, date: str, hour: int) -> dict[str, float | None]:
    for prior_hour in range(hour - 1, -1, -1):
        previous = _state_path(root, symbol, date, prior_hour)
        if previous.exists():
            return _load_state(previous)
        output = final_path(root=root, symbol=symbol, date=date, hour=prior_hour)
        if output.exists():
            frame = pl.read_parquet(output)
            state_columns = [c for c in frame.columns if c.startswith(STATE_PREFIXES)]
            if frame.height and state_columns:
                last = frame.row(frame.height - 1, named=True)
                return {column: last.get(column) for column in state_columns}
    return {}


def _ffill_hour(
    frame: pl.DataFrame, carry: dict[str, float | None],
) -> tuple[pl.DataFrame, dict[str, float | None]]:
    state_columns = [c for c in frame.columns if c.startswith(STATE_PREFIXES)]
    if not state_columns:
        return frame, carry
    expressions = []
    for column in state_columns:
        expression = pl.col(column)
        if column in carry and carry[column] is not None:
            expression = expression.fill_null(carry[column])
        expressions.append(expression.alias(column))
    result = frame.with_columns(expressions)
    result = result.with_columns(pl.col(state_columns).forward_fill())
    if result.height:
        last = result.row(result.height - 1, named=True)
        carry = {column: last.get(column) for column in state_columns}
    return result, carry


def _prune_empty(path: Path, stop: Path) -> None:
    """Remove empty generated directories without crossing ``stop``."""
    stop = stop.resolve()
    current = (path if path.is_dir() else path.parent).resolve()
    while current != stop and current.is_dir():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _cleanup_work_inputs(
    *, root: Path, symbol: str, date: str, hours: list[int], all_hours: bool,
) -> None:
    work = pipeline_work_dir(root, symbol)
    for hour in hours:
        (work / "lob_state" / date / f"{hour:02d}.parquet").unlink(missing_ok=True)
        (work / "source" / "orderbook" / date / f"{hour:02d}.parquet").unlink(missing_ok=True)
        (root / "matched" / symbol.upper() / "sparse" / date / f"{hour:02d}.parquet").unlink(missing_ok=True)
    if all_hours:
        for path in (
            work / "source" / "aggTrades" / f"{date}.parquet",
            work / "source" / "bookDepth" / f"{date}.parquet",
            work / "day_seed" / f"{date}.parquet",
            work / "day_seed_state" / date,
            work / "sparse" / date,
            work / "state" / date,
        ):
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    for path in (
        work / "lob_state" / date,
        work / "source" / "orderbook" / date,
        work / "source" / "aggTrades",
        work / "source" / "bookDepth",
        work / "day_seed" / date,
        work / "day_seed_state" / date,
        work / "sparse" / date,
        work / "state" / date,
        root / "matched" / symbol.upper() / "sparse" / date,
    ):
        _prune_empty(path, work if path.is_relative_to(work) else root / "matched" / symbol.upper())


def _finalize_file(
    *, root: Path, symbol: str, date: str, hour: int, state_path: Path,
) -> Path:
    sparse = sparse_path(root=root, symbol=symbol, date=date, hour=hour)
    output = final_path(root=root, symbol=symbol, date=date, hour=hour)
    if not sparse.exists():
        raise FileNotFoundError(sparse)
    frame = pl.read_parquet(sparse)
    if frame.height == 0:
        raise ValueError(f"empty sparse merge: {sparse}")
    if frame["timestamp"].null_count():
        raise ValueError(f"null timestamp in sparse merge: {sparse}")
    carry = _previous_state(root, symbol, date, hour)
    filled, carry = _ffill_hour(frame, carry)
    temp = output.with_suffix(".tmp.parquet")
    filled.write_parquet(temp)
    temp.replace(output)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(carry, sort_keys=True), encoding="utf-8")
    return output


def run_merge(
    *, root=None, symbol="BTCUSDT", source="ingest_parquet", date: str,
    hour: int | None = None, lob_levels=20, keep_work_inputs=False,
    matched_lob_root=None,
) -> list[Path]:
    root_p = Path(root) if root is not None else paths.MARKET_DATA_ROOT
    symbol = symbol.upper()
    command = [
        str(_resolve_bin()), "--root", str(root_p), "--symbol", symbol,
        "--source", source, "--date", date, "--hour", "all" if hour is None else str(hour),
        "--lob-levels", str(lob_levels),
    ]
    if matched_lob_root is not None:
        command.extend(["--matched-lob-root", str(matched_lob_root)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"event_merger failed:\n{result.stderr[-4000:]}")
    print(result.stdout.strip())
    hours = [hour] if hour is not None else sorted(p.stem for p in (root_p / "matched" / symbol / "sparse" / date).glob("*.parquet"))
    outputs = [
        _finalize_file(
            root=root_p, symbol=symbol, date=date, hour=int(h),
            state_path=_state_path(root_p, symbol, date, int(h)),
        )
        for h in hours
    ]
    _cleanup_work_inputs(
        root=root_p, symbol=symbol, date=date, hours=[int(h) for h in hours], all_hours=hour is None,
    ) if not keep_work_inputs else None
    return outputs


def ffill_state(frame: pl.DataFrame) -> pl.DataFrame:
    state_columns = [c for c in frame.columns if c.startswith(STATE_PREFIXES)]
    return frame.with_columns(pl.col(state_columns).forward_fill()) if state_columns else frame


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge three source streams and finalize hourly state.")
    parser.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    parser.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    parser.add_argument("--source", choices=["ingest_parquet", "matched"], default="ingest_parquet")
    parser.add_argument("--date", required=True)
    parser.add_argument("--hour", type=int, default=None, help="0..23; omit for all hours")
    parser.add_argument("--lob-levels", type=int, default=20)
    parser.add_argument("--keep-work-inputs", action="store_true")
    parser.add_argument("--matched-lob-root", default=None)
    args = parser.parse_args(argv)
    outputs = run_merge(
        root=args.root, symbol=args.symbol, source=args.source, date=args.date,
        hour=args.hour, lob_levels=args.lob_levels, keep_work_inputs=args.keep_work_inputs,
        matched_lob_root=args.matched_lob_root,
    )
    for output in outputs:
        print(f"merged -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
