from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from config.ingestion import config as cfg


PathLike = str | os.PathLike[str] | Path


def _as_path(path: PathLike) -> Path:
    return Path(path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp.parquet"
    )


def _atomic_replace(temp_path: Path, final_path: Path) -> None:
    os.replace(temp_path, final_path)


# ============================================================
# LIST PARQUET
# ============================================================

def _default_path() -> Path:
    """
    Resolve the default matched path without depending on the CWD.

    ``config.ingestion.config.MATCHED_PATH`` is built from a *relative*
    ``DATA_ROOT`` (``Path("market_data")``). Running an interpreter from
    ``market_data/matched/<TICKER>`` therefore resolved it to a
    non-existent nested path, ``list_parquet()`` returned ``[]`` and callers
    subscripted ``None``. Anchoring on the repo root (derived from this
    file's location) makes the default work from any directory.
    """
    project_root = Path(__file__).resolve().parents[4]
    configured = cfg.MATCHED_PATH
    if configured is not None:
        configured = Path(configured)
        if configured.is_absolute():
            return configured
        # Prefer the CWD-relative path when it actually exists (e.g. running
        # from the repo root), otherwise anchor it at the repo root.
        if configured.exists():
            return configured
        return project_root / configured
    return project_root / "market_data" / "matched" / cfg.TICKER


def list_parquet(
    path: PathLike | None = None,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Return parquet files in sorted deterministic order.

    If path is omitted, the default matched path is used.
    """

    if path is None:
        path = _default_path()

    root = _as_path(path)

    if not root.exists():
        return []

    if root.is_file():
        if (
            root.suffix == ".parquet"
            and not root.name.endswith(".tmp.parquet")
        ):
            return [root]

        return []

    iterator = (
        root.rglob("*.parquet")
        if recursive
        else root.glob("*.parquet")
    )

    files = [
        file
        for file in iterator
        if file.is_file()
        and not file.name.endswith(".tmp.parquet")
    ]

    return sorted(files)


# ============================================================
# READ PARQUET
# ============================================================

def read_parquet(
    path: PathLike | None = None,
    *,
    batch_size: int | None = None,
    columns: Sequence[str] | None = None,
    recursive: bool = False,
    ignore_index: bool = True,
):
    """
    Read parquet data.

    If batch_size is omitted:
        read_parquet("file.parquet")
        -> DataFrame

    If batch_size is supplied:
        read_parquet(batch_size=1)
        -> iterator of DataFrames

    When path is omitted:
        the default matched path is used.

    CSV/other extensions are routed to ``pandas.read_*`` so a sample
    exported from the matched tree can be read with the same call.
    """

    if path is None:
        path = _default_path()

    path = _as_path(path)

    # --------------------------------------------------------
    # Batched read
    # --------------------------------------------------------

    if batch_size is not None:

        return read_batched_parquet(
            list_parquet(
                path,
                recursive=recursive,
            ),
            batch_size=batch_size,
            columns=columns,
            ignore_index=ignore_index,
        )

    # --------------------------------------------------------
    # Single file
    # --------------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise ValueError(
            f"read_parquet() without batch_size expects "
            f"a parquet file, got: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            usecols=columns,
            index_col=False if ignore_index else None,
        )

    if suffix in (".json", ".jsonl"):
        return pd.read_json(path)

    return pd.read_parquet(
        path,
        columns=columns,
    )


# ============================================================
# READ BATCHED PARQUET
# ============================================================

def read_batched_parquet(
    files: Iterable[PathLike],
    *,
    batch_size: int = 1,
    columns: Sequence[str] | None = None,
    ignore_index: bool = True,
) -> Iterator[pd.DataFrame]:

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1"
        )

    batch: list[Path] = []

    for file in files:

        batch.append(
            _as_path(file)
        )

        if len(batch) == batch_size:

            yield _read_file_batch(
                batch,
                columns=columns,
                ignore_index=ignore_index,
            )

            batch.clear()

    if batch:

        yield _read_file_batch(
            batch,
            columns=columns,
            ignore_index=ignore_index,
        )


def _read_file_batch(
    files: Sequence[Path],
    *,
    columns: Sequence[str] | None,
    ignore_index: bool,
) -> pd.DataFrame:

    frames = [
        read_parquet(
            file,
            columns=columns,
        )
        for file in files
    ]

    if len(frames) == 1:
        return frames[0]

    return pd.concat(
        frames,
        ignore_index=ignore_index,
    )


# ============================================================
# WRITE PARQUET
# ============================================================

def write_parquet(
    path: PathLike,
    df: pd.DataFrame,
    *,
    index: bool = False,
) -> None:

    final_path = _as_path(path)

    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = _temporary_path(
        final_path
    )

    try:

        df.to_parquet(
            temp_path,
            index=index,
        )

        with temp_path.open("rb") as f:
            os.fsync(
                f.fileno()
            )

        _atomic_replace(
            temp_path,
            final_path,
        )

    except Exception:

        try:
            temp_path.unlink()

        except FileNotFoundError:
            pass

        raise


# ============================================================
# WRITE BATCHED PARQUET
# ============================================================

def write_parquet_batched(
    batches: Iterable[
        tuple[PathLike, pd.DataFrame]
    ],
    *,
    index: bool = False,
) -> None:

    for path, df in batches:

        write_parquet(
            path,
            df,
            index=index,
        )


# ============================================================
# TERMINAL PIPELINE WORK TREE
# ============================================================

def pipeline_work_dir(
    root: PathLike | None = None,
    symbol: str = "BTCUSDT",
) -> Path:
    """Disposable pipeline tree; canonical ingest and legacy samples stay read-only."""
    import components.python.data_pipeline.CryptoL2.datapaths as paths

    root_p = _as_path(root) if root is not None else paths.MARKET_DATA_ROOT
    return paths.matched_dir(root_p, symbol.upper()) / "_pipeline"


def pipeline_source_dir(
    root: PathLike | None = None,
    symbol: str = "BTCUSDT",
    dataset: str | None = None,
) -> Path:
    base = pipeline_work_dir(root, symbol) / "source"
    return base if dataset is None else base / dataset


def _materialize_work_file(src: Path, dest: Path) -> None:
    """Create a disposable work file without modifying the canonical source."""
    import shutil

    if not src.is_file():
        raise FileNotFoundError(src)
    if src.resolve() == dest.resolve():
        raise ValueError(f"source and destination are identical: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(dest)
    try:
        try:
            os.link(src, temp)
        except OSError:
            shutil.copy2(src, temp)
        os.replace(temp, dest)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


# ============================================================
# FETCH ALL THREE SOURCE DATASETS INTO THE WORK TREE
# ============================================================

def fetch_source_data(
    *,
    root: PathLike | None = None,
    symbol: str = "BTCUSDT",
    start: str,
    end: str,
) -> list[Path]:
    """Materialize aggTrades, bookDepth, and orderbook for a UTC date window.

    Canonical files under ``ingest_parquet`` are only read. Work files live under
    ``matched/<SYMBOL>/_pipeline/source`` so they can be removed safely after a
    verified merge without deleting source data.
    """
    import components.python.data_pipeline.CryptoL2.datapaths as paths

    root_p = _as_path(root) if root is not None else paths.MARKET_DATA_ROOT
    symbol = symbol.upper()
    work = pipeline_source_dir(root_p, symbol)
    materialized: list[Path] = []

    days = set(paths.binance_parquet_days(root_p, symbol, "aggTrades"))
    days.update(paths.binance_parquet_days(root_p, symbol, "bookDepth"))
    days.update(paths.orderbook_parquet_days(root_p, symbol))
    days = {day for day in days if start <= day <= end}
    for day in sorted(days):
        for dataset in ("aggTrades", "bookDepth"):
            src = paths.parquet_dir(root_p, symbol, dataset) / f"{day}.parquet"
            if src.exists():
                dest = work / dataset / f"{day}.parquet"
                _materialize_work_file(src, dest)
                materialized.append(dest)
                print(f"[fetch] {dataset} {day} -> {dest}")

        for hour in range(24):
            src = paths.parquet_chd_file(root_p, symbol, day, hour)
            if src.exists():
                dest = work / "orderbook" / day / f"{hour:02d}.parquet"
                _materialize_work_file(src, dest)
                materialized.append(dest)
                print(f"[fetch] orderbook {day} {hour:02d} -> {dest}")

    if not materialized:
        raise FileNotFoundError(
            f"no aggTrades/bookDepth/orderbook files in [{start}, {end}] "
            f"under {paths.parquet_root(root_p) / symbol}"
        )
    return sorted(materialized)

