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

def list_parquet(
    path: PathLike | None = None,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Return parquet files in sorted deterministic order.

    If path is omitted, cfg.MATCHED_PATH is used.
    """

    if path is None:
        path = cfg.MATCHED_PATH

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
        cfg.MATCHED_PATH is used.
    """

    if path is None:
        path = cfg.MATCHED_PATH

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
    # Single parquet file
    # --------------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise ValueError(
            f"read_parquet() without batch_size expects "
            f"a parquet file, got: {path}"
        )

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