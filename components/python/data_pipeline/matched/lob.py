"""
Multilevel L2 order-book reconstruction.

Responsibilities:
    - normalize CryptoHFTData L2 columns
    - invoke the Rust lob_replay executable
    - maintain replay state between chunks
    - load/store day-start checkpoints
    - return reconstructed top-N book states

The actual L2 replay logic lives in Rust.

Python is only responsible for:
    - preparing parquet input
    - calling the Rust executable
    - reading the Rust output
    - maintaining the cross-chunk checkpoint/resume information
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from config.ingestion import config as cfg


# ===========================================================
# CONSTANTS
# ===========================================================

LOB_TICK_SIZE = 1.0 / 100.0

LOB_CHECKPOINT_HOUR_UTC = 0
SNAPSHOT_DIRNAME = "snapshots"

SNAPSHOT_POLICY_DAY_START = "day_start"
SNAPSHOT_POLICY_EVERY_EVENT = "every_snapshot"

SNAPSHOT_POLICIES = (
    SNAPSHOT_POLICY_DAY_START,
    SNAPSHOT_POLICY_EVERY_EVENT,
)

REPLAY_STATS_KEYS = (
    "snapshot_events",
    "update_events",
    "gaps",
    "malformed_events",
    "crossed_books",
    "rows_before_initial_snapshot",
    "snapshots",
    "skipped_checkpoints",
    "batches",
    "emitted_rows",
    "skipped_state_rows",
    "warm_start_batches",
)

LOB_REPLAY_BIN = Path(
    os.environ.get(
        "LOB_REPLAY_BIN",
        "components/target/debug/lob_replay.exe"
        if os.name == "nt"
        else "components/target/debug/lob_replay",
    )
)


# ===========================================================
# REPLAY STATE
# ===========================================================

@dataclass
class LobReplayState:
    """
    State carried from one parquet chunk to the next.

    The actual order-book contents live in Rust.

    Python only needs to remember:
        - the last emitted timestamp
        - the most recent checkpoint used for reseeding
    """

    skip_until: int = 0
    seed_snapshot_path: Path | None = None


# ===========================================================
# TIME / SNAPSHOT HELPERS
# ===========================================================

def day_string(event_time_ms: int) -> str:
    """
    UTC YYYY-MM-DD for an epoch-ms timestamp.
    """
    return datetime.fromtimestamp(
        event_time_ms / 1000.0,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")


def snapshot_path(
    day: str,
    snapshot_dir: Path | None = None,
) -> Path:
    """
    Return the checkpoint path for a UTC day.
    """
    if snapshot_dir is None:
        snapshot_dir = (
            Path(cfg.MATCHED_PATH)
            / SNAPSHOT_DIRNAME
        )

    return Path(snapshot_dir) / f"{day}.parquet"


def is_day_start_event(event_time_ms: int) -> bool:
    """
    True when an epoch-ms timestamp falls inside UTC hour 00.
    """
    if event_time_ms < 0:
        return False

    ms_of_day = event_time_ms % 86_400_000
    hour = ms_of_day // 3_600_000

    return hour == LOB_CHECKPOINT_HOUR_UTC


# ===========================================================
# SNAPSHOT LOADING
# ===========================================================

def load_day_snapshot(
    day: str,
    snapshot_dir: Path | None = None,
) -> dict | None:
    """
    Load the first stored snapshot for a day.

    Returns:

        {
            "day": ...,
            "event_time": ...,
            "final_update_id": ...,
            "bids": [(price, qty), ...],
            "asks": [(price, qty), ...],
            "path": Path(...)
        }

    or None if the snapshot does not exist.
    """
    path = snapshot_path(
        day,
        snapshot_dir,
    )

    if not path.exists():
        return None

    frame = pl.read_parquet(path)

    if frame.height == 0:
        return None

    def read_side(prefix: str) -> list[tuple[float, float]]:
        price_columns = sorted(
            (
                name
                for name in frame.columns
                if name.startswith(f"{prefix}_price_")
            ),
            key=lambda name: int(
                name.rsplit("_", 1)[1]
            ),
        )

        result = []

        for price_column in price_columns:
            price = frame[price_column][0]

            if price is None:
                continue

            qty_column = price_column.replace(
                "_price_",
                "_qty_",
            )

            quantity = frame[qty_column][0]

            if quantity is None:
                continue

            result.append(
                (
                    float(price),
                    float(quantity),
                )
            )

        return result

    return {
        "day": day,
        "event_time": int(
            frame["event_time"][0]
        ),
        "final_update_id": int(
            frame["final_update_id"][0]
        ),
        "bids": read_side("bid"),
        "asks": read_side("ask"),
        "path": path,
    }


# ===========================================================
# COLUMN NORMALIZATION
# ===========================================================

def lob_columns(lob) -> pl.DataFrame:
    """
    Normalize CryptoHFTData's L2 columns into the schema expected by Rust.

    Accepts either:
        - Polars LazyFrame
        - Polars DataFrame
    """
    normalized = lob.select(
        pl.col("event_time").cast(pl.Int64),

        (pl.col("side") == "bid").alias(
            "is_bid"
        ),

        (pl.col("event_type") == "snapshot").alias(
            "is_snapshot"
        ),

        pl.col("price").cast(pl.Float64),
        pl.col("quantity").cast(pl.Float64),

        pl.col("first_update_id").cast(pl.Int64),
        pl.col("final_update_id").cast(pl.Int64),
        pl.col("prev_final_update_id").cast(pl.Int64),
    )

    if isinstance(normalized, pl.LazyFrame):
        return normalized.collect()

    return normalized


# ===========================================================
# RUST REPLAY
# ===========================================================

def _parse_stats(stdout: str) -> dict[str, int]:
    """
    Parse:

        STATS key=value key=value ...

    emitted by lob_replay.
    """
    for line in stdout.splitlines():
        if not line.startswith("STATS "):
            continue

        result = {}

        for token in line[len("STATS "):].split():
            key, value = token.split("=", 1)
            result[key] = int(value)

        return result

    raise RuntimeError(
        "lob_replay produced no STATS line"
    )


def _run_lob_replay(
    lob,
    *,
    lob_levels: int,
    snapshot_policy: str,
    snapshot_dir: Path | None,
    snapshot_levels: int,
    state: LobReplayState | None,
    tick_size: float,
    warm_start: bool,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """
    Run one bounded LOB chunk through the Rust executable.
    """
    frame = lob_columns(lob)

    with tempfile.TemporaryDirectory(
        prefix="lob_replay_",
    ) as tmp_dir:

        tmp_dir = Path(tmp_dir)

        input_path = (
            tmp_dir / "input.parquet"
        )

        output_path = (
            tmp_dir / "book_state.parquet"
        )

        frame.write_parquet(input_path)

        command = [
            str(LOB_REPLAY_BIN),
            "columnar",

            "--input",
            str(input_path),

            "--output",
            str(output_path),

            "--tick-size",
            str(tick_size),

            "--levels",
            str(lob_levels),

            "--snapshot-policy",
            snapshot_policy,
        ]

        if warm_start:
            command.append("--warm-start")

        if snapshot_dir is not None:
            command.extend(
                [
                    "--snapshot-dir",
                    str(snapshot_dir),
                ]
            )

        if (
            snapshot_policy
            == SNAPSHOT_POLICY_EVERY_EVENT
        ):
            command.extend(
                [
                    "--snapshot-levels",
                    str(snapshot_levels),
                ]
            )

        if state is not None:
            if (
                state.seed_snapshot_path
                is not None
            ):
                command.extend(
                    [
                        "--seed-snapshot",
                        str(
                            state.seed_snapshot_path
                        ),
                    ]
                )

            if state.skip_until:
                command.extend(
                    [
                        "--skip-until",
                        str(state.skip_until),
                    ]
                )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "lob_replay failed "
                f"(exit {result.returncode})\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        stats = _parse_stats(
            result.stdout
        )

        if output_path.exists():
            book = pl.read_parquet(
                output_path
            )
        else:
            book = pl.DataFrame(
                {
                    "event_time": pl.Series(
                        [],
                        dtype=pl.Int64,
                    )
                }
            )

    return book, stats


# ===========================================================
# MULTILEVEL LOB BUILDER
# ===========================================================

def build_multilevel_lob(
    lob,
    *,
    lob_levels: int = 20,
    state: LobReplayState | None = None,
    snapshot_dir: Path | None = None,
    write_snapshots: bool = True,
    snapshot_policy: str = SNAPSHOT_POLICY_DAY_START,
    snapshot_levels: int = 50,
    tick_size: float = LOB_TICK_SIZE,
    warm_start: bool = True,
) -> tuple[pl.DataFrame, LobReplayState, dict[str, int]]:
    """
    Reconstruct the multilevel L2 order book for one bounded chunk.

    The Rust executable performs the actual replay.

    Parameters
    ----------
    lob:
        Polars LazyFrame or DataFrame containing raw CryptoHFTData L2 rows.

    lob_levels:
        Number of top bid/ask levels emitted into the book-state output.

    state:
        State carried from the previous chunk.

    snapshot_policy:
        "day_start"
            Store one full-book checkpoint per UTC day.

        "every_event"
            Store every valid reconstructed event.

    Returns
    -------
    book:
        One reconstructed book-state row per valid event_time.

    next_state:
        State to pass into the next chunk.

    stats:
        Statistics generated by this chunk.
    """
    if lob_levels < 1:
        raise ValueError(
            "lob_levels must be >= 1"
        )

    if snapshot_levels < 1:
        raise ValueError(
            "snapshot_levels must be >= 1"
        )

    if snapshot_policy not in SNAPSHOT_POLICIES:
        raise ValueError(
            f"unknown snapshot policy "
            f"{snapshot_policy!r}; expected "
            f"{SNAPSHOT_POLICIES}"
        )

    if not LOB_REPLAY_BIN.exists():
        raise FileNotFoundError(
            f"lob_replay binary not found: "
            f"{LOB_REPLAY_BIN}"
        )

    if write_snapshots:
        if snapshot_dir is None:
            snapshot_dir = (
                Path(cfg.MATCHED_PATH)
                / SNAPSHOT_DIRNAME
            )

        snapshot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    current_state = (
        state
        if state is not None
        else LobReplayState()
    )

    # Don't pass snapshot/resume state when the
    # caller explicitly disabled snapshot writing.
    replay_state = (
        current_state
        if write_snapshots
        else LobReplayState(
            skip_until=current_state.skip_until
        )
    )

    book, stats = _run_lob_replay(
        lob,
        lob_levels=lob_levels,
        snapshot_policy=snapshot_policy,
        snapshot_dir=(
            snapshot_dir
            if write_snapshots
            else None
        ),
        snapshot_levels=snapshot_levels,
        state=replay_state,
        tick_size=tick_size,
        warm_start=warm_start,
    )

    # Advance skip state to the last reconstructed
    # event in this chunk.
    next_state = LobReplayState(
        skip_until=current_state.skip_until,
        seed_snapshot_path=current_state.seed_snapshot_path,
    )

    if book.height > 0:
        last_event_time = int(
            book["event_time"][-1]
        )

        next_state.skip_until = max(
            current_state.skip_until,
            last_event_time,
        )

        # Day-start mode:
        # if Rust materialized a checkpoint during this
        # chunk, remember it for the next chunk.
        if (
            write_snapshots
            and snapshot_policy
            == SNAPSHOT_POLICY_DAY_START
        ):
            candidate = snapshot_path(
                day_string(last_event_time),
                snapshot_dir,
            )

            if candidate.exists():
                next_state.seed_snapshot_path = (
                    candidate
                )

    # Return a stable stats dictionary even if the
    # binary omitted a zero-valued counter.
    normalized_stats = {
        key: int(stats.get(key, 0))
        for key in REPLAY_STATS_KEYS
    }

    return (
        book,
        next_state,
        normalized_stats,
    )


# ===========================================================
# RESUME
# ===========================================================

def resume_from_day_snapshot(
    day: str,
    *,
    snapshot_dir: Path | None = None,
) -> LobReplayState | None:
    """
    Create replay state from an existing day-start
    snapshot.

    Returns None when there is no checkpoint.
    """
    snapshot = load_day_snapshot(
        day,
        snapshot_dir,
    )

    if snapshot is None:
        return None

    return LobReplayState(
        skip_until=snapshot["event_time"],
        seed_snapshot_path=snapshot["path"],
    )