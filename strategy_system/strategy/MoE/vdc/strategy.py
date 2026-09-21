import pandas as pd

from strategy_system.strategy.MoE.vdc.state_builder import build_state
from components.python.diagnostic.lossless import lob_report
from components.python.data_pipeline.matched.reconstruct import (
    reconstruct,
    rule,
    join_timeframes,
)
from components.python.data_pipeline.matched.lob import (
    LobReplayState,
    build_multilevel_lob,
    resume_from_day_snapshot,
)
from components.python.data_pipeline.matched.parquet import (
    list_parquet,
    read_parquet,
    read_batched_parquet,
    write_parquet,
    write_parquet_batched,
)

from pathlib import Path


# ============================================================
# READ MULTIPLE PARQUET FILES
# ============================================================

files = list_parquet()

files = files[:10]

df = pd.concat(
    (
        read_parquet(file)
        for file in files
    ),
    ignore_index=True,
)

# ============================================================
# SORT RAW DATA
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    errors="coerce",
)

df = (
    df
    .dropna(subset=["timestamp"])
    .sort_values(
        ["timestamp", "agg_trade_id"],
        kind="mergesort",
    )
    .reset_index(drop=True)
)


# ============================================================
# RECONSTRUCT
# ============================================================

df = build_state(df)