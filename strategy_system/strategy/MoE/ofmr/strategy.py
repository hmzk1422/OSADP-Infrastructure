import pandas as pd

from strategy_system.strategy.MoE.ofmr.state_builder import build_state
from components.python.diagnostic.lossless import lob_report
from components.python.data_pipeline.matched.reconstruct import (
    normalize_merged_frame,
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
def load_state(*, file_limit: int | None = 10) -> pd.DataFrame:
    """Load matched files and build the causal OFMR state table.

    The merged matched files use namespaced columns and nanosecond epoch
    timestamps.  ``normalize_merged_frame`` is the single boundary adapter
    for that storage schema; in particular, timestamps must not be forced to
    milliseconds here.
    """
    files = list_parquet()
    if file_limit is not None:
        files = files[:file_limit]
    if not files:
        raise FileNotFoundError(
            "No matched parquet files were found in the configured matched path."
        )

    df = pd.concat(
        (read_parquet(file) for file in files),
        ignore_index=True,
    )

    df = normalize_merged_frame(df)

    sort_columns = ["timestamp"]
    if "agg_trade_id" in df.columns:
        sort_columns.append("agg_trade_id")
    df = (
        df
        .dropna(subset=["timestamp"])
        .sort_values(sort_columns, kind="mergesort")
        .reset_index(drop=True)
    )

    return build_state(df)


if __name__ == "__main__":
    state = load_state()
    print(state)
