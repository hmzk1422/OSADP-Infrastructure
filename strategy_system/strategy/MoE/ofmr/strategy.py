import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from strategy_system.strategy.MoE.ofmr.state_builder import build_state
from components.pipelines.diagnostic.lossless import lob_report
from components.pipelines.event_merger.matched.reconstruct import (
    normalize_merged_frame,
    reconstruct,
    rule,
    join_timeframes,
)
from components.pipelines.event_merger.matched.lob import (
    LobReplayState,
    build_multilevel_lob,
    resume_from_day_snapshot,
)
from components.pipelines.event_merger.matched.parquet import (
    list_parquet,
    read_parquet,
    read_batched_parquet,
    write_parquet,
    write_parquet_batched,
)
from components.pipelines.ingestion.CryptoL1.price import get_ohlcv

from pathlib import Path


# ============================================================
def load_state_L2(file_limit=10, start=0) -> pd.DataFrame:
    """Load matched files and build the causal OFMR state table.

    The merged matched files use namespaced columns and nanosecond epoch
    timestamps.  ``normalize_merged_frame`` is the single boundary adapter
    for that storage schema; in particular, timestamps must not be forced to
    milliseconds here.
    """
    files = list_parquet()
    files = files[start:start + file_limit]

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

def load_state_L1() -> pd.DataFrame:
    df = get_ohlcv(
            symbol="BTC/USDT",
            timeframe="1m",
            since="2025-01-01T00:00:00Z",
            until="2025-06-01T00:00:00Z",
            exchange_id="binance",
        )
    print(f"Total candles retrieved: {len(df)}")

    return build_state(df)

# ============================================================

if __name__ == "__main__":
    state = load_state_L1()