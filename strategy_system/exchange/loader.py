"""Data loaders for the OFMR scalp strategy.

This module is the only place that knows how raw market data is fetched:

    load_L2  matched L2 parquet (merged trades + depth)
    load_L1  CryptoL1 OHLCV candles

The strategy engine (``strategy.py``) calls these through ``load_data``;
nothing else in the package touches storage.
"""

from __future__ import annotations

import pandas as pd

from components.pipelines.event_merger.matched.parquet import (
    list_parquet,
    read_parquet,
)
from components.pipelines.event_merger.matched.reconstruct import (
    normalize_merged_frame,
)
from components.pipelines.ingestion.CryptoL1.price import get_ohlcv


def load_L2(file_limit: int = 10, start: int = 0) -> pd.DataFrame:
    """Load matched L2 parquet files and build the causal OFMR state table.

    The merged matched files use namespaced columns and nanosecond epoch
    timestamps.  ``normalize_merged_frame`` is the single boundary adapter
    for that storage schema; in particular, timestamps must not be forced to
    milliseconds here.

    Parameters
    ----------
    file_limit:
        Maximum number of parquet files to read (one file per hour).
    start:
        Offset into the sorted file list.
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

    return df


def load_L1(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    since: str = "2025-01-01T00:00:00Z",
    until: str = "2025-06-01T00:00:00Z",
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Load OHLCV candles (L1) for the configured symbol/timeframe."""
    df = get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        until=until,
        exchange_id=exchange_id,
    )
    print(f"Total candles retrieved: {len(df)}")

    return df
