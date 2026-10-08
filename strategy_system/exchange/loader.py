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


# Book columns after normalize_merged_frame() (flattened from book_state__*).
# Trade execution price lives in the SEPARATE ``price`` (agg__price) column
# and must never be mixed into these.
_BOOK_COL_PREFIXES = (
    "best_",
    "bid_price_",
    "bid_qty_",
    "ask_price_",
    "ask_qty_",
)

# BTCUSDT instrument tick: 2 decimals ($0.01). mid/order prices are snapped
# with (price / tick).round() * tick -- the same formula as
# uets/executor/exchange/exchange.rs VenueSpec::round_price. Centralised here
# so loader mid and strategy order prices can never drift apart.
BTCTICK = 0.01


def quantize_price(price, tick: float = BTCTICK):
    """Snap price(s) to the instrument tick: (price/tick).round()*tick.

    Scalar- and Series-safe. NaN passes through (no fake quote invented).

    Uses Decimal(str(value)) with ROUND_HALF_UP so the result is the exact
    decimal grid value (108613.2), not the nearest float64 to it
    (108613.20000000001): plain float ``(p / 0.01).round() * 0.01`` re-dusts
    because 0.01 itself is inexact in binary.
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(str(tick))

    def _one(v):
        import math
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        if not math.isfinite(f):
            return v
        if isinstance(v, float) and math.isnan(f):
            return v
        return float(Decimal(str(f)).quantize(q, rounding=ROUND_HALF_UP))

    try:
        if hasattr(price, "apply"):  # pandas Series
            return price.apply(_one)
        if hasattr(price, "__iter__") and not isinstance(price, (str, bytes)):
            import pandas as _pd
            return _pd.Series(list(price)).apply(_one)
        return _one(price)
    except Exception:
        return price


def check_tick_cadence(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    warn_ms: float = 150.0,
    fail_ms: float = 200.0,
) -> dict:
    """Lossless-cadence guard: median inter-row gap must stay ~100ms or below.

    Warns at >warn_ms, raises at >fail_ms (your 'it broke' tripwire: a 200ms
    median means rows are being dropped/duplicated upstream). Returns the
    measured stats dict. Read-only w.r.t. df.
    """
    import logging
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    dt_ms = ts.sort_values().diff().dt.total_seconds() * 1000.0
    dt_ms = dt_ms.dropna()
    stats = {
        "rows": int(len(df)),
        "median_ms": float(dt_ms.median()) if len(dt_ms) else float("nan"),
        "mean_ms": float(dt_ms.mean()) if len(dt_ms) else float("nan"),
        "max_ms": float(dt_ms.max()) if len(dt_ms) else float("nan"),
    }
    msg = (
        f"[tick-cadence] rows={stats['rows']} median={stats['median_ms']:.1f}ms "
        f"mean={stats['mean_ms']:.1f}ms max={stats['max_ms']:.1f}ms"
    )
    if stats["median_ms"] > fail_ms:
        raise ValueError(msg + f" -- median > {fail_ms}ms: LOSSLESS BREAK")
    if stats["median_ms"] > warn_ms:
        logging.getLogger(__name__).warning(msg + " -- degraded cadence")
    return stats


def _book_columns(df: pd.DataFrame) -> list[str]:
    """All live-book columns (best + depth levels) present in this frame."""
    return [
        c for c in df.columns
        if c in ("best_bid_price", "best_ask_price")
        or c.startswith(_BOOK_COL_PREFIXES)
    ]


def _repair_one_file(df: pd.DataFrame) -> pd.DataFrame:
    """Mask stale book on non-LOB rows, causal-ffill live quotes, clean mid.

    Causal: LOB rows keep their book, TRADE/BOOKDEPTH rows inherit the last
    LIVE quote via forward-fill. Trade execution price stays strictly in the
    separate ``price`` column and never leaks into the book.
    """
    if "event_type" not in df.columns:
        return df
    if {"best_bid_price", "best_ask_price"} - set(df.columns):
        return df
    book_cols = _book_columns(df)
    if not book_cols:
        return df
    is_lob = df["event_type"].astype(str).str.upper() == "LOB"
    df.loc[~is_lob, book_cols] = pd.NA
    for c in book_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ffill = last live quote (causal). bfill covers only the edge case
    # where a file starts with TRADE rows before its first same-hour LOB.
    df[book_cols] = df[book_cols].ffill().bfill()
    # Passive mid on the instrument tick: (price/tick).round()*tick kills
    # IEEE dust like 108613.20000000001 -> 108613.2 AND guarantees every
    # downstream equality comparison / order price is tick-valid.
    df["mid_price"] = quantize_price(
        (df["best_bid_price"] + df["best_ask_price"]) / 2.0, BTCTICK
    )
    return df


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

    # --- Causal book repair (needle + dust fix) ---
    # Merged files carry a STALE book_state on TRADE/BOOKDEPTH rows (e.g. all
    # TRADE rows in merged_2025-07-09_09.parquet pin best_bid=108612.2 while
    # LOB rows move 108624..108737). Repair is applied PER FILE (before the
    # cross-file concat) so one hour's stale book can never overwrite another
    # hour's live book. Downstream (strategy.py:183-186, features.py:126)
    # reuses the precomputed ``mid_price`` because of its
    # ``if "mid_price" not in df.columns`` guard.
    frames = []
    sort_columns = ["timestamp"]
    for _file in files:
        _one = normalize_merged_frame(read_parquet(_file))
        _cols = ["timestamp"]
        if "event_sequence" in _one.columns:
            _cols.append("event_sequence")
        elif "agg_trade_id" in _one.columns:
            _cols.append("agg_trade_id")
        if len(sort_columns) == 1 and len(_cols) > 1:
            sort_columns = list(_cols)
        _one = (
            _one.dropna(subset=["timestamp"])
            .sort_values(_cols, kind="mergesort")
            .reset_index(drop=True)
        )
        frames.append(_repair_one_file(_one))
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(sort_columns, kind="mergesort")
        .reset_index(drop=True)
    )

    # Lossless-cadence tripwire: median gap must stay ~100ms or below.
    # check_tick_cadence raises if median > 200ms (rows dropped/duplicated).
    check_tick_cadence(df)

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
