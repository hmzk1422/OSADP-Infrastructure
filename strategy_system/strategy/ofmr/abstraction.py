"""Matched-database adapter for the OFMR feature pipeline.

Bridges ``market_data/matched/<SYMBOL>/merged_*.parquet`` (the matcher output:
one event-stream frame per hour with ``event_type`` in {LOB, TRADE, BOOKDEPTH})
into the exact eager-frame schema that
:func:`ml_state_builder._load_and_build_state_space_from_frames` already
consumes — without touching any other file in this folder:

    matched merged_* parquet
        -> :func:`load_matched_frames`       (this module: split + rename)
        -> ``_load_and_build_state_space_from_frames``  (existing, untouched)
        -> state-space parquet(s) under ``strategy_system/strategy/ofmr/models/``

Schema mapping (verified against a real ``merged_2025-06-30_00.parquet``):

    LOB rows   -> orderbook frame
        book_state__best_bid_price -> bid_price      (f64)
        book_state__best_bid_qty   -> bid_amount     (f64)
        book_state__best_ask_price -> ask_price      (f64)
        book_state__best_ask_qty   -> ask_amount     (f64)
        timestamp (ns, i64)        -> timestamp      (ms, i64; // 1_000_000,
                                      matching _with_datetime's // 1000 on ms)
    TRADE rows -> trades frame
        agg__price          -> price    (f64)
        agg__quantity        -> amount   (f64)
        agg__is_buyer_maker  -> side     ("buy"/"sell"; see _taker_side below)
        agg__agg_trade_id    -> id       (str)

Timestamps in matched/ are nanoseconds; the legacy pipeline stores
milliseconds (``_with_datetime`` does ``timestamp // 1000 -> s``). The adapter
divides by 1_000_000 so downstream epoch math is unchanged.

BOOKDEPTH rows (snapshot depth levels) carry no top-of-book/trade print and
are ignored — orderflow/volume features derive from LOB L1 + TRADE prints.

``agg__is_buyer_maker`` semantics (Binance aggTrade ``m`` flag): True means
the *buyer* was the maker, i.e. the taker SOLD into the bid — so the
aggressor side is "sell". False (buyer was taker) -> "buy". This matches the
taker-side convention ``compute_orderflow_features`` expects
(``side == "buy"`` counts into ``buy_vol``).

Outputs (all under ``strategy_system/strategy/ofmr/models/``):

    models/state_space_<symbol>_<tag>.parquet   full state space, datetime-sorted
    models/manifest_<symbol>_<tag>.json         row counts, time bounds, config

Run:

    python -m strategy_system.strategy.ofmr.abstraction --help
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    # Appended (not prepended): `ofmr/config.py` must keep shadowing the
    # repo-root `config/` stub package, exactly as when ml_state_optimized.py
    # runs with ofmr/ as the script dir.
    sys.path.append(PROJECT_ROOT)

OFMR_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.abspath(os.path.join(OFMR_DIR, ".."))
if STRATEGY_DIR not in sys.path:
    # ml_state_builder does `from ofmr.feature_builder import ...` and
    # `from indicators import ...` — resolvable only with
    # strategy_system/strategy on sys.path (CWD when the runner launches).
    sys.path.insert(0, STRATEGY_DIR)
if OFMR_DIR not in sys.path:
    sys.path.insert(0, OFMR_DIR)

import polars as pl

os.environ.setdefault("POLARS_MAX_THREADS", "4")

from ml_state_builder import (  # noqa: E402  (sys.path wired above)
    BAR_FREQ,
    SessionFeatureContext,
    _load_and_build_state_space_from_frames,
    _rsi_features_from_price_bars,
    dense_resample,
)

MODELS_DIR = os.path.join(OFMR_DIR, "models")
MATCHED_ROOT = os.path.join(PROJECT_ROOT, "market_data", "matched")

_NS_PER_MS = 1_000_000
_BOOK_COLS = (
    "book_state__best_bid_price",
    "book_state__best_bid_qty",
    "book_state__best_ask_price",
    "book_state__best_ask_qty",
)
_TRADE_COLS = ("agg__price", "agg__quantity", "agg__is_buyer_maker")

# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_matched_files(symbol: str, tag: str | None = None) -> list[str]:
    """Return sorted ``merged_*.parquet`` paths for ``symbol`` under matched/.

    ``symbol`` is matched case-insensitively (``BTCUSDT`` dir vs ``btcusdt``
    ticker). When ``tag`` is given, only files whose basename contains it are
    returned (e.g. a date ``2025-06-30``); otherwise every merged file.
    """
    sym_dir = None
    if os.path.isdir(MATCHED_ROOT):
        for entry in os.listdir(MATCHED_ROOT):
            if entry.upper() == symbol.upper() and os.path.isdir(
                os.path.join(MATCHED_ROOT, entry)
            ):
                sym_dir = os.path.join(MATCHED_ROOT, entry)
                break
    if sym_dir is None:
        raise FileNotFoundError(
            f"no matched dir for symbol {symbol!r} under {MATCHED_ROOT!r}"
        )
    paths = sorted(glob.glob(os.path.join(sym_dir, "merged_*.parquet")))
    if tag:
        paths = [p for p in paths if tag in os.path.basename(p)]
    if not paths:
        raise FileNotFoundError(
            f"no merged_*.parquet for symbol {symbol!r}"
            + (f" with tag {tag!r}" if tag else "")
            + f" under {sym_dir!r}"
        )
    return paths


# ---------------------------------------------------------------------------
# frame adapter: matched event stream -> (orderbook, trades) eager frames
# ---------------------------------------------------------------------------

def _taker_side(is_buyer_maker: pl.Expr) -> pl.Expr:
    # Binance aggTrade `m`: buyer-is-maker. Taker sold into the bid -> "sell".
    return (
        pl.when(is_buyer_maker.fill_null(True))
        .then(pl.lit("sell"))
        .otherwise(pl.lit("buy"))
    )


def load_matched_frames(paths: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split matched merged files into parquet-schema (orderbook, trades).

    Returns eager frames with exactly the columns
    ``_load_and_build_state_space_from_frames`` documents
    (``timestamp`` ms-int, bid/ask price+amount / price, amount, side, id)
    plus a ``datetime`` column, sorted by (datetime, timestamp) like the
    chunk loader produces. BOOKDEPTH rows are ignored (no L1/trade print).
    """
    if not paths:
        raise ValueError("load_matched_frames: no input paths given")

    books: list[pl.DataFrame] = []
    trades: list[pl.DataFrame] = []
    for path in paths:
        df = pl.read_parquet(path)
        missing = [
            c
            for c in (*_BOOK_COLS, *_TRADE_COLS, "timestamp", "event_type")
            if c not in df.columns
        ]
        if missing:
            raise KeyError(f"{path!r} is missing columns: {missing}")

        lob = df.filter(pl.col("event_type") == "LOB").select(
            ((pl.col("timestamp") // _NS_PER_MS).cast(pl.Int64)).alias("timestamp"),
            pl.col(_BOOK_COLS[0]).cast(pl.Float64).alias("bid_price"),
            pl.col(_BOOK_COLS[1]).cast(pl.Float64).alias("bid_amount"),
            pl.col(_BOOK_COLS[2]).cast(pl.Float64).alias("ask_price"),
            pl.col(_BOOK_COLS[3]).cast(pl.Float64).alias("ask_amount"),
        )
        tr = df.filter(pl.col("event_type") == "TRADE").select(
            ((pl.col("timestamp") // _NS_PER_MS).cast(pl.Int64)).alias("timestamp"),
            pl.col(_TRADE_COLS[0]).cast(pl.Float64).alias("price"),
            pl.col(_TRADE_COLS[1]).cast(pl.Float64).alias("amount"),
            _taker_side(pl.col(_TRADE_COLS[2])).alias("side"),
            pl.col("agg__agg_trade_id").cast(pl.String).alias("id"),
        )
        books.append(lob)
        trades.append(tr)
        del df
        gc.collect()

    df_book = pl.concat(books, how="vertical_relaxed")
    df_trades = pl.concat(trades, how="vertical_relaxed")
    del books, trades
    gc.collect()

    # Mirror _load_and_clean_chunk_events: visible counts, drop null keys.
    for frame, name, keys in (
        (df_book, "orderbook", ["bid_price", "ask_price"]),
        (df_trades, "trades", ["price", "amount"]),
    ):
        n_null = frame.filter(
            pl.col("timestamp").is_null()
            | pl.any_horizontal(*[pl.col(k).is_null() for k in keys])
        ).height
        if n_null:
            print(f"  [DATA WARNING] matched adapter: dropping {n_null} {name} row(s)")
    df_book = df_book.filter(
        pl.col("timestamp").is_not_null()
        & pl.col("bid_price").is_not_null()
        & pl.col("ask_price").is_not_null()
    )
    df_trades = df_trades.filter(
        pl.col("timestamp").is_not_null()
        & pl.col("price").is_not_null()
        & pl.col("amount").is_not_null()
    )
    if df_book.is_empty():
        raise ValueError("matched adapter: no usable LOB rows after cleaning")
    if df_trades.is_empty():
        raise ValueError("matched adapter: no usable TRADE rows after cleaning")


    # Same datetime derivation as _with_datetime (ms -> s -> datetime),
    # then the same (datetime, timestamp) ordering the loaders produce.
    # Timestamps fit int32 seconds; the ms timestamp stays Int64 (id-like,
    # never cast — mirrors the chunk loader's cs.by_name exclusion).
    df_book = df_book.with_columns(
        pl.from_epoch((pl.col("timestamp") // 1000).cast(pl.Int32),
                      time_unit="s").alias("datetime")
    ).filter(pl.col("datetime").is_not_null()).sort(
        ["datetime", "timestamp"], maintain_order=True
    ).with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price")
    )
    df_trades = df_trades.with_columns(
        pl.from_epoch((pl.col("timestamp") // 1000).cast(pl.Int32),
                      time_unit="s").alias("datetime")
    ).filter(pl.col("datetime").is_not_null()).sort(
        ["datetime", "timestamp"], maintain_order=True
    )
    return df_book, df_trades


def build_matched_session_context(
    df_book: pl.DataFrame, df_trades: pl.DataFrame
) -> SessionFeatureContext:
    """Session context over the whole matched span (no chunk-reset rule).

    Mirrors ``build_session_feature_context`` but from in-memory frames:
    VPIN buckets sized by total session trade volume, and the recursive RSI
    family over one continuous BAR_FREQ close series (sparse last-mid bars ->
    dense grid -> ``_rsi_features_from_price_bars``), exactly like
    ``_build_session_rsi_features``.
    """
    total_volume = float(
        df_trades.select(pl.col("amount").cast(pl.Float64).sum()).item()
    )
    sparse_bars = (
        df_book.sort(["datetime", "timestamp"], maintain_order=True)
        .group_by_dynamic("datetime", every=BAR_FREQ, closed="left", label="left")
        .agg(pl.col("mid_price").last().alias("close"))
    )
    price_bars = dense_resample(
        sparse_bars, BAR_FREQ, [pl.col("close").last().alias("close")])
    rsi_features = _rsi_features_from_price_bars(price_bars)
    print(f"  [CTX] total_volume={total_volume:,.3f}, rsi_bars={rsi_features.height}")
    return SessionFeatureContext(total_volume=total_volume, rsi_features=rsi_features)


# ---------------------------------------------------------------------------
# run hook: matched/ -> ofmr/models/
# ---------------------------------------------------------------------------

def run_matched_to_models(
    symbol: str,
    tag: str | None = None,
    session_id: int = 0,
    paths: list[str] | None = None,
    out_name: str | None = None,
) -> dict:
    """Build the state space from matched/ and persist it under ofmr/models/.

    Writes ``state_space_<symbol>_<tag>.parquet`` (datetime-sorted) plus a
    ``manifest_<symbol>_<tag>.json`` sidecar, and returns the manifest dict.
    """
    files = list(paths) if paths else discover_matched_files(symbol, tag)
    stem = out_name or (tag or f"{len(files)}files")
    print(f"[MATCHED] {len(files)} file(s) for {symbol} -> models/{stem}")

    df_book, df_trades = load_matched_frames(files)
    print(f"  [MATCHED] book={df_book.height:,} trades={df_trades.height:,}")
    session_features = build_matched_session_context(df_book, df_trades)

    state_space = _load_and_build_state_space_from_frames(
        df_book, df_trades, session_id, session_features=session_features
    )
    del df_book, df_trades
    gc.collect()
    print(f"  [MATCHED] state_space={state_space.height:,} rows "
          f"x {len(state_space.columns)} cols")
    if state_space.is_empty():
        # Single-hour input cannot satisfy recursive warmup (RSI-14/SMA-14 +
        # shift(1) lags + drop_nulls): ~13 five-minute bars < 14-bar SMA seed.
        # This is a data-span issue, not a transform bug — ingest a wider tag.
        raise ValueError(
            "matched adapter: state_space is EMPTY after drop_nulls. "
            f"{len(files)} file(s) give too short a span for warmup "
            "(need RSI-14/SMA-14 + shift lags). Rerun with a wider --tag "
            "(e.g. --tag 2025-06-30 for the full day) or omit --tag for all files."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)
    out_parquet = os.path.join(MODELS_DIR, f"state_space_{symbol.lower()}_{safe}.parquet")
    state_space = state_space.sort("datetime")
    state_space.write_parquet(out_parquet)
    bounds = state_space.select(
        pl.col("datetime").min().alias("start"), pl.col("datetime").max().alias("end")
    ).row(0)
    manifest = {
        "symbol": symbol,
        "tag": stem,
        "inputs": files,
        "rows": state_space.height,
        "columns": state_space.columns,
        "start": str(bounds[0]),
        "end": str(bounds[1]),
        "session_id": session_id,
        "bar_freq": BAR_FREQ,
        "produced_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "output": out_parquet,
    }
    manifest_path = os.path.join(MODELS_DIR, f"manifest_{symbol.lower()}_{safe}.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  [MATCHED] wrote {out_parquet}")
    print(f"  [MATCHED] wrote {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ingest market_data/matched merged_* parquet into "
                    "strategy_system/strategy/ofmr/models/ state space."
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tag", default=None,
                    help="only use merged files whose name contains TAG (e.g. a date)")
    ap.add_argument("--session-id", type=int, default=0)
    ap.add_argument("--out-name", default=None,
                    help="output stem override (default: TAG or '<N>files')")
    ap.add_argument("--files", nargs="*", default=None,
                    help="explicit merged parquet paths (overrides discovery)")
    args = ap.parse_args(argv)
    manifest = run_matched_to_models(
        symbol=args.symbol, tag=args.tag, session_id=args.session_id,
        paths=args.files, out_name=args.out_name,
    )
    print(json.dumps(
        {k: manifest[k] for k in ("rows", "start", "end", "output")}, indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

