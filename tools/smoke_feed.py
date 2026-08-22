#!/usr/bin/env python3
"""Bounded smoke test for the streaming live-test feed builder."""
import sys, tempfile, os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from unified_runner import _build_live_test_feed  # noqa: E402

TRADES = pd.DataFrame({
    "timestamp": [100, 102, 105, 110],
    "price": [1.0, 2.0, 3.0, 9.0],
    "amount": [0.5, 0.2, 0.1, 1.0],
    "side": ["buy", "sell", "buy", "buy"],
    "id": ["1", "2", "3", "9"],
})
BOOK = pd.DataFrame({
    "timestamp": [101, 103, 110],
    "bid_price": [99.9, 99.8, 50.0],
    "bid_amount": [0.1, 0.2, 0.5],
    "ask_price": [100.1, 100.2, 51.0],
    "ask_amount": [0.2, 0.3, 0.6],
})


def write_parquet(df, path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def write_csv(df, path):
    df.to_csv(path, index=False)


def run_case(fmt):
    tmp = Path(tempfile.mkdtemp())
    ext = ".parquet" if fmt == "parquet" else ".csv"
    write_parquet(TRADES, tmp / f"BTCUSDT_trades{ext}") if fmt == "parquet" \
        else write_csv(TRADES, tmp / f"BTCUSDT_trades{ext}")
    write_parquet(BOOK, tmp / f"BTCUSDT_orderbook{ext}") if fmt == "parquet" \
        else write_csv(BOOK, tmp / f"BTCUSDT_orderbook{ext}")
    out = tmp / "feed.replay"
    err = _build_live_test_feed("btcusdt", feed_dir=tmp, out_path=out)
    assert err is None, f"builder returned error ({fmt}): {err}"
    lines = out.read_text().splitlines()
    assert lines[0] == "#TICKER btcusdt", lines[0]
    body = lines[1:]
    assert len(body) == 7, f"[{fmt}] expected 7 events got {len(body)}: {body}"
    # ts, trades-before-book-on-tie ordering
    order = [(l[0], int(l.split()[1])) for l in body]
    assert order == [("T", 100), ("B", 101), ("T", 102), ("B", 103), ("T", 105), ("T", 110), ("B", 110)], order
    # tie at 110: T line precedes B line
    idx_t110 = next(i for i, l in enumerate(body) if l.startswith("T 110"))
    idx_b110 = next(i for i, l in enumerate(body) if l.startswith("B 110"))
    assert idx_t110 < idx_b110, "trade must precede book top at equal ms"
    print(f"[{fmt}] merge+ordering+tie OK ({len(body)} events)")


def run_window(fmt):
    tmp = Path(tempfile.mkdtemp())
    ext = ".parquet" if fmt == "parquet" else ".csv"
    (tmp / f"BTCUSDT_trades{ext}").write_text("")  # not used
    if fmt == "parquet":
        write_parquet(TRADES, tmp / f"BTCUSDT_trades{ext}")
        write_parquet(BOOK, tmp / f"BTCUSDT_orderbook{ext}")
    else:
        write_csv(TRADES, tmp / f"BTCUSDT_trades{ext}")
        write_csv(BOOK, tmp / f"BTCUSDT_orderbook{ext}")
    out = tmp / "feed.replay"
    err = _build_live_test_feed("btcusdt", feed_dir=tmp, out_path=out, start_ms=102, end_ms=103)
    assert err is None, err
    body = out.read_text().splitlines()[1:]
    ts = [int(l.split()[1]) for l in body]
    assert ts == [102, 103], f"[{fmt}] window mismatch: {ts}"
    print(f"[{fmt}] window [102,103] OK ({len(body)} events)")


def run_limit(fmt):
    tmp = Path(tempfile.mkdtemp())
    ext = ".parquet" if fmt == "parquet" else ".csv"
    if fmt == "parquet":
        write_parquet(TRADES, tmp / f"BTCUSDT_trades{ext}")
        write_parquet(BOOK, tmp / f"BTCUSDT_orderbook{ext}")
    else:
        write_csv(TRADES, tmp / f"BTCUSDT_trades{ext}")
        write_csv(BOOK, tmp / f"BTCUSDT_orderbook{ext}")
    out = tmp / "feed.replay"
    err = _build_live_test_feed("btcusdt", feed_dir=tmp, out_path=out, limit=3)
    assert err is None, err
    body = out.read_text().splitlines()[1:]
    assert len(body) == 3, f"[{fmt}] limit mismatch: {len(body)}"
    assert [l[0] for l in body] == ["T", "B", "T"], body
    print(f"[{fmt}] limit=3 OK ({len(body)} events)")


def run_trades_only(fmt):
    tmp = Path(tempfile.mkdtemp())
    ext = ".parquet" if fmt == "parquet" else ".csv"
    if fmt == "parquet":
        write_parquet(TRADES, tmp / f"BTCUSDT_trades{ext}")
    else:
        write_csv(TRADES, tmp / f"BTCUSDT_trades{ext}")
    out = tmp / "feed.replay"
    err = _build_live_test_feed("btcusdt", feed_dir=tmp, out_path=out)
    assert err is None, err
    body = out.read_text().splitlines()[1:]
    assert len(body) == 4, body
    print(f"[{fmt}] trades-only OK ({len(body)} events)")


if __name__ == "__main__":
    for fmt in ("parquet", "csv"):
        run_case(fmt)
        run_window(fmt)
        run_limit(fmt)
        run_trades_only(fmt)
    print("\nALL SMOKE TESTS PASSED")
