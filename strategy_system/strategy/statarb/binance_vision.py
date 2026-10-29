"""
binance_vision.py -- pull aggTrades + bookDepth straight from Binance Data Vision
(https://data.binance.vision) and aggregate them to fixed-width bars.

Only does I/O + aggregation. All feature engineering lives in flow_features.py.

Notes
-----
* There is no "aggDepth" dataset. The depth feed is `bookDepth`
  (futures only; `um` supported, `cm` rejected, daily files, columns:
  timestamp, percentage, depth, notional). Spot only has aggTrades.
* Daily files are processed one day at a time, reduced to bars immediately and the
  raw rows thrown away, so a month of BTCUSDT never sits in memory.
* The per-day bar frames are cached as parquet, so a re-run costs nothing.
* Row t of the output is labelled by the bar START and holds information up to the
  bar END (depth = last snapshot inside the bar).
* bookDepth is NOT a per-price-level book. Each snapshot is ~10 rows: percentage in
  {-5..-1, 1..5} with CUMULATIVE depth/notional inside that % band around mid. One
  timestamp = several rows, which `bookdepth_to_bars` pivots to one wide row.
* bookTicker (best bid/ask price+qty on every top-of-book change) IS on Data Vision for
  um/cm futures. It is what a genuine Cont-Kukanov-Stoikov (2014) L1 OFI needs, so it is
  optional here (`with_ticker=True`). It is large: BTCUSDT is tens of millions of rows/day.
* If you have a true per-level L2 source (rows = time, price, side, qty; e.g. Tardis, your own
  websocket recorder) use `lob_levels_to_bands` to reduce it to the same bid_n{k}/ask_n{k} columns.
* COIN-M (`cm`) quantities are contracts, not base units, so price*qty is not notional there.
  That market is rejected rather than silently mis-scaled.
"""
from __future__ import annotations

import io
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data"
CHUNK_ROWS = 2_000_000  # aggTrades rows parsed at a time
Fetch = Callable[[str], "bytes | None"]  # url -> zip bytes, or None if the file does not exist


# --------------------------------------------------------------------------- urls / http
def norm_symbol(symbol: str) -> str:
    """'BTC/USDT' or 'BTC/USDT:USDT' -> 'BTCUSDT'."""
    return symbol.split(":")[0].replace("/", "").upper()


def vision_url(market: str, dtype: str, symbol: str, day: pd.Timestamp) -> str:
    root = "spot" if market == "spot" else f"futures/{market}"
    s = norm_symbol(symbol)
    return f"{BASE_URL}/{root}/daily/{dtype}/{s}/{s}-{dtype}-{day:%Y-%m-%d}.zip"


def http_fetch(url: str, retries: int = 4, timeout: int = 120) -> bytes | None:
    import requests

    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
        except requests.RequestException:
            time.sleep(2**i)
            continue
        if r.status_code == 404:
            return None
        if r.ok:
            return r.content
        time.sleep(2**i)
    raise RuntimeError(f"download failed after {retries} tries: {url}")


def _bar_ms(bar: str) -> int:
    ms = int(pd.Timedelta(bar).total_seconds() * 1000)
    if ms <= 0 or 86_400_000 % ms:
        raise ValueError(f"bar={bar!r} must divide one day evenly")
    return ms


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# --------------------------------------------------------------------------- parsers
_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "vol": "sum", "notional": "sum", "buy_vol": "sum", "buy_notional": "sum",
    "n_trades": "sum", "n_buy": "sum", "rv": "sum", "max_trade": "max",
}


def aggtrades_to_bars(blob: bytes, bar: str) -> pd.DataFrame:
    """aggTrades zip -> bar frame. Every column is mergeable across chunks.

    taker buy  <=> is_buyer_maker == False  (the aggressor was the buyer)
    rv         = sum of squared log tick-to-tick returns inside the bar
    """
    bar_ms = _bar_ms(bar)
    parts: list[pd.DataFrame] = []
    prev_lp: float | None = None

    with zipfile.ZipFile(io.BytesIO(blob)) as zf, zf.open(zf.namelist()[0]) as fh:
        has_header = not fh.peek(1)[:1].isdigit()  # older files have no header row
        reader = pd.read_csv(
            fh,
            header=None,
            skiprows=1 if has_header else 0,
            usecols=[1, 2, 5, 6],  # price, quantity, transact_time, is_buyer_maker
            names=["price", "qty", "time", "ibm"],
            dtype={"price": "float64", "qty": "float64", "time": "int64", "ibm": "bool"},
            true_values=["true", "True", "TRUE"],
            false_values=["false", "False", "FALSE"],
            chunksize=CHUNK_ROWS,
        )
        for ch in reader:
            t = ch["time"].to_numpy(np.int64)
            t = np.where(t > 10**14, t // 1000, t)  # spot files moved to microseconds
            p = ch["price"].to_numpy(float)
            q = ch["qty"].to_numpy(float)
            buy = ~ch["ibm"].to_numpy(bool)

            lp = np.log(p)
            r = np.diff(lp, prepend=lp[0] if prev_lp is None else prev_lp)
            prev_lp = lp[-1]
            nv = p * q

            g = (
                pd.DataFrame(
                    {
                        "bar": t // bar_ms * bar_ms, "p": p, "q": q, "nv": nv,
                        "bq": q * buy, "bn": nv * buy, "buy": buy.astype(np.int64), "r2": r * r,
                    }
                )
                .groupby("bar", sort=True)
                .agg(
                    open=("p", "first"), high=("p", "max"), low=("p", "min"), close=("p", "last"),
                    vol=("q", "sum"), notional=("nv", "sum"), buy_vol=("bq", "sum"),
                    buy_notional=("bn", "sum"), n_trades=("p", "size"), n_buy=("buy", "sum"),
                    rv=("r2", "sum"), max_trade=("nv", "max"),
                )
            )
            parts.append(g)

    out = pd.concat(parts).groupby(level=0, sort=True).agg(_AGG)
    out.index = pd.to_datetime(out.index, unit="ms", utc=True)
    out.index.name = "datetime"
    return out


def bookdepth_to_bars(blob: bytes, bar: str) -> pd.DataFrame:
    """bookDepth zip -> last snapshot per bar, columns bid_n{k}/ask_n{k} (notional within k% of mid)."""
    _bar_ms(bar)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        df = pd.read_csv(zf.open(zf.namelist()[0]))
    need = {"timestamp", "percentage", "notional"}
    if not need <= set(df.columns):
        raise ValueError(f"unexpected bookDepth columns: {list(df.columns)}")

    ts = df["timestamp"]
    ts = pd.to_datetime(ts, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True)
    piv = (
        df.assign(ts=ts)
        .pivot_table(index="ts", columns="percentage", values="notional", aggfunc="last")
        .sort_index()
    )
    piv.columns = [f"{'bid' if c < 0 else 'ask'}_n{abs(c):g}" for c in piv.columns]
    out = piv.groupby(piv.index.floor(bar)).last()
    out.index.name = "datetime"
    return out



_TICKER_COLS = ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time"]
_TICKER_AGG = {
    "bt_ofi": "sum", "bt_bid": "last", "bt_bq": "last", "bt_ask": "last", "bt_aq": "last", "bt_n": "sum",
}


def bookticker_to_bars(blob: bytes, bar: str) -> pd.DataFrame:
    """bookTicker zip -> per-bar L1 order-flow imbalance (Cont, Kukanov & Stoikov 2014).

    For every top-of-book update n (previous update n-1):
        e_n =  1[Pb_n >= Pb_n-1] * qb_n  -  1[Pb_n <= Pb_n-1] * qb_n-1
             - 1[Pa_n <= Pa_n-1] * qa_n  +  1[Pa_n >= Pa_n-1] * qa_n-1
    OFI over a bar = sum of e_n inside the bar. Units: base asset (normalise downstream).
    Chunks are stitched by carrying the last row of the previous chunk, so no event is lost.
    """
    bar_ms = _bar_ms(bar)
    parts: list[pd.DataFrame] = []
    carry: np.ndarray | None = None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf, zf.open(zf.namelist()[0]) as fh:
        reader = pd.read_csv(fh, usecols=_TICKER_COLS, chunksize=CHUNK_ROWS)
        for ch in reader:
            arr = ch[_TICKER_COLS].to_numpy(np.float64)
            skip = 0
            if carry is not None:
                arr = np.vstack([carry, arr])
                skip = 1
            pb, qb, pa, qa, t = arr.T
            e = np.zeros(len(arr))
            e[1:] = (
                (pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
                - (pa[1:] <= pa[:-1]) * qa[1:] + (pa[1:] >= pa[:-1]) * qa[:-1]
            )
            carry = arr[-1:].copy()
            arr, e = arr[skip:], e[skip:]
            if not len(arr):
                continue
            tt = arr[:, 4].astype(np.int64)
            tt = np.where(tt > 10**14, tt // 1000, tt)
            g = (
                pd.DataFrame({"bar": tt // bar_ms * bar_ms, "e": e, "pb": arr[:, 0], "qb": arr[:, 1],
                              "pa": arr[:, 2], "qa": arr[:, 3]})
                .groupby("bar", sort=True)
                .agg(bt_ofi=("e", "sum"), bt_bid=("pb", "last"), bt_bq=("qb", "last"),
                     bt_ask=("pa", "last"), bt_aq=("qa", "last"), bt_n=("e", "size"))
            )
            parts.append(g)
    out = pd.concat(parts).groupby(level=0, sort=True).agg(_TICKER_AGG)
    out.index = pd.to_datetime(out.index, unit="ms", utc=True)
    out.index.name = "datetime"
    return out


def lob_levels_to_bands(
    df: pd.DataFrame,
    bar: str,
    bands: tuple[float, ...] = (1, 2, 3, 4, 5),
    time_col: str = "time",
    price_col: str = "price",
    side_col: str = "side",
    qty_col: str = "qty",
) -> pd.DataFrame:
    """Per-level L2 SNAPSHOT rows (time, price, side, qty; one row per resting level) -> the same
    bid_n{k}/ask_n{k} columns bookDepth gives, so flow_features works with either source.

    Every timestamp must be a full snapshot. If your rows are incremental updates you must
    replay them into a book first (a dict price -> qty per side) and snapshot that.
    notional within k% = sum(price*qty) over levels with price >= mid*(1-k%) (bids) or <= mid*(1+k%) (asks);
    mid = (best bid + best ask)/2 of the same snapshot. Last snapshot inside each bar is kept.
    """
    _bar_ms(bar)
    d = df[[time_col, price_col, side_col, qty_col]].copy()
    ts = d[time_col]
    ts = pd.to_datetime(ts, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True)
    side = d[side_col].astype(str).str.lower()
    is_bid = side.isin(["bid", "b", "buy"])
    if not (is_bid | side.isin(["ask", "a", "sell", "offer"])).all():
        raise ValueError(f"unrecognised side values: {sorted(side.unique())}")
    p, nv = d[price_col].astype(float), d[price_col].astype(float) * d[qty_col].astype(float)
    key = ts.rename("ts")
    best_bid = p.where(is_bid).groupby(key).transform("max")
    best_ask = p.where(~is_bid).groupby(key).transform("min")
    mid = (best_bid + best_ask) / 2
    cols = {}
    for k in bands:
        cols[f"bid_n{k:g}"] = nv.where(is_bid & (p >= mid * (1 - k / 100)), 0.0).groupby(key).sum()
        cols[f"ask_n{k:g}"] = nv.where(~is_bid & (p <= mid * (1 + k / 100)), 0.0).groupby(key).sum()
    snap = pd.DataFrame(cols).sort_index()
    out = snap.groupby(snap.index.floor(bar)).last()
    out.index.name = "datetime"
    return out


# --------------------------------------------------------------------------- per-day cache
def _day_frame(symbol, day, dtype, market, bar, cache: Path, fetch: Fetch, parser) -> pd.DataFrame | None:
    folder = cache / market / dtype / bar / norm_symbol(symbol)
    path = folder / f"{day:%Y-%m-%d}.parquet"
    miss = path.with_suffix(".missing")
    if path.exists():
        return pd.read_parquet(path)
    if miss.exists():
        return None

    blob = fetch(vision_url(market, dtype, symbol, day))
    folder.mkdir(parents=True, exist_ok=True)
    if blob is None:
        # files for the last ~2 days are not published yet: don't remember those as missing
        if day < pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2):
            miss.touch()
        return None

    df = parser(blob, bar)
    df.to_parquet(path)
    return df


# --------------------------------------------------------------------------- public API
def load_bars(
    symbol: str,
    start,
    end,
    bar: str = "5min",
    market: str = "um",
    with_depth: bool = True,
    cache_dir: str | Path = "~/.cache/binance_vision",
    workers: int = 4,
    depth_ffill_limit: int = 12,
    fetch: Fetch = http_fetch,
    with_ticker: bool = False,
    ticker_ffill_limit: int = 12,
) -> pd.DataFrame:
    """Download (or read from cache) aggTrades [+ bookDepth] for every day in [start, end]
    and return one bar frame (UTC index). Days that don't exist on Data Vision are skipped."""
    cache = Path(cache_dir).expanduser()
    if market == "cm":
        raise NotImplementedError(
            "COIN-M quantities are contracts (USD per contract), so price*qty is not notional. "
            "Use market='um' (or 'spot') or add a contract-size conversion first."
        )
    if with_ticker and market == "spot":
        warnings.warn("bookTicker is a futures dataset here; continuing without L1 OFI.")
        with_ticker = False
    if with_depth and market == "spot":
        warnings.warn("bookDepth is not published for spot; continuing with trade features only.")
        with_depth = False

    days = pd.date_range(_utc(start).normalize(), _utc(end).normalize(), freq="D")

    def job(day):
        tr = _day_frame(symbol, day, "aggTrades", market, bar, cache, fetch, aggtrades_to_bars)
        dp = _day_frame(symbol, day, "bookDepth", market, bar, cache, fetch, bookdepth_to_bars) if with_depth else None
        bt = _day_frame(symbol, day, "bookTicker", market, bar, cache, fetch, bookticker_to_bars) if with_ticker else None
        return day, tr, dp, bt

    with ThreadPoolExecutor(workers) as ex:
        results = list(tqdm(ex.map(job, days), total=len(days), desc=norm_symbol(symbol), leave=False))

    trade_parts, depth_parts, tick_parts = [], [], []
    for day, tr, dp, bt in results:
        if tr is None or tr.empty:
            continue
        grid = pd.date_range(day, day + pd.Timedelta(days=1), freq=bar, inclusive="left")
        trade_parts.append(tr.reindex(grid))
        if dp is not None and not dp.empty:
            depth_parts.append(dp.reindex(grid))
        if bt is not None and not bt.empty:
            tick_parts.append(bt.reindex(grid))
    if not trade_parts:
        raise ValueError(f"No aggTrades found for {symbol} ({market}) between {start} and {end}.")

    bars = pd.concat(trade_parts)
    bars.index.name = "datetime"
    bars["close"] = bars["close"].ffill()
    for c in ("open", "high", "low"):
        bars[c] = bars[c].fillna(bars["close"])
    additive = ["vol", "notional", "buy_vol", "buy_notional", "n_trades", "n_buy", "rv", "max_trade"]
    bars[additive] = bars[additive].fillna(0.0)

    if depth_parts:
        depth = pd.concat(depth_parts).ffill(limit=depth_ffill_limit)
        if {"bid_n1", "bid_n5"} <= set(depth.columns):
            if (depth["bid_n5"] < 0.999 * depth["bid_n1"]).mean() > 0.2:
                warnings.warn(
                    f"{symbol}: bookDepth bands do not look cumulative (n5 < n1 often). "
                    "Check the file format before trusting depth_slope."
                )
        bars = bars.join(depth)

    if tick_parts:
        tick = pd.concat(tick_parts)
        tick[["bt_ofi", "bt_n"]] = tick[["bt_ofi", "bt_n"]].fillna(0.0)   # no update in bar -> no flow
        quote = ["bt_bid", "bt_bq", "bt_ask", "bt_aq"]
        tick[quote] = tick[quote].ffill(limit=ticker_ffill_limit)
        bars = bars.join(tick)
    return bars