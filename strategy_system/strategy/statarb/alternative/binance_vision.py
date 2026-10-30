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
* Extra futures datasets (all optional, all `um` only; a missing file just leaves NaN columns):
    metrics            5-min snapshots: open interest, top-trader / account long-short ratios, taker buy/sell vol ratio
    fundingRate        MONTHLY file of funding settlements, forward-filled onto the bars from the settlement time
    premiumIndexKlines premium of the perp over the index (basis), same interval as the bars
    liquidationSnapshot forced orders. Binance keeps only the LAST liquidation per second per symbol, so this
                       UNDER-counts large cascades, and the daily files stop/gap for some periods. Treat as a lower bound.
  Snapshots are labelled with the bar that CONTAINS their timestamp (last value in the bar), so a value stamped
  exactly on a bar boundary is only usable at the end of that bar -- never earlier. I could not verify the exact
  column layout of these files against the live site from the sandbox, so every parser maps columns by NAME and
  fails loudly (ValueError) instead of guessing.
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


def vision_url(market: str, dtype: str, symbol: str, day: pd.Timestamp,
               interval: str | None = None, freq: str = "daily") -> str:
    """Data Vision url. `interval` is for the kline-style datasets (folder + file carry the interval);
    freq='monthly' is for datasets that only exist as monthly files (fundingRate)."""
    root = "spot" if market == "spot" else f"futures/{market}"
    s = norm_symbol(symbol)
    stamp = f"{day:%Y-%m}" if freq == "monthly" else f"{day:%Y-%m-%d}"
    tail = f"{s}/{interval}/{s}-{interval}-{stamp}" if interval else f"{s}/{s}-{dtype}-{stamp}"
    return f"{BASE_URL}/{root}/{freq}/{dtype}/{tail}.zip"


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


# --------------------------------------------------------------------------- extra futures datasets
_KLINE_INTERVALS = {"1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h",
                    "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d"}


def kline_interval(bar: str) -> str:
    key = str(pd.Timedelta(bar).to_pytimedelta().total_seconds())
    for k, v in _KLINE_INTERVALS.items():
        if str(pd.Timedelta(k).total_seconds()) == key:
            return v
    raise ValueError(f"bar={bar!r} is not a Binance kline interval; premium index needs one of {list(_KLINE_INTERVALS)}")


def _to_utc(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(ts):
        unit = "us" if ts.iloc[0] > 10**14 else "ms"
        return pd.to_datetime(ts, unit=unit, utc=True)
    return pd.to_datetime(ts, utc=True)


def _read_zip_csv(blob: bytes, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return pd.read_csv(zf.open(zf.namelist()[0]), **kw)


_METRICS_MAP = {
    "sum_open_interest": "oi",                           # base-asset units
    "sum_open_interest_value": "oi_val",                 # USD
    "count_toptrader_long_short_ratio": "ls_top_cnt",    # top traders, by ACCOUNT count
    "sum_toptrader_long_short_ratio": "ls_top_pos",      # top traders, by POSITION size
    "count_long_short_ratio": "ls_acct",                 # all accounts
    "sum_taker_long_short_vol_ratio": "taker_ls",        # taker buy vol / taker sell vol
}


def metrics_to_bars(blob: bytes, bar: str) -> pd.DataFrame:
    """metrics zip (5-min snapshots) -> last snapshot per bar. Columns: oi, oi_val, ls_top_cnt, ls_top_pos, ls_acct, taker_ls."""
    _bar_ms(bar)
    df = _read_zip_csv(blob)
    if "create_time" not in df.columns or "sum_open_interest" not in df.columns:
        raise ValueError(f"unexpected metrics columns: {list(df.columns)}")
    ts = _to_utc(df["create_time"])
    out = df[[c for c in _METRICS_MAP if c in df.columns]].rename(columns=_METRICS_MAP).apply(pd.to_numeric, errors="coerce")
    out.index = ts
    out = out.sort_index().groupby(out.index.floor(bar)).last()
    out.index.name = "datetime"
    return out


def funding_to_events(blob: bytes, bar: str = "") -> pd.DataFrame:
    """fundingRate monthly zip -> one row per settlement (index = settlement time). Not bar-aggregated."""
    df = _read_zip_csv(blob)
    need = {"calc_time", "last_funding_rate"}
    if not need <= set(df.columns):
        raise ValueError(f"unexpected fundingRate columns: {list(df.columns)}")
    out = pd.DataFrame({"funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce").to_numpy()},
                       index=_to_utc(df["calc_time"]))
    if "funding_interval_hours" in df.columns:
        out["funding_hours"] = pd.to_numeric(df["funding_interval_hours"], errors="coerce").to_numpy()
    out.index.name = "datetime"
    return out.sort_index()


_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
               "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def premium_to_bars(blob: bytes, bar: str) -> pd.DataFrame:
    """premiumIndexKlines zip -> `prem` = last premium-index close inside the bar (fraction, 0.001 = 10 bps)."""
    _bar_ms(bar)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf, zf.open(zf.namelist()[0]) as fh:
        has_header = not fh.peek(1)[:1].isdigit()
        df = pd.read_csv(fh, header=None, skiprows=1 if has_header else 0)
    df = df.iloc[:, :len(_KLINE_COLS)]
    df.columns = _KLINE_COLS[: df.shape[1]]
    if "close" not in df.columns:
        raise ValueError(f"unexpected premiumIndexKlines shape: {df.shape}")
    out = pd.DataFrame({"prem": pd.to_numeric(df["close"], errors="coerce").to_numpy()}, index=_to_utc(df["open_time"]))
    out = out.sort_index().groupby(out.index.floor(bar)).last()
    out.index.name = "datetime"
    return out


_LIQ_NAMES = ["time", "side", "order_type", "time_in_force", "original_quantity", "price", "average_price",
              "order_status", "last_fill_quantity", "accumulated_fill_quantity"]


def liquidation_to_bars(blob: bytes, bar: str) -> pd.DataFrame:
    """liquidationSnapshot zip -> forced-order notional per bar.

    A liquidated LONG is a forced SELL order, a liquidated SHORT a forced BUY order.
    Columns: liq_long_n, liq_short_n (quote notional), liq_long_c, liq_short_c (counts).
    """
    _bar_ms(bar)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf, zf.open(zf.namelist()[0]) as fh:
        head = fh.peek(64)[:64]
        has_header = not head[:1].isdigit()
        df = pd.read_csv(fh, header=0 if has_header else None)
    if not has_header:
        df.columns = _LIQ_NAMES[: df.shape[1]]
    if not {"time", "side"} <= set(df.columns):
        raise ValueError(f"unexpected liquidationSnapshot columns: {list(df.columns)}")
    px_ = pd.to_numeric(df.get("average_price", 0), errors="coerce")
    px_ = px_.where(px_ > 0, pd.to_numeric(df.get("price", np.nan), errors="coerce"))
    qty = pd.to_numeric(df.get("accumulated_fill_quantity", np.nan), errors="coerce")
    qty = qty.where(qty > 0, pd.to_numeric(df.get("original_quantity", np.nan), errors="coerce"))
    nv = (px_ * qty).to_numpy(float)
    is_sell = df["side"].astype(str).str.upper().eq("SELL").to_numpy()
    ts = _to_utc(df["time"])
    d = pd.DataFrame({"liq_long_n": np.where(is_sell, nv, 0.0), "liq_short_n": np.where(~is_sell, nv, 0.0),
                      "liq_long_c": is_sell.astype(float), "liq_short_c": (~is_sell).astype(float)}, index=ts)
    out = d.sort_index().groupby(d.sort_index().index.floor(bar)).sum()
    out.index.name = "datetime"
    return out


# --------------------------------------------------------------------------- per-day cache
def _day_frame(symbol, day, dtype, market, bar, cache: Path, fetch: Fetch, parser,
               interval: str | None = None, freq: str = "daily") -> pd.DataFrame | None:
    """Fetch + parse + parquet-cache one file (one day, or one month when freq='monthly')."""
    folder = cache / market / dtype / (interval or bar) / norm_symbol(symbol)
    stamp = f"{day:%Y-%m}" if freq == "monthly" else f"{day:%Y-%m-%d}"
    path = folder / f"{stamp}.parquet"
    miss = path.with_suffix(".missing")
    if path.exists():
        return pd.read_parquet(path)
    if miss.exists():
        return None

    blob = fetch(vision_url(market, dtype, symbol, day, interval=interval, freq=freq))
    folder.mkdir(parents=True, exist_ok=True)
    if blob is None:
        # the newest files are not published yet: don't remember those as missing
        lag = pd.Timedelta(days=2) if freq == "daily" else pd.Timedelta(days=40)
        if day < pd.Timestamp.now(tz="UTC").normalize() - lag:
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
    with_metrics: bool = True,
    with_funding: bool = True,
    with_premium: bool = True,
    with_liq: bool = True,
    extra_ffill_limit: int = 12,
) -> pd.DataFrame:
    """Download (or read from cache) aggTrades [+ bookDepth, bookTicker, metrics, premium index, funding,
    liquidations] for every day in [start, end] and return one bar frame (UTC index).
    Days that don't exist on Data Vision are skipped; a missing EXTRA dataset leaves its columns NaN
    (or absent when no day of it exists at all), it never drops trade bars."""
    cache = Path(cache_dir).expanduser()
    if market == "cm":
        raise NotImplementedError(
            "COIN-M quantities are contracts (USD per contract), so price*qty is not notional. "
            "Use market='um' (or 'spot') or add a contract-size conversion first."
        )
    if market == "spot":
        for name, on in (("bookTicker", with_ticker), ("bookDepth", with_depth), ("metrics", with_metrics),
                         ("fundingRate", with_funding), ("premiumIndexKlines", with_premium),
                         ("liquidationSnapshot", with_liq)):
            if on:
                warnings.warn(f"{name} is a futures dataset; continuing without it on spot.")
        with_ticker = with_depth = with_metrics = with_funding = with_premium = with_liq = False
    interval = kline_interval(bar) if with_premium else None

    days = pd.date_range(_utc(start).normalize(), _utc(end).normalize(), freq="D")

    def job(day):
        tr = _day_frame(symbol, day, "aggTrades", market, bar, cache, fetch, aggtrades_to_bars)
        if tr is None or tr.empty:
            return day, None, {}
        ex = {}
        if with_depth:
            ex["depth"] = _day_frame(symbol, day, "bookDepth", market, bar, cache, fetch, bookdepth_to_bars)
        if with_ticker:
            ex["tick"] = _day_frame(symbol, day, "bookTicker", market, bar, cache, fetch, bookticker_to_bars)
        if with_metrics:
            ex["metrics"] = _day_frame(symbol, day, "metrics", market, bar, cache, fetch, metrics_to_bars)
        if with_premium:
            ex["prem"] = _day_frame(symbol, day, "premiumIndexKlines", market, bar, cache, fetch,
                                    premium_to_bars, interval=interval)
        if with_liq:
            ex["liq"] = _day_frame(symbol, day, "liquidationSnapshot", market, bar, cache, fetch, liquidation_to_bars)
        return day, tr, ex

    with ThreadPoolExecutor(workers) as ex_:
        results = list(tqdm(ex_.map(job, days), total=len(days), desc=norm_symbol(symbol), leave=False))

    trade_parts: list[pd.DataFrame] = []
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in ("depth", "tick", "metrics", "prem", "liq")}
    for day, tr, ex in results:
        if tr is None or tr.empty:
            continue
        grid = pd.date_range(day, day + pd.Timedelta(days=1), freq=bar, inclusive="left")
        trade_parts.append(tr.reindex(grid))
        for k, df in ex.items():
            if df is not None and not df.empty:
                parts[k].append(df.reindex(grid))
    if not trade_parts:
        raise ValueError(f"No aggTrades found for {symbol} ({market}) between {start} and {end}.")

    bars = pd.concat(trade_parts)
    bars.index.name = "datetime"
    bars["close"] = bars["close"].ffill()
    for c in ("open", "high", "low"):
        bars[c] = bars[c].fillna(bars["close"])
    additive = ["vol", "notional", "buy_vol", "buy_notional", "n_trades", "n_buy", "rv", "max_trade"]
    bars[additive] = bars[additive].fillna(0.0)

    if parts["depth"]:
        depth = pd.concat(parts["depth"]).ffill(limit=depth_ffill_limit)
        if {"bid_n1", "bid_n5"} <= set(depth.columns):
            if (depth["bid_n5"] < 0.999 * depth["bid_n1"]).mean() > 0.2:
                warnings.warn(
                    f"{symbol}: bookDepth bands do not look cumulative (n5 < n1 often). "
                    "Check the file format before trusting depth_slope."
                )
        bars = bars.join(depth)

    if parts["tick"]:
        tick = pd.concat(parts["tick"])
        tick[["bt_ofi", "bt_n"]] = tick[["bt_ofi", "bt_n"]].fillna(0.0)   # no update in bar -> no flow
        quote = ["bt_bid", "bt_bq", "bt_ask", "bt_aq"]
        tick[quote] = tick[quote].ffill(limit=ticker_ffill_limit)
        bars = bars.join(tick)

    if parts["metrics"]:                       # open interest, long/short ratios (state variables: carry forward)
        bars = bars.join(pd.concat(parts["metrics"]).ffill(limit=extra_ffill_limit))
    if parts["prem"]:
        bars = bars.join(pd.concat(parts["prem"]).ffill(limit=extra_ffill_limit))
    if parts["liq"]:                           # file exists for the day -> no forced order means zero, not NaN
        bars = bars.join(pd.concat(parts["liq"]))
        lc = [c for c in bars.columns if c.startswith("liq_")]
        have_day = bars.index.floor("D").isin([d for d, tr, ex in results if ex.get("liq") is not None])
        bars.loc[have_day, lc] = bars.loc[have_day, lc].fillna(0.0)

    if with_funding:
        fund = _load_funding(symbol, bars.index[0], bars.index[-1], market, cache, fetch)
        if fund is not None:
            # settlement value is known from its settlement time on: carry it forward onto the bars
            bars = bars.join(fund.reindex(bars.index, method="ffill"))
    return bars


def _load_funding(symbol, t0: pd.Timestamp, t1: pd.Timestamp, market: str, cache: Path, fetch: Fetch):
    """Monthly fundingRate files covering [t0, t1], plus the month before so the first bars have a value."""
    first = t0.normalize().replace(day=1) - pd.offsets.MonthBegin(1)
    months = pd.date_range(first, t1.normalize().replace(day=1), freq="MS")
    frames = [_day_frame(symbol, m, "fundingRate", market, "", cache, fetch, funding_to_events, freq="monthly")
              for m in months]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None
    ev = pd.concat(frames)
    return ev[~ev.index.duplicated(keep="last")].sort_index()