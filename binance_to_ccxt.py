"""
binance_to_ccxt.py
===================
Converts raw Binance historical exports (aggTrades + bookDepth CSVs) into
the CCXT-style trades / orderbook CSV format expected by
`load_and_build_state_space()` in ml_state_optimized.py:

    ticker_trades.csv     -> timestamp, price, amount, side, id
    ticker_orderbook.csv  -> timestamp, bid_price, bid_amount, ask_price, ask_amount

--------------------------------------------------------------------------
IMPORTANT: bookDepth is NOT a real top-of-book (L1 / NBBO) feed
--------------------------------------------------------------------------
Binance's `bookDepth` export gives, per timestamp, CUMULATIVE resting size
in whole-percent bands away from the mid price (percentage = -5..-1 for
bids, +1..+5 for asks; `depth` = cumulative base-asset quantity from the
mid price out to that band, `notional` = cumulative quote value). It does
NOT contain the actual best bid/ask price or size, and it carries no price
of its own — only the percentage offset.

There is therefore no way to recover a true NBBO from this file alone.
This script builds the closest available approximation:
  - reference price = nearest aggTrades trade price at each bookDepth
    timestamp (merge_asof, direction='nearest')
  - bid_price / ask_price = that reference price shifted by the percentage
    band closest to 0% on each side (the finest resolution the file has)
  - bid_amount / ask_amount = the cumulative `depth` of that same closest
    band (this is "size within ~1% of mid", not "size at the best price")

Treat the resulting quotes as a coarse liquidity proxy, not literal
tick-level NBBO. If the research/live pipeline needs real L1 quotes,
source them from Binance's bookTicker stream (best bid/ask push updates)
instead of bookDepth.
--------------------------------------------------------------------------

Usage (CLI):
    python binance_to_ccxt.py \
        --aggtrades ./BTCUSDT_aggtrades.csv \
        --bookdepth ./BTCUSDT_bookdepth.csv \
        --ticker BTCUSDT \
        --outdir ./research/csv

    -> writes ./research/csv/BTCUSDT_trades.csv
             ./research/csv/BTCUSDT_orderbook.csv

Usage (import):
    from binance_to_ccxt import convert
    convert("aggtrades.csv", "bookdepth.csv", "BTCUSDT", "./research/csv")
"""

import argparse
import os

import numpy as np
import pandas as pd


# ============================================================================
# TRADES: Binance aggTrades -> CCXT trades
# ============================================================================
def convert_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    """Core conversion, operating on an already-loaded DataFrame.

    Binance aggTrades columns:
        agg_trade_id, price, quantity, first_trade_id, last_trade_id,
        transact_time, is_buyer_maker

    -> CCXT trades columns:
        timestamp, price, amount, side, id

    `side` is the TAKER (aggressor) side, matching CCXT's convention.
    Binance confirms: is_buyer_maker=True  -> buyer was resting (maker),
    seller was the aggressor -> side='sell'.
                       is_buyer_maker=False -> seller was resting (maker),
    buyer was the aggressor -> side='buy'.
    """
    required = {'agg_trade_id', 'price', 'quantity', 'transact_time', 'is_buyer_maker'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"aggTrades data missing columns: {sorted(missing)}")

    is_buyer_maker = df['is_buyer_maker']
    if is_buyer_maker.dtype != bool:
        parsed = is_buyer_maker.astype(str).str.strip().str.lower().map(
            {'true': True, 'false': False, '1': True, '0': False}
        )
        if parsed.isna().any():
            bad = is_buyer_maker[parsed.isna()].unique()[:5]
            raise ValueError(f"Could not parse is_buyer_maker values, e.g. {bad}")
        is_buyer_maker = parsed

    out = pd.DataFrame({
        'timestamp': df['transact_time'].astype('int64'),
        'price': df['price'].astype(float),
        'amount': df['quantity'].astype(float),
        'side': np.where(is_buyer_maker, 'sell', 'buy'),
        'id': df['agg_trade_id'].astype(str),
    })

    out = out.sort_values('timestamp').reset_index(drop=True)
    return out


def convert_trades(aggtrades_path: str) -> pd.DataFrame:
    """CSV-path convenience wrapper around convert_trades_df()."""
    return convert_trades_df(pd.read_csv(aggtrades_path))


# ============================================================================
# ORDERBOOK: Binance bookDepth -> CCXT orderbook (approximate NBBO)
# ============================================================================
def convert_orderbook_df(df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Core conversion, operating on an already-loaded DataFrame.

    Binance bookDepth columns:
        timestamp, percentage, depth, notional

    -> CCXT orderbook columns:
        timestamp, bid_price, bid_amount, ask_price, ask_amount

    See module docstring for the approximation this relies on. `trades_df`
    must already be in CCXT format (timestamp in ms, price) — pass the
    output of convert_trades_df().

    `timestamp` is accepted either as an epoch (ms) integer or as a
    formatted datetime string (e.g. "2026-08-07 00:00:01") — both forms
    have been seen in Binance's historical exports.
    """
    required = {'timestamp', 'percentage', 'depth', 'notional'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bookDepth data missing columns: {sorted(missing)}")

    df = df.copy()
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype('datetime64[ns]')
    df['percentage'] = df['percentage'].astype(float)

    bids = df[df['percentage'] < 0]
    asks = df[df['percentage'] > 0]
    if bids.empty or asks.empty:
        raise ValueError(
            "bookDepth file has no negative (bid) or positive (ask) "
            "percentage rows — cannot derive a two-sided quote."
        )

    # Band closest to 0% on each side = finest resolution available = best
    # approximation of top-of-book.
    best_bid = bids.loc[bids.groupby('timestamp')['percentage'].idxmax()]
    best_ask = asks.loc[asks.groupby('timestamp')['percentage'].idxmin()]

    best_bid = best_bid.set_index('timestamp')[['percentage', 'depth']].rename(
        columns={'percentage': 'bid_pct', 'depth': 'bid_amount'})
    best_ask = best_ask.set_index('timestamp')[['percentage', 'depth']].rename(
        columns={'percentage': 'ask_pct', 'depth': 'ask_amount'})

    ob = best_bid.join(best_ask, how='outer').sort_index().reset_index()

    # bookDepth carries no price of its own -> borrow a reference price from
    # the nearest aggTrade in time.
    trades_sorted = trades_df[['timestamp', 'price']].sort_values('timestamp').copy()
    trades_sorted['datetime'] = pd.to_datetime(
        trades_sorted['timestamp'], unit='ms'
    ).astype('datetime64[ns]')

    ob = pd.merge_asof(
        ob.sort_values('timestamp'),
        trades_sorted[['datetime', 'price']].sort_values('datetime'),
        left_on='timestamp', right_on='datetime',
        direction='nearest',
    )

    n_missing_ref = ob['price'].isna().sum()
    if n_missing_ref:
        print(f"  Warning: {n_missing_ref} bookDepth snapshot(s) had no nearby "
              f"trade to reference a price from and were dropped.")
        ob = ob.dropna(subset=['price'])

    ob['bid_price'] = ob['price'] * (1.0 + ob['bid_pct'] / 100.0)
    ob['ask_price'] = ob['price'] * (1.0 + ob['ask_pct'] / 100.0)

    ob['timestamp'] = ob['timestamp'].astype('int64') // 1_000_000  # ns -> ms

    out = ob[['timestamp', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount']]
    out = out.dropna(subset=['bid_price', 'ask_price', 'bid_amount', 'ask_amount'])
    out = out.sort_values('timestamp').reset_index(drop=True)

    crossed = out['bid_price'] >= out['ask_price']
    if crossed.any():
        print(f"  Warning: {crossed.sum()} row(s) have bid_price >= ask_price. "
              f"This is inherent to the coarse percentage-band approximation "
              f"(not a bug) — it means the closest bid/ask bands overlapped "
              f"the reference price at that instant.")

    return out


def convert_orderbook(bookdepth_path: str, trades_df: pd.DataFrame) -> pd.DataFrame:
    """CSV-path convenience wrapper around convert_orderbook_df()."""
    return convert_orderbook_df(pd.read_csv(bookdepth_path), trades_df)


# ============================================================================
# ORCHESTRATION
# ============================================================================
def convert(aggtrades_path: str, bookdepth_path: str, ticker: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    print(f"Reading aggTrades: {aggtrades_path}")
    trades = convert_trades(aggtrades_path)

    print(f"Reading bookDepth: {bookdepth_path}")
    orderbook = convert_orderbook(bookdepth_path, trades)

    trades_out = os.path.join(outdir, f"{ticker}_trades.csv")
    ob_out = os.path.join(outdir, f"{ticker}_orderbook.csv")

    trades.to_csv(trades_out, index=False)
    orderbook.to_csv(ob_out, index=False)

    print(f"Wrote {len(trades):,} trades      -> {trades_out}")
    print(f"Wrote {len(orderbook):,} orderbook rows -> {ob_out}")
    return trades_out, ob_out


def main():
    parser = argparse.ArgumentParser(
        description="Convert Binance aggTrades/bookDepth CSVs to CCXT-style trades/orderbook CSVs"
    )
    parser.add_argument('--aggtrades', required=True, help="Path to Binance aggTrades CSV")
    parser.add_argument('--bookdepth', required=True, help="Path to Binance bookDepth CSV")
    parser.add_argument('--ticker', required=True,
                         help="Ticker/session label, e.g. BTCUSDT or BTCUSDT_2 — "
                              "outputs are named {ticker}_trades.csv / {ticker}_orderbook.csv")
    parser.add_argument('--outdir', default='./research/csv',
                         help="Output directory (default: ./research/csv, matching "
                              "load_and_build_state_space()'s default paths)")
    args = parser.parse_args()
    convert(args.aggtrades, args.bookdepth, args.ticker, args.outdir)


if __name__ == '__main__':
    main()