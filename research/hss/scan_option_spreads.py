"""
scan_option_spreads.py

Live-scans all active Deribit options for a given currency over websocket
ticker streams and ranks them by bid-ask spread. This is step 1 of the MM
workflow: find the widest-spread candidates. Hedging (using other options
or the underlying to control Greeks, aimed at maximizing Sortino) is a
separate piece -- not attempted here, see the note at the bottom.

Meant to live in research/hss/ alongside its CSV output.

Usage:
    python scan_option_spreads.py --currency BTC
    python scan_option_spreads.py --currency BTC --min-oi 5 --interval 15
    python scan_option_spreads.py --currency ETH --top 20 --max-days-to-expiry 30

Notes:
    - Watches EVERY active option matching the filters, live, over
      websocket -- there's no way to know who has the widest spread right
      now without watching broadly. Use --max-days-to-expiry and --min-oi
      to cut the universe down to something liquid enough to actually be
      tradeable; otherwise you'll rank near-dead, far-OTM contracts with a
      "wide spread" that's really just an absent order book.
    - ccxt.pro multiplexes all watch_ticker calls for one exchange instance
      over a single websocket connection, so this isn't opening hundreds of
      sockets -- but subscription messages are staggered on startup to
      avoid bursting the rate limit.
    - `open_interest` / `mark_iv` / `underlying_price` are pulled from the
      raw Deribit payload (ticker['info']). Field names there come straight
      from Deribit's API and haven't been verified against a live response
      in this environment -- print(ticker['info']) once to confirm if the
      snapshot columns come back empty.
    - Deribit's ticker payload also carries exchange-computed Greeks
      (info['greeks']: delta, gamma, vega, theta, rho), fitted off their own
      vol surface -- more accurate than recomputing Black-Scholes off a flat
      mark_iv. Captured here so live hedging decisions can read them
      straight from the stream instead of re-deriving them.
"""

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import ccxt.pro as ccxtpro
import ccxt


def parse_args():
    p = argparse.ArgumentParser(description="Live-scan Deribit options for widest bid-ask spread")
    p.add_argument("--currency", default="BTC", help="BTC or ETH (default: BTC)")
    p.add_argument("--top", type=int, default=25,
                    help="How many widest-spread options to keep in the snapshot (default: 25)")
    p.add_argument("--interval", type=float, default=10.0,
                    help="Seconds between snapshot writes (default: 10)")
    p.add_argument("--max-days-to-expiry", type=int, default=None,
                    help="Skip options expiring further out than this many days (default: no filter)")
    p.add_argument("--min-oi", type=float, default=0,
                    help="Skip options with open interest below this in the ranking "
                         "(default: 0, no filter). Guards against dead markets with a "
                         "'wide spread' that's really just an empty book.")
    p.add_argument("--output", default=None,
                    help="Snapshot CSV filename (default: <currency>_option_spreads.csv)")
    return p.parse_args()


def days_to_expiry(market_info, now_ms):
    expiry = market_info.get("expiry")
    if not expiry:
        return None
    return (expiry - now_ms) / (1000 * 60 * 60 * 24)


async def watch_one(exchange, symbol, state):
    while True:
        try:
            ticker = await exchange.watch_ticker(symbol)
        except Exception as e:
            print(f"[{symbol}] watch error: {e}")
            await asyncio.sleep(3)
            continue

        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            continue

        mid = (bid + ask) / 2
        spread_abs = ask - bid
        spread_pct = (spread_abs / mid) * 100 if mid else None

        info = ticker.get("info", {}) or {}
        greeks = info.get("greeks", {}) or {}
        state[symbol] = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_abs": spread_abs,
            "spread_pct": spread_pct,
            "open_interest": info.get("open_interest"),
            "mark_iv": info.get("mark_iv"),
            "underlying_price": info.get("underlying_price"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "vega": greeks.get("vega"),
            "theta": greeks.get("theta"),
            "rho": greeks.get("rho"),
        }


def write_snapshot(state, output_path, top_n, min_oi):
    rows = list(state.values())
    if min_oi:
        rows = [r for r in rows if (r.get("open_interest") or 0) >= min_oi]

    rows.sort(key=lambda r: r["spread_pct"] or 0, reverse=True)
    rows = rows[:top_n]

    if not rows:
        print("No data yet, skipping snapshot.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] snapshot: top {len(rows)} by spread% -> {output_path}")
    for r in rows[:5]:
        spread_pct = r["spread_pct"] if r["spread_pct"] is not None else float("nan")
        print(f"  {r['symbol']:<28} bid={r['bid']:.4f} ask={r['ask']:.4f} spread%={spread_pct:.2f}")


async def main():
    args = parse_args()

    # Load the option universe via REST first -- cheap, one-time.
    rest_exchange = ccxt.deribit({"enableRateLimit": True})
    markets = rest_exchange.load_markets()
    now_ms = rest_exchange.milliseconds()

    symbols = []
    for symbol, m in markets.items():
        if not m.get("active") or m.get("type") != "option":
            continue
        if not symbol.startswith(f"{args.currency}/"):
            continue
        if args.max_days_to_expiry is not None:
            dte = days_to_expiry(m, now_ms)
            if dte is None or dte > args.max_days_to_expiry:
                continue
        symbols.append(symbol)

    if not symbols:
        print("No matching option symbols found. Check --currency / --max-days-to-expiry.")
        return

    print(f"Watching {len(symbols)} {args.currency} options over websocket...")

    exchange = ccxtpro.deribit({"enableRateLimit": True})
    state = {}

    watchers = []
    for i, s in enumerate(symbols):
        watchers.append(asyncio.create_task(watch_one(exchange, s, state)))
        # Stagger subscription bursts so we don't slam the rate limit on startup.
        if i % 20 == 19:
            await asyncio.sleep(0.5)

    output_filename = args.output or f"{args.currency}_option_spreads.csv"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)

    try:
        while True:
            await asyncio.sleep(args.interval)
            write_snapshot(state, output_path, args.top, args.min_oi)
    except KeyboardInterrupt:
        pass
    finally:
        for w in watchers:
            w.cancel()
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())