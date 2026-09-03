"""
live_sim_runner.py

Live paper-trading loop: watches your option position(s) + hedge candidates
(underlying + other options) over websocket, reads Greeks straight off
Deribit's ticker feed, calls hedge_solver whenever delta/gamma drift out of
band, and SIMULATES the resulting fills.

SIMPLIFICATION IN EFFECT (explicit, per your instruction):
Any hedge trade the solver decides on is assumed to fill instantly and
completely at the current bid/ask -- i.e. "if price hits our limit order
it's a guaranteed fill." No partial fills, no missed fills, no adverse
selection, no latency between decision and execution. This will make P&L
look better than reality, especially for the option leg where you're
providing liquidity in the first place (their whole edge is that YOUR
fills aren't guaranteed against real flow). Fine for shaking out the
mechanics; revisit before trusting the P&L number for anything real.

This still does not touch a real exchange order -- it maintains an
in-memory paper book only. Say the word when you want real execution
wired in.

Usage:
    python live_sim_runner.py \\
        --mm-position BTC-27JUN25-70000-C -1 \\
        --underlying BTC/USD:BTC \\
        --hedge-option BTC-27JUN25-72000-C \\
        --delta-band 0.02 --gamma-band 0.000005
"""

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import ccxt.pro as ccxtpro

from hedge_solver import portfolio_greeks, hedge_book_greeks, build_candidates, solve_hedge


def parse_args():
    p = argparse.ArgumentParser(description="Live paper-trading hedge simulation over websocket")
    p.add_argument("--mm-position", nargs=2, action="append", metavar=("SYMBOL", "QTY"),
                    required=True, help="Option position being market-made. Repeatable. QTY signed.")
    p.add_argument("--underlying", required=True,
                    help="CCXT unified symbol for the underlying perp/spot, e.g. BTC/USD:BTC")
    p.add_argument("--hedge-option", action="append", default=[],
                    help="Other option symbols available as hedge candidates. Repeatable.")
    p.add_argument("--delta-band", type=float, default=0.02)
    p.add_argument("--gamma-band", type=float, default=None)
    p.add_argument("--vega-band", type=float, default=None)
    p.add_argument("--check-every-seconds", type=float, default=5.0)
    p.add_argument("--log", default=None,
                    help="CSV log of fills + running P&L (default: sim_log.csv, alongside this script)")
    return p.parse_args()


async def watch_symbol(exchange, symbol, state):
    while True:
        try:
            ticker = await exchange.watch_ticker(symbol)
        except Exception as e:
            print(f"[{symbol}] watch error: {e}")
            await asyncio.sleep(3)
            continue

        info = ticker.get("info", {}) or {}
        greeks = info.get("greeks", {}) or {}
        state[symbol] = {
            "timestamp": datetime.now(timezone.utc),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "vega": greeks.get("vega"),
        }


def mid(state, symbol):
    s = state.get(symbol)
    if s is None or s.get("bid") is None or s.get("ask") is None:
        return None
    return (s["bid"] + s["ask"]) / 2.0


def mark_to_market(book: dict, state: dict) -> float:
    """book: {symbol: qty}. Returns total value using current mid prices; None-mid legs are skipped (not yet warmed up)."""
    total = 0.0
    for symbol, qty in book.items():
        m = mid(state, symbol)
        if m is not None:
            total += qty * m
    return total


def log_row(log_path, row: dict, fieldnames: list):
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


async def main():
    args = parse_args()
    mm_positions = {sym: float(qty) for sym, qty in args.mm_position}

    # Paper hedge book: what we've bought/sold to hedge, separate from the MM'd option(s).
    hedge_book = {}
    cumulative_hedge_cost = 0.0

    watch_symbols = list(mm_positions.keys()) + [args.underlying] + list(args.hedge_option)
    exchange = ccxtpro.deribit({"enableRateLimit": True})
    state = {}

    tasks = [asyncio.create_task(watch_symbol(exchange, s, state)) for s in watch_symbols]

    log_path = args.log or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_log.csv")
    fieldnames = ["timestamp", "event", "delta", "gamma", "vega", "trades",
                  "hedge_cost_this_event", "cumulative_hedge_cost", "book_value"]

    print(f"Watching {len(watch_symbols)} symbols (simulated fills, no real orders)...")

    try:
        while True:
            await asyncio.sleep(args.check_every_seconds)

            summary = portfolio_greeks(mm_positions, state)
            if summary is None:
                print("Still warming up (missing Greeks for MM position)...")
                continue

            # Fold in the hedge book's own Greeks. Uses hedge_book_greeks (not portfolio_greeks)
            # because the hedge book can contain the underlying, which never carries exchange
            # Greeks in the ticker -- portfolio_greeks would incorrectly treat that as "not warmed
            # up yet" and stall forever after the first underlying hedge trade.
            hedge_summary = (hedge_book_greeks(hedge_book, args.underlying, state)
                              if hedge_book else {"delta": 0.0, "gamma": 0.0, "vega": 0.0})
            if hedge_summary is None:
                print("Hedge book data not fully warmed up, skipping this round.")
                continue

            total_delta = summary["delta"] + hedge_summary["delta"]
            total_gamma = summary["gamma"] + hedge_summary["gamma"]
            total_vega = summary["vega"] + hedge_summary["vega"]

            in_band = abs(total_delta) <= args.delta_band
            if args.gamma_band is not None:
                in_band = in_band and abs(total_gamma) <= args.gamma_band
            if args.vega_band is not None:
                in_band = in_band and abs(total_vega) <= args.vega_band

            print(f"[{datetime.now().strftime('%H:%M:%S')}] delta={total_delta:.4f} "
                  f"gamma={total_gamma:.7f} vega={total_vega:.4f} in_band={in_band}")

            if in_band:
                continue

            candidates = build_candidates(args.underlying, args.hedge_option, state)
            if not candidates:
                print("  No warmed-up hedge candidates yet, skipping.")
                continue

            result = solve_hedge(
                portfolio_delta=total_delta, portfolio_gamma=total_gamma,
                candidates=candidates, delta_band=args.delta_band,
                gamma_band=args.gamma_band, portfolio_vega=total_vega, vega_band=args.vega_band,
            )

            book_value = mark_to_market(mm_positions, state) + mark_to_market(hedge_book, state)

            if not result.feasible:
                print(f"  INFEASIBLE: {result.message}")
                log_row(log_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(), "event": "infeasible",
                    "delta": total_delta, "gamma": total_gamma, "vega": total_vega,
                    "trades": "", "hedge_cost_this_event": 0.0,
                    "cumulative_hedge_cost": cumulative_hedge_cost, "book_value": book_value,
                }, fieldnames)
                continue

            if not result.trades:
                continue

            # SIMULATED FILL: apply every trade instantly, in full, at the price the solver costed it at.
            for sym, qty in result.trades.items():
                hedge_book[sym] = hedge_book.get(sym, 0.0) + qty
                side = "BUY" if qty > 0 else "SELL"
                print(f"  [SIMULATED FILL] {side} {abs(qty):.4f} {sym}")

            cumulative_hedge_cost += result.total_cost
            book_value = mark_to_market(mm_positions, state) + mark_to_market(hedge_book, state)

            print(f"  cost this event: {result.total_cost:.4f} | cumulative hedge cost: {cumulative_hedge_cost:.4f} "
                  f"| book value: {book_value:.4f}")

            log_row(log_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(), "event": "hedge",
                "delta": total_delta, "gamma": total_gamma, "vega": total_vega,
                "trades": str(result.trades), "hedge_cost_this_event": result.total_cost,
                "cumulative_hedge_cost": cumulative_hedge_cost, "book_value": book_value,
            }, fieldnames)

    except KeyboardInterrupt:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())