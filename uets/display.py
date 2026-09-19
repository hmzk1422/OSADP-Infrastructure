"""Self-updating orderbook + tape view for the Rust live monitor.

Terminal 1:  cd uets
             cargo run --bin dev_test -- monitor btcusdt
Terminal 2:  python uets/display.py btcusdt

Flags:
    --once     render one frame and exit (no live loop)
    --file     explicit frame path (default: uets/out/live/<symbol>.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

FRAME_DIR = Path(__file__).resolve().parent / "out" / "live"

CLEAR = "\033[2J\033[H"
GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def fmt_px(p):
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:.4f}"
    return f"{p:.6f}"


def fmt_qty(q):
    if q >= 100:
        return f"{q:,.2f}"
    if q >= 1:
        return f"{q:.4f}"
    return f"{q:.6f}"


def fmt_ms(ms):
    return time.strftime("%H:%M:%S", time.localtime(ms / 1000)) + f".{ms % 1000:03d}"


def signed(val, pct):
    sign = "+" if val >= 0 else ""
    color = GREEN if val >= 0 else RED
    return f"{color}{sign}{val:,.2f} ({sign}{pct:.4f}%){RESET}"


def render(f):
    tr = f.get("trades", {})
    last = tr.get("last_price") or 0.0
    first = tr.get("first_price") or 0.0
    prev = tr.get("prev_price") or last
    trade_delta = last - prev
    trade_pct = (trade_delta / prev * 100.0) if prev else 0.0
    session_delta = last - first
    session_pct = (session_delta / first * 100.0) if first else 0.0

    lines = [
        f"{BOLD}{f.get('symbol', '?')}{RESET}   {DIM}frame {fmt_ms(f.get('ts', 0))}{RESET}",
        f"last {fmt_px(last)}   last-trade {signed(trade_delta, trade_pct)}   "
        f"session {signed(session_delta, session_pct)}",
        f"buy vol {tr.get('buy_vol', 0):.6f}   sell vol {tr.get('sell_vol', 0):.6f}   "
        f"trades {tr.get('trade_count', 0)}   last trade {fmt_ms(tr.get('last_trade_ms', 0))}   "
        f"book age {(f.get('ts', 0) - f.get('book_ms', 0)) / 1000:.1f}s",
        "",
    ]

    recent = tr.get("recent", [])
    lines.append(f"{DIM}tape (last {len(recent)}){RESET}")
    for t in reversed(recent):
        color = GREEN if t["side"] == "buy" else RED
        lines.append(
            f"  {fmt_ms(t['t'])}  {fmt_px(t['p']):>13}  {fmt_qty(t['q']):>12}  {color}{t['side']:>4}{RESET}"
        )
    lines.append("")

    bids, asks = f.get("bids", []), f.get("asks", [])
    lines.append(f"orderbook ({len(bids)} bid / {len(asks)} ask levels, snapshot {f.get('book_update_id', 0)})")
    lines.append(f"  {'bid qty':>12}  {'bid px':>13}  |  {'ask px':>13}  {'ask qty':>12}")
    for i in range(max(len(bids), len(asks))):
        b = bids[i] if i < len(bids) else [None, None]
        a = asks[i] if i < len(asks) else [None, None]
        bs = f"{fmt_qty(b[1]):>12} {fmt_px(b[0]):>13}" if b[0] is not None else f"{'--':>12} {'--':>13}"
        as_ = f"{fmt_px(a[0]):>13} {fmt_qty(a[1]):>12}" if a[0] is not None else f"{'--':>13} {'--':>12}"
        lines.append(f"  {bs}  |  {as_}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="live orderbook + tape view (uets monitor)")
    ap.add_argument("symbol", nargs="?", default="btcusdt")
    ap.add_argument("--once", action="store_true", help="render one frame and exit")
    ap.add_argument("--file", type=Path, help="frame file (default: uets/out/live/<symbol>.json)")
    args = ap.parse_args()
    path = args.file or FRAME_DIR / f"{args.symbol.lower()}.json"

    while True:
        try:
            frame = json.loads(path.read_text())
        except (OSError, ValueError):
            sys.stdout.write(CLEAR)
            print(f"waiting for frames at {path}")
            print("start the monitor in another terminal:")
            print(f"  cd uets && cargo run --bin dev_test -- monitor {args.symbol}")
            sys.stdout.flush()
            if args.once:
                return
            time.sleep(0.5)
            continue

        sys.stdout.write(CLEAR + render(frame) + "\n")
        sys.stdout.flush()
        if args.once:
            return
        time.sleep(0.1)


if __name__ == "__main__":
    main()