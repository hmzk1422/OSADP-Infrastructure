#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latency_diag.py — end-to-end feed-latency diagnostic for the Rust executor
==========================================================================

Drives ONE strategy (Bitfinex/Bybit, paper mode) through the exact path a real
strategy uses (unified_runner.Strategy -> engine (PyO3) -> run_execution_loop())
and samples `strat.status()` every `interval` seconds, measuring, at sub-millisecond
resolution:

    feed_staleness_ms      AGE of the newest market event (time since it was
                           committed to the engine). This is what Python actually
                           sees as "fresh". Grows on a silent feed; resets only when
                           a new event lands.
    last_event_latency_ms  MONOTONIC local code latency: WS dequeue -> fully
                           processed (parse + strategy + db). Clock-independent;
                           sub-ms on a healthy box. NOT the venue network RTT.
    clock_skew_ms          wall-clock delta (local - exchange) on the newest event.
                           ~0 when the box is NTP-synced to the venue; a large value
                           (e.g. -64s in this sandbox) is a CLOCK/NTP problem, not a
                           code problem. Surfaced separately on purpose.
    loop_liveness_ms       time since the loop last woke (incl. the 2s WS-read
                           timeout). Pure liveness/stall detector, independent of
                           feed freshness.
    loop_cadence_ms        full Rust loop iteration time (feed wait + work).
    batch_process_ms       pure per-batch processing time (db write + strategy).
    status_rt_ms           Python -> Rust status() round-trip.
    event_rate_s           events consumed per second between consecutive polls.
    exchange_dt_ms         delta between consecutive exchange report timestamps
                           (venue delivery cadence as observed).

What a quant dev should test (the "should-have-been" reference)
---------------------------------------------------------------
  feed_staleness       on an ACTIVE feed, single-digit ms (age of the newest tick)
                       between updates. On a QUIET feed it grows until the next
                       trade/book-top — that is correct and means "data is old",
                       NOT a stall. A sawtooth that resets every ~2s is the
                       feed-read timeout waking the loop (see loop_liveness_ms).
  clock_skew_ms        should be a few ms; |skew| > 100ms => the machine clock is
                       out of sync with the venue (run NTP). A -64s value is PURE
                       clock skew, surfaced here so it is never mistaken for
                       ingest latency.
  last_event_latency    sub-ms to a few ms. Persistent >5ms => a blocking call or
                       CPU-bound handler on the hot path.
  batch_process        sub-ms; monotonic growth => CPU-bound or blocking db write.
  status_rt_ms          typically 0.01-0.1ms; if it dominates, status() serializes
                       too much state.
  loop_cadence         <= the WS-read timeout (2s); a spike with low batch_process
                       => the WS reader blocked or lock contention.

Usage
-----
    python latency_diag.py                         # 30s btcusdt @ bitfinex
    python latency_diag.py --ticker ethusdt --duration 60 --interval 0.05
    python latency_diag.py --exchange bybit --duration 30
    python latency_diag.py --no-persist            # delete temp RocksDB

Per-sample CSV + JSON artifacts are OPT-IN (`--save`) and land under build/ —
by default only the console report is printed. The same staleness / latency /
skew numbers are also persisted per tick in the strategy's RocksDB diag
column family, and are live-visible in the portfolio dashboard via the
children's periodic `[net]` console line.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from unified_runner import DEFAULT_TICKERS, SUPPORTED_EXCHANGES, Strategy

REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "build"

logger = logging.getLogger("latency_diag")


def pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[idx]


def summary(values: List[float]) -> Dict[str, float]:
    return {
        "min": min(values) if values else float("nan"),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "max": max(values) if values else float("nan"),
    }


def fmt_pct(d: Dict[str, float]) -> str:
    if d["p50"] != d["p50"]:  # NaN
        return "   (no data)"
    return (
        f"min {d['min']:8.2f} | p50 {d['p50']:8.2f} | p95 {d['p95']:8.2f} | "
        f"p99 {d['p99']:8.2f} | max {d['max']:8.2f}"
    )


def collect_samples(
    strat: Strategy,
    duration: float,
    interval: float,
    warmup_s: float,
) -> List[Dict[str, Any]]:
    """Poll `strat.status()` every `interval` seconds for `duration` seconds.

    Each row captures the Python-visible snapshot plus derived rates. Rows taken
    during the warm-up window are marked `in_warmup=True` and excluded from the
    summary (connection + seed phase).
    """
    samples: List[Dict[str, Any]] = []
    start = time.monotonic()
    warmup_until = start + warmup_s
    deadline = start + duration
    prev_wall: Optional[float] = None
    prev_events: Optional[int] = None
    prev_ev_ts: Optional[int] = None

    while time.monotonic() < deadline:
        t_before = time.monotonic()
        raw = strat.status()
        rt_ms = (time.monotonic() - t_before) * 1000.0
        if isinstance(raw, str):
            st: Dict[str, Any] = json.loads(raw) if raw else {}
        elif isinstance(raw, dict):
            st = raw
        else:
            st = {}
        t_now = time.monotonic()

        row: Dict[str, Any] = {
            "t_mono_s": round(t_now - start, 6),
            "wall_iso": datetime.now().isoformat(timespec="milliseconds"),
            "in_warmup": t_now < warmup_until,
            "mid": st.get("mid") or 0.0,
            "events_seen": st.get("events_seen") or 0,
            "feed_staleness_ms": st.get("feed_staleness_ms"),
            "last_event_latency_ms": st.get("last_event_latency_ms"),
            "clock_skew_ms": st.get("clock_skew_ms"),
            "loop_liveness_ms": st.get("loop_liveness_ms"),
            "loop_cadence_ms": st.get("loop_cadence_ms"),
            "batch_process_ms": st.get("batch_process_ms"),
            "last_event_ts_ms": st.get("last_event_ts_ms"),
            "status_rt_ms": round(rt_ms, 3),
            "event_rate_s": None,
            "exchange_dt_ms": None,
        }
        if prev_wall is not None and prev_events is not None:
            wall_dt = t_now - prev_wall
            if wall_dt > 0:
                seen = int(st.get("events_seen") or prev_events)
                row["event_rate_s"] = round((seen - prev_events) / wall_dt, 1)
        if prev_ev_ts is not None and row["last_event_ts_ms"] is not None:
            row["exchange_dt_ms"] = row["last_event_ts_ms"] - prev_ev_ts
        samples.append(row)

        prev_wall = t_now
        prev_events = row["events_seen"]
        prev_ev_ts = row["last_event_ts_ms"]

        if time.monotonic() < deadline:
            time.sleep(interval)
    return samples


def _field(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [
        float(r[key]) for r in rows
        if not r["in_warmup"]
        and isinstance(r.get(key), (int, float))
        and r[key] is not None
    ]


def build_report(
    samples: List[Dict[str, Any]], ticker: str, exchange: str,
) -> Dict[str, Any]:
    stal = _field(samples, "feed_staleness_ms")
    lat = _field(samples, "last_event_latency_ms")
    skew = _field(samples, "clock_skew_ms")
    live = _field(samples, "loop_liveness_ms")
    cad = _field(samples, "loop_cadence_ms")
    batch = _field(samples, "batch_process_ms")
    rt = _field(samples, "status_rt_ms")
    seen = _field(samples, "events_seen") or [0.0]
    total_events = int(max(seen))

    rates = [r for r in _field(samples, "event_rate_s") if r > 0]
    ex_dt = [
        r["exchange_dt_ms"] for r in samples
        if not r["in_warmup"]
        and isinstance(r.get("exchange_dt_ms"), (int, float))
        and r["exchange_dt_ms"] is not None
        and r["exchange_dt_ms"] > 0
    ]

    report: Dict[str, Any] = {
        "symbol": ticker,
        "exchange": exchange,
        "samples": len(samples),
        "duration_s": round(samples[-1]["t_mono_s"], 3) if samples else 0.0,
        "events_seen": total_events,
        "event_rate_avg_s": round(sum(rates) / len(rates), 1) if rates else None,
        "feed_staleness_ms": summary(stal),
        "exchange_latency_ms": summary(lat),
        "clock_skew_ms": summary(skew),
        "loop_liveness_ms": summary(live),
        "loop_cadence_ms": summary(cad),
        "batch_process_ms": summary(batch),
        "status_rt_ms": summary(rt),
        "exchange_delivery_dt_ms": {"observed": summary(ex_dt) if ex_dt else None},
    }
    report["verdict"] = _verdict(report)
    return report


def _verdict(r: Dict[str, Any]) -> str:
    if r["samples"] == 0:
        return "FAIL - no samples collected (native module not built?)"
    if r["events_seen"] <= 0:
        return "WARNING - no market events received (feed down / no data?)"
    stal = r["feed_staleness_ms"]
    lat = r["exchange_latency_ms"]
    skew = r["clock_skew_ms"]
    if stal["p50"] > 100:
        return "WARNING - Python sees the feed >100ms stale; Rust loop stalled"
    skew_is_nan = skew["p50"] != skew["p50"]
    if not skew_is_nan and abs(skew["p50"]) > 100:
        return f"WARNING - machine clock skewed vs venue ({skew['p50']:.0f}ms); NTP sync required"
    if lat["p50"] > 5:
        return "WARNING - exchange->local code latency unusually high (>5ms median)"
    sk_str = f", skew {skew['p50']:.0f}ms" if not skew_is_nan else ""
    return (
        "HEALTHY - feed fresh (p50 staleness "
        f"{stal['p50']:.1f}ms), code latency p50 {lat['p50']:.2f}ms{sk_str}, "
        f"batch p50 {r['batch_process_ms']['p50']:.2f}ms"
    )


def print_report(r: Dict[str, Any]) -> None:
    stal = r["feed_staleness_ms"]
    lat = r["exchange_latency_ms"]
    skew = r["clock_skew_ms"]
    live = r["loop_liveness_ms"]
    print("\n" + "=" * 66)
    print("latency_diag - end-to-end feed latency (Rust executor)")
    print("=" * 66)
    print(f"  symbol  : {r['symbol']} @ {r['exchange']}")
    print(f"  window  : {r['samples']} samples over {r['duration_s']:.1f}s")
    if r.get("event_rate_avg_s") is not None:
        print(f"  events  : {r['events_seen']:,} ingested "
              f"({r['event_rate_avg_s']:.1f} events/s)")
    else:
        print(f"  events  : {r['events_seen']:,}")
    print()
    print("  feed_staleness_ms          " + fmt_pct(stal))
    print("  last_event_latency_ms      " + fmt_pct(lat))
    print("  clock_skew_ms              " + fmt_pct(skew))
    print("  loop_liveness_ms           " + fmt_pct(live))
    print("  loop_cadence_ms            " + fmt_pct(r["loop_cadence_ms"]))
    print("  batch_process_ms           " + fmt_pct(r["batch_process_ms"]))
    print("  status()_rt_ms             " + fmt_pct(r["status_rt_ms"]))
    ex = r["exchange_delivery_dt_ms"]["observed"]
    if ex:
        print("  exchange_delivery_dt_ms    " + fmt_pct(ex))
    print()
    print("  verdict : " + r["verdict"])
    print("=" * 66)


def write_artifacts(
    samples: List[Dict[str, Any]], report: Dict[str, Any], ticker: str,
) -> List[Path]:
    """Persist per-sample CSV + a machine-readable JSON report in build/.

    OPT-IN ONLY (``--save``): the console report above is the default output —
    writing one CSV+JSON pair per run into build/ cluttered the folder, and
    everything these files carry is also persisted per-tick in the strategy's
    RocksDB ``diag`` column family (see uets_rs/live-trade/src/db.rs).
    """
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = BUILD_DIR / f"latency_diag_{ticker}_{stamp}"

    cols = [
        "t_mono_s", "wall_iso", "in_warmup", "mid", "events_seen",
        "feed_staleness_ms", "last_event_latency_ms", "clock_skew_ms",
        "loop_liveness_ms", "loop_cadence_ms", "batch_process_ms",
        "last_event_ts_ms", "status_rt_ms", "event_rate_s", "exchange_dt_ms",
    ]
    csv_path = base.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in samples:
            w.writerow([s.get(c) for c in cols])

    json_path = base.with_suffix(".json")
    with json_path.open("w") as fh:
        json.dump(report, fh, indent=2)

    print()
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")
    return [csv_path, json_path]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="latency_diag",
        description="measure end-to-end feed latency through the Rust execution loop",
    )
    ap.add_argument("--ticker", default=DEFAULT_TICKERS[0])
    ap.add_argument("--exchange", default="bitfinex", choices=SUPPORTED_EXCHANGES)
    ap.add_argument("--duration", type=float, default=30.0,
                    help="sampling window in seconds (default 30)")
    ap.add_argument("--interval", type=float, default=0.05,
                    help="status() sampling interval in seconds (default 0.05 -> 50Hz)")
    ap.add_argument("--warmup", type=float, default=3.0,
                    help="seconds of warm-up samples to exclude from stats (default 3)")
    ap.add_argument("--capital", type=float, default=10_000.0,
                    help="paper capital (default 10000)")
    ap.add_argument("--db", default=None,
                    help="RocksDB path (default: data/latency_diag_<ticker>.rocksdb)")
    ap.add_argument("--no-persist", action="store_true",
                    help="delete the temp RocksDB store after the run")
    ap.add_argument("--save", action="store_true",
                    help="also write latency_diag_<ticker>_<ts>.csv/.json into build/ "
                         "(default OFF: console report only — build/ stays clean)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    ticker = args.ticker.lower()
    db_path = args.db or str(REPO_ROOT / "data" / f"latency_diag_{ticker}.rocksdb")

    strat: Optional[Strategy] = None
    try:
        strat = Strategy(ticker, exchange=args.exchange, mode="paper",
                         capital=args.capital, db_path=db_path)
        strat.run()

        deadline = time.monotonic() + 8.0
        while not strat.is_running and time.monotonic() < deadline:
            time.sleep(0.05)
        if not strat.is_running:
            print("ERROR: strategy did not start within 8s", file=sys.stderr)
            print("  (is the native engine built? see latency_diag.py docstring)",
                  file=sys.stderr)
            return 1

        print(f"sampling {ticker} @ {args.exchange} - {args.duration:.0f}s, "
              f"status() every {args.interval * 1000:.0f}ms "
              f"(warmup {args.warmup:.0f}s)")
        print("  Ctrl-C aborts gracefully; the Rust store is flushed + closed.\n")

        samples = collect_samples(
            strat, args.duration, args.interval, args.warmup
        )
        report = build_report(samples, ticker, args.exchange)
        print_report(report)
        if args.save:
            write_artifacts(samples, report, ticker)
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted - shutting down cleanly...")
        return 130
    finally:
        if strat is not None:
            logger.info("stopping strategy (joins Rust thread, flushes + closes RocksDB)")
            strat.stop()
        if args.no_persist:
            shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())