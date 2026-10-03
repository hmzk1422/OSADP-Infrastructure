"""Step 1 CLI: build 00-hour LOB seeds in the matched work tree."""
from __future__ import annotations
import argparse
from typing import Optional
import components.pipelines.ingestion.CryptoL2.datapaths as paths
from components.pipelines.event_merger.matched import lob as _lob
from components.pipelines.event_merger.matched.snapshots import build_day_start_snapshots


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="00-hour snapshots in ingest_parquet.")
    p.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    p.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--lob-levels", type=int, default=20)
    p.add_argument("--tick-size", type=float, default=_lob.LOB_TICK_SIZE)
    a = p.parse_args(argv)
    days = paths.orderbook_parquet_days(a.root, a.symbol)
    start = a.start or days[0]
    end = a.end or days[-1]
    print(build_day_start_snapshots(root=a.root, symbol=a.symbol, start=start,
                                    end=end, lob_levels=a.lob_levels,
                                    tick_size=a.tick_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
