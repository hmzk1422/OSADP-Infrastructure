"""Step 2 CLI: materialize all canonical source datasets into the work tree."""
from __future__ import annotations
import argparse
from typing import Optional
import components.pipelines.ingestion.CryptoL2.datapaths as paths
from components.pipelines.event_merger.matched.parquet import fetch_source_data


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Materialize all three source datasets into matched work.")
    p.add_argument("--root", default=str(paths.MARKET_DATA_ROOT))
    p.add_argument("--symbol", default=paths.DEFAULT_SYMBOL)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    a = p.parse_args(argv)
    out = fetch_source_data(root=a.root, symbol=a.symbol, start=a.start, end=a.end)
    print(f"materialized {len(out)} source file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
