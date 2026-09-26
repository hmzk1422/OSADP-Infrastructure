"""Mock research: a fake "research script" that defines features on a DataFrame.

This stands in for the real research code. It is deliberately written the way a
quant writes research — declare features against columns, aggregate over
trailing windows, reconstruct bars — and then hands the whole thing to the
compiler. Nothing here knows about Rust; that is the point of the parity test.

Run directly to regenerate the fixtures::

    python stresstest/mock_research.py

Outputs (next to this file):
    spec.json   the compiled feature spec consumed by the Rust executor
    rows.csv    the input observations
    py_out.csv  the feature values Python computed
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from live import (  # noqa: E402
    FeaturePipeline,
    bid_ask,
    col,
    evaluate,
    imbalance,
    lit,
    microprice,
    reconstruct,
    window,
    zscore,
)

# Columns the mock "market data" carries. The Rust side receives exactly these.
INPUT_COLUMNS = [
    "event_time_ms",
    "price",
    "quantity",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
]


def synthetic_rows(count: int = 24) -> list[dict]:
    """Deterministic pseudo-market data.

    A fixed linear congruential generator keeps the fixture byte-identical
    between runs, so a parity failure is always a real divergence rather than
    input drift.
    """
    rows = []
    price = 100.0
    seed = 20250630
    for index in range(count):
        seed = (1103515245 * seed + 12345) % 2147483648
        price += ((seed % 2001) - 1000) / 100_000.0
        spread = 0.05 + (index % 3) * 0.01
        bid_qty = round(0.5 + ((seed >> 7) % 400) / 100.0, 4)
        ask_qty = round(0.5 + ((seed >> 11) % 400) / 100.0, 4)
        rows.append(
            {
                "event_time_ms": 1751245200000 + index * 250,
                "price": round(price, 6),
                "quantity": round(0.001 + ((seed >> 3) % 500) / 1000.0, 6),
                "best_bid_price": round(price - spread, 6),
                "best_bid_qty": bid_qty,
                "best_ask_price": round(price + spread, 6),
                "best_ask_qty": ask_qty,
            }
        )
    return rows


def build_pipeline() -> FeaturePipeline:
    """The features under test.

    One of every supported construct, so a divergence anywhere in the op set
    shows up as a specific failing cell rather than a blanket mismatch:

    * arithmetic between columns (``bid_ask``, ``imbalance``)
    * a weighted microprice (mul/add/div)
    * a literal constant
    * a trailing window, mean and last
    * a causal z-score
    * bar reconstruction
    """
    pipeline = FeaturePipeline()
    pipeline.AddFeature("bid_ask", bid_ask())
    pipeline.AddFeature("imbalance", imbalance())
    pipeline.AddFeature("microprice", microprice())
    pipeline.AddFeature("mid_offset", col("price") - (col("best_bid_price") + col("best_ask_price")) / 2.0)
    pipeline.AddFeature("tick_size_scaled", col("quantity") * 100.0)
    pipeline.AddFeature("one", lit(1.0))
    pipeline.AddFeature("size_ma", window(col("price"), 5, "mean"))
    pipeline.AddFeature("size_last", window(col("price"), 5, "last"))
    pipeline.AddFeature("size_max", window(bid_ask(), 3, "max"))
    pipeline.AddFeature("qty_sum", window(col("quantity"), 4, "sum"))
    pipeline.AddFeature("price_z", zscore(col("price"), 5))
    pipeline.AddFeature("bar_last", reconstruct(col("price"), 4, "last"))
    return pipeline


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row"] + INPUT_COLUMNS)
        for index, row in enumerate(rows):
            writer.writerow([index] + [repr(row[column]) for column in INPUT_COLUMNS])


def write_py_out_csv(path: Path, results: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "feature", "value"])
        for index, features in enumerate(results):
            for name, value in features.items():
                writer.writerow([index, name, repr(float(value))])


def main() -> int:
    rows = synthetic_rows()
    pipeline = build_pipeline()
    spec = pipeline.compile()
    results = evaluate(spec, rows)

    (HERE / "spec.json").write_text(spec, encoding="utf-8")
    write_rows_csv(HERE / "rows.csv", rows)
    write_py_out_csv(HERE / "py_out.csv", results)

    print(f"spec:     {len(pipeline.feature_names)} features")
    print(f"rows:     {len(rows)}")
    print(f"features: {', '.join(pipeline.feature_names)}")
    print()
    header = "row  " + "  ".join(f"{n[:11]:>11}" for n in pipeline.feature_names)
    print(header)
    for index, features in enumerate(results):
        cells = "  ".join(
            f"{(f'{v:.6f}' if math.isfinite(v) else 'nan'):>11}"
            for v in features.values()
        )
        print(f"{index:<4} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
