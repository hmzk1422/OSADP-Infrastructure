from __future__ import annotations

import csv
import os
from pathlib import Path

from features import ExampleFeatures
from strategy import ExampleStrategy


def main() -> None:
    input_path = Path(os.environ["PARITY_DATA_PATH"])
    output_path = Path(os.environ["PARITY_PYTHON_OUT"])
    symbol = os.environ.get("PARITY_SYMBOL", "example")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = ExampleFeatures()
    strategy = ExampleStrategy()
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["timestamp_ms", "price"]:
            raise ValueError("input CSV must have timestamp_ms,price columns")
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(["timestamp_ms", "dz", "decision", "position"])
            for row in reader:
                timestamp_ms = int(row["timestamp_ms"])
                price = float(row["price"])
                if not (price == price and abs(price) != float("inf")):
                    raise ValueError(f"non-finite price at {timestamp_ms}")
                dz = features.compute(price)
                decision, position = strategy.decide(dz)
                writer.writerow([timestamp_ms, dz, decision, position])
    print(f"Python example strategy processed {input_path} for {symbol}")


if __name__ == "__main__":
    main()
