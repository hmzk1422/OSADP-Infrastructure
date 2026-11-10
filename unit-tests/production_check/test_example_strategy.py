from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = ROOT / "strategy_system" / "strategy" / "example_strategy" / "parity.py"
FIXTURE = ROOT / "unit-tests" / "production_check" / "fixtures" / "example_strategy.csv"


class ExampleStrategyParityTests(unittest.TestCase):
    def test_python_strategy_emits_expected_schema_and_bounded_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "python.csv"
            environment = os.environ.copy()
            environment.update(
                PARITY_DATA_PATH=str(FIXTURE),
                PARITY_PYTHON_OUT=str(output),
                PARITY_SYMBOL="EXAMPLE",
            )
            subprocess.run(
                [sys.executable, str(PARITY_SCRIPT)],
                cwd=ROOT,
                env=environment,
                check=True,
            )

            with output.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(
                    reader.fieldnames,
                    ["timestamp_ms", "dz", "decision", "position"],
                )
                rows = list(reader)
            self.assertEqual(len(rows), 20)
            self.assertTrue(any(row["decision"] == "1" for row in rows))
            self.assertTrue(all(row["position"] in {"-1", "0", "1"} for row in rows))


if __name__ == "__main__":
    unittest.main()
