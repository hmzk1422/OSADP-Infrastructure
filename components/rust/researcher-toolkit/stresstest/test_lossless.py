"""Exact Python-to-Rust parity for the feature pipeline.

The old tests compared Python against *itself*, which cannot detect a divergence
between the two implementations. These tests compile the spec once, run the
**real executor sources** in Rust, and compare every output cell for exact
`float` equality.

Every check is a full (all rows x all features) comparison. A tolerance would
hide exactly the class of bug this exists to catch.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from live import (  # noqa: E402
    FeaturePipeline,
    col,
    evaluate,
    lit,
    reconstruct,
    reconstruct_rows,
    window,
    zscore,
)
from mock_research import build_pipeline, synthetic_rows, write_rows_csv  # noqa: E402

CRATE_DIR = HERE
SPEC_PATH = HERE / "spec.json"
ROWS_PATH = HERE / "rows.csv"
RUST_OUT_PATH = HERE / "rust_out.csv"


def _cargo(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cargo", *args], cwd=CRATE_DIR, capture_output=True, text=True
    )


def read_cells(path: Path) -> dict:
    """Read a `row,feature,value` CSV into a ``(row, feature) -> value`` map."""
    cells = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            cells[(int(record["row"]), record["feature"])] = float(record["value"])
    return cells


def run_rust(spec_path: Path, rows_path: Path, out_path: Path) -> None:
    """Run the Rust executor over the given inputs and write its output CSV."""
    result = _cargo("run", "--quiet", "--", str(spec_path), str(rows_path), str(out_path))
    if result.returncode != 0:
        raise RuntimeError(
            "rust parity runner failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


class RustParityTest(unittest.TestCase):
    """Python and Rust must agree exactly, cell for cell."""

    @classmethod
    def setUpClass(cls) -> None:
        build = _cargo("build", "--quiet")
        if build.returncode != 0:
            raise RuntimeError(f"cargo build failed:\n{build.stdout}\n{build.stderr}")

        cls.rows = synthetic_rows()
        cls.pipeline = build_pipeline()
        cls.spec = cls.pipeline.compile()

        # Hand Rust exactly the inputs Python used.
        write_rows_csv(ROWS_PATH, cls.rows)
        SPEC_PATH.write_text(cls.spec, encoding="utf-8")
        run_rust(SPEC_PATH, ROWS_PATH, RUST_OUT_PATH)

        cls.python_results = evaluate(cls.spec, cls.rows)
        cls.rust_cells = read_cells(RUST_OUT_PATH)

    def _python_cells(self) -> dict:
        cells = {}
        for index, features in enumerate(self.python_results):
            for name, value in features.items():
                cells[(index, name)] = float(value)
        return cells

    def test_every_cell_matches_exactly(self) -> None:
        """The core contract: identical values, not merely close ones."""
        python_cells = self._python_cells()

        self.assertEqual(
            set(python_cells),
            set(self.rust_cells),
            "python and rust disagree on WHICH cells exist "
            "(a feature was emitted on one side only)",
        )
        self.assertTrue(python_cells, "no cells were produced; the test is vacuous")

        mismatches = []
        for key, expected in sorted(python_cells.items()):
            actual = self.rust_cells[key]
            if math.isnan(expected) and math.isnan(actual):
                continue
            # Exact equality, including the bit pattern: no tolerance.
            if not (expected == actual):
                mismatches.append(
                    f"row {key[0]:>2} {key[1]:<16} py={expected!r} rust={actual!r}"
                )

        self.assertEqual(
            [],
            mismatches,
            f"{len(mismatches)} of {len(python_cells)} cells diverged:\n"
            + "\n".join(mismatches[:20]),
        )

    def test_parity_covers_every_declared_feature(self) -> None:
        """Guards against a feature silently going missing on one side."""
        for name in self.pipeline.feature_names:
            with self.subTest(feature=name):
                self.assertTrue(
                    any(key[1] == name for key in self.rust_cells),
                    f"rust never emitted feature {name!r}",
                )

    def test_nan_cells_agree(self) -> None:
        """Undefined must be undefined in both, not one side's zero.

        `price_z` has no spread on the first row, and `bar_last` has no
        completed group until row 3.
        """
        for index, name in ((0, "price_z"), (0, "bar_last"), (1, "bar_last"), (2, "bar_last")):
            with self.subTest(row=index, feature=name):
                self.assertTrue(math.isnan(self.rust_cells[(index, name)]))

    def test_window_is_causal(self) -> None:
        """A feature must not change when future rows are appended.

        This is the regression test for the original bug: the old `window()`
        read outside the current history, so a value depended on later rows.
        """
        prefix = evaluate(self.spec, self.rows[:10])[9]
        full = evaluate(self.spec, self.rows)[9]
        for name, value in prefix.items():
            with self.subTest(feature=name):
                if math.isnan(value) and math.isnan(full[name]):
                    continue
                self.assertEqual(value, full[name])

    def test_window_uses_current_row_plus_previous(self) -> None:
        """Pin the exact trailing-window definition.

        A `mean` over the last 3 prices at row i must equal the mean of
        prices[i-2:i+1] â€” current row included, oldest dropped.
        """
        spec = FeaturePipeline()
        spec.AddFeature("ma3", window(col("price"), 3, "mean"))
        results = evaluate(spec.compile(), self.rows)
        prices = [row["price"] for row in self.rows]
        for index in range(3, 10):
            expected = sum(prices[index - 2: index + 1]) / 3.0
            self.assertEqual(results[index]["ma3"], expected)

    def test_reconstruct_drops_incomplete_trailing_bar(self) -> None:
        """24 rows at size 4 gives 6 bars; the partial remainder is dropped."""
        bars = reconstruct_rows(self.rows, 4, "last")
        self.assertEqual(len(bars), len(self.rows) // 4)
        spec = FeaturePipeline()
        spec.AddFeature("bar", reconstruct(col("price"), 4, "last"))
        results = evaluate(spec.compile(), self.rows)
        emitted = [i for i, r in enumerate(results) if not math.isnan(r["bar"])]
        self.assertEqual(emitted, [3, 7, 11, 15, 19, 23])

    def test_div_by_zero_raises(self) -> None:
        """An exact zero denominator is a hard error, not `inf`."""
        spec = FeaturePipeline()
        spec.AddFeature("boom", col("price") / col("quantity"))
        rows = [{"price": 1.0, "quantity": 0.0}]
        with self.assertRaises(ZeroDivisionError):
            evaluate(spec.compile(), rows)

    def test_reconstruct_matches_rust_reconstructor(self) -> None:
        """`strategy::Reconstructor` must agree with the Python reconstruction.

        The second cross-language contract: same bars, same close timestamps,
        and the same rows dropped.
        """
        result = _cargo(
            "run", "--quiet", "--", "--reconstruct", str(ROWS_PATH), "4", "mean"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rust_bars = json.loads(result.stdout.strip().splitlines()[-1])
        python_bars = reconstruct_rows(self.rows, 4, "mean")

        self.assertTrue(rust_bars, "rust produced no bars; the check is vacuous")
        self.assertEqual(len(rust_bars), len(python_bars))
        for got, want in zip(rust_bars, python_bars):
            self.assertEqual(got["timestamp_ms"], want["event_time_ms"])
            self.assertEqual(got["value"], want["price"])
            self.assertEqual(got["row_count"], 4)


class NegativeControlTest(unittest.TestCase):
    """Prove the parity check actually detects divergence.

    A test that only ever passes proves nothing. These cases perturb the
    pipeline in a way that *should* break parity, and assert the comparison
    notices. If these ever start passing silently, the harness is broken.
    """

    def _compare(self, spec: str, rows: list) -> dict:
        write_rows_csv(ROWS_PATH, rows)
        SPEC_PATH.write_text(spec, encoding="utf-8")
        run_rust(SPEC_PATH, ROWS_PATH, RUST_OUT_PATH)
        return read_cells(RUST_OUT_PATH)

    def _diverges(self, py_spec, rust_spec, rows: list) -> bool:
        py_results = evaluate(py_spec, rows)
        rust_cells = self._compare(rust_spec, rows)
        for index, features in enumerate(py_results):
            name = next(iter(features))
            if not (features[name] == rust_cells[(index, name)]):
                return True
        return False

    def test_detects_off_by_one_window(self) -> None:
        """A 3-row window and a 2-row window must not look alike."""
        rows = synthetic_rows(8)
        py = FeaturePipeline().AddFeature("f", window(col("price"), 3, "mean"))
        rs = FeaturePipeline().AddFeature("f", window(col("price"), 2, "mean"))
        self.assertTrue(
            self._diverges(py.compile(), rs.compile(), rows),
            "a 3-row and a 2-row window produced identical output; the comparison "
            "cannot be sensitive to window size",
        )

    def test_detects_window_size_change(self) -> None:
        rows = synthetic_rows(10)
        py = FeaturePipeline().AddFeature("f", window(col("price"), 5, "mean"))
        rs = FeaturePipeline().AddFeature("f", window(col("price"), 4, "mean"))
        self.assertTrue(
            self._diverges(py.compile(), rs.compile(), rows),
            "changing the window size did not change the output; parity check is blind",
        )

    def test_detects_mean_masquerading_as_zscore(self) -> None:
        """A non-causal or wrong-formula implementation must not slip through."""
        rows = synthetic_rows(10)
        py = FeaturePipeline().AddFeature("z", zscore(col("price"), 5))
        rs = FeaturePipeline().AddFeature("z", window(col("price"), 5, "mean"))
        self.assertTrue(
            self._diverges(py.compile(), rs.compile(), rows),
            "a mean was indistinguishable from a z-score; the check is too weak",
        )


class SpecTest(unittest.TestCase):
    """The compiled spec must be deterministic, strict, and validated."""

    def test_compile_is_deterministic(self) -> None:
        self.assertEqual(build_pipeline().compile(), build_pipeline().compile())

    def test_spec_is_strict_json(self) -> None:
        document = json.loads(build_pipeline().compile())
        self.assertEqual(document["version"], 1)
        self.assertEqual(
            [f["name"] for f in document["features"]],
            build_pipeline().feature_names,
        )

    def test_literals_and_arithmetic_compile(self) -> None:
        spec = FeaturePipeline()
        spec.AddFeature("const", lit(2.5))
        spec.AddFeature("scaled", col("price") * 2.0)
        document = json.loads(spec.compile())
        self.assertEqual(document["features"][0]["expr"]["op"], "literal")
        self.assertEqual(document["features"][1]["expr"]["op"], "mul")

    def test_window_rejects_bad_arguments(self) -> None:
        for bad in (0, -1, 2.5):
            with self.subTest(size=bad), self.assertRaises(ValueError):
                window(col("price"), bad)
        with self.assertRaises(ValueError):
            window(col("price"), 3, "median")

    def test_add_feature_rejects_non_expression(self) -> None:
        """A user callable cannot be made deterministic, so it is refused."""
        spec = FeaturePipeline()
        with self.assertRaises(TypeError):
            spec.AddFeature("bad", lambda row: 1.0)
        with self.assertRaises(ValueError):
            spec.AddFeature("", col("price"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

