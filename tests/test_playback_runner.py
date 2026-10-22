from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _write_cryptol2_fixtures(directory: Path) -> None:
    rng = np.random.default_rng(20261022)
    timestamps = np.arange(1_800_000_000_000, 1_800_000_080_000, 1_000)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        log_returns = rng.normal(0.0, 0.001, size=timestamps.size)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        table = pa.table({
            "timestamp": pa.array(timestamps, type=pa.int64()),
            "event_type": pa.array(["LOB"] * timestamps.size),
            "best_bid_price": pa.array(prices - 0.01),
            "best_ask_price": pa.array(prices + 0.01),
        })
        path = directory / symbol / "merged" / "20261022" / "00.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


class PlaybackRunnerTests(unittest.TestCase):
    def test_all_playback_modes_run_and_emit_reports(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uets-playback-") as temp:
            work = Path(temp)
            cryptol2 = work / "cryptol2"
            _write_cryptol2_fixtures(cryptol2)

            cases = (
                {
                    "method": "purged_walk_forward",
                    "source": "synthetic",
                    "n_splits": 4,
                    "synthetic": {"generator": "synth2", "paths": 4, "n_bars": 96, "seed": 10},
                },
                {
                    "method": "parametric_monte_carlo",
                    "source": "synthetic",
                    "max_paths": 8,
                    "synthetic": {"generator": "synth2", "paths": 8, "n_bars": 96, "seed": 100},
                },
                {
                    "method": "cpcv",
                    "source": "cryptol2",
                    "max_abs_correlation": 1.0,
                    "n_splits": 4,
                    "n_test_splits": 2,
                    "cryptol2": {
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "path_glob": str(cryptol2 / "{symbol}" / "merged" / "**" / "*.parquet"),
                    },
                },
            )
            for case in cases:
                with self.subTest(method=case["method"]):
                    config = {
                        "mode": "playback",
                        "playback": {
                            "workers": 2,
                            "strategy": {"type": "momentum", "lookback_bars": 3},
                            **case,
                        },
                    }
                    config_path = work / f"{case['method']}.yaml"
                    config_path.write_text(
                        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "uets.playback",
                            "--config",
                            str(config_path),
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                    )
                    self.assertIn(f"Playback report: {case['method']}", result.stdout)
                    self.assertIn("Cumulative return:", result.stdout)
                    self.assertIn("Average expectancy per trade:", result.stdout)
                    self.assertIn("MDD:", result.stdout)
                    self.assertIn("Sortino:", result.stdout)
                    print(f"{case['method']}: SUCCESSFUL")

    def test_rust_cli_dispatches_playback_module(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uets-playback-cli-") as temp:
            work = Path(temp)
            cryptol2 = work / "cryptol2"
            _write_cryptol2_fixtures(cryptol2)
            config_path = work / "playback.yaml"
            config_path.write_text(
                yaml.safe_dump({
                    "mode": "playback",
                    "playback": {
                        "method": "cpcv",
                        "source": "cryptol2",
                        "max_abs_correlation": 1.0,
                        "n_splits": 4,
                        "n_test_splits": 2,
                        "cryptol2": {
                            "symbols": ["BTCUSDT", "ETHUSDT"],
                            "path_glob": str(
                                cryptol2 / "{symbol}" / "merged" / "**" / "*.parquet"
                            ),
                        },
                    },
                }),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--manifest-path",
                    "uets/Cargo.toml",
                    "--bin",
                    "dev_test",
                    "--",
                    "playback",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("Playback report: cpcv", result.stdout)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PlaybackRunnerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("SUCCESSFUL")
