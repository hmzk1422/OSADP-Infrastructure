from __future__ import annotations

import json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uets.playback import PathMetrics, _report, _run_one_path


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


def _write_state_space_fixture(path: Path) -> tuple[int, int]:
    row_count = 8_000
    interval_ms = 300_000
    start_ms = (1_800_000_000_000 // interval_ms) * interval_ms
    timestamps_ms = start_ms + np.arange(row_count, dtype=np.int64) * 1_000
    volume = np.zeros(row_count, dtype=np.float64)
    volume[[10, 310]] = 1.0
    columns = {
        "datetime": pa.array(timestamps_ms * 1_000, type=pa.timestamp("us")),
        "session_id": pa.array(np.zeros(row_count, dtype=np.int32)),
        "execution_price": pa.array(np.full(row_count, 100.0)),
        "volume": pa.array(volume),
        "bar_vol_std": pa.array(np.full(row_count, 0.01)),
    }
    feature_names = (
        "spread_bps_z",
        "ma_slope_5",
        "rsi_x_ma",
        "ma_accel",
        "price_roc_3",
        "rsi_2",
        "sma_14",
        "ofi_z",
        "cvd_delta_z",
        "cvd_slope_z",
        "absorption_z",
        "tick_autocorr",
        "inst_ratio",
        "delta_divergence_z",
        "depth_imbalance_z",
    )
    for name in feature_names:
        columns[name] = pa.array(
            np.full(row_count, 20.0 if name == "rsi_2" else 0.0)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return start_ms, start_ms + interval_ms


class PlaybackRunnerTests(unittest.TestCase):
    def test_ofmr_playback_uses_rust_model_on_cryptol2(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uets-ofmr-playback-") as temp:
            work = Path(temp)
            cryptol2 = work / "cryptol2"
            _write_cryptol2_fixtures(cryptol2)
            catalog_path = work / "catalog.json"
            catalog_path.write_text(
                json.dumps({
                    "models": [{
                        "fold": 1,
                        "model": str(ROOT / "uets" / "models" / "ofmr" / "trader.txt"),
                        "test_start_ms": 1_800_000_000_000,
                        "test_end_ms": 1_800_000_080_000,
                    }],
                }),
                encoding="utf-8",
            )
            config_path = work / "ofmr-playback.yaml"
            config_path.write_text(
                yaml.safe_dump({
                    "mode": "playback",
                    "playback": {
                        "method": "purged_walk_forward",
                        "source": "cryptol2",
                        "n_splits": 4,
                        "cryptol2": {
                            "symbols": ["BTCUSDT"],
                            "path_glob": str(
                                cryptol2 / "{symbol}" / "merged" / "**" / "*.parquet"
                            ),
                            "walk_forward": {"model_catalog": str(catalog_path)},
                        },
                        "strategy": {"type": "ofmr"},
                    },
                }),
                encoding="utf-8",
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
            self.assertIn("Strategy: OFMR (Rust)", result.stdout)
            self.assertNotIn("Strategy: momentum baseline", result.stdout)
            self.assertIn("Playback report: purged_walk_forward", result.stdout)
            self.assertIn("folds/combinations: 1", result.stdout)
            self.assertIn("Average expectancy per trade:", result.stdout)
            self.assertIn(" R +- ", result.stdout)
            print("OFMR CryptoL2 playback: SUCCESSFUL")

    def test_ofmr_expectancy_is_reported_in_risk_units(self) -> None:
        prices = np.array([[100.0], [101.0], [101.0]])
        positions = np.array([[1.0], [0.0], [0.0]])
        risk = np.array([[0.01], [0.0], [0.0]])
        metrics = _run_one_path(
            prices,
            {"fee_bps": 0.0, "starting_capital": 10_000.0},
            positions,
            risk,
        )
        self.assertAlmostEqual(metrics.expectancy_per_trade, 1.0)
        self.assertEqual(metrics.trades, 1)

    def test_report_weights_average_expectancy_by_trade_count(self) -> None:
        report = _report(
            "purged_walk_forward",
            [
                PathMetrics(0.0, 2.0, 0.0, 0.0, 9),
                PathMetrics(0.0, -4.0, 0.0, 0.0, 1),
            ],
            strategy="ofmr",
        )
        self.assertIn(
            "Average expectancy per trade: 1.400000 R +- "
            "[CVaR05 -4.000000 R, CVaR95 2.000000 R]",
            report,
        )

    def test_state_space_playback_uses_catalogued_research_folds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uets-ofmr-state-space-") as temp:
            work = Path(temp)
            state_space = work / "state_space.parquet"
            start_ms, end_ms = _write_state_space_fixture(state_space)
            catalog_path = work / "catalog.json"
            catalog_path.write_text(
                json.dumps({
                    "models": [{
                        "fold": 1,
                        "model": str(ROOT / "uets" / "models" / "ofmr" / "trader.txt"),
                        "test_start_ms": start_ms,
                        "test_end_ms": end_ms,
                        "test_rows": 2,
                    }],
                }),
                encoding="utf-8",
            )
            config_path = work / "state-space-playback.yaml"
            config_path.write_text(
                yaml.safe_dump({
                    "mode": "playback",
                    "playback": {
                        "method": "purged_walk_forward",
                        "source": "state_space",
                        "starting_capital": 10_000.0,
                        "fee_bps": 4.0,
                        "periods_per_year": 252.0,
                        "state_space": {
                            "path": str(state_space),
                            "walk_forward": {"model_catalog": str(catalog_path)},
                        },
                        "strategy": {"type": "ofmr"},
                    },
                }),
                encoding="utf-8",
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
            self.assertIn("Strategy: OFMR (Rust)", result.stdout)
            self.assertIn("Paths run: 1; rejected: 0; folds/combinations: 1", result.stdout)
            self.assertIn("Average expectancy per trade:", result.stdout)
            print("OFMR state-space playback: SUCCESSFUL")

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
