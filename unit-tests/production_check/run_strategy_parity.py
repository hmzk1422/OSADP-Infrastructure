from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER_MANIFEST = ROOT / "unit-tests" / "production_check" / "live_parity_test" / "Cargo.toml"

PROFILES = {
    "ofmr": {
        "python_args": [
            "strategy_system/strategy/ofmr/test_strategy_parity.py",
            "--model",
            "uets/models/ofmr/trader.txt",
        ],
        "rust_args": [
            "run",
            "--quiet",
            "--manifest-path",
            "components/pipelines/uets_trades/Cargo.toml",
            "--bin",
            "playback_ofmr",
            "--",
            "--model",
            "uets/models/ofmr/trader.txt",
        ],
        "timestamp_col": "timestamp_ms",
        "feature_cols": ",".join(f"Column_{index}" for index in range(15)),
        "trade_cols": "decision,position",
        "python_out": "build/production_check/ofmr/python.csv",
        "rust_out": "build/production_check/ofmr/rust.csv",
        "rust_env": ["OFMR_CONFIDENCE_THRESHOLD=0.2"],
        "default_data": "market_data/matched",
        "default_symbol": "BTCUSDT",
        "default_tag": "2025-06-30",
    },
    "example_strategy": {
        "python_args": ["strategy_system/strategy/example_strategy/parity.py"],
        "rust_args": [
            "run",
            "--quiet",
            "--manifest-path",
            "uets/Cargo.toml",
            "--bin",
            "example_strategy_parity",
            "--",
        ],
        "timestamp_col": "timestamp_ms",
        "feature_cols": "dz",
        "trade_cols": "decision,position",
        "python_out": "build/production_check/example_strategy/python.csv",
        "rust_out": "build/production_check/example_strategy/rust.csv",
        "rust_env": [],
        "default_data": "unit-tests/production_check/fixtures/example_strategy.csv",
        "default_symbol": "EXAMPLE",
        "default_tag": "smoke",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the configured Python/Rust live parity check for a strategy."
    )
    parser.add_argument("--strategy", choices=sorted(PROFILES), required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--symbol")
    parser.add_argument("--tag")
    parser.add_argument("--threshold", type=float, default=0.9999)
    parser.add_argument("--max-mae", type=float, default=1e-5)
    args = parser.parse_args()
    profile = PROFILES[args.strategy]
    data_path = (ROOT / (args.data_path or profile["default_data"])).resolve()
    symbol = args.symbol or profile["default_symbol"]
    tag = args.tag if args.tag is not None else profile["default_tag"]
    output_dir = ROOT / "build" / "production_check" / args.strategy
    python_out = output_dir / "python.csv"
    rust_out = output_dir / "rust.csv"

    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(RUNNER_MANIFEST),
        "--",
        "--data-path",
        str(data_path),
        "--symbol",
        symbol,
        "--tag",
        tag,
        "--python-program",
        sys.executable,
        "--python-arg",
        profile["python_args"][0],
        "--rust-program",
        "cargo",
    ]
    for argument in profile["python_args"][1:]:
        command.extend(["--python-arg", argument])
    for argument in profile["rust_args"]:
        command.extend(["--rust-arg", argument])
    for key, value in (
        ("timestamp-col", profile["timestamp_col"]),
        ("feature-cols", profile["feature_cols"]),
        ("trade-cols", profile["trade_cols"]),
        ("python-out", str(python_out)),
        ("rust-out", str(rust_out)),
        ("threshold", str(args.threshold)),
        ("max-mae", str(args.max_mae)),
    ):
        command.extend([f"--{key}", value])
    for override in profile["rust_env"]:
        command.extend(["--rust-env", override])

    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
