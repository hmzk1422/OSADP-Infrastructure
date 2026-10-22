"""Real-data smoke test for the native Rust GP path; run from the repository root."""

from __future__ import annotations

import json
import tempfile
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy_system.exchange.loader import load_L2
from strategy_system.strategy.moe.ops.export import export_gp_features
from strategy_system.strategy.moe.ops.paths import models_dir_from_cfg
from strategy_system.strategy.moe.ops.settings import load_settings
from strategy_system.strategy.moe.strategy import run_gp_new


def main() -> None:
    settings = load_settings()
    with tempfile.TemporaryDirectory(prefix="rust-gp-smoke-") as output_dir:
        settings.setdefault("ops", {})["models_dir"] = output_dir
        data = load_L2(**settings["data"].get("orderbook", {}))
        result = run_gp_new(data, settings)

        if result.features.empty or len(result.features) != len(result.feat_df):
            raise AssertionError("Rust GP returned no features or misaligned feature rows")
        screen = result.feature_set.stats["correlation_screen"]
        verified_rows = int(screen["verification_rows"])
        if screen.get("verification_scope") != "all_rows" or verified_rows != len(result.features):
            raise AssertionError(
                f"correlation screen did not verify every feature row: {screen}"
            )

        library_path = Path(output_dir) / "gp" / "library.json"
        if not library_path.is_file():
            raise AssertionError(f"Rust GP did not save its feature library: {library_path}")
        model = SimpleNamespace(
            feature_columns=tuple(result.features.columns),
            target_columns=("future_return__smoke",),
            feature_mean=np.zeros(result.features.shape[1]),
            feature_scale=np.ones(result.features.shape[1]),
            coefficients=np.zeros((result.features.shape[1], 1)),
            intercept=np.zeros(1),
            n_train=verified_rows,
        )
        payload_path = export_gp_features(
            models_dir_from_cfg(settings), "smoke", result.feature_set,
            model, list(result.terminals.columns), ["smoke"],
        )
        if not payload_path.is_file():
            raise AssertionError("Rust GP feature export was not created")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not payload["input_terminals"] or len(payload["features"]) != len(result.features.columns):
            raise AssertionError("Rust GP model export omitted feature trees or required terminals")
        if int(screen["selected_features"]) != result.features.shape[1]:
            raise AssertionError("the final GP feature matrix differs from the screened library")

        print(
            f"rows={len(result.features)} terminals={result.terminals.shape[1]} "
            f"features={result.features.shape[1]} "
            f"pearson_spearman_screen={screen['threshold']} "
            f"verified_rows={verified_rows}"
        )
        print("SUCCESSFUL")


if __name__ == "__main__":
    main()
