"""Verify production feature mapping and model inference against study artifacts.

This checks the saved purged-walk-forward models and their out-of-sample rows.
The study's label-aware trade selection is deliberately not treated as a live
trade path; the causal Python/Rust path is covered by test_strategy_parity.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

_OFMR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _OFMR_DIR.parents[2]
for _path in (str(_REPO_ROOT), str(_OFMR_DIR.parent), str(_OFMR_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_SOURCE_COLUMNS,
    _as_feature_result,
)
from strategy import OfmrStrategy  # noqa: E402
from config import RSI_FILTER_THRESHOLD  # noqa: E402

BAR_SECONDS = 300
SIGNAL_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.20, 0.96, 0.05))
DEFAULT_STATE_SPACE = (
    _OFMR_DIR / "models" / "state_space_btcusdt_720files.parquet"
)
DEFAULT_CATALOG = _REPO_ROOT / "build" / "walkforward_models" / "catalog.json"
DEFAULT_MODEL_DIR = _REPO_ROOT / "build" / "walkforward_models"


def _read_study_candidates(
    state_space_path: Path, models: list[dict]
) -> pl.DataFrame:
    if not state_space_path.is_file():
        raise FileNotFoundError(
            f"Study state-space artifact not found: {state_space_path}. "
            "Run ml_state_optimized.py first or pass --state-space."
        )

    timestamp_ms = pl.col("datetime").cast(pl.Int64) // 1_000
    bucket = pl.col("datetime").dt.truncate(f"{BAR_SECONDS}s")
    in_fold_window = pl.lit(False)
    for model in models:
        start_ms = int(model["test_start_ms"])
        end_ms = int(model["test_end_ms"])
        in_fold_window |= (
            (timestamp_ms >= start_ms)
            & (timestamp_ms < end_ms + BAR_SECONDS * 1_000)
        )

    selected_columns = [
        "datetime",
        "volume",
        "execution_price",
        "bar_vol_std",
        *FEATURE_SOURCE_COLUMNS,
    ]
    candidates = (
        pl.scan_parquet(state_space_path)
        .select(selected_columns)
        .filter(in_fold_window)
        .filter(pl.col("volume") > 0)
        .sort("datetime")
        .with_columns(bucket.alias("_bucket"))
        .group_by("_bucket", maintain_order=True)
        .last()
        .with_columns(pl.col("_bucket").alias("datetime"))
        .drop("_bucket")
        .filter(pl.col("rsi_2") < RSI_FILTER_THRESHOLD)
        .sort("datetime")
        .collect()
    )
    return candidates.with_columns(
        (pl.col("datetime").cast(pl.Int64) // 1_000).alias("timestamp_ms")
    )


def evaluate_study_parity(
    state_space_path: Path = DEFAULT_STATE_SPACE,
    catalog_path: Path = DEFAULT_CATALOG,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, float | int | bool]:
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Walk-forward model catalog not found: {catalog_path}. "
            "Run ml_state_optimized.py first or pass --catalog."
        )
    with catalog_path.open(encoding="utf-8") as file:
        catalog = json.load(file)
    models = catalog.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"No walk-forward models listed in {catalog_path}")

    candidates = _read_study_candidates(state_space_path, models)
    if candidates.is_empty():
        raise ValueError(
            f"No eligible out-of-sample rows found in {state_space_path}"
        )

    total_rows = 0
    max_feature_error = 0.0
    max_probability_error = 0.0
    signal_comparisons = 0
    from lightgbm import Booster

    print(
        f"[STUDY] state_space={state_space_path}; catalog={catalog_path}; "
        f"eligible_rows={candidates.height}"
    )

    for fold in models:
        start_ms = int(fold["test_start_ms"])
        end_ms = int(fold["test_end_ms"])
        fold_rows = candidates.filter(
            pl.col("timestamp_ms").is_between(start_ms, end_ms, closed="both")
        )
        expected_rows = int(fold["test_rows"])
        if fold_rows.height != expected_rows:
            raise AssertionError(
                f"Fold {fold['fold']} has {fold_rows.height} eligible study rows; "
                f"catalog expects {expected_rows}. State-space/catalog mismatch."
            )
        if int(fold_rows["timestamp_ms"][0]) != start_ms or int(
            fold_rows["timestamp_ms"][-1]
        ) != end_ms:
            raise AssertionError(
                f"Fold {fold['fold']} candidate timestamps do not match catalog bounds"
            )

        model_path = model_dir / str(fold["model"])
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold['fold']} model not found: {model_path}"
            )

        mapped = _as_feature_result(fold_rows).frame
        reference_features = fold_rows.select(FEATURE_SOURCE_COLUMNS).to_numpy()
        production_features = mapped.select(FEATURE_NAMES).to_numpy()
        feature_error = float(
            np.max(np.abs(reference_features - production_features))
        )
        max_feature_error = max(max_feature_error, feature_error)
        if feature_error != 0.0:
            raise AssertionError(
                f"Fold {fold['fold']} production feature mapping differs from "
                f"the study state space (max abs error {feature_error})"
            )

        matrix = production_features.astype(np.float64, copy=False)
        study_probabilities = Booster(model_file=str(model_path)).predict(matrix)
        production_strategy = OfmrStrategy(
            model_path=str(model_path),
            threshold=1.1,
        )
        production_probabilities = np.fromiter(
            (
                production_strategy._predict_proba(row)
                for row in matrix
            ),
            dtype=np.float64,
            count=len(matrix),
        )
        probability_error = float(
            np.max(np.abs(study_probabilities - production_probabilities))
        )
        max_probability_error = max(max_probability_error, probability_error)
        if not np.allclose(
            study_probabilities,
            production_probabilities,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                f"Fold {fold['fold']} production inference differs from the "
                f"study model (max abs error {probability_error})"
            )

        for threshold in SIGNAL_THRESHOLDS:
            study_signals = study_probabilities >= threshold
            production_signals = production_probabilities >= threshold
            if not np.array_equal(study_signals, production_signals):
                raise AssertionError(
                    f"Fold {fold['fold']} signals differ at threshold {threshold}"
                )
            signal_comparisons += len(study_signals)

        total_rows += len(matrix)
        print(
            f"  fold {fold['fold']:>2}: rows={len(matrix)}; "
            f"max_feature_error={feature_error:.3g}; "
            f"max_probability_error={probability_error:.3g}; "
            "all threshold signals MATCH"
        )

    print(
        f"[STUDY] rows={total_rows}; signal comparisons={signal_comparisons}; "
        f"max feature error={max_feature_error:.3g}; "
        f"max probability error={max_probability_error:.3g}; RESULT: PASS"
    )
    return {
        "rows": total_rows,
        "signal_comparisons": signal_comparisons,
        "max_feature_error": max_feature_error,
        "max_probability_error": max_probability_error,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare OFMR Python features and model scores with saved "
        "ml_state_optimized walk-forward artifacts."
    )
    parser.add_argument("--state-space", type=Path, default=DEFAULT_STATE_SPACE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    evaluate_study_parity(args.state_space, args.catalog, args.model_dir)
    return 0


def test_ml_state_optimized_parity() -> None:
    evaluate_study_parity()


if __name__ == "__main__":
    raise SystemExit(main())
