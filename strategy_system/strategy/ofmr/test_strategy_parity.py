"""Compare production OFMR features and emit a causal path for Rust parity.

The separate test_ml_state_optimized_parity.py uses the saved study folds to
check feature mapping and model-score parity. This module's trade path is the
live Python path consumed by the Python/Rust comparator, not the study's
label-aware walk-forward trade log.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

_OFMR_DIR = Path(__file__).resolve().parent
_STRATEGY_DIR = _OFMR_DIR.parent
_REPO_ROOT = _OFMR_DIR.parents[2]
for _path in (str(_REPO_ROOT), str(_STRATEGY_DIR), str(_OFMR_DIR)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

from features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_SOURCE_COLUMNS,
    _matched_files,
    compute_features_from_matched,
)
from strategy import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    OfmrStrategy,
    TradeDecision,
)

COSINE_THRESHOLD = 0.9999
MAX_ABSOLUTE_ERROR = 1e-6
DEFAULT_TAG = "2025-06-30"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"cosine inputs must have equal shape: {a.shape} != {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return 0.0
    a_valid, b_valid = a[valid], b[valid]
    norm_a = np.linalg.norm(a_valid)
    norm_b = np.linalg.norm(b_valid)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if np.array_equal(a_valid, b_valid) else 0.0
    return float(np.dot(a_valid, b_valid) / (norm_a * norm_b))


def _build_walkforward_reference(
    matched_root: str, symbol: str, tag: Optional[str]
) -> pl.DataFrame:
    from abstraction import (  # noqa: PLC0415
        build_matched_session_context,
        load_matched_frames,
    )
    from ml_state_builder import (  # noqa: PLC0415
        _load_and_build_state_space_from_frames,
    )

    files = _matched_files(matched_root, symbol, tag)
    orderbook, trades = load_matched_frames(files)
    state_space = _load_and_build_state_space_from_frames(
        orderbook,
        trades,
        session_id=0,
        session_features=build_matched_session_context(orderbook, trades),
    )
    return state_space.select("datetime", *FEATURE_SOURCE_COLUMNS).with_columns(
        pl.col("datetime").cast(pl.Datetime("us"))
    )


def _strategy_path(
    frame: pl.DataFrame, model_path: str
) -> tuple[pl.DataFrame, int]:
    strategy = OfmrStrategy(model_path=model_path)
    timestamps_ms = frame["datetime"].cast(pl.Int64).to_numpy() // 1000
    features = frame.select(FEATURE_NAMES).to_numpy()
    decisions: list[int] = []
    positions: list[int] = []
    probabilities: list[float] = []
    for row, timestamp_ms in zip(frame.iter_rows(named=True), timestamps_ms):
        x = features[len(decisions)]
        decision, info = strategy.update(
            x,
            float(timestamp_ms),
            float(row["price"]),
            bar_vol_std=float(row["bar_vol_std"]),
        )
        decisions.append(
            {
                TradeDecision.HOLD: 0,
                TradeDecision.ENTER_LONG: 1,
                TradeDecision.EXIT: 2,
            }[decision]
        )
        positions.append(1 if strategy.position.value == "long" else 0)
        probabilities.append(float(info.get("probability", 0.0)))

    output = frame.select("datetime", "price", *FEATURE_NAMES, "bar_vol_std").with_columns(
        pl.Series("timestamp_ms", timestamps_ms),
        pl.Series("decision", decisions),
        pl.Series("position", positions),
        pl.Series("probability", probabilities),
    )
    return output, sum(value == 1 for value in decisions)


def evaluate_strategy_parity(
    matched_root: str,
    symbol: str = "BTCUSDT",
    tag: Optional[str] = DEFAULT_TAG,
    reference_state_space: Optional[str] = None,
    python_out_csv: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = COSINE_THRESHOLD,
) -> dict[str, float | bool]:
    production = compute_features_from_matched(matched_root, symbol=symbol, tag=tag).frame
    if production.is_empty():
        raise ValueError("Production feature builder returned no rows")
    if production["datetime"].n_unique() != production.height:
        raise ValueError("Production feature frame contains duplicate timestamps")

    if reference_state_space:
        reference_path = Path(reference_state_space)
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference state-space file not found: {reference_path}")
        reference = pl.read_parquet(
            reference_path,
            columns=["datetime", *FEATURE_SOURCE_COLUMNS],
        ).with_columns(pl.col("datetime").cast(pl.Datetime("us")))
        reference_label = str(reference_path)
    else:
        reference = _build_walkforward_reference(matched_root, symbol, tag)
        reference_label = "fresh ml_state_optimized state-space build"
    if reference["datetime"].n_unique() != reference.height:
        raise ValueError(
            f"Reference state space contains duplicate timestamps: {reference_label}"
        )

    reference = reference.select(
        "datetime",
        *[
            pl.col(source).alias(target)
            for source, target in zip(FEATURE_SOURCE_COLUMNS, FEATURE_NAMES)
        ],
    )
    aligned = production.join(
        reference,
        on="datetime",
        how="left",
        suffix="_reference",
        maintain_order="left",
    )
    if aligned.height != production.height:
        raise ValueError(
            f"Timestamp alignment changed row count: production={production.height}, "
            f"aligned={aligned.height}"
        )
    missing = aligned.select(
        pl.any_horizontal(
            *[pl.col(f"{name}_reference").is_null() for name in FEATURE_NAMES]
        ).sum()
    ).item()
    if missing:
        raise ValueError(
            f"{missing} production timestamp(s) have no matching reference row "
            f"in {reference_label}"
        )

    production_matrix = production.select(FEATURE_NAMES).to_numpy()
    reference_matrix = aligned.select(
        *[f"{name}_reference" for name in FEATURE_NAMES]
    ).to_numpy()

    scores: dict[str, float | bool] = {}
    errors: dict[str, float] = {}
    print(
        f"[PARITY] matched rows={production.height:,}; reference={reference_label}; "
        f"symbol={symbol}; tag={tag}"
    )
    for index, name in enumerate(FEATURE_NAMES):
        left = production_matrix[:, index]
        right = reference_matrix[:, index]
        similarity = _cosine_sim(left, right)
        max_error = float(np.max(np.abs(left - right)))
        scores[name] = similarity
        errors[name] = max_error
        status = (
            "MATCH"
            if similarity >= threshold and max_error <= MAX_ABSOLUTE_ERROR
            else "DIVERGED"
        )
        print(
            f"  {name:12s}: cosine={similarity:.8f}, "
            f"max_abs_error={max_error:.3g} -> {status}"
        )

    average = float(np.mean([float(scores[name]) for name in FEATURE_NAMES]))
    average_error = float(np.mean(list(errors.values())))
    passed = all(
        float(scores[name]) >= threshold
        and errors[name] <= MAX_ABSOLUTE_ERROR
        for name in FEATURE_NAMES
    )

    strategy_output, entries = _strategy_path(production, model_path)
    print(f"  trader.txt entries: {entries}")
    print(f"Overall average cosine similarity: {average:.8f}")
    print(f"Mean max absolute feature error:    {average_error:.3g}")
    print(f"FEATURE RESULT: {'PASS' if passed else 'FAIL'}")
    print("Python live trade path generated separately from study walk-forward trades.")

    if python_out_csv:
        strategy_output.write_csv(python_out_csv)
        print(f"Python strategy path written to: {python_out_csv}")
    return {
        "average": average,
        "mean_max_absolute_error": average_error,
        "rows": float(production.height),
        "entries": float(entries),
        "passed": passed,
        **scores,
    }


def test_strategy_parity() -> None:
    matched_root = os.environ.get(
        "PARITY_DATA_PATH",
        os.environ.get("OFMR_MATCHED_ROOT", str(_REPO_ROOT / "market_data" / "matched")),
    )
    tag = os.environ.get("PARITY_TAG", os.environ.get("OFMR_PARITY_TAG", DEFAULT_TAG)) or None
    result = evaluate_strategy_parity(matched_root, tag=tag)
    assert result["passed"], (
        f"OFMR feature parity failed: average cosine={result['average']:.8f}, "
        f"mean max error={result['mean_max_absolute_error']:.3g}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OFMR feature and strategy parity test")
    parser.add_argument(
        "--matched-root",
        default=os.environ.get(
            "PARITY_DATA_PATH",
            os.environ.get("OFMR_MATCHED_ROOT", str(_REPO_ROOT / "market_data" / "matched")),
        ),
    )
    parser.add_argument("--symbol", default=os.environ.get("PARITY_SYMBOL", "BTCUSDT"))
    parser.add_argument(
        "--tag",
        default=os.environ.get("PARITY_TAG", DEFAULT_TAG),
    )
    parser.add_argument("--reference-state-space", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--python-out", default=os.environ.get("PARITY_PYTHON_OUT"))
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD)
    args = parser.parse_args()

    result = evaluate_strategy_parity(
        matched_root=args.matched_root,
        symbol=args.symbol,
        tag=args.tag or None,
        reference_state_space=args.reference_state_space,
        python_out_csv=args.python_out,
        model_path=args.model,
        threshold=args.threshold,
    )
    if not result["passed"]:
        sys.exit(1)
