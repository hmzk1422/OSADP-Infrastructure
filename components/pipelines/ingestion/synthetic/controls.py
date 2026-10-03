"""Shared controls for configurable synthetic market generators."""

from __future__ import annotations

import numpy as np


def validate_generator_controls(
    volatility_scale: float,
    directional_bias: float,
    regime_persistence: float | None,
) -> None:
    """Validate common controls shared by Synth1, Synth2, and Synth3.

    ``directional_bias`` is an additive log-return bias per generated bar.
    ``regime_persistence`` is the desired diagonal probability of the base
    regime transition matrix. ``None`` preserves that generator's original
    transition matrix.
    """
    if not np.isfinite(volatility_scale) or volatility_scale <= 0.0:
        raise ValueError("volatility_scale must be finite and > 0")
    if not np.isfinite(directional_bias):
        raise ValueError("directional_bias must be finite")
    if regime_persistence is not None and (
        not np.isfinite(regime_persistence)
        or not 0.0 <= regime_persistence <= 1.0
    ):
        raise ValueError("regime_persistence must be in [0, 1] or None")


def with_regime_persistence(
    transition: np.ndarray,
    persistence: float | None,
) -> np.ndarray:
    """Set each row's diagonal probability while preserving off-diagonal mix."""
    matrix = np.asarray(transition, dtype=np.float64)
    if persistence is None:
        return matrix.copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be a square matrix")
    if len(matrix) == 0:
        raise ValueError("transition must contain at least one regime")
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("persistence must be in [0, 1]")
    if len(matrix) == 1:
        if persistence != 1.0:
            raise ValueError("a one-regime process can only have persistence=1")
        return np.ones((1, 1), dtype=np.float64)

    adjusted = matrix.copy()
    for row in range(len(adjusted)):
        off_diagonal = adjusted[row].copy()
        off_diagonal[row] = 0.0
        off_sum = float(off_diagonal.sum())
        if off_sum > 0.0:
            off_diagonal *= (1.0 - persistence) / off_sum
        elif persistence < 1.0:
            off_diagonal[:] = (1.0 - persistence) / (len(adjusted) - 1)
            off_diagonal[row] = 0.0
        adjusted[row] = off_diagonal
        adjusted[row, row] = persistence
    return adjusted
