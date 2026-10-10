"""Interchangeable barrier mathematics and estimator factories.

Every calculation the triple-barrier engine delegates about *widths,
targets, timeouts and default models* lives here so it can be replaced
without touching the orchestration in ``triple_barrier.py``:

* :func:`rr_unit`           -- baseline risk unit U_t ("1R") per row
* :func:`loss_function`     -- stop width from a risk unit
* :func:`tp_function`       -- take-profit width from the stop width
* :func:`timeout_function`  -- timeout (bars) -> validated holding horizon
* :func:`classifier_model`  -- default P(TP first) estimator
* :func:`regressor_model`   -- default MFE/MAE estimator
* :func:`raw_fraction_unit` -- legacy unit U_t = 1 (raw price fractions)

Edit the body of any function to change the math, or pass a replacement
callable into ``tb_classifier()`` / ``tb_regressor()`` for a one-off
experiment. ``triple_barrier.py`` resolves these defaults at *call time*
(never as bound argument defaults), so monkeypatching e.g.
``models.rr_unit`` also affects subsequent runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "classifier_model",
    "loss_function",
    "raw_fraction_unit",
    "regressor_model",
    "rr_unit",
    "timeout_function",
    "tp_function",
]


def rr_unit(
    df: pd.DataFrame,
    *,
    price_col: str,
    window: int = 14,
    vol_col: str | None = None,
    sqrt_horizon: float | None = None,
) -> np.ndarray:
    """Baseline risk unit U_t for every row (fraction of price, causal).

    Default math: trailing per-bar volatility -- the column ``vol_col`` if
    given, otherwise the rolling standard deviation of simple bar returns
    of ``price_col`` -- optionally multiplied by ``sqrt_horizon`` (pass
    ``np.sqrt(horizon_bars)`` to widen a per-bar unit to the holding path;
    the ``tb_*`` orchestrators do exactly that). Row i only uses data up to
    i, so early rows without enough history are NaN and stay unscored.

    **Edit this body to change how "1R" is defined** (ATR, Parkinson, a
    fixed grid, ...). The contract: one finite-or-NaN value per row of
    ``df``; the orchestrator passes ``window``/``vol_col``/``sqrt_horizon``
    through from the settings section (keys rr_window|vol_window,
    rr_col|vol_col, sqrt_horizon).
    """
    if vol_col is not None:
        if vol_col not in df.columns:
            raise KeyError(f"rr_unit vol_col {vol_col!r} not found in frame")
        unit = pd.to_numeric(df[vol_col], errors="coerce").to_numpy(dtype=float)
    else:
        if price_col not in df.columns:
            raise KeyError(f"rr_unit price_col {price_col!r} not found in frame")
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError(f"rr_unit window must be a positive integer; got {window!r}")
        price = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
        returns = pd.Series(price, index=df.index).pct_change(fill_method=None)
        unit = returns.rolling(window, min_periods=max(2, window // 3)).std(ddof=0).to_numpy()
    if sqrt_horizon is not None:
        unit = unit * sqrt_horizon
    return unit


def loss_function(rr_width, barrier_size: float):
    """Stop width: the risk unit scaled by ``barrier_size``.

    ``sl_t = U_t * barrier_size`` -- all of the stop math happens on this
    one line. **Edit this body** to floor widths, round to ticks, cap risk,
    or otherwise reshape how the stop distance is derived.
    """
    return rr_width * barrier_size


def tp_function(sl_width, tp_size: float):
    """Take-profit width: the stop width scaled by ``tp_size``.

    ``tp_t = sl_t * tp_size`` (tp_size = 1 -> 1R target, 2 -> 2R).
    **Edit this body** to change the R-multiple relationship, e.g. make
    targets asymmetric in volatility or trade count.
    """
    return sl_width * tp_size


def timeout_function(timeout_bars):
    """Timeout in bars -> the holding horizon used everywhere (validated).

    The returned value drives *everything* consistently: the purge distance
    of the chronological split, ``path_ok`` (rows that cannot enter),
    the simulation horizon, and therefore the labels and realized R.
    Default: identity after validating a positive integer bar count.
    **Edit this body** to cap, round, or scale configured timeouts instead.
    """
    if (isinstance(timeout_bars, bool)
            or not isinstance(timeout_bars, (int, np.integer))
            or timeout_bars <= 0):
        raise ValueError(
            f"timeout must be a positive integer number of bars; got {timeout_bars!r}"
        )
    return int(timeout_bars)


def raw_fraction_unit(
    df: pd.DataFrame,
    *,
    price_col: str,
    window: int = 14,
    vol_col: str | None = None,
    sqrt_horizon: float | None = None,
) -> np.ndarray:
    """Legacy unit for ``evaluate_regressor``: raw price fractions (U_t = 1).

    The original regressor fitted MFE/MAE as plain price fractions with no
    baseline unit; this shim keeps that exact behaviour selectable while the
    default ``tb_regressor`` normalises its targets by :func:`rr_unit`.
    Arguments are accepted for signature compatibility and ignored.
    """
    return np.ones(len(df))


def classifier_model():
    """Default P(TP first) estimator: scaled logistic regression.

    **Edit this body to change the default model** for every
    ``tb_classifier`` run that does not receive an explicit model.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def regressor_model():
    """Default excursion estimator: scaled linear regression.

    One clone is fitted per target (MFE_R and MAE_R). **Edit this body to
    change the default model** for every ``tb_regressor`` run without an
    explicit model.
    """
    return make_pipeline(StandardScaler(), LinearRegression())
