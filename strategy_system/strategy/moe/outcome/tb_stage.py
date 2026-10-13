"""Triple-barrier stage machinery (everything ``run_tb`` delegates to).

Per latent: features = [pred__<h>] + [err__<h>] only (no raw / GP columns),
fitted through ``tb_regressor`` which predicts MFE_R / MAE_R.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .triple_barrier import tb_regressor

_SIDE_COL = "__tb_side"

# =========================================================================== #
# TRAINING MECHANISM  --  how much of each latent's history tb_regressor fits on.
# tb_regressor needs train_fraction in (0, 1); this value = "all episodes"
# (everything except the final purge window it must keep). Override with
# ``moe.tb_train_fraction`` (e.g. 0.7) to get a chronological holdout instead.
# =========================================================================== #
TB_TRAIN_FRACTION_ALL = 1.0 - 1e-9

ERR_RULE = (
    "err[<h>] = realized future_return[<h>] - pred[<h>] of the SAME latent, taken from the most "
    "recent prediction whose endpoint (future_timestamp[<h>]) is <= now; 0.0 until one has elapsed"
)


class TBFit(NamedTuple):
    feature_order: list
    tp_model: object        # sklearn pipeline -> MFE_R
    sl_model: object        # sklearn pipeline -> MAE_R
    n_fit: int
    parity_x: np.ndarray    # sample of the exact (zero-filled) training matrix
    stats: dict
    barrier: dict           # metadata block for Rust


def elapsed_errors(index: pd.Index, latent_id: pd.Series, realized: pd.Series,
                   predicted: pd.Series, endpoint: pd.Series, state: str) -> np.ndarray:
    """Most recent ELAPSED residual (realized - predicted) of one latent; 0.0 before the first.

    Causal: a residual is only visible at rows whose clock >= the predicting row's
    endpoint timestamp (backward merge_asof). Same-latent residuals only.
    """
    n = len(index)
    zeros = np.zeros(n, dtype=np.float64)
    if not isinstance(index, pd.DatetimeIndex):
        return zeros
    clock = pd.to_datetime(index, utc=True, errors="coerce")
    state_values = latent_id.astype("object").to_numpy()
    state_rows = np.flatnonzero(state_values == state)
    residual = pd.to_numeric(realized, errors="coerce") - pd.to_numeric(predicted, errors="coerce")
    known = pd.DataFrame({
        "_endpoint": pd.to_datetime(endpoint, utc=True, errors="coerce"),
        "_resid": residual.to_numpy(dtype=np.float64),
        "_state": state_values,
    })
    known = known[known["_state"] == state].dropna(subset=["_endpoint", "_resid"])
    if known.empty or state_rows.size == 0:
        return zeros
    left = pd.DataFrame({"_time": clock[state_rows], "_pos": state_rows})
    merged = pd.merge_asof(
        left.sort_values("_time", kind="mergesort"),
        known.sort_values("_endpoint", kind="mergesort"),
        left_on="_time", right_on="_endpoint", direction="backward",
    )
    values = merged["_resid"].to_numpy(dtype=np.float64)
    out = zeros.copy()
    out[merged["_pos"].to_numpy(dtype=np.int64)] = np.where(np.isfinite(values), values, 0.0)
    return out


def tb_params(cfg: dict) -> dict:
    """Barrier settings (outcome.triple_barrier.*) with the same defaults as before."""
    tb = dict(((cfg.get("outcome") or {}).get("triple_barrier", {})) or {})
    reg = dict(tb.get("regressor", {}) or {})
    return {
        "price_col": tb.get("price_col", "price_close"),
        "high_col": tb.get("high_col"),
        "low_col": tb.get("low_col"),
        "side": tb.get("side", 1),
        "horizon": int(tb.get("horizon", 30)),
        "rr_window": int(tb.get("rr_window", tb.get("vol_window", 60))),
        "rr_col": tb.get("rr_col", tb.get("vol_col")),
        "barrier_size": float(reg.get("barrier_size", 1.0)),
        "tp_size": float(reg.get("tp_size", reg.get("rr", 1.0))),
        "timestamp_col": tb.get("timestamp_col"),
        "train_fraction": float((cfg.get("moe") or {}).get("tb_train_fraction", TB_TRAIN_FRACTION_ALL)),
    }


def build_tb_base_frame(df: pd.DataFrame, feat_df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Price / high / low / side columns on the feature index (raw columns attached causally)."""
    need = [c for c in (p["price_col"], p["high_col"], p["low_col"],
                        p["side"] if isinstance(p["side"], str) else None) if c is not None]
    frame = feat_df
    missing = [c for c in need if c not in frame.columns]
    if missing:
        from .alignment import attach_raw_columns
        frame = attach_raw_columns(df, frame, set(missing), p["timestamp_col"])
    return frame[need].copy()


def fit_latent_tb(base: pd.DataFrame, predictions: pd.DataFrame, targets: pd.DataFrame,
                  latent_id: pd.Series, state: str, horizons: list, p: dict) -> TBFit:
    """Fit tb_regressor for one latent on [pred__h..., err__h...] only."""
    frame = base.copy()
    for h in horizons:
        frame[f"pred__{h}"] = pd.to_numeric(predictions[f"future_return__{h}"], errors="coerce")
    for h in horizons:
        endpoint = f"future_timestamp__{h}"
        frame[f"err__{h}"] = (
            elapsed_errors(frame.index, latent_id,
                           pd.to_numeric(targets[f"future_return__{h}"], errors="coerce"),
                           frame[f"pred__{h}"], targets[endpoint], state)
            if endpoint in targets else 0.0
        )
    feature_order = [f"pred__{h}" for h in horizons] + [f"err__{h}" for h in horizons]

    in_latent = (latent_id.astype("object") == state).to_numpy()
    base_side = (pd.to_numeric(frame[p["side"]], errors="coerce").fillna(0).to_numpy()
                 if isinstance(p["side"], str) else np.full(len(frame), float(p["side"])))
    frame[_SIDE_COL] = np.where(in_latent, base_side, 0.0)  # side 0 = not an event = ignored

    result = tb_regressor(
        frame,
        feature_cols=feature_order,
        side=_SIDE_COL,
        price_col=p["price_col"], high_col=p["high_col"], low_col=p["low_col"],
        timeout=p["horizon"], rr_window=p["rr_window"], rr_col=p["rr_col"],
        barrier_size=p["barrier_size"], tp_size=p["tp_size"],
        train_fraction=p["train_fraction"],
    )
    models = result.model
    tp_model, sl_model = models["tp_model"], models["sl_model"]

    raw = frame[feature_order].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    raw = np.where(np.isfinite(raw), raw, 0.0)  # tb_regressor's own zero-fill rule
    rows = np.flatnonzero(in_latent)
    sample = raw[rows[:: max(1, len(rows) // 2000)]]
    n_fit = int(tp_model.steps[0][1].n_samples_seen_)

    barrier = {
        "side": p["side"],
        "horizon_bars": p["horizon"],
        "price_col": p["price_col"], "high_col": p["high_col"], "low_col": p["low_col"],
        "barrier_size": p["barrier_size"], "tp_size": p["tp_size"],
        "rr_unit": {
            "formula": ("U_t = rolling_std(simple_returns(price_col), window=rr_window, ddof=0, "
                        "min_periods=max(2, rr_window // 3)) * sqrt_horizon  (rr_col replaces the rolling std if set)"),
            "rr_window": p["rr_window"], "rr_col": p["rr_col"],
            "sqrt_horizon": float(np.sqrt(p["horizon"])),
        },
        "sl_width": "MAE_R * U_t * barrier_size   (fraction of entry price)",
        "tp_width": "MFE_R * U_t * barrier_size * tp_size   (fraction of entry price)",
        "levels_long": "tp = p * (1 + tp_width); sl = p * (1 - sl_width); stop wins ties; timeout after horizon_bars",
        "source": "models.py loss_function / tp_function defaults",
    }
    return TBFit(feature_order, tp_model, sl_model, n_fit, sample, result.stats, barrier)