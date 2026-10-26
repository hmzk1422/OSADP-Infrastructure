"""TEST FIXTURE ONLY -- synthetic stand-in for ``load_L2`` used to validate infonet_study (positive / null controls).

World "sign_flip": 3 persistent regimes with observably different spread/depth/vol; order-flow pressure predicts
forward returns + in A, - in B (identical mutual information, opposite sign), nothing in C.
World "null": same microstructure differences, nothing predicts returns.

Usage:
    from <pkg>.synthetic_loader_stub import load_L2, set_world
    set_world("sign_flip", seed=2, mean_dwell=1500, beta=0.35)      # or "null"
    DataConfig(train_files=6, test_files=4, step_files=3, n_folds=3, decimate=2, loader=load_L2, feature_sets=("20",))
    RunConfig(segmenter="oracle", horizons=(4, 16, 32), block=100, min_episode_len=200)   # ground-truth regimes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROWS_PER_FILE = 10_000
_WORLD = {"name": "sign_flip", "seed": 0, "mean_dwell": 5000, "beta": 0.35, "total_files": 36}
_CACHE: dict = {}


def set_world(name: str = "sign_flip", seed: int = 0, mean_dwell: int = 5000, beta: float = 0.35,
              rows_per_file: int = 10_000, total_files: int = 36) -> None:
    """``total_files`` must cover the largest ``start + file_limit`` any fold will request:
    start + (n_folds - 1) * step_files + train_files + gap_files + test_files.  (The run_infonet_study.py defaults
    need 34.)  FIX: the world used to be generated with ``max(20, start + file_limit)`` files, i.e. a DIFFERENT random
    world (different price path) whenever a fold reached past file 20, so the train and test pieces of one fold, and the
    same file seen from two folds, did not belong to the same price series."""
    global ROWS_PER_FILE
    _WORLD.update(name=name, seed=seed, mean_dwell=mean_dwell, beta=beta, total_files=total_files)
    ROWS_PER_FILE = rows_per_file
    _CACHE.clear()


def _master(n_files: int | None = None) -> pd.DataFrame:
    n_files = _WORLD["total_files"] if n_files is None else n_files
    key = (_WORLD["name"], _WORLD["seed"], _WORLD["mean_dwell"], _WORLD["beta"], ROWS_PER_FILE, n_files)
    if key in _CACHE:
        return _CACHE[key]
    rng = np.random.default_rng(_WORLD["seed"])
    n = n_files * ROWS_PER_FILE
    # persistent regime path
    reg = np.empty(n, dtype=np.int64)
    i, cur = 0, 0
    while i < n:
        d = int(rng.geometric(1.0 / _WORLD["mean_dwell"]))
        cur = int(rng.choice([k for k in range(3) if k != cur])) if i else 0
        reg[i:i + d] = cur
        i += d
    beta_r = np.array([_WORLD["beta"], -_WORLD["beta"], 0.0]) if _WORLD["name"] == "sign_flip" else np.zeros(3)
    sig_r = np.array([1.0, 1.3, 1.8]) * 0.02
    spread_r = np.array([1.0, 1.5, 2.5])
    depth_r = np.array([1.0, 0.65, 0.4])
    # latent order-flow pressure s_t: AR(1)
    s = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        s[t] = 0.97 * s[t - 1] + 0.25 * eps[t]
    dm = beta_r[reg][1:] * 0.02 * s[:-1] + sig_r[reg][1:] * rng.standard_normal(n - 1)
    mid = 100.0 + np.concatenate([[0.0], np.cumsum(dm)])
    tick = 0.01
    spread = np.maximum(tick, np.round(spread_r[reg] * tick * (1 + rng.poisson(0.3, n))/tick) * tick)
    bid = np.floor((mid - spread / 2) / tick) * tick
    ask = bid + spread
    cols = {}
    cols["timestamp"] = pd.to_datetime(1_700_000_000_000 + np.cumsum(rng.integers(5, 120, n)), unit="ms")
    cols["event_type"] = np.where(rng.random(n) < 0.4, "trade", "book")
    cols["event_sequence"] = np.arange(n)
    base = 5.0 * depth_r[reg]
    for lvl in range(1, 21):
        bk, ak = ("best_bid", "best_ask") if lvl == 1 else (f"bid", f"ask")
        pb = bid - (lvl - 1) * tick
        pa = ask + (lvl - 1) * tick
        qb = base * np.exp(0.5 * s) * rng.lognormal(0, 0.4, n) * (1 + 0.1 * lvl)
        qa = base * np.exp(-0.5 * s) * rng.lognormal(0, 0.4, n) * (1 + 0.1 * lvl)
        if lvl == 1:
            cols["best_bid_price"], cols["best_bid_qty"], cols["best_ask_price"], cols["best_ask_qty"] = pb, qb, pa, qa
        else:
            cols[f"bid_price_{lvl}"], cols[f"bid_qty_{lvl}"], cols[f"ask_price_{lvl}"], cols[f"ask_qty_{lvl}"] = pb, qb, pa, qa
    is_trade = cols["event_type"] == "trade"
    p_buy = 1 / (1 + np.exp(-1.5 * s))
    buyer_maker = rng.random(n) > p_buy          # False => buyer aggressor
    qty = rng.lognormal(-1.0, 0.8, n)
    tid = np.cumsum(is_trade)
    nan = np.full(n, np.nan)
    cols["agg_trade_id"] = np.where(is_trade, tid, nan)
    cols["price"] = np.where(is_trade, np.where(buyer_maker, bid, ask), nan)
    cols["quantity"] = np.where(is_trade, qty, nan)
    cols["agg__qty"] = np.where(is_trade, qty, nan)
    cols["first_trade_id"] = cols["agg_trade_id"]
    cols["last_trade_id"] = cols["agg_trade_id"]
    cols["transact_time"] = np.where(is_trade, np.arange(n).astype(float), nan)
    cols["is_buyer_maker"] = np.where(is_trade, buyer_maker.astype(float), nan)
    cols["percentage"] = nan
    cols["depth"] = nan
    cols["notional"] = np.where(is_trade, qty * cols["price"], nan)
    cols["mid_price"] = (bid + ask) / 2
    df = pd.DataFrame(cols)
    _CACHE[key] = df
    _CACHE[("truth",)+key] = reg
    return df


def load_L2(file_limit: int = 1, start: int = 0, **_) -> pd.DataFrame:
    """Return ``file_limit`` consecutive 'files' starting at file index ``start``."""
    n_total = _WORLD["total_files"]
    if start + file_limit > n_total:
        raise ValueError(f"requested files [{start}, {start + file_limit}) but the synthetic world has {n_total}; "
                         f"call set_world(..., total_files=...) with a larger value")
    master = _master(n_total)
    out = master.iloc[start * ROWS_PER_FILE:(start + file_limit) * ROWS_PER_FILE].copy()
    key = (_WORLD["name"], _WORLD["seed"], _WORLD["mean_dwell"], _WORLD["beta"], ROWS_PER_FILE, n_total)
    out["regime_truth"] = _CACHE[("truth",) + key][start * ROWS_PER_FILE:(start + file_limit) * ROWS_PER_FILE]
    return out