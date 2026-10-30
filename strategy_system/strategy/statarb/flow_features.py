"""
flow_features.py -- feature engineering on the bar frames from binance_vision.load_bars().

Every feature at bar t uses information up to the END of bar t only (causal).

What each family is, and what it is NOT
---------------------------------------
* Trade flow (aggTrades). The exchange flags the aggressor, so no Lee-Ready tick test is needed.
    tfi        signed-volume ratio in [-1,1]               (order imbalance, Chordia-Roll-Subrahmanyam 2002)
    net_flow   signed notional / rolling avg notional      (scale-free net taker volume)
    cvd_s/_l   rolling sum of net_flow                     (cumulative volume delta over a window)
    cvd        full cumulative sum of net_flow             (I(1): the only flow quantity that can cointegrate with a price LEVEL)
  These are NOT the Cont-Kukanov-Stoikov OFI. They never see the book.
* True L1 OFI (needs bookTicker, `with_ticker=True`): ofi / ofi_s / ofi_l / ofi_cum, from
  Cont, Kukanov & Stoikov (2014), normalised by average top-of-book size. Only built if bt_* columns exist.
* Depth-band pressure (bookDepth): bid/ask notional inside +-k% of mid, cumulative bands, snapshots.
    di{k}, depth_flow{k}, ... are coarse book-pressure proxies, not queue-level OFI and not
    Xu-Gould-Howison multi-level OFI (their levels are price ticks, ours are % bands around a moving mid).
* Heuristics (no paper behind them): absorb, large_share, depth_slope.

Stationarity matters when a feature goes into a LEVEL regression (log price ~ ... ):
INTEGRATED_FEATURES lists the I(1) ones; everything else is treated as I(0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INTEGRATED_FEATURES = frozenset({"cvd", "ofi_cum"})   # cumulative sums -> unit-root processes


@dataclass
class FeatureConfig:
    w_short: int = 12            # short horizon, bars
    w_long: int = 96             # long horizon, bars
    z_window: int = 288          # normalisation window, bars
    bands: tuple[int, ...] = (1, 5)  # depth bands (% from mid) turned into features; first = nearest


def warmup_bars(cfg: FeatureConfig) -> int:
    return max(cfg.z_window, cfg.w_long)


def _r(s: pd.Series, w: int) -> pd.core.window.rolling.Rolling:
    return s.rolling(w, min_periods=max(2, w // 4))


def features(bars: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    f = pd.DataFrame(index=bars.index)

    # ---- price / volatility ------------------------------------------------------------
    f["logp"] = np.log(bars["close"])
    f["ret"] = f["logp"].diff()
    f["vol"] = f["ret"].rolling(cfg.w_long, min_periods=cfg.w_short).std()
    # NOTE: tick-to-tick returns include bid-ask bounce, so this is biased UP as a variance estimate
    # (the usual remedy is sparse sampling, Andersen et al. 2001 / Hansen-Lunde 2006). Not used in the spread search.
    f["rv_tick"] = np.sqrt(bars["rv"])
    f["range"] = np.log(bars["high"] / bars["low"])
    f["parkinson"] = f["range"] / np.sqrt(4 * np.log(2))     # Parkinson (1980) per-bar sigma

    # ---- trade-based order flow (aggressor side) -----------------------------------------
    notional = bars["notional"]
    buy_n = bars["buy_notional"]
    sell_n = notional - buy_n
    n_tr = bars["n_trades"]                                   # number of aggTrade rows (aggregated fills), not raw fills
    avg_notional = _r(notional, cfg.z_window).mean().where(lambda s: s > 0)

    f["tfi"] = ((buy_n - sell_n) / notional.where(notional > 0)).fillna(0.0)             # [-1, 1]
    f["tfi_cnt"] = ((2 * bars["n_buy"] - n_tr) / n_tr.where(n_tr > 0)).fillna(0.0)       # by aggTrade count
    f["net_flow"] = (buy_n - sell_n) / avg_notional                                      # signed notional, scale-free
    f["tfi_s"] = _r(f["tfi"], cfg.w_short).mean()
    f["tfi_l"] = _r(f["tfi"], cfg.w_long).mean()
    f["cvd_s"] = _r(f["net_flow"], cfg.w_short).sum()
    f["cvd_l"] = _r(f["net_flow"], cfg.w_long).sum()
    f["cvd"] = f["net_flow"].fillna(0.0).cumsum()                                        # I(1); level is arbitrary (absorbed by the intercept)

    # ---- true L1 OFI (Cont-Kukanov-Stoikov 2014), only with bookTicker ------------------------
    if {"bt_ofi", "bt_bq", "bt_aq", "bt_bid", "bt_ask"} <= set(bars.columns):
        ad = ((bars["bt_bq"] + bars["bt_aq"]) / 2)
        ad_avg = _r(ad, cfg.z_window).mean().where(lambda s: s > 0)  # average top-of-book size (CKS's AD)
        f["ofi"] = bars["bt_ofi"] / ad_avg
        f["ofi_s"] = _r(f["ofi"], cfg.w_short).sum()
        f["ofi_l"] = _r(f["ofi"], cfg.w_long).sum()
        f["ofi_cum"] = f["ofi"].fillna(0.0).cumsum()                                     # I(1)
        f["qi1"] = (bars["bt_bq"] - bars["bt_aq"]) / (bars["bt_bq"] + bars["bt_aq"])     # L1 queue imbalance
        f["rel_spread"] = (bars["bt_ask"] - bars["bt_bid"]) / ((bars["bt_ask"] + bars["bt_bid"]) / 2)

    # ---- activity / size -------------------------------------------------------------------
    lognot = np.log1p(notional)
    f["notional_z"] = (lognot - _r(lognot, cfg.z_window).mean()) / _r(lognot, cfg.z_window).std()
    avg_trade = (notional / n_tr.where(n_tr > 0))
    f["trade_size_ratio"] = np.log(avg_trade / _r(avg_trade, cfg.z_window).mean())
    f["large_share"] = (bars["max_trade"] / notional.where(notional > 0)).fillna(0.0)    # heuristic: biggest print / bar notional

    # ---- price impact / liquidity ----------------------------------------------------------
    # Kyle-lambda style: rolling slope of return on signed flow (cov/var), as in Hasbrouck (2009)-type regressions.
    f["kyle_lambda"] = f["ret"].rolling(cfg.w_long, min_periods=cfg.w_short).cov(f["net_flow"]) / \
        f["net_flow"].rolling(cfg.w_long, min_periods=cfg.w_short).var()
    rel_act = (notional / avg_notional).where(lambda s: s > 0)
    # Amihud (2002) is |r|/dollar volume AVERAGED over a window; this is the per-bar term, normalised by relative activity.
    f["illiq"] = f["ret"].abs() / rel_act

    # ---- depth-band book pressure (needs bookDepth) -------------------------------------------
    for k in cfg.bands:
        b, a = bars.get(f"bid_n{k}"), bars.get(f"ask_n{k}")
        if b is None or a is None:
            continue
        tot = b + a
        f[f"di{k}"] = (b - a) / tot                          # depth imbalance in [-1, 1]
        f[f"ddi{k}"] = f[f"di{k}"].diff()
        f[f"liq{k}"] = np.log(tot)
        f[f"dliq{k}"] = f[f"liq{k}"].diff()
        # band content also changes mechanically when mid moves (levels cross the band edge), so this is a proxy only
        f[f"depth_flow{k}"] = (b.diff() - a.diff()) / tot.shift(1)

    near, far = cfg.bands[0], cfg.bands[-1]
    if f"liq{near}" in f and f"liq{far}" in f and near != far:
        f["depth_slope"] = f[f"liq{far}"] - f[f"liq{near}"]         # bands are cumulative, so this is log(n_far/n_near) >= 0
    if f"di{near}" in f:
        f["absorb"] = f["tfi"] * (-f[f"di{near}"])                 # heuristic: >0 when aggressive flow meets opposing resting depth

    return f.replace([np.inf, -np.inf], np.nan)