"""
Regime-switching random walk market generator.

The true bias of the random walk is a hidden Markov chain that switches
between negative / neutral / positive regimes. The agent never sees the true
bias directly. It only sees NOISY, LAGGED estimates (EMAs, volatility,
momentum) computed from the price path.

This is the crux of the study: the variables are informative but never
perfect, because the market is non-stationary. By the time the variables
confirm a regime shift, the regime may already be shifting back.
"""

from typing import List, Optional, Tuple

import numpy as np

from config import MarketConfig


class RegimeSwitchingMarket:
    """Generates a regime-switching random walk and its observable variables."""

    def __init__(self, cfg: MarketConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _ema(self, x: np.ndarray, span: int) -> np.ndarray:
        """Exponential moving average (pandas-style span)."""
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out

    def generate(self) -> dict:
        """
        Generate one episode of market data.

        Returns a dict with:
          - returns:      per-step raw returns (noise + regime bias)
          - price:        cumulative price path
          - true_bias:    the hidden true per-step bias (for analysis only)
          - regime:       the hidden true regime index per step
          - obs:          (T, n_vars) array of observable variables
          - obs_names:    names of the observable variables
        """
        cfg = self.cfg
        T = cfg.n_steps
        n_regimes = len(cfg.regime_biases)

        # --- Hidden Markov regime chain -------------------------------------
        regime = np.empty(T, dtype=int)
        regime[0] = cfg.start_regime
        for t in range(1, T):
            stay = self.rng.random() < cfg.stay_probs[regime[t - 1]]
            if stay:
                regime[t] = regime[t - 1]
            else:
                # Move to a different regime uniformly.
                options = [r for r in range(n_regimes) if r != regime[t - 1]]
                regime[t] = self.rng.choice(options)

        true_bias = np.array([cfg.regime_biases[r] for r in regime])

        # --- Returns: bias + noise ------------------------------------------
        noise = self.rng.normal(0.0, cfg.base_vol, T)
        returns = true_bias + noise
        price = np.exp(np.cumsum(returns))

        # --- Observable variables (noisy, lagged estimates of bias) ---------
        # These are the "variables that may or may not affect random walk
        # bias". They are computed from the price path, so they lag the true
        # regime and are contaminated by noise.
        ema_short = self._ema(returns, cfg.ema_short_span)
        ema_long = self._ema(returns, cfg.ema_long_span)

        # Rolling volatility (std of returns over window).
        vol = np.empty(T)
        for t in range(T):
            lo = max(0, t - cfg.vol_span + 1)
            vol[t] = np.std(returns[lo : t + 1])

        # Momentum: cumulative return over the momentum window.
        momentum = np.empty(T)
        for t in range(T):
            lo = max(0, t - cfg.momentum_span + 1)
            momentum[t] = np.sum(returns[lo : t + 1])

        # Normalize variables to roughly [-1, 1] / [0, 1] scale for the
        # discretizer. Use the base_vol as the natural scale.
        bv = cfg.base_vol
        obs = np.stack(
            [
                ema_short / bv,          # ~[-1, 1]
                ema_long / bv,           # ~[-1, 1]
                vol / bv,                # ~[0, 2]
                momentum / (bv * np.sqrt(cfg.momentum_span)),  # ~[-1, 1]
            ],
            axis=1,
        )
        obs_names = ["ema_short", "ema_long", "vol", "momentum"]

        return {
            "returns": returns,
            "price": price,
            "true_bias": true_bias,
            "regime": regime,
            "obs": obs,
            "obs_names": obs_names,
        }