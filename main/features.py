"""
Live Feature Pipeline (1-Second Resolution) — Optimized Feature Set
====================================================================
Real-time feature computation mirroring the research pipeline in ml_state_optimized.py.

Architecture (matches research exactly):
    1. Ingest raw trades → aggregate into 1-second micro bars
    2. 1-minute macro features (RSI, MA slope, spread, etc.) updated every minute
    3. Micro features (price_roc_3) updated EVERY SECOND
    4. Join micro + macro → full 7-feature vector
    5. Run inference EVERY SECOND

Research reference: ml_state_optimized.py lines 386-439
    Features (by PFI ranking):
        spread_bps_z  (0.081) — DeepLOB spread z-score
        ma_slope_5    (0.024) — MA velocity
        rsi_x_ma      (0.006) — RSI × MA slope interaction
        ma_accel      (0.005) — MA acceleration
        price_roc_3   (0.005) — Price rate of change
        rsi_2         (filter) — RSI(2)
        sma_14        (context) — SMA of RSI(14)
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Optional
from utils.indicators import get_wilders_rsi, sma, calculate_zscore

import sys
sys.path.append('.')

from main.config import *


EPS = 1e-9


class LiveFeatureEngine:
    """
    Streaming feature computation engine at 1-SECOND resolution.
    Computes the 7 optimized features from ml_state_optimized.py.

    Each call to on_trade() updates micro-bar data per-second.
    Calling compute_1s_feature_vector() produces the full 7-feature vector
    for the current 1-second bar.

    Usage:
        engine = LiveFeatureEngine()
        engine.on_trade(timestamp, price, amount, side)   # each trade
        engine.on_orderbook(bid_px, bid_amt, ask_px, ask_amt)  # each OB update

        # EVERY SECOND (on the 1-second clock tick):
        features = engine.compute_1s_feature_vector()
        # → dict with all 7 features, ready for model.predict_proba()
    """

    def __init__(self):
        # --- 1-second micro bar state ---
        self._current_second: Optional[int] = None
        self._second_open: Optional[float] = None
        self._second_high: Optional[float] = None
        self._second_low: Optional[float] = None
        self._second_close: Optional[float] = None
        self._second_buy_vol: float = 0.0
        self._second_sell_vol: float = 0.0

        # --- Rolling buffers for micro features (updated every second) ---
        # 1-second prices (for price_roc_3 and price diffs)
        self._price_buffer = RingBuffer(300)

        # --- 1-minute macro state (OHLC aggregates) ---
        self._minute_open: Optional[float] = None
        self._minute_high: Optional[float] = None
        self._minute_low: Optional[float] = None
        self._minute_close: Optional[float] = None
        self._minute_buy_vol: float = 0.0
        self._minute_sell_vol: float = 0.0
        self._minute_volume: float = 0.0
        self._minute_largest_buy: float = 0.0
        self._minute_largest_sell: float = 0.0

        # --- Current orderbook snapshot (latest) ---
        self._last_bid_price: float = 0.0
        self._last_bid_amount: float = 0.0
        self._last_ask_price: float = 0.0
        self._last_ask_amount: float = 0.0
        self._last_mid_price: float = 0.0

        # --- Computed 1-minute features (cached, updated on minute rollover) ---
        self._macro_features: Dict[str, float] = {
            'rsi_2': 50.0,
            'sma_14': 50.0,
            'ma_slope_5': 0.0,
            'ma_accel': 0.0,
            'spread_bps_z': 0.0,
            'rsi_x_ma': 0.0,
        }

        # --- Buffers for macro feature computation ---
        self._close_prices = RingBuffer(200)       # For MA slope, price_roc
        self._price_diffs = RingBuffer(300)         # For RSI computation
        self._rsi_2_buffer = RingBuffer(50)         # For rsi_z computation
        self._spread_bps_buffer = RingBuffer(50)    # For spread_bps_z computation

        # Track which minute we're currently accumulating
        self._current_minute_key: Optional[int] = None
        self._last_macro_update_minute: int = -1

        # --- Bar counter ---
        self._second_count = 0

    def on_trade(self, timestamp_ms: int, price: float, amount: float, side: str) -> None:
        """Ingest a single trade. Updates both micro (1s) and macro (1min) state."""
        second_key = timestamp_ms // 1000

        # On second rollover, flush the completed second's data
        if self._current_second is not None and second_key != self._current_second:
            self._flush_second()

        # Update 1-second micro bar
        self._current_second = second_key
        if self._second_open is None:
            self._second_open = price
        self._second_high = price if self._second_high is None else max(self._second_high, price)
        self._second_low = price if self._second_low is None else min(self._second_low, price)
        self._second_close = price
        if side == 'buy':
            self._second_buy_vol += amount
        else:
            self._second_sell_vol += amount

        # Also update 1-minute macro aggregates
        if self._minute_open is None:
            self._minute_open = price
        self._minute_high = price if self._minute_high is None else max(self._minute_high, price)
        self._minute_low = price if self._minute_low is None else min(self._minute_low, price)
        self._minute_close = price
        self._minute_volume += amount
        if side == 'buy':
            self._minute_buy_vol += amount
            self._minute_largest_buy = max(self._minute_largest_buy, amount)
        else:
            self._minute_sell_vol += amount
            self._minute_largest_sell = max(self._minute_largest_sell, amount)

    def on_orderbook(self, timestamp_ms: int, bid_price: float, bid_amount: float,
                     ask_price: float, ask_amount: float) -> None:
        """Ingest an orderbook snapshot."""
        self._last_bid_price = bid_price
        self._last_bid_amount = bid_amount
        self._last_ask_price = ask_price
        self._last_ask_amount = ask_amount
        self._last_mid_price = (bid_price + ask_price) / 2

    def _flush_second(self) -> None:
        """
        Called when a new second starts. Flushes the completed second's data
        into the rolling buffers.
        """
        close = self._second_close if self._second_close is not None else self._last_mid_price

        # Store price for diff computation
        self._price_buffer.append(close)

        self._second_count += 1

        # Check if we need to compute new macro features (minute rollover)
        minute_key = int(self._current_second // 60) if self._current_second else 0
        if minute_key != self._last_macro_update_minute and self._minute_close is not None:
            self._compute_macro_features()
            # Reset minute accumulators for next minute
            self._reset_minute()

        # Reset second
        self._second_open = None
        self._second_high = None
        self._second_low = None
        self._second_close = None
        self._second_buy_vol = 0.0
        self._second_sell_vol = 0.0

    def _compute_macro_features(self) -> None:
        """
        Compute the 1-minute-resolution features matching ml_state_optimized.py.
        Called when the current minute rolls over.

        These are forward-filled until the next minute update — exactly
        matching research: df_micro.join(df_macro, how='left').ffill()
        """
        close = self._minute_close if self._minute_close is not None else self._last_mid_price
        if close <= 0:
            return

        open_ = self._minute_open if self._minute_open is not None else close
        high = self._minute_high if self._minute_high is not None else close
        low = self._minute_low if self._minute_low is not None else close

        # --- Price diff for RSI ---
        if self._close_prices._count > 0:
            prev_close = self._close_prices.values[-1]
            delta = close - prev_close
        else:
            delta = 0.0

        self._price_diffs.append(delta)
        self._close_prices.append(close)

        # ================================================================
        # RSI features (ml_state_optimized.py lines 397-400)
        # ================================================================
        diffs = self._price_diffs.values
        rsi_2_arr = get_wilders_rsi(diffs, RSI_2_WINDOW) if len(diffs) >= RSI_2_WINDOW + 1 else np.array([np.nan])
        rsi_14_arr = get_wilders_rsi(diffs, RSI_14_WINDOW) if len(diffs) >= RSI_14_WINDOW + 1 else np.array([np.nan])

        rsi_2_val = float(rsi_2_arr[-1]) if len(rsi_2_arr) > 0 else 50.0
        rsi_14_val = float(rsi_14_arr[-1]) if len(rsi_14_arr) > 0 else 50.0
        sma_14_val = float(np.nanmean(rsi_14_arr[-SMA_14_WINDOW:])) if len(rsi_14_arr) >= SMA_14_WINDOW else 50.0

        # Store rsi_2 for z-score computation
        self._rsi_2_buffer.append(rsi_2_val)

        # ================================================================
        # MA Slope (velocity) — ml_state_optimized.py lines 404-406
        # ma_5 = np.convolve(close_arr, np.ones(5)/5, mode='same')
        # ma_slope_5 = np.diff(ma_5)
        # ================================================================
        close_arr = self._close_prices.values
        if len(close_arr) >= MA_SLOPE_WINDOW + 1:
            # Compute 5-period SMA
            ma_5 = np.convolve(close_arr, np.ones(MA_SLOPE_WINDOW) / MA_SLOPE_WINDOW, mode='valid')
            # MA slope = difference of MA
            ma_slope_5 = float(np.diff(ma_5)[-1]) if len(ma_5) >= 2 else 0.0
        else:
            ma_slope_5 = 0.0

        # ================================================================
        # MA Acceleration (2nd derivative) — ml_state_optimized.py lines 409-412
        # ma_accel = np.diff(ma_slope)
        # ================================================================
        if len(close_arr) >= MA_SLOPE_WINDOW + 2:
            ma_5 = np.convolve(close_arr, np.ones(MA_SLOPE_WINDOW) / MA_SLOPE_WINDOW, mode='valid')
            ma_slope_arr = np.diff(ma_5)
            ma_accel = float(np.diff(ma_slope_arr)[-1]) if len(ma_slope_arr) >= 2 else 0.0
        else:
            ma_accel = 0.0

        # ================================================================
        # DeepLOB: Spread in bps, z-scored — ml_state_optimized.py lines 430-431
        # spread_bps_z = calculate_zscore(depth_1m['spread_bps'], window=10)
        # ================================================================
        if self._last_bid_price > 0 and self._last_ask_price > 0:
            mid = (self._last_bid_price + self._last_ask_price) / 2
            spread_bps = (self._last_ask_price - self._last_bid_price) / mid * 10000.0
        else:
            spread_bps = 0.0

        self._spread_bps_buffer.append(spread_bps)

        # Rolling z-score of spread_bps (window=10)
        spread_vals = self._spread_bps_buffer.values
        if len(spread_vals) >= DEEPLOB_Z_WINDOW:
            spread_mean = np.nanmean(spread_vals[-DEEPLOB_Z_WINDOW:])
            spread_std = np.nanstd(spread_vals[-DEEPLOB_Z_WINDOW:])
            spread_bps_z = (spread_bps - spread_mean) / (spread_std + EPS)
        else:
            spread_bps_z = 0.0

        # ================================================================
        # RSI × MA slope interaction — ml_state_optimized.py lines 437-439
        # rsi_z = calculate_zscore(rsi_2, window=20)
        # rsi_x_ma = rsi_z * ma_slope_5
        # ================================================================
        rsi_vals = self._rsi_2_buffer.values
        if len(rsi_vals) >= RSI_Z_WINDOW:
            rsi_mean = np.nanmean(rsi_vals[-RSI_Z_WINDOW:])
            rsi_std = np.nanstd(rsi_vals[-RSI_Z_WINDOW:])
            rsi_z = (rsi_2_val - rsi_mean) / (rsi_std + EPS)
        else:
            rsi_z = 0.0

        rsi_x_ma = rsi_z * ma_slope_5

        # Store macro features (these will be forward-filled across 1s bars)
        self._macro_features = {
            'rsi_2': rsi_2_val,
            'sma_14': sma_14_val,
            'ma_slope_5': ma_slope_5,
            'ma_accel': ma_accel,
            'spread_bps_z': spread_bps_z,
            'rsi_x_ma': rsi_x_ma,
        }

        self._last_macro_update_minute = int((self._current_second or 0) // 60)
        self._macro_features['_close'] = close
        self._macro_features['_high'] = high
        self._macro_features['_low'] = low

    def _reset_minute(self) -> None:
        """Reset all per-minute accumulators."""
        self._minute_open = None
        self._minute_high = None
        self._minute_low = None
        self._minute_close = None
        self._minute_buy_vol = 0.0
        self._minute_sell_vol = 0.0
        self._minute_volume = 0.0
        self._minute_largest_buy = 0.0
        self._minute_largest_sell = 0.0

    def compute_1s_feature_vector(self) -> Dict[str, float]:
        """
        Compute the full 7-feature vector for the CURRENT 1-second bar.
        Called every second (on the 1-second clock tick).

        Uses last known price if current second has no trades.
        Macro features forward-filled from last minute update.

        Returns dict with 7 features + 'close' price, or {} if no data at all.
        """
        # Use current second's close, or fallback to last mid price, or macro close
        close = self._second_close
        if close is None or close == 0.0:
            close = self._last_mid_price
        if close is None or close == 0.0:
            close = self._macro_features.get('_close', 0.0)
        if close is None or close == 0.0:
            return {}  # No data at all yet

        # ================================================================
        # Micro features (updated every second)
        # ================================================================

        # price_roc_3 — ml_state_optimized.py line 415
        # df_price_1m['price_roc_3'] = df_price_1m['close'].pct_change(3)
        price_vals = self._price_buffer.values
        if len(price_vals) >= PRICE_ROC_WINDOW_3 + 1:
            price_roc_3 = (close - price_vals[-PRICE_ROC_WINDOW_3]) / (price_vals[-PRICE_ROC_WINDOW_3] + EPS)
        else:
            price_roc_3 = 0.0

        # ================================================================
        # Build full 7-feature vector (matching ml_state_optimized.py)
        # ================================================================
        features = {
            'spread_bps_z': self._macro_features.get('spread_bps_z', 0.0),
            'ma_slope_5': self._macro_features.get('ma_slope_5', 0.0),
            'rsi_x_ma': self._macro_features.get('rsi_x_ma', 0.0),
            'ma_accel': self._macro_features.get('ma_accel', 0.0),
            'price_roc_3': price_roc_3,
            'rsi_2': self._macro_features.get('rsi_2', 50.0),
            'sma_14': self._macro_features.get('sma_14', 50.0),
            'close': close,
        }

        return features

    @property
    def current_mid_price(self) -> float:
        return self._last_mid_price

    def get_feature_columns(self) -> list:
        """Return the list of feature column names in order."""
        try:
            from main.config import FEATURE_COLUMNS
            return FEATURE_COLUMNS
        except (ImportError, AttributeError):
            return [
                'spread_bps_z',
                'ma_slope_5',
                'rsi_x_ma',
                'ma_accel',
                'price_roc_3',
                'rsi_2',
                'sma_14',
            ]

    def extract_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to model-ready numpy array (1D)."""
        cols = self.get_feature_columns()
        vec = np.array([features.get(c, 0.0) for c in cols], dtype=float)
        return vec.reshape(1, -1)


class RingBuffer:
    """
    Fixed-size NumPy ring buffer for incremental stat computation.
    Stores values and provides rolling mean/std on demand.
    """
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._data = np.full(maxlen, np.nan)
        self._ptr = 0
        self._count = 0

    def append(self, value: float) -> None:
        self._data[self._ptr] = value
        self._ptr = (self._ptr + 1) % self.maxlen
        if self._count < self.maxlen:
            self._count += 1

    @property
    def values(self) -> np.ndarray:
        if self._count < self.maxlen:
            return self._data[:self._count].copy()
        return np.concatenate([
            self._data[self._ptr:],
            self._data[:self._ptr],
        ])

    @property
    def mean(self) -> float:
        return np.nanmean(self.values)

    @property
    def std(self) -> float:
        return np.nanstd(self.values)

    @property
    def full(self) -> bool:
        return self._count >= self.maxlen