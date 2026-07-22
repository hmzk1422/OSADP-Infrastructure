"""
Live Feature Pipeline (1-Second Resolution)
=============================================
Real-time feature computation mirroring the research pipeline in ml-state.ipynb.

Architecture (matches research exactly):
    1. Ingest raw trades → aggregate into 1-second micro bars
    2. 1-minute macro features (RSI, candle, etc.) updated every 60 trades
    3. Micro features (OFI, CVD) updated EVERY SECOND
    4. Join micro + macro → full 12-feature vector
    5. Run inference EVERY SECOND

Research reference: ml-state.ipynb lines 80-142
    df_micro = df_trades.resample('1s')       → price, micro_OFI, micro_CVD
    df_macro = df_ob_1m (resample '1min')     → RSI, candle, OFI_LOB, etc.
    df_state_space = df_micro.join(df_macro, how='left').ffill()
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Optional
from utils.indicators import get_wilders_rsi, sma, calculate_cvd

import sys
sys.path.append('.')

from main.config import *


EPS = 1e-9


class LiveFeatureEngine:
    """
    Streaming feature computation engine at 1-SECOND resolution.

    Each call to on_trade() updates micro-bar data per-second.
    Calling compute_1s_feature_vector() produces the full 12-feature vector
    for the current 1-second bar.

    Usage:
        engine = LiveFeatureEngine()
        engine.on_trade(timestamp, price, amount, side)   # each trade
        engine.on_orderbook(bid_px, bid_amt, ask_px, ask_amt)  # each OB update

        # EVERY SECOND (on the 1-second clock tick):
        features = engine.compute_1s_feature_vector()
        # → dict with all 12 features, ready for model.predict_proba()
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
        # 1-second micro OFI values (last 60 seconds)
        self._ofi_buffer = RingBuffer(60)
        # 1-second prices (for micro_CVD and price diffs)
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
        self._minute_ob_bid_sum: float = 0.0
        self._minute_ob_ask_sum: float = 0.0
        self._minute_ob_bid_count: int = 0
        self._minute_ob_ask_count: int = 0

        # --- Current orderbook snapshot (latest) ---
        self._last_bid_price: float = 0.0
        self._last_bid_amount: float = 0.0
        self._last_ask_price: float = 0.0
        self._last_ask_amount: float = 0.0
        self._last_mid_price: float = 0.0

        # --- Computed 1-minute features (cached, updated on minute rollover) ---
        self._macro_features: Dict[str, float] = {
            'rsi_2': 50.0,
            'rsi_14': 50.0,
            'sma_14': 50.0,
            'largest_buyer': 0.0,
            'largest_seller': 0.0,
            'OFI_LOB': 0.0,
            'OFI_AGG': 0.0,
            'CVD_z': 0.0,
            'candle_type': 0.5,
            'candle_size': 1.0,
        }

        # --- Buffers for macro feature computation ---
        self._close_prices = RingBuffer(200)
        self._price_diffs = RingBuffer(300)
        self._cvd_deltas = RingBuffer(OFI_LOOKBACK)
        self._candle_sizes = RingBuffer(14)

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

        self._minute_ob_bid_sum += bid_amount
        self._minute_ob_ask_sum += ask_amount
        self._minute_ob_bid_count += 1
        self._minute_ob_ask_count += 1

    def _flush_second(self) -> None:
        """
        Called when a new second starts. Flushes the completed second's data
        into the rolling buffers.
        """
        close = self._second_close if self._second_close is not None else self._last_mid_price
        net_OFI = self._second_buy_vol - self._second_sell_vol

        # Store price for diff computation
        self._price_buffer.append(close)
        self._ofi_buffer.append(net_OFI)

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
        Compute the 1-minute-resolution features (RSI, candle, orderflow).
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

        # --- RSI features ---
        diffs = self._price_diffs.values
        rsi_2_arr = get_wilders_rsi(diffs, RSI_2_WINDOW) if len(diffs) >= RSI_2_WINDOW + 1 else np.array([np.nan])
        rsi_14_arr = get_wilders_rsi(diffs, RSI_14_WINDOW) if len(diffs) >= RSI_14_WINDOW + 1 else np.array([np.nan])

        rsi_2_val = float(rsi_2_arr[-1]) if len(rsi_2_arr) > 0 else 50.0
        rsi_14_val = float(rsi_14_arr[-1]) if len(rsi_14_arr) > 0 else 50.0
        sma_14_val = float(np.nanmean(rsi_14_arr[-SMA_14_WINDOW:])) if len(rsi_14_arr) >= SMA_14_WINDOW else 50.0

        # --- Candle features ---
        candle_range = high - low if high > low else EPS
        candle_type = abs(close - open_) / candle_range
        self._candle_sizes.append(candle_range)
        candle_size = candle_range / (self._candle_sizes.mean + EPS) if self._candle_sizes.full else 1.0

        # --- Aggregated trade features ---
        total_vol = self._minute_volume if self._minute_volume > 0 else EPS
        largest_buyer_pct = (self._minute_largest_buy * 100.0) / total_vol
        largest_seller_pct = (self._minute_largest_sell * 100.0) / total_vol

        # --- Orderbook imbalance ---
        ob_bid_avg = self._minute_ob_bid_sum / max(self._minute_ob_bid_count, 1)
        ob_ask_avg = self._minute_ob_ask_sum / max(self._minute_ob_ask_count, 1)
        OFI_LOB = 100.0 * (ob_bid_avg - ob_ask_avg) / (ob_bid_avg + ob_ask_avg + EPS)

        # --- Aggregated trade imbalance ---
        agg_buyer = self._minute_buy_vol
        agg_seller = self._minute_sell_vol
        OFI_AGG = 100.0 * (agg_buyer - agg_seller) / (agg_buyer + agg_seller + EPS)

        # --- CVD Z-score ---
        cvd_raw = agg_buyer - agg_seller
        cvd_delta = cvd_raw - (self._cvd_deltas.values[-1] if self._cvd_deltas._count > 0 else cvd_raw)
        self._cvd_deltas.append(cvd_delta)
        CVD_z = (cvd_delta - self._cvd_deltas.mean) / (self._cvd_deltas.std + EPS) if self._cvd_deltas.full else 0.0

        # Store macro features (these will be forward-filled across 1s bars)
        self._macro_features = {
            'rsi_2': rsi_2_val,
            'rsi_14': rsi_14_val,
            'sma_14': sma_14_val,
            'largest_buyer': largest_buyer_pct,
            'largest_seller': largest_seller_pct,
            'OFI_LOB': OFI_LOB,
            'OFI_AGG': OFI_AGG,
            'CVD_z': CVD_z,
            'candle_type': candle_type,
            'candle_size': candle_size,
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
        self._minute_ob_bid_sum = 0.0
        self._minute_ob_ask_sum = 0.0
        self._minute_ob_bid_count = 0
        self._minute_ob_ask_count = 0

    def compute_1s_feature_vector(self) -> Dict[str, float]:
        """
        Compute the full 12-feature vector for the CURRENT 1-second bar.
        Called every second (on the 1-second clock tick).

        Uses last known price if current second has no trades.
        Micro features read from rolling buffers (may lag by 1 second).
        Macro features forward-filled from last minute update.

        Returns dict with 12 features + 'close' price, or {} if no data at all.
        """
        # Use current second's close, or fallback to last mid price, or macro close
        close = self._second_close
        if close is None or close == 0.0:
            close = self._last_mid_price
        if close is None or close == 0.0:
            close = self._macro_features.get('_close', 0.0)
        if close is None or close == 0.0:
            return {}  # No data at all yet

        # --- Micro features (1-second resolution) ---
        # micro_OFI = buy_vol - sell_vol for this second
        micro_OFI = self._second_buy_vol - self._second_sell_vol

        # micro_CVD = sum of last 10 seconds of micro_OFI (research: rolling(10).sum())
        if self._ofi_buffer._count >= CVD_ROLLING:
            micro_CVD = float(np.nansum(self._ofi_buffer.values[-CVD_ROLLING:]))
        else:
            micro_CVD = float(micro_OFI)

        # --- Build full feature vector ---
        # Micro features (update every second)
        # Macro features (forward-filled from last minute update, matching research .ffill())
        features = {
            'micro_OFI': micro_OFI,
            'micro_CVD': micro_CVD,
            'rsi_2': self._macro_features.get('rsi_2', 50.0),
            'rsi_14': self._macro_features.get('rsi_14', 50.0),
            'sma_14': self._macro_features.get('sma_14', 50.0),
            'largest_buyer': self._macro_features.get('largest_buyer', 0.0),
            'largest_seller': self._macro_features.get('largest_seller', 0.0),
            'OFI_LOB': self._macro_features.get('OFI_LOB', 0.0),
            'OFI_AGG': self._macro_features.get('OFI_AGG', 0.0),
            'CVD_z': self._macro_features.get('CVD_z', 0.0),
            'candle_type': self._macro_features.get('candle_type', 0.5),
            'candle_size': self._macro_features.get('candle_size', 1.0),
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
                'micro_OFI', 'micro_CVD',
                'rsi_2', 'rsi_14', 'sma_14',
                'largest_buyer', 'largest_seller',
                'OFI_LOB', 'OFI_AGG', 'CVD_z',
                'candle_type', 'candle_size',
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