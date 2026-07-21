"""
Live Feature Pipeline
======================
Real-time feature computation mirroring the research pipeline in corrected_state.py.

Outputs a 1-row feature vector ready for model inference on each 1-minute bar.

Feature set (same as research):
    - price: last tick price
    - micro_OFI: Order Flow Imbalance (1s resolution)
    - micro_CVD: Cumulative Volume Delta (10s rolling)
    - rsi_2: Wilder's RSI(2) on price differences
    - rsi_14: Wilder's RSI(14) on price differences
    - sma_14: SMA of RSI(14)
    - largest_buyer / largest_seller: % of trade volume from largest trade
    - OFI_LOB: Order Book imbalance
    - OFI_AGG: Aggregated trade imbalance
    - CVD_z: Z-score of CVD change
    - candle_type: |close - open| / |high - low|
    - candle_size: (high - low) / mean(high-low, 14)
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


class RingBuffer:
    """
    Fixed-size NumPy ring buffer for incremental stat computation.
    Stores values and provides rolling mean/std on demand.
    """
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._data = np.full(maxlen, np.nan)
        self._ptr = 0          # next write index
        self._count = 0        # number of values written so far

    def append(self, value: float) -> None:
        self._data[self._ptr] = value
        self._ptr = (self._ptr + 1) % self.maxlen
        if self._count < self.maxlen:
            self._count += 1

    @property
    def values(self) -> np.ndarray:
        """Return in chronological order (oldest first)."""
        if self._count < self.maxlen:
            return self._data[:self._count].copy()
        # Rotate so that oldest is at index 0
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


class LiveFeatureEngine:
    """
    Streaming feature computation engine.
    Ingests tick-level data and produces 1-min bar feature vectors on demand.

    Usage:
        engine = LiveFeatureEngine()
        engine.on_trade(timestamp, price, amount, side)
        engine.on_orderbook(timestamp, bid_price, bid_amount, ask_price, ask_amount)
        features = engine.on_bar_close(timestamp, open_, high, low, close)
    """

    def __init__(self):
        # --- 1-second micro bar buffers ---
        self._second_bars: Dict[int, dict] = {}  # second_key -> {buy_vol, sell_vol, price}

        # --- 1-minute OHLC state ---
        self._minute_open: Optional[float] = None
        self._minute_high: Optional[float] = None
        self._minute_low: Optional[float] = None
        self._minute_close: Optional[float] = None
        self._minute_buy_vol: float = 0.0
        self._minute_sell_vol: float = 0.0
        self._minute_volume: float = 0.0
        self._minute_num_trades: int = 0
        self._minute_largest_buy: float = 0.0
        self._minute_largest_sell: float = 0.0
        self._minute_ob_bid_sum: float = 0.0
        self._minute_ob_ask_sum: float = 0.0
        self._minute_ob_bid_count: int = 0
        self._minute_ob_ask_count: int = 0

        # --- Orderbook snapshot (latest) ---
        self._last_bid_price: float = 0.0
        self._last_bid_amount: float = 0.0
        self._last_ask_price: float = 0.0
        self._last_ask_amount: float = 0.0
        self._last_mid_price: float = 0.0

        # --- Rolling buffers for features (chronological) ---
        # 1-second micro features
        self._micro_OFI_buffer = RingBuffer(60)    # last 60 sec
        self._micro_CVD_buffer = RingBuffer(CVD_ROLLING)

        # Price diffs for RSI
        self._price_diffs = RingBuffer(300)         # enough for RSI(14) + SMA(14)

        # Candle-size history for candle_size feature
        self._candle_sizes = RingBuffer(14)

        # Price for barrier vol estimation
        self._prices_1s = RingBuffer(BARRIER_VOL_WINDOW)

        # CVD change buffer for CVD_z
        self._cvd_deltas = RingBuffer(OFI_LOOKBACK)

        # Close price history for return calculation
        self._close_prices = RingBuffer(200)

        # --- Bar counter ---
        self._bar_count = 0

    def on_trade(self, timestamp_ms: int, price: float, amount: float, side: str) -> None:
        """Ingest a single trade."""
        second_key = timestamp_ms // 1000

        if second_key not in self._second_bars:
            self._second_bars[second_key] = {
                'buy_vol': 0.0,
                'sell_vol': 0.0,
                'price': price,
            }

        bar = self._second_bars[second_key]
        bar['price'] = price
        bar['buy_vol'] += amount if side == 'buy' else 0
        bar['sell_vol'] += amount if side == 'sell' else 0

        # Also accumulate into minute aggregates
        self._minute_buy_vol += amount if side == 'buy' else 0
        self._minute_sell_vol += amount if side == 'sell' else 0
        self._minute_volume += amount
        self._minute_num_trades += 1
        if side == 'buy' and amount > self._minute_largest_buy:
            self._minute_largest_buy = amount
        if side == 'sell' and amount > self._minute_largest_sell:
            self._minute_largest_sell = amount

        # Update minute OHLC
        if self._minute_open is None:
            self._minute_open = price
        self._minute_close = price
        if self._minute_high is None or price > self._minute_high:
            self._minute_high = price
        if self._minute_low is None or price < self._minute_low:
            self._minute_low = price

        # Store price for diff computation
        self._prices_1s.append(price)

    def on_orderbook(self, timestamp_ms: int, bid_price: float, bid_amount: float,
                     ask_price: float, ask_amount: float) -> None:
        """Ingest an orderbook snapshot."""
        self._last_bid_price = bid_price
        self._last_bid_amount = bid_amount
        self._last_ask_price = ask_price
        self._last_ask_amount = ask_amount
        self._last_mid_price = (bid_price + ask_price) / 2

        # Accumulate OB sums for 1-min aggregate
        self._minute_ob_bid_sum += bid_amount
        self._minute_ob_ask_sum += ask_amount
        self._minute_ob_bid_count += 1
        self._minute_ob_ask_count += 1

    def _flush_second_bars(self) -> None:
        """
        Flush completed 1-second micro bars into rolling feature buffers.
        A second is 'completed' when its key is < current_second - 1.
        """
        now_s = int(pd.Timestamp.now().timestamp() * 1000) // 1000
        completed_keys = [k for k in self._second_bars if k < now_s - 1]
        completed_keys.sort()

        for key in completed_keys:
            bar = self._second_bars.pop(key)
            net_OFI = bar['buy_vol'] - bar['sell_vol']
            self._micro_OFI_buffer.append(net_OFI)
            self._prices_1s.append(bar['price'])

        # Recompute micro_CVD from last N OFI entries
        if self._micro_OFI_buffer._count >= CVD_ROLLING:
            cvd = np.nansum(self._micro_OFI_buffer.values[-CVD_ROLLING:])
            self._micro_CVD_buffer.append(cvd)

    def compute_micro_features(self) -> Dict[str, float]:
        """Compute the 1s-resolution features (micro_OFI, micro_CVD)."""
        self._flush_second_bars()

        return {
            'micro_OFI': self._micro_OFI_buffer.values[-1] if self._micro_OFI_buffer._count > 0 else 0.0,
            'micro_CVD': self._micro_CVD_buffer.values[-1] if self._micro_CVD_buffer._count > 0 else 0.0,
        }

    def on_bar_close(self) -> Dict[str, float]:
        """
        Called when a 1-minute bar closes.
        Computes the full feature vector for this bar.

        Returns a dict matching the feature columns used in corrected_state.py.
        """
        close = self._minute_close if self._minute_close is not None else self._last_mid_price
        if close is None or close == 0.0:
            return {}  # not enough data yet

        open_ = self._minute_open if self._minute_open is not None else close
        high = self._minute_high if self._minute_high is not None else close
        low = self._minute_low if self._minute_low is not None else close

        # --- Price features ---
        if self._close_prices._count > 0:
            prev_close = self._close_prices.values[-1]
            delta = close - prev_close
        else:
            delta = 0.0

        self._price_diffs.append(delta)
        self._close_prices.append(close)

        diffs = self._price_diffs.values
        rsi_2_arr = get_wilders_rsi(diffs, RSI_2_WINDOW) if len(diffs) >= RSI_2_WINDOW + 1 else np.array([np.nan])
        rsi_14_arr = get_wilders_rsi(diffs, RSI_14_WINDOW) if len(diffs) >= RSI_14_WINDOW + 1 else np.array([np.nan])

        rsi_2_val = float(rsi_2_arr[-1]) if len(rsi_2_arr) > 0 else np.nan
        rsi_14_val = float(rsi_14_arr[-1]) if len(rsi_14_arr) > 0 else np.nan
        sma_14_val = float(np.nanmean(rsi_14_arr[-SMA_14_WINDOW:])) if len(rsi_14_arr) >= SMA_14_WINDOW else np.nan

        # --- Aggregated trade features ---
        total_vol = self._minute_volume if self._minute_volume > 0 else EPS
        largest_buyer_pct = (self._minute_largest_buy * 100.0) / total_vol
        largest_seller_pct = (self._minute_largest_sell * 100.0) / total_vol

        # --- Orderbook features ---
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
        cvd_z = (cvd_delta - self._cvd_deltas.mean) / (self._cvd_deltas.std + EPS) if self._cvd_deltas.full else 0.0

        # --- Candle type / size ---
        candle_range = high - low if high > low else EPS
        candle_type = abs(close - open_) / candle_range
        self._candle_sizes.append(candle_range)
        candle_size = candle_range / (self._candle_sizes.mean + EPS) if self._candle_sizes.full else 1.0

        # --- Build feature dict ---
        micro = self.compute_micro_features()

        features = {
            'price': close,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'micro_OFI': micro['micro_OFI'],
            'micro_CVD': micro['micro_CVD'],
            'rsi_2': rsi_2_val,
            'rsi_14': rsi_14_val,
            'sma_14': sma_14_val,
            'largest_buyer': largest_buyer_pct,
            'largest_seller': largest_seller_pct,
            'OFI_LOB': OFI_LOB,
            'OFI_AGG': OFI_AGG,
            'CVD_z': cvd_z,
            'candle_type': candle_type,
            'candle_size': candle_size,
        }

        # Reset minute accumulators
        self._reset_minute()

        return features

    def _reset_minute(self) -> None:
        """Reset all per-minute accumulators."""
        self._minute_open = None
        self._minute_high = None
        self._minute_low = None
        self._minute_close = None
        self._minute_buy_vol = 0.0
        self._minute_sell_vol = 0.0
        self._minute_volume = 0.0
        self._minute_num_trades = 0
        self._minute_largest_buy = 0.0
        self._minute_largest_sell = 0.0
        self._minute_ob_bid_sum = 0.0
        self._minute_ob_ask_sum = 0.0
        self._minute_ob_bid_count = 0
        self._minute_ob_ask_count = 0

    @property
    def current_mid_price(self) -> float:
        return self._last_mid_price

    def get_feature_columns(self) -> list:
        """Return the list of feature column names (excluding price OHLC)."""
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