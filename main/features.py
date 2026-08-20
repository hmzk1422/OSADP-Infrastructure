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
from scripts.ingestion.utils.indicators import get_wilders_rsi, sma, calculate_zscore

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
            'bar_vol_std': 0.0,
            # --- Orderflow (optional, see USE_ORDERFLOW_FEATURES) ---
            'ofi_z': 0.0,
            'cvd_delta_z': 0.0,
            'cvd_slope_z': 0.0,
            'vpin': 0.0,
            'absorption_z': 0.0,
            'tick_autocorr': 0.0,
            'inst_ratio': 0.0,
            'delta_divergence_z': 0.0,
            'depth_imbalance_z': 0.0,
        }

        # --- Buffers for macro feature computation ---
        self._close_prices = RingBuffer(200)       # For MA slope, price_roc
        self._price_diffs = RingBuffer(300)         # For RSI computation
        self._rsi_2_buffer = RingBuffer(50)         # For rsi_z computation
        self._spread_bps_buffer = RingBuffer(50)    # For spread_bps_z computation

        # --- Orderflow feature state (optional 9-feature block; gated by
        # config.USE_ORDERFLOW_FEATURES). Mirrors ml_state_optimized.py's
        # compute_orderflow_features() + Step 4 z-scoring, adapted to run
        # incrementally per tick instead of over a completed CSV session.
        self._use_orderflow = bool(globals().get('USE_ORDERFLOW_FEATURES', False))
        self._retail_threshold = globals().get('RETAIL_THRESHOLD', 0.05)
        self._orderflow_z_window = globals().get('ORDERFLOW_Z_WINDOW', 10)
        self._vpin_target_buckets = globals().get('VPIN_TARGET_BUCKETS_PER_SESSION', 50)
        self._vpin_window_buckets = globals().get('VPIN_WINDOW_BUCKETS', 10)
        self._vpin_min_periods = globals().get('VPIN_MIN_PERIODS', 5)
        vpin_lookback_s = globals().get('VPIN_SESSION_LOOKBACK_SECONDS', 86400)
        vpin_lookback_bars = max(int(vpin_lookback_s // max(TIMEFRAME_BARRIER, 1)), 3)

        # OFI (order flow imbalance): needs the PREVIOUS orderbook tick to
        # compute a delta — tracked separately from _last_bid_price/etc below
        # (those get overwritten every tick and are also used for spread/depth).
        self._ofi_prev_bid_price: Optional[float] = None
        self._ofi_prev_bid_amount: Optional[float] = None
        self._ofi_prev_ask_price: Optional[float] = None
        self._ofi_prev_ask_amount: Optional[float] = None
        self._bar_ofi_sum: float = 0.0          # raw OFI accumulated so far this bar
        self._prev_bar_ofi: float = 0.0         # last completed bar's total (for diff)
        self._ofi_diff_buffer = RingBuffer(max(30, self._orderflow_z_window * 3))

        # CVD (cumulative volume delta): the per-bar net (buy - sell) IS
        # cvd_delta_1m (diff of a cumsum equals the per-bar sum it was built
        # from — see ml_state_optimized.py). cvd_slope_5 is the sum of the
        # last 5 per-bar deltas.
        self._bar_cvd_net: float = 0.0
        self._cvd_delta_buffer = RingBuffer(max(30, self._orderflow_z_window * 3))
        self._cvd_slope_buffer = RingBuffer(max(30, self._orderflow_z_window * 3))

        # VPIN: volume-clock buckets. Bucket size self-calibrates from a
        # trailing window of realized bar volume (VPIN_SESSION_LOOKBACK_SECONDS)
        # instead of a fixed total-session volume — live has no session
        # endpoint to divide by up front the way a finished CSV file does.
        self._vpin_bucket_buy: float = 0.0
        self._vpin_bucket_sell: float = 0.0
        self._vpin_bucket_vol: float = 0.0
        self._vpin_bucket_size: Optional[float] = None
        self._vpin_bar_volumes = RingBuffer(vpin_lookback_bars)
        self._vpin_ratio_buffer = RingBuffer(max(30, self._vpin_window_buckets * 3))
        self._vpin_smoothed: float = 0.0

        # Absorption ratio: per-bar CVD delta normalized by the bar's
        # orderbook price range (ask_high - bid_low seen during the bar).
        self._bar_ask_high: Optional[float] = None
        self._bar_bid_low: Optional[float] = None
        self._absorption_buffer = RingBuffer(max(30, self._orderflow_z_window * 3))

        # Trade sign autocorrelation: per-second "last trade tick sign" this
        # bar, collected into a plain list (needs the full in-bar series, not
        # a fixed rolling window) and lag-1-autocorrelated at bar rollover.
        self._last_trade_price: Optional[float] = None
        self._second_last_tick_sign: int = 0
        self._bar_tick_signs: list = []

        # Retail vs institutional trade-size cohorts.
        self._bar_retail_buy: float = 0.0
        self._bar_retail_sell: float = 0.0
        self._bar_inst_buy: float = 0.0
        self._bar_inst_sell: float = 0.0
        self._delta_divergence_buffer = RingBuffer(max(30, self._orderflow_z_window * 3))

        # Depth imbalance: reuses _last_bid_amount/_last_ask_amount (latest
        # snapshot at bar close, tracked below), just needs its own z-score history.
        self._depth_imbalance_buffer = RingBuffer(max(30, self._orderflow_z_window * 3))

        # --- Aggregated-bar volatility (for barrier sizing, NOT an ML feature) ---
        # Std-dev of bar-to-bar % returns, measured on the SAME TIMEFRAME_BARRIER
        # aggregated candle that RSI(2) reads from, instead of approximating it
        # from 1s-tick std x sqrt(horizon).
        self._bar_vol_window = globals().get('BARRIER_VOL_BAR_WINDOW', 20)
        self._bar_vol_std: float = 0.0

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

        # ================================================================
        # Orderflow accumulation (optional — see USE_ORDERFLOW_FEATURES)
        # ================================================================
        if self._use_orderflow:
            # --- CVD: cvd_tick = buy_vol - sell_vol, summed over the bar ---
            self._bar_cvd_net += amount if side == 'buy' else -amount

            # --- Retail vs institutional cohorts (ml_state_optimized.py
            # compute_orderflow_features Step 5) ---
            if amount < self._retail_threshold:
                if side == 'buy':
                    self._bar_retail_buy += amount
                else:
                    self._bar_retail_sell += amount
            else:
                if side == 'buy':
                    self._bar_inst_buy += amount
                else:
                    self._bar_inst_sell += amount

            # --- Trade sign / tick rule (vs previous TRADE price, not
            # previous second) — resample('1s').last() means only the last
            # trade's sign within a second survives, so just overwrite. ---
            if self._last_trade_price is not None:
                if price > self._last_trade_price:
                    self._second_last_tick_sign = 1
                elif price < self._last_trade_price:
                    self._second_last_tick_sign = -1
                else:
                    self._second_last_tick_sign = 0
            self._last_trade_price = price

            # --- VPIN: volume-clock bucket accumulation ---
            self._vpin_bucket_buy += amount if side == 'buy' else 0.0
            self._vpin_bucket_sell += amount if side == 'sell' else 0.0
            self._vpin_bucket_vol += amount
            if self._vpin_bucket_size is not None and self._vpin_bucket_vol >= self._vpin_bucket_size:
                vpin_raw = abs(self._vpin_bucket_buy - self._vpin_bucket_sell) / (self._vpin_bucket_vol + EPS)
                self._vpin_ratio_buffer.append(vpin_raw)
                if self._vpin_ratio_buffer._count >= self._vpin_min_periods:
                    vals = self._vpin_ratio_buffer.values
                    window = min(len(vals), self._vpin_window_buckets)
                    self._vpin_smoothed = float(np.nanmean(vals[-window:]))
                # Reset the bucket (any overshoot past the target size is
                # dropped rather than carried forward — a minor approximation
                # of the offline bucket-boundary logic, negligible at these
                # trade sizes relative to bucket size).
                self._vpin_bucket_buy = 0.0
                self._vpin_bucket_sell = 0.0
                self._vpin_bucket_vol = 0.0

    def on_orderbook(self, timestamp_ms: int, bid_price: float, bid_amount: float,
                     ask_price: float, ask_amount: float) -> None:
        """Ingest an orderbook snapshot."""
        # --- OFI (Cont et al.) — needs the PREVIOUS tick, so compute this
        # tick's contribution BEFORE overwriting _last_bid_price/etc below.
        # Mirrors ml_state_optimized.py compute_orderflow_features() Step 1,
        # applied per-tick instead of per-row-of-a-dataframe (sum is
        # associative, so accumulating raw ticks gives the same bar total as
        # resample('1s').sum() -> resample(BAR_FREQ).sum()).
        if self._use_orderflow:
            if self._ofi_prev_bid_price is not None:
                if bid_price > self._ofi_prev_bid_price:
                    delta_bid = bid_amount
                elif bid_price < self._ofi_prev_bid_price:
                    delta_bid = -self._ofi_prev_bid_amount
                else:
                    delta_bid = bid_amount - self._ofi_prev_bid_amount

                if ask_price < self._ofi_prev_ask_price:
                    delta_ask = ask_amount
                elif ask_price > self._ofi_prev_ask_price:
                    delta_ask = -self._ofi_prev_ask_amount
                else:
                    delta_ask = ask_amount - self._ofi_prev_ask_amount

                self._bar_ofi_sum += (delta_bid - delta_ask)
            # else: first tick ever (no previous snapshot) — matches
            # research's shift(1)-produced NaN row, which contributes 0.

            self._ofi_prev_bid_price = bid_price
            self._ofi_prev_bid_amount = bid_amount
            self._ofi_prev_ask_price = ask_price
            self._ofi_prev_ask_amount = ask_amount

            # --- Absorption ratio's price range: bar's ask high / bid low ---
            self._bar_ask_high = ask_price if self._bar_ask_high is None else max(self._bar_ask_high, ask_price)
            self._bar_bid_low = bid_price if self._bar_bid_low is None else min(self._bar_bid_low, bid_price)

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

        # Tick sign autocorrelation input: research resamples per-trade tick
        # rule to 1s via .last().fillna(0) — a second with no trade counts as
        # neutral (0), not carried forward from the prior second.
        if self._use_orderflow:
            self._bar_tick_signs.append(self._second_last_tick_sign)
            self._second_last_tick_sign = 0

        # Check if we need to compute new macro features (timeframe rollover)
        # TIMEFRAME_BARRIER is already in seconds (300 = 5 minutes)
        timeframe_key = int(self._current_second // TIMEFRAME_BARRIER) if self._current_second else 0
        if timeframe_key != self._last_macro_update_minute and self._minute_close is not None:
            self._compute_macro_features()
            # Reset accumulators for next timeframe
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
        # Aggregated-bar volatility for barrier sizing
        # Std-dev of returns BETWEEN consecutive aggregated (TIMEFRAME_BARRIER)
        # bar closes — same close series RSI(2) is built from — rather than a
        # 1s-tick std x sqrt(horizon) approximation.
        # ================================================================
        bar_closes = self._close_prices.values
        if len(bar_closes) >= 3:
            bar_rets = np.diff(bar_closes) / (bar_closes[:-1] + EPS)
            window = min(len(bar_rets), self._bar_vol_window)
            self._bar_vol_std = float(np.nanstd(bar_rets[-window:])) if window >= 2 else 0.0
        else:
            self._bar_vol_std = 0.0

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

        # ================================================================
        # Orderflow features (optional — see USE_ORDERFLOW_FEATURES).
        # Mirrors ml_state_optimized.py compute_orderflow_features() + the
        # Step 4 z-scoring, computed from this bar's accumulated state
        # instead of a resampled dataframe.
        # ================================================================
        if self._use_orderflow:
            zw = self._orderflow_z_window

            def _rolling_zscore(buffer: 'RingBuffer', value: float) -> float:
                buffer.append(value)
                vals = buffer.values
                if len(vals) < zw:
                    return 0.0
                window_vals = vals[-zw:]
                mean = np.nanmean(window_vals)
                std = np.nanstd(window_vals)
                return float((value - mean) / (std + EPS))

            # --- OFI: this bar's total minus last bar's total, z-scored ---
            ofi_diff = self._bar_ofi_sum - self._prev_bar_ofi
            ofi_z = _rolling_zscore(self._ofi_diff_buffer, ofi_diff)
            self._prev_bar_ofi = self._bar_ofi_sum

            # --- CVD: per-bar net delta, and 5-bar slope, each z-scored ---
            cvd_delta_z = _rolling_zscore(self._cvd_delta_buffer, self._bar_cvd_net)
            # Slope needs the raw (un-zscored) delta history, which
            # _rolling_zscore already appended to _cvd_delta_buffer above.
            cvd_hist = self._cvd_delta_buffer.values
            cvd_slope_5 = float(np.sum(cvd_hist[-5:])) if len(cvd_hist) >= 5 else 0.0
            cvd_slope_z = _rolling_zscore(self._cvd_slope_buffer, cvd_slope_5)

            # --- VPIN: bucket-close smoothing already maintained in
            # on_trade(); recalibrate the bucket size from trailing bar
            # volume so it tracks changing liquidity. ---
            self._vpin_bar_volumes.append(self._minute_volume)
            if self._vpin_bar_volumes._count >= 3:
                trailing_vol = float(np.nansum(self._vpin_bar_volumes.values))
                self._vpin_bucket_size = max(trailing_vol / self._vpin_target_buckets, EPS)
            vpin = self._vpin_smoothed

            # --- Absorption ratio: this bar's CVD delta / orderbook price range ---
            if self._bar_ask_high is not None and self._bar_bid_low is not None:
                price_range = self._bar_ask_high - self._bar_bid_low
            else:
                price_range = 0.0
            abs_ratio_raw = self._bar_cvd_net / (price_range + EPS)
            absorption_z = _rolling_zscore(self._absorption_buffer, abs_ratio_raw)

            # --- Trade sign autocorrelation (lag-1) over this bar's
            # per-second tick-sign series ---
            tick_arr = np.asarray(self._bar_tick_signs, dtype=float)
            if len(tick_arr) > 5 and np.nanstd(tick_arr) > 0:
                corr = np.corrcoef(tick_arr[:-1], tick_arr[1:])[0, 1]
                tick_autocorr = float(corr) if not np.isnan(corr) else 0.0
            else:
                tick_autocorr = 0.0

            # --- Retail vs institutional flow ---
            bar_volume = (self._bar_retail_buy + self._bar_retail_sell
                          + self._bar_inst_buy + self._bar_inst_sell)
            inst_delta = self._bar_inst_buy - self._bar_inst_sell
            retail_delta = self._bar_retail_buy - self._bar_retail_sell
            delta_divergence = inst_delta - retail_delta
            inst_ratio = (self._bar_inst_buy + self._bar_inst_sell) / (bar_volume + EPS)
            delta_divergence_z = _rolling_zscore(self._delta_divergence_buffer, delta_divergence)

            # --- Depth imbalance: latest orderbook snapshot at bar close ---
            bid_amt, ask_amt = self._last_bid_amount, self._last_ask_amount
            depth_imbalance = (bid_amt - ask_amt) / (bid_amt + ask_amt + EPS)
            depth_imbalance_z = _rolling_zscore(self._depth_imbalance_buffer, depth_imbalance)

            orderflow_features = {
                'ofi_z': ofi_z,
                'cvd_delta_z': cvd_delta_z,
                'cvd_slope_z': cvd_slope_z,
                'vpin': vpin,
                'absorption_z': absorption_z,
                'tick_autocorr': tick_autocorr,
                'inst_ratio': inst_ratio,
                'delta_divergence_z': delta_divergence_z,
                'depth_imbalance_z': depth_imbalance_z,
            }

            # Reset this bar's raw accumulators for the next bar.
            self._bar_ofi_sum = 0.0
            self._bar_cvd_net = 0.0
            self._bar_ask_high = None
            self._bar_bid_low = None
            self._bar_tick_signs = []
            self._bar_retail_buy = 0.0
            self._bar_retail_sell = 0.0
            self._bar_inst_buy = 0.0
            self._bar_inst_sell = 0.0
        else:
            orderflow_features = {}

        # Store macro features (these will be forward-filled across 1s bars)
        self._macro_features = {
            'rsi_2': rsi_2_val,
            'sma_14': sma_14_val,
            'ma_slope_5': ma_slope_5,
            'ma_accel': ma_accel,
            'spread_bps_z': spread_bps_z,
            'rsi_x_ma': rsi_x_ma,
            'bar_vol_std': self._bar_vol_std,  # NOT an ML feature — barrier sizing only
            **orderflow_features,
        }

        # Update with TIMEFRAME_BARRIER-based timeframe key
        self._last_macro_update_minute = int((self._current_second or 0) // TIMEFRAME_BARRIER)
        self._macro_features['_close'] = close
        self._macro_features['_high'] = high
        self._macro_features['_low'] = low

    def _reset_minute(self) -> None:
        """Reset all per-timeframe accumulators."""
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
            # Barrier-sizing only — NOT one of the 7 ML features (see get_feature_columns)
            'bar_vol_std': self._macro_features.get('bar_vol_std', 0.0),
        }

        # Optional orderflow block (see USE_ORDERFLOW_FEATURES) — always
        # copied out of _macro_features if present so the dict has them
        # whenever they've been computed, regardless of the toggle; whether
        # they actually reach the model is decided by get_feature_columns().
        if self._use_orderflow:
            for col in ('ofi_z', 'cvd_delta_z', 'cvd_slope_z', 'vpin', 'absorption_z',
                        'tick_autocorr', 'inst_ratio', 'delta_divergence_z', 'depth_imbalance_z'):
                features[col] = self._macro_features.get(col, 0.0)

        return features

    @property
    def current_mid_price(self) -> float:
        return self._last_mid_price

    def get_feature_columns(self) -> list:
        """Return the list of feature column names in order.

        Base 7 features, plus the 9 orderflow features when
        USE_ORDERFLOW_FEATURES is on (config.py) — matching the exact
        FEATURE_COLUMNS + ORDERFLOW_FEATURE_COLUMNS order ml_state_optimized.py
        exports in <model>.features.json.
        """
        try:
            from main.config import FEATURE_COLUMNS
            cols = list(FEATURE_COLUMNS)
        except (ImportError, AttributeError):
            cols = [
                'spread_bps_z',
                'ma_slope_5',
                'rsi_x_ma',
                'ma_accel',
                'price_roc_3',
                'rsi_2',
                'sma_14',
            ]

        try:
            from main.config import USE_ORDERFLOW_FEATURES, ORDERFLOW_FEATURE_COLUMNS
            if USE_ORDERFLOW_FEATURES:
                cols = cols + list(ORDERFLOW_FEATURE_COLUMNS)
        except (ImportError, AttributeError):
            pass

        return cols

    def extract_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to model-ready numpy array (1D)."""
        cols = self.get_feature_columns()
        vec = np.array([features.get(c, 0.0) for c in cols], dtype=float)
        return vec.reshape(1, -1)

    def seed_historical_data(self, symbol: str, limit: int = 150) -> None:
        """
        Fetches historical 1-minute closes to warm up the RSI and MA buffers,
        preventing 'cold start' math errors on initialization.
        """
        import logging
        from scripts.ingestion.utils.bybit import BybitConnector
        from main.config import API_KEY, API_SECRET, TESTNET
        
        logger = logging.getLogger("features")
        # RSI(2)/bar_vol_std are computed on TIMEFRAME_BARRIER-aggregated bars
        # (see _compute_macro_features). Seeding MUST use the same bar size,
        # or the ring buffers get warmed up with mismatched-timeframe closes
        # that then take up to 200 x TIMEFRAME_BARRIER seconds to fully flush.
        interval_str = f"{max(1, TIMEFRAME_BARRIER // 60)}m"
        logger.info(
            f"Seeding macro buffers with {limit} historical {interval_str} bars "
            f"for {symbol} (TIMEFRAME_BARRIER={TIMEFRAME_BARRIER}s)..."
        )

        connector = BybitConnector(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)
        df_ohlcv = connector.get_ohlcv(symbol=symbol, interval=interval_str, total_limit=limit)
        
        if df_ohlcv.empty:
            logger.error("Failed to fetch historical data for seeding! RSI may be inaccurate initially.")
            return
            
        closes = df_ohlcv['close'].astype(float).values
        
        # Push historical data through the buffers exactly like live data
        for close in closes:
            if self._close_prices._count > 0:
                prev_close = self._close_prices.values[-1]
                delta = close - prev_close
            else:
                delta = 0.0
                
            self._close_prices.append(close)
            self._price_diffs.append(delta)
            
        logger.info(f"Successfully seeded buffers. _close_prices count: {self._close_prices._count}")
        
        # Force a macro feature computation so the initial RSI is ready immediately
        self._last_mid_price = closes[-1]
        self._minute_close = closes[-1]
        self._compute_macro_features()


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