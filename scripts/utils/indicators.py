"""
Technical Indicators and ML Feature Engineering

This module provides a suite of technical indicators and mathematical 
transformations to normalize and prepare raw market data for machine 
learning models.

Role:
    - Provides a library of signal processing and statistical indicators.
    - Normalizes raw price data (OHLC) and predictive data (OF, Liquidation, etc) into feature sets for ML training.
    - Implements custom kinematics calculations to analyze market momentum.

Key Responsibilities:
    - Feature Calculation: Volume spread analysis, TA, Liquidation, Orderflow imbalance, etc
    - Market Analysis: Measures price velocity and acceleration relative to volatility.
    - Regime Identification: Categorizes market states based on signal strength 
      and recovery speed.
    - Data Preparation: Facilitates the extraction of predictive features 
      from raw ticker arrays.

Usage:
    import numpy as np
    from indicators import calculate_zscore, get_market_kinematics
    
    # Calculate rolling Z-Score for normalized pricing
    z_scores = calculate_zscore(price_data, window=20)
    
    # Extract kinematic features for ML input
    ma, slope, acc, rsi = get_market_kinematics(price, atr, ma_window=50)

Dependencies:
    - numpy: For optimized numerical computation.
    - polars: For rolling-window statistics and structured data (replaces pandas).
"""

import numpy as np
import polars as pl


"""
------------------------------------------------
PRICE INDICATORS
------------------------------------------------
"""

def calculate_zscore(data, window=None):
    """
    Computes Z-Score.
    If window is None: calculates Global Z-score.
    If window is int: calculates Rolling Z-score.

    NOTE ON THE OLD IMPLEMENTATION: the rolling branch used to build an
    explicit (N - window + 1) x window matrix via sliding_window_view,
    then call np.nanmean/np.nanstd(axis=1) on it. The view itself is
    cheap, but nanmean/nanstd have to build a full NaN-mask array the
    same shape as that matrix to do the "nan-aware" part — for a large
    N with a wide window that's a multi-GB allocation in one shot. This
    is the likely source of the near-crash you saw.

    Polars' rolling_mean/rolling_std compute in O(N) memory (a proper
    streaming window algorithm, not a materialized matrix), which fixes
    that at the source.
    """
    # fill_nan(None) turns float NaN into polars null, so mean/std below
    # ignore missing values the same way np.nanmean/np.nanstd did.
    s = pl.Series(np.asarray(data, dtype=float)).fill_nan(None)

    if window is None:
        mean = s.mean()
        std = s.std(ddof=0)
        if not std:  # None (all-null) or exactly 0
            return np.zeros(len(s))
        return ((s - mean) / std).to_numpy()

    mean = s.rolling_mean(window_size=window)
    std = s.rolling_std(window_size=window, ddof=0)
    return ((s - mean) / std).to_numpy()


def get_wilders_rsi(delta, window=14):
    """
    Wilder's smoothing is a recursion (each value depends on the value
    before it), so there's no vectorized form in polars that's any
    better than a numpy loop here — it's O(N) memory either way, and
    this loop was never the source of the memory spike. Left as a plain
    numpy loop for that reason.
    """
    delta = np.asarray(delta, dtype=float)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)

    avg_gain[window] = np.mean(gain[:window])
    avg_loss[window] = np.mean(loss[:window])

    # Wilder's Smoothing formula:
    # Next_Avg = ((Prev_Avg * (window - 1)) + Current) / window
    for i in range(window + 1, len(delta)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(high, low, close, window=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Forward-fill NaN true-range values BEFORE they enter the recursion,
    # so a bad/missing bar just carries forward the last known range
    # instead of poisoning every value after it.
    true_range = (
        pl.Series(true_range)
        .fill_nan(None)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .to_numpy()
    )

    atr = np.full_like(true_range, np.nan)
    atr[window - 1] = np.mean(true_range[:window])
    for i in range(window, len(true_range)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window

    return atr


def sma(price_data, ma_window):
    s = pl.Series(np.asarray(price_data, dtype=float))
    # rolling_mean gives null for the first ma_window-1 bars (same NaN
    # padding the old np.convolve + concatenate version produced).
    return s.rolling_mean(window_size=ma_window).to_numpy()


def get_market_kinematics(price_data, atr_data, ma_window=50, lookback=9, rsi_window=2):
    """
    Computes indicators matching the provided Pine Script logic.

    NOTE: the MA here is deliberately still np.convolve(..., mode='valid')
    rather than a polars rolling_mean. The slope/acceleration slicing
    below is derived against the *compressed* length that 'valid' mode
    produces (len == N - ma_window + 1, no NaN padding). Swapping to a
    full-length, NaN-padded rolling_mean would shift every one of those
    offsets and silently change what slope/acc/rsi line up against — not
    something to guess at without re-deriving the alignment. Ping me if
    you want this one converted too and I'll work through the offsets.
    """
    price = np.asanyarray(price_data, dtype=float)
    atr = np.asanyarray(atr_data, dtype=float)

    # 1. Moving Average (SMA for example, use ta-lib or custom EMA for others)
    ma = np.convolve(price, np.ones(ma_window) / ma_window, mode='valid')

    # 2. MA Slope (Velocity)
    # Logic: (MA[t] - MA[t - lookback]) / ATR[t]
    ma_slope = (ma[lookback:] - ma[:-lookback]) / atr[lookback + ma_window - 1:]

    # 3. Acceleration (Change in Slope)
    acceleration = np.diff(ma_slope)

    # 4. RSI
    delta = np.diff(price)
    rsi = get_wilders_rsi(delta, rsi_window)

    # Alignment: Acceleration is 1 shorter than Slope
    ma_final = ma[lookback + 1:]
    slope_final = ma_slope[1:]

    return ma_final, slope_final, acceleration, rsi[-(len(acceleration)):]


def analyze_regime_speed(df: pl.DataFrame):
    """
    Expects a polars DataFrame with 'rsi', 'ma_slope', 'acc' columns.

    The old version indexed a pandas DataFrame with .iloc inside a
    Python while-loop — every .iloc[i] call rebuilds a Series (index,
    dtype checks, etc.), so on any real-sized dataset that loop gets
    very slow, which can look like a hang/near-freeze even without
    technically leaking memory. Pulling each column to a plain numpy
    array once, up front, and indexing those instead removes that
    per-row overhead entirely — same logic, much lighter.
    """
    rsi = df["rsi"].to_numpy()
    ma_slope = df["ma_slope"].to_numpy()
    acc = df["acc"].to_numpy()

    results = []
    n = len(df)
    i = 0
    while i < n - 20:
        # Long Entry: RSI < 30
        if rsi[i] < 30:
            slope_dir = "Slope_UP" if ma_slope[i] > 0 else "Slope_DOWN"
            force_dir = "Force_POS" if acc[i] > 0 else "Force_NEG"

            for bars in range(1, 21):
                if rsi[i + bars] >= 50:
                    results.append({
                        'bars_to_neutral': bars,
                        'regime': f"Long_{slope_dir}_{force_dir}"
                    })
                    break

        # Short Entry: RSI > 70
        elif rsi[i] > 70:
            slope_dir = "Slope_UP" if ma_slope[i] > 0 else "Slope_DOWN"
            force_dir = "Force_POS" if acc[i] > 0 else "Force_NEG"

            for bars in range(1, 21):
                if rsi[i + bars] <= 50:
                    results.append({
                        'bars_to_neutral': bars,
                        'regime': f"Short_{slope_dir}_{force_dir}"
                    })
                    break
        i += 1

    if not results:
        return pl.DataFrame({'bars_to_neutral': [], 'regime': []})
    return pl.DataFrame(results)


def volatility_ratio(high, low, close, atr1_window=4, atr2_window=14):
    atr1 = calculate_atr(high, low, close, atr1_window)
    atr2 = calculate_atr(high, low, close, atr2_window)
    return atr1 / (atr2 + 1e-9)


"""
------------------------------------------------
ORDERFLOW INDICATORS
------------------------------------------------
"""

def calculate_cvd(agg_buyer, agg_seller):
    agg_buyer = np.asarray(agg_buyer, dtype=float)
    agg_seller = np.asarray(agg_seller, dtype=float)
    return pl.Series(agg_buyer - agg_seller).cum_sum().to_numpy()

