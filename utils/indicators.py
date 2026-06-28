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
    - numpy: For optimized numerical computation and windowing operations.
    - pandas: For structured data manipulation and regime analysis.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd


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
    """
    arr = np.asanyarray(data, dtype=float)
    
    # 1. Global Z-Score (Original behavior)
    if window is None:
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        return (arr - mean) / std if std != 0 else np.zeros_like(arr)
    
    # 2. Rolling Z-Score (Using window)
    else:
        # Create sliding windows of size 'window'
        windows = sliding_window_view(arr, window_shape=window)
        
        # Calculate mean and std for each window
        means = np.nanmean(windows, axis=1)
        stds = np.nanstd(windows, axis=1)
        
        # To match input length, pad the start with NaNs
        pad_width = window - 1
        zscores = np.full(len(arr), np.nan)
        
        # Avoid division by zero
        valid = stds != 0
        zscores[pad_width:][valid] = (arr[pad_width:][valid] - means[valid]) / stds[valid]
        
        return zscores

def get_wilders_rsi(delta, window=14):
    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Initialize with simple mean for the first value
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[window] = np.mean(gain[:window])
    avg_loss[window] = np.mean(loss[:window])
    
    # Wilder's Smoothing formula:
    # Next_Avg = ((Prev_Avg * (window - 1)) + Current) / window
    for i in range(window + 1, len(delta)):
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i]) / window
        
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

import numpy as np

def calculate_atr(high, low, close, window=14):
    """
    Calculates ATR matching the Pine Script ta.rma logic.
    """
    # 1. Calculate True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0] # Handle first element
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # 2. Wilder's Smoothing (RMA)
    atr = np.zeros_like(true_range)
    atr[window-1] = np.mean(true_range[:window]) # Initial SMA
    
    for i in range(window, len(true_range)):
        atr[i] = (atr[i-1] * (window - 1) + true_range[i]) / window
        
    return atr

def sma(price_data, ma_window):
    price = np.asanyarray(price_data, dtype=float)
    
    # 1. Use mode='valid' to get the clean mathematical convolution
    ma_valid = np.convolve(price, np.ones(ma_window) / ma_window, mode='valid')
    
    # 2. Create an array of NaNs for the starting bars that don't have a full window
    padding = np.full(ma_window - 1, np.nan)
    
    # 3. Stack the padding at the front so the most recent data stays at the end
    return np.concatenate([padding, ma_valid])

def get_market_kinematics(price_data, atr_data, ma_window=50, lookback=9, rsi_window=2):
    """
    Computes indicators matching the provided Pine Script logic.
    """
    price = np.asanyarray(price_data, dtype=float)
    atr = np.asanyarray(atr_data, dtype=float)
    
    # 1. Moving Average (SMA for example, use ta-lib or custom EMA for others)
    ma = np.convolve(price, np.ones(ma_window) / ma_window, mode='valid')
    
    # 2. MA Slope (Velocity) 
    # Logic: (MA[t] - MA[t - lookback]) / ATR[t]
    # We slice MA to align with the lookback window
    ma_slope = (ma[lookback:] - ma[:-lookback]) / atr[lookback + ma_window - 1:]
    
    # 3. Acceleration (Change in Slope)
    # This is the 1st derivative of the slope
    acceleration = np.diff(ma_slope)
    
    # 4. RSI (using the corrected RSI function)
    delta = np.diff(price)
    rsi = get_wilders_rsi(delta, rsi_window)
    
    # Alignment: Acceleration is 1 shorter than Slope
    # Truncate all to the shortest length
    ma_final = ma[lookback + 1:] # Align with slope
    slope_final = ma_slope[1:]   # Align with acceleration
    
    return ma_final, slope_final, acceleration, rsi[-(len(acceleration)):]

def analyze_regime_speed(df):
    results = []
    i = 0
    while i < len(df) - 20:
        # Long Entry: RSI < 30
        if df.iloc[i]['rsi'] < 30:
            # Categorize the regime
            slope_dir = "Slope_UP" if df.iloc[i]['ma_slope'] > 0 else "Slope_DOWN"
            force_dir = "Force_POS" if df.iloc[i]['acc'] > 0 else "Force_NEG"
            
            # Find recovery speed
            for bars in range(1, 21):
                if df.iloc[i + bars]['rsi'] >= 50:
                    results.append({
                        'bars_to_neutral': bars,
                        'regime': f"Long_{slope_dir}_{force_dir}"
                    })
                    break
        
        # Short Entry: RSI > 70
        elif df.iloc[i]['rsi'] > 70:
            slope_dir = "Slope_UP" if df.iloc[i]['ma_slope'] > 0 else "Slope_DOWN"
            force_dir = "Force_POS" if df.iloc[i]['acc'] > 0 else "Force_NEG"
            
            for bars in range(1, 21):
                if df.iloc[i + bars]['rsi'] <= 50:
                    results.append({
                        'bars_to_neutral': bars,
                        'regime': f"Short_{slope_dir}_{force_dir}"
                    })
                    break
        i += 1
    return pd.DataFrame(results)


"""
------------------------------------------------
ORDERFLOW INDICATORS
------------------------------------------------
"""
def calculate_cvd(agg_buyer, agg_seller):
    return pd.Series(agg_buyer - agg_seller).cumsum()