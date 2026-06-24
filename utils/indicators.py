import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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
def get_market_kinematics(price_data, ma_window=50, rsi_window=2):
    """
    Computes MA, MA Slope (1st deriv), MA Acceleration (2nd deriv), and RSI.
    Returns: (ma, slope, acceleration, rsi)
    """
    price = np.asanyarray(price_data, dtype=float)
    
    # 1. Moving Average (Simple)
    # Using 'valid' means the output is shorter by (ma_window - 1)
    ma = np.convolve(price, np.ones(ma_window)/ma_window, mode='valid')
    
    # 2. MA Slope (Velocity) & Acceleration
    slope = np.diff(ma)
    acceleration = np.diff(slope)
    
    # 3. RSI Calculation
    delta = np.diff(price)
 
    rsi = get_wilders_rsi(delta)
    
    # Align lengths:
    # Since diff() reduces length by 1, and convolution reduces by (window-1)
    # acceleration is 2 shorter than ma, RSI is 1 shorter than price.
    # To keep this clean, truncate everything to the shortest common length:
    min_len = len(acceleration)
    return ma[-min_len:], slope[-min_len:], acceleration, rsi[-min_len:]

