import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def calculate_zscore(data, window=None):
    arr = np.asanyarray(data, dtype=float)

    if window is None:
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        return (arr - mean) / std if std != 0 else np.zeros_like(arr)

    windows = sliding_window_view(arr, window_shape=window)
    means = np.nanmean(windows, axis=1)
    stds = np.nanstd(windows, axis=1)

    pad_width = window - 1
    zscores = np.full(len(arr), np.nan)
    valid = stds != 0
    zscores[pad_width:][valid] = (arr[pad_width:][valid] - means[valid]) / stds[valid]
    return zscores


def get_wilders_rsi(delta, window=14):
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)

    avg_gain[window] = np.mean(gain[:window])
    avg_loss[window] = np.mean(loss[:window])

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

    true_range = np.where(np.isnan(true_range), np.nan, true_range)
    mask = np.isnan(true_range)
    if np.any(mask):
        idx = np.where(~mask, np.arange(len(true_range)), 0)
        np.maximum.accumulate(idx, out=idx)
        true_range = true_range[idx]

    atr = np.full_like(true_range, np.nan)
    atr[window - 1] = np.mean(true_range[:window])
    for i in range(window, len(true_range)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window
    return atr


def sma(price_data, ma_window):
    price = np.asanyarray(price_data, dtype=float)
    ma_valid = np.convolve(price, np.ones(ma_window) / ma_window, mode='valid')
    padding = np.full(ma_window - 1, np.nan)
    return np.concatenate([padding, ma_valid])


def get_market_kinematics(price_data, atr_data, ma_window=50, lookback=9, rsi_window=2):
    price = np.asanyarray(price_data, dtype=float)
    atr = np.asanyarray(atr_data, dtype=float)

    ma = sma(price, ma_window)

    ma_slope = (ma[lookback:] - ma[:-lookback]) / atr[lookback + ma_window - 1:]

    acceleration = np.diff(ma_slope)

    delta = np.diff(price)
    rsi = get_wilders_rsi(delta, rsi_window)

    ma_final = ma[lookback + 1:]
    slope_final = ma_slope[1:]

    return ma_final, slope_final, acceleration, rsi[-(len(acceleration)):]


def analyze_regime_speed(df):
    results = []
    i = 0
    while i < len(df) - 20:
        rsi_val = df.iloc[i]['rsi']
        slope_val = df.iloc[i]['ma_slope']
        acc_val = df.iloc[i]['acc']

        if rsi_val < 30:
            slope_dir = "Slope_UP" if slope_val > 0 else "Slope_DOWN"
            force_dir = "Force_POS" if acc_val > 0 else "Force_NEG"
            for bars in range(1, 21):
                if df.iloc[i + bars]['rsi'] >= 50:
                    results.append({
                        'bars_to_neutral': bars,
                        'regime': f"Long_{slope_dir}_{force_dir}"
                    })
                    break

        elif rsi_val > 70:
            slope_dir = "Slope_UP" if slope_val > 0 else "Slope_DOWN"
            force_dir = "Force_POS" if acc_val > 0 else "Force_NEG"
            for bars in range(1, 21):
                if df.iloc[i + bars]['rsi'] <= 50:
                    results.append({
                        'bars_to_neutral': bars,
                        'regime': f"Short_{slope_dir}_{force_dir}"
                    })
                    break
        i += 1
    return pd.DataFrame(results)


def volatility_ratio(high, low, close, atr1_window=4, atr2_window=14):
    atr1 = calculate_atr(high, low, close, atr1_window)
    atr2 = calculate_atr(high, low, close, atr2_window)
    return atr1 / (atr2 + 1e-9)


def calculate_cvd(agg_buyer, agg_seller):
    diff = np.asarray(agg_buyer, dtype=float) - np.asarray(agg_seller, dtype=float)
    result = np.cumsum(diff)
    if isinstance(agg_buyer, pd.Series):
        return pd.Series(result, index=agg_buyer.index)
    return result
