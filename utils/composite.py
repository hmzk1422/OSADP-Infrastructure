import numpy as np
import pandas as pd
from .indicators import get_wilders_rsi, sma
from .screener import trend_persistency


def membership_bias(df, period=7):
    close = df['close'].values
    delta = np.diff(close, prepend=close[0])

    rsi_14 = get_wilders_rsi(delta, 14)
    sma_14 = sma(rsi_14, 14)

    rsi_sma_dist = rsi_14 - sma_14
    bias = sma(rsi_sma_dist, period)

    return bias


def _time_index(df, timestamp_col="timestamp"):
    if timestamp_col in df.columns:
        return pd.Index(df[timestamp_col].values)
    return df.index


def align_assets(screened_data, timestamp_col="timestamp", max_ffill=3):
    frames = {}
    for ticker, data in screened_data.items():
        df = data["df"]
        idx = _time_index(df, timestamp_col)
        s = pd.Series(df["close"].values, index=idx, name=ticker)
        frames[ticker] = s

    aligned = pd.concat(frames.values(), axis=1, join="outer").sort_index()
    aligned = aligned.ffill(limit=max_ffill)
    return aligned


def _apply_hysteresis(state, confirm_bars):
    if confirm_bars <= 1:
        return state

    rolling_sum = state.astype(int).rolling(window=confirm_bars).sum()
    triggers = pd.Series(index=state.index, dtype=object)

    triggers[rolling_sum == confirm_bars] = True
    triggers[rolling_sum == 0] = False

    out = triggers.ffill().fillna(state.iloc[0]).astype(bool)
    return out


def build_membership_matrix(screened_data, aligned_index, period=7,
                            timestamp_col="timestamp", confirm_bars=1):
    cols = {}
    for ticker, data in screened_data.items():
        df = data["df"]
        idx = _time_index(df, timestamp_col)

        sig = membership_bias(df, period=period)
        raw_in = pd.Series(sig, index=idx) > 0
        raw_in = raw_in.reindex(aligned_index).ffill().fillna(False)

        if confirm_bars > 1:
            raw_in = _apply_hysteresis(raw_in, confirm_bars)

        cols[ticker] = raw_in

    return pd.DataFrame(cols)


def build_composite(screened_data, period=7, base_value=100.0,
                    timestamp_col="timestamp", confirm_bars=1):
    aligned_prices = align_assets(screened_data, timestamp_col=timestamp_col)
    membership = build_membership_matrix(
        screened_data, aligned_prices.index, period=period,
        timestamp_col=timestamp_col, confirm_bars=confirm_bars,
    )

    rets = aligned_prices.pct_change()
    active = membership.shift(1).fillna(False).astype(bool)

    masked_rets = rets.where(active)
    composite_ret = masked_rets.mean(axis=1, skipna=True).fillna(0.0)

    composite_level = base_value * (1 + composite_ret).cumprod()
    n_constituents = active.sum(axis=1)

    composite_df = pd.DataFrame({
        "timestamp": composite_level.index,
        "close": composite_level.values,
        "n_constituents": n_constituents.values,
    })

    return composite_df, membership, active


def test_composite_persistency(composite_df):
    bias, rsi_sma_dist = trend_persistency(composite_df)
    return bias, rsi_sma_dist


def membership_changes(membership):
    diff = membership.astype(int).diff().stack()
    changes = diff[diff != 0]

    if changes.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "event"])

    events_df = changes.reset_index()
    events_df.columns = ["timestamp", "ticker", "event_code"]
    events_df["event"] = np.where(events_df["event_code"] == 1, "ADD", "DROP")

    return events_df[["timestamp", "ticker", "event"]].sort_values("timestamp").reset_index(drop=True)