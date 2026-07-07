"""
Composite Index Builder — Trend Persistency Membership
========================================================

Turns the per-asset trend_persistency() signal from screener.py into a
rules-based, dynamically-reconstituted synthetic index:

    signal(t) = SMA_7( RSI_14(t) - SMA_14(RSI_14)(t) )

    signal(t) > 0   ->  asset is IN the index
    signal(t) < 0   ->  asset is OUT of the index

While an asset is IN, its return is blended into the composite
(equal-weighted, chain-linked). The moment its signal crosses below 0 it's
dropped and the composite reweights across whoever's left — the same
mechanism real indices use (via their divisor) to absorb additions and
removals without the price-scale of any single name distorting the index
level. That's what makes this work across assets with wildly different
prices (BTC at $60k next to a coin at $0.0002).

Once the composite itself is built as a price series, you can run the
*original* 200-length trend_persistency() on it to test whether "a basket
of currently-trending assets" is itself trend-persistent — a recursive/
meta version of the same question.
"""

import numpy as np
import pandas as pd
from utils.indicators import *
from utils.screener import *


# ---------------------------------------------------------------------------
# 1. Membership signal: identical math to trend_persistency() in screener.py,
#    just with a 7-bar lookback on the final smoothing instead of 200. This
#    does NOT touch or replace trend_persistency() — screener.py is left
#    exactly as-is, and we import it directly later for the recursive test.
# ---------------------------------------------------------------------------
def membership_bias(df, period=7):
    close = df['close'].values
    delta = np.diff(close, prepend=close[0])

    rsi_14 = get_wilders_rsi(delta, 14)
    sma_14 = sma(rsi_14, 14)

    rsi_sma_dist = rsi_14 - sma_14
    bias = sma(rsi_sma_dist, period)

    return bias


# ---------------------------------------------------------------------------
# 2. Time axis helper. Assumes each df has a 'timestamp' column (typical for
#    ccxt-style OHLCV frames). Falls back to the df's own index if not found
#    — adjust `timestamp_col` if your connector names it differently.
# ---------------------------------------------------------------------------
def _time_index(df, timestamp_col="timestamp"):
    if timestamp_col in df.columns:
        return pd.Index(df[timestamp_col].values)
    return df.index


# ---------------------------------------------------------------------------
# 3. Align every candidate asset's close price onto one shared time axis.
#    Outer-join + limited ffill so one asset's missing candle doesn't nuke
#    the whole row for every other asset.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Membership matrix (bool, one column per ticker) on that same aligned
#    time axis. Computed off each asset's OWN native series (not the
#    ffilled/aligned one), then reindexed.
#
#    confirm_bars is optional hysteresis: require the raw signal to sit on
#    one side for N consecutive bars before the membership flag actually
#    flips. Real indices use an analogous "buffer rule" (S&P, Russell, etc.)
#    specifically to stop names from churning in and out on noise. Default
#    is 1 = no hysteresis, i.e. exactly your original ">0 / <0" rule.
# ---------------------------------------------------------------------------
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


def _apply_hysteresis(state, confirm_bars):
    """Vectorized hysteresis using rolling windows and forward-filling."""
    if confirm_bars <= 1:
        return state
        
    # 1. Convert boolean to 1/0 and calculate rolling sum
    rolling_sum = state.astype(int).rolling(window=confirm_bars).sum()
    
    # 2. Create an empty Series to hold our definitive "trigger" points
    triggers = pd.Series(index=state.index, dtype=object)
    
    # 3. Trigger True when the last N bars were all True
    triggers[rolling_sum == confirm_bars] = True
    
    # 4. Trigger False when the last N bars were all False
    triggers[rolling_sum == 0] = False
    
    # 5. Forward-fill the triggers to maintain state during noisy/mixed periods.
    # The initial un-triggered period defaults to the very first raw state.
    out = triggers.ffill().fillna(state.iloc[0]).astype(bool)
    
    return out

# ---------------------------------------------------------------------------
# 5. Build the composite. Membership decided at the CLOSE of bar t governs
#    participation in bar t+1's return (shift(1)) — you can't trade a bar
#    you haven't seen close yet, so this avoids look-ahead bias.
#    Equal-weighted, chain-linked (index-style) across whoever is active.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 6. Recursive/meta test: does the composite ITSELF show trend persistency?
#    Reuses your original, unmodified trend_persistency() from screener.py
#    (the 200-length version) on the composite's own price series.
# ---------------------------------------------------------------------------
def test_composite_persistency(composite_df):
    from utils.screener import trend_persistency
    bias, rsi_sma_dist = trend_persistency(composite_df)
    return bias, rsi_sma_dist


# ---------------------------------------------------------------------------
# 7. Reconstitution log — every add/drop event, same idea as S&P publishing
#    index changes. Useful for auditing turnover / sanity-checking the rule.
# ---------------------------------------------------------------------------
def membership_changes(membership):
    """Vectorized extraction of ADD/DROP events."""
    # diff() gives 1 for ADD, -1 for DROP, 0 for no change.
    # stack() flattens the 2D matrix into a 1D MultiIndex Series (timestamp, ticker)
    diff = membership.astype(int).diff().stack()
    
    # Keep only the rows where a change actually occurred
    changes = diff[diff != 0]
    
    if changes.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "event"])
        
    # Reset index turns the MultiIndex into standard columns
    events_df = changes.reset_index()
    events_df.columns = ["timestamp", "ticker", "event_code"]
    
    # Map the integers back to readable strings
    events_df["event"] = np.where(events_df["event_code"] == 1, "ADD", "DROP")
    
    return events_df[["timestamp", "ticker", "event"]].sort_values("timestamp").reset_index(drop=True)

