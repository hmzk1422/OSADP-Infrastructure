"""Feature pipeline for the OFMR scalp strategy.

``compute_features`` is the public entry point.  It joins the higher
timeframe (macro) and lower timeframe (micro) feature groups and applies
the causal delay/ffill rules that prevent lookahead bias.

The per-column math lives in the ``features/`` package:

    features/rsidprice.py  RSI(2) + d² price trigger (macro group)
    features/orderflow.py  order-flow / microstructure families (micro group)

Note on the layout
------------------
``features.py`` (this module) and the ``features/`` folder intentionally
share a name.  Python resolves the module first, so this file grants
itself the folder's ``__path__`` below; that lets the submodules inside
``features/`` import cleanly while ``from .features import
compute_features`` still reaches this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from components.pipelines.event_merger.matched.reconstruct import (
    join_timeframes,
    reconstruct,
)

# ---------------------------------------------------------------------
# Coexistence trick: make `...ofmr_scalp.features.<name>` resolve to the
# files inside the `features/` folder.  Must run before the imports
# immediately below.
# ---------------------------------------------------------------------
__path__ = [str(Path(__file__).resolve().parent / "features")]

from .features.orderflow import add_orderflow
from .features.rsidprice import add_rsi_d2_trigger


def report_time_precision(df: pd.DataFrame) -> None:
    """Print the average interval between consecutive rows."""
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.to_series()
    else:
        raise ValueError("No 'timestamp' column or DatetimeIndex found in DataFrame.")

    timestamps = timestamps.sort_values()
    avg_interval = timestamps.diff().mean()

    print(f"Average time interval: {avg_interval}")

    avg_seconds = avg_interval.total_seconds() if pd.notna(avg_interval) else 0.0
    print(f"Average interval in seconds: {avg_seconds:.6f} s")
    print(f"Average interval in milliseconds: {avg_seconds * 1000:.3f} ms")


def report_nan_counts(
    df: pd.DataFrame,
    micro_feat_col: str | list[str],
) -> None:
    """Print total NaN counts and NaN counts for the micro feature columns."""
    print("Total NaNs:", df.isna().sum().sum())
    print("\nColumns with NaNs:")
    print(df.isna().sum().loc[lambda x: x > 0])

    print("\nMicro feature:", micro_feat_col)
    if isinstance(micro_feat_col, list):
        print("Micro feature NaNs:\n", df[micro_feat_col].isna().sum())
    else:
        print("Micro feature NaNs:\n", df[micro_feat_col].isna().sum())


def compute_macro_features(
    df: pd.DataFrame,
    timeframe: str,
    *,
    rsi_period: int = 2,
    d2_ewm_span: int = 3,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Higher timeframe (HTF) feature group.

    Reconstructs ``df`` on ``timeframe``, then computes the RSI(2) + d²
    price trigger features.
    """
    df = df.copy()
    df = reconstruct(df, timeframe)

    df, feat_col = add_rsi_d2_trigger(
        df,
        price_col="price_close",
        rsi_period=rsi_period,
        d2_ewm_span=d2_ewm_span,
        oversold=oversold,
        overbought=overbought,
    )

    feat_cols = [feat_col] if isinstance(feat_col, str) else list(feat_col)
    return df, feat_cols


def compute_micro_features(
    df: pd.DataFrame,
    timeframe: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Lower timeframe (LTF) feature group.

    Reconstructs ``df`` on ``timeframe``, then computes the order-flow /
    microstructure features.
    """
    df = df.copy()
    df = reconstruct(df, timeframe)
    df, feat_col = add_orderflow(df)

    feat_cols = [feat_col] if isinstance(feat_col, str) else list(feat_col)
    return df, feat_cols


def compute_features(
    df: pd.DataFrame,
    delay: int = 1,
    htf_timeframe: str = "1min",
    ltf_timeframe: str = "100ms",
    rsi_period: int = 2,
    d2_ewm_span: int = 3,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Join macro and micro features into one causal frame.

    Parameters mirror the ``features`` section of ``settings.yaml``.

    Parameters
    ----------
    df:
        Source tick/bar data.
    delay:
        Lookahead prevention delay (in rows/bars).

    Returns
    -------
    (df, feat_cols)
        ``price_close`` plus the feature columns, ready for
        ``strategy()`` to label with the triple-barrier sequence.
    """
    df = df.copy()

    # 1. Mid price on the original frame.
    df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2.0

    # 2. Feature groups.
    df_1, macro_feat_cols = compute_macro_features(
        df,
        htf_timeframe,
        rsi_period=rsi_period,
        d2_ewm_span=d2_ewm_span,
        oversold=oversold,
        overbought=overbought,
    )
    df_2, micro_feat_cols = compute_micro_features(df, ltf_timeframe)
    df_merged = join_timeframes(df_2, df_1, htf_timeframe, causal_col=None)

    # 3. HTF columns are namespaced by `join_timeframes`.
    macro_feat_cols = [f"htf_{col}" for col in macro_feat_cols]
    if "mid_price" not in df_merged.columns and "mid_price_close" in df_merged.columns:
        df_merged["mid_price"] = df_merged["mid_price_close"]

    # 4. Lookahead prevention: shift by `delay`, then carry the last known
    #    value forward until fresh data arrives.
    feat_cols = macro_feat_cols + micro_feat_cols + ["mid_price"]
    df_merged[feat_cols] = df_merged[feat_cols].shift(delay)
    df_merged[feat_cols] = df_merged[feat_cols].ffill()
    df_merged = df_merged.dropna(subset=feat_cols)
    df_merged["price_close"] = df_merged["price_close"].ffill().fillna(df_merged["mid_price"])

    return df_merged[["price_close"] + feat_cols], feat_cols
