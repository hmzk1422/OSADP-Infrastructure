"""OFMR production features built by the walk-forward state-space pipeline."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

_OFMR_DIR = Path(__file__).resolve().parent
_STRATEGY_DIR = _OFMR_DIR.parent
_REPO_ROOT = _OFMR_DIR.parents[2]
for _path in (_OFMR_DIR, _STRATEGY_DIR, _REPO_ROOT):
    if str(_path) not in os.sys.path:
        os.sys.path.insert(0, str(_path))

from ml_state_builder import (  # noqa: E402
    SessionFeatureContext,
    _load_and_build_state_space_from_frames,
    _rsi_features_from_price_bars,
    dense_resample,
)
from ml_shared_config import BAR_FREQ  # noqa: E402

N_FEATURES = 15
FEATURE_NAMES = [f"Column_{index}" for index in range(N_FEATURES)]
FEATURE_SOURCE_COLUMNS = (
    "spread_bps_z",
    "ma_slope_5",
    "rsi_x_ma",
    "ma_accel",
    "price_roc_3",
    "rsi_2",
    "sma_14",
    "ofi_z",
    "cvd_delta_z",
    "cvd_slope_z",
    "absorption_z",
    "tick_autocorr",
    "inst_ratio",
    "delta_divergence_z",
    "depth_imbalance_z",
)


@dataclass
class FeatureResult:
    frame: pl.DataFrame
    feature_names: list[str]


def _session_context(
    orderbook: pl.DataFrame, trades: pl.DataFrame
) -> SessionFeatureContext:
    total_volume = float(trades.select(pl.col("amount").cast(pl.Float64).sum()).item())
    sparse_bars = (
        orderbook.sort(["datetime", "timestamp"], maintain_order=True)
        .group_by_dynamic("datetime", every=BAR_FREQ, closed="left", label="left")
        .agg(pl.col("mid_price").last().alias("close"))
    )
    price_bars = dense_resample(
        sparse_bars,
        BAR_FREQ,
        [pl.col("close").last().alias("close")],
    )
    return SessionFeatureContext(
        total_volume=total_volume,
        rsi_features=_rsi_features_from_price_bars(price_bars),
    )


def _build_state_space(
    orderbook: pl.DataFrame, trades: pl.DataFrame
) -> pl.DataFrame:
    if orderbook.is_empty() or trades.is_empty():
        raise ValueError("OFMR features require non-empty orderbook and trade data")
    if "datetime" not in orderbook.columns:
        orderbook = orderbook.with_columns(
            pl.from_epoch(
                (pl.col("timestamp") // 1000).cast(pl.Int32), time_unit="s"
            ).alias("datetime")
        )
    if "datetime" not in trades.columns:
        trades = trades.with_columns(
            pl.from_epoch(
                (pl.col("timestamp") // 1000).cast(pl.Int32), time_unit="s"
            ).alias("datetime")
        )
    if "mid_price" not in orderbook.columns:
        orderbook = orderbook.with_columns(
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid_price")
        )

    orderbook = orderbook.sort(["datetime", "timestamp"], maintain_order=True)
    trades = trades.sort(["datetime", "timestamp"], maintain_order=True)
    context = _session_context(orderbook, trades)
    return _load_and_build_state_space_from_frames(
        orderbook,
        trades,
        session_id=0,
        session_features=context,
    )


def _as_feature_result(state_space: pl.DataFrame) -> FeatureResult:
    required = [
        "datetime",
        "execution_price",
        "bar_vol_std",
        *FEATURE_SOURCE_COLUMNS,
    ]
    missing = [column for column in required if column not in state_space.columns]
    if missing:
        raise KeyError(f"OFMR state-space builder omitted columns: {missing}")
    frame = state_space.select(
        pl.col("datetime").cast(pl.Datetime("us")),
        pl.col("execution_price").alias("price"),
        *[
            pl.col(source).alias(target)
            for source, target in zip(FEATURE_SOURCE_COLUMNS, FEATURE_NAMES)
        ],
        pl.col("bar_vol_std"),
    )
    return FeatureResult(frame=frame, feature_names=FEATURE_NAMES.copy())


def compute_features(
    orderbook: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    bar_freq: str = "1s",
    symbol: str = "BTCUSDT",
) -> FeatureResult:
    """Build the same point-in-time state space used by ml_state_optimized."""
    del bar_freq, symbol
    return _as_feature_result(_build_state_space(orderbook, trades))


def _matched_files(
    matched_root: str | os.PathLike[str], symbol: str, tag: Optional[str]
) -> list[str]:
    root = Path(matched_root)
    symbol_dir = root
    if root.name.upper() != symbol.upper():
        matches = [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.upper() == symbol.upper()
        ] if root.is_dir() else []
        if not matches:
            raise FileNotFoundError(
                f"No matched directory for {symbol!r} under {str(root)!r}"
            )
        symbol_dir = matches[0]

    files = sorted(glob.glob(str(symbol_dir / "merged_*.parquet")))
    if tag:
        files = [path for path in files if tag in Path(path).name]
    if not files:
        raise FileNotFoundError(
            f"No matched parquet files for symbol={symbol!r}, tag={tag!r} "
            f"under {str(symbol_dir)!r}"
        )
    return files


def compute_features_from_matched(
    matched_root: str | os.PathLike[str],
    symbol: str = "BTCUSDT",
    tag: Optional[str] = None,
) -> FeatureResult:
    """Load matched event files and run the walk-forward state-space pipeline."""
    from abstraction import load_matched_frames  # noqa: PLC0415

    files = _matched_files(matched_root, symbol, tag)
    orderbook, trades = load_matched_frames(files)
    return _as_feature_result(_build_state_space(orderbook, trades))


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SOURCE_COLUMNS",
    "N_FEATURES",
    "FeatureResult",
    "compute_features",
    "compute_features_from_matched",
]
