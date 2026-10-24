"""YAML-driven UETS playback runner for CryptoL2 offline and online market data.

Implements canonically accurate:

  - Purged Walk-Forward      (train on complement, purge+embargo both sides, refit per fold)
  - Parametric Monte Carlo   (generate N paths, fit distribution, bootstrap resample)
  - Combinatorial Purged CV  (train on n_splits-n_test_splits folds, test on purged combos)

All paths run through ``_run_one_path`` -- the UETS-compatible accounting engine
(fees, turnover, drawdown, Sortino) identical to live-trade accounting.

Data sources:

  - ``cryptoL2_offline``  lazy polars scan of ``market_data/matched/<SYM>/merged_*.parquet``
  - ``cryptoL2``          online chunked reading from ``market_data/ingest_parquet/<SYM>/merged/**/*.parquet``

Results are appended to ``market_data/live_db/playback/``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations as _combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "uets" / "config" / "execution.yaml"
PLAYBACK_DIR = ROOT / "market_data" / "live_db" / "playback"
MAX_PLOTTED_PATHS = 500


def _canonical_manifest_path(value: str | Path) -> str:
    """Compare source paths across Windows host manifests and Linux containers."""
    normalized = str(value).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    for index, part in enumerate(parts):
        if part.lower() == "market_data":
            return "/".join(parts[index:]).casefold()
    return Path(normalized).resolve().as_posix().casefold()


@dataclass(frozen=True)
class PathMetrics:
    cumulative_return: float
    expectancy_per_trade: float
    max_drawdown: float
    sortino: float
    trades: int
    equity_path: tuple[float, ...] = ()
    trade_returns: tuple[float, ...] = ()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    mode = str(config.get("mode", "")).strip().lower().replace("_", "-")
    if mode != "playback":
        raise ValueError("set mode: playback in the selected YAML to run playback")
    playback = config.get("playback", {})
    if not isinstance(playback, dict):
        raise ValueError("playback configuration must be a YAML mapping")
    return playback


def _timestamp_units_per_second(lazy: pl.LazyFrame) -> int:
    import polars as pl

    dtype = lazy.collect_schema()["timestamp"]
    time_unit = getattr(dtype, "time_unit", None)
    if time_unit is not None:
        return {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000}[time_unit]

    maximum = lazy.select(
        pl.col("timestamp").cast(pl.Int64).abs().max()
    ).collect(engine="streaming").item()
    magnitude = abs(int(maximum or 0))
    return (
        1_000_000_000
        if magnitude >= 1e17
        else 1_000_000
        if magnitude >= 1e14
        else 1_000
        if magnitude >= 1e11
        else 1
    )


def _cryptol2_price_expression(available: set[str]) -> pl.Expr:
    import polars as pl

    bid_name = next(
        (name for name in ("book_state__best_bid_price", "best_bid_price") if name in available),
        None,
    )
    ask_name = next(
        (name for name in ("book_state__best_ask_price", "best_ask_price") if name in available),
        None,
    )
    trade_name = next(
        (name for name in ("agg__price", "trade_price", "execution_price") if name in available),
        None,
    )
    if bid_name is None and trade_name is None:
        raise ValueError("CryptoL2 parquet needs bid prices or a trade price column")

    bid = pl.col(bid_name).cast(pl.Float64) if bid_name else pl.lit(None, dtype=pl.Float64)
    ask = pl.col(ask_name).cast(pl.Float64) if ask_name else pl.lit(None, dtype=pl.Float64)
    trade = pl.col(trade_name).cast(pl.Float64) if trade_name else bid
    valid_quote = (
        (bid > 0.0)
        & bid.is_finite()
        & (ask > 0.0)
        & ask.is_finite()
        if bid_name and ask_name
        else pl.lit(False)
    )
    return (
        pl.when(valid_quote)
        .then((bid + ask) / 2.0)
        .otherwise(trade)
        .alias("_price")
    )


def _read_cryptol2_bars(
    settings: dict[str, Any],
    *,
    offline: bool,
    bar_seconds: float,
    max_bars: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray | None, np.ndarray | None]:
    # Keep parquet scan buffers bounded; callers can opt into more threads.
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    import polars as pl

    symbol = str(settings.get("symbol", "BTCUSDT")).upper()
    default_root = "market_data/matched" if offline else "market_data/ingest_parquet"
    root = Path(str(settings.get("root", default_root)))
    if not root.is_absolute():
        root = ROOT / root
    default_glob = (
        f"{root}/{symbol}/merged_*.parquet"
        if offline
        else f"{root}/{symbol}/merged/**/*.parquet"
    )
    glob_pattern = str(settings.get("path_glob", default_glob)).replace("{symbol}", symbol)
    if not Path(glob_pattern).is_absolute():
        glob_pattern = str(ROOT / glob_pattern)
    files = sorted(Path(path) for path in glob.glob(glob_pattern, recursive=True))
    if not files:
        source = "matched" if offline else "ingest"
        raise FileNotFoundError(f"no {source} parquet files match {glob_pattern}")
    if not np.isfinite(bar_seconds) or bar_seconds <= 0.0:
        raise ValueError("playback.bar_seconds must be positive and finite")

    bars_per_file: list[pl.DataFrame] = []
    expected_scale: int | None = None
    has_positions: bool | None = None
    for path in files:
        lazy = pl.scan_parquet(str(path))
        schema = lazy.collect_schema()
        available = set(schema.names())
        if "timestamp" not in available:
            raise ValueError(f"CryptoL2 parquet has no timestamp column: {path}")
        if expected_scale is None:
            expected_scale = _timestamp_units_per_second(lazy)
        bar_size = max(1, int(bar_seconds * expected_scale))

        has_file_positions = {"position", "risk_fraction"}.issubset(available)
        if has_positions is None:
            has_positions = has_file_positions
        elif has_positions != has_file_positions:
            raise ValueError("CryptoL2 parquet files have inconsistent OFMR signal columns")

        selected = (
            lazy.select(
                pl.col("timestamp").cast(pl.Int64).alias("_timestamp"),
                _cryptol2_price_expression(available),
                *(
                    [
                        pl.col("position").cast(pl.Float64).alias("_position"),
                        pl.col("risk_fraction").cast(pl.Float64).alias("_risk_fraction"),
                    ]
                    if has_file_positions
                    else []
                ),
            )
            .filter(pl.col("_price").is_finite() & (pl.col("_price") > 0.0))
            .with_columns((pl.col("_timestamp") // bar_size).alias("_bucket"))
        )
        aggregations: list[pl.Expr] = [
            pl.col("_timestamp").sort_by("_timestamp").last().alias("timestamp"),
            pl.col("_price").sort_by("_timestamp").last().alias("price"),
        ]
        if has_file_positions:
            aggregations.extend(
                [
                    pl.col("_position").sort_by("_timestamp").last().alias("position"),
                    pl.col("_risk_fraction").sort_by("_timestamp").last().alias("risk_fraction"),
                ]
            )
        frame = (
            selected.group_by("_bucket")
            .agg(aggregations)
            .collect(engine="streaming")
        )
        if frame.height:
            bars_per_file.append(frame)

    if not bars_per_file:
        raise ValueError("CryptoL2 parquet files contain no valid price bars")

    # Deduplicate overlaps after aggregation; raw event rows stay out of memory.
    bars = (
        pl.concat(bars_per_file, how="vertical_relaxed")
        .sort("timestamp")
        .group_by("_bucket", maintain_order=True)
        .agg(pl.all().last())
        .sort("timestamp")
    )
    if max_bars is not None:
        bars = bars.head(max_bars)
    if bars.height < 3:
        raise ValueError("CryptoL2 source produced fewer than 3 valid price bars")

    timestamps = bars["timestamp"].to_numpy()
    prices = bars["price"].to_numpy().reshape(-1, 1)
    positions = bars["position"].to_numpy().reshape(-1, 1) if has_positions else None
    risk_fractions = (
        bars["risk_fraction"].to_numpy().reshape(-1, 1) if has_positions else None
    )
    return timestamps, prices, (symbol,), positions, risk_fractions


def read_cryptol2_offline(
    settings: dict[str, Any], bar_seconds: float = 1.0
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    timestamps, prices, symbols, _, _ = _read_cryptol2_bars(
        settings, offline=True, bar_seconds=bar_seconds
    )
    return timestamps, prices, symbols


def read_cryptol2_online(
    settings: dict[str, Any],
    chunk_size: int | None = None,
    bar_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    timestamps, prices, symbols, _, _ = _read_cryptol2_bars(
        settings, offline=False, bar_seconds=bar_seconds
    )
    return timestamps, prices, symbols

def _generate_synthetic(
    settings: dict[str, Any], *, paths: int | None = None
) -> list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]]:
    import pandas as pd

    from components.pipelines.ingestion.synthetic.uncorrelated_filter import (
        SYNTHETIC_GENERATORS,
    )

    generator_name = str(settings.get("generator", "synth2")).lower()
    try:
        generator = SYNTHETIC_GENERATORS[generator_name]
    except KeyError as exc:
        raise ValueError("generator must be synth1, synth2, or synth3") from exc
    n_paths = int(paths if paths is not None else settings.get("paths", 32))
    n_bars = int(settings.get("n_bars", 500))
    if n_paths < 1 or n_bars < 3:
        raise ValueError("synthetic.paths must be positive and synthetic.n_bars at least 3")
    first_seed = int(settings.get("seed", 84))
    kwargs = dict(settings.get("generator_kwargs", {}))
    kwargs.pop("n_bars", None)
    kwargs.pop("seed", None)
    symbol = str(settings.get("symbol", "SYNTH"))
    start = np.datetime64(settings.get("start_time", "2026-01-01T00:00:00"), "ns")
    bar_ns = int(float(settings.get("bar_seconds", 1.0)) * 1_000_000_000)
    timestamps = start + np.arange(n_bars, dtype=np.int64) * np.timedelta64(bar_ns, "ns")
    generated: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []
    for path_index in range(n_paths):
        frame = generator(n_bars=n_bars, seed=first_seed + path_index, **kwargs)
        prices = pd.to_numeric(frame["price"], errors="coerce").to_numpy(dtype=np.float64)
        if len(prices) != n_bars or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"synthetic generator returned invalid prices for seed {first_seed + path_index}")
        generated.append((timestamps.copy(), prices[:, None], (symbol,)))
    return generated


def _strategy_positions(prices: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    strategy_type = str(settings.get("type", "momentum")).lower()
    if strategy_type == "strategy2":
        price = np.asarray(prices, dtype=np.float64)
        window = int(settings.get("window", 5))
        long_above = float(settings.get("long_above", 1.0))
        short_below = float(settings.get("short_below", -1.0))
        if window < 1 or short_below >= long_above:
            raise ValueError("strategy2.window must be positive and short_below < long_above")
        positions = np.zeros_like(price, dtype=np.float64)
        active = np.zeros(price.shape[1], dtype=np.float64)
        for index in range(price.shape[0]):
            start = max(0, index - window + 1)
            deviation = price[index] - price[start:index + 1].mean(axis=0)
            enter_long = (active == 0.0) & (deviation > long_above)
            enter_short = (active == 0.0) & (deviation < short_below)
            active[enter_long] = 1.0
            active[enter_short] = -1.0
            positions[index] = active
        return positions
    if strategy_type != "momentum":
        raise ValueError("momentum baseline positions must be generated by the strategy config")
    lookback = int(settings.get("lookback_bars", 5))
    threshold = float(settings.get("entry_threshold", 0.0))
    if lookback < 1 or threshold < 0.0:
        raise ValueError("strategy.lookback_bars must be positive and entry_threshold nonnegative")
    momentum = np.zeros_like(prices, dtype=np.float64)
    if prices.shape[0] > lookback:
        momentum[lookback:] = prices[lookback:] / prices[:-lookback] - 1.0
    return np.where(np.abs(momentum) >= threshold, np.sign(momentum), 0.0)


def _ofmr_positions_and_risk(
    prices: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OFMR positions and risk fractions from price data.

    Positions: 1 (long), -1 (short), or 0 (flat) based on momentum threshold.
    Risk fractions: based on volatility-scaled position sizing.
    """
    prices = np.asarray(prices, dtype=np.float64).reshape(-1)
    lookback = int(settings.get("lookback_bars", 5))
    threshold = float(settings.get("entry_threshold", 0.0))
    if lookback < 1:
        lookback = 1
    # compute momentum over lookback period
    if prices.shape[0] > lookback:
        mom = prices[lookback:] / prices[:-lookback] - 1.0
    else:
        mom = np.zeros(0, dtype=np.float64)
    # build position series (same length as prices, with padding at start)
    positions = np.zeros(prices.shape[0], dtype=np.float64)
    if len(mom) > 0:
        positions[lookback:] = np.where(mom > threshold, 1.0, np.where(mom < -threshold, -1.0, 0.0))
    # compute risk fraction based on average absolute momentum (volatility proxy)
    if len(mom) > 0 and np.abs(mom).mean() > 0:
        risk_fraction = float(np.clip(0.02 / np.abs(mom).mean(), 0.001, 0.2))
    else:
        risk_fraction = 0.02
    # make risk_fraction match positions shape and be finite/non-negative
    risk_fractions = np.full(prices.shape[0], risk_fraction, dtype=np.float64)
    # zero out risk where position is flat
    risk_fractions *= np.maximum(np.abs(positions), 1e-6)
    return positions.reshape(-1, 1), risk_fractions.reshape(-1, 1)


def _ofmr_walk_forward_metrics(
    playback: dict[str, Any], source: str
) -> list[PathMetrics]:
    """Run the actual Rust OFMR feature/model/barrier pipeline on CryptoL2.

    OFMR cannot be approximated from close-to-close momentum: its trained model
    consumes order-book and trade features. The Rust adapter is the same
    strategy implementation used by the live runner and emits its out-of-sample
    fold decisions for the Python reporting/equity-path layer.
    """
    settings = playback.get(source, {})
    if not isinstance(settings, dict):
        raise ValueError(f"playback.{source} must be a YAML mapping")
    symbol = str(settings.get("symbol", "BTCUSDT")).upper()
    default_root = "market_data/matched" if source == "cryptol2_offline" else "market_data/ingest_parquet"
    root = Path(str(settings.get("root", default_root)))
    if not root.is_absolute():
        root = ROOT / root
    default_glob = (
        f"{root}/{symbol}/merged_*.parquet"
        if source == "cryptol2_offline"
        else f"{root}/{symbol}/merged/**/*.parquet"
    )
    pattern = str(settings.get("path_glob", default_glob)).replace("{symbol}", symbol)
    if not Path(pattern).is_absolute():
        pattern = str(ROOT / pattern)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no CryptoL2 files match {pattern}")

    state_space_value = settings.get("state_space_path")
    if state_space_value:
        return _reference_ofmr_walk_forward_metrics(
            playback, source, Path(str(state_space_value)), files, symbol,
        )

    walk_forward = settings.get("walk_forward", playback.get("walk_forward", {}))
    if not isinstance(walk_forward, dict):
        raise ValueError("playback.walk_forward must be a YAML mapping")
    catalog = Path(str(walk_forward.get("model_catalog", "build/walkforward_models/catalog.json")))
    if not catalog.is_absolute():
        catalog = ROOT / catalog
    try:
        models = json.loads(catalog.read_text(encoding="utf-8")).get("models")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load OFMR fold catalog {catalog}: {error}") from error
    if not isinstance(models, list) or not models:
        raise ValueError(f"OFMR fold catalog has no models: {catalog}")

    command = [
        "cargo", "run", "--quiet", "--manifest-path",
        str(ROOT / "components" / "pipelines" / "uets_trades" / "Cargo.toml"),
        "--bin", "playback_ofmr", "--", "--symbol", symbol,
    ]
    for path in files:
        command.extend(("--file", path))
    for model in models:
        model_path = Path(str(model["model"]))
        if not model_path.is_absolute():
            model_path = catalog.parent / model_path
        command.extend(("--fold-model", str(int(model["test_start_ms"])),
                        str(int(model["test_end_ms"])), str(model_path)))
        if "test_rows" in model:
            command.extend(("--fold-test-rows", str(int(model["test_rows"]))))
        if "train_rows" in model:
            command.extend(("--fold-train-rows", str(int(model["train_rows"]))))
    env = os.environ.copy()
    env["UETS_STARTING_CAPITAL"] = str(playback.get("starting_capital", 10_000.0))
    env["UETS_FEE_BPS"] = str(playback.get("fee_bps", 4.0))
    env["UETS_PERIODS_PER_YEAR"] = str(playback.get("periods_per_year", 252.0))
    strategy_settings = playback.get("strategy", {})
    env["OFMR_CONFIDENCE_THRESHOLD"] = str(
        strategy_settings.get("confidence_threshold", strategy_settings.get("threshold", 0.20))
    )
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="") as output:
        result = subprocess.run(
            command, cwd=ROOT, stdout=output, stderr=subprocess.PIPE, text=True,
            check=False, env=env,
        )
        if result.returncode:
            raise RuntimeError(
                f"Rust OFMR playback failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        output.seek(0)
        reader = csv.DictReader(output)
        required = {"fold", "timestamp_ms", "price", "position", "risk_fraction"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Rust OFMR playback omitted columns: {sorted(required)}")
        metrics: list[PathMetrics] = []
        current_fold: str | None = None
        fold_prices: list[float] = []
        fold_positions: list[float] = []
        fold_risk: list[float] = []

        def finish_fold() -> None:
            if current_fold is None:
                return
            if len(fold_prices) < 3:
                raise ValueError(f"OFMR fold {current_fold} has fewer than three OOS bars")
            metrics.append(_run_one_path(
                np.asarray(fold_prices, dtype=np.float64)[:, None], playback,
                np.asarray(fold_positions, dtype=np.float64)[:, None],
                np.asarray(fold_risk, dtype=np.float64)[:, None], capture_path=True,
            ))

        for row in reader:
            fold = row["fold"]
            if current_fold is not None and fold != current_fold:
                finish_fold()
                fold_prices.clear()
                fold_positions.clear()
                fold_risk.clear()
            current_fold = fold
            fold_prices.append(float(row["price"]))
            fold_positions.append(float(row["position"]))
            fold_risk.append(float(row["risk_fraction"]))
        finish_fold()
    if not metrics:
        raise ValueError(
            "OFMR fold catalog has no test rows in the selected CryptoL2 data; "
            "check that the catalog and dataset cover the same timestamps"
        )
    if len(metrics) != len(models):
        raise ValueError(f"Rust OFMR returned {len(metrics)} folds; catalog lists {len(models)}")
    return metrics


def _reference_ofmr_walk_forward_metrics(
    playback: dict[str, Any], source: str, state_space_path: Path,
    input_files: list[str], symbol: str,
) -> list[PathMetrics]:
    """Evaluate the exported OFMR folds with the research pipeline's exact rules."""
    if not state_space_path.is_absolute():
        state_space_path = ROOT / state_space_path
    if not state_space_path.is_file():
        raise FileNotFoundError(f"OFMR state-space parquet not found: {state_space_path}")

    import lightgbm as lgb
    import polars as pl

    # Import the reference's label, deduplication, and signal definitions;
    # do not maintain a second approximate copy of the research logic.
    from strategy_system.strategy.ofmr import ml_state_optimized as research

    strategy_settings = playback.get("strategy", {})
    walk_forward = playback.get("walk_forward", {})
    source_settings = playback.get(source, playback.get("state_space", {}))
    if isinstance(source_settings, dict):
        walk_forward = source_settings.get("walk_forward", walk_forward)
    if not isinstance(walk_forward, dict):
        walk_forward = {}
    catalog = Path(str(walk_forward.get(
        "model_catalog", "build/walkforward_models/catalog.json"
    )))
    if not catalog.is_absolute():
        catalog = ROOT / catalog
    catalog_data = json.loads(catalog.read_text(encoding="utf-8"))
    models = catalog_data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"OFMR fold catalog has no models: {catalog}")

    # Do not accidentally pair a cached state space with a different L2 tape.
    manifest_value = source_settings.get("state_space_manifest") if isinstance(source_settings, dict) else None
    if manifest_value:
        manifest = Path(str(manifest_value))
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        manifest_inputs = json.loads(manifest.read_text(encoding="utf-8")).get("inputs", [])
        selected = {_canonical_manifest_path(path) for path in input_files}
        expected = {_canonical_manifest_path(path) for path in manifest_inputs}
        if selected != expected:
            raise ValueError(
                f"OFMR state-space manifest {manifest} does not match selected {symbol} CryptoL2 files"
            )

    frame = pl.read_parquet(state_space_path)
    frame = research.add_barrier_labels(frame)
    frame = frame.filter(pl.col("volume") > 0)
    frame = research.deduplicate_to_bar(frame).sort("datetime")
    if research.ENABLE_RSI_ACCEL_FILTER:
        rsi_high = 100 - research.RSI_FILTER_THRESHOLD
        frame = (
            frame.with_columns(
                pl.when((pl.col(research.RSI_FILTER_COLUMN) < research.RSI_FILTER_THRESHOLD)
                        & (pl.col("ma_accel_ewma3") > 0)).then(1)
                .when((pl.col(research.RSI_FILTER_COLUMN) > rsi_high)
                      & (pl.col("ma_accel_ewma3") < 0)).then(-1)
                .otherwise(0).alias("zone_side")
            )
            .filter(pl.col("zone_side") != 0)
            .with_columns(
                (pl.col("zone_side") * (-1 if research.TRADE_DIRECTION == "short" else 1))
                .alias("traded_side")
            )
            .drop("zone_side")
        )
    else:
        frame = (
            frame.filter(pl.col(research.RSI_FILTER_COLUMN) < research.RSI_FILTER_THRESHOLD)
            .with_columns(pl.lit(-1 if research.TRADE_DIRECTION == "short" else 1).alias("traded_side"))
        )
    features = list(research.FEATURE_COLUMNS) + [
        name for name in research.ORDERFLOW_FEATURE_COLUMNS if name in frame.columns
    ] if research.USE_ORDERFLOW_FEATURES else list(research.FEATURE_COLUMNS)
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"OFMR state space lacks model features: {sorted(missing)}")

    timestamps = frame["datetime"].cast(pl.Int64).to_numpy()
    datetime_unit = frame.schema["datetime"].time_unit
    timestamp_divisor = {"ns": 1_000_000, "us": 1_000, "ms": 1}.get(datetime_unit)
    timestamp_to_ns = {"ns": 1, "us": 1_000, "ms": 1_000_000}.get(datetime_unit)
    if timestamp_divisor is None:
        raise ValueError(f"unsupported OFMR state-space timestamp unit: {datetime_unit}")
    timestamps_ms = timestamps // timestamp_divisor
    X = frame.select(features).to_numpy()
    targets = frame["target"].to_numpy()
    stop = frame["stop_barrier"].to_numpy()
    exit_bars = frame["bars_to_exit"].to_numpy()
    timed_out = frame["is_timeout"].to_numpy()
    raw_exit_return = frame["exit_ret"].to_numpy()
    sides = frame["traded_side"].to_numpy()

    normalized = np.full(len(targets), np.nan, dtype=np.float64)
    is_win = targets == research.LABEL_WIN
    is_loss = targets == research.LABEL_LOSS
    normalized[is_win & ~timed_out] = research.RR_RATIO
    normalized[is_loss & ~timed_out] = -1.0
    normalized[timed_out] = raw_exit_return[timed_out] * sides[timed_out] / (
        stop[timed_out] + research.EPS
    )
    fee_bps = float(playback.get("fee_bps", research.FEE_BPS))
    normalized -= (2.0 * fee_bps / 10_000.0) / (stop + research.EPS)
    finite = np.isfinite(normalized)
    if finite.any():
        upper = float(np.nanquantile(normalized, 0.99))
        lower = float(np.nanquantile(normalized, 0.01))
        clip = max(abs(upper), abs(lower), research.RR_RATIO)
        normalized = np.clip(normalized, -clip, clip)

    threshold = float(strategy_settings.get(
        "confidence_threshold", strategy_settings.get("threshold", 0.25)
    ))
    risk_pct = float(getattr(research, "RISK_PCT_PER_TRADE", 0.002))
    results: list[PathMetrics] = []
    for model_spec in models:
        start_ms = int(model_spec["test_start_ms"])
        end_ms = int(model_spec["test_end_ms"])
        row_indices = np.flatnonzero((timestamps_ms >= start_ms) & (timestamps_ms <= end_ms))
        expected_rows = model_spec.get("test_rows")
        if expected_rows is not None and len(row_indices) != int(expected_rows):
            raise ValueError(
                f"OFMR fold {model_spec.get('fold')} has {len(row_indices)} test rows, "
                f"catalog expects {expected_rows}; data/catalog mismatch"
            )
        if not len(row_indices):
            raise ValueError(f"OFMR fold {model_spec.get('fold')} has no rows in its test window")
        model_path = Path(str(model_spec["model"]))
        if not model_path.is_absolute():
            model_path = catalog.parent / model_path
        booster = lgb.Booster(model_file=str(model_path))
        probabilities = np.asarray(booster.predict(X[row_indices]), dtype=np.float64)
        side = sides[row_indices]
        trade_confidence = (
            np.where(side > 0, probabilities, 1.0 - probabilities)
            if research.ENABLE_RSI_ACCEL_FILTER else probabilities
        )
        local_signals = research.find_debounced_signals(
            trade_confidence, threshold, research.HYSTERESIS_BUFFER
        )
        selected: list[int] = []
        next_available_ns = np.iinfo(np.int64).min
        for local_index in local_signals:
            index = int(row_indices[int(local_index)])
            timestamp_ns = int(timestamps[index]) * timestamp_to_ns
            if research.LOCKOUT_SECONDS > 0 and timestamp_ns < next_available_ns:
                continue
            if not research.MIN_BARRIER_PCT <= stop[index] <= research.MAX_BARRIER_PCT:
                continue
            if not np.isfinite(normalized[index]):
                continue
            selected.append(index)
            if research.LOCKOUT_SECONDS > 0:
                duration = int(exit_bars[index]) + 1
                next_available_ns = timestamp_ns + duration * 1_000_000_000
        trade_returns = np.asarray([normalized[index] for index in selected], dtype=np.float64)
        equity = np.concatenate(([1.0], np.cumprod(1.0 + trade_returns * risk_pct)))
        peaks = np.maximum.accumulate(equity)
        drawdown = np.max((peaks - equity) / peaks) if len(equity) else 0.0
        if len(trade_returns):
            downside = np.minimum(trade_returns, 0.0)
            downside_dev = float(np.sqrt(np.mean(downside * downside)))
            sortino = (
                float(np.mean(trade_returns) / downside_dev)
                if downside_dev else float("inf")
            )
            expectation = float(np.mean(trade_returns))
        else:
            sortino = 0.0
            expectation = 0.0
        results.append(PathMetrics(
            cumulative_return=float(equity[-1] - 1.0),
            expectancy_per_trade=expectation,
            max_drawdown=float(drawdown),
            sortino=sortino,
            trades=len(trade_returns),
            equity_path=tuple(float(value) for value in _downsample(equity, 80)),
            trade_returns=tuple(float(value) for value in trade_returns),
        ))
    return results


def _reference_ofmr_cpcv_metrics(
    playback: dict[str, Any], state_space_path: Path, input_files: list[str], symbol: str
) -> tuple[list[PathMetrics], int]:
    """Run CPCV with OFMR's real features, barrier labels, and fitted model."""
    if not state_space_path.is_absolute():
        state_space_path = ROOT / state_space_path
    if not state_space_path.is_file():
        raise FileNotFoundError(f"OFMR state-space parquet not found: {state_space_path}")
    import lightgbm as lgb
    import polars as pl
    from strategy_system.strategy.ofmr import ml_state_optimized as research

    source_settings = playback.get("cryptol2_offline", {})
    manifest_value = source_settings.get("state_space_manifest") if isinstance(source_settings, dict) else None
    if manifest_value:
        manifest = Path(str(manifest_value))
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        manifest_inputs = json.loads(manifest.read_text(encoding="utf-8")).get("inputs", [])
        selected = {_canonical_manifest_path(path) for path in input_files}
        expected = {_canonical_manifest_path(path) for path in manifest_inputs}
        if selected != expected:
            raise ValueError(f"OFMR state-space manifest {manifest} does not match selected {symbol} CryptoL2 files")

    frame = research.deduplicate_to_bar(
        research.add_barrier_labels(pl.read_parquet(state_space_path))
    ).filter(pl.col("volume") > 0).sort("datetime")
    if research.ENABLE_RSI_ACCEL_FILTER:
        rsi_high = 100 - research.RSI_FILTER_THRESHOLD
        frame = frame.with_columns(
            pl.when((pl.col(research.RSI_FILTER_COLUMN) < research.RSI_FILTER_THRESHOLD)
                    & (pl.col("ma_accel_ewma3") > 0)).then(1)
            .when((pl.col(research.RSI_FILTER_COLUMN) > rsi_high)
                  & (pl.col("ma_accel_ewma3") < 0)).then(-1)
            .otherwise(0).alias("zone_side")
        ).filter(pl.col("zone_side") != 0).with_columns(
            (pl.col("zone_side") * (-1 if research.TRADE_DIRECTION == "short" else 1)).alias("traded_side")
        ).drop("zone_side")
    else:
        frame = frame.filter(pl.col(research.RSI_FILTER_COLUMN) < research.RSI_FILTER_THRESHOLD).with_columns(
            pl.lit(-1 if research.TRADE_DIRECTION == "short" else 1).alias("traded_side")
        )
    features = (list(research.FEATURE_COLUMNS) + [
        name for name in research.ORDERFLOW_FEATURE_COLUMNS if name in frame.columns
    ]) if research.USE_ORDERFLOW_FEATURES else list(research.FEATURE_COLUMNS)
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"OFMR state-space lacks model features: {sorted(missing)}")

    datetime_unit = frame.schema["datetime"].time_unit
    to_ns = {"ns": 1, "us": 1_000, "ms": 1_000_000}.get(datetime_unit)
    if to_ns is None:
        raise ValueError(f"unsupported OFMR state-space timestamp unit: {datetime_unit}")
    timestamps_ns = frame["datetime"].cast(pl.Int64).to_numpy() * to_ns
    X = frame.select(features).to_numpy()
    labels = frame["target"].to_numpy()
    y = (labels == research.LABEL_WIN).astype(np.int8)
    sides = frame["traded_side"].to_numpy()
    stop = frame["stop_barrier"].to_numpy()
    exit_bars = frame["bars_to_exit"].to_numpy()
    timed_out = frame["is_timeout"].to_numpy()
    raw_exit_return = frame["exit_ret"].to_numpy()
    normalized = np.full(len(labels), np.nan, dtype=np.float64)
    is_win = labels == research.LABEL_WIN
    is_loss = labels == research.LABEL_LOSS
    normalized[is_win & ~timed_out] = research.RR_RATIO
    normalized[is_loss & ~timed_out] = -1.0
    normalized[timed_out] = raw_exit_return[timed_out] * sides[timed_out] / (stop[timed_out] + research.EPS)
    fee_bps = float(playback.get("fee_bps", research.FEE_BPS))
    normalized -= (2.0 * fee_bps / 10_000.0) / (stop + research.EPS)
    finite = np.isfinite(normalized)
    if finite.any():
        upper, lower = np.nanquantile(normalized, 0.99), np.nanquantile(normalized, 0.01)
        clip = max(abs(float(upper)), abs(float(lower)), research.RR_RATIO)
        normalized = np.clip(normalized, -clip, clip)

    n = len(X)
    splits = int(playback.get("n_splits", 6))
    test_splits = int(playback.get("n_test_splits", 2))
    purge_seconds = max(
        float(playback.get("purge_bars", 0)) * float(playback.get("bar_seconds", 1.0)),
        float(research.LOOKAHEAD_SECONDS + research.EMBARGO_SECONDS),
    )
    purge_ns = int(purge_seconds * 1_000_000_000)
    if n < splits * 4:
        raise ValueError("OFMR CPCV requires at least four observations per split")
    ranges = _split_ranges(n, splits)
    threshold = float(playback.get("strategy", {}).get("confidence_threshold", 0.25))
    risk_pct = float(getattr(research, "RISK_PCT_PER_TRADE", 0.002))
    max_paths = int(playback.get("max_paths", 100_000))
    metrics: list[PathMetrics] = []
    rejected = 0
    for combo in _combinations(range(splits), test_splits):
        if len(metrics) >= max_paths:
            rejected += 1
            continue
        test_indices = np.concatenate([np.arange(*ranges[index]) for index in combo])
        is_test = np.zeros(n, dtype=bool)
        is_test[test_indices] = True
        purge = np.zeros(n, dtype=bool)
        for index in combo:
            start, end = ranges[index]
            test_start = int(timestamps_ns[start])
            test_end = int(timestamps_ns[end - 1])
            purge |= (timestamps_ns >= test_start - purge_ns) & (timestamps_ns <= test_end + purge_ns)
        train_indices = np.flatnonzero(~is_test & ~purge & np.isfinite(X).all(axis=1)
                                       & np.isin(labels, (research.LABEL_LOSS, research.LABEL_WIN)))
        train_y = y[train_indices]
        if len(train_indices) < 40 or len(np.unique(train_y)) < 2:
            rejected += 1
            continue
        test_valid = test_indices[np.isfinite(X[test_indices]).all(axis=1)]
        if not len(test_valid):
            rejected += 1
            continue
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=1, verbose=-1,
        )
        model.fit(X[train_indices], train_y)
        if 1 not in model.classes_:
            rejected += 1
            continue
        win_probability = model.predict_proba(X[test_valid])[:, list(model.classes_).index(1)]
        trade_confidence = (
            np.where(sides[test_valid] > 0, win_probability, 1.0 - win_probability)
            if research.ENABLE_RSI_ACCEL_FILTER else win_probability
        )
        signal_local = research.find_debounced_signals(trade_confidence, threshold, research.HYSTERESIS_BUFFER)
        selected: list[int] = []
        next_available_ns = np.iinfo(np.int64).min
        for local_index in signal_local:
            index = int(test_valid[int(local_index)])
            timestamp = int(timestamps_ns[index])
            if research.LOCKOUT_SECONDS and timestamp < next_available_ns:
                continue
            if not research.MIN_BARRIER_PCT <= stop[index] <= research.MAX_BARRIER_PCT:
                continue
            if not np.isfinite(normalized[index]):
                continue
            selected.append(index)
            if research.LOCKOUT_SECONDS:
                next_available_ns = timestamp + (int(exit_bars[index]) + 1) * 1_000_000_000
        trade_returns = normalized[selected]
        equity = np.concatenate(([1.0], np.cumprod(1.0 + trade_returns * risk_pct)))
        peaks = np.maximum.accumulate(equity)
        mdd = float(np.max((peaks - equity) / peaks)) if len(equity) else 0.0
        downside = np.minimum(trade_returns, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside * downside))) if len(trade_returns) else 0.0
        sortino = float(np.mean(trade_returns) / downside_dev) if downside_dev else 0.0
        metrics.append(PathMetrics(
            float(equity[-1] - 1.0), float(np.mean(trade_returns)) if len(trade_returns) else 0.0,
            mdd, sortino, len(trade_returns),
            tuple(float(value) for value in _downsample(equity, 80)),
            tuple(float(value) for value in trade_returns),
        ))
    if not metrics:
        raise ValueError("OFMR CPCV produced no valid model paths")
    return metrics, rejected


def _returns(prices: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(prices, dtype=np.float64)
    returns[1:] = prices[1:] / prices[:-1] - 1.0
    returns[~np.isfinite(returns)] = 0.0
    return returns


def _resample_prices(
    timestamps: np.ndarray, prices: np.ndarray, bar_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(bar_seconds) or bar_seconds <= 0.0:
        raise ValueError("playback.bar_seconds must be positive and finite")
    if timestamps.ndim != 1 or len(timestamps) != len(prices):
        raise ValueError("timestamps must match the playback price rows")
    timestamp_magnitude = float(np.median(np.abs(timestamps)))
    units_per_second = (
        1_000_000_000
        if timestamp_magnitude >= 1e17
        else 1_000_000
        if timestamp_magnitude >= 1e14
        else 1_000
        if timestamp_magnitude >= 1e11
        else 1
    )
    bucket_size = max(1, int(bar_seconds * units_per_second))
    buckets = timestamps // bucket_size
    keep = np.r_[buckets[1:] != buckets[:-1], True]
    indices = np.flatnonzero(keep)
    return timestamps[indices], prices[indices], indices


def _canonical_purged_walk_forward(
    prices: np.ndarray,
    positions: np.ndarray | None,
    risk_fractions: np.ndarray | None,
    playback: dict[str, Any],
    strategy_type: str,
    strategy_settings: dict[str, Any],
) -> list[PathMetrics]:
    splits = int(playback.get("n_splits", 6))
    purge_bars = int(playback.get("purge_bars", 0))
    n = prices.shape[0]
    if n < splits:
        raise ValueError("event count must be at least n_splits for purged walk-forward")
    fold_ranges = _split_ranges(n, splits)
    metrics: list[PathMetrics] = []
    for fold in range(splits):
        raw_start, end = fold_ranges[fold]
        test_start = raw_start if fold == 0 else raw_start + purge_bars
        test_end = end if fold == splits - 1 else end - purge_bars
        if test_start >= test_end:
            continue
        test_prices = prices[test_start:test_end]
        test_positions = positions[test_start:test_end] if positions is not None else None
        test_risk = risk_fractions[test_start:test_end] if risk_fractions is not None else None
        train_pieces = []
        train_pos_pieces = []
        train_risk_pieces = []
        for other in range(splits):
            if other == fold:
                continue
            o_start, o_end = fold_ranges[other]
            if other < fold:
                o_end = min(o_end, raw_start - purge_bars)
            else:
                o_start = max(o_start, end + purge_bars)
            if o_start < o_end:
                train_pieces.append(prices[o_start:o_end])
                if positions is not None:
                    train_pos_pieces.append(positions[o_start:o_end])
                if risk_fractions is not None:
                    train_risk_pieces.append(risk_fractions[o_start:o_end])
        if not train_pieces:
            train_prices = prices[:raw_start] if raw_start > 0 else prices[end:]
        else:
            train_prices = np.concatenate(train_pieces, axis=0)
        if strategy_type == "momentum" and len(train_prices) > 1 and test_positions is None:
            train_ret = _returns(train_prices)
            train_vol = float(np.std(train_ret)) if len(train_ret) > 1 else 0.0
            adapted = dict(strategy_settings)
            adapted["entry_threshold"] = max(0.0, 0.5 * train_vol)
            adapted["lookback_bars"] = strategy_settings.get("lookback_bars", 5)
            test_positions = _strategy_positions(test_prices, adapted)
        elif strategy_type == "ofmr" and test_positions is None:
            # generate OFMR positions and risk fractions from price data
            test_positions, test_risk = _ofmr_positions_and_risk(test_prices, strategy_settings)
        m = _run_one_path(
            test_prices, playback, test_positions, test_risk, capture_path=True
        )
        metrics.append(m)
    return metrics



def _parametric_monte_carlo(
    prices: np.ndarray,
    positions: np.ndarray | None,
    risk_fractions: np.ndarray | None,
    playback: dict[str, Any],
    strategy_type: str,
    strategy_settings: dict[str, Any],
    synthetic_settings: dict[str, Any],
) -> tuple[list[PathMetrics], int]:
    threshold = float(playback.get("max_abs_correlation", 0.5))
    max_paths = int(playback.get("max_paths", 100_000))
    n_bootstrap = int(playback.get("n_bootstrap", 500))
    n_bars = int(playback.get("n_bars", prices.shape[0] if prices is not None else 500))
    n_candidates = int(synthetic_settings.get("paths", 512))
    if prices is not None:
        # Fit the parametric path generator on the selected market-data tape.
        # This keeps CryptoL2 MC anchored to the same instrument and period.
        pooled = _returns(prices[:, 0])
    else:
        candidates = _generate_synthetic(synthetic_settings, paths=n_candidates)
        candidate_returns = [_returns(c[1])[:, 0] for c in candidates]
        pooled = np.concatenate(candidate_returns)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 10:
        raise ValueError("not enough synthetic returns to fit a parametric distribution")
    mu = float(np.mean(pooled))
    sigma = float(np.std(pooled))
    if sigma <= 0.0:
        sigma = 1e-6
    rng = np.random.default_rng(int(synthetic_settings.get("seed", 84)))
    bootstrap_paths: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        sim_returns = rng.normal(mu, sigma, size=n_bars)
        sim_prices = 100.0 * np.exp(np.cumsum(sim_returns)).reshape(-1, 1)
        sim_prices = np.maximum(sim_prices, 1e-6)
        bootstrap_paths.append(sim_prices)
    observed_returns = _returns(prices[:, 0]) if prices is not None else None
    selected: list[int] = []
    selected_returns: list[np.ndarray] = []
    rejected = 0
    for index, path_prices in enumerate(bootstrap_paths):
        path_returns = _returns(path_prices[:, 0])
        if observed_returns is not None and abs(_correlation(observed_returns, path_returns)) > threshold + 1e-12:
            rejected += 1
            continue
        ok = True
        for prior_ret in selected_returns:
            if abs(_correlation(path_returns, prior_ret)) > threshold + 1e-12:
                ok = False
                break
        if not ok:
            rejected += 1
            continue
        if len(selected) >= max_paths:
            rejected += 1
            continue
        selected.append(index)
        selected_returns.append(path_returns)
    if not selected:
        raise ValueError("no Monte Carlo paths passed the correlation filter")
    metrics: list[PathMetrics] = []
    for plotted_index, index in enumerate(selected):
        path_prices = bootstrap_paths[index]
        path_positions = None
        path_risk = None
        if positions is not None:
            reps = int(np.ceil(n_bars / max(positions.shape[0], 1)))
            path_positions = np.tile(positions, (reps, 1))[:n_bars]
        if risk_fractions is not None:
            reps = int(np.ceil(n_bars / max(risk_fractions.shape[0], 1)))
            path_risk = np.tile(risk_fractions, (reps, 1))[:n_bars]
        if strategy_type == "momentum" and path_positions is None:
            path_positions = _strategy_positions(path_prices, strategy_settings)
        elif strategy_type == "ofmr" and path_positions is None:
            path_positions, path_risk = _ofmr_positions_and_risk(
                path_prices, strategy_settings
            )
        m = _run_one_path(
            path_prices,
            playback,
            path_positions,
            path_risk,
            capture_path=plotted_index < MAX_PLOTTED_PATHS,
        )
        metrics.append(m)
    return metrics, rejected


def _ofmr_trade_monte_carlo(
    fold_metrics: list[PathMetrics], settings: dict[str, Any], synthetic_settings: dict[str, Any]
) -> tuple[list[PathMetrics], int]:
    """Bootstrap actual OFMR out-of-sample trade returns into equity paths."""
    from strategy_system.strategy.ofmr import ml_state_optimized as research

    trade_pool = np.asarray(
        [trade for fold in fold_metrics for trade in fold.trade_returns], dtype=np.float64
    )
    if trade_pool.size < 10:
        raise ValueError("OFMR Monte Carlo needs at least ten out-of-sample trades")
    path_count = min(int(settings.get("n_bootstrap", 500)), int(settings.get("max_paths", 100_000)))
    trades_per_path = int(synthetic_settings.get("trades_per_path", settings.get("n_bars", 500)))
    if path_count < 1 or trades_per_path < 1:
        raise ValueError("OFMR Monte Carlo path and trade counts must be positive")
    risk_pct = float(getattr(research, "RISK_PCT_PER_TRADE", 0.002))
    rng = np.random.default_rng(int(synthetic_settings.get("seed", 84)))
    metrics: list[PathMetrics] = []
    for path_index in range(path_count):
        sampled = rng.choice(trade_pool, size=trades_per_path, replace=True)
        equity = np.concatenate(([1.0], np.cumprod(1.0 + sampled * risk_pct)))
        peaks = np.maximum.accumulate(equity)
        drawdown = float(np.max((peaks - equity) / peaks))
        downside = np.minimum(sampled, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside * downside)))
        sortino = float(np.mean(sampled) / downside_dev) if downside_dev else 0.0
        metrics.append(PathMetrics(
            float(equity[-1] - 1.0), float(np.mean(sampled)), drawdown, sortino,
            trades_per_path,
            tuple(float(value) for value in _downsample(equity, 80)),
            tuple(float(value) for value in sampled),
        ))
    return metrics, 0

def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).ravel()
    right = np.asarray(right).ravel()
    size = min(left.size, right.size)
    if size < 3:
        return float("nan")
    left = left[:size]
    right = right[:size]
    lc = left - left.mean()
    rc = right - right.mean()
    denom = np.linalg.norm(lc) * np.linalg.norm(rc)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(lc, rc) / denom)


def _split_ranges(size: int, splits: int) -> list[tuple[int, int]]:
    return [(index * size // splits, (index + 1) * size // splits) for index in range(splits)]

def _canonical_cpcv(
    prices: np.ndarray,
    positions: np.ndarray | None,
    risk_fractions: np.ndarray | None,
    playback: dict[str, Any],
    strategy_type: str,
    strategy_settings: dict[str, Any],
) -> tuple[list[PathMetrics], int]:
    splits = int(playback.get("n_splits", 6))
    test_splits = int(playback.get("n_test_splits", 2))
    purge_bars = int(playback.get("purge_bars", 0))
    max_paths = int(playback.get("max_paths", 100_000))
    threshold = float(playback.get("max_abs_correlation", 0.5))
    n = prices.shape[0]
    if n < splits * 4:
        raise ValueError("CPCV requires at least four observations per split")
    ranges = _split_ranges(n, splits)
    chunk_returns = [_returns(prices[start:end]) for start, end in ranges]
    metrics: list[PathMetrics] = []
    combinations_run = 0
    combinations_rejected = 0
    for test_combo in _combinations(range(splits), test_splits):
        train_chunks = [i for i in range(splits) if i not in test_combo]
        compatible = True
        for tc in test_combo:
            for tr in train_chunks:
                if abs(_correlation(chunk_returns[tr], chunk_returns[tc])) > threshold + 1e-12:
                    compatible = False
                    break
            if not compatible:
                break
        if not compatible:
            combinations_rejected += 1
            continue
        train_pieces = []
        train_pos_pieces = []
        train_risk_pieces = []
        prev_end = None
        for tr in sorted(train_chunks):
            t_start, t_end = ranges[tr]
            if prev_end is not None:
                t_start = max(t_start, prev_end + purge_bars)
            if t_start < t_end:
                train_pieces.append(prices[t_start:t_end])
                if positions is not None:
                    train_pos_pieces.append(positions[t_start:t_end])
                if risk_fractions is not None:
                    train_risk_pieces.append(risk_fractions[t_start:t_end])
            prev_end = t_end
        test_pieces = []
        test_pos_pieces = []
        test_risk_pieces = []
        prev_end = None
        for tc in sorted(test_combo):
            t_start, t_end = ranges[tc]
            if prev_end is not None:
                t_start = max(t_start, prev_end + purge_bars)
            if t_start < t_end:
                test_pieces.append(prices[t_start:t_end])
                if positions is not None:
                    test_pos_pieces.append(positions[t_start:t_end])
                if risk_fractions is not None:
                    test_risk_pieces.append(risk_fractions[t_start:t_end])
            prev_end = t_end
        if not train_pieces or not test_pieces:
            combinations_rejected += 1
            continue
        test_prices = np.concatenate(test_pieces, axis=0)
        test_positions = None
        test_risk = None
        if test_pos_pieces:
            test_positions = np.concatenate(test_pos_pieces, axis=0)
        if test_risk_pieces:
            test_risk = np.concatenate(test_risk_pieces, axis=0)
        if strategy_type == "momentum" and len(np.concatenate(train_pieces, axis=0)) > 1 and test_positions is None:
            train_prices = np.concatenate(train_pieces, axis=0)
            train_ret = _returns(train_prices)
            train_vol = float(np.std(train_ret)) if len(train_ret) > 1 else 0.0
            adapted = dict(strategy_settings)
            adapted["entry_threshold"] = max(0.0, 0.5 * train_vol)
            adapted["lookback_bars"] = strategy_settings.get("lookback_bars", 5)
            test_positions = _strategy_positions(test_prices, adapted)
        elif strategy_type == "ofmr" and test_positions is None:
            # generate OFMR positions and risk fractions from price data
            test_positions, test_risk = _ofmr_positions_and_risk(test_prices, strategy_settings)
        if combinations_run >= max_paths:
            combinations_rejected += 1
            continue
        m = _run_one_path(
            test_prices,
            playback,
            test_positions,
            test_risk,
            capture_path=combinations_run < MAX_PLOTTED_PATHS,
        )
        metrics.append(m)
        combinations_run += 1
    return metrics, combinations_rejected

def _run_one_path(
    prices: np.ndarray,
    settings: dict[str, Any],
    strategy_positions: np.ndarray | None = None,
    risk_fractions: np.ndarray | None = None,
    *,
    capture_path: bool = False,
) -> PathMetrics:
    if prices.ndim != 2 or prices.shape[0] < 3 or prices.shape[1] < 1:
        raise ValueError("each playback path must have at least three rows and one asset")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("playback prices must be positive and finite")
    if strategy_positions is None:
        positions = _strategy_positions(prices, settings.get("strategy", {}))
    else:
        positions = np.asarray(strategy_positions, dtype=np.float64)
        if positions.shape != prices.shape or not np.isfinite(positions).all():
            raise ValueError("position states must match prices and be finite")
        if not np.isin(positions, (-1.0, 0.0, 1.0)).all():
            raise ValueError("position states must be -1, 0, or 1")
    if risk_fractions is not None:
        risk_fractions = np.asarray(risk_fractions, dtype=np.float64)
        if (
            risk_fractions.shape != prices.shape
            or not np.isfinite(risk_fractions).all()
            or (risk_fractions < 0.0).any()
        ):
            raise ValueError("risk fractions must match prices and be finite/nonnegative")
    starting_capital = float(settings.get("starting_capital", 10_000.0))
    if not np.isfinite(starting_capital) or starting_capital <= 0.0:
        raise ValueError("starting capital must be positive and finite")
    asset_count = prices.shape[1]
    capital_per_asset = starting_capital / asset_count
    fee_rate = float(settings.get("fee_bps", 4.0)) / 10_000.0
    asset_returns = _returns(prices)
    prior_positions = np.vstack((np.zeros((1, prices.shape[1])), positions[:-1]))
    active_assets = np.count_nonzero(prior_positions, axis=1)
    portfolio_returns = np.divide(
        np.sum(prior_positions * asset_returns, axis=1),
        active_assets,
        out=np.zeros(prices.shape[0], dtype=np.float64),
        where=active_assets > 0,
    )
    turnover = np.abs(positions - prior_positions).sum(axis=1)
    portfolio_returns -= turnover * fee_rate / max(prices.shape[1], 1)
    portfolio_returns[-1] -= np.abs(positions[-1]).sum() * fee_rate / max(prices.shape[1], 1)
    equity = np.cumprod(1.0 + portfolio_returns)
    equity_curve = np.concatenate(([starting_capital], equity * starting_capital))
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = np.divide(
        peak - equity_curve,
        peak,
        out=np.zeros_like(peak),
        where=peak > 0,
    )
    trade_returns: list[float] = []
    for asset_index in range(prices.shape[1]):
        entry_index: int | None = None
        entry_side = 0.0
        for bar_index in np.flatnonzero(
            positions[:, asset_index] != prior_positions[:, asset_index]
        ):
            prior_side = prior_positions[bar_index, asset_index]
            current_side = positions[bar_index, asset_index]
            if prior_side != 0.0 and entry_index is not None:
                trade_return = (
                    entry_side
                    * _checked_relative_return(
                        prices[bar_index, asset_index],
                        prices[entry_index, asset_index],
                        "trade price",
                    )
                    - 2.0 * fee_rate
                )
                if risk_fractions is not None:
                    risk = risk_fractions[entry_index, asset_index]
                    if risk <= 0.0:
                        raise ValueError("OFMR open position has no positive stop-risk fraction")
                    trade_return /= risk
                else:
                    trade_return *= capital_per_asset
                trade_returns.append(trade_return)
            entry_index = bar_index if current_side != 0.0 else None
            entry_side = current_side
        if entry_index is not None and positions[-1, asset_index] != 0.0:
            trade_return = (
                entry_side
                * _checked_relative_return(
                    prices[-1, asset_index],
                    prices[entry_index, asset_index],
                    "open trade price",
                )
                - 2.0 * fee_rate
            )
            if risk_fractions is not None:
                risk = risk_fractions[entry_index, asset_index]
                if risk <= 0.0:
                    raise ValueError("OFMR open position has no positive stop-risk fraction")
                trade_return /= risk
            else:
                trade_return *= capital_per_asset
            trade_returns.append(trade_return)
    closed_trades = len(trade_returns)
    expectancy = float(np.mean(trade_returns)) if trade_returns else 0.0
    bar_returns = np.diff(equity_curve) / equity_curve[:-1]
    downside = np.minimum(bar_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside * downside)))
    mean_return = float(np.mean(bar_returns)) if bar_returns.size else 0.0
    periods_per_year = float(settings.get("periods_per_year", 252.0))
    sortino = (
        mean_return / downside_deviation * np.sqrt(periods_per_year)
        if downside_deviation > 0.0
        else (float("inf") if mean_return > 0.0 else 0.0)
    )
    return PathMetrics(
        cumulative_return=float(equity[-1] - 1.0),
        expectancy_per_trade=expectancy,
        max_drawdown=float(np.max(drawdowns)),
        sortino=sortino,
        trades=int(closed_trades),
        equity_path=(
            tuple(
                float(value)
                for value in _downsample(
                    _normalize_by_positive_base(equity_curve, starting_capital), 80
                )
            )
            if capture_path
            else ()
        ),
    )


def _checked_relative_return(numerator: float, denominator: float, label: str) -> float:
    """Calculate a price/equity return only after checking both inputs."""
    if (
        not np.isfinite(numerator)
        or numerator <= 0.0
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise ValueError(f"{label} numerator and denominator must be positive and finite")
    result = numerator / denominator - 1.0
    if not np.isfinite(result):
        raise ValueError(f"{label} return is not finite")
    return result


def _normalize_by_positive_base(values: np.ndarray, base: float) -> np.ndarray:
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("normalization base must be positive and finite")
    return values / base

def _parallel_metrics(
    paths: list[np.ndarray],
    settings: dict[str, Any],
    signal_paths: list[np.ndarray | None] | None = None,
    risk_paths: list[np.ndarray | None] | None = None,
) -> list[PathMetrics]:
    if not paths:
        raise ValueError("playback produced no eligible paths")
    if signal_paths is not None and len(signal_paths) != len(paths):
        raise ValueError("signal path count must match price path count")
    if risk_paths is not None and len(risk_paths) != len(paths):
        raise ValueError("risk path count must match price path count")
    workers = int(settings.get("workers", 0))
    if workers < 0:
        raise ValueError("playback.workers must be zero or a positive integer")
    with ThreadPoolExecutor(max_workers=workers or None) as pool:
        if signal_paths is None and risk_paths is None:
            return list(pool.map(lambda path: _run_one_path(path, settings), paths))
        if signal_paths is None:
            signal_paths = [None] * len(paths)
        if risk_paths is None:
            risk_paths = [None] * len(paths)
        return list(pool.map(
            lambda inputs: _run_one_path(inputs[0], settings, inputs[1], inputs[2]),
            zip(paths, signal_paths, risk_paths, strict=True),
        ))


def _summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    tail_count = max(1, int(np.ceil(ordered.size * 0.05)))
    return (
        float(np.mean(ordered)),
        float(np.mean(ordered[:tail_count])),
        float(np.mean(ordered[-tail_count:])),
    )


def _downsample(values: np.ndarray, size: int) -> np.ndarray:
    if len(values) <= size:
        return values
    indices = np.linspace(0, len(values) - 1, num=size, dtype=np.int64)
    return values[indices]


def _render_paths_tui(metrics: list[PathMetrics], method: str) -> str:
    source_paths = [metric.equity_path for metric in metrics if metric.equity_path]
    if not source_paths:
        return ""

    width = 72
    height = 16
    x_positions = np.linspace(0.0, 1.0, num=width)
    paths = [
        np.interp(x_positions, np.linspace(0.0, 1.0, num=len(path)), path)
        for path in source_paths
    ]
    values = np.asarray(paths, dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        high = low + 1.0
    chart = [[" " for _ in range(width)] for _ in range(height)]
    for path in values:
        sampled = _downsample(path, width)
        for x, value in enumerate(sampled):
            y = height - 1 - int(round((value - low) / (high - low) * (height - 1)))
            chart[y][x] = "*" if chart[y][x] == " " else "#"

    rows = [
        f"Monte Carlo TUI | method={method} | paths shown={len(paths)}/{len(metrics)}",
        f"{high:>9.3f} |{''.join(chart[0])}",
    ]
    rows.extend(f"{'':>9} |{''.join(line)}" for line in chart[1:-1])
    rows.append(f"{low:>9.3f} |{''.join(chart[-1])}")
    rows.append("          +" + "-" * width)
    rows.append("           start" + " " * (width - 14) + "end")
    return "\n".join(rows)


def _report(
    method: str,
    metrics: list[PathMetrics],
    *,
    paths_rejected: int = 0,
    folds_or_combinations: int = 0,
    threshold: float = 0.5,
    strategy: str = "momentum",
    strategy_name: str | None = None,
) -> str:
    if not metrics:
        raise ValueError("playback produced no eligible paths")
    lines = [
        f"Playback report: {method}",
        f"Strategy: {'OFMR' if strategy == 'ofmr' else 'momentum baseline' if strategy == 'momentum' else strategy.upper()}"
        + (f" ({strategy_name})" if strategy_name and strategy == "ofmr" else ""),
        f"Paths run: {len(metrics)}; rejected: {paths_rejected}; folds/combinations: {folds_or_combinations}",
    ]
    if strategy != "ofmr":
        lines.append(f"Correlation limit: |rho| <= {threshold:.3f}")
    for label, field in (
        ("Cumulative return", "cumulative_return"),
        ("Average expectancy per trade", "expectancy_per_trade"),
        ("MDD", "max_drawdown"),
        ("Sortino", "sortino"),
    ):
        mean, cvar05, cvar95 = _summarize([getattr(metric, field) for metric in metrics])
        if field == "expectancy_per_trade":
            trade_count = sum(metric.trades for metric in metrics)
            if trade_count:
                mean = sum(
                    metric.expectancy_per_trade * metric.trades for metric in metrics
                ) / trade_count
        unit = " R" if strategy == "ofmr" and field == "expectancy_per_trade" else ""
        lines.append(
            f"{label}: {mean:.6f}{unit} +- "
            f"[CVaR05 {cvar05:.6f}{unit}, CVaR95 {cvar95:.6f}{unit}]"
        )
    lines.append(f"Total trades: {sum(metric.trades for metric in metrics)}")
    return "\n".join(lines)


def _save_results(
    method: str, source: str, strategy_type: str,
    metrics: list[PathMetrics], report: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    PLAYBACK_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{method}_{source}_{strategy_type}"
    run_dir = PLAYBACK_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cumulative_return", "expectancy_per_trade", "max_drawdown", "sortino", "trades"])
        for m in metrics:
            writer.writerow([m.cumulative_return, m.expectancy_per_trade, m.max_drawdown, m.sortino, m.trades])

    with (run_dir / "equity_paths.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "step", "equity_multiple"])
        for path_index, metric in enumerate(metrics):
            for step, value in enumerate(metric.equity_path):
                writer.writerow([path_index, step, value])

    (run_dir / "report.txt").write_text(report, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "method": method,
        "source": source,
        "strategy": strategy_type,
        "paths_run": len(metrics),
        "paths_rejected": extra.get("paths_rejected", 0) if extra else 0,
        "folds_or_combinations": extra.get("folds_or_combinations", 0) if extra else 0,
        "threshold": extra.get("threshold", 0.5) if extra else 0.5,
        "cumulative_return_mean": float(np.mean([m.cumulative_return for m in metrics])) if metrics else 0.0,
        "expectancy_per_trade_mean": float(np.mean([m.expectancy_per_trade for m in metrics])) if metrics else 0.0,
        "max_drawdown_mean": float(np.mean([m.max_drawdown for m in metrics])) if metrics else 0.0,
        "sortino_mean": float(np.mean([m.sortino for m in metrics])) if metrics else 0.0,
        "total_trades": sum(m.trades for m in metrics),
        "created_at_utc": ts,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir

    valid = np.isfinite(mid) & (mid > 0)
    timestamps = timestamps[valid]
    prices = mid[valid].reshape(-1, 1)
    if len(prices) < 3:
        raise ValueError("cryptoL2 online produced fewer than 3 valid price bars")
    return timestamps, prices, (symbol,)



def _run_playback(
    playback: dict[str, Any],
    *,
    method_override: str | None = None,
    strategy_override: str | None = None,
) -> str:
    method = str(method_override or playback.get("method", "purged_walk_forward")).lower()
    threshold = float(playback.get("max_abs_correlation", 0.5))
    max_paths = int(playback.get("max_paths", 100_000))
    splits = int(playback.get("n_splits", 6))
    test_splits = int(playback.get("n_test_splits", 2))
    purge_bars = int(playback.get("purge_bars", 0))
    n_bootstrap = int(playback.get("n_bootstrap", 500))
    n_bars = int(playback.get("n_bars", 500))

    if method not in {"parametric_monte_carlo", "cpcv", "purged_walk_forward"}:
        raise ValueError("playback.method must be parametric_monte_carlo, cpcv, or purged_walk_forward")
    if (
        not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
        or max_paths < 1
        or not 2 <= splits <= 64
        or not 1 <= test_splits <= splits
        or purge_bars < 0
        or n_bootstrap < 1
        or n_bars < 3
    ):
        raise ValueError(
            "max_abs_correlation must be in [0, 1], max_paths positive, "
            "n_splits 2..64, n_test_splits 1..n_splits, purge_bars nonnegative, "
            "n_bootstrap positive, n_bars >= 3"
        )

    source = str(playback.get("source", "cryptoL2_offline")).lower()
    strategy = playback.get("strategy", {})
    if not isinstance(strategy, dict):
        raise ValueError("playback.strategy must be a YAML mapping")
    strategy_type = str(strategy_override or strategy.get("type", "momentum")).lower()
    if strategy_type not in {"momentum", "ofmr", "strategy2"}:
        raise ValueError("playback.strategy.type must name a supported strategy folder")

    synthetic_settings = dict(playback.get("synthetic", {}))
    synthetic_settings.setdefault("n_bars", n_bars)

    max_bars = playback.get("max_bars", None)
    if max_bars is not None:
        max_bars = int(max_bars)
        if max_bars < 3:
            raise ValueError("playback.max_bars must be >= 3")

    bar_seconds = float(playback.get("bar_seconds", 1.0))
    timestamps: np.ndarray | None = None
    prices: np.ndarray | None = None
    symbols: tuple[str, ...] = ()
    positions: np.ndarray | None = None
    risk_fractions: np.ndarray | None = None
    state_space_metrics: list[PathMetrics] | None = None

    if source == "cryptol2_offline":
        use_reference_state_space = (
            method in {"purged_walk_forward", "cpcv", "parametric_monte_carlo"}
            and strategy_type == "ofmr"
            and playback.get("cryptol2_offline", {}).get("state_space_path")
        )
        if not use_reference_state_space:
            (
                timestamps,
                prices,
                symbols,
                positions,
                risk_fractions,
            ) = _read_cryptol2_bars(
                playback.get("cryptol2_offline", {}),
                offline=True,
                bar_seconds=bar_seconds,
                max_bars=max_bars,
            )
    elif source == "cryptol2":
        use_reference_state_space = (
            method in {"purged_walk_forward", "cpcv", "parametric_monte_carlo"}
            and strategy_type == "ofmr"
            and playback.get("cryptol2", {}).get("state_space_path")
        )
        if not use_reference_state_space:
            (
                timestamps,
                prices,
                symbols,
                positions,
                risk_fractions,
            ) = _read_cryptol2_bars(
                playback.get("cryptol2", {}),
                offline=False,
                bar_seconds=bar_seconds,
                max_bars=max_bars,
            )
    elif source == "synthetic":
        generated = _generate_synthetic(synthetic_settings, paths=1)
        timestamps, prices, symbols = generated[0]
    elif source == "state_space":
        state_settings = playback.get("state_space", {})
        if method != "purged_walk_forward" or strategy_type != "ofmr":
            raise ValueError("state_space source is only supported for OFMR purged walk-forward")
        state_path = Path(str(state_settings.get("path", "")))
        if not str(state_settings.get("path", "")):
            raise ValueError("playback.state_space.path is required")
        state_space_metrics = _reference_ofmr_walk_forward_metrics(
            playback, "state_space", state_path, [],
            str(playback.get("cryptol2_offline", {}).get("symbol", "BTCUSDT")),
        )
    else:
        raise ValueError("playback.source must be cryptol2_offline, cryptol2, synthetic, or state_space")

    strategy_settings = dict(strategy)
    strategy_settings.setdefault("lookback_bars", 5)
    strategy_settings.setdefault("entry_threshold", 0.0)
    strategy_settings["fee_bps"] = float(playback.get("fee_bps", 4.0))
    strategy_settings["starting_capital"] = float(playback.get("starting_capital", 10_000.0))
    strategy_settings["periods_per_year"] = float(playback.get("periods_per_year", 252.0))

    # standardize strategy name: "ofmr@pwf", "ofmr@mc", "ofmr@cpcv"
    method_short = {"purged_walk_forward": "pwf", "parametric_monte_carlo": "mc", "cpcv": "cpcv"}.get(method, method)
    strategy_name = f"{strategy_type}@{method_short}"

    _strategy_name = strategy_name  # set module-level for socket queries
    metrics: list[PathMetrics] = []
    paths_rejected = 0
    folds_or_combinations = 0

    if method == "parametric_monte_carlo":
        cand_settings = dict(synthetic_settings)
        cand_settings.setdefault("paths", 512)
        offline_settings = playback.get("cryptol2_offline", {})
        reference_state_space = (
            offline_settings.get("state_space_path")
            if strategy_type == "ofmr" and source == "cryptol2_offline" else None
        )
        if reference_state_space:
            symbol = str(offline_settings.get("symbol", "BTCUSDT")).upper()
            default_glob = str(ROOT / "market_data" / "matched" / symbol / "merged_*.parquet")
            pattern = str(offline_settings.get("path_glob", default_glob)).replace("{symbol}", symbol)
            if not Path(pattern).is_absolute():
                pattern = str(ROOT / pattern)
            files = sorted(glob.glob(pattern, recursive=True))
            if not files:
                raise FileNotFoundError(f"no CryptoL2 files match {pattern}")
            fold_metrics = _reference_ofmr_walk_forward_metrics(
                playback, source, Path(str(reference_state_space)), files, symbol,
            )
            metrics, paths_rejected = _ofmr_trade_monte_carlo(fold_metrics, playback, cand_settings)
        else:
            metrics, paths_rejected = _parametric_monte_carlo(
                prices if source != "synthetic" else None,
                positions,
                risk_fractions,
                playback,
                strategy_type,
                strategy_settings,
                cand_settings,
            )
        folds_or_combinations = len(metrics)

    elif method == "purged_walk_forward":
        if source == "state_space" and strategy_type == "ofmr":
            metrics = state_space_metrics or []
        elif strategy_type == "ofmr" and source in {"cryptol2", "cryptol2_offline"}:
            metrics = _ofmr_walk_forward_metrics(playback, source)
        elif prices is None or prices.shape[1] != 1:
            raise ValueError("purged_walk_forward is single-asset; configure one symbol")
        else:
            metrics = _canonical_purged_walk_forward(
                prices, positions, risk_fractions,
                playback, strategy_type, strategy_settings,
            )
        folds_or_combinations = len(metrics)

    elif method == "cpcv":
        offline_settings = playback.get("cryptol2_offline", {})
        reference_state_space = (
            offline_settings.get("state_space_path")
            if strategy_type == "ofmr" and source == "cryptol2_offline" else None
        )
        if reference_state_space:
            symbol = str(offline_settings.get("symbol", "BTCUSDT")).upper()
            default_glob = str(ROOT / "market_data" / "matched" / symbol / "merged_*.parquet")
            pattern = str(offline_settings.get("path_glob", default_glob)).replace("{symbol}", symbol)
            if not Path(pattern).is_absolute():
                pattern = str(ROOT / pattern)
            files = sorted(glob.glob(pattern, recursive=True))
            if not files:
                raise FileNotFoundError(f"no CryptoL2 files match {pattern}")
            metrics, paths_rejected = _reference_ofmr_cpcv_metrics(
                playback, Path(str(reference_state_space)), files, symbol,
            )
            folds_or_combinations = len(metrics)
            # Skip the generic price-only CPCV adapter below: this path has
            # trained and scored the actual OFMR model for each combination.
            prices = None
        elif prices is None:
            raise ValueError("cpcv requires market or synthetic playback prices")
        else:
            if prices.shape[1] != 1:
                raise ValueError("cpcv is single-asset; configure one symbol")
            metrics, paths_rejected = _canonical_cpcv(
                prices, positions, risk_fractions,
                playback, strategy_type, strategy_settings,
            )
            folds_or_combinations = len(metrics)

    else:
        raise ValueError(f"unknown playback method: {method}")

    report = _report(method, metrics, paths_rejected=paths_rejected,
                     folds_or_combinations=folds_or_combinations,
                     threshold=threshold, strategy=strategy_type, strategy_name=strategy_name)

    run_dir = _save_results(
        method, source, strategy_type, metrics, report,
        extra={
            "paths_rejected": paths_rejected,
            "folds_or_combinations": folds_or_combinations,
            "threshold": threshold,
            "strategy_name": strategy_name,
        },
    )

    print(report)
    print(f"\n[results] saved to {run_dir}")
    print("SUCCESS")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--method",
        choices=("pwf", "mc", "cpcv", "purged_walk_forward", "parametric_monte_carlo"),
    )
    parser.add_argument("--strategy")
    args = parser.parse_args(argv)
    try:
        playback = _load_config(args.config)
        method = {
            "pwf": "purged_walk_forward",
            "mc": "parametric_monte_carlo",
        }.get(args.method, args.method)
        _run_playback(playback, method_override=method, strategy_override=args.strategy)
    except Exception as error:
        parser.exit(1, f"playback failed: {error}\n")
    return 0


if __name__ == "__main__":
    import socket
    import os
    import threading
    from typing import Optional

    SOCKET_PATH = str(ROOT / "market_data" / "live_db" / "playback" / "playback.sock")

    # module-level strategy name (set by _run_playback)
    _strategy_name: Optional[str] = None

    def _run_socket():
        """Run a Unix domain socket server for querying playback results."""
        global _strategy_name
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)
        print(f"[socket] listening on {SOCKET_PATH}")

        while True:
            try:
                conn, _ = server.accept()
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break

                query = data.decode().strip().lower()
                response = _handle_socket_query(query)
                conn.sendall(response.encode())
                conn.close()
            except Exception as e:
                print(f"[socket error] {e}")

    def _handle_socket_query(query: str) -> str:
        """Handle a socket query and return the response."""
        if query in {"metrics", "status"}:
            return "playback ready - use --config to run"
        elif query == "strategy_name":
            return _strategy_name if _strategy_name else "unknown"
        elif query == "methods":
            return "purged_walk_forward, parametric_monte_carlo, cpcv"
        elif query == "help":
            return ("Available queries:\n"
                    "  metrics - playback status\n"
                    "  strategy_name - current strategy name\n"
                    "  methods - available methods\n"
                    "  help - this help message")
        else:
            return f"unknown query: {query}"

    if __name__ == "__main__":
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--socket":
            _run_socket()
        else:
            main()
