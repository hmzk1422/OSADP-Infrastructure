"""YAML-driven UETS playback runner for synthetic and CryptoL2 market prices."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import glob
from io import StringIO
from itertools import combinations
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "uets" / "config" / "execution.yaml"


@dataclass(frozen=True)
class PathMetrics:
    cumulative_return: float
    expectancy_per_trade: float
    max_drawdown: float
    sortino: float
    trades: int


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    if config.get("mode") != "playback":
        raise ValueError("set mode: playback in the selected YAML to run playback")
    playback = config.get("playback", {})
    if not isinstance(playback, dict):
        raise ValueError("playback configuration must be a YAML mapping")
    return playback


def _generate_synthetic(
    settings: dict[str, Any], *, paths: int | None = None
) -> list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]]:
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


def _read_cryptol2(settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    configured_symbols = settings.get("symbols")
    pattern = str(settings.get(
        "path_glob", "market_data/ingest_parquet/{symbol}/merged/**/*.parquet"
    ))

    def matching_files(raw_pattern: str) -> list[Path]:
        path_pattern = Path(raw_pattern)
        if not path_pattern.is_absolute():
            path_pattern = ROOT / path_pattern
        return sorted(Path(path) for path in glob.glob(str(path_pattern), recursive=True))

    paths_by_symbol: dict[str, list[Path]] = {}
    if configured_symbols:
        for symbol in configured_symbols:
            symbol = str(symbol).upper()
            paths_by_symbol[symbol] = matching_files(pattern.format(symbol=symbol))
    else:
        for path in matching_files(str(settings.get(
            "all_assets_glob", "market_data/ingest_parquet/*/merged/**/*.parquet"
        ))):
            pieces = path.parts
            if "ingest_parquet" in pieces:
                symbol = pieces[pieces.index("ingest_parquet") + 1].upper()
                paths_by_symbol.setdefault(symbol, []).append(path)
    paths_by_symbol = {
        symbol: sorted(paths)
        for symbol, paths in paths_by_symbol.items()
        if paths
    }
    if not paths_by_symbol:
        raise FileNotFoundError("no CryptoL2 merged Parquet files matched playback.path_glob")
    command = [
        "cargo", "run", "--quiet",
        "--manifest-path", str(ROOT / "components" / "pipelines" / "uets_trades" / "Cargo.toml"),
        "--bin", "playback_data", "--",
    ]
    for symbol, files in paths_by_symbol.items():
        command.extend(("--symbol", symbol))
        for path in files:
            command.extend(("--file", str(path)))
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"CryptoL2 Rust loader failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    frame = pd.read_csv(StringIO(result.stdout))
    if frame.empty:
        raise ValueError("CryptoL2 Parquet files contained no usable LOB mid prices")
    aligned = (
        frame.pivot_table(index="timestamp_ms", columns="symbol", values="price", aggfunc="last")
        .sort_index()
        .ffill()
        .dropna()
    )
    if len(aligned) < 2:
        raise ValueError("CryptoL2 data must contain at least two aligned timestamps")
    return (
        aligned.index.to_numpy(dtype=np.int64),
        aligned.to_numpy(dtype=np.float64),
        tuple(str(symbol) for symbol in aligned.columns),
    )


def _returns(prices: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(prices, dtype=np.float64)
    returns[1:] = prices[1:] / prices[:-1] - 1.0
    returns[~np.isfinite(returns)] = 0.0
    return returns


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    size = min(left.size, right.size)
    if size < 3:
        return float("nan")
    left = left[:size]
    right = right[:size]
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def _sequences_compatible(left: np.ndarray, right: np.ndarray, threshold: float) -> bool:
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    for left_asset in range(left.shape[1]):
        for right_asset in range(right.shape[1]):
            correlation = _correlation(left[:, left_asset], right[:, right_asset])
            if not np.isfinite(correlation) or abs(correlation) > threshold + 1e-12:
                return False
    return True


def _correlation_mask(
    returns: list[np.ndarray], threshold: float
) -> np.ndarray:
    count = len(returns)
    mask = np.eye(count, dtype=np.bool_)
    for left in range(count):
        for right in range(left + 1, count):
            compatible = _sequences_compatible(returns[left], returns[right], threshold)
            mask[left, right] = mask[right, left] = compatible
    return mask


def _select_monte_carlo(
    paths: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]],
    threshold: float,
    max_paths: int,
) -> tuple[list[int], int]:
    returns = [_returns(path[1]) for path in paths]
    mask = _correlation_mask(returns, threshold)
    selected: list[int] = []
    for index in range(len(paths)):
        if len(selected) >= max_paths:
            break
        if all(mask[index, prior] for prior in selected):
            selected.append(index)
    return selected, len(paths) - len(selected)


def _strategy_positions(prices: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    strategy_type = str(settings.get("type", "momentum")).lower()
    if strategy_type != "momentum":
        raise ValueError("playback.strategy.type currently supports only 'momentum'")
    lookback = int(settings.get("lookback_bars", 5))
    threshold = float(settings.get("entry_threshold", 0.0))
    if lookback < 1 or threshold < 0.0:
        raise ValueError("strategy.lookback_bars must be positive and entry_threshold nonnegative")
    momentum = np.zeros_like(prices, dtype=np.float64)
    if prices.shape[0] > lookback:
        momentum[lookback:] = prices[lookback:] / prices[:-lookback] - 1.0
    return np.where(np.abs(momentum) >= threshold, np.sign(momentum), 0.0)


def _run_one_path(
    prices: np.ndarray,
    settings: dict[str, Any],
) -> PathMetrics:
    if prices.ndim != 2 or prices.shape[0] < 3 or prices.shape[1] < 1:
        raise ValueError("each playback path must have at least three rows and one asset")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("playback prices must be positive and finite")
    positions = _strategy_positions(prices, settings.get("strategy", {}))
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
    fee_rate = float(settings.get("fee_bps", 4.0)) / 10_000.0
    portfolio_returns -= turnover * fee_rate / max(prices.shape[1], 1)
    portfolio_returns[-1] -= (
        np.abs(positions[-1]).sum() * fee_rate / max(prices.shape[1], 1)
    )
    portfolio_returns = np.clip(portfolio_returns, -0.999999, None)
    starting_capital = float(settings.get("starting_capital", 10_000.0))
    if not np.isfinite(starting_capital) or starting_capital <= 0:
        raise ValueError("starting_capital must be finite and positive")
    equity = starting_capital * np.cumprod(1.0 + portfolio_returns)
    equity_curve = np.concatenate(([starting_capital], equity))
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
                trade_returns.append(
                    entry_side * (prices[bar_index, asset_index] / prices[entry_index, asset_index] - 1.0)
                    - 2.0 * fee_rate
                )
            entry_index = bar_index if current_side != 0.0 else None
            entry_side = current_side
        if entry_index is not None and positions[-1, asset_index] != 0.0:
            trade_returns.append(
                entry_side * (prices[-1, asset_index] / prices[entry_index, asset_index] - 1.0)
                - 2.0 * fee_rate
            )
    closed_trades = len(trade_returns)
    expectancy = (
        float(np.mean(trade_returns)) * starting_capital / prices.shape[1]
        if trade_returns
        else 0.0
    )
    bar_returns = np.diff(equity_curve) / equity_curve[:-1]
    downside = np.minimum(bar_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside * downside)))
    mean_return = float(np.mean(bar_returns)) if bar_returns.size else 0.0
    sortino = (
        mean_return / downside_deviation * np.sqrt(float(settings.get("periods_per_year", 252.0)))
        if downside_deviation > 0.0
        else (float("inf") if mean_return > 0.0 else 0.0)
    )
    return PathMetrics(
        cumulative_return=float(equity[-1] / starting_capital - 1.0),
        expectancy_per_trade=expectancy,
        max_drawdown=float(np.max(drawdowns)),
        sortino=sortino,
        trades=int(closed_trades),
    )


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


def _report(
    method: str,
    metrics: list[PathMetrics],
    *,
    paths_rejected: int = 0,
    folds_or_combinations: int = 0,
    threshold: float = 0.5,
) -> str:
    if not metrics:
        raise ValueError("playback produced no eligible paths")
    lines = [
        f"Playback report: {method}",
        f"Paths run: {len(metrics)}; rejected: {paths_rejected}; folds/combinations: {folds_or_combinations}",
        f"Correlation limit: |rho| <= {threshold:.3f}",
    ]
    for label, field in (
        ("Cumulative return", "cumulative_return"),
        ("Average expectancy per trade", "expectancy_per_trade"),
        ("MDD", "max_drawdown"),
        ("Sortino", "sortino"),
    ):
        mean, cvar05, cvar95 = _summarize([getattr(metric, field) for metric in metrics])
        lines.append(f"{label}: {mean:.6f} +- [CVaR05 {cvar05:.6f}, CVaR95 {cvar95:.6f}]")
    lines.append(f"Total trades: {sum(metric.trades for metric in metrics)}")
    return "\n".join(lines)


def _split_ranges(size: int, splits: int) -> list[tuple[int, int]]:
    return [(index * size // splits, (index + 1) * size // splits) for index in range(splits)]


def _run_playback(playback: dict[str, Any]) -> str:
    method = str(playback.get("method", "purged_walk_forward")).lower()
    threshold = float(playback.get("max_abs_correlation", 0.5))
    max_paths = int(playback.get("max_paths", 100_000))
    splits = int(playback.get("n_splits", 6))
    test_splits = int(playback.get("n_test_splits", 2))
    purge_bars = int(playback.get("purge_bars", 0))
    if method not in {"parametric_monte_carlo", "cpcv", "purged_walk_forward"}:
        raise ValueError("playback.method must be parametric_monte_carlo, cpcv, or purged_walk_forward")
    if (
        not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
        or max_paths < 1
        or not 2 <= splits <= 64
        or not 1 <= test_splits <= splits
        or purge_bars < 0
    ):
        raise ValueError(
            "max_abs_correlation must be in [0, 1], max_paths positive, "
            "n_splits 2..64, n_test_splits 1..n_splits, and purge_bars nonnegative"
        )

    source = str(playback.get("source", "synthetic")).lower()
    synthetic_settings = playback.get("synthetic", {})
    if method == "parametric_monte_carlo":
        if source != "synthetic":
            raise ValueError("parametric_monte_carlo requires playback.source: synthetic")
        candidates = _generate_synthetic(synthetic_settings)
        selected, rejected = _select_monte_carlo(candidates, threshold, max_paths)
        metrics = _parallel_metrics(
            [candidates[index][1] for index in selected],
            playback,
        )
        return _report(
            method,
            metrics,
            paths_rejected=rejected,
            threshold=threshold,
        )

    if method == "cpcv":
        if source != "cryptol2":
            raise ValueError("cpcv requires playback.source: cryptol2")
        timestamps, prices, symbols = _read_cryptol2(playback.get("cryptol2", {}))
    elif source == "synthetic":
        timestamps, prices, symbols = _generate_synthetic(synthetic_settings, paths=1)[0]
    elif source == "cryptol2":
        timestamps, prices, symbols = _read_cryptol2(playback.get("cryptol2", {}))
    else:
        raise ValueError("playback.source must be synthetic or cryptol2")

    if method == "purged_walk_forward":
        if prices.shape[1] != 1:
            raise ValueError("purged_walk_forward is single-asset; configure one symbol")
        if prices.shape[0] < splits:
            raise ValueError("event count must be at least n_splits")
        folds = []
        for index, (raw_start, end) in enumerate(_split_ranges(prices.shape[0], splits)):
            start = raw_start if index == 0 else raw_start + purge_bars
            if start < end:
                folds.append(prices[start:end])
        metrics = _parallel_metrics(folds, playback)
        return _report(
            method,
            metrics,
            folds_or_combinations=len(folds),
            threshold=threshold,
        )

    if prices.shape[0] < splits * 4:
        raise ValueError("CPCV requires at least four observations per split")
    ranges = _split_ranges(prices.shape[0], splits)
    chunk_returns = [
        _returns(prices[start:end])
        for start, end in ranges
    ]
    compatible = _correlation_mask(chunk_returns, threshold)
    combinations_to_run: list[tuple[int, ...]] = []
    for path in combinations(range(splits), test_splits):
        if all(compatible[left, right] for left, right in combinations(path, 2)):
            combinations_to_run.append(path)
            if len(combinations_to_run) >= max_paths:
                break
    if not combinations_to_run:
        raise ValueError("no CPCV combinations met the precomputed correlation mask")

    def build_path(path: tuple[int, ...]) -> np.ndarray:
        pieces = []
        for position, chunk in enumerate(path):
            start, end = ranges[chunk]
            if position > 0:
                start += purge_bars
            if position + 1 < len(path):
                end -= purge_bars
            if start < end:
                pieces.append(prices[start:end])
        if not pieces:
            raise ValueError("purge_bars removed all events from a CPCV path")
        return np.concatenate(pieces, axis=0)

    metrics = _parallel_metrics(
        [build_path(path) for path in combinations_to_run],
        playback,
    )
    return _report(
        method,
        metrics,
        folds_or_combinations=len(metrics),
        threshold=threshold,
    )


def _parallel_metrics(
    paths: list[np.ndarray], settings: dict[str, Any]
) -> list[PathMetrics]:
    if not paths:
        raise ValueError("playback produced no eligible paths")
    workers = int(settings.get("workers", 0))
    if workers < 0:
        raise ValueError("playback.workers must be zero or a positive integer")
    with ThreadPoolExecutor(max_workers=workers or None) as pool:
        return list(pool.map(lambda path: _run_one_path(path, settings), paths))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        report = _report_wrapper(args.config)
    except Exception as error:
        parser.exit(1, f"playback failed: {error}\n")
    print(report)
    return 0


def _report_wrapper(config_path: Path) -> str:
    playback = _load_config(config_path)
    return _run_playback(playback)


if __name__ == "__main__":
    raise SystemExit(main())
