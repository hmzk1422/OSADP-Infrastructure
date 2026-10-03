"""Generate synthetic price paths and keep a pairwise low-correlation set."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from components.pipelines.ingestion.synthetic.synth1 import simulate_market_data
from components.pipelines.ingestion.synthetic.synth2 import simulate_market_stochastic
from components.pipelines.ingestion.synthetic.synth3 import simulate_market_realistic


SYNTHETIC_GENERATORS = {
    "synth1": simulate_market_data,
    "synth2": simulate_market_stochastic,
    "synth3": simulate_market_realistic,
}


def _returns_from_prices(frame: pd.DataFrame, price_column: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("each candidate asset must be a pandas DataFrame")
    if price_column not in frame.columns:
        raise ValueError(f"candidate is missing price column {price_column!r}")
    prices = pd.to_numeric(frame[price_column], errors="coerce")
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    return returns.astype(np.float64)


def _correlation(left: pd.Series, right: pd.Series, min_observations: int) -> float:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
    if len(aligned) < min_observations:
        return np.nan
    if aligned["left"].nunique() < 2 or aligned["right"].nunique() < 2:
        return np.nan
    return float(aligned["left"].corr(aligned["right"]))


def filter_uncorrelated_synthetic_data(
    candidate_data: Mapping[str, pd.DataFrame],
    max_abs_correlation: float = 0.10,
    price_column: str = "price",
    min_observations: int = 30,
    max_assets: int | None = None,
) -> dict:
    """Greedily select paths whose pairwise absolute return correlation is low.

    Correlation is Pearson correlation of aligned simple returns from the
    supplied price column. A candidate is retained only when its correlation
    with every previously retained path is finite and no greater than
    ``max_abs_correlation`` in absolute value. The filter does not claim
    independence; it enforces this explicit pairwise sample-correlation rule.
    """
    if not isinstance(candidate_data, Mapping) or not candidate_data:
        raise ValueError("candidate_data must be a nonempty mapping of IDs to frames")
    if not np.isfinite(max_abs_correlation) or not 0.0 <= max_abs_correlation <= 1.0:
        raise ValueError("max_abs_correlation must be in [0, 1]")
    if not isinstance(min_observations, int) or min_observations < 3:
        raise ValueError("min_observations must be an integer >= 3")
    if max_assets is not None and (not isinstance(max_assets, int) or max_assets <= 0):
        raise ValueError("max_assets must be a positive integer or None")

    selected: dict[str, pd.DataFrame] = {}
    selected_returns: dict[str, pd.Series] = {}
    rejected: dict[str, dict] = {}
    for asset_id, frame in candidate_data.items():
        asset_id = str(asset_id)
        try:
            returns = _returns_from_prices(frame, price_column)
        except (TypeError, ValueError) as exc:
            rejected[asset_id] = {"reason": str(exc), "max_abs_corr": np.nan}
            continue
        if len(returns) < min_observations:
            rejected[asset_id] = {
                "reason": f"only {len(returns)} aligned returns; need {min_observations}",
                "max_abs_corr": np.nan,
            }
            continue

        correlations = {
            prior_id: _correlation(returns, prior_returns, min_observations)
            for prior_id, prior_returns in selected_returns.items()
        }
        finite_corrs = [abs(value) for value in correlations.values() if np.isfinite(value)]
        maximum = max(finite_corrs, default=0.0)
        if any(not np.isfinite(value) for value in correlations.values()):
            rejected[asset_id] = {"reason": "correlation undefined on aligned returns", "max_abs_corr": np.nan}
        elif maximum > max_abs_correlation:
            rejected[asset_id] = {"reason": "pairwise correlation threshold exceeded", "max_abs_corr": maximum}
        elif max_assets is not None and len(selected) >= max_assets:
            rejected[asset_id] = {"reason": "selection limit reached", "max_abs_corr": maximum}
        else:
            selected[asset_id] = frame
            selected_returns[asset_id] = returns

    if selected:
        returns_frame = pd.concat(selected_returns, axis=1, join="inner")
        correlation_matrix = returns_frame.corr(min_periods=min_observations)
    else:
        correlation_matrix = pd.DataFrame(dtype=np.float64)
    return {
        "asset_data": selected,
        "correlation_matrix": correlation_matrix,
        "accepted_ids": list(selected),
        "rejected": rejected,
        "n_candidates": len(candidate_data),
        "n_accepted": len(selected),
        "max_abs_correlation": float(max_abs_correlation),
    }


def generate_uncorrelated_synthetic_assets(
    n_assets: int,
    generator: str = "synth2",
    max_abs_correlation: float = 0.10,
    max_candidates: int | None = None,
    seed: int = 0,
    generator_kwargs: Mapping | None = None,
    bar_seconds: float = 1.0,
    start_time="2026-01-01T00:00:00Z",
    min_observations: int = 30,
) -> dict:
    """Generate candidate paths until ``n_assets`` pass the correlation filter.

    Candidate paths are generated one at a time, so rejected frames are not
    retained in memory. Correlation is measured on the generated underlying
    price returns, before VDC is run.
    """
    if not isinstance(n_assets, int) or n_assets <= 0:
        raise ValueError("n_assets must be a positive integer")
    if not np.isfinite(max_abs_correlation) or not 0.0 <= max_abs_correlation <= 1.0:
        raise ValueError("max_abs_correlation must be in [0, 1]")
    if not np.isfinite(bar_seconds) or bar_seconds <= 0.0:
        raise ValueError("bar_seconds must be finite and > 0")
    if max_candidates is None:
        max_candidates = n_assets * 10
    if not isinstance(max_candidates, int) or max_candidates < n_assets:
        raise ValueError("max_candidates must be an integer >= n_assets")
    try:
        generator_fn = SYNTHETIC_GENERATORS[generator.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("generator must be 'synth1', 'synth2', or 'synth3'") from exc

    kwargs = dict(generator_kwargs or {})
    selected: dict[str, pd.DataFrame] = {}
    selected_returns: dict[str, pd.Series] = {}
    rejected: dict[str, dict] = {}
    first_time = pd.Timestamp(start_time)
    if first_time.tzinfo is None:
        first_time = first_time.tz_localize("UTC")
    else:
        first_time = first_time.tz_convert("UTC")

    for candidate_number in range(max_candidates):
        candidate_seed = int(seed + candidate_number)
        frame = generator_fn(**{**kwargs, "seed": candidate_seed})
        if not isinstance(frame, pd.DataFrame) or "price" not in frame:
            raise TypeError("synthetic generator must return a DataFrame with a 'price' column")
        frame = frame.copy()
        frame.index = pd.date_range(
            first_time,
            periods=len(frame),
            freq=pd.to_timedelta(bar_seconds, unit="s"),
        )
        asset_id = f"{generator.lower()}_seed_{candidate_seed}"
        returns = _returns_from_prices(frame, "price")
        if len(returns) < min_observations:
            rejected[asset_id] = {"reason": f"only {len(returns)} returns; need {min_observations}", "max_abs_corr": np.nan}
            continue
        correlations = {
            prior_id: _correlation(returns, prior_returns, min_observations)
            for prior_id, prior_returns in selected_returns.items()
        }
        if any(not np.isfinite(value) for value in correlations.values()):
            rejected[asset_id] = {"reason": "correlation undefined on aligned returns", "max_abs_corr": np.nan}
            continue
        maximum = max((abs(value) for value in correlations.values()), default=0.0)
        if maximum > max_abs_correlation:
            rejected[asset_id] = {"reason": "pairwise correlation threshold exceeded", "max_abs_corr": maximum}
            continue
        selected[asset_id] = frame
        selected_returns[asset_id] = returns
        if len(selected) >= n_assets:
            break

    if len(selected) < n_assets:
        raise RuntimeError(
            f"Only {len(selected)} of {n_assets} requested assets passed the pairwise "
            f"|correlation| <= {max_abs_correlation:.3f} filter after {max_candidates} candidates"
        )
    returns_frame = pd.concat(selected_returns, axis=1, join="inner")
    return {
        "asset_data": selected,
        "correlation_matrix": returns_frame.corr(min_periods=min_observations),
        "accepted_ids": list(selected),
        "rejected": rejected,
        "n_candidates": len(selected) + len(rejected),
        "n_accepted": len(selected),
        "max_abs_correlation": float(max_abs_correlation),
        "generator": generator.lower(),
    }
