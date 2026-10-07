"""Portfolio aggregation and user-defined VDC risk/execution hooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _series_from_path(item: dict, key: str) -> tuple[pd.Series, pd.DataFrame | None]:
    """Normalize an asset/fold record to fractional per-observation returns."""
    values = item.get("returns")
    path = item.get("path_fraction", item.get("path"))
    if values is None:
        if path is None:
            raise ValueError(f"portfolio path {key!r} has neither 'returns' nor 'path'")
        cumulative = np.asarray(path, dtype=np.float64)
        if "path_fraction" not in item and item.get("path_unit", "percent") == "percent":
            cumulative = cumulative / 100.0
        values = np.diff(cumulative, prepend=0.0)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"returns for {key!r} must be a finite one-dimensional array")

    timestamps = item.get("timestamps")
    if timestamps is None and isinstance(path, (pd.Series, pd.DataFrame)):
        timestamps = path.index
    if timestamps is None:
        timestamps = pd.RangeIndex(len(values), name="observation")
    if len(timestamps) != len(values):
        raise ValueError(f"timestamps and returns length differ for {key!r}")

    returns = pd.Series(values, index=pd.Index(timestamps), name=key, dtype=np.float64)
    returns = returns[~returns.index.duplicated(keep="last")]
    context = item.get("context")
    if context is not None:
        if isinstance(context, pd.Series):
            context = context.to_frame()
        if not isinstance(context, pd.DataFrame):
            raise TypeError(f"context for {key!r} must be a DataFrame or Series")
        if len(context) != len(values):
            raise ValueError(f"context and returns length differ for {key!r}")
        if not context.index.equals(pd.Index(timestamps)):
            context = context.copy()
            context.index = pd.Index(timestamps)
        context = context.loc[~context.index.duplicated(keep="last")]
    return returns, context


def _weight_scores(value, all_ids: list[str], active_ids: list[str]) -> pd.Series:
    """Normalize nonnegative regime scores into target weights."""
    if isinstance(value, pd.DataFrame):
        column = next((name for name in ("score", "weight", "eligible") if name in value), None)
        if column is None:
            raise ValueError("regime_detection DataFrame must include score, weight, or eligible")
        value = value[column]
    if isinstance(value, Mapping):
        scores = pd.Series(value, dtype=np.float64)
    elif isinstance(value, pd.Series):
        scores = value.astype(np.float64)
    else:
        array = np.asarray(value)
        if array.ndim == 0:
            scores = pd.Series(np.repeat(array.item(), len(active_ids)), index=active_ids, dtype=np.float64)
        elif len(array) == len(active_ids):
            scores = pd.Series(array, index=active_ids, dtype=np.float64)
        elif len(array) == len(all_ids):
            scores = pd.Series(array, index=all_ids, dtype=np.float64)
        else:
            raise ValueError("regime_detection output length must match active assets")
    scores = scores.reindex(active_ids).fillna(0.0)
    if not np.isfinite(scores.to_numpy()).all() or (scores < 0.0).any():
        raise ValueError("regime_detection scores must be finite and nonnegative")
    total = float(scores.sum())
    return scores / total if total > 0.0 else pd.Series(0.0, index=active_ids, dtype=np.float64)


def _apply_stop_loss(value, targets: pd.Series) -> pd.Series:
    """Apply a portfolio-wide or per-asset gate to next-bar target weights."""
    if value is None:
        return targets
    if isinstance(value, (bool, np.bool_)):
        return targets * (0.0 if value else 1.0)
    if isinstance(value, (int, float, np.number)):
        factor = float(value)
        if not np.isfinite(factor) or not 0.0 <= factor <= 1.0:
            raise ValueError("stop_loss scalar output must be in [0, 1]")
        return targets * factor
    if isinstance(value, Mapping):
        factors = pd.Series(value, dtype=np.float64)
    elif isinstance(value, pd.Series):
        factors = value.astype(np.float64)
    else:
        raise TypeError("stop_loss must return bool, a [0, 1] scalar, mapping, or Series")
    factors = factors.reindex(targets.index).fillna(1.0)
    if not np.isfinite(factors.to_numpy()).all() or ((factors < 0.0) | (factors > 1.0)).any():
        raise ValueError("stop_loss per-asset factors must be in [0, 1]")
    return targets * factors


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def _finite_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _finite_positive_pct(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values > 0.0) * 100.0) if len(values) else np.nan


def vdc_portfolio(
    vdc_results: dict,
    weighting: str = "equal",
    initial_capital: float = 1.0,
    regime_detection: Callable[[pd.DataFrame], object] | None = None,
    stop_loss: Callable[[pd.DataFrame, dict], object] | None = None,
    executor: Callable[[pd.DataFrame], object] | None = None,
    plot_output: bool = True,
    report_output: bool = True,
) -> dict:
    """Build portfolios per scenario from VDC asset/fold return paths.

    Records with the same ``scenario_id`` and ``fold`` are contemporaneous
    assets and are combined cross-sectionally. Monte Carlo scenario IDs remain
    separate; independent paths are never pooled and misreported as assets.
    Folds within a scenario are processed in fold order, carrying NAV forward.
    NAV compounds as invested capital: ``NAV[t] = NAV[t-1] * (1 + return[t])``.

    Hooks see only the current bar and choose holdings for the following bar:

    * ``regime_detection(snapshot)`` returns nonnegative scores, weights, or
      eligibility flags indexed by ``asset_id``; scores are normalized.
    * ``stop_loss(snapshot, portfolio_state)`` returns a bool, scalar in
      ``[0, 1]``, or per-asset factors in ``[0, 1]``.
    * ``executor(orders)`` returns ``None``, executed weights, or a DataFrame
      with ``executed_weight`` and optional ``transaction_cost`` as a fraction
      of post-return NAV at that rebalance. Executed holdings apply from the
      next observation.

    Input strategy returns are fractional P&L per unit sleeve capital. An
    observation return at or below -100% is rejected because the data contains
    no liquidation/margin-call model for a bankrupt account.
    """
    if not isinstance(vdc_results, dict):
        raise TypeError("vdc_results must be a result dictionary from vdc_asset_test or vdc_monte_carlo")
    if weighting != "equal":
        raise ValueError("currently only weighting='equal' is supported")
    if not np.isfinite(initial_capital) or initial_capital <= 0.0:
        raise ValueError("initial_capital must be finite and > 0")
    for name, callback in (("regime_detection", regime_detection), ("stop_loss", stop_loss), ("executor", executor)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None")

    raw_paths = vdc_results.get("plotted_paths")
    if not raw_paths:
        raise ValueError("vdc_results must contain nonempty 'plotted_paths'")

    # A missing scenario ID on old MC records means one scenario per path;
    # old asset-test records belong to one shared walk-forward set.
    scenarios: dict[object, dict[object, list[dict]]] = {}
    for serial, item in enumerate(raw_paths):
        if not isinstance(item, dict):
            raise TypeError("each plotted_paths item must be a dictionary")
        scenario_id = item.get("scenario_id")
        if scenario_id is None:
            if item.get("source") == "monte_carlo":
                scenario_id = item.get("ticker", f"monte_carlo_path_{serial}")
            else:
                scenario_id = "asset_test"
        fold = item.get("fold", 1)
        scenarios.setdefault(scenario_id, {}).setdefault(fold, []).append({**item, "_serial": serial})

    all_rows: list[dict] = []
    fold_records: list[dict] = []
    portfolio_paths: list[dict] = []
    final_scenario_returns: list[float] = []
    all_asset_final_returns = [float(item.get("final_return_pct", np.nan)) for item in raw_paths]

    for scenario_id, folds in scenarios.items():
        nav = float(initial_capital)
        scenario_peak = float(initial_capital)
        scenario_rows: list[dict] = []
        scenario_timestamps: list[object] = []
        scenario_returns: list[float] = []
        scenario_navs: list[float] = []
        scenario_turnover: list[float] = []
        scenario_costs: list[float] = []

        try:
            ordered_folds = sorted(folds.items(), key=lambda pair: pair[0])
        except TypeError:
            ordered_folds = sorted(folds.items(), key=lambda pair: str(pair[0]))
        for fold, items in ordered_folds:
            fold_start_nav = nav
            normalized = []
            seen_ids: set[str] = set()
            for item in items:
                base_id = str(item.get("ticker", f"path_{item['_serial']}"))
                asset_id = base_id if base_id not in seen_ids else f"{base_id}#{item['_serial']}"
                seen_ids.add(asset_id)
                returns, context = _series_from_path(item, asset_id)
                normalized.append((asset_id, item, returns, context))

            return_frame = pd.concat([entry[2] for entry in normalized], axis=1).sort_index()
            return_frame = return_frame.loc[:, ~return_frame.columns.duplicated(keep="last")]
            observed = return_frame.notna()
            return_frame = return_frame.fillna(0.0)
            asset_ids = list(return_frame.columns)
            timeline = return_frame.index
            if len(timeline) == 0:
                continue
            context_by_id = {asset_id: context for asset_id, _, _, context in normalized}
            item_by_id = {asset_id: item for asset_id, item, _, _ in normalized}

            first_active = observed.iloc[0]
            current_weights = pd.Series(0.0, index=asset_ids, dtype=np.float64)
            if first_active.any():
                current_weights.loc[first_active[first_active].index] = 1.0 / int(first_active.sum())

            if regime_detection is None and stop_loss is None and executor is None:
                active_values = observed.to_numpy(dtype=bool)
                target_values = active_values / np.maximum(active_values.sum(axis=1, keepdims=True), 1)
                held_values = target_values.copy()
                if len(timeline) > 1:
                    held_values[1:] = target_values[:-1]
                bar_returns = return_frame.to_numpy(dtype=np.float64)
                gross = np.sum(held_values * bar_returns, axis=1)
                net = gross.copy()
                turnover = np.abs(target_values - held_values).sum(axis=1)
                costs = np.zeros(len(timeline), dtype=np.float64)
                if np.any(net <= -1.0):
                    bad_time = timeline[int(np.flatnonzero(net <= -1.0)[0])]
                    raise ValueError(f"portfolio return <= -100% at {bad_time!r}; liquidation is not modeled")
                nav_values = nav * np.cumprod(1.0 + net)
                nav = float(nav_values[-1])
                peak_values = np.maximum.accumulate(np.r_[scenario_peak, nav_values])
                dd_values = nav_values / peak_values[1:] - 1.0
                scenario_peak = float(peak_values[-1])
                for i, timestamp in enumerate(timeline):
                    scenario_rows.append({
                        "scenario_id": scenario_id,
                        "fold": fold,
                        "timestamp": timestamp,
                        "portfolio_return": float(net[i]),
                        "gross_portfolio_return": float(gross[i]),
                        "transaction_cost": 0.0,
                        "transaction_cost_amount": 0.0,
                        "cumulative_return": float(nav_values[i] / initial_capital - 1.0),
                        "strategy_nav": float(nav_values[i]),
                        "drawdown": float(dd_values[i]),
                        "gross_turnover": float(turnover[i]),
                    })
                scenario_timestamps.extend(timeline.tolist())
                scenario_returns.extend(net.tolist())
                scenario_navs.extend(nav_values.tolist())
                scenario_turnover.extend(turnover.tolist())
                scenario_costs.extend(costs.tolist())
            else:
                fold_records_for_summary = []
                for row_number, timestamp in enumerate(timeline):
                    bar_returns = return_frame.iloc[row_number]
                    gross_return = float((current_weights * bar_returns).sum())
                    if gross_return <= -1.0:
                        raise ValueError(f"portfolio return <= -100% at {timestamp!r}; liquidation is not modeled")
                    nav_before_trade = nav * (1.0 + gross_return)
                    peak_before_trade = max(initial_capital, max(scenario_navs, default=initial_capital), nav_before_trade)
                    active_ids = list(observed.columns[observed.iloc[row_number]])
                    snapshot_rows = []
                    for asset_id in active_ids:
                        item = item_by_id[asset_id]
                        row = {
                            "asset_id": asset_id,
                            "ticker": item.get("ticker", asset_id),
                            "source": item.get("source", "asset_test"),
                            "asset_class": item.get("asset_class"),
                            "scenario_id": scenario_id,
                            "fold": fold,
                            "asset_return": float(bar_returns.loc[asset_id]),
                            "current_weight": float(current_weights.loc[asset_id]),
                            "directional_bias": item.get("directional_bias", np.nan),
                            "volatility_scale": item.get("volatility_scale", np.nan),
                            "regime_persistence": item.get("regime_persistence", np.nan),
                        }
                        context = context_by_id.get(asset_id)
                        if context is not None and timestamp in context.index:
                            context_row = context.loc[timestamp]
                            if isinstance(context_row, pd.DataFrame):
                                context_row = context_row.iloc[-1]
                            for column, value in context_row.items():
                                row[f"context_{column}" if column in row else column] = value
                        snapshot_rows.append(row)
                    snapshot = pd.DataFrame(snapshot_rows).set_index("asset_id") if snapshot_rows else pd.DataFrame(index=pd.Index([], name="asset_id"))

                    if active_ids:
                        if regime_detection is None:
                            targets = pd.Series(1.0 / len(active_ids), index=active_ids, dtype=np.float64)
                        else:
                            targets = _weight_scores(regime_detection(snapshot.copy()), asset_ids, active_ids)
                        portfolio_state = {
                            "timestamp": timestamp,
                            "nav": float(nav_before_trade),
                            "cumulative_return": float(nav_before_trade / initial_capital - 1.0),
                            "drawdown": float(nav_before_trade / peak_before_trade - 1.0),
                            "current_weights": current_weights.copy(),
                            "target_weights": targets.copy(),
                        }
                        if stop_loss is not None:
                            targets = _apply_stop_loss(stop_loss(snapshot.copy(), portfolio_state), targets)
                        order_rows = []
                        for asset_id in asset_ids:
                            data = snapshot.loc[asset_id].to_dict() if asset_id in snapshot.index else {}
                            target = float(targets.get(asset_id, 0.0))
                            current = float(current_weights.get(asset_id, 0.0))
                            order_rows.append({**data, "asset_id": asset_id, "target_weight": target,
                                               "current_weight": current, "trade_weight": target - current,
                                               "asset_return": float(bar_returns.loc[asset_id])})
                        orders = pd.DataFrame(order_rows).set_index("asset_id")
                        executed = executor(orders.copy()) if executor is not None else None
                        execution_cost = 0.0
                        if executed is None:
                            next_weights = pd.Series([row["target_weight"] for row in order_rows], index=asset_ids, dtype=np.float64)
                        elif isinstance(executed, pd.DataFrame):
                            if "executed_weight" not in executed.columns:
                                raise ValueError("executor DataFrame must include 'executed_weight'")
                            next_weights = executed["executed_weight"].reindex(asset_ids).fillna(0.0).astype(np.float64)
                            if "transaction_cost" in executed:
                                execution_cost = float(executed["transaction_cost"].fillna(0.0).sum())
                        elif isinstance(executed, pd.Series):
                            next_weights = executed.reindex(asset_ids).fillna(0.0).astype(np.float64)
                        elif isinstance(executed, Mapping):
                            next_weights = pd.Series(executed, dtype=np.float64).reindex(asset_ids).fillna(0.0)
                        else:
                            raise TypeError("executor must return None, a Series/Mapping, or a DataFrame")
                        if not np.isfinite(next_weights.to_numpy()).all():
                            raise ValueError("executor weights must be finite")
                        if not np.isfinite(execution_cost) or execution_cost < 0.0:
                            raise ValueError("executor transaction_cost must be finite and nonnegative")
                        turnover = float((next_weights - current_weights).abs().sum())
                        current_weights = next_weights
                    else:
                        execution_cost = 0.0
                        turnover = 0.0
                    transaction_cost_amount = nav_before_trade * execution_cost
                    net_return = (1.0 + gross_return) * (1.0 - execution_cost) - 1.0
                    if net_return <= -1.0:
                        raise ValueError(f"net portfolio return <= -100% at {timestamp!r}; liquidation is not modeled")
                    nav = nav_before_trade - transaction_cost_amount
                    scenario_rows.append({
                        "scenario_id": scenario_id, "fold": fold, "timestamp": timestamp,
                        "portfolio_return": net_return, "gross_portfolio_return": gross_return,
                        "transaction_cost": execution_cost,
                        "transaction_cost_amount": transaction_cost_amount,
                        "cumulative_return": nav / initial_capital - 1.0,
                        "strategy_nav": nav,
                        "drawdown": nav / max(peak_before_trade, nav) - 1.0,
                        "gross_turnover": turnover,
                    })
                    scenario_timestamps.append(timestamp)
                    scenario_returns.append(net_return)
                    scenario_navs.append(nav)
                    scenario_turnover.append(turnover)
                    scenario_costs.append(execution_cost)

            fold_slice = [row for row in scenario_rows if row["fold"] == fold]
            fold_final_return = (
                float((nav / fold_start_nav - 1.0) * 100.0)
                if fold_slice and fold_start_nav > 0.0
                else np.nan
            )
            asset_final = np.asarray([float(item.get("final_return_pct", np.nan)) for item in items], dtype=np.float64)
            fold_records.append({
                "scenario_id": scenario_id,
                "fold": fold,
                "n_assets": len(items),
                "final_return_pct": fold_final_return,
                "average_asset_return_pct": _finite_mean(asset_final),
                "median_asset_return_pct": _finite_median(asset_final),
                "profitable_asset_pct": _finite_positive_pct(asset_final),
            })

        if scenario_rows:
            final_return = float(scenario_rows[-1]["cumulative_return"])
            final_scenario_returns.append(final_return)
            portfolio_paths.append({
                "scenario_id": scenario_id,
                "path_fraction": np.asarray([row["cumulative_return"] for row in scenario_rows], dtype=np.float64),
                "returns": np.asarray([row["portfolio_return"] for row in scenario_rows], dtype=np.float64),
                "timestamps": [row["timestamp"] for row in scenario_rows],
                "folds": [row["fold"] for row in scenario_rows],
                "n_assets": max(row["n_assets"] for row in fold_records if row["scenario_id"] == scenario_id),
            })
            all_rows.extend(scenario_rows)

    if not all_rows:
        raise RuntimeError("Unable to construct portfolio paths")

    portfolio_index = pd.DataFrame(all_rows).set_index(["scenario_id", "timestamp"])
    portfolio_returns = portfolio_index["portfolio_return"].to_numpy(dtype=np.float64)
    transaction_costs = portfolio_index["transaction_cost"].to_numpy(dtype=np.float64)
    active_returns = portfolio_returns[portfolio_returns != 0.0]
    downside_deviation = float(np.sqrt(np.mean(np.minimum(portfolio_returns, 0.0) ** 2)))
    avg_return = float(np.mean(portfolio_returns))
    sortino = avg_return / downside_deviation if downside_deviation > 0.0 else (np.inf if avg_return > 0.0 else np.nan)
    fold_df = pd.DataFrame(fold_records)
    finals_pct = np.asarray(final_scenario_returns, dtype=np.float64) * 100.0
    asset_finals = np.asarray(all_asset_final_returns, dtype=np.float64)
    summary = {
        "initial_capital": float(initial_capital),
        "n_assets": len({str(item.get("ticker", i)) for i, item in enumerate(raw_paths)}),
        "n_paths": len(raw_paths),
        "n_scenarios": len(final_scenario_returns),
        "n_folds": int(fold_df["fold"].nunique()),
        "final_portfolio_return_pct": float(np.mean(finals_pct)),
        "median_scenario_return_pct": float(np.median(finals_pct)),
        "best_scenario_return_pct": float(np.max(finals_pct)),
        "worst_scenario_return_pct": float(np.min(finals_pct)),
        "final_nav": float(initial_capital * (1.0 + np.mean(final_scenario_returns))),
        "average_asset_final_return_pct": _finite_mean(asset_finals),
        "median_asset_final_return_pct": _finite_median(asset_finals),
        "profitable_asset_path_pct": _finite_positive_pct(asset_finals),
        "portfolio_mdd_pct": float(portfolio_index["drawdown"].min() * 100.0),
        "portfolio_sortino": float(sortino),
        "portfolio_win_rate_pct": float(np.mean(active_returns > 0.0) * 100.0) if len(active_returns) else np.nan,
        "portfolio_downside_deviation": downside_deviation,
        "total_transaction_cost": float(portfolio_index["transaction_cost_amount"].sum()),
        "total_turnover": float(portfolio_index["gross_turnover"].sum()),
        "best_fold_return_pct": float(fold_df["final_return_pct"].max()),
        "worst_fold_return_pct": float(fold_df["final_return_pct"].min()),
        "regime_detection_enabled": regime_detection is not None,
        "stop_loss_enabled": stop_loss is not None,
        "executor_enabled": executor is not None,
    }

    if report_output:
        print("\n" + "=" * 60)
        print("VDC PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"Assets / paths              : {summary['n_assets']} / {summary['n_paths']}")
        print(f"Scenarios / folds           : {summary['n_scenarios']} / {summary['n_folds']}")
        print(f"Mean scenario return        : {summary['final_portfolio_return_pct']:.4f}%")
        print(f"Median scenario return      : {summary['median_scenario_return_pct']:.4f}%")
        print(f"Mean final NAV              : {summary['final_nav']:.6f}")
        print(f"Portfolio MDD               : {summary['portfolio_mdd_pct']:.4f}%")
        print(f"Portfolio Sortino           : {summary['portfolio_sortino']:.4f}")
        print(f"Portfolio Win Rate          : {summary['portfolio_win_rate_pct']:.2f}%")
        print(f"Transaction Cost            : {summary['total_transaction_cost']:.6f}")
        print(f"Regime Detection            : {summary['regime_detection_enabled']}")
        print(f"Stop Loss                   : {summary['stop_loss_enabled']}")
        print(f"Executor                    : {summary['executor_enabled']}")
        print("=" * 60)

    if plot_output:
        plt.figure(figsize=(14, 7))
        for path in portfolio_paths:
            plt.plot(path["timestamps"], path["path_fraction"] * 100.0, alpha=0.35, linewidth=1.0)
        plt.axhline(0.0, linestyle="--", linewidth=1.5, label="Breakeven")
        plt.xlabel("Portfolio Test Observation")
        plt.ylabel("Cumulative Return (%)")
        plt.title("VDC Diversified Portfolio by Scenario")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    return {
        "portfolio_index": portfolio_index,
        "fold_results": fold_df,
        "portfolio_paths": portfolio_paths,
        "summary": summary,
    }


vdc_portfolio_index = vdc_portfolio
