"""OFMR scalp strategy engine (S3E: thin orchestration layer).

Responsibilities, in order:

    1. Load ``settings.yaml``                             -> ``load_settings``
    2. Load raw market data through ``exchange/``         -> ``load_data``
    3. Compute features                                   -> ``features.compute_features``
    4. Detect OOD states from raw temporal primitives     -> ``run_ood`` (``ood/``)
    5. Build supervision targets                          -> ``outcome.build_supervision_targets``
    6. Optional per-state GP search                       -> ``run_gp``
    7. Assemble the single modeling frame (S1 contract)   -> ``outcome.build_model_frame``
    8. Outcome pipeline (linreg -> metalabel -> barriers) -> ``run_triple_barrier``
    9. Apply the strategy rules                           -> ``policies/``

All mechanics -- causal alignment, train masks, fits, barrier building -- live
in ``outcome/alignment.py`` (S3A), ``outcome/outcomes.py`` (S3B/S3C) and
``outcome/triple_barrier.py`` (S3D); this module only wires them together.

Column contract (S1) -- ``StrategyResult.features`` is assembled once by
``build_model_frame``:

    [causal features] + [raw__ primitives] + [gp_feat*] + [regime flags] + [targets]

Forward columns carry the ``future_`` prefix and are supervision targets only:
the GP evolves from ``build_gp_source`` (features + ``raw__`` primitives, with
``future_*`` / ``latent`` / ``label`` / ... excluded by name) and the policy
consumes the causal feature list, so targets can never leak into live signals.

Run from the repository root:

    python -m strategy_system.strategy.moe.strategy [path/to/settings.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple
from tqdm import tqdm
import pandas as pd
import yaml

from ...exchange.loader import load_L1, load_L2
from .features import compute_features
from .gp import GPEvolutionResult, extrapolate_features
from .ood import OODSystem
from .outcome import (
    MetalabelResult,
    OutcomeResult,
    align_regimes_to_index,
    build_gp_source,
    build_model_frame,
    build_supervision_targets,
    build_triple_barrier_sequence,
    chronological_train_mask,
    run_outcomes,
)
from .policies.orderflow_prediction import (
    evaluate_orderflow_policy,
    format_report,
)

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"

DATA_LOADERS = {
    "orderbook": load_L2,
    "ohlcv": load_L1,
}


class StrategyResult(NamedTuple):
    """Everything the engine produced in one run.

    ``features`` is the S1 assembled model frame (column contract in the
    module docstring), not the bare ``compute_features`` output.
    """

    features: pd.DataFrame  # S1 frame: features + raw__ + gp_feat* + regime flags + targets
    regimes: pd.DataFrame   # causal raw-market OOD state assignments
    events: pd.DataFrame    # triple-barrier sequence (one row per signal)
    outcomes: pd.DataFrame  # all-row forward-return targets for supervised learning
    policy: pd.DataFrame    # policy report (the strategy's rules)
    gp: GPEvolutionResult | None        # optional state-specific evolved features
    metalabel: MetalabelResult | None   # fitted metalabel tree + labels + OOS report


def load_settings(path: str | Path | None = None) -> dict:
    """Read the engine's ``settings.yaml`` (defaults to the sibling file)."""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH

    with settings_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    if not isinstance(settings, dict):
        raise ValueError(f"{settings_path} must contain a YAML mapping.")

    return settings


def load_data(settings: dict) -> pd.DataFrame:
    """Load raw market data through the exchange layer.

    Uses ``settings['data']['source']`` to pick the loader and forwards the
    matching parameter section to it.
    """
    data_cfg = settings["data"]
    source = data_cfg["source"]

    try:
        loader = DATA_LOADERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown data source {source!r}; expected one of {sorted(DATA_LOADERS)}."
        ) from None

    return loader(**data_cfg.get(source, {}))


def run_ood(
    raw: pd.DataFrame,
    cfg: dict,
    detector: OODSystem | None = None,
    process_from: int = 0,
) -> pd.DataFrame:
    """Detect OOD states from the original raw stream (never a derived frame).

    Null states are explicitly non-tradeable. Pass the same ``detector``
    across data chunks to preserve learned latent prototypes; pass its
    ``state_dict()`` to persist and later restore those prototypes.
    """
    detector = detector if detector is not None else OODSystem(**cfg.get("ood", {}))
    return detector.detect(raw, process_from=process_from)


def run_gp(
    raw: pd.DataFrame,
    feat_df: pd.DataFrame,
    targets: pd.DataFrame,
    regimes: pd.DataFrame,
    cfg: dict,
) -> GPEvolutionResult | None:
    """Opt-in per-state GP search (extracted from ``strategy()``).

    Returns ``None`` while ``gp.enabled`` is false. The source build, the
    chronological mask and the config knobs live here so ``strategy()`` stays
    a straight-line call sequence (S3E: no config surgery in ``strategy()``).
    Targets are passed as-is: ``extrapolate_features`` only uses
    ``future_return__*`` columns as objectives, never the ``future_`` frame's
    neighbours as inputs (GP input selection forbids ``future_``/``latent``/...).
    """
    gp_cfg = dict(cfg.get("gp", {}))
    if not gp_cfg.pop("enabled", False):
        return None

    ood_cfg = cfg.get("ood") or {}
    source = build_gp_source(raw, feat_df, ood_cfg.get("timestamp_col", "timestamp"))
    target_columns = gp_cfg.get("target_columns")
    train_fraction = float(gp_cfg.pop("train_fraction", 0.70))
    purge_rows = int(gp_cfg.pop("purge_rows", 0))
    split_position = int(len(feat_df) * train_fraction)
    train_end_time = (
        feat_df.index[split_position]
        if isinstance(feat_df.index, pd.DatetimeIndex) and split_position < len(feat_df)
        else None
    )
    train_mask = chronological_train_mask(feat_df, train_fraction, purge_rows)
    # Control columns configure the split but are not arguments to GP.
    gp_cfg.pop("target_columns", None)
    aligned_regimes = align_regimes_to_index(regimes, feat_df.index)
    return extrapolate_features(
        source,
        targets,
        aligned_regimes["latent_id"],
        train_mask,
        target_columns=target_columns,
        train_end_time=train_end_time,
        purge_rows=purge_rows,
        **gp_cfg,
    )


def run_triple_barrier(
    model_frame: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    *,
    raw: pd.DataFrame | None = None,
) -> OutcomeResult:
    """Compose linreg -> metalabel -> barriers (one call into ``outcomes.run_outcomes``).

    Settings files that still use the legacy ``barriers`` section instead get
    a fixed barrier sequence straight from ``triple_barrier.py`` with empty
    fits, matching the pre-refactor behaviour.

    Either way, labelled events are annotated with their entry state so
    research can tell known-state, tradeable entries from the rest.
    """
    if cfg.get("outcome") is None:
        # Backward compatibility for settings files that still use `barriers`.
        barrier_cfg = dict(cfg["barriers"])
        horizon_bars = barrier_cfg.pop("horizon", barrier_cfg.pop("horizon_bars", 30))
        events = build_triple_barrier_sequence(
            model_frame,
            mode="fixed",
            horizon_bars=horizon_bars,
            **barrier_cfg,
        )
        outcome = OutcomeResult(
            predictions=pd.DataFrame(index=model_frame.index),
            excursions=pd.DataFrame(index=model_frame.index),
            metalabel=None,
            events=events,
            report=pd.DataFrame(columns=["latent_id", "n_train", "n_features", "rank"]),
        )
    else:
        outcome = run_outcomes(model_frame, targets, cfg, raw=raw)

    # Keep labeled events for model research, while exposing whether their
    # entry state was known and therefore eligible for execution.
    events = outcome.events
    if not events.empty and "t0" in events.columns:
        entry_regimes = align_regimes_to_index(model_frame, pd.Index(events["t0"]))
        events["latent_id"] = entry_regimes["latent_id"].to_numpy()
        events["ood_similarity"] = entry_regimes["similarity"].to_numpy()
        events["trade_allowed"] = entry_regimes["trade_allowed"].fillna(False).to_numpy(dtype=bool)
        events["state_mature"] = entry_regimes["state_mature"].fillna(False).to_numpy(dtype=bool)
        events["model_ready"] = entry_regimes["model_ready"].fillna(False).to_numpy(dtype=bool)
    return outcome


def strategy(
    df: pd.DataFrame,
    settings: dict | None = None,
    ood_detector: OODSystem | None = None,
    ood_process_from: int = 0,
    show_progress: bool = True
) -> StrategyResult:
    """Run the OFMR scalp pipeline on a raw state table.

    Steps
    -----
    1. ``compute_features``            raw frame -> causal feature frame
    2. ``run_ood``                     raw temporal primitives -> latent states
    3. ``build_supervision_targets``   future returns + excursion labels
    4. ``run_gp``                      optional state-specific evolved features
    5. ``build_model_frame``           single assembly point (S1 contract)
    6. ``run_triple_barrier``          linreg -> metalabel -> barrier events
    7. ``evaluate_orderflow_policy``   the strategy's rules

    Parameters
    ----------
    df:
        Raw state table from ``load_data``.
    settings:
        Parsed ``settings.yaml``; loaded from disk when omitted.
    ood_detector:
        Optional stateful detector. Pass the same instance across data chunks
        to preserve learned latent prototypes; pass its ``state_dict()`` to
        persist and later restore those prototypes.
    ood_process_from:
        First row in ``df`` that should update a supplied detector. Include a
        raw context prefix for encoder warmup and set this to its first new row.
    """
    cfg = load_settings() if settings is None else settings

    steps = [
        "Features",
        "OOD states",
        "Supervision targets",
        "GP search",
        "Model frame",
        "Triple barrier / outcomes",
        "Policy",
    ]

    progress = tqdm(
        total=len(steps),
        desc="OFMR pipeline",
        unit="step",
        disable=not show_progress,
    )

    try:
        # 1. Features.
        progress.set_description("OFMR pipeline")
        progress.set_postfix_str("Features", refresh=True)

        feat_df, feat_cols = compute_features(
            df,
            **cfg["features"],
        )

        progress.update(1)

        # 2. OOD states.
        progress.set_postfix_str("OOD states", refresh=True)

        regimes = run_ood(
            df,
            cfg,
            detector=ood_detector,
            process_from=ood_process_from,
        )

        progress.update(1)

        # 3. Supervision targets.
        progress.set_postfix_str("Supervision targets", refresh=True)

        targets = build_supervision_targets(
            df,
            feat_df,
            cfg.get("outcome"),
        )

        progress.update(1)

        # 4. Optional GP.
        progress.set_postfix_str("GP search", refresh=True)

        gp_result = run_gp(
            df,
            feat_df,
            targets,
            regimes,
            cfg,
        )

        progress.update(1)

        # 5. Assemble model frame.
        progress.set_postfix_str("Model frame", refresh=True)

        model_frame, _aligned_regimes, _gp_feature_names = build_model_frame(
            df,
            feat_df,
            regimes,
            gp_result.model if gp_result is not None else None,
            targets=targets,
        )

        progress.update(1)

        # 6. Outcomes / barriers.
        progress.set_postfix_str("Triple barrier / outcomes", refresh=True)

        outcome = run_triple_barrier(
            model_frame,
            targets,
            cfg,
            raw=df,
        )

        progress.update(1)

        # 7. Policy.
        progress.set_postfix_str("Policy", refresh=True)

        """
        policy = evaluate_orderflow_policy(
            feat_df,
            feat_cols,
            **cfg["policy"],
        )
        """
        policy = pd.DataFrame()

        progress.update(1)

    finally:
        progress.close()

    return StrategyResult(
        features=model_frame,
        regimes=regimes,
        events=outcome.events,
        outcomes=targets,
        policy=policy,
        gp=gp_result,
        metalabel=outcome.metalabel,
    )


def strategy_testing(
    df: pd.DataFrame,
    settings: dict | None = None,
    ood_detector: OODSystem | None = None,
    ood_process_from: int = 0,
    show_progress: bool = True
) -> StrategyResult:
    """Run the OFMR scalp pipeline on a raw state table.

    Steps
    -----
    1. ``compute_features``            raw frame -> causal feature frame
    2. ``run_ood``                     raw temporal primitives -> latent states
    3. ``build_supervision_targets``   future returns + excursion labels
    4. ``run_gp``                      optional state-specific evolved features
    5. ``build_model_frame``           single assembly point (S1 contract)
    6. ``run_triple_barrier``          linreg -> metalabel -> barrier events
    7. ``evaluate_orderflow_policy``   the strategy's rules

    Parameters
    ----------
    df:
        Raw state table from ``load_data``.
    settings:
        Parsed ``settings.yaml``; loaded from disk when omitted.
    ood_detector:
        Optional stateful detector. Pass the same instance across data chunks
        to preserve learned latent prototypes; pass its ``state_dict()`` to
        persist and later restore those prototypes.
    ood_process_from:
        First row in ``df`` that should update a supplied detector. Include a
        raw context prefix for encoder warmup and set this to its first new row.
    """
    cfg = load_settings() if settings is None else settings

    steps = [
        "Features",
        "OOD states",
        "Supervision targets",
        "GP search",
        "Model frame",
        "Triple barrier / outcomes",
        "Policy",
    ]

    progress = tqdm(
        total=len(steps),
        desc="OFMR pipeline",
        unit="step",
        disable=not show_progress,
    )

    try:
        # 1. Features.
        progress.set_description("OFMR pipeline")
        progress.set_postfix_str("Features", refresh=True)

        feat_df, feat_cols = compute_features(
            df,
            **cfg["features"],
        )

        progress.update(1)

        # ------------------------------------------------------------------
        # Forecast-model comparison (research harness). Everything here is
        # LOCAL to strategy_testing -- including the horizon index ``ti`` -- so
        # it never collides with the coexisting module code. Three estimators
        # predict the SAME target (forward simple return over t+1..t+n) from
        # the causal feature frame:
        #   linreg : batch OLS             (sklearn LinearRegression)
        #   ridge  : batch L2 least squares(sklearn Ridge)
        #   kalman : online recursive least squares (a Kalman filter over the
        #            regression weights; predict-then-update -> out-of-sample)
        # Each model's t+1 prediction signs a side and the triple barrier runs
        # once per model. Results print and are stashed on the function
        # (strategy_testing.forecast_df / .barrier_summary). Never breaks the
        # main pipeline -- any failure is caught and reported.
        # ------------------------------------------------------------------
        def _compare_forecast_models(feat_df, feat_cols, fc_cfg):
            import numpy as np  # local: keeps numpy out of module scope
            from sklearn.linear_model import LinearRegression, Ridge

            n_steps = int(fc_cfg.get("n_steps", 3))
            ti = list(range(1, n_steps + 1))          # horizons t+1..t+n (LOCAL)
            train_fraction = float(fc_cfg.get("train_fraction", 0.70))
            ridge_alpha = float(fc_cfg.get("ridge_alpha", 1.0))
            forgetting = float(fc_cfg.get("kalman_forgetting", 0.99))
            p0 = float(fc_cfg.get("kalman_p0", 1.0e3))
            top_k = int(fc_cfg.get("kalman_top_k", 8))
            price_col = fc_cfg.get("price_col") or (
                "mid_price" if "mid_price" in feat_df.columns else "price_close")

            tb_cfg = dict((cfg.get("outcome") or {}).get("triple_barrier") or {})
            fixed_cfg = dict(tb_cfg.get("fixed") or {})
            pt = float(fixed_cfg.get("pt", 0.005))
            sl = float(fixed_cfg.get("sl", 0.003))
            horizon_bars = int(tb_cfg.get("horizon_bars", 30))

            price = pd.to_numeric(feat_df[price_col], errors="coerce")
            idx = feat_df.index
            n = len(feat_df)

            # forward simple returns r_t(h) = price[t+h] / price[t] - 1
            targets = {h: (price.shift(-h) / price - 1.0) for h in ti}

            # causal feature matrix (never the price target itself)
            feats = [c for c in feat_cols if c in feat_df.columns and c != price_col]
            X = feat_df[feats].apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan)

            train_rows = np.arange(0, max(int(n * train_fraction), 1))

            # standardise with TRAIN statistics only (no leakage into the tail)
            mu = X.iloc[train_rows].mean()
            sd = X.iloc[train_rows].std().replace(0.0, 1.0).fillna(1.0)
            Xz = ((X - mu) / sd).fillna(0.0)

            # top-k features by |corr| with the t+1 target on the train slice
            base_target = targets[ti[0]]
            corr = Xz.iloc[train_rows].corrwith(base_target.iloc[train_rows]).abs()
            sel = [c for c in corr.sort_values(ascending=False).index
                   if np.isfinite(corr[c])][:top_k]
            if not sel:
                sel = list(Xz.columns[:top_k])

            Y = np.column_stack([targets[h].to_numpy(dtype=float) for h in ti])  # (n, m)
            Xk = Xz[sel].to_numpy(dtype=float)                                   # (n, k)

            forecast = pd.DataFrame(index=idx)
            for h in ti:
                forecast[f"fwd_ret__t+{h}"] = targets[h].to_numpy()
            preds = {name: np.full((n, len(ti)), np.nan)
                     for name in ("linreg", "ridge", "kalman")}

            # ---- batch models: fit linreg + ridge on the chronological prefix ----
            fit_rows = [r for r in train_rows
                        if np.isfinite(Y[r]).all() and np.isfinite(Xk[r]).all()]
            if len(fit_rows) >= top_k + 1:
                ols = LinearRegression().fit(Xk[fit_rows], Y[fit_rows])
                rdg = Ridge(alpha=ridge_alpha).fit(Xk[fit_rows], Y[fit_rows])
                ok = np.isfinite(Xk).all(axis=1)
                preds["linreg"][ok] = ols.predict(Xk[ok])
                preds["ridge"][ok] = rdg.predict(Xk[ok])


            # ---- kalman filter: online recursive least squares ----------------
            # state: weights W (k, m); covariance P (k, k). Predict BEFORE the
            # update, so preds[t] never sees weights learned from row t.
            k, m = len(sel), len(ti)
            W = np.zeros((k, m), dtype=float)
            P = np.eye(k) * p0
            lam = forgetting
            kal = preds["kalman"]
            for t in range(n):
                x = Xk[t]
                if not np.isfinite(x).all():
                    continue
                kal[t] = x @ W                                   # out-of-sample prediction
                y = Y[t]
                if not np.isfinite(y).all():
                    continue
                Px = P @ x
                denom = lam + float(x @ Px)
                if denom <= 1e-12:
                    continue
                gain = Px / denom
                err = y - (x @ W)
                W = W + np.outer(gain, err)
                P = (P - np.outer(gain, Px)) / lam


            for name in ("linreg", "ridge", "kalman"):
                for j, h in enumerate(ti):
                    forecast[f"pred_{name}__t+{h}"] = preds[name][:, j]
            # linreg prediction error (actual - predicted), per horizon
            for h in ti:
                forecast[f"linreg_err__t+{h}"] = (
                    forecast[f"fwd_ret__t+{h}"] - forecast[f"pred_linreg__t+{h}"])

            # ---- triple barrier, run once per model --------------------------
            summary_rows = []
            for name in ("linreg", "ridge", "kalman"):
                side = np.sign(forecast[f"pred_{name}__t+{ti[0]}"]).fillna(0.0)
                tb = pd.DataFrame({price_col: price.to_numpy(),
                                   "side": side.to_numpy()}, index=idx)
                events = build_triple_barrier_sequence(
                    tb, mode="fixed", signal_col="side", price_col=price_col,
                    pt=pt, sl=sl, horizon_bars=horizon_bars,
                )
                if events.empty:
                    summary_rows.append({"model": name, "n_events": 0,
                                         "hit_rate": np.nan, "mean_return": np.nan,
                                         "profitable_frac": np.nan})
                    continue
                summary_rows.append({
                    "model": name,
                    "n_events": int(len(events)),
                    "hit_rate": float((events["label"] == 1).mean()),
                    "mean_return": float(events["return"].mean()),
                    "profitable_frac": float(events["meta_label"].mean()),
                })
            return forecast, pd.DataFrame(summary_rows)

        try:
            forecast_df, barrier_summary = _compare_forecast_models(
                feat_df, feat_cols, cfg.get("forecast_compare") or {})
            strategy_testing.forecast_df = forecast_df
            strategy_testing.barrier_summary = barrier_summary
            if show_progress:
                print("\n=== forecast-model comparison (t+1..t+n) ===")
                print(forecast_df.filter(like="pred_").head().round(6).to_string())
                print("\nlinreg error (head):")
                print(forecast_df.filter(like="linreg_err").head().round(6).to_string())
                print("\n=== triple-barrier comparison (one run per model) ===")
                print(barrier_summary.round(4).to_string(index=False))
                if barrier_summary["mean_return"].notna().any():
                    best = barrier_summary.loc[barrier_summary["mean_return"].idxmax(), "model"]
                    print(f"\nbest model by mean barrier return: {best}")
        except Exception as exc:  # research harness must never break the pipeline
            if show_progress:
                print(f"(forecast-model comparison skipped: {exc})")


        # 2. OOD states.
        progress.set_postfix_str("OOD states", refresh=True)

        regimes = run_ood(
            df,
            cfg,
            detector=ood_detector,
            process_from=ood_process_from,
        )

        progress.update(1)

        # 3. Supervision targets.
        progress.set_postfix_str("Supervision targets", refresh=True)

        targets = build_supervision_targets(
            df,
            feat_df,
            cfg.get("outcome"),
        )

        progress.update(1)

        # 4. Optional GP.
        progress.set_postfix_str("GP search", refresh=True)

        gp_result = run_gp(
            df,
            feat_df,
            targets,
            regimes,
            cfg,
        )

        progress.update(1)

        # 5. Assemble model frame.
        progress.set_postfix_str("Model frame", refresh=True)

        model_frame, _aligned_regimes, _gp_feature_names = build_model_frame(
            df,
            feat_df,
            regimes,
            gp_result.model if gp_result is not None else None,
            targets=targets,
        )

        progress.update(1)

        # 6. Outcomes / barriers.
        progress.set_postfix_str("Triple barrier / outcomes", refresh=True)

        outcome = run_triple_barrier(
            model_frame,
            targets,
            cfg,
            raw=df,
        )

        progress.update(1)

        # 7. Policy.
        progress.set_postfix_str("Policy", refresh=True)

        """
        policy = evaluate_orderflow_policy(
            feat_df,
            feat_cols,
            **cfg["policy"],
        )
        """
        policy = pd.DataFrame()

        progress.update(1)

    finally:
        progress.close()

    return StrategyResult(
        features=model_frame,
        regimes=regimes,
        events=outcome.events,
        outcomes=targets,
        policy=policy,
        gp=gp_result,
        metalabel=outcome.metalabel,
    )



if __name__ == "__main__":
    settings = load_settings(sys.argv[1] if len(sys.argv) > 1 else None)
    raw = load_data(settings)
    result = strategy_testing(raw, settings)

    print("\n=== features ===")
    print(f"{len(result.features)} rows, {len(result.features.columns)} columns")

    print("\n=== OOD latent states ===")
    print(result.regimes["latent_id"].value_counts(dropna=False).to_string())
    print(f"tradeable rows: {int(result.regimes['trade_allowed'].sum())}/{len(result.regimes)}")

    print("\n=== triple-barrier events ===")
    print(f"{len(result.events)} events")
    if not result.events.empty:
        print(result.events["barrier"].value_counts().to_string())
        if "trade_allowed" in result.events:
            print(f"known-state eligible events: {int(result.events['trade_allowed'].sum())}/{len(result.events)}")

    print("\n=== forward-return targets ===")
    print(f"{len(result.outcomes)} rows, {len(result.outcomes.columns)} target columns")

    print("\n=== metalabel ===")
    if result.metalabel is None or result.metalabel.report.empty:
        print("(no state had enough labeled training rows)")
    else:
        print(result.metalabel.report.to_string(index=False))

    print("\n=== policy report ===")
    if result.policy.empty:
        print("(not enough data for the policy study)")
    else:
        print(format_report(result.policy).to_string(index=False))

    if result.gp is not None:
        print("\n=== GP evolved features ===")
        if result.gp.report.empty:
            print("(no state/target pair had enough complete training samples)")
        else:
            print(result.gp.report.to_string(index=False))
