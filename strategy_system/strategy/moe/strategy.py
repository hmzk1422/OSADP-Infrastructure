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

        policy = evaluate_orderflow_policy(
            feat_df,
            feat_cols,
            **cfg["policy"],
        )

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
    result = strategy(raw, settings)

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
