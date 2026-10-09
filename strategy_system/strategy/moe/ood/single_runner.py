"""Run ONLY the OOD / latent-regime stage and print recurrence + similarity diagnostics.

No features, no GP, no targets, no barriers, no policy.

Put this file next to strategy.py (strategy_system/strategy/moe/) and run from
the repository root:

    python -m strategy_system.strategy.moe.run_ood_only
    python -m strategy_system.strategy.moe.run_ood_only path/to/settings.yaml --rows 50000
    python -m strategy_system.strategy.moe.run_ood_only --train-frac 0.6 --out ood_out

Pipeline: load_data -> run_ood (OODSystem.detect) -> validate_regimes -> report.

Preferred running method: python -m strategy_system.strategy.moe.ood.single_runner strategy_system/strategy/moe/settings.yaml --train-frac 0.7 --out ood_out 
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ....exchange.loader import load_L1, load_L2
from .ood import OODSystem

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"
DATA_LOADERS = {"orderbook": load_L2, "ohlcv": load_L1}


# --------------------------------------------------------------------------- #
# Same two steps strategy.py does before OOD: load settings, load raw data.
# --------------------------------------------------------------------------- #
def load_settings(path) -> dict:
    with Path(path or DEFAULT_SETTINGS_PATH).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(settings: dict) -> pd.DataFrame:
    data_cfg = settings["data"]
    source = data_cfg["source"]
    return DATA_LOADERS[source](**data_cfg.get(source, {}))


# --------------------------------------------------------------------------- #
# The OOD call itself (identical to strategy.run_ood).
# --------------------------------------------------------------------------- #
def run_ood(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, OODSystem]:
    detector = OODSystem(**cfg.get("ood", {}))
    regimes = detector.detect(raw)  # one causal pass over the whole stream
    return regimes, detector


# --------------------------------------------------------------------------- #
# Recurrence: does each latent come back, and does it come back OUT OF SAMPLE?
# --------------------------------------------------------------------------- #
def episode_runs(labels: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """Consecutive runs of the same label -> {state: [(start, end), ...]}; null excluded."""
    runs: dict[str, list[tuple[int, int]]] = {}
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != "null":
                runs.setdefault(labels[start], []).append((start, i))
            start = i
    return runs


def recurrence_table(regimes: pd.DataFrame, split: int) -> pd.DataFrame:
    """Per-latent first-seen, rows/episodes before vs after ``split`` (train vs test).

    The detector is a single causal online pass, so rows after ``split`` were
    labelled using only prototypes learned from data available at that time:
    slicing the output at ``split`` IS an honest train/test split.
    """
    labels = regimes["latent_id"].astype(str).to_numpy()
    runs = episode_runs(labels)
    rows = []
    for state, spans in sorted(runs.items()):
        first = spans[0][0]
        tr = [s for s in spans if s[0] < split]
        te = [s for s in spans if s[0] >= split]
        lengths = np.array([e - s for s, e in spans])
        if first >= split:
            verdict = "BORN IN TEST (never seen in train)"
        elif te:
            verdict = "RECURS IN TEST"
        else:
            verdict = "train only (did not return)"
        rows.append({
            "latent_id": state, "first_seen_row": first,
            "rows_train": int(sum(e - s for s, e in tr)), "rows_test": int(sum(e - s for s, e in te)),
            "episodes_train": len(tr), "episodes_test": len(te), "episodes_total": len(spans),
            "median_episode_len": float(np.median(lengths)), "verdict": verdict,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("settings", nargs="?", default=None)
    ap.add_argument("--rows", type=int, default=None, help="use only the first N rows (quick test)")
    ap.add_argument("--train-frac", type=float, default=0.70, help="chronological train/test split")
    ap.add_argument("--out", default="ood_out", help="output folder for CSVs / PNG")
    args = ap.parse_args()

    pd.set_option("display.width", 200, "display.max_columns", 50)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_settings(args.settings)
    raw = load_data(cfg)
    if args.rows:
        raw = raw.iloc[: args.rows]
    split = int(len(raw) * args.train_frac)
    print(f"raw rows: {len(raw)} | train: [0,{split}) | test: [{split},{len(raw)})")

    regimes, detector = run_ood(raw, cfg)
    regimes.to_csv(out / "regimes.csv")

    # 1) latent counts, with null
    print("\n=== latent state row counts (null included) ===")
    counts = regimes["latent_id"].value_counts(dropna=False)
    print(pd.DataFrame({"rows": counts, "share": (counts / len(regimes)).round(3)}).to_string())
    print(f"trade_allowed rows: {int(regimes['trade_allowed'].sum())}/{len(regimes)}")
    print("status breakdown:", regimes["status"].value_counts().to_dict())
    print("null share  train: %.3f | test: %.3f" % (
        (regimes["latent_id"].iloc[:split] == "null").mean(),
        (regimes["latent_id"].iloc[split:] == "null").mean()))

    # 2) recurrence, train vs test
    print("\n=== recurrence (train -> test) ===")
    rec = recurrence_table(regimes, split)
    if rec.empty:
        print("No latent was ever promoted (everything is null). See 'status' above; "
              "try lowering ood.min_novel_observations / min_novel_episodes / raising ood.novel_threshold.")
    else:
        print(rec.to_string(index=False))
        rec.to_csv(out / "recurrence.csv", index=False)

    # 3) validator diagnostics (similarity matrices etc.)
    print("\n=== validator: persistence ===")
    report = detector.validate_regimes()
    print(report["persistence"].round(2).to_string())
    print(f"\ntransition_stability (post-smoothing): {report['transition_stability']:.4f}")
    print(f"null_regime_detected (>=50% null): {report['null_regime_detected']}")
    print(f"within/between separation ratio: {report.get('within_between_separation_ratio')}")

    matrices = {k: report[k] for k in ("embedding_similarity", "temporal_pattern_similarity",
                                        "episode_duration_similarity", "feature_similarity")
                if k in report}
    protos = detector.segmenter.get_regime_prototypes()
    if len(protos) >= 2:
        matrices["prototype_pearson_corr"] = pd.DataFrame(protos).corr()
    for name, m in matrices.items():
        print(f"\n=== {name} ===")
        print(m.round(2).to_string())
        m.to_csv(out / f"{name}.csv")

    # 4) optional picture
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n in ("embedding_similarity", "temporal_pattern_similarity") if n in matrices]
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(6 * (len(names) + 1), 5))
        ids = sorted(set(regimes["latent_id"].astype(str)) - {"null"})
        code = regimes["latent_id"].astype(str).map({s: i + 1 for i, s in enumerate(ids)}).fillna(0)
        axes[0].plot(code.to_numpy(), drawstyle="steps-post", lw=0.6)
        axes[0].axvline(split, color="r", ls="--", label="train|test")
        axes[0].set_yticks(range(len(ids) + 1), ["null"] + ids)
        axes[0].set_title("latent timeline"); axes[0].legend()
        for ax, name in zip(axes[1:], names):
            m = matrices[name]
            ax.imshow(m.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(m)), m.columns, rotation=90)
            ax.set_yticks(range(len(m)), m.index)
            ax.set_title(name + "\n(diagonal = same state, different episodes)")
        fig.tight_layout(); fig.savefig(out / "ood_overview.png", dpi=110)
        print(f"\nsaved plot -> {out / 'ood_overview.png'}")
    except Exception as exc:  # plotting is optional
        print(f"(plot skipped: {exc})")

    print(f"\nall outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()