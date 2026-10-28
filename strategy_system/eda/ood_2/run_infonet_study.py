"""
run_infonet_study.py  --  drives the three hypotheses with ``infonet_study``.

Put next to base_regime_encoder.py etc. and run   python -m <your.package>.run_infonet_study

Fold geometry is in FILES (what ``load_L2(file_limit, start)`` understands).  Size it so that a TEST window holds
well over ~30 tradable episodes; with fewer, tau / I^2 / rank statistics are not interpretable (the report prints
episode counts so you can check).  Fold k loads files [start + k*step, +train) | [gap) | [test).
"""
from __future__ import annotations

import os
import pickle
import warnings

import pandas as pd

from .infonet_study import DataConfig, RunConfig, run_simple_study

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220, "display.max_columns", 60, "display.float_format", "{:.3f}".format)

# ---- EDIT ME ---------------------------------------------------------------------------------------------
DATA = DataConfig(
    train_files=10, gap_files=0, test_files=4, step_files=4, n_folds=4, start=0,
    decimate=5,                      # retain every fifth event after full-resolution feature calculation
    ts_unit="ms",                    # only used if load_L2's ``timestamp`` is numeric
)
CHECKPOINT = "regime_metrics_checkpoint_v3.pkl"
COMMON = dict(
    window=32, horizons=(1, 3, 13), trade_horizon=3, vol_window=20, min_dwell=10,
    min_episode_len=80, min_expert_rows=100, metrics=(), fee_bps=0.0,
    encoder_kwargs={"epochs": 5, "batch_size": 256},
)
# -----------------------------------------------------------------------------------------------------------

ENCODERS = ("ts2vec", "tcn", "vq-vae")
SEGMENTERS = ("kmeans", "hdbscan")
RUNS = []
for encoder in ENCODERS:
    for segmenter in SEGMENTERS:
        kwargs = {"backend_params": {"max_fit_samples": 50_000}} if segmenter == "kmeans" else {}
        RUNS.append(RunConfig(name=f"{encoder}_{segmenter}", feature_set="20", encoder=encoder,
                              segmenter=segmenter, segmenter_kwargs=kwargs, **COMMON))

if __name__ == "__main__":
    results = run_simple_study(DATA, RUNS, checkpoint_path=CHECKPOINT)
    tmp = "regime_metrics_results.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, "regime_metrics_results.pkl")
    print("\nExpectancy is trade-weighted; CVaR bounds are tail means of held-out episode expectancy.")
    print("Parity rows are experts; columns are regimes assigned to test episodes.")
    print("Correlation is between experts' episode-level net RR on the same test episodes.\n")
    for name, result in results.items():
        print(f"##### {name} #####")
        for fold in result["folds"]:
            if fold.get("skipped") or fold.get("error"):
                print(f"fold {fold['fold']}: {fold.get('skipped', fold.get('error'))}")
                continue
            print(f"fold {fold['fold']}: recurring latents {fold['recurring_latents']}")
            if fold["expectancy"].empty:
                if fold["recurring_latents"]:
                    if fold.get("eligible_test_rows", 0) == 0:
                        print("No valid held-out RR/cost rows (flat prices, invalid quotes, or costs above the 5R gate).")
                    else:
                        print("No eligible test trades for these trained regimes.")
                else:
                    print("No recurring training regime had enough rows to fit an expert.")
            else:
                print("Expectancy: latent | episodes | trades | mean RR | CVaR 0.05 | CVaR 0.95")
                print(fold["expectancy"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            if fold["persistence"].empty:
                print("No recurring test episodes.")
            else:
                print("Persistence: latent | episodes | mean seconds | CVaR 0.05 seconds | CVaR 0.95 seconds")
                print(fold["persistence"][["latent", "episodes", "mean_duration_sec", "cvar005_sec", "cvar095_sec"]]
                      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("Parity expectancy matrix (expert × assigned regime):")
            print(fold["parity"].to_string(float_format=lambda x: f"{x:.4f}", na_rep="—")
                  if not fold["parity"].empty else "No parity observations.")
            print("Episode RR correlation matrix:")
            print(fold["correlation"].to_string(float_format=lambda x: f"{x:.4f}")
                  if not fold["correlation"].empty else "Not enough common episodes for correlation.")
            print()
