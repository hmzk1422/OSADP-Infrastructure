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

from .infonet_study import DataConfig, RunConfig, final_report, run_simple_study

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.set_option("display.width", 220, "display.max_columns", 60, "display.float_format", "{:.3f}".format)

# ---- EDIT ME ---------------------------------------------------------------------------------------------
DATA = DataConfig(
    train_files=6, gap_files=0, test_files=4, step_files=10, n_folds=72, start=0,
    # Non-overlapping 10-file folds (6 train + 4 test) tile the full 720-file matched month
    # exactly (step_files == train_files + test_files).  Keeps each fold ~660k rows so
    # add_orderflow/add_flow_price_features (which copy & consolidate before decimation)
    # cannot allocate the ~2.4 GiB array that 30-file folds triggered.
    decimate=5,                      # retain every fifth event after full-resolution feature calculation
    ts_unit="ms",                    # only used if load_L2's ``timestamp`` is numeric
)
CHECKPOINT = "regime_forecast_mae_checkpoint_v1.pkl"
REPORT_TXT = "regime_report.txt"
COMMON = dict(
    strategy="forecast_mae", window=32, horizons=(1, 2, 3, 4, 5),
    trade_horizon=1,          # legacy RR holding slot; forecast_mae uses cfg.horizons, keep coherent with horizons
    min_dwell=10, min_episode_len=80, min_expert_rows=100, fee_bps=0.0,
    block=40,                                             # retained for the optional InfoNET metrics path
    metrics=(), max_test_episodes=40, min_latents=2,
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
    text = final_report(results)
    print("\n" + text)
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write(text)
