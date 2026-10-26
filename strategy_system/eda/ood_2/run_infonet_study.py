"""
run_infonet_study.py  --  drives the three hypotheses with ``infonet_study``.

Put next to base_regime_encoder.py etc. and run   python -m <your.package>.run_infonet_study

Fold geometry is in FILES (what ``load_L2(file_limit, start)`` understands).  Size it so that a TEST window holds
well over ~30 tradable episodes; with fewer, tau / I^2 / rank statistics are not interpretable (the report prints
episode counts so you can check).  Fold k loads files [start + k*step, +train) | [gap) | [test).
"""
from __future__ import annotations

import pickle
import warnings

import pandas as pd

from .infonet_study import (DataConfig, RunConfig, compare, economics, metric_value, report, run_study)

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220, "display.max_columns", 60, "display.float_format", "{:.3f}".format)

# ---- EDIT ME ---------------------------------------------------------------------------------------------
DATA = DataConfig(
    train_files=10, gap_files=0, test_files=4, step_files=4, n_folds=6, start=0,
    decimate=5,                      # retain every fifth event after full-resolution feature calculation
    ts_unit="ms",                    # only used if load_L2's ``timestamp`` is numeric
)
CHECKPOINT = "infonet_study_checkpoint_v2.pkl"  # saved after each run/fold and resumed automatically
COMMON = dict(
    # Scale row-based settings to keep roughly the same event-time horizons
    # and episode lengths after retaining every fifth event.
    window=8, horizons=(1, 3, 13), trade_horizon=3, block=40, min_dwell=10,
    min_episode_len=80, min_expert_rows=100,   # InfoNet: block >= 2*max(horizon)+1
    fee_bps=0.0,                     # per side, ON TOP of crossing the spread -- set to your venue's taker fee
    max_test_episodes=60, max_ref_episodes=6, n_placebo=20,
)
# -----------------------------------------------------------------------------------------------------------

# H1 -- feature count.  Same encoder / segmenter, all four metrics.  decoy40 = 10 real + 30 time-shifted copies:
#       if decoy40 "improves" distinctness as much as the real 40, it was dimensionality, not economics.
H1 = [RunConfig(name=f"H1_{fs}", feature_set=fs, encoder="temporal_window", segmenter="kmeans", **COMMON)
      for fs in ("10", "20", "40", "decoy40")]

# H2 -- encoder x segmenter, judged on persistence (metrics=() skips the expensive similarity step).
#       Add "tcn" / "ts2vec" / "vq-vae" if torch is available (pass epochs etc. via encoder_kwargs).
H2 = [RunConfig(name=f"H2_{enc}_{seg}", feature_set="20", encoder=enc, segmenter=seg, metrics=(), **COMMON)
      for enc in ("temporal_window", "pca") for seg in ("kmeans", "custom", "sticky_hmm", "hdbscan")]

# H3 -- metric comparison needs no extra runs: every H1 run scores each (episode, regime) pair with cosine,
#       geometric, tdmi and infonet.  Read metric_value(results["H1_20"]).

if __name__ == "__main__":
    results = run_study(DATA, H1 + H2, checkpoint_path=CHECKPOINT)
    with open("infonet_study_results.pkl", "wb") as fh:
        pickle.dump(results, fh)

    table = compare(results)
    print("\n##### H1: feature count (10/20/40 vs decoy40) #####")
    print(table.loc[[r.name for r in H1]].T)
    print("\n##### H2: persistence by encoder x segmenter (sorted by tradable share) #####")
    cols = ["test_median_len", "test_median_sec", "test_tradable_share", "test_flicker_share", "test_n_latents",
            "test_share_null", "test_recurring_latents", "own_rr", "cross_rr", "own_minus_cross", "p_vs_placebo"]
    print(table.loc[[r.name for r in H2], [c for c in cols if c in table]]
          .sort_values("test_tradable_share", ascending=False))
    print("\n##### H3: similarity metrics (from H1_20) #####")
    print(metric_value(results["H1_20"]).T)
    report({"H1_20": results["H1_20"]})
