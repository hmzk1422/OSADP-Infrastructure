"""
run_infonet_study.py  --  drives the three hypotheses with ``infonet_study``.

Put next to base_regime_encoder.py etc. and run   python -m <your.package>.run_infonet_study

Fold geometry is in FILES (what ``load_L2(file_limit, start)`` understands).  Size it so that a TEST window holds
well over ~30 tradable episodes; with fewer, tau / I^2 / rank statistics are not interpretable (the report prints
episode counts so you can check).  Fold k loads files [start + k*step, +train) | [gap) | [test).

Changes vs the first version
  * step 0 runs ``causality_check`` on every (encoder, segmenter) pair: a segmenter that peeks at future rows
    invalidates every number below it, so the grid is only trusted for pairs that pass.
  * n_placebo 20 -> 200 (with 20 draws the smallest attainable p-value is 1/21 = 0.048).
  * H2 is no longer ranked on tradable share alone: one giant cluster has a tradable share of 100%.  The table is
    filtered to segmenters that actually produce >= 2 latents and ranked on the LIVE own-vs-global edge.
  * the LIVE arm (no look-ahead on episode length, confirmation lag, regime-free expert on the same rows) is shown.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import replace

import pandas as pd

from ...exchange.loader import load_L2
from .infonet_study import (DataConfig, RunConfig, causality_check, compare, economics, load_fold, metric_value,
                            report, run_study)

# silence only the expected all-NaN-slice noise; do NOT blanket-ignore RuntimeWarning (it hides real numeric problems)
for msg in ("Mean of empty slice", "invalid value encountered", "Degrees of freedom <= 0"):
    warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)
pd.set_option("display.width", 220, "display.max_columns", 60, "display.float_format", "{:.3f}".format)

# ---- EDIT ME ---------------------------------------------------------------------------------------------
DATA = DataConfig(
    train_files=10, gap_files=0, test_files=4, step_files=4, n_folds=6, start=0,
    decimate=1,                      # raise (e.g. 5) to thin events if encoding the full stream is too slow
    ts_unit="ms",                    # only used if load_L2's ``timestamp`` is numeric
)
COMMON = dict(
    horizons=(4, 16, 64), trade_horizon=16, block=200, min_episode_len=400,   # InfoNet: block >= 2*max(horizon)+1
    fee_bps=0.0,                     # per side, ON TOP of crossing the spread -- set to your venue's taker fee (!)
    exec_delay=1,                    # events between signal and fill; 0 trades on the very quote that made the signal
    confirm_rows=50,                 # LIVE arm: trade a regime only after its label has been on this many rows
    max_test_episodes=60, max_ref_episodes=6, n_placebo=200,
)
# -----------------------------------------------------------------------------------------------------------

# H1 -- feature count.  Same encoder / segmenter, all four metrics.  decoy40 = 10 real + 30 time-shifted copies:
#       if decoy40 "improves" distinctness as much as the real 40, it was dimensionality, not economics.
H1 = [RunConfig(name=f"H1_{fs}", feature_set=fs, encoder="temporal_window", segmenter="kmeans", **COMMON)
      for fs in ("10", "20", "40", "decoy40")]

# H2 -- encoder x segmenter, judged on persistence AND on whether the persistent states carry a distinct edge
#       (metrics=() skips the expensive similarity step).
#       Add "tcn" / "ts2vec" / "vq-vae" if torch is available (pass epochs etc. via encoder_kwargs).
H2 = [RunConfig(name=f"H2_{enc}_{seg}", feature_set="20", encoder=enc, segmenter=seg, metrics=(), **COMMON)
      for enc in ("temporal_window", "pca") for seg in ("kmeans", "custom", "sticky_hmm", "hdbscan")]

# H3 -- metric comparison needs no extra runs: every H1 run scores each (episode, regime) pair with cosine,
#       geometric, tdmi and infonet.  Read metric_value(results["H1_20"]).

if __name__ == "__main__":
    # ---- step 0: causality gate (fold 0, every distinct encoder x segmenter) --------------------------------
    fd0 = load_fold(load_L2, 0, replace(DATA, feature_sets=("20",)))
    print("\n##### step 0: causality check (test_agreement must be exactly 1.000 and emb_max_abs_diff 0) #####")
    gate = {}
    for r in H1[:1] + H2:
        gate[r.name] = causality_check(fd0, r)
    gate = pd.DataFrame(gate).T
    print(gate)
    bad = gate[(gate["test_agreement"] < 1.0) | (gate["emb_max_abs_diff"] > 1e-9)].index.tolist()
    if bad:
        print("!! LOOK-AHEAD (or non-deterministic fit) in:", bad, "-- their results below are NOT tradable evidence.")
    del fd0

    results = run_study(DATA, H1 + H2)
    with open("infonet_study_results.pkl", "wb") as fh:
        pickle.dump(results, fh)

    table = compare(results)
    live_cols = ["live_own_rr", "live_global_rr", "live_own_minus_global", "live_lo", "live_hi"]
    print("\n##### H1: feature count (10/20/40 vs decoy40) #####")
    print(table.loc[[r.name for r in H1]].T)

    print("\n##### H2: persistence + edge by encoder x segmenter #####")
    cols = ["test_n_latents", "test_median_len", "test_median_sec", "test_tradable_share", "test_flicker_share",
            "test_share_null", "test_recurring_latents", "own_rr", "cross_rr", "own_minus_cross", "omc_folds_positive",
            "p_vs_placebo"] + live_cols
    h2 = table.loc[[r.name for r in H2], [c for c in cols if c in table]]
    h2["degenerate"] = h2["test_n_latents"] < 2         # one giant cluster is 'persistent' by construction
    h2["look_ahead"] = [n in bad for n in h2.index]
    print(h2.sort_values("test_tradable_share", ascending=False))
    ok = h2[~h2["degenerate"] & ~h2["look_ahead"] & (h2["test_recurring_latents"] >= 1)]
    print("\n--- eligible (>= 2 latents, >= 1 recurring, causal), ranked by LIVE own-minus-global edge ---")
    print(ok.sort_values("live_own_minus_global", ascending=False) if len(ok) else "none eligible")

    print("\n##### H3: similarity metrics (from H1_20) #####")
    print(metric_value(results["H1_20"]).T)
    report({"H1_20": results["H1_20"]})
