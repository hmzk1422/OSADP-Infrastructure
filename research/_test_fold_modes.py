"""Smoke test for the fold-scheme switch in regime_data.make_folds (monthly vs purged)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import numpy as np
import pandas as pd
import research.regime_data as rd

# Load the light parts of the frame only (make_folds needs dt + year_month).
df = rd.load_trade_log()
df = rd.add_entry_state(df)
df["_complete"] = True
print(f"stream: {len(df)} trades, {df['dt'].min()} .. {df['dt'].max()}")

# --- monthly mode (default) -------------------------------------------------
rd.FOLD_MODE = "monthly"
folds_m = rd.make_folds(df)
print(f"\n[monthly] {len(folds_m)} folds")
for tr, te, lab in folds_m[:3]:
    print(f"  {lab}: train={int(tr.sum())} test={int(te.sum())}")
assert len(folds_m) > 0
# masks must be boolean Series aligned to df.index
tr, te, lab = folds_m[0]
assert isinstance(tr, pd.Series) and tr.index.equals(df.index)

# --- purged mode -------------------------------------------------------------
rd.FOLD_MODE = "purged"
folds_p = rd.make_folds(df)
print(f"\n[purged] {len(folds_p)} folds (n_samples={rd.FOLD_N_SAMPLES}, "
      f"block={rd.FOLD_BLOCK_SIZE}, horizon={rd.FOLD_HORIZON_SECONDS}s, "
      f"embargo={rd.FOLD_EMBARGO_SECONDS}s)")
for tr, te, lab in folds_p:
    print(f"  {lab}: train={int(tr.sum())} test={int(te.sum())}")

assert len(folds_p) > 0
dts = df["dt"].to_numpy()
test_starts, test_ends = [], []
for tr, te, lab in folds_p:
    tr_pos = np.where(tr.to_numpy())[0]
    te_pos = np.where(te.to_numpy())[0]
    # 1. test blocks are contiguous and non-overlapping, after the train window
    assert (np.diff(te_pos) == 1).all(), "test block not contiguous"
    assert te_pos[0] >= tr_pos[-1], "test block overlaps train window"
    # 2. purge respected: last train trade is > horizon+embargo before test start
    gap_s = (dts[te_pos[0]] - dts[tr_pos[-1]]) / np.timedelta64(1, "s")
    assert gap_s > rd.FOLD_HORIZON_SECONDS + rd.FOLD_EMBARGO_SECONDS, \
        f"purge violated: gap={gap_s}s"
    test_starts.append(dts[te_pos[0]])
    test_ends.append(dts[te_pos[-1]])
    # 3. consumer-idiom compatibility (mirrors regime_trees / regime_hmm usage)
    use = tr & df["_complete"]
    assert use.sum() == tr.sum()
    assert len(df.index[np.where(use.to_numpy())[0]]) == int(use.sum())

# strictly chronological, non-overlapping test blocks
test_starts = np.array(test_starts)
test_ends = np.array(test_ends)
assert (np.diff(test_starts) > np.timedelta64(0, "s")).all(), "test blocks overlap in time"
assert (test_starts[1:] > test_ends[:-1]).all(), "test blocks overlap in time"

print("\n[OK] monthly + purged fold schemes validated:")
print(f"  test block 1: {test_starts[0]} .. {test_ends[0]}")
print(f"  test block N: {test_starts[-1]} .. {test_ends[-1]}")
