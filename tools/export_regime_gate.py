#!/usr/bin/env python3
"""Export the REGIME-DETECTION GATE (research DecisionTree) consumed by live-trade.

The gate sits BETWEEN the main executor and execution: at every entry candidate
the engine classifies the current entry-state+strategy-state vector into a leaf
of the walk-forward tree, and voids the trade when the leaf is not favorable.
Voided candidates are shadow-recorded (build/gate_log_<id>.jsonl) so their data
never disappears.

This is the EXACT research artifact (research/ofmr/regime_trees.py):
  * feature set = regime_trees.model_features() (9 entry-state + price_rel_sma14
    + vol_prior20{range,mae} + hour_of_day + 15 rolling strat_prev* vars = 28),
  * learner = DecisionTreeRegressor(max_depth=3, min_samples_leaf=150,
    random_state=42, criterion="squared_error"),
  * walk-forward folds = regime_data.make_folds() (expanding train, test = next
    month), fit on the TRAIN segment of the LAST fold only -> zero lookahead,
  * leaf label = favorable iff leaf train mean-R > fold train baseline with
    n_leaf >= 150 (>= MIN_LEAF_MODEL).

Tree is exported as plain JSON with sklearn NaN semantics: feature <= threshold
-> left child; NaN -> left child; leaves have left==right==-1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ofmr import regime_data  # noqa: E402
from research.ofmr import regime_trees  # noqa: E402

MIN_LEAF_MODEL = 150


def export_regime_gate(out_path: str = "build/regime_gate.json",
                       log_path: str = "build/walk_forward_trade_log_v3.csv") -> int:
    df = regime_data.load_study_frame()
    if len(df) == 0:
        print(f"ERROR: empty study frame from {log_path}", file=sys.stderr)
        return 1

    feats = list(regime_trees.model_features())
    x = df[feats].to_numpy(dtype=float)
    y = df["ret"].to_numpy(dtype=float)
    folds = regime_data.make_folds(df)
    if not folds:
        print("ERROR: no folds produced", file=sys.stderr)
        return 1

    # Fit on the TRAIN side of the LAST fold - the only slice the gate is allowed
    # to see (research label rule). The test segment is never touched.
    train_mask, _test_mask, label = folds[-1]
    usable = train_mask & df["_complete"] if "_complete" in df.columns else train_mask
    idx = usable.to_numpy().nonzero()[0]
    Xtr = pd.DataFrame(x[idx], columns=feats, index=df.index[idx])
    ytr = pd.Series(y[idx], index=df.index[idx])
    if len(Xtr) < 4 * MIN_LEAF_MODEL:
        print(f"ERROR: too little train data ({len(Xtr)} rows)", file=sys.stderr)
        return 1

    clf = regime_trees.fit_tree(Xtr, ytr, max_depth=regime_trees.MAX_DEPTH)
    base = float(ytr.mean())

    # Export the sklearn tree_ as plain JSON.
    t = clf.tree_
    nodes = []
    favorable = []
    leaf_stats: dict = {}
    for i in range(t.node_count):
        is_leaf = t.children_left[i] == t.children_right[i]
        if is_leaf:
            nodes.append({"feature": -2, "threshold": None, "left": -1, "right": -1})
            favorable.append(False)
        else:
            nodes.append({
                "feature": int(t.feature[i]),
                "threshold": float(t.threshold[i]),
                "left": int(t.children_left[i]),
                "right": int(t.children_right[i]),
            })
            favorable.append(False)

    # Label the leaves: favorable iff mean-R > baseline AND n >= MIN_LEAF_MODEL.
    tr_leaf = clf.apply(Xtr)
    for leaf_id in range(t.node_count):
        if t.children_left[leaf_id] != t.children_right[leaf_id]:
            continue
        mask = tr_leaf == leaf_id
        if mask.sum() == 0:
            continue
        n_i = int(mask.sum())
        m_i = float(ytr[mask].mean())
        is_fav = n_i >= MIN_LEAF_MODEL and not pd.isna(m_i) and m_i > base
        favorable[leaf_id] = is_fav
        leaf_stats[leaf_id] = {"n": n_i, "mean_r": m_i}

    out = {
        "schema": "regime_tree_gate_v1",
        "max_depth": regime_trees.MAX_DEPTH,
        "features": feats,
        "nodes": nodes,
        "favorable": favorable,
        "leaf_stats": {str(k): v for k, v in leaf_stats.items()},
        "train_baseline_r": base,
        "train_label": label,
        "n_train": int(len(Xtr)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    fav_n = sum(1 for f in favorable if f)
    print(f"regime tree gate -> {out_path}")
    print(f"  features: {len(feats)}  nodes: {t.node_count}  train: {len(Xtr)} trades  fold: {label}")
    print(f"  favorable leaves: {fav_n}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="build/regime_gate.json")
    ap.add_argument("--log", default="build/walk_forward_trade_log_v3.csv")
    a = ap.parse_args()
    return export_regime_gate(a.out, a.log)


if __name__ == "__main__":
    raise SystemExit(main())