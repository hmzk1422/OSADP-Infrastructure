"""
regime_trees.py
===============
Method B - shallow supervised regression trees as an interpretability tool.

Goal: discover simple regions of entry-state space with materially different
future trade expectancy (e.g. "high vol + strong trend + positive OFI").
Not a predictive-accuracy exercise: `max_depth` is 2-4, `min_samples_leaf` is
large, and every tree is fit on the training segment of an expanding monthly
window and scored on the following unseen month.

The final "candidate filter" uses each fold's tree (fit on data strictly before
the test month) to label favorable leaves, evaluated pooled out-of-sample plus
per month.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ofmr.regime_data import (  # noqa: E402
    load_study_frame, make_folds, ret_metrics, bootstrap_ci, fmt_metrics,
    MIN_LEAF, ensure_out, ENTRY_STATE_FEATURES, STRAT_COLS,
)

MAX_DEPTH = 3
MIN_LEAF_MODEL = 150


def model_features() -> list[str]:
    """Entry-state + strategy-state + calendar features for the trees."""
    feats = list(ENTRY_STATE_FEATURES) + [
        "price_rel_sma14", "vol_prior20_range", "vol_prior20_mae",
        "hour_of_day",
    ]
    feats += [c for c in STRAT_COLS() if c.startswith(("strat_prev",))]
    return feats


def extract_rule_text(tree, feature_names, node=0, path=None, rules=None):
    """Recursively build readable if-else rules per tree leaf."""
    if path is None:
        path = []
        rules = {}
    t = tree.tree_
    if t.children_left[node] == t.children_right[node]:  # leaf
        rules[node] = " AND ".join(path) if path else "all"
        return rules
    feat = feature_names[t.feature[node]]
    thr = t.threshold[node]
    path.append(f"{feat} <= {thr:.5g}")
    extract_rule_text(tree, feature_names, t.children_left[node], path, rules)
    path.pop()
    path.append(f"{feat} > {thr:.5g}")
    extract_rule_text(tree, feature_names, t.children_right[node], path, rules)
    path.pop()
    return rules


def fit_tree(train_x: pd.DataFrame, train_y: np.ndarray, max_depth: int = MAX_DEPTH):
    """Small tree, big leaves, fixed seed - pure interpretability."""
    return DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=MIN_LEAF_MODEL,
        random_state=42, criterion="squared_error",
    ).fit(train_x, train_y)


def run_trees(df: pd.DataFrame, depths=(2, 3, 4)) -> dict:
    """Walk-forward tree study for several depths.

    Returns fold_rows (per fold+leaf train/test stats) and filter_parts
    (each fold's train-defined favorable leaves + their test sample count).
    """
    ensure_out()
    folds = make_folds(df)
    feats = model_features()
    x = df[feats].copy()
    y = df["ret"].to_numpy()

    fold_rows, filter_parts = [], []

    for depth in depths:
        for train_mask, test_mask, label in folds:
            usable_train = train_mask & df["_complete"]
            usable_test = test_mask & df["_complete"]
            Xtr = x.loc[usable_train]; ytr = y[usable_train.to_numpy()]
            Xte = x.loc[usable_test]; yte = y[usable_test.to_numpy()]
            if len(Xtr) < 4 * MIN_LEAF_MODEL:
                continue
            clf = fit_tree(Xtr, ytr, max_depth=depth)
            rules = extract_rule_text(clf, feats)

            tr_leaf = clf.apply(Xtr)
            te_leaf = clf.apply(Xte)
            tr_means = pd.Series(ytr).groupby(tr_leaf).mean()
            tr_ns = pd.Series(ytr).groupby(tr_leaf).size()
            base_tr = ytr.mean()

            for leaf in np.unique(te_leaf):
                mask = te_leaf == leaf
                m = ret_metrics(yte[mask])
                n_tr = int(tr_ns.get(leaf, 0))
                train_mean = float(tr_means.get(leaf, np.nan))
                fold_rows.append({
                    "depth": depth, "fold": label, "leaf": int(leaf),
                    "rule": rules.get(int(leaf), "?"),
                    "n_train": n_tr, "train_mean_r": train_mean,
                    "train_baseline_r": base_tr,
                    "n_test": m["n"], "test_mean_r": m["mean_r"],
                    "test_win_rate": m["win_rate"], "test_p_3r": m["p_3r"],
                    "test_pf": m["pf"], "test_total_r": m["total_r"],
                    "ci_lo": bootstrap_ci(yte[mask])[0],
                    "ci_hi": bootstrap_ci(yte[mask])[1],
                    "favorable_train": bool(n_tr >= MIN_LEAF_MODEL and train_mean > base_tr),
                })

            for leaf in np.unique(tr_leaf):
                tr_n = int(tr_ns.get(leaf, 0))
                tr_m = float(tr_means.get(leaf, np.nan))
                if tr_n < MIN_LEAF_MODEL:
                    continue
                n_te = int((te_leaf == leaf).sum())
                if n_te == 0:
                    continue
                filter_parts.append({
                    "fold": label, "depth": depth, "leaf": int(leaf),
                    "train_mean_r": tr_m, "train_n": tr_n,
                    "good_train": bool(tr_m > base_tr), "n_test": n_te,
                })

    fold_df = pd.DataFrame(fold_rows) if fold_rows else pd.DataFrame()
    fold_df.to_csv(os.path.join(ensure_out(), "trees_folds.csv"), index=False)
    return {"fold_rows": fold_df,
            "filter_parts": pd.DataFrame(filter_parts) if filter_parts else pd.DataFrame()}
def evaluate_filter(df: pd.DataFrame, depth: int = 3) -> dict:
    """Pooled OOS comparison of favorable-leaf vs rest.

    Each fold's tree (trained strictly before the test month) marks leaves
    favorable if leaf train meanR > fold train baseline with sufficient n.
    """
    ensure_out()
    folds = make_folds(df)
    feats = model_features()
    x = df[feats].copy()
    labels = pd.Series("unmodeled", index=df.index, dtype=object)
    param_rows = []
    for train_mask, test_mask, label in folds:
        usable_train = train_mask & df["_complete"]
        usable_test = test_mask & df["_complete"]
        Xtr = x.loc[usable_train]; ytr = df.loc[usable_train, "ret"].to_numpy()
        Xte = x.loc[usable_test]
        clf = fit_tree(Xtr, ytr, max_depth=depth)
        tr_leaf = clf.apply(Xtr)
        te_leaf = clf.apply(Xte)
        tr_means = pd.Series(ytr).groupby(tr_leaf).mean()
        tr_ns = pd.Series(ytr).groupby(tr_leaf).size()
        base = ytr.mean()
        for leaf in np.unique(te_leaf):
            tr_n = int(tr_ns.get(leaf, 0))
            tr_m = float(tr_means.get(leaf, np.nan))
            if tr_n < MIN_LEAF_MODEL or np.isnan(tr_m):
                lab = "small_leaf"
            elif tr_m > base:
                lab = "favorable"
            elif tr_m < base:
                lab = "unfavorable"
            else:
                lab = "neutral"
            te_positions = np.where(usable_test.to_numpy())[0]
            leaf_idx = df.index[te_positions[te_leaf == leaf]]
            labels.loc[leaf_idx] = lab
            param_rows.append({"fold": label, "leaf": int(leaf), "label": lab,
                               "train_n": tr_n, "train_mean_r": tr_m})
    pred_df = df.copy()
    pred_df["leaf_label"] = labels
    pred_df.to_csv(os.path.join(ensure_out(), "trees_predictions.csv"), index=False)
    pd.DataFrame(param_rows).to_csv(os.path.join(ensure_out(), "trees_labels.csv"),
                                    index=False)
    out = {"labels": labels, "pred_df": pred_df,
           "metrics": {lab: ret_metrics(pred_df[pred_df["leaf_label"] == lab]["ret"].to_numpy())
                       for lab in ["favorable", "unfavorable", "small_leaf", "neutral"]
                       if (pred_df["leaf_label"] == lab).sum() > 0}}
    out["metrics"]["baseline"] = ret_metrics(pred_df["ret"].to_numpy())
    return out


def stability_of_filter(df: pd.DataFrame, labels: pd.Series, lab: str = "favorable"):
    """Monthly stats for a labeled subset."""
    tmp = df.loc[labels.notna()].copy()
    tmp["_lab"] = labels.loc[tmp.index]
    sub = tmp[tmp["_lab"] == lab]
    rows = []
    for p, g in sub.groupby("year_month"):
        m = ret_metrics(g["ret"].to_numpy())
        lo, hi = bootstrap_ci(g["ret"].to_numpy())
        rows.append({"year_month": p, "n": m["n"], "mean_r": m["mean_r"],
                     "ci_lo": lo, "ci_hi": hi, "win_rate": m["win_rate"],
                     "pf": m["pf"], "total_r": m["total_r"]})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(ensure_out(), f"trees_filter_{lab}_monthly.csv"), index=False)
    return t


if __name__ == "__main__":
    d = load_study_frame()
    r = run_trees(d)
    print("trees fold rows:", len(r["fold_rows"]))
    ev = evaluate_filter(d)
    for k, v in ev["metrics"].items():
        print(k, fmt_metrics(v))
    print()
    print(stability_of_filter(d, ev["labels"]).to_string())