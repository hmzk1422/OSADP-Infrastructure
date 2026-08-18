"""
Feature Redundancy Diagnostics — VIF, Orthogonalization & Ablation
====================================================================
Addresses the multicollinearity question raised for the top-5
head/tail PFI + SHAP features and turns "run VIF or Gram-Schmidt" into
an actual, falsifiable test instead of a vibe.

WHY THIS SCRIPT EXISTS (read this before trusting a single number below)
--------------------------------------------------------------------------
PFI and SHAP answer "how much does the *fitted* model rely on this
feature", not "how much unique information does this feature carry".
Both are known to mis-attribute importance under collinearity:

  - PFI: permuting one of two correlated features barely hurts the
    model, because the model can still route through its correlated
    twin. This DEFLATES the permuted feature's PFI even if it is
    genuinely informative. https://scikit-learn.org/stable/modules/permutation_importance.html#misleading-values-on-strongly-correlated-features
  - SHAP (TreeExplainer, interventional/tree_path_dependent default):
    Shapley values are "true to the model", not "true to the data" —
    under correlation, credit gets *split* between correlated features
    in a way that depends on tree structure, not on which feature is
    causally upstream.

So before this codebase concludes "orderflow decays" or "these 5
features matter and these 5 don't", we need to know whether the
head/tail split is a real information gradient or a collinearity
artifact of splitting credit across near-duplicate features
(e.g. cvd_delta_z and cvd_slope_z are both derivatives of the same
underlying signed-volume series as ofi_z).

THREE TOOLS, USED FOR THREE DIFFERENT JOBS (do not conflate them):
  1. VIF            -> DIAGNOSIS ONLY. Flags *that* a feature's variance
                        is linearly explained by the others. Says
                        nothing about whether that shared variance is
                        useful, and nothing about nonlinear redundancy
                        (which is most of what a GBM actually exploits).
  2. QR / Gram-Schmidt orthogonalization -> a TREATMENT for the linear
                        part of the redundancy, used here narrowly to
                        ask "does a tail feature still explain the
                        target once head-feature variance is removed
                        from it?" i.e. a residual-alpha test. It is
                        NOT used to rebuild the production feature set,
                        because it destroys interpretability of the
                        transformed feature and is order-dependent.
  3. Leave-one-out ablation (retrain, not just re-score) -> the only
                        one of the three that is model-agnostic and
                        captures *nonlinear* redundancy too. This is
                        the closest thing to a ground-truth answer to
                        "does removing this feature actually cost
                        performance", and is reported against BOTH the
                        full model and a no-orderflow baseline, exactly
                        as requested.

Bottom line stated up front (see the printed VERDICT section for the
data-driven version): VIF/orthogonalization are good, cheap, standard
DIAGNOSTIC hygiene and should be run -- but they are not, by
themselves, evidence for or against "orderflow decay". That claim can
only be supported by the ablation deltas (retrained, out-of-sample)
and by the longitudinal test in `longitudinal_decay_testing.py`. If
this script is used to justify RQ2 on its own, that is a methods
error -- flag it.

WHAT THIS SCRIPT DOES NOT DO
------------------------------
It does not touch `ml_state_optimized.py`'s training path. STEP 1-6
below is a *minimal, read-only mirror* of that script's data
preparation (same functions, same config, same purged split scheme)
just to reconstruct X/y/features/datetimes for diagnostics. If you
change feature engineering, barrier labeling, or the split scheme in
ml_state_optimized.py, mirror the change here too -- or better,
refactor STEP 1-6 in ml_state_optimized.py's __main__ into a
`prepare_training_matrix()` function that both scripts import, so this
duplication can't silently drift. That refactor is out of scope here
because ml_state_optimized.py's main block interleaves data prep with
report-capture/atexit side effects that shouldn't be imported blindly.

Usage:
    python -m research.feature_redundancy_diagnostics
"""
import os
import sys

import numpy as np
import polars as pl
import scipy.stats as st
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ml_shared_config import (  # noqa: E402
    BUILD_DIR, ORDERFLOW_FEATURE_COLUMNS, N_RANDOM_SAMPLES, TEST_BLOCK_SIZE,
    EMBARGO_SECONDS,
)

ALPHA = 0.05
TOP_K = 5  # head AND tail, per ranking (PFI, SHAP) -> up to 4*5=20 candidates, deduped


# ----------------------------------------------------------------------------
# STEP 0 — pull candidate features from the existing feature_importance_v3.csv
# (written by ml_state_optimized.py's STEP 7). We do NOT recompute SHAP/PFI
# here; we consume what the production walk-forward already produced, so
# this script's candidate set always matches whatever run generated the
# report you're reading.
# ----------------------------------------------------------------------------
def load_candidate_features(path: str = None) -> tuple[list[str], dict[str, set[str]]]:
    path = path or os.path.join(BUILD_DIR, "feature_importance_v3.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run ml_state_optimized.py first (STEP 7 writes this file) "
            f"before running feature_redundancy_diagnostics.py -- it consumes that output."
        )
    imp = pl.read_csv(path)

    membership: dict[str, set[str]] = {}

    def head_tail(col: str, tag: str):
        ranked = imp.sort(col, descending=True)
        head = ranked.head(TOP_K)["Feature"].to_list()
        tail = ranked.tail(TOP_K)["Feature"].to_list()
        for f in head:
            membership.setdefault(f, set()).add(f"{tag}_head")
        for f in tail:
            membership.setdefault(f, set()).add(f"{tag}_tail")

    head_tail("PFI_Mean_AUC_Drop", "PFI")
    head_tail("SHAP_Importance", "SHAP")

    candidates = sorted(membership.keys())
    return candidates, membership


# ----------------------------------------------------------------------------
# STEP 1-6 (read-only mirror of ml_state_optimized.py's data prep) — only far
# enough to reconstruct X / y / features / datetimes for diagnostics. See the
# module docstring for why this is a mirror rather than an import.
# ----------------------------------------------------------------------------
def prepare_training_matrix():
    from research.ml_state_optimized import (  # noqa: E402
        build_historical_state_space, add_barrier_labels, deduplicate_to_bar,
    )
    from main.config import (  # noqa: E402
        RSI_FILTER_COLUMN, RSI_FILTER_THRESHOLD, ENABLE_RSI_ACCEL_FILTER,
        TRADE_DIRECTION, FEATURE_COLUMNS,
    )
    from research.ml_shared_config import CHUNK_SESSIONS, N_CHUNKS_PER_SESSION, USE_ORDERFLOW_FEATURES

    df = build_historical_state_space(chunk_sessions=CHUNK_SESSIONS, chunks_per_session=N_CHUNKS_PER_SESSION)
    df = add_barrier_labels(df)
    df = df.filter(pl.col("volume") > 0)
    df = deduplicate_to_bar(df)

    rsi_high_thr = 100 - RSI_FILTER_THRESHOLD
    if ENABLE_RSI_ACCEL_FILTER:
        df_zone = (
            df.with_columns(
                pl.when((pl.col(RSI_FILTER_COLUMN) < RSI_FILTER_THRESHOLD) & (pl.col("ma_accel_ewma3") > 0)).then(1)
                  .when((pl.col(RSI_FILTER_COLUMN) > rsi_high_thr) & (pl.col("ma_accel_ewma3") < 0)).then(-1)
                  .otherwise(0).alias("zone_side")
            ).filter(pl.col("zone_side") != 0)
            .with_columns((pl.col("zone_side") * (-1 if TRADE_DIRECTION == "short" else 1)).alias("traded_side"))
            .drop("zone_side")
        )
    else:
        df_zone = (
            df.filter(pl.col(RSI_FILTER_COLUMN) < RSI_FILTER_THRESHOLD)
              .with_columns(pl.lit(-1 if TRADE_DIRECTION == "short" else 1).alias("traded_side"))
        )

    if USE_ORDERFLOW_FEATURES:
        features = list(FEATURE_COLUMNS) + [c for c in ORDERFLOW_FEATURE_COLUMNS if c in df_zone.columns]
    else:
        features = list(FEATURE_COLUMNS)

    datetimes = df_zone["datetime"].to_numpy()
    X = df_zone.select(features).to_numpy()
    y = df_zone["target"].to_numpy()
    return X, y, features, datetimes


# ----------------------------------------------------------------------------
# STEP 2 — VIF, computed PER FOLD (train slice only), not on the pooled
# dataset. Two reasons: (a) matches what the model actually saw at fit time
# (no leakage from test-period collinearity structure), and (b) the spread
# of VIF across folds is itself diagnostic -- if VIF for a feature swings
# wildly fold to fold, its collinearity structure is regime-dependent, which
# is directly relevant to RQ1.
# ----------------------------------------------------------------------------
def compute_vif(X_sub: np.ndarray) -> np.ndarray:
    """Manual VIF via auxiliary R^2 (equivalent to statsmodels'
    variance_inflation_factor but avoids re-fitting statsmodels OLS objects
    len(features) times per fold -- this is the closed-form: VIF_j = 1/(1-R_j^2)
    where R_j^2 comes from regressing column j on all other columns."""
    n, k = X_sub.shape
    Xz = (X_sub - X_sub.mean(axis=0)) / (X_sub.std(axis=0) + 1e-12)
    vifs = np.full(k, np.nan)
    for j in range(k):
        y_j = Xz[:, j]
        others = np.delete(Xz, j, axis=1)
        others_ = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(others_, y_j, rcond=None)
        y_hat = others_ @ coef
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / max(1e-6, (1.0 - r2))
    return vifs


def vif_report(X: np.ndarray, features: list[str], candidates: list[str], folds) -> pl.DataFrame:
    idx = [features.index(f) for f in candidates]
    per_fold = []
    for train_idx, _, _ in folds:
        if len(train_idx) < 5 * len(candidates):
            continue
        X_sub = X[train_idx][:, idx]
        if not np.all(np.isfinite(X_sub)):
            X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
        per_fold.append(compute_vif(X_sub))
    if not per_fold:
        return pl.DataFrame({"Feature": candidates, "VIF_mean": [float("nan")] * len(candidates),
                              "VIF_std": [float("nan")] * len(candidates)})
    arr = np.array(per_fold)  # (n_folds, n_candidates)
    return pl.DataFrame({
        "Feature": candidates,
        "VIF_mean": np.nanmean(arr, axis=0),
        "VIF_std": np.nanstd(arr, axis=0),
        "VIF_folds_used": [int(np.sum(np.isfinite(arr[:, j]))) for j in range(arr.shape[1])],
    }).sort("VIF_mean", descending=True)


def cluster_report(X: np.ndarray, features: list[str], candidates: list[str]) -> tuple[np.ndarray, list[int]]:
    """Hierarchical clustering on |correlation| distance -- VIF tells you a
    feature IS collinear, this tells you WITH WHOM."""
    idx = [features.index(f) for f in candidates]
    X_sub = X[:, idx]
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(X_sub, rowvar=False)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry (float noise)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=0.5, criterion="distance")  # merge features with |corr| > 0.5
    return corr, list(cluster_ids)


# ----------------------------------------------------------------------------
# STEP 3 — QR / Gram-Schmidt residual-alpha test.
# Order: HEAD features first (kept as-is -- they are the established
# primary signal), then TAIL features are orthogonalized against the head
# block via QR. We then compare each tail feature's |Spearman rho| with the
# target BEFORE vs AFTER orthogonalization:
#   - rho barely moves     -> the tail feature carries information the head
#                              block does NOT already have (real, if small,
#                              residual alpha -- supports RQ3's "does raw
#                              orderflow retain residual content" framing).
#   - rho collapses to ~0  -> the tail feature's small PFI/SHAP was
#                              borrowed/shared variance from the head
#                              features, not unique content. It is a
#                              redundant duplicate, not weak-but-real alpha.
# ----------------------------------------------------------------------------
def orthogonalize_tail_against_head(X: np.ndarray, y: np.ndarray, features: list[str],
                                     head: list[str], tail: list[str]) -> pl.DataFrame:
    head = [f for f in head if f not in tail]  # a feature can't be both; head wins the slot
    if not head or not tail:
        return pl.DataFrame({"Feature": [], "raw_abs_rho": [], "residual_abs_rho": [], "retained_pct": []})

    head_idx = [features.index(f) for f in head]
    tail_idx = [features.index(f) for f in tail]

    X_head = np.nan_to_num(X[:, head_idx], nan=0.0, posinf=0.0, neginf=0.0)
    X_tail = np.nan_to_num(X[:, tail_idx], nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize (Gram-Schmidt/QR on raw-scale mixed-unit columns is
    # meaningless -- the projection would be dominated by whichever feature
    # happens to have the largest numeric range).
    X_head_z = (X_head - X_head.mean(axis=0)) / (X_head.std(axis=0) + 1e-12)
    X_tail_z = (X_tail - X_tail.mean(axis=0)) / (X_tail.std(axis=0) + 1e-12)

    # QR of the head block gives an orthonormal basis Q spanning the same
    # column space as X_head; projecting tail columns onto Q and subtracting
    # is numerically-stable Gram-Schmidt (avoids the classical algorithm's
    # well-known loss-of-orthogonality on correlated columns).
    Q, _ = np.linalg.qr(X_head_z)
    tail_residual = X_tail_z - Q @ (Q.T @ X_tail_z)

    y_bin = (y == np.max(y)).astype(float)  # WIN vs not-WIN, matches target semantics elsewhere
    rows = []
    for j, feat in enumerate(tail):
        raw_rho, _ = st.spearmanr(X_tail_z[:, j], y_bin)
        res_rho, _ = st.spearmanr(tail_residual[:, j], y_bin)
        raw_abs = abs(raw_rho) if np.isfinite(raw_rho) else 0.0
        res_abs = abs(res_rho) if np.isfinite(res_rho) else 0.0
        retained_pct = (res_abs / raw_abs * 100.0) if raw_abs > 1e-9 else float("nan")
        rows.append({"Feature": feat, "raw_abs_rho": raw_abs, "residual_abs_rho": res_abs,
                      "retained_pct": retained_pct})
    return pl.DataFrame(rows).sort("retained_pct")


# ----------------------------------------------------------------------------
# STEP 4 — Leave-one-out ablation, retrained (not re-scored) per fold, on
# the SAME purged split scheme as ml_state_optimized.py. This is the one
# test in this file that is model-agnostic to linear-vs-nonlinear
# redundancy, because it asks the actual estimator to relearn without the
# feature rather than inferring redundancy from a linear proxy.
# ----------------------------------------------------------------------------
def run_ablation(X: np.ndarray, y: np.ndarray, features: list[str], candidates: list[str],
                  folds, n_folds_cap: int = 40) -> pl.DataFrame:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    orderflow_idx = [i for i, f in enumerate(features) if f in ORDERFLOW_FEATURE_COLUMNS]
    baseline_idx = [i for i in range(len(features)) if i not in orderflow_idx]

    variants = {"FULL": list(range(len(features))), "BASELINE_no_orderflow": baseline_idx}
    for f in candidates:
        drop_idx = features.index(f)
        variants[f"minus_{f}"] = [i for i in range(len(features)) if i != drop_idx]

    fold_list = [fld for fld in folds if len(fld[0]) > 0 and len(np.unique(y[fld[0]])) > 1][:n_folds_cap]

    scores = {name: [] for name in variants}
    for train_idx, test_idx, _ in fold_list:
        if len(np.unique(y[test_idx])) < 2:
            continue
        for name, cols in variants.items():
            clf = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=42, n_jobs=1, verbose=-1,
            )
            clf.fit(X[train_idx][:, cols], y[train_idx])
            proba = clf.predict_proba(X[test_idx][:, cols])
            target_pos = list(clf.classes_).index(max(clf.classes_))
            try:
                auc = roc_auc_score(y[test_idx], proba[:, target_pos])
            except ValueError:
                continue
            scores[name].append(auc)

    full = np.array(scores["FULL"])
    base = np.array(scores["BASELINE_no_orderflow"])
    rows = []
    for name, arr in scores.items():
        arr = np.array(arr)
        if len(arr) == 0:
            continue
        n = min(len(arr), len(full))
        delta_full = arr[:n] - full[:n]
        delta_base = arr[:n] - base[:n] if len(base) >= n else np.full(n, np.nan)
        try:
            _, p_full = st.wilcoxon(delta_full) if np.any(delta_full != 0) else (np.nan, 1.0)
        except ValueError:
            p_full = float("nan")
        rows.append({
            "Variant": name, "n_folds": len(arr),
            "AUC_mean": float(arr.mean()), "AUC_std": float(arr.std()),
            "delta_vs_FULL": float(delta_full.mean()) if name != "FULL" else 0.0,
            "delta_vs_BASELINE": float(delta_base.mean()) if name != "BASELINE_no_orderflow" else 0.0,
            "p_vs_FULL_wilcoxon": float(p_full) if name != "FULL" else float("nan"),
        })
    return pl.DataFrame(rows).sort("delta_vs_FULL")


# ----------------------------------------------------------------------------
# Report writer (duplicated intentionally -- matches this codebase's own
# convention of small, self-contained report helpers per script rather than
# a shared import, e.g. _pwe/_sortino already duplicated between
# ml_state_optimized.py and hypothesis_session_testing.py).
# ----------------------------------------------------------------------------
class _Report:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")

    def add(self, *entries):
        text = " ".join(str(e) for e in entries)
        self._fh.write(text + "\n")
        print(text)

    def table(self, df: pl.DataFrame):
        text = str(df)
        self._fh.write(text + "\n\n")
        print(text)

    def close(self):
        self._fh.close()
        print(f"\n[report] written: {self.path}")


def main():
    report_path = os.path.join(BUILD_DIR, "feature_redundancy_diagnostics_report.md")
    rep = _Report(report_path)
    rep.add("# Feature Redundancy Diagnostics — VIF / Orthogonalization / Ablation")
    rep.add("")
    rep.add("_Read-only diagnostic. Mirrors ml_state_optimized.py STEP 1-6 for X/y/features; "
            "does not modify the training pipeline._")
    rep.add("")

    candidates, membership = load_candidate_features()
    rep.add(f"## Candidate features (top-{TOP_K} head+tail, PFI and SHAP, deduped): {len(candidates)}")
    rep.add("")
    for f in candidates:
        rep.add(f"  - {f}: {sorted(membership[f])}")
    disagreements = [f for f, tags in membership.items()
                      if any("head" in t for t in tags) and any("tail" in t for t in tags)]
    if disagreements:
        rep.add("")
        rep.add(f"**Ranking-instability flag**: {disagreements} rank as HEAD under one importance "
                f"measure and TAIL under the other. That disagreement is itself evidence the two "
                f"measures are splitting credit differently across correlated features -- treat "
                f"the raw head/tail label for these as unreliable until the ablation numbers below "
                f"confirm a direction.")
    rep.add("")

    X, y, features, datetimes = prepare_training_matrix()
    from research.ml_state_optimized import purged_chronological_splits  # noqa: E402
    folds = list(purged_chronological_splits(
        datetimes, n_samples=N_RANDOM_SAMPLES, block_size=TEST_BLOCK_SIZE,
        horizon_seconds=EMBARGO_SECONDS, embargo_seconds=EMBARGO_SECONDS,
    ))

    rep.add("## 1. VIF (per-fold train slice, mean ± std across folds)")
    rep.add("")
    rep.add("Rule of thumb used below: VIF > 10 = severe, 5-10 = moderate, < 5 = low. "
            "This is a LINEAR diagnostic only -- see module docstring.")
    rep.add("")
    vdf = vif_report(X, features, candidates, folds)
    rep.table(vdf)

    rep.add("## 2. Correlation clustering (|corr| distance, merge threshold 0.5)")
    rep.add("")
    corr, cluster_ids = cluster_report(X, features, candidates)
    clus_df = pl.DataFrame({"Feature": candidates, "Cluster": cluster_ids}).sort("Cluster")
    rep.table(clus_df)
    n_singletons = sum(1 for c in set(cluster_ids) if cluster_ids.count(c) == 1)
    rep.add(f"_{len(set(cluster_ids))} clusters from {len(candidates)} candidates "
            f"({n_singletons} singleton / not-collinear-with-anything-here)._")
    rep.add("")

    rep.add("## 3. Gram-Schmidt (QR) residual-alpha test: tail orthogonalized against head")
    rep.add("")
    head_all = [f for f, tags in membership.items() if any("head" in t for t in tags)]
    tail_all = [f for f, tags in membership.items() if any("tail" in t for t in tags) and f not in head_all]
    orth_df = orthogonalize_tail_against_head(X, y, features, head_all, tail_all)
    rep.table(orth_df)
    rep.add("Interpretation: `retained_pct` near 100 -> tail feature's association with the target "
            "survives removing everything explainable by the head block (real, if weak, residual "
            "alpha). Near 0 -> the tail feature's apparent (small) importance was head-block "
            "variance the model happened to route through it; it is redundant, not incremental.")
    rep.add("")

    rep.add("## 4. Leave-one-out ablation (retrained per fold; capped folds for runtime)")
    rep.add("")
    rep.add("`delta_vs_FULL` = mean(AUC_variant - AUC_full) across paired folds; negative = "
            "removing the feature hurt. `delta_vs_BASELINE` compares to the no-orderflow model.")
    rep.add("")
    abl_df = run_ablation(X, y, features, candidates, folds)
    rep.table(abl_df)
    rep.add("")

    # ------------------------------------------------------------------
    # Objective verdict — data-driven, not just asserted.
    # ------------------------------------------------------------------
    rep.add("## 5. VERDICT — is VIF/Gram-Schmidt the right method here?")
    rep.add("")
    severe = vdf.filter(pl.col("VIF_mean") > 10)["Feature"].to_list()
    moderate = vdf.filter((pl.col("VIF_mean") <= 10) & (pl.col("VIF_mean") > 5))["Feature"].to_list()
    rep.add(f"- VIF flags **{len(severe)} severe** (>10) and **{len(moderate)} moderate** (5-10) "
            f"features out of {len(candidates)} candidates.")
    if severe:
        rep.add(f"  -> Severe: {severe}. Orthogonalization/dropping is warranted for these specifically, "
                f"not for the whole candidate set.")
    else:
        rep.add("  -> No severe multicollinearity detected among these candidates. Running VIF was "
                "still correct due diligence (it's what rules this OUT), but it means "
                "orthogonalization would not have been buying you much here -- don't apply it "
                "wholesale just because it was on the checklist.")
    rep.add("")
    ablation_disagrees = abl_df.filter(
        (pl.col("Variant").str.starts_with("minus_")) & (pl.col("delta_vs_FULL") > 0)
    )["Variant"].to_list()
    if ablation_disagrees:
        rep.add(f"- **Ablation contradicts PFI/SHAP for**: {ablation_disagrees} -- removing these "
                f"features did not hurt (or helped) retrained OOS AUC, despite nonzero PFI/SHAP. "
                f"This is the collinearity-splitting failure mode described in the module docstring "
                f"made concrete: trust the ablation number over the raw importance rank for these.")
    else:
        rep.add("- Ablation direction is consistent with the PFI/SHAP ranking for all candidates "
                "checked -- no evidence the head/tail split is a pure collinearity artifact.")
    rep.add("")
    rep.add("**Overall**: VIF + clustering is appropriate, cheap, standard diagnostic hygiene and "
            "should stay in the pipeline as a gate before trusting any single-feature importance "
            "claim. Gram-Schmidt/QR orthogonalization is appropriate ONLY as the narrow residual-"
            "alpha test above (RQ3), not as a wholesale feature-set transformation -- it is linear, "
            "order-dependent, and destroys interpretability if applied broadly. Neither tool is, by "
            "itself, evidence of longitudinal decay (RQ2) or session dependency (RQ1); those need "
            "the retrained ablation deltas above and the longitudinal/session tests in "
            "`hypothesis_session_testing.py` / `longitudinal_decay_testing.py`.")
    rep.add("")
    rep.add("---")
    rep.add("_Run: read-only diagnostics; training pipeline untouched._")
    rep.close()


if __name__ == "__main__":
    main()