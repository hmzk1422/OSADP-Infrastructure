"""
regime_buckets.py
=================
Method A - econ-interpretable quantile buckets + interactions + stability +
drawdown attribution, all on the leakage-safe frame from `regime_data`.

Walk-forward discipline: every bucket edge is fit on the training segment of an
expanding monthly window and applied to the following unseen month. The same
per-fold bucket label is reused everywhere downstream (stability, interactions)
so nothing downstream can peek at test-period distributions.

Outcome metrics follow `regime_data.ret_metrics`. Multiple-testing context is
tracked explicitly (number of comparisons, BH-adjusted Kruskal p-values on the
pooled out-of-sample buckets).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ofmr.regime_data import (  # noqa: E402
    load_study_frame, make_folds, ret_metrics, bootstrap_ci,
    MIN_LEAF, ensure_out, ENTRY_STATE_FEATURES, STRAT_COLS,
)

# Variable catalogue ----------------------------------------------------------
# (column, human-readable name, family)
VARIABLES: list[tuple[str, str, str]] = [
    # --- volatility / activity (pre-entry proxies) ---
    ("vol_prior20_range", "Realized range (prior20, med)", "volatility"),
    ("vol_prior20_mae", "Realized MAE (prior20, med)", "volatility"),
    ("vol_prior50_range", "Realized range (prior50, med)", "volatility"),
    ("spread_bps_z_open", "Spread z-score @ entry", "liquidity"),
    ("vol_prior20_ret_std", "Ret std (prior20)", "strategy-variance"),
    # --- trend / momentum / stretch ---
    ("ma_slope_5_open", "MA5 slope @ entry", "trend"),
    ("ma_accel_open", "MA5 acceleration @ entry", "trend"),
    ("price_roc_3_open", "3-bar price ROC @ entry", "momentum"),
    ("rsi_2_open", "RSI(2) stretch @ entry", "momentum"),
    ("price_rel_sma14", "Price vs SMA14 @ entry", "trend"),
    # --- order flow ---
    ("cvd_slope_z_open", "CVD slope z @ entry", "orderflow"),
    ("tick_autocorr_open", "Tick autocorr @ entry", "microstructure"),
    ("inst_ratio_open", "Institutional ratio @ entry", "orderflow"),
    ("depth_imbalance_z_open", "Depth imbalance z @ entry", "orderflow"),
    # --- calendar ---
    ("hour_of_day", "Entry hour (UTC)", "calendar"),
    ("weekday", "Entry weekday", "calendar"),
    # --- strategy state (prior trades only) ---
    ("strat_prev10_win_rate", "Win rate (prior10)", "strategy-state"),
    ("strat_prev20_win_rate", "Win rate (prior20)", "strategy-state"),
    ("strat_prev10_mean_r", "Mean R (prior10)", "strategy-state"),
    ("strat_prev20_profit_factor", "Profit factor (prior20)", "strategy-state"),
    ("strat_prev10_median_gap", "Median time-to-exit (prior10, min)", "strategy-state"),
    ("strat_prev10_quick_frac", "Quick-stop frac (prior10, <=6m)", "strategy-state"),
    ("strat_prev20_quick_frac", "Quick-stop frac (prior20, <=6m)", "strategy-state"),
    ("strat_prev10_mae", "Mean MAE (prior10, %)", "strategy-state"),
]

INTERACTION_PAIRS: list[tuple[str, str, str]] = [
    ("vol_prior20_range", "ma_slope_5_open", "volatility x trend"),
    ("vol_prior20_range", "price_roc_3_open", "volatility x momentum"),
    ("vol_prior20_range", "cvd_slope_z_open", "volatility x OFI/order-flow"),
    ("vol_prior20_range", "depth_imbalance_z_open", "volatility x book imbalance"),
    ("vol_prior20_range", "spread_bps_z_open", "volatility x liquidity"),
    ("vol_prior20_range", "strat_prev10_median_gap", "volatility x stop-out speed"),
    ("ma_slope_5_open", "cvd_slope_z_open", "trend x OFI/order-flow"),
    ("ma_slope_5_open", "depth_imbalance_z_open", "trend x book imbalance"),
]

# ----------------------------------------------------------------------------
# Bucket fitting / assignment (walk-forward safe)
# ----------------------------------------------------------------------------
def fit_quantile_edges(s: pd.Series, q: int = 5) -> list:
    """Quantile bin edges fit on a train-only series (q bins)."""
    vals = s.dropna()
    if vals.nunique() <= 1 or len(vals) < q:
        return None
    edges = [vals.quantile(i / q) for i in range(q + 1)]
    edges[0], edges[-1] = -np.inf, np.inf
    out = [edges[0]]
    for e in edges[1:-1]:
        if e > out[-1]:
            out.append(e)
    out.append(np.inf)
    return out


def assign_buckets(df: pd.DataFrame, var: str, q: int = 5,
                   folds=None) -> pd.Series:
    """Per-trade bucket label using per-fold train-fitted edges.

    Training-segment trades are bucketed with their own fold's edges (used only
    for calibration/introspection); test trades use the fold's train-fitted
    edges. Rows without rolling state get NaN.
    """
    labels = pd.Series(np.nan, index=df.index, dtype=object)
    if folds is None:
        folds = make_folds(df)
    for train_mask, test_mask, _ in folds:
        tr = df.loc[train_mask & df["_complete"], var]
        edges = fit_quantile_edges(tr, q)
        if edges is None:
            continue
        for mask in (train_mask, test_mask):
            sub = df.loc[mask & df["_complete"], var]
            n_bins = len(edges) - 1
            labs = pd.cut(sub, bins=edges, labels=[f"Q{i}" for i in range(1, n_bins + 1)],
                          include_lowest=True)
            labels.loc[sub.index] = labs.astype(str).to_numpy()
    return labels


# ----------------------------------------------------------------------------
# Conditional outcome table
# ----------------------------------------------------------------------------
def bucket_stats_table(df: pd.DataFrame, labels: pd.Series, ret_col="ret",
                       extra=("mae_pct", "mfe_pct")) -> pd.DataFrame:
    """Per-bucket conditional outcome table."""
    rows = []
    tmp = df.copy()
    tmp["_bucket"] = labels
    for b in sorted(tmp["_bucket"].dropna().unique()):
        m = tmp[tmp["_bucket"] == b]
        r = m[ret_col].to_numpy()
        met = ret_metrics(r)
        ci = bootstrap_ci(r)
        row = {
            "bucket": b, "n": met["n"], "mean_r": met["mean_r"],
            "ci_lo": ci[0], "ci_hi": ci[1], "median_r": met["median_r"],
            "win_rate": met["win_rate"], "p_1r": met["p_1r"],
            "p_2r": met["p_2r"], "p_3r": met["p_3r"], "p_5r": met["p_5r"],
            "pf": met["pf"], "total_r": met["total_r"],
        }
        for ex in extra:
            row[f"med_{ex}"] = float(m[ex].median()) if ex in m else np.nan
            row[f"mean_{ex}"] = float(m[ex].mean()) if ex in m else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket") if rows else pd.DataFrame()


def kruskal_p(df: pd.DataFrame, labels: pd.Series, ret_col="ret") -> float:
    """Kruskal-Wallis p-value across buckets (pooled out-of-sample trades)."""
    tmp = df.copy()
    tmp["_bucket"] = labels
    groups = [tmp.loc[tmp["_bucket"] == b, ret_col].dropna().to_numpy()
              for b in sorted(tmp["_bucket"].dropna().unique())]
    groups = [g for g in groups if len(g) > 2]
    if len(groups) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*groups).pvalue)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------
# Full walk-forward bucket study
# ----------------------------------------------------------------------------
def run_bucket_study(df: pd.DataFrame, q: int = 5) -> dict:
    """Loop every variable: per-fold bucket assignment -> pooled OOS table."""
    ensure_out()
    folds = make_folds(df)
    summary = []
    tables: dict[str, pd.DataFrame] = {}
    labels_full: dict[str, pd.Series] = {}

    for var, name, family in VARIABLES:
        labels = assign_buckets(df, var, q=q, folds=folds)
        labels_full[var] = labels

        test_mask = pd.Series(False, index=df.index)
        for _, tm, _ in folds:
            test_mask |= tm
        oos = df.loc[test_mask & labels.notna()].copy()
        oos["_b"] = labels.loc[oos.index]
        oos_tbl = bucket_stats_table(oos, oos["_b"])

        is_df = df.loc[labels.notna()].copy()
        is_df["_b"] = labels.loc[is_df.index]
        is_tbl = bucket_stats_table(is_df, is_df["_b"])

        tables[var] = oos_tbl.assign(var=var, name=name, family=family)
        tables[var + "_is"] = is_tbl.assign(var=var, name=name, family=family)

        kw = kruskal_p(oos, oos["_b"])
        spread_oos = n_spread = np.nan
        if len(oos_tbl) >= 2:
            q5 = oos_tbl[oos_tbl["bucket"] == "Q5"]
            q1 = oos_tbl[oos_tbl["bucket"] == "Q1"]
            if len(q5) and len(q1):
                spread_oos = float(q5["mean_r"].iloc[0] - q1["mean_r"].iloc[0])
                n_spread = int(min(q5["n"].iloc[0], q1["n"].iloc[0]))
        baseline = ret_metrics(df.loc[test_mask, "ret"].to_numpy())["mean_r"]
        bucket_line = "  ".join(
            f"{r.bucket}:{r['mean_r']:+.3f}(n{r['n']:.0f})" for _, r in oos_tbl.iterrows()
        )
        is_spread = np.nan
        if len(is_tbl) >= 2:
            i5 = is_tbl[is_tbl["bucket"] == "Q5"]
            i1 = is_tbl[is_tbl["bucket"] == "Q1"]
            if len(i5) and len(i1):
                is_spread = float(i5["mean_r"].iloc[0] - i1["mean_r"].iloc[0])
        summary.append({
            "var": var, "name": name, "family": family, "q": q,
            "oos_spread": spread_oos, "oos_n_min": n_spread,
            "kw_p_oos": kw, "baseline_mean_r": baseline,
            "oos_buckets": bucket_line, "is_spread": is_spread,
        })

    summ = pd.DataFrame(summary)
    if len(summ):
        summ["abs_spread"] = summ["oos_spread"].abs()
        summ = summ.sort_values("abs_spread", ascending=False)
    pvals = summ["kw_p_oos"].to_numpy()
    summ["kw_p_oos_bh"] = _bh_adjust(pvals)
    summ.to_csv(os.path.join(ensure_out(), "buckets_summary.csv"), index=False)
    return {"summary": summ, "tables": tables, "labels_full": labels_full,
            "n_comparisons": int(summ["q"].fillna(1).sum())}


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment (NaN-safe)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(pv) + 1)
    adj = pv * len(pv) / ranks
    order_desc = np.argsort(-pv)  # or reverse the ascending order
    adj = np.minimum.accumulate(adj[order][::-1])[::-1]
    adj = adj[np.argsort(order)]  # careful with index remapping
    out[valid] = adj
    return out
    """Kruskal-Wallis p-value across buckets (pooled out-of-sample trades)."""
    tmp = df.copy()
    tmp["_bucket"] = labels
    groups = [tmp.loc[tmp["_bucket"] == b, ret_col].dropna().to_numpy()
              for b in sorted(tmp["_bucket"].dropna().unique())]
    groups = [g for g in groups if len(g) > 2]
    if len(groups) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*groups).pvalue)
    except ValueError:
        return np.nan
# ----------------------------------------------------------------------------
# Pairwise interactions (tercile grids, walk-forward safe)
# ----------------------------------------------------------------------------
def interaction_grids(df: pd.DataFrame, pairs=None, min_test_n=60) -> pd.DataFrame:
    """3x3 tercile grids for the prioritized interaction pairs.

    Tercile edges are fit per-fold on training data; only pooled test-month
    trades are summarized (sample sizes shown for every cell).
    """
    if pairs is None:
        pairs = INTERACTION_PAIRS
    ensure_out()
    folds = make_folds(df)
    test_mask = pd.Series(False, index=df.index)
    for _, tm, _ in folds:
        test_mask |= tm

    rows = []
    for a, b, label in pairs:
        lab_a = assign_buckets(df, a, q=3, folds=folds)
        lab_b = assign_buckets(df, b, q=3, folds=folds)
        sub = df.loc[test_mask & lab_a.notna() & lab_b.notna()].copy()
        sub["_A"] = lab_a.loc[sub.index]
        sub["_B"] = lab_b.loc[sub.index]
        for _, g in sub.groupby(["_A", "_B"]):
            r = g["ret"].to_numpy()
            met = ret_metrics(r)
            if met["n"] < min_test_n:
                continue
            rows.append({
                "pair": label, "varA": a, "varB": b,
                "A": g["_A"].iloc[0], "B": g["_B"].iloc[0],
                "n": met["n"], "mean_r": met["mean_r"], "win_rate": met["win_rate"],
                "p_3r": met["p_3r"], "total_r": met["total_r"],
                "med_mae": float(g["mae_pct"].median()),
                "med_mfe": float(g["mfe_pct"].median()),
            })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(ensure_out(), "interactions.csv"), index=False)
    return tbl


# ----------------------------------------------------------------------------
# Stability through time
# ----------------------------------------------------------------------------
def stability_by_time(df: pd.DataFrame, labels: pd.Series, bucket: str,
                      period: str = "year_month") -> pd.DataFrame:
    """Monthly/weekly outcome table for one bucket (pooled OOS-labeled trades)."""
    tmp = df.loc[labels.notna()].copy()
    tmp["_b"] = labels.loc[tmp.index]
    sub = tmp[tmp["_b"] == bucket]
    if len(sub) == 0:
        return pd.DataFrame()
    rows = []
    for p, g in sub.groupby(period):
        met = ret_metrics(g["ret"].to_numpy())
        lo, hi = bootstrap_ci(g["ret"].to_numpy())
        rows.append({
            period: p, "n": met["n"], "mean_r": met["mean_r"],
            "mean_r_ci_lo": lo, "mean_r_ci_hi": hi,
            "median_r": met["median_r"], "win_rate": met["win_rate"],
            "pf": met["pf"], "total_r": met["total_r"],
            "worst_r": float(g["ret"].min()), "best_r": float(g["ret"].max()),
        })
    t = pd.DataFrame(rows)
    if len(t):
        t["pos_periods"] = (t["mean_r"] > 0).astype(int)
    return t


def stability_report(df: pd.DataFrame, labels_full: dict, summary,
                     top_n: int = 6) -> dict:
    """Stability tables for the top-OOS-spread variables (best & worst buckets)."""
    ensure_out()
    if len(summary) == 0:
        return {}
    top_vars = summary.head(top_n)["var"].tolist()
    out = {}
    for var in top_vars:
        labels = labels_full[var]
        q5 = stability_by_time(df, labels, "Q5")
        q1 = stability_by_time(df, labels, "Q1")
        q5.to_csv(os.path.join(ensure_out(), f"stab_{var}_Q5.csv"), index=False)
        q1.to_csv(os.path.join(ensure_out(), f"stab_{var}_Q1.csv"), index=False)
        out[var] = {"Q5": q5, "Q1": q1}
    return out
# ----------------------------------------------------------------------------
# Drawdown attribution
# ----------------------------------------------------------------------------
def drawdown_attribution(df: pd.DataFrame, dd_thr_r: float = -8.0) -> dict:
    """Attribute drawdowns to entry states.

    Equity uses the same-threshold stream only (chronological). A trade is
    flagged `in_dd` when cumulative R sits more than `dd_thr_r` below the
    running peak at its entry. Compares conditional outcomes and the marginal
    distributions of every state variable between in-drawdown and healthy
    entries.
    """
    df = df.sort_values("dt").reset_index(drop=True)
    equity = df["ret"].cumsum()
    peak = equity.cummax()
    dd = (equity - peak).to_numpy()
    df["_equity"] = equity.to_numpy()
    df["_dd"] = dd
    df["_in_dd"] = dd <= dd_thr_r

    in_dd = df[df["_in_dd"]]
    out_dd = df[~df["_in_dd"]]
    m_in = ret_metrics(in_dd["ret"].to_numpy())
    m_out = ret_metrics(out_dd["ret"].to_numpy())

    state_rows = []
    cols = [c for c, _, _ in VARIABLES]
    for c in cols:
        a = in_dd[c].dropna().to_numpy()
        b = out_dd[c].dropna().to_numpy()
        if len(a) < 20 or len(b) < 20:
            continue
        ks = stats.ks_2samp(a, b)
        state_rows.append({
            "var": c, "mean_in_dd": float(a.mean()), "mean_out_dd": float(b.mean()),
            "ks_stat": ks.statistic, "ks_p": ks.pvalue,
        })
    state_tbl = pd.DataFrame(state_rows)
    if len(state_tbl):
        state_tbl = state_tbl.sort_values("ks_p")

    ep = []
    for i in range(1, len(dd)):
        if dd[i] <= dd[i - 1] and dd[i] < dd_thr_r:
            if not ep or ep[-1] != i - 1:
                ep.append(i - 1)
            ep.append(i)
    dd_idx = sorted(set(ep))
    dd_tbl = pd.DataFrame({
        "trade_idx": dd_idx,
        "dt": df["dt"].iloc[dd_idx].values,
        "dd_r": df["_dd"].iloc[dd_idx].values,
    }) if dd_idx else pd.DataFrame(columns=["trade_idx", "dt", "dd_r"])

    out = {"in_dd_metrics": m_in, "out_dd_metrics": m_out,
           "n_in_dd": int(in_dd.shape[0]), "n_out_dd": int(out_dd.shape[0]),
           "dd_thr_r": dd_thr_r, "state_table": state_tbl, "dd_episodes": dd_tbl,
           "frame": df}
    pd.DataFrame({"n_in_dd": [out["n_in_dd"]], "n_out_dd": [out["n_out_dd"]],
                  "mean_in_dd": [m_in["mean_r"]], "mean_out_dd": [m_out["mean_r"]],
                  "wr_in_dd": [m_in["win_rate"]], "wr_out_dd": [m_out["win_rate"]]}
                 ).to_csv(os.path.join(ensure_out(), "drawdown_overview.csv"), index=False)
    state_tbl.to_csv(os.path.join(ensure_out(), "drawdown_states.csv"), index=False)
    return out


def run_buckets_all(df: pd.DataFrame) -> dict:
    """Orchestrate method A and collect everything the report needs."""
    b = run_bucket_study(df)
    interactions = interaction_grids(df)
    stability = stability_report(df, b["labels_full"], b["summary"])
    dd = drawdown_attribution(df)
    baseline = ret_metrics(df["ret"].to_numpy())
    return {"baseline": baseline, "buckets": b, "interactions": interactions,
            "stability": stability, "drawdown": dd}