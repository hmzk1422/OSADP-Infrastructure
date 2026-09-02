"""
regime_hmm.py
=============
Method C - Hidden Markov Model experiment for persistent latent market states.

Inputs    : a compact, deduplicated, economically meaningful set of entry-time
            market variables (vol proxy, spread, trend, momentum, RSI stretch,
            CVD slope, depth imbalance). Strictly pre-entry only.
Fitting   : per-fold expanding-window: parameters fit on the training segment,
            then applied to the following unseen month (observations z-scored
            with training statistics only).
Inference : BOTH estimates are reported, because they answer different
            questions:
              1. viterbi (hindsight / full-sequence, `decode`) - can look
                 good but is NOT available in real time.
              2. forward-filtered (`predict_proba`) - state probabilities given
                 only observations up to the current trade. This is the only
                 deployable estimate.
Question  : does the probability of being in a particular latent state predict
            subsequent trade expectancy out of sample?
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats as sps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ofmr.regime_data import (  # noqa: E402
    load_study_frame, make_folds, ret_metrics, bootstrap_ci, fmt_metrics,
    ensure_out,
)

# Compact, interpretable, deduplicated market-state observation set.
HMM_OBS = [
    "vol_prior20_range", "spread_bps_z_open", "ma_slope_5_open",
    "price_roc_3_open", "rsi_2_open", "cvd_slope_z_open",
    "depth_imbalance_z_open",
]

N_STATES_RANGE = (2, 3, 4, 5)

OBS_LABELS = {
    "vol_prior20_range": "vol(range20)",
    "spread_bps_z_open": "spread",
    "ma_slope_5_open": "trend(MA5)",
    "price_roc_3_open": "mom(roc3)",
    "rsi_2_open": "rsi2",
    "cvd_slope_z_open": "cvd_slope",
    "depth_imbalance_z_open": "depth_imb",
}


def zscore_fit(series: pd.Series):
    """Training-only z-score parameters."""
    s = series.dropna()
    return float(s.mean()), float(s.std())


def fit_hmm(Xtr: np.ndarray, n_states: int):
    """Small diagonal Gaussian HMM, fixed seed."""
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=200, tol=1e-3, random_state=42,
    )
    model.fit(Xtr)
    return model
def run_hmm_study(df: pd.DataFrame, n_states_range=N_STATES_RANGE) -> dict:
    """Walk-forward HMM for several state counts.

    Returns pooled-test metrics per (fold, K, state, inference method), plus
    transition matrices and standardized observation profiles.
    """
    ensure_out()
    folds = make_folds(df)
    rows, state_rows, trans_rows = [], [], []

    for K in n_states_range:
        for train_mask, test_mask, label in folds:
            use_train = train_mask & df["_complete"]
            use_test = test_mask & df["_complete"]
            Xtr_raw = df.loc[use_train, HMM_OBS]
            Xte_raw = df.loc[use_test, HMM_OBS]
            if len(Xtr_raw) < 200 or len(Xte_raw) < 100:
                continue

            mus, stds = {}, {}
            Xtr = np.empty((len(Xtr_raw), len(HMM_OBS)))
            Xte = np.empty((len(Xte_raw), len(HMM_OBS)))
            for j, col in enumerate(HMM_OBS):
                mu, sd = zscore_fit(Xtr_raw[col])
                mus[col], stds[col] = mu, sd
                Xtr[:, j] = (Xtr_raw[col].to_numpy() - mu) / max(sd, 1e-9)
                Xte[:, j] = (Xte_raw[col].to_numpy() - mu) / max(sd, 1e-9)

            try:
                model = fit_hmm(Xtr, K)
            except Exception as exc:  # noqa: BLE001 - hmmlearn convergence quirks
                print(f"  [hmm] fit failed K={K} fold={label}: {exc}")
                continue

            filtered = model.predict_proba(Xte)   # forward-filter (real-time)
            viterbi = model.predict(Xte)          # hindsight path
            test_ret = df.loc[use_test, "ret"].to_numpy()

            for i in range(K):
                for j in range(K):
                    trans_rows.append({"K": K, "fold": label, "from": i, "to": j,
                                       "prob": model.transmat_[i, j]})

            for s in range(K):
                hi_conf = filtered[:, s] >= 0.6
                m_hc = ret_metrics(test_ret[hi_conf])
                m_am = ret_metrics(test_ret[np.argmax(filtered, axis=1) == s])
                m_vb = ret_metrics(test_ret[viterbi == s])
                for method, m in (("filtered_ge0.6", m_hc),
                                  ("filtered_argmax", m_am), ("viterbi", m_vb)):
                    rows.append({
                        "K": K, "fold": label, "state": s, "method": method,
                        "n": m["n"], "mean_r": m["mean_r"],
                        "median_r": m["median_r"], "win_rate": m["win_rate"],
                        "p_3r": m["p_3r"], "pf": m["pf"], "total_r": m["total_r"],
                        "frac": float((viterbi == s).mean()) if method == "viterbi"
                                 else float((np.argmax(filtered, axis=1) == s).mean())
                                 if method == "filtered_argmax"
                                 else float(hi_conf.mean()),
                    })

                prof = pd.Series(model.means_[s], index=HMM_OBS)
                state_rows.append({
                    "K": K, "fold": label, "state": s,
                    **{OBS_LABELS[c]: round(float(prof[c]), 2) for c in HMM_OBS},
                })

    res = {"rows": pd.DataFrame(rows), "trans": pd.DataFrame(trans_rows),
           "states": pd.DataFrame(state_rows)}
    for k, v in res.items():
        if len(v):
            v.to_csv(os.path.join(ensure_out(), f"hmm_{k}.csv"), index=False)
    return res
def summarize_hmm(res: dict) -> pd.DataFrame:
    """Pooled OOS summary per (K, state) across folds (filtered argmax)."""
    r = res["rows"]
    if not len(r):
        return pd.DataFrame()
    f = r[r["method"] == "filtered_argmax"].copy()
    if not len(f):
        return pd.DataFrame()
    out = []
    for (K, s), sub in f.groupby(["K", "state"]):
        out.append({
            "K": K, "state": s, "n_folds": int(sub["fold"].nunique()),
            "n": int(sub["n"].sum()),
            "mean_r": float(sub["mean_r"].mean()),
            "win_rate": float(sub["win_rate"].mean()),
            "p_3r": float(sub["p_3r"].mean()),
            "pf": float(sub["pf"].mean()),
            "total_r": float(sub["total_r"].sum()),
            "frac": float(sub["frac"].mean()),
        })
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(ensure_out(), "hmm_summary.csv"), index=False)
def _align_map(res: dict, key_col: str = "trend(MA5)") -> dict:
    """Map each (K, fold, state) to an aligned rank by economic profile.

    HMM state numbers are arbitrary (permutation invariant); pooling across
    folds requires aligning by the fitted observation profile (train params).
    We sort states by `key_col` ascending so aligned-state 0 is always the
    deepest-dip / most-stretched regime for every fold.
    """
    st = res["states"]
    mp = {}
    for (K, fold), g in st.groupby(["K", "fold"], sort=False):
        g = g.sort_values(key_col)
        mp[(int(K), fold)] = {int(row.state): i for i, row in enumerate(g.itertuples())}
    return mp


def summarize_hmm_aligned(res: dict, key_col: str = "trend(MA5)",
                          method: str = "filtered_argmax") -> pd.DataFrame:
    """Pooled OOS summary per (K, aligned state) + average economic profile."""
    r = res["rows"]
    if not len(r):
        return pd.DataFrame()
    f = r[r["method"] == method].copy()
    if not len(f):
        return pd.DataFrame()
    mp = _align_map(res, key_col)
    f["astate"] = [mp.get((int(K), fold), {}).get(int(s), np.nan)
                   for K, fold, s in zip(f["K"], f["fold"], f["state"])]
    f = f.dropna(subset=["astate"])

    st = res["states"].copy()
    st["astate"] = [mp.get((int(K), fold), {}).get(int(s), np.nan)
                    for K, fold, s in zip(st["K"], st["fold"], st["state"])]
    st = st.dropna(subset=["astate"])

    out = []
    for (K, a), sub in f.groupby(["K", "astate"]):
        prof = st[(st["K"] == K) & (st["astate"] == a)][list(OBS_LABELS.values())].mean()
        out.append({
            "K": K, "aligned_state": int(a), "n": int(sub["n"].sum()),
            "n_folds": int(sub["fold"].nunique()),
            "mean_r": float(sub["mean_r"].mean()),
            "win_rate": float(sub["win_rate"].mean()),
            "p_3r": float(sub["p_3r"].mean()),
            "pf": float(sub["pf"].mean()),
            "total_r": float(sub["total_r"].sum()),
            "frac": float(sub["frac"].mean()),
            "mean_trend": float(prof["trend(MA5)"]),
            "mean_rsi2": float(prof["rsi2"]),
            "mean_vol": float(prof["vol(range20)"]),
            "mean_cvd": float(prof["cvd_slope"]),
        })
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(ensure_out(), "hmm_summary_aligned.csv"), index=False)
    return t


def hmm_state_prob_predicts_ret(df: pd.DataFrame,
                                n_states_range=N_STATES_RANGE) -> pd.DataFrame:
    """Does the filtered (real-time) state probability predict next ret?

    Refits each fold's HMM (fixed seed -> identical fits), then computes
    Spearman between each state's forward-filtered probability and the current
    trade's ret, plus a pooled OLS slope per state. Deterministic rerun.
    """
    ensure_out()
    folds = make_folds(df)
    rows = []
    for K in n_states_range:
        for train_mask, test_mask, label in folds:
            use_train = train_mask & df["_complete"]
            use_test = test_mask & df["_complete"]
            Xtr_raw = df.loc[use_train, HMM_OBS]
            Xte_raw = df.loc[use_test, HMM_OBS]
            if len(Xtr_raw) < 200 or len(Xte_raw) < 100:
                continue
            mus, stds = {}, {}
            Xtr = np.empty((len(Xtr_raw), len(HMM_OBS)))
            Xte = np.empty((len(Xte_raw), len(HMM_OBS)))
            for j, col in enumerate(HMM_OBS):
                mu, sd = zscore_fit(Xtr_raw[col])
                mus[col], stds[col] = mu, sd
                Xtr[:, j] = (Xtr_raw[col].to_numpy() - mu) / max(sd, 1e-9)
                Xte[:, j] = (Xte_raw[col].to_numpy() - mu) / max(sd, 1e-9)
            model = fit_hmm(Xtr, K)
            filt = model.predict_proba(Xte)
            rte = df.loc[use_test, "ret"].to_numpy()
            for s in range(K):
                rho = sps.spearmanr(filt[:, s], rte)
                # pooled OLS slope across folds (ret ~ prob), stored per fold
                denom = float(np.cov(filt[:, s], rte)[0, 1])
                var = float(np.var(filt[:, s]))
                slope = denom / var if var > 1e-12 else 0.0
                # high-confidence state performance
                hi = filt[:, s] >= 0.6
                m_hi = ret_metrics(rte[hi])
                rows.append({
                    "K": K, "fold": label, "state": s,
                    "spearman": rho.statistic if rho is not None else np.nan,
                    "spearman_p": rho.pvalue if rho is not None else np.nan,
                    "ols_slope": slope,
                    "n_hi": m_hi["n"], "hi_mean_r": m_hi["mean_r"],
                    "hi_win_rate": m_hi["win_rate"],
                })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(ensure_out(), "hmm_predictability.csv"), index=False)
    return t


def hmm_confidence_filter(df: pd.DataFrame, K: int = 2,
                          prob_thr: float = 0.6) -> dict:
    """OOS filter: trade only when the deepest-dip state's prob >= thr.

    The deep-dip state is identified per fold by the lowest fitted trend mean.
    """
    ensure_out()
    folds = make_folds(df)
    labels = pd.Series("unmodeled", index=df.index, dtype=object)
    for train_mask, test_mask, label in folds:
        use_train = train_mask & df["_complete"]
        use_test = test_mask & df["_complete"]
        Xtr_raw = df.loc[use_train, HMM_OBS]
        Xte_raw = df.loc[use_test, HMM_OBS]
        mus, stds = {}, {}
        Xtr = np.empty((len(Xtr_raw), len(HMM_OBS)))
        Xte = np.empty((len(Xte_raw), len(HMM_OBS)))
        for j, col in enumerate(HMM_OBS):
            mu, sd = zscore_fit(Xtr_raw[col])
            mus[col], stds[col] = mu, sd
            Xtr[:, j] = (Xtr_raw[col].to_numpy() - mu) / max(sd, 1e-9)
            Xte[:, j] = (Xte_raw[col].to_numpy() - mu) / max(sd, 1e-9)
        model = fit_hmm(Xtr, K)
        dip_state = int(np.argmin(model.means_[:, HMM_OBS.index("ma_slope_5_open")]))
        filt = model.predict_proba(Xte)
        idx = df.index[np.where(use_test.to_numpy())[0]]
        labels.loc[idx] = np.where(filt[:, dip_state] >= prob_thr,
                                   "deep_dip", "else").astype(object)
    pred = df.copy()
    pred["hmm_filter"] = labels
    out = {lab: ret_metrics(pred[pred["hmm_filter"] == lab]["ret"].to_numpy())
           for lab in ["deep_dip", "else"] if (pred["hmm_filter"] == lab).sum() > 0}
    out["baseline"] = ret_metrics(pred["ret"].to_numpy())
    out["labels"] = labels
    pred.to_csv(os.path.join(ensure_out(), "hmm_filter_predictions.csv"), index=False)
    return out


def transition_persistence(res: dict) -> pd.DataFrame:
    """Max self-transition probability per (K, fold) as a persistence measure."""
    t = res["trans"]
    if not len(t):
        return pd.DataFrame()
    self_t = t[t["from"] == t["to"]].groupby(["K", "fold"])["prob"].max().reset_index()
    self_t.to_csv(os.path.join(ensure_out(), "hmm_persistence.csv"), index=False)
    return self_t


if __name__ == "__main__":
    d = load_study_frame()
    res = run_hmm_study(d)
    print("hmm rows:", len(res["rows"]))
    summ = summarize_hmm(res)
    print(summ.to_string())
    print()
    print(transition_persistence(res).to_string())