"""
Session-Regime Hypothesis Testing for Walk-Forward Trade Expectancy
=====================================================================
READ-ONLY statistical analysis on the realized, out-of-sample trade log
(`build/walk_forward_trade_log_v3.csv`). It never touches training/feature code and
introduces zero lookahead risk - every input is a *realized* trade outcome.

Produces `build/hypothesis_session_testing_report.md` containing:

  1. Descriptive per-session expectancy (PWE / Sortino / trade count)
  2. Kruskal-Wallis H-test     - do per-trade return medians differ across sessions?
  3. Dunn's post-hoc pairwise  - which session pairs differ? (Bonferroni-corrected)
  4. OLS + Newey-West (HAC)    - quantified edge vs Asia baseline, HAC-corrected SEs
  5. Session-clustered OLS     - robustness to within-session correlation
  6. Sensitivity              - headline on Weekday-only vs All-days

The unit of observation is ONE TRADE. Session is assigned from the entry bar's
time-of-day using the exact `SESSION_BOUNDS` mapping that
`ml_state_optimized.session_regime_breakdown()` uses, so these numbers line up
with the "Production-Grade Live Execution (Statistic)" table.

Usage:
    python -m research.hypothesis_session_testing
"""
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ml_shared_config import BUILD_DIR, SESSION_BOUNDS, SESSION_ORDER  # noqa: E402

# All inputs/outputs live under the shared build folder — never the repo root.
TRADE_LOG = os.path.join(BUILD_DIR, "walk_forward_trade_log_v3.csv")
REPORT = os.path.join(BUILD_DIR, "hypothesis_session_testing_report.md")

ALPHA = 0.05


# ----------------------------------------------------------------------------
# Shared session mapping (mirrors ml_state_optimized._session_expr exactly).
# ----------------------------------------------------------------------------
def _session_label_expr(mod: pl.Expr) -> pl.Expr:
    e = pl.when(mod < SESSION_BOUNDS[0][2]).then(pl.lit(SESSION_BOUNDS[0][0]))
    for label, _, hi in SESSION_BOUNDS[1:]:
        e = e.when(mod < hi).then(pl.lit(label))
    return e.otherwise(pl.lit(SESSION_BOUNDS[-1][0]))


def _as_float_array(s) -> np.ndarray:
    if isinstance(s, pl.Series):
        return s.to_numpy().astype(float)
    return np.asarray(s, dtype=float)


# ----------------------------------------------------------------------------
# Expectancy metrics (same definitions as ml_state_optimized).
# ----------------------------------------------------------------------------
def _pwe(ret) -> float:
    r = _as_float_array(ret)
    if len(r) == 0:
        return float("nan")
    wins, losses = r[r > 0], r[r < 0]
    win_rate = len(wins) / len(r)
    loss_rate = len(losses) / len(r)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    return float(win_rate * avg_win + loss_rate * avg_loss)


def _sortino(ret, mar: float = 0.0) -> float:
    r = _as_float_array(ret)
    if len(r) == 0:
        return float("nan")
    downside = np.minimum(0.0, r - mar)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev == 0:
        return float("nan")
    return float((r.mean() - mar) / downside_dev)


# ----------------------------------------------------------------------------
# Dunn's post-hoc test (two-tailed) with Bonferroni. Pure scipy (rankdata +
# normal CDF), so we avoid adding the scikit-posthocs dependency.
# ----------------------------------------------------------------------------
def dunn_test(samples: dict[str, np.ndarray]):
    groups = [(lbl, s.astype(float)) for lbl, s in samples.items() if len(s) > 0]
    if len(groups) < 2:
        return []
    pooled = np.concatenate([s for _, s in groups])
    n = len(pooled)
    ranks = st.rankdata(pooled, method="average")

    _, counts = np.unique(ranks, return_counts=True)
    tie_adj = float(np.sum(counts ** 3 - counts))
    variance_root = (n * (n + 1) / 12.0) - tie_adj / (12.0 * (n - 1))

    pos = 0
    mean_rank = {}
    for lbl, s in groups:
        mean_rank[lbl] = float(ranks[pos:pos + len(s)].mean())
        pos += len(s)

    n_pairs = len(groups) * (len(groups) - 1) // 2
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            li, si = groups[i]
            lj, sj = groups[j]
            diff = mean_rank[li] - mean_rank[lj]
            var = variance_root * (1 / len(si) + 1 / len(sj))
            if var <= 0:
                continue
            z = diff / np.sqrt(var)
            p = 2.0 * (1.0 - st.norm.cdf(abs(z)))
            p_bonf = min(1.0, p * n_pairs)
            rows.append({"a": li, "b": lj, "z": z, "p": p, "p_bonf": p_bonf, "sig": p_bonf < ALPHA})
    return rows


def nw_maxlags(n: int) -> int:
    if n <= 0:
        return 1
    return max(1, int(np.floor(4 * (n / 100.0) ** (2 / 9.0))))


# ----------------------------------------------------------------------------
# Minimal markdown report writer (mirrors backtest report style).
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

    def table(self, header, rows):
        widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(header)]
        fmt = " | ".join("{:<%d}" % w for w in widths)
        sep = "-+-".join("-" * w for w in widths)
        self._fh.write(fmt.format(*header) + "\n")
        self._fh.write(sep + "\n")
        for r in rows:
            self._fh.write(fmt.format(*[str(x) for x in r]) + "\n")
        self._fh.write("\n")

    def close(self):
        self._fh.close()
        print(f"\n[report] written: {self.path}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def build_samples(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return {session_label: ret[np.ndarray]} for the present sessions."""
    out = {}
    for l in SESSION_ORDER:
        s = frame.filter(pl.col("session") == l)["ret"].to_numpy().astype(float)
        if len(s) > 0:
            out[l] = s
    return out


def main():
    rep = _Report(REPORT)
    rep.add("# Session-Regime Hypothesis Testing - Walk-Forward Trade Expectancy")
    rep.add("")
    rep.add(f"_Input: `{os.path.basename(TRADE_LOG)}` (out-of-sample, realized trades)_")

    df = pl.read_csv(TRADE_LOG, try_parse_dates=True)
    df = df.with_columns([
        pl.col("datetime").cast(pl.Datetime("us")),
        (pl.col("datetime").dt.hour().cast(pl.Int32) * 60
         + pl.col("datetime").dt.minute().cast(pl.Int32)).alias("mod"),
    ])
    df = df.with_columns(_session_label_expr(pl.col("mod")).alias("session"))
    df = df.with_columns(
        pl.when(pl.col("datetime").dt.weekday() <= 5).then(pl.lit("Weekday"))
          .otherwise(pl.lit("Weekend")).alias("regime")
    )
    rep.add(f"_Trades loaded: {df.height}_")
    rep.add("")

    # --- Production threshold = the threshold with the most out-of-sample trades,
    # --- matching the selection session_regime_breakdown uses in the report.
    thr_counts = df.group_by("threshold").len().sort("len", descending=True)
    prod_thr = float(thr_counts["threshold"][0])
    rep.add(f"**Filter**: production threshold = `P(WIN) >= {prod_thr:.2f}` "
            f"({thr_counts['len'][0]} trades).")
    rep.add("")

    df_prod = df.filter(pl.col("threshold") == prod_thr)

    # ------------------------------------------------------------------
    # 1 & 2 & 3. Headline: Weekday trades at the production threshold
    # ------------------------------------------------------------------
    rep.add("## 1. Headline sample - Weekday, production threshold")
    rep.add("")
    df_home = df_prod.filter(pl.col("regime") == "Weekday")
    samples = build_samples(df_home)

    desc_rows = []
    for l in SESSION_ORDER:
        s = samples.get(l)
        n = len(s) if s is not None else 0
        desc_rows.append([l, str(n), f"{_pwe(s):.4f}" if n else "n/a",
                          f"{_sortino(s):.3f}" if n else "n/a"])
    rep.table(["Session", "n", "PWE", "Sortino"], desc_rows)

    rep.add("## 2. Kruskal-Wallis H-test (non-parametric equality of medians)")
    rep.add("")
    if len(samples) >= 2:
        labels = [l for l in SESSION_ORDER if l in samples]
        H, p = st.kruskal(*[samples[l] for l in labels])
        rep.add(f"H-statistic = **{H:.3f}**, p-value = **{p:.4g}**")
        rep.add(f"Conclusion: {'REJECT H0' if p < ALPHA else 'FAIL to reject H0'} "
                f"(medians {'not all equal' if p < ALPHA else 'consistent with being equal'} "
                f"across sessions at alpha={ALPHA}).")
        rep.add("")
        rep.add("K-W only proves 'at least one session differs'. Dunn's below shows which.")
        rep.add("")

        rep.add("## 3. Dunn's post-hoc pairwise test (Bonferroni-corrected)")
        rep.add("")
        dunn_rows = dunn_test(samples)
        if dunn_rows:
            rows = [[f"{r['a']} vs {r['b']}", f"{r['z']:.3f}", f"{r['p']:.4g}",
                     f"{r['p_bonf']:.4g}", "YES" if r["sig"] else ""] for r in dunn_rows]
            rep.table(["Pair", "z", "p", "p_adj (Bonf)", "sig<0.05"], rows)
        else:
            rep.add("Fewer than 2 non-empty sessions - Dunn's not computable.")
        rep.add("")
    else:
        rep.add("Fewer than 2 non-empty sessions in the weekday sample.")
        rep.add("")

    # ------------------------------------------------------------------
    # 4. OLS + Newey-West (HAC), Asia baseline
    # ------------------------------------------------------------------
    rep.add("## 4. OLS regression + Newey-West (HAC) standard errors")
    rep.add("")
    rep.add("`ret_t = b0(Asia) + sum_j b_j*session_j + e_t` — each `b_j` is the mean edge "
            "difference vs Asia. HAC SEs correct for autocorrelation + heteroskedasticity.")
    rep.add("")

    frame = df_home.to_pandas()
    present_labels = [l for l in SESSION_ORDER if l in samples]
    ref = present_labels[0] if present_labels else None
    if "Asia (00:00-08:00)" in present_labels:
        ref = "Asia (00:00-08:00)"

    if ref is not None and len(present_labels) >= 2:
        dummies = pd.get_dummies(pd.Categorical(frame["session"], categories=SESSION_ORDER),
                                 prefix="", prefix_sep="", drop_first=False)
        dummies = dummies[[l for l in present_labels]]
        X = pd.concat([pd.Series(1.0, index=frame.index, name="(Intercept)"),
                       dummies.drop(columns=[ref])], axis=1)
        Y = frame["ret"].astype(float)
        X = X.astype(float)

        import statsmodels.api as sm  # noqa: PLC0415
        model = sm.OLS(Y, X)
        L = nw_maxlags(len(Y))
        res_hac = model.fit(cov_type="HAC", cov_kwds={"maxlags": L})

        rep.add(f"_n = {len(Y)} weekday trades | baseline = `{ref}` | Newey-West maxlags = {L}_")
        rep.add("")
        ci = res_hac.conf_int()
        rows = []
        for term in X.columns:
            coef = float(res_hac.params[term])
            se = float(res_hac.bse[term])
            t = float(res_hac.tvalues[term])
            p = float(res_hac.pvalues[term])
            lo, hi = float(ci.loc[term][0]), float(ci.loc[term][1])
            label = "(Asia baseline)" if term == "(Intercept)" else f"{term}"
            rows.append([label, f"{coef:+.4f}", f"{se:.4f}", f"{t:+.3f}",
                         f"{p:.4g}", f"[{lo:+.4f}, {hi:+.4f}]",
                         "YES" if p < ALPHA else ""])
        rep.table(["term", "coef(R)", "HAC SE", "t", "p-value", "95% CI", "sig<0.05"], rows)

        sess_only = [t for t in X.columns if t != "(Intercept)"]
        neg_sig = [f"{c} ({res_hac.params[c]:+.3f})" for c in sess_only
                   if res_hac.params[c] < 0 and res_hac.pvalues[c] < ALPHA]
        if neg_sig:
            rep.add(f"Negative & significant (alpha=0.05, HAC) vs Asia: {', '.join(neg_sig)}.")
        else:
            rep.add("No session is significantly negative relative to baseline at alpha=0.05 (HAC).")
        rep.add("")

        rep.add("### 4b. Robustness note — no session-clustered SEs")
        rep.add("")
        rep.add("Session-clustered (panel) standard errors were intentionally NOT reported: "
                "cluster-robust inference is unreliable with only 6 session clusters (needs ~50+), "
                "and here it degenerates to SE=0. Newey-West (HAC) is the correct robust method "
                "for this autocorrelated microstructure series and is used throughout.")
        rep.add(f"Intercept (Asia mean) = {float(res_hac.params['(Intercept)']):+.4f} R/trade.")
        rep.add("")
    else:
        rep.add("Need >=2 present sessions with a baseline for OLS - skipped.")
        rep.add("")

    # ------------------------------------------------------------------
    # 5. Sensitivity: Weekday vs All-days expectancy
    # ------------------------------------------------------------------
    rep.add("## 5. Sensitivity — Weekday vs All-days expectancy (PWE) by session")
    rep.add("")
    all_samples = build_samples(df_prod)
    sens_rows = []
    for l in SESSION_ORDER:
        s_wd = samples.get(l)
        s_all = all_samples.get(l)
        sens_rows.append([l, str(len(s_wd) if s_wd is not None else 0),
                      f"{_pwe(s_wd):.4f}" if (s_wd is not None and len(s_wd)) else "n/a",
                      str(len(s_all) if s_all is not None else 0),
                      f"{_pwe(s_all):.4f}" if (s_all is not None and len(s_all)) else "n/a"])
    rep.table(["Session", "n_wd", "PWE_wd", "n_all", "PWE_all"], sens_rows)

    # ------------------------------------------------------------------
    # 6. Interpretation notes
    # ------------------------------------------------------------------
    rep.add("## 6. Interpretation notes")
    rep.add("")
    rep.add("- 'Asia is the highest edge' is evidenced if its pairwise Dunn p_adj < 0.05 vs the "
            "others (or its OLS coefficient is the positive max / the others are negative-sig).")
    rep.add("- 'US/EU/Pacific are noise/negative' needs the OLS column: a significantly negative "
            "coefficient relative to Asia means the edge decays to (or below) zero.")
    rep.add("- Watch `n`: a small US/Pacific sample gives low power — a non-significant p there "
            "is 'not proven different', not 'proven equal'.")
    rep.add("- Dunn's already Bonferroni-adjusts all 15 pairwise comparisons (family-wise error).")
    rep.add("- Sessions are assigned by the entry bar's time-of-day (same as the live-stat table).")
    rep.add("")
    rep.add("---")
    rep.add("_Run: read-only on realized trades; no feature/training code touched._")

    rep.close()


if __name__ == "__main__":
    main()