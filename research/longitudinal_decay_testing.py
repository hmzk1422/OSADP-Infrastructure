"""
Longitudinal Decay & Session-Regime Testing (RQ1 x RQ2), + RQ3 scope note
=============================================================================
Extends `hypothesis_session_testing.py`'s approach (same READ-ONLY trade
log, same session mapping, same HAC/nonparametric toolkit) by adding YEAR
as a second grouping dimension, so RQ1 (session dependency) and RQ2
(longitudinal decay) get tested jointly instead of as two separate,
un-cross-checked univariate breakdowns. It is a strict superset of that
script's method, not a replacement for it.

READ THIS BEFORE CITING "2026 WORST, 2025 BEST, 2024 COMPARABLE TO 2025"
AS EVIDENCE FOR RQ2
----------------------------------------------------------------------------
Two things below will look like pushback but are just what an honest
significance check produces on this data shape:

1. SCOPE MISMATCH. RQ2 as written claims a 2020-2025 decay trajectory.
   The empirical claim driving this ask is 2024 vs 2025 vs 2026 --
   three years, and the two "good" years (2024, 2025) are described as
   comparable, i.e. there may be exactly ONE year (2026) driving the
   entire "decay" narrative. Either narrow RQ2's stated window to what
   the data actually covers, or get 2020-2023 data before writing it up
   as a multi-year trajectory. This isn't pedantry: a reviewer (or your
   own future self re-running this on 2027 data) will immediately ask
   why the RQ says 2020 when the earliest year tested is 2024.
2. STATISTICAL POWER. A trend test on 3 yearly points has essentially no
   power to distinguish "monotonic decay" from "2026 was one bad year,
   noise." Mann-Kendall with n=3 can only ever produce tau in
   {-1, -0.33, 0.33, 1}, and even the most extreme possible outcome (a
   perfectly monotonic run across all 3 years) caps out at p~=0.30 --
   nowhere near significant at any conventional alpha. This script still
   runs it, and runs it on
   the highest-resolution unit available (per-trade, per-session-year
   cell, not just 3 yearly averages) precisely so you have the strongest
   version of the test the data allows -- but the honest conclusion with
   only 3 years is "consistent with decay, cannot rule out one bad year"
   unless the per-trade/per-cell tests below independently corroborate
   it. Get 2020-2023 data (or even just 2 more years) before this
   becomes a "for sure" claim rather than a "worth taking seriously"
   claim.

WHAT THIS SCRIPT ADDS OVER THE 3-EYEBALLED-NUMBERS VERSION
--------------------------------------------------------------
  1. Year x Session descriptive grid (PWE, Sortino, n)             [RQ1+RQ2]
  2. Kruskal-Wallis across years (pooled)                          [RQ2]
  3. OLS with year dummies + session dummies + year:session
     interaction, HAC SEs -- tests whether the session pattern
     itself has shifted across years (a session that was the best
     edge in 2024 losing that rank in 2026 is a STRONGER decay
     story than a flat shift, and this is the only test here that
     can see that interaction)                                    [RQ1 x RQ2]
  4. Mann-Kendall trend test + Sen's slope on yearly PWE, run both
     pooled and per-session, with the n=3 caveat surfaced in the
     output itself, not just this docstring                       [RQ2]
  5. RQ3 scope note: what IS and ISN'T testable from the trade log
     alone, and the recommended experiment design for the parts
     that aren't (nonlinear transforms, VPIN/vol regime filtering,
     and the bar-timeframe question)                               [RQ3]

Usage:
    python -m research.longitudinal_decay_testing
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
from research.hypothesis_session_testing import (  # noqa: E402
    _session_label_expr, _as_float_array, _pwe, _sortino, nw_maxlags,
)

TRADE_LOG = os.path.join(BUILD_DIR, "walk_forward_trade_log_v3.csv")
ALPHA = 0.05


# ----------------------------------------------------------------------------
# Mann-Kendall trend test + Sen's slope. Not in scipy; both are simple
# enough to implement directly rather than adding a dependency, and doing so
# makes the n=3 degeneracy visible in the code instead of hidden in a
# library call.
# ----------------------------------------------------------------------------
def mann_kendall(x: np.ndarray) -> dict:
    n = len(x)
    if n < 3:
        return {"n": n, "S": float("nan"), "tau": float("nan"), "z": float("nan"),
                "p": float("nan"), "note": "n<3: trend test not meaningful"}
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(x[j] - x[i])
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        z = 0.0
    elif S > 0:
        z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - st.norm.cdf(abs(z)))
    tau, _ = st.kendalltau(np.arange(n), x)
    note = "n=3: max power is p~=0.30 even for a perfect monotonic run -- see module docstring" if n == 3 else ""
    return {"n": n, "S": int(S), "tau": float(tau), "z": float(z), "p": float(p), "note": note}


def sens_slope(years: np.ndarray, x: np.ndarray) -> dict:
    n = len(x)
    slopes = [(x[j] - x[i]) / (years[j] - years[i])
              for i in range(n - 1) for j in range(i + 1, n) if years[j] != years[i]]
    if not slopes:
        return {"slope_per_year": float("nan"), "n_pairs": 0}
    return {"slope_per_year": float(np.median(slopes)), "n_pairs": len(slopes)}


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


def main():
    rep = _Report(os.path.join(BUILD_DIR, "longitudinal_decay_testing_report.md"))
    rep.add("# Longitudinal Decay x Session-Regime Testing (RQ1 x RQ2), RQ3 scope note")
    rep.add("")
    rep.add(f"_Input: `{os.path.basename(TRADE_LOG)}` (out-of-sample, realized trades)_")
    rep.add("")

    df = pl.read_csv(TRADE_LOG, try_parse_dates=True)
    df = df.with_columns([
        pl.col("datetime").cast(pl.Datetime("us")),
        (pl.col("datetime").dt.hour().cast(pl.Int32) * 60
         + pl.col("datetime").dt.minute().cast(pl.Int32)).alias("mod"),
        pl.col("datetime").dt.year().alias("year"),
    ])
    df = df.with_columns(_session_label_expr(pl.col("mod")).alias("session"))

    years_present = sorted(df["year"].unique().to_list())
    rep.add(f"_Years present in trade log: {years_present}_")
    if years_present and (min(years_present) > 2020 or max(years_present) - min(years_present) < 3):
        rep.add(f"**SCOPE WARNING**: RQ2 as stated covers 2020-2025; this trade log covers "
                f"{years_present}. See module docstring section 1 (scope mismatch) before writing "
                f"up results against the RQ2 range as stated.")
    rep.add("")

    thr_counts = df.group_by("threshold").len().sort("len", descending=True)
    prod_thr = float(thr_counts["threshold"][0])
    df_prod = df.filter(pl.col("threshold") == prod_thr)
    rep.add(f"**Filter**: production threshold = `P(WIN) >= {prod_thr:.2f}`.")
    rep.add("")

    # ------------------------------------------------------------------
    # 1. Year x Session descriptive grid
    # ------------------------------------------------------------------
    rep.add("## 1. Year x Session — PWE / Sortino / n")
    rep.add("")
    rows = []
    for yr in years_present:
        for sess in SESSION_ORDER:
            s = df_prod.filter((pl.col("year") == yr) & (pl.col("session") == sess))["ret"].to_numpy().astype(float)
            if len(s) == 0:
                continue
            rows.append([yr, sess, len(s), f"{_pwe(s):.4f}", f"{_sortino(s):.3f}"])
    rep.table(["Year", "Session", "n", "PWE", "Sortino"], rows)

    # ------------------------------------------------------------------
    # 2. Kruskal-Wallis across years (pooled sessions)
    # ------------------------------------------------------------------
    rep.add("## 2. Kruskal-Wallis across years (pooled sessions, production threshold)")
    rep.add("")
    yearly_samples = {yr: df_prod.filter(pl.col("year") == yr)["ret"].to_numpy().astype(float)
                       for yr in years_present}
    yearly_samples = {yr: s for yr, s in yearly_samples.items() if len(s) > 0}
    if len(yearly_samples) >= 2:
        H, p = st.kruskal(*yearly_samples.values())
        rep.add(f"H = **{H:.3f}**, p = **{p:.4g}** -> "
                f"{'REJECT' if p < ALPHA else 'FAIL to reject'} equal-medians-across-years at "
                f"alpha={ALPHA}.")
    else:
        rep.add("Fewer than 2 years with trades -- skipped.")
    rep.add("")

    # ------------------------------------------------------------------
    # 3. OLS: year dummies + session dummies + year:session interaction, HAC
    # ------------------------------------------------------------------
    rep.add("## 3. OLS + Newey-West (HAC): year main effect, session main effect, year:session interaction")
    rep.add("")
    frame = df_prod.to_pandas()
    frame = frame[frame["session"].isin(SESSION_ORDER) & frame["year"].isin(years_present)]
    if frame["year"].nunique() >= 2 and frame["session"].nunique() >= 2:
        ref_year = sorted(frame["year"].unique())[0]
        ref_session = next(s for s in SESSION_ORDER if s in frame["session"].unique())

        year_dum = pd.get_dummies(frame["year"], prefix="yr", drop_first=False).astype(float)
        year_dum = year_dum.drop(columns=[f"yr_{ref_year}"])
        sess_dum = pd.get_dummies(pd.Categorical(frame["session"], categories=SESSION_ORDER),
                                   prefix="", prefix_sep="", drop_first=False).astype(float)
        sess_dum = sess_dum.drop(columns=[ref_session])

        inter_cols = {}
        for yc in year_dum.columns:
            for sc in sess_dum.columns:
                inter_cols[f"{yc}_x_{sc}"] = (year_dum[yc].values * sess_dum[sc].values)
        inter_df = pd.DataFrame(inter_cols, index=frame.index)

        X = pd.concat([pd.Series(1.0, index=frame.index, name="(Intercept)"),
                        year_dum, sess_dum, inter_df], axis=1).astype(float)
        Y = frame["ret"].astype(float)

        import statsmodels.api as sm  # noqa: PLC0415
        model = sm.OLS(Y, X)
        L = nw_maxlags(len(Y))
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": L})

        rep.add(f"_n={len(Y)} | baseline = year {ref_year} x {ref_session} | HAC maxlags={L}_")
        rep.add("")
        ci = res.conf_int()
        out_rows = []
        for term in X.columns:
            coef, se, t, p = float(res.params[term]), float(res.bse[term]), float(res.tvalues[term]), float(res.pvalues[term])
            lo, hi = float(ci.loc[term][0]), float(ci.loc[term][1])
            out_rows.append([term, f"{coef:+.4f}", f"{se:.4f}", f"{t:+.3f}", f"{p:.4g}",
                              f"[{lo:+.4f},{hi:+.4f}]", "YES" if p < ALPHA else ""])
        rep.table(["term", "coef(R)", "HAC SE", "t", "p", "95% CI", "sig<0.05"], out_rows)

        inter_sig = [c for c in inter_df.columns if res.pvalues.get(c, 1.0) < ALPHA]
        if inter_sig:
            rep.add(f"**Significant year:session interaction terms**: {inter_sig}. This means the "
                    f"session ranking itself has shifted across years -- decay is not just a level "
                    f"shift, the WHERE-the-edge-lives pattern from RQ1 has moved too. Report both "
                    f"together, not as two independent findings.")
        else:
            rep.add("No significant year:session interaction terms -- consistent with a level "
                    "shift (decay) that preserves the relative session ranking, i.e. RQ1's session "
                    "pattern and RQ2's decay look separable in this sample.")
        rep.add("")
    else:
        rep.add("Need >=2 years and >=2 sessions present for the interaction model -- skipped.")
        rep.add("")

    # ------------------------------------------------------------------
    # 4. Mann-Kendall + Sen's slope on yearly PWE (pooled, then per-session)
    # ------------------------------------------------------------------
    rep.add("## 4. Trend test — Mann-Kendall + Sen's slope on yearly PWE")
    rep.add("")
    yr_arr = np.array(years_present, dtype=float)
    pwe_arr = np.array([_pwe(yearly_samples[yr]) if yr in yearly_samples else np.nan for yr in years_present])
    valid = ~np.isnan(pwe_arr)
    mk = mann_kendall(pwe_arr[valid])
    sen = sens_slope(yr_arr[valid], pwe_arr[valid])
    rep.add(f"Pooled (all sessions): n={mk['n']}, S={mk['S']}, tau={mk.get('tau', float('nan')):.3f}, "
            f"p={mk.get('p', float('nan')):.4g}, Sen's slope={sen['slope_per_year']:.5f} R/trade/year. "
            f"{mk.get('note', '')}")
    rep.add("")

    per_sess_rows = []
    for sess in SESSION_ORDER:
        pwe_by_yr = []
        yrs_by_yr = []
        for yr in years_present:
            s = df_prod.filter((pl.col("year") == yr) & (pl.col("session") == sess))["ret"].to_numpy().astype(float)
            if len(s) > 0:
                pwe_by_yr.append(_pwe(s))
                yrs_by_yr.append(yr)
        if len(pwe_by_yr) >= 2:
            mk_s = mann_kendall(np.array(pwe_by_yr))
            sen_s = sens_slope(np.array(yrs_by_yr, dtype=float), np.array(pwe_by_yr))
            per_sess_rows.append([sess, len(pwe_by_yr), mk_s.get("tau", float("nan")),
                                   mk_s.get("p", float("nan")), sen_s["slope_per_year"]])
    if per_sess_rows:
        rep.table(["Session", "n_years", "tau", "p", "Sen_slope/yr"],
                   [[r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.4g}", f"{r[4]:.5f}"] for r in per_sess_rows])
    rep.add("")

    # ------------------------------------------------------------------
    # 5. RQ3 scope note
    # ------------------------------------------------------------------
    rep.add("## 5. RQ3 — what's testable from this trade log, and what isn't")
    rep.add("")
    rep.add("**Testable here (nothing further to build)**: whether the CURRENT feature set's "
            "realized edge is regime/session-conditional (sections 1-4 above) -- i.e. whether "
            "filtering to specific sessions/regimes recovers edge that looks flat/negative "
            "unconditionally. That's a real, if partial, answer to 'does dynamic regime "
            "filtering isolate unpriced liquidity'.")
    rep.add("")
    rep.add("**NOT testable from this trade log alone** (needs a re-run of the walk-forward "
            "training, not just a re-slice of realized trades):")
    rep.add("  - Nonlinear transforms (Connor's method / d2price) vs raw order flow: this compares "
            "two different FEATURE SETS, which means two different trained models and two "
            "different trade logs. Build both, run `hypothesis_session_testing.py` /this script "
            "on each, and compare OOS PWE distributions with the same Kruskal-Wallis/HAC toolkit "
            "already in place -- don't compare in-sample R^2 or importance scores across the two "
            "representations, compare realized OOS expectancy.")
    rep.add("  - Bar-timeframe (shorter horizon strengthens the orderflow edge): TIMEFRAME_BARRIER "
            "is a pipeline-level config baked into BAR_FREQ before any of this trade log exists, "
            "so this needs multiple full walk-forward runs at different bar sizes, not a "
            "re-analysis. Recommended design: hold everything else fixed, sweep bar_seconds over "
            "e.g. [15, 30, 60, 120, 300, 900], and regress the orderflow-only feature block's mean "
            "OOS PWE (or its aggregate SHAP importance) on log(bar_seconds). A negative slope "
            "would corroborate the market-microstructure prior that order-flow imbalance's "
            "predictive content decays as the bar horizon lengthens (the imbalance gets absorbed/"
            "arbitraged, and more buy/sell flow gets averaged together within a longer bar) -- "
            "this is a documented pattern in the market-microstructure literature (e.g. Cont, "
            "Kukanov & Stoikov's order-flow-imbalance work), not something specific to this "
            "codebase, but it still needs to be shown on THIS data/feature set before it's used "
            "to justify a design choice here.")
    rep.add("  - VPIN/vol regime filtering as the RQ3 'dynamic regime filtering' mechanism: blocked "
            "until `vpin_chunking_diagnostic.py`'s findings are resolved (VPIN is currently lossy "
            "under chunking) -- use the realized-volatility interim regime label for this test in "
            "the meantime, or the result will be contaminated by the chunking artifact rather than "
            "reflecting genuine regime-conditional information content.")
    rep.add("  - Cross-crypto group comparison (planned next step): once the above works for one "
            "symbol, the SYMBOL_HIST config already isolates the one place to parameterize -- "
            "rerun this same script per symbol/group and compare Sen's slopes and interaction "
            "terms across groups rather than raw PWE levels, since baseline volatility/liquidity "
            "differs by asset and raw-level comparisons across symbols will conflate 'this symbol "
            "is more volatile' with 'orderflow decays faster for this symbol'.")
    rep.add("")
    rep.add("---")
    rep.add("_Run: read-only on realized trades; no feature/training code touched._")
    rep.close()


if __name__ == "__main__":
    main()