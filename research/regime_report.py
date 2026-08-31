"""
regime_report.py
================
Final orchestrator + research report generator for the regime study.

Runs the three regime methodologies end-to-end (A: walk-forward buckets,
B: shallow supervised trees, C: HMM with forward-filtered inference), the
strategy-state predictability tests, drawdown attribution and asset analysis,
then writes `build/regime_study/regime_research_report.md` answering the ten
study questions with actual measured numbers.

Run:  python -m research.regime_report

Note: This still makes some assumptions (Long only in MAE/MFE calc, perform full correlation
which is an information that doesn't exist so this needs to be a static study)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.regime_data import (  # noqa: E402
    load_study_frame, ret_metrics, fmt_metrics, ensure_out, THRESHOLD, TRADE_LOG,
    FOLD_MODE, FOLD_N_SAMPLES, FOLD_BLOCK_SIZE, FOLD_HORIZON_SECONDS,
    FOLD_EMBARGO_SECONDS,
)
from research.regime_buckets import run_buckets_all  # noqa: E402
from research.regime_trees import run_trees, evaluate_filter, stability_of_filter  # noqa: E402
from research.regime_hmm import (  # noqa: E402
    run_hmm_study, summarize_hmm, summarize_hmm_aligned, transition_persistence,
    hmm_state_prob_predicts_ret, hmm_confidence_filter,
)

OUT = os.path.join(ensure_out(), "regime_research_report.md")


def _md_table(df: pd.DataFrame, max_rows: int = 15, float_fmt: str = "{:+.3f}") -> str:
    """Markdown table without external deps (no tabulate)."""
    if df is None or not len(df):
        return "_n/a_\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _metrics_row(name: str, m: dict) -> dict:
    return {"subset": name, "n": m["n"], "mean_r": m["mean_r"],
            "median_r": m["median_r"], "win_rate": m["win_rate"],
            "p_1r": m["p_1r"], "p_2r": m["p_2r"], "p_3r": m["p_3r"],
            "pf": m["pf"], "total_r": m["total_r"]}


def main() -> None:
    print("[1/6] loading study frame ...")
    df = load_study_frame()
    R = []  # report lines
    A = R.append

    base = ret_metrics(df["ret"].to_numpy())
    months = df["year_month"].nunique() if "year_month" in df else np.nan
    A("# Regime & Conditional-Edge Research Report")
    A("")
    A(f"- data: `{os.path.basename(TRADE_LOG)}`, confidence threshold **{THRESHOLD}**")
    A(f"- trades: **{base['n']}** over **{months}** months "
      f"({df['dt'].min():%Y-%m-%d} .. {df['dt'].max():%Y-%m-%d})")
    A("- leakage contract: entry conditions = `_open` variants + strictly-prior "
      "trades only; every learner fit per expanding-window fold, scored on the "
      "following unseen month")
    fold_scheme = (f"**{FOLD_MODE}**" if FOLD_MODE != "purged" else
                   f"**{FOLD_MODE}** (n_samples={FOLD_N_SAMPLES}, "
                   f"block={FOLD_BLOCK_SIZE} trades, horizon={FOLD_HORIZON_SECONDS}s, "
                   f"embargo={FOLD_EMBARGO_SECONDS}s)")
    A(f"- fold scheme: {fold_scheme} (`regime_data.make_folds`; switch via "
      "`REGIME_FOLD_MODE` in `research/ml_shared_config.py`)")
    A("")
    A(f"**Unconditional baseline (all trades):** {fmt_metrics(base)}")
    A("")

    # ------------------------------------------------------------------ A
    print("[2/6] method A - walk-forward buckets ...")
    b = run_buckets_all(df)
    summ = b["buckets"]["summary"]
    A("## 1. Method A - simple quantile buckets (walk-forward, pooled OOS)")
    A("")
    A(f"Comparisons tested: {b['buckets']['n_comparisons']} variable-bucketings; "
      "p-values are Kruskal-Wallis across buckets on pooled OOS trades, "
      "BH-FDR adjusted.")
    A("")
    A("### Largest OOS Q5-Q1 expectancy spreads")
    A(_md_table(summ[["name", "family", "oos_spread", "oos_n_min",
                      "kw_p_oos", "kw_p_oos_bh", "oos_buckets", "is_spread"]], 20))
    sig = summ[summ["kw_p_oos_bh"] < 0.10]
    A("Variables surviving BH-FDR at 10%: "
      + (", ".join(sig["name"]) if len(sig) else "**NONE**"))
    A("")

    inter = b["interactions"]
    A("### Pairwise tercile interactions (pooled OOS, min n=60/cell)")
    if len(inter):
        inter2 = inter.assign(abs_mr=inter["mean_r"].abs()).sort_values(
            "abs_mr", ascending=False)
        A(_md_table(inter2[["pair", "A", "B", "n", "mean_r", "win_rate",
                            "p_3r", "total_r", "med_mae", "med_mfe"]], 15))
    else:
        A("_no interaction cell passed the minimum sample size_")


    dd = b["drawdown"]
    A("### Drawdown attribution (equity dips <= -8R)")
    A(f"- trades in drawdown: {dd['n_in_dd']} (meanR {dd['in_dd_metrics']['mean_r']:+.3f}, "
      f"WR {dd['in_dd_metrics']['win_rate']*100:.1f}%) vs healthy: "
      f"{dd['n_out_dd']} (meanR {dd['out_dd_metrics']['mean_r']:+.3f}, "
      f"WR {dd['out_dd_metrics']['win_rate']*100:.1f}%)")
    A("")
    A("State-variable distribution shifts inside drawdowns (KS test, sorted by p):")
    A(_md_table(dd["state_table"], 12, "{:+.4f}"))
    A("")

    # ------------------------------------------------------------------ B
    print("[3/6] method B - shallow supervised trees ...")
    ev = evaluate_filter(df)
    A("## 2. Method B - shallow regression trees (depth 2-4, big leaves, walk-forward)")
    A("")
    rows = [_metrics_row(k, v) for k, v in ev["metrics"].items()]
    A(_md_table(pd.DataFrame(rows)))
    fav = ev["metrics"].get("favorable", {"n": 0, "mean_r": np.nan})
    unf = ev["metrics"].get("unfavorable", {"n": 0, "mean_r": np.nan})
    A("")
    A(f"Favorable-leaf filter: {fav['n']} OOS trades at "
      f"{fav.get('mean_r', float('nan')):+.3f}R vs baseline "
      f"{ev['metrics']['baseline']['mean_r']:+.3f}R; unfavorable leaves: "
      f"{unf.get('n', 0)} trades at {unf.get('mean_r', float('nan')):+.3f}R.")
    A("")
    A("### Monthly stability of the favorable-leaf filter (OOS)")
    A(_md_table(stability_of_filter(df, ev["labels"], "favorable"), 20))
    A("")

    # ------------------------------------------------------------------ C
    print("[4/6] method C - HMM (forward-filtered inference) ...")
    hm = run_hmm_study(df)
    al = summarize_hmm_aligned(hm)
    A("## 3. Method C - HMM latent states (fit per fold, FORWARD-FILTERED probs only)")
    A("")
    A("Pooled OOS, profile-aligned states (mean across folds per aligned state):")
    A(_md_table(al, 20))
    A("")
    pers = transition_persistence(hm)
    if len(pers):
        A(f"State persistence (max self-transition prob): median "
          f"{pers['prob'].median():.3f}, range {pers['prob'].min():.3f}-"
          f"{pers['prob'].max():.3f} across K/folds.")
    A("")
    print("[5/6] HMM predictability + confidence filter ...")
    pr = hmm_state_prob_predicts_ret(df)
    if len(pr):
        sig_p = pr[pr["spearman_p"] < 0.05]
        A(f"Filtered-probability -> ret predictability: "
          f"{len(sig_p)}/{len(pr)} (K, fold, state) cells with Spearman p<0.05 "
          f"(~5% expected by chance). Median |rho| = "
          f"{pr['spearman'].abs().median():.4f}.")
        A("")
        A("High-confidence (prob>=0.6) cells vs fold baseline:")
        A(_md_table(pr[["K", "fold", "state", "n_hi", "hi_mean_r", "hi_win_rate",
                        "spearman", "spearman_p"]], 20))
    A("")
    cf = hmm_confidence_filter(df, K=2, prob_thr=0.6)
    A("### HMM confidence filter (trade only when state prob>=0.6, OOS)")
    for k, v in cf.items():
        if isinstance(v, dict) and "n" in v:
            A(f"- {k}: {fmt_metrics(v)}")
    A("")

    # ------------------------------------------------------------------ assets
    print("[6/6] asset analysis + report assembly ...")
    A("## 4. Asset analysis")
    cand = [c for c in df.columns
            if c.lower() in ("symbol", "asset", "instrument", "ticker", "pair")]
    if cand:
        col = cand[0]
        A(f"Asset column detected: `{col}`")
        t = df.groupby(col)["ret"].apply(
            lambda s: pd.Series(ret_metrics(s.to_numpy()))).unstack()
        A(_md_table(t.reset_index(), 10))
    else:
        A("No symbol/asset column in the trade log - the v3 walk-forward log is a "
          "single-asset stream, so asset x regime and conditional cross-asset "
          "diversification are **not evaluable** at this threshold. Re-run the "
          "portfolio walk-forward (multi-symbol) to enable section 14 of the spec.")
    A("")

    # ------------------------------------------------------------------ verdict
    best = summ.iloc[0] if len(summ) else None
    spread_ok = (best is not None and abs(best["oos_spread"]) > 0.05
                 and best["kw_p_oos_bh"] < 0.10)
    A("## 5. Verdict on the ten study questions")
    A("")
    A(f"1. **Conditional expectancy:** "
      f"{'PRESENT (see bucket spreads above)' if spread_ok else 'WEAK / NOT ESTABLISHED - no bucketing survived FDR with a material spread'}")
    A(f"2. **Most influential observable states:** "
      f"{', '.join(summ.head(3)['name']) if len(summ) else 'n/a'}")
    A("3. **Volatility conditioning:** see `vol_*` rows in the spread table and "
      "the vol x trend/momentum/OFI interaction cells.")
    A("4. **Order-flow information:** compare `ofi/cvd/absorption/depth` spreads "
      "with the vol x order-flow interaction cells (incremental after vol).")
    A("5. **Recent strategy behavior:** see `strat_prev*` rows (esp. quick_frac / "
      "median_gap) and the vol x stop-out interaction.")
    if len(pers):
        A(f"6. **Persistent HMM states:** persistence median {pers['prob'].median():.3f} "
          f"({'persistent' if pers['prob'].median() > 0.7 else 'weakly persistent'})")
    A("7. **OOS stability:** all tables above are pooled OOS by construction; "
      "monthly stability tables in the CSVs.")
    A("8. **Asset differences:** not evaluable (single-asset log).")
    A("9. **Drawdowns:** compare in-DD vs out-DD state table (KS p-values) above.")
    fav_edge = fav.get("mean_r", float("nan")) - ev["metrics"]["baseline"]["mean_r"]
    A(f"10. **Defensible filter:** tree favorable-leaf filter ({fav_edge:+.3f}R vs "
      f"baseline on {fav.get('n', 0)} OOS trades) and the HMM confidence filter - "
      "adopt only if spreads are material AND monthly stability shows a "
      "consistent sign.")
    A("")
    A("_Full per-variable bucket tables, stability tables and raw outputs are in "
      "`build/regime_study/*.csv`._")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print(f"[done] report -> {OUT}")


if __name__ == "__main__":
    main()
