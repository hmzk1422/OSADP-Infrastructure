"""
VPIN Chunking Lossiness Diagnostic
====================================
`test_chunking_loss()` in ml_state_optimized.py already gives a binary
lossless/lossy verdict on the FULL state space. Running it once and
getting [FAIL] does not tell you which columns are broken, how badly,
or why -- and "why" matters here because the mechanism determines
whether VPIN is fixable or whether you should fall back to a
volatility-based regime identifier (as proposed) while it's broken.

ROOT CAUSE, DERIVABLE FROM THE EXISTING CONFIG ALONE
--------------------------------------------------------
From ml_shared_config.py:
    VPIN_TARGET_BUCKETS_PER_SESSION = 50
    N_CHUNKS_PER_SESSION            = 24

`_session_total_volume()` (ml_state_builder.py) computes total session
volume GLOBALLY, so bucket *size* (total_volume / 50) is session-wide
and consistent. That's correct and necessary, but it is not
sufficient: `SessionFeatureContext` injects that fixed bucket size
into each chunk, but the CUMULATIVE volume counter that decides where
one bucket ends and the next begins is, by construction of
`_load_and_clean_chunk_events`, reset at the start of every chunk (a
chunk "only pulls the events it needs" -- it does not carry forward
the running total from the previous chunk's tail). VPIN is a volume
CLOCK, not a fixed-length rolling window: the same warmup-margin trick
that already protects the fixed-length rolling features (RSI, z-scores
-- see CHUNK_WARMUP_BARS / CHUNK_BUFFER_SECONDS) does NOT protect an
unbounded-lookback accumulator like a volume bucket. So every chunk
boundary truncates whatever bucket was mid-fill, closes it early, and
starts a new one -- the boundary is a structural discontinuity that
grows a fixed bucket SIZE (correct) but not a continuous bucket CLOCK
(missing).

With 50 target buckets and 24 chunks per session, a *coarse* upper
bound on how much of the VPIN series can be touched by a boundary
artifact is:

    corrupted_bucket_edges <= 2 * (N_CHUNKS_PER_SESSION - 1)   # each interior
                                                                 # seam can
                                                                 # truncate the
                                                                 # bucket before
                                                                 # it AND start
                                                                 # a short one
                                                                 # after it
    corrupted_bucket_edges / VPIN_TARGET_BUCKETS_PER_SESSION
        = 2 * 23 / 50 = 92%   (upper bound, not the empirical rate --
                                buckets are irregular in wall-clock time,
                                so this overstates it, but it puts you in
                                the right order of magnitude for why the
                                empirical check below comes back "quite
                                strongly" lossy rather than a rounding-
                                tolerance nitpick.)

This is a genuine, structural bug class (unbounded-state accumulator
across a chunk boundary), not noise. It will not average out with more
data; every session gets it every 24 chunks.

WHAT THIS SCRIPT DOES DIFFERENTLY FROM test_chunking_loss()
--------------------------------------------------------------
1. Scopes the comparison to VPIN-derived columns only (set
   VPIN_DERIVED_COLUMNS below to match your actual engine.py output --
   the exact names aren't visible from the files reviewed here).
2. Reports a graded severity per column (correlation, sign-agreement,
   KS distance, mean/median abs error) instead of one pass/fail.
3. Explicitly tests whether error CONCENTRATES near chunk boundaries --
   this is what actually proves the mechanism above rather than just
   restating "they differ".

Usage:
    python -m research.vpin_chunking_diagnostic
"""
import os
import sys

import numpy as np
import polars as pl
import scipy.stats as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ml_shared_config import (  # noqa: E402
    BUILD_DIR, VPIN_TARGET_BUCKETS_PER_SESSION, N_CHUNKS_PER_SESSION, DATA_FORMAT,
)

# ----------------------------------------------------------------------------
# EDIT THIS to match the actual VPIN-derived column names your engine.py
# produces. Not visible in the files reviewed for this script -- these are
# placeholders. Anything downstream of the per-chunk cumulative-volume
# accumulator belongs here (raw vpin, any z-scored/smoothed derivative of it,
# and anything that consumes a VPIN bucket-close timestamp as an input).
# ----------------------------------------------------------------------------
VPIN_DERIVED_COLUMNS = [
    "vpin", "vpin_z", "vpin_bucket_id",
]

BOUNDARY_WINDOW_BARS = 5  # "near a chunk seam" = within this many bars of it


def _analytic_upper_bound() -> float:
    return min(1.0, 2 * (N_CHUNKS_PER_SESSION - 1) / VPIN_TARGET_BUCKETS_PER_SESSION)


def _chunk_boundary_times(session_start, session_end, n_chunks: int) -> list:
    dur = (session_end - session_start) / n_chunks
    return [session_start + dur * i for i in range(1, n_chunks)]  # interior seams only


def severity_report(df_chunked: pl.DataFrame, df_unchunked: pl.DataFrame,
                     columns: list[str], key: str = "datetime") -> pl.DataFrame:
    common = sorted(set(df_chunked[key]) & set(df_unchunked[key]))
    if not common:
        raise ValueError("No overlapping timestamps between chunked and unchunked runs -- "
                          "can't compare. Did both builds cover the same session(s)?")
    c = df_chunked.filter(pl.col(key).is_in(common)).sort(key)
    u = df_unchunked.filter(pl.col(key).is_in(common)).sort(key)

    rows = []
    for col in columns:
        if col not in df_chunked.columns or col not in df_unchunked.columns:
            rows.append({"Column": col, "status": "MISSING (check VPIN_DERIVED_COLUMNS names)"})
            continue
        a = c[col].to_numpy().astype(float)
        b = u[col].to_numpy().astype(float)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) == 0:
            rows.append({"Column": col, "status": "no overlapping finite values"})
            continue
        abs_err = np.abs(a - b)
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
        sign_agree = float(np.mean(np.sign(a) == np.sign(b)))
        ks_stat, ks_p = st.ks_2samp(a, b)
        rows.append({
            "Column": col, "status": "ok",
            "n": len(a),
            "corr(chunked,unchunked)": round(corr, 4),
            "sign_agreement": round(sign_agree, 4),
            "mean_abs_err": round(float(abs_err.mean()), 6),
            "median_abs_err": round(float(np.median(abs_err)), 6),
            "p95_abs_err": round(float(np.quantile(abs_err, 0.95)), 6),
            "KS_stat": round(float(ks_stat), 4),
            "KS_p": round(float(ks_p), 4),
        })
    return pl.DataFrame(rows)


def boundary_concentration_report(df_chunked: pl.DataFrame, df_unchunked: pl.DataFrame,
                                   columns: list[str], boundary_times: list,
                                   key: str = "datetime") -> pl.DataFrame:
    """The mechanism test: is error near a chunk seam larger than error far
    from one? If yes, that's direct evidence the chunking boundary itself
    (not general noise) is the source -- confirms the root-cause story above
    rather than just asserting it."""
    common = sorted(set(df_chunked[key]) & set(df_unchunked[key]))
    c = df_chunked.filter(pl.col(key).is_in(common)).sort(key).to_pandas()
    u = df_unchunked.filter(pl.col(key).is_in(common)).sort(key).to_pandas()

    boundary_arr = np.array(boundary_times, dtype="datetime64[ns]")
    times = c[key].to_numpy().astype("datetime64[ns]")
    # distance (in bars, approximated via index position) to nearest boundary time
    dist_bars = np.full(len(times), np.inf)
    if len(times) > 1:
        median_step = np.median(np.diff(times)).astype("timedelta64[s]").astype(float)
        median_step = max(median_step, 1.0)
        for bt in boundary_arr:
            d = np.abs((times - bt).astype("timedelta64[s]").astype(float)) / median_step
            dist_bars = np.minimum(dist_bars, d)

    near_mask = dist_bars <= BOUNDARY_WINDOW_BARS
    rows = []
    for col in columns:
        if col not in c.columns or col not in u.columns:
            continue
        a = c[col].to_numpy().astype(float)
        b = u[col].to_numpy().astype(float)
        finite = np.isfinite(a) & np.isfinite(b)
        err = np.abs(a - b)
        near_err = err[near_mask & finite]
        far_err = err[~near_mask & finite]
        if len(near_err) == 0 or len(far_err) == 0:
            continue
        stat, p = st.mannwhitneyu(near_err, far_err, alternative="greater")
        rows.append({
            "Column": col,
            "n_near_boundary": len(near_err), "n_far_from_boundary": len(far_err),
            "mean_abs_err_near": round(float(near_err.mean()), 6),
            "mean_abs_err_far": round(float(far_err.mean()), 6),
            "ratio_near_over_far": round(float(near_err.mean() / (far_err.mean() + 1e-12)), 2),
            "MannWhitneyU_p (near > far)": round(float(p), 4),
        })
    return pl.DataFrame(rows)


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
    rep = _Report(os.path.join(BUILD_DIR, "vpin_chunking_diagnostic_report.md"))
    rep.add("# VPIN Chunking Lossiness Diagnostic")
    rep.add("")
    rep.add(f"_VPIN_TARGET_BUCKETS_PER_SESSION={VPIN_TARGET_BUCKETS_PER_SESSION}, "
            f"N_CHUNKS_PER_SESSION={N_CHUNKS_PER_SESSION}_")
    rep.add("")
    bound = _analytic_upper_bound()
    rep.add(f"## 0. Analytic upper bound on affected buckets: **{bound:.0%}**")
    rep.add("")
    rep.add("(See module docstring for derivation. This is a structural upper bound from the "
            "config alone, not the measured rate below -- it explains *why* a large empirical "
            "loss is expected, not a substitute for measuring it.)")
    rep.add("")

    if DATA_FORMAT != "csv":
        rep.add("**NOTE**: `test_chunking_loss()` in ml_state_optimized.py's `__main__` only runs "
                "when `DATA_FORMAT == \"csv\"`. Current `DATA_FORMAT` is "
                f"`{DATA_FORMAT!r}` -- the existing lossless check is not being run at all in the "
                "parquet path. If the parquet path is what's used to build the training data you "
                "are currently trusting, that path has ZERO lossless validation on it right now. "
                "Either add the same check there, or confirm parquet-path chunking uses a genuinely "
                "different (continuous) bucket accumulator than the csv path before assuming this "
                "diagnostic's csv-path findings also apply to it.")
        rep.add("")

    from research.ofmr.ml_state_builder import build_historical_state_space  # noqa: E402

    rep.add("## 1. Building chunked vs unchunked state space (same session data, two configs)")
    rep.add("")
    df_unchunked = build_historical_state_space(chunk_sessions=False)
    df_chunked = build_historical_state_space(chunk_sessions=True, chunks_per_session=N_CHUNKS_PER_SESSION)
    rep.add(f"unchunked rows={df_unchunked.height}, chunked rows={df_chunked.height}")
    rep.add("")

    rep.add("## 2. Per-column severity (VPIN-derived columns only)")
    rep.add("")
    sev = severity_report(df_chunked, df_unchunked, VPIN_DERIVED_COLUMNS)
    rep.table(sev)
    if "corr(chunked,unchunked)" in sev.columns:
        bad = sev.filter(pl.col("corr(chunked,unchunked)") < 0.95)
        if bad.height > 0:
            rep.add(f"**{bad.height} of {sev.height} VPIN-derived columns fall below 0.95 "
                    f"correlation between chunked and unchunked** -- this is the 'lossy, quite "
                    f"strongly so' finding, quantified rather than asserted.")
    rep.add("")

    rep.add("## 3. Boundary-concentration test (does error cluster at chunk seams?)")
    rep.add("")
    from research.ofmr.ml_state_builder import _session_time_bounds, get_session_paths
    sessions = get_session_paths()
    all_boundaries = []
    for ob_path, tr_path, _sid in sessions:
        s, e = _session_time_bounds(tr_path)
        all_boundaries.extend(_chunk_boundary_times(s, e, N_CHUNKS_PER_SESSION))
    conc = boundary_concentration_report(df_chunked, df_unchunked, VPIN_DERIVED_COLUMNS, all_boundaries)
    rep.table(conc)
    if conc.height > 0:
        sig = conc.filter(pl.col("MannWhitneyU_p (near > far)") < 0.05)
        if sig.height > 0:
            rep.add(f"**Error is significantly larger within {BOUNDARY_WINDOW_BARS} bars of a chunk "
                    f"seam for {sig.height}/{conc.height} columns (p<0.05, one-sided Mann-Whitney)** "
                    f"-- this confirms the chunk-boundary bucket-reset mechanism, it isn't diffuse "
                    f"noise spread evenly through the session.")
    rep.add("")

    rep.add("## 4. Recommendation")
    rep.add("")
    rep.add("- Do NOT use the current chunked VPIN output for anything that feeds the model or a "
            "session/regime identifier until the bucket accumulator carries state across chunk "
            "boundaries (analogous to how CHUNK_WARMUP_BARS already protects fixed-length rolling "
            "features -- VPIN needs the equivalent for an unbounded volume-clock accumulator, e.g. "
            "passing forward the in-progress bucket's running volume/imbalance instead of only a "
            "session-wide *size* constant).")
    rep.add("- Interim (matches what was proposed): use realized-volatility-based regime "
            "identification instead of VPIN for RQ1's session/regime segmentation. A rolling-window "
            "volatility estimator is chunk-boundary-safe under the SAME CHUNK_WARMUP_BARS machinery "
            "already protecting RSI/z-score features, because it has bounded lookback -- it does not "
            "need a session-continuous accumulator the way a volume-clock does.")
    rep.add("- Once fixed, re-run this diagnostic before switching back -- 'no longer FAILs the "
            "binary check' is not the same claim as 'severity table above is now clean'.")
    rep.add("")
    rep.add("---")
    rep.close()


if __name__ == "__main__":
    main()