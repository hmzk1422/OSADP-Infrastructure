"""
Replay Parity Diagnostics — Python research txt vs Rust live-test txt
=====================================================================
Answers one question: "when the SAME historical feed is replayed through the
research walk-forward pipeline and the Rust `live-test` engine, how similar is
every trade column?"

It compares, COLUMN BY COLUMN, the two realized trade logs:

  Python (txt):  `build/walk_forward_trade_log_v3.csv`
                 written by STEP 10 of research/ofmr/ml_state_optimized.py
                 columns: datetime, ret, threshold, cum_pnl   (ret in R units)

  Rust (txt):    the `pnl` column family of the live-test RocksDB store
                 (same JSON records research/extract_rocksdb.py exports)
                 columns: entry_ms, exit_ms, side, entry_price, exit_price,
                 reason, qty, net_usd, r_multiple, r_raw, equity_after,
                 duration_s, session_id, barrier_pct, target_price, stop_price
                 (`r_raw` = R BEFORE the research STEP 5 dataset clip; the
                 dual-pass cap is reconstructed from it before comparison)

Each comparable column gets a similarity verdict: what % of aligned trade
pairs agree (within a tolerance), the mean/max absolute difference, and the
Pearson correlation. Trades are aligned BY ORDER (research emits trades
chronologically; the Rust engine closes them chronologically) — a trade-count
mismatch is itself reported as the first divergence signal.

READ-ONLY: never touches training/feature code, never writes to either source.
Output: `build/parity_diagnostic_report.md` + a console summary.

Runs fully ASYNCHRONOUSLY: both sources are loaded on worker threads and every
column comparison is dispatched as an asyncio task (asyncio.gather).

Usage:
    python -m research.parity_diagnostics
    python -m research.parity_diagnostics --rust-db build/diag_rocksdb
    python -m research.parity_diagnostics --rust-csv build/pnl_trades.csv
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.ml_shared_config import BUILD_DIR  # noqa: E402

# All inputs/outputs live under the shared build folder — never the repo root.
PY_TRADE_LOG = os.path.join(BUILD_DIR, "walk_forward_trade_log_v3.csv")
REPORT = os.path.join(BUILD_DIR, "parity_diagnostic_report.md")
DEFAULT_RUST_DB = os.path.join(BUILD_DIR, "diag_rocksdb")

# Similarity thresholds for the per-column verdict.
PASS_TOL = 0.999   # >= 99.9% of pairs agree  -> PASS  (~100%)
WARN_TOL = 0.90    # >= 90%  of pairs agree  -> WARN
# else -> FAIL

# Research STEP 5 R-clip floor: research clamps normalized_return dataset-wide
# with cap = max(|q(0.99)|, |q(0.01)|, RR_RATIO) BEFORE expectancy. The Rust
# engine streams trades one at a time and records the UNCLIPPED R per trade
# (`r_raw`), so parity reconstructs the same cap post-hoc from the replay's raw
# values (DUAL PASS) and clips before comparing. This must match
# config/strategy.py RR_RATIO / the rr_ratio handed to the live-test engine.
RR_RATIO = 10.0

# ----------------------------------------------------------------------------
# Source loaders (each runs on a worker thread via asyncio.to_thread)
# ----------------------------------------------------------------------------
def load_python_trades(path: str) -> pd.DataFrame:
    """Python txt: the walk-forward trade log (one row per OOS trade)."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def load_rust_trades_from_db(db_path: str) -> pd.DataFrame:
    """Rust txt: closed-trade JSON records from the live-test RocksDB `pnl` CF.

    Same extraction logic as research/extract_rocksdb.py (lossless, no
    rescaling): records carrying `entry_ms` are closed trades, `ts_ms` rows are
    equity snapshots and are skipped here.
    """
    from rocksdict import Rdict

    db = Rdict(db_path)
    try:
        cf = db.get_column_family("pnl")
        records = []
        for _k, v in cf.items():
            v_str = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
            try:
                rec = json.loads(v_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if "entry_ms" in rec:
                records.append(rec)
        return pd.DataFrame(records)
    finally:
        db.close()


def load_rust_trades_from_csv(path: str) -> pd.DataFrame:
    """Rust txt alternative: a `pnl_trades.csv` written by extract_rocksdb.py."""
    df = pd.read_csv(path)
    if "exit_ms_utc" in df.columns:  # extract_rocksdb exports derived UTC strings
        df["exit_ms"] = pd.to_datetime(df["exit_ms_utc"], utc=True).astype("int64") // 10**6
    return df


# ----------------------------------------------------------------------------
# Comparison primitives (pure functions — safe to run on any thread)
# ----------------------------------------------------------------------------
def _verdict(exact_frac: float) -> str:
    if exact_frac >= PASS_TOL:
        return "PASS"
    if exact_frac >= WARN_TOL:
        return "WARN"
    return "FAIL"


def compare_numeric(py: np.ndarray, rust: np.ndarray, label: str, tol: float) -> dict:
    """Column similarity for a numeric pair: % pairs within `tol`, mean/max
    absolute difference, Pearson correlation."""
    n = min(len(py), len(rust))
    if n == 0:
        return {"column": label, "n": 0, "exact_pct": 0.0, "verdict": "FAIL"}
    a, b = py[:n].astype(float), rust[:n].astype(float)
    diff = np.abs(a - b)
    exact = float(np.mean(diff <= tol))
    if n >= 2 and np.std(a) > 0 and np.std(b) > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = float("nan")
    return {
        "column": label,
        "n": n,
        "exact_pct": exact,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "corr": corr,
        "verdict": _verdict(exact),
    }


def compare_time_ms(py_ms: np.ndarray, rust_ms: np.ndarray, label: str,
                    tol_ms: float) -> dict:
    """Column similarity for a timestamp pair. Reports BOTH the raw agreement
    and the offset-corrected agreement: a constant clock offset (timezone or
    feed-start mismatch) shifts every pair equally, so the corrected number
    isolates per-trade divergence from a systematic shift. The median offset
    itself is surfaced in the report."""
    res = compare_numeric(py_ms, rust_ms, label, tol=tol_ms)
    n = res["n"]
    if n == 0:
        return res
    delta = py_ms[:n] - rust_ms[:n]
    offset = float(np.median(delta))
    corrected = compare_numeric(py_ms[:n] - offset, rust_ms[:n], label, tol=tol_ms)
    res.update({
        "median_offset_s": offset / 1000.0,
        "exact_pct_corrected": corrected["exact_pct"],
        "max_abs_diff_corrected_s": corrected.get("max_abs_diff", float("nan")) / 1000.0,
        # The verdict is based on the offset-CORRECTED agreement: a uniform
        # offset is a labeling/timezone issue, not an engine divergence.
        "verdict": corrected["verdict"],
    })
    return res


# ----------------------------------------------------------------------------
# Async orchestration
# ----------------------------------------------------------------------------
async def _load_python(path: str) -> pd.DataFrame:
    return await asyncio.to_thread(load_python_trades, path)


async def _load_rust(db_path: str | None, csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        return await asyncio.to_thread(load_rust_trades_from_csv, csv_path)
    return await asyncio.to_thread(load_rust_trades_from_db, db_path)


async def _compare_all_columns(py_df: pd.DataFrame, rust_df: pd.DataFrame,
                               tol_time_s: float, tol_ret: float,
                               rr_ratio: float = RR_RATIO) -> tuple[list[dict], dict | None]:
    """Dispatch every column comparison as a concurrent asyncio task.

    Returns (results, clip_info): `clip_info` is None when the Rust side has no
    `r_raw` column (legacy store); otherwise it documents the dual-pass
    research STEP 5 clip applied to the Rust R-multiples before comparison.
    """
    py_ms = (py_df["datetime"].astype("int64") // 10**6).to_numpy()
    rust_ms = rust_df["exit_ms"].to_numpy().astype(float)
    rust_ret = rust_df["r_multiple"].to_numpy().astype(float)

    # ---- dual-pass research-parity R clip (STEP 5) ---------------------------
    # The Rust engine cannot know research's dataset-wide clip cap while
    # streaming, so it persists the uncapped R (`r_raw`). Recompute the exact
    # same cap from those raw values now and clip — this is the second pass.
    clip_info: dict | None = None
    rust_ret_clipped = rust_ret
    if "r_raw" in rust_df.columns:
        raw = rust_df["r_raw"].to_numpy(dtype=float)  # NaN = legacy row
        have = raw[~np.isnan(raw)]
        if have.size:
            cap = max(abs(np.quantile(have, 0.99)),
                      abs(np.quantile(have, 0.01)),
                      rr_ratio)
            rust_ret_clipped = np.where(
                np.isnan(raw), rust_ret, np.clip(raw, -cap, cap))
            clip_info = {
                "cap": float(cap),
                "rr": rr_ratio,
                "n_raw": int(have.size),
                "n_clipped": int(np.sum(np.abs(have) > cap)),
            }

    rust_cum_clipped = np.cumsum(rust_ret_clipped)
    rust_cum = np.cumsum(rust_ret)  # unclipped, for the informational row

    tasks = [
        # exit timestamp: python `datetime` (entry bar) vs rust `exit_ms`
        asyncio.to_thread(
            compare_time_ms, py_ms, rust_ms, "exit_time", tol_time_s * 1000.0),
        # per-trade return in R: python `ret` (STEP 5-clipped) vs rust
        # `r_raw` clipped with the reconstructed dataset cap
        asyncio.to_thread(
            compare_numeric, py_df["ret"].to_numpy(), rust_ret_clipped,
            "return_r", tol_ret),
        # cumulative return in R: python `cum_pnl` vs running sum
        asyncio.to_thread(
            compare_numeric, py_df["cum_pnl"].to_numpy(), rust_cum_clipped,
            "cum_return_r", tol_ret * max(1, len(rust_ret_clipped))),
        # informational only: raw (uncapped) rust R vs python — expected to
        # diverge on timeout outliers exactly because research clips them
        asyncio.to_thread(
            compare_numeric, py_df["ret"].to_numpy(), rust_ret,
            "return_r_uncapped_info", tol_ret),
    ]
    results = list(await asyncio.gather(*tasks))
    results = sorted(results, key=lambda r: r["column"])
    return results, clip_info


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def run(py_csv: str, rust_db: str | None, rust_csv: str | None,
        tol_time_s: float, tol_ret: float, rr_ratio: float = RR_RATIO) -> int:
    async def _main() -> tuple[list[dict], dict, dict | None]:
        # Both sources load CONCURRENTLY on worker threads.
        py_df, rust_df = await asyncio.gather(
            _load_python(py_csv), _load_rust(rust_db, rust_csv))
        summary = {
            "py_trades": len(py_df),
            "rust_trades": len(rust_df),
            "count_match_pct": (min(len(py_df), len(rust_df))
                                / max(1, len(py_df), len(rust_df))),
        }
        cols, clip_info = await _compare_all_columns(
            py_df, rust_df, tol_time_s, tol_ret, rr_ratio)
        return cols, summary, clip_info

    cols, summary, clip_info = asyncio.run(_main())

    lines = []
    ok = True
    lines.append("# Replay Parity Diagnostic — Python research txt vs Rust live-test txt\n")
    lines.append(f"- Python txt (`{os.path.relpath(py_csv, PROJECT_ROOT)}`): {summary['py_trades']:,} trades")
    src = rust_csv or rust_db
    lines.append(f"- Rust txt   (`{os.path.relpath(src, PROJECT_ROOT)}`): {summary['rust_trades']:,} closed trades")
    lines.append(f"- Trade-count agreement: {_format_pct(summary['count_match_pct'])} "
                 f"({summary['py_trades']:,} vs {summary['rust_trades']:,})\n")
    ok &= summary["count_match_pct"] >= WARN_TOL

    if clip_info:
        lines.append(
            f"- Dual-pass R clip (research STEP 5): cap = ±{clip_info['cap']:.4f}R "
            f"(reconstructed from {clip_info['n_raw']:,} raw-R values, "
            f"{clip_info['n_clipped']} trade(s) clamped; RR_RATIO floor {clip_info['rr']:.1f}). "
            "`return_r` compares python's clipped ret against rust r_raw clipped "
            "with this cap; `return_r_uncapped_info` shows the raw comparison.\n")

    lines.append("## Per-column similarity\n")
    lines.append("| Column | Compared | Agreement | Mean abs diff | Max abs diff | Corr | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in cols:
        # `_info` rows are diagnostic-only (expected to diverge — e.g. the
        # uncapped rust R vs python's clipped ret) and never gate the verdict.
        if not c["column"].endswith("_info"):
            ok &= c["verdict"] != "FAIL"
        extra = ""
        if "exact_pct_corrected" in c:
            extra = (f" (offset-corrected: {_format_pct(c['exact_pct_corrected'])}, "
                     f"median offset {c['median_offset_s']:+.1f}s)")
        corr = c.get("corr", float("nan"))
        corr_s = "n/a" if corr != corr else f"{corr:.4f}"
        lines.append(
            f"| {c['column']} | {c['n']} | {_format_pct(c['exact_pct'])}{extra} "
            f"| {c.get('mean_abs_diff', float('nan')):.6g} "
            f"| {c.get('max_abs_diff', float('nan')):.6g} | {corr_s} | {c['verdict']} |"
        )
    lines.append("")
    lines.append("Agreement = fraction of aligned trade pairs equal within tolerance "
                 f"({tol_time_s:.0f}s for timestamps, {tol_ret:g}R for returns). "
                 "Time verdicts use the offset-corrected agreement; the raw number "
                 "and the median clock offset are shown next to them.\n")

    rust_only = ["side", "entry_price", "exit_price", "reason", "qty", "net_usd",
                 "equity_after", "duration_s", "barrier_pct", "target_price",
                 "stop_price"]
    lines.append("## Columns with no counterpart (informational, not compared)\n")
    lines.append("- Python-only: `threshold` (research-only sweep artifact)")
    lines.append(f"- Rust-only: {', '.join(f'`{c}`' for c in rust_only)} "
                 "(engine/execution fields the research pipeline never materializes)\n")

    overall = "PASS" if ok else "DIVERGENCE DETECTED"
    lines.append(f"## Overall verdict: **{overall}**\n")
    lines.append("Interpretation: a FAIL on `return_r`/`cum_return_r` with a PASS on "
                 "`exit_time` means the engines take the same trades but price/exit "
                 "differ (fees, barrier placement, model parity). A FAIL on "
                 "`exit_time` itself means the trade SEQUENCES diverge — check entry "
                 "gates and fold coverage first.\n")

    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # Console summary
    print("\n".join(lines))
    print(f"Report written to: {REPORT}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Column-by-column parity: research walk-forward trade log "
                    "vs the Rust live-test RocksDB trade log. Async."
    )
    ap.add_argument("--py-csv", default=PY_TRADE_LOG,
                    help="Python txt (walk_forward_trade_log_v3.csv)")
    ap.add_argument("--rust-db", default=DEFAULT_RUST_DB,
                    help="Rust txt source: live-test RocksDB store path")
    ap.add_argument("--rust-csv", default=None,
                    help="Rust txt alternative: pnl_trades.csv from extract_rocksdb.py")
    ap.add_argument("--tol-time-s", type=float, default=5.0,
                    help="Timestamp agreement tolerance in seconds (default 5)")
    ap.add_argument("--tol-ret", type=float, default=1e-6,
                    help="Per-trade return agreement tolerance in R (default 1e-6)")
    ap.add_argument("--rr-ratio", type=float, default=RR_RATIO,
                    help="RR_RATIO floor of the STEP 5 R-clip cap (default 10, "
                         "must match config/strategy.py)")
    args = ap.parse_args()
    sys.exit(run(args.py_csv, args.rust_db, args.rust_csv,
                 args.tol_time_s, args.tol_ret, args.rr_ratio))


if __name__ == "__main__":
    main()
