"""meta_labeller.py — ONE take/avoid labeller for the ENTIRE portfolio.

Executor-style mechanism, exactly mirroring how the production LightGBM
executor model works: the executor is trained on BTCUSDT walk-forward data
only (no symbol-identity features) and scores all three portfolio assets.
This labeller does the same:

  1. trains ONE LightGBM "labeller" on TRAIN_SYMBOL's walk-forward trade log
     (the executor's REAL, already-OOS taken-trade universe at confidence
     threshold THRESHOLD) over 16 entry-time features (the same 15 the live
     engine computes + UTC hour-of-day; order pinned below so the Rust engine
     replicates the vector 1:1 and loads the export with the SAME `LightGbm`
     parser live-trade uses),
  2. tunes the take/avoid threshold on the TRAIN slice only,
  3. evaluates on the train symbol's chronological holdout AND — when their
     logs exist — transfers the SAME model to every other portfolio asset:
     per-asset filtered-vs-unfiltered improvement is how the one model
     "observes each crypto's respective problems" (the features are
     asset-relative z-scores; there is deliberately NO symbol-identity
     feature, exactly like the executor),
  4. exports ONE model — uets_rs/models/lgb_labeller.txt +
     .features.json + build/meta_labeller_manifest.json ("portfolio" entry) —
     loaded by EVERY asset's live process.

Live semantics: a missing/un-deployed labeller = gate OPEN (executor
unchanged). The labeller only ever FURTHER filters the executor's candidates.

Run:  python -m research.ofmr.meta_labeller [--train-symbol btcusdt]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json as _json  # noqa: E402

with open(os.path.join(PROJECT_ROOT, "config", "platform.json"),
          "r", encoding="utf-8-sig") as _pf:
    BUILD_DIR = os.path.join(
        PROJECT_ROOT, _json.load(_pf)["build_dir"]
    )  # noqa: E402

# ---------------------------------------------------------------------------
# Study config
# ---------------------------------------------------------------------------
THRESHOLD = 0.20  # executor confidence threshold stream (matches regime_data)

# ONE model, portfolio-wide: trained on TRAIN_SYMBOL only (executor-style),
# transferred to every other asset in EVAL_SYMBOLS for evaluation.
TRAIN_SYMBOL = "btcusdt"
EVAL_SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]  # portfolio.json assets

MODEL_DIR = os.path.join(PROJECT_ROOT, "uets_rs", "models")
LABELLER_TXT = "lgb_labeller.txt"
MANIFEST_PATH = os.path.join(BUILD_DIR, "meta_labeller_manifest.json")
REPORT_PATH = os.path.join(BUILD_DIR, "meta_labeller_report.md")

# Per-symbol walk-forward trade-log locations. v3 is the btcusdt log; other
# symbols get their own `<symbol>` variant when produced.
DEFAULT_LOGS = {TRAIN_SYMBOL: os.path.join(BUILD_DIR, "walk_forward_trade_log_v3.csv")}
for _sym in EVAL_SYMBOLS:
    DEFAULT_LOGS.setdefault(
        _sym, os.path.join(BUILD_DIR, f"walk_forward_trade_log_v3_{_sym}.csv")
    )

# ---------------------------------------------------------------------------
# Labeller feature vector — ORDER IS PART OF THE CONTRACT
# ---------------------------------------------------------------------------
# Every name is an `_open` (entry-time, point-in-time) column in the
# walk-forward trade log. The Rust engine builds the SAME 16 values in the
# SAME order inside `compute_features` (engine/src/lib.rs `meta_features`
# block), so live scoring needs no feature-name lookup. NOTE: no
# symbol-identity feature — the executor has none either; each asset is
# observed through its own asset-relative z-scores.
LABELLER_FEATURES: list[str] = [
    "spread_bps_z_open", "ma_slope_5_open", "rsi_x_ma_open", "ma_accel_open",
    "price_roc_3_open", "rsi_2_open", "sma_14_open", "ofi_z_open",
    "cvd_delta_z_open", "cvd_slope_z_open", "absorption_z_open",
    "tick_autocorr_open", "inst_ratio_open", "delta_divergence_z_open",
    "depth_imbalance_z_open", "hour_of_day",
]
META_FEATURE_COUNT = len(LABELLER_FEATURES)

# Labeller hyper-params mirror the executor (same model class, same
# regularization reflex) — a small, heavily-regularized booster.
LABELLER_PARAMS = dict(
    n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
)

# Take/avoid threshold grid — swept on TRAIN ONLY, never the holdout.
THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]

TRAIN_FRAC = 0.60      # chronological train slice (threshold tuning)
MIN_FILTER_N = 20      # minimum labelled trades before a gate is meaningful

# ---------------------------------------------------------------------------
# Loading + features
# ---------------------------------------------------------------------------


def load_symbol_log(symbol: str, path: str | None = None) -> pd.DataFrame | None:
    """Load <symbol>'s walk-forward trade log at THRESHOLD, chronological."""
    path = path or DEFAULT_LOGS.get(symbol)
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["threshold"] == THRESHOLD].copy()
    if not len(df):
        return None
    df["dt"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    df["hour_of_day"] = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    df["year_month"] = df["dt"].dt.strftime("%Y-%m")
    df["is_win"] = (df["ret"] > 0).astype(np.int8)
    return df


def meta_matrix(df: pd.DataFrame) -> np.ndarray:
    """16-dim labeller input (entry-time only), NaN -> 0 (engine-computable)."""
    x = df[LABELLER_FEATURES].to_numpy(dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Training + threshold tuning
# ---------------------------------------------------------------------------


def train_labeller(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(**LABELLER_PARAMS)
    clf.fit(X, y)
    return clf


def best_threshold(clf, Xtr: np.ndarray, ytr: np.ndarray,
                   probs: np.ndarray) -> tuple[float, float]:
    """Pick threshold maximizing profit factor on TRAIN only (n>=MIN_FILTER_N)."""
    best_t, best_pf = 0.5, -1.0
    for t in THRESHOLD_GRID:
        keep = probs >= t
        n = int(keep.sum())
        if n < MIN_FILTER_N:
            continue
        wins = int((ytr[keep] > 0).sum())
        losses = n - wins
        pf = (wins / max(losses, 1)) if losses else float("inf")
        if pf > best_pf:
            best_t, best_pf = t, pf
    return best_t, best_pf


# ---------------------------------------------------------------------------
# Evaluation — the ONE trained model measured per asset
# ---------------------------------------------------------------------------


def evaluate_stream(symbol: str, role: str, df: pd.DataFrame, clf, thr: float,
                    log_path: str) -> dict:
    """Filtered-vs-unfiltered stats of an ALREADY-TRAINED labeller on one
    asset's trade stream (`role` = 'train' holdout or 'transfer' full stream)."""
    from research.ofmr.regime_data import ret_metrics

    X = meta_matrix(df)
    probs = clf.predict_proba(X)[:, 1]
    lab = pd.Series(probs >= thr, index=df.index)

    base = ret_metrics(df["ret"].to_numpy())
    kept = df[lab]
    filt = (ret_metrics(kept["ret"].to_numpy()) if len(kept) else
            {"n": 0, "win_rate": 0.0, "mean_r": 0.0, "pf": 0.0, "total_r": 0.0})

    mo = []
    for p, g in df.groupby("year_month"):
        gm = g[lab[g.index]]
        em = ret_metrics(g["ret"].to_numpy())
        fm = (ret_metrics(gm["ret"].to_numpy()) if len(gm) else
              {"n": 0, "win_rate": 0.0, "mean_r": 0.0})
        mo.append({"year_month": p, "exec_n": em["n"], "exec_mean": em["mean_r"],
                   "gated_n": fm["n"], "gated_mean": fm["mean_r"],
                   "gated_wr": fm["win_rate"]})

    # Same deployment rule as before: material lift AND no month where the
    # gate turns a profitable executor month negative.
    month_sign_ok = all((m["gated_n"] == 0) or (m["gated_mean"] >= 0)
                        or (m["exec_mean"] <= 0) for m in mo)
    is_gate = bool(filt["n"] >= MIN_FILTER_N and filt["mean_r"] > base["mean_r"]
                   and filt["win_rate"] > base["win_rate"] and month_sign_ok)
    # A transfer asset is 'harmful' when the gate would flip a profitable
    # executor stream negative there — that blocks portfolio-wide deployment.
    harmful = bool(filt["n"] >= MIN_FILTER_N and base["mean_r"] > 0
                   and filt["mean_r"] < 0)

    return {"symbol": symbol, "role": role, "log": log_path, "is_gate": is_gate,
            "harmful": harmful, "n_total": int(len(df)),
            "months": int(df["year_month"].nunique()),
            "dt_min": str(df["dt"].min()), "dt_max": str(df["dt"].max()),
            "threshold": float(thr), "base": base, "filt": filt,
            "monthly": mo}


def run(train_symbol: str, log_override: str | None = None) -> int:
    console: list[str] = []
    train_log = log_override or DEFAULT_LOGS.get(train_symbol)
    df_tr = load_symbol_log(train_symbol, train_log)
    if df_tr is None:
        console.append(f"  {train_symbol}: no walk-forward trade log -> gate OPEN")
        write_report([{"symbol": train_symbol, "role": "train", "log": None}],
                     console)
        return 0

    X, y = meta_matrix(df_tr), df_tr["is_win"].to_numpy()
    n = len(df_tr)
    cut = int(n * TRAIN_FRAC)
    clf_train = train_labeller(X[:cut], y[:cut])
    thr, _pf = best_threshold(clf_train, X[:cut], y[:cut],
                              clf_train.predict_proba(X[:cut])[:, 1])

    # 1) train symbol: chronological holdout (the honest OOS slice).
    results = [evaluate_stream(train_symbol, "train", df_tr.iloc[cut:],
                               clf_train, thr,
                               train_log or DEFAULT_LOGS[train_symbol])]
    # 2) transfer: the SAME model on every other portfolio asset's stream
    #    (all of it is OOS with respect to the train symbol's data).
    for sym in EVAL_SYMBOLS:
        if sym == train_symbol:
            continue
        dfe = load_symbol_log(sym, DEFAULT_LOGS.get(sym))
        if dfe is None:
            results.append({"symbol": sym, "role": "transfer", "log": None})
            console.append(f"  {sym}: no log -> transfer evaluation skipped")
            continue
        results.append(evaluate_stream(sym, "transfer", dfe, clf_train, thr,
                                       DEFAULT_LOGS[sym]))

    train_res = results[0]
    deploy_ok = bool(train_res.get("is_gate")) and not any(
        r.get("harmful", False) for r in results[1:])

    # Final LIVE model: fit on ALL train-symbol trades, export with the
    # train-picked threshold. ONE file for the whole portfolio.
    clf_all = train_labeller(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    txt_path = os.path.join(MODEL_DIR, LABELLER_TXT)
    feat_path = os.path.join(MODEL_DIR, "lgb_labeller.features.json")
    clf_all.booster_.save_model(txt_path)
    with open(feat_path, "w", encoding="utf-8") as fh:
        json.dump({"train_symbol": train_symbol, "features": LABELLER_FEATURES,
                   "includes_orderflow": True, "version": 1,
                   "model": LABELLER_TXT,
                   "meta_threshold": round(float(thr), 3), "n_trades": n,
                   "trained_at": datetime.now(timezone.utc).isoformat()},
                  fh, indent=2)

    man: dict = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as fh:
                man = json.load(fh)
        except (json.JSONDecodeError, OSError):
            man = {}
    # One portfolio entry; retire any legacy per-symbol labeller entries.
    for sym in EVAL_SYMBOLS:
        man.pop(sym, None)
    man["meta_threshold"] = round(float(thr), 3)
    man["portfolio"] = {"model": LABELLER_TXT,
                        "features": LABELLER_FEATURES,
                        "meta_threshold": round(float(thr), 3),
                        "train_symbol": train_symbol, "n_trades": n,
                        "gate_deployed": deploy_ok,
                        "trained_at": datetime.now(timezone.utc).isoformat()}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)

    for r in results:
        if r.get("log") is None:
            continue
        d = r["filt"]["mean_r"] - r["base"]["mean_r"] if r["filt"]["n"] else 0.0
        role = "train-holdout" if r["role"] == "train" else "transfer"
        console.append(
            f"  {r['symbol']} ({role}): {r['n_total']} trades | "
            f"{r['base']['n']}->{r['filt']['n']} "
            f"| meanR {r['base']['mean_r']:+.3f}->{r['filt']['mean_r']:+.3f} "
            f"(d={d:+.3f}) | thr={r['threshold']:.2f}")
    console.append(f"  deployment: gate_deployed={deploy_ok} "
                   f"(train verdict + no transfer harm) -> {LABELLER_TXT}")
    write_report(results, console, deploy_ok, train_symbol)
    return 0


def _fmt(m) -> str:
    from research.ofmr.regime_data import fmt_metrics
    return fmt_metrics(m)


def write_report(results: list[dict], console: list[str], deploy_ok: bool,
                 train_symbol: str) -> None:
    R = []
    A = R.append
    A("# Meta-Labeller Report — ONE portfolio-wide take/avoid gate over ml_state_optimized")
    A("")
    A(f"- generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    A(f"- ONE model for the ENTIRE portfolio, executor-style: trained on "
      f"{train_symbol.upper()} only (no symbol-identity features), deployed to "
      f"every asset; each asset is observed through its own asset-relative features.")
    A("- addition-only: the primary walk-forward (ml_state_optimized) is untouched;")
    A("  this report evaluates 'executor alone' vs 'executor + labeller gate'.")
    A("- leakage contract: labels come from the executor's own OOS trades;")
    A("  threshold tuned on train slice only; holdout = strictly later trades.")
    A("")
    for r in results:
        sym = r["symbol"].upper()
        if r.get("log") is None:
            A(f"## {sym}")
            A("- no walk-forward trade log found -> transfer evaluation skipped;")
            A("  the live gate still applies once deployed (missing file = OPEN).")
            A("")
            continue
        role = r.get("role", "train")
        if role == "train":
            A(f"## {sym} — TRAIN SYMBOL (chronological holdout)")
        else:
            A(f"## {sym} — transfer evaluation (full stream is OOS wrt training)")
        A(f"- universe: {r['n_total']} executor trades @ threshold 0.20 "
          f"({r['months']} months, {r['dt_min']} .. {r['dt_max']})")
        A(f"- executor={r['base']['n']} -> gated={r['filt']['n']} "
          f"({100 * r['filt']['n'] / max(r['base']['n'], 1):.1f}% kept)")
        A("")
        A("  | metric | executor alone | + labeller gate | delta |")
        A("  |---|---:|---:|---:|")
        b, f = r["base"], r["filt"]
        if f["n"] == 0:
            A("  | (gate kept 0 trades) | | | |")
        else:
            A(f"  | n | {b['n']} | {f['n']} | {f['n'] - b['n']:+d} |")
            A(f"  | win rate | {b['win_rate']:.1%} | {f['win_rate']:.1%} "
              f"| {f['win_rate'] - b['win_rate']:+.1%} |")
            A(f"  | mean R | {b['mean_r']:+.3f} | {f['mean_r']:+.3f} "
              f"| {f['mean_r'] - b['mean_r']:+.3f} |")
            A(f"  | profit factor | {b['pf']:.3f} | {f['pf']:.3f} "
              f"| {f['pf'] - b['pf']:+.3f} |")
            A(f"  | total R | {b['total_r']:+.1f} | {f['total_r']:+.1f} "
              f"| {f['total_r'] - b['total_r']:+.1f} |")
            A(f"  | threshold (train-picked) | - | {r['threshold']:.2f} | - |")
        A("")
        A("  monthly stability (gated vs executor):")
        A("  | month | exec n | exec meanR | gated n | gated meanR | gated WR |")
        A("  |---|---:|---:|---:|---:|---:|")
        for m in r["monthly"]:
            A(f"  | {m['year_month']} | {m['exec_n']} | {m['exec_mean']:+.3f} "
              f"| {m['gated_n']} | {m['gated_mean']:+.3f} | {m['gated_wr']:.1%} |")
        A("")
        if role == "train":
            if r["is_gate"]:
                A("- verdict: **GATE** (material + month-consistent lift).")
            else:
                A("- verdict: no material/stable lift on holdout — labeller still")
                A("  exported, but this blocks `gate_deployed` until it earns it.")
        else:
            if r.get("harmful"):
                A("- verdict: **HARMFUL on this asset** (gate flips a profitable")
                A("  executor stream negative) — blocks portfolio-wide deployment.")
            else:
                A("- verdict: no harm detected on this asset.")
        A("")
    A("- deployment: `gate_deployed=" + str(deploy_ok).lower() + "` — train-symbol")
    A("  verdict AND no transfer harm -> build/meta_labeller_manifest.json 'portfolio'.")
    A("")
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(R))
    print(f"[meta-labeller] report -> {REPORT_PATH}")
    for ln in console:
        print(ln)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="ONE portfolio-wide meta-labeller (addition-only)")
    ap.add_argument("--train-symbol", default=TRAIN_SYMBOL,
                    help="symbol whose walk-forward log trains the labeller")
    ap.add_argument("--log", default=None,
                    help="override the train-symbol trade-log path")
    args = ap.parse_args(argv)
    return run(args.train_symbol.strip().lower(), args.log)


if __name__ == "__main__":
    sys.exit(main())
