"""Empirical proof of the tb_* remake of moe/outcome/triple_barrier.py.

Sections:
  1. Legacy parity  -- the pre-remake implementation (git HEAD) vs the new
     one on identical data with identical legacy settings (classifier +
     regressor): trades and stats must match bit-exactly.
  2. Invariants     -- tp_size=2 -> +2R winners; barrier_size doubles widths.
  3. Interchangeability -- custom model / model_fn / tp_fn / loss_fn /
     rr_unit_fn / timeout_fn must CHANGE the behaviour, not be merely accepted.
  4. Timeout        -- exit at exactly entry+timeout (R formula), no entries
     in the last `timeout` bars, purge distance tracks the timeout, one
     resolved timeout drives purge + path_ok + simulation + labels.
  5. Pipeline       -- run_outcomes with legacy keys, new keys, a callable
     evaluator, and the untouched mode-fixed branch.

Run from the repo root:  python _tb_remake_proof.py
"""
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import strategy_system.strategy.moe.outcome.triple_barrier as tb  # noqa: E402
from strategy_system.strategy.moe.outcome import (  # noqa: E402
    classifier_model,
    evaluate_classifier,
    evaluate_regressor,
    loss_function,
    raw_fraction_unit,
    regressor_model,
    rr_unit,
    run_outcomes,
    tb_classifier,
    tb_regressor,
    timeout_function,
    tp_function,
)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def stats_exact(a: dict, b: dict, label: str):
    assert set(a) == set(b), f"{label}: stats keys differ: {set(a) ^ set(b)}"
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and np.isnan(va):
            ok = isinstance(vb, float) and np.isnan(vb)
        else:
            ok = va == vb
        assert ok, f"{label}: stats[{k!r}] differs: {va!r} != {vb!r}"


def make_df(n=900, seed=7):
    """Random-walk bars with a mildly predictive feature."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    shocks = rng.normal(0.0, 5e-4, n)
    price = 100.0 * np.exp(np.cumsum(shocks))
    mom = pd.Series(price).pct_change(5).fillna(0.0).to_numpy()
    df = pd.DataFrame(
        {
            "price_close": price,
            "side_signal": rng.choice([-1, 1], n),
            "feat_a": 0.6 * mom + 0.02 * rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "feat_c": np.sin(np.arange(n) / 25.0) + 0.1 * rng.normal(size=n),
        },
        index=idx,
    )
    return df


FCOLS = ["feat_a", "feat_b", "feat_c"]
df = make_df()

TB_SETTINGS = {
    "triple_barrier": {
        "side": "side_signal",
        "price_col": "price_close",
        "horizon": 25,
        "train_fraction": 0.70,
        "classifier": {"rr": 1.5, "barrier_size": 1.2, "vol_col": None,
                       "vol_window": 60, "threshold": None},
        "regressor": {"rr": 1.4, "barrier_size": 1.1},
    }
}


def load_old_module():
    """Extract the pre-remake triple_barrier.py from git HEAD and import it."""
    src = subprocess.check_output(
        ["git", "show", "HEAD:strategy_system/strategy/moe/outcome/triple_barrier.py"],
        cwd=ROOT,
    )
    path = ROOT / "_tb_old_tmp.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("_tb_old_tmp", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tb_old_tmp"] = mod  # dataclasses resolve names via sys.modules
    spec.loader.exec_module(mod)
    return mod


print("=" * 72)
print("PROOF 1: legacy parity (old implementation from git HEAD vs new)")
print("=" * 72)
old = load_old_module()
thr0_settings = {"triple_barrier": {**TB_SETTINGS["triple_barrier"],
                                    "classifier": {**TB_SETTINGS["triple_barrier"]["classifier"],
                                                   "threshold": 0.0}}}
parity_runs = [
    ("classifier", TB_SETTINGS, "parity[classifier] settings.yaml-like", False),
    ("classifier", thr0_settings, "parity[classifier] threshold=0 (non-empty trades)", True),
    ("regressor", TB_SETTINGS, "parity[regressor] settings.yaml-like", False),
]
for which, settings, label, require_trades in parity_runs:
    r_old = old.run_triple_barrier(
        df, FCOLS, getattr(old, f"evaluate_{which}"), settings=settings)
    r_new = tb.run_triple_barrier(
        df, FCOLS, getattr(tb, f"evaluate_{which}"), settings=settings)
    pd.testing.assert_frame_equal(r_old.trades, r_new.trades, check_exact=True)
    stats_exact(r_old.stats, r_new.stats, label)
    if require_trades:
        assert len(r_new.trades) > 0, f"{label}: expected non-empty trades"
    check(
        f"{label} trades+stats bit-exact",
        True,
        f"n_trades={r_new.stats['n_trades']} "
        f"expectancy_r={r_new.stats['expectancy_r']!r}",
    )

print()
print("=" * 72)
print("PROOF 2: invariants (tp_size -> R-multiple, barrier_size -> widths)")
print("=" * 72)


def cls(**kw):
    args = {"feature_cols": FCOLS, "side": "side_signal", "timeout": 20,
            "tp_size": 1.0, "barrier_size": 1.0, "threshold": 0.0, "rr_window": 60}
    model = kw.pop("model", None)
    args.update(kw)
    return tb.tb_classifier(df, model, **args)


def reg(**kw):
    args = {"feature_cols": FCOLS, "side": "side_signal", "timeout": 20,
            "barrier_size": 1.0, "rr_window": 60}
    model = kw.pop("model", None)
    args.update(kw)
    return tb.tb_regressor(df, model, **args)


base = cls()                      # threshold=0 -> trade every valid test row
two = cls(tp_size=2.0)            # 2R targets
check("threshold=0 trades every valid test row", len(base.trades) > 0,
      f"n_trades={len(base.trades)}")
check("tp_size=2 keeps the same traded rows", two.trades.index.equals(base.trades.index))
wins = two.trades["outcome"] == 1
losses = two.trades["outcome"] == -1
check("tp_size=2: winners pay exactly +2R", bool(wins.any())
      and bool(np.allclose(two.trades.loc[wins, "r"], 2.0, rtol=1e-12)),
      f"n_win={int(wins.sum())}")
check("losers still pay exactly -1R", bool(losses.any())
      and bool(np.allclose(two.trades.loc[losses, "r"], -1.0)))

bs2 = cls(barrier_size=2.0)
check("barrier_size=2 keeps the same traded rows", bs2.trades.index.equals(base.trades.index))
sl1 = base.trades["sl_width"].to_numpy()
sl2 = bs2.trades["sl_width"].to_numpy()
tp1 = base.trades["tp_width"].to_numpy()
tp2 = bs2.trades["tp_width"].to_numpy()
check("barrier_size=2 exactly doubles the stop width",
      bool(np.allclose(sl2, sl1 * 2.0, rtol=1e-12)))
check("barrier_size=2 exactly doubles the target width",
      bool(np.allclose(tp2, tp1 * 2.0, rtol=1e-12)))

print()
print("=" * 72)
print("PROOF 3: interchangeability (every seam changes behaviour)")
print("=" * 72)
b = cls()
m = cls(model=make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000)))
check("custom model changes P(win)",
      not np.allclose(b.trades["prob"].to_numpy(), m.trades["prob"].to_numpy()))
mf = cls(model_fn=lambda: DecisionTreeClassifier(max_depth=2, random_state=0))
check("custom model_fn is used (DecisionTree fitted)",
      isinstance(mf.model, DecisionTreeClassifier)
      and not np.allclose(b.trades["prob"].to_numpy(), mf.trades["prob"].to_numpy()))
half = cls(tp_fn=lambda sl, tp_size: sl * tp_size * 0.5)
hw = half.trades["outcome"] == 1
check("custom tp_fn changes the R-multiple (halved)",
      bool(hw.any()) and bool(np.allclose(half.trades.loc[hw, "rr"], 0.5, rtol=1e-12)))
dbl = cls(loss_fn=lambda u, bs: u * bs * 2.0)
check("custom loss_fn doubles the stop width",
      bool(np.allclose(dbl.trades["sl_width"].to_numpy(),
                       base.trades["sl_width"].to_numpy() * 2.0, rtol=1e-12)))
const = cls(rr_unit_fn=lambda frame, *, price_col, window=14, vol_col=None,
            sqrt_horizon=None: np.full(len(frame), 0.001))
check("custom rr_unit_fn sets a constant 0.001 risk unit",
      bool(np.allclose(const.trades["sl_width"].to_numpy(), 0.001, rtol=1e-12)))
quarter = cls(timeout=40, timeout_fn=lambda bars: bars // 4)
q_pos = df.index.get_indexer(quarter.trades.index)
q_to = quarter.trades["outcome"].to_numpy() == 0
p_arr = df["price_close"].to_numpy()
s_arr = df["side_signal"].to_numpy().astype(float)
q_exp = (s_arr[q_pos[q_to]] * (p_arr[q_pos[q_to] + 10] / p_arr[q_pos[q_to]] - 1.0)
         / quarter.trades["sl_width"].to_numpy()[q_to])
check("custom timeout_fn re-horizons the run (40//4=10, exit formula holds)",
      bool(q_to.any()) and bool(np.allclose(
          quarter.trades["r"].to_numpy()[q_to], q_exp, rtol=1e-10)))

rb = reg()
rridge = reg(model=make_pipeline(StandardScaler(), Ridge(alpha=100.0)))
common = rb.trades.index.intersection(rridge.trades.index)
check("regressor: custom model changes the barrier widths",
      len(common) > 0 and not np.allclose(
          rb.trades.loc[common, "sl_width"], rridge.trades.loc[common, "sl_width"]))
r3 = reg(loss_fn=lambda w, bs: w * bs * 3.0)
check("regressor: custom loss_fn triples the stop width",
      bool(np.allclose(r3.trades["sl_width"].to_numpy(),
                       rb.trades["sl_width"].to_numpy() * 3.0, rtol=1e-12)))
rconst = reg(rr_unit_fn=lambda frame, *, price_col, window=14, vol_col=None,
             sqrt_horizon=None: np.full(len(frame), 0.01))
rcommon = rb.trades.index.intersection(rconst.trades.index)
check("regressor: custom rr_unit_fn changes widths",
      len(rcommon) > 0
      and bool((rconst.trades.loc[rcommon, "sl_width"] > 0).all())
      and not np.allclose(rconst.trades.loc[rcommon, "sl_width"].to_numpy(),
                          rb.trades.loc[rcommon, "sl_width"].to_numpy()))
# factories still produce the documented defaults
check("model factories return the documented pipelines",
      type(classifier_model()).__name__ == "Pipeline"
      and type(regressor_model()).__name__ == "Pipeline"
      and type(raw_fraction_unit(df, price_col="price_close")).__module__ == "numpy")

print()
print("=" * 72)
print("PROOF 4: timeout drives purge + path_ok + simulation + labels")
print("=" * 72)
H = 20
res = reg(timeout=H)
tr = res.trades
pos = df.index.get_indexer(tr.index)
n = len(df)
check("no entry within the last `timeout` bars (path_ok boundary)",
      len(tr) > 0 and bool((pos + H < n).all()))
to = tr["outcome"].to_numpy() == 0
exp_r = (s_arr[pos[to]] * (p_arr[pos[to] + H] / p_arr[pos[to]] - 1.0)
         / tr["sl_width"].to_numpy()[to])
check("timeout trades exit at EXACTLY entry+timeout (R formula)",
      bool(to.any()) and bool(np.allclose(tr["r"].to_numpy()[to], exp_r, rtol=1e-10)),
      f"n_timeout={int(to.sum())}")

d10 = tb._build_barrier_data(df, FCOLS, side="side_signal", horizon=10,
                             price_col="price_close", high_col=None, low_col=None,
                             train_fraction=0.70)
d45 = tb._build_barrier_data(df, FCOLS, side="side_signal", horizon=45,
                             price_col="price_close", high_col=None, low_col=None,
                             train_fraction=0.70)
split = int(n * 0.70)
check("purge distance == timeout (train ends `timeout` bars before split)",
      int(d10.train.sum()) == split - 10 and int(d45.train.sum()) == split - 45
      and not np.array_equal(d10.train, d45.train))
check("path_ok excludes the last `timeout` rows",
      bool((~d45.path_ok[-45:]).all()) and bool(d45.path_ok[-46]))

cfg_t10 = {"triple_barrier": {
    "side": "side_signal", "price_col": "price_close", "horizon": 30,
    "train_fraction": 0.70,
    "classifier": {"rr": 1.5, "barrier_size": 1.2, "vol_window": 60,
                   "threshold": None, "timeout": 10},
}}
cfg_t30 = {"triple_barrier": {
    "side": "side_signal", "price_col": "price_close", "horizon": 30,
    "train_fraction": 0.70,
    "classifier": {"rr": 1.5, "barrier_size": 1.2, "vol_window": 60,
                   "threshold": None},
}}
e10 = tb.run_triple_barrier(df, FCOLS, tb.evaluate_classifier, settings=cfg_t10)
e30 = tb.run_triple_barrier(df, FCOLS, tb.evaluate_classifier, settings=cfg_t30)
check("settings timeout overrides shared horizon in run_triple_barrier",
      e10.stats["n_eligible"] > e30.stats["n_eligible"],
      f"n_eligible(H=10)={e10.stats['n_eligible']} > n_eligible(H=30)={e30.stats['n_eligible']}")

# A prebuilt BarrierData must not silently disagree with the timeout.
try:
    tb.tb_classifier(d10, timeout=30)
    check("timeout/BarrierData mismatch raises", False)
except ValueError:
    check("timeout/BarrierData mismatch raises", True)
try:
    tb.tb_classifier(d10, side="side_signal")
    check("build args with a prebuilt BarrierData raise", False)
except ValueError:
    check("build args with a prebuilt BarrierData raise", True)
try:
    timeout_function(0)
    check("timeout_function rejects non-positive bars", False)
except ValueError:
    check("timeout_function rejects non-positive bars", True)
try:
    timeout_function(True)
    check("timeout_function rejects booleans", False)
except ValueError:
    check("timeout_function rejects booleans", True)

print()
print("=" * 72)
print("PROOF 5: run_outcomes pipeline (legacy keys, new keys, mode-fixed)")
print("=" * 72)


def make_frames(n=1200, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=n, freq="1min", tz="UTC")
    # Noisier than PROOF 1-4 data so 2R barriers at H=12 actually get hit.
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 2e-3, n)))
    frame = pd.DataFrame(
        {
            "price_close": price,
            "htf_rsi_d2_signal": rng.choice([-1, 0, 1], n, p=[0.45, 0.10, 0.45]),
            "latent_id": np.where(np.arange(n) < n // 2, 0, 1).astype(int),
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
        },
        index=idx,
    )
    targets = pd.DataFrame(index=idx)
    for k in (1, 2):
        endpoints = np.minimum(np.arange(n) + k, n - 1)
        targets[f"future_return__{k}"] = price[endpoints] / price - 1.0
    # S1 contract: targets live alongside in the model frame (never live inputs;
    # resolve_feature_cols excludes future_*).
    return frame.join(targets), targets


def outcomes_cfg(tb_section):
    return {"outcome": {
        "model": {"train_fraction": 0.70, "purge_rows": 0, "ridge": 1e-6,
                  "min_state_rows": 30, "side": 1,
                  "tree_max_depth": 3, "tree_min_samples_leaf": 5,
                  "tree_min_train_rows": 10 ** 9,  # keep the metalabel tree off
                  "seed": 7},
        "triple_barrier": tb_section,
    }}


frame, targets = make_frames()

LEG = {"side": "htf_rsi_d2_signal", "price_col": "price_close", "horizon": 20,
       "train_fraction": 0.70,
       "classifier": {"rr": 1.5, "barrier_size": 1.0, "vol_col": None,
                      "vol_window": 60, "threshold": 0.0},
       "regressor": {"rr": 1.4, "barrier_size": 1.0}}
out1 = run_outcomes(frame, targets, outcomes_cfg(LEG))
check("pipeline: legacy classifier keys",
      len(out1.events) > 0
      and {"return", "return_rr", "barrier", "t0", "t1", "side"} <= set(out1.events.columns),
      f"n_events={len(out1.events)}")

LEG_REG = dict(LEG, evaluator="regressor")
out2 = run_outcomes(frame, targets, outcomes_cfg(LEG_REG))
check("pipeline: legacy regressor keys (evaluator: regressor)",
      len(out2.events) > 0, f"n_events={len(out2.events)}")

NEW = {"side": "htf_rsi_d2_signal", "price_col": "price_close", "horizon": 20,
       "train_fraction": 0.70,
       # barrier_size=0.3 keeps tp = 2R within reach of the synthetic path,
       # so the +2R winner invariant below is actually exercised.
       "classifier": {"tp_size": 2, "barrier_size": 0.3, "timeout": 12,
                      "rr_window": 40, "threshold": 0.0}}
out3 = run_outcomes(frame, targets, outcomes_cfg(NEW))
ev = out3.events
w = ev["outcome"] == 1
check("pipeline: new keys (tp_size/timeout/rr_window)",
      len(ev) > 0 and bool(w.any())
      and bool(np.allclose(ev.loc[w, "return"], 2.0, rtol=1e-12)),
      f"n_events={len(ev)} n_win={int(w.sum())}")

NEW_REG = {"side": "htf_rsi_d2_signal", "price_col": "price_close", "horizon": 20,
           "train_fraction": 0.70,
           "evaluator": tb_regressor,  # callable -> new RR-unit regressor
           "regressor": {"barrier_size": 1.0, "timeout": 12, "rr_window": 40}}
out4 = run_outcomes(frame, targets, outcomes_cfg(NEW_REG))
check("pipeline: callable evaluator (tb_regressor, new RR-unit semantics)",
      len(out4.events) > 0, f"n_events={len(out4.events)}")

FIX = {"mode": "fixed",
       "fixed": {"signal_col": "htf_rsi_d2_signal", "price_col": "price_close",
                 "pt": 0.002, "sl": 0.001, "horizon_bars": 10}}
out5 = run_outcomes(frame, targets, outcomes_cfg(FIX))
check("pipeline: untouched mode-fixed branch still works",
      len(out5.events) > 0, f"n_events={len(out5.events)}")

print()
print("=" * 72)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
print("=" * 72)

# Cleanup temp artifacts.
if (ROOT / "_tb_old_tmp.py").exists():
    (ROOT / "_tb_old_tmp.py").unlink()
