"""Verification tests for trade-level Sortino/Sharpe, restart-safe MDD,
and the clock-offset guard. Run: python _test_stats.py"""

from collections import deque
import sys

sys.path.insert(0, ".")

from vim_dashboard import _compute_sortino, _compute_sharpe  # noqa: E402
from portfolio_supervisor import PortfolioSupervisor, AssetHandle  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---- 1. Trade-level Sortino / Sharpe --------------------------------------
print("[1] trade-level Sortino/Sharpe")
# Known series: +2R, -1R, +3R, -2R  -> mean=0.5
rs = [2.0, -1.0, 3.0, -2.0]
n = len(rs)
mean = sum(rs) / n
dm = sum(min(0.0, r) for r in rs) / n
dd = (sum((min(0.0, r) - dm) ** 2 for r in rs) / (n - 1)) ** 0.5
sd = (sum((r - mean) ** 2 for r in rs) / (n - 1)) ** 0.5
check("sortino matches hand-computed trade-level value",
      abs(_compute_sortino(rs) - mean / dd) < 1e-12,
      f"got {_compute_sortino(rs)} want {mean/dd}")
check("sharpe matches hand-computed trade-level value",
      abs(_compute_sharpe(rs) - mean / sd) < 1e-12,
      f"got {_compute_sharpe(rs)} want {mean/sd}")
check("no annualization applied (sortino == mean/dd exactly)",
      _compute_sortino(rs) == mean / dd)
check("n < 2 -> N/A (None), not 0.0", _compute_sortino([1.0]) is None)
check("empty -> N/A (None)", _compute_sortino([]) is None)
check("all-winners (no downside obs) -> N/A (None), not 0.0",
      _compute_sortino([1.0, 2.0, 0.5]) is None)
check("varied all-winners: sharpe IS defined (full sd, unlike sortino)",
      _compute_sharpe([1.0, 2.0, 0.5]) is not None)
check("sharpe n < 2 -> N/A (None)", _compute_sharpe([1.0]) is None)
check("constant series (sd=0) -> N/A", _compute_sharpe([1.0, 1.0]) is None)

# ---- 2. Restart-safe MDD ----------------------------------------------------
print("[2] restart-safe MDD")
sup = PortfolioSupervisor([], restart=False, dashboard=False)


def mdd_of(vals):
    return PortfolioSupervisor._mdd_from_series(vals)


# Restart artifact: +5R then child restart resets to 0 then rebuilds to +1R
curve = [5.0, 5.0, 5.0, None, 0.0, 0.5, 1.0]
check("restart split: no phantom 5R drawdown", mdd_of(curve) == 0.0,
      f"got {mdd_of(curve)}")
# Same curve WITHOUT the break marker would (correctly) show the artifact —
# guard against the marker being dropped by future edits:
check("without marker the collapse WOULD be a 5R drawdown (sanity)",
      mdd_of([5.0, 5.0, 5.0, 0.0, 0.5, 1.0]) == 5.0)
# Real drawdown inside a segment is still measured:
check("real drawdown within a segment still measured",
      mdd_of([2.0, 1.0, 3.0, None, 4.0, 3.0]) == 1.0, f"got {mdd_of([2.0, 1.0, 3.0, None, 4.0, 3.0])}")

# Full pipeline through _record_metrics with real AssetHandles
specs = [{"ticker": "ethusdt", "id": "ethusdt"}, {"ticker": "ethusdt", "id": "eth-raw"}]
assets = [AssetHandle(s, {}, "bin") for s in specs]
sup2 = PortfolioSupervisor(assets, restart=False, dashboard=False)


def sim_phase(idx, total_r, n_trades, pending):
    """Only asset `idx` trades; the twin stays live at 0 (realistic twin A/B)."""
    for i, a in enumerate(assets):
        if i == idx:
            a.total_r = total_r
            a.n_trades = n_trades
            a._pending_trade_rs = list(pending)
        else:
            a.total_r = 0.0
            a.n_trades = 0
            a._pending_trade_rs = []


# Phase A: 5 losses of -1R then the child restarts
sim_phase(0, -5.0, 5, [-1.0] * 5)
sup2._record_metrics()
sim_phase(0, -5.0, 5, [])
sup2._record_metrics()
# Phase B: restart -> counters reset
sim_phase(0, 0.0, 0, [])
sup2._record_metrics()
# Phase C: rebuild to +1R
sim_phase(0, 1.0, 1, [1.0])
sup2._record_metrics()

check("trade_rs collected across drain (5 losses + 1 win)",
      len(sup2.trade_rs) == 6, f"got {list(sup2.trade_rs)}")
check("pending drained to empty", all(len(a._pending_trade_rs) == 0 for a in assets))
check("restart did NOT create phantom 5R MDD (portfolio)",
      sup2.mdd_r() == 0.0, f"got {sup2.mdd_r()}")
check("per-asset MDD also restart-safe", assets[0] is not None and
      sup2.asset_mdd_r("ethusdt") == 0.0, f"got {sup2.asset_mdd_r('ethusdt')}")
check("trade-close resolution captured the -5R trough pre-restart (segment split only at marker)",
      -5.0 in sup2.r_hist and None in sup2.r_hist)
# Now a real losing streak WITHOUT restart: trough must be measured
sim_phase(0, -3.0, 3, [-1.0, -1.0, -1.0])
sup2._record_metrics()
sim_phase(0, 0.0, 4, [3.0])
sup2._record_metrics()
check("real drawdown measured at trade-close resolution",
      sup2.mdd_r() == 4.0, f"got {sup2.mdd_r()}")  # peak +1R -> trough -3R = 4R
# equity MDD with breaks
sup3 = PortfolioSupervisor([], restart=False, dashboard=False)
sup3.eq_hist.extend([10000.0, 11000.0, 10500.0, None, 10000.0, 10200.0])
check("equity MDD ignores restart break (10k->11k->10.5k = 4.55%, second seg 0)",
      abs(sup3.mdd_pct() - (11000.0 - 10500.0) / 11000.0) < 1e-12,
      f"got {sup3.mdd_pct()}")

# ---- 3. Clock-offset guard ---------------------------------------------------
print("[3] clock-offset guard")
sup4 = PortfolioSupervisor(assets, restart=False, dashboard=False)
assets[0].skew_ms = 1_400_000.0   # ~23 min off — the garbage you saw
assets[0].lat_ms = 0.4
assets[1].skew_ms = 2500.0        # sane
sup4._record_metrics()
sk0 = list(sup4.net_hist_by_asset["ethusdt"]["sk"])
sk1 = list(sup4.net_hist_by_asset["eth-raw"]["sk"])
check("23-minute offset sample discarded", sk0 == [], f"got {sk0}")
check("sane offset sample kept", sk1 == [2500.0])
check("discard counted, not silent",
      sup4.sk_discarded_by_asset.get("ethusdt", 0) == 1)
assets[0].skew_ms = 1300.0
sup4._record_metrics()
check("recovered clock resumes recording",
      list(sup4.net_hist_by_asset["ethusdt"]["sk"]) == [1300.0])

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
