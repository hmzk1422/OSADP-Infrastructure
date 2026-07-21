"""
Quick validation tests for the live execution pipeline.
Run: python -m main.test_pipeline
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.execution import PositionTracker, QuantSimulator
from main.features import LiveFeatureEngine
from main.config import *

print("=" * 60)
print("PIPELINE VALIDATION TESTS")
print("=" * 60)

# ============================================================
# Test 1: PositionTracker barrier calculation
# ============================================================
print("\n[Test 1] PositionTracker — Barrier Calculation")
tracker = PositionTracker()
for _ in range(100):
    tracker.update_volatility(60000 + np.random.randn() * 100)

bp = tracker.get_barrier_pct()
print(f"  Barrier pct: {bp*100:.4f}%")
assert bp > 0, "Barrier pct should be positive"
tracker.enter_short(60000.0, 0.01)
print(f"  Entry: {tracker.entry_price:.2f}")
print(f"  Stop barrier: {tracker.stop_barrier:.2f}")
print(f"  Profit barrier: {tracker.profit_barrier:.2f}")
assert tracker.active == True
print("  PASSED ✓")

# ============================================================
# Test 2: PositionTracker exit scenarios
# ============================================================
print("\n[Test 2] PositionTracker — Exit Scenarios")

# Exit via profit (price goes down for short) — use smaller vol data
tracker2 = PositionTracker()
for _ in range(100):
    tracker2.update_volatility(60000 + np.random.randn() * 5)  # 5x smaller noise
tracker2.enter_short(60000.0, 0.01)
# Simulate price going down hard (hits profit barrier)
tracker2.update_tick(59900.0)
tracker2.update_tick(59700.0)
exit1 = tracker2.check_exit(59700.0)
assert exit1 is not None and exit1['reason'] == 'take_profit', f"Expected take_profit, got {exit1}"
print(f"  Profit exit: {exit1['reason']} | PnL: {exit1['gross_pnl_pct']*100:.3f}% ✓")

# Exit via stop (price goes up for short) — use smaller vol data
tracker3 = PositionTracker()
for _ in range(100):
    tracker3.update_volatility(60000 + np.random.randn() * 5)  # 5x smaller noise
tracker3.enter_short(60000.0, 0.01)
# Simulate price going up hard (hits stop barrier)
tracker3.update_tick(60100.0)
tracker3.update_tick(60200.0)
exit2 = tracker3.check_exit(60200.0)
assert exit2 is not None and exit2['reason'] == 'stop_loss', f"Expected stop_loss, got {exit2}"
print(f"  Stop exit: {exit2['reason']} | PnL: {exit2['gross_pnl_pct']*100:.3f}% ✓")

# Exit via timeout — set entry time far in the past to force timeout
from datetime import datetime, timedelta
tracker4 = PositionTracker()
for _ in range(100):
    tracker4.update_volatility(60000 + np.random.randn() * 100)
past_timestamp = datetime.now() - timedelta(seconds=LOOKAHEAD_SECONDS + 10)
tracker4.enter_short(60000.0, 0.01, timestamp=past_timestamp)
# Price hasn't moved much, but timeout should fire
exit3 = tracker4.check_exit(60001.0)
assert exit3 is not None and exit3['reason'] == 'timeout', f"Expected timeout, got {exit3}"
print(f"  Timeout exit: {exit3['reason']} | PnL: {exit3['gross_pnl_pct']*100:.3f}% ✓")

print("  PASSED ✓")

# ============================================================
# Test 3: QuantSimulator dry run
# ============================================================
print("\n[Test 3] QuantSimulator — Professional Dry Run Mode")
sim = QuantSimulator(quick_init=True)
assert sim.can_trade() == True
assert sim.position.active == False
print(f"  Can trade: {sim.can_trade()}")

receipt = sim.enter_short(60000.0, 0.01)
assert receipt is not None
assert sim.position.active == True
print(f"  Enter short: fill={receipt['fill_price']:.2f}, slippage={receipt['slippage_bps']:.2f}bps")

# Simulate price going in our favor (down)
result = sim.force_exit('take_profit', 59900.0)
assert result is not None
print(f"  Exit PnL: gross={result['gross_pnl_pct']*100:.3f}%, net={result['net_pnl_pct']*100:.3f}%, R={result['r_multiple']:.4f}")
assert len(sim.trades) == 1

status = sim.get_status()
assert status['total_trades'] == 1
assert status['in_position'] == False
print(f"  Status: total_trades={status['total_trades']}, net_pnl=${status['total_net_pnl_usd']:.4f}")
print("  PASSED ✓")

# ============================================================
# Test 4: FeatureEngine with simple data
# ============================================================
print("\n[Test 4] LiveFeatureEngine — Data Processing")
import pandas as pd

engine = LiveFeatureEngine()
for min_idx in range(3):
    base_price = 60000 + min_idx * 10
    ts_base = int(pd.Timestamp('2026-07-21 12:00:00').timestamp() * 1000) + min_idx * 60000
    for sec in range(60):
        ts = ts_base + sec * 100
        price = base_price + np.random.randn() * 5
        side = 'buy' if np.random.rand() > 0.5 else 'sell'
        amt = np.random.rand() * 0.5
        engine.on_trade(ts, price, amt, side)
        engine.on_orderbook(ts, price * 0.999, 10.0, price * 1.001, 10.0)

    features = engine.on_bar_close()
    assert features != {}, "Features should not be empty"
    print(f"  Bar {min_idx}: RSI14={features.get('rsi_14', 'N/A')} | Close={features.get('close', 0):.2f}")

    vec = engine.extract_feature_vector(features)
    assert vec.shape == (1, len(engine.get_feature_columns())), f"Shape mismatch: {vec.shape}"
    print(f"  Feature vector: {vec.shape} ✓")

print("  PASSED ✓")

# ============================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nPipeline is ready for live execution.")
print("Run: python -m main.executor")