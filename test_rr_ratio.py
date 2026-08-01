import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.config import *
from main.execution import PositionTracker, SlippageModel

# Test RR_RATIO implementation
tracker = PositionTracker()
slippage = SlippageModel()
tracker.attach_slippage(slippage)

# Simulate price data
mid_price = 1865.00
np.random.seed(42)
returns = np.random.normal(0, 0.0001, 200)

prices = [mid_price]
for ret in returns:
    prices.append(prices[-1] * (1 + ret))

for price in prices:
    tracker.update_volatility(price)

# Test SHORT with RR_RATIO = 2.0
print("=" * 70)
print("RR_RATIO TEST")
print("=" * 70)
print(f"RR_RATIO: {RR_RATIO}")
print()

entry_mid = prices[-1]
qty = MAX_POSITION_USD / entry_mid

print("Testing SHORT position:")
receipt_short = tracker.enter_short(entry_mid, qty)

fill_short = receipt_short['fill_price']
stop_short = receipt_short['stop_barrier']
profit_short = receipt_short['profit_barrier']
b_pct = receipt_short['barrier_pct']

print(f"  Entry: {fill_short:.2f}")
print(f"  Stop: {stop_short:.2f} (+{((stop_short/fill_short)-1)*100:.4f}%)")
print(f"  Profit: {profit_short:.2f} (-{((fill_short-profit_short)/fill_short)*100:.4f}%)")

stop_dist = stop_short - fill_short
profit_dist = fill_short - profit_short
rr_actual = profit_dist / stop_dist if stop_dist != 0 else 0

print(f"  Stop distance: {stop_dist:.2f} points")
print(f"  Profit distance: {profit_dist:.2f} points")
print(f"  Actual RR: {rr_actual:.2f}:1")
print()

# Test LONG with RR_RATIO = 2.0
tracker2 = PositionTracker()
tracker2.attach_slippage(slippage)
for price in prices:
    tracker2.update_volatility(price)

print("Testing LONG position:")
receipt_long = tracker2.enter_long(entry_mid, qty)

fill_long = receipt_long['fill_price']
stop_long = receipt_long['stop_barrier']
profit_long = receipt_long['profit_barrier']

print(f"  Entry: {fill_long:.2f}")
print(f"  Stop: {stop_long:.2f} (-{((fill_long-stop_long)/fill_long)*100:.4f}%)")
print(f"  Profit: {profit_long:.2f} (+{((profit_long/fill_long)-1)*100:.4f}%)")

stop_dist_long = fill_long - stop_long
profit_dist_long = profit_long - fill_long
rr_actual_long = profit_dist_long / stop_dist_long if stop_dist_long != 0 else 0

print(f"  Stop distance: {stop_dist_long:.2f} points")
print(f"  Profit distance: {profit_dist_long:.2f} points")
print(f"  Actual RR: {rr_actual_long:.2f}:1")
print()

# Verify RR is correct
if 1.8 < rr_actual < 2.2 and 1.8 < rr_actual_long < 2.2:
    print("PASS: RR_RATIO is correctly implemented (≈2:1)")
    sys.exit(0)
else:
    print(f"FAIL: Expected RR=2.0, got SHORT={rr_actual:.2f}, LONG={rr_actual_long:.2f}")
    sys.exit(1)
