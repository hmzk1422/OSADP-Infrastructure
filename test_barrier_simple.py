"""
Simple diagnostic to verify barrier_pct calculation.
"""
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.config import *
from main.execution import PositionTracker, SlippageModel

# Create tracker
tracker = PositionTracker()
slippage = SlippageModel()
tracker.attach_slippage(slippage)

# Simulate realistic mid-prices (avoiding bid/ask alternation)
mid_price = 1865.00
np.random.seed(42)
returns = np.random.normal(0, 0.0001, 200)

prices = [mid_price]
for ret in returns:
    prices.append(prices[-1] * (1 + ret))

# Feed to vol buffer
for price in prices:
    tracker.update_volatility(price)

# Calculate barrier
barrier_pct = tracker.get_barrier_pct()

print(f"BARRIER_MULTIPLIER: {BARRIER_MULTIPLIER}")
print(f"barrier_pct: {barrier_pct:.6f} ({barrier_pct*100:.4f}%)")
print(f"Expected: ~0.04% - 0.08%")

# Test entry
entry_mid = prices[-1]
qty = MAX_POSITION_USD / entry_mid
receipt = tracker.enter_long(entry_mid, qty)

fill = receipt['fill_price']
b_pct = receipt['barrier_pct']
stop = receipt['stop_barrier']
profit = receipt['profit_barrier']

print(f"\nEntry fill: {fill:.2f}")
print(f"barrier_pct: {b_pct:.6f} ({b_pct*100:.4f}%)")
print(f"Stop: {stop:.2f} ({(stop/fill-1)*100:.4f}%)")
print(f"Profit: {profit:.2f} ({(profit/fill-1)*100:.4f}%)")

if b_pct < 0.001:
    print("\nPASS: Barriers correctly scaled with MULTIPLIER=0.5")
    sys.exit(0)
else:
    print(f"\nFAIL: barrier_pct={b_pct*100:.3f}% still too large")
    sys.exit(1)

