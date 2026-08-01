import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.config import *
from main.execution import PositionTracker, SlippageModel

tracker = PositionTracker()
slippage = SlippageModel()
tracker.attach_slippage(slippage)

mid_price = 1865.00
np.random.seed(42)
returns = np.random.normal(0, 0.0001, 200)

prices = [mid_price]
for ret in returns:
    prices.append(prices[-1] * (1 + ret))

for price in prices:
    tracker.update_volatility(price)

barrier_pct = tracker.get_barrier_pct()

print("BARRIER_MULTIPLIER:", BARRIER_MULTIPLIER)
print("barrier_pct:", barrier_pct)
print("barrier_pct %:", barrier_pct * 100)

entry_mid = prices[-1]
qty = MAX_POSITION_USD / entry_mid
receipt = tracker.enter_long(entry_mid, qty)

fill = receipt['fill_price']
b_pct = receipt['barrier_pct']
stop = receipt['stop_barrier']
profit = receipt['profit_barrier']

print("Entry fill:", fill)
print("barrier_pct:", b_pct)
print("barrier_pct %:", b_pct * 100)
print("Stop:", stop)
print("Profit:", profit)

stop_diff = stop - fill
profit_diff = profit - fill
print("Stop distance:", stop_diff)
print("Profit distance:", profit_diff)

if b_pct < 0.001:
    print("PASS: Barriers correctly scaled")
    sys.exit(0)
else:
    print("FAIL: barrier_pct too large:", b_pct * 100)
    sys.exit(1)
