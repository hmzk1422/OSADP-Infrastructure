"""
Quick diagnostic to verify barrier_pct calculation with the fix.
This simulates what happens in live trading without needing the full pipeline.
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.config import *
from main.execution import PositionTracker, SlippageModel

def test_barrier_calculation():
    """Test that barrier_pct scales correctly with BARRIER_MULTIPLIER."""
    
    print("=" * 70)
    print("BARRIER CALCULATION DIAGNOSTIC")
    print("=" * 70)
    print(f"BARRIER_MULTIPLIER : {BARRIER_MULTIPLIER}")
    print(f"TIMEFRAME_BARRIER  : {TIMEFRAME_BARRIER}s")
    print(f"LOOKAHEAD_SECONDS  : {LOOKAHEAD_SECONDS}s")
    print()
    
    # Create a position tracker
    tracker = PositionTracker()
    slippage = SlippageModel()
    tracker.attach_slippage(slippage)
    
    # Simulate realistic price data (like ETH/USDT around 1865)
    # Use MID-PRICES to avoid artificial volatility from bid/ask alternation
    print("Simulating 200 seconds of mid-price data...")
    
    # Start with a realistic mid-price
    mid_price = 1865.00
    
    # Simulate small realistic movements (0.01% per second typical)
    # This is much smaller than the 0.5% swings you get from bid/ask alternation
    np.random.seed(42)
    returns = np.random.normal(0, 0.0001, 200)  # 0.01% std dev
    
    prices = [mid_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Feed prices to volatility buffer (simulating what executor does)
    for price in prices:
        tracker.update_volatility(price)
    
    # Calculate barrier_pct
    barrier_pct = tracker.get_barrier_pct()
    
    print(f"\nResults after feeding {len(prices)} price points:")
    print(f"  barrier_pct        : {barrier_pct:.6f} ({barrier_pct*100:.4f}%)")
    print(f"  Expected range     : ~0.04% - 0.08% (with MULTIPLIER=0.5)")
    print()
    
    # Simulate an entry at current mid
    entry_mid = prices[-1]
    qty = MAX_POSITION_USD / entry_mid
    
    print(f"Simulating LONG entry at mid={entry_mid:.2f}:")
    receipt = tracker.enter_long(entry_mid, qty)
    
    fill = receipt['fill_price']
    b_pct = receipt['barrier_pct']
    stop = receipt['stop_barrier']
    profit = receipt['profit_barrier']
    
    print(f"  Entry fill price   : {fill:.2f}")
    print(f"  barrier_pct used   : {b_pct:.6f} ({b_pct*100:.4f}%)")
    stop_pct = (stop / fill - 1) * 100
    profit_pct = (profit / fill - 1) * 100
    print(f"  Stop barrier       : {stop:.2f} ({stop_pct:.4f}%)")
    print(f"  Profit barrier     : {profit:.2f} ({profit_pct:.4f}%)")
    print()
    
    # Calculate what the barriers should be
    stop_distance = fill * b_pct
    profit_distance = fill * b_pct
    
    print("Barrier distances from entry:")
    print(f"  Stop distance      : {stop_distance:.2f} points")
    print(f"  Profit distance    : {profit_distance:.2f} points")
    print()
    
    # Check if barriers are reasonable
    if b_pct < 0.001:  # Less than 0.1%
        print("PASS: Barriers are small (as expected with MULTIPLIER=0.5)")
    elif b_pct < 0.003:  # Less than 0.3%
        print("WARNING: Barriers are smaller than before but still > 0.1%")
    else:
        print("FAIL: Barriers are still too large!")
        print(f"  Expected ~0.05%, got {b_pct*100:.2f}%")
    
    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    return b_pct

if __name__ == "__main__":
    b_pct = test_barrier_calculation()
    
    # Exit with status code based on result
    if b_pct < 0.001:
        print("\nFix verified: barrier_pct is correctly scaled with MULTIPLIER=0.5")
        sys.exit(0)
    else:
        print(f"\nIssue remains: barrier_pct={b_pct*100:.3f}% is still too large")
        sys.exit(1)

