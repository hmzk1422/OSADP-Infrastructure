import numpy as np
import pandas as pd
from .indicators import calculate_zscore
import matplotlib.pyplot as plt

def run_historical_strategy(betas, price_a, price_b, window=50):
    # 1. FORCE CONVERSION: If these are DataFrames, .values strips the 
    # column labels and returns a pure NumPy array.
    betas = np.asanyarray(betas).astype(float)
    price_a = np.asanyarray(price_a).astype(float)
    price_b = np.asanyarray(price_b).astype(float)
    # 2. Calculate z-score using your existing function
    beta_zscore = calculate_zscore(betas, window)
    
    n = len(betas)
    pnl_history = np.zeros(n)
    in_position = False
    entry_beta = 0.0
    entry_pa = 0.0
    entry_pb = 0.0
    side = 0
    
    # 3. Now the loop is safe because price_a and price_b are pure NumPy
    for i in range(window, n):
        z = beta_zscore[i]
        pa = price_a[i]
        pb = price_b[i]
        
        if not in_position:
            if z <= -2.0:
                in_position, side, entry_beta = True, 1, betas[i]
                entry_pa, entry_pb = pa, pb
            elif z >= 2.0:
                in_position, side, entry_beta = True, -1, betas[i]
                entry_pa, entry_pb = pa, pb
        
        elif in_position:
            current_spread = pa - (entry_beta * pb)
            
            diff_a = pa - entry_pa
            diff_b = pb - entry_pb
            pnl_history[i] = side * (diff_a - (entry_beta * diff_b))
            
            if abs(current_spread) < 0.01:
                in_position = False
                
    return pnl_history


# --- Plotting Tools ---

def plot_pnl_history(pnl_history):
    plt.figure(figsize=(10, 6))
    plt.plot(pnl_history, label="Strategy PnL", color='blue', linewidth=1)
    plt.title("Historical Strategy PnL")
    plt.xlabel("Time Index")
    plt.ylabel("Cumulative PnL")
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8) # Baseline at zero
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()