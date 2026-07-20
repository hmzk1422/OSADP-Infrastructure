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

def plot_one(pnl_history):
    plt.figure(figsize=(10, 6))
    plt.plot(pnl_history, label="Strategy PnL", color='blue', linewidth=1)
    plt.title("Historical Strategy PnL")
    plt.xlabel("Time Index")
    plt.ylabel("Cumulative PnL")
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8) # Baseline at zero
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_two(primary_data, secondary_data_1, secondary_data_2=None, 
             primary_label="Primary Metric", secondary_label_1="Indicator 1", 
             secondary_label_2="Indicator 2", title="Historical Analysis"):
    """
    Plots a primary metric (e.g., PnL) on the top panel and up to two 
    secondary metrics (e.g., RSI and its SMA) on the bottom panel, sharing the same X-axis.
    """
    # Create a figure with 2 subplots sharing the same X-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # ==========================================
    # Panel 1: Primary Metric (e.g., PnL, Portfolio Value)
    # ==========================================
    ax1.plot(primary_data, label=primary_label, color='blue', linewidth=1.2)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel(primary_label, fontsize=11)
    ax1.axhline(0, color='red', linestyle='--', linewidth=0.8) # Baseline reference
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # Panel 2: Secondary Metrics (e.g., Indicators, Macro Data)
    # ==========================================
    ax2.plot(secondary_data_1, label=secondary_label_1, color='purple', linewidth=1)
    
    # Optional second curve on the bottom panel (like your SMA)
    if secondary_data_2 is not None:
        ax2.plot(secondary_data_2, label=secondary_label_2, color='orange', linewidth=1.2)
        
    ax2.set_xlabel("Time Index", fontsize=11)
    ax2.set_ylabel("Value / Scale", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()