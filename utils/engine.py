import numpy as np
from .indicators import calculate_zscore


def run_historical_strategy(betas, price_a, price_b, window=50):
    betas = np.asanyarray(betas, dtype=float)
    price_a = np.asanyarray(price_a, dtype=float)
    price_b = np.asanyarray(price_b, dtype=float)

    beta_zscore = calculate_zscore(betas, window)

    n = len(betas)
    pnl_history = np.zeros(n)
    in_position = False
    entry_beta = 0.0
    entry_pa = 0.0
    entry_pb = 0.0
    side = 0

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
            diff_a = pa - entry_pa
            diff_b = pb - entry_pb
            pnl_history[i] = side * (diff_a - (entry_beta * diff_b))

            current_spread = pa - (entry_beta * pb)
            if abs(current_spread) < 0.01:
                in_position = False

    return pnl_history