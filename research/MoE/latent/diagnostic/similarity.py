"""
similarity.py
=====================================

Numerical similarity/distance function for the MoE latent layer. Once all the metrics are coded,
perform ablation test. The easiest way is to perform OLS or L2 regularization with expectancy.

Design:
------------
- Use numpy only for future implementation so its consistent with the conversion to polars.
- All final similarity functions should return values of [0,1] where practical.
- Main metrics: Cosine similarity/TS-SS, Mahalanobis, Wasserstein
"""

import numpy as np

# ===========================================================
# Similarity Scoring Method Initialization
# ===========================================================

def cosine_similarity(x,y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)

    if x_norm == 0 or y_norm == 0:
        return 0.0

    return np.dot(x, y) / (x_norm * y_norm)

def mahalanobis_similarity(x, state):
    # Mahalanobis measuring outlier based on skewness of data (with covariance and inv cov)
    d = x - state.mean
    d2 = d @ state.inv_cov @ d
    return np.exp(-d2 / 2)

def wasserstein_distance(x, y):
    # Wasserstein = ∫₀¹ |F_p⁻¹(u) - F_q⁻¹(u)| du
    # 1D Wasserstein. Multivariate Wasserstein might be too expensive — experiment later.

    x = np.sort(np.asarray(x))
    y = np.sort(np.asarray(y))

    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    return np.mean(np.abs(x - y))

def ts_ss_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)

    if x_norm == 0 or y_norm == 0:
        return np.nan

    cosine = np.dot(x, y) / (x_norm * y_norm)

    theta = np.degrees(
        np.arccos(
            np.clip(cosine, -1.0, 1.0)
        )
    )

    theta_prime = theta + 10.0

    ed = np.linalg.norm(x - y)
    md = abs(x_norm - y_norm)

    ts = (
        x_norm
        * y_norm
        * np.sin(np.radians(theta_prime))
        / 2.0
    )

    ss = (
        np.pi
        * (ed + md) ** 2
        * theta_prime
        / 360.0
    )

    return ts * ss

def similarity_metric(x, y, metric="cosine"):
    """
    Unified similarity score.
    Note: In using mahalanobis, y should be state.

    All metrics are converted to:
        0.0 = least similar
        1.0 = most similar
    """

    if metric == "cosine":
        cosine = cosine_similarity(x, y)

        # Cosine: [-1, 1] -> [0, 1]
        return (
            np.clip(cosine, -1.0, 1.0) + 1.0
        ) / 2.0

    elif metric == "wasserstein":
        distance = wasserstein_distance(x, y)

        # Distance >= 0.
        # Convert to similarity.
        return distance

    elif metric == "ts_ss":
        distance = ts_ss_distance(x, y)

        # Distance >= 0.
        # Convert to similarity.
        return distance

    elif metric == "mahalanobis":
        return mahalanobis_similarity(x, y)

    else:
        raise ValueError(
            f"Unknown similarity metric: {metric}"
        )

# ============================================================
# Test cases
# Used to test the effectiveness of similarity scores.
# ============================================================

def synth_pred():
    """
    Simple test to check similarity test. True test is observed in encoder.py
    """
    x = np.linspace(0, 20, 999)

    outlier = 0.9
    noise = np.repeat([0.1, outlier, 0.1], len(x) // 3)
    y = np.sin(x + noise) * np.cos(x + noise)

    # Three latent/state representations
    a = np.array([np.mean(y[:333]), np.std(y[:333]), 0.1])
    b = np.array([np.mean(y[333:666]), np.std(y[333:666]), outlier])
    c = np.array([np.mean(y[666:]), np.std(y[666:]), 0.1])

    print("A vs B:", cosine_similarity(a, b))
    print("A vs C:", cosine_similarity(a, c))
    print("B vs C:", cosine_similarity(b, c))

if __name__ == "__main__":
    synth_pred()