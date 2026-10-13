import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ...strategy.moe.ood.metrics import make_tdmi_similarity, tdmi_fingerprint, tdmi_fingerprint_similarity
from ...exchange.loader import load_L1

from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error


# ================================================================
# 1. Define regimes: predictive weights + temporal persistence
# ================================================================

regime_specs = {
    "Regime 1": {
        "weights":     [1.5, 1.0, 0.1, 0.1],
        "persistence": [0.9, 0.8, 0.1, 0.1],
    },
    "Regime 2": {
        "weights":     [0.8, 0.5, 1.2, 1.5],
        "persistence": [0.1, 0.1, 0.8, 0.9],
    },
    "Regime 3": {
        "weights":     [0.1, 0.2, 0.5, 2.0],
        "persistence": [0.1, 0.1, 0.4, 0.95],
    },
}

def regime_maker(n=500, spec=None, seed=42, noise=0.5):
    rng = np.random.default_rng(seed)
    w = np.asarray(spec["weights"])
    rho = np.asarray(spec["persistence"])

    x = np.zeros((n, len(w)))
    x[0] = rng.normal(size=len(w))

    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * rng.normal(size=len(w))

    # y represents the return associated with features at time t
    y = x @ w + rng.normal(0, noise, n)
    return x, y

# Recurring episodes: same regime settings, different seeds
N_TRAIN_EPISODES = 5
N_TEST_EPISODES = 5

train_episodes, test_episodes = {}, {}

for r, (name, spec) in enumerate(regime_specs.items()):
    train_episodes[name] = [
        regime_maker(spec=spec, seed=1000 + r * 100 + k)
        for k in range(N_TRAIN_EPISODES)
    ]
    test_episodes[name] = [
        regime_maker(spec=spec, seed=10000 + r * 100 + k)
        for k in range(N_TEST_EPISODES)
    ]

names = list(regime_specs)

# ================================================================
# 2. TDMI similarity matrix across recurring regime episodes
# ================================================================

tdmi_sim = make_tdmi_similarity(
    max_lag=10,
    bins=8,
    min_pairs=30,
    n_surrogates=200,
    seed=42,
)

tdmi_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i == j:
            # Compare different episodes of the same regime
            pairs = combinations(test_episodes[a], 2)
        else:
            # Compare episodes across two different regimes
            pairs = (
                (ea, eb)
                for ea in test_episodes[a]
                for eb in test_episodes[b]
            )

        scores = [
            tdmi_sim(xa, xb)
            for (xa, _), (xb, _) in pairs
        ]
        scores = [s for s in scores if np.isfinite(s)]
        tdmi_matrix.loc[a, b] = np.mean(scores) if scores else np.nan

print("\nMean TDMI similarity matrix:")
print(tdmi_matrix.round(3))

# ================================================================
# 3. Train one expert per regime
# ================================================================

experts = {}

for name in names:
    x_train = np.concatenate([x for x, _ in train_episodes[name]])
    y_train = np.concatenate([y for _, y in train_episodes[name]])

    experts[name] = Ridge(alpha=1.0).fit(x_train, y_train)

# ================================================================
# 4. Cross-regime evaluation: MI and MSE
# ================================================================

mi_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
mse_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

for expert_name, model in experts.items():
    for test_name in names:
        x_test = np.concatenate([x for x, _ in test_episodes[test_name]])
        y_test = np.concatenate([y for _, y in test_episodes[test_name]])

        y_pred = model.predict(x_test)

        mi = mutual_info_regression(
            y_pred.reshape(-1, 1),
            y_test,
            random_state=42,
        )[0]

        mi_matrix.loc[expert_name, test_name] = mi
        mse_matrix.loc[expert_name, test_name] = mean_squared_error(
            y_test, y_pred
        )

print("\nPredictive MI matrix (nats):")
print(mi_matrix.round(3))

print("\nMSE matrix (lower is better):")
print(mse_matrix.round(3))

# ================================================================
# 5. Visualize all three matrices
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, matrix, title in zip(
    axes,
    [tdmi_matrix, mi_matrix, mse_matrix],
    ["TDMI Similarity", "Predictive MI (nats)", "Predictive MSE"],
):
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title(title)

    for i in range(len(names)):
        for j in range(len(names)):
            value = values[i, j]
            label = f"{value:.2f}" if np.isfinite(value) else "NaN"
            ax.text(j, i, label, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# ==================================================================
# 6. Compare TDMI fingerprints using Jaccard, cosine, and distance
# ==================================================================

from itertools import combinations, product

# Compute each episode's TDMI fingerprint once
fp_episodes = {
    name: [
        np.ravel(tdmi_fingerprint(x, 10, 8, 30, 200, 42)).astype(float)
        for x, _ in test_episodes[name]
    ]
    for name in names
}

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)

    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0

    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

# Calibrate the distance scale across all episode pairs
all_fp = [fp for group in fp_episodes.values() for fp in group]
distances = np.array([
    np.linalg.norm(a - b)
    for a, b in combinations(all_fp, 2)
])
positive = distances[distances > 1e-12]
dist_scale = np.median(positive) if len(positive) else 1.0

def distance_sim(a, b):
    d = np.linalg.norm(a - b)
    return float(np.exp(-d / dist_scale))

def make_similarity_matrix(metric):
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                pairs = combinations(fp_episodes[a], 2)
            else:
                pairs = product(fp_episodes[a], fp_episodes[b])

            scores = [metric(u, v) for u, v in pairs]
            matrix.loc[a, b] = np.mean(scores)

    return matrix

jaccard_matrix = make_similarity_matrix(
    lambda a, b: tdmi_fingerprint_similarity(a, b, 0.0)
)
cosine_matrix = make_similarity_matrix(cosine_sim)
distance_matrix = make_similarity_matrix(distance_sim)

print("\nWeighted Jaccard:")
print(jaccard_matrix.round(3))

print("\nCosine similarity:")
print(cosine_matrix.round(3))

print("\nEuclidean-distance similarity:")
print(distance_matrix.round(3))