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
# 1. Define regimes: brute-force block discovery on real L1 data
#    (replaces the synthetic regime_specs / regime_maker)
# ================================================================

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
SINCE = "2025-01-01T00:00:00Z"
UNTIL = "2025-06-01T00:00:00Z"

BLOCK_LEN = 288           # 1 day of 5m bars; one block = one candidate "episode"
N_REGIMES = 3             # how many recurring regimes we keep
MIN_EPISODES = 6          # a regime must recur at least this many times
TRAIN_FRAC = 0.6          # chronological split inside each regime (no shuffling)
Y_SCALE = 1e4             # forward return in bps so MSE is readable

df = load_L1(symbol=SYMBOL, timeframe=TIMEFRAME, since=SINCE, until=UNTIL)
df = df.sort_index()

close = df["close"].astype(float)
ret = np.log(close).diff()

# Features known at time t (4 features, same width as the synthetic version)
feat = pd.DataFrame(index=df.index)
feat["ret"] = ret
feat["vol"] = ret.rolling(20).std()
feat["rng"] = (df["high"] - df["low"]) / close
feat["vlm"] = np.log1p(df["volume"].astype(float))

# y = NEXT bar return (features at t -> return t+1), so no look-ahead
y_all = ret.shift(-1) * Y_SCALE

data = pd.concat([feat, y_all.rename("y")], axis=1).dropna()

# Standardise with statistics from the first TRAIN_FRAC of the sample only
cut = int(len(data) * TRAIN_FRAC)
mu = data.iloc[:cut][feat.columns].mean()
sd = data.iloc[:cut][feat.columns].std().replace(0, 1.0)
X_all = ((data[feat.columns] - mu) / sd).to_numpy(float)
Y_all = data["y"].to_numpy(float)

# Cut into non-overlapping candidate blocks
n_blocks = len(data) // BLOCK_LEN
blocks = [
    (X_all[k * BLOCK_LEN:(k + 1) * BLOCK_LEN],
     Y_all[k * BLOCK_LEN:(k + 1) * BLOCK_LEN])
    for k in range(n_blocks)
]
print(f"Loaded {len(data)} bars -> {n_blocks} candidate blocks of {BLOCK_LEN} bars")

# Fingerprint every block once (x only, no y -> clustering never sees labels)
block_fp = [
    np.ravel(tdmi_fingerprint(x, 10, 8, 30, 200, 42)).astype(float)
    for x, _ in blocks
]


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)

    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0

    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


# Calibrate the distance scale across all block pairs
_d = np.array([np.linalg.norm(a - b) for a, b in combinations(block_fp, 2)])
_pos = _d[_d > 1e-12]
dist_scale = np.median(_pos) if len(_pos) else 1.0


def distance_sim(a, b):
    d = np.linalg.norm(a - b)
    return float(np.exp(-d / dist_scale))


# The two similarity definitions we run the whole pipeline with
SIMILARITIES = {
    "TDMI (Jaccard)":      lambda a, b: tdmi_fingerprint_similarity(a, b, 0.0),
    "Geometric (distance)": distance_sim,
}


def block_similarity_matrix(sim):
    n = len(block_fp)
    S = np.eye(n)
    for i, j in combinations(range(n), 2):
        S[i, j] = S[j, i] = sim(block_fp[i], block_fp[j])
    return S


def leader_cluster(S, thr):
    """Greedy: a block joins a cluster if its mean similarity to members >= thr."""
    clusters = []
    for i in range(len(S)):
        placed = False
        for c in clusters:
            if np.mean(S[i, c]) >= thr:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def brute_force_regimes(S):
    """Sweep thresholds; keep the one whose recurring clusters are most separated."""
    off = S[np.triu_indices_from(S, k=1)]
    best, best_score = None, -np.inf

    for thr in np.unique(np.quantile(off, np.linspace(0.05, 0.95, 40))):
        clusters = [c for c in leader_cluster(S, thr) if len(c) >= MIN_EPISODES]
        if len(clusters) < N_REGIMES:
            continue

        clusters = sorted(clusters, key=len, reverse=True)[:N_REGIMES]
        within = np.mean([
            S[np.ix_(c, c)][np.triu_indices(len(c), k=1)].mean() for c in clusters
        ])
        between = np.mean([
            S[np.ix_(a, b)].mean() for a, b in combinations(clusters, 2)
        ])
        score = within - between

        if score > best_score:
            best, best_score = clusters, score

    return best, best_score


names = [f"Regime {r + 1}" for r in range(N_REGIMES)]

results = {}

# ================================================================
# Run the full pipeline twice: 1st = TDMI similarity, 2nd = geometric
# ================================================================

for sim_name, sim in SIMILARITIES.items():
    print("\n" + "#" * 66)
    print(f"# RUN: {sim_name}")
    print("#" * 66)

    S = block_similarity_matrix(sim)
    clusters, score = brute_force_regimes(S)

    if clusters is None:
        print("No recurring regimes found with these settings "
              "(try lowering MIN_EPISODES or BLOCK_LEN).")
        continue

    print(f"Regime separation score (within - between): {score:.3f}")

    # Recurring episodes: blocks of the same cluster, split chronologically
    train_episodes, test_episodes = {}, {}
    for name, idx in zip(names, clusters):
        idx = sorted(idx)
        k = max(1, int(len(idx) * TRAIN_FRAC))
        train_episodes[name] = [blocks[i] for i in idx[:k]]
        test_episodes[name] = [blocks[i] for i in idx[k:]]
        print(f"{name}: {len(idx)} blocks "
              f"(train {len(train_episodes[name])}, test {len(test_episodes[name])})")

    if any(len(test_episodes[n]) < 2 for n in names):
        print("Not enough test episodes per regime; skipping this run.")
        continue

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
    expectancy_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

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
            # Expectancy: mean bps per bar of trading sign(prediction), zero fees
            expectancy_matrix.loc[expert_name, test_name] = np.mean(
                np.sign(y_pred) * y_test
            )

    # Baseline: predict the regime's train mean (what "no skill" looks like)
    baseline_mse = pd.Series(index=names, dtype=float)
    for test_name in names:
        y_test = np.concatenate([y for _, y in test_episodes[test_name]])
        y_tr = np.concatenate([y for _, y in train_episodes[test_name]])
        baseline_mse[test_name] = mean_squared_error(
            y_test, np.full_like(y_test, y_tr.mean())
        )

    print("\nPredictive MI matrix (nats):")
    print(mi_matrix.round(4))

    print("\nMSE matrix (lower is better):")
    print(mse_matrix.round(3))

    print("\nBaseline MSE (train-mean predictor) per test regime:")
    print(baseline_mse.round(3))

    print("\nExpectancy matrix (bps / bar, gross):")
    print(expectancy_matrix.round(4))

    # ================================================================
    # 5. Visualize all three matrices (+ expectancy)
    # ================================================================

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"{sim_name} regime discovery on {SYMBOL} {TIMEFRAME}")

    for ax, matrix, title in zip(
        axes,
        [tdmi_matrix, mi_matrix, mse_matrix, expectancy_matrix],
        ["TDMI Similarity", "Predictive MI (nats)", "Predictive MSE",
         "Expectancy (bps/bar)"],
    ):
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_title(title)

        for i in range(len(names)):
            for j in range(len(names)):
                value = values[i, j]
                label = f"{value:.3f}" if np.isfinite(value) else "NaN"
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

    # Calibrate the distance scale across all episode pairs
    all_fp = [fp for group in fp_episodes.values() for fp in group]
    distances = np.array([
        np.linalg.norm(a - b)
        for a, b in combinations(all_fp, 2)
    ])
    positive = distances[distances > 1e-12]
    dist_scale = np.median(positive) if len(positive) else 1.0

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

    results[sim_name] = {
        "mi": mi_matrix,
        "mse": mse_matrix,
        "baseline_mse": baseline_mse,
        "expectancy": expectancy_matrix,
    }

# ================================================================
# 7. Head-to-head: TDMI run vs geometric run
#    Does the matched expert (diagonal) beat mismatched experts
#    (off-diagonal) and the no-skill baseline?
# ================================================================

for sim_name, res in results.items():
    mi, mse, exp_, base = res["mi"], res["mse"], res["expectancy"], res["baseline_mse"]
    diag = np.eye(len(names), dtype=bool)

    print(f"\n=== {sim_name} ===")
    print(f"MI   diag {mi.to_numpy()[diag].mean():.4f} | off-diag {mi.to_numpy()[~diag].mean():.4f}")
    print(f"MSE  diag {mse.to_numpy()[diag].mean():.3f} | off-diag {mse.to_numpy()[~diag].mean():.3f}"
          f" | baseline {base.mean():.3f}")
    print(f"Expectancy diag {exp_.to_numpy()[diag].mean():.4f} bps | "
          f"off-diag {exp_.to_numpy()[~diag].mean():.4f} bps")