import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from strategy_system.strategy.MoE.latent.diagnostic.similarity import similarity_metric
from components.python.data_pipeline.synthetic.synth1 import simulate_market_data
from components.python.data_pipeline.synthetic.synth2 import simulate_market_stochastic
from strategy_system.strategy.MoE.latent.vqvae.vqvae import LatentStateModel
from strategy_system.strategy.MoE.latent.diagnostic.hypothesis import informative_temporality, test_latent_clusterability, find_best_similarity_model
import random
import torch

# ============================================================
# Keep seeds stable
# ============================================================

seed = 84

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Configurations
# ============================================================

configs = [
    {"window_size": 16, "latent_dim": 8,  "hidden_dim": 128},
    {"window_size": 16, "latent_dim": 16, "hidden_dim": 128},
    {"window_size": 32, "latent_dim": 8,  "hidden_dim": 128},
    {"window_size": 32, "latent_dim": 16, "hidden_dim": 128},
    {"window_size": 32, "latent_dim": 32, "hidden_dim": 256},
    {"window_size": 64, "latent_dim": 8,  "hidden_dim": 128},
    {"window_size": 64, "latent_dim": 16, "hidden_dim": 128},
]


# ============================================================
# Synthetic market
# ============================================================

df = simulate_market_data(
    n_bars=5_000,
    n_features=50,
    n_states=4,
    seed=seed,
    feature_noise=0.5,
)

features = [
    c for c in df.columns
    if c.startswith("f")
]

X = df[features].values
true_state = df["state"].values

#informative_temporality(X, df, true_state, seed)

# ============================================================
# Run experiments
# ============================================================

results = []

for i, config in enumerate(configs, 1):

    vqvae = LatentStateModel(
        n_features=len(features),
        window_size=config["window_size"],
        latent_dim=config["latent_dim"],
        n_codes=4,
        hidden_dim=config["hidden_dim"],
        beta=0.25,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=200,
        batch_size=256,
    )

    history = vqvae.fit(X)
    output = vqvae.encode(X)

    latent_z = output["z_e"]
    latent_state = output["state"]
    quantization_distance = output["quantization_distance"]

    true_window_state = true_state[
        config["window_size"] - 1:
    ]


    # ========================================================
    # Align arbitrary VQ code IDs to true HMM state IDs
    #
    # VQ code 0 does not necessarily mean true state 0.
    # Test all possible mappings and keep the best one.
    # ========================================================

    best_accuracy = -1.0
    best_mapping = None

    for permutation in permutations(range(4)):

        mapping = np.asarray(permutation)
        mapped = mapping[latent_state]

        accuracy = np.mean(
            mapped == true_window_state
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping

    mapped_state = best_mapping[latent_state]


    # ========================================================
    # Transition count
    # ========================================================

    true_transitions = np.sum(
        true_window_state[1:]
        != true_window_state[:-1]
    )

    discovered_transitions = np.sum(
        mapped_state[1:]
        != mapped_state[:-1]
    )


    # ========================================================
    # VQ code usage
    # ========================================================

    usage = np.bincount(
        latent_state,
        minlength=4,
    ) / len(latent_state)


    results.append({
        "config": config,
        "accuracy": best_accuracy,
        "transitions": int(discovered_transitions),
        "true_transitions": int(true_transitions),
        "quantization_error": float(
            quantization_distance.mean()
        ),
        "final_loss": float(history[-1]),
        "usage": usage,
        "vqvae": vqvae,
        "history": history,
        "latent_z": latent_z,
        "latent_state": latent_state,
        "mapped_state": mapped_state,
        "true_state": true_window_state,
    })


    print(
        f"[{i}/{len(configs)}] "
        f"window={config['window_size']} | "
        f"latent={config['latent_dim']} | "
        f"hidden={config['hidden_dim']} | "
        f"accuracy={best_accuracy:.3%} | "
        f"transitions={discovered_transitions}/{true_transitions} | "
        f"q_error={quantization_distance.mean():.4f} | "
        f"loss={history[-1]:.4f} | "
        f"usage={np.round(usage, 2)}"
    )

"""
# ============================================================
# Best configuration
# ============================================================

best = max(
    results,
    key=lambda x: x["accuracy"],
)
clusterability_results = (
    test_latent_clusterability(
        latent_z=best["latent_z"],
        true_state=best["true_state"],
        n_clusters=4,
        train_frac=0.8,
        seed=seed,
    )
)


print()

print("=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    f"Window size         : "
    f"{best['config']['window_size']}"
)

print(
    f"Latent dimension    : "
    f"{best['config']['latent_dim']}"
)

print(
    f"Hidden dimension    : "
    f"{best['config']['hidden_dim']}"
)

print(
    f"State accuracy      : "
    f"{best['accuracy']:.3%}"
)

print(
    f"True transitions    : "
    f"{best['true_transitions']}"
)

print(
    f"VQ transitions      : "
    f"{best['transitions']}"
)

print(
    f"Quantization error  : "
    f"{best['quantization_error']:.6f}"
)

print(
    f"Final loss          : "
    f"{best['final_loss']:.6f}"
)

print(
    f"Codebook usage      : "
    f"{np.round(best['usage'], 3)}"
)
"""
# ============================================================
# Best configuration by MARKET-SPACE SIMILARITY
# ============================================================

similarity_metric_name = "cosine"

best_similarity, similarity_results = (
    find_best_similarity_model(
        X=X,
        results=results,
        metric=similarity_metric_name,
        max_samples_per_state=200,
        seed=seed,
    )
)

best = next(
    result
    for result in results
    if result["config"] == best_similarity["config"]
)

print()
print("=" * 70)
print("BEST CONFIGURATION BY MARKET-SPACE SIMILARITY")
print("=" * 70)

print(
    f"Metric              : "
    f"{similarity_metric_name}"
)

print(
    f"Window size         : "
    f"{best['config']['window_size']}"
)

print(
    f"Latent dimension    : "
    f"{best['config']['latent_dim']}"
)

print(
    f"Hidden dimension    : "
    f"{best['config']['hidden_dim']}"
)

print(
    f"Mean diagonal score : "
    f"{best_similarity['mean_diagonal_similarity']:.6f}"
)

print(
    f"Per-state diagonal  : "
    f"{np.round(best_similarity['diagonal_scores'], 6)}"
)

# ============================================================
# Final comparison table
# ============================================================

print()

print("=" * 70)
print("ALL CONFIGURATIONS")
print("=" * 70)

for result in sorted(
    results,
    key=lambda x: x["accuracy"],
    reverse=True,
):

    config = result["config"]

    print(
        f"window={config['window_size']:>2} | "
        f"latent={config['latent_dim']:>2} | "
        f"hidden={config['hidden_dim']:>3} | "
        f"accuracy={result['accuracy']:.3%} | "
        f"transitions={result['transitions']:>3} | "
        f"q_error={result['quantization_error']:.4f} | "
        f"loss={result['final_loss']:.4f}"
    )

# ============================================================
# DIAGNOSTICS FOR BEST MODEL
# ============================================================

true_best = best["true_state"]
vq_best = best["mapped_state"]


# ============================================================
# 0. Continuous latent-space diagnostics
#
# What are we trying to find out?
# → Did the encoder separate the true states in z_e BEFORE
#   the codebook/quantization step?
#
# If z_e separates the states well but VQ does not:
# → likely codebook / quantization problem.
#
# If z_e does NOT separate the states:
# → likely encoder / reconstruction-objective problem.
# ============================================================

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


latent_z = best["latent_z"]


# ============================================================
# PCA visualization
#
# z_e may have 8, 16, or 32 dimensions.
# PCA reduces it to 2 dimensions so we can visualize it.
#
# PCA itself does NOT know the true states.
# We only color the points afterwards using the HMM state.
# ============================================================

pca = PCA(
    n_components=2
)

z_2d = pca.fit_transform(
    latent_z
)

print()
print("=" * 70)
print("PCA OF z_e")
print("=" * 70)

print(
    f"PCA 1 explained variance : "
    f"{pca.explained_variance_ratio_[0]:.3%}"
)

print(
    f"PCA 2 explained variance : "
    f"{pca.explained_variance_ratio_[1]:.3%}"
)

print(
    f"Total explained variance : "
    f"{pca.explained_variance_ratio_.sum():.3%}"
)


plt.figure(figsize=(8, 6))

for state in range(4):

    mask = true_best == state

    plt.scatter(
        z_2d[mask, 0],
        z_2d[mask, 1],
        label=f"True state {state}",
        alpha=0.5,
    )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Continuous Latent Space (z_e) by True State")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Numeric separability test
#
# Instead of relying only on the 2-D PCA picture, use the FULL
# z_e representation to predict the true HMM state.
#
# We use a chronological split rather than random CV because
# overlapping market windows contain many of the same bars.
#
# Train  = earlier observations
# Test   = later observations
#
# If k-NN performs well on z_e but VQ accuracy is poor:
# → encoder contains the state information
# → codebook/quantization is the bottleneck.
#
# If k-NN also performs poorly:
# → state information was not cleanly separated by encoder.
# ============================================================

split = int(
    len(latent_z) * 0.8
)

z_train = latent_z[:split]
z_test = latent_z[split:]

state_train = true_best[:split]
state_test = true_best[split:]


knn = KNeighborsClassifier(
    n_neighbors=15
)

knn.fit(
    z_train,
    state_train,
)

knn_accuracy = knn.score(
    z_test,
    state_test,
)

print()
print("=" * 70)
print("k-NN STATE SEPARABILITY IN z_e")
print("=" * 70)

print(
    f"k-NN accuracy on z_e : "
    f"{knn_accuracy:.3%}"
)

print(
    f"Random 4-state chance : "
    f"{1 / 4:.3%}"
)

# ============================================================
# 1. Confusion matrix
#
# What are we trying to find out?
# → Which VQ state did the model assign to each true state?
#
# This lets us see whether:
# - each true state has its own VQ state, or
# - multiple true states are being assigned to the same VQ state.
#
# Rows    = true HMM state
# Columns = VQ predicted state
#
# Diagonal     = correct classifications
# Off-diagonal = state confusion
# ============================================================

confusion = np.zeros((4, 4), dtype=int)

for true_value, vq_value in zip(
    true_best,
    vq_best,
):
    confusion[true_value, vq_value] += 1

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print(
    "Rows = true state | Columns = VQ state"
)
print(confusion)


# ============================================================
# 2. True vs VQ state frequencies
#
# What are we trying to find out?
# → How often does each state actually occur?
#
# True frequency tells us the real distribution of the
# four HMM states.
#
# VQ frequency tells us how often the VQ-VAE uses each
# discovered state.
#
# This helps distinguish:
# - a genuinely rare true state, from
# - a VQ state that is being underused.
#
# Example:
# If true state 2 occurs 5% of the time and VQ state 2
# is also used ~5%, low usage may simply reflect the data.
# ============================================================

true_frequency = (
    np.bincount(
        true_best,
        minlength=4,
    )
    / len(true_best)
)

vq_frequency = (
    np.bincount(
        vq_best,
        minlength=4,
    )
    / len(vq_best)
)

print()
print("=" * 70)
print("STATE FREQUENCIES")
print("=" * 70)

for state in range(4):
    print(
        f"State {state} | "
        f"True={true_frequency[state]:.3%} | "
        f"VQ={vq_frequency[state]:.3%}"
    )


# ============================================================
# 3. Per-state recall
#
# What are we trying to find out?
# → How well can VQ-VAE recognize EACH individual state?
#
# Overall accuracy can hide a weak state.
#
# Example:
# State 0 → 95% recall
# State 1 → 80% recall
# State 2 → 10% recall
# State 3 → 85% recall
#
# This tells us that state 2 is difficult for the model,
# even if overall accuracy looks acceptable.
#
# recall =
# correctly identified observations of that state
# --------------------------------------------------
# total observations of that state
# ============================================================

print()
print("=" * 70)
print("PER-STATE RECALL")
print("=" * 70)

for state in range(4):

    true_count = np.sum(
        true_best == state
    )

    correct_count = np.sum(
        (true_best == state)
        & (vq_best == state)
    )

    recall = correct_count / true_count

    print(
        f"State {state} | "
        f"correct={correct_count:>4} | "
        f"total={true_count:>4} | "
        f"recall={recall:.3%}"
    )


# ============================================================
# 4. True vs VQ state sequence
#
# What are we trying to find out?
# → Does VQ change state when the underlying HMM changes?
#
# This lets us inspect the TIMING of state changes.
#
# Too many rapid VQ changes while the true state is stable
# suggest the model is sensitive to noise / flickering.
#
# Too few VQ changes suggest the model is failing to detect
# genuine state transitions.
#
# The transition count gives a summary; the sequence plot
# shows us exactly where those differences happen.
# ============================================================

plt.figure(figsize=(14, 5))

plt.step(
    np.arange(len(true_best)),
    true_best,
    where="post",
    label="True HMM state",
)

plt.step(
    np.arange(len(vq_best)),
    vq_best,
    where="post",
    label="VQ state",
    alpha=0.7,
)

plt.yticks(range(4))
plt.xlabel("Bar")
plt.ylabel("State")
plt.title("True vs VQ Latent State Sequence")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 5. Confusion matrix visualization
#
# Same information as the table above, but visualized so
# large concentrations of predictions are immediately visible.
#
# A strong diagonal = good state separation.
# Large off-diagonal blocks = state confusion.
# One dominant column = possible over-reliance on one VQ state.
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(confusion)

plt.colorbar(
    label="Count"
)

plt.xticks(
    range(4),
    [f"VQ {i}" for i in range(4)],
)

plt.yticks(
    range(4),
    [f"True {i}" for i in range(4)],
)

plt.xlabel("VQ predicted state")
plt.ylabel("True HMM state")
plt.title("Confusion Matrix")

for i in range(4):
    for j in range(4):
        plt.text(
            j,
            i,
            confusion[i, j],
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.show()


# ============================================================
# 6. State frequency comparison
#
# What are we trying to find out?
# → Does the VQ-VAE use the states in roughly the same
#   proportions as the underlying HMM?
#
# This is mainly a diagnostic for state usage imbalance.
#
# It does NOT tell us whether transitions happened at the
# correct times — that is what the sequence/transition
# diagnostic above is for.
# ============================================================

x = np.arange(4)
width = 0.35

plt.figure(figsize=(7, 5))

plt.bar(
    x - width / 2,
    true_frequency,
    width,
    label="True HMM",
)

plt.bar(
    x + width / 2,
    vq_frequency,
    width,
    label="VQ",
)

plt.xticks(
    x,
    [f"State {i}" for i in range(4)],
)

plt.ylabel("Frequency")
plt.title("True vs VQ State Frequency")

plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# MARKET-SPACE SIMILARITY TEST
# ============================================================
#
# VQ state is used ONLY as a grouping label.
#
# Similarity is measured on the ACTUAL MARKET WINDOW.
#
# Change the metric here for ablation testing:
#
#     "cosine"
#     "mahalanobis"
#     "wasserstein"
#     "ts_ss"
#
# ============================================================


window_size = best["config"]["window_size"]

# ------------------------------------------------------------
# Construct actual market windows.
# ------------------------------------------------------------

market_windows = np.asarray([
    X[t - window_size + 1:t + 1]
    for t in range(
        window_size - 1,
        len(X),
    )
])

market_windows = market_windows.reshape(
    len(market_windows),
    -1,
)

# ------------------------------------------------------------
# Sanity check.
# ------------------------------------------------------------

if len(market_windows) != len(best["latent_state"]):
    raise RuntimeError(
        "Market windows and VQ states are misaligned: "
        f"{len(market_windows)} market windows vs "
        f"{len(best['latent_state'])} VQ states."
    )

true_market_state = best["true_state"]
vq_market_state = best["latent_state"]

rng = np.random.default_rng(seed)

max_samples_per_state = 200


def _sample_group_indices(labels, state):
    indices = np.where(
        labels == state
    )[0]

    if len(indices) > max_samples_per_state:
        indices = rng.choice(
            indices,
            size=max_samples_per_state,
            replace=False,
        )

    return indices


def _mean_within_group_similarity(
    vectors,
    metric,
):
    if len(vectors) < 2:
        return np.nan

    similarities = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarities.append(
                similarity_metric(
                    vectors[i],
                    vectors[j],
                    metric=metric,
                )
            )

    if len(similarities) == 0:
        return np.nan

    return float(
        np.mean(similarities)
    )


# ============================================================
# A. TRUE STATE → ACTUAL MARKET SIMILARITY
# ============================================================

print()
print("=" * 70)
print(
    "TRUE STATE → ACTUAL MARKET SIMILARITY"
)
print("=" * 70)

true_state_similarity = np.full(
    4,
    np.nan,
)

for state in range(4):

    indices = _sample_group_indices(
        true_market_state,
        state,
    )

    vectors = market_windows[indices]

    true_state_similarity[state] = (
        _mean_within_group_similarity(
            vectors,
            similarity_metric_name,
        )
    )

    print(
        f"True state {state} | "
        f"n={len(vectors):>3} | "
        f"mean {similarity_metric_name}="
        f"{true_state_similarity[state]:.4f}"
    )


# ============================================================
# B. VQ STATE → ACTUAL MARKET SIMILARITY
# ============================================================

print()
print("=" * 70)
print(
    "VQ STATE → ACTUAL MARKET SIMILARITY"
)
print("=" * 70)

vq_state_similarity = np.full(
    4,
    np.nan,
)

for state in range(4):

    indices = _sample_group_indices(
        vq_market_state,
        state,
    )

    vectors = market_windows[indices]

    vq_state_similarity[state] = (
        _mean_within_group_similarity(
            vectors,
            similarity_metric_name,
        )
    )

    print(
        f"VQ state {state} | "
        f"n={len(vectors):>3} | "
        f"mean {similarity_metric_name}="
        f"{vq_state_similarity[state]:.4f}"
    )


# ============================================================
# C. DIRECT COMPARISON
# ============================================================

print()
print("=" * 70)
print(
    "TRUE VS VQ MARKET-SPACE COHERENCE"
)
print("=" * 70)

print(
    f"Metric = {similarity_metric_name}"
)

print(
    "Higher = more similar market windows."
)

for state in range(4):

    true_value = true_state_similarity[state]
    vq_value = vq_state_similarity[state]

    print(
        f"State {state} | "
        f"True={true_value:.4f} | "
        f"VQ={vq_value:.4f}"
    )


# ============================================================
# D. TRUE STATE-TO-STATE MARKET SIMILARITY
# ============================================================

print()
print("=" * 70)
print(
    "TRUE STATE — ACTUAL MARKET SIMILARITY MATRIX"
)
print("=" * 70)

true_market_similarity_matrix = np.full(
    (4, 4),
    np.nan,
)

for state_a in range(4):

    indices_a = _sample_group_indices(
        true_market_state,
        state_a,
    )

    vectors_a = market_windows[indices_a]

    for state_b in range(4):

        indices_b = _sample_group_indices(
            true_market_state,
            state_b,
        )

        vectors_b = market_windows[indices_b]

        if (
            len(vectors_a) == 0
            or len(vectors_b) == 0
        ):
            continue

        similarities = []

        for vector_a in vectors_a:

            for vector_b in vectors_b:

                similarities.append(
                    similarity_metric(
                        vector_a,
                        vector_b,
                        metric=similarity_metric_name,
                    )
                )

        # Remove self-comparisons on diagonal.
        if state_a == state_b:

            values = []

            for i in range(len(vectors_a)):
                for j in range(i + 1, len(vectors_a)):
                    values.append(
                        similarity_metric(
                            vectors_a[i],
                            vectors_a[j],
                            metric=similarity_metric_name,
                        )
                    )

            similarities = values

        if len(similarities) > 0:

            true_market_similarity_matrix[
                state_a,
                state_b,
            ] = np.mean(
                similarities
            )

print(
    np.round(
        true_market_similarity_matrix,
        3,
    )
)


# ============================================================
# E. VQ STATE-TO-STATE MARKET SIMILARITY
# ============================================================

print()
print("=" * 70)
print(
    "VQ STATE — ACTUAL MARKET SIMILARITY MATRIX"
)
print("=" * 70)

vq_market_similarity_matrix = np.full(
    (4, 4),
    np.nan,
)

for state_a in range(4):

    indices_a = _sample_group_indices(
        vq_market_state,
        state_a,
    )

    vectors_a = market_windows[indices_a]

    for state_b in range(4):

        indices_b = _sample_group_indices(
            vq_market_state,
            state_b,
        )

        vectors_b = market_windows[indices_b]

        if (
            len(vectors_a) == 0
            or len(vectors_b) == 0
        ):
            continue

        similarities = []

        if state_a == state_b:

            for i in range(len(vectors_a)):
                for j in range(i + 1, len(vectors_a)):

                    similarities.append(
                        similarity_metric(
                            vectors_a[i],
                            vectors_a[j],
                            metric=similarity_metric_name,
                        )
                    )

        else:

            for vector_a in vectors_a:
                for vector_b in vectors_b:

                    similarities.append(
                        similarity_metric(
                            vector_a,
                            vector_b,
                            metric=similarity_metric_name,
                        )
                    )

        if len(similarities) > 0:

            vq_market_similarity_matrix[
                state_a,
                state_b,
            ] = np.mean(
                similarities
            )

print(
    np.round(
        vq_market_similarity_matrix,
        3,
    )
)

"""

======================================================================
BEST CONFIGURATION
======================================================================
Window size         : 32
Latent dimension    : 32
Hidden dimension    : 256
State accuracy      : 43.651%
True transitions    : 87
VQ transitions      : 46
Quantization error  : 3.746241
Final loss          : 0.984449
Codebook usage      : [0.073 0.186 0.051 0.69 ]

======================================================================
ALL CONFIGURATIONS
======================================================================
window=32 | latent=32 | hidden=256 | accuracy=43.651% | transitions= 46 | q_error=3.7462 | loss=0.9844
window=32 | latent=16 | hidden=128 | accuracy=41.095% | transitions= 58 | q_error=1.6960 | loss=0.9953
window=64 | latent= 8 | hidden=128 | accuracy=38.850% | transitions= 48 | q_error=0.5637 | loss=0.9628
window=32 | latent= 8 | hidden=128 | accuracy=38.499% | transitions= 54 | q_error=0.8148 | loss=0.9945
window=64 | latent=16 | hidden=128 | accuracy=31.396% | transitions= 13 | q_error=0.3005 | loss=0.9448
window=16 | latent=16 | hidden=128 | accuracy=30.752% | transitions=  4 | q_error=0.1824 | loss=0.9816
window=16 | latent= 8 | hidden=128 | accuracy=28.566% | transitions=  0 | q_error=0.0005 | loss=0.9989

======================================================================
CONFUSION MATRIX
======================================================================
Rows = true state | Columns = VQ state
[[1154   27    7    0]
 [ 263  788  346   27]
 [1110   58    0    0]
 [ 902   52    8  227]]

======================================================================
STATE FREQUENCIES
======================================================================
State 0 | True=23.908% | VQ=69.008%
State 1 | True=28.658% | VQ=18.615%
State 2 | True=23.506% | VQ=7.265%
State 3 | True=23.928% | VQ=5.112%

======================================================================
PER-STATE RECALL
======================================================================
State 0 | correct=1154 | total=1188 | recall=97.138%
State 1 | correct= 788 | total=1424 | recall=55.337%
State 2 | correct=   0 | total=1168 | recall=0.000%
State 3 | correct= 227 | total=1189 | recall=19.092%

Noteable problems:
- Model is too reliant on VQ state 0. It learned nothing about state 2.
- State 0-3 is actually quite balanced but the VQ-VAE used 69.008% for state 0 and everything else underweighted.
- underswitching is a problem (46 vs 87)

Result from PCA test and KNN:
VQ accuracy = 52.3%
k-NN on z_e (chronological holdout) = 84.9%

With n_neighbours 15, it shows that its not overfitting to lookalikes so encoder genuinely
learned how to represent state information. Meaning this is a codebook/quantization problem
- PCA state shows that state 0 and 3 overlap based on variance only. However state distinction might
live in feature z6 for example, but it doesn't vary much across all observation.
- KNN looks at the entire vector z_e shows there is substantial information in z_e representation that
could distinguish the states.

Since K-NN method can extract 84.9% from z_e but VQ only 52.3%, the problem isn't in state information of z_e
but rather 4-code quantization is failing to preserve it. Proposed solution:
1. Use K-means to initialize VQ codebook rather than starting with rand values (-1,1). only during training.
When running live, use VQ-VAE to update the codebook, making z_e and codebook prototype closer and reducing initial 
q_error.
2. Use EMA instead of gradient descent which would make it less adaptive but more stable.
3. Track every epoch.

======================================================================
ALL CONFIGURATIONS
======================================================================
window=32 | latent= 8 | hidden=128 | accuracy=76.555% | transitions= 92 | q_error=1.5173 | loss=0.8571
window=16 | latent= 8 | hidden=128 | accuracy=71.795% | transitions= 95 | q_error=1.6687 | loss=0.8330
window=32 | latent=16 | hidden=128 | accuracy=54.578% | transitions= 85 | q_error=2.2447 | loss=0.8583
window=16 | latent=16 | hidden=128 | accuracy=40.863% | transitions= 43 | q_error=2.1127 | loss=0.8751
window=64 | latent=16 | hidden=128 | accuracy=38.687% | transitions= 41 | q_error=1.3246 | loss=0.8896
window=64 | latent= 8 | hidden=128 | accuracy=38.242% | transitions= 33 | q_error=0.5588 | loss=0.8849
window=32 | latent=32 | hidden=256 | accuracy=37.533% | transitions= 20 | q_error=2.7780 | loss=0.8891

======================================================================
ALL CONFIGURATIONS
======================================================================
window=16 | latent= 8 | hidden=128 | accuracy=85.938% | transitions=101 | q_error=1.1714 | loss=0.8365
window=32 | latent= 8 | hidden=128 | accuracy=76.857% | transitions= 88 | q_error=1.1290 | loss=0.8644
window=64 | latent=16 | hidden=128 | accuracy=61.171% | transitions= 63 | q_error=1.9286 | loss=0.9105
window=32 | latent=16 | hidden=128 | accuracy=59.771% | transitions= 83 | q_error=2.1562 | loss=0.8854
window=32 | latent=32 | hidden=256 | accuracy=57.235% | transitions= 61 | q_error=2.7422 | loss=0.9069
window=16 | latent=16 | hidden=128 | accuracy=56.951% | transitions=116 | q_error=1.2832 | loss=0.8588
window=64 | latent= 8 | hidden=128 | accuracy=47.276% | transitions= 30 | q_error=0.6448 | loss=0.9154

======================================================================
ALL CONFIGURATIONS
======================================================================
window=16 | latent= 8 | hidden=128 | accuracy=75.707% | transitions=106 | q_error=1.3577 | loss=0.8322
window=16 | latent=16 | hidden=128 | accuracy=73.561% | transitions=108 | q_error=2.3585 | loss=0.8358
window=32 | latent= 8 | hidden=128 | accuracy=65.446% | transitions= 87 | q_error=1.1116 | loss=0.8605
window=32 | latent=16 | hidden=128 | accuracy=65.265% | transitions= 83 | q_error=2.1777 | loss=0.8641
window=64 | latent= 8 | hidden=128 | accuracy=57.667% | transitions= 69 | q_error=0.8312 | loss=0.8959
window=32 | latent=32 | hidden=256 | accuracy=41.578% | transitions= 45 | q_error=2.7119 | loss=0.8908
window=64 | latent=16 | hidden=128 | accuracy=41.381% | transitions= 96 | q_error=1.0408 | loss=0.9112

Verdict: Green flag
- Higher latent dimensionality tends to be associated with lower state accuracy. Preferably, small, compact latent
representation and relatively short market window.
- 16x8 or 32x8 configurations performs the best. Consider using it as a baseline
- Transition counts are encouraging. q_error or total loss shouldn't determine model yet.
- Cosine similarity returned a positive result where recurring latent states e_k returned significantly higher score
over other latent states.

True strategy_system test:

Method 1:
- In the beginning, brute force market windows that returns similarity scores placed into buckets [90% similarity, 
80% similarity, 70% similarity, ...]. Check if underlying strategy's backtested edge associates with similarity
score. Perform ablation test on the 4 candidates of similarity scores
- Label the brute forced market windows. Encode these latent states into VQ-VAE and store it into the codebook to keep
VQ-VAE static. Since these market windows have been pre-validated to have high similarity scores, we don't have to concern
ourselves with recurring market states having high similarity scores (Provided latent null is implemented such that
it knows how to isolate regimes, isolating the problem down into a model-only problem)

Method 2:
- Use VQ-VAE to perform the labelling rather than brute forcing market windows. Simpler, faster, but doesn't guarantee
similarity scores being high. By focusing on the model becoming great in identifying latent states with high similarity
scores, it becomes intelligent enough to develop its own states. But this might be a harder problem to solve. Possible
solutions:
    - In training and wf, use cosine similarity to eliminate regimes with low recurring similarity scores (keep 90%).

Problems in Live Market: 

State similarity score != edge expectancy. We cannot change the objective of the model (for example to high expectancy
regimes) because it will cause the model to overfit to the regimes. Unless our strategy trades in a distinct environment. Change the objcetive
from purely state similarity with
    - Strategy-based latent states: where strategy_k fulfills expectancy > friction, set that as latent_k. 
    VQ-VAE could still identify the latent states provided that strategy_k relies on a persistent condition to have
    positive expectancy (Simplest method)
    - Add TCN: Strategy's edge may depend on temporal patterns and causal TCN could encode these patterns into latent
    states more consistently than MLP
    - Increasing latent dimensionality: Increases the representational capacity  of the feature such that similarity score becomes more meaningful,
    at the cost of VQ-VAE accuracy. However this must be tested strictly.

VQ-VAE identification is still probabilistic and may still decay. Though VQ-VAE is used only as a regime detection tool,
the overall MoE system might still suffer from overfitting problems. 
    - Cross validating different experts to different latent states is important. Ensure they could perform well even when model misidentifies them.
    - This means reducing tightness of experts (Keep parameters shallow to avoid overfitting)

After testing with a more stochastic data, its observed that vq-vae isn't that good for interpreting regimes. Might have change models for this.    

Doesn't seem to work on stochastic data. VQ-VAE might not be the right tool for this (TCN, etc)
- Turn vq-vae to a supervised learner. use brute forcing to extract windows where objective is fulfilled (Objectve = similarity score such that correlates with expectancy)
- If that doesn't work either, move to other method or cancel this project
"""