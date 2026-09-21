
from itertools import permutations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.cluster import KMeans

import numpy as np

from strategy_system.strategy.MoE.latent.diagnostic.similarity import similarity_metric


def informative_temporality(X, df, true_state, seed):
    """
    Args:
        X -> Pandas df of features
        df -> Synthetic data to interpret
        true_state -> Correct latent states in synthetic data

    Function to test for the entropy of states in data, use this to test
    if the data itself has information on each state and if VQ-VAE is
    failing because information exhibits temporal behavior.
    """

    # ============================================================
    # STEP 0 — INFORMATION CEILING
    # ============================================================
    #
    # Before touching VQ-VAE, determine how much state information
    # is actually present in the synthetic observations.
    #
    # We test:
    #
    #   0A. Individual latent observable factors
    #   0B. latent observable factors combined
    #   0C. actual noisy 50-feature representation
    #   0D. temporal context from the actual 50 features
    #
    # Everything is chronological.
    # ============================================================

    latent_columns = [
        f"latent_observable_{i + 1}"
        for i in range(8)
    ]

    latent_observable = df[latent_columns].values

    split = int(len(df) * 0.8)

    X_latent_train = latent_observable[:split]
    X_latent_test = latent_observable[split:]

    X_raw_train = X[:split]
    X_raw_test = X[split:]

    y_train = true_state[:split]
    y_test = true_state[split:]


    # ============================================================
    # 0A. INDIVIDUAL LATENT FACTOR TESTS
    # ============================================================

    print()
    print("=" * 70)
    print("STEP 0A — INDIVIDUAL LATENT FACTOR INFORMATION")
    print("=" * 70)

    for feature_idx in range(8):

        feature_train = X_latent_train[:, feature_idx].reshape(-1, 1)
        feature_test = X_latent_test[:, feature_idx].reshape(-1, 1)

        # --------------------------------------------------------
        # One-vs-rest state-2 classifier
        # --------------------------------------------------------

        y2_train = (y_train == 2).astype(int)
        y2_test = (y_test == 2).astype(int)

        logistic_state2 = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=seed,
            ),
        )

        logistic_state2.fit(
            feature_train,
            y2_train,
        )

        state2_probability = logistic_state2.predict_proba(
            feature_test
        )[:, 1]

        state2_prediction = (
            state2_probability >= 0.5
        ).astype(int)

        auc = roc_auc_score(
            y2_test,
            state2_probability,
        )

        accuracy = accuracy_score(
            y2_test,
            state2_prediction,
        )

        balanced_accuracy = balanced_accuracy_score(
            y2_test,
            state2_prediction,
        )

        print(
            f"latent_{feature_idx + 1:>2} | "
            f"state2 AUC={auc:.3f} | "
            f"accuracy={accuracy:.3f} | "
            f"balanced_acc={balanced_accuracy:.3f}"
        )

    # ============================================================
    # STEP 0B — ALL 8 LATENT OBSERVABLE FACTORS
    # ============================================================

    print()
    print("=" * 70)
    print("STEP 0B — ALL LATENT OBSERVABLE FACTORS")
    print("=" * 70)

    latent_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )

    latent_rf.fit(
        X_latent_train,
        y_train,
    )

    latent_prediction = latent_rf.predict(
        X_latent_test
    )

    latent_accuracy = accuracy_score(
        y_test,
        latent_prediction,
    )

    latent_balanced_accuracy = balanced_accuracy_score(
        y_test,
        latent_prediction,
    )

    print(
        f"RF accuracy            : "
        f"{latent_accuracy:.3%}"
    )

    print(
        f"RF balanced accuracy   : "
        f"{latent_balanced_accuracy:.3%}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            latent_prediction,
        )
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            latent_prediction,
            digits=3,
        )
    )

    print()
    print(
        "Feature importance:"
    )

    for i, importance in enumerate(
        latent_rf.feature_importances_
    ):
        print(
            f"latent_{i + 1:>2} : "
            f"{importance:.4f}"
        )
    # ============================================================
    # STEP 0C — ACTUAL 50 NOISY OBSERVABLE FEATURES
    # ============================================================

    print()
    print("=" * 70)
    print("STEP 0C — ACTUAL 50 FEATURE INFORMATION")
    print("=" * 70)

    raw_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )

    raw_rf.fit(
        X_raw_train,
        y_train,
    )

    raw_prediction = raw_rf.predict(
        X_raw_test
    )

    raw_accuracy = accuracy_score(
        y_test,
        raw_prediction,
    )

    raw_balanced_accuracy = balanced_accuracy_score(
        y_test,
        raw_prediction,
    )

    print(
        f"RF accuracy            : "
        f"{raw_accuracy:.3%}"
    )

    print(
        f"RF balanced accuracy   : "
        f"{raw_balanced_accuracy:.3%}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            raw_prediction,
        )
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            raw_prediction,
            digits=3,
        )
    )

    # ============================================================
    # STEP 0D — TEMPORAL INFORMATION TEST
    # ============================================================
    #
    # Use only the actual observable 50 features.
    #
    # At time t, the model receives:
    #
    #   X[t-L+1 : t]
    #
    # Therefore there is no future information.
    #
    # This asks:
    #
    #   "Does temporal context make the hidden state much easier
    #    to identify?"
    #
    # Especially important for state 2, where the defining property
    # is mean reversion.
    # ============================================================

    temporal_window = 16

    temporal_X = []
    temporal_y = []

    for t in range(
        temporal_window - 1,
        len(X),
    ):

        window = X[
            t - temporal_window + 1:
            t + 1
        ]

        temporal_X.append(
            window.reshape(-1)
        )

        temporal_y.append(
            true_state[t]
        )

    temporal_X = np.asarray(
        temporal_X
    )

    temporal_y = np.asarray(
        temporal_y
    )

    temporal_split = int(
        len(temporal_X) * 0.8
    )

    temporal_X_train = (
        temporal_X[:temporal_split]
    )

    temporal_X_test = (
        temporal_X[temporal_split:]
    )

    temporal_y_train = (
        temporal_y[:temporal_split]
    )

    temporal_y_test = (
        temporal_y[temporal_split:]
    )

    print()
    print("=" * 70)
    print("STEP 0D — TEMPORAL INFORMATION")
    print("=" * 70)

    temporal_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )

    temporal_rf.fit(
        temporal_X_train,
        temporal_y_train,
    )

    temporal_prediction = temporal_rf.predict(
        temporal_X_test
    )

    temporal_accuracy = accuracy_score(
        temporal_y_test,
        temporal_prediction,
    )

    temporal_balanced_accuracy = (
        balanced_accuracy_score(
            temporal_y_test,
            temporal_prediction,
        )
    )

    print(
        f"Temporal RF accuracy          : "
        f"{temporal_accuracy:.3%}"
    )

    print(
        f"Temporal RF balanced accuracy : "
        f"{temporal_balanced_accuracy:.3%}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            temporal_y_test,
            temporal_prediction,
        )
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            temporal_y_test,
            temporal_prediction,
            digits=3,
        )
    )

    # ============================================================
    # STEP 0E — STATE 2 RECOVERABILITY
    # ============================================================

    print()
    print("=" * 70)
    print("STEP 0E — STATE 2 RECOVERABILITY")
    print("=" * 70)

    # ------------------------------------------------------------
    # Current-feature RF
    # ------------------------------------------------------------

    state2_train = (
        y_train == 2
    ).astype(int)

    state2_test = (
        y_test == 2
    ).astype(int)

    state2_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )

    state2_rf.fit(
        X_raw_train,
        state2_train,
    )

    state2_probability = (
        state2_rf.predict_proba(
            X_raw_test
        )[:, 1]
    )

    state2_prediction = (
        state2_probability >= 0.5
    ).astype(int)

    print(
        f"Current features, state-2 AUC : "
        f"{roc_auc_score(state2_test, state2_probability):.3%}"
    )

    print(
        f"Current features, state-2 acc : "
        f"{accuracy_score(state2_test, state2_prediction):.3%}"
    )

    print(
        f"Current features, state-2 bal : "
        f"{balanced_accuracy_score(state2_test, state2_prediction):.3%}"
    )

    # ------------------------------------------------------------
    # Temporal RF
    # ------------------------------------------------------------

    temporal_state2_train = (
        temporal_y_train == 2
    ).astype(int)

    temporal_state2_test = (
        temporal_y_test == 2
    ).astype(int)

    temporal_state2_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )

    temporal_state2_rf.fit(
        temporal_X_train,
        temporal_state2_train,
    )

    temporal_state2_probability = (
        temporal_state2_rf.predict_proba(
            temporal_X_test
        )[:, 1]
    )

    temporal_state2_prediction = (
        temporal_state2_probability >= 0.5
    ).astype(int)

    print(
        f"Temporal features, state-2 AUC : "
        f"{roc_auc_score(temporal_state2_test, temporal_state2_probability):.3%}"
    )

    print(
        f"Temporal features, state-2 acc : "
        f"{accuracy_score(temporal_state2_test, temporal_state2_prediction):.3%}"
    )

    print(
        f"Temporal features, state-2 bal : "
        f"{balanced_accuracy_score(temporal_state2_test, temporal_state2_prediction):.3%}"
    )

"""
informative_temporality result

- State 2 is observable — it is not missing from the features.
- Raw 50 features recover state 2 with 75.9% AUC.
- Adding 16-bar temporal context improves state-2 AUC to 85.2% → temporal information matters.
- All 8 clean latent factors recover states at 87.8% accuracy / 80.2% balanced accuracy.
- Your current VQ-VAE performs much worse than the information available in the data.
- z_e → kNN at 84.9% shows the encoder contains strong state information.
- Therefore the main suspect is VQ discretization/codebook geometry, not lack of state information.
- State 2 failure is not simply a feature-information failure.
- Next test: z_e → KMeans. This tells us whether VQ itself is destroying good latent clustering or whether the latent information is nonlinear/non-clustered.

Reasoning
- Reconstruction objective: VQ-VAE is rewarded to reconstruct  X->z_e->z_q->X^ but not necessarily
for making 4 codes correspond to the 4 hidden states. So if drift + volatility explains most of the input,
the model can only use the codes to represent those dimensions while mixing states that have similar
observable behaviour.
- Codebook got trapped compressing into one of the nearest 4 points where the prototypes can't truly divide it
to 4 states.
- Encoder's latent geometry failed to form 4 clean clusters. K-means helped distinguish nonlinear regions but the
nearest prototype VQ cannot.
- K-means initialization runs on randomly-initialized encoder. Initial codebook starts in places that makes sense
for random encoder but not for trained encoder (Not necessarily the problem)
- EMA doesn't prevent the collapse. It just makes it stable.
- Commitment/reconstruction balance with beta=0.25 means to make reconstruction good while staying reasonably
close to assigned code. If reconstruction benefits from continuous latent representation, encoder may not naturally
organize itself around four well-separated prototypes.
"""
def _find_best_mapping(
    cluster_labels: np.ndarray,
    true_state: np.ndarray,
    n_clusters: int,
    n_states: int,
) -> tuple[np.ndarray, float]:
    """
    Find the permutation mapping from arbitrary KMeans cluster IDs
    to true state IDs that maximizes TRAIN accuracy.

    Example:

        cluster 0 -> state 2
        cluster 1 -> state 0
        cluster 2 -> state 3
        cluster 3 -> state 1

    Parameters
    ----------
    cluster_labels:
        KMeans cluster IDs.
    true_state:
        Ground-truth state labels.
    n_clusters:
        Number of KMeans clusters.
    n_states:
        Number of true states.

    Returns
    -------
    best_mapping:
        Array where mapping[cluster_id] = mapped_state_id.
    best_accuracy:
        Accuracy achieved on the data used to find the mapping.
    """

    if n_clusters != n_states:
        raise ValueError(
            "This permutation-based alignment requires "
            "n_clusters == n_states."
        )

    best_accuracy = -1.0
    best_mapping = None

    for permutation in permutations(range(n_states)):
        mapping = np.asarray(
            permutation,
            dtype=np.int64,
        )

        mapped = mapping[cluster_labels]

        accuracy = np.mean(
            mapped == true_state
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping.copy()

    if best_mapping is None:
        raise RuntimeError(
            "Failed to find a cluster-to-state mapping."
        )

    return best_mapping, best_accuracy


"""
Hypothesis: Latent Clusterability

Question:
    Does the continuous encoder representation z_e naturally contain
    four clusterable regions corresponding to the four true synthetic
    market states?

Experimental logic:
    z_e -> KMeans(4) -> permutation alignment -> OOS evaluation

Important:
    - KMeans is fit on TRAIN z_e only.
    - The cluster-to-true-state mapping is learned on TRAIN only.
    - The same frozen KMeans + mapping is applied to TEST.
    - No true-state information is given to KMeans.
"""

def test_latent_clusterability(
    latent_z: np.ndarray,
    true_state: np.ndarray,
    n_clusters: int = 4,
    train_frac: float = 0.8,
    seed: int = 84,
    n_init: int = 20,
    normalize_latent: bool = True,
) -> dict:
    """
    Test whether the continuous latent representation z_e is
    naturally clusterable into the expected number of states.

    Parameters
    ----------
    latent_z:
        Continuous encoder representation.

        Shape:
            (n_samples, latent_dim)

    true_state:
        Ground-truth synthetic state for each latent vector.

        Shape:
            (n_samples,)

    n_clusters:
        Number of KMeans clusters.
        Expected to be 4 for the current synthetic generator.

    train_frac:
        Chronological fraction used for KMeans fitting and
        cluster-to-state mapping.

    seed:
        Random seed for KMeans.

    n_init:
        Number of KMeans initializations.

    Returns
    -------
    results:
        Dictionary containing train/test metrics, mapping,
        confusion matrix, cluster usage, and predictions.

    Interpretation
    --------------
    KMeans(test) approximately equal to VQ:
        The continuous latent geometry itself may not contain
        cleanly separated four-state clusters.

    KMeans(test) substantially better than VQ:
        The encoder contains clusterable state structure, but
        the VQ codebook/quantization is failing to preserve it.

    Intermediate result:
        Some states may form clean clusters while others overlap.
        This is especially relevant for the synthetic state-2/state-3
        ambiguity.
    """

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    latent_z = np.asarray(
        latent_z,
        dtype=np.float64,
    )

    true_state = np.asarray(
        true_state,
        dtype=np.int64,
    )

    if latent_z.ndim != 2:
        raise ValueError(
            "latent_z must be a 2D array with shape "
            "(n_samples, latent_dim)."
        )

    if true_state.ndim != 1:
        raise ValueError(
            "true_state must be a 1D array."
        )

    if len(latent_z) != len(true_state):
        raise ValueError(
            "latent_z and true_state must contain the "
            "same number of samples."
        )

    if len(latent_z) < 10:
        raise ValueError(
            "Not enough samples for clusterability testing."
        )

    if not 0.0 < train_frac < 1.0:
        raise ValueError(
            "train_frac must be between 0 and 1."
        )

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    # ============================================================
    # CHRONOLOGICAL SPLIT
    # ============================================================

    split = int(
        len(latent_z) * train_frac
    )

    if split <= 0 or split >= len(latent_z):
        raise ValueError(
            "train_frac produces an empty train or test set."
        )

    z_train = latent_z[:split]
    z_test = latent_z[split:]

    state_train = true_state[:split]
    state_test = true_state[split:]

    # ============================================================
    # OPTIONAL: NORMALIZE LATENT VECTORS
    # ============================================================
    # Unit-normalized vectors make Euclidean distance equivalent
    # to cosine distance:
    #
    #   ||u - v||^2 = 2(1 - cosine_similarity(u, v))
    #
    # Therefore this tests whether latent states are separated
    # primarily by direction rather than magnitude.
    # ============================================================

    if normalize_latent:

        train_norm = np.linalg.norm(
            z_train,
            axis=1,
            keepdims=True,
        )

        test_norm = np.linalg.norm(
            z_test,
            axis=1,
            keepdims=True,
        )

        train_norm = np.maximum(
            train_norm,
            1e-12,
        )

        test_norm = np.maximum(
            test_norm,
            1e-12,
        )

        z_train = z_train / train_norm
        z_test = z_test / test_norm

    n_states = len(
        np.unique(true_state)
    )

    if n_states != n_clusters:
        raise ValueError(
            f"Expected the number of true states "
            f"({n_states}) to equal n_clusters "
            f"({n_clusters}) for permutation alignment."
        )

    # ============================================================
    # FIT KMEANS ON TRAIN ONLY
    # ============================================================

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=n_init,
    )

    train_clusters = kmeans.fit_predict(
        z_train
    )

    # ============================================================
    # LEARN CLUSTER -> TRUE STATE MAPPING ON TRAIN ONLY
    # ============================================================

    best_mapping, train_mapping_accuracy = (
        _find_best_mapping(
            cluster_labels=train_clusters,
            true_state=state_train,
            n_clusters=n_clusters,
            n_states=n_states,
        )
    )

    # ============================================================
    # APPLY FROZEN KMEANS + FROZEN MAPPING TO TEST
    # ============================================================

    test_clusters = kmeans.predict(
        z_test
    )

    mapped_train = (
        best_mapping[train_clusters]
    )

    mapped_test = (
        best_mapping[test_clusters]
    )

    # ============================================================
    # METRICS
    # ============================================================

    train_accuracy = accuracy_score(
        state_train,
        mapped_train,
    )

    test_accuracy = accuracy_score(
        state_test,
        mapped_test,
    )

    train_balanced_accuracy = (
        balanced_accuracy_score(
            state_train,
            mapped_train,
        )
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            state_test,
            mapped_test,
        )
    )

    test_confusion = confusion_matrix(
        state_test,
        mapped_test,
        labels=np.arange(n_states),
    )

    # ============================================================
    # PER-STATE RECALL
    # ============================================================

    per_state_recall = {}

    for state in range(n_states):

        total = np.sum(
            state_test == state
        )

        correct = np.sum(
            (state_test == state)
            & (mapped_test == state)
        )

        if total == 0:
            recall = np.nan
        else:
            recall = correct / total

        per_state_recall[state] = float(
            recall
        )

    # ============================================================
    # CLUSTER USAGE
    # ============================================================

    train_cluster_usage = (
        np.bincount(
            train_clusters,
            minlength=n_clusters,
        )
        / len(train_clusters)
    )

    test_cluster_usage = (
        np.bincount(
            test_clusters,
            minlength=n_clusters,
        )
        / len(test_clusters)
    )

    # ============================================================
    # MAPPING TABLE
    # ============================================================

    mapping = {
        int(cluster_id): int(
            best_mapping[cluster_id]
        )
        for cluster_id in range(n_clusters)
    }

    # ============================================================
    # REPORT
    # ============================================================

    metric_name = (
        "COSINE / NORMALIZED EUCLIDEAN"
        if normalize_latent
        else "EUCLIDEAN"
    )

    print()
    print("=" * 70)
    print(
        f"LATENT CLUSTERABILITY — KMEANS(z_e) [{metric_name}]"
    )
    print("=" * 70)

    print()
    print(
        f"Samples             : {len(latent_z)}"
    )

    print(
        f"Latent dimension    : {latent_z.shape[1]}"
    )

    print(
        f"Train samples       : {len(z_train)}"
    )

    print(
        f"Test samples        : {len(z_test)}"
    )

    print(
        f"KMeans clusters     : {n_clusters}"
    )

    print()

    print(
        "Cluster -> true-state mapping:"
    )

    for cluster_id in range(n_clusters):
        print(
            f"  KMeans {cluster_id} -> "
            f"True state {best_mapping[cluster_id]}"
        )

    print()

    print(
        f"Train accuracy      : "
        f"{train_accuracy:.3%}"
    )

    print(
        f"Train balanced acc  : "
        f"{train_balanced_accuracy:.3%}"
    )

    print()

    print(
        f"Test accuracy       : "
        f"{test_accuracy:.3%}"
    )

    print(
        f"Test balanced acc   : "
        f"{test_balanced_accuracy:.3%}"
    )

    print()

    print("Test confusion matrix:")
    print(
        test_confusion
    )

    print()

    print("Per-state recall:")
    for state in range(n_states):
        print(
            f"  State {state}: "
            f"{per_state_recall[state]:.3%}"
        )

    print()

    print("Train cluster usage:")
    print(
        np.round(
            train_cluster_usage,
            4,
        )
    )

    print()

    print("Test cluster usage:")
    print(
        np.round(
            test_cluster_usage,
            4,
        )
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            state_test,
            mapped_test,
            labels=np.arange(n_states),
            digits=3,
            zero_division=0,
        )
    )

    # ============================================================
    # RETURN RESULTS
    # ============================================================

    return {
        "kmeans": kmeans,
        "mapping": mapping,

        "train_accuracy": float(
            train_accuracy
        ),

        "test_accuracy": float(
            test_accuracy
        ),

        "train_balanced_accuracy": float(
            train_balanced_accuracy
        ),

        "test_balanced_accuracy": float(
            test_balanced_accuracy
        ),

        "test_confusion_matrix": (
            test_confusion
        ),

        "per_state_recall": (
            per_state_recall
        ),

        "train_cluster_usage": (
            train_cluster_usage
        ),

        "test_cluster_usage": (
            test_cluster_usage
        ),

        "train_clusters": (
            train_clusters
        ),

        "test_clusters": (
            test_clusters
        ),

        "mapped_train_state": (
            mapped_train
        ),

        "mapped_test_state": (
            mapped_test
        ),

        "true_train_state": (
            state_train
        ),

        "true_test_state": (
            state_test
        ),

        "cluster_centers": (
            kmeans.cluster_centers_
        ),

        "train_mapping_accuracy": float(
            train_mapping_accuracy
        ),
    }

def find_best_similarity_model(
    X,
    results,
    metric="ts_ss",
    max_samples_per_state=200,
    seed=42,
):
    """
    Find the VQ model whose states have the highest
    within-state market-space similarity on the diagonal.

    For similarity metrics:
        higher diagonal mean = better

    For distance metrics:
        lower diagonal mean = better

    Returns:
        best_result
        all_results
    """

    rng = np.random.default_rng(seed)

    # Metrics where lower = more similar.
    distance_metrics = {
        "wasserstein",
    }

    model_scores = []

    for result in results:

        window_size = result["config"]["window_size"]
        vq_state = result["latent_state"]

        # --------------------------------------------------------
        # Construct actual market windows
        # --------------------------------------------------------

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

        if len(market_windows) != len(vq_state):
            raise RuntimeError(
                "Market windows and VQ states are misaligned: "
                f"{len(market_windows)} market windows vs "
                f"{len(vq_state)} VQ states."
            )

        # --------------------------------------------------------
        # Sample each VQ state
        # --------------------------------------------------------

        state_vectors = {}

        for state in range(4):

            indices = np.where(
                vq_state == state
            )[0]

            if len(indices) > max_samples_per_state:
                indices = rng.choice(
                    indices,
                    size=max_samples_per_state,
                    replace=False,
                )

            state_vectors[state] = market_windows[indices]

        # --------------------------------------------------------
        # Diagonal similarity
        # --------------------------------------------------------

        diagonal_scores = []

        for state in range(4):

            vectors = state_vectors[state]

            if len(vectors) < 2:
                diagonal_scores.append(np.nan)
                continue

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
                diagonal_scores.append(np.nan)
            else:
                diagonal_scores.append(
                    float(np.mean(similarities))
                )

        diagonal_scores = np.asarray(
            diagonal_scores,
            dtype=float,
        )

        valid_scores = diagonal_scores[
            ~np.isnan(diagonal_scores)
        ]

        if len(valid_scores) == 0:
            overall_score = np.nan
        else:
            overall_score = float(
                np.mean(valid_scores)
            )

        model_scores.append({
            "config": result["config"],
            "window_size": window_size,
            "metric": metric,
            "diagonal_scores": diagonal_scores,
            "mean_diagonal_similarity": overall_score,
        })

    # ------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------

    if metric in distance_metrics:
        # Lower = more similar
        best_result = min(
            model_scores,
            key=lambda x: (
                np.inf
                if np.isnan(
                    x["mean_diagonal_similarity"]
                )
                else x["mean_diagonal_similarity"]
            ),
        )
    else:
        # Higher = more similar
        best_result = max(
            model_scores,
            key=lambda x: (
                -np.inf
                if np.isnan(
                    x["mean_diagonal_similarity"]
                )
                else x["mean_diagonal_similarity"]
            ),
        )

    return best_result, model_scores