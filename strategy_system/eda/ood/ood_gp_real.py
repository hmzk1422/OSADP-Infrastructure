"""BTC/USDT L2 regime discovery plus a multi-model, cross-regime evaluation.

Run this file from the package context where the two project-relative imports
below are valid. It keeps the original TDMI/geometric regime search and evaluates
multiple linear, tree, boosting, kernel, neighbour, and neural-network models.
"""

import warnings
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicTransformer
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ...strategy.moe.ood.metrics import (
    make_tdmi_similarity,
    tdmi_fingerprint,
    tdmi_fingerprint_similarity,
)
from ...exchange.loader import load_L2


# =====================================================================
# 0. Configuration
# =====================================================================

SYMBOL = "BTC/USDT"
BAR = "1s"
TIMEFRAME = BAR

BLOCK_LEN = 600
N_REGIMES = 3
MIN_EPISODES = 6
MIN_TEST_BLOCKS = 2
MAX_CLUSTER_FRAC = 0.6
TRAIN_FRAC = 0.6
Y_SCALE = 1e4
HORIZON = 5

PERIODS = (5, 10, 30)
MAX_DEPTH = 4

GP_POP = 3000
GP_GEN = 8
GP_HOF = 300
GP_COMPONENTS = 100
GP_MAX_ROWS = 20000

N_CAND = 40
K_FEATURES = 4
N_SUBSETS = 300
HILL_STEPS = 60
CORR_CAP = 0.9
SEARCH_SURR = 20
EVAL_SURR = 200

# Tree ensembles are intentionally moderately sized. Training rows are capped
# per regime for runtime; SVR gets a smaller cap because kernel fitting scales
# poorly with sample count. Set these to None to disable the general cap.
MAX_MODEL_TRAIN_ROWS = 20000
SVR_MAX_TRAIN_ROWS = 4000
RANDOM_STATE = 42

PLOT_RESULTS = True
PLOT_EACH_MODEL = False  # Set True to plot MI/MSE/MAE/expectancy heatmaps for every model.
PRINT_ALL_MODEL_MATRICES = False

rng = np.random.default_rng(RANDOM_STATE)


# =====================================================================
# 1. Event stream -> bars -> features -> target
# =====================================================================

print("Loading L2 events...")
df = load_L2(30)
if df is None or len(df) == 0:
    raise ValueError("load_L2(30) returned no rows.")

required_cols = {"timestamp", "event_sequence", "event_type"}
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"L2 data is missing required columns: {sorted(missing)}")

df = df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["timestamp", "event_sequence"]).set_index("timestamp")

eps = 1e-9
levels = np.arange(1, 21)

lob = df[df["event_type"] == "LOB"]
if lob.empty:
    raise ValueError("No LOB events were returned by load_L2(30).")

lob_required = ["best_bid_qty", "best_ask_qty", "best_bid_price", "best_ask_price"]
lob_required += [f"bid_qty_{k}" for k in range(2, 21)]
lob_required += [f"ask_qty_{k}" for k in range(2, 21)]
missing = set(lob_required).difference(lob.columns)
if missing:
    raise ValueError(f"LOB data is missing required columns: {sorted(missing)}")

bid_q = np.nan_to_num(
    np.column_stack([lob["best_bid_qty"]] + [lob[f"bid_qty_{k}"] for k in range(2, 21)]).astype(float),
    nan=0.0, posinf=0.0, neginf=0.0,
)
ask_q = np.nan_to_num(
    np.column_stack([lob["best_ask_qty"]] + [lob[f"ask_qty_{k}"] for k in range(2, 21)]).astype(float),
    nan=0.0, posinf=0.0, neginf=0.0,
)
bp = lob["best_bid_price"].to_numpy(float)
ap = lob["best_ask_price"].to_numpy(float)
mid = (bp + ap) / 2.0


def depth_imb(n):
    bid_depth = bid_q[:, :n].sum(axis=1)
    ask_depth = ask_q[:, :n].sum(axis=1)
    return (bid_depth - ask_depth) / (bid_depth + ask_depth + eps)


bq1, aq1 = bid_q[:, 0], ask_q[:, 0]
dbp = np.diff(bp, prepend=bp[0])
dap = np.diff(ap, prepend=ap[0])
bq1_prev = np.r_[bq1[0], bq1[:-1]]
aq1_prev = np.r_[aq1[0], aq1[:-1]]

# Level-1 order-flow imbalance (Cont et al.).
ofi = (
    np.where(dbp >= 0, bq1, 0.0)
    - np.where(dbp <= 0, bq1_prev, 0.0)
    - np.where(dap <= 0, aq1, 0.0)
    + np.where(dap >= 0, aq1_prev, 0.0)
)

bid_c = (bid_q * levels).sum(axis=1) / (bid_q.sum(axis=1) + eps)
ask_c = (ask_q * levels).sum(axis=1) / (ask_q.sum(axis=1) + eps)

lob_ev = pd.DataFrame(
    {
        "mid": mid,
        "spread": (ap - bp) / (mid + eps) * 1e4,
        "imb1": depth_imb(1),
        "imb5": depth_imb(5),
        "imb10": depth_imb(10),
        "imb20": depth_imb(20),
        "micro_dev": ((bp * aq1 + ap * bq1) / (bq1 + aq1 + eps) - mid)
        / (mid + eps) * 1e4,
        "skew": bid_c - ask_c,
        "ofi": ofi,
    },
    index=lob.index,
)

state_cols = [
    "mid", "spread", "imb1", "imb5", "imb10", "imb20", "micro_dev", "skew"
]
lob_bar = lob_ev.resample(BAR).agg(
    {**{col: "last" for col in state_cols}, "ofi": "sum"}
)
lob_bar[state_cols] = lob_bar[state_cols].ffill()

tr = df[df["event_type"] == "TRADE"]
if not tr.empty:
    trade_required = {"quantity", "is_buyer_maker"}
    missing = trade_required.difference(tr.columns)
    if missing:
        raise ValueError(f"TRADE data is missing required columns: {sorted(missing)}")
    qty = tr["quantity"].astype(float)
    # Buyer is maker => seller was aggressor.
    sgn = np.where(tr["is_buyer_maker"].astype(bool), -1.0, 1.0)
    tr_ev = pd.DataFrame(
        {"signed": sgn * qty, "tot": qty, "n": 1.0}, index=tr.index
    )
    tr_bar = tr_ev.resample(BAR).sum()
else:
    tr_bar = pd.DataFrame(
        index=pd.DatetimeIndex([], name=lob_bar.index.name),
        columns=["signed", "tot", "n"],
        dtype=float,
    )

bars = lob_bar.join(tr_bar, how="left")
for col in ["signed", "tot", "n"]:
    if col not in bars:
        bars[col] = 0.0
bars[["signed", "tot", "n"]] = bars[["signed", "tot", "n"]].fillna(0.0)
bars = bars.replace([np.inf, -np.inf], np.nan)
bars = bars.dropna(subset=["mid"])
bars = bars[bars["mid"] > 0]
if bars.empty:
    raise ValueError("No valid time bars remain after cleaning the L2 data.")

base = pd.DataFrame(index=bars.index)
base["ret1"] = np.log(bars["mid"]).diff() * Y_SCALE
for col in ["spread", "imb1", "imb5", "imb10", "imb20", "micro_dev", "skew", "ofi"]:
    base[col] = bars[col]
base["signed_vol"] = bars["signed"]
base["tot_vol"] = np.log1p(bars["tot"].clip(lower=0.0))
base["n_trades"] = bars["n"]
base["trade_imb"] = bars["signed"] / (bars["tot"] + eps)

# Raw terminals plus backward-looking rolling means and changes.
terms = {}
for col in base.columns:
    series = base[col]
    terms[col] = series
    for window in PERIODS:
        terms[f"{col}_m{window}"] = series.rolling(window).mean()
        terms[f"{col}_d{window}"] = series - series.shift(window)

T_df = pd.DataFrame(terms)
term_names = list(T_df.columns)

# Features at t predict the forward mid-price return from t to t+HORIZON.
log_mid = np.log(bars["mid"])
y_all = (log_mid.shift(-HORIZON) - log_mid) * Y_SCALE

data = pd.concat([T_df, y_all.rename("y")], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).dropna()

n_blocks = len(data) // BLOCK_LEN
if n_blocks < max(N_REGIMES * MIN_EPISODES, 10):
    raise ValueError(
        f"Only {n_blocks} complete blocks are available; too few for the configured "
        f"regime search. Check the loaded data, BAR, and BLOCK_LEN."
    )

n_train_blocks = int(n_blocks * TRAIN_FRAC)
cut = n_train_blocks * BLOCK_LEN
if n_train_blocks < 2 or n_blocks - n_train_blocks < MIN_TEST_BLOCKS * N_REGIMES:
    raise ValueError(
        f"Chronological split gives {n_train_blocks} train blocks and "
        f"{n_blocks - n_train_blocks} test blocks; increase the data or adjust split settings."
    )

# Fit scaling only on the global training period.
mu = data.iloc[:cut][term_names].mean()
sd = data.iloc[:cut][term_names].std().replace(0, 1.0).fillna(1.0)
T_all = ((data[term_names] - mu) / sd).clip(-5, 5).to_numpy(dtype=float)
Y_all = data["y"].to_numpy(dtype=float)

# Only complete blocks are used by the episode/regime pipeline.
used_rows = n_blocks * BLOCK_LEN
T_all = T_all[:used_rows]
Y_all = Y_all[:used_rows]
cut = n_train_blocks * BLOCK_LEN

print(
    f"{len(data)} usable rows, {len(term_names)} GP terminals -> {n_blocks} blocks "
    f"({n_train_blocks} train / {n_blocks - n_train_blocks} test)"
)


# =====================================================================
# 2. GP feature generation, training-period only
# =====================================================================

def _abs_spearman(y, y_pred, sample_weight):
    if np.std(y_pred) < 1e-12:
        return 0.0
    rho = spearmanr(y, y_pred)[0]
    return 0.0 if not np.isfinite(rho) else abs(float(rho))


abs_spearman = make_fitness(function=_abs_spearman, greater_is_better=True)
fit_idx = np.sort(rng.choice(cut, size=min(cut, GP_MAX_ROWS), replace=False))

print("Fitting gplearn SymbolicTransformer...")
gp = SymbolicTransformer(
    population_size=GP_POP,
    generations=GP_GEN,
    hall_of_fame=GP_HOF,
    n_components=GP_COMPONENTS,
    function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min"),
    metric=abs_spearman,
    init_depth=(2, MAX_DEPTH),
    init_method="half and half",
    const_range=(-1.0, 1.0),
    parsimony_coefficient=0.0005,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    feature_names=term_names,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
)
gp.fit(T_all[fit_idx], Y_all[fit_idx])

# Harvest programs from every generation; mutations can grow deeper trees.
pool = {}
for generation in gp._programs:
    for program in generation:
        if program is None or program.depth_ > MAX_DEPTH:
            continue
        key = str(program)
        if key not in pool or program.raw_fitness_ > pool[key].raw_fitness_:
            pool[key] = program
print(f"GP produced {len(pool)} distinct programs with depth <= {MAX_DEPTH}")

cand_programs, cand_cols, z_train_kept = [], [], []
for program in sorted(pool.values(), key=lambda item: item.raw_fitness_, reverse=True):
    values = np.nan_to_num(
        program.execute(T_all), nan=0.0, posinf=0.0, neginf=0.0
    )
    train_std = values[:cut].std()
    if train_std < 1e-9:
        continue
    z = (values[:cut] - values[:cut].mean()) / train_std
    if z_train_kept:
        correlations = np.abs(np.asarray(z_train_kept) @ z / len(z))
        if np.max(correlations) > CORR_CAP:
            continue
    cand_programs.append(program)
    cand_cols.append(values)
    z_train_kept.append(z)
    if len(cand_programs) >= N_CAND:
        break

if len(cand_cols) < K_FEATURES:
    raise ValueError(
        f"Only {len(cand_cols)} decorrelated GP features survived; K_FEATURES={K_FEATURES}. "
        "Lower CORR_CAP, lower K_FEATURES, or increase the GP search size."
    )

F_raw = np.column_stack(cand_cols)
f_mu = F_raw[:cut].mean(axis=0)
f_sd = F_raw[:cut].std(axis=0)
f_sd = np.where(f_sd < 1e-12, 1.0, f_sd)
F_all = np.clip((F_raw - f_mu) / f_sd, -5, 5)
F_all = np.nan_to_num(F_all, nan=0.0, posinf=0.0, neginf=0.0)
F_names = [str(program) for program in cand_programs]
print(f"{F_all.shape[1]} decorrelated GP features kept")

Fb = [F_all[b * BLOCK_LEN:(b + 1) * BLOCK_LEN] for b in range(n_blocks)]
Yb = [Y_all[b * BLOCK_LEN:(b + 1) * BLOCK_LEN] for b in range(n_blocks)]


def fingerprint_vector(x, n_surrogates, seed):
    """Return a flat finite TDMI fingerprint for a block or episode."""
    fp = tdmi_fingerprint(x, 10, 8, 30, n_surrogates, seed)
    return np.nan_to_num(np.ravel(fp).astype(float), nan=0.0, posinf=0.0, neginf=0.0)


print("Computing candidate-block fingerprints...")
FP = [
    [fingerprint_vector(Fb[b][:, [j]], SEARCH_SURR, RANDOM_STATE) for j in range(F_all.shape[1])]
    for b in range(n_blocks)
]


# =====================================================================
# 3. Regime-similarity and clustering helpers
# =====================================================================

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def pair_sim(sim_name, vectors):
    """Return a similarity function calibrated for the chosen feature subset."""
    if sim_name.startswith("TDMI"):
        return lambda a, b: float(tdmi_fingerprint_similarity(a, b, 0.0))

    pair_distances = np.array(
        [np.linalg.norm(a - b) for a, b in combinations(vectors, 2)], dtype=float
    )
    positive = pair_distances[pair_distances > 1e-12]
    distance_scale = float(np.median(positive)) if len(positive) else 1.0
    # The scale is closed over here: no global dist_scale dependency.
    return lambda a, b: float(np.exp(-np.linalg.norm(a - b) / distance_scale))


def block_similarity_matrix(vectors, similarity):
    n = len(vectors)
    matrix = np.eye(n, dtype=float)
    for i, j in combinations(range(n), 2):
        matrix[i, j] = matrix[j, i] = similarity(vectors[i], vectors[j])
    return matrix


def leader_cluster(similarity_matrix, threshold):
    """Greedy clustering: join a cluster if mean similarity to members >= threshold."""
    clusters = []
    for i in range(len(similarity_matrix)):
        for cluster in clusters:
            if np.mean(similarity_matrix[i, cluster]) >= threshold:
                cluster.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def regime_matrix(similarity_matrix, clusters):
    k = len(clusters)
    matrix = np.zeros((k, k), dtype=float)
    for i, left in enumerate(clusters):
        for j, right in enumerate(clusters):
            sub = similarity_matrix[np.ix_(left, right)]
            if i == j:
                vals = sub[np.triu_indices(len(left), k=1)]
                matrix[i, j] = float(np.mean(vals)) if len(vals) else np.nan
            else:
                matrix[i, j] = float(np.mean(sub)) if sub.size else np.nan
    return matrix


def identity_margin(matrix):
    """Minimum diagonal-minus-best-off-diagonal gap, ignoring non-finite rivals."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return np.nan
    margins = []
    for i in range(matrix.shape[0]):
        own = matrix[i, i]
        rivals = np.delete(matrix[i], i)
        rivals = rivals[np.isfinite(rivals)]
        if not np.isfinite(own) or not len(rivals):
            return np.nan
        margins.append(own - np.max(rivals))
    return float(np.min(margins))


def valid_cluster(cluster):
    n_test = sum(i >= n_train_blocks for i in cluster)
    n_train = len(cluster) - n_test
    return (
        len(cluster) >= MIN_EPISODES
        and len(cluster) <= MAX_CLUSTER_FRAC * n_blocks
        and n_test >= MIN_TEST_BLOCKS
        and n_train >= 2
    )


def brute_force_regimes(similarity_matrix):
    """Sweep clustering thresholds, selecting recurring clusters with an identity-like matrix."""
    off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
    if not len(off_diagonal):
        return None, -np.inf

    best_clusters, best_score = None, -np.inf
    thresholds = np.unique(np.quantile(off_diagonal, np.linspace(0.05, 0.95, 40)))
    for threshold in thresholds:
        clusters = [c for c in leader_cluster(similarity_matrix, threshold) if valid_cluster(c)]
        if len(clusters) < N_REGIMES:
            continue
        clusters = sorted(clusters, key=len, reverse=True)[:N_REGIMES]
        score = identity_margin(regime_matrix(similarity_matrix, clusters))
        if np.isfinite(score) and score > best_score:
            best_clusters, best_score = clusters, score
    return best_clusters, best_score


_subset_cache = {}


def eval_subset(features, sim_name):
    canonical_features = tuple(sorted(int(f) for f in features))
    cache_key = (sim_name, canonical_features)
    if cache_key not in _subset_cache:
        vectors = [
            np.concatenate([FP[b][j] for j in canonical_features])
            for b in range(n_blocks)
        ]
        similarity_matrix = block_similarity_matrix(vectors, pair_sim(sim_name, vectors))
        clusters, score = brute_force_regimes(similarity_matrix)
        _subset_cache[cache_key] = (
            score if clusters is not None else -np.inf,
            clusters,
            similarity_matrix,
        )
    return _subset_cache[cache_key]


def search_feature_subset(sim_name):
    """Random K-feature subsets followed by one-feature-at-a-time hill climbing."""
    n_features = F_all.shape[1]
    if n_features < K_FEATURES:
        raise ValueError(f"Need at least {K_FEATURES} features; found {n_features}.")

    best_features = None
    best_result = (-np.inf, None, None)
    for _ in range(N_SUBSETS):
        features = tuple(sorted(rng.choice(n_features, K_FEATURES, replace=False).tolist()))
        result = eval_subset(features, sim_name)
        if result[0] > best_result[0]:
            best_features, best_result = features, result

    if best_features is None:
        return None, None, None, -np.inf

    for _ in range(HILL_STEPS):
        trial = list(best_features)
        slot = int(rng.integers(K_FEATURES))
        trial[slot] = int(rng.integers(n_features))
        if len(set(trial)) != K_FEATURES:
            continue
        trial = tuple(sorted(trial))
        result = eval_subset(trial, sim_name)
        if result[0] > best_result[0]:
            best_features, best_result = trial, result

    return list(best_features), best_result[1], best_result[2], best_result[0]


# =====================================================================
# 4. Model zoo
# =====================================================================

# Each template is cloned before fitting, so every regime/model has an independent estimator.
MODEL_TEMPLATES = {
    # Linear and robust linear models
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Ridge (alpha=10)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.005, max_iter=10000)),
    "ElasticNet": make_pipeline(
        StandardScaler(), ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE)
    ),
    "Huber": make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)),
    "BayesianRidge": make_pipeline(StandardScaler(), BayesianRidge()),

    # Single trees and ensembles
    "DecisionTree": DecisionTreeRegressor(
        max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_leaf=10,
        max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE,
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.03, max_depth=2,
        min_samples_leaf=20, loss="squared_error", random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=1.0, random_state=RANDOM_STATE,
    ),

    # Nonlinear baselines
    "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, kernel="rbf")),
    "KNN": make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=40, weights="distance", n_jobs=-1)
    ),
    "MLP": make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=400,
            early_stopping=True, random_state=RANDOM_STATE,
        ),
    ),
}


def sample_training_rows(x, y, model_name, seed):
    """Cap training rows for expensive estimators without touching test data."""
    limit = MAX_MODEL_TRAIN_ROWS
    if model_name == "SVR (RBF)":
        limit = SVR_MAX_TRAIN_ROWS
    if limit is None or len(y) <= limit:
        return x, y
    local_rng = np.random.default_rng(seed)
    idx = np.sort(local_rng.choice(len(y), size=limit, replace=False))
    return x[idx], y[idx]


def safe_mutual_information(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return 0.0
    try:
        value = mutual_info_regression(
            np.asarray(y_pred).reshape(-1, 1),
            np.asarray(y_true),
            random_state=RANDOM_STATE,
        )[0]
        return float(value) if np.isfinite(value) else np.nan
    except (ValueError, FloatingPointError):
        return np.nan


def metric_matrices():
    return {
        "mi": pd.DataFrame(index=REGIME_NAMES, columns=REGIME_NAMES, dtype=float),
        "mse": pd.DataFrame(index=REGIME_NAMES, columns=REGIME_NAMES, dtype=float),
        "mae": pd.DataFrame(index=REGIME_NAMES, columns=REGIME_NAMES, dtype=float),
        "r2": pd.DataFrame(index=REGIME_NAMES, columns=REGIME_NAMES, dtype=float),
        "expectancy": pd.DataFrame(index=REGIME_NAMES, columns=REGIME_NAMES, dtype=float),
    }


def evaluate_model_zoo(train_episodes, test_episodes):
    """Fit one copy of each model per training regime, then cross-evaluate all regimes."""
    train_arrays = {}
    test_arrays = {}
    for regime_name in REGIME_NAMES:
        if not train_episodes[regime_name]:
            raise ValueError(f"No training episodes for {regime_name}.")
        if not test_episodes[regime_name]:
            raise ValueError(f"No test episodes for {regime_name}.")
        train_arrays[regime_name] = (
            np.concatenate([x for x, _ in train_episodes[regime_name]], axis=0),
            np.concatenate([y for _, y in train_episodes[regime_name]], axis=0),
        )
        test_arrays[regime_name] = (
            np.concatenate([x for x, _ in test_episodes[regime_name]], axis=0),
            np.concatenate([y for _, y in test_episodes[regime_name]], axis=0),
        )

    # The no-skill baseline predicts the matching regime's training-target mean.
    baseline_mse = pd.Series(index=REGIME_NAMES, dtype=float)
    baseline_mae = pd.Series(index=REGIME_NAMES, dtype=float)
    for test_regime in REGIME_NAMES:
        y_test = test_arrays[test_regime][1]
        y_train = train_arrays[test_regime][1]
        baseline_prediction = np.full(y_test.shape, float(np.mean(y_train)))
        baseline_mse[test_regime] = mean_squared_error(y_test, baseline_prediction)
        baseline_mae[test_regime] = mean_absolute_error(y_test, baseline_prediction)

    all_model_results = {}
    model_count = len(MODEL_TEMPLATES)
    print(f"\nFitting {model_count} models for each of {len(REGIME_NAMES)} regimes...")

    for model_no, (model_name, template) in enumerate(MODEL_TEMPLATES.items(), start=1):
        print(f"  [{model_no:02d}/{model_count:02d}] {model_name}")
        fitted_experts = {}
        metric_values = metric_matrices()

        for train_regime_no, train_regime in enumerate(REGIME_NAMES):
            x_train, y_train = train_arrays[train_regime]
            x_fit, y_fit = sample_training_rows(
                x_train, y_train, model_name,
                seed=RANDOM_STATE + 1000 * model_no + train_regime_no,
            )
            model = clone(template)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_fit, y_fit)
            fitted_experts[train_regime] = model

        for expert_regime, model in fitted_experts.items():
            for test_regime in REGIME_NAMES:
                x_test, y_test = test_arrays[test_regime]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred = np.asarray(model.predict(x_test), dtype=float)

                if not np.all(np.isfinite(y_pred)):
                    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

                metric_values["mi"].loc[expert_regime, test_regime] = safe_mutual_information(
                    y_test, y_pred
                )
                metric_values["mse"].loc[expert_regime, test_regime] = mean_squared_error(
                    y_test, y_pred
                )
                metric_values["mae"].loc[expert_regime, test_regime] = mean_absolute_error(
                    y_test, y_pred
                )
                try:
                    metric_values["r2"].loc[expert_regime, test_regime] = r2_score(y_test, y_pred)
                except ValueError:
                    metric_values["r2"].loc[expert_regime, test_regime] = np.nan
                # Gross directional expectancy in bps/bar; fees/slippage are not deducted.
                metric_values["expectancy"].loc[expert_regime, test_regime] = float(
                    np.mean(np.sign(y_pred) * y_test)
                )

        all_model_results[model_name] = {
            **metric_values,
            "experts": fitted_experts,
        }

    return all_model_results, baseline_mse, baseline_mae


def summarize_models(model_results, baseline_mse):
    """One row per model, using matched-regime (diagonal) and mismatched averages."""
    rows = []
    diag_mask = np.eye(len(REGIME_NAMES), dtype=bool)
    off_mask = ~diag_mask
    baseline_mean = float(baseline_mse.mean())

    for model_name, values in model_results.items():
        mi = values["mi"].to_numpy(dtype=float)
        mse = values["mse"].to_numpy(dtype=float)
        mae = values["mae"].to_numpy(dtype=float)
        r2 = values["r2"].to_numpy(dtype=float)
        expectancy = values["expectancy"].to_numpy(dtype=float)
        diag_mse = float(np.nanmean(mse[diag_mask]))
        improvement = (
            100.0 * (baseline_mean - diag_mse) / baseline_mean
            if baseline_mean > 1e-12 else np.nan
        )
        rows.append(
            {
                "model": model_name,
                "diag_mi": float(np.nanmean(mi[diag_mask])),
                "offdiag_mi": float(np.nanmean(mi[off_mask])),
                "diag_mse": diag_mse,
                "offdiag_mse": float(np.nanmean(mse[off_mask])),
                "baseline_mse": baseline_mean,
                "mse_improvement_vs_baseline_%": improvement,
                "diag_mae": float(np.nanmean(mae[diag_mask])),
                "diag_r2": float(np.nanmean(r2[diag_mask])),
                "diag_expectancy_bps": float(np.nanmean(expectancy[diag_mask])),
                "offdiag_expectancy_bps": float(np.nanmean(expectancy[off_mask])),
            }
        )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    return table.sort_values("diag_mse", ascending=True, na_position="last").reset_index(drop=True)


def plot_heatmaps(sim_name, tdmi_matrix, model_name, model_result):
    matrices = [
        tdmi_matrix,
        model_result["mi"],
        model_result["mse"],
        model_result["expectancy"],
    ]
    titles = [
        "Held-out TDMI similarity",
        f"{model_name}: predictive MI",
        f"{model_name}: MSE",
        f"{model_name}: expectancy (bps/bar)",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    fig.suptitle(f"{sim_name} | {SYMBOL} L2 {TIMEFRAME} | best by diagonal MSE: {model_name}")
    for ax, matrix, title in zip(axes, matrices, titles):
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto")
        ax.set_xticks(range(len(REGIME_NAMES)), REGIME_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(REGIME_NAMES)), REGIME_NAMES)
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                text = f"{value:.3f}" if np.isfinite(value) else "NaN"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_all_model_heatmaps(sim_name, model_results):
    for model_name, model_result in model_results.items():
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle(f"{sim_name} | {model_name}")
        for ax, key, title in zip(
            axes,
            ["mi", "mse", "mae", "expectancy"],
            ["Predictive MI", "MSE", "MAE", "Expectancy (bps/bar)"],
        ):
            matrix = model_result[key]
            values = matrix.to_numpy(dtype=float)
            image = ax.imshow(values, aspect="auto")
            ax.set_xticks(range(len(REGIME_NAMES)), REGIME_NAMES, rotation=45, ha="right")
            ax.set_yticks(range(len(REGIME_NAMES)), REGIME_NAMES)
            ax.set_title(title)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    value = values[i, j]
                    ax.text(j, i, f"{value:.3f}" if np.isfinite(value) else "NaN",
                            ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()


# =====================================================================
# 5. Run regime discovery twice; fit and test every model for each run
# =====================================================================

SIMILARITIES = ["TDMI (Jaccard)", "Geometric (distance)"]
REGIME_NAMES = [f"Regime {r + 1}" for r in range(N_REGIMES)]
results = {}

for sim_name in SIMILARITIES:
    print("\n" + "#" * 78)
    print(f"# REGIME DISCOVERY: {sim_name}")
    print("#" * 78)

    features, clusters, block_sim_matrix, search_score = search_feature_subset(sim_name)
    if clusters is None:
        print(
            "No recurring regimes found with these settings. Try reducing MIN_EPISODES, "
            "reducing N_REGIMES, increasing data, or changing BLOCK_LEN."
        )
        continue

    print(f"Identity margin (search): {search_score:.4f}")
    print("Selected GP features:")
    for feature_idx in features:
        print(f"  [{feature_idx}] {F_names[feature_idx]}")

    selected_regime_matrix = regime_matrix(block_sim_matrix, clusters)
    lo = np.nanmin(selected_regime_matrix)
    hi = np.nanmax(selected_regime_matrix)
    contrast = (selected_regime_matrix - lo) / (hi - lo + 1e-12)
    print("\nRegime-vs-regime block similarity:")
    print(pd.DataFrame(selected_regime_matrix, index=REGIME_NAMES, columns=REGIME_NAMES).round(3))
    print("\nMin-max contrast (identity-like separation is desired):")
    print(pd.DataFrame(contrast, index=REGIME_NAMES, columns=REGIME_NAMES).round(2))

    blocks = [(Fb[b][:, features], Yb[b]) for b in range(n_blocks)]
    train_episodes, test_episodes = {}, {}
    for regime_name, block_indices in zip(REGIME_NAMES, clusters):
        sorted_indices = sorted(block_indices)
        train_episodes[regime_name] = [
            blocks[i] for i in sorted_indices if i < n_train_blocks
        ]
        test_episodes[regime_name] = [
            blocks[i] for i in sorted_indices if i >= n_train_blocks
        ]
        print(
            f"{regime_name}: {len(sorted_indices)} blocks "
            f"(train {len(train_episodes[regime_name])}, test {len(test_episodes[regime_name])})"
        )

    if any(len(test_episodes[name]) < MIN_TEST_BLOCKS for name in REGIME_NAMES):
        print("Not enough held-out episodes per regime; skipping this similarity run.")
        continue
    if any(len(train_episodes[name]) == 0 for name in REGIME_NAMES):
        print("A discovered regime has no training episode; skipping this similarity run.")
        continue

    # ---------------------------------------------------------------
    # 5a. Held-out TDMI similarity matrix across test episodes
    # ---------------------------------------------------------------
    print("\nComputing held-out TDMI episode-similarity matrix...")
    tdmi_similarity = make_tdmi_similarity(
        max_lag=10,
        bins=8,
        min_pairs=30,
        n_surrogates=EVAL_SURR,
        seed=RANDOM_STATE,
    )
    tdmi_matrix = pd.DataFrame(index=REGIME_NAMES, columns=REGIME_NAMES, dtype=float)

    for i, regime_a in enumerate(REGIME_NAMES):
        for j, regime_b in enumerate(REGIME_NAMES):
            if i == j:
                episode_pairs = combinations(test_episodes[regime_a], 2)
            else:
                episode_pairs = product(test_episodes[regime_a], test_episodes[regime_b])
            scores = []
            for (x_a, _), (x_b, _) in episode_pairs:
                value = tdmi_similarity(x_a, x_b)
                if np.isfinite(value):
                    scores.append(float(value))
            tdmi_matrix.loc[regime_a, regime_b] = float(np.mean(scores)) if scores else np.nan

    print("\nMean TDMI similarity matrix (held-out test episodes):")
    print(tdmi_matrix.round(3))
    heldout_tdmi_margin = identity_margin(tdmi_matrix.to_numpy(dtype=float))
    print(f"Held-out TDMI identity margin: {heldout_tdmi_margin:.4f}")

    # ---------------------------------------------------------------
    # 5b. Fit/evaluate all model families
    # ---------------------------------------------------------------
    model_results, baseline_mse, baseline_mae = evaluate_model_zoo(
        train_episodes, test_episodes
    )
    leaderboard = summarize_models(model_results, baseline_mse)
    print("\nMODEL LEADERBOARD (sorted by matched-regime diagonal MSE; lower is better):")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(leaderboard.round(4).to_string(index=False))

    print("\nTrain-mean baseline MSE per test regime:")
    print(baseline_mse.round(4))
    print("\nTrain-mean baseline MAE per test regime:")
    print(baseline_mae.round(4))

    if not leaderboard.empty:
        best_model_name = str(leaderboard.iloc[0]["model"])
        print(f"\nBest model by diagonal MSE: {best_model_name}")
        best_model = model_results[best_model_name]
        print("\nBest model predictive MI matrix:")
        print(best_model["mi"].round(4))
        print("\nBest model MSE matrix:")
        print(best_model["mse"].round(4))
        print("\nBest model MAE matrix:")
        print(best_model["mae"].round(4))
        print("\nBest model R2 matrix:")
        print(best_model["r2"].round(4))
        print("\nBest model expectancy matrix (gross bps/bar):")
        print(best_model["expectancy"].round(4))

        if PRINT_ALL_MODEL_MATRICES:
            for model_name, model_result in model_results.items():
                print(f"\n--- {model_name} ---")
                for key in ["mi", "mse", "mae", "r2", "expectancy"]:
                    print(f"{key.upper()} matrix:")
                    print(model_result[key].round(4))

        if PLOT_RESULTS:
            plot_heatmaps(sim_name, tdmi_matrix, best_model_name, best_model)
        if PLOT_EACH_MODEL:
            plot_all_model_heatmaps(sim_name, model_results)

    # ---------------------------------------------------------------
    # 5c. Compare test-episode fingerprints: weighted Jaccard, cosine, distance
    # ---------------------------------------------------------------
    print("\nComputing test-episode fingerprint similarity matrices...")
    episode_fingerprints = {
        regime_name: [
            fingerprint_vector(x, EVAL_SURR, RANDOM_STATE)
            for x, _ in test_episodes[regime_name]
        ]
        for regime_name in REGIME_NAMES
    }

    all_fingerprints = [fp for group in episode_fingerprints.values() for fp in group]
    all_pair_distances = np.array(
        [np.linalg.norm(a - b) for a, b in combinations(all_fingerprints, 2)],
        dtype=float,
    )
    positive_distances = all_pair_distances[all_pair_distances > 1e-12]
    # This value is computed before constructing/calling distance similarity.
    dist_scale = float(np.median(positive_distances)) if len(positive_distances) else 1.0

    def distance_similarity(a, b, scale=dist_scale):
        return float(np.exp(-np.linalg.norm(a - b) / scale))

    def make_fingerprint_similarity_matrix(metric):
        matrix = pd.DataFrame(index=REGIME_NAMES, columns=REGIME_NAMES, dtype=float)
        for regime_a in REGIME_NAMES:
            for regime_b in REGIME_NAMES:
                if regime_a == regime_b:
                    episode_pairs = combinations(episode_fingerprints[regime_a], 2)
                else:
                    episode_pairs = product(
                        episode_fingerprints[regime_a], episode_fingerprints[regime_b]
                    )
                scores = [metric(a, b) for a, b in episode_pairs]
                matrix.loc[regime_a, regime_b] = float(np.mean(scores)) if scores else np.nan
        return matrix

    jaccard_matrix = make_fingerprint_similarity_matrix(
        lambda a, b: float(tdmi_fingerprint_similarity(a, b, 0.0))
    )
    cosine_matrix = make_fingerprint_similarity_matrix(cosine_sim)
    distance_matrix = make_fingerprint_similarity_matrix(distance_similarity)

    print("\nWeighted Jaccard fingerprint similarity:")
    print(jaccard_matrix.round(3))
    print("\nCosine fingerprint similarity:")
    print(cosine_matrix.round(3))
    print(f"\nEuclidean-distance similarity (dist_scale={dist_scale:.6g}):")
    print(distance_matrix.round(3))

    results[sim_name] = {
        "features": [F_names[j] for j in features],
        "feature_indices": features,
        "clusters": clusters,
        "block_similarity": pd.DataFrame(block_sim_matrix),
        "regime_similarity": pd.DataFrame(
            selected_regime_matrix, index=REGIME_NAMES, columns=REGIME_NAMES
        ),
        "identity_margin": search_score,
        "tdmi": tdmi_matrix,
        "test_tdmi_margin": heldout_tdmi_margin,
        "fingerprint_jaccard": jaccard_matrix,
        "fingerprint_cosine": cosine_matrix,
        "fingerprint_distance": distance_matrix,
        "dist_scale": dist_scale,
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
        "models": model_results,
        "leaderboard": leaderboard,
    }


# =====================================================================
# 6. Final comparison across model families and similarity approaches
# =====================================================================

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)

all_leaderboards = []
for sim_name, run_result in results.items():
    print(f"\n=== Similarity method: {sim_name} ===")
    leaderboard = run_result["leaderboard"].copy()
    if leaderboard.empty:
        print("No model results for this run.")
        continue
    leaderboard.insert(0, "similarity", sim_name)
    leaderboard["search_identity_margin"] = run_result["identity_margin"]
    leaderboard["heldout_tdmi_margin"] = run_result["test_tdmi_margin"]
    all_leaderboards.append(leaderboard)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(leaderboard.round(4).to_string(index=False))

if all_leaderboards:
    combined_leaderboard = pd.concat(all_leaderboards, ignore_index=True)
    combined_leaderboard = combined_leaderboard.sort_values(
        "diag_mse", ascending=True, na_position="last"
    ).reset_index(drop=True)
    print("\n=== COMBINED LEADERBOARD (lower diagonal MSE is better) ===")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(combined_leaderboard.round(4).to_string(index=False))
else:
    combined_leaderboard = pd.DataFrame()
    print(
        "No runs completed successfully. Regime search may need relaxed settings, "
        "more data, or fewer required episodes."
    )

# Convenient variables remain in the execution namespace for notebook inspection:
# results[similarity_name]["models"][model_name]["mse"], ["mi"], ["mae"], ["r2"], ["expectancy"]
# results[similarity_name]["leaderboard"]
# combined_leaderboard
