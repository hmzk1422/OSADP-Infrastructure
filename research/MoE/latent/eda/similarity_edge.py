"""
similarity_edge.py

Mixture-of-Experts (MoE) trading layer built on top of the VQ-VAE
regime-discovery pipeline.

Each discovered latent regime gets its own "home" expert strategy:

    regime 0 (bullish trend)      -> long-bias expert
    regime 1 (bearish trend)      -> short-bias expert
    regime 2 (mean reversion)     -> fade-the-move expert
    regime 3 (high vol/weak dir)  -> tree-based (RandomForest) expert

A router assigns every window to a discovered regime (the VQ-VAE's
mapped code). In production only the "home" expert for a regime
would actually be allowed to trade it -- but for these experiments
we deliberately run EVERY expert on EVERY regime's windows, because
both hypotheses below are fundamentally about cross-expert,
cross-regime behavior, not just about the home pairing.

`state` (ground truth) is used ONCE, offline, on the TRAIN split
only, to map raw VQ code IDs to semantic regime labels -- a
labeling step, not something available live. That fixed mapping is
then applied mechanically to test data. Test labels are never used
to fit anything.

Two hypotheses are tested:

H1 -- "regime specialization":
    Does expert_n, deployed on latent regime n, outperform expert_n
    deployed on every OTHER latent regime? Tested via a full
    cross-deployment matrix on the TEST split.

H2 -- "similarity predicts edge transfer" (REVISED):

    The original version of this test asked a REGIME-level question:
    "does regime r's overall backtested expectancy transfer to
    regime r's overall test-period expectancy, and does that
    transfer correlate with regime r's average internal coherence?"
    That only gives one data point per regime (n=4 regimes total),
    and -- more importantly -- it silently assumes that "this
    window resembles regime r" implies "this window's future return
    behaves like regime r's historical average", without ever
    testing that assumption at the level where it actually has to
    hold: the INDIVIDUAL WINDOW.

    The corrected hypothesis is:

        high similarity(window, historical regime-r prototype)
            => this window's outcome, under ANY expert e, lands
               close to expert e's HISTORICAL (train) expectancy
               for regime r

    Concretely, for every TEST window:
      1. It has a mapped regime r (from the VQ router).
      2. Its "similarity" is cosine_similarity(z_e, codebook[r]) --
         how close this window's continuous encoding is to the
         VQ-VAE's own learned prototype for regime r. The codebook
         vector is used as the prototype (rather than a plain
         centroid) because it IS the model's own reference point --
         literally what "recurring configuration r" means to it.
      3. For EVERY expert e (not just regime r's home expert), we
         compute what e's position and realized PnL would have been
         on this window, and compare that PnL to e's TRAIN-period
         backtested expectancy for regime r specifically (an
         expert x regime matrix -- exactly the shape of the example
         table in the prompt: each expert has ITS OWN historical
         number for EACH regime, not just its own).
      4. Test windows are bucketed into low/medium/high similarity
         terciles WITHIN each regime (since similarity ranges can
         differ across regimes), and realized expectancy per
         (expert, regime, bucket) is compared against that expert's
         single historical (expert, regime) number.

    This produces two outputs:
      - An interpretable bucketed table (matches the low/med/high
        narrative in the prompt).
      - A POOLED per-window-per-expert correlation between
        similarity and |realized pnl - historical expectancy|,
        with n = (test windows) x (n_codes experts) -- thousands of
        data points instead of 4, which is what actually lets you
        test this statistically rather than eyeballing 4 numbers.
"""

import numpy as np
from itertools import permutations

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from research.MoE.data.synthetic import simulate_market_data

try:
    from research.MoE.latent.vqvae.vqvae import LatentStateModel
except ImportError:
    from research.MoE.latent.vqvae.vqvae_gd import LatentStateModel


# ============================================================
# CONFIG
# ============================================================

SEED = 84

N_BARS = 5_000
N_FEATURES = 50
N_STATES = 4
FEATURE_NOISE = 0.5

WINDOW_SIZE = 16
LATENT_DIM = 8
HIDDEN_DIM = 128
N_CODES = 4
EPOCHS = 200
BATCH_SIZE = 256

TRAIN_FRAC = 0.7

# Which SEMANTIC regime (post train-only mapping) gets the
# tree-based expert instead of a hand-written heuristic.
TREE_EXPERT_REGIME = 3

LONG_THRESH = 0.55
SHORT_THRESH = 0.45

# Minimum test windows in a regime before we bother bucketing it.
MIN_WINDOWS_FOR_BUCKETING = 30


# ============================================================
# Windowing / alignment helper
#
# See earlier turns for the full rationale. A window ending at
# local bar index `e` decides a position using information at the
# close of bar e; the outcome realized is the return on bar e+1.
# Called separately per slice so no bar ever crosses the train/
# test boundary.
# ============================================================

def build_dataset(X_slice, state_slice, return_slice, window_size):

    n = len(X_slice)

    if n < window_size + 1:
        raise ValueError(
            "Slice too short for this window size plus one "
            "lookahead bar."
        )

    end_indices = np.arange(window_size - 1, n - 1)

    current_features = X_slice[end_indices]
    true_regime = state_slice[end_indices]
    next_return = return_slice[end_indices + 1]
    most_recent_return = return_slice[end_indices]

    return current_features, true_regime, next_return, most_recent_return


# ============================================================
# Heuristic experts (vectorized)
# ============================================================

def expert_bull(current_features, most_recent_return):
    return np.ones(len(current_features))


def expert_bear(current_features, most_recent_return):
    return -np.ones(len(current_features))


def expert_mean_reversion(current_features, most_recent_return):
    return -np.sign(most_recent_return)


HEURISTIC_EXPERTS = {
    0: expert_bull,
    1: expert_bear,
    2: expert_mean_reversion,
}


# ============================================================
# Tree-based expert
# ============================================================

class TreeExpert:

    def __init__(self, random_state=SEED, **rf_kwargs):

        default_kwargs = dict(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=20,
            random_state=random_state,
        )
        default_kwargs.update(rf_kwargs)

        self.model = RandomForestClassifier(**default_kwargs)
        self.fitted = False

    def fit(self, current_features, next_return):
        target = (next_return > 0).astype(int)
        self.model.fit(current_features, target)
        self.fitted = True

    def predict_position(
        self,
        current_features,
        long_thresh=LONG_THRESH,
        short_thresh=SHORT_THRESH,
    ):

        if not self.fitted:
            raise RuntimeError("TreeExpert must be fit before use.")

        proba_up = self.model.predict_proba(current_features)[:, 1]

        position = np.zeros(len(current_features))
        position[proba_up > long_thresh] = 1.0
        position[proba_up < short_thresh] = -1.0

        return position


# ============================================================
# Cross-deployment matrix
#
# Shared helper: run EVERY expert on EVERY regime's windows within
# a given slice (train or test), returning:
#   matrix[e, r]  = mean(position_e * next_return) restricted to
#                   windows whose mapped regime is r
#   counts[e, r]  = number of such windows (same for every e,
#                   since it only depends on r -- kept per-cell for
#                   clarity when printing)
# ============================================================

def cross_deployment_matrix(
    experts,
    current_features,
    most_recent_return,
    next_return,
    mapped_regime,
    n_codes,
):

    matrix = np.full((n_codes, n_codes), np.nan)
    counts = np.zeros((n_codes, n_codes), dtype=int)

    for r in range(n_codes):

        mask = mapped_regime == r
        n_windows = int(mask.sum())

        if n_windows == 0:
            continue

        for e in range(n_codes):
            position = experts[e](
                current_features[mask], most_recent_return[mask]
            )
            pnl = position * next_return[mask]
            matrix[e, r] = pnl.mean()
            counts[e, r] = n_windows

    return matrix, counts


# ============================================================
# Cosine similarity, batched against a per-row prototype
# ============================================================

def cosine_similarity_batch(vectors, prototypes_per_row):

    numerator = np.sum(vectors * prototypes_per_row, axis=1)

    denom = (
        np.linalg.norm(vectors, axis=1)
        * np.linalg.norm(prototypes_per_row, axis=1)
    )
    denom = np.where(denom == 0, 1e-8, denom)

    return numerator / denom


# ============================================================
# Cluster-robust bootstrap for the pooled correlation
#
# The pooled (window, expert) dataset has one row per expert per
# test window, so the SAME window contributes n_codes rows. Those
# rows are not independent -- a plain pearsonr() p-value on the
# stacked data treats them as if they were, which overstates the
# effective sample size.
#
# This resamples at the WINDOW level (with replacement), keeping
# every expert's row for a chosen window together each time, then
# recomputes the pooled correlation on each resample. The spread
# of the resulting distribution reflects the TRUE number of
# independent units (test windows), not (test windows x experts).
# ============================================================

def cluster_bootstrap_correlation(
    similarity,
    deviation,
    window_id,
    n_boot=2000,
    rng=None,
):

    if rng is None:
        rng = np.random.default_rng(0)

    unique_windows = np.unique(window_id)
    n_windows = len(unique_windows)

    cluster_to_rows = {
        w: np.where(window_id == w)[0] for w in unique_windows
    }

    boot_r = np.empty(n_boot)

    for b in range(n_boot):

        sampled_windows = rng.choice(
            unique_windows, size=n_windows, replace=True
        )

        idx = np.concatenate(
            [cluster_to_rows[w] for w in sampled_windows]
        )

        boot_r[b], _ = pearsonr(similarity[idx], deviation[idx])

    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    # Two-sided bootstrap p-value: proportion of resamples on the
    # opposite side of zero from the observed statistic, doubled.
    observed_r, _ = pearsonr(similarity, deviation)

    if observed_r >= 0:
        p_boot = 2 * np.mean(boot_r <= 0)
    else:
        p_boot = 2 * np.mean(boot_r >= 0)

    p_boot = min(p_boot, 1.0)

    return {
        "observed_r": observed_r,
        "boot_r": boot_r,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_boot": p_boot,
        "n_clusters": n_windows,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    import random
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # Data + chronological split
    # --------------------------------------------------------

    df = simulate_market_data(
        n_bars=N_BARS,
        n_features=N_FEATURES,
        n_states=N_STATES,
        seed=SEED,
        feature_noise=FEATURE_NOISE,
    )

    feature_cols = [c for c in df.columns if c.startswith("f")]

    X = df[feature_cols].values
    true_state = df["state"].values
    returns = df["return"].values

    split = int(len(X) * TRAIN_FRAC)

    X_train, X_test = X[:split], X[split:]
    state_train, state_test = true_state[:split], true_state[split:]
    return_train, return_test = returns[:split], returns[split:]

    # --------------------------------------------------------
    # Fit VQ-VAE on TRAIN ONLY, encode both slices with the
    # frozen, fitted model.
    # --------------------------------------------------------

    vqvae = LatentStateModel(
        n_features=len(feature_cols),
        window_size=WINDOW_SIZE,
        latent_dim=LATENT_DIM,
        n_codes=N_CODES,
        hidden_dim=HIDDEN_DIM,
        beta=0.25,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    vqvae.fit(X_train)

    out_train = vqvae.encode(X_train)
    out_test = vqvae.encode(X_test)

    raw_vq_train_full = out_train["state"]
    raw_vq_test_full = out_test["state"]

    z_e_test_full = out_test["z_e"]

    codebook = (
        vqvae.model.codebook.weight.detach().cpu().numpy()
    )  # (n_codes, latent_dim), indexed by RAW code id

    # --------------------------------------------------------
    # TRAIN-only raw-code -> semantic-regime mapping.
    # --------------------------------------------------------

    true_train_full = state_train[WINDOW_SIZE - 1:]

    best_accuracy = -1.0
    best_mapping = None

    for perm in permutations(range(N_CODES)):
        mapping = np.asarray(perm)
        mapped = mapping[raw_vq_train_full]
        acc = np.mean(mapped == true_train_full)
        if acc > best_accuracy:
            best_accuracy = acc
            best_mapping = mapping

    print("=" * 70)
    print("TRAIN-ONLY CODE -> REGIME MAPPING")
    print("=" * 70)
    print(f"Mapping (raw code -> regime): {best_mapping}")
    print(f"Train-set mapping accuracy   : {best_accuracy:.3%}")

    mapped_vq_train_full = best_mapping[raw_vq_train_full]
    mapped_vq_test_full = best_mapping[raw_vq_test_full]

    # Prototype vector for each SEMANTIC regime = the codebook
    # vector of whichever raw code maps to it. best_mapping is a
    # permutation of range(N_CODES), so argsort gives the inverse:
    # inverse_mapping[r] = the raw code c such that
    # best_mapping[c] == r.
    inverse_mapping = np.argsort(best_mapping)
    regime_prototype = codebook[inverse_mapping]  # (n_codes, latent_dim), indexed by REGIME

    # --------------------------------------------------------
    # Build trading datasets (drops each slice's final window;
    # VQ-code arrays trimmed by one to match).
    # --------------------------------------------------------

    cf_train, regime_train_true, next_ret_train, mrr_train = build_dataset(
        X_train, state_train, return_train, WINDOW_SIZE
    )
    cf_test, regime_test_true, next_ret_test, mrr_test = build_dataset(
        X_test, state_test, return_test, WINDOW_SIZE
    )

    mapped_regime_train = mapped_vq_train_full[:-1]
    mapped_regime_test = mapped_vq_test_full[:-1]
    z_e_test = z_e_test_full[:-1]

    assert len(mapped_regime_train) == len(cf_train)
    assert len(mapped_regime_test) == len(cf_test)
    assert len(z_e_test) == len(cf_test)

    # --------------------------------------------------------
    # Build one "home" expert per regime.
    # --------------------------------------------------------

    experts = {}

    for regime in range(N_CODES):

        if regime == TREE_EXPERT_REGIME:
            mask = mapped_regime_train == regime
            tree = TreeExpert()
            tree.fit(cf_train[mask], next_ret_train[mask])
            experts[regime] = (
                lambda cf, mrr, model=tree: model.predict_position(cf)
            )
        else:
            experts[regime] = HEURISTIC_EXPERTS[regime]

    # --------------------------------------------------------
    # H1: cross-deployment matrix on TEST.
    # --------------------------------------------------------

    test_matrix, test_counts = cross_deployment_matrix(
        experts, cf_test, mrr_test, next_ret_test, mapped_regime_test, N_CODES
    )

    print()
    print("=" * 70)
    print("H1: CROSS-DEPLOYMENT EXPECTANCY MATRIX (TEST)")
    print("=" * 70)
    print("Rows = expert e | Columns = deployed-on regime r")
    print(np.round(test_matrix, 6))
    print()
    print("Window counts per column (same for every expert):")
    print(test_counts[0])

    print()
    for e in range(N_CODES):
        row = test_matrix[e]
        best_r = int(np.nanargmax(row))
        verdict = "SUPPORTS H1" if best_r == e else "CONTRADICTS H1"
        print(
            f"Expert {e}: expectancy on own regime = {row[e]:+.6f} | "
            f"best regime for this expert = {best_r} "
            f"(expectancy {row[best_r]:+.6f}) -> {verdict}"
        )

    # --------------------------------------------------------
    # H2 (revised): TRAIN-side expert x regime backtest matrix
    # -- this is the "historical expectancy table" from the
    # prompt's example (every expert has a number for every
    # regime, not just its own).
    # --------------------------------------------------------

    backtest_matrix, backtest_counts = cross_deployment_matrix(
        experts, cf_train, mrr_train, next_ret_train, mapped_regime_train, N_CODES
    )

    print()
    print("=" * 70)
    print("H2 INPUT: BACKTESTED (TRAIN) EXPERT x REGIME EXPECTANCY")
    print("=" * 70)
    print("Rows = expert e | Columns = regime r (this is the table")
    print("shape from the prompt's example -- every expert has its")
    print("own historical number for every regime).")
    print(np.round(backtest_matrix, 6))

    # --------------------------------------------------------
    # Per-window similarity: for every TEST window, how similar
    # is its z_e to the codebook prototype of ITS OWN mapped
    # regime.
    # --------------------------------------------------------

    prototype_per_window = regime_prototype[mapped_regime_test]
    similarity_test = cosine_similarity_batch(z_e_test, prototype_per_window)

    # --------------------------------------------------------
    # Bucket within each regime into low/medium/high similarity
    # terciles, then compute realized expectancy for EVERY
    # expert within each (regime, bucket) cell, and compare to
    # that expert's single TRAIN number for that regime.
    # --------------------------------------------------------

    bucket_names = ["low", "medium", "high"]
    bucket_results = {}  # (expert, regime, bucket_name) -> dict

    # Per-expert full-series arrays, used later for a per-expert-only
    # correlation that doesn't stack multiple experts' rows for the
    # same window (a less-inflated, though still not perfectly
    # independent, secondary check).
    per_expert_similarity = {e: [] for e in range(N_CODES)}
    per_expert_abs_deviation = {e: [] for e in range(N_CODES)}

    # Pooled arrays PLUS the window id each row came from, so the
    # cluster-robust bootstrap below can resample at the window
    # level instead of the row level.
    pooled_similarity = []
    pooled_abs_deviation = []
    pooled_window_id = []

    for r in range(N_CODES):

        mask_r = mapped_regime_test == r
        n_r = int(mask_r.sum())

        if n_r < MIN_WINDOWS_FOR_BUCKETING:
            print(
                f"\nRegime {r}: only {n_r} test windows -- skipping "
                f"bucketing (need >= {MIN_WINDOWS_FOR_BUCKETING})."
            )
            continue

        window_id_r = np.where(mask_r)[0]  # global test-window indices

        sim_r = similarity_test[mask_r]
        q_low, q_high = np.quantile(sim_r, [1 / 3, 2 / 3])
        bucket_r = np.digitize(sim_r, [q_low, q_high])  # 0=low,1=med,2=high

        cf_r = cf_test[mask_r]
        mrr_r = mrr_test[mask_r]
        next_ret_r = next_ret_test[mask_r]

        for e in range(N_CODES):

            position_e = experts[e](cf_r, mrr_r)
            pnl_e = position_e * next_ret_r

            backtest_val = backtest_matrix[e, r]
            abs_dev_e = np.abs(pnl_e - backtest_val)

            pooled_similarity.append(sim_r)
            pooled_abs_deviation.append(abs_dev_e)
            pooled_window_id.append(window_id_r)

            per_expert_similarity[e].append(sim_r)
            per_expert_abs_deviation[e].append(abs_dev_e)

            for b, name in enumerate(bucket_names):

                bmask = bucket_r == b

                if bmask.sum() == 0:
                    continue

                realized = pnl_e[bmask].mean()
                deviation = abs(realized - backtest_val)

                bucket_results[(e, r, name)] = {
                    "realized": realized,
                    "backtest": backtest_val,
                    "deviation": deviation,
                    "n": int(bmask.sum()),
                }

    pooled_similarity = np.concatenate(pooled_similarity)
    pooled_abs_deviation = np.concatenate(pooled_abs_deviation)
    pooled_window_id = np.concatenate(pooled_window_id)

    for e in range(N_CODES):
        per_expert_similarity[e] = np.concatenate(per_expert_similarity[e])
        per_expert_abs_deviation[e] = np.concatenate(
            per_expert_abs_deviation[e]
        )

    print()
    print("=" * 70)
    print("H2: BUCKETED REALIZED EXPECTANCY vs BACKTESTED EXPECTANCY")
    print("=" * 70)
    print(
        "For each regime, ALL experts are cross-deployed, then test\n"
        "windows are bucketed by similarity to that regime's codebook\n"
        "prototype. If H2 holds, 'deviation' should shrink going\n"
        "low -> medium -> high within most (expert, regime) rows."
    )

    for r in range(N_CODES):

        rows = [
            (e, name)
            for e in range(N_CODES)
            for name in bucket_names
            if (e, r, name) in bucket_results
        ]

        if not rows:
            continue

        print(f"\n--- Regime {r} ---")

        for e in range(N_CODES):

            cells = [
                bucket_results.get((e, r, name)) for name in bucket_names
            ]

            if all(c is None for c in cells):
                continue

            pieces = []
            for name, cell in zip(bucket_names, cells):
                if cell is None:
                    pieces.append(f"{name}=n/a")
                else:
                    pieces.append(
                        f"{name}(n={cell['n']:>3d}): "
                        f"realized={cell['realized']:+.4f} "
                        f"dev={cell['deviation']:.4f}"
                    )

            backtest_val = backtest_matrix[e, r]
            print(
                f"  expert {e} | backtest={backtest_val:+.4f} | "
                + " | ".join(pieces)
            )

    # --------------------------------------------------------
    # THE PRIMARY H2 TEST: does HIGH similarity land closer to
    # the historical (expert, regime) number than LOW similarity
    # does, for every (expert, regime) pair? This is the actual
    # question -- simpler and more robust to read than any
    # correlation coefficient, and it's what the bucketed table
    # above was building toward.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("H2 PRIMARY TEST: HIGH-SIMILARITY EDGE vs LOW-SIMILARITY EDGE")
    print("=" * 70)
    print(
        "For each (expert, regime) pair with both a low and a high\n"
        "bucket: realized[high] - realized[low], and whether high\n"
        "similarity lands CLOSER to the historical backtest number\n"
        "than low similarity does (|dev_high| < |dev_low|)."
    )
    print()

    n_pairs_checked = 0
    n_pairs_supporting = 0

    for r in range(N_CODES):
        for e in range(N_CODES):

            low_cell = bucket_results.get((e, r, "low"))
            high_cell = bucket_results.get((e, r, "high"))

            if low_cell is None or high_cell is None:
                continue

            n_pairs_checked += 1

            edge_diff = high_cell["realized"] - low_cell["realized"]
            high_closer = high_cell["deviation"] < low_cell["deviation"]

            if high_closer:
                n_pairs_supporting += 1

            print(
                f"  regime {r} | expert {e} | backtest={low_cell['backtest']:+.4f} | "
                f"low_realized={low_cell['realized']:+.4f} "
                f"(dev={low_cell['deviation']:.4f}) | "
                f"high_realized={high_cell['realized']:+.4f} "
                f"(dev={high_cell['deviation']:.4f}) | "
                f"high-low={edge_diff:+.4f} | "
                f"{'SUPPORTS H2' if high_closer else 'contradicts H2'}"
            )

    if n_pairs_checked > 0:
        print()
        print(
            f"SUMMARY: {n_pairs_supporting}/{n_pairs_checked} (expert, "
            f"regime) pairs show high similarity landing closer to the "
            f"historical backtest number than low similarity "
            f"({n_pairs_supporting / n_pairs_checked:.1%})."
        )
        print(
            "This tally is directional evidence, not a rigorous test on "
            "its own -- pairs sharing the same regime's test windows, or "
            "the same expert across regimes, are not fully independent "
            "of each other either. Treat it the same way as the pooled "
            "statistics below: suggestive from a single run, not proof."
        )
    else:
        print("No (expert, regime) pairs had both a low and high bucket.")

    # --------------------------------------------------------
    # Per-expert-only correlation: similarity vs |deviation|,
    # computed SEPARATELY within each expert's own series (each
    # test window contributes exactly ONE row per expert here,
    # rather than being stacked with the other 3 experts' rows
    # for that same window). Still not perfectly independent --
    # overlapping windows and autocorrelated returns mean
    # consecutive rows within one expert's series aren't fully
    # independent either -- but it removes the specific
    # cross-expert pseudo-replication problem.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("H2 SECONDARY CHECK: PER-EXPERT-ONLY CORRELATION")
    print("=" * 70)
    print(
        "Computed separately per expert so no test window is stacked "
        "more than once within a single correlation."
    )

    for e in range(N_CODES):
        sim_e = per_expert_similarity[e]
        dev_e = per_expert_abs_deviation[e]
        r_e, p_e = pearsonr(sim_e, dev_e)
        print(
            f"  expert {e} | n={len(sim_e):>4d} | "
            f"Pearson r = {r_e:+.4f} (p = {p_e:.3e})"
        )

    # --------------------------------------------------------
    # Pooled correlation -- EXPLORATORY ONLY.
    #
    # The naive pearsonr() on the fully pooled (window, expert)
    # data is NOT statistically valid on its own: each test window
    # contributes n_codes correlated rows (one per expert), so the
    # nominal n overstates the true number of independent units.
    # The cluster-robust bootstrap below resamples at the WINDOW
    # level instead, giving a p-value / CI that actually reflects
    # the number of independent test windows rather than
    # (test windows x experts).
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("H2 EXPLORATORY: POOLED CORRELATION (NAIVE, then CLUSTER-ROBUST)")
    print("=" * 70)

    naive_pearson_r, naive_pearson_p = pearsonr(
        pooled_similarity, pooled_abs_deviation
    )
    naive_spearman_r, naive_spearman_p = spearmanr(
        pooled_similarity, pooled_abs_deviation
    )

    n_windows_pooled = len(np.unique(pooled_window_id))

    print(
        f"Naive pooled n = {len(pooled_similarity)} rows, but only "
        f"{n_windows_pooled} independent test windows underlie them "
        f"(each window repeated {N_CODES}x, once per expert)."
    )
    print(
        f"Naive Pearson  r = {naive_pearson_r:+.4f} "
        f"(p = {naive_pearson_p:.3e}) <- p-value NOT valid, "
        f"treat as descriptive only"
    )
    print(
        f"Naive Spearman rho = {naive_spearman_r:+.4f} "
        f"(p = {naive_spearman_p:.3e}) <- same caveat"
    )

    boot = cluster_bootstrap_correlation(
        pooled_similarity,
        pooled_abs_deviation,
        pooled_window_id,
        n_boot=2000,
        rng=np.random.default_rng(SEED),
    )

    print()
    print(
        f"Cluster-robust bootstrap (resampling {boot['n_clusters']} "
        f"test windows, 2000 resamples):"
    )
    print(
        f"  observed r = {boot['observed_r']:+.4f} | "
        f"95% CI = [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] | "
        f"bootstrap p = {boot['p_boot']:.3f}"
    )

    if boot["ci_low"] < 0 < boot["ci_high"]:
        print(
            "The 95% CI includes 0 -- once the true clustering is "
            "accounted for, this run does NOT give strong statistical "
            "support for H2 from the pooled data alone. The bucketed "
            "table and the high-vs-low tally above are more trustworthy "
            "reads of the same effect."
        )
    elif boot["observed_r"] < 0:
        print(
            "The 95% CI is entirely negative -- consistent with H2 even "
            "after accounting for window-level clustering."
        )
    else:
        print(
            "The 95% CI is entirely positive -- this CONTRADICTS H2 even "
            "after accounting for window-level clustering."
        )

    # --------------------------------------------------------
    # Plot: pooled similarity vs deviation, plus bucket means for
    # visual clarity.
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        pooled_similarity,
        pooled_abs_deviation,
        s=8,
        alpha=0.15,
        label="individual (window, expert) pairs",
    )

    q_edges = np.quantile(pooled_similarity, [0, 1 / 3, 2 / 3, 1])
    bucket_all = np.digitize(pooled_similarity, q_edges[1:-1])

    bucket_mean_sim = [
        pooled_similarity[bucket_all == b].mean() for b in range(3)
    ]
    bucket_mean_dev = [
        pooled_abs_deviation[bucket_all == b].mean() for b in range(3)
    ]

    plt.plot(
        bucket_mean_sim,
        bucket_mean_dev,
        "o-",
        color="red",
        linewidth=2,
        markersize=10,
        label="tercile means (low/med/high)",
    )

    plt.xlabel("Cosine similarity to regime's codebook prototype")
    plt.ylabel("|realized pnl - historical (expert, regime) expectancy|")
    plt.title(
        "H2 (exploratory): similarity vs deviation\n"
        "(points are NOT independent -- see cluster-robust bootstrap)"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "test_matrix": test_matrix,
        "backtest_matrix": backtest_matrix,
        "bucket_results": bucket_results,
        "pooled_similarity": pooled_similarity,
        "pooled_abs_deviation": pooled_abs_deviation,
        "pooled_window_id": pooled_window_id,
        "cluster_bootstrap": boot,
        "per_expert_similarity": per_expert_similarity,
        "per_expert_abs_deviation": per_expert_abs_deviation,
        "n_pairs_checked": n_pairs_checked,
        "n_pairs_supporting": n_pairs_supporting,
    }


if __name__ == "__main__":
    main()

"""
ATTEMPT 1
Rejected hypothesis:
Higher similarity score != closer edge expectancy to backtested result.

Why:
Similar market states do not necessarily preserve the conditions
that generated the strategy's edge. Representation similarity
is not equivalent to payoff similarity.

Possible solutions:

1. Edge-based regimes
   Make regime discovery strategy-dependent.
   
   Objective:
   Use AE to identify market regions where the strategy has positive
   conditional expectancy after costs/slippage.
   
   Advantage:
   Much simpler system. Instead of:
       latent state → similarity → edge transfer
   directly estimate:
       state → conditional strategy expectancy

   Risk:
   Edge is non-stationary, so regime definitions and expectancy
   must be updated chronologically. Charting edge expectancy
   of a strategy shows that its not persistent.

2. Temporal encoder / TCN
   Hypothesis:
   Strategy edge may depend on temporal dynamics rather than
   static feature configuration. A causal TCN may encode these
   patterns more efficiently and consistently than the MLP. 
   
   Objective:
   Use AE to identify and derive latent spaces such that they're 
   persistent, then have experts train on it.

   Advantage:
    Possibly doesn't violate the persistency assumption as:
        n market data -> AE encodes persistent dimensions -> persistent state
    Could adapt to non-stationarity of the market as objective
    focuses on "trading only when you've seen this market before"

    Disadvantage:
        Overfitting risk, which may violate assumption of recurring
        Much more complex so tracking executional mistakes might be
        harder.

   Test:
   Ordered MLP vs shuffled MLP vs parameter-matched causal TCN.
"""