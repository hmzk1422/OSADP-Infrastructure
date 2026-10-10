"""Empirical proof of the live TDMI temporal-matrix bridging bug.

Builds a regimes frame where ONE latent's "episode" is interrupted by another
latent, so its rows are split into two contiguous runs with different means.
If the estimator bridges across the separator, MI is spuriously large.
"""
import numpy as np
import pandas as pd

from strategy_system.strategy.moe.ood import metrics as m
from strategy_system.strategy.moe.ood.regime_validator import RegimeValidator


def split_runs(path):
    runs, cur = [], []
    for row in path:
        if np.isfinite(row).all():
            cur.append(row)
        elif cur:
            runs.append(np.array(cur)); cur = []
    if cur:
        runs.append(np.array(cur))
    return runs


print("=" * 70)
print("PROOF 1: estimator is correct when given SEPARATE runs")
print("=" * 70)
rng = np.random.default_rng(13)
a = rng.normal(-5.0, 1.0, 1500)
b = rng.normal(5.0, 1.0, 1500)
runs = split_runs(np.vstack([a[:, None], np.full((1, 1), np.nan), b[:, None]]))
print("  runs detected from NaN-separated path:", len(runs))
est = m.tdmi_estimate(runs, bins=4, min_pairs=4, n_surrogates=0)
print("  within-episode MI (expect ~0):", round(float(np.nanmean(est.mi)), 4))

pooled = np.concatenate([a[:-1], b[:-1]])
nxt = np.concatenate([a[1:], b[1:]])
bridged = m._mi_batch(m._rank_bins(pooled, 4), m._rank_bins(nxt, 4)[None, :], 4)[0]
print("  bridged/pooled MI (BUG if >>0):", round(float(bridged), 4))

print()
print("=" * 70)
print("PROOF 2: LIVE build_temporal_similarity_matrix bridges the two runs")
print("=" * 70)
# Latent A occupies rows 0..1499 and 1501..3000; latent Z interrupts at row 1500.
n = 3001
emb = np.zeros((n, 2))
emb[:1500] = a[:, None]
emb[1500] = np.nan
emb[1501:] = b[:, None]
labels = np.array(["A"] * n, dtype=object)
labels[1500] = "Z"  # the interrupting latent
regimes = pd.DataFrame({"latent_id": labels}, index=pd.RangeIndex(n))
embed = pd.DataFrame(emb, index=regimes.index)

v = RegimeValidator(regime_column="latent_id")
# Monkey-check what the matrix actually feeds tdmi for latent A vs itself:
# it masks rows to A, dropping the Z row and CONCATENATING the two runs.
ids = pd.Series(labels).astype(str).to_numpy()
masked = emb[(ids == "A")]  # exactly what the live loop does at L304-305
print("  masked A rows:", masked.shape, "(Z separator row dropped, runs joined)")
print("  masked A is fully finite everywhere:", bool(np.isfinite(masked).all()))

sim = v.build_temporal_similarity_matrix(embed, labels)
aa = sim.loc["A", "A"]
print("  A-A temporal similarity:", round(float(aa), 4))

# Compare: fingerprint over the JOINED run vs the two SEPARATE runs.
try:
    m.tdmi_fingerprint(masked, bins=4, min_pairs=4, n_surrogates=0)
    print("  fingerprint with n_surrogates=0: OK")
except Exception as exc:  # noqa: BLE001
    print("  fingerprint with n_surrogates=0 RAISED:", type(exc).__name__, "-", exc)

joined_fp = m.tdmi_fingerprint(masked, bins=4, min_pairs=4, n_surrogates=49, seed=0)
split_fp = m.tdmi_fingerprint(split_runs(emb), bins=4, min_pairs=4, n_surrogates=49, seed=0)
print("  fingerprint (JOINED) max cell:", round(float(np.nanmax(joined_fp)), 4))
print("  fingerprint (SEPARATE runs) max cell:", round(float(np.nanmax(split_fp)), 4))
print()
if float(np.nanmax(joined_fp)) > 5 * max(float(np.nanmax(split_fp)), 1e-9):
    print("  => BUG CONFIRMED: joining runs inflates TDMI (spurious dependence).")
else:
    print("  => joining did not inflate TDMI here.")

print()
print("=" * 70)
print("PROOF 3: resolve_metrics accepts 'tdmi' but _temporal_metric() rejects it")
print("=" * 70)
print("  SEQUENCE_METRICS has tdmi:", "tdmi" in m.SEQUENCE_METRICS)
resolved = m.resolve_metrics({"temporal_pattern_similarity": "tdmi"})
print("  resolve_metrics accepts tdmi ->", resolved["temporal_pattern_similarity"])
try:
    v2 = RegimeValidator(regime_column="latent_id",
                         metrics={"temporal_pattern_similarity": "tdmi"})
    fn = v2._temporal_metric()
    print("  _temporal_metric() returned:", fn)
except Exception as exc:  # noqa: BLE001
    print("  _temporal_metric() RAISED:", type(exc).__name__, "-", exc)
