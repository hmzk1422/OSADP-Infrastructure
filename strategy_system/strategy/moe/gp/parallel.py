"""Pair-loop parallelism for offline GP search (threads, never per live row).

Two entry points:

* ``search_pairs_sequential`` -- the ORIGINAL loop (kept verbatim and
  bit-identical) for audits and single-core machines.
* ``search_pairs_threaded`` -- the same pair computations run with a
  ``ThreadPoolExecutor`` over independent (state, target) pairs.

Threading is justified by the repo-local micro-benchmark
(``_bench_gp_micro.py``, 16 cores): walk-forward ridge/ols fits speed up
~1.8x and pandas expression evaluation ~2-2.9x with 2-8 workers, because the
hot loops live in numpy/pandas C code that releases the GIL. Gains plateau
past 4 workers, so ``workers`` defaults to min(4, cpu_count).

Determinism contract
--------------------
The original search shares ONE ``np.random.default_rng(seed)`` across all
pairs, so run order (and worker completion order) would leak into results.
Both entry points therefore derive a per-pair generator from
``np.random.SeedSequence(seed)`` keyed by ``(state_id, target_column)``.
Sequential and threaded runs are bit-identical to each other; they differ
from the legacy shared-RNG stream (documented in the GP README caveat
"Run order affects results").
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np


def derive_pair_seed(seed: int, state_id: str, target_column: str) -> int:
    """Deterministic per-pair seed; independent of pair evaluation order.

    ``np.random.SeedSequence`` entropy must be integers, so the string identity
    of the pair is folded in with a stable (non-salted) hash -- ``hash()`` is
    randomized per process and would break reproducibility across runs.
    """
    import hashlib

    digest = hashlib.blake2b(
        f"{int(seed)}|{state_id}|{target_column}".encode("utf-8"), digest_size=8
    ).digest()
    pair_tag = int.from_bytes(digest, "big")
    seq = np.random.SeedSequence([int(seed), pair_tag])
    return int(seq.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def default_worker_count(n_pairs: int) -> int:
    """Cap parallelism where the benchmark shows gains plateau (~4)."""
    import os

    cpu = os.cpu_count() or 4
    return int(max(1, min(4, cpu, max(1, n_pairs))))


def run_pairs(
    pair_fn,
    pairs: list[tuple[str, str]],
    *,
    workers: int = 1,
    worker_count_fn=default_worker_count,
    show_progress: bool = False,
) -> list[Any]:
    """Run ``pair_fn(state_id, target_column)`` over pairs, in input order.

    ``workers=1`` is the sequential path (legacy behavior, same order).
    ``workers>1`` uses a thread pool sized by ``worker_count_fn``; results
    are returned in input order regardless of completion order. ``pair_fn``
    must be thread-safe: no shared RNG, no shared mutable caches.
    """
    pairs = list(pairs)
    if not pairs:
        return []
    if workers is None or workers <= 1:
        return [pair_fn(state_id, target) for state_id, target in pairs]
    max_workers = worker_count_fn(len(pairs))
    if max_workers <= 1:
        return [pair_fn(state_id, target) for state_id, target in pairs]
    if show_progress:
        from tqdm.auto import tqdm

        pairs = tqdm(pairs, desc="GP pair search")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda pair: pair_fn(*pair), pairs))
