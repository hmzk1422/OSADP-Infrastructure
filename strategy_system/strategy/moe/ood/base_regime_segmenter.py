"""Swappable persistent segmenters and recurrence-aware prototype registry.

The current hysteresis / sticky-HMM implementations are kept verbatim as the
``"custom"`` segmenter path: ``HysteresisClusterRegimeSegmenter.predict``
only dispatches, and its body moved to ``predict_custom``. New library
backends (``"kmeans"`` baseline, ``"hdbscan"``) implement the same
``BaseRegimeSegmenter`` contract via ``make_segmenter``: offline fit on a
training embedding slice, frozen prototypes + radius rule at inference, and
the same causal min-dwell debounce.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from . import _jit


_COLUMNS = ["latent_id", "raw_latent_id", "similarity", "distance", "is_known",
            "candidate_count", "candidate_episodes", "known_state_count", "state_support",
            "transition_pending", "status"]

#: Segmenter backends accepted by ``backend``: the online state machine
#: (``"custom"``) plus the offline sklearn discovery backends.
SEGMENTER_BACKENDS = ("custom", "kmeans", "hdbscan", "sklearn")

#: Backends whose prototypes are discovered offline and then frozen.  These
#: share one assignment geometry (see ``_SklearnBackend._assign``) and do not
#: support online novelty learning.
FROZEN_BACKENDS = ("kmeans", "hdbscan", "sklearn")


def _match_one_to_one(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost one-to-one row/column matching (Hungarian when available).

    ``cost`` is ``(n_old, n_new)``; returns ``(old_index, new_index)`` pairs.
    Falls back to a greedy nearest assignment if SciPy is unavailable so the
    dependency stays optional.
    """
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except Exception:
        pairs: list[tuple[int, int]] = []
        used_cols: set[int] = set()
        for r in range(cost.shape[0]):
            order = np.argsort(cost[r])
            for c in order:
                if int(c) not in used_cols:
                    used_cols.add(int(c))
                    pairs.append((r, int(c)))
                    break
        return pairs


class BaseRegimeSegmenter(ABC):
    """Interface for embedding-to-regime assignment.

    Implementations own prototype IDs and must treat them as persistent stored
    identities, not semantic/numeric labels. ``predict`` may update online
    recurrence and persistence state.
    """

    schema_version = 1

    @classmethod
    def available_backends(cls) -> tuple[str, ...]:
        """Names accepted by ``backend`` (online custom + library fits)."""
        return SEGMENTER_BACKENDS

    @abstractmethod
    def fit(self, embeddings: pd.DataFrame, y=None) -> "BaseRegimeSegmenter":
        """Initialize from a chronological embedding sequence."""

    @abstractmethod
    def predict(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Return aligned assignments, retaining null until recurrence support."""

    @abstractmethod
    def predict_proba(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Return confidence by known prototype without mutating online state."""

    @abstractmethod
    def assign_regime(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Find nearest known prototype and its normalized distance."""

    @abstractmethod
    def detect_regime_transition(self, embedding: np.ndarray) -> bool:
        """Report whether the embedding supports a state transition."""

    @abstractmethod
    def enforce_persistence(self, labels: Sequence[str]) -> np.ndarray:
        """Apply causal persistence/hysteresis and return aligned labels."""

    @abstractmethod
    def update_state(self, embedding: np.ndarray, position: int) -> dict:
        """Update known prototypes or recurrence candidates for one row."""

    @abstractmethod
    def get_regime_prototypes(self) -> dict[str, np.ndarray]:
        """Return a copy of the stored regime prototypes."""

    @abstractmethod
    def get_regime_counts(self) -> dict[str, int]:
        """Return support counts for promoted regimes."""

    @abstractmethod
    def get_observations_seen(self) -> int:
        """Return the stream position for the next online observation."""

    @abstractmethod
    def get_model_ready_states(self) -> set[str]:
        """Return states with separately fitted/deployed models."""

    @abstractmethod
    def set_model_ready_states(self, state_ids: Sequence[str]) -> None:
        """Set model-ready regime IDs."""

class HysteresisClusterRegimeSegmenter(BaseRegimeSegmenter):
    """Online prototype clustering with persistence and multi-episode promotion.

    Scale semantics: with ``centroid_update_rate == 0`` (default) the per-state
    scale is the RMS deviation of assigned rows around the fixed prototype.  With a
    nonzero rate the accumulator is not a standard Welford variance of a moving
    centre, so scales are a heuristic dispersion, not a calibrated estimate.
    """

    def __init__(
        self, max_regimes: int = 10, match_threshold: float = 1.5,
        candidate_threshold: float = 0.25, min_candidate_observations: int = 200,
        min_candidate_episodes: int = 3, candidate_episode_gap: int = 32,
        max_candidates: int = 128, candidate_expiry: int = 5000,
        min_dwell: int = 5, switch_margin: float = 0.0,
        centroid_update_rate: float = 0.0, min_state_scale: float = 0.1,
        state_expiry_observations: int = 100000,
        learn_novel: bool = True,
        model_ready_states: Sequence[str] | None = None,
        *,
        backend: str = "custom",
        backend_params: dict | None = None,
    ):
        for name, value, minimum in (("max_regimes", max_regimes, 1),
                                     ("min_candidate_observations", min_candidate_observations, 2),
                                     ("min_candidate_episodes", min_candidate_episodes, 2),
                                     ("candidate_episode_gap", candidate_episode_gap, 1),
                                     ("max_candidates", max_candidates, 1), ("candidate_expiry", candidate_expiry, 1),
                                     ("state_expiry_observations", state_expiry_observations, 1),
                                     ("min_dwell", min_dwell, 1)):
            if not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (("match_threshold", match_threshold), ("candidate_threshold", candidate_threshold)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.max_regimes = int(max_regimes)
        self.match_threshold, self.candidate_threshold = float(match_threshold), float(candidate_threshold)
        self.min_candidate_observations, self.min_candidate_episodes = int(min_candidate_observations), int(min_candidate_episodes)
        self.candidate_episode_gap, self.min_dwell = int(candidate_episode_gap), int(min_dwell)
        self.max_candidates, self.candidate_expiry = int(max_candidates), int(candidate_expiry)
        self.switch_margin = float(switch_margin)
        # rate == 1 is degenerate: the center jumps to the observation, so the
        # Welford scale accumulator receives zero for that row while
        # scale_counts still grows, collapsing the scale to its floor.  Reject
        # it rather than silently producing a meaningless dispersion estimate.
        if not np.isfinite(centroid_update_rate) or not 0.0 <= centroid_update_rate < 1.0:
            raise ValueError("centroid_update_rate must be in [0, 1) (1.0 is degenerate)")
        if not np.isfinite(min_state_scale) or min_state_scale <= 0:
            raise ValueError("min_state_scale must be finite and positive")
        self.centroid_update_rate = float(centroid_update_rate)
        self.min_state_scale = float(min_state_scale)
        self.state_expiry_observations = int(state_expiry_observations)
        self.learn_novel = bool(learn_novel)
        self.model_ready_states = set(map(str, model_ready_states or ()))
        if backend not in SEGMENTER_BACKENDS:
            raise ValueError(f"backend must be one of {SEGMENTER_BACKENDS}")
        self.backend = backend
        self.backend_params = dict(backend_params or {})
        self._backend_obj = None
        self.prototypes: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.scales: dict[str, np.ndarray] = {}
        self.scale_m2: dict[str, np.ndarray] = {}
        self.scale_counts: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.candidates: list[dict] = []
        self._next_id = 1
        self.observations_seen = 0
        self.current_state: str | None = None
        self._pending_state: str | None = None
        self._pending_count = 0
        self._persistent_label: str | None = "null"
        self._smooth_pending: str | None = None
        self._smooth_pending_count = 0
        self._transition_pending_rows: list[bool] = []
        # Frozen-backend metadata (populated by _SklearnBackend.fit):
        #   _fitted_columns  -> exact feature order validated at inference.
        #   _radius_meta     -> per-prototype fitted radius + whether the
        #                       min_state_scale floor bound the metric.
        #   _support_detail  -> {sid: {"cluster": H, "in_radius": I, ...}}.
        #   _fit_meta        -> discovery/inference bookkeeping for diagnostics.
        self._fitted_columns: list[str] | None = None
        self._radius_meta: dict[str, dict] = {}
        self._support_detail: dict[str, dict] = {}
        self._fit_meta: dict = {}
        # Every regime ID ever issued (monotonic, never reused across refits) so
        # a retired latent_xxx cannot be silently reassigned to a new regime.
        self._issued_ids: set[str] = set()

    def _distance(self, value: np.ndarray, center: np.ndarray, scale: np.ndarray | None = None) -> float:
        if value.shape != center.shape or not np.isfinite(value).all():
            return np.inf
        normalized = value - center if scale is None else (value - center) / np.maximum(scale, self.min_state_scale)
        return float(np.sqrt(np.mean(normalized ** 2)))

    def _prune_states(self, position: int) -> None:
        expired = [state_id for state_id, seen in self.last_seen.items()
                   if position - seen > self.state_expiry_observations]
        for state_id in expired:
            self.prototypes.pop(state_id, None)
            self.scales.pop(state_id, None)
            self.scale_m2.pop(state_id, None)
            self.scale_counts.pop(state_id, None)
            self.counts.pop(state_id, None)
            self.last_seen.pop(state_id, None)
            self.model_ready_states.discard(state_id)
        if self.current_state not in self.prototypes:
            self.current_state = None

    def assign_regime(self, embedding: np.ndarray) -> tuple[str | None, float]:
        if self.backend in FROZEN_BACKENDS and self.prototypes:
            return self._backend().assign_regime(embedding)
        value = np.asarray(embedding, dtype=np.float64)
        pairs = [(key, self._distance(value, center, self.scales.get(key))) for key, center in self.prototypes.items()]
        return min(pairs, key=lambda pair: pair[1]) if pairs else (None, np.inf)

    def detect_regime_transition(self, embedding: np.ndarray) -> bool:
        if self.backend in FROZEN_BACKENDS and self.prototypes:
            return self._backend().detect_regime_transition(embedding)
        state_id, distance = self.assign_regime(np.asarray(embedding, dtype=np.float64))
        return state_id != self.current_state and distance <= self.match_threshold

    def predict_proba(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        if self.backend in FROZEN_BACKENDS and self.prototypes:
            return self._backend().predict_proba(embeddings)
        ids = list(self.prototypes)
        probabilities = pd.DataFrame(0.0, index=embeddings.index, columns=ids)
        for row, value in enumerate(embeddings.to_numpy(dtype=np.float64)):
            distances = np.asarray([self._distance(value, self.prototypes[key], self.scales[key]) for key in ids])
            valid = np.isfinite(distances)
            if valid.any():
                scores = np.zeros(len(ids), dtype=np.float64)
                scores[valid] = np.exp(-0.5 * distances[valid] ** 2)
                if scores.sum() > 0:
                    probabilities.iloc[row] = scores / scores.sum()
        return probabilities

    def get_regime_prototypes(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self.prototypes.items()}

    def get_regime_counts(self) -> dict[str, int]:
        return dict(self.counts)

    def get_observations_seen(self) -> int:
        return int(self.observations_seen)

    def get_model_ready_states(self) -> set[str]:
        return set(self.model_ready_states)

    def set_model_ready_states(self, state_ids: Sequence[str]) -> None:
        self.model_ready_states = set(map(str, state_ids))

    def enforce_persistence(self, labels: Sequence[str]) -> np.ndarray:
        """Causally debounce every label, including null, for ``min_dwell`` rows.

        Numba path: labels are factorized to integer codes, the debounce runs as
        one compiled loop, and codes map back to the original strings.  State
        (``_persistent_label`` etc.) is carried across calls exactly as before.
        Falls back to the reference loop when numba is unavailable or the
        carried state is not a plain string label.
        """
        if (not _jit.NUMBA_AVAILABLE or not isinstance(self._persistent_label, str)
                or not self._persistent_label
                or not (self._smooth_pending is None or isinstance(self._smooth_pending, str))):
            return self._enforce_persistence_python(labels)
        raw = np.asarray(labels, dtype=object)
        if raw.size == 0:
            return self._enforce_persistence_python(labels)
        if pd.api.types.infer_dtype(raw, skipna=False) != "string":
            raw = np.array([str(label) for label in raw], dtype=object)
        codes, uniques = pd.factorize(raw)
        names = list(uniques)
        for extra in (self._persistent_label, self._smooth_pending):
            if extra is not None and extra not in names:
                names.append(extra)
        lookup = {name: code for code, name in enumerate(names)}
        state = np.array([lookup[self._persistent_label],
                          -1 if self._smooth_pending is None else lookup[self._smooth_pending],
                          self._smooth_pending_count], dtype=np.int64)
        out = np.empty(len(codes), dtype=np.int64)
        pending_rows = np.empty(len(codes), dtype=np.bool_)
        _jit.persistence_kernel(codes.astype(np.int64), state, self.min_dwell, out, pending_rows)
        self._persistent_label = names[int(state[0])]
        self._smooth_pending = None if state[1] < 0 else names[int(state[1])]
        self._smooth_pending_count = int(state[2])
        self._transition_pending_rows = pending_rows.tolist()
        return np.asarray(names, dtype=object)[out]

    def _enforce_persistence_python(self, labels: Sequence[str]) -> np.ndarray:
        """Causally debounce every label, including null, for ``min_dwell`` rows."""
        output = []
        pending_rows = []
        for label in map(str, labels):
            if label == self._persistent_label:
                output.append(self._persistent_label)
                self._smooth_pending, self._smooth_pending_count = None, 0
            elif label == self._smooth_pending:
                self._smooth_pending_count += 1
                if self._smooth_pending_count >= self.min_dwell:
                    self._persistent_label = label
                    self._smooth_pending, self._smooth_pending_count = None, 0
                output.append(self._persistent_label or "null")
            else:
                self._smooth_pending, self._smooth_pending_count = label, 1
                if self.min_dwell <= 1:
                    self._persistent_label, self._smooth_pending, self._smooth_pending_count = label, None, 0
                output.append(self._persistent_label or "null")
            pending_rows.append(label != self._persistent_label)
        self._transition_pending_rows = pending_rows
        return np.asarray(output, dtype=object)

    def _promote(self, candidate: dict) -> str | None:
        if len(self.prototypes) >= self.max_regimes:
            return None
        state_id = f"latent_{self._next_id:03d}"
        self._next_id += 1
        self._issued_ids.add(state_id)   # retired IDs must never be re-issued by any backend
        self.prototypes[state_id] = candidate["center"].copy()
        variance = candidate["m2"] / max(candidate["count"] - 1, 1)
        self.scales[state_id] = np.maximum(np.sqrt(np.maximum(variance, 0)), self.min_state_scale)
        self.scale_m2[state_id] = candidate["m2"].copy()
        self.scale_counts[state_id] = int(candidate["count"])
        self.counts[state_id] = 0
        self.last_seen[state_id] = int(candidate["last_seen"])
        return state_id

    def _candidate_update(self, value: np.ndarray, position: int) -> tuple[int, int, str]:
        if not self.learn_novel:
            return 0, 0, "learning_disabled"
        self.candidates = [candidate for candidate in self.candidates
                           if position - candidate["last_seen"] <= self.candidate_expiry]
        pairs = [(i, self._distance(value, c["center"])) for i, c in enumerate(self.candidates)]
        valid = [(i, d) for i, d in pairs if np.isfinite(d)]
        idx, dist = min(valid, key=lambda pair: pair[1]) if valid else (None, np.inf)
        if idx is None or dist > self.candidate_threshold:
            if len(self.candidates) >= self.max_candidates:
                oldest = min(range(len(self.candidates)), key=lambda i: self.candidates[i]["last_seen"])
                self.candidates.pop(oldest)
            candidate = {"center": value.copy(), "m2": np.zeros_like(value), "count": 1,
                         "episodes": 1, "last_seen": position}
            self.candidates.append(candidate)
            idx = len(self.candidates) - 1
        else:
            candidate = self.candidates[idx]
            if position - candidate["last_seen"] > self.candidate_episode_gap:
                candidate["episodes"] += 1
            candidate["count"] += 1
            delta = value - candidate["center"]
            candidate["center"] += delta / candidate["count"]
            candidate["m2"] += delta * (value - candidate["center"])
            candidate["last_seen"] = position
        count, episodes = int(candidate["count"]), int(candidate["episodes"])
        if count >= self.min_candidate_observations and episodes >= self.min_candidate_episodes:
            existing = [
                (state_id, self._distance(candidate["center"], center, self.scales[state_id]))
                for state_id, center in self.prototypes.items()
            ]
            if existing:
                state_id, distance = min(existing, key=lambda item: item[1])
                if distance <= self.match_threshold:
                    self.candidates.pop(idx)
                    return count, episodes, "merged"
            state_id = self._promote(candidate)
            if state_id is not None:
                self.candidates.pop(idx)
                return count, episodes, "promoted"
            return count, episodes, "capacity"
        return count, episodes, "learning"

    def update_state(self, embedding: np.ndarray, position: int) -> dict:
        if self.backend in FROZEN_BACKENDS:
            raise NotImplementedError(
                "Frozen segmenters do not support online state updates; "
                "use predict() for inference.")
        self._prune_states(position)
        value = np.asarray(embedding, dtype=np.float64)
        if value.ndim != 1 or not np.isfinite(value).all():
            return {"latent_id": "null", "similarity": 0.0, "distance": np.nan,
                    "candidate_count": 0, "candidate_episodes": 0, "state_support": 0, "status": "warmup"}
        state_id, distance = self.assign_regime(value)
        allowed_distance = self.match_threshold
        if state_id == self.current_state:
            allowed_distance += self.switch_margin
        if state_id is not None and distance <= allowed_distance:
            self.counts[state_id] += 1
            residual = value - self.prototypes[state_id]
            updated_center = self.prototypes[state_id] + self.centroid_update_rate * residual
            adjusted_residual = value - updated_center
            self.scale_m2[state_id] += residual * adjusted_residual
            self.prototypes[state_id] = updated_center
            self.scale_counts[state_id] += 1
            self.scales[state_id] = np.maximum(
                np.sqrt(np.maximum(self.scale_m2[state_id], 0) / max(self.scale_counts[state_id] - 1, 1)),
                self.min_state_scale)
            self.last_seen[state_id] = position
            # Candidate learned prototype remains frozen as an identity anchor.
            self.current_state = state_id
            return {"latent_id": state_id, "similarity": float(np.exp(-distance / max(allowed_distance, 1e-12))),
                    "distance": distance, "candidate_count": 0, "candidate_episodes": 0,
                    "state_support": self.counts[state_id], "status": "known"}
        count, episodes, status = self._candidate_update(value, position)
        self.current_state = None
        return {"latent_id": "null", "similarity": float(np.exp(-max(distance, 0) / self.match_threshold)) if np.isfinite(distance) else 0.0,
                "distance": distance if np.isfinite(distance) else np.nan, "candidate_count": count,
                "candidate_episodes": episodes, "state_support": 0, "status": status}

    def _backend(self):
        """Return (and cache) the sklearn backend for non-custom backends."""
        if self.backend == "custom":
            return None
        if self._backend_obj is None:
            self._backend_obj = _SklearnBackend(self, self.backend)
        return self._backend_obj

    def fit(self, embeddings: pd.DataFrame, y=None) -> "HysteresisClusterRegimeSegmenter":
        """Dispatch to the selected backend; always returns the model (self)."""
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        if self.backend == "custom":
            return self.fit_custom(embeddings)
        return self._backend().fit(embeddings)

    def fit_custom(self, embeddings: pd.DataFrame) -> "HysteresisClusterRegimeSegmenter":
        """Current online candidate/promotion fit (kept verbatim)."""
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        self.predict_custom(embeddings)
        return self

    def predict(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Dispatch to the selected backend; always returns the assignment frame."""
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        if self.backend == "custom":
            return self.predict_custom(embeddings)
        return self._backend().predict(embeddings)

    #: Methods the compiled kernel re-implements.  If a subclass overrides any
    #: of them (e.g. ``StickyHMMRegimeSegmenter.assign_regime``) the kernel
    #: would silently skip that behaviour, so those classes use the Python path.
    _JIT_INLINED = ("update_state", "assign_regime", "_distance", "_prune_states",
                    "_candidate_update", "_promote")

    def predict_custom(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Online state machine; runs as a compiled numba kernel when possible.

        Results and end-of-run state are identical to the reference Python loop
        (``_predict_custom_python``), which is used automatically when numba is
        missing, a subclass overrides the inlined methods, or the carried state
        is not representable in the kernel's array layout.
        """
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        result = self._predict_custom_jit(embeddings)
        return result if result is not None else self._predict_custom_python(embeddings)

    #: Target size of the float64 working block copied per kernel call.  Keeps the
    #: extra RAM of the JIT path O(chunk) instead of O(rows * features).
    _JIT_CHUNK_BYTES = 64 * 2**20

    def _predict_custom_jit(self, embeddings: pd.DataFrame, chunk_rows: int | None = None):
        base = HysteresisClusterRegimeSegmenter
        if (not _jit.NUMBA_AVAILABLE or self.backend != "custom" or len(embeddings) == 0
                or any(getattr(type(self), name) is not getattr(base, name) for name in self._JIT_INLINED)
                or not isinstance(self._persistent_label, str) or not self._persistent_label):
            return None
        if self.prototypes and embeddings.shape[1] != len(next(iter(self.prototypes.values()))):
            raise ValueError("embedding dimensions differ from fitted prototypes")
        # Validate dtypes up front so a bad column raises (via the Python path)
        # before any state is mutated, as the original did.  The float64 matrix
        # itself is converted per chunk below -- never as one full-size copy.
        if not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                   for dtype in embeddings.dtypes):
            return None
        n, size = embeddings.shape
        if chunk_rows is None:
            chunk_rows = int(max(1024, min(250_000, self._JIT_CHUNK_BYTES // max(size * 8, 1))))
        ids = list(self.prototypes)
        # Everything carried in must already be (size,)-shaped and complete.
        if (any(set(table) != set(ids) for table in (self.scales, self.scale_m2, self.scale_counts,
                                                      self.counts, self.last_seen))
                or any(np.shape(self.prototypes[k]) != (size,) or np.shape(self.scales[k]) != (size,)
                       or np.shape(self.scale_m2[k]) != (size,) for k in ids)
                or any(np.shape(c["center"]) != (size,) or np.shape(c["m2"]) != (size,) for c in self.candidates)):
            return None

        capacity = max(self.max_regimes, len(ids)) + 1
        cand_capacity = max(self.max_candidates, len(self.candidates)) + 1
        P = np.zeros((capacity, size)); S = np.zeros((capacity, size)); M2 = np.zeros((capacity, size))
        SC = np.zeros(capacity, np.int64); CNT = np.zeros(capacity, np.int64)
        LAST = np.zeros(capacity, np.int64); UID = np.zeros(capacity, np.int64)
        for k, key in enumerate(ids):
            P[k], S[k], M2[k] = self.prototypes[key], self.scales[key], self.scale_m2[key]
            SC[k], CNT[k], LAST[k], UID[k] = self.scale_counts[key], self.counts[key], self.last_seen[key], k
        CC = np.zeros((cand_capacity, size)); CM2 = np.zeros((cand_capacity, size))
        CCNT = np.zeros(cand_capacity, np.int64); CEP = np.zeros(cand_capacity, np.int64)
        CLAST = np.zeros(cand_capacity, np.int64)
        for k, c in enumerate(self.candidates):
            CC[k], CM2[k] = c["center"], c["m2"]
            CCNT[k], CEP[k], CLAST[k] = c["count"], c["episodes"], c["last_seen"]
        uid_names = list(ids)
        current = ids.index(self.current_state) if self.current_state in self.prototypes else -1
        ints = np.array([len(ids), len(self.candidates), len(ids), current, 0, 0], dtype=np.int64)
        cf = np.array([self.match_threshold, self.candidate_threshold, self.switch_margin,
                       self.centroid_update_rate, self.min_state_scale], dtype=np.float64)
        ci = np.array([self.max_regimes, self.min_candidate_observations, self.min_candidate_episodes,
                       self.candidate_episode_gap, self.max_candidates, self.candidate_expiry,
                       self.state_expiry_observations, int(self.learn_novel)], dtype=np.int64)
        uid_out = np.empty(n, np.int64); sim_arg = np.empty(n); dist_out = np.empty(n)
        cand_count = np.empty(n, np.int64); cand_eps = np.empty(n, np.int64)
        support = np.empty(n, np.int64); status = np.empty(n, np.int8); known_count = np.empty(n, np.int64)
        pruned_log = np.empty(capacity + n + 1, np.int64)
        pos0 = self.observations_seen
        next_id = self._next_id

        with tqdm(total=n, desc="Segmenting Regimes") as bar:
            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                promoted_before = int(ints[5])
                block = np.ascontiguousarray(embeddings.iloc[start:stop].to_numpy(dtype=np.float64))
                _jit.segment_chunk(block, 0, stop - start, pos0 + start, P, S, M2, SC, CNT, LAST, UID,
                                   CC, CM2, CCNT, CEP, CLAST, ints, cf, ci,
                                   uid_out[start:stop], sim_arg[start:stop], dist_out[start:stop],
                                   cand_count[start:stop], cand_eps[start:stop], support[start:stop],
                                   status[start:stop], known_count[start:stop], pruned_log)
                del block
                for _ in range(int(ints[5]) - promoted_before):   # ids are issued in promotion order
                    uid_names.append(f"latent_{next_id:03d}")
                    self._issued_ids.add(uid_names[-1])
                    next_id += 1
                bar.update(stop - start)

        # ---- write the carried state back in the original dict/list shapes ----
        nk, nc = int(ints[0]), int(ints[1])
        keys = [uid_names[int(UID[k])] for k in range(nk)]
        self.prototypes = {key: P[k].copy() for k, key in enumerate(keys)}
        self.scales = {key: S[k].copy() for k, key in enumerate(keys)}
        self.scale_m2 = {key: M2[k].copy() for k, key in enumerate(keys)}
        self.scale_counts = {key: int(SC[k]) for k, key in enumerate(keys)}
        self.counts = {key: int(CNT[k]) for k, key in enumerate(keys)}
        self.last_seen = {key: int(LAST[k]) for k, key in enumerate(keys)}
        self.candidates = [{"center": CC[k].copy(), "m2": CM2[k].copy(), "count": int(CCNT[k]),
                            "episodes": int(CEP[k]), "last_seen": int(CLAST[k])} for k in range(nc)]
        for k in range(int(ints[4])):
            self.model_ready_states.discard(uid_names[int(pruned_log[k])])
        self._next_id = next_id
        self.observations_seen = pos0 + n
        self.current_state = uid_names[int(ints[3])] if ints[3] >= 0 else None

        names = np.array(uid_names + ["null"], dtype=object)
        raw_ids = names[uid_out].tolist()                    # uid -1 -> "null"
        status_names = np.array(_jit.STATUS_NAMES, dtype=object)[status].tolist()
        result = pd.DataFrame({
            "latent_id": raw_ids, "raw_latent_id": raw_ids,
            "similarity": np.exp(sim_arg),                   # np.exp (not in-kernel) keeps bits identical
            "distance": dist_out, "is_known": [value != "null" for value in raw_ids],
            "candidate_count": cand_count, "candidate_episodes": cand_eps,
            "known_state_count": known_count, "state_support": support,
            "transition_pending": np.zeros(n, dtype=bool), "status": status_names,
        }, index=embeddings.index, columns=_COLUMNS)
        result["latent_id"] = self.enforce_persistence(result["raw_latent_id"])
        result["transition_pending"] = self._transition_pending_rows
        result["is_known"] = result["latent_id"].ne("null")
        return result

    def _predict_custom_python(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Reference (pure Python) state-machine body, kept verbatim for fallback/testing."""
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        if self.prototypes and embeddings.shape[1] != len(next(iter(self.prototypes.values()))):
            raise ValueError("embedding dimensions differ from fitted prototypes")
        rows = []
        values = embeddings.to_numpy(dtype=np.float64)
        for value in tqdm(values, desc="Segmenting Regimes"):
            position = self.observations_seen
            self.observations_seen += 1
            raw = self.update_state(value, position)
            rows.append({**raw, "raw_latent_id": raw["latent_id"], "is_known": raw["latent_id"] != "null",
                         "known_state_count": len(self.prototypes)})
        result = pd.DataFrame(rows, index=embeddings.index, columns=_COLUMNS)
        result["latent_id"] = self.enforce_persistence(result["raw_latent_id"])
        result["transition_pending"] = self._transition_pending_rows
        result["is_known"] = result["latent_id"].ne("null")
        return result

    def state_dict(self) -> dict:
        return {"type": type(self).__name__, "schema_version": self.schema_version,
                "config": {"max_regimes": self.max_regimes, "match_threshold": self.match_threshold,
                           "candidate_threshold": self.candidate_threshold,
                           "min_candidate_observations": self.min_candidate_observations,
                           "min_candidate_episodes": self.min_candidate_episodes,
                           "candidate_episode_gap": self.candidate_episode_gap,
                           "max_candidates": self.max_candidates, "candidate_expiry": self.candidate_expiry,
                           "min_dwell": self.min_dwell, "learn_novel": self.learn_novel,
                           "switch_margin": self.switch_margin,
                           "centroid_update_rate": self.centroid_update_rate,
                           "min_state_scale": self.min_state_scale,
                           "state_expiry_observations": self.state_expiry_observations,
                           "model_ready_states": sorted(self.model_ready_states),
                           "backend": self.backend, "backend_params": dict(self.backend_params)},
                "prototypes": {k: v.tolist() for k, v in self.prototypes.items()},
                "scales": {k: v.tolist() for k, v in self.scales.items()}, "counts": self.counts,
                "scale_m2": {k: v.tolist() for k, v in self.scale_m2.items()},
                "scale_counts": self.scale_counts,
                "last_seen": self.last_seen,
                "candidates": [{**c, "center": c["center"].tolist(), "m2": c["m2"].tolist()} for c in self.candidates],
                "next_id": self._next_id, "observations_seen": self.observations_seen,
                "persistent_label": self._persistent_label, "smooth_pending": self._smooth_pending,
                "smooth_pending_count": self._smooth_pending_count,
                "current_state": self.current_state,
                "fitted_columns": self._fitted_columns,
                "radius_meta": self._radius_meta,
                "support_detail": self._support_detail,
                "fit_meta": self._fit_meta,
                "issued_ids": sorted(self._issued_ids)}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "HysteresisClusterRegimeSegmenter":
        if state.get("schema_version") != cls.schema_version:
            raise ValueError("segmenter schema version is incompatible")
        result = cls(**state["config"])
        result.prototypes = {k: np.asarray(v, dtype=np.float64) for k, v in state["prototypes"].items()}
        result.scales = {k: np.asarray(v, dtype=np.float64) for k, v in state["scales"].items()}
        result.scale_m2 = {k: np.asarray(v, dtype=np.float64) for k, v in state.get("scale_m2", {}).items()}
        result.scale_counts = {k: int(v) for k, v in state.get("scale_counts", {}).items()}
        result.last_seen = {k: int(v) for k, v in state.get("last_seen", {}).items()}
        for key in result.prototypes:
            result.scale_m2.setdefault(key, result.scales[key] ** 2)
            result.scale_counts.setdefault(key, 1)
            result.last_seen.setdefault(key, result.observations_seen)
        result.counts = {k: int(v) for k, v in state["counts"].items()}
        result.candidates = [{**c, "center": np.asarray(c["center"], dtype=np.float64),
                              "m2": np.asarray(c["m2"], dtype=np.float64)} for c in state["candidates"]]
        result._next_id, result.observations_seen = int(state["next_id"]), int(state["observations_seen"])
        result._persistent_label = state.get("persistent_label")
        result._smooth_pending = state.get("smooth_pending")
        result._smooth_pending_count = int(state.get("smooth_pending_count", 0))
        result.current_state = state.get("current_state")
        result._fitted_columns = state.get("fitted_columns")
        result._radius_meta = dict(state.get("radius_meta", {}))
        result._support_detail = dict(state.get("support_detail", {}))
        result._fit_meta = dict(state.get("fit_meta", {}))
        result._issued_ids = set(state.get("issued_ids", [])) | set(result.prototypes)
        return result


class StickyHMMRegimeSegmenter(HysteresisClusterRegimeSegmenter):
    """Causal diagonal-Gaussian HMM filter with a sticky transition prior.

    Prototypes provide Gaussian emissions; ``sticky_bias`` ``b`` sets the
    self-transition probability to ``sigmoid(b)`` and spreads the remaining
    ``1 - sigmoid(b)`` uniformly across the other states.  Note that ``b`` is
    therefore the *aggregate* log-odds of staying versus switching to *any*
    other state; the log-odds of staying versus switching to one *specific*
    alternative is ``b + log(K - 1)`` for ``K`` states.  Prototype creation
    still requires independent recurring episodes.  This HMM filter only
    composes with the online ``"custom"`` backend (it defines its own
    forward-filter assignment); combining it with an offline frozen backend is
    rejected.
    """

    def __init__(self, sticky_bias: float = 2.5, **kwargs):
        if not np.isfinite(sticky_bias) or sticky_bias < 0:
            raise ValueError("sticky_bias must be finite and >= 0")
        self.sticky_bias = float(sticky_bias)
        self._forward_probabilities: dict[str, float] = {}
        super().__init__(**kwargs)
        if self.backend != "custom":
            raise ValueError(
                "StickyHMMRegimeSegmenter only supports the 'custom' online "
                f"backend; got backend={self.backend!r}. Its forward-filter "
                "assignment is incompatible with offline frozen backends.")

    def _filter_step(self, embedding: np.ndarray, previous: dict[str, float]) -> dict[str, float]:
        ids = list(self.prototypes)
        if not ids:
            return {}
        value = np.asarray(embedding, dtype=np.float64)
        if not np.isfinite(value).all():
            return previous
        stay = 1.0 / (1.0 + np.exp(-self.sticky_bias))
        count = len(ids)
        emission = {}
        for state_id in ids:
            scale = np.maximum(self.scales[state_id], self.min_state_scale)
            standardized = (value - self.prototypes[state_id]) / scale
            emission[state_id] = float(-0.5 * np.sum(standardized ** 2) - np.sum(np.log(scale)))
        if previous:
            prior = {}
            for target in ids:
                probability = 0.0
                for source, source_probability in previous.items():
                    if source not in ids:
                        continue
                    transition = stay if source == target else (1.0 - stay) / max(count - 1, 1)
                    probability += source_probability * transition
                prior[target] = max(probability, 1e-300)
        else:
            prior = {state_id: 1.0 / count for state_id in ids}
        log_alpha = {state_id: np.log(prior[state_id]) + emission[state_id] for state_id in ids}
        maximum = max(log_alpha.values())
        alpha = {state_id: float(np.exp(value - maximum)) for state_id, value in log_alpha.items()}
        total = sum(alpha.values())
        return {state_id: probability / total for state_id, probability in alpha.items()}

    def assign_regime(self, embedding: np.ndarray) -> tuple[str | None, float]:
        value = np.asarray(embedding, dtype=np.float64)
        if not self.prototypes or not np.isfinite(value).all():
            return None, np.inf
        ids = list(self.prototypes)
        self._forward_probabilities = self._filter_step(value, self._forward_probabilities)
        state_id = max(self._forward_probabilities, key=self._forward_probabilities.get)
        return state_id, self._distance(value, self.prototypes[state_id], self.scales[state_id])

    def detect_regime_transition(self, embedding: np.ndarray) -> bool:
        value = np.asarray(embedding, dtype=np.float64)
        posterior = self._filter_step(value, self._forward_probabilities)
        if not posterior:
            return False
        state_id = max(posterior, key=posterior.get)
        distance = self._distance(value, self.prototypes[state_id], self.scales[state_id])
        return state_id != self.current_state and distance <= self.match_threshold

    def predict_proba(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        ids = list(self.prototypes)
        result = pd.DataFrame(0.0, index=embeddings.index, columns=ids)
        forward = dict(self._forward_probabilities)
        for row, value in enumerate(embeddings.to_numpy(dtype=np.float64)):
            forward = self._filter_step(value, forward)
            for state_id, probability in forward.items():
                result.iloc[row, result.columns.get_loc(state_id)] = probability
        return result

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["type"] = type(self).__name__
        state["config"]["sticky_bias"] = self.sticky_bias
        state["forward_probabilities"] = self._forward_probabilities
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "StickyHMMRegimeSegmenter":
        if state.get("schema_version") != cls.schema_version:
            raise ValueError("segmenter schema version is incompatible")
        result = cls(**state["config"])
        result.prototypes = {k: np.asarray(v, dtype=np.float64) for k, v in state["prototypes"].items()}
        result.scales = {k: np.asarray(v, dtype=np.float64) for k, v in state["scales"].items()}
        result.scale_m2 = {k: np.asarray(v, dtype=np.float64) for k, v in state.get("scale_m2", {}).items()}
        result.scale_counts = {k: int(v) for k, v in state.get("scale_counts", {}).items()}
        result.last_seen = {k: int(v) for k, v in state.get("last_seen", {}).items()}
        for key in result.prototypes:
            result.scale_m2.setdefault(key, result.scales[key] ** 2)
            result.scale_counts.setdefault(key, 1)
            result.last_seen.setdefault(key, result.observations_seen)
        result.counts = {k: int(v) for k, v in state["counts"].items()}
        result.candidates = [{**c, "center": np.asarray(c["center"], dtype=np.float64),
                              "m2": np.asarray(c["m2"], dtype=np.float64)} for c in state["candidates"]]
        result._next_id, result.observations_seen = int(state["next_id"]), int(state["observations_seen"])
        result._persistent_label = state.get("persistent_label")
        result._smooth_pending = state.get("smooth_pending")
        result._smooth_pending_count = int(state.get("smooth_pending_count", 0))
        result.current_state = state.get("current_state")
        result._forward_probabilities = {str(k): float(v) for k, v in state.get("forward_probabilities", {}).items()}
        return result


# ---------------------------------------------------------------------------
# Library backends: offline discovery on a training slice, frozen inference.
# ---------------------------------------------------------------------------


class _SklearnBackend:
    """Offline sklearn clustering (KMeans / HDBSCAN) for the hysteresis owner.

    Fit discovers prototypes once on a training embedding slice and mirrors
    the fitted state into the owning segmenter: raw-space centers in
    ``prototypes`` plus radius-encoded per-dimension ``scales`` chosen so the
    owner's RMS distance approximates ``d_standardized / radius`` (scale_j =
    unit_j * radius / sqrt(D)).  When the ``min_state_scale`` floor binds, that
    equality no longer holds exactly -- the effective acceptance metric is the
    post-floor RMS (a distance ``<= 1`` test); the original quantile radius and
    a ``floor_bound`` flag are stored per prototype in ``_radius_meta`` so the
    geometry is inspectable rather than assumed.

    Predict is frozen: nearest prototype by that distance, ``"null"`` when
    outside the radius (distance > 1), then the owner's causal min-dwell
    debounce.  This is a *prototype-based approximation* of the discovery
    method, NOT its native inference: for HDBSCAN in particular, a training row
    labeled noise (-1) can later be assigned to a known regime if it falls
    within a frozen prototype's radius, because inference uses centroid+radius
    geometry, not HDBSCAN's density reachability.  Support counts, debounce
    state, and prototypes live on the owner so ``state_dict`` / pickle round-
    trips the fitted backend.
    """

    METHODS = ("kmeans", "hdbscan", "sklearn")

    def __init__(self, owner: "HysteresisClusterRegimeSegmenter", method: str):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.owner = owner
        self.method = method
        params = dict(owner.backend_params or {})
        # Defaults follow the reference implementation; n_clusters defaults
        # to the owner's max_regimes so OOD settings map onto the backend.
        self.n_clusters = int(params.pop("n_clusters", owner.max_regimes))
        self.min_cluster_size = int(params.pop("min_cluster_size", 500))
        min_samples = params.pop("min_samples", None)
        self.min_samples = None if min_samples is None else int(min_samples)
        # HDBSCAN is time/memory heavy -> default to a 50k subsample; KMeans
        # can take a larger slice.
        self.max_fit_samples = int(params.pop("max_fit_samples", 50_000 if method == "hdbscan" else 300_000))
        self.radius_quantile = float(params.pop("radius_quantile", 0.99))
        self.radius_slack = float(params.pop("radius_slack", 1.5))
        self.standardize = bool(params.pop("standardize", True))
        self.seed = int(params.pop("seed", 7))
        # Max standardized-center distance for a refit cluster to inherit an
        # existing regime ID (one-to-one).  Only used when refitting over an
        # already-populated owner; defaults to the owner's match_threshold.
        identity_threshold = params.pop("identity_threshold", None)
        self.identity_threshold = (float(identity_threshold) if identity_threshold is not None
                                   else float(owner.match_threshold))
        if not np.isfinite(self.identity_threshold) or self.identity_threshold <= 0:
            raise ValueError("identity_threshold must be finite and positive")
        # method == "sklearn": any clusterer instance (AgglomerativeClustering, DBSCAN,
        # GaussianMixture, Birch, ...). Cloned at fit; labels come from ``labels_`` or
        # ``fit_predict``; inference is the same frozen centroid + radius rule.
        self.estimator = params.pop("estimator", None)
        if method == "sklearn" and self.estimator is None:
            raise ValueError("backend 'sklearn' requires backend_params={'estimator': <sklearn clusterer>}")
        if params:
            raise TypeError(f"unknown {method} backend params: {sorted(params)}")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if not 0.0 < self.radius_quantile <= 1.0:
            raise ValueError("radius_quantile must be in (0, 1]")
        if not np.isfinite(self.radius_slack) or self.radius_slack <= 0:
            raise ValueError("radius_slack must be finite and positive")
        if self.max_fit_samples < 1:
            raise ValueError("max_fit_samples must be >= 1")
        self.model_ = None  # fitted sklearn object, kept for inspection only

    @staticmethod
    def _pairwise(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """(n, K) Euclidean distances from rows of ``x`` to ``centers``."""
        d2 = (x ** 2).sum(1, keepdims=True) - 2 * x @ centers.T + (centers ** 2).sum(1)
        return np.sqrt(np.maximum(d2, 0.0))

    def fit(self, embeddings: pd.DataFrame, y=None) -> "HysteresisClusterRegimeSegmenter":
        """Cluster the training slice offline; mirror frozen state to the owner."""
        owner = self.owner
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        x = embeddings.to_numpy(dtype=np.float64)
        x = x[np.isfinite(x).all(axis=1)]
        required = (self.n_clusters if self.method == "kmeans"
                    else 50 if self.method == "sklearn"
                    else max(self.n_clusters, self.min_cluster_size)) * 2
        if len(x) < required:
            raise ValueError(f"not enough finite embeddings to fit the segmenter ({len(x)} < {required})")
        mean = x.mean(0)
        std = np.maximum(x.std(0), 1e-8)
        xs = (x - mean) / std if self.standardize else x
        rng = np.random.default_rng(self.seed)
        sub = xs if len(xs) <= self.max_fit_samples else xs[
            np.sort(rng.choice(len(xs), self.max_fit_samples, replace=False))]
        # Validate the ACTUAL fitting subset, not just the finite population:
        # a too-small max_fit_samples must fail here, not deep inside sklearn.
        if len(sub) < required:
            raise ValueError(
                f"fit subsample has {len(sub)} rows but {required} are required; "
                f"lower max_fit_samples ({self.max_fit_samples}) or provide more data")
        if self.method == "kmeans":
            from sklearn.cluster import KMeans
            self.model_ = KMeans(self.n_clusters, n_init=10, random_state=self.seed).fit(sub)
        elif self.method == "hdbscan":
            from sklearn.cluster import HDBSCAN
            self.model_ = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples).fit(sub)
        else:
            from sklearn.base import clone
            self.model_ = clone(self.estimator)
            if hasattr(self.model_, "labels_") or not hasattr(self.model_, "fit_predict"):
                self.model_.fit(sub)
                labels = self.model_.labels_ if hasattr(self.model_, "labels_") else self.model_.predict(sub)
            else:
                labels = self.model_.fit_predict(sub)
            self.model_.labels_ = np.asarray(labels)
        labels = np.asarray(self.model_.labels_)
        if labels.shape[0] != sub.shape[0]:
            raise ValueError(
                "sklearn estimator must yield one label per fitted sample; "
                f"got {labels.shape[0]} labels for {sub.shape[0]} samples")
        if labels.ndim != 1:
            raise ValueError("sklearn estimator must yield a 1-D label array")
        # The k>=0 / noise(-1) filtering below assumes integer cluster labels.
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(
                "sklearn estimator must yield integer cluster labels "
                f"(-1 = noise, >= 0 = cluster); got dtype {labels.dtype}")
        # Rank non-noise clusters by support, then HARD-cap at max_regimes so a
        # density method (HDBSCAN) cannot silently exceed the configured budget.
        cluster_sizes = {int(k): int((labels == k).sum()) for k in np.unique(labels) if k >= 0}
        kept = sorted(cluster_sizes, key=lambda k: -cluster_sizes[k])
        if not kept:
            raise ValueError("segmenter found no clusters; lower min_cluster_size or check the embeddings")
        discarded = kept[owner.max_regimes:]
        kept = kept[:owner.max_regimes]
        noise_count = int((labels < 0).sum())
        if discarded:
            print(f"[segmenter] max_regimes={owner.max_regimes} enforced: dropped "
                  f"{len(discarded)} cluster(s) with support "
                  f"{ {k: cluster_sizes[k] for k in discarded} }")
        centers = np.stack([sub[labels == k].mean(0) for k in kept])
        distances = self._pairwise(sub, centers)
        radii = np.array([max(np.quantile(distances[labels == k, i], self.radius_quantile) * self.radius_slack, 1e-6)
                          for i, k in enumerate(kept)])
        dim = xs.shape[1]
        unit = std if self.standardize else np.ones(dim)
        # Raw-space prototypes + radius-encoded scales.  When the min_state_scale
        # floor binds, the owner's RMS distance is no longer exactly
        # d_standardized / radius, so record that per prototype in radius_meta.
        raw_centers = [(mean + std * centers[i]) if self.standardize else centers[i].copy()
                       for i in range(len(kept))]
        pre_floor = np.outer(radii, unit) / np.sqrt(dim)
        scales = [np.maximum(pre_floor[i], owner.min_state_scale) for i in range(len(kept))]

        # Refit identity matching: preserve continuing regime IDs so a refit on a
        # newer window does not silently relabel stable regimes.
        ids = self._assign_ids(old_ids=list(owner.prototypes), raw_centers=raw_centers,
                               std=std if self.standardize else None)

        owner.prototypes = {sid: raw_centers[i] for i, sid in enumerate(ids)}
        owner.scales = {sid: scales[i] for i, sid in enumerate(ids)}
        owner.scale_counts = {sid: cluster_sizes[k] for sid, k in zip(ids, kept)}
        owner.scale_m2 = {sid: owner.scales[sid] ** 2 * max(owner.scale_counts[sid] - 1, 1) for sid in ids}
        # Split support: discovery membership (HDBSCAN/KMeans) vs. in-radius
        # acceptance vs. explicit noise.  counts stays the frozen inference
        # support (in-radius) so predict()/persistence are unchanged.
        owner._radius_meta = {sid: {"radius": float(radii[i]),
                                    "floor_bound": bool((pre_floor[i] < owner.min_state_scale).any())}
                              for i, sid in enumerate(ids)}
        in_radius_counts = self._in_radius_support(x, np.stack(raw_centers), np.stack(scales))
        owner.counts = {sid: int(in_radius_counts[i]) for i, sid in enumerate(ids)}
        # NOTE: cluster_support is counted over the FIT SUBSAMPLE (len(sub));
        # in_radius_support is counted over the FULL finite training set (len(x)).
        # They are different populations -- record both sizes so they are not
        # mistakenly compared.
        owner._support_detail = {sid: {"cluster_support": int(cluster_sizes[k]),
                                       "cluster_support_population": int(len(sub)),
                                       "in_radius_support": int(in_radius_counts[i]),
                                       "in_radius_population": int(len(x)),
                                       "noise_count": noise_count}
                                 for i, (sid, k) in enumerate(zip(ids, kept))}
        owner._fit_meta = {"method": self.method, "n_fit_rows": int(len(x)),
                            "n_fit_subsample": int(len(sub)), "n_clusters": len(ids),
                            "max_regimes": owner.max_regimes, "dropped_clusters": len(discarded),
                            "noise_count": noise_count, "standardize": self.standardize}
        owner._fitted_columns = list(embeddings.columns)
        owner.last_seen = {sid: 0 for sid in ids}
        owner.candidates = []
        owner._next_id = len(ids) + 1
        owner.observations_seen = 0
        owner.current_state = None
        owner._pending_state, owner._pending_count = None, 0
        owner._persistent_label, owner._smooth_pending, owner._smooth_pending_count = "null", None, 0
        owner._transition_pending_rows = []
        return owner

    def _in_radius_support(self, xs: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Count rows accepted by each frozen prototype's radius rule over ``xs``."""
        d = np.empty((xs.shape[0], centers.shape[0]), dtype=np.float64)
        for k in range(centers.shape[0]):
            diff = (xs - centers[k]) / scales[k]
            d[:, k] = np.sqrt(np.mean(diff ** 2, axis=1))
        nearest = d.argmin(1)
        dn = d[np.arange(xs.shape[0]), nearest]
        inside = dn <= 1.0
        return np.array([int(((nearest == k) & inside).sum()) for k in range(centers.shape[0])])

    def _fresh_id(self) -> str:
        """Next never-before-issued ``latent_xxx`` ID (monotonic, no reuse).

        Reserves the ID in ``_issued_ids`` immediately so repeated calls yield
        distinct IDs even before the caller bulk-registers them.
        """
        n = 1
        while f"latent_{n:03d}" in self.owner._issued_ids:
            n += 1
        new_id = f"latent_{n:03d}"
        self.owner._issued_ids.add(new_id)
        return new_id

    def _assign_ids(self, old_ids: list[str], raw_centers: list[np.ndarray],
                    std: np.ndarray | None) -> list[str]:
        """Map new clusters to persistent IDs, matching survivors to ``old_ids``.

        One-to-one minimum-cost match on standardized **RMS** center distance
        (``sqrt(mean(z^2))``, the same geometry as ``_distance`` so a single
        ``identity_threshold`` means the same thing in both places).  A new
        cluster within ``identity_threshold`` of an old prototype keeps that
        regime's ID; every unmatched new cluster receives a **fresh, never-
        reused** ID from the monotonic ``_issued_ids`` registry (a retired
        ``latent_xxx`` is never reassigned).  ``model_ready_states`` is
        reconciled so a retired prototype's readiness is dropped (downstream
        models must be reviewed), while surviving IDs retain readiness.
        """
        owner = self.owner
        if not old_ids:
            ids = [self._fresh_id() for _ in range(len(raw_centers))]
            owner._issued_ids.update(ids)
            return ids
        old_matrix = np.stack([owner.prototypes[o] for o in old_ids])
        new_matrix = np.stack(raw_centers)
        unit = std if std is not None else np.ones(new_matrix.shape[1])
        # RMS standardized distance: matches _distance/match_threshold semantics.
        cost = np.sqrt((((new_matrix[:, None, :] - old_matrix[None, :, :]) / unit) ** 2).mean(-1)).T
        pairs = _match_one_to_one(cost)
        id_map: dict[int, str] = {}
        matched_old: set[int] = set()
        for oi, ni in pairs:
            if cost[oi, ni] <= self.identity_threshold:
                id_map[ni] = old_ids[oi]
                matched_old.add(oi)
        # Unmatched new clusters get fresh, never-reused IDs (each _fresh_id
        # call reserves its ID, so they are guaranteed distinct).
        for ni in range(len(raw_centers)):
            if ni in id_map:
                continue
            id_map[ni] = self._fresh_id()
        owner._issued_ids.update(id_map.values())
        # Reconcile model_ready_states: keep an old ready ID only when it was
        # matched to a continuing prototype.
        surviving_ready = {old_ids[oi] for oi in matched_old if old_ids[oi] in owner.model_ready_states}
        dropped_ready = owner.model_ready_states - surviving_ready
        if dropped_ready:
            print(f"[segmenter] refit retired model-ready states {sorted(dropped_ready)} "
                  f"(prototype drift > identity_threshold); downstream models need review")
        owner.model_ready_states = surviving_ready
        return [id_map[ni] for ni in range(len(raw_centers))]

    def _assign(self, value: np.ndarray) -> tuple[str | None, float, bool, float]:
        """Shared frozen-geometry assignment.

        Returns ``(regime_id, normalized_distance, in_radius, confidence)`` where
        ``normalized_distance`` is the owner's RMS scaled distance (so ``<= 1``
        is exactly the radius test) and ``confidence`` is a distance-derived
        score ``exp(-d)`` -- a ranking/novelty signal, NOT a calibrated
        membership probability.
        """
        owner = self.owner
        ids = list(owner.prototypes)
        value = np.asarray(value, dtype=np.float64)
        dim = owner.prototypes[ids[0]].shape[0] if ids else None
        if not ids or value.ndim != 1 or value.shape[0] != dim or not np.isfinite(value).all():
            return None, np.inf, False, 0.0
        centers = np.stack([owner.prototypes[sid] for sid in ids])
        scales = np.stack([owner.scales[sid] for sid in ids])
        d = np.empty(len(ids), dtype=np.float64)
        for k in range(len(ids)):
            diff = (value - centers[k]) / scales[k]
            d[k] = np.sqrt(np.mean(diff ** 2))
        best = int(np.argmin(d))
        dn = float(d[best])
        in_radius = bool(np.isfinite(dn) and dn <= 1.0)
        regime_id = ids[best] if in_radius else None
        confidence = float(np.exp(-dn)) if np.isfinite(dn) else 0.0
        return regime_id, dn, in_radius, confidence

    def _validate_embeddings(self, embeddings: pd.DataFrame) -> np.ndarray:
        """Shared input validation for the frozen inference methods.

        Enforces DataFrame type, the exact fitted feature names and order, and
        the embedding dimension.  Returns the float64 value matrix.  Equal column
        counts are not enough -- a reordered frame must not silently assign
        against the wrong features.
        """
        owner = self.owner
        ids = list(owner.prototypes)
        if not ids:
            raise RuntimeError("fit must be called before predict")
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        if embeddings.shape[1] != owner.prototypes[ids[0]].shape[0]:
            raise ValueError("embedding dimensions differ from fitted prototypes")
        if owner._fitted_columns is not None and list(embeddings.columns) != list(owner._fitted_columns):
            raise ValueError(
                "embedding columns/order changed since fit; expected "
                f"{owner._fitted_columns}, got {list(embeddings.columns)}")
        return embeddings.to_numpy(dtype=np.float64)

    def assign_regime(self, embedding: np.ndarray) -> tuple[str | None, float]:
        regime_id, dn, _, _ = self._assign(embedding)
        return regime_id, dn

    def detect_regime_transition(self, embedding: np.ndarray) -> bool:
        regime_id, _, in_radius, _ = self._assign(embedding)
        return regime_id is not None and in_radius and regime_id != self.owner.current_state

    def predict_proba(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Distance-derived known-state scores plus an explicit ``null`` column.

        In-radius rows receive normalized similarity scores over known regimes
        (a ranking signal, not a calibrated probability); out-of-radius or
        non-finite rows place their mass in ``null`` so an OOD observation is
        never given a misleadingly confident known-regime assignment.
        """
        owner = self.owner
        ids = list(owner.prototypes)
        columns = list(ids) + ["null"]
        result = pd.DataFrame(0.0, index=embeddings.index, columns=columns)
        values = self._validate_embeddings(embeddings)
        for row, value in enumerate(values):
            regime_id, dn, in_radius, _ = self._assign(value)
            if not in_radius:
                result.iloc[row, result.columns.get_loc("null")] = 1.0
                continue
            # Recompute per-prototype distances for a proper known-state spread.
            centers = np.stack([owner.prototypes[sid] for sid in ids])
            scales = np.stack([owner.scales[sid] for sid in ids])
            d = np.array([np.sqrt(np.mean(((value - centers[k]) / scales[k]) ** 2)) for k in range(len(ids))])
            scores = np.exp(-0.5 * np.nan_to_num(d) ** 2)
            total = scores.sum()
            if total > 0:
                scores = scores / total
            for k, sid in enumerate(ids):
                result.iloc[row, result.columns.get_loc(sid)] = scores[k]
        return result

    def predict(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Frozen radius-rule assignment plus the owner's causal min-dwell debounce.

        Semantics (kept distinct so support is unambiguous):
          * ``raw_latent_id`` / ``status`` reflect the RAW assignment: a finite
            row inside a prototype's radius is ``"assigned"`` to that regime,
            a finite row outside every radius is ``"outside_radius"`` (raw
            ``"null"``), a non-finite row is ``"not_ready"``.
          * ``latent_id`` is the debounced label after min-dwell hysteresis, so a
            row can carry ``status="outside_radius"`` while ``is_known=True``
            (the previous regime is held) -- ``is_known`` reflects the debounced
            label, NOT that the current observation is in-distribution.
          * ``counts`` / ``state_support`` track IN-RADIUS assigned support only
            (raw accepted, seeded from fit); debounced ``emitted`` support is
            tracked separately in ``_support_detail`` and never mixed in.
        """
        owner = self.owner
        values = self._validate_embeddings(embeddings)
        ids = list(owner.prototypes)
        n = len(values)
        finite = np.isfinite(values).all(axis=1)
        rows = np.flatnonzero(finite)
        norm_dist = np.full(n, np.nan)
        nearest = np.full(n, -1, dtype=np.int64)
        if rows.size:
            centers = np.stack([owner.prototypes[sid] for sid in ids])
            scales = np.stack([owner.scales[sid] for sid in ids])
            d = np.empty((rows.size, len(ids)), dtype=np.float64)
            for k in range(len(ids)):
                diff = (values[rows] - centers[k]) / scales[k]
                d[:, k] = np.sqrt(np.mean(diff ** 2, axis=1))
            best = d.argmin(1)
            dn = d[np.arange(rows.size), best]
            inside = np.isfinite(dn) & (dn <= 1.0)
            norm_dist[rows] = dn
            nearest[rows[inside]] = best[inside]
        known = np.asarray(ids, dtype=object)
        raw = np.where(nearest >= 0, known[np.maximum(nearest, 0)], "null")
        status = np.where(~finite, "not_ready", np.where(nearest < 0, "outside_radius", "assigned"))
        latent = owner.enforce_persistence(raw)
        # IN-RADIUS assigned support (raw == sid); debounced emitted tracked apart.
        assigned_support = np.zeros(n, dtype=np.int64)
        emitted_support = np.zeros(n, dtype=np.int64)
        for sid in ids:
            assigned_mask = raw == sid
            if assigned_mask.any():
                assigned_support[assigned_mask] = owner.counts.get(sid, 0) + np.cumsum(assigned_mask)[assigned_mask]
                owner.counts[sid] = owner.counts.get(sid, 0) + int(assigned_mask.sum())
            emitted_mask = latent == sid
            if emitted_mask.any():
                prev_emitted = owner._support_detail.get(sid, {}).get("emitted_support", 0)
                emitted_support[emitted_mask] = prev_emitted + np.cumsum(emitted_mask)[emitted_mask]
                owner._support_detail.setdefault(sid, {})["emitted_support"] = (
                    prev_emitted + int(emitted_mask.sum()))
        owner.observations_seen += n
        if n:
            owner.current_state = None if raw[-1] == "null" else str(raw[-1])
        out = pd.DataFrame({
            "latent_id": latent, "raw_latent_id": raw,
            "similarity": np.where(np.isfinite(norm_dist), np.exp(-np.nan_to_num(norm_dist)), 0.0),
            "distance": norm_dist, "is_known": latent != "null",
            "candidate_count": 0, "candidate_episodes": 0, "known_state_count": len(ids),
            "state_support": assigned_support, "transition_pending": owner._transition_pending_rows,
            "status": status}, index=embeddings.index)
        return out[_COLUMNS]


class KMeansRegimeSegmenter(HysteresisClusterRegimeSegmenter):
    """Hysteresis segmenter with offline KMeans discovery (frozen prototypes).

    Accepts the owner's keywords plus ``backend_params`` for the library
    backend (``n_clusters``, ``radius_quantile``, ``radius_slack``,
    ``max_fit_samples``, ``standardize``, ``seed``).
    """

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = "kmeans"
        super().__init__(*args, **kwargs)
        self._backend()  # fail fast on bad backend_params


class HDBSCANRegimeSegmenter(HysteresisClusterRegimeSegmenter):
    """Hysteresis segmenter with offline HDBSCAN discovery (frozen prototypes).

    ``backend_params`` additionally accepts ``min_cluster_size`` and
    ``min_samples``.  During fit, HDBSCAN noise rows (-1) are excluded from
    prototypes and counted in ``_support_detail['noise_count']``.  Inference is
    the frozen centroid+radius rule (see ``_SklearnBackend``): it is an
    approximation of HDBSCAN, so a noise-labeled training point can still be
    assigned to a regime at predict time if it lies within a prototype radius.
    """

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = "hdbscan"
        super().__init__(*args, **kwargs)
        self._backend()  # fail fast on bad backend_params


class SklearnRegimeSegmenter(HysteresisClusterRegimeSegmenter):
    """Hysteresis segmenter with offline discovery by ANY scikit-learn clusterer.

    ``backend_params={"estimator": <clusterer>, "radius_quantile": ..., "radius_slack": ...}``.
    Noise labels (-1) map to the ``"null"`` regime.  Note: ``state_dict`` is not
    JSON-serializable with an estimator inside; use pickle for these.
    """

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = "sklearn"
        super().__init__(*args, **kwargs)
        self._backend()  # fail fast on bad backend_params


def make_segmenter(kind: str = "custom", **kwargs) -> BaseRegimeSegmenter:
    """THE switch: pick the segmenter by name.  Everything downstream is unchanged.

    ``"custom"`` (alias ``"hysteresis"``) keeps the online learning state machine;
    ``"sticky_hmm"`` is its HMM filter variant; ``"kmeans"`` and ``"hdbscan"``
    discover prototypes offline in ``fit`` and freeze them for inference;
    ``"sklearn"`` does the same with any clusterer you pass:

        make_segmenter("kmeans", max_regimes=8)
        make_segmenter("hdbscan", backend_params={"min_cluster_size": 800})
        make_segmenter("sklearn", backend_params={"estimator": AgglomerativeClustering(6)})
    """
    if kind in ("custom", "hysteresis"):
        return HysteresisClusterRegimeSegmenter(**kwargs)
    elif kind == "sticky_hmm":
        return StickyHMMRegimeSegmenter(**kwargs)
    elif kind == "kmeans":
        return KMeansRegimeSegmenter(**kwargs)
    elif kind == "hdbscan":
        return HDBSCANRegimeSegmenter(**kwargs)
    elif kind == "sklearn":
        return SklearnRegimeSegmenter(**kwargs)
    raise ValueError("kind must be one of 'custom', 'hysteresis', 'sticky_hmm', 'kmeans', 'hdbscan', 'sklearn'")