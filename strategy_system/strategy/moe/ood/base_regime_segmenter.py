"""Swappable persistent segmenters and recurrence-aware prototype registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from tqdm.auto import tqdm

import numpy as np
import pandas as pd


_COLUMNS = ["latent_id", "raw_latent_id", "similarity", "distance", "is_known",
            "candidate_count", "candidate_episodes", "known_state_count", "state_support",
            "transition_pending", "status"]


class BaseRegimeSegmenter(ABC):
    """Interface for embedding-to-regime assignment.

    Implementations own prototype IDs and must treat them as persistent stored
    identities, not semantic/numeric labels. ``predict`` may update online
    recurrence and persistence state.
    """

    schema_version = 1

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
    """Online prototype clustering with persistence and multi-episode promotion."""

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
        if not np.isfinite(centroid_update_rate) or not 0.0 <= centroid_update_rate <= 1.0:
            raise ValueError("centroid_update_rate must be in [0, 1]")
        if not np.isfinite(min_state_scale) or min_state_scale <= 0:
            raise ValueError("min_state_scale must be finite and positive")
        self.centroid_update_rate = float(centroid_update_rate)
        self.min_state_scale = float(min_state_scale)
        self.state_expiry_observations = int(state_expiry_observations)
        self.learn_novel = bool(learn_novel)
        self.model_ready_states = set(map(str, model_ready_states or ()))
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
        value = np.asarray(embedding, dtype=np.float64)
        pairs = [(key, self._distance(value, center, self.scales.get(key))) for key, center in self.prototypes.items()]
        return min(pairs, key=lambda pair: pair[1]) if pairs else (None, np.inf)

    def detect_regime_transition(self, embedding: np.ndarray) -> bool:
        state_id, distance = self.assign_regime(np.asarray(embedding, dtype=np.float64))
        return state_id != self.current_state and distance <= self.match_threshold

    def predict_proba(self, embeddings: pd.DataFrame) -> pd.DataFrame:
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

    def fit(self, embeddings: pd.DataFrame, y=None) -> "HysteresisClusterRegimeSegmenter":
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        self.predict(embeddings)
        return self

    def predict(self, embeddings: pd.DataFrame) -> pd.DataFrame:
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
                           "model_ready_states": sorted(self.model_ready_states)},
                "prototypes": {k: v.tolist() for k, v in self.prototypes.items()},
                "scales": {k: v.tolist() for k, v in self.scales.items()}, "counts": self.counts,
                "scale_m2": {k: v.tolist() for k, v in self.scale_m2.items()},
                "scale_counts": self.scale_counts,
                "last_seen": self.last_seen,
                "candidates": [{**c, "center": c["center"].tolist(), "m2": c["m2"].tolist()} for c in self.candidates],
                "next_id": self._next_id, "observations_seen": self.observations_seen,
                "persistent_label": self._persistent_label, "smooth_pending": self._smooth_pending,
                "smooth_pending_count": self._smooth_pending_count,
                "current_state": self.current_state}

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
        return result


class StickyHMMRegimeSegmenter(HysteresisClusterRegimeSegmenter):
    """Causal diagonal-Gaussian HMM filter with a sticky transition prior.

    Prototypes provide Gaussian emissions; ``sticky_bias`` is the log prior
    advantage of a self-transition over switching to another known state.
    Prototype creation still requires independent recurring episodes.
    """

    def __init__(self, sticky_bias: float = 2.5, **kwargs):
        if not np.isfinite(sticky_bias) or sticky_bias < 0:
            raise ValueError("sticky_bias must be finite and >= 0")
        self.sticky_bias = float(sticky_bias)
        self._forward_probabilities: dict[str, float] = {}
        super().__init__(**kwargs)

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
