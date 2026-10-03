"""Online recurring latent-state registry with explicit null learning states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .distance import diagonal_mahalanobis, normalized_euclidean, similarity_from_distance


_RESULT_COLUMNS = [
    "timestamp", "latent_id", "similarity", "distance", "is_known",
    "state_mature", "model_ready", "trade_allowed", "status",
    "nearest_latent_id", "candidate_count", "candidate_episodes",
    "known_state_count",
]


class OnlineRegimeRegistry:
    """Match known embeddings and promote only recurrent, mature candidates.

    A candidate must accumulate enough observations across distinct episodes;
    consecutive overlapping windows count as one episode. Promotion itself
    remains null and non-tradeable. Known states are matched using a
    per-state diagonal Mahalanobis distance. Tradeability additionally requires
    a mature state and an explicitly registered fitted model by default.
    """

    def __init__(
        self,
        max_states: int = 10,
        match_threshold: float = 3.0,
        novel_threshold: float | None = None,
        min_novel_observations: int = 200,
        min_novel_episodes: int = 3,
        candidate_episode_gap: int = 32,
        candidate_expiry_observations: int = 5000,
        max_novel_clusters: int = 128,
        mature_state_observations: int = 200,
        state_expiry_observations: int = 100000,
        min_state_scale: float = 0.1,
        centroid_update_rate: float = 0.0,
        learn_novel: bool = True,
        model_ready_states: Sequence[str] | None = None,
        require_model_for_trade: bool = True,
    ):
        if not isinstance(max_states, int) or max_states < 1:
            raise ValueError("max_states must be an integer >= 1")
        if not np.isfinite(match_threshold) or match_threshold <= 0.0:
            raise ValueError("match_threshold must be finite and > 0")
        if novel_threshold is not None and (not np.isfinite(novel_threshold) or novel_threshold <= 0.0):
            raise ValueError("novel_threshold must be finite and > 0 or None")
        for name, value, minimum in (
            ("min_novel_observations", min_novel_observations, 2),
            ("min_novel_episodes", min_novel_episodes, 2),
            ("candidate_episode_gap", candidate_episode_gap, 1),
            ("candidate_expiry_observations", candidate_expiry_observations, 1),
            ("max_novel_clusters", max_novel_clusters, 1),
            ("mature_state_observations", mature_state_observations, 1),
            ("state_expiry_observations", state_expiry_observations, 1),
        ):
            if not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(min_state_scale) or min_state_scale <= 0.0:
            raise ValueError("min_state_scale must be finite and > 0")
        if not np.isfinite(centroid_update_rate) or not 0.0 <= centroid_update_rate <= 1.0:
            raise ValueError("centroid_update_rate must be in [0, 1]")

        self.max_states = int(max_states)
        self.match_threshold = float(match_threshold)
        self.novel_threshold = float(novel_threshold if novel_threshold is not None else 0.8)
        self.min_novel_observations = int(min_novel_observations)
        self.min_novel_episodes = int(min_novel_episodes)
        self.candidate_episode_gap = int(candidate_episode_gap)
        self.candidate_expiry_observations = int(candidate_expiry_observations)
        self.max_novel_clusters = int(max_novel_clusters)
        self.mature_state_observations = int(mature_state_observations)
        self.state_expiry_observations = int(state_expiry_observations)
        self.min_state_scale = float(min_state_scale)
        self.centroid_update_rate = float(centroid_update_rate)
        self.learn_novel = bool(learn_novel)
        self.require_model_for_trade = bool(require_model_for_trade)
        self.model_ready_states = set(str(state_id) for state_id in (model_ready_states or ()))
        self.prototypes: dict[str, np.ndarray] = {}
        self.scales: dict[str, np.ndarray] = {}
        self.radius_m2: dict[str, np.ndarray] = {}
        self.radius_counts: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.maturity_counts: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.novel_candidates: list[dict] = []
        self._next_id = 1
        self.observations_seen = 0

    def set_model_ready_states(self, state_ids: Sequence[str]) -> None:
        """Set IDs for which a fitted, deployable state model is available."""
        self.model_ready_states = {str(state_id) for state_id in state_ids}

    def _new_state_id(self) -> str:
        state_id = f"latent_{self._next_id:03d}"
        self._next_id += 1
        return state_id

    def _prune(self, position: int) -> None:
        self.novel_candidates = [
            item for item in self.novel_candidates
            if position - item["last_seen"] <= self.candidate_expiry_observations
        ]
        expired = [
            state_id for state_id, last_seen in self.last_seen.items()
            if position - last_seen > self.state_expiry_observations
        ]
        for state_id in expired:
            self.prototypes.pop(state_id, None)
            self.scales.pop(state_id, None)
            self.radius_m2.pop(state_id, None)
            self.radius_counts.pop(state_id, None)
            self.counts.pop(state_id, None)
            self.maturity_counts.pop(state_id, None)
            self.last_seen.pop(state_id, None)
            self.model_ready_states.discard(state_id)

    def _nearest_known(self, embedding: np.ndarray) -> tuple[str | None, float]:
        distances = [
            (state_id, diagonal_mahalanobis(embedding, center, self.scales[state_id], self.min_state_scale))
            for state_id, center in self.prototypes.items()
        ]
        finite = [(state_id, distance) for state_id, distance in distances if np.isfinite(distance)]
        return min(finite, key=lambda item: item[1]) if finite else (None, np.nan)

    def _promote_candidate(self, candidate: dict, position: int) -> tuple[str | None, str]:
        self._prune(position)
        # A candidate that has drifted inside an existing state's local radius
        # is redundant; keep the established identity instead of minting a twin.
        nearest_id, distance = self._nearest_known(candidate["center"])
        if nearest_id is not None and distance <= self.match_threshold:
            return None, "merged"
        if len(self.prototypes) >= self.max_states:
            return None, "capacity"
        state_id = self._new_state_id()
        count = int(candidate["count"])
        variance = candidate["m2"] / max(count - 1, 1)
        self.prototypes[state_id] = candidate["center"].copy()
        self.scales[state_id] = np.maximum(np.sqrt(np.maximum(variance, 0.0)), self.min_state_scale)
        self.radius_m2[state_id] = candidate["m2"].copy()
        self.radius_counts[state_id] = count
        self.counts[state_id] = count
        # Maturity counts only post-promotion observations, so the new state
        # cannot become mature on the same observations that taught its center.
        self.maturity_counts[state_id] = 0
        self.last_seen[state_id] = position
        return state_id, "promoted"

    def _learn_unknown(self, embedding: np.ndarray, position: int) -> tuple[int | None, int | None, str]:
        """Update recurrence buffers; consecutive windows count as one episode."""
        if not self.learn_novel:
            return None, None, "learning_disabled"
        self._prune(position)
        distances = [normalized_euclidean(embedding, item["center"]) for item in self.novel_candidates]
        finite = [(index, value) for index, value in enumerate(distances) if np.isfinite(value)]
        if finite:
            candidate_index, distance = min(finite, key=lambda item: item[1])
        else:
            candidate_index, distance = None, np.nan
        if candidate_index is None or distance > self.novel_threshold:
            if len(self.novel_candidates) >= self.max_novel_clusters:
                oldest = min(range(len(self.novel_candidates)), key=lambda idx: self.novel_candidates[idx]["last_seen"])
                self.novel_candidates.pop(oldest)
            candidate = {
                "center": embedding.copy(), "m2": np.zeros_like(embedding),
                "count": 1, "episodes": 1, "episode_count": 1,
                "last_seen": position,
            }
            self.novel_candidates.append(candidate)
            candidate_index = len(self.novel_candidates) - 1
        else:
            candidate = self.novel_candidates[candidate_index]
            intervening = position - candidate["last_seen"] - 1
            if intervening > self.candidate_episode_gap:
                candidate["episodes"] += 1
                candidate["episode_count"] = 0
            candidate["count"] += 1
            candidate["episode_count"] += 1
            delta = embedding - candidate["center"]
            candidate["center"] += delta / candidate["count"]
            candidate["m2"] += delta * (embedding - candidate["center"])
            candidate["last_seen"] = position

        count = int(candidate["count"])
        episodes = int(candidate["episodes"])
        if count >= self.min_novel_observations and episodes >= self.min_novel_episodes:
            _, status = self._promote_candidate(candidate, position)
            if status in ("promoted", "merged"):
                self.novel_candidates.pop(candidate_index)
            return count, episodes, status
        return count, episodes, "learning"

    def process(self, embeddings: pd.DataFrame, timestamps=None) -> pd.DataFrame:
        """Assign IDs sequentially, updating known states and novel candidates."""
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("embeddings must be a pandas DataFrame")
        if timestamps is not None and len(timestamps) != len(embeddings):
            raise ValueError("timestamps length must match embeddings")
        if self.prototypes and embeddings.shape[1] != next(iter(self.prototypes.values())).shape[0]:
            raise ValueError("embedding dimensions differ from loaded regime prototypes")
        if embeddings.empty:
            return pd.DataFrame(index=embeddings.index, columns=_RESULT_COLUMNS)

        records = []
        for position, embedding in enumerate(embeddings.to_numpy(dtype=np.float64)):
            global_position = self.observations_seen
            self.observations_seen += 1
            timestamp = timestamps[position] if timestamps is not None else embeddings.index[position]
            self._prune(global_position)
            if not np.isfinite(embedding).all():
                records.append({
                    "timestamp": timestamp, "latent_id": "null", "similarity": 0.0,
                    "distance": np.nan, "is_known": False, "state_mature": False,
                    "model_ready": False, "trade_allowed": False, "status": "warmup",
                    "nearest_latent_id": None, "candidate_count": 0,
                    "candidate_episodes": 0, "known_state_count": len(self.prototypes),
                })
                continue

            nearest_id, distance = self._nearest_known(embedding)
            similarity = similarity_from_distance(distance, self.match_threshold)
            if nearest_id is not None and distance <= self.match_threshold:
                self.counts[nearest_id] += 1
                self.maturity_counts[nearest_id] += 1
                self.last_seen[nearest_id] = global_position
                if self.centroid_update_rate > 0.0:
                    self.prototypes[nearest_id] += self.centroid_update_rate * (embedding - self.prototypes[nearest_id])
                # Adapt the local diagonal radius more slowly than the centroid;
                # frozen centers remain stable identity anchors by default.
                residual = embedding - self.prototypes[nearest_id]
                self.radius_m2[nearest_id] += residual ** 2
                self.radius_counts[nearest_id] += 1
                variance = self.radius_m2[nearest_id] / max(self.radius_counts[nearest_id] - 1, 1)
                self.scales[nearest_id] = np.maximum(np.sqrt(np.maximum(variance, 0.0)), self.min_state_scale)
                state_mature = self.maturity_counts[nearest_id] >= self.mature_state_observations
                model_ready = nearest_id in self.model_ready_states
                trade_allowed = state_mature and (model_ready or not self.require_model_for_trade)
                records.append({
                    "timestamp": timestamp, "latent_id": nearest_id, "similarity": similarity,
                    "distance": distance, "is_known": True, "state_mature": state_mature,
                    "model_ready": model_ready, "trade_allowed": trade_allowed,
                    "status": "known" if trade_allowed else "maturing" if not state_mature else "model_unavailable",
                    "nearest_latent_id": nearest_id, "candidate_count": 0,
                    "candidate_episodes": 0, "known_state_count": len(self.prototypes),
                })
            else:
                candidate_count, episodes, learn_status = self._learn_unknown(embedding, global_position)
                status = {
                    "promoted": "learned", "merged": "merged", "capacity": "capacity",
                    "learning_disabled": "learning_disabled",
                }.get(learn_status, "learning")
                records.append({
                    "timestamp": timestamp, "latent_id": "null", "similarity": similarity,
                    "distance": distance, "is_known": False, "state_mature": False,
                    "model_ready": False, "trade_allowed": False, "status": status,
                    "nearest_latent_id": nearest_id, "candidate_count": candidate_count or 0,
                    "candidate_episodes": episodes or 0, "known_state_count": len(self.prototypes),
                })
        result = pd.DataFrame(records, index=embeddings.index, columns=_RESULT_COLUMNS)
        result.index.name = embeddings.index.name
        return result

    def state_dict(self) -> dict:
        """Return a JSON/YAML-serializable copy of known and learning states."""
        return {
            "config": {
                "max_states": self.max_states,
                "match_threshold": self.match_threshold,
                "novel_threshold": self.novel_threshold,
                "min_novel_observations": self.min_novel_observations,
                "min_novel_episodes": self.min_novel_episodes,
                "candidate_episode_gap": self.candidate_episode_gap,
                "candidate_expiry_observations": self.candidate_expiry_observations,
                "max_novel_clusters": self.max_novel_clusters,
                "mature_state_observations": self.mature_state_observations,
                "state_expiry_observations": self.state_expiry_observations,
                "min_state_scale": self.min_state_scale,
                "centroid_update_rate": self.centroid_update_rate,
                "learn_novel": self.learn_novel,
                "model_ready_states": sorted(self.model_ready_states),
                "require_model_for_trade": self.require_model_for_trade,
            },
            "next_id": self._next_id,
            "observations_seen": self.observations_seen,
            "prototypes": {key: value.tolist() for key, value in self.prototypes.items()},
            "scales": {key: value.tolist() for key, value in self.scales.items()},
            "radius_m2": {key: value.tolist() for key, value in self.radius_m2.items()},
            "radius_counts": dict(self.radius_counts),
            "counts": dict(self.counts),
            "maturity_counts": dict(self.maturity_counts),
            "last_seen": dict(self.last_seen),
            "novel_candidates": [
                {
                    "center": item["center"].tolist(), "m2": item["m2"].tolist(),
                    "count": int(item["count"]), "episodes": int(item["episodes"]),
                    "episode_count": int(item["episode_count"]), "last_seen": int(item["last_seen"]),
                }
                for item in self.novel_candidates
            ],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "OnlineRegimeRegistry":
        config = dict(state.get("config", {}))
        registry = cls(**config)
        registry._next_id = int(state.get("next_id", 1))
        registry.observations_seen = int(state.get("observations_seen", 0))
        registry.prototypes = {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in state.get("prototypes", {}).items()
        }
        registry.scales = {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in state.get("scales", {}).items()
        }
        for state_id, prototype in registry.prototypes.items():
            registry.scales.setdefault(state_id, np.ones_like(prototype))
        registry.radius_m2 = {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in state.get("radius_m2", {}).items()
        }
        registry.radius_counts = {
            str(key): int(value) for key, value in state.get("radius_counts", {}).items()
        }
        for state_id in registry.prototypes:
            registry.radius_m2.setdefault(state_id, registry.scales[state_id] ** 2)
            registry.radius_counts.setdefault(state_id, 1)
        registry.counts = {str(key): int(value) for key, value in state.get("counts", {}).items()}
        registry.maturity_counts = {
            str(key): int(value) for key, value in state.get("maturity_counts", {}).items()
        }
        registry.last_seen = {str(key): int(value) for key, value in state.get("last_seen", {}).items()}
        for state_id in registry.prototypes:
            registry.counts.setdefault(state_id, 0)
            registry.maturity_counts.setdefault(state_id, 0)
            registry.last_seen.setdefault(state_id, registry.observations_seen)
        registry.novel_candidates = [
            {
                "center": np.asarray(item["center"], dtype=np.float64),
                "m2": np.asarray(item.get("m2", np.zeros_like(item["center"])), dtype=np.float64),
                "count": int(item["count"]), "episodes": int(item.get("episodes", 1)),
                "episode_count": int(item.get("episode_count", item["count"])),
                "last_seen": int(item.get("last_seen", registry.observations_seen)),
            }
            for item in state.get("novel_candidates", [])
        ]
        dimensions = {value.shape for value in registry.prototypes.values()} | {value.shape for value in registry.scales.values()}
        if len(dimensions) > 1:
            raise ValueError("saved prototypes or scales have inconsistent embedding dimensions")
        if set(registry.prototypes) != set(registry.scales):
            raise ValueError("saved prototypes and state scales must have identical IDs")
        return registry
