"""OOD detector orchestration: raw primitives -> temporal embedding -> states."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import pandas as pd

from .encoder import TemporalPrimitiveEncoder
from .regimes import OnlineRegimeRegistry


class OODRegimeDetector:
    """Stateful detector that can preserve learned latent prototypes between runs."""

    def __init__(
        self,
        sequence_window: int = 32,
        embedding_bins: int = 8,
        normalization_window: int = 500,
        normalization_min_periods: int = 50,
        clip_z: float = 8.0,
        slow_feature_window: int = 5000,
        slow_min_periods: int = 50,
        slow_level_weight: float = 1.0,
        primitive_columns: Sequence[str] | None = None,
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
        timestamp_col: str | None = None,
    ):
        if candidate_episode_gap < sequence_window:
            raise ValueError("candidate_episode_gap must be >= sequence_window to distinguish non-overlapping episodes")
        self.timestamp_col = timestamp_col
        self.encoder = TemporalPrimitiveEncoder(
            sequence_window=sequence_window,
            embedding_bins=embedding_bins,
            normalization_window=normalization_window,
            normalization_min_periods=normalization_min_periods,
            clip_z=clip_z,
            slow_feature_window=slow_feature_window,
            slow_min_periods=slow_min_periods,
            slow_level_weight=slow_level_weight,
            primitive_columns=primitive_columns,
        )
        self.registry = OnlineRegimeRegistry(
            max_states=max_states,
            match_threshold=match_threshold,
            novel_threshold=novel_threshold,
            min_novel_observations=min_novel_observations,
            min_novel_episodes=min_novel_episodes,
            candidate_episode_gap=candidate_episode_gap,
            candidate_expiry_observations=candidate_expiry_observations,
            max_novel_clusters=max_novel_clusters,
            mature_state_observations=mature_state_observations,
            state_expiry_observations=state_expiry_observations,
            min_state_scale=min_state_scale,
            centroid_update_rate=centroid_update_rate,
            learn_novel=learn_novel,
            model_ready_states=model_ready_states,
            require_model_for_trade=require_model_for_trade,
        )
        self._expected_primitive_names: tuple[str, ...] | None = None

    def set_model_ready_states(self, state_ids: Sequence[str]) -> None:
        """Mark states with fitted, deployable models as execution-ready."""
        self.registry.set_model_ready_states(state_ids)

    def detect(self, df: pd.DataFrame, process_from: int = 0) -> pd.DataFrame:
        """Encode raw context, updating registry only from ``process_from`` onward.

        For chunked use, include a raw normalization/sequence warmup prefix and
        set ``process_from`` to the first new row. Prefix rows are returned as
        non-tradeable context and do not update prototypes a second time.
        """
        if isinstance(process_from, bool) or not isinstance(process_from, Integral) or not 0 <= process_from <= len(df):
            raise ValueError("process_from must be an integer in [0, len(df)]")
        process_from = int(process_from)
        timestamp_name = self.timestamp_col or ("timestamp" if "timestamp" in df.columns else None)
        embeddings = self.encoder.encode(df, timestamp_col=timestamp_name)
        primitive_names = self.encoder.primitive_names
        if self._expected_primitive_names is not None and primitive_names != self._expected_primitive_names:
            raise ValueError(
                "raw OOD primitive schema changed from the saved registry; "
                "keep primitive columns and their order fixed"
            )
        self._expected_primitive_names = primitive_names
        if timestamp_name is not None and timestamp_name in df.columns:
            timestamps = pd.to_datetime(df[timestamp_name], utc=True, errors="coerce").array
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True, errors="coerce")).array
        else:
            timestamps = df.index.array
        context_index = embeddings.index[:process_from]
        result = self.registry.process(
            embeddings.iloc[process_from:],
            timestamps=timestamps[process_from:],
        )
        if process_from:
            context = pd.DataFrame({
                "timestamp": timestamps[:process_from],
                "latent_id": "null",
                "similarity": 0.0,
                "distance": float("nan"),
                "is_known": False,
                "state_mature": False,
                "model_ready": False,
                "trade_allowed": False,
                "status": "context",
                "nearest_latent_id": None,
                "candidate_count": 0,
                "candidate_episodes": 0,
                "known_state_count": len(self.registry.prototypes),
            }, index=context_index)
            result = pd.concat([context, result])
            result.index.name = embeddings.index.name
        result["timestamp"] = timestamps
        result["embedding_ready"] = embeddings.notna().all(axis=1).to_numpy()
        result["primitive_count"] = len(self.encoder.primitive_names)
        return result

    def state_dict(self) -> dict:
        """Serialize detector configuration and learned latent prototypes."""
        return {
            "timestamp_col": self.timestamp_col,
            "encoder_config": {
                "sequence_window": self.encoder.sequence_window,
                "embedding_bins": self.encoder.embedding_bins,
                "normalization_window": self.encoder.normalization_window,
                "normalization_min_periods": self.encoder.normalization_min_periods,
                "clip_z": self.encoder.clip_z,
                "slow_feature_window": self.encoder.slow_feature_window,
                "slow_min_periods": self.encoder.slow_min_periods,
                "slow_level_weight": self.encoder.slow_level_weight,
                "primitive_columns": list(self.encoder.primitive_columns) if self.encoder.primitive_columns is not None else None,
            },
            "primitive_names": list(self._expected_primitive_names or ()),
            "registry": self.registry.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "OODRegimeDetector":
        detector = cls(**state.get("encoder_config", {}), timestamp_col=state.get("timestamp_col"))
        detector.registry = OnlineRegimeRegistry.from_state_dict(state["registry"])
        saved_names = state.get("primitive_names")
        detector._expected_primitive_names = tuple(saved_names) if saved_names else None
        return detector


def detect_ood_regimes(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Convenience one-shot OOD detection; use the class to retain a registry."""
    return OODRegimeDetector(**kwargs).detect(df)
