"""Top-level OOD pipeline: causal encoder -> persistent segmenter -> diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .base_regime_encoder import (
    BaseRegimeEncoder, TemporalWindowEncoder, TorchRegimeEncoder, SklearnRegimeEncoder,
    prepare_regime_input, build_temporal_windows, complete_window_mask,
    gather_trailing_windows, causal_zscore_matrix, make_encoder,
)
from .base_regime_segmenter import (
    BaseRegimeSegmenter, HysteresisClusterRegimeSegmenter, StickyHMMRegimeSegmenter,
    KMeansRegimeSegmenter, HDBSCANRegimeSegmenter, make_segmenter,
)

from .regime_validator import RegimeValidator

def build_primitive_observations(df: pd.DataFrame, primitive_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Build raw primitives for OOD and the existing GP source adapter.

    Auto mode derives price return, spread, queue/depth imbalance, signed trade
    flow, activity, and numeric liquidation measurements where available.
    """
    if primitive_columns is not None:
        return prepare_regime_input(df, primitive_columns)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("raw market frame must be a non-empty DataFrame")
    out: dict[str, pd.Series] = {}
    cols = df.columns
    def first(names):
        return next((name for name in names if name in cols), None)
    bid, ask = first(("best_bid_price", "best_bid_price_close", "bid_price")), first(("best_ask_price", "best_ask_price_close", "ask_price"))
    price = first(("mid_price", "price_close", "price", "close"))
    if bid and ask:
        b, a = pd.to_numeric(df[bid], errors="coerce"), pd.to_numeric(df[ask], errors="coerce")
        mid = (a + b) / 2
        out["mid_return"] = np.log(mid.where(mid > 0)).diff()
        out["spread_bps"] = (a - b) / mid.where(mid > 0) * 10_000
    elif price:
        p = pd.to_numeric(df[price], errors="coerce")
        out["mid_return"] = np.log(p.where(p > 0)).diff()
    bq, aq = first(("best_bid_qty", "bid_qty_1", "bid_size")), first(("best_ask_qty", "ask_qty_1", "ask_size"))
    if bq and aq:
        b, a = pd.to_numeric(df[bq], errors="coerce").clip(lower=0), pd.to_numeric(df[aq], errors="coerce").clip(lower=0)
        total = b + a
        out["queue_imbalance"] = (b - a) / total.replace(0, np.nan)
        out["top_depth_log"] = np.log1p(total)
    levels = sorted({c[len("bid_qty_"):] for c in cols if str(c).startswith("bid_qty_") and str(c)[len("bid_qty_"):].isdigit()} |
                    {c[len("ask_qty_"):] for c in cols if str(c).startswith("ask_qty_") and str(c)[len("ask_qty_"):].isdigit()}, key=int)
    for level in levels:
        bn, an = f"bid_qty_{level}", f"ask_qty_{level}"
        if bn in cols and an in cols and not (level == "1" and bq and aq):
            b, a = pd.to_numeric(df[bn], errors="coerce").clip(lower=0), pd.to_numeric(df[an], errors="coerce").clip(lower=0)
            total = b + a
            out[f"depth_imbalance_{level}"] = (b - a) / total.replace(0, np.nan)
            out[f"depth_log_{level}"] = np.log1p(total)
    qty, maker = first(("quantity", "trade_qty", "volume")), first(("is_buyer_maker", "buyer_is_maker"))
    if qty:
        volume = pd.to_numeric(df[qty], errors="coerce").clip(lower=0)
        out["trade_activity_log"] = np.log1p(volume)
        if maker:
            signed = volume.where(~df[maker].fillna(False).astype(bool), -volume)
            out["signed_trade_flow"] = signed
            out["trade_direction"] = np.sign(signed)
    for column in cols:
        if "liquidat" in str(column).lower() and pd.api.types.is_numeric_dtype(df[column]):
            out[f"raw_{column}"] = pd.to_numeric(df[column], errors="coerce")
    if not out:
        primitives = prepare_regime_input(df)
    else:
        primitives = pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)
    # Causal LOCF across the interleaved event stream. The matched L2 table
    # mixes LOB / TRADE / BOOKDEPTH rows, so each primitive is only populated on
    # its own event type and ~71% of rows carry at least one NaN. The temporal
    # encoder marks a row ready only when an ALL-finite trailing window exists,
    # so un-filled gaps made ~97% of embeddings NaN (embedding_ready=False ->
    # trade_allowed=0 everywhere). Carrying the last known value forward is
    # causal (never reads a future row) and mirrors the ffill the loader and
    # features.py already apply. Leading rows before a primitive is first seen
    # stay NaN and remain honestly not-ready.
    return primitives.ffill()


class OODSystem:
    """Orchestrate independent encoder, segmenter, and validator interfaces.

    Regime discovery only sees raw/primitive input and embeddings. Forward
    outcomes are accepted by validation methods only and never enter ``fit``.
    """

    schema_version = 1

    def __init__(
        self,
        encoder: BaseRegimeEncoder | None = None,
        segmenter: BaseRegimeSegmenter | None = None,
        validator: RegimeValidator | None = None,
        sequence_window: int = 32,
        embedding_bins: int = 8,
        normalization_window: int = 500,
        normalization_min_periods: int = 50,
        clip_z: float = 8.0,
        slow_feature_window: int | None = None,
        slow_min_periods: int | None = None,
        slow_level_weight: float | None = None,
        primitive_columns: Sequence[str] | None = None,
        max_states: int = 10,
        match_threshold: float = 1.5,
        novel_threshold: float = 0.25,
        min_novel_observations: int = 200,
        min_novel_episodes: int = 3,
        candidate_episode_gap: int = 32,
        min_dwell: int = 5,
        mature_state_observations: int = 200,
        model_ready_states: Sequence[str] | None = None,
        require_model_for_trade: bool = True,
        timestamp_col: str | None = None,
        segmenter_type: str = "hysteresis",
        encoder_type: str = "temporal_window",
        encoder_params: dict | None = None,
        segmenter_params: dict | None = None,
        metrics: dict | None = None,
        sticky_bias: float = 2.5,
        candidate_expiry_observations: int | None = None,
        max_novel_clusters: int | None = None,
        state_expiry_observations: int | None = None,
        min_state_scale: float | None = None,
        centroid_update_rate: float | None = None,
        learn_novel: bool | None = None,
        fit_rows: int | float | None = None,
        **unsupported,
    ):
        if unsupported:
            keys = ", ".join(sorted(unsupported))
            raise TypeError(f"unsupported OOD settings: {keys}")
        if encoder is not None:
            self.encoder = encoder
        else:
            # Legacy spellings ("tcn_backend", ...) still work.
            name = str(encoder_type).replace("_backend", "")
            params = dict(encoder_params or {})
            if name == "temporal_window":
                params.update(bins=embedding_bins, normalization_window=normalization_window,
                              normalization_min_periods=normalization_min_periods, clip_z=clip_z,
                              slow_feature_window=5000 if slow_feature_window is None else slow_feature_window,
                              slow_min_periods=50 if slow_min_periods is None else slow_min_periods,
                              slow_level_weight=1.0 if slow_level_weight is None else slow_level_weight)
            elif any(value is not None for value in (slow_feature_window, slow_min_periods, slow_level_weight)):
                raise ValueError("slow-level channels are currently supported by encoder_type='temporal_window' only")
            self.encoder = make_encoder(name, window=sequence_window, feature_columns=primitive_columns,
                                        timestamp_col=timestamp_col, **params)
        if segmenter is None:
            kwargs = dict(max_regimes=max_states, match_threshold=match_threshold,
                          candidate_threshold=novel_threshold,
                          min_candidate_observations=min_novel_observations,
                          min_candidate_episodes=min_novel_episodes,
                          candidate_episode_gap=candidate_episode_gap,
                          max_candidates=128 if max_novel_clusters is None else max_novel_clusters,
                          candidate_expiry=5000 if candidate_expiry_observations is None else candidate_expiry_observations,
                          min_dwell=min_dwell,
                          centroid_update_rate=0.0 if centroid_update_rate is None else centroid_update_rate,
                          min_state_scale=0.1 if min_state_scale is None else min_state_scale,
                          state_expiry_observations=100000 if state_expiry_observations is None else state_expiry_observations,
                          learn_novel=True if learn_novel is None else learn_novel,
                          model_ready_states=model_ready_states,
                          backend_params=dict(segmenter_params or {}))
            if segmenter_type == "sticky_hmm":
                kwargs["sticky_bias"] = sticky_bias
            segmenter = make_segmenter(segmenter_type, **kwargs)
        self.segmenter = segmenter
        # Leak guard for ``detect``: fit encoder/segmenter on only the first ``fit_rows``
        # rows (float in (0,1) = fraction, int = row count); None = whole frame (old behaviour).
        if fit_rows is not None and not (0 < fit_rows < 1 or (float(fit_rows).is_integer() and fit_rows >= 2)):
            raise ValueError("fit_rows must be a fraction in (0, 1) or an integer row count >= 2")
        self.fit_rows = fit_rows
        encoder_window = int(self.encoder.window_size or sequence_window)
        if candidate_episode_gap < encoder_window:
            raise ValueError("candidate_episode_gap must be >= encoder window to count non-overlapping recurrence episodes")
        self.validator = validator or RegimeValidator(metrics=metrics)
        self.mature_state_observations = int(mature_state_observations)
        self.require_model_for_trade = bool(require_model_for_trade)
        self.timestamp_col = timestamp_col
        self._fitted = False
        # Histories are stored as an optional materialized base plus a list of
        # per-call chunks. Appends push a chunk (O(chunk)); the full frame is
        # concatenated lazily only when read. This avoids O(T^2) pd.concat when
        # transform()/update_regime() are called once per row over ~1.1M rows.
        # Lazily materialized history storage.
        self._history_base: pd.DataFrame | None = None
        self._embedding_base: pd.DataFrame | None = None
        self._feature_base: pd.DataFrame | None = None

        self._history_chunks: list[pd.DataFrame] = []
        self._embedding_chunks: list[pd.DataFrame] = []
        self._feature_chunks: list[pd.DataFrame] = []

        # Buffer single-row update_regime() calls. Flush in bounded batches.
        self._history_chunk_size = 4096
        self._max_history_chunks = 64

        self._history_pending_rows: list[dict] = []
        self._history_pending_index: list[object] = []
        self._history_pending_index_template = None

        self._embedding_pending_values: list[np.ndarray] = []
        self._embedding_pending_index: list[object] = []
        self._embedding_pending_index_template = None
        self._embedding_pending_columns = None

    @staticmethod
    def _build_pending_index(labels, template) -> pd.Index:
        """Reconstruct a chunk index from buffered labels."""
        if isinstance(template, pd.MultiIndex):
            return pd.MultiIndex.from_tuples(labels, names=template.names)
        if isinstance(template, pd.DatetimeIndex):
            return pd.DatetimeIndex(
                labels, name=template.name, tz=template.tz
            )
        if isinstance(template, pd.TimedeltaIndex):
            return pd.TimedeltaIndex(labels, name=template.name)
        if isinstance(template, pd.PeriodIndex):
            return pd.PeriodIndex(
                labels, name=template.name, freq=template.freq
            )
        return pd.Index(labels, name=template.name)

    def _append_history_chunk(self, chunks_name: str, frame: pd.DataFrame) -> None:
        """Append a detached chunk and periodically compact recent chunks."""
        if frame.empty:
            return

        chunks = getattr(self, chunks_name)
        chunks.append(frame.copy())

        # Bound the number of DataFrame objects without repeatedly copying
        # the complete historical dataset.
        if len(chunks) >= self._max_history_chunks:
            chunks[:] = [pd.concat(chunks, axis=0, copy=False)]

    def _flush_pending_history(self) -> None:
        """Convert pending single-row records into bounded DataFrame chunks."""
        if self._history_pending_rows:
            index = self._build_pending_index(
                self._history_pending_index,
                self._history_pending_index_template,
            )
            frame = pd.DataFrame(self._history_pending_rows, index=index)
            self._append_history_chunk("_history_chunks", frame)

            self._history_pending_rows.clear()
            self._history_pending_index.clear()
            self._history_pending_index_template = None

        if self._embedding_pending_values:
            index = self._build_pending_index(
                self._embedding_pending_index,
                self._embedding_pending_index_template,
            )
            values = np.stack(self._embedding_pending_values, axis=0)
            frame = pd.DataFrame(
                values,
                index=index,
                columns=self._embedding_pending_columns,
            )
            self._append_history_chunk("_embedding_chunks", frame)

            self._embedding_pending_values.clear()
            self._embedding_pending_index.clear()
            self._embedding_pending_index_template = None
            self._embedding_pending_columns = None

    def _materialize_history(
        self, base_name: str, chunks_name: str
    ) -> pd.DataFrame:
        """Flush pending records and materialize the requested history."""
        self._flush_pending_history()

        base = getattr(self, base_name)
        chunks = getattr(self, chunks_name)

        if chunks:
            parts = ([base] if base is not None else []) + chunks
            combined = pd.concat(parts, axis=0, copy=False)
            setattr(self, base_name, combined)
            chunks.clear()
            base = combined

        return base if base is not None else pd.DataFrame()

    @property
    def _history(self) -> pd.DataFrame:
        return self._materialize_history(
            "_history_base", "_history_chunks"
        )

    @property
    def _embedding_history(self) -> pd.DataFrame:
        return self._materialize_history(
            "_embedding_base", "_embedding_chunks"
        )

    @property
    def _feature_history(self) -> pd.DataFrame:
        return self._materialize_history(
            "_feature_base", "_feature_chunks"
        )

    @property
    def embeddings_(self) -> pd.DataFrame:
        """Public, lazily materialized embedding history."""
        return self._embedding_history

    def fit(self, df: pd.DataFrame, y=None) -> "OODSystem":
        """Fit encoder schema/parameters. ``y`` is intentionally ignored."""
        if y is not None:
            raise ValueError("OOD discovery is unsupervised; forward outcomes are not accepted")
        self._validate_timestamps(df)
        timestamp_name = self.timestamp_col or getattr(self.encoder, "timestamp_col", None) or ("timestamp" if "timestamp" in df else None)
        source = df.drop(columns=[timestamp_name]) if timestamp_name is not None and timestamp_name in df else df
        primitives = build_primitive_observations(source, self.encoder.feature_columns)
        if timestamp_name is not None and timestamp_name in df:
            primitives[timestamp_name] = df[timestamp_name].to_numpy()
        self.encoder.fit_encoder(primitives)
        if getattr(self.segmenter, "backend", "custom") != "custom":
            # Library segmenters discover prototypes offline during fit; the
            # custom segmenter keeps learning lazily inside predict.
            self.segmenter.fit(self.encoder.transform(primitives))
        self._fitted = True
        return self

    def fit_train_apply(self, train_df: pd.DataFrame, new_df: pd.DataFrame,
                        context_rows: int = 600) -> pd.DataFrame:
        """Leak-free fit/apply: fit encoder AND segmenter on the train slice.

        ``OODSystem.fit`` fits only the encoder, so sklearn-style segmenters
        fitted offline would otherwise never be fitted. Fit both on
        ``train_df`` only, then assign ``new_df`` with the last
        ``context_rows`` of ``train_df`` warming the causal normalization
        (context rows are not assigned).
        """
        self.fit(train_df)
        if getattr(self.segmenter, "backend", "custom") == "custom":
            # Warm the online learner on the train slice; library backends
            # were already fitted offline inside fit().
            self.segmenter.fit(self.encoder.transform(self._primitives(train_df)))
        context = train_df.iloc[-int(context_rows):] if int(context_rows) > 0 else train_df.iloc[0:0]
        return self.transform(pd.concat([context, new_df]), process_from=len(context)).iloc[len(context):]

    def _primitives(self, df: pd.DataFrame) -> pd.DataFrame:
        timestamp_name = self.timestamp_col or getattr(self.encoder, "timestamp_col", None) or (
            "timestamp" if "timestamp" in df else None)
        source = df.drop(columns=[timestamp_name]) if timestamp_name is not None and timestamp_name in df else df
        primitives = build_primitive_observations(source, self.encoder.feature_columns)
        if timestamp_name is not None and timestamp_name in df:
            primitives[timestamp_name] = df[timestamp_name].to_numpy()
        return primitives

    def transform(self, df: pd.DataFrame, process_from: int = 0) -> pd.DataFrame:
        """Encode and segment a causal chunk, optionally skipping context rows."""
        if not self._fitted:
            raise RuntimeError("fit must be called before transform")
        if isinstance(process_from, bool) or not isinstance(process_from, (int, np.integer)) or not 0 <= process_from <= len(df):
            raise ValueError("process_from must be an integer in [0, len(df)]")
        self._validate_timestamps(df)
        timestamp_name = self.timestamp_col or getattr(self.encoder, "timestamp_col", None) or ("timestamp" if "timestamp" in df else None)
        source = df.drop(columns=[timestamp_name]) if timestamp_name is not None and timestamp_name in df else df
        primitives = build_primitive_observations(source, self.encoder.feature_columns)
        if timestamp_name is not None and timestamp_name in df:
            primitives[timestamp_name] = df[timestamp_name].to_numpy()
        embeddings = self.encoder.transform(primitives)
        assignments = self.segmenter.predict(embeddings.iloc[int(process_from):])
        timestamps = self._timestamps(df)
        if process_from:
            context = pd.DataFrame({"latent_id": "null", "raw_latent_id": "null", "similarity": 0.0,
                                    "distance": np.nan, "is_known": False, "candidate_count": 0,
                                    "candidate_episodes": 0, "known_state_count": len(self.segmenter.get_regime_prototypes()),
                                    "state_support": 0, "transition_pending": False,
                                    "status": "context"}, index=embeddings.index[:process_from])
            assignments = pd.concat([context, assignments])
        assignments["timestamp"] = timestamps
        assignments["embedding_ready"] = embeddings.notna().all(axis=1).to_numpy()
        assignments["state_mature"] = assignments["state_support"].fillna(0).ge(self.mature_state_observations) & assignments["latent_id"].ne("null")
        ready_states = self.segmenter.get_model_ready_states()
        assignments["model_ready"] = assignments["latent_id"].map(
            lambda state: str(state) in ready_states if str(state) != "null" else False)
        stable_assignment = (assignments["raw_latent_id"].astype(str) == assignments["latent_id"].astype(str)) & ~assignments["transition_pending"].fillna(False)
        assignments["trade_allowed"] = assignments["state_mature"] & stable_assignment & (
            assignments["model_ready"] | (not self.require_model_for_trade))
        assignments["timestamp"] = timestamps
        # Append this chunk only; do NOT concat the whole accumulated history.
        # .copy() detaches the stored chunk from buffers the caller may mutate.
        self._append_history_chunk(
            "_history_chunks", assignments.iloc[process_from:]
        )

        self._append_history_chunk(
            "_embedding_chunks", embeddings.iloc[process_from:]
        )

        selected = primitives.iloc[process_from:].drop(
            columns=(
                [timestamp_name]
                if timestamp_name is not None
                and timestamp_name in primitives
                else []
            ),
            errors="ignore",
        )
        self._append_history_chunk("_feature_chunks", selected)

        return assignments

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y=y).transform(df)

    def identify_regimes(self, df: pd.DataFrame, process_from: int = 0) -> pd.DataFrame:
        return self.transform(df, process_from=process_from)

    def detect(self, df: pd.DataFrame, process_from: int = 0) -> pd.DataFrame:
        """Backward-compatible strategy entry point."""
        if not self._fitted:
            n_fit = len(df) if self.fit_rows is None else (
                int(len(df) * self.fit_rows) if self.fit_rows < 1 else min(int(self.fit_rows), len(df)))
            self.fit(df.iloc[:n_fit])
        return self.transform(df, process_from=process_from)

    def set_model_ready_states(self, state_ids: Sequence[str]) -> None:
        self.segmenter.set_model_ready_states(state_ids)

    def identify_current_regime(self, df: pd.DataFrame) -> pd.Series:
        # Treat all preceding rows as encoder context and update the segmenter
        # with only the latest row, avoiding duplicate history assignments.
        result = self.transform(df, process_from=max(len(df) - 1, 0))
        return result.iloc[-1] if len(result) else pd.Series(dtype=object)

    def detect_null_regime(self, regimes: pd.DataFrame | None = None) -> bool:
        frame = self._history if regimes is None else regimes
        return self.validator.detect_null_regime(frame)

    def update_regime(self, embedding: Sequence[float], timestamp=None) -> dict:
        values = np.asarray(embedding, dtype=np.float64)
        columns = getattr(self.encoder, "embedding_columns_", None)
        if columns is None or len(columns) != len(values):
            columns = [f"embedding_{i}" for i in range(len(values))]
        index_value = timestamp if timestamp is not None else self.segmenter.get_observations_seen()
        indexed = pd.DataFrame([values], index=pd.Index([index_value]), columns=columns)
        assigned = self.segmenter.predict(indexed).iloc[0].to_dict()
        state_id = str(assigned["latent_id"])
        assigned["timestamp"] = timestamp
        assigned["embedding_ready"] = bool(np.isfinite(values).all())
        assigned["state_mature"] = bool(
            state_id != "null" and assigned.get("state_support", 0) >= self.mature_state_observations
        )
        assigned["model_ready"] = state_id in self.segmenter.get_model_ready_states()
        assigned["trade_allowed"] = bool(
            assigned["state_mature"]
            and assigned["raw_latent_id"] == state_id
            and not assigned.get("transition_pending", False)
            and (assigned["model_ready"] or not self.require_model_for_trade)
        )
        # Append one-row chunks; no full-history concat. update_regime never
        # added primitive features to _feature_history, and still does not.
        # update_regime() handles one indexed row at a time.
        if len(indexed.index) != 1:
            raise ValueError("update_regime expects exactly one indexed row")

        row_index = indexed.index[0]
        embedding_columns = pd.Index(columns)

        # Keep buffered embedding rows in one stable column order.
        if (
            self._embedding_pending_values
            and not embedding_columns.equals(
                self._embedding_pending_columns
            )
        ):
            self._flush_pending_history()

        # Preserve the original row mapping without creating a DataFrame.
        if not self._history_pending_rows:
            self._history_pending_index_template = indexed.index.copy()

        self._history_pending_rows.append(dict(assigned))
        self._history_pending_index.append(row_index)

        if not self._embedding_pending_values:
            self._embedding_pending_index_template = indexed.index.copy()
            self._embedding_pending_columns = embedding_columns.copy()

        self._embedding_pending_values.append(
            np.asarray(values).reshape(-1).copy()
        )
        self._embedding_pending_index.append(row_index)

        # Materialize only every 4096 single-row updates.
        if len(self._history_pending_rows) >= self._history_chunk_size:
            self._flush_pending_history()

        return assigned

    def get_regime(self, regime_id: str | None = None):
        if regime_id is None:
            return self._history.iloc[-1] if len(self._history) else None
        return self.segmenter.get_regime_prototypes().get(str(regime_id))

    def get_regime_history(self) -> pd.DataFrame:
        return self._history.copy()

    def validate_regimes(self, regimes: pd.DataFrame | None = None,
                         features: pd.DataFrame | None = None,
                         outcomes: pd.DataFrame | None = None,
                         outcome_columns: Sequence[str] | None = None,
                         ground_truth: Sequence[str] | None = None) -> dict:
        frame = self._history if regimes is None else regimes
        embeddings = self._embedding_history if len(self._embedding_history) else self.embeddings_
        if embeddings is not None and len(embeddings) != len(frame):
            embeddings = None
        validation_features = self._feature_history if features is None else features
        if validation_features is not None and len(validation_features) != len(frame):
            validation_features = None
        return self.validator.validate_regime(frame, embeddings=embeddings,
                                              features=validation_features,
                                              outcomes=outcomes, outcome_columns=outcome_columns,
                                              ground_truth=ground_truth)

    def get_similarity_scores(self) -> dict:
        result = self.validate_regimes()
        return {key: value for key, value in result.items() if "similarity" in key}

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path) -> "OODSystem":
        with Path(path).open("rb") as stream:
            result = pickle.load(stream)
        if not isinstance(result, cls) or result.schema_version != cls.schema_version:
            raise ValueError("saved OOD system type or schema version is incompatible")
        return result

    def __setstate__(self, state):
        """Migrate pickles written before the chunked-history layout.

        Old instances stored full DataFrames under _history / _embedding_history
        / _feature_history / embeddings_. The new layout stores _*_base plus
        _*_chunks and exposes the history via properties, so those legacy keys
        are folded into the corresponding *_base and the rest dropped.
        """
        state = dict(state)
        layouts = {
            "_history_base": ("_history",),
            "_embedding_base": ("_embedding_history", "embeddings_"),
            "_feature_base": ("_feature_history",),
        }
        chunks_for_base = {
            "_history_base": "_history_chunks",
            "_embedding_base": "_embedding_chunks",
            "_feature_base": "_feature_chunks",
        }
        for base_name, legacy_names in layouts.items():
            if base_name not in state:
                for old_name in legacy_names:
                    if old_name in state and state[old_name] is not None:
                        state[base_name] = state[old_name]
                        break
            state.pop("embeddings_", None)
            for old_name in legacy_names:
                state.pop(old_name, None)
            state.setdefault(base_name, None)
            state.setdefault(chunks_for_base[base_name], [])
        # Complete the migration: legacy pickles have none of the new
        # buffer/compaction attrs, and every history read/update touches them.
        # (Not in the spec's __setstate__ verbatim, but its PROOF step 4
        # requires reading history from a migrated instance.)
        state.setdefault("_history_chunk_size", 4096)
        state.setdefault("_max_history_chunks", 64)
        state.setdefault("_history_pending_rows", [])
        state.setdefault("_history_pending_index", [])
        state.setdefault("_history_pending_index_template", None)
        state.setdefault("_embedding_pending_values", [])
        state.setdefault("_embedding_pending_index", [])
        state.setdefault("_embedding_pending_index_template", None)
        state.setdefault("_embedding_pending_columns", None)
        self.__dict__.update(state)

    def _timestamps(self, df: pd.DataFrame):
        name = self.timestamp_col or getattr(self.encoder, "timestamp_col", None) or ("timestamp" if "timestamp" in df else None)
        if name is not None and name in df:
            return pd.to_datetime(df[name], utc=True, errors="coerce").array
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.DatetimeIndex(pd.to_datetime(df.index, utc=True, errors="coerce")).array
        return df.index.array

    def _validate_timestamps(self, df: pd.DataFrame) -> None:
        name = self.timestamp_col or getattr(self.encoder, "timestamp_col", None) or ("timestamp" if "timestamp" in df else None)
        if name is not None:
            if name not in df:
                if isinstance(df.index, pd.DatetimeIndex):
                    values = pd.to_datetime(df.index, utc=True, errors="coerce")
                else:
                    raise KeyError(f"timestamp column {name!r} is missing")
            else:
                values = pd.to_datetime(df[name], utc=True, errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            values = pd.to_datetime(df.index, utc=True, errors="coerce")
        else:
            return
        if values.isna().any() or not values.is_monotonic_increasing:
            raise ValueError("OOD input timestamps must be valid and monotonically increasing")


# Compatibility name for existing callers during the migration.
OODRegimeDetector = OODSystem