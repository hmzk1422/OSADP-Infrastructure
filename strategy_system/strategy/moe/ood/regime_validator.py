"""Diagnostics for recurring regimes; forward outcomes are an optional later input."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import metrics as _metrics


class RegimeValidator:
    """Measure representation, temporal, duration, persistence, and expectancy separately."""

    def __init__(self, regime_column: str = "latent_id", null_id: str = "null",
                 episode_gap_tolerance: int = 1, metrics: dict | None = None):
        if episode_gap_tolerance < 0:
            raise ValueError("episode_gap_tolerance must be >= 0")
        self.regime_column, self.null_id = regime_column, null_id
        self.episode_gap_tolerance = int(episode_gap_tolerance)
        # Per-slot similarity metric names (ood.metrics), validated on resolve.
        self.metrics = _metrics.resolve_metrics(metrics)

    def _vector_metric(self):
        """Resolve the pairwise vector metric for the given slot via if/elif."""
        name = self.metrics["embedding_similarity"]
        if name == "cosine":
            return self._cosine_similarity
        elif name == "euclidean":
            return _metrics.euclidean_similarity
        elif name == "pearson":
            return _metrics.pearson_similarity
        raise ValueError(f"unsupported embedding_similarity metric: {name}")

    def _feature_metric(self):
        name = self.metrics["feature_similarity"]
        if name == "cosine":
            return self._cosine_similarity
        elif name == "euclidean":
            return _metrics.euclidean_similarity
        elif name == "pearson":
            return _metrics.pearson_similarity
        raise ValueError(f"unsupported feature_similarity metric: {name}")

    def _temporal_metric(self):
        name = self.metrics["temporal_pattern_similarity"]
        if name == "resampled":
            return self.calculate_temporal_similarity
        elif name == "dtw":
            return _metrics.dtw_similarity
        elif name == "flattened_cosine":
            return _metrics.flattened_cosine_similarity
        raise ValueError(f"unsupported temporal_pattern_similarity metric: {name}")

    def _duration_metric(self):
        name = self.metrics["episode_duration_similarity"]
        if name == "quantile":
            return self._distribution_similarity
        elif name == "mean":
            return _metrics.mean_similarity
        elif name == "overlap":
            return _metrics.overlap_similarity
        raise ValueError(f"unsupported episode_duration_similarity metric: {name}")

    def prototype_correlation(self, prototypes: dict) -> pd.DataFrame:
        """Pairwise correlation of regime prototypes; metric chosen by ood.metrics."""
        name = self.metrics["prototype_pearson_corr"]
        if name == "pearson":
            return _metrics.prototype_pearson(prototypes)
        elif name == "spearman":
            return _metrics.prototype_spearman(prototypes)
        elif name == "kendall":
            return _metrics.prototype_kendall(prototypes)
        raise ValueError(f"unsupported prototype_pearson_corr metric: {name}")

    @staticmethod
    def _as_matrix(values) -> np.ndarray:
        return values.to_numpy(dtype=np.float64) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=np.float64)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            return np.nan
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.clip((np.dot(left, right) / denom + 1.0) / 2.0, 0.0, 1.0)) if denom > 1e-12 else float(np.allclose(left, right))

    def _groups(self, labels: pd.Series, values) -> dict[str, np.ndarray]:
        matrix = self._as_matrix(values)
        if len(labels) != len(matrix):
            raise ValueError("labels and values must have equal row counts")
        groups = {}
        ids = labels.astype(str).to_numpy()
        for state_id in pd.unique(ids):
            if state_id == self.null_id:
                continue
            selected = matrix[(ids == state_id) & np.isfinite(matrix).all(axis=1)]
            if len(selected):
                groups[state_id] = selected
        return groups

    def calculate_embedding_similarity(self, left, right) -> float:
        a, b = self._as_matrix(left), self._as_matrix(right)
        if a.ndim == 2:
            a = np.nanmean(a, axis=0)
        if b.ndim == 2:
            b = np.nanmean(b, axis=0)
        return self._cosine_similarity(a, b)

    def calculate_temporal_similarity(self, left, right) -> float:
        """Compare normalized time-ordered embedding paths with equal alignment."""
        a, b = self._as_matrix(left), self._as_matrix(right)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
            return np.nan
        # Resample episodes to a common relative-time grid; values retain order.
        points = min(max(len(a), len(b)), 64)
        grid = np.linspace(0.0, 1.0, points)
        def resample(array):
            source = np.linspace(0.0, 1.0, len(array))
            return np.stack([np.interp(grid, source, array[:, col]) for col in range(array.shape[1])], axis=1)
        aa, bb = resample(a), resample(b)
        if not np.isfinite(aa).all() or not np.isfinite(bb).all():
            return np.nan
        scale = np.sqrt(np.mean(np.concatenate([aa, bb], axis=0) ** 2))
        distance = np.sqrt(np.mean((aa - bb) ** 2)) / max(float(scale), 1e-12)
        return float(np.exp(-distance))

    def calculate_feature_similarity(self, left, right) -> float:
        return self.calculate_embedding_similarity(left, right)

    def _episodes(self, labels: Sequence[str], include_null: bool = False) -> dict[str, list[tuple[int, int]]]:
        ids = list(map(str, labels))
        runs: list[tuple[str, int, int]] = []
        start = 0
        while start < len(ids):
            end = start + 1
            while end < len(ids) and ids[end] == ids[start]:
                end += 1
            runs.append((ids[start], start, end))
            start = end
        episodes: dict[str, list[tuple[int, int]]] = {}
        for state_id, start, end in runs:
            if state_id == self.null_id and not include_null:
                continue
            spans = episodes.setdefault(state_id, [])
            if spans:
                previous_start, previous_end = spans[-1]
                gap = start - previous_end
                if (gap <= self.episode_gap_tolerance
                        and all(value == self.null_id for value in ids[previous_end:start])):
                    spans[-1] = (previous_start, end)
                    continue
            spans.append((start, end))
        return episodes

    def calculate_regime_persistence(self, regimes: pd.DataFrame | Sequence[str]) -> pd.DataFrame:
        labels = regimes[self.regime_column] if isinstance(regimes, pd.DataFrame) else pd.Series(regimes)
        episodes = self._episodes(labels, include_null=True)
        durations = self._duration_profile(regimes, include_null=True)
        rows = []
        for state_id, spans in episodes.items():
            lengths = durations[state_id]
            rows.append({"latent_id": state_id, "episode_count": len(lengths),
                         "mean_duration": float(lengths.mean()), "median_duration": float(np.median(lengths)),
                         "duration_std": float(lengths.std(ddof=0)),
                         "transition_rate": float(len(lengths) / max(len(labels), 1))})
        return pd.DataFrame(rows).set_index("latent_id") if rows else pd.DataFrame(
            columns=["episode_count", "mean_duration", "median_duration", "duration_std", "transition_rate"]
        )

    def _duration_profile(self, regimes: pd.DataFrame | Sequence[str], include_null: bool = False) -> dict[str, np.ndarray]:
        labels = regimes[self.regime_column] if isinstance(regimes, pd.DataFrame) else regimes
        episodes = self._episodes(labels, include_null=include_null)
        if isinstance(regimes, pd.DataFrame) and "timestamp" in regimes:
            times = pd.to_datetime(regimes["timestamp"], utc=True, errors="coerce")
            valid = times.notna().all()
            if valid:
                seconds = times.astype("int64").to_numpy(dtype=np.float64) / 1e9
                cadence = float(np.nanmedian(np.diff(seconds))) if len(seconds) > 1 else 1.0
                return {key: np.asarray([
                    max(seconds[end - 1] - seconds[start] + cadence, 0.0)
                    for start, end in spans
                ], dtype=np.float64) for key, spans in episodes.items()}
        return {key: np.asarray([end - start for start, end in spans], dtype=np.float64)
                for key, spans in episodes.items()}

    @staticmethod
    def _distribution_similarity(left: np.ndarray, right: np.ndarray) -> float:
        if not len(left) or not len(right):
            return np.nan
        quantiles = np.linspace(0.0, 1.0, 21)
        a, b = np.quantile(left, quantiles), np.quantile(right, quantiles)
        distance = float(np.mean(np.abs(a - b))) / max(float(np.mean(np.concatenate([left, right]))), 1.0)
        return float(np.exp(-distance))

    def calculate_expectancy_stability(self, regimes: pd.DataFrame, outcomes: pd.DataFrame,
                                       outcome_columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Summarize episode-level forward outcomes; never called by discovery."""
        if len(regimes) != len(outcomes):
            raise ValueError("regimes and outcomes must be row-aligned")
        cols = list(outcome_columns) if outcome_columns is not None else list(outcomes.select_dtypes(include=[np.number]).columns)
        if not cols:
            raise ValueError("no numeric forward outcome columns supplied")
        rows = []
        ids = regimes[self.regime_column].astype(str).to_numpy()
        spans_by_state = self._episodes(ids)
        for state_id in pd.unique(ids):
            if state_id == self.null_id:
                continue
            state_values = outcomes.loc[ids == state_id, cols]
            spans = spans_by_state.get(state_id, [])
            episode_means = pd.DataFrame([
                outcomes.iloc[start:end].loc[
                    regimes[self.regime_column].astype(str).iloc[start:end].to_numpy() == state_id,
                    cols,
                ].mean(axis=0)
                for start, end in spans
            ]) if spans else pd.DataFrame(columns=cols)
            row = {"latent_id": state_id, "observations": int(len(state_values)),
                   "episode_count": int(len(episode_means))}
            for col in cols:
                row[f"mean__{col}"] = float(episode_means[col].mean())
                row[f"std__{col}"] = float(episode_means[col].std(ddof=0))
            rows.append(row)
        return pd.DataFrame(rows).set_index("latent_id") if rows else pd.DataFrame()

    def calculate_transition_stability(self, regimes: pd.DataFrame | Sequence[str]) -> float:
        labels = regimes[self.regime_column] if isinstance(regimes, pd.DataFrame) else list(regimes)
        values = list(map(str, labels))
        if len(values) < 2:
            return 1.0
        return float(1.0 - sum(a != b for a, b in zip(values, values[1:])) / (len(values) - 1))

    def calculate_regime_similarity(self, embeddings, labels: Sequence[str] | pd.Series,
                                    metric=None) -> pd.DataFrame:
        label_series = pd.Series(labels)
        matrix_values = self._as_matrix(embeddings)
        if len(label_series) != len(matrix_values):
            raise ValueError("labels and embeddings must have equal row counts")
        pair_metric = metric or self._vector_metric()
        episodes = self._episodes(label_series)
        label_array = label_series.astype(str).to_numpy()
        episode_means = {}
        for state_id, spans in episodes.items():
            means = []
            for start, end in spans:
                sample = matrix_values[start:end]
                sample = sample[(label_array[start:end] == state_id)
                                & np.isfinite(sample).all(axis=1)]
                if len(sample):
                    means.append(sample.mean(axis=0))
            if means:
                episode_means[state_id] = means
        ids = list(episode_means)
        matrix = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
        for i, left in enumerate(ids):
            for j in range(i, len(ids)):
                right = ids[j]
                pairs = [(a, b) for li, a in enumerate(episode_means[left])
                         for ri, b in enumerate(episode_means[right])
                         if left != right or li < ri]
                scores = [pair_metric(a, b) for a, b in pairs]
                finite = [score for score in scores if np.isfinite(score)]
                score = float(np.mean(finite)) if finite else np.nan
                matrix.loc[left, right] = matrix.loc[right, left] = score
        return matrix

    def build_regime_similarity_matrix(self, embeddings, labels: Sequence[str] | pd.Series) -> pd.DataFrame:
        return self.calculate_regime_similarity(embeddings, labels)

    def build_feature_similarity_matrix(self, features, labels: Sequence[str] | pd.Series) -> pd.DataFrame:
        """Compare episode-aggregated input features independently of embeddings."""
        values = self._as_matrix(features)
        center = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        standardized = (values - center) / scale
        standardized = np.clip(standardized, -10.0, 10.0)
        return self.calculate_regime_similarity(standardized, labels, metric=self._feature_metric())

    def build_temporal_similarity_matrix(self, embeddings: pd.DataFrame, labels: Sequence[str] | pd.Series) -> pd.DataFrame:
        """Pairwise similarity of within-episode temporal embedding paths."""
        ids = pd.Series(labels, index=embeddings.index).astype(str)
        episodes = self._episodes(ids)
        matrix = pd.DataFrame(np.nan, index=list(episodes), columns=list(episodes), dtype=float)
        arrays = embeddings.to_numpy(dtype=np.float64)
        ids_array = ids.to_numpy()
        pair_metric = self._temporal_metric()
        for i, left in enumerate(matrix.index):
            spans_l = episodes[left]
            for j in range(i, len(matrix.columns)):
                right = matrix.columns[j]
                spans_r = episodes[right]
                pattern_scores = []
                episode_pairs = [(li, ri, lspan, rspan)
                                 for li, lspan in enumerate(spans_l)
                                 for ri, rspan in enumerate(spans_r)
                                 if left != right or li < ri]
                for _, _, (la, lb), (ra, rb) in episode_pairs[:64]:
                        # A diagonal compares different recurrences of the same ID;
                        # it never compares an episode with itself.
                        left_values = arrays[la:lb]
                        right_values = arrays[ra:rb]
                        left_values = left_values[(ids_array[la:lb] == left)]
                        right_values = right_values[(ids_array[ra:rb] == right)]
                        pattern_scores.append(pair_metric(left_values, right_values))
                finite = [score for score in pattern_scores if np.isfinite(score)]
                pattern = float(np.mean(finite)) if finite else np.nan
                matrix.loc[left, right] = matrix.loc[right, left] = pattern
        return matrix

    def build_duration_similarity_matrix(self, regimes: pd.DataFrame | Sequence[str]) -> pd.DataFrame:
        """Pairwise similarity of each regime's episode-duration distribution."""
        labels = regimes[self.regime_column] if isinstance(regimes, pd.DataFrame) else regimes
        profiles = self._duration_profile(regimes)
        ids = list(profiles)
        matrix = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
        pair_metric = self._duration_metric()
        for i, left in enumerate(ids):
            for j in range(i, len(ids)):
                right = ids[j]
                score = pair_metric(profiles[left], profiles[right])
                matrix.loc[left, right] = matrix.loc[right, left] = score
        return matrix

    def build_expectancy_matrix(self, regimes: pd.DataFrame, outcomes: pd.DataFrame,
                                outcome_columns: Sequence[str] | None = None) -> pd.DataFrame:
        summary = self.calculate_expectancy_stability(regimes, outcomes, outcome_columns)
        means = [c for c in summary.columns if c.startswith("mean__")]
        ids = list(summary.index)
        matrix = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
        for i, left in enumerate(ids):
            for j in range(i, len(ids)):
                right = ids[j]
                a, b = summary.loc[left, means].to_numpy(dtype=float), summary.loc[right, means].to_numpy(dtype=float)
                scale = np.nanmean(np.abs(np.concatenate([a, b])))
                score = float(np.exp(-np.nanmean(np.abs(a - b)) / max(scale, 1e-12)))
                matrix.loc[left, right] = matrix.loc[right, left] = score
        return matrix

    def calculate_separation_ratio(self, embeddings, labels: Sequence[str] | pd.Series) -> float:
        """Between-regime embedding variance divided by within-regime variance."""
        values = self._as_matrix(embeddings)
        ids = pd.Series(labels).astype(str).to_numpy()
        if len(values) != len(ids):
            raise ValueError("labels and embeddings must have equal row counts")
        valid = np.isfinite(values).all(axis=1) & (ids != self.null_id)
        values, ids = values[valid], ids[valid]
        unique = pd.unique(ids)
        if len(unique) < 2 or not len(values):
            return np.nan
        global_mean = values.mean(axis=0)
        between, within = 0.0, 0.0
        for state_id in unique:
            group = values[ids == state_id]
            center = group.mean(axis=0)
            between += len(group) * float(np.sum((center - global_mean) ** 2))
            within += float(np.sum((group - center) ** 2))
        dof = max(len(values) - len(unique), 1)
        return float((between / max(len(unique) - 1, 1)) / (within / dof + 1e-12))

    @staticmethod
    def _contingency(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        a, b = np.asarray(left, dtype=str), np.asarray(right, dtype=str)
        if a.shape != b.shape:
            raise ValueError("predicted labels and ground truth must have equal length")
        rows, row_index = np.unique(a, return_inverse=True)
        cols, col_index = np.unique(b, return_inverse=True)
        table = np.zeros((len(rows), len(cols)), dtype=np.int64)
        np.add.at(table, (row_index, col_index), 1)
        return table

    def calculate_ground_truth_scores(self, predicted: Sequence[str], ground_truth: Sequence[str]) -> dict[str, float]:
        """Permutation-invariant clustering scores plus assigned-label purity."""
        predicted = np.asarray(predicted, dtype=str)
        truth = np.asarray(ground_truth, dtype=str)
        table = self._contingency(predicted, truth)
        n = int(table.sum())
        choose2 = lambda counts: float(np.sum(counts * (counts - 1) / 2.0))
        cell_pairs = choose2(table)
        row_pairs, col_pairs = choose2(table.sum(axis=1)), choose2(table.sum(axis=0))
        total_pairs = n * (n - 1) / 2.0
        expected = row_pairs * col_pairs / total_pairs if total_pairs > 0 else 0.0
        maximum = 0.5 * (row_pairs + col_pairs)
        ari = (cell_pairs - expected) / (maximum - expected) if maximum != expected else 1.0

        probabilities = table / max(n, 1)
        row_prob, col_prob = probabilities.sum(axis=1), probabilities.sum(axis=0)
        nz = probabilities > 0
        mi = float(np.sum(probabilities[nz] * np.log(probabilities[nz] /
                         (row_prob[:, None] * col_prob[None, :])[nz])))
        h_pred = float(-np.sum(row_prob[row_prob > 0] * np.log(row_prob[row_prob > 0])))
        h_true = float(-np.sum(col_prob[col_prob > 0] * np.log(col_prob[col_prob > 0])))
        nmi = mi / np.sqrt(h_pred * h_true) if h_pred > 0 and h_true > 0 else float(np.array_equal(predicted, truth))

        assigned = predicted != self.null_id
        coverage = float(assigned.mean()) if n else 0.0
        if assigned.any():
            purity_table = self._contingency(predicted[assigned], truth[assigned])
            purity = float(purity_table.max(axis=1).sum() / purity_table.sum())
        else:
            purity = np.nan
        return {"ari": float(ari), "nmi": float(nmi), "recurrence_purity": purity,
                "assigned_coverage": coverage}

    def validate_regime_recurrence(self, regimes: pd.DataFrame, min_episodes: int = 2) -> pd.DataFrame:
        persistence = self.calculate_regime_persistence(regimes)
        if persistence.empty:
            return persistence.assign(recurrent=False)
        persistence["recurrent"] = (persistence["episode_count"] >= int(min_episodes)) & (persistence.index != self.null_id)
        return persistence

    def detect_null_regime(self, regimes: pd.DataFrame, null_fraction_threshold: float = 0.5) -> bool:
        if not 0 <= null_fraction_threshold <= 1:
            raise ValueError("null_fraction_threshold must be in [0, 1]")
        if regimes.empty:
            return True
        return bool(regimes[self.regime_column].astype(str).eq(self.null_id).mean() >= null_fraction_threshold)

    def validate_regime(self, regimes: pd.DataFrame, embeddings: pd.DataFrame | None = None,
                        features: pd.DataFrame | None = None,
                        outcomes: pd.DataFrame | None = None,
                        outcome_columns: Sequence[str] | None = None,
                        ground_truth: Sequence[str] | None = None) -> dict[str, pd.DataFrame | float | bool]:
        """Return diagnostics by dimension; outcomes are analyzed only if passed."""
        labels = regimes[self.regime_column]
        result: dict[str, pd.DataFrame | float | bool] = {
            "recurrence": self.validate_regime_recurrence(regimes),
            "persistence": self.calculate_regime_persistence(regimes),
            "transition_stability": self.calculate_transition_stability(labels),
            "null_regime_detected": self.detect_null_regime(regimes),
        }
        if "raw_latent_id" in regimes:
            raw_validator = RegimeValidator(regime_column="raw_latent_id", null_id=self.null_id)
            result["persistence_pre_smoothing"] = raw_validator.calculate_regime_persistence(regimes)
            result["transition_stability_pre_smoothing"] = raw_validator.calculate_transition_stability(regimes["raw_latent_id"])
            result["persistence_post_smoothing"] = result["persistence"]
            result["transition_stability_post_smoothing"] = result["transition_stability"]
        if embeddings is not None:
            result["embedding_similarity"] = self.build_regime_similarity_matrix(embeddings, labels)
            result["temporal_pattern_similarity"] = self.build_temporal_similarity_matrix(embeddings, labels)
            result["episode_duration_similarity"] = self.build_duration_similarity_matrix(regimes)
            result["within_between_separation_ratio"] = self.calculate_separation_ratio(embeddings, labels)
        if features is not None:
            result["feature_similarity"] = self.build_feature_similarity_matrix(features, labels)
        if outcomes is not None:
            result["expectancy_stability"] = self.calculate_expectancy_stability(regimes, outcomes, outcome_columns)
            result["expectancy_similarity"] = self.build_expectancy_matrix(regimes, outcomes, outcome_columns)
        if ground_truth is not None:
            result["ground_truth_scores"] = self.calculate_ground_truth_scores(labels, ground_truth)
        return result
