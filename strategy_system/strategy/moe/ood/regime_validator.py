"""Diagnostics for recurring regimes; forward outcomes are an optional later input."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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

    def _temporal_metric(self, tdmi_kwargs: dict | None = None):
        """Resolve the temporal metric; pass fixed TDMI settings through consistently."""
        name = self.metrics["temporal_pattern_similarity"]
        if name == "resampled":
            return self.calculate_temporal_similarity
        elif name == "dtw":
            return _metrics.dtw_similarity
        elif name == "flattened_cosine":
            return _metrics.flattened_cosine_similarity
        elif name == "tdmi":
            options = dict(tdmi_kwargs or {})
            allowed = {"max_lag", "bins", "min_pairs", "n_surrogates", "seed", "floor"}
            options = {key: value for key, value in options.items()
                       if key in allowed and value is not None}
            return _metrics.make_tdmi_similarity(**options)
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
        """Vectorized episode run extractor using NumPy."""
        arr = np.asarray(labels, dtype=str)
        if arr.size == 0:
            return {}
        
        # Find continuous run boundaries instantly via vectorization
        change = np.flatnonzero(arr[:-1] != arr[1:])
        starts = np.concatenate([[0], change + 1])
        ends = np.concatenate([change + 1, [len(arr)]])
        run_states = arr[starts]
        
        episodes: dict[str, list[tuple[int, int]]] = {}
        for state_id, start, end in zip(run_states, starts, ends):
            if state_id == self.null_id and not include_null:
                continue
            spans = episodes.setdefault(state_id, [])
            if spans and self.episode_gap_tolerance > 0:
                previous_start, previous_end = spans[-1]
                gap = start - previous_end
                if (gap <= self.episode_gap_tolerance
                        and np.all(arr[previous_end:start] == self.null_id)):
                    spans[-1] = (previous_start, end)
                    continue
            spans.append((int(start), int(end)))
        return episodes
    
    @staticmethod
    def _contiguous_runs(ids: np.ndarray, state_id: str, start: int, end: int) -> list[tuple[int, int]]:
        """Maximal runs of ``state_id`` inside ``[start, end)``; null gaps are never bridged."""
        mask = (ids[start:end] == state_id).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask, [0]])))
        return [(start + a, start + b) for a, b in zip(edges[::2], edges[1::2])]

    @staticmethod
    def _check_alignment(labels, values) -> None:
        """Refuse inputs whose pandas indexes disagree (equal length is not alignment).

        A default ``RangeIndex`` on exactly one side is treated as "positional" and passes.
        """
        pandas_types = (pd.Series, pd.DataFrame)
        if not (isinstance(labels, pandas_types) and isinstance(values, pandas_types)):
            return
        left, right = labels.index, values.index
        if left.equals(right):
            return
        if isinstance(left, pd.RangeIndex) != isinstance(right, pd.RangeIndex):
            return
        raise ValueError("labels and values have different indexes; align them before validating")

    @staticmethod
    def _finite_mean(array) -> float:
        array = np.asarray(array, dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(array.mean()) if len(array) else np.nan

    def _tdmi_collect(self, regimes, values, max_lag=None, bins=None, min_pairs=None,
                      n_surrogates=None, seed: int = 0,
                      bin_selection_end: int | None = None) -> dict:
        """Per-episode within-episode TDMI estimates (the shared first pass).

        Episodes follow ``_episodes`` (null gaps within tolerance are merged) but lag pairs
        are formed only inside contiguous runs of the latent, so neither episode boundaries
        nor null gaps create pairs. One bin count is used for every episode and latent.

        If ``bin_selection_end`` is supplied, automatic bin selection uses only complete
        episodes ending at or before that row (the training period); later episodes are
        still estimated, with the fixed training-selected bin count.
        """
        labels = regimes[self.regime_column] if isinstance(regimes, pd.DataFrame) else pd.Series(regimes)
        self._check_alignment(regimes if isinstance(regimes, pd.DataFrame) else labels, values)
        matrix = self._as_matrix(values)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if len(labels) != len(matrix):
            raise ValueError("labels and values must have equal row counts")
        names = list(values.columns) if isinstance(values, pd.DataFrame) else [f"f{i}" for i in range(matrix.shape[1])]
        ids = labels.astype(str).to_numpy()
        times = None
        if isinstance(regimes, pd.DataFrame) and "timestamp" in regimes:
            times = pd.to_datetime(regimes["timestamp"], utc=True, errors="coerce").to_numpy()
        n_surrogates = _metrics.TDMI_SURROGATES if n_surrogates is None else int(n_surrogates)
        eff_pairs = _metrics.TDMI_MIN_PAIRS if min_pairs is None else int(min_pairs)
        max_lag, bins, _ = _metrics._check_tdmi_args(max_lag, bins, eff_pairs)
        if n_surrogates < 0:
            raise ValueError("n_surrogates must be >= 0")

        raw = []
        for state_id, spans in self._episodes(ids).items():
            for start, end in spans:
                segments = [matrix[a:b] for a, b in self._contiguous_runs(ids, state_id, start, end)]
                if segments:
                    raw.append((state_id, start, end, segments))
        if bins is None:
            sizes = [
                len(seg) - 1
                for _, start, end, segments in raw
                if bin_selection_end is None or end <= int(bin_selection_end)
                for seg in segments
                if len(seg) - 1 >= eff_pairs
            ]
            bins = _metrics.tdmi_bins(np.median(sizes)) if sizes else 2
        episodes = []
        for index, (state_id, start, end, segments) in enumerate(tqdm(raw, desc="TDMI Estimation")):
            estimate = _metrics.tdmi_estimate(segments, max_lag, bins, min_pairs, n_surrogates, seed=[seed, index])
            row = {"latent_id": state_id, "start": start, "end": end,
                   "observations": int(sum(len(seg) for seg in segments)), "estimate": estimate}
            if times is not None:
                row["start_time"], row["end_time"] = times[start], times[end - 1]
            episodes.append(row)
        lags = episodes[0]["estimate"].mi.shape[0] if episodes else int(max_lag or _metrics.TDMI_MAX_LAG)
        return {"episodes": episodes, "bins": bins, "names": names, "lags": lags,
                "n_rows": len(matrix), "n_surrogates": n_surrogates, "bin_selection_end": bin_selection_end}

    def _cells(self, estimate) -> tuple[np.ndarray, np.ndarray | None]:
        """(raw cells clipped at 0, null-calibrated cells or None)."""
        raw = np.maximum(estimate.mi, 0.0)
        return raw, (_metrics.tdmi_calibrate(estimate) if estimate.null is not None else None)

    def _tdmi_tables(self, collected: dict, min_episodes: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
        columns = ["episode_count", "observations", "recurrent", "tdmi", "tdmi_lag1", "tdmi_null", "tdmi_net", "tdmi_p",
                   "tdmi_sig_frac", "tdmi_se", "tdmi_others", "tdmi_excess", "tdmi_gt_others", "tdmi_rank"]
        episodes = collected["episodes"]
        if not episodes:
            empty = pd.DataFrame(columns=columns)
            empty.index.name = "latent_id"
            return empty, pd.DataFrame()

        episode_rows, signed = [], {}
        for index, ep in enumerate(episodes):
            est = ep["estimate"]
            raw, cal = self._cells(est)
            signed.setdefault(ep["latent_id"], []).append(self._finite_mean(est.mi))
            row = {k: v for k, v in ep.items() if k != "estimate"}
            row.update({"n_pairs": int(est.weight[0].max()), "tdmi": self._finite_mean(raw),
                        "tdmi_lag1": self._finite_mean(raw[0]),
                        "tdmi_cal": self._finite_mean(cal) if cal is not None else np.nan,
                        "tdmi_sig_frac": float(np.mean(cal[np.isfinite(cal)] > 0)) if cal is not None and np.isfinite(cal).any() else np.nan})
            episode_rows.append(row)
        episode_frame = pd.DataFrame(episode_rows)
        episode_frame.insert(1, "episode", episode_frame.groupby("latent_id").cumcount())

        rows = {}
        for state_id in pd.unique(episode_frame["latent_id"]):
            members = [ep for ep in episodes if ep["latent_id"] == state_id]
            est = _metrics.combine_tdmi_estimates([ep["estimate"] for ep in members])
            raw, cal = self._cells(est)
            score = self._finite_mean(raw)
            row = {"episode_count": len(members), "observations": int(sum(ep["observations"] for ep in members)),
                   "tdmi": score, "tdmi_lag1": self._finite_mean(raw[0]),
                   "tdmi_null": np.nan, "tdmi_p": np.nan, "tdmi_sig_frac": np.nan}
            if est.null is not None:
                clipped = np.maximum(est.null, 0.0)
                finite = np.isfinite(clipped)
                flat = clipped.reshape(len(clipped), -1)
                counts = finite.reshape(len(clipped), -1).sum(axis=1)
                null_scores = np.where(counts > 0, np.where(finite.reshape(len(clipped), -1), flat, 0.0).sum(axis=1)
                                       / np.maximum(counts, 1), np.nan)
                row["tdmi_null"] = self._finite_mean(null_scores)
                row["tdmi_p"] = float((1 + np.sum(null_scores >= score)) / (len(null_scores) + 1)) if np.isfinite(score) else np.nan
                row["tdmi_sig_frac"] = float(np.mean(cal[np.isfinite(cal)] > 0)) if np.isfinite(cal).any() else np.nan
            per_episode = np.asarray([v for v in signed[state_id] if np.isfinite(v)])
            row["tdmi_se"] = float(per_episode.std(ddof=1) / np.sqrt(len(per_episode))) if len(per_episode) > 1 else np.nan
            rows[state_id] = row
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "latent_id"
        table["recurrent"] = table["episode_count"] >= int(min_episodes)
        table["tdmi_net"] = table["tdmi"] - table["tdmi_null"]
        # Contrast on the null-adjusted score when surrogates exist, so latents whose episodes are
        # shorter (higher noise floor) are not advantaged; otherwise on the raw score.
        basis = table["tdmi_net"] if table["tdmi_net"].notna().any() else table["tdmi"]
        others = []
        for state_id in table.index:
            rest = basis[basis.index != state_id].dropna()
            others.append(float(rest.mean()) if len(rest) else np.nan)
        table["tdmi_others"] = others
        table["tdmi_excess"] = basis - table["tdmi_others"]
        table["tdmi_gt_others"] = basis > table["tdmi_others"]
        table["tdmi_rank"] = basis.rank(ascending=False, method="min")
        return table[columns], episode_frame

    def calculate_tdmi(self, regimes: pd.DataFrame | Sequence[str], values,
                       max_lag: int | None = None, bins: int | None = None,
                       min_pairs: int | None = None, min_episodes: int = 2,
                       n_surrogates: int | None = None, seed: int = 0,
                       return_episodes: bool = False):
        """Time-delayed mutual information (TDMI) per latent regime.

        ``values`` are the row-aligned features (or embeddings) whose temporal
        self-dependence I(x[t]; x[t+tau]) is measured for tau = 1..max_lag bars. Estimation
        is within-episode (level shifts between recurrences cannot create dependence), and
        every estimate has a circular-shift surrogate null. Returns a frame indexed by
        ``latent_id`` (joinable with ``recurrence`` / ``persistence`` / ``expectancy_stability``):

            episode_count, observations, recurrent
            tdmi            pair-weighted within-episode TDMI, mean over lags and features (nats)
            tdmi_lag1       the same at lag 1
            tdmi_null       mean TDMI of the surrogates (what "no temporal structure" scores)
            tdmi_net        tdmi - tdmi_null (null-adjusted)
            tdmi_p          surrogate p-value of ``tdmi`` against that null (vs. no structure,
                            NOT a test between latents)
            tdmi_sig_frac   share of (lag, feature) cells above the null's 95th percentile
            tdmi_se         standard error of per-episode TDMI across recurrences
            tdmi_others / tdmi_excess / tdmi_gt_others / tdmi_rank   exploratory contrasts with
                            the other latents, on ``tdmi_net`` when surrogates are on; descriptive,
                            not significance tests

        With ``return_episodes=True`` also returns the per-episode frame. ``n_surrogates=0``
        skips the null (the null/p/sig columns are then NaN).
        """
        collected = self._tdmi_collect(regimes, values, max_lag, bins, min_pairs, n_surrogates, seed)
        table, episodes = self._tdmi_tables(collected, min_episodes)
        return (table, episodes) if return_episodes else table

    def calculate_tdmi_fingerprints(self, regimes: pd.DataFrame | Sequence[str], values,
                                    max_lag: int | None = None, bins: int | None = None,
                                    min_pairs: int | None = None, n_surrogates: int | None = None,
                                    seed: int = 0) -> pd.DataFrame:
        """Null-calibrated (lag x feature) TDMI fingerprint of each latent.

        Rows are ``latent_id``; columns are a (lag, feature) MultiIndex, so *which* feature
        carries the temporal dependence is preserved (``tdmi`` averages it away).
        """
        collected = self._tdmi_collect(regimes, values, max_lag, bins, min_pairs, n_surrogates, seed)
        episodes = collected["episodes"]
        columns = pd.MultiIndex.from_product([range(1, collected["lags"] + 1), collected["names"]],
                                             names=["lag", "feature"])
        rows = {}
        for state_id in dict.fromkeys(ep["latent_id"] for ep in episodes):
            est = _metrics.combine_tdmi_estimates([ep["estimate"] for ep in episodes if ep["latent_id"] == state_id])
            raw, cal = self._cells(est)
            rows[state_id] = (cal if cal is not None else raw).ravel()
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns) if rows else pd.DataFrame(columns=columns)
        frame.index.name = "latent_id"
        return frame

    def _tdmi_recurrence(self, collected: dict, train_fraction: float = 0.5) -> pd.DataFrame:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        cut = int(round(collected["n_rows"] * train_fraction))
        episodes = collected["episodes"]
        # Purged chronological split: episodes straddling the cut belong to neither side.
        train = [ep for ep in episodes if ep["end"] <= cut]
        test = [ep for ep in episodes if ep["start"] >= cut]

        def fingerprint(est):
            raw, cal = self._cells(est)
            return (cal if cal is not None else raw).ravel()

        prints = {}
        for state_id in dict.fromkeys(ep["latent_id"] for ep in train):
            prints[state_id] = fingerprint(_metrics.combine_tdmi_estimates(
                [ep["estimate"] for ep in train if ep["latent_id"] == state_id]))
        columns = ["test_episodes", "own_similarity", "other_similarity", "similarity_gap", "top1_hit_rate"]
        test = [ep for ep in test if ep["latent_id"] in prints]
        out = pd.DataFrame(columns=columns)
        out.index.name = "latent_id"
        out.attrs.update({"top1": np.nan, "chance": np.nan, "majority_baseline": np.nan, "n_test": len(test)})
        if len(test) == 0 or len(prints) < 2:
            return out
        known = list(prints)
        rows, hits = {}, []
        for ep in test:
            fp = fingerprint(ep["estimate"])
            sims = np.asarray([_metrics.tdmi_fingerprint_similarity(fp, prints[k]) for k in known])
            own = known.index(ep["latent_id"])
            best = np.flatnonzero(np.isclose(sims, np.nanmax(sims))) if np.isfinite(sims).any() else np.array([], dtype=int)
            hit = (1.0 / len(best)) if own in best else 0.0          # ties share the credit
            others = np.delete(sims, own)
            rows.setdefault(ep["latent_id"], []).append((sims[own], self._finite_mean(others), hit))
            hits.append(hit)
        for state_id, vals in rows.items():
            arr = np.asarray(vals, dtype=np.float64)
            out.loc[state_id] = [len(arr), self._finite_mean(arr[:, 0]), self._finite_mean(arr[:, 1]),
                                 self._finite_mean(arr[:, 0] - arr[:, 1]), float(arr[:, 2].mean())]
        shares = pd.Series([ep["latent_id"] for ep in test]).value_counts(normalize=True)
        out.attrs.update({"top1": float(np.mean(hits)), "chance": 1.0 / len(known),
                          "majority_baseline": float(shares.max()), "n_test": len(test)})
        return out

    def calculate_tdmi_recurrence(self, regimes: pd.DataFrame | Sequence[str], values,
                                  train_fraction: float = 0.5, max_lag: int | None = None,
                                  bins: int | None = None, min_pairs: int | None = None,
                                  n_surrogates: int | None = None, seed: int = 0) -> pd.DataFrame:
        """Out-of-sample recurrence of TDMI fingerprints.

        Latent fingerprints are built from episodes in the first ``train_fraction`` of the
        rows only; each later episode is then matched against every latent's fingerprint.
        ``top1_hit_rate`` is how often an episode is closest to its own latent (ties share
        credit); compare ``attrs['top1']`` with ``attrs['chance']`` (1 / #latents) and
        ``attrs['majority_baseline']``. ``own_similarity`` vs ``other_similarity`` gives the
        gap per latent. Note: if the latent labels were themselves fitted on the full sample,
        this tests the labels' temporal structure, not the representation's out-of-sample
        stability -- fit the regime model on the training window for that.
        """
        collected = self._tdmi_collect(regimes, values, max_lag, bins, min_pairs, n_surrogates, seed)
        return self._tdmi_recurrence(collected, train_fraction)

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
        """Vectorized transition stability calculation."""
        labels = regimes[self.regime_column] if isinstance(regimes, pd.DataFrame) else regimes
        arr = np.asarray(labels, dtype=str)
        if len(arr) < 2:
            return 1.0
        changes = np.sum(arr[:-1] != arr[1:])
        return float(1.0 - changes / (len(arr) - 1))
    
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
        for i, left in enumerate(tqdm(ids, desc="Embedding Similarity")):
            for j in range(i, len(ids)):
                right = ids[j]
                scores = [pair_metric(a, b)
                          for li, a in enumerate(episode_means[left])
                          for ri, b in enumerate(episode_means[right])
                          if left != right or li < ri]
                finite = [score for score in scores if np.isfinite(score)]
                score = float(np.mean(finite)) if finite else np.nan
                matrix.loc[left, right] = matrix.loc[right, left] = score
        return matrix
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

    def build_temporal_similarity_matrix(self, embeddings: pd.DataFrame, labels: Sequence[str] | pd.Series,
                                         max_episode_pairs: int = 64, pair_sampling: str = "spread",
                                         sample_seed: int = 0, tdmi_kwargs: dict | None = None) -> pd.DataFrame:
        """Pairwise similarity of within-episode temporal paths.

        At most ``max_episode_pairs`` episode pairs are scored per cell. ``"spread"`` (default)
        draws them with a seeded RNG over all pairs, so every recurrence is represented;
        ``"first"`` is the legacy behaviour (first N in chronological order, which
        over-represents the earliest episodes).

        For TDMI, all-invalid rows are kept as separators (never deleted and bridged over),
        so two runs interrupted by another latent are compared as independent runs, and the
        same estimator settings (``tdmi_kwargs``) are used for every episode.
        """
        if pair_sampling not in ("spread", "first"):
            raise ValueError("pair_sampling must be 'spread' or 'first'")
        if max_episode_pairs < 1:
            raise ValueError("max_episode_pairs must be >= 1")
        if isinstance(labels, pd.Series):
            self._check_alignment(labels, embeddings)
        ids = pd.Series(np.asarray(labels), index=embeddings.index).astype(str)
        episodes = self._episodes(ids)
        matrix = pd.DataFrame(np.nan, index=list(episodes), columns=list(episodes), dtype=float)
        arrays = embeddings.to_numpy(dtype=np.float64)
        ids_array = ids.to_numpy()
        rng = np.random.default_rng(sample_seed)
        is_tdmi = self.metrics["temporal_pattern_similarity"] == "tdmi"
        options = dict(tdmi_kwargs or {})
        options.pop("bin_selection_end", None)
        pair_metric = self._temporal_metric(options) if is_tdmi else self._temporal_metric()

        # Cache each episode path once: the same episode is compared in many
        # pairs, so slice+mask is computed a single time per episode.
        path_cache: dict[tuple[int, int, str], np.ndarray] = {}

        def cached_path(a, b, state_id):
            key = (a, b, state_id)
            path = path_cache.get(key)
            if path is None:
                keep = ids_array[a:b] == state_id
                raw = arrays[a:b]
                path = np.where(keep[:, None], raw, np.nan) if is_tdmi else raw[keep]
                path_cache[key] = path
            return path

        for i, left in enumerate(tqdm(matrix.index, desc="Temporal Similarity")):
            spans_l = episodes[left]
            for j in range(i, len(matrix.columns)):
                right = matrix.columns[j]
                spans_r = episodes[right]
                diagonal = left == right
                n_left = len(spans_l)
                # Pair count WITHOUT materialising the O(E^2) list.
                total = n_left * (n_left - 1) // 2 if diagonal else n_left * len(spans_r)
                if total > max_episode_pairs:
                    if pair_sampling == "first":
                        indices = range(max_episode_pairs)
                    else:
                        indices = np.sort(rng.choice(total, max_episode_pairs, replace=False))
                else:
                    indices = range(total)
                # Decode each flat index to (lspan, rspan) in the exact order the
                # list comprehension produced, so seeding/order are unchanged.
                offsets = (np.array([li * (2 * n_left - li - 1) // 2 for li in range(n_left)],
                                    dtype=np.int64) if diagonal else None)
                pattern_scores = []
                for k in indices:
                    if diagonal:
                        li = int(np.searchsorted(offsets, k, side="right") - 1)
                        ri = li + 1 + (int(k) - int(offsets[li]))
                    else:
                        li, ri = divmod(int(k), len(spans_r))
                    (la, lb), (ra, rb) = spans_l[li], spans_r[ri]
                    pattern_scores.append(pair_metric(cached_path(la, lb, left),
                                                       cached_path(ra, rb, right)))
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
                        ground_truth: Sequence[str] | None = None,
                        include_tdmi: bool | None = None,
                        tdmi_kwargs: dict | None = None) -> dict[str, pd.DataFrame | float | bool]:
        """Return diagnostics by dimension; outcomes are analyzed only if passed.

        ``include_tdmi`` adds ``tdmi`` (per latent), ``tdmi_episodes`` (per episode),
        ``tdmi_recurrence`` (out-of-sample fingerprint matching) and ``tdmi_recurrence_top1``,
        computed on ``features`` if given, else ``embeddings``. ``None`` (default) turns it on
        only when ``temporal_pattern_similarity`` is set to ``tdmi``. ``tdmi_kwargs`` may hold
        max_lag, bins, min_pairs, n_surrogates, seed, min_episodes, train_fraction, and floor.

        The same ``tdmi_kwargs`` are threaded into the temporal similarity matrix and the TDMI
        tables so every TDMI output is estimated under one configuration. When ``bins`` is left
        automatic it is chosen from complete episodes ending within the initial
        ``train_fraction`` of the rows only, never from the held-out period.
        """
        labels = regimes[self.regime_column]
        for obj in (embeddings, features, outcomes):
            if obj is not None:
                self._check_alignment(regimes, obj)
        options = dict(tdmi_kwargs or {})
        min_episodes = int(options.pop("min_episodes", 2))
        train_fraction = float(options.pop("train_fraction", 0.5))
        floor = options.pop("floor", None)
        allowed = {"max_lag", "bins", "min_pairs", "n_surrogates", "seed"}
        unknown = set(options) - allowed
        if unknown:
            raise TypeError(f"unknown tdmi_kwargs: {sorted(unknown)}")
        tdmi_options = {key: value for key, value in options.items() if value is not None}
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
        bin_cut = int(round(len(regimes) * train_fraction))
        if embeddings is not None:
            result["embedding_similarity"] = self.build_regime_similarity_matrix(embeddings, labels)
            use_tdmi = self.metrics["temporal_pattern_similarity"] == "tdmi"
            matrix_options = dict(tdmi_options)
            if floor is not None:
                matrix_options["floor"] = floor
            if use_tdmi:
                matrix_options["bin_selection_end"] = bin_cut
            # TDMI is defined on observed order-flow/features when supplied; other
            # temporal metrics continue to compare embeddings.
            temporal_source = features if use_tdmi and features is not None else embeddings
            result["temporal_pattern_similarity"] = self.build_temporal_similarity_matrix(
                temporal_source, labels, tdmi_kwargs=matrix_options
            )
            result["episode_duration_similarity"] = self.build_duration_similarity_matrix(regimes)
            result["within_between_separation_ratio"] = self.calculate_separation_ratio(embeddings, labels)
        if features is not None:
            result["feature_similarity"] = self.build_feature_similarity_matrix(features, labels)
        if outcomes is not None:
            result["expectancy_stability"] = self.calculate_expectancy_stability(regimes, outcomes, outcome_columns)
            result["expectancy_similarity"] = self.build_expectancy_matrix(regimes, outcomes, outcome_columns)
        if ground_truth is not None:
            result["ground_truth_scores"] = self.calculate_ground_truth_scores(labels, ground_truth)
        explicit_tdmi = include_tdmi is not None
        if include_tdmi is None:
            include_tdmi = self.metrics["temporal_pattern_similarity"] == "tdmi"
        if include_tdmi:
            source = features if features is not None else embeddings
            if source is None:
                if explicit_tdmi:
                    raise ValueError("include_tdmi=True requires features or embeddings")
            else:
                collected = self._tdmi_collect(
                    regimes, source, **tdmi_options,
                    bin_selection_end=bin_cut if not options.get("bins") else None
                )
                result["tdmi"], result["tdmi_episodes"] = self._tdmi_tables(collected, min_episodes)
                result["tdmi"].attrs["source"] = "features" if features is not None else "embeddings"
                recurrence = self._tdmi_recurrence(collected, train_fraction)
                result["tdmi_recurrence"] = recurrence
                result["tdmi_recurrence_top1"] = recurrence.attrs["top1"]
        return result