"""Thin Python wrapper over the Rust crate. Only column selection / DataFrame plumbing lives here."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import gpcore  # built from gpcore/ with `maturin build --release`

def prepare_terminals(frame: pd.DataFrame, input_columns=None, normalization_window: int = 500,
                      normalization_min_periods: int = 50, clip_z: float = 8.0) -> pd.DataFrame:
    """Convert selected numeric source columns to causal normalized GP terminals in Rust."""
    names = [str(column) for column in frame.columns]
    raw = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values, selected_names = gpcore.prepare_terminals(
        np.ascontiguousarray(raw), names,
        None if input_columns is None else [str(column) for column in input_columns],
        normalization_window, normalization_min_periods, clip_z,
    )
    return pd.DataFrame(values, index=frame.index, columns=selected_names)


def default_gp_config() -> dict:
    """Return native Rust defaults plus terminal preprocessing settings."""
    config = json.loads(gpcore.default_config_json())
    config.update(
        normalization_window=500,
        normalization_min_periods=50,
        clip_z=8.0,
        correlation_threshold=0.5,
    )
    return config


@dataclass
class GPFeatureSet:
    library: dict
    terminal_names: list
    _json: str = field(repr=False, default="")
    normalization: dict = field(default_factory=lambda: {
        "normalization_window": 500, "normalization_min_periods": 50, "clip_z": 8.0,
    })

    @property
    def names(self) -> list:
        return [f["name"] for f in self.library["features"]]

    @property
    def formulas(self) -> dict:
        return {f["name"]: f["formula"] for f in self.library["features"]}

    @property
    def stats(self) -> dict:
        return self.library.get("stats", {})

    def transform(self, terminals: pd.DataFrame, threads: int = 0) -> pd.DataFrame:
        """Evaluate every selected formula on any frame of terminals (causal, same code as live)."""
        x = np.ascontiguousarray(terminals[self.terminal_names].to_numpy(float))
        out = gpcore.transform(x, self._json, threads)
        return pd.DataFrame(out, index=terminals.index, columns=self.names)

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        library = dict(self.library, normalization=self.normalization)
        p.write_text(json.dumps(library, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return p


def load_feature_set(path) -> GPFeatureSet:
    txt = Path(path).read_text(encoding="utf-8"); lib = json.loads(txt)
    normalization = lib.get("normalization", {})
    return GPFeatureSet(lib, list(lib["terminals"]), txt, normalization)


def search_features(terminals: pd.DataFrame, cfg: dict | None = None, fit_rows: int | float | None = None,
                    target: pd.Series | None = None) -> GPFeatureSet:
    """Target-free novelty search. ``fit_rows`` restricts the SEARCH to the first rows (leak guard:
    selection then never sees the evaluation period); features are still computable on every row."""
    c = default_gp_config(); c.update(cfg or {})
    n = len(terminals)
    k = n if fit_rows is None else (int(n * fit_rows) if fit_rows < 1 else min(int(fit_rows), n))
    sub = terminals.iloc[:k]
    first = int(sub.notna().all(axis=1).to_numpy().argmax()) if sub.notna().all(axis=1).any() else 0
    x = np.ascontiguousarray(sub.iloc[first:].to_numpy(float))
    tgt = None if target is None else np.ascontiguousarray(target.iloc[:k].iloc[first:].to_numpy(float))
    txt = gpcore.search(x, [str(c_) for c_ in terminals.columns], json.dumps(c), tgt)
    lib = json.loads(txt)
    return GPFeatureSet(lib, [str(c_) for c_ in terminals.columns], txt)


def enforce_correlation_screen(
    feature_set: GPFeatureSet,
    terminals: pd.DataFrame,
    threshold: float = 0.5,
    verify_rows: int | None = None,
) -> GPFeatureSet:
    """Apply exact Rust Pearson and Spearman screens over all rows by default."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    features = feature_set.transform(terminals)
    values = np.ascontiguousarray(features.to_numpy(dtype=np.float64))
    if values.shape[1] <= 1:
        return feature_set

    sample_rows = values.shape[0] if verify_rows is None else min(verify_rows, values.shape[0])
    if sample_rows < 2:
        raise ValueError(f"not enough rows ({values.shape[0]}) for GP correlation screening")

    def keep_indices(matrix: np.ndarray, metric: str) -> list[int]:
        return list(gpcore.correlation_prune(
            np.ascontiguousarray(matrix), threshold, metric, sample_rows, 0
        ))

    pearson_indices = keep_indices(values, "pearson")
    if not pearson_indices:
        raise ValueError("Rust Pearson correlation screen rejected every GP feature")
    pearson_values = values[:, pearson_indices]
    spearman_local = keep_indices(pearson_values, "spearman")
    if not spearman_local:
        raise ValueError("Rust Spearman correlation screen rejected every GP feature")
    selected_indices = {pearson_indices[i] for i in spearman_local}

    feature_set.library["features"] = [
        item for i, item in enumerate(feature_set.library["features"])
        if i in selected_indices
    ]
    feature_set.library.setdefault("stats", {})["correlation_screen"] = {
        "threshold": threshold,
        "verification_rows": sample_rows,
        "verification_scope": "all_rows" if sample_rows == values.shape[0] else "sampled_rows",
        "input_features": int(values.shape[1]),
        "pearson_features": len(pearson_indices),
        "selected_features": len(selected_indices),
    }
    feature_set._json = json.dumps(feature_set.library, separators=(",", ":"), allow_nan=False)
    return feature_set