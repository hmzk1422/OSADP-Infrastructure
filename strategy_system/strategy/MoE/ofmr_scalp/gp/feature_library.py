"""Versioned persistence for offline validated features, grouped by latent ID."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .grammar import node_from_dict


SCHEMA_VERSION = 1


def build_feature_library(selected_features=(), *, metadata=None):
    library = {"schema_version": SCHEMA_VERSION, "metadata": dict(metadata or {}), "regimes": {}}
    for item in selected_features:
        state = str(item.state_id)
        library["regimes"].setdefault(state, []).append({
            "target_column": item.target_column,
            "output_column": item.output_column,
            "expression": item.expression,
            "train_ic": float(item.train_ic),
            "validation_ic": float(item.validation_ic),
            "fitness": float(item.fitness),
            "direction": int(item.direction),
            "n_train": int(item.n_train),
            "n_validation": int(item.n_validation),
            "model_metrics": dict(getattr(item, "model_metrics", {}) or {}),
        })
    return library


def select_validated_features(report, *, min_fitness=None, min_validation_ic=None):
    if report is None or report.empty:
        return report.copy() if report is not None else report
    selected = report.copy()
    if min_fitness is not None and "fitness" in selected:
        selected = selected[selected["fitness"] >= min_fitness]
    if min_validation_ic is not None and "validation_ic" in selected:
        selected = selected[selected["validation_ic"].abs() >= min_validation_ic]
    return selected.reset_index(drop=True)


def _json_default(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: getattr(value, key) for key in value.__dataclass_fields__}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_feature_library(library, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(library)
    data.setdefault("schema_version", SCHEMA_VERSION)
    target.write_text(json.dumps(data, default=_json_default, indent=2), encoding="utf-8")
    return target


def load_feature_library(path):
    library = json.loads(Path(path).read_text(encoding="utf-8"))
    if library.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported feature-library schema: {library.get('schema_version')}")
    for records in library.get("regimes", {}).values():
        for record in records:
            if isinstance(record.get("expression"), dict):
                record["expression"] = node_from_dict(record["expression"])
    return library


def get_features_for_regime(library, regime_id):
    return list(library.get("regimes", {}).get(str(regime_id), ()))


def register_regime_features(library, regime_id, features, *, replace=True):
    result = {**library, "regimes": dict(library.get("regimes", {}))}
    key = str(regime_id)
    existing = [] if replace else list(result["regimes"].get(key, ()))
    result["regimes"][key] = existing + list(features)
    return result

