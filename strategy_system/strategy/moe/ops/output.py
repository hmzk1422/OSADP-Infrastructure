"""Minimal overwrite outputs: one small fixed-name CSV per stage.

``ops/output/`` holds at most these files (overwritten every run, never
timestamped)::

    features.csv  feature-column list (column)
    ood.csv       latent counts (latent_id, n, share)
    targets.csv   target coverage (column, n_finite, coverage)
    gp.csv        GP selection report (as returned, may be empty)
    frame.csv     model-frame head (first 500 rows, kept small on purpose)
    barrier.csv   triple-barrier events (as returned, may be empty)
    policy.csv    policy report (as returned, may be empty)

Full frames stay in memory on ``StrategyResult``; these CSVs are the
small reviewable slice. Pass ``enabled=False`` (``settings["ops"]["output"]``)
to skip disk writes (tests / smoke runs).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .logging import get_logger
from .paths import ops_paths

OUTPUT_FILES = (
    "features.csv",
    "ood.csv",
    "targets.csv",
    "gp.csv",
    "frame.csv",
    "barrier.csv",
    "policy.csv",
)

FRAME_CSV_ROWS = 500


def output_enabled(cfg: dict | None = None) -> bool:
    ops_cfg = dict((cfg or {}).get("ops") or {})
    if "output" in ops_cfg and isinstance(ops_cfg["output"], bool):
        return bool(ops_cfg["output"])
    return bool(ops_cfg.get("output_enabled", True))


def write_output(name: str, frame: pd.DataFrame, cfg: dict | None = None) -> Path | None:
    """Overwrite ``ops/output/<name>`` with ``frame``; return path or None when disabled."""
    if not output_enabled(cfg):
        return None
    out_dir = ops_paths(cfg).output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    frame.to_csv(path, index=False)
    get_logger("strategy", cfg).info("wrote output/%s (%d rows x %d cols)", name, len(frame), len(frame.columns))
    return path


def write_frame_head(frame: pd.DataFrame, cfg: dict | None = None, rows: int = FRAME_CSV_ROWS) -> Path | None:
    """Overwrite ``ops/output/frame.csv`` with the first ``rows`` of the model frame."""
    return write_output("frame.csv", frame.head(rows), cfg)


__all__ = ["OUTPUT_FILES", "FRAME_CSV_ROWS", "output_enabled", "write_output", "write_frame_head"]
