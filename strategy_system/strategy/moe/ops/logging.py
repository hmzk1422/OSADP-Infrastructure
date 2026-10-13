"""Per-file loggers: one fixed file per stage, no timestamps in the name.

Files (all under ``ops/log/``)::

    strategy.log, features.log, ood.log, targets.log,
    gp.log, frame.log, barrier.log, policy.log, export.log

``get_logger("ood")`` appends to ``ops/log/ood.log`` and also echoes to
stdout. Each pipeline run writes a ``=== run <utc> ===`` header first so a
single file keeps the full history without time-based filenames.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paths import ops_paths

_LOGGERS: dict[str, logging.Logger] = {}

STAGES = (
    "strategy",
    "features",
    "ood",
    "targets",
    "gp",
    "frame",
    "barrier",
    "policy",
    "export",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_logger(stage: str, cfg: dict | None = None) -> logging.Logger:
    """Return (creating once) the logger for ``stage`` -> ``ops/log/<stage>.log``."""
    name = f"moe.{stage}"
    if name in _LOGGERS:
        return _LOGGERS[name]
    log_dir = ops_paths(cfg).log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
    _LOGGERS[name] = logger
    return logger


def log_header(stage: str, cfg: dict | None = None) -> logging.Logger:
    """Write the ``=== run <utc> ===`` separator and return the logger."""
    logger = get_logger(stage, cfg)
    logger.info("=== run %s ===", _utc_now())
    return logger


__all__ = ["STAGES", "get_logger", "log_header"]
