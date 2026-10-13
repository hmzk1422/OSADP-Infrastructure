"""Single place that resolves the ops/ directories.

``settings["ops"]`` may override any of these; otherwise everything anchors
at ``strategy_system/strategy/moe/ops/`` so runs are reproducible without
passing paths around:

    ops/log/     per-stage logs, fixed names (strategy.log, ood.log, ...)
    ops/output/  overwrite CSVs, fixed names (ood.csv, gp.csv, ...)
    ops/report/  full text reports, fixed names (ood.txt, gp.txt, run.txt)
    ops/models/  Rust bundle (regime/, latent_*/, manifest.json)

Relative overrides in settings anchor at the ops/ folder; absolute paths
are used verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parent
LOG_DIR_DEFAULT = OPS_DIR / "log"
OUTPUT_DIR_DEFAULT = OPS_DIR / "output"
REPORT_DIR_DEFAULT = OPS_DIR / "report"
MODELS_DIR_DEFAULT = OPS_DIR / "models"


@dataclass(frozen=True)
class OpsPaths:
    log_dir: Path
    output_dir: Path
    report_dir: Path
    models_dir: Path


def _resolve(base: Path, default: Path, raw: str | Path | None) -> Path:
    if raw is None or str(raw).strip() == "":
        return default
    path = Path(raw)
    return path if path.is_absolute() else (base / path)


def ops_paths(cfg: dict | None = None) -> OpsPaths:
    """Resolve log/output/report/models dirs from ``settings["ops"]``."""
    ops_cfg = dict((cfg or {}).get("ops") or {})
    return OpsPaths(
        log_dir=_resolve(OPS_DIR, LOG_DIR_DEFAULT, ops_cfg.get("log_dir", "log")),
        output_dir=_resolve(OPS_DIR, OUTPUT_DIR_DEFAULT, ops_cfg.get("output_dir", "output")),
        report_dir=_resolve(OPS_DIR, REPORT_DIR_DEFAULT, ops_cfg.get("report_dir", "report")),
        models_dir=_resolve(OPS_DIR, MODELS_DIR_DEFAULT, ops_cfg.get("models_dir")),
    )


def models_dir_from_cfg(cfg: dict | None = None) -> Path:
    """Back-compat: ``moe.models_dir`` wins, else ``ops.models_dir``, else ``ops/models``."""
    cfg = cfg or {}
    raw = (cfg.get("moe") or {}).get("models_dir")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (OPS_DIR.parent / path)
    return ops_paths(cfg).models_dir


__all__ = [
    "OPS_DIR",
    "LOG_DIR_DEFAULT",
    "OUTPUT_DIR_DEFAULT",
    "REPORT_DIR_DEFAULT",
    "MODELS_DIR_DEFAULT",
    "OpsPaths",
    "ops_paths",
    "models_dir_from_cfg",
]
