"""StageReport: the individual report each ``run_*`` returns.

``text`` is exactly what gets printed to stdout AND written to
``ops/report/<stage>.txt`` — one source of truth, no drift between the
console and the file. ``summary`` holds the small scalar dict for the
combined ``run.txt``; ``table`` holds the full frame (latent counts,
metalabel row, GP rows, policy rows) that also lands in ``ops/output/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .logging import get_logger
from .paths import ops_paths


@dataclass
class StageReport:
    stage: str                       # features | ood | targets | gp | frame | barrier | policy
    title: str                       # header line printed + stored
    summary: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)
    text: str = ""                   # full printable body (built by build_text)

    def build_text(self) -> str:
        lines = [f"=== {self.title} ==="]
        for key, value in self.summary.items():
            lines.append(f"{key}: {value}")
        if self.table is not None and not self.table.empty:
            lines.append("")
            lines.append(self.table.to_string(index=False))
        elif self.table is not None:
            lines.append("")
            lines.append("(empty)")
        for note in self.notes:
            lines.append(f"note: {note}")
        self.text = "\n".join(lines)
        return self.text


def emit_report(report: StageReport, cfg: dict | None = None, *, verbose: bool = True) -> StageReport:
    """Print ``report.text`` (= write it to ``ops/report/<stage>.txt``)."""
    if not report.text:
        report.build_text()
    logger = get_logger(report.stage, cfg)
    if verbose:
        logger.info("\n%s", report.text)
    else:
        # still record to the file handler without echoing
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(report.text + "\n")
                handler.flush()
    report_dir = ops_paths(cfg).report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{report.stage}.txt").write_text(report.text + "\n", encoding="utf-8")
    return report


def emit_run_report(reports: dict[str, StageReport], cfg: dict | None = None) -> Path:
    """Concatenate every stage text into ``ops/report/run.txt``; return its path."""
    parts = []
    for stage in ("features", "ood", "targets", "gp", "frame", "barrier", "policy"):
        rep = reports.get(stage)
        if rep is not None:
            if not rep.text:
                rep.build_text()
            parts.append(rep.text)
    run_text = "\n\n".join(parts) + "\n"
    report_dir = ops_paths(cfg).report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "run.txt"
    path.write_text(run_text, encoding="utf-8")
    get_logger("strategy", cfg).info("\n%s", run_text)
    return path


__all__ = ["StageReport", "emit_report", "emit_run_report"]
