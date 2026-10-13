"""Ops root for the MoE strategy: filesystem-facing helpers.

Layout (all fixed names, anchored at ``strategy_system/strategy/moe/ops/``)::

    ops/
      log/       strategy.log, features.log, ood.log, gp.log,
                 targets.log, barrier.log, policy.log, export.log
      output/    features.csv, ood.csv, gp.csv, targets.csv,
                 barrier.csv, policy.csv, frame.csv  (overwrite each run)
      report/    features.txt, ood.txt, gp.txt, targets.txt,
                 barrier.txt, policy.txt, run.txt   (what was printed)
      models/    regime/, latent_*/, manifest.json (Rust bundle, via export.py)
      paths.py   single place that resolves the four dirs + models dir
      logging.py per-file loggers (no timestamps in the filename)
      output.py  minimal overwrite CSV writer
      report.py  StageReport container + writer (print == file content)
      settings.py load_settings()
      export.py  model-bundle export

Nothing here fits a model; it only moves bytes to disk.

Import either the package (``from .ops import ops_paths``) or the
submodule (``from .ops.paths import ops_paths``); both work.
"""

from __future__ import annotations

from . import export as export
from . import logging as logging
from . import output as output
from . import paths as paths
from . import report as report
from . import settings as settings
from .export import (
    GP_FEATURES_FILE,
    MANIFEST_FILE,
    METADATA_FILE,
    REGIME_DIR,
    REGIME_METADATA_FILE,
    REGIME_MODEL_FILE,
    TB_MODEL_FILE,
    build_gp_features_payload,
    build_regime_metadata,
    export_gp_features,
    export_latent,
    export_manifest,
    export_regime_bundle,
    export_tb_onnx,
    latent_dir_name,
    sha256_file,
    verify_regime_contract,
    write_json,
)
from .logging import STAGES, get_logger, log_header
from .output import (
    FRAME_CSV_ROWS,
    OUTPUT_FILES,
    output_enabled,
    write_frame_head,
    write_output,
)
from .paths import (
    LOG_DIR_DEFAULT,
    MODELS_DIR_DEFAULT,
    OPS_DIR,
    OUTPUT_DIR_DEFAULT,
    REPORT_DIR_DEFAULT,
    OpsPaths,
    models_dir_from_cfg,
    ops_paths,
)
from .report import StageReport, emit_report, emit_run_report
from .settings import DEFAULT_SETTINGS_PATH, load_settings
