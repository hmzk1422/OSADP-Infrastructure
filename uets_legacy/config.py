"""Legacy compatibility shim for the strategy configuration.

The canonical, source-of-truth parameters now live at ``config/strategy.py``
(repo root) so the toolchain does NOT depend on the legacy ``main/`` package.
This file only re-exports those values so older modules that still do
``from main.config import *`` (e.g. ``main/execution.py``, ``research/*.py``)
keep working unchanged.

DO NOT edit parameters here - edit ``config/strategy.py`` instead.
"""
from config.strategy import *  # noqa: F401,F403  (constants + helpers)
