"""settings.yaml loading for the MoE pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"


def load_settings(path: str | Path | None = None) -> dict:
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    with settings_path.open("r", encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    if not isinstance(settings, dict):
        raise ValueError(f"{settings_path} must contain a YAML mapping.")
    return settings