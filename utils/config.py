from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_root_from_config(config: dict[str, Any]) -> Path:
    root_value = config.get("project_root")
    if root_value:
        return Path(root_value).expanduser().resolve()
    config_path = config.get("_config_path")
    if config_path:
        return Path(config_path).resolve().parents[1]
    return Path.cwd().resolve()


def resolve_path(path_value: str | Path, project_root: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path(project_root).expanduser().resolve() / path


def apply_common_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: value for key, value in overrides.items() if value is not None}
    return deep_update(config, cleaned)

