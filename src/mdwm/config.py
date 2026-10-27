from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not contain a YAML mapping")
    return cfg


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_override_value(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        lowered = parsed.lower()
        if lowered in {"none", "null"}:
            return None
        if lowered in {"true", "false"}:
            return lowered == "true"
    return parsed


def ensure_dirs(cfg: dict[str, Any]) -> None:
    if "data" in cfg and "processed_dir" in cfg["data"]:
        Path(cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    if "outputs" in cfg:
        if "root" in cfg["outputs"]:
            Path(cfg["outputs"]["root"]).mkdir(parents=True, exist_ok=True)
        if "checkpoint_dir" in cfg["outputs"]:
            Path(cfg["outputs"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
