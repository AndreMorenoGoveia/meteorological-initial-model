"""Configuration loading.

Configs are YAML files. A config may set ``defaults: <path>`` (relative to its own
location) to inherit from a base config; values in the child override values in the parent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _load_one(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Top-level config in {path} must be a mapping, got {type(cfg)}")
    return cfg


def load_config(path: str | Path) -> DictConfig:
    """Load a config file, recursively resolving the ``defaults:`` chain.

    Parent values are merged first, then overridden by the child.
    """
    path = Path(path).resolve()
    cfg = _load_one(path)
    parent_path = cfg.pop("defaults", None)
    if parent_path is None:
        return cfg
    parent = load_config(path.parent / str(parent_path))
    return OmegaConf.merge(parent, cfg)  # type: ignore[return-value]


def to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
