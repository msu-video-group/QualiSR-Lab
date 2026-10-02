"""Shared helpers for file- and directory-based config execution."""

from __future__ import annotations

from pathlib import Path


def discover_config_paths(path: Path) -> list[Path]:
    """Return one config file or recursively discovered JSON configs in stable order."""

    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Config path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Config path must be a file or directory: {path}")

    configs = sorted(
        (item for item in path.rglob("*.json") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not configs:
        raise ValueError(f"Config directory contains no JSON files: {path}")
    return configs
