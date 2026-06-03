"""Public Python API for QualiSR."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "extract_regressor_config",
    "load_config",
    "run_experiment",
    "run_pipeline",
]


def __getattr__(name: str) -> Any:
    if name == "run_pipeline":
        return import_module("qualisr.pipeline").run_pipeline
    if name in {"extract_regressor_config", "load_config", "run_experiment"}:
        return getattr(import_module("qualisr.regressors"), name)
    raise AttributeError(f"module 'qualisr' has no attribute {name!r}")
