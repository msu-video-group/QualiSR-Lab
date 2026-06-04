"""Public Python API for QualiSR."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PipelineOptions",
    "extract_regressor_config",
    "load_config",
    "load_pipeline_config",
    "load_regressor_config",
    "run_experiment",
    "run_pipeline",
    "run_pipeline_config",
    "run_regressor_experiment",
]


def __getattr__(name: str) -> Any:
    if name in {
        "PipelineOptions",
        "load_config",
        "load_pipeline_config",
        "load_regressor_config",
        "run_pipeline",
        "run_pipeline_config",
        "run_regressor_experiment",
    }:
        return getattr(import_module("qualisr.api"), name)
    if name in {"extract_regressor_config", "run_experiment"}:
        return getattr(import_module("qualisr.regressors"), name)
    raise AttributeError(f"module 'qualisr' has no attribute {name!r}")
