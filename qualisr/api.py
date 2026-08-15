"""Stable public API helpers for using QualiSR as a Python package."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from qualisr.pipeline import PipelineOptions, config_base_dir

__all__ = [
    "PipelineOptions",
    "load_config",
    "load_pipeline_config",
    "load_regressor_config",
    "run_pipeline",
    "run_pipeline_config",
    "run_regressor_experiment",
]


def load_regressor_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a regressor config or the packaged sample experiment."""
    from qualisr.regressors import load_config

    return load_config(Path(path) if path is not None else None)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Backward-compatible alias for :func:`load_regressor_config`."""
    return load_regressor_config(path)


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a unified pipeline config from a path or from the packaged default."""
    from qualisr.pipeline import load_pipeline_config as _load_pipeline_config

    return _load_pipeline_config(Path(path) if path is not None else None)


def run_regressor_experiment(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    make_plots: bool = True,
    overrides: dict[str, Any] | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a regressor experiment from a config object or config path.

    Parameters
    ----------
    config:
        Already loaded unified pipeline or regressor config.
    config_path:
        Optional path to a unified pipeline or regressor JSON config. If both
        config arguments are omitted, the packaged sample experiment is used.
    make_plots:
        Whether to generate configured plots.
    overrides:
        Optional nested config updates applied before running.
    """
    from qualisr.datasets import load_datasets
    from qualisr.regressors import (
        deep_update,
        extract_regressor_config,
        load_config,
        load_config_with_samples,
        resolve_regressor_config_paths,
        run_experiment,
    )

    if config is not None and config_path is not None:
        raise ValueError("Specify config or config_path, not both")
    if config_path is not None:
        if samples is None:
            cfg, configured_samples = load_config_with_samples(Path(config_path))
            samples = configured_samples
        else:
            cfg = load_config(Path(config_path))
    elif config is not None:
        pipeline_cfg = deepcopy(config)
        config_root = Path.cwd() if base_dir is None else Path(base_dir)
        cfg = resolve_regressor_config_paths(
            extract_regressor_config(pipeline_cfg),
            config_root,
        )
        if samples is None:
            dataset_entries = pipeline_cfg.get("datasets")
            if not isinstance(dataset_entries, list):
                raise ValueError("Config without explicit samples must define datasets as a list")
            samples = load_datasets(dataset_entries, base_dir=config_root)
    else:
        cfg, samples = load_config_with_samples()
    if overrides:
        cfg = deep_update(cfg, overrides)
    return run_experiment(cfg, samples=samples, make_plots=make_plots)


def run_pipeline_config(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    options: PipelineOptions | None = None,
    only_section: list[str] | None = None,
    skip_section: list[str] | None = None,
    experiment_name: str | None = None,
    plots_root: str | None = None,
    no_plots: bool = False,
    save_svg: bool = False,
    samples: list[dict[str, Any]] | None = None,
) -> None:
    """Run the unified pipeline from a config object or config path."""
    from qualisr.pipeline import run_pipeline

    path = Path(config_path) if config_path is not None else None
    cfg = deepcopy(config) if config is not None else load_pipeline_config(path)
    run_options = options or PipelineOptions(
        only_section=only_section,
        skip_section=skip_section,
        experiment_name=experiment_name,
        plots_root=plots_root,
        no_plots=no_plots,
        save_svg=save_svg,
    )
    run_pipeline(
        cfg,
        base_dir=base_dir or config_base_dir(path),
        options=run_options,
        samples=samples,
    )


def run_pipeline(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    options: PipelineOptions | None = None,
    **kwargs: Any,
) -> None:
    """Convenience alias for :func:`run_pipeline_config`."""
    run_pipeline_config(
        config=config,
        config_path=config_path,
        base_dir=base_dir,
        options=options,
        **kwargs,
    )
