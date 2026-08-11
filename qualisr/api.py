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
    """Load a regressor config from a path or from the packaged default."""
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
    make_plots: bool = True,
    overrides: dict[str, Any] | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a regressor experiment from a config object or config path.

    Parameters
    ----------
    config:
        Already loaded regressor config. If omitted, ``config_path`` or the
        packaged default config is used.
    config_path:
        Optional path to a regressor or unified pipeline JSON config.
    make_plots:
        Whether to generate configured plots.
    overrides:
        Optional nested config updates applied before running.
    """
    from qualisr.regressors import deep_update, load_config_with_samples, run_experiment

    if config is not None:
        cfg = deepcopy(config)
    elif config_path is not None:
        cfg, configured_samples = load_config_with_samples(Path(config_path))
        if samples is None:
            samples = configured_samples
    else:
        cfg = load_regressor_config()
    if overrides:
        cfg = deep_update(cfg, overrides)
    return run_experiment(cfg, make_plots=make_plots, samples=samples)


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
