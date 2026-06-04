from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources


def run_cli(*args: str, tmp_path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["PYTHONPATH"] = os.getcwd()
    return subprocess.run(
        [sys.executable, "-m", "qualisr.cli", *args],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=tmp_path,
    )


def test_public_api_imports() -> None:
    import qualisr
    from qualisr import (
        PipelineOptions,
        load_config,
        load_pipeline_config,
        run_pipeline,
        run_regressor_experiment,
    )

    assert "load_config" in qualisr.__all__
    assert PipelineOptions.__module__ == "qualisr.pipeline"
    assert load_config.__module__ == "qualisr.api"
    assert load_pipeline_config.__module__ == "qualisr.api"
    assert run_pipeline.__module__ == "qualisr.api"
    assert run_regressor_experiment.__module__ == "qualisr.api"


def test_packaged_configs_are_available() -> None:
    config_root = resources.files("qualisr.configs")
    assert config_root.joinpath("default.json").is_file()
    assert config_root.joinpath("pipeline.json").is_file()


def test_public_api_loads_packaged_configs() -> None:
    from qualisr import load_pipeline_config, load_regressor_config

    regressor_cfg = load_regressor_config()
    pipeline_cfg = load_pipeline_config()
    assert "models" in regressor_cfg
    assert "regressors" in pipeline_cfg
    assert "qualisr/sample_data" in regressor_cfg["paths"]["features_root"].replace("\\", "/")


def test_cli_help(tmp_path) -> None:
    result = run_cli("--help", tmp_path=tmp_path)
    assert "run-regressors" in result.stdout


def test_module_dispatcher_help(tmp_path) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, "-m", "qualisr", "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    assert "run-regressors" in result.stdout


def test_regressor_help(tmp_path) -> None:
    result = run_cli("run-regressors", "--help", tmp_path=tmp_path)
    assert "qualisr-run-regressors" in result.stdout


def test_feature_help_does_not_require_feature_extras(tmp_path) -> None:
    result = run_cli("extract-features", "--help", tmp_path=tmp_path)
    assert "qualisr-extract-features" in result.stdout or "Compute FR/NR" in result.stdout


def test_regressors_default_uses_packaged_sample_data(tmp_path) -> None:
    result = run_cli(
        "run-regressors",
        "--no-plots",
        "--plots-root",
        str(tmp_path / "plots"),
        tmp_path=tmp_path,
    )
    assert "Saved results to" in result.stdout
    assert (tmp_path / "plots" / "baseline@pca5" / "correlations" / "correlations.csv").is_file()
