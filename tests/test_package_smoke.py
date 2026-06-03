from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources


def run_cli(*args: str, tmp_path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    return subprocess.run(
        [sys.executable, "-m", "qualisr.cli", *args],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def test_public_api_imports() -> None:
    import qualisr
    from qualisr import load_config, run_experiment, run_pipeline

    assert "load_config" in qualisr.__all__
    assert load_config.__module__ == "qualisr.regressors"
    assert run_experiment.__module__ == "qualisr.regressors"
    assert run_pipeline.__module__ == "qualisr.pipeline"


def test_packaged_configs_are_available() -> None:
    config_root = resources.files("qualisr.configs")
    assert config_root.joinpath("default.json").is_file()
    assert config_root.joinpath("pipeline.json").is_file()


def test_cli_help(tmp_path) -> None:
    result = run_cli("--help", tmp_path=tmp_path)
    assert "run-regressors" in result.stdout


def test_regressor_help(tmp_path) -> None:
    result = run_cli("run-regressors", "--help", tmp_path=tmp_path)
    assert "qualisr-run-regressors" in result.stdout


def test_feature_help_does_not_require_feature_extras(tmp_path) -> None:
    result = run_cli("extract-features", "--help", tmp_path=tmp_path)
    assert "qualisr-extract-features" in result.stdout or "Compute FR/NR" in result.stdout
