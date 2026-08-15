from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources
from pathlib import Path


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
        load_dataset,
        load_datasets,
        load_pipeline_config,
        run_pipeline,
        run_regressor_experiment,
    )

    assert "load_config" in qualisr.__all__
    assert "load_dataset" in qualisr.__all__
    assert "load_datasets" in qualisr.__all__
    assert PipelineOptions.__module__ == "qualisr.pipeline"
    assert load_config.__module__ == "qualisr.api"
    assert load_dataset.__module__ == "qualisr.datasets"
    assert load_datasets.__module__ == "qualisr.datasets"
    assert load_pipeline_config.__module__ == "qualisr.api"
    assert run_pipeline.__module__ == "qualisr.api"
    assert run_regressor_experiment.__module__ == "qualisr.api"


def test_packaged_configs_are_available() -> None:
    config_root = resources.files("qualisr.configs")
    assert config_root.joinpath("default.json").is_file()
    assert config_root.joinpath("pipeline.json").is_file()
    sample_root = resources.files("qualisr.sample_data")
    assert sample_root.joinpath("features", "fr.csv").is_file()
    assert sample_root.joinpath("features", "nr.csv").is_file()
    assert sample_root.joinpath("features", "pca", "vgg_pca5.csv").is_file()
    assert sample_root.joinpath("features", "pca", "resnet_pca5.csv").is_file()
    assert sample_root.joinpath("scores", "labels.csv").is_file()


def test_public_api_loads_packaged_configs() -> None:
    from qualisr import load_pipeline_config, load_regressor_config

    regressor_cfg = load_regressor_config()
    pipeline_cfg = load_pipeline_config()
    assert "models" in regressor_cfg
    assert "regressors" in pipeline_cfg
    dataset_cfg = pipeline_cfg["datasets"][0]
    assert dataset_cfg["name"] == "QualiSR-Set120"
    assert dataset_cfg["features_root"] == "features"
    assert dataset_cfg["regressors"] == {"train": True, "validate": True, "test_size": 0.2}
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
    assert "--config CONFIG" in result.stdout
    assert "required" not in result.stderr


def test_feature_help_does_not_require_feature_extras(tmp_path) -> None:
    result = run_cli("extract-features", "--help", tmp_path=tmp_path)
    assert "qualisr-extract-features" in result.stdout or "Compute FR/NR" in result.stdout


def test_embedding_difference_help(tmp_path) -> None:
    result = run_cli("embedding-difference", "--help", tmp_path=tmp_path)
    assert "SR-minus-reference" in result.stdout


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


def test_explicit_config_paths_are_resolved_from_config_directory(tmp_path, monkeypatch) -> None:
    from qualisr.regressors import load_config

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    cfg = load_config(repo_root / "configs" / "pipeline.json")

    assert Path(cfg["paths"]["plots_root"]) == repo_root / "plots"


def test_missing_explicit_config_does_not_fall_back_to_packaged_default(tmp_path, monkeypatch) -> None:
    from qualisr.regressors import load_config

    monkeypatch.chdir(tmp_path)

    try:
        load_config(Path("configs/pipeline.json"))
    except FileNotFoundError:
        return

    raise AssertionError("missing explicit config unexpectedly loaded packaged default")


def test_regressors_load_samples_from_unified_pipeline_config() -> None:
    from qualisr.regressors import load_config_with_samples

    repo_root = Path(__file__).resolve().parents[1]
    cfg, samples = load_config_with_samples(repo_root / "configs" / "pipeline.json")

    assert len(samples) == 120
    assert {sample["regressors"]["test_size"] for sample in samples} == {0.2}
    assert {sample["features_root"] for sample in samples} == {str(repo_root / "features")}
    assert "dataset" not in cfg
    assert "features_root" not in cfg["paths"]
