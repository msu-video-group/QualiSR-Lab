"""Public commands, installation metadata and packaged resources."""

from __future__ import annotations

import csv
import os
import runpy
import subprocess
import sys
from importlib import metadata, resources
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


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
    assert pipeline_cfg["regressors"]["config"]["cross_validation"] == {
        "enabled": False,
        "n_splits": 5,
    }
    assert pipeline_cfg["regressors"]["config"]["split_seed"] == 42
    assert pipeline_cfg["regressors"]["config"]["imputation"] == {
        "enabled": True,
        "strategy": "median",
    }
    assert regressor_cfg["imputation"] == {"enabled": True, "strategy": "median"}
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
    output_dir = tmp_path / "plots" / "baseline@pca5"
    assert (output_dir / "correlations" / "correlations.csv").is_file()
    console_output = (output_dir / "log.txt").read_text(encoding="utf-8")
    assert console_output.count("Saved results to") == 1


def test_regressor_console_output_is_duplicated(tmp_path, capsys) -> None:
    from qualisr.regressors import duplicate_regressor_console_output

    cfg = {
        "experiment_name": "logging-test",
        "paths": {"plots_root": str(tmp_path)},
    }

    with duplicate_regressor_console_output(cfg) as log_path:
        print("standard output")
        print("standard error", file=sys.stderr)

    captured = capsys.readouterr()
    saved = log_path.read_text(encoding="utf-8")
    assert "standard output" in captured.out
    assert "standard error" in captured.err
    assert "standard output" in saved
    assert "standard error" in saved


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


def test_regressors_load_samples_from_unified_pipeline_config(tmp_path) -> None:
    from PIL import Image

    from qualisr.regressors import load_config_with_samples

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    labels_path = dataset_root / "labels.csv"
    labels_path.write_bytes((repo_root / "dataset/labels.csv").read_bytes())
    for row in csv.DictReader(labels_path.open(encoding="utf-8")):
        for relative in (row["image"], f'hr/{row["test_case"]}.png', f'lr/{row["test_case"]}.png'):
            path = dataset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(path)
    config_path = tmp_path / "configs/pipeline.json"
    config_path.parent.mkdir()
    config_path.write_bytes((repo_root / "configs/pipeline.json").read_bytes())
    cfg, samples = load_config_with_samples(config_path)

    assert len(samples) == 120
    assert {sample["regressors"]["test_size"] for sample in samples} == {0.2}
    assert {sample["features_root"] for sample in samples} == {str(tmp_path / "features")}
    assert "dataset" not in cfg
    assert "features_root" not in cfg["paths"]


def test_config_directory_discovery_is_recursive_and_sorted(tmp_path) -> None:
    from qualisr.config_paths import discover_config_paths

    config_root = tmp_path / "suite"
    nested = config_root / "nested"
    nested.mkdir(parents=True)
    (config_root / "z.json").write_text("{}", encoding="utf-8")
    (nested / "a.json").write_text("{}", encoding="utf-8")
    (config_root / "README.md").write_text("ignored", encoding="utf-8")

    discovered = discover_config_paths(config_root)

    assert [path.relative_to(config_root).as_posix() for path in discovered] == [
        "nested/a.json",
        "z.json",
    ]


def test_pipeline_main_runs_config_directory_in_order(tmp_path, monkeypatch) -> None:
    import qualisr.pipeline as pipeline

    config_root = tmp_path / "suite"
    nested = config_root / "nested"
    nested.mkdir(parents=True)
    (config_root / "z.json").write_text('{"name": "second"}', encoding="utf-8")
    (nested / "a.json").write_text('{"name": "first"}', encoding="utf-8")
    runs = []

    def fake_run_pipeline(cfg, base_dir, options) -> None:
        runs.append((cfg["name"], Path(base_dir), options.no_plots))

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    pipeline.main(["--config", str(config_root), "--no-plots"])

    assert runs == [
        ("first", nested, True),
        ("second", config_root, True),
    ]


def test_regressor_main_runs_config_directory_in_order(tmp_path, monkeypatch) -> None:
    import qualisr.regressors as regressors

    class DummyResults:
        def to_string(self, index=False) -> str:
            return "results"

    config_root = tmp_path / "suite"
    nested = config_root / "nested"
    nested.mkdir(parents=True)
    (config_root / "z.json").write_text("{}", encoding="utf-8")
    (nested / "a.json").write_text("{}", encoding="utf-8")
    loaded = []
    runs = []

    def fake_load_config_with_samples(path):
        loaded.append(path.relative_to(config_root).as_posix())
        return {
            "experiment_name": path.stem,
            "paths": {"plots_root": str(tmp_path / "plots")},
        }, []

    def fake_run_experiment(cfg, make_plots, samples):
        runs.append((cfg["experiment_name"], make_plots, samples))
        return {"output_dir": Path("plots") / cfg["experiment_name"], "results": DummyResults()}

    monkeypatch.setattr(regressors, "load_config_with_samples", fake_load_config_with_samples)
    monkeypatch.setattr(regressors, "run_experiment", fake_run_experiment)

    regressors.main(["--config", str(config_root), "--no-plots"])

    assert loaded == ["nested/a.json", "z.json"]
    assert runs == [("a", False, []), ("z", False, [])]
    assert (tmp_path / "plots" / "a" / "log.txt").is_file()
    assert (tmp_path / "plots" / "z" / "log.txt").is_file()


def test_packaged_resources_match_repository() -> None:
    resource_pairs = runpy.run_path(str(ROOT / "scripts/sync_package_data.py"))["resource_pairs"]
    package_root = resources.files("qualisr")
    for source, target in resource_pairs(ROOT):
        relative = target.relative_to(ROOT / "qualisr")
        assert package_root.joinpath(*relative.parts).read_bytes() == source.read_bytes(), str(relative)


def test_packaged_configs_have_no_removed_sources() -> None:
    expected = {path.relative_to(ROOT / "configs") for path in (ROOT / "configs").rglob("*")
                if path.is_file() and path.suffix in {".json", ".md"}}
    actual = {path.relative_to(ROOT / "qualisr/configs") for path in (ROOT / "qualisr/configs").rglob("*")
              if path.is_file() and path.suffix in {".json", ".md"}}
    assert actual == expected


def test_distribution_version_and_dependencies() -> None:
    import tomllib

    pins = {
        canonicalize_name(requirement.name): str(requirement.specifier)
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for requirement in [Requirement(line)]
    }
    project = metadata.metadata("qualisr-lab")
    assert project["Version"] == tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    python_versions = SpecifierSet(project["Requires-Python"])
    assert python_versions == SpecifierSet(">=3.12,<3.14")
    assert "3.12" in python_versions
    assert "3.13" in python_versions
    assert "3.14" not in python_versions
    all_dependencies = set()
    for line in project.get_all("Requires-Dist", []):
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        assert str(requirement.specifier) == pins[name], line
        if requirement.marker is None or requirement.marker.evaluate({"extra": "all"}):
            all_dependencies.add(name)
    assert all_dependencies == set(pins)


def test_paper_assets_are_available_after_installation():
    import csv
    import hashlib
    import json

    experiments = resources.files("experiments")
    flags = list(csv.DictReader(experiments.joinpath("inputs/set120_controls.csv").read_text().splitlines()))
    assert len(flags) == 120
    assert sum(row["is_control_item"] == "True" for row in flags) == 3
    index = json.loads(experiments.joinpath("artifacts.json").read_text())
    configs = resources.files("qualisr.configs")
    assert experiments.joinpath(index["protocol"]).read_bytes() == configs.joinpath("paper.json").read_bytes()
    for group in index["results"].values():
        for item in group:
            assert hashlib.sha256(experiments.joinpath(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    paper = json.loads(configs.joinpath("paper.json").read_text())
    feature_config = json.loads(configs.joinpath("paper_features.json").read_text())
    assert paper["datasets"] == feature_config["datasets"]
    assert set(feature_config["features"]["groups"]) == {"fr", "nr", "vgg", "ref_vgg", "resnet", "ref_resnet", "siglip"}


def test_installed_grouped_runner_reads_packaged_evaluation(monkeypatch, tmp_path):
    from experiments import run_grouped

    calls = []
    monkeypatch.setattr(run_grouped, "run_grouped_experiments",
                        lambda config, base, output, options: calls.append(config["evaluation"]))
    entry, = metadata.distribution("qualisr-lab").entry_points.select(name="qualisr-run-experiments")
    with resources.as_file(resources.files("qualisr.configs").joinpath("paper.json")) as config:
        entry.load()(["--config", str(config), "--output", str(tmp_path), "--audit-only"])
    settings, = calls
    assert (settings["outer_folds"], settings["inner_folds"], settings["split_seeds"]) == (5, 3, [42, 43, 44])
