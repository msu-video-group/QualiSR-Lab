from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from qualisr.datasets import load_dataset, load_datasets


def write_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 128, 192)).save(path)


def write_qualisr_dataset(root: Path, method: str = "PASD") -> None:
    rows = []
    for test_case, score in (("0001", 2.0), ("0002", 4.0)):
        for subdir in ("hr", "lr"):
            write_image(root / subdir / f"{test_case}.png")
        write_image(root / "sr" / method / f"{test_case}.png")
        rows.append(
            {
                "test_case": test_case,
                "method": method,
                "score": score,
                "image": f"sr/{method}/{test_case}.png",
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_qualisr_samples_have_stable_ids_and_default_heatmaps(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_qualisr_dataset(root)

    samples = load_dataset(
        {
            "name": "QualiSR-Set120",
            "root": str(root),
            "features_root": str(tmp_path / "features"),
            "regressors": {"train": False, "validate": False},
        }
    )

    assert [sample["score"] for sample in samples] == [0.0, 1.0]
    assert samples[0]["sample_id"] == "QualiSR-Set120/PASD/0001.npy.gz"
    assert samples[0]["score_type"] == "mos"
    assert samples[0]["heatmap_path"] == str(root / "heatmaps" / "PASD" / "0001.npy.gz")
    assert Path(samples[0]["sr_path"]).is_absolute()


def test_custom_file_parser_and_multiple_datasets(tmp_path: Path) -> None:
    builtin_root = tmp_path / "builtin"
    custom_root = tmp_path / "custom"
    write_qualisr_dataset(builtin_root)
    write_image(custom_root / "lr.png")
    write_image(custom_root / "sr.png")
    parser_path = tmp_path / "parser.py"
    parser_path.write_text(
        """
from pathlib import Path

def parse_dataset(root, score=0.25):
    root = Path(root)
    return [{
        "dataset": "ignored-parser-name",
        "test_case": "case",
        "method": "custom-method",
        "rel_path": "sr.png",
        "lr_path": root / "lr.png",
        "sr_path": root / "sr.png",
        "score": score,
    }]
""",
        encoding="utf-8",
    )

    samples = load_datasets(
        [
            {
                "name": "QualiSR-Set120",
                "root": str(builtin_root),
                "features_root": str(tmp_path / "builtin-features"),
                "regressors": {"train": False, "validate": False},
            },
            {
                "name": "custom",
                "root": str(custom_root),
                "features_root": str(tmp_path / "custom-features"),
                "regressors": {"train": False, "validate": False},
                "parser": {"path": str(parser_path), "function": "parse_dataset"},
                "kwargs": {"score": 0.75},
            },
        ]
    )

    assert len(samples) == 3
    assert samples[-1]["dataset"] == "custom"
    assert samples[-1]["score"] == 0.75
    assert samples[-1]["sample_id"] == "custom/custom-method/sr.npy.gz"


def test_directory_dataset_uses_explicit_directories(tmp_path: Path) -> None:
    root = tmp_path / "layout"
    hr_dir = tmp_path / "images" / "hr"
    lr_dir = tmp_path / "images" / "lr"
    sr_dir = tmp_path / "images" / "outputs"
    heatmap_dir = tmp_path / "images" / "masks"
    for directory in (hr_dir, lr_dir, sr_dir):
        write_image(directory / "case.png")
    heatmap_dir.mkdir(parents=True)
    with gzip.open(heatmap_dir / "case.npy.gz", "wb") as handle:
        np.save(handle, np.ones((2, 2), dtype=np.float32))
    root.mkdir(parents=True)
    pd.DataFrame([{"test_case": "case", "method": "method-a", "score": 0.5, "image": "case.png"}]).to_csv(
        root / "labels.csv", index=False
    )

    samples = load_dataset(
        {
            "name": "directory-data",
            "root": str(root),
            "features_root": str(tmp_path / "directory-features"),
            "regressors": {"train": False, "validate": False},
            "labels": "labels.csv",
            "directories": {
                "hr": str(hr_dir),
                "lr": str(lr_dir),
                "sr": {"method-a": str(sr_dir)},
                "heatmaps": {"method-a": str(heatmap_dir)},
            },
        }
    )

    assert len(samples) == 1
    assert samples[0]["sr_path"] == str((sr_dir / "case.png").resolve())
    assert samples[0]["heatmap_path"] == str((heatmap_dir / "case.npy.gz").resolve())


def test_realsrq_heatmaps_use_flat_heatmaps_directory(tmp_path: Path, monkeypatch) -> None:
    import scipy.io

    from qualisr.datasets import parse_realsrq

    scores = np.arange(60 * 27, dtype=float).reshape(60, 27)
    monkeypatch.setattr(scipy.io, "loadmat", lambda _: {"score_matrix": scores})

    samples = parse_realsrq(str(tmp_path), include_refs=False)

    assert {sample["score_type"] for sample in samples} == {"bradley_terry"}
    assert samples[0]["heatmap_path"] == str(
        (tmp_path / "heatmaps" / "Buildings_001_LR2_AIS.npy.gz").resolve()
    )


def test_sample_aware_statistics_uses_sample_id(tmp_path: Path) -> None:
    from qualisr.statistics import main as statistics_main

    heatmap_path = tmp_path / "case.npy.gz"
    with gzip.open(heatmap_path, "wb") as handle:
        np.save(handle, np.arange(4, dtype=np.float32).reshape(2, 2))
    output = tmp_path / "stats.csv"

    statistics_main(
        ["--output", str(output), "--no-progress"],
        samples=[{"sample_id": "data/method/case.npy.gz", "heatmap_path": str(heatmap_path)}],
    )

    frame = pd.read_csv(output)
    assert frame.loc[0, "sample_id"] == "data/method/case.npy.gz"
    assert frame.loc[0, "max"] == 3.0


def test_sample_aware_bicubic_generation_updates_reference_paths(tmp_path: Path) -> None:
    from qualisr.references import generate_references

    root = tmp_path / "dataset"
    write_image(root / "lr.png", size=(4, 4))
    write_image(root / "sr.png", size=(8, 8))
    sample = {
        "dataset": "data",
        "dataset_root": str(root),
        "method": "method-a",
        "lr_path": str(root / "lr.png"),
        "sr_path": str(root / "sr.png"),
        "ref_paths": {},
    }

    stats = generate_references([sample], {"refs": ["bicubic"], "no_progress": True})

    assert stats["bicubic_ok"] == 1
    assert Path(sample["ref_paths"]["bicubic"]).is_file()


def regressor_test_config() -> dict:
    return {
        "seed": 42,
        "split_seed": 7,
        "scale_features": False,
        "features": {
            "pca_n": 0,
            "include": ["nr"],
            "include_stats": False,
            "stats_columns": [],
            "fr_refs": [],
            "exclude_columns": [],
            "feature_files": {"nr": "{features_root}/nr.csv"},
        },
    }


def regressor_samples(
    dataset: str,
    features_root: Path,
    usage: dict,
    count: int,
) -> list[dict]:
    samples = []
    feature_rows = []
    for index in range(count):
        sample_id = f"{dataset}/method/{index}.npy.gz"
        samples.append(
            {
                "dataset": dataset,
                "features_root": str(features_root),
                "regressors": usage,
                "sample_id": sample_id,
                "test_case": str(index),
                "score": index / max(count - 1, 1),
            }
        )
        feature_rows.append({"sample_id": sample_id, "quality": float(index)})
    features_root.mkdir(parents=True)
    pd.DataFrame(feature_rows).to_csv(features_root / "nr.csv", index=False)
    return samples


def test_regressors_use_multiple_dataset_feature_roots_and_whole_validation_dataset(
    tmp_path: Path,
) -> None:
    from qualisr.regressors import build_dataset, split_dataset

    train_samples = regressor_samples(
        "train-data",
        tmp_path / "train-features",
        {"train": True, "validate": True, "test_size": 0},
        4,
    )
    validation_samples = regressor_samples(
        "validation-data",
        tmp_path / "validation-features",
        {"train": False, "validate": True},
        3,
    )
    samples = train_samples + validation_samples

    dataset = build_dataset(regressor_test_config(), samples)
    X_train, X_validation, y_train, y_validation = split_dataset(
        dataset,
        regressor_test_config(),
        samples,
    )

    assert len(X_train) == len(y_train) == 4
    assert len(X_validation) == len(y_validation) == 3
    assert set(dataset.loc[X_train.index, "dataset"]) == {"train-data"}
    assert set(dataset.loc[X_validation.index, "dataset"]) == {"validation-data"}


def test_training_dataset_requires_test_size(tmp_path: Path) -> None:
    from qualisr.regressors import build_dataset, split_dataset

    samples = regressor_samples(
        "train-data",
        tmp_path / "train-features",
        {"train": True, "validate": True},
        4,
    )
    dataset = build_dataset(regressor_test_config(), samples)

    try:
        split_dataset(dataset, regressor_test_config(), samples)
    except ValueError as error:
        assert "must define regressors.test_size" in str(error)
        return
    raise AssertionError("training dataset without test_size was accepted")


def test_training_dataset_test_split_is_grouped(tmp_path: Path) -> None:
    from qualisr.regressors import build_dataset, split_dataset

    samples = regressor_samples(
        "train-data",
        tmp_path / "train-features",
        {"train": True, "validate": True, "test_size": 0.5},
        6,
    )
    dataset = build_dataset(regressor_test_config(), samples)
    X_train, X_validation, _, _ = split_dataset(dataset, regressor_test_config(), samples)

    assert len(X_train) == 3
    assert len(X_validation) == 3
    assert set(X_train.index).isdisjoint(X_validation.index)


def test_training_split_uses_split_seed_and_keeps_gt_groups_together(tmp_path: Path) -> None:
    from qualisr.regressors import build_dataset, split_dataset

    samples = regressor_samples(
        "train-data",
        tmp_path / "train-features",
        {"train": True, "validate": True, "test_size": 0.5},
        8,
    )
    for index, sample in enumerate(samples):
        sample["test_case"] = f"gt-{index // 2}"
    dataset = build_dataset(regressor_test_config(), samples)

    first_cfg = {**regressor_test_config(), "seed": 1, "split_seed": 17}
    second_cfg = {**regressor_test_config(), "seed": 999, "split_seed": 17}
    first_train, first_validation, _, _ = split_dataset(dataset, first_cfg, samples)
    second_train, second_validation, _, _ = split_dataset(dataset, second_cfg, samples)

    assert first_train.index.tolist() == second_train.index.tolist()
    assert first_validation.index.tolist() == second_validation.index.tolist()
    train_groups = set(dataset.loc[first_train.index, "test_case"])
    validation_groups = set(dataset.loc[first_validation.index, "test_case"])
    assert train_groups.isdisjoint(validation_groups)


def test_validation_correlations_pool_mos_and_average_bt_per_gt() -> None:
    from qualisr.regressors import validation_correlations

    metadata = pd.DataFrame(
        {
            "dataset": ["mos-data"] * 4 + ["bt-data"] * 6,
            "test_case": ["gt-a", "gt-a", "gt-b", "gt-b", "gt-1", "gt-1", "gt-1", "gt-2", "gt-2", "gt-2"],
            "score_type": ["mos"] * 4 + ["bradley_terry"] * 6,
        }
    )
    target = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
    prediction = [0, 1, 2, 3, 0, 1, 2, 2, 1, 0]

    plcc, srcc, per_dataset, details = validation_correlations(target, prediction, metadata)

    mos = per_dataset.set_index("dataset").loc["mos-data"]
    bt = per_dataset.set_index("dataset").loc["bt-data"]
    assert mos["aggregation"] == "pooled"
    assert mos["n_groups"] == 1
    assert mos["plcc"] == 1.0
    assert mos["srcc"] == 1.0
    assert bt["aggregation"] == "mean_per_gt"
    assert bt["n_groups"] == 2
    assert np.isclose(bt["plcc"], 0.0)
    assert np.isclose(bt["srcc"], 0.0)
    assert np.isclose(plcc, 0.5)
    assert np.isclose(srcc, 0.5)
    assert len(details[details["dataset"] == "mos-data"]) == 1
    assert len(details[details["dataset"] == "bt-data"]) == 2


def test_grouped_cross_validation_keeps_source_groups_together() -> None:
    from qualisr.regressors import grouped_cross_validation_splits

    dataset = pd.DataFrame(
        [
            {
                "sample_id": f"dataset-{dataset_index}/method-{method}/{source}",
                "dataset": f"dataset-{dataset_index}",
                "test_case": source,
                "score": 0.5,
                "quality": 1.0,
            }
            for dataset_index in range(2)
            for source in ("source-a", "source-b", "source-c")
            for method in range(2)
        ]
    )
    cfg = {
        "seed": 42,
        "split_seed": 42,
        "cross_validation": {"enabled": True, "n_splits": 3},
    }

    splits = grouped_cross_validation_splits(dataset, cfg)
    validation_indices = []
    for train_indices, fold_validation_indices in splits:
        train_groups = {
            (dataset.loc[index, "dataset"], dataset.loc[index, "test_case"])
            for index in train_indices
        }
        validation_groups = {
            (dataset.loc[index, "dataset"], dataset.loc[index, "test_case"])
            for index in fold_validation_indices
        }
        assert train_groups.isdisjoint(validation_groups)
        validation_indices.extend(fold_validation_indices)

    assert sorted(validation_indices) == dataset.index.tolist()


def test_cross_validation_run_saves_fold_and_aggregate_outputs(tmp_path: Path) -> None:
    from qualisr.regressors import run_experiment

    samples = regressor_samples(
        "train-data",
        tmp_path / "features",
        {"train": True, "validate": False, "test_size": 0},
        10,
    )
    cfg = regressor_test_config()
    cfg.update(
        {
            "experiment_name": "cv-test",
            "cross_validation": {"enabled": True, "n_splits": 5},
            "save_dataset_snapshot": True,
            "save_mean_correlations": False,
            "save_best_correlations": False,
            "profiling": {"regressors": False},
            "analysis": {
                "outliers": {"enabled": False},
                "feature_metrics": {"enabled": False},
                "feature_selection": {"enabled": False},
            },
            "correlation_metrics": {"enabled": False, "items": []},
            "paths": {"plots_root": str(tmp_path / "plots")},
            "models": {"linear": {"enabled": True, "params": {}}},
            "plot": {"enabled": False},
        }
    )

    result = run_experiment(cfg, samples, make_plots=False)

    assert len(result["fold_outputs"]) == 5
    assert set(result["fold_results"]["fold"]) == {1, 2, 3, 4, 5}
    assert result["results"].loc[0, "n_folds"] == 5
    output_dir = Path(result["output_dir"])
    assignments = pd.read_csv(output_dir / "metadata" / "cross_validation_folds.csv")
    predictions = pd.read_csv(output_dir / "predictions" / "predictions_linear.csv")
    assert len(assignments) == len(predictions) == 10
    assert set(assignments["fold"]) == {1, 2, 3, 4, 5}
    assert (output_dir / "correlations" / "cross_validation_folds.csv").is_file()
    assert (output_dir / "correlations" / "correlations.csv").is_file()


def test_regressor_run_saves_separate_per_dataset_validation_outputs(tmp_path: Path) -> None:
    from qualisr.regressors import run_experiment

    samples = regressor_samples(
        "train-data",
        tmp_path / "train-features",
        {"train": True, "validate": False, "test_size": 0},
        6,
    )
    samples += regressor_samples(
        "validation-a",
        tmp_path / "validation-a-features",
        {"train": False, "validate": True},
        4,
    )
    validation_b = regressor_samples(
        "validation-b",
        tmp_path / "validation-b-features",
        {"train": False, "validate": True},
        4,
    )
    for index, sample in enumerate(validation_b):
        sample["score_type"] = "bradley_terry"
        sample["test_case"] = f"gt-{index // 2}"
    samples += validation_b
    cfg = regressor_test_config()
    cfg.update(
        {
            "experiment_name": "per-dataset-test",
            "cross_validation": {"enabled": False, "n_splits": 5},
            "permutation_repeats": 3,
            "save_dataset_snapshot": False,
            "save_mean_correlations": False,
            "save_best_correlations": False,
            "profiling": {"regressors": False},
            "analysis": {
                "outliers": {"enabled": False},
                "feature_metrics": {"enabled": False},
                "feature_selection": {"enabled": False},
            },
            "correlation_metrics": {"enabled": False, "items": []},
            "paths": {"plots_root": str(tmp_path / "plots")},
            "models": {"linear": {"enabled": True, "params": {}}},
            "plot": {
                "enabled": {
                    "importance": True,
                    "all_importances": False,
                    "shap": False,
                    "all_shap_importances": False,
                    "correlations": False,
                    "correlations_without_metrics": False,
                    "feature_correlations": False,
                    "feature_cross_correlation_matrix": False,
                    "prediction_scatter": False,
                },
                "importance_figsize": [4, 3],
                "dpi": 72,
            },
        }
    )

    result = run_experiment(cfg, samples, make_plots=True)

    assert set(result["per_dataset"]) == {"validation-a", "validation-b"}
    output_dir = Path(result["output_dir"])
    for dataset_name in ("validation-a", "validation-b"):
        dataset_dir = output_dir / "per_dataset" / dataset_name
        assert (dataset_dir / "correlations" / "correlations.csv").is_file()
        assert (dataset_dir / "predictions" / "predictions_linear.csv").is_file()
        assert (dataset_dir / "importances" / "importance_linear.csv").is_file()
        assert (dataset_dir / "importances" / "importance_linear.png").is_file()
        assert (dataset_dir / "feature_analysis" / "feature_correlations.csv").is_file()
    bt_details = pd.read_csv(
        output_dir
        / "per_dataset"
        / "validation-b"
        / "correlations"
        / "correlations_per_gt.csv"
    )
    assert len(bt_details) == 2
    assert set(bt_details["correlation_group"]) == {"gt-0", "gt-1"}
