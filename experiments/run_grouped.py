"""Source-grouped, nested experiments for the QualiSR-Lab paper.

Run with ``qualisr-run-experiments --config configs/paper.json --output NEW_DIRECTORY``.
The installed example and paper experiments share the same feature and regression components.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from experiments.data import NR_COLUMNS, load_experiment_data
from experiments.fold_features import FoldTransformer
from experiments.presets import experiment_specs, search_candidates
from experiments.selection import select_ridge_features
from qualisr.regressors import MODEL_FACTORIES, validation_correlations

METADATA = ["sample_id", "dataset", "test_case", "source_group", "correlation_group", "score_type", "score", "method"]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def grouped_splits(frame: pd.DataFrame, indices: np.ndarray, folds: int, seed: int):
    """Keep shared content together while balancing the participating datasets."""
    data = frame.loc[indices, ["dataset", "source_group"]]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, valid in splitter.split(data, data["dataset"], groups=data["source_group"]):
        train_ids, valid_ids = indices[train], indices[valid]
        if set(frame.loc[train_ids, "source_group"]) & set(frame.loc[valid_ids, "source_group"]):
            raise AssertionError("Source groups cross an evaluation partition")
        if set(frame.loc[train_ids, "dataset"]) != set(data["dataset"]) or set(
            frame.loc[valid_ids, "dataset"]
        ) != set(data["dataset"]):
            raise ValueError("Every grouped fold must contain each participating training dataset")
        yield train_ids, valid_ids


def feature_columns(spec: dict, dimensions: dict[str, int]) -> list[str]:
    result = list(spec["scalars"])
    for family, count in dimensions.items():
        for representation in spec["representations"]:
            result.extend(f"{family}_{representation}_{i:03d}" for i in range(count))
    if not result or len(result) != len(set(result)):
        raise ValueError("Experiment features must be nonempty and unique")
    return result


def make_model(name: str, params: dict, seed: int, threads: int):
    execution = {"n_jobs": threads} if name in ("randomforest", "xgb") else {
        "thread_count": threads, "allow_writing_files": False, "task_type": "CPU"
    }
    cfg = {"seed": seed, "models": {name: {"params": {**params, **execution}}}}
    return MODEL_FACTORIES[name](cfg, name)


def fit_transform_fold(raw, train, evaluate, blocks, scalars, dimensions, seed, directory):
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    transformer = FoldTransformer(scalars, blocks, dimensions, seed=seed).fit(raw.loc[train])
    train_x = transformer.transform(raw.loc[train])
    eval_x = transformer.transform(raw.loc[evaluate])
    joblib.dump(transformer, directory / "transformer.joblib", compress=3)
    pd.DataFrame({"sample_id": transformer.fit_sample_ids_}).to_csv(directory / "fit_samples.csv", index=False)
    write_json(directory / "transform.json", {
        "dimensions": dimensions, "seed": seed, "basis": "stacked_SR_and_bicubic_reference_training_rows",
        "n_training_rows": len(train), "n_evaluation_rows": len(evaluate),
        "seconds": time.monotonic() - start,
        "explained_variance": {k: float(v.explained_variance_ratio_.sum()) for k, v in transformer.pcas_.items()},
    })
    return train_x, eval_x


def tune_model(name, spec, inner, seed, threads, *, candidates=None):
    if candidates is None:
        candidates = search_candidates(name, bool(spec["representations"]))
    history = []
    for number, candidate in enumerate(candidates):
        columns = feature_columns(spec, candidate["dimensions"])
        scores, plccs = [], []
        for train_x, train_y, valid_x, valid_y, metadata in inner:
            model = make_model(name, candidate["params"], seed, threads)
            model.fit(train_x[columns], train_y)
            plcc, srcc, _, _ = validation_correlations(valid_y, model.predict(valid_x[columns]), metadata)
            if not np.isfinite(srcc):
                raise ValueError(f"Undefined inner SRCC for {name}/{spec['name']}/candidate {number}")
            scores.append(srcc)
            plccs.append(plcc)
        history.append({"candidate": number, "params": json.dumps(candidate["params"], sort_keys=True),
                        "dimensions": json.dumps(candidate["dimensions"], sort_keys=True),
                        "mean_srcc": float(np.mean(scores)), "mean_plcc": float(np.mean(plccs)),
                        "fold_srcc": json.dumps(scores)})
    best = min(range(len(history)), key=lambda i: (-history[i]["mean_srcc"], i))
    table = pd.DataFrame(history)
    table["selected"] = np.arange(len(table)) == best
    return candidates[best], table


def save_evaluation(directory, model, columns, eval_x, metadata, params):
    directory.mkdir(parents=True, exist_ok=False)
    prediction = np.asarray(model.predict(eval_x[columns]), dtype=float)
    result = metadata[METADATA + ["evaluation"]].copy()
    result["prediction"] = prediction
    result.to_csv(directory / "predictions.csv", index=False)
    all_metrics = []
    for scope, group in result.groupby("evaluation", sort=True):
        plcc, srcc, datasets, comparison_groups = validation_correlations(
            group["score"], group["prediction"], group
        )
        datasets.insert(0, "evaluation", scope)
        datasets.to_csv(directory / f"metrics_{scope}.csv", index=False)
        comparison_groups.to_csv(directory / f"comparison_groups_{scope}.csv", index=False)
        all_metrics.append({"dataset": "macro", "evaluation": scope, "plcc": plcc, "srcc": srcc})
        all_metrics.extend(datasets.to_dict("records"))
    pd.DataFrame(all_metrics).to_csv(directory / "metrics.csv", index=False)
    joblib.dump(model, directory / "model.joblib", compress=3)
    actual = {key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
              for key, value in model.get_params().items()}
    write_json(directory / "parameters.json", {**params, "columns": columns, "actual_model_params": actual})
    return all_metrics


def run_grouped_experiments(config: dict, base_dir: Path, output: Path, options: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=False)
    settings = config["evaluation"]
    seed = int(settings["model_seed"])
    threads = int(settings["threads"])
    if not 1 <= threads <= 4:
        raise ValueError("CPU experiment jobs require between one and four CPU threads")
    dimensions = settings["max_pca_dimensions"]
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib"
    )}
    versions["xgboost_module"] = xgboost.__version__
    write_json(output / "run.json", {"config": config, "options": vars(options), "versions": versions,
                                     "python": platform.python_version(), "phase_at_creation": "loading",
                                     "external_evaluation": "Predictions from each source-only outer-training fit; no target-label selection or full-source refit",
                                     "installed_distributions": {
                                         d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                     }})
    print("Loading raw features and auditing source content", flush=True)
    raw, blocks, scalars, provenance = load_experiment_data(config, base_dir)
    write_json(output / "data_provenance.json", provenance)
    raw[METADATA + ["role", "gt_path", "lr_path", "content_sha256", "content_kind"]].to_csv(
        output / "samples.csv", index=False
    )
    train_pool = raw.index[raw["role"] == "train"].to_numpy()
    external = raw.index[raw["role"] == "external"].to_numpy()
    specs = experiment_specs()
    if options.experiments:
        requested = options.experiments.split(",")
        if set(requested) - {s["name"] for s in specs}:
            raise ValueError("Unknown requested experiment name")
        specs = [s for s in specs if s["name"] in requested]
    write_json(output / "experiment_specs.json", specs)
    if options.audit_only:
        write_json(output / "status.json", {"status": "audit_complete", "n_samples": len(raw)})
        return
    summary_rows = []
    assignments = []
    split_seeds = options.split_seeds or settings["split_seeds"]
    models = options.models or settings["models"]
    for split_seed in split_seeds:
        outer = list(grouped_splits(raw, train_pool, settings["outer_folds"], split_seed))
        for fold, (train, heldout) in enumerate(outer, start=1):
            if options.fold_limit and fold > options.fold_limit:
                break
            root = output / f"seed_{split_seed}" / f"fold_{fold:02d}"
            evaluate = np.concatenate([heldout, external])
            meta = raw.loc[evaluate, METADATA].copy()
            meta["evaluation"] = np.where(meta.index.isin(heldout), "internal", "external")
            assignment = raw.loc[train_pool, METADATA].copy()
            assignment["split_seed"], assignment["fold"] = split_seed, fold
            assignment["partition"] = np.where(assignment.index.isin(train), "train", "heldout")
            assignments.append(assignment)
            pd.concat(assignments, ignore_index=True).to_csv(output / "fold_assignments.csv", index=False)
            print(f"seed={split_seed} fold={fold}: preparing outer features", flush=True)
            train_x, eval_x = fit_transform_fold(
                raw, train, evaluate, blocks, scalars, dimensions, seed, root / "outer_transform"
            )
            train_y = raw.loc[train, "score"]
            inner = []
            if not options.skip_selection or (not options.skip_tuning and any(s["tune"] for s in specs)):
                for inner_fold, (inner_train, inner_valid) in enumerate(
                    grouped_splits(raw, train, settings["inner_folds"], split_seed), start=1
                ):
                    print(f"seed={split_seed} fold={fold}: inner features {inner_fold}", flush=True)
                    ix, vx = fit_transform_fold(raw, inner_train, inner_valid, blocks, scalars, dimensions,
                                                seed, root / f"inner_{inner_fold:02d}")
                    raw.loc[inner_valid, METADATA].to_csv(
                        root / f"inner_{inner_fold:02d}" / "validation_samples.csv", index=False
                    )
                    inner.append((ix, raw.loc[inner_train, "score"], vx, raw.loc[inner_valid, "score"],
                                  raw.loc[inner_valid, METADATA]))
            for spec in specs:
                for name in models:
                    print(f"seed={split_seed} fold={fold} {spec['name']} {name} fixed", flush=True)
                    columns = feature_columns(spec, {"vgg": 5, "resnet": 5})
                    model = make_model(name, {}, seed, threads)
                    model.fit(train_x[columns], train_y)
                    params = {"track": "fixed", "dimensions": {"vgg": 5, "resnet": 5},
                              "seed": seed, "split_seed": split_seed, "fold": fold}
                    metrics = save_evaluation(root / spec["name"] / name / "fixed", model, columns, eval_x, meta, params)
                    summary_rows.extend({"experiment": spec["name"], "model": name, **params, **m} for m in metrics)
                    if spec["tune"] and not options.skip_tuning:
                        print(f"seed={split_seed} fold={fold} {spec['name']} {name} inner search", flush=True)
                        candidate, history = tune_model(name, spec, inner, seed, threads)
                        columns = feature_columns(spec, candidate["dimensions"])
                        model = make_model(name, candidate["params"], seed, threads)
                        model.fit(train_x[columns], train_y)
                        params = {**params, "track": "tuned", **candidate}
                        target = root / spec["name"] / name / "tuned"
                        metrics = save_evaluation(target, model, columns, eval_x, meta, params)
                        history.to_csv(target / "inner_search.csv", index=False)
                        summary_rows.extend({"experiment": spec["name"], "model": name, **params, **m} for m in metrics)
                    pd.DataFrame(summary_rows).to_csv(output / "metrics_all_folds.csv", index=False)
            if not options.skip_selection:
                print(f"seed={split_seed} fold={fold}: nested Ridge selection", flush=True)
                candidates = ["musiq", "arniqa", "unique", "paq2piq", "lpips-vgg_rlfn", "ahiq_rlfn",
                              "vgg_diff_000", "resnet_diff_000", "max", "median", "std", "area00"]
                selected, history = select_ridge_features(inner, candidates)
                model = Ridge(alpha=1.0).fit(train_x[selected], train_y)
                params = {"track": "nested_selection", "seed": seed, "split_seed": split_seed, "fold": fold}
                target = root / "compact_selection" / "ridge"
                metrics = save_evaluation(target, model, selected, eval_x, meta, params)
                history.to_csv(target / "inner_selection.csv", index=False)
                summary_rows.extend({"experiment": "compact_selection", "model": "ridge", **params, **m} for m in metrics)
            # Direct NR metrics share exactly the predictors' evaluation populations.
            direct = meta.copy()
            for column in NR_COLUMNS:
                direct["prediction"] = raw.loc[evaluate, column]
                for scope, group in direct.groupby("evaluation", sort=True):
                    plcc, srcc, datasets, _ = validation_correlations(group["score"], group["prediction"], group)
                    rows = [{"dataset": "macro", "plcc": plcc, "srcc": srcc}, *datasets.to_dict("records")]
                    summary_rows.extend({"experiment": "direct_nr", "model": column, "track": "direct",
                                         "split_seed": split_seed, "fold": fold, "evaluation": scope, **m} for m in rows)
            pd.DataFrame(summary_rows).to_csv(output / "metrics_all_folds.csv", index=False)
            write_json(output / "status.json", {"status": "running", "last_completed_seed": split_seed,
                                                "last_completed_fold": fold})
    table = pd.DataFrame(summary_rows)
    table.groupby(["experiment", "model", "track", "evaluation", "dataset"], dropna=False).agg(
        plcc_mean=("plcc", "mean"), plcc_std=("plcc", "std"), srcc_mean=("srcc", "mean"),
        srcc_std=("srcc", "std"), n_folds=("srcc", "size")
    ).to_csv(output / "metrics_summary.csv")
    write_json(output / "status.json", {"status": "complete", "n_metric_rows": len(table)})
    print("Paper experiments complete", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--split-seeds", type=int, nargs="+")
    parser.add_argument("--models", nargs="+", choices=["randomforest", "xgb", "catboost"])
    parser.add_argument("--experiments", help="Comma-separated preset names; omitted means the complete grouped experiment matrix")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    options = parser.parse_args(argv)
    config = json.loads(options.config.read_text())
    if not (3, 12) <= tuple(sys.version_info[:2]) < (3, 14):
        raise RuntimeError("Paper experiments require Python 3.12 or 3.13")
    if options.fold_limit is not None and options.fold_limit < 1:
        parser.error("--fold-limit must be positive")
    with threadpool_limits(limits=int(config["evaluation"]["threads"])):
        run_grouped_experiments(config, options.config.resolve().parent, options.output.resolve(), options)


if __name__ == "__main__":
    main()
