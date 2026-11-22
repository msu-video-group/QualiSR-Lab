"""Matched nested compact-subset selection across the four paper regressors.

Prepare small matrices from verified, frozen training-only transforms; run each
backend separately; summarize only after every requested fold has completed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from experiments.data import load_experiment_data, sha256_file
from experiments.run_grouped import METADATA, grouped_splits, make_model, write_json
from experiments.selection import select_features
from qualisr.regressors import validation_correlations

CANDIDATES = ["musiq", "arniqa", "unique", "paq2piq", "lpips-vgg_rlfn", "ahiq_rlfn",
              "vgg_diff_000", "resnet_diff_000", "max", "median", "std", "area00"]
BACKENDS = ["ridge", "randomforest", "xgb", "catboost"]
FAMILIES = dict(zip(CANDIDATES, ["NR"] * 4 + ["pseudo-FR"] * 2 + ["VGG Diff", "ResNet Diff"]
                    + ["WASD"] * 4, strict=True))
KEY = ["split_seed", "fold"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def read_predictions(path, backend):
    frame = read(path)
    # XGBoost emits float32. Pandas writes its shortest float32-round-trip
    # decimals, so restore that dtype before widening to the scoring dtype.
    if backend == "xgb":
        for name in ("selected", "full_pool", "ridge_selected"):
            frame[name] = frame[name].astype(np.float32).astype(np.float64)
    return frame


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def estimator(backend):
    return Ridge(alpha=1.0) if backend == "ridge" else make_model(backend, {}, 42, 2)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    require(config["evaluation"]["model_seed"] == 42, "Unexpected model seed")
    require(json.loads((args.parent / "status.json").read_text())["status"] == "complete",
            "Parent run is incomplete")
    raw, _, _, provenance = load_experiment_data(config, args.config.resolve().parent)
    original = read(args.parent / "samples.csv")
    pd.testing.assert_frame_equal(raw[METADATA], original[METADATA], check_exact=True)
    parent_provenance = json.loads((args.parent / "data_provenance.json").read_text())
    require([(f["dataset"], f["sha256"]) for f in provenance["files"]] ==
            [(f["dataset"], f["sha256"]) for f in parent_provenance["files"]], "Feature cache hashes changed")
    raw[METADATA + ["role"]].to_csv(args.output / "samples.csv", index=False)
    write_json(args.output / "data_provenance.json", provenance)
    assignments = read(args.parent / "fold_assignments.csv")
    train_pool = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    inputs, keys = [], []
    for seed in config["evaluation"]["split_seeds"]:
        for fold, (train, heldout) in enumerate(grouped_splits(raw, train_pool, 5, seed), 1):
            key = dict(split_seed=seed, fold=fold)
            folder = args.output / f"seed_{seed}_fold_{fold:02d}"
            folder.mkdir()
            parent = args.parent / f"seed_{seed}/fold_{fold:02d}"
            given = assignments.loc[assignments.split_seed.eq(seed) & assignments.fold.eq(fold)]
            require(set(given.loc[given.partition.eq("train"), "sample_id"]) == set(raw.loc[train, "sample_id"]),
                    "Parent outer training IDs changed")
            require(set(given.loc[given.partition.eq("heldout"), "sample_id"]) == set(raw.loc[heldout, "sample_id"]),
                    "Parent outer holdout IDs changed")
            evaluate = np.concatenate([heldout, external])
            paths = [parent / "outer_transform/transformer.joblib"]
            transform = joblib.load(paths[0])
            require(transform.fit_sample_ids_ == raw.loc[train, "sample_id"].tolist(), "Outer transform fit IDs differ")
            train_x = transform.transform(raw.loc[train])[CANDIDATES]
            eval_x = transform.transform(raw.loc[evaluate])[CANDIDATES]
            meta = raw.loc[evaluate, METADATA].copy()
            meta["evaluation"] = np.where(meta.index.isin(heldout), "internal", "external")
            inner, partitions = [], []
            partitions.append(raw.loc[train_pool, METADATA].assign(scope="outer", inner_fold=0))
            partitions[-1]["partition"] = np.where(partitions[-1].index.isin(train), "train", "heldout")
            for number, (it, iv) in enumerate(grouped_splits(raw, train, 3, seed), 1):
                path = parent / f"inner_{number:02d}/transformer.joblib"
                paths.append(path)
                frozen = joblib.load(path)
                require(frozen.fit_sample_ids_ == raw.loc[it, "sample_id"].tolist(), "Inner transform fit IDs differ")
                expected_valid = read(parent / f"inner_{number:02d}/validation_samples.csv")
                pd.testing.assert_frame_equal(raw.loc[iv, METADATA].reset_index(drop=True),
                                              expected_valid[METADATA], check_exact=True)
                inner.append((frozen.transform(raw.loc[it])[CANDIDATES], raw.loc[it, "score"],
                              frozen.transform(raw.loc[iv])[CANDIDATES], raw.loc[iv, "score"],
                              raw.loc[iv, METADATA]))
                part = raw.loc[train, METADATA].assign(scope="inner", inner_fold=number)
                part["partition"] = np.where(part.index.isin(it), "train", "heldout")
                partitions.append(part)
            pd.concat(partitions).assign(**key).to_csv(folder / "partitions.csv.gz", index=False)
            ridge = parent / "compact_selection/ridge"
            selected = json.loads((ridge / "parameters.json").read_text())["columns"]
            joblib.dump(dict(inner=inner, train_x=train_x, train_y=raw.loc[train, "score"],
                             eval_x=eval_x, metadata=meta, ridge_selected=selected), folder / "matrices.joblib", compress=3)
            for name in ("inner_selection.csv", "predictions.csv"):
                read(ridge / name).to_csv(folder / f"parent_{name}", index=False)
                paths.append(ridge / name)
            inputs.extend(dict(path=str(path.resolve()), sha256=sha256_file(path)) for path in paths)
            keys.append(key)
            print(f"prepared {seed}/{fold}", flush=True)
    write_json(args.output / "run.json", dict(config=config, parent=str(args.parent.resolve()),
        candidates=CANDIDATES, families=FAMILIES, folds=keys, inputs=inputs,
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "catboost")},
        python=platform.python_version(), model_seed=42, threads=2,
        protocol="Frozen parent training-only paired PCA, median imputation, MinMaxScaler; independent complete "
                 "greedy forward/backward paths on three inner grouped folds; mean MOS-dataset SRCC; "
                 "ties: smaller size, forward, lexical. Fixed library-default tree settings. No external selection."))
    write_json(args.output / "status.json", dict(status="complete", folds=len(keys)))


def score_predictions(frame):
    rows = []
    for variant in ("selected", "full_pool", "ridge_selected"):
        for scope, group in frame.groupby("evaluation", sort=True):
            _, _, datasets, _ = validation_correlations(group.score, group[variant], group)
            rows.extend(datasets.assign(variant=variant, evaluation=scope).to_dict("records"))
    return pd.DataFrame(rows)


def verify_selection(history, trials):
    """Reconstruct both greedy paths and the final choice from all unique trials."""
    require(not trials.features.duplicated().any(), "Duplicate trial subset")
    lookup = trials.set_index("features")
    expected = []
    for direction in ("forward", "backward"):
        current = [] if direction == "forward" else sorted(CANDIDATES)
        for size in (range(1, 13) if direction == "forward" else range(12, 0, -1)):
            if direction == "backward" and size == 12:
                options = [current]
            elif direction == "forward":
                options = [sorted(current + [name]) for name in CANDIDATES if name not in current]
            else:
                options = [[name for name in current if name != removed] for removed in current]
            scored = [(names, lookup.loc["|".join(names)]) for names in options]
            scored = [(names, row) for names, row in scored if np.isfinite(row.mean_srcc)]
            require(bool(scored), "No eligible trial")
            current, row = min(scored, key=lambda pair: (-pair[1].mean_srcc, tuple(pair[0])))
            require(len(current) == size, "Greedy size differs")
            expected.append(dict(direction=direction, features="|".join(current), n_features=size,
                                 mean_srcc=row.mean_srcc))
    for actual, wanted in zip(history.to_dict("records"), expected, strict=True):
        for field in wanted:
            require(actual[field] == wanted[field], f"Greedy path mismatch: {field}")
    winner = min(range(24), key=lambda i: (-expected[i]["mean_srcc"], expected[i]["n_features"],
                 expected[i]["direction"] != "forward", expected[i]["features"]))
    require(history.selected.sum() == 1 and bool(history.iloc[winner].selected), "Wrong selected trial")
    selected = expected[winner]["features"].split("|")
    for row in trials.itertuples():
        scores = np.array(json.loads(row.fold_srcc))
        require(len(scores) == 3, "Expected three inner scores")
        np.testing.assert_allclose(row.mean_srcc, scores.mean(), atol=1e-15, rtol=0)
    return selected


def select_fold(args, key):
    with threadpool_limits(limits=2):
        seed, fold = key["split_seed"], key["fold"]
        source = args.prepared / f"seed_{seed}_fold_{fold:02d}"
        target = args.output / source.name
        target.mkdir()
        data = joblib.load(source / "matrices.joblib")
        trials = []
        start = time.monotonic()
        def on_trial(row, trials=trials, seed=seed, fold=fold, start=start):
            trials.append(row)
            if len(trials) % 10 == 0:
                print(f"{args.backend} {seed}/{fold}: {len(trials)} unique subsets, "
                      f"{time.monotonic() - start:.1f}s", flush=True)
        selected, history = select_features(data["inner"], CANDIDATES,
                                            lambda: estimator(args.backend), on_trial)
        trial_table = pd.DataFrame(trials)
        require(selected == verify_selection(history, trial_table), "Selection reconstruction differs")
        history.to_csv(target / "history.csv", index=False)
        trial_table.to_csv(target / "trials.csv", index=False)
        frame = data["metadata"].copy()
        fitted_settings = {}
        for name, columns in dict(selected=selected, full_pool=sorted(CANDIDATES),
                                  ridge_selected=data["ridge_selected"]).items():
            model = estimator(args.backend)
            model.fit(data["train_x"][columns], data["train_y"])
            frame[name] = np.asarray(model.predict(data["eval_x"][columns]), dtype=np.float64)
            require(np.isfinite(frame[name]).all(), "Nonfinite prediction")
            if args.backend == "catboost":
                fitted_settings[name] = model.get_all_params()
            elif args.backend == "xgb":
                fitted_settings[name] = json.loads(model.get_booster().save_config())
        write_json(target / "fitted_settings.json", fitted_settings)
        if args.backend == "ridge":
            parent_history = read(source / "parent_inner_selection.csv")
            require(selected == parent_history.loc[parent_history.selected, "features"].iloc[0].split("|"),
                    "Ridge subset differs from parent")
            np.testing.assert_allclose(history.mean_srcc, parent_history.mean_srcc, atol=1e-12, rtol=0)
            parent = read(source / "parent_predictions.csv").set_index("sample_id").loc[frame.sample_id]
            np.testing.assert_allclose(frame.selected, parent.prediction, atol=1e-12, rtol=0)
        frame.to_csv(target / "predictions.csv.gz", index=False)
        score_predictions(frame).to_csv(target / "metrics.csv", index=False)
        params = {name: str(value) if isinstance(value, float) and not np.isfinite(value) else value
                  for name, value in estimator(args.backend).get_params().items()}
        write_json(target / "result.json", dict(**key, backend=args.backend, selected=selected,
            n_features=len(selected), unique_trials=len(trials), inner_fits=3 * len(trials),
            seconds=time.monotonic() - start, actual_model_params=params,
            ridge_parent_parity=args.backend == "ridge"))
        print(f"completed {args.backend} {seed}/{fold}: {len(selected)} features, "
              f"{time.monotonic() - start:.1f}s", flush=True)
    return key


def select(args):
    prepared = json.loads((args.prepared / "run.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    folds = prepared["folds"][:1] if args.pilot else prepared["folds"][1:]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(select_fold, args, key) for key in folds]
        for number, future in enumerate(as_completed(futures), 1):
            key = future.result()
            write_json(args.output / "status.json", dict(status="running", last_completed=key,
                                                        completed_folds=number, workers=args.workers))
    write_json(args.output / "status.json", dict(status="complete", backend=args.backend,
                                                folds=len(folds), workers=args.workers))


def overlap(a, b):
    left, right = set(a), set(b)
    observed = len(left & right) / len(left | right)
    intersection = np.arange(max(0, len(left) + len(right) - 12), min(len(left), len(right)) + 1)
    chance = float(np.sum(hypergeom.pmf(intersection, 12, len(left), len(right)) *
                          intersection / (len(left) + len(right) - intersection)))
    fa, fb = {FAMILIES[x] for x in left}, {FAMILIES[x] for x in right}
    return dict(feature_jaccard=observed, size_matched_chance_jaccard=chance,
                excess_jaccard=observed - chance, family_jaccard=len(fa & fb) / len(fa | fb),
                same_subset=left == right)


def verify_prepared(prepared, run):
    samples = read(prepared / "samples.csv").set_index("sample_id", drop=False)
    for key in run["folds"]:
        folder = prepared / f"seed_{key['split_seed']}_fold_{key['fold']:02d}"
        parts = read(folder / "partitions.csv.gz")
        outer = parts.loc[parts.scope.eq("outer")]
        train = set(outer.loc[outer.partition.eq("train"), "sample_id"])
        heldout = set(outer.loc[outer.partition.eq("heldout"), "sample_id"])
        require(not train & heldout, "Outer samples overlap")
        require(train | heldout == set(samples.loc[samples.role.eq("train")].index), "Outer pool differs")
        validation_ids = set()
        for (_, number), group in parts.groupby(["scope", "inner_fold"], sort=True):
            require(group.sample_id.is_unique, "Duplicate partition sample")
            fit = group.loc[group.partition.eq("train")]
            valid = group.loc[group.partition.eq("heldout")]
            require(not set(fit.source_group) & set(valid.source_group), "Source groups cross a partition")
            if number:
                require(set(group.sample_id) == train, "Inner rows are not exactly outer-training rows")
                require(not validation_ids & set(valid.sample_id), "Inner validation rows overlap")
                validation_ids.update(valid.sample_id)
        require(validation_ids == train, "Inner validation folds do not partition outer training")
        pd.testing.assert_frame_equal(parts[METADATA].reset_index(drop=True),
            samples.loc[parts.sample_id, METADATA].reset_index(drop=True), check_exact=True)


def summarize(args):
    args.output.mkdir(parents=True, exist_ok=False)
    run = json.loads((args.prepared / "run.json").read_text())
    verify_prepared(args.prepared, run)
    results, metrics, checked, precision = [], [], [], []
    for backend in BACKENDS:
        for key in run["folds"]:
            stage = "pilot" if key == run["folds"][0] else "full"
            folder = args.runs / f"{backend}-{stage}" / f"seed_{key['split_seed']}_fold_{key['fold']:02d}"
            require(json.loads((folder.parent / "status.json").read_text())["status"] == "complete",
                    "Incomplete selection run")
            result = json.loads((folder / "result.json").read_text())
            require(result["selected"] == verify_selection(read(folder / "history.csv"), read(folder / "trials.csv")),
                    "Saved subset differs")
            frame = read_predictions(folder / "predictions.csv.gz", backend)
            sample = read(args.prepared / "samples.csv").set_index("sample_id", drop=False).loc[frame.sample_id]
            pd.testing.assert_frame_equal(frame[METADATA].reset_index(drop=True),
                                          sample[METADATA].reset_index(drop=True), check_exact=True)
            computed = score_predictions(frame)
            saved_metrics = read(folder / "metrics.csv")
            pd.testing.assert_frame_equal(computed, saved_metrics, check_exact=False,
                                          atol=1e-12, rtol=0)
            if backend == "xgb":
                decimals = read(folder / "predictions.csv.gz")
                precision.append(dict(**key, backend=backend, native_prediction_dtype="float32",
                    max_decimal_to_original_error=max(float(np.max(np.abs(frame[name] - decimals[name])))
                        for name in ("selected", "full_pool", "ridge_selected")),
                    max_saved_plcc_error=float(np.max(np.abs(computed.plcc - saved_metrics.plcc))),
                    max_saved_srcc_error=float(np.max(np.abs(computed.srcc - saved_metrics.srcc)))))
            results.append({k: result[k] for k in (*KEY, "backend", "n_features", "inner_fits", "seconds")}
                           | {"features": "|".join(result["selected"])})
            metrics.append(computed.assign(backend=backend, **key))
            checked.extend(dict(path=str(path.resolve()), sha256=sha256_file(path)) for path in folder.iterdir())
    subsets = pd.DataFrame(results)
    subsets.to_csv(args.output / "subsets.csv", index=False)
    pd.DataFrame(precision).to_csv(args.output / "prediction_precision_audit.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(args.output / "metrics.csv", index=False)
    subsets.groupby("backend").agg(n_folds=("fold", "size"), min_features=("n_features", "min"),
        median_features=("n_features", "median"), max_features=("n_features", "max"),
        inner_fits=("inner_fits", "sum"), seconds=("seconds", "sum")).to_csv(args.output / "sizes.csv")
    family, features, pairs = [], [], []
    for backend, group in subsets.groupby("backend", sort=True):
        sets = [row.features.split("|") for row in group.itertuples()]
        for name in sorted(set(FAMILIES.values())):
            counts = [sum(FAMILIES[x] == name for x in selected) for selected in sets]
            family.append(dict(backend=backend, family=name, folds_included=sum(x > 0 for x in counts),
                               mean_features=float(np.mean(counts)), min_features=min(counts), max_features=max(counts)))
        for feature in CANDIDATES:
            features.append(dict(backend=backend, feature=feature, folds_included=sum(feature in x for x in sets)))
        for a, b in itertools.combinations(group.to_dict("records"), 2):
            pairs.append(dict(scope="across_splits", backend_a=backend, backend_b=backend,
                seed_a=a["split_seed"], fold_a=a["fold"], seed_b=b["split_seed"], fold_b=b["fold"],
                **overlap(a["features"].split("|"), b["features"].split("|"))))
    for (seed, fold), group in subsets.groupby(KEY, sort=True):
        for a, b in itertools.combinations(group.to_dict("records"), 2):
            pairs.append(dict(scope="across_regressors", backend_a=a["backend"], backend_b=b["backend"],
                seed_a=seed, fold_a=fold, seed_b=seed, fold_b=fold,
                **overlap(a["features"].split("|"), b["features"].split("|"))))
    pd.DataFrame(family).to_csv(args.output / "families.csv", index=False)
    pd.DataFrame(features).to_csv(args.output / "features.csv", index=False)
    pair_table = pd.DataFrame(pairs)
    pair_table.to_csv(args.output / "overlap_pairs.csv", index=False)
    pair_table.groupby(["scope", "backend_a", "backend_b"]).agg(
        n_pairs=("feature_jaccard", "size"), feature_mean=("feature_jaccard", "mean"),
        feature_min=("feature_jaccard", "min"), feature_max=("feature_jaccard", "max"),
        family_mean=("family_jaccard", "mean"), chance_mean=("size_matched_chance_jaccard", "mean"),
        excess_mean=("excess_jaccard", "mean"), identical=("same_subset", "sum")
    ).to_csv(args.output / "overlap_summary.csv")
    pd.concat(metrics).groupby(["backend", "variant", "evaluation", "dataset"]).agg(
        plcc_mean=("plcc", "mean"), plcc_sd=("plcc", "std"), srcc_mean=("srcc", "mean"),
        srcc_sd=("srcc", "std"), n_folds=("srcc", "size")).to_csv(args.output / "performance.csv")
    write_json(args.output / "verification.json", dict(status="passed", selected_subsets=len(subsets),
        inner_fits=int(subsets.inner_fits.sum()), inputs=checked,
        checks=["all outer/inner source partitions and targets", "60 independent greedy path and final winner reconstructions", "all prediction targets and IDs",
                "PLCC/SRCC recomputed from saved predictions", "no external selection"],
        interpretation="Descriptive overlap; repeated folds and subset pairs are dependent. PCA-coordinate identity "
                       "is meaningful between regressors in a common fold; family-level overlap is primary across folds. "
                       "Chance expectation conditions on sizes and 12 candidates; it is not a p-value."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("config", "parent", "output"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    selection = sub.add_parser("select")
    for name in ("prepared", "output"):
        selection.add_argument(f"--{name}", type=Path, required=True)
    selection.add_argument("--backend", choices=BACKENDS, required=True)
    selection.add_argument("--pilot", action="store_true")
    selection.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    summary = sub.add_parser("summarize")
    for name in ("prepared", "runs", "output"):
        summary.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {"prepare": prepare, "select": select, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
