"""Test pseudo-reference FR scores on top of the archived five-NR + Ref model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import xgboost
from threadpoolctl import threadpool_limits

from experiments.analyze_bt_pairs import paired_bt
from experiments.analyze_statistics import BACKENDS, BT, MACRO, MOS, SEED, Analysis, estimate, macro
from experiments.data import FR_METRICS, NR_COLUMNS, load_experiment_data, read_csv, require, sha256_file
from experiments.run_grouped import METADATA, feature_columns, grouped_splits, make_model, write_json
from experiments.transfer import check_partition

BASELINE = "nr_ref"
PRIMARY = "span_lpips_stlpips"
REF_DIMS = {"vgg": 5, "resnet": 50}
COMPACT = ("lpips-vgg", "stlpips-vgg")
PAIR_ARMS = {f"{ref}_lpips_stlpips": [f"{metric}_{ref}" for metric in COMPACT] for ref in ("bicubic", "rlfn")}
ADDITIONS = {
    BASELINE: [],
    PRIMARY: [f"{metric}_span" for metric in COMPACT],
    "span_six": [f"{metric}_span" for metric in FR_METRICS],
    **PAIR_ARMS,
}
EXTENSION = {f"{ref}_six": [f"{metric}_{ref}" for metric in FR_METRICS] for ref in ("bicubic", "rlfn")}


def feature_sets(extend=False):
    baseline = feature_columns({"scalars": NR_COLUMNS, "representations": ["ref"]}, REF_DIMS)
    additions = ADDITIONS | EXTENSION if extend else ADDITIONS
    return {name: baseline + fr for name, fr in additions.items()}


def read_json(path):
    return json.loads(path.read_text())


def prepare_fold(raw, train, evaluate, archive):
    check_partition(raw, train, evaluate)
    path = archive / "outer"
    partition = read_json(path / "partition.json")
    transform = joblib.load(path / "transformer.joblib")
    require(partition["train"] == train.tolist() and partition["evaluate"] == evaluate.tolist(),
            "Archived outer partition differs")
    require(partition["fit_sample_ids"] == transform.fit_sample_ids_ == raw.loc[train, "sample_id"].tolist(),
            "Archived preprocessing fit IDs differ")
    require(partition["dimensions"] == transform.dimensions == {"vgg": 96, "resnet": 96},
            "Expected the shared paired PCA basis")
    with np.load(path / "evaluation.npz") as data:
        require(np.array_equal(data["indices"], evaluate) and data["columns"].tolist() == transform.feature_names_out_,
                "Archived evaluation projection differs in rows or columns")
        evaluation = pd.DataFrame(data["values"], columns=data["columns"], index=evaluate)
    training = transform.transform(raw.loc[train])
    manifest = {name: sha256_file(path / name) for name in ("partition.json", "transformer.joblib", "evaluation.npz")}
    return training, evaluation, manifest


def reuse_fold(path, manifest, metadata, n_fit, seed, fold):
    proof_path = path / "verification.json"
    proof = read_json(proof_path)
    require(proof["status"] == "passed" and proof["archive_sha256"] == manifest,
            f"Different or unverified archived preprocessing: {path}")
    require(proof["predictions_sha256"] == sha256_file(path / "predictions.csv.gz"),
            f"Archived forecasts failed their checksum: {path}")
    predictions = read_csv(path, "predictions.csv.gz")
    fits = proof["fits"]
    columns = feature_sets()
    groups = dict(tuple(predictions.groupby(["arm", "model"])))
    expected = {(arm, backend) for arm in columns for backend in BACKENDS}
    require(groups.keys() == expected and len(fits) == len(expected)
            and {(fit["arm"], fit["backend"]) for fit in fits} == expected
            and len(predictions) == proof["prediction_rows"], f"Incomplete archived forecasts: {path}")
    expected_rows = metadata.reset_index(drop=True).assign(split_seed=seed, fold=fold)
    for fit in fits:
        arm, backend = fit["arm"], fit["backend"]
        require(fit["columns"] == columns[arm] and fit["n_fit"] == n_fit and fit["n_predict"] == len(metadata),
                f"Different archived fit: {path}/{backend}/{arm}")
        rows = groups[arm, backend].reset_index(drop=True)
        pd.testing.assert_frame_equal(rows[expected_rows.columns], expected_rows, check_exact=True)
    return predictions, [dict(fit, refit=False) for fit in fits], sha256_file(proof_path)


def run_backend(backend, settings, columns_by_arm, training, evaluation, target, metadata,
                archived_predictions, archive, output):
    output.mkdir()
    threads = settings["threads"]
    prediction_options = {"thread_count": threads} if backend == "catboost" else {}
    forecasts, records = [], []
    for arm, columns in columns_by_arm.items():
        print(f"  {backend}/{arm}", flush=True)
        if arm == BASELINE:
            baseline = archive / backend / BASELINE / "vgg05_resnet50"
            require(read_json(baseline / "parameters.json")["columns"] == columns, "Archived control columns differ")
            model_path = baseline / "model.joblib"
        else:
            model_path = output / f"{arm}.joblib"
            model = make_model(backend, {}, settings["model_seed"], threads).fit(training[columns], target)
            fitted_prediction = model.predict(evaluation[columns], **prediction_options)
            joblib.dump(model, model_path, compress=3)
        model = joblib.load(model_path)
        replay = model.predict(evaluation[columns], **prediction_options)
        require(np.isfinite(replay).all(), f"Nonfinite predictions from {backend}/{arm}")
        if arm == BASELINE:
            original = archived_predictions.loc[archived_predictions.model.eq(backend), METADATA + ["prediction"]]
            pd.testing.assert_frame_equal(original[METADATA].reset_index(drop=True), metadata[METADATA].reset_index(drop=True))
            fitted_prediction = original.prediction.to_numpy()
        np.testing.assert_allclose(replay, fitted_prediction, rtol=0, atol=1e-12)
        forecasts.append(metadata.assign(arm=arm, model=backend, prediction=fitted_prediction))
        records.append(dict(arm=arm, backend=backend, columns=columns, model_path=str(model_path),
                            model_sha256=sha256_file(model_path), refit=arm != BASELINE,
                            n_fit=len(training), n_predict=len(evaluation),
                            max_replay_error=float(np.max(np.abs(replay - fitted_prediction)))))
    return pd.concat(forecasts, ignore_index=True), records


def run(config_path, prior, output, pilot, workers, extend_from=None, refit_pairs=False):
    require(not refit_pairs or extend_from is not None, "Pair refits require the completed SPAN run")
    config = read_json(config_path)
    settings = config["evaluation"]
    require(tuple(settings["models"]) == BACKENDS and settings["threads"] == 2 and 1 <= workers <= 2,
            "Expected the three paper backends, two threads per backend, and at most two workers")
    archive = prior / "complete"
    proof = read_json(prior / "complete-verified/verification.json")
    n_full_folds = settings["outer_folds"] * len(settings["split_seeds"])
    require(proof["status"] == "passed" and proof["folds"] == n_full_folds == 15, "Need the verified 15-fold PCA run")
    require(read_json(archive / "run.json")["config"] == config, "The archived run uses a different paper config")
    columns_by_arm = feature_sets(extend=extend_from is not None and not refit_pairs)
    fit_arms = columns_by_arm
    if extend_from is not None:
        fit_arms = {arm: columns_by_arm[arm] for arm in (PAIR_ARMS if refit_pairs else EXTENSION)}
        saved = read_json(extend_from / "run.json")
        status = read_json(extend_from / "status.json")
        expected_folds = [[seed, fold] for seed in settings["split_seeds"] for fold in range(1, settings["outer_folds"] + 1)]
        require(status["status"] == "complete" and status["completed_folds"] == expected_folds,
                "Need the completed 15-fold pseudo-FR run")
        require(saved["config"] == config and saved["prior"] == str(prior) and saved["features"] == feature_sets(),
                "The archived pseudo-FR run uses a different protocol")
    output.mkdir()
    write_json(output / "status.json", dict(status="loading_inputs"))
    raw, _, _, provenance = load_experiment_data(config, config_path.parent, include_all_rr=True)
    require(provenance == read_json(archive / "data_provenance.json"), "The source or feature-cache provenance changed")
    if extend_from is not None:
        require(provenance == read_json(extend_from / "data_provenance.json"), "The archived pseudo-FR inputs differ")
    original = pd.read_csv(archive / "samples.csv", float_precision="round_trip",
                           dtype={col: str for col in METADATA if col != "score"})
    pd.testing.assert_frame_equal(raw[METADATA + ["role"]], original[METADATA + ["role"]], check_exact=True)
    write_json(output / "data_provenance.json", provenance)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "catboost")}
    versions["xgboost"] = xgboost.__version__
    write_json(output / "run.json", dict(config=config, prior=str(prior), pilot=pilot, workers=workers,
               primary=dict(arm=PRIMARY, metric="srcc", population=MACRO), features=columns_by_arm,
               extend_from=str(extend_from) if extend_from is not None else None, fit_arms=list(fit_arms),
               code_sha256=sha256_file(Path(__file__)), python=platform.python_version(), versions=versions))
    pool = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    completed = []
    new_fits = 0
    for seed in settings["split_seeds"][:1] if pilot else settings["split_seeds"]:
        splits = grouped_splits(raw, pool, settings["outer_folds"], seed)
        for fold, (train, valid) in enumerate(splits, 1):
            if pilot and fold > 1:
                break
            print(f"Seed {seed}, fold {fold}", flush=True)
            name = f"seed_{seed}_fold_{fold:02d}"
            evaluate = np.concatenate([valid, external])
            training, evaluation, manifest = prepare_fold(raw, train, evaluate, archive / name)
            metadata = raw.loc[evaluate, METADATA]
            proof = {}
            if extend_from is None:
                previous = read_csv(archive / name, "predictions.csv.gz").query(
                    "pool == 'nr_ref' and track == 'vgg05_resnet50'"
                )
            else:
                reused, reused_fits, digest = reuse_fold(extend_from / name, manifest, metadata, len(train), seed, fold)
                proof["reused_verification_sha256"] = digest
                previous = None
            folder = output / name
            folder.mkdir()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_backend, backend, settings, fit_arms, training, evaluation,
                                           raw.loc[train, "score"], metadata, previous, archive / name, folder / backend)
                           for backend in BACKENDS]
                results = [future.result() for future in futures]
            predictions = pd.concat([rows for rows, _ in results], ignore_index=True).assign(split_seed=seed, fold=fold)
            fits = [fit for _, records in results for fit in records]
            new_fits += sum(fit["refit"] for fit in fits)
            if extend_from is not None:
                if refit_pairs:
                    keys = ["arm", "model", "sample_id"]
                    archived = reused.set_index(keys).prediction
                    fresh = predictions.set_index(keys).prediction
                    proof["max_refit_prediction_delta"] = float((fresh - archived.loc[fresh.index]).abs().max())
                    reused = reused.loc[~reused.arm.isin(fit_arms)]
                    reused_fits = [fit for fit in reused_fits if fit["arm"] not in fit_arms]
                predictions = pd.concat([reused, predictions], ignore_index=True)
                fits = reused_fits + fits
            path = folder / "predictions.csv.gz"
            predictions.to_csv(path, index=False)
            pd.testing.assert_frame_equal(read_csv(folder, path.name), predictions, check_exact=True)
            write_json(folder / "verification.json", dict(status="passed", fits=fits, archive_sha256=manifest,
                       prediction_rows=len(predictions), predictions_sha256=sha256_file(path), **proof))
            completed.append([seed, fold])
            write_json(output / "status.json", dict(status="running", completed_folds=completed))
            print(f"Verified seed {seed}, fold {fold}: {len(fits)} prediction sets", flush=True)
    require(len(completed) == (1 if pilot else n_full_folds), "Incomplete fold grid")
    write_json(output / "status.json", dict(status="complete", completed_folds=completed,
               prediction_sets=len(completed) * len(columns_by_arm) * len(BACKENDS), new_fits=new_fits))


def analyze(run_path, output, n_reps):
    status = read_json(run_path / "status.json")
    require(status["status"] == "complete", "Cannot analyze an incomplete run")
    features = read_json(run_path / "run.json")["features"]
    require(features in (feature_sets(), feature_sets(extend=True)), "Unexpected pseudo-FR feature sets")
    frames = []
    for seed, fold in status["completed_folds"]:
        path = run_path / f"seed_{seed}_fold_{fold:02d}"
        proof = read_json(path / "verification.json")
        require(proof["status"] == "passed" and proof["predictions_sha256"] == sha256_file(path / "predictions.csv.gz"),
                f"Unverified predictions: {path}")
        frames.append(read_csv(path, "predictions.csv.gz"))
    data = dict(tuple(pd.concat(frames, ignore_index=True).groupby(["arm", "model"])))
    require(set(data) == {(arm, backend) for arm in features for backend in BACKENDS}, "Incomplete feature arms")
    analysis = Analysis(n_reps)
    scores = {}
    for (arm, backend), rows in data.items():
        values = estimate(rows, n_reps)
        values[MACRO] = macro(values, MOS)
        analysis.record(analysis.cells, values, table="pseudo_fr", name=arm, model=backend, kind="score")
        scores[arm, backend] = values
    for arm in features:
        scores[arm, "backend_mean"] = analysis.mean_backend([scores[arm, b] for b in BACKENDS], "pseudo_fr", arm)
    bt_data = {key: rows.loc[rows.dataset.eq(BT)] for key, rows in data.items()}
    bt_summary, bt_folds = [], []
    for arm in features:
        if arm == BASELINE:
            continue
        for model in (*BACKENDS, "backend_mean"):
            values = analysis.pair(scores[BASELINE, model], scores[arm, model], "pseudo_fr", arm, model)
            if arm == PRIMARY and model == "backend_mean":
                analysis.test(values[MACRO], arm, MACRO, "pseudo_fr", model)
            summary, folds = paired_bt(bt_data, SimpleNamespace(name=arm, model=model), keys=((BASELINE,), (arm,)),
                                       n_folds=len(status["completed_folds"]), n_reps=n_reps)
            bt_summary.extend(dict(arm=arm, model=model, **row) for row in summary)
            bt_folds.append(folds.assign(arm=arm, model=model))
    output.mkdir()
    analysis.primary = [row for row in analysis.primary if row["metric"] == "srcc"]
    analysis.write(output, expected_primary=1)
    pd.DataFrame(bt_summary).to_csv(output / "bt_common_groups_deltas.csv", index=False)
    pd.concat(bt_folds, ignore_index=True).to_csv(output / "bt_common_groups_folds.csv", index=False)
    write_json(output / "status.json", dict(status="complete", bootstrap_replicates=n_reps, bootstrap_seed=SEED,
               n_folds=len(status["completed_folds"]), primary_tests=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fit = stages.add_parser("fit")
    for name in ("config", "prior", "output"):
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--pilot", action="store_true")
    fit.add_argument("--workers", type=int, default=2)
    fit.add_argument("--extend-from", type=Path, help="Reuse the completed SPAN run and add bicubic/RLFN six-metric arms")
    fit.add_argument("--refit-pairs", action="store_true",
                     help="Fit the two-metric bicubic/RLFN arms again using the SPAN archive for controls")
    stats = stages.add_parser("analyze")
    stats.add_argument("--run", type=Path, required=True)
    stats.add_argument("--output", type=Path, required=True)
    stats.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == "fit":
            run(args.config, args.prior, args.output, args.pilot, args.workers, args.extend_from, args.refit_pairs)
        else:
            analyze(args.run, args.output, args.replicates)


if __name__ == "__main__":
    main()
