"""Evaluate two fixed WASD additions on the verified 15-fold PCA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.analyze_bt_pairs import paired_bt
from experiments.analyze_statistics import (
    BACKENDS,
    BT,
    MACRO,
    MOS,
    SEED,
    Analysis,
    estimate,
    holm,
    macro,
    read,
)
from experiments.data import STAT_COLUMNS, load_experiment_data, require, sha256_file
from experiments.main_feature_fusion import (
    DEFAULT_DIMS,
    MAX_DIMS,
    POOLS,
    checked_prior,
    fit_predict,
    reuse_transform,
)
from experiments.run_grouped import METADATA, feature_columns, grouped_splits, write_json
from experiments.transfer import metric_rows
from experiments.transfer_features import predict_model, read_metadata

ARCHIVED = {"nr_ref": "without_wasd", "all_rr_no_fr": "with_wasd"}
ARMS = ("without_wasd", "with_wasd")
TRACK = "vgg05_resnet05"
REPLICATES = 10_000


def columns_for(pool, arm):
    spec = POOLS[pool]
    if arm == "without_wasd":
        scalars = [name for name in spec["scalars"] if name not in STAT_COLUMNS]
    else:
        scalars = spec["scalars"] if pool == "all_rr_no_fr" else spec["scalars"] + STAT_COLUMNS
    return feature_columns({**spec, "scalars": scalars}, DEFAULT_DIMS)


def fit_fold(raw, settings, train, valid, external, prior, assignments, output, seed, fold):
    require(pd.Index(train).is_unique and pd.Index(valid).is_unique, "Duplicate partition rows")
    for partition, ids in (("train", train), ("heldout", valid)):
        rows = assignments.loc[assignments.partition.eq(partition), "sample_id"]
        require(rows.is_unique and set(rows) == set(raw.loc[ids, "sample_id"]), "Archived split differs")
    evaluate = np.concatenate((valid, external))
    require(pd.Index(evaluate).is_unique, "Duplicate evaluation rows")
    (tx, vx), hashes = reuse_transform(raw, train, evaluate, prior / "outer", MAX_DIMS)
    meta = raw.loc[evaluate, METADATA].reset_index(drop=True)
    meta["evaluation"] = np.where(raw.loc[evaluate, "role"].eq("train"), "internal", "external")
    archive_path = prior / "predictions.csv.gz"
    archive = read_metadata(archive_path).query("track == @TRACK")
    require(set(archive.model) == set(BACKENDS) and set(archive.pool) == set(ARCHIVED)
            and archive.split_seed.eq(seed).all() and archive.fold.eq(fold).all(), "Unexpected archived fold")

    output.mkdir()
    forecasts, metrics, fits = [], [], []
    for pool, archived_arm in ARCHIVED.items():
        new_arm = next(arm for arm in ARMS if arm != archived_arm)
        old_columns, new_columns = columns_for(pool, archived_arm), columns_for(pool, new_arm)
        require([c for c in columns_for(pool, "with_wasd") if c not in STAT_COLUMNS] ==
                columns_for(pool, "without_wasd"), "Non-WASD columns differ")
        for backend in BACKENDS:
            print(f"{output.name} {pool}/{backend}/{new_arm}", flush=True)
            source_path = prior / backend / pool / TRACK
            parameters_path = source_path / "parameters.json"
            saved = json.loads(parameters_path.read_text())
            require(saved["columns"] == old_columns, "Archived control columns differ")
            control = archive.loc[archive.pool.eq(pool) & archive.model.eq(backend)]
            require(len(control) == len(meta), "Incorrect control row count")
            control = control.set_index("sample_id", verify_integrity=True).loc[meta.sample_id].reset_index()
            pd.testing.assert_frame_equal(control[meta.columns], meta, check_exact=True)
            control_path = source_path / "model.joblib"
            control_replay = predict_model(joblib.load(control_path), vx[old_columns], backend, settings["threads"])
            np.testing.assert_allclose(control_replay, control.prediction, atol=1e-12, rtol=0)
            model, prediction = fit_predict(backend, {}, settings, tx[new_columns], vx[new_columns],
                                            raw.loc[train, "score"], None)
            actual = {key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
                      for key, value in model.get_params().items()}
            require(actual == saved["actual"], "Head parameters differ across the pair")
            model_path = output / f"{pool}_{backend}.joblib"
            joblib.dump(model, model_path, compress=3)
            replay = predict_model(joblib.load(model_path), vx[new_columns], backend, settings["threads"])
            np.testing.assert_allclose(replay, prediction, atol=1e-12, rtol=0)
            fits.append(dict(pool=pool, model=backend, new_arm=new_arm, model_file=model_path.name,
                             control_arm=archived_arm, parameters=actual, columns=new_columns,
                             control_model_sha256=sha256_file(control_path), new_model_sha256=sha256_file(model_path),
                             control_max_error=float(np.max(np.abs(control_replay - control.prediction))),
                             new_model_max_error=float(np.max(np.abs(replay - prediction)))))
            for arm, values in ((archived_arm, control.prediction.to_numpy()), (new_arm, prediction)):
                labels = dict(pool=pool, model=backend, arm=arm, split_seed=seed, fold=fold)
                frame = meta.assign(prediction=values)
                forecasts.append(frame.assign(**labels))
                scores, _ = metric_rows(frame)
                metrics.append(scores.assign(**labels))
    predictions = pd.concat(forecasts, ignore_index=True)
    path = output / "predictions.csv.gz"
    predictions.to_csv(path, index=False)
    pd.testing.assert_frame_equal(read_metadata(path), predictions, check_exact=True)
    pd.concat(metrics, ignore_index=True).to_csv(output / "metrics.csv", index=False)
    write_json(output / "verification.json", dict(
        status="passed", seed=seed, fold=fold, new_fits=len(fits), reused_controls=len(fits), fits=fits,
        prediction_rows=len(predictions), transform_sha256=hashes, control_forecasts_sha256=sha256_file(archive_path),
        predictions_sha256=sha256_file(path),
    ))
    return len(fits), len(predictions)


def analyze(run, output):
    paths = sorted(run.glob("seed_*_fold_*/predictions.csv.gz"))
    status = json.loads((run / "status.json").read_text())
    require(status["status"] == "fits_verified" and len(paths) == status["folds"] == 15,
            "Need all 15 verified fold pairs")
    for path in paths:
        proof = json.loads((path.parent / "verification.json").read_text())
        require(proof["status"] == "passed" and sha256_file(path) == proof["predictions_sha256"],
                f"Prediction hash differs: {path}")
    predictions = pd.concat([read_metadata(path) for path in paths], ignore_index=True)
    require(len(predictions) == status["prediction_rows"], "Prediction count differs from verified matrix")
    output.mkdir()
    analysis = Analysis(REPLICATES)
    bt = predictions.loc[predictions.dataset.eq(BT)]
    bt_arms = dict(tuple(bt.groupby(["pool", "arm", "model"])))
    bt_rows, bt_folds = [], []
    for pool in ARCHIVED:
        print(f"Scene bootstrap for {pool}", flush=True)
        subset = predictions.loc[predictions.pool.eq(pool) & predictions.dataset.isin(MOS)]
        scores = {}
        for arm in ARMS:
            cells = {}
            for backend in BACKENDS:
                panel = subset.loc[subset.arm.eq(arm) & subset.model.eq(backend)]
                cells[backend] = estimate(panel, REPLICATES)
                cells[backend][MACRO] = macro(cells[backend], MOS)
                analysis.record(analysis.cells, cells[backend], table=pool, name=arm, model=backend, kind="score")
            cells["backend_mean"] = analysis.mean_backend(list(cells.values()), pool, arm)
            scores[arm] = cells
        left, right = ARMS
        for backend in (*BACKENDS, "backend_mean"):
            result = analysis.pair(scores[left][backend], scores[right][backend], pool, "wasd_addition", backend)
            if backend == "backend_mean":
                analysis.test(result[MACRO], "wasd_addition", MACRO, pool, backend)
            label = SimpleNamespace(name="wasd_addition", model=backend)
            rows, folds = paired_bt(bt_arms, label, keys=((pool, left), (pool, right)), n_reps=REPLICATES)
            bt_rows.extend(dict(pool=pool, model=backend, **row) for row in rows)
            bt_folds.append(folds.assign(pool=pool, model=backend))
    analysis.primary = [row for row in analysis.primary if row["metric"] == "srcc"]
    analysis.write(output, expected_primary=2)
    primary = read(output / "primary_tests.csv")
    primary["p_holm"] = holm(primary.p_raw)
    primary.to_csv(output / "primary_tests.csv", index=False)
    pd.DataFrame(bt_rows).to_csv(output / "bt_common_groups_deltas.csv", index=False)
    pd.concat(bt_folds, ignore_index=True).to_csv(output / "bt_common_groups_folds.csv", index=False)
    write_json(output / "status.json", dict(status="complete", replicates=REPLICATES, seed=SEED, primary_tests=2))
    print(primary[["table", "conditional_point", "ci_low", "ci_high", "p_holm"]].to_string(index=False), flush=True)


def run(config_path, verified_pca, output):
    config = json.loads(config_path.read_text())
    settings = config["evaluation"]
    require((settings["outer_folds"], settings["split_seeds"], settings["model_seed"], settings["threads"])
            == (5, [42, 43, 44], 42, 2) and tuple(settings["models"]) == BACKENDS, "Unexpected evaluation protocol")
    output.mkdir()
    write_json(output / "status.json", dict(status="loading_inputs", folds=0))
    print("Loading cached features and checking source provenance", flush=True)
    raw, _, _, provenance = load_experiment_data(config, config_path.resolve().parent, include_all_rr=True)
    prior, assignments = checked_prior(raw, provenance, verified_pca)
    require(config == json.loads((prior / "run.json").read_text())["config"], "Archived paper config differs")
    write_json(output / "data_provenance.json", provenance)
    write_json(output / "run.json", dict(
        config=config, backends=BACKENDS, dimensions=DEFAULT_DIMS, shared_basis_dimensions=MAX_DIMS,
        archived_arms=ARCHIVED, archive_track=TRACK, verified_pca=verified_pca,
        archive_proof_sha256=sha256_file(verified_pca / "complete-verified/verification.json"),
        feature_columns={pool: {arm: columns_for(pool, arm) for arm in ARMS} for pool in ARCHIVED},
        fit_weighting="ordinary rows", primary_metric="MOS macro SRCC averaged across backends",
        bootstrap_replicates=REPLICATES, source_sha256=sha256_file(Path(__file__)),
        inference="retrospective and conditional on saved fits",
    ))
    internal = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    folds = count = rows = 0
    for seed in settings["split_seeds"]:
        for fold, (train, valid) in enumerate(grouped_splits(raw, internal, settings["outer_folds"], seed), 1):
            key = f"seed_{seed}_fold_{fold:02d}"
            reference = assignments.loc[assignments.split_seed.eq(seed) & assignments.fold.eq(fold)]
            write_json(output / "status.json", dict(status="pilot" if not folds else "fitting",
                                                   folds=folds, active=[seed, fold]))
            fits, predictions = fit_fold(raw, settings, train, valid, external, prior / key, reference,
                                         output / key, seed, fold)
            require(fits == 6 and predictions == 12 * (len(valid) + len(external)), "Incomplete fold")
            folds += 1
            count += fits
            rows += predictions
            if folds == 1:
                write_json(output / "pilot_verification.json", dict(
                    status="passed", split_seed=seed, fold=fold, new_fits=fits, reused_controls=fits,
                    prediction_rows=predictions,
                    proof_sha256=sha256_file(output / key / "verification.json"),
                ))
                print("Pilot verified. Continuing with the remaining 14 folds.", flush=True)
    require(folds == 15 and count == 90, "Incomplete WASD matrix")
    summary = dict(folds=folds, new_fits=count, reused_controls=count, prediction_rows=rows)
    write_json(output / "status.json", dict(status="fits_verified", **summary))
    analyze(output, output / "analysis")
    write_json(output / "status.json", dict(status="complete", **summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "verified-pca", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.config, args.verified_pca, args.output)
