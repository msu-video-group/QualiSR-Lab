"""Nested feature fusion with balanced model loss and a prediction ensemble."""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

from experiments.data import load_experiment_data, read_csv, require, sha256_file
from experiments.run_grouped import METADATA, feature_columns, grouped_splits, make_model, write_json
from experiments.transfer import check_partition, metric_rows
from experiments.transfer_features import feature_pools, predict_model, read_metadata
from qualisr.regressors import validation_correlations

BACKENDS = ("ridge", "randomforest", "xgb", "catboost")
POOLS = {name: feature_pools()[name] for name in ("nr", "nr_ref", "nr_ref_diff", "all_rr_no_fr")}
WEIGHT_MODES = ("rows", "partial", "balanced")
DEFAULT_DIMS = {"vgg": 5, "resnet": 5}
MAX_DIMS = {"vgg": 96, "resnet": 96}
SEARCH_SEED = 20261116
SPACES = {
    "randomforest": dict(n_estimators=[300, 500, 800], max_depth=[6, 10, 20, None],
                         min_samples_leaf=[1, 2, 4, 8], max_features=[.5, .7, 1.]),
    "xgb": dict(max_depth=[2, 3, 4, 6], learning_rate=[.02, .05, .1], n_estimators=[200, 500, 800],
                subsample=[.7, 1.], colsample_bytree=[.7, 1.], reg_alpha=[0., .01, .1],
                reg_lambda=[.1, 1., 10.], min_child_weight=[1, 3, 10]),
    "catboost": dict(depth=[3, 4, 6, 8], learning_rate=[.02, .05, .1],
                     l2_leaf_reg=[1., 3., 10., 30.], iterations=[250, 500, 800, 1000]),
}


def candidates(backend, pool):
    if backend == "ridge":
        params = [{}] + [{"alpha": alpha} for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 10., 100., 1e4)]
    else:
        params = [{}, *ParameterSampler(SPACES[backend], n_iter=7, random_state=SEARCH_SEED)]
    rng = np.random.default_rng(SEARCH_SEED)
    # Use the same PCA schedule across embedding pools and model families.
    levels = (5, 25, 50, 96)
    vgg = rng.permutation([*levels, 5, 25])
    resnet = rng.permutation([*levels, 25, 50])
    dims = [DEFAULT_DIMS, *(dict(vgg=int(v), resnet=int(r)) for v, r in zip(vgg, resnet, strict=True)), MAX_DIMS]
    return [
        dict(candidate=w * len(params) + i, weight=weight, params=setting,
             dimensions=dim if POOLS[pool]["representations"] else {})
        for w, weight in enumerate(WEIGHT_MODES)
        for i, (setting, dim) in enumerate(zip(params, dims, strict=True))
    ]


def training_weights(rows):
    by_dataset = rows.groupby("dataset", sort=False)
    sources_per_dataset = by_dataset["source_group"].transform("nunique")
    outputs_per_source = rows.groupby(["dataset", "source_group"], sort=False)["sample_id"].transform("size")
    equal = 1.0 / (sources_per_dataset * outputs_per_source)
    equal = equal.to_numpy() / equal.mean()
    return dict(rows=np.ones(len(rows)), partial=(1. + equal) / 2., balanced=equal)


def checked_prior(raw, provenance, prior):
    complete = prior / "complete"
    proof = json.loads((prior / "complete-verified/verification.json").read_text())
    require(proof["status"] == "passed" and proof["folds"] == 15, "Need the verified 15-fold PCA run")
    previous = read_metadata(complete / "samples.csv")
    pd.testing.assert_frame_equal(previous[METADATA], raw[METADATA], check_exact=True)
    previous_provenance = json.loads((complete / "data_provenance.json").read_text())
    require(provenance["files"] == previous_provenance["files"] and
            provenance["source_content_audit"] == previous_provenance["source_content_audit"],
            "Source or feature cache differs from the verified PCA run")
    return complete, read_csv(complete, "fold_assignments.csv")


def reuse_transform(raw, train, evaluate, path, dimensions):
    check_partition(raw, train, evaluate)
    record = json.loads((path / "partition.json").read_text())
    transform = joblib.load(path / "transformer.joblib")
    require(record["train"] == train.tolist() and record["evaluate"] == evaluate.tolist(),
            f"Archived partition changed: {path}")
    require(record["fit_sample_ids"] == transform.fit_sample_ids_ == raw.loc[train, "sample_id"].tolist()
            and record["dimensions"] == transform.dimensions == dimensions, f"Wrong fit IDs or dimensions: {path}")
    with np.load(path / "evaluation.npz") as archive:
        require(np.array_equal(archive["indices"], evaluate), f"Wrong projected evaluation IDs: {path}")
        require(archive["columns"].tolist() == transform.feature_names_out_, f"Wrong projected columns: {path}")
        valid_x = pd.DataFrame(archive["values"], columns=archive["columns"], index=evaluate)
    train_x = transform.transform(raw.loc[train])
    return (train_x, valid_x), {name: sha256_file(path / name)
                              for name in ("partition.json", "transformer.joblib", "evaluation.npz")}


def make_head(backend, params, settings):
    if backend == "ridge":
        return Ridge(**params)
    return make_model(backend, params, settings["model_seed"], settings["threads"])


def fit_predict(backend, params, settings, tx, vx, target, weights):
    model = make_head(backend, params, settings).fit(tx, target, sample_weight=weights)
    values = np.asarray(predict_model(model, vx, backend, settings["threads"]), dtype=np.float64)
    require(np.isfinite(values).all(), f"Nonfinite {backend} prediction")
    return model, values


def inner_metrics(raw, indices, prediction):
    metadata = raw.loc[indices, METADATA]
    plcc, srcc, datasets, _ = validation_correlations(metadata.score, prediction, metadata)
    require(np.isfinite(datasets[["plcc", "srcc"]]).all().all(), "Undefined inner MOS correlation")
    return np.array([plcc, srcc])


def inner_replay(raw, indices, prediction):
    targets = raw.loc[indices, ["dataset", "score"]].assign(prediction=prediction)
    return np.mean([(pearsonr(rows.score, rows.prediction).statistic,
                     spearmanr(rows.score, rows.prediction).statistic)
                    for _, rows in targets.groupby("dataset", sort=True)], axis=0)


def inner_alignment(train_ids, inner):
    """Resolve inner row positions once before fitting in worker threads."""
    train_index = pd.Index(train_ids)
    require(train_index.is_unique, "Duplicate outer-training rows")
    split_ids = [vx.index.to_numpy() for _, vx, _ in inner]
    positions = [train_index.get_indexer(ids) for ids in split_ids]
    require(np.array_equal(np.sort(np.concatenate(positions)), np.arange(len(train_index))),
            "Each outer-training sample must have exactly one inner prediction")
    return split_ids, positions


def fit_pool(raw, settings, backend, pool, budget, outer, legacy, inner, alignment, folder):
    started = time.monotonic()
    folder.mkdir()
    train_x, eval_x, train_weights = outer
    train_ids = train_x.index
    oof = np.full((len(budget), len(train_ids)), np.nan)
    split_ids, positions = alignment
    history, scores = [], []
    spec = POOLS[pool]
    for i, trial in enumerate(budget):
        columns = feature_columns(spec, trial["dimensions"])
        fold_metrics = []
        for split, ((tx, vx, weights), ix, p) in enumerate(zip(inner, split_ids, positions, strict=True), 1):
            loss_weights = None if trial["weight"] == "rows" else weights[trial["weight"]]
            _, forecast = fit_predict(backend, trial["params"], settings, tx[columns], vx[columns],
                                      raw.loc[tx.index, "score"], loss_weights)
            oof[i, p] = forecast
            coefficients = inner_metrics(raw, ix, forecast)
            fold_metrics.append(coefficients)
            scores.append(dict(candidate=trial["candidate"], inner_fold=split, plcc=coefficients[0],
                               srcc=coefficients[1]))
        plcc, srcc = np.mean(fold_metrics, axis=0)
        history.append(dict(candidate=trial["candidate"], weight=trial["weight"], plcc=plcc, srcc=srcc,
                            params=json.dumps(trial["params"], sort_keys=True),
                            dimensions=json.dumps(trial["dimensions"], sort_keys=True)))
        print(f"{folder.parent.parent.name} {pool}/{backend} candidate {i + 1}/{len(budget)} "
              f"({time.monotonic() - started:.1f}s)", flush=True)
    np.savez_compressed(folder / "out_of_fold.npz", indices=train_ids.to_numpy(),
                        candidates=[t["candidate"] for t in budget], predictions=oof)
    pd.DataFrame(scores).to_csv(folder / "inner_scores.csv", index=False)
    trials = pd.DataFrame(history)
    trials.to_csv(folder / "trials.csv", index=False)

    with np.load(folder / "out_of_fold.npz") as saved:
        require(np.array_equal(saved["indices"], train_ids), "Saved OOF row order differs")
        require(np.array_equal(saved["candidates"], trials.candidate), "Saved OOF candidates differ")
        replay = np.array([inner_replay(raw, ix, prediction[p]) for prediction in saved["predictions"]
                           for ix, p in zip(split_ids, positions, strict=True)])
    np.testing.assert_allclose(replay, read_csv(folder, "inner_scores.csv")[["plcc", "srcc"]],
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(replay.reshape(len(budget), len(inner), 2).mean(axis=1),
                               read_csv(folder, "trials.csv")[["plcc", "srcc"]], atol=1e-12, rtol=0)
    # idxmax keeps the first candidate in budget order when scores tie.
    best = int(trials.srcc.idxmax())
    best_modes = {f"best_{mode}": int(rows.srcc.idxmax()) for mode, rows in trials.groupby("weight", sort=False)}
    tracks = {"fixed": 0, **best_modes, "selected": best}
    forecasts, fitted, fits = {}, {}, []
    for track, i in tracks.items():
        trial = budget[i]
        if i not in fitted:
            columns = feature_columns(spec, trial["dimensions"])
            loss_weights = None if trial["weight"] == "rows" else train_weights[trial["weight"]]
            model, forecast = fit_predict(backend, trial["params"], settings, train_x[columns], eval_x[columns],
                                          raw.loc[train_ids, "score"], loss_weights)
            model_path = folder / f"candidate_{trial['candidate']:02d}.joblib"
            joblib.dump(model, model_path, compress=3)
            np.testing.assert_allclose(predict_model(joblib.load(model_path), eval_x[columns], backend,
                                                     settings["threads"]), forecast, atol=1e-12, rtol=0)
            fitted[i] = forecast, model_path.name
        forecasts[track], model_name = fitted[i]
        fits.append(dict(track=track, **trial, model=model_name, basis="shared_96"))

    if spec["representations"]:
        lx, lv = legacy
        columns = feature_columns(spec, DEFAULT_DIMS)
        model, forecast = fit_predict(backend, {}, settings, lx[columns], lv[columns], raw.loc[train_ids, "score"], None)
        model_path = folder / "legacy_basis_fixed.joblib"
        joblib.dump(model, model_path, compress=3)
        np.testing.assert_allclose(predict_model(joblib.load(model_path), lv[columns], backend,
                                                 settings["threads"]), forecast, atol=1e-12, rtol=0)
        forecasts["legacy_basis_fixed"] = forecast
        fits.append(dict(track="legacy_basis_fixed", **budget[0], model=model_path.name, basis="legacy_10_50"))

    write_json(folder / "fits.json", fits)
    return forecasts, oof[best], len(fitted) + bool(spec["representations"])


def simplex_grid():
    weights = [(a, b, c, d) for a, b, c, d in itertools.product(range(5), repeat=len(BACKENDS))
               if a + b + c + d == 4]
    return np.asarray([(1, 1, 1, 1), *(w for w in weights if w != (1, 1, 1, 1))]) / 4.


def fit_ensemble(raw, alignment, selected_oof, selected_outer, folder):
    grid = simplex_grid()
    validation_ids, positions = alignment
    scores, replay = [], []
    for weights in grid:
        prediction = selected_oof @ weights
        scores.append(np.mean([inner_metrics(raw, ids, prediction[p])
                               for ids, p in zip(validation_ids, positions, strict=True)], axis=0))
        replay.append(np.mean([inner_replay(raw, ids, prediction[p])
                               for ids, p in zip(validation_ids, positions, strict=True)], axis=0))
    table = pd.DataFrame(grid, columns=BACKENDS).assign(plcc=np.asarray(scores)[:, 0], srcc=np.asarray(scores)[:, 1])
    winner = table.srcc.idxmax()
    table["selected"] = table.index == winner
    table.to_csv(folder / "ensemble_search.csv", index=False)
    np.testing.assert_allclose(replay, read_csv(folder, "ensemble_search.csv")[["plcc", "srcc"]],
                               atol=1e-12, rtol=0)
    write_json(folder / "ensemble.json", dict(backends=BACKENDS, grid_resolution=.25,
               selected=int(winner), weights=grid[winner].tolist(), inner_srcc=float(table.srcc.iat[winner])))
    return {"equal": selected_outer @ grid[0], "selected": selected_outer @ grid[winner]}


def control_parity(raw, forecasts, reference, pool, evaluate):
    if pool not in ("nr_ref", "all_rr_no_fr"):
        return 0.
    maximum = 0.
    for backend, tracks in forecasts.items():
        if backend == "ridge":
            continue
        for track, archived in (("fixed", "vgg05_resnet05"), ("legacy_basis_fixed", "legacy_basis_5_5")):
            rows = reference.loc[reference.pool.eq(pool) & reference.model.eq(backend) & reference.track.eq(archived)]
            rows = rows.set_index("sample_id", verify_integrity=True).loc[raw.loc[evaluate, "sample_id"]]
            delta = np.max(np.abs(rows.prediction.to_numpy() - tracks[track]))
            require(delta <= 1e-12, f"Archived default replay differs for {backend}/{pool}/{track}: {delta}")
            maximum = max(maximum, float(delta))
    return maximum


def fit_fold(raw, settings, train, evaluate, inner, legacy, prior, target, workers, pilot):
    target.mkdir()
    pools = ("nr", "nr_ref") if pilot else POOLS
    default = read_metadata(prior / "predictions.csv.gz")
    transform_x, transform_v = train
    weights = training_weights(raw.loc[transform_x.index])
    outer = transform_x, transform_v, weights
    inner_weighted = [(tx, vx, training_weights(raw.loc[tx.index])) for tx, vx in inner]
    alignment = inner_alignment(transform_x.index, inner_weighted)
    for tx, vx in (train, legacy, *inner):
        require(tx.columns.is_unique and vx.columns.is_unique, "Duplicate projected feature columns")
    weight_records = [pd.DataFrame(weights).assign(role="outer", sample_id=raw.loc[transform_x.index, "sample_id"].to_numpy())]
    for i, (tx, _, weights) in enumerate(inner_weighted, 1):
        weight_records.append(pd.DataFrame(weights).assign(role=f"inner_{i}",
                              sample_id=raw.loc[tx.index, "sample_id"].to_numpy()))
    pd.concat(weight_records, ignore_index=True).to_csv(target / "training_weights.csv.gz", index=False)
    metrics, counts = [], dict(inner_fits=0, outer_models=0, ensemble_weights=0, max_default_parity_error=0.)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for pool in pools:
            folder = target / pool
            folder.mkdir()
            tasks = {}
            for backend in BACKENDS:
                budget = candidates(backend, pool)
                if pilot:
                    # Verify both PCA bases and every fit API; exercise all three weights on the NR pilot.
                    budget = budget[:1] if pool == "nr_ref" else [t for t in budget if t["candidate"] % 8 == 0]
                counts["inner_fits"] += len(budget) * len(inner)
                tasks[backend] = executor.submit(fit_pool, raw, settings, backend, pool, budget, outer,
                                                 legacy, inner_weighted, alignment, folder / backend)
            forecasts, oof = {}, []
            for backend, future in tasks.items():
                forecasts[backend], prediction, n_models = future.result()
                oof.append(prediction)
                counts["outer_models"] += n_models
            parity = control_parity(raw, forecasts, default, pool, evaluate)
            counts["max_default_parity_error"] = max(counts["max_default_parity_error"], parity)
            selected = np.column_stack([forecasts[b]["selected"] for b in BACKENDS])
            forecasts["ensemble"] = fit_ensemble(raw, alignment, np.column_stack(oof), selected, folder)
            counts["ensemble_weights"] += len(simplex_grid())
            results = raw.loc[evaluate, METADATA].copy()
            bt_groups = []
            for model, tracks in forecasts.items():
                for track, prediction in tracks.items():
                    results[f"{model}_{track}"] = prediction
                    scored, bt = metric_rows(results[METADATA].assign(prediction=prediction))
                    metrics.append(scored.assign(pool=pool, model=model, track=track))
                    bt_groups.append(bt.assign(model=model, track=track))
            results.to_csv(folder / "predictions.csv.gz", index=False)
            pd.concat(bt_groups, ignore_index=True).to_csv(folder / "bt_groups.csv.gz", index=False)
            print(f"{target.name} verified {pool}", flush=True)
    return pd.concat(metrics, ignore_index=True), counts


def restore_completed(source, output, manifest, folds):
    """Carry verified folds into a fresh run, leaving any partial fold behind."""
    old_manifest = json.loads((source / "run.json").read_text())
    original_source = old_manifest.pop("source_sha256")
    require(old_manifest == {k: v for k, v in manifest.items() if k != "source_sha256"},
            "Resume source used a different experiment protocol or runtime")
    for name in ("samples.csv", "data_provenance.json"):
        require(sha256_file(source / name) == sha256_file(output / name), f"Resume input differs: {name}")
    source_full = source / "full"
    status = json.loads((source_full / "status.json").read_text())
    checks = read_csv(source_full, "verification_by_fold.csv")
    keys = list(checks[["split_seed", "fold"]].itertuples(index=False, name=None))
    require(status["status"] == "running" and 0 < status["folds"] == len(keys) < len(folds)
            and keys == folds[:len(keys)] and tuple(status["last_completed"]) == keys[-1],
            "Resume source has no contiguous verified fold prefix")
    require(checks.inner_fits.eq(1152).all() and checks.ensemble_weights.eq(140).all()
            and checks.max_default_parity_error.between(0, 1e-12).all(), "Incomplete fold verification")
    table = read_csv(source_full, "metrics.csv")
    sizes = table.groupby(["split_seed", "fold"], sort=False).size()
    require(list(sizes.index) == keys and sizes.eq(500).all()
            and not table.duplicated(["split_seed", "fold", "pool", "model", "track", "dataset"]).any(),
            "Wrong saved metric coverage for verified folds")
    proof = json.loads((source / "pilot/verification.json").read_text())
    require(proof["status"] == "passed" and proof["inner_fits"] == 48 and proof["ensemble_weights"] == 70,
            "Resume source pilot did not pass")
    full = output / "full"
    full.mkdir()
    shutil.copytree(source / "pilot", output / "pilot", copy_function=os.link)
    for seed, fold in keys:
        name = f"seed_{seed}_fold_{fold:02d}"
        shutil.copytree(source_full / name, full / name, copy_function=os.link)
    shutil.copyfile(source_full / "metrics.csv", full / "metrics.csv")
    shutil.copyfile(source_full / "verification_by_fold.csv", full / "verification_by_fold.csv")
    write_json(output / "resume.json", dict(source=source, carried_folds=keys,
               original_source_sha256=original_source,
               original_fold_checks_sha256=sha256_file(source_full / "verification_by_fold.csv"),
               original_fold_metrics_sha256=sha256_file(source_full / "metrics.csv")))
    write_json(full / "status.json", dict(status="running", folds=len(keys), last_completed=keys[-1],
                                         active=folds[len(keys)]))
    return [table], checks.to_dict("records")


def run(config_path, verified_pca, output, workers, resume_from=None):
    config = json.loads(config_path.read_text())
    settings = config["evaluation"]
    protocol = (settings["outer_folds"], settings["inner_folds"], settings["split_seeds"],
                settings["model_seed"], settings["threads"])
    require(protocol == (5, 3, [42, 43, 44], 42, 2) and workers in (1, 2), "Unexpected fusion protocol")
    output.mkdir()
    print("Loading cached features and verifying original inputs", flush=True)
    raw, _, _, provenance = load_experiment_data(config, config_path.resolve().parent, include_all_rr=True)
    prior, prior_assignments = checked_prior(raw, provenance, verified_pca)
    raw[METADATA].to_csv(output / "samples.csv", index=False)
    write_json(output / "data_provenance.json", provenance)
    write_json(output / "run.json", dict(config=config, feature_pools=POOLS, candidate_seed=SEARCH_SEED,
               budgets={p: {b: candidates(b, p) for b in BACKENDS} for p in POOLS}, workers=workers,
               verified_pca=verified_pca, pca_proof_sha256=sha256_file(verified_pca / "complete-verified/verification.json"),
               package_versions={name: importlib.metadata.version(name) for name in
                                 ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "xgboost-cpu", "joblib")},
               source_sha256=sha256_file(Path(__file__)), weight_modes=WEIGHT_MODES,
               selection="Mean inner dataset-macro SRCC. Ties go to the first candidate, then ensemble vector order."))
    indices = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    full = output / "full"
    metrics, counts = [], []
    if resume_from is not None:
        fold_order = list(itertools.product(settings["split_seeds"], range(1, settings["outer_folds"] + 1)))
        manifest = json.loads((output / "run.json").read_text())
        metrics, counts = restore_completed(resume_from, output, manifest, fold_order)
        print(f"Preserved {len(counts)} verified fold(s); starting at {fold_order[len(counts)]}", flush=True)
    completed = {(row["split_seed"], row["fold"]) for row in counts}
    for seed in settings["split_seeds"]:
        for fold, (train, valid) in enumerate(grouped_splits(raw, indices, settings["outer_folds"], seed), 1):
            if (seed, fold) in completed:
                continue
            name = f"seed_{seed}_fold_{fold:02d}"
            print(f"{name} verifying original group partitions and bases", flush=True)
            require(shutil.disk_usage(output).free > 20 * 1024**3, "Insufficient free space for fusion outputs")
            reference = prior_assignments.loc[prior_assignments.split_seed.eq(seed) & prior_assignments.fold.eq(fold)]
            for partition, selected in (("train", train), ("heldout", valid)):
                require(set(reference.loc[reference.partition.eq(partition), "sample_id"]) ==
                        set(raw.loc[selected, "sample_id"]), "Original outer split differs")
            evaluate = np.concatenate((valid, external))
            fold_dir = prior / name
            outer, hashes = reuse_transform(raw, train, evaluate, fold_dir / "outer", MAX_DIMS)
            legacy, legacy_hashes = reuse_transform(raw, train, evaluate, fold_dir / "legacy",
                                                    settings["max_pca_dimensions"])
            inner, inner_hashes = [], {}
            for i, (it, iv) in enumerate(grouped_splits(raw, train, settings["inner_folds"], seed), 1):
                projected, basis_hashes = reuse_transform(raw, it, iv, fold_dir / f"inner_{i}", MAX_DIMS)
                inner.append(projected)
                inner_hashes[f"inner_{i}"] = basis_hashes

            if not counts:
                pilot_metrics, pilot_counts = fit_fold(raw, settings, outer, evaluate, inner, legacy, fold_dir,
                                                       output / "pilot", workers, pilot=True)
                require(pilot_counts["inner_fits"] == 48 and pilot_counts["ensemble_weights"] == 70,
                        "Incomplete launch pilot")
                pilot_metrics.to_csv(output / "pilot" / "metrics.csv", index=False)
                write_json(output / "pilot" / "verification.json", dict(status="passed", **pilot_counts))
                print("Verified launch pilot. Starting full 15-fold matrix.", flush=True)
                full.mkdir()
                write_json(full / "status.json", dict(status="running", folds=0, active=[seed, fold]))
            destination = full / name
            table, checked = fit_fold(raw, settings, outer, evaluate, inner, legacy, fold_dir,
                                      destination, workers, pilot=False)
            write_json(destination / "transform_hashes.json", dict(outer=hashes, legacy=legacy_hashes, **inner_hashes))
            metrics.append(table.assign(split_seed=seed, fold=fold))
            counts.append(dict(split_seed=seed, fold=fold, **checked))
            pd.concat(metrics, ignore_index=True).to_csv(full / "metrics.csv", index=False)
            pd.DataFrame(counts).to_csv(full / "verification_by_fold.csv", index=False)
            write_json(full / "status.json", dict(status="running", folds=len(counts), last_completed=[seed, fold]))
    result = pd.concat(metrics, ignore_index=True)
    total_inner = sum(row["inner_fits"] for row in counts)
    total_ensemble = sum(row["ensemble_weights"] for row in counts)
    require(len(counts) == 15 and total_inner == 17280 and total_ensemble == 2100, "Wrong full matrix size")
    summary = result.groupby(["pool", "model", "track", "dataset"])[["plcc", "srcc"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(full / "metrics_summary.csv")
    write_json(full / "status.json", dict(status="complete", folds=len(counts), inner_fits=total_inner,
               ensemble_weights=total_ensemble, outer_models=sum(row["outer_models"] for row in counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "verified-pca", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume-from", type=Path, help="Verified fold prefix in a previous run's results directory")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.config, args.verified_pca, args.output, args.workers, args.resume_from)


if __name__ == "__main__":
    main()
