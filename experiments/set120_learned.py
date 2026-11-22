"""Run the five matched learned feature pools on the original Set120-only folds."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS
from experiments.data import load_experiment_data, read_csv, require, sha256_file
from experiments.finetune_musiq import FOLD, SPEC, partition_table, partitions
from experiments.run_grouped import METADATA, feature_columns, write_json
from experiments.transfer import DISRQAD, ISRGEN, SET120, metric_rows
from experiments.transfer_features import (
    candidates_for,
    choose_candidate,
    feature_pools,
    fit_backend,
    predict_model,
    prepare,
    read_metadata,
)

KEY = FOLD + ["sample_id"]
POOLS = {name: spec for name, spec in feature_pools().items() if name != "legacy_original"}
POOLS["legacy"] = SPEC


def sorted_assignments(table):
    return table.sort_values(FOLD + ["role", "sample_id"], ignore_index=True)


def load_controls(root, raw, assignments, config):
    """Reuse archived predictions after verifying all targets and split IDs."""
    tabular = root / "musiq-full"
    run_info = json.loads((tabular / "run.json").read_text())
    prior_config = run_info["config"]
    require(prior_config["datasets"] == config["datasets"], "Historical dataset config differs")
    require(prior_config["resubmission"] == config["evaluation"], "Historical split config differs")
    require(run_info["columns"] == feature_columns(SPEC, run_info["fixed_dimensions"]),
            "Historical feature pool differs")
    require(run_info["params"] == {} and not run_info["pilot"], "Historical defaults differ")
    predictions = []
    fingerprints = {}
    expected = assignments.loc[assignments.role.eq("evaluate"), KEY]
    expected_index = pd.MultiIndex.from_frame(expected)
    tabular_hash = sha256_file(tabular / "predictions.csv.gz")
    for name in ("musiq-full", "arniqa-full", "unique-full", "paq2piq-full"):
        path = root / name
        proof = path / ("verification.json" if name == "musiq-full" else "analysis/verification.json")
        verification = json.loads(proof.read_text())
        require(verification["status"] == "passed" and verification["prediction_rows"] == len(expected),
                f"Unverified control {name}")
        if name != "musiq-full":
            require(verification["parent_tabular_predictions_sha256"] == tabular_hash,
                    f"Control tabular parent differs: {name}")
        prior = read_metadata(path / "samples.csv").set_index("sample_id", verify_integrity=True)
        require(set(prior.index) == set(raw.sample_id), f"Control sample coverage differs: {name}")
        pd.testing.assert_frame_equal(prior.loc[raw.sample_id].reset_index()[METADATA],
                                      raw[METADATA], check_exact=True)
        pd.testing.assert_frame_equal(sorted_assignments(read_csv(path, "partitions.csv.gz")),
                                      sorted_assignments(assignments))
        values = read_csv(path, "predictions.csv.gz").set_index(KEY, verify_integrity=True)
        require(len(values) == len(expected) and expected_index.isin(values.index).all(),
                f"Control prediction coverage differs: {name}")
        values = values.loc[expected_index]
        if name != "musiq-full":
            metric = name.removesuffix("-full")
            values = values[["direct", "finetuned"]].add_prefix(f"{metric}_")
        require(np.isfinite(values.drop(columns="qalign_direct", errors="ignore")).all().all()
                and not np.isinf(values).any().any(), f"Invalid control forecasts: {name}")
        predictions.append(values)
        fingerprints[name] = {file.name: sha256_file(file) for file in (
            path / "samples.csv", path / "partitions.csv.gz", path / "predictions.csv.gz", proof,
        )}
    return pd.concat(predictions, axis=1).reset_index(), fingerprints


def score(frame):
    table, bt = metric_rows(frame)
    external = table.loc[table.dataset.isin([DISRQAD, ISRGEN])]
    require(len(external) == 2, "External MOS datasets are required")
    table.loc[len(table)] = dict(dataset="external_mos_macro", score_type="mos", aggregation="mean_per_dataset",
                                 n_samples=external.n_samples.sum(), n_groups=2,
                                 plcc=external.plcc.mean(), srcc=external.srcc.mean())
    return table, bt


def verify_pool(output, folder, backend, pool, fit, raw, train, evaluate, splits, threads):
    """Replay saved outer models and recompute inner statistics from binary forecasts."""
    trials = read_csv(folder, "trials.csv")
    scores = read_csv(folder, "inner_scores.csv")
    candidates = candidates_for(backend, pool)
    require(trials.candidate.tolist() == list(range(len(candidates))), "Candidate coverage differs")
    with np.load(folder / "inner_predictions.npz") as archive:
        keys = set()
        for number, candidate in enumerate(candidates):
            trial = trials.iloc[number]
            dims = candidate["dimensions"] if POOLS[pool]["representations"] else {}
            require(json.loads(trial.params) == candidate["params"] and json.loads(trial.dimensions) == dims,
                    "Candidate settings differ")
            inner_srcc = []
            for name, _, _, valid in splits:
                key = f"c{number:02d}_{name}"
                keys.add(key)
                prediction = archive[key]
                target = raw.loc[valid, "score"]
                require(prediction.shape == (len(valid),) and np.isfinite(prediction).all(),
                        "Invalid inner forecasts")
                actual = scores.loc[scores.candidate.eq(number) & scores.partition.eq(name)]
                plcc, srcc = pearsonr(target, prediction).statistic, spearmanr(target, prediction).statistic
                require(len(actual) == 1, "Inner score coverage differs")
                np.testing.assert_allclose(actual[["plcc", "srcc"]].to_numpy()[0],
                                           [plcc, srcc], atol=1e-12, rtol=0)
                inner_srcc.append(srcc)
            np.testing.assert_allclose(trial.grouped, np.mean(inner_srcc), atol=1e-12, rtol=0)
        require(set(archive.files) == keys and len(scores) == len(keys), "Inner prediction coverage differs")
    winner = choose_candidate(trials, "grouped")
    require(fit["candidate"] == winner and fit["params"] == candidates[winner]["params"], "Wrong candidate selected")
    np.testing.assert_allclose(fit["selection_score"], trials.iloc[winner].grouped, atol=1e-12, rtol=0)
    require(fit["columns"] == feature_columns(POOLS[pool], fit["dimensions"]), "Fitted columns differ")
    transform = output / fit["transform"]
    transformer = joblib.load(transform / "transformer.joblib")
    require(transformer.fit_sample_ids_ == raw.loc[train, "sample_id"].tolist(), "Wrong outer fit IDs")
    require(fit["dimensions"] == json.loads(trials.iloc[winner].dimensions) == transformer.dimensions,
            "Fitted PCA dimensions differ")
    forecast = read_metadata(folder / "grouped_predictions.csv.gz")
    pd.testing.assert_frame_equal(forecast[METADATA], raw.loc[evaluate, METADATA].reset_index(drop=True),
                                  check_exact=True)
    with np.load(transform / "evaluation.npz") as matrix:
        require(matrix["indices"].tolist() == evaluate.tolist(), "Replay row order differs")
        inputs = pd.DataFrame(matrix["values"], columns=matrix["columns"])
        model = joblib.load(folder / fit["model"])
        values = predict_model(model, inputs[fit["columns"]], backend, threads)
    np.testing.assert_allclose(values, forecast.prediction, atol=1e-12, rtol=0)
    table, bt = score(forecast)
    pd.testing.assert_frame_equal(read_csv(folder, "grouped_metrics.csv"),
                                  table.loc[table.dataset.ne("external_mos_macro")].reset_index(drop=True),
                                  check_exact=False, atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(read_csv(folder, "grouped_bt_groups.csv"), bt.reset_index(drop=True),
                                  check_exact=False, atol=1e-12, rtol=0)
    return table, len(keys)


def run(config_path, controls_path, output, pilot):
    config = json.loads(config_path.read_text())
    settings = config["evaluation"]
    seed, threads = settings["model_seed"], settings["threads"]
    require((settings["outer_folds"], settings["inner_folds"], settings["split_seeds"], seed, threads)
            == (5, 3, [42, 43, 44], 42, 2), "Unexpected Set120 protocol settings")
    output.mkdir()
    print("Loading the four cached datasets and auditing source identities", flush=True)
    raw, blocks, scalars, provenance = load_experiment_data(config, config_path.resolve().parent, include_all_rr=True)
    folds = list(partitions(raw, settings, pilot=False))
    assignments = pd.concat([partition_table(raw, *fold) for fold in folds], ignore_index=True)
    controls, hashes = load_controls(controls_path, raw, assignments, config)
    folds = folds[:1] if pilot else folds
    fold_keys = pd.DataFrame([(s, f) for s, f, *_ in folds], columns=FOLD)
    controls = controls.merge(fold_keys, on=FOLD, validate="many_to_one")
    controls.to_csv(output / "controls.csv.gz", index=False)
    raw[METADATA].to_csv(output / "samples.csv", index=False)
    assignments.merge(fold_keys, on=FOLD, validate="many_to_one").to_csv(output / "partitions.csv.gz", index=False)
    write_json(output / "data_provenance.json", provenance)
    budgets = {b: {p: candidates_for(b, p) for p in POOLS} for b in BACKENDS}
    write_json(output / "run.json", dict(config=config, pilot=pilot, pools=POOLS, candidates=budgets,
               archived_controls=hashes, python=platform.python_version(), source_sha256=sha256_file(Path(__file__)),
               selection="Mean inner SRCC; ties by candidate then pool order. Archived controls excluded."))
    metrics, records, selections = [], [], []
    inner_checks = 0
    dimensions = sorted({(), *(tuple(c["dimensions"].items()) for backend in budgets.values()
                              for pool, candidates in backend.items() if POOLS[pool]["representations"]
                              for c in candidates)})
    for split_seed, fold, train, evaluate, inner in folds:
        tags = dict(split_seed=split_seed, fold=fold)
        folder = output / f"seed_{split_seed}_fold_{fold}"
        (folder / "transforms").mkdir(parents=True)
        require(raw.loc[train, "dataset"].eq(SET120).all(), "Non-Set120 training rows")
        splits = [(f"inner_{i}", "grouped", it, iv) for i, (it, iv) in enumerate(inner, 1)]
        prepared = {}
        for name, _, it, iv in [("outer", "outer", train, evaluate), *splits]:
            for number, dims in enumerate(dimensions):
                path = folder / "transforms" / f"{name}_d{number:02d}"
                tx, vx = prepare(raw, it, iv, blocks, scalars, dict(dims), seed, path)
                info = json.loads((path / "partition.json").read_text())
                fitted = joblib.load(path / "transformer.joblib")
                require(info["fit_sample_ids"] == fitted.fit_sample_ids_ == raw.loc[it, "sample_id"].tolist()
                        and info["evaluate"] == iv.tolist(), "Wrong preprocessing fit or evaluation IDs")
                prepared[(name, dims)] = (tx, vx, path.relative_to(output))
        for backend in BACKENDS:
            fits, _ = fit_backend(raw, train, evaluate, settings, backend, prepared, splits,
                                  folder / backend, POOLS, pool_specs=POOLS, strategies=("grouped",))
            tables = {}
            for fit in fits:
                pool = fit["pool"]
                tables[pool], checked = verify_pool(output, folder / backend / pool, backend, pool,
                                                   fit, raw, train, evaluate, splits, threads)
                inner_checks += checked
                records.append(dict(**tags, **fit))
                metrics.append(tables[pool].assign(**tags, model=backend, pool=pool))
            winner = max(fits, key=lambda fit: fit["selection_score"])
            selections.append(dict(**tags, **winner))
            metrics.append(tables[winner["pool"]].assign(**tags, model=backend, pool="selected"))
        archived = controls.loc[controls.split_seed.eq(split_seed) & controls.fold.eq(fold)]
        archived = archived.set_index("sample_id").loc[raw.loc[evaluate, "sample_id"]]
        for name in archived.columns.drop(FOLD):
            table, _ = score(raw.loc[evaluate, METADATA].assign(prediction=archived[name].to_numpy()))
            metrics.append(table.assign(**tags, model=name, pool="archived"))
        write_json(output / "fits.json", records)
        write_json(output / "selections.json", selections)
        table = pd.concat(metrics, ignore_index=True)
        table.to_csv(output / "metrics.csv", index=False)
        write_json(output / "status.json", dict(status="running", verified_folds=len(records) // 15,
                                               fitted_models=len(records), inner_forecasts=inner_checks))
        print(f"VERIFIED FOLD {split_seed}/{fold}: {len(records)} outer models, {inner_checks} inner forecasts",
              flush=True)
    require(len(records) == len(folds) * 15 and len(selections) == len(folds) * 3
            and inner_checks == len(records) * 24, "Unexpected experiment matrix size")
    summary = table.groupby(["model", "pool", "dataset"])[["plcc", "srcc"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(output / "metrics_summary.csv")
    write_json(output / "status.json", dict(status="complete", verified_folds=len(folds), fitted_models=len(records),
                                           inner_forecasts=inner_checks, selected_models=len(selections)))


def verify_pilot_parity(root):
    """Check the reused pilot fold against the completed full run without fitting."""
    pilot, full = root / "pilot", root / "full"
    for stage, folds in ((pilot, 1), (full, 15)):
        require(json.loads((stage / "status.json").read_text()) ==
                dict(status="complete", verified_folds=folds, fitted_models=15 * folds,
                     inner_forecasts=360 * folds, selected_models=3 * folds), f"Incomplete {stage.name}")
        require((root / f"{stage.name}.exit").read_text().strip() == "0", f"Failed {stage.name} fitting")
    original = json.loads((pilot / "run.json").read_text())
    replay = json.loads((full / "run.json").read_text())
    require(original.pop("pilot") and not replay.pop("pilot") and original == replay,
            "Pilot and full protocols differ")
    require(sha256_file(pilot / "samples.csv") == sha256_file(full / "samples.csv")
            and sha256_file(pilot / "data_provenance.json") == sha256_file(full / "data_provenance.json"),
            "Pilot and full data differ")

    def first_fold(table):
        return table.loc[table.split_seed.eq(42) & table.fold.eq(1)].reset_index(drop=True)

    for name in ("fits.json", "selections.json"):
        before = json.loads((pilot / name).read_text())
        after = json.loads((full / name).read_text())
        require(before == [row for row in after if (row["split_seed"], row["fold"]) == (42, 1)],
                f"Fold 42/1 fit decisions differ: {name}")
    for name in ("partitions.csv.gz", "controls.csv.gz"):
        pd.testing.assert_frame_equal(read_csv(pilot, name), first_fold(read_csv(full, name)), check_exact=True)

    atol = 1e-12  # Same absolute tolerance as the per-fold prediction and metric replay.
    prediction_rows, inner_vectors = 0, 0
    max_forecast_delta, max_inner_delta, max_inner_metric_delta = 0., 0., 0.
    cols = ["plcc", "srcc"]
    for backend in BACKENDS:
        for pool in POOLS:
            before = pilot / "seed_42_fold_1" / backend / pool
            after = full / before.relative_to(pilot)
            pd.testing.assert_frame_equal(read_csv(before, "trials.csv"), read_csv(after, "trials.csv"),
                                          check_exact=True)
            first, repeated = read_csv(before, "inner_scores.csv"), read_csv(after, "inner_scores.csv")
            pd.testing.assert_frame_equal(first.drop(columns=cols), repeated.drop(columns=cols), check_exact=True)
            np.testing.assert_allclose(first[cols], repeated[cols], atol=atol, rtol=0, equal_nan=False)
            max_inner_metric_delta = max(max_inner_metric_delta, float((first[cols] - repeated[cols]).abs().max().max()))
            p = read_metadata(before / "grouped_predictions.csv.gz")
            f = read_metadata(after / "grouped_predictions.csv.gz")
            pd.testing.assert_frame_equal(p[METADATA], f[METADATA], check_exact=True)
            np.testing.assert_allclose(p.prediction, f.prediction, atol=atol, rtol=0, equal_nan=False)
            prediction_rows += len(p)
            max_forecast_delta = max(max_forecast_delta, float((p.prediction - f.prediction).abs().max()))
            with np.load(before / "inner_predictions.npz") as a, np.load(after / "inner_predictions.npz") as b:
                require(set(a.files) == set(b.files) and len(a.files) == 24, "Inner forecast coverage differs")
                for key in a.files:
                    np.testing.assert_allclose(a[key], b[key], atol=atol, rtol=0, equal_nan=False)
                    max_inner_delta = max(max_inner_delta, float(np.max(np.abs(a[key] - b[key]))))
                    inner_vectors += 1

    p = read_csv(pilot, "metrics.csv")
    all_metrics = read_csv(full, "metrics.csv")
    f = first_fold(all_metrics)
    pd.testing.assert_frame_equal(p.drop(columns=cols), f.drop(columns=cols), check_exact=True)
    require(p[cols].isna().equals(f[cols].isna()), "Undefined metrics differ")
    np.testing.assert_allclose(p[cols], f[cols], rtol=0, atol=atol, equal_nan=True)
    require(len(p) == 180 and len(all_metrics) == 2700 and inner_vectors == 360, "Wrong metric or inner coverage")
    summary = all_metrics.groupby(["model", "pool", "dataset"])[cols].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    pd.testing.assert_frame_equal(read_csv(full, "metrics_summary.csv"), summary.reset_index(),
                                  check_exact=False, atol=atol, rtol=0)
    return dict(status="passed", absolute_tolerance=atol, matched_metric_rows=len(p),
                verified_full_metric_rows=len(all_metrics), matched_outer_predictions=prediction_rows,
                matched_inner_vectors=inner_vectors, max_outer_prediction_delta=max_forecast_delta,
                max_inner_prediction_delta=max_inner_delta, max_inner_metric_delta=max_inner_metric_delta,
                max_plcc_delta=float((p.plcc - f.plcc).abs().max()),
                max_srcc_delta=float((p.srcc - f.srcc).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.config, args.controls, args.output, args.pilot)


if __name__ == "__main__":
    main()
