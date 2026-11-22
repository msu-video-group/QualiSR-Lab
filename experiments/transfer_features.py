"""Matched full-source transfer with inner-selected feature pools and dimensions.

The original transfer runner is unchanged. This experiment compares five pools
with the existing eight-candidate budget, plus an exact three-candidate legacy
control. External targets never enter selection. Inner forecasts are retained
in binary form so every selection score can be independently recomputed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS
from experiments.data import (
    FR_METRICS,
    NR_COLUMNS,
    SIGLIP_COLUMNS,
    STAT_COLUMNS,
    load_experiment_data,
    read_csv,
    require,
    sha256_file,
)
from experiments.fold_features import FoldTransformer
from experiments.presets import search_candidates
from experiments.run_grouped import METADATA, feature_columns, grouped_splits, make_model, write_json
from experiments.transfer import (
    ISRGEN,
    MOS_DATASETS,
    REALSRQ,
    SET120,
    check_partition,
    metric_rows,
    transfer_partitions,
)
from qualisr.regressors import validation_correlations

STRATEGIES = ("grouped", "transfer_aware")


def feature_pools():
    legacy = dict(scalars=NR_COLUMNS + [f"{m}_rlfn" for m in FR_METRICS], representations=["sr"])
    return {
        "nr": dict(scalars=NR_COLUMNS, representations=[]),
        "nr_ref": dict(scalars=NR_COLUMNS, representations=["ref"]),
        "nr_ref_diff": dict(scalars=NR_COLUMNS, representations=["ref", "diff"]),
        "all_rr_no_fr": dict(scalars=NR_COLUMNS + STAT_COLUMNS + SIGLIP_COLUMNS,
                             representations=["sr", "ref", "diff"]),
        "legacy": legacy,
        "legacy_original": legacy,
    }


def candidates_for(backend, pool):
    if pool == "legacy_original":
        return search_candidates(backend, False)[:3]
    return search_candidates(backend, bool(feature_pools()[pool]["representations"]))


def selection_partitions(raw, train, settings):
    result = [(f"grouped_{i:02d}", "grouped", it, iv)
              for i, (it, iv) in enumerate(grouped_splits(raw, train, settings["inner_folds"],
                                                        settings["model_seed"]), 1)]
    datasets = [d for d in MOS_DATASETS if raw.loc[train, "dataset"].eq(d).any()]
    if len(datasets) > 1:
        for dataset in datasets:
            heldout = raw.loc[train, "dataset"].eq(dataset).to_numpy()
            result.append((f"lodo_{dataset}", "lodo", train[~heldout], train[heldout]))
    return result


def selection_scores(scores):
    require(np.isfinite(scores.srcc).all(), "Undefined inner correlation; folds must not be omitted")
    grouped = float(scores.loc[scores.kind.eq("grouped"), "srcc"].mean())
    lodo = scores.loc[scores.kind.eq("lodo"), "srcc"]
    require(np.isfinite(grouped) and np.isfinite(lodo).all(), "Undefined training selection score")
    return dict(grouped=grouped, transfer_aware=(grouped + float(lodo.mean())) / 2 if len(lodo) else grouped)


def choose_candidate(table, strategy):
    # Stable, explicit tie rule: earlier recorded candidate.
    return int(table.sort_values([strategy, "candidate"], ascending=[False, True]).candidate.iat[0])


def predict_model(model, values, backend, threads):
    # CatBoost prediction defaults to all CPUs, independently of its fit setting.
    if backend == "catboost":
        return model.predict(values, thread_count=threads)
    return model.predict(values)


def prepare(raw, train, evaluate, blocks, scalars, dimensions, seed, folder):
    check_partition(raw, train, evaluate)
    folder.mkdir()
    fitted = FoldTransformer(scalars, blocks if dimensions else {}, dimensions, seed=seed)
    tx = fitted.fit_transform(raw.loc[train])
    vx = fitted.transform(raw.loc[evaluate])
    joblib.dump(fitted, folder / "transformer.joblib", compress=3)
    write_json(folder / "partition.json", dict(train=train.tolist(), evaluate=evaluate.tolist(),
                                               fit_sample_ids=fitted.fit_sample_ids_, dimensions=dimensions))
    if folder.name.startswith("outer_"):
        np.savez_compressed(folder / "evaluation.npz", values=vx.to_numpy(), columns=vx.columns.to_numpy(dtype=str),
                            indices=evaluate)
    return tx, vx


def fit_backend(raw, train, evaluate, settings, backend, prepared, splits, folder, pool_names,
                *, pool_specs=None, strategies=STRATEGIES):
    folder.mkdir(exist_ok=True)
    fit_rows, all_metrics = [], []
    pool_specs = feature_pools() if pool_specs is None else pool_specs
    for pool in pool_names:
        spec = pool_specs[pool]
        started = time.monotonic()
        path = folder / pool
        path.mkdir()
        candidates = candidates_for(backend, pool)
        forecasts, scores, aggregate = {}, [], []
        for number, candidate in enumerate(candidates):
            dims = candidate["dimensions"] if spec["representations"] else {}
            columns = feature_columns(spec, dims)
            for name, kind, it, iv in splits:
                if pool == "legacy_original" and kind == "lodo":
                    continue
                tx, vx, _ = prepared[(name, tuple(dims.items()))]
                model = make_model(backend, candidate["params"], settings["model_seed"], settings["threads"])
                model.fit(tx[columns], raw.loc[it, "score"])
                prediction = np.asarray(predict_model(model, vx[columns], backend, settings["threads"]), dtype=np.float64)
                require(np.isfinite(prediction).all(), "Nonfinite inner forecast")
                plcc, srcc, _, _ = validation_correlations(raw.loc[iv, "score"], prediction, raw.loc[iv])
                scores.append(dict(candidate=number, partition=name, kind=kind, plcc=plcc, srcc=srcc))
                forecasts[f"c{number:02d}_{name}"] = prediction
            table = pd.DataFrame(scores)
            aggregate.append(dict(candidate=number, params=json.dumps(candidate["params"], sort_keys=True),
                                  dimensions=json.dumps(dims, sort_keys=True),
                                  **selection_scores(table.loc[table.candidate.eq(number)])))
            print(f"{folder.parent.name}/{backend}/{pool}: candidate {number + 1}/{len(candidates)}", flush=True)
        np.savez_compressed(path / "inner_predictions.npz", **forecasts)
        pd.DataFrame(scores).to_csv(path / "inner_scores.csv", index=False)
        trials = pd.DataFrame(aggregate)
        trials.to_csv(path / "trials.csv", index=False)
        fitted = {}
        for strategy in (("grouped",) if pool == "legacy_original" else strategies):
            number = choose_candidate(trials, strategy)
            candidate = candidates[number]
            dims = candidate["dimensions"] if spec["representations"] else {}
            columns = feature_columns(spec, dims)
            tx, vx, transform_path = prepared[("outer", tuple(dims.items()))]
            if number not in fitted:
                model = make_model(backend, candidate["params"], settings["model_seed"], settings["threads"])
                model.fit(tx[columns], raw.loc[train, "score"])
                prediction = np.asarray(predict_model(model, vx[columns], backend, settings["threads"]), dtype=np.float64)
                require(np.isfinite(prediction).all(), "Nonfinite external forecast")
                joblib.dump(model, path / f"model_{number:02d}.joblib", compress=3)
                fitted[number] = (model, prediction)
            model, prediction = fitted[number]
            actual = model.get_params()
            for key, value in candidate["params"].items():
                require(actual[key] == value, f"Model ignored {key}")
            frame = raw.loc[evaluate, METADATA].copy().assign(prediction=prediction)
            frame.to_csv(path / f"{strategy}_predictions.csv.gz", index=False)
            metrics, groups = metric_rows(frame)
            metrics.to_csv(path / f"{strategy}_metrics.csv", index=False)
            groups.to_csv(path / f"{strategy}_bt_groups.csv", index=False)
            record = dict(backend=backend, pool=pool, strategy=strategy, candidate=number,
                          selection_score=float(trials.loc[trials.candidate.eq(number), strategy].iat[0]),
                          params=candidate["params"], dimensions=dims, columns=columns,
                          transform=str(transform_path), model=f"model_{number:02d}.joblib")
            write_json(path / f"{strategy}_fit.json", record)
            fit_rows.append(record)
            all_metrics.append(metrics.assign(backend=backend, pool=pool, strategy=strategy))
        print(f"{folder.parent.name}/{backend}/{pool}: done in {time.monotonic() - started:.1f}s", flush=True)
    return fit_rows, pd.concat(all_metrics, ignore_index=True)


def run(config, base, output, *, pilot=False, workers=1, data=None):
    output.mkdir(parents=True)
    settings = config["evaluation"]
    require(settings["model_seed"] == 42, "This protocol fixes model/selection seed 42")
    require(1 <= workers <= 3 and settings["threads"] == 2, "Use 1–3 workers and two threads per model")
    raw, blocks, scalars, provenance = load_experiment_data(config, base, include_all_rr=True) if data is None else data
    require(set(raw.dataset) == {*MOS_DATASETS, REALSRQ}, "All four paper datasets are required")
    raw.to_csv(output / "samples.csv.gz", index=False, columns=METADATA + ["role", "content_sha256", "content_kind"])
    write_json(output / "data_provenance.json", provenance)
    partitions = list(transfer_partitions(raw))
    if pilot:
        partitions = [p for p in partitions if p[0] in (SET120, f"{SET120}+{ISRGEN}")]
    run_info = dict(config=config, pilot=pilot, workers=workers, pools=feature_pools(),
                    candidates={b: {p: candidates_for(b, p) for p in feature_pools()} for b in BACKENDS},
                    cases=[p[0] for p in partitions], python=platform.python_version(),
                    versions={**{n: importlib.metadata.version(n) for n in
                                 ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib")},
                              "xgboost_module": xgboost.__version__},
                    code_sha256=sha256_file(Path(__file__)),
                    primary="grouped: mean SRCC across three grouped inner folds; MOS datasets equal within fold",
                    sensitivity="transfer_aware: equal mean of grouped and leave-one-training-dataset-out SRCC; "
                                "equals grouped for one training dataset",
                    legacy_control="original three regressor candidates, shared paired PCA-5 and grouped-only selection",
                    ties="earlier candidate, then earlier feature pool; no external-label selection")
    write_json(output / "run.json", run_info)
    fit_records, metric_records = [], []
    for case, train, evaluate in partitions:
        started = time.monotonic()
        case_path = output / case
        case_path.mkdir()
        splits = selection_partitions(raw, train, settings)
        write_json(case_path / "partitions.json", dict(train=train.tolist(), evaluate=evaluate.tolist(),
                  inner={name: dict(kind=kind, train=it.tolist(), evaluate=iv.tolist()) for name, kind, it, iv in splits}))
        dimensions = {()}
        for backend in BACKENDS:
            for pool, spec in feature_pools().items():
                if spec["representations"]:
                    dimensions.update(tuple(c["dimensions"].items()) for c in candidates_for(backend, pool))
        prepared = {}
        (case_path / "transforms").mkdir()
        for name, _, it, iv in [("outer", "outer", train, evaluate), *splits]:
            for number, dims in enumerate(sorted(dimensions)):
                folder = case_path / "transforms" / f"{name}_d{number:02d}"
                print(f"{case}: preparing {folder.name}", flush=True)
                tx, vx = prepare(raw, it, iv, blocks, scalars, dict(dims), settings["model_seed"], folder)
                prepared[(name, dims)] = (tx, vx, folder.relative_to(output))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = [executor.submit(fit_backend, raw, train, evaluate, settings, backend,
                                       prepared, splits, case_path / backend, [pool])
                       for backend in BACKENDS for pool in feature_pools()]
            for future in pending:
                fits, metrics = future.result()
                fit_records.extend(dict(case=case, **f) for f in fits)
                metric_records.append(metrics.assign(case=case))
        direct = []
        for metric in NR_COLUMNS:
            frame = raw.loc[evaluate, METADATA].copy().assign(prediction=raw.loc[evaluate, metric])
            frame.to_csv(case_path / f"direct_{metric}.csv.gz", index=False)
            metrics, _ = metric_rows(frame)
            direct.append(metrics.assign(case=case, backend="direct", pool=metric, strategy="frozen"))
        metric_records.extend(direct)
        pd.concat(metric_records, ignore_index=True).to_csv(output / "metrics.csv", index=False)
        write_json(output / "fits.json", fit_records)
        write_json(output / "status.json", dict(status="running", completed_cases=case,
                                                elapsed_case_seconds=time.monotonic() - started))
        print(f"COMPLETED CASE {case}: {time.monotonic() - started:.1f}s", flush=True)
    write_json(output / "status.json", dict(status="complete", cases=len(partitions), fitted_rows=len(fit_records)))


def read_metadata(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={c: str for c in METADATA if c != "score"})


def verify(root, destination=None):
    """Recompute selection, metrics, coverage and preprocessing fit provenance."""
    require(json.loads((root / "status.json").read_text())["status"] == "complete", "Incomplete run")
    run_info = json.loads((root / "run.json").read_text())
    samples = read_metadata(root / "samples.csv.gz")
    require(samples.sample_id.is_unique, "Duplicate sample IDs")
    fits = json.loads((root / "fits.json").read_text())
    expected_cases = list(transfer_partitions(samples))
    if run_info["pilot"]:
        expected_cases = [p for p in expected_cases if p[0] in (SET120, f"{SET120}+{ISRGEN}")]
    require(run_info["cases"] == [p[0] for p in expected_cases], "Missing transfer conditions")
    all_metrics, selections, checks = [], [], []
    observed = {(f["case"], f["backend"], f["pool"], f["strategy"]): f for f in fits}
    require(len(observed) == len(fits), "Duplicate fits")
    for case, train, evaluate in expected_cases:
        path = root / case
        saved = json.loads((path / "partitions.json").read_text())
        require(saved["train"] == train.tolist() and saved["evaluate"] == evaluate.tolist(), "Changed outer IDs")
        check_partition(samples, train, evaluate)
        expected_inner = selection_partitions(samples, train, run_info["config"]["evaluation"])
        require(set(saved["inner"]) == {x[0] for x in expected_inner}, "Missing inner folds")
        for name, kind, it, iv in expected_inner:
            check_partition(samples, it, iv)
            require(saved["inner"][name] == dict(kind=kind, train=it.tolist(), evaluate=iv.tolist()), "Changed inner IDs")
        for transform_path in (path / "transforms").iterdir():
            info = json.loads((transform_path / "partition.json").read_text())
            name = transform_path.name.rsplit("_d", 1)[0]
            ids = train if name == "outer" else np.array(saved["inner"][name]["train"])
            require(info["train"] == ids.tolist(), "Transform partition differs")
            transformer = joblib.load(transform_path / "transformer.joblib")
            expected_ids = samples.loc[ids, "sample_id"].tolist()
            require(info["fit_sample_ids"] == expected_ids == transformer.fit_sample_ids_, "Transform fit IDs differ")
        for backend in BACKENDS:
            for pool in feature_pools():
                folder = path / backend / pool
                trials = read_csv(folder, "trials.csv")
                scores = read_csv(folder, "inner_scores.csv")
                candidates = run_info["candidates"][backend][pool]
                require(trials.candidate.tolist() == list(range(len(candidates))), "Candidate budget differs")
                with np.load(folder / "inner_predictions.npz") as archive:
                    expected_keys = set()
                    for number, candidate in enumerate(candidates):
                        measured = []
                        row = trials.iloc[number]
                        dims = candidate["dimensions"] if feature_pools()[pool]["representations"] else {}
                        require(json.loads(row.params) == candidate["params"] and json.loads(row.dimensions) == dims,
                                "Trial settings differ")
                        for name, kind, _, iv in expected_inner:
                            if pool == "legacy_original" and kind == "lodo":
                                continue
                            key = f"c{number:02d}_{name}"
                            expected_keys.add(key)
                            prediction = archive[key]
                            require(prediction.shape == (len(iv),) and np.isfinite(prediction).all(), "Invalid inner forecast")
                            plcc, srcc, _, _ = validation_correlations(samples.loc[iv, "score"], prediction, samples.loc[iv])
                            record = scores.loc[scores.candidate.eq(number) & scores.partition.eq(name)]
                            require(len(record) == 1 and record.kind.iat[0] == kind, "Inner score coverage differs")
                            np.testing.assert_allclose(record[["plcc", "srcc"]].to_numpy()[0], [plcc, srcc], atol=1e-12, rtol=0)
                            measured.append(dict(kind=kind, srcc=srcc))
                        expected = selection_scores(pd.DataFrame(measured))
                        np.testing.assert_allclose([row.grouped, row.transfer_aware], list(expected.values()), atol=1e-12, rtol=0)
                    require(set(archive.files) == expected_keys, "Extra/missing inner forecasts")
                require(len(scores) == len(expected_keys), "Extra inner scores")
                for strategy in (("grouped",) if pool == "legacy_original" else STRATEGIES):
                    fit = observed.pop((case, backend, pool, strategy))
                    number = choose_candidate(trials, strategy)
                    require(fit["candidate"] == number and fit["params"] == candidates[number]["params"], "Wrong selected candidate")
                    np.testing.assert_allclose(fit["selection_score"], trials.iloc[number][strategy], atol=1e-12, rtol=0)
                    expected_dims = candidates[number]["dimensions"] if feature_pools()[pool]["representations"] else {}
                    require(fit["dimensions"] == expected_dims, "Wrong selected dimensions")
                    require(fit["columns"] == feature_columns(feature_pools()[pool], fit["dimensions"]), "Wrong selected columns")
                    require((folder / fit["model"]).is_file(), "Missing fitted model")
                    prediction = read_metadata(folder / f"{strategy}_predictions.csv.gz")
                    pd.testing.assert_frame_equal(prediction[METADATA], samples.loc[evaluate, METADATA].reset_index(drop=True), check_exact=True)
                    require(np.isfinite(prediction.prediction).all(), "Nonfinite final predictions")
                    model = joblib.load(folder / fit["model"])
                    with np.load(root / fit["transform"] / "evaluation.npz") as matrix:
                        require(matrix["indices"].tolist() == evaluate.tolist(), "Replay row order differs")
                        inputs = pd.DataFrame(matrix["values"], columns=matrix["columns"])
                        np.testing.assert_allclose(predict_model(model, inputs[fit["columns"]], backend,
                                                                run_info["config"]["evaluation"]["threads"]),
                                                   prediction.prediction,
                                                   atol=1e-12, rtol=0)
                    metrics, groups = metric_rows(prediction)
                    pd.testing.assert_frame_equal(read_csv(folder, f"{strategy}_metrics.csv"), metrics.reset_index(drop=True),
                                                  check_exact=False, atol=1e-12, rtol=0)
                    saved_groups = read_csv(folder, f"{strategy}_bt_groups.csv")
                    pd.testing.assert_frame_equal(saved_groups, groups.reset_index(drop=True), check_exact=False, atol=1e-12, rtol=0)
                    all_metrics.append(metrics.assign(case=case, backend=backend, pool=pool, strategy=strategy))
                    selections.append(dict(case=case, backend=backend, pool=pool, strategy=strategy,
                                           candidate=number, selection_score=float(trials.iloc[number][strategy])))
                    checks.append(dict(case=case, backend=backend, pool=pool, strategy=strategy, rows=len(prediction)))
            # Feature-pool selection is confined to the recorded training score.
            for strategy in STRATEGIES:
                options = [s for s in selections if s["case"] == case and s["backend"] == backend
                           and s["strategy"] == strategy and s["pool"] != "legacy_original"]
                winner = max(options, key=lambda x: x["selection_score"])
                selections.append(dict(winner, pool="selected", selected_pool=winner["pool"]))
                selected_metrics = next(
                    x for x in all_metrics if x.case.iat[0] == case and x.backend.iat[0] == backend
                    and x.pool.iat[0] == winner["pool"] and x.strategy.iat[0] == strategy)
                all_metrics.append(selected_metrics.assign(pool="selected"))
        for metric in NR_COLUMNS:
            prediction = read_metadata(path / f"direct_{metric}.csv.gz")
            pd.testing.assert_frame_equal(prediction[METADATA], samples.loc[evaluate, METADATA].reset_index(drop=True), check_exact=True)
            require(not np.isinf(prediction.prediction).any(), "Infinite direct score")
            metrics, _ = metric_rows(prediction)
            all_metrics.append(metrics.assign(case=case, backend="direct", pool=metric, strategy="frozen"))
    require(not observed, "Unexpected final fits")
    summary = pd.concat(all_metrics, ignore_index=True)
    destination = root / "verified" if destination is None else destination
    destination.mkdir()
    summary.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(destination / "selections.csv", index=False)
    learned = summary.loc[summary.backend.isin(BACKENDS)]
    means = learned.groupby(["case", "pool", "strategy", "dataset", "score_type"], sort=False).agg(
        plcc=("plcc", "mean"), srcc=("srcc", "mean"), backends=("backend", "nunique")).reset_index()
    require(means.backends.eq(3).all(), "Missing backend in summary")
    means.to_csv(destination / "backend_mean.csv", index=False)
    write_json(destination / "verification.json", dict(status="passed", cases=len(expected_cases),
               final_fits=len(checks), final_prediction_rows=sum(c["rows"] for c in checks),
               checks=["exact source/target and grouped/LODO partitions", "every transform fit ID",
                       "every inner correlation from binary forecasts", "candidate budgets, parameters and winners",
                       "final coverage, targets and MOS/BT metrics", "training-only feature-pool selection"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--pilot", action="store_true")
    fit.add_argument("--workers", type=int, default=1)
    check = commands.add_parser("verify")
    check.add_argument("root", type=Path)
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.root, args.output)
        return
    config = json.loads(args.config.read_text())
    with threadpool_limits(limits=config["evaluation"]["threads"]):
        run(config, args.config.resolve().parent, args.output, pilot=args.pilot, workers=args.workers)


if __name__ == "__main__":
    main()
