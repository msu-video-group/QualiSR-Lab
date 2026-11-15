"""Run grouped feature-family comparisons, PCA sweeps and preprocessing controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from experiments.data import (
    FR_COLUMNS,
    GT_COLUMNS,
    NR_COLUMNS,
    SIGLIP_COLUMNS,
    STAT_COLUMNS,
    load_experiment_data,
)
from experiments.fold_features import FoldTransformer
from experiments.presets import experiment_specs, search_candidates
from experiments.run_grouped import (
    METADATA,
    feature_columns,
    grouped_splits,
    make_model,
    tune_model,
    write_json,
)
from experiments.transfer import control_ids
from qualisr.regressors import validation_correlations

PCA_GRID = (2, 5, 10, 25, 50, 75, 96)
FIXED = {"vgg": 5, "resnet": 5}
RESNET50 = {"vgg": 5, "resnet": 50}


def append(table, path):
    table.to_csv(path, index=False, mode="a", header=not path.exists())


def run(args):
    output = args.output
    output.mkdir()
    config = json.loads(args.config.read_text())
    settings = config["evaluation"]
    seed, threads = settings["model_seed"], settings["threads"]
    models = settings["models"]
    prior = pd.read_csv(args.prior / "fold_assignments.csv", float_precision="round_trip")
    raw, blocks, scalars, provenance = load_experiment_data(config, args.config.parent, include_all_rr=True)
    previous_files = json.loads((args.prior / "data_provenance.json").read_text())["files"]
    fingerprints = {(item["dataset"], item["path"]): item["sha256"] for item in provenance["files"]}
    for item in previous_files:
        if fingerprints.get((item["dataset"], item["path"])) != item["sha256"]:
            raise ValueError(f"Changed cached input since the reference run: {item['path']}")
    meta = raw[METADATA + ["role", "gt_path", "lr_path", "sr_path", "declared_scale", "content_sha256", "content_kind"]]
    original = pd.read_csv(args.prior / "samples.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(meta[original.columns], original, check_exact=True)
    controls = control_ids(raw, pd.read_csv(args.control_flags))
    meta.to_csv(output / "samples.csv", index=False)
    write_json(output / "data_provenance.json", provenance)
    raw.groupby("dataset")[scalars + GT_COLUMNS].agg(lambda x: int(np.isfinite(x).sum())).to_csv(
        output / "finite_feature_coverage.csv"
    )

    full = {"name": "all_rr", "scalars": scalars, "representations": ["sr", "ref", "diff"]}
    specs = {s["name"]: s for s in experiment_specs()}
    variants = {
        "all_rr": full,
        "all_rr_no_wasd": {**full, "scalars": NR_COLUMNS + FR_COLUMNS + SIGLIP_COLUMNS},
        "all_rr_no_fr": {**full, "scalars": NR_COLUMNS + STAT_COLUMNS + SIGLIP_COLUMNS},
        "all_rr_no_siglip": {**full, "scalars": NR_COLUMNS + FR_COLUMNS + STAT_COLUMNS},
        "all_rr_no_qalign": {**full, "scalars": [name for name in scalars if name != "qalign"]},
    }
    write_json(output / "run.json", {
        "config": config, "prior": str(args.prior), "pilot": args.pilot, "stage": args.stage,
        "variants": variants, "controls": sorted(controls), "pca_grid": PCA_GRID,
        "tuning": {name: search_candidates(name, True) for name in models},
    })

    pool = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    def run_fold(split_seed, fold, train, valid):
        print(f"Seed {split_seed}, fold {fold}, stage {args.stage}", flush=True)
        assignment = prior.loc[(prior.split_seed == split_seed) & (prior.fold == fold)]
        for name, ids in (("train", train), ("heldout", valid)):
            assert set(assignment.loc[assignment.partition.eq(name), "sample_id"]) == set(raw.loc[ids, "sample_id"])
        append(assignment, output / "fold_assignments.csv")
        evaluate = np.concatenate([valid, external])
        evaluation = raw.loc[evaluate, METADATA].copy()
        evaluation["evaluation"] = np.where(evaluation.index.isin(valid), "internal", "external")
        fit_rows, metric_rows, prediction_rows = [], [], []

        def record_fit(ids, case):
            append(raw.loc[ids, ["sample_id"]].assign(split_seed=split_seed, fold=fold, case=case),
                   output / "fit_ids.csv.gz")

        def evaluate_fit(experiment, name, track, train_x, eval_x, *, params=None, cohort="released",
                         n_components=None):
            print(f"  {experiment}/{name}/{track}", flush=True)
            train_y = raw.loc[train_x.index, "score"]
            model = Ridge(alpha=1.0) if name == "ridge" else make_model(name, params or {}, seed, threads)
            model.fit(train_x, train_y)
            prediction = model.predict(eval_x)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite prediction")
            key = {"split_seed": split_seed, "fold": fold, "experiment": experiment, "model": name, "track": track}
            fit_rows.append({**key, "n_fit": len(train_x), "n_predict": len(eval_x), "cohort": cohort,
                             "n_components": n_components, "columns": "|".join(train_x.columns),
                             "params": json.dumps(model.get_params(), default=str, sort_keys=True)})
            pred = evaluation.loc[eval_x.index].assign(prediction=prediction, **key)
            prediction_rows.append(pred)
            for scope, group in pred.groupby("evaluation"):
                _, _, datasets, _ = validation_correlations(group.score, group.prediction, group)
                metric_rows.extend(datasets.assign(evaluation=scope, **key).to_dict("records"))

        record_fit(train, "released")
        if args.stage == "pca":
            for family, (columns, _) in blocks.items():
                rank_bound = min(len(train) - 1, len(columns))
                maximum = min(max(PCA_GRID), rank_bound)
                pca = PCA(n_components=maximum, svd_solver="randomized", random_state=seed).fit(
                    raw.loc[train, columns]
                )
                rank = int((pca.singular_values_ > pca.singular_values_[0] *
                            max(len(train), len(columns)) * np.finfo(pca.singular_values_.dtype).eps).sum())
                append(pd.DataFrame([{"family": family, "split_seed": split_seed, "fold": fold,
                                     "rank_bound": rank_bound, "numerical_rank_within_basis": rank}]),
                       output / "pca_rank.csv")
                columns_out = [f"{family}_sr_{i:03d}" for i in range(maximum)]
                train_z = pca.transform(raw.loc[train, columns])
                scaler = MinMaxScaler().fit(train_z)
                tx = pd.DataFrame(scaler.transform(train_z), index=train, columns=columns_out)
                vx = pd.DataFrame(scaler.transform(pca.transform(raw.loc[evaluate, columns])),
                                  index=evaluate, columns=columns_out)
                grid = (2, 96) if args.pilot else PCA_GRID
                for count in grid:
                    if count <= rank:
                        for name in models:
                            evaluate_fit(f"{family}_sr", name, f"pca_{count}", tx.iloc[:, :count],
                                         vx.iloc[:, :count], n_components=count)
        else:
            def prepare(ids, transform_ids, columns=scalars, scaler="minmax"):
                transformer = FoldTransformer(columns, blocks, settings["max_pca_dimensions"], seed=seed,
                                              scaler=scaler)
                return transformer.fit_transform(raw.loc[ids]), transformer.transform(raw.loc[transform_ids])

            tx, vx = prepare(train, evaluate)
            std_tx, std_vx = prepare(train, evaluate, scaler="standard")
            ordinary_train = train[~raw.loc[train, "sample_id"].isin(controls)]
            ordinary_eval = evaluate[~raw.loc[evaluate, "sample_id"].isin(controls)]
            ox, ov = prepare(ordinary_train, ordinary_eval)
            record_fit(ordinary_train, "ordinary")
            inner = []
            for i, (it, iv) in enumerate(grouped_splits(raw, train, settings["inner_folds"], split_seed), 1):
                record_fit(it, f"inner_{i}_train")
                record_fit(iv, f"inner_{i}_validation")
                ix, ivx = prepare(it, iv)
                inner.append((ix, raw.loc[it, "score"], ivx, raw.loc[iv, "score"], raw.loc[iv, METADATA]))
            for name in models:
                for experiment, spec in variants.items():
                    columns = feature_columns(spec, FIXED)
                    evaluate_fit(experiment, name, "fixed", tx[columns], vx[columns])
                columns = feature_columns(full, FIXED)
                evaluate_fit("all_rr", name, "standard", std_tx[columns], std_vx[columns])
                evaluate_fit("all_rr", name, "ordinary", ox[columns], ov[columns], cohort="ordinary")
                for spec in (full, specs["nr_ref"], specs["nr_diff"], specs["nr_all"]):
                    columns = feature_columns(spec, RESNET50)
                    evaluate_fit(spec["name"], name, "resnet50", tx[columns], vx[columns])
                candidate, history = tune_model(name, full, inner, seed, threads)
                append(history.assign(split_seed=split_seed, fold=fold, model=name), output / "inner_search.csv")
                columns = feature_columns(full, candidate["dimensions"])
                evaluate_fit("all_rr", name, "tuned", tx[columns], vx[columns], params=candidate["params"])
            columns = feature_columns(full, FIXED)
            evaluate_fit("all_rr", "ridge", "fixed", tx[columns], vx[columns])
            evaluate_fit("all_rr", "ridge", "standard", std_tx[columns], std_vx[columns])

            # Compare RR and true-HR FR on exactly the rows with all six HR metrics.
            common = np.isfinite(raw[GT_COLUMNS]).all(axis=1) & raw.gt_path.ne("")
            oracle_train, oracle_eval = train[common.loc[train]], evaluate[common.loc[evaluate]]
            if len(oracle_train) and len(oracle_eval):
                record_fit(oracle_train, "hr_common")
                hx, hv = prepare(oracle_train, oracle_eval, columns=scalars + GT_COLUMNS)
                for name in models:
                    for label, cols in (("all_rr_hr_common", columns), ("true_hr_oracle", columns + GT_COLUMNS)):
                        evaluate_fit(label, name, "fixed", hx[cols], hv[cols], cohort="hr_common")

        append(pd.DataFrame(fit_rows), output / "fits.csv")
        append(pd.DataFrame(metric_rows), output / "metrics.csv")
        append(pd.concat(prediction_rows, ignore_index=True), output / "predictions.csv.gz")
        write_json(output / "status.json", {"status": "running", "last_completed": [split_seed, fold]})

    for split_seed in settings["split_seeds"][:1] if args.pilot else settings["split_seeds"]:
        for fold, (train, valid) in enumerate(grouped_splits(raw, pool, settings["outer_folds"], split_seed), 1):
            if args.pilot and fold > 1:
                break
            run_fold(split_seed, fold, train, valid)

    write_json(output / "status.json", {"status": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "prior", "control-flags", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("core", "pca"), required=True)
    parser.add_argument("--pilot", action="store_true")
    with threadpool_limits(limits=2):
        run(parser.parse_args())
