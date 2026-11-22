"""Fit, verify and summarize matched transfer and robustness experiments."""

from __future__ import annotations

import argparse
import json
import platform
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS
from experiments.data import NR_COLUMNS, load_experiment_data, read_csv, require, sha256_file
from experiments.fold_features import FoldTransformer
from experiments.presets import COMPACT_STAT_COLUMNS, experiment_specs, search_candidates
from experiments.run_grouped import (
    METADATA,
    feature_columns,
    grouped_splits,
    make_model,
    tune_model,
    write_json,
)
from qualisr.regressors import safe_corr, validation_correlations

SET120, DISRQAD, ISRGEN, REALSRQ = "QualiSR-Set120", "dsr-dataset", "isrgenqa", "realsrq"
MOS_DATASETS = (SET120, DISRQAD, ISRGEN)
MODEL = "randomforest"
DIMS = {"vgg": 5, "resnet": 5}
MAIN = ("nr", "nr_fr", "nr_ref", "nr_diff", "nr_stats", "nr_all")
SEED_SPECS = ("nr", "nr_all")
EXTRA_SEEDS = (43, 44)
TRANSFER_CANDIDATES = search_candidates(MODEL, False)[:3]


BASELINE = "historical_baseline"
STATS = "historical_baseline_stats"
KEY = ["fit_id", "track", "case", "split_seed", "fold", "spec", "model_seed"]


def control_ids(raw, flags):
    """Join per-image control flags by source and SR method."""
    keys = ["test_case", "method"]
    samples = raw.loc[raw.dataset == SET120, ["sample_id", *keys]].copy()
    flags = flags[[*keys, "is_control_item"]].copy()
    for table in (samples, flags):
        table["method"] = table.method.str.lower()
    joined = samples.merge(flags, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or set(joined.is_control_item) != {True, False}:
        raise ValueError("Control flags must provide exactly one boolean for every Set120 sample")
    controls = set(joined.loc[joined.is_control_item, "sample_id"])
    if len(controls) != 3:
        raise ValueError(f"Expected three screened Set120 images, found {len(controls)}")
    return controls


def transfer_partitions(raw):
    """Full-source transfer with invariant external targets across source mixtures."""
    for size in range(1, len(MOS_DATASETS) + 1):
        for sources in combinations(MOS_DATASETS, size):
            source = raw.dataset.isin(sources)
            yield "+".join(sources), raw.index[source].to_numpy(), raw.index[~source].to_numpy()


def matched_partitions(raw, train, heldout):
    """Table 13 factorial uses one evaluation population per outer split."""
    dsr_train = train[raw.loc[train, "dataset"].eq(DISRQAD)]
    external = raw.index[raw.dataset.isin([ISRGEN, REALSRQ])].to_numpy()
    return dsr_train, np.concatenate([heldout, external])


def check_partition(raw, train, evaluate):
    if not len(train) or not len(evaluate):
        raise ValueError("Training and evaluation must be nonempty")
    if set(raw.loc[train, "source_group"]) & set(raw.loc[evaluate, "source_group"]):
        raise ValueError("A source group crosses the training/evaluation boundary")
    if not np.isfinite(raw.loc[train, "score"]).all():
        raise ValueError("A training target is nonfinite")


def run(config, base, flags_path, output, pilot, *, include_qalign=False, backend=MODEL,
        data=None, prepared_cache=None):
    candidates = search_candidates(backend, False)[:3]
    output.mkdir(parents=True, exist_ok=False)
    cfg = config["evaluation"]
    seed, threads = cfg["model_seed"], cfg["threads"]
    print("Loading raw features and auditing source content", flush=True)
    raw, blocks, scalars, provenance = load_experiment_data(config, base) if data is None else data
    if set(raw.dataset) != {*MOS_DATASETS, REALSRQ}:
        raise ValueError("Transfer and robustness experiments require the four paper datasets")
    controls = control_ids(raw, pd.read_csv(flags_path))
    raw["screened_control"] = raw.sample_id.isin(controls)
    raw[METADATA + ["screened_control", "content_sha256", "content_kind"]].to_csv(
        output / "samples.csv", index=False
    )
    write_json(output / "run.json", {
        "config": config, "python": platform.python_version(), "pilot": pilot, "backend": backend,
        "fixed_pca_dimensions": DIMS, "transfer_candidates": candidates,
        "controls": sorted(controls), "flags_sha256": sha256_file(flags_path),
        "robustness_specs": MAIN, "seed_specs": SEED_SPECS, "extra_model_seeds": EXTRA_SEEDS,
        "pca_seed": seed,
        "include_qalign": include_qalign,
    })
    write_json(output / "data_provenance.json", provenance)
    specs = {spec["name"]: spec for spec in experiment_specs()}
    # Match the historical Table 11–13 feature families and Q-Align exclusion.
    baseline = {
        "name": "baseline_qalign" if include_qalign else "historical_baseline",
        "scalars": [c for c in specs["nr_fr"]["scalars"] if include_qalign or c != "qalign"],
        "representations": ["sr"],
    }
    with_stats = {**baseline, "name": f"{baseline['name']}_stats",
                  "scalars": baseline["scalars"] + COMPACT_STAT_COLUMNS}

    fits, assignments, predictions, searches = [], [], [], []
    fit_count = 0

    def save_batch(last_completed):
        nonlocal fit_count
        for name, records in (("fits.csv", fits), ("partitions.csv.gz", assignments),
                              ("predictions.csv.gz", predictions), ("inner_search.csv", searches)):
            path = output / name
            table = pd.DataFrame(records) if records is fits else pd.concat(records, ignore_index=True)
            table.to_csv(path, index=False, mode="a", header=not path.exists())
        fit_count += len(fits)
        for records in (fits, assignments, predictions, searches):
            records.clear()
        write_json(output / "status.json", {"status": "running", "fits": fit_count,
                                            "last_completed": last_completed})

    def prepare(train, evaluate, dimensions=DIMS):
        check_partition(raw, train, evaluate)
        key = (tuple(train), tuple(evaluate), tuple(dimensions.items()), seed)
        if prepared_cache is not None and key in prepared_cache:
            return prepared_cache[key]
        transformer = FoldTransformer(
            scalars, blocks, dimensions, seed=seed
        )
        training_features = transformer.fit_transform(raw.loc[train])
        result = training_features, transformer.transform(raw.loc[evaluate]), transformer
        if prepared_cache is not None:
            prepared_cache[key] = result
        return result

    def record_partition(partition_id, train, evaluate, split_seed):
        for ids, role in ((train, "train"), (evaluate, "evaluate")):
            assignments.append(pd.DataFrame({
                "partition_id": partition_id, "role": role, "sample_id": raw.loc[ids, "sample_id"],
            }))
        # Every selection split is confined to this partition's training rows.
        inner = []
        if split_seed is not None:
            for fold, (it, iv) in enumerate(grouped_splits(raw, train, cfg["inner_folds"], split_seed), 1):
                print(f"{partition_id}: preparing inner fold {fold}", flush=True)
                ix, vx, _ = prepare(it, iv)
                inner.append((ix, raw.loc[it, "score"], vx, raw.loc[iv, "score"], raw.loc[iv, METADATA]))
                for ids, role in ((it, f"inner_{fold}_train"), (iv, f"inner_{fold}_validate")):
                    assignments.append(pd.DataFrame({
                        "partition_id": partition_id, "role": role, "sample_id": raw.loc[ids, "sample_id"],
                    }))
        return inner

    def fit(spec, partition_id, track, case, split_seed, fold, model_seed, train, evaluate, prepared, inner=None):
        tx, ex, transformer = prepared
        params = {}
        fit_id = f"{partition_id}/{spec['name']}/{model_seed}"
        print(f"{fit_id}: fitting", flush=True)
        if inner is not None:
            selected, history = tune_model(backend, spec, inner, model_seed, threads, candidates=candidates)
            params = selected["params"]
            searches.append(history.assign(fit_id=fit_id))
        columns = feature_columns(spec, DIMS)
        model = make_model(backend, params, model_seed, threads).fit(tx[columns], raw.loc[train, "score"])
        fits.append({"fit_id": fit_id, "partition_id": partition_id, "track": track, "case": case,
                     "split_seed": split_seed, "fold": fold, "spec": spec["name"], "model_seed": model_seed,
                     "params": json.dumps(params), "columns": json.dumps(columns)})
        predictions.append(pd.DataFrame({"fit_id": fit_id, "sample_id": raw.loc[evaluate, "sample_id"],
                                         "prediction": model.predict(ex[columns])}))
        if track == "robustness" and case == "released" and split_seed == seed and fold == 1 and model_seed == seed:
            if spec["name"] in {"nr", "nr_diff", "nr_stats"}:
                folder = output / "profile_models"
                folder.mkdir(exist_ok=True)
                joblib.dump((transformer, model, columns), folder / f"{spec['name']}.joblib", compress=3)

    def direct(partition_id, case, split_seed, fold, evaluate):
        for metric in NR_COLUMNS:
            fit_id = f"{partition_id}/{metric}/direct"
            fits.append({"fit_id": fit_id, "partition_id": partition_id, "track": "direct", "case": case,
                         "split_seed": split_seed, "fold": fold, "spec": metric})
            predictions.append(pd.DataFrame({"fit_id": fit_id, "sample_id": raw.loc[evaluate, "sample_id"],
                                             "prediction": raw.loc[evaluate, metric]}))

    transfer = list(transfer_partitions(raw))
    for case, train, evaluate in transfer[:1] if pilot else transfer:
        print(f"Transfer: {case}", flush=True)
        partition = f"transfer/{case}"
        prepared = prepare(train, evaluate)
        inner = record_partition(partition, train, evaluate, seed)
        fit(baseline, partition, "transfer", case, seed, 0, seed, train, evaluate, prepared, inner)
        direct(partition, case, seed, 0, evaluate)
        save_batch(partition)

    pool = raw.index[raw.dataset.isin([SET120, DISRQAD])].to_numpy()
    for split_seed in cfg["split_seeds"][:1] if pilot else cfg["split_seeds"]:
        for fold, (train, heldout) in enumerate(grouped_splits(raw, pool, cfg["outer_folds"], split_seed), 1):
            print(f"Matched mixture and robustness: seed={split_seed}, fold={fold}", flush=True)
            dsr_train, evaluate = matched_partitions(raw, train, heldout)
            stem = f"seed_{split_seed}/fold_{fold:02d}"
            for case, subset in (("dsr_only", dsr_train), ("dsr_plus_set120", train)):
                partition = f"impact/{stem}/{case}"
                prepared = prepare(subset, evaluate)
                inner = record_partition(partition, subset, evaluate, split_seed)
                for spec in (baseline, with_stats):
                    fit(spec, partition, "impact", case, split_seed, fold, seed, subset, evaluate, prepared, inner)
            direct(partition, "matched", split_seed, fold, evaluate)

            if include_qalign:
                save_batch(stem)
                if pilot:
                    break
                continue

            for case in ("released", "ordinary"):
                subset = train
                target = evaluate
                if case == "ordinary":
                    subset = train[~raw.loc[train, "screened_control"]]
                    target = evaluate[~raw.loc[evaluate, "screened_control"]]
                partition = f"robustness/{stem}/{case}"
                prepared = prepare(subset, target, cfg["max_pca_dimensions"])
                record_partition(partition, subset, target, None)
                for name in MAIN:
                    fit(specs[name], partition, "robustness", case, split_seed, fold, seed, subset, target, prepared)
                if case == "released" and split_seed == seed:
                    for name in SEED_SPECS:
                        for model_seed in EXTRA_SEEDS:
                            fit(specs[name], partition, "robustness", case, split_seed, fold, model_seed,
                                subset, target, prepared)
            save_batch(stem)
            if pilot:
                break

    write_json(output / "status.json", {"status": "complete", "fits": fit_count})
    print(f"Completed {fit_count} prediction sets", flush=True)


def expected_partitions(samples, run):
    cfg = run["config"]["evaluation"]
    seed = cfg["model_seed"]
    include_qalign = run.get("include_qalign", False)
    baseline = "baseline_qalign" if include_qalign else BASELINE
    stats = f"{baseline}_stats" if include_qalign else STATS
    transfers = list(transfer_partitions(samples))
    for case, train, target in transfers[:1] if run["pilot"] else transfers:
        expected = [(baseline, seed), *((name, None) for name in NR_COLUMNS)]
        yield f"transfer/{case}", train, target, seed, expected
    pool = samples.index[samples.dataset.isin([SET120, DISRQAD])].to_numpy()
    for split_seed in cfg["split_seeds"][:1] if run["pilot"] else cfg["split_seeds"]:
        for fold, (train, heldout) in enumerate(grouped_splits(samples, pool, cfg["outer_folds"], split_seed), 1):
            dsr_train, target = matched_partitions(samples, train, heldout)
            stem = f"seed_{split_seed}/fold_{fold:02d}"
            expected = [(baseline, seed), (stats, seed)]
            yield f"impact/{stem}/dsr_only", dsr_train, target, split_seed, expected
            yield f"impact/{stem}/dsr_plus_set120", train, target, split_seed, [
                *expected, *((name, None) for name in NR_COLUMNS)
            ]
            if include_qalign:
                if run["pilot"]:
                    break
                continue
            expected = [(name, seed) for name in MAIN]
            extras = [(name, extra) for name in SEED_SPECS for extra in EXTRA_SEEDS] if split_seed == seed else []
            yield f"robustness/{stem}/released", train, target, None, expected + extras
            ordinary_train = train[~samples.loc[train, "screened_control"]]
            ordinary_target = target[~samples.loc[target, "screened_control"]]
            yield f"robustness/{stem}/ordinary", ordinary_train, ordinary_target, None, expected
            if run["pilot"]:
                break


def verify(root):
    run = json.loads((root / "run.json").read_text())
    status = json.loads((root / "status.json").read_text())
    require(status["status"] == "complete", "Experiment must be complete")
    samples = read_csv(root, "samples.csv")
    fits = read_csv(root, "fits.csv")
    partitions = read_csv(root, "partitions.csv.gz")
    predictions = read_csv(root, "predictions.csv.gz")
    searches = read_csv(root, "inner_search.csv")
    require(samples.sample_id.is_unique and fits.fit_id.is_unique, "Duplicate sample or fit ID")
    require(set(samples.loc[samples.screened_control, "sample_id"]) == set(run["controls"])
            and len(run["controls"]) == 3, "Incorrect control IDs")
    require(set(predictions.fit_id) == set(fits.fit_id) and len(fits) == status["fits"], "Fit coverage differs")
    require(not predictions.duplicated(["fit_id", "sample_id"]).any(), "Duplicate prediction")
    require(not partitions.duplicated().any(), "Duplicate partition assignment")
    learned = set(fits.loc[fits.track.ne("direct"), "fit_id"])
    require(np.isfinite(predictions.loc[predictions.fit_id.isin(learned), "prediction"]).all(),
            "Nonfinite learned predictions")
    require(not np.isinf(predictions.prediction).any(), "Infinite direct-metric predictions")
    source = samples.set_index("sample_id").source_group
    observed = partitions.groupby(["partition_id", "role"]).sample_id.agg(set).to_dict()
    pred_ids = predictions.groupby("fit_id").sample_id.agg(set).to_dict()
    expected_fits = {}
    expected_searches = set()
    for partition, train, target, split_seed, fit_specs in expected_partitions(samples, run):
        ids = {"train": set(samples.loc[train, "sample_id"]), "evaluate": set(samples.loc[target, "sample_id"])}
        require(not set(source.loc[list(ids["train"])]) & set(source.loc[list(ids["evaluate"])]),
                f"Source leakage in {partition}")
        if split_seed is not None:
            n = run["config"]["evaluation"]["inner_folds"]
            for i, (it, iv) in enumerate(grouped_splits(samples, train, n, split_seed), 1):
                ids[f"inner_{i}_train"] = set(samples.loc[it, "sample_id"])
                ids[f"inner_{i}_validate"] = set(samples.loc[iv, "sample_id"])
        for role, sample_ids in ids.items():
            require(observed.pop((partition, role), None) == sample_ids, f"Incorrect {partition}/{role}")
        for spec, seed in fit_specs:
            fit_id = f"{partition}/{spec}/{'direct' if seed is None else seed}"
            expected_fits[fit_id] = partition
            require(pred_ids.pop(fit_id, None) == ids["evaluate"], f"Incorrect prediction coverage in {fit_id}")
            if split_seed is not None and seed is not None:
                expected_searches.add(fit_id)
    require(not observed and not pred_ids, "Unexpected partition or predictions")
    require(fits.set_index("fit_id").partition_id.to_dict() == expected_fits, "Fit manifest disagrees")
    require(set(searches.fit_id) == expected_searches, "Incorrect tuning coverage")
    require(not searches.duplicated(["fit_id", "candidate"]).any(), "Duplicate tuning candidate")
    selected = searches.sort_values(["mean_srcc", "candidate"], ascending=[False, True]).groupby("fit_id").head(1)
    require(set(zip(selected.fit_id, selected.candidate, strict=False)) ==
            set(zip(searches.loc[searches.selected, "fit_id"], searches.loc[searches.selected, "candidate"], strict=False)),
            "Saved tuning choices disagree with inner SRCC")
    require(searches.groupby("fit_id").size().eq(len(run["transfer_candidates"])).all(), "Candidate budget differs")
    fitted = fits.set_index("fit_id")
    for row in selected.itertuples():
        require(json.loads(row.params) == json.loads(fitted.at[row.fit_id, "params"]),
                f"Selected parameters differ for {row.fit_id}")
    joined = predictions.merge(samples, on="sample_id", validate="many_to_one")
    require(len(joined) == len(predictions), "Unknown prediction sample")
    return run, fits, joined


def metric_rows(group):
    _, _, per_dataset, groups = validation_correlations(group.score, group.prediction, group)
    mos = per_dataset[per_dataset.score_type == "mos"]
    if len(mos):
        macro = pd.DataFrame([{"dataset": "macro_mos", "score_type": "mos", "aggregation": "mean_per_dataset",
                               "n_samples": mos.n_samples.sum(), "n_groups": len(mos),
                               "plcc": mos.plcc.mean(), "srcc": mos.srcc.mean()}])
        per_dataset = pd.concat([per_dataset, macro], ignore_index=True)
    return per_dataset, groups[groups.score_type != "mos"]


def dataset_method(group):
    """Parse the algorithm prefix in both ISRGen-QA filename variants."""
    if group.dataset.iat[0] != ISRGEN:
        return group.method
    names = group.method.str.extract(r"^(.+?)(?:_x[0-9]+)?_[0-9]+x[0-9]+$", expand=False)
    require(names.notna().all(), "An ISRGen-QA method ID does not match its documented filename grammar")
    return names


def analyze(root):
    run, fits, predictions = verify(root)
    seed = run["config"]["evaluation"]["model_seed"]
    include_qalign = run.get("include_qalign", False)
    baseline = "baseline_qalign" if include_qalign else BASELINE
    all_metrics, all_bt, ordinary, methods, influence, bt_methods, common_direct = [], [], [], [], [], [], []
    fit_info = fits.set_index("fit_id")
    by_fit = predictions.groupby("fit_id", sort=False)
    for fit_id, group in by_fit:
        info = fit_info.loc[fit_id]
        per_dataset, bt = metric_rows(group)
        all_metrics.append(per_dataset.assign(fit_id=fit_id))
        all_bt.append(bt.assign(fit_id=fit_id))
        if info.track in ("transfer", "impact"):
            direct_partition = info.partition_id.replace("/dsr_only", "/dsr_plus_set120")
            for metric in NR_COLUMNS:
                direct = by_fit.get_group(f"{direct_partition}/{metric}/direct").set_index("sample_id").prediction
                # Save learned scores on exactly the rows supported by each direct metric.
                covered = group[group.sample_id.map(direct).notna()]
                require(covered.dataset.nunique() == group.dataset.nunique(), f"No direct {metric} scores for a target")
                covered_metrics, _ = metric_rows(covered)
                common_direct.append(covered_metrics.assign(fit_id=fit_id, direct_metric=metric))
            scored = group.loc[group.score_type.ne("mos") & group.score.notna()].copy()
            # Rank within the linked BT comparison group before method-level diagnostics.
            ranks = scored.groupby("correlation_group")[["score", "prediction"]].rank(pct=True)
            scored["rank_error"] = ranks.prediction - ranks.score
            scored["absolute_rank_error"] = scored.rank_error.abs()
            by_method = scored.groupby("method").agg(
                n_groups=("correlation_group", "nunique"), n_sources=("source_group", "nunique"),
                mean_rank_error=("rank_error", "mean"), mean_absolute_rank_error=("absolute_rank_error", "mean"),
            ).reset_index()
            bt_methods.append(by_method.assign(fit_id=fit_id))
        if info.track in ("transfer", "impact") or (info.track == "robustness" and info.model_seed == seed):
            for dataset, part in group[group.score_type == "mos"].groupby("dataset"):
                for method, rows in part.groupby(dataset_method(part)):
                    n_sources = rows.source_group.nunique()
                    if n_sources >= 3:
                        plcc, srcc = safe_corr(rows.score, rows.prediction)
                        methods.append({"fit_id": fit_id, "dataset": dataset, "method": method, "plcc": plcc,
                                        "srcc": srcc, "n_samples": len(rows), "n_sources": n_sources})
        if info.track != "robustness" or info.model_seed != seed:
            continue
        subset = group.loc[group.dataset.eq(SET120) & ~group.screened_control]
        plcc, srcc = safe_corr(subset.score, subset.prediction)
        ordinary.append({"fit_id": fit_id, "n_samples": len(subset), "plcc": plcc, "srcc": srcc})
        if info.case == "ordinary":
            for omit in subset.source_group.unique():
                keep = subset[subset.source_group != omit]
                p, s = safe_corr(keep.score, keep.prediction)
                influence.append({"fit_id": fit_id, "omitted_source": omit,
                                  "delta_plcc": p - plcc, "delta_srcc": s - srcc})
    target = root / "analysis"
    target.mkdir(exist_ok=False)
    info = fits[KEY]
    metrics = pd.concat(all_metrics, ignore_index=True).merge(info, on="fit_id", validate="many_to_one")
    metrics.to_csv(target / "metrics.csv", index=False)
    direct_comparison = pd.concat(common_direct, ignore_index=True).merge(info, on="fit_id", validate="many_to_one")
    direct_comparison.to_csv(target / "learned_on_direct_coverage.csv", index=False)
    pd.concat(bt_methods, ignore_index=True).merge(info, on="fit_id", validate="many_to_one").to_csv(
        target / "realsrq_method_rank_errors.csv", index=False
    )
    pd.concat(all_bt, ignore_index=True).to_csv(target / "realsrq_group_metrics.csv.gz", index=False)
    group_cols = ["track", "case", "spec", "model_seed", "dataset", "score_type"]
    metrics.groupby(group_cols, dropna=False).agg(
        plcc_mean=("plcc", "mean"), srcc_mean=("srcc", "mean"),
        plcc_sd=("plcc", "std"), srcc_sd=("srcc", "std"),
        n_evaluations=("srcc", "size"), n_finite_srcc=("srcc", "count"),
        min_scored=("n_samples", "min"), max_scored=("n_samples", "max"),
    ).to_csv(target / "summary.csv")
    method_table = pd.DataFrame(methods).merge(info, on="fit_id", validate="many_to_one")
    method_table.to_csv(target / "mos_method_metrics.csv.gz", index=False)
    method_table.groupby([*group_cols[:-1], "method"], dropna=False).agg(
        plcc_mean=("plcc", "mean"), srcc_mean=("srcc", "mean"),
        n_evaluations=("srcc", "size"), min_sources=("n_sources", "min"),
    ).to_csv(target / "mos_method_summary.csv")
    def contrasts(table, axis, left, right, keys, label):
        columns = ["plcc", "srcc"]
        pairs = table.loc[table[axis].eq(left), keys + columns].merge(
            table.loc[table[axis].eq(right), keys + columns], on=keys, how="outer",
            validate="one_to_one", suffixes=("_left", "_right"), indicator=True,
        )
        require(pairs["_merge"].eq("both").all(), f"Unmatched rows for {label}")
        return pairs[keys].assign(contrast=label, delta_plcc=pairs.plcc_right - pairs.plcc_left,
                                  delta_srcc=pairs.srcc_right - pairs.srcc_left)

    split = ["split_seed", "fold"]
    impact = metrics[metrics.track == "impact"]
    diffs = [
        contrasts(impact, "case", "dsr_only", "dsr_plus_set120", [*split, "spec", "dataset"], "add_set120"),
        contrasts(impact, "spec", baseline, f"{baseline}_stats", [*split, "case", "dataset"], "add_stats"),
    ]
    if not include_qalign:
        pd.DataFrame(influence).merge(info, on="fit_id", validate="many_to_one").to_csv(
            target / "set120_leave_one_source_out.csv", index=False
        )
        ordinary_table = pd.DataFrame(ordinary).merge(info, on="fit_id", validate="one_to_one")
        ordinary_table.to_csv(target / "set120_common_ordinary_metrics.csv", index=False)
        robust = metrics[metrics.track.eq("robustness") & metrics.model_seed.eq(seed)]
        seeds = metrics[metrics.track.eq("robustness") & metrics.case.eq("released") &
                        metrics.split_seed.eq(seed) & metrics.spec.isin(SEED_SPECS)]
        diffs.extend([
            contrasts(robust, "case", "released", "ordinary", [*split, "spec", "dataset"], "exclude_controls"),
            contrasts(ordinary_table.assign(dataset=SET120), "case", "released", "ordinary",
                      [*split, "spec", "dataset"], "exclude_controls_common_ordinary"),
        ])
        for extra in EXTRA_SEEDS:
            diffs.append(contrasts(seeds, "model_seed", seed, extra, [*split, "spec", "dataset"],
                                   f"model_seed_{extra}"))
        for name in MAIN[1:]:
            subset = robust[robust.case == "ordinary"]
            diffs.append(contrasts(subset, "spec", "nr", name, [*split, "dataset"], f"ordinary_{name}_vs_nr"))
    paired = pd.concat(diffs, ignore_index=True)
    paired.to_csv(target / "paired_differences.csv", index=False)
    paired.groupby(["contrast", "dataset", "spec", "case"], dropna=False).agg(
        delta_plcc_mean=("delta_plcc", "mean"), delta_srcc_mean=("delta_srcc", "mean"),
        delta_srcc_min=("delta_srcc", "min"), delta_srcc_max=("delta_srcc", "max"),
        n_finite=("delta_srcc", "count"),
    ).to_csv(target / "paired_summary.csv")
    write_json(target / "verification.json", {
        "status": "passed", "pilot": run["pilot"], "prediction_sets": len(fits), "prediction_rows": len(predictions),
        "checks": ["exact outer and inner IDs", "source separation", "matched evaluation coverage",
                   "screened controls", "finite learned predictions", "inner-only selected parameters and candidate budget"],
        "direct_policy": "Missing direct scores are retained as NaN. Learned-on-direct-coverage metrics use "
                         "the same available rows as each direct baseline.",
        "scores": "MOS pooled within dataset, separate MOS macro; RealSRQ BT correlated within comparison group",
        "method_diagnostics": "MOS only, at least three sources; ISRGen-QA algorithm from documented filename grammar",
    })


def run_matrix(config, args):
    print("Loading and auditing the common matrix once for all backends", flush=True)
    data = load_experiment_data(config, args.config.resolve().parent)
    prepared = {}
    for backend in BACKENDS:
        for pool in ("historical", "qalign"):
            root = args.output / f"{'pilot' if args.pilot else 'full'}-{backend}-{pool}"
            print(f"Running {backend}, {pool}; shared prepared partitions={len(prepared)}", flush=True)
            run(config, args.config.resolve().parent, args.control_flags, root, args.pilot,
                include_qalign=pool == "qalign", backend=backend, data=data, prepared_cache=prepared)
            analyze(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    matrix = commands.add_parser("matrix", help="Run and summarize both feature pools for all three backends")
    fit = commands.add_parser("fit", help="Fit one backend and feature pool")
    for cmd in (matrix, fit):
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--control-flags", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--pilot", action="store_true")
    fit.add_argument("--backend", choices=BACKENDS, default=MODEL)
    fit.add_argument("--include-qalign", action="store_true", help="Include Q-Align in the transfer and factorial study")
    summary = commands.add_parser("summarize", help="Verify and summarize a saved fit")
    summary.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        analyze(args.root)
        return
    config = json.loads(args.config.read_text())
    with threadpool_limits(limits=config["evaluation"]["threads"]):
        if args.command == "matrix":
            run_matrix(config, args)
        else:
            run(config, args.config.resolve().parent, args.control_flags, args.output, args.pilot,
                include_qalign=args.include_qalign, backend=args.backend)


if __name__ == "__main__":
    main()
