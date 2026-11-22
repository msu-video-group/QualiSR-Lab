"""Verify the supplementary runs and estimate paired WASD uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_grouped import METADATA, write_json
from experiments.transfer import dataset_method
from experiments.verify_grouped import read_results_csv
from qualisr.regressors import safe_corr, validation_correlations

KEY = ["split_seed", "fold", "experiment", "model", "track"]
PAIRS = [
    ("all_rr_no_wasd", "fixed", "all_rr", "fixed"),
    ("all_rr_no_fr", "fixed", "all_rr", "fixed"),
    ("all_rr_no_siglip", "fixed", "all_rr", "fixed"),
    ("all_rr_no_qalign", "fixed", "all_rr", "fixed"),
    ("all_rr", "fixed", "all_rr", "standard"),
    ("all_rr", "fixed", "all_rr", "resnet50"),
    ("all_rr", "fixed", "all_rr", "tuned"),
    ("all_rr_hr_common", "fixed", "true_hr_oracle", "fixed"),
]


def verify(root, output, *, check_metrics=True):
    assert json.loads((root / "status.json").read_text())["status"] == "complete"
    run = json.loads((root / "run.json").read_text())
    data = read_results_csv(root / "samples.csv").set_index("sample_id")
    assignments = read_results_csv(root / "fold_assignments.csv")
    predictions = read_results_csv(root / "predictions.csv.gz")
    fits = read_results_csv(root / "fits.csv")
    metrics = read_results_csv(root / "metrics.csv")
    fit_ids = read_results_csv(root / "fit_ids.csv.gz")
    assert not fits.duplicated(KEY).any()
    assert set(map(tuple, fits[KEY].to_numpy())) == set(map(tuple, predictions[KEY].to_numpy()))
    assert set(map(tuple, fits[KEY].to_numpy())) == set(map(tuple, metrics[KEY].to_numpy()))
    assert not predictions.duplicated(KEY + ["sample_id"]).any()
    assert np.isfinite(predictions.prediction).all()
    pd.testing.assert_frame_equal(
        predictions[METADATA[1:]].reset_index(drop=True),
        data.loc[predictions.sample_id, METADATA[1:]].reset_index(drop=True),
        check_exact=True,
    )

    folds = assignments.groupby(KEY[:2])
    expected = 1 if run["pilot"] else 15
    assert len(folds) == expected
    external = set(data.index[data.role.eq("external")])
    pool = set(data.index[data.role.eq("train")])
    controls = set(run["controls"])
    n_sets = 0
    method_rows, influence_rows, ordinary_rows = [], [], []
    search = read_results_csv(root / "inner_search.csv") if run["stage"] == "core" else None
    for key, assignment in folds:
        assert set(assignment.sample_id) == pool and not assignment.sample_id.duplicated().any()
        train = set(assignment.loc[assignment.partition.eq("train"), "sample_id"])
        holdout = pool - train
        assert not (set(data.loc[list(train), "source_group"]) &
                    set(data.loc[list(holdout | external), "source_group"]))
        groups = fit_ids[(fit_ids.split_seed == key[0]) & (fit_ids.fold == key[1])].groupby("case")
        sets = {label: set(part.sample_id) for label, part in groups}
        assert sets["released"] == train
        if run["stage"] == "core":
            assert sets["ordinary"] == train - controls
            inner_validation = []
            for i in range(1, 4):
                it, iv = sets[f"inner_{i}_train"], sets[f"inner_{i}_validation"]
                assert not it & iv and it | iv == train
                assert not set(data.loc[list(it), "source_group"]) & set(data.loc[list(iv), "source_group"])
                inner_validation.extend(iv)
            assert set(inner_validation) == train and len(inner_validation) == len(train)
            histories = search.loc[(search.split_seed == key[0]) & (search.fold == key[1])]
            assert set(histories.model) == set(run["config"]["evaluation"]["models"])
            for _, history in histories.groupby("model"):
                ordered = history.sort_values(["mean_srcc", "candidate"], ascending=[False, True])
                assert len(ordered) == 8 and ordered.selected.sum() == 1 and ordered.iloc[0].selected
        def keyed(table, split=key):
            return table[(table.split_seed == split[0]) & (table.fold == split[1])]
        target = keyed(fits).set_index(KEY[2:])
        saved = keyed(metrics).set_index(KEY[2:])
        hr_coverage = None
        for model_key, pred in keyed(predictions).groupby(KEY[2:]):
            n_sets += 1
            model = target.loc[model_key]
            expected_ids = holdout | external
            if model.cohort == "ordinary":
                expected_ids -= controls
            if model.cohort == "hr_common":
                if hr_coverage is None:
                    hr_coverage = set(pred.sample_id)
                assert set(pred.sample_id) == hr_coverage <= expected_ids
                expected_ids = hr_coverage
            assert set(pred.sample_id) == expected_ids and len(pred) == model.n_predict
            assert model.n_fit == len(sets[model.cohort])
            assert set(pred.loc[pred.evaluation.eq("external"), "sample_id"]) == expected_ids & external
            assert set(pred.loc[pred.evaluation.eq("internal"), "sample_id"]) == expected_ids & holdout
            if check_metrics:
                for scope, group in pred.groupby("evaluation"):
                    _, _, actual, _ = validation_correlations(group.score, group.prediction, group)
                    columns = ["plcc", "srcc", "n_groups", "n_samples"]
                    stored = saved.loc[[model_key]]
                    stored = stored.loc[stored.evaluation.eq(scope)].set_index("dataset")
                    assert set(stored.index) == set(actual.dataset)
                    np.testing.assert_allclose(stored[columns].sort_index(), actual.set_index("dataset")[columns].sort_index(),
                                               rtol=1e-10, atol=1e-12, equal_nan=True)
            if run["stage"] != "core":
                continue
            experiment, name, track = model_key
            tags = dict(zip(KEY, (*key, *model_key), strict=True))
            if track == "fixed" and name != "ridge":
                for dataset, part in pred.loc[pred.score_type.eq("mos")].groupby("dataset"):
                    for method, group in part.groupby(dataset_method(part)):
                        if group.source_group.nunique() >= 3:
                            p, s = safe_corr(group.score, group.prediction)
                            method_rows.append({**tags, "dataset": dataset, "method": method,
                                                "n_sources": group.source_group.nunique(), "plcc": p, "srcc": s})
            if experiment == "all_rr" and track in ("ordinary", "fixed") and name != "ridge":
                subset = pred.loc[pred.dataset.eq("QualiSR-Set120") & ~pred.sample_id.isin(controls)]
                p, s = safe_corr(subset.score, subset.prediction)
                ordinary_rows.append({**tags, "n": len(subset), "plcc": p, "srcc": s})
                if track == "ordinary":
                    for omit in subset.source_group.unique():
                        keep = subset.loc[subset.source_group.ne(omit)]
                        q, t = safe_corr(keep.score, keep.prediction)
                        influence_rows.append({**tags, "omitted_source": omit, "delta_plcc": q - p, "delta_srcc": t - s})
    assert n_sets == len(fits)
    if run["stage"] == "core":
        pd.DataFrame(method_rows).to_csv(output / "mos_methods.csv", index=False)
        pd.DataFrame(influence_rows).to_csv(output / "set120_source_influence.csv", index=False)
        pd.DataFrame(ordinary_rows).to_csv(output / "set120_ordinary_common.csv", index=False)
        if not run["pilot"]:
            intervals(predictions, Path(run["prior"]), output)
    summary = metrics.groupby(KEY[2:] + ["evaluation", "dataset"]).agg(
        plcc_mean=("plcc", "mean"), plcc_sd=("plcc", "std"),
        srcc_mean=("srcc", "mean"), srcc_sd=("srcc", "std"), n_folds=("srcc", "size")
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    differences = []
    for base, bt, added, at in PAIRS:
        columns = ["split_seed", "fold", "model", "evaluation", "dataset"]
        a = metrics.loc[metrics.experiment.eq(base) & metrics.track.eq(bt)]
        b = metrics.loc[metrics.experiment.eq(added) & metrics.track.eq(at)]
        joined = a.merge(b, on=columns, validate="one_to_one", suffixes=("_base", "_added"))
        differences.append(joined[columns].assign(
            baseline=base, baseline_track=bt, comparison=added, comparison_track=at,
            delta_plcc=joined.plcc_added - joined.plcc_base, delta_srcc=joined.srcc_added - joined.srcc_base))
    pd.concat(differences, ignore_index=True).to_csv(output / "paired_deltas.csv", index=False)
    if run["stage"] == "pca":
        plot_sweep(summary, output)
    write_json(output / "verification.json", {"status": "passed", "folds": expected, "prediction_sets": n_sets,
                                             "prediction_rows": len(predictions), "metrics_recomputed": check_metrics})


def weighted_corr(a, b, weights):
    total = weights.sum(axis=1, keepdims=True)
    x = a - (weights * a).sum(axis=1, keepdims=True) / total
    y = b - (weights * b).sum(axis=1, keepdims=True) / total
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights * x * y).sum(axis=1) / np.sqrt((weights * x * x).sum(axis=1) *
                                                      (weights * y * y).sum(axis=1))


def weighted_ranks(values, weights):
    order = np.argsort(values, kind="stable")
    ties = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    sorted_weights = weights[:, order]
    masses = np.add.reduceat(sorted_weights, ties, axis=1)
    midpoints = np.cumsum(masses, axis=1) - masses / 2
    ranks = np.empty_like(weights, dtype=float)
    ranks[:, order] = np.repeat(midpoints, np.diff(np.r_[ties, len(order)]), axis=1)
    return ranks


def clustered_delta(frame, n_reps=10_000):
    group_codes, sources = pd.factorize(frame.source_group, sort=True)
    n = len(sources)
    rng = np.random.default_rng(20261105)
    draws = rng.multinomial(n, np.full(n, 1 / n), size=n_reps)
    if frame.score_type.iat[0] == "bradley_terry":
        a, b = [], []
        for _, part in frame.groupby(["source_group", "correlation_group"], sort=True):
            a.append(safe_corr(part.score, part.base))
            b.append(safe_corr(part.score, part.added))
        groups = frame[["source_group", "correlation_group"]].drop_duplicates().sort_values(
            ["source_group", "correlation_group"])
        indices = sources.get_indexer(groups.source_group)
        a, b = np.array(a), np.array(b)
        def bt_avg(values, counts):
            valid = np.isfinite(values)
            numerator = counts @ np.nan_to_num(values)
            denominator = counts @ valid.astype(float)
            return numerator / denominator
        point = np.nanmean(b, axis=0) - np.nanmean(a, axis=0)
        reps = np.concatenate([bt_avg(b, batch[:, indices]) - bt_avg(a, batch[:, indices])
                               for batch in np.array_split(draws, 40)])
    else:
        y, a, b = (frame[col].to_numpy() for col in ("score", "base", "added"))
        point = np.array(safe_corr(y, b)) - np.array(safe_corr(y, a))
        results = []
        for batch in np.array_split(draws, 40):
            weights = batch[:, group_codes]
            yr, ar, br = (weighted_ranks(z, weights) for z in (y, a, b))
            p = weighted_corr(y, b, weights) - weighted_corr(y, a, weights)
            s = weighted_corr(yr, br, weights) - weighted_corr(yr, ar, weights)
            results.append(np.column_stack([p, s]))
        reps = np.concatenate(results)
    quantiles = np.nanquantile(reps, [0.025, 0.975], axis=0)
    return [{"metric": name, "delta": point[i], "ci_low": quantiles[0, i], "ci_high": quantiles[1, i],
             "n_sources": n, "n_samples": len(frame), "valid_replicates": np.isfinite(reps[:, i]).sum()}
            for i, name in enumerate(("plcc", "srcc"))]


def intervals(predictions, prior, output):
    results = []
    pair_keys = ["split_seed", "fold", *METADATA, "evaluation"]
    for name in ("randomforest", "xgb", "catboost"):
        contrasts = [("historical_candidate_with_qalign", "full_without_stats", "full_with_stats")]
        if predictions is not None:
            contrasts.append(("all_rr", "all_rr_no_wasd", "all_rr"))
        for contrast, left, right in contrasts:
            parts = []
            for experiment in (left, right):
                if experiment.startswith("all_rr"):
                    subset = predictions.query("model == @name and experiment == @experiment and track == 'fixed'")
                else:
                    pieces = []
                    for path in sorted(prior.glob(f"seed_*/fold_*/{experiment}/{name}/fixed/predictions.csv")):
                        seed_path, fold_path = path.relative_to(prior).parts[:2]
                        seed = int(seed_path.split("_")[-1])
                        fold = int(fold_path.split("_")[-1])
                        pieces.append(read_results_csv(path).assign(split_seed=seed, fold=fold))
                    assert len(pieces) == 15
                    subset = pd.concat(pieces, ignore_index=True)
                parts.append(subset[pair_keys + ["prediction"]])
            pair = parts[0].merge(parts[1], on=pair_keys, how="outer", indicator=True, validate="one_to_one",
                                  suffixes=("_base", "_added"))
            assert pair._merge.eq("both").all()
            scored = pair.loc[pair.score.notna()]
            for dataset, group in scored.groupby("dataset"):
                folds = group.groupby("sample_id").size()
                assert folds.eq(3 if group.evaluation.iat[0] == "internal" else 15).all()
                rows = group.groupby(["sample_id", "source_group", "correlation_group", "score_type", "score"])[
                    ["prediction_base", "prediction_added"]].mean().reset_index()
                rows = rows.rename(columns={"prediction_base": "base", "prediction_added": "added"})
                results.extend({"contrast": contrast, "model": name, "dataset": dataset, **row}
                               for row in clustered_delta(rows))
    pd.DataFrame(results).to_csv(output / "wasd_cluster_intervals.csv", index=False)


def plot_sweep(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
    train_datasets = ("QualiSR-Set120", "dsr-dataset")
    for col, dataset in enumerate(train_datasets):
        for row, metric in enumerate(("plcc", "srcc")):
            ax = axes[row, col]
            for (family, model), part in summary.loc[summary.dataset.eq(dataset)].groupby(["experiment", "model"]):
                series = part.assign(n=part.track.str.rsplit("_", n=1).str[-1].astype(int)).sort_values("n")
                label = family.replace("_", " ")
                line, = ax.plot(series.n, series[f"{metric}_mean"], marker=".", label=f"{label} · {model}")
                avg, sd = series[f"{metric}_mean"], series[f"{metric}_sd"]
                ax.fill_between(series.n, avg - sd, avg + sd, alpha=0.1, color=line.get_color())
            for n in (5, 25, 50):
                ax.axvline(n, color="0.85", lw=0.6, zorder=0)
            ax.set(title=dataset if row == 0 else "", ylabel=metric.upper())
            if row == 1:
                ax.set_xlabel("PCA components")
    axes[0, 1].legend(fontsize=7, ncols=2)
    fig.savefig(output / "pca_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--prior-wasd", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    if args.run:
        verify(args.run, args.output)
    else:
        intervals(None, args.prior_wasd, args.output)
