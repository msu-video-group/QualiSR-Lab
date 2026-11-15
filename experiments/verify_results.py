"""Independently check collected backend metrics using NumPy and pandas only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BACKENDS = ("randomforest", "xgb", "catboost")


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def correlation(frame):
    frame = frame.dropna(subset=["score", "prediction"])
    values = frame[["score", "prediction"]].to_numpy()
    if len(values) < 2 or any(np.allclose(x, x[0]) for x in values.T):
        return np.full(2, np.nan)
    ranks = frame[["score", "prediction"]].rank(method="average").to_numpy()
    return np.array([np.corrcoef(matrix.T)[0, 1] for matrix in (values, ranks)])


def check_metrics(root):
    samples = read(root / "samples.csv")
    fits = read(root / "fits.csv")
    predictions = read(root / "predictions.csv.gz")
    if not fits.fit_id.is_unique or predictions.duplicated(["fit_id", "sample_id"]).any():
        raise ValueError("Duplicate fit or prediction")
    joined = predictions.merge(samples, on="sample_id", validate="many_to_one")
    computed = []
    for fit, group in joined.groupby("fit_id", sort=False):
        mos, mos_samples = [], 0
        for dataset, rows in group.groupby("dataset"):
            n_samples = len(rows.dropna(subset=["score", "prediction"]))
            if rows.score_type.iat[0] == "mos":
                values = correlation(rows)
                mos.append(values)
                mos_samples += n_samples
                n_groups = 1
            else:
                values = np.array([correlation(part) for _, part in rows.groupby("correlation_group")])
                values = np.nanmean(values, axis=0)
                n_groups = rows.correlation_group.nunique()
            computed.append(dict(fit_id=fit, dataset=dataset, plcc=values[0], srcc=values[1],
                                 n_samples=n_samples, n_groups=n_groups))
        if mos:
            values = np.nanmean(mos, axis=0)
            computed.append(dict(fit_id=fit, dataset="macro_mos", plcc=values[0], srcc=values[1],
                                 n_samples=mos_samples, n_groups=len(mos)))
    reference = read(root / "analysis/metrics.csv")
    keys = ["fit_id", "dataset"]
    reference = reference.sort_values(keys).reset_index(drop=True)
    computed = pd.DataFrame(computed).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(computed[keys], reference[keys], check_exact=True)
    np.testing.assert_array_equal(computed[["n_samples", "n_groups"]], reference[["n_samples", "n_groups"]])
    np.testing.assert_allclose(computed[["plcc", "srcc"]], reference[["plcc", "srcc"]],
                               rtol=0, atol=1e-12, equal_nan=True)
    return dict(root=root.name, prediction_sets=len(fits), prediction_rows=len(predictions),
                metric_rows=len(reference))


def parent_parity(child, parent):
    keys = ["fit_id", "sample_id"]
    a, b = (read(root / "predictions.csv.gz").sort_values(keys).reset_index(drop=True)
            for root in (child, parent))
    pd.testing.assert_frame_equal(a[keys], b[keys], check_exact=True)
    np.testing.assert_allclose(a.prediction, b.prediction, rtol=0, atol=1e-12, equal_nan=True)
    prediction_error = float(np.nanmax(np.abs(a.prediction - b.prediction)))
    changes = []
    for filename in ("samples.csv", "fits.csv", "partitions.csv.gz", "inner_search.csv"):
        a, b = (read(root / filename) for root in (child, parent))
        if filename == "inner_search.csv":
            # RF's parallel floating-point summation can change ranks of tied
            # predictions. Audit inner scores separately; retain strict checks
            # on selected candidates, final parameters and outer predictions.
            scores = [np.array(frame.fold_srcc.map(json.loads).tolist()) for frame in (a, b)]
            for frame, values in zip((a, b), scores, strict=True):
                np.testing.assert_allclose(frame.mean_srcc, values.mean(axis=1), atol=1e-12, rtol=0)
            np.testing.assert_allclose(a.mean_plcc, b.mean_plcc, atol=1e-12, rtol=0)
            columns = ["fold_srcc", "mean_srcc", "mean_plcc"]
            pd.testing.assert_frame_equal(a.drop(columns=columns), b.drop(columns=columns), check_exact=True)
            for row, fold in np.argwhere(np.abs(scores[0] - scores[1]) > 1e-12):
                changes.append(dict(fit_id=a.iloc[row].fit_id, candidate=int(a.iloc[row].candidate),
                                    inner_fold=int(fold) + 1, old_srcc=float(scores[1][row, fold]),
                                    new_srcc=float(scores[0][row, fold]),
                                    delta=float(scores[0][row, fold] - scores[1][row, fold])))
        else:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
    return dict(root=child.name, prediction_max_abs_error=prediction_error,
                selected_candidates="identical", targets_splits_features="identical",
                inner_srcc_differences=changes)


def check_statistics(root):
    analysis = root / "analysis-full"
    folds = read(analysis / "fold_metrics.csv.gz")
    keys = ["table", "name", "model", "population", "kind"]
    expected = pd.concat([read(analysis / "correlations.csv"), read(analysis / "paired_deltas.csv")])
    count = 0
    for key, group in folds.groupby(keys):
        selected = expected.loc[np.logical_and.reduce([expected[k].eq(v) for k, v in zip(keys, key, strict=True)])]
        for metric in ("plcc", "srcc"):
            row, = selected.loc[selected.metric.eq(metric)].to_dict("records")
            np.testing.assert_allclose([group[metric].mean(), group[metric].std()],
                                       [row["fold_mean"], row["fold_sd"]], atol=1e-12, rtol=0, equal_nan=True)
            count += 1
    primary = read(analysis / "primary_tests.csv")
    assert len(primary) == 12
    order = np.argsort(primary.p_raw.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(primary.p_raw.to_numpy()[order] * np.arange(12, 0, -1)))
    np.testing.assert_allclose(primary.p_holm.to_numpy()[order], adjusted, rtol=0, atol=1e-15)
    samples = read(root / "musiq-full/samples.csv")
    predictions = read(root / "musiq-full/predictions.csv.gz")
    columns = ["musiq_finetuned", *(f"qualisr_{b}" for b in BACKENDS)]
    collapsed = predictions.groupby("sample_id")[columns].mean().reset_index().merge(
        samples, on="sample_id", validate="one_to_one")
    points = {}
    for column in columns:
        scores = {dataset: correlation(part.assign(prediction=part[column]))
                  for dataset, part in collapsed.loc[collapsed.dataset.ne("realsrq")].groupby("dataset")}
        points[column] = scores["QualiSR-Set120"] - (scores["dsr-dataset"] + scores["isrgenqa"]) / 2
    gap = points["musiq_finetuned"] - np.mean([points[f"qualisr_{b}"] for b in BACKENDS], axis=0)
    for index, metric in enumerate(("plcc", "srcc")):
        row, = primary.loc[primary.name.eq("finetuned_minus_qualisr_gap") & primary.metric.eq(metric)].to_dict("records")
        np.testing.assert_allclose(gap[index], row["conditional_point"], atol=1e-12, rtol=0)
    return dict(fold_summaries=count, holm_tests=12, independent_musiq_gap_points=2)


def check_cost(root):
    points = read(root / "cost-analysis/front_all_backends.csv")
    oof = read(root / "cost-analysis/oof.csv")
    assert len(points) == 23 and oof.groupby("sample_id").size().eq(3).all()
    collapsed = oof.groupby("sample_id").mean(numeric_only=True)
    extended = pd.concat([read(root / folder / "measurements.csv")
                          for folder in ("extended-pilot", "extended-full")], ignore_index=True)
    compact = read(root / "compact-full/measurements.csv")
    compact = compact.loc[compact.sample_id.isin(extended.sample_id)]
    for row in points.itertuples():
        actual = correlation(collapsed.assign(prediction=collapsed[row.pipeline]))
        np.testing.assert_allclose(actual, [row.plcc, row.srcc], atol=1e-12, rtol=0)
        folds = np.array([correlation(part.assign(prediction=part[row.pipeline]))
                          for _, part in oof.groupby(["split_seed", "fold"])])
        np.testing.assert_allclose(folds.mean(axis=0), [row.plcc_fold_mean, row.srcc_fold_mean],
                                   atol=1e-12, rtol=0)
        dominated = ((points.p50_ms.le(row.p50_ms) & points.srcc.ge(row.srcc))
                     & (points.p50_ms.lt(row.p50_ms) | points.srcc.gt(row.srcc))).any()
        assert row.pareto_all_backends == (not dominated)
        total = f"total_{row.pipeline}"
        measured = extended if total in extended else compact
        np.testing.assert_allclose(measured[total].quantile([.25, .5, .75]),
                                   [row.p25_ms, row.p50_ms, row.p75_ms], atol=1e-8, rtol=0)
        assert len(measured) == row.n_latency
    stages_checked = 0
    for folder in ("compact-full", "extended-pilot", "extended-full"):
        protocol = json.loads((root / folder / "protocol.json").read_text())
        measured = read(root / folder / "measurements.csv")
        for name, stages in protocol["stages"].items():
            np.testing.assert_allclose(measured[stages].sum(axis=1), measured[f"total_{name}"],
                                       atol=1e-8, rtol=0)
            stages_checked += len(measured)
    return dict(quality_points=len(points), fold_quality_pairs=len(points) * 15,
                stage_sums=stages_checked, global_frontier=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "historical-rf", "qalign-rf", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--include-cost", action="store_true")
    args = parser.parse_args()
    checks = []
    for backend in BACKENDS:
        for pool in ("historical", "qalign"):
            root = args.root / f"full-{backend}-{pool}"
            checks.append(check_metrics(root))
            print(f"Verified independent correlations for {root.name}", flush=True)
    parity = [parent_parity(args.root / "full-randomforest-historical", args.historical_rf),
              parent_parity(args.root / "full-randomforest-qalign", args.qalign_rf)]
    statistics = check_statistics(args.root)
    if args.include_cost:
        statistics["cost"] = check_cost(args.root)
    args.output.write_text(json.dumps(dict(status="passed", tables=checks, statistics=statistics,
        rf_parent_parity=parity,
        parity_scope="Predictions within 1e-12; exact targets, splits, features and selections. Inner score differences are recorded explicitly."),
        indent=2) + "\n")
