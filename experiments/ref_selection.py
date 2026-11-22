"""Evaluate within-source Ref selection and its sensitivity to image geometry."""

import argparse
import json
from functools import cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.analyze_statistics import (
    BACKENDS,
    DSR,
    META,
    MOS,
    SET,
    combine,
    describe,
    estimate,
    read,
)
from experiments.data import _read_block, require, sha256_file
from experiments.run_grouped import write_json
from qualisr.regressors import safe_corr

SEED = 20261112
KEY = ["split_seed", "fold", "model", "sample_id"]


def load_predictions(prior, pilot=False):
    pieces, inputs = [], []
    for seed in ([42] if pilot else [42, 43, 44]):
        for fold in ([1] if pilot else range(1, 6)):
            for backend in BACKENDS:
                pair = []
                for name in ("nr", "nr_ref"):
                    path = prior / f"seed_{seed}/fold_{fold:02d}" / name / backend / "fixed/predictions.csv"
                    frame = read(path)
                    require(frame.sample_id.is_unique, f"Duplicate predictions: {path}")
                    pair.append(frame.set_index("sample_id").sort_index())
                    inputs.append(dict(path=str(path.resolve()), sha256=sha256_file(path)))
                pd.testing.assert_frame_equal(pair[0].drop(columns="prediction"),
                                              pair[1].drop(columns="prediction"), check_exact=True)
                frame = pair[0].rename(columns={"prediction": "nr"})
                frame["nr_ref"] = pair[1].prediction
                frame = frame.loc[frame.dataset.isin(MOS)].reset_index()
                require(np.isfinite(frame[["score", "nr", "nr_ref"]]).all().all(), "Nonfinite MOS input")
                pieces.append(frame.assign(split_seed=seed, fold=fold, model=backend))
    return pd.concat(pieces, ignore_index=True), inputs


def pair_counts(frame):
    """Exact MOS ties are excluded; an exact prediction tie earns half credit."""
    require(frame.sample_id.is_unique, "Duplicate output in a comparison group")
    i, j = np.triu_indices(len(frame), 1)
    target = np.sign(frame.score.to_numpy()[i] - frame.score.to_numpy()[j])
    result = {"n_outputs": len(frame), "n_pairs": len(i), "target_ties": int((target == 0).sum())}
    valid = target != 0
    for name in ("nr", "nr_ref"):
        delta = np.sign(frame[name].to_numpy()[i] - frame[name].to_numpy()[j])
        c = int(((delta == target) & valid).sum())
        d = int(((delta == -target) & valid).sum())
        t = int(((delta == 0) & valid).sum())
        result.update({f"{name}_concordant": c, f"{name}_discordant": d, f"{name}_prediction_ties": t,
                       f"{name}_accuracy": (c + t / 2) / (c + d + t) if c + d + t else np.nan})
    return result


def within_groups(frame):
    rows = []
    keys = ["dataset", "model", "split_seed", "source_group", "comparison_group"]
    for key, group in frame.groupby(keys, sort=True):
        if len(group) < 2:
            continue
        require(group.fold.nunique() == 1, "Outputs of a source cross outer folds")
        rows.append(dict(zip(keys, key, strict=True), **pair_counts(group)))
    return pd.DataFrame(rows)


def comparison_groups(predictions, samples):
    """Use the known Set120 scale and declared DISRQAD scales; external rows are pooled only."""
    scale = predictions.sample_id.map(samples.set_index("sample_id").declared_scale)
    require(scale.loc[predictions.dataset.eq(DSR)].isin([2, 4]).all(), "Missing or invalid DISRQAD scale")
    scale.loc[predictions.dataset.eq(SET)] = 4.0
    groups = predictions.source_group.copy()
    internal = predictions.dataset.isin([SET, DSR])
    groups.loc[internal] += "/scale=" + scale.loc[internal].astype(str)
    return groups


def accuracy_summary(groups, repetitions):
    rows = []
    for dataset, population in groups.dropna(subset=["nr_accuracy", "nr_ref_accuracy"]).groupby("dataset", sort=True):
        # Each source has equal weight. Average scales, then repeated splits, within a source.
        scores = population.groupby(["model", "source_group"])[["nr_accuracy", "nr_ref_accuracy"]].mean()
        sources = sorted(scores.index.get_level_values("source_group").unique())
        rng = np.random.default_rng(SEED + (dataset == DSR))
        draws = rng.multinomial(len(sources), np.full(len(sources), 1 / len(sources)), size=repetitions)
        panels = {}
        for backend in BACKENDS:
            values = scores.loc[backend].reindex(sources).to_numpy()
            require(np.isfinite(values).all(), "Unequal or undefined source coverage across predictors/backends")
            panels[backend] = values
        panels["backend_mean"] = np.mean(list(panels.values()), axis=0)
        for backend, values in panels.items():
            values = np.column_stack([values, values[:, 1] - values[:, 0]])
            reps = draws @ values / len(sources)
            for k, name in enumerate(("nr", "nr_ref", "delta")):
                lo, hi = np.quantile(reps[:, k], [0.025, 0.975])
                rows.append(dict(dataset=dataset, model=backend, predictor=name, accuracy=values[:, k].mean(),
                                 ci_low=lo, ci_high=hi, n_sources=len(sources), repetitions=repetitions))
    return pd.DataFrame(rows)


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    predictions, inputs = load_predictions(args.prior, args.pilot)
    samples = read(args.samples)
    predictions["comparison_group"] = comparison_groups(predictions, samples)
    predictions.to_csv(args.output / "paired_predictions.csv.gz", index=False)
    within = predictions.loc[predictions.dataset.isin([SET, DSR]) & predictions.evaluation.eq("internal")].copy()
    require(not within.duplicated(["model", "split_seed", "sample_id"]).any(), "Non-OOF internal predictions")
    require(within.groupby(["model", "sample_id"]).split_seed.nunique().eq(1 if args.pilot else 3).all(),
            "Incomplete repeated OOF coverage")
    if not args.pilot:
        require(within.loc[within.dataset.eq(SET)].source_group.nunique() == 40, "Expected all 40 Set120 sources")
    groups = within_groups(within)
    groups.to_csv(args.output / "within_source_counts.csv", index=False)
    summary = accuracy_summary(groups, args.repetitions).assign(estimand="mean_split_accuracy")
    # Match the paper's conditional pooled intervals: average each image's fixed forecasts first.
    collapsed = within.groupby(["model", "sample_id"], as_index=False).agg(
        {**dict.fromkeys(["dataset", "source_group", "comparison_group", "score"], "first"),
         "nr": "mean", "nr_ref": "mean"})
    collapsed["split_seed"], collapsed["fold"] = 0, 0
    mean_groups = within_groups(collapsed)
    mean_groups.to_csv(args.output / "within_source_mean_forecast_counts.csv", index=False)
    summary = pd.concat([summary, accuracy_summary(mean_groups, args.repetitions).assign(estimand="mean_forecast")])
    summary.to_csv(args.output / "within_source_summary.csv", index=False)
    coverage = []
    for dataset, frame in within.groupby("dataset"):
        metadata = frame.drop_duplicates("sample_id")
        eligible = groups.loc[groups.dataset.eq(dataset)]
        coverage.append(dict(dataset=dataset, n_outputs=metadata.sample_id.nunique(),
                             n_sources=metadata.source_group.nunique(),
                             eligible_sources=eligible.source_group.nunique(),
                             eligible_groups=eligible[["source_group", "comparison_group"]].drop_duplicates().shape[0],
                             eligible_outputs=int(eligible.loc[(eligible.model == BACKENDS[0]) &
                                                               (eligible.split_seed == 42), "n_outputs"].sum()),
                             zero_target_variation_groups=int(eligible.nr_accuracy.isna().sum())))
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False)
    pooled, fold_rows, panels = [], [], {}
    for name in ("nr", "nr_ref"):
        panels[name] = {}
        for backend in BACKENDS:
            frame = predictions.loc[predictions.model.eq(backend), [*KEY, *[x for x in META if x != "sample_id"], name]]
            frame = frame.rename(columns={name: "prediction"})
            tables = estimate(frame, args.repetitions)
            panels[name][backend] = tables
            for dataset, item in tables.items():
                pooled.extend(dict(dataset=dataset, model=backend, predictor=name, **row) for row in describe(item))
                fold_rows.append(item.folds.reset_index().assign(dataset=dataset, model=backend, predictor=name))
        panels[name]["backend_mean"] = {
            d: combine([(1 / 3, panels[name][b][d]) for b in BACKENDS], average_backends=True) for d in MOS}
        for dataset, item in panels[name]["backend_mean"].items():
            pooled.extend(dict(dataset=dataset, model="backend_mean", predictor=name, **r) for r in describe(item))
    for backend in (*BACKENDS, "backend_mean"):
        for dataset in MOS:
            item = combine([(-1, panels["nr"][backend][dataset]), (1, panels["nr_ref"][backend][dataset])])
            pooled.extend(dict(dataset=dataset, model=backend, predictor="delta", **r) for r in describe(item))
    pd.DataFrame(pooled).to_csv(args.output / "pooled_correlations.csv", index=False)
    pd.concat(fold_rows).to_csv(args.output / "fold_correlations.csv", index=False)
    # Pooled OOF correlation per repeat is descriptive: it combines differently fitted heads.
    repeated = []
    for key, frame in within.groupby(["dataset", "model", "split_seed"]):
        for name in ("nr", "nr_ref"):
            plcc, srcc = safe_corr(frame.score, frame[name])
            repeated.append(dict(zip(["dataset", "model", "split_seed"], key, strict=True),
                                 predictor=name, plcc=plcc, srcc=srcc))
    pd.DataFrame(repeated).to_csv(args.output / "pooled_oof_per_repeat.csv", index=False)
    write_json(args.output / "protocol.json", dict(
        within_source_seed=SEED, pooled_bootstrap_seed=20261106,
        bootstrap_repetitions=args.repetitions, pilot=args.pilot,
        feature_pools={"nr": "five NR including Q-Align", "nr_ref": "five NR + bicubic VGG/ResNet PCA-5 Ref"},
        primary="source-equal pair accuracy, averaged across split seeds and then backends",
        ties="Exclude exact MOS ties; exact prediction ties earn half credit; no tolerance",
        bootstrap="Paired source multinomial resampling; same draws for both predictors and all backends/splits",
        inference="Exploratory within-source comparisons conditional on saved predictions",
        pooled="Paper fold mean/SD and conditional mean-forecast PLCC/SRCC; datasets remain separate",
        samples=dict(path=str(args.samples.resolve()), sha256=sha256_file(args.samples)), inputs=inputs,
        code_sha256=sha256_file(Path(__file__))))
    write_json(args.output / "status.json", dict(status="complete", prediction_rows=len(predictions),
                                                 within_group_rows=len(groups)))


def audit_geometry(args):
    args.output.mkdir()
    samples = pd.read_csv(args.core / "samples.csv", float_precision="round_trip").set_index("sample_id")
    samples = samples.loc[samples.dataset.isin(["QualiSR-Set120", "dsr-dataset"])]

    @cache
    def digest(path):
        return sha256_file(Path(path))

    @cache
    def size(path):
        with Image.open(path) as image:
            return image.size

    geometry = samples[["dataset", "source_group", "declared_scale", "method"]].copy()
    geometry["lr_sha256"] = samples.lr_path.map(digest)
    geometry["sr_dimensions"] = samples.sr_path.map(size).map(lambda value: f"{value[0]}x{value[1]}")
    geometry.to_csv(args.output / "image_geometry.csv")
    set120 = samples.loc[samples.dataset.eq("QualiSR-Set120")]
    records = json.loads((args.core / "data_provenance.json").read_text())["files"]
    frames, provenance = [], []
    for name in ["ref_vgg", "ref_resnet"]:
        record, = [r for r in records if r["dataset"] == "QualiSR-Set120" and Path(r["path"]).stem == name]
        frame, actual = _read_block(Path(record["path"]), set120.index, prefix=name + "_")
        assert actual["sha256"] == record["sha256"]
        frames.append(frame)
        provenance.append(record)
    features = pd.concat(frames, axis=1)
    rows = []
    for source, group in set120.groupby("source_group"):
        values = features.loc[group.index].to_numpy()
        rows.append(dict(source_group=source, n_outputs=len(group),
                         cached_ref_max_abs_difference=float(np.max(np.abs(values - values[0]))),
                         n_lr_inputs=geometry.loc[group.index].lr_sha256.nunique(),
                         n_sr_dimensions=geometry.loc[group.index].sr_dimensions.nunique()))
    pd.DataFrame(rows).to_csv(args.output / "set120_reference_constancy.csv", index=False)
    metadata = dict(samples_sha256=sha256_file(args.core / "samples.csv"), inputs=provenance,
                    code_sha256=sha256_file(Path(__file__)), n_samples=len(samples))
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(rows).groupby("n_sr_dimensions").cached_ref_max_abs_difference.agg(["count", "min", "max"]))


def geometry_sensitivity(args):
    args.output.mkdir()
    predictions = pd.read_csv(args.selection / "paired_predictions.csv.gz", float_precision="round_trip")
    predictions = predictions.loc[predictions.evaluation.eq("internal")].copy()
    geometry = pd.read_csv(args.geometry / "image_geometry.csv")
    predictions = predictions.merge(geometry[["sample_id", "lr_sha256", "sr_dimensions"]],
                                    on="sample_id", validate="many_to_one")
    summaries, coverage = [], []
    for scope in ["same_lr", "same_lr_dimensions"]:
        frame = predictions.copy()
        frame["comparison_group"] = frame.source_group + "/lr=" + frame.lr_sha256
        if scope == "same_lr_dimensions":
            frame["comparison_group"] += "/size=" + frame.sr_dimensions
        counts = within_groups(frame)
        counts.to_csv(args.output / f"{scope}_counts.csv", index=False)
        summaries.append(accuracy_summary(counts, 10000).assign(scope=scope))
        for dataset, rows in counts.loc[(counts.model == "randomforest") & (counts.split_seed == 42)].groupby("dataset"):
            coverage.append(dict(scope=scope, dataset=dataset, n_groups=len(rows),
                                 n_sources=rows.source_group.nunique(), n_outputs=int(rows.n_outputs.sum()),
                                 n_pairs=int(rows.n_pairs.sum()), target_ties=int(rows.target_ties.sum()),
                                 defined_sources=rows.dropna(subset=["nr_accuracy"]).source_group.nunique()))
    pd.concat(summaries).to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False)
    # Preserve every Set120 method pair and its geometric stratum for inspection.
    records = []
    for key, rows in predictions.loc[predictions.dataset.eq("QualiSR-Set120")].groupby(
        ["source_group", "model", "split_seed"]
    ):
        for left, right in combinations(rows.sort_values("method").to_dict("records"), 2):
            record = dict(zip(["source_group", "model", "split_seed"], key, strict=True),
                          method_pair=left["method"] + "/" + right["method"],
                          sample_a=left["sample_id"], sample_b=right["sample_id"],
                          same_lr=left["lr_sha256"] == right["lr_sha256"],
                          same_dimensions=left["sr_dimensions"] == right["sr_dimensions"])
            for name in ["score", "nr", "nr_ref"]:
                record[name + "_difference"] = left[name] - right[name]
            records.append(record)
    pd.DataFrame(records).to_csv(args.output / "set120_method_pairs.csv", index=False)
    metadata = dict(reason="Added after discovering cached Ref varies with output dimensions in 23/40 Set120 sources",
                    interpretation="Exploratory sensitivity; original prespecified all-pair results retained",
                    geometry_sha256=sha256_file(args.geometry / "image_geometry.csv"),
                    predictions_sha256=sha256_file(args.selection / "paired_predictions.csv.gz"),
                    code_sha256=sha256_file(Path(__file__)))
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare source-equal pair accuracy with and without Ref")
    compare.add_argument("--prior", type=Path, required=True)
    compare.add_argument("--samples", type=Path, required=True)
    compare.add_argument("--repetitions", type=int, default=10000)
    compare.add_argument("--pilot", action="store_true")
    geometry = commands.add_parser("geometry", help="Audit LR identity, dimensions and cached Ref embeddings")
    geometry.add_argument("--core", type=Path, required=True)
    sensitivity = commands.add_parser("sensitivity", help="Restrict pair comparisons to matched geometry")
    sensitivity.add_argument("--selection", type=Path, required=True)
    sensitivity.add_argument("--geometry", type=Path, required=True)
    for cmd in (compare, geometry, sensitivity):
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        if args.repetitions < 1:
            compare.error("--repetitions must be positive")
        analyze(args)
    elif args.command == "geometry":
        audit_geometry(args)
    else:
        geometry_sensitivity(args)


if __name__ == "__main__":
    main()
