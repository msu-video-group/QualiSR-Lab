"""Compare MUSIQ and tabular backends and analyze their transfer distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS, BT, FOLD, read
from experiments.data import load_experiment_data, read_csv, require, sha256_file
from experiments.finetune_musiq import MODELS, SPEC, partitions
from experiments.fold_features import FoldTransformer
from experiments.run_grouped import METADATA, feature_columns, make_model, write_json
from experiments.summarize_musiq import verify
from experiments.transfer import DIMS, DISRQAD, ISRGEN, REALSRQ, SET120, metric_rows

DATASETS = {SET120: "Set120 holdout", DISRQAD: "DISRQAD", ISRGEN: "ISRGen-QA", REALSRQ: "RealSRQ"}
COLORS = dict(zip(DATASETS, ["#27272a", "#0072b2", "#d55e00", "#009e73"], strict=True))
TITLES = dict(zip(MODELS, ["Original MUSIQ", "Fine-tuned MUSIQ", "QualiSR-Lab RF", "Direct Q-Align"], strict=True))
PERCENTILES = np.linspace(0, 1, 101)


def feature_plot(root, out):
    import matplotlib.pyplot as plt

    frame = read_csv(root, "feature_quantiles.csv.gz")
    cols = FOLD + ["family", "feature"]
    training = frame[frame.scope.eq("train")][cols + ["p25", "p50", "p75"]]
    evaluation = frame[frame.scope.ne("train")].merge(training, on=cols, validate="many_to_one",
                                                    suffixes=("", "_train"))
    iqr = evaluation.p75_train - evaluation.p25_train
    evaluation["median_shift"] = (evaluation.p50 - evaluation.p50_train).div(iqr.where(iqr > 0))
    evaluation["coverage"] = evaluation.finite / evaluation.rows
    summary = evaluation.groupby(["dataset", "family", "feature"], sort=False).agg(
        median_shift=("median_shift", "median"),
        min_finite_fraction=("coverage", "min"),
        n_nonzero_train_iqr=("median_shift", "count"),
    ).reset_index()
    summary.to_csv(out / "feature_shift_summary.csv", index=False)
    families = frame[["family", "feature"]].drop_duplicates().reset_index(drop=True)
    features = families.feature.tolist()
    fig, (ax, coverage) = plt.subplots(1, 2, figsize=(10, 7), sharey=True, width_ratios=[2.6, 1])
    for i, (dataset, title) in enumerate(DATASETS.items()):
        rows = summary[summary.dataset.eq(dataset)].set_index("feature").reindex(features)
        y = np.arange(len(features)) + (i - 1.5) * 0.17
        ax.scatter(rows.median_shift, y, s=14, color=COLORS[dataset], label=title)
        coverage.scatter(100 * rows.min_finite_fraction, y, s=14, color=COLORS[dataset])
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yticks(range(len(features)), features)
    ax.invert_yaxis()
    ax.set_xlabel("Median displacement from train (training IQR units)")
    coverage.set_xlabel("Minimum finite coverage (%)")
    coverage.set_xlim(-2, 104)
    coverage.grid(axis="x", color="#dddddd", linewidth=0.6)
    fig.legend(loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out / "feature_shift.svg")
    plt.close(fig)


def prediction_plot(root, out, models=MODELS, titles=TITLES):
    import matplotlib.pyplot as plt

    predictions = read_csv(root, "predictions.csv.gz")
    samples = read_csv(root, "samples.csv")
    frame = predictions.merge(samples[["sample_id", "dataset"]], on="sample_id", validate="many_to_one")
    n_rows = len(models) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(9.5, 3 * n_rows), sharey=True)
    for ax, model in zip(axes.flat, models, strict=True):
        for dataset, title in DATASETS.items():
            part = frame[frame.dataset.eq(dataset)]
            quantiles = part.groupby(FOLD)[model].quantile(PERCENTILES).unstack(FOLD)
            ax.plot(quantiles.median(axis=1), PERCENTILES, color=COLORS[dataset], linewidth=1.6, label=title)
        ax.set_title(titles[model], loc="left", fontsize=11)
        ax.set_xlabel("Predicted score")
        ax.grid(alpha=0.22)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative fraction")
    top_limits = [ax.get_xlim() for ax in axes[0]]
    for ax in axes[0]:
        ax.set_xlim(min(bounds[0] for bounds in top_limits), max(bounds[1] for bounds in top_limits))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "prediction_distributions.svg")
    fig.savefig(out / "prediction_distributions.png", dpi=160)
    plt.close(fig)


def fit_backends(args):
    original_run, original = verify(args.parent)
    cfg = json.loads(args.config.read_text())
    require(cfg == original_run["config"], "MUSIQ comparison config changed")
    raw, blocks, _, provenance = load_experiment_data(cfg, args.config.resolve().parent, include_all_rr=True)
    pd.testing.assert_frame_equal(raw[METADATA], original[METADATA].drop_duplicates().sort_values("sample_id")
                                  .set_index("sample_id").loc[raw.sample_id].reset_index()[METADATA],
                                  check_exact=True)
    args.output.mkdir()
    raw[METADATA].to_csv(args.output / "samples.csv", index=False)
    write_json(args.output / "data_provenance.json", provenance)
    assignments = pd.read_csv(args.parent / "partitions.csv.gz")
    columns = feature_columns(SPEC, DIMS)
    predictions, metrics, replays = [], [], []
    folds = list(partitions(raw, cfg["evaluation"], args.pilot))
    keys = pd.DataFrame([(seed, fold) for seed, fold, *_ in folds], columns=FOLD)
    assignments.merge(keys, on=FOLD, validate="many_to_one").to_csv(
        args.output / "partitions.csv.gz", index=False)
    for split_seed, fold, train, evaluate, _ in folds:
        key = {"split_seed": split_seed, "fold": fold}
        parent = original.loc[original.split_seed.eq(split_seed) & original.fold.eq(fold)]
        parent = parent.set_index("sample_id").loc[raw.loc[evaluate, "sample_id"]]
        frozen = FoldTransformer(SPEC["scalars"], blocks, DIMS, seed=cfg["evaluation"]["model_seed"])
        train_x = frozen.fit_transform(raw.loc[train])[columns]
        eval_x = frozen.transform(raw.loc[evaluate])[columns]
        frame = raw.loc[evaluate, ["sample_id"]].assign(**key).reset_index(drop=True)
        for name in ("musiq_direct", "musiq_finetuned", "qalign_direct"):
            frame[name] = parent[name].to_numpy()
        for backend in BACKENDS:
            print(f"MUSIQ comparison {split_seed}/{fold}: {backend}", flush=True)
            model = make_model(backend, {}, cfg["evaluation"]["model_seed"], cfg["evaluation"]["threads"])
            model.fit(train_x, raw.loc[train, "score"])
            values = model.predict(eval_x)
            require(np.isfinite(values).all(), "Nonfinite tabular prediction")
            if backend == "randomforest":
                np.testing.assert_allclose(values, parent.qualisr, atol=1e-12, rtol=0)
            frame[f"qualisr_{backend}"] = values
            # Recreate one fold from the config and compare every evaluation row
            # after serializing the newly fitted transform and model.
            if (split_seed, fold) == (42, 1):
                model_file = args.output / f"{backend}.joblib"
                joblib.dump((frozen, model, columns), model_file, compress=3)
                restored, head, saved_columns = joblib.load(model_file)
                actual = head.predict(restored.transform(raw.loc[evaluate])[saved_columns])
                np.testing.assert_allclose(actual, values, atol=1e-12, rtol=0)
                fresh = make_model(backend, {}, cfg["evaluation"]["model_seed"], cfg["evaluation"]["threads"])
                fresh.fit(train_x, raw.loc[train, "score"])
                repeated = fresh.predict(eval_x)
                np.testing.assert_allclose(repeated, values, atol=1e-12, rtol=0)
                replays.append({"backend": backend, **key, "prediction_rows": len(values),
                                "max_abs_error": float(np.max(np.abs(repeated - values))),
                                "model_sha256": sha256_file(model_file)})
        require(set(frame.sample_id) == set(parent.index), "MUSIQ evaluation IDs changed")
        predictions.append(frame)
        for name in ("musiq_direct", "musiq_finetuned", "qalign_direct",
                     *(f"qualisr_{backend}" for backend in BACKENDS)):
            scored = frame[["sample_id", name]].rename(columns={name: "prediction"}).merge(
                raw[METADATA], on="sample_id", validate="one_to_one")
            table, _ = metric_rows(scored)
            external = table[table.dataset.isin(["dsr-dataset", "isrgenqa"])]
            if len(external) == 2:
                table.loc[len(table)] = dict(dataset="external_mos_macro", score_type="mos",
                    aggregation="mean_per_dataset", n_samples=external.n_samples.sum(),
                    n_groups=2, plcc=external.plcc.mean(), srcc=external.srcc.mean())
            metrics.append(table.assign(model=name, **key))
    values = pd.concat(predictions, ignore_index=True)
    require(not values.duplicated(FOLD + ["sample_id"]).any(), "Duplicate comparison row")
    values.to_csv(args.output / "predictions.csv.gz", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(replays).to_csv(args.output / "reproduction.csv", index=False)
    write_json(args.output / "run.json", {
        "config": cfg, "pilot": args.pilot, "backends": BACKENDS, "params": {},
        "parent": str(args.parent), "parent_predictions_sha256": sha256_file(args.parent / "predictions.csv.gz"),
        "parent_weights_sha256": sha256_file(args.parent / "weights.csv"),
        "columns": columns, "fixed_dimensions": DIMS,
        "neural_scores": "Verified original MUSIQ predictions and checkpoints reused without refitting",
    })
    write_json(args.output / "verification.json", {
        "status": "passed", "folds": len(folds), "prediction_rows": len(values),
        "rf_parent_parity": True, "reproduction_backends": len(replays),
        "checks": ["parent full verification", "identical parser targets and evaluated IDs",
                   "original Set120-only train IDs", "train-only paired PCA, imputation and scaler",
                   "all RF parent predictions within 1e-12", "three one-fold deterministic replays"],
    })


def summarize_backends(args):
    require(json.loads((args.root / "verification.json").read_text())["status"] == "passed",
            "Unverified tabular MUSIQ comparison")
    predictions = read(args.root / "predictions.csv.gz")
    frame = predictions.merge(read(args.root / "samples.csv"), on="sample_id", validate="many_to_one")
    titles = {
        "musiq_direct": "Original MUSIQ", "musiq_finetuned": "Fine-tuned MUSIQ",
        "qualisr_randomforest": "QualiSR-Lab RF", "qualisr_xgb": "QualiSR-Lab XGBoost",
        "qualisr_catboost": "QualiSR-Lab CatBoost", "qalign_direct": "Direct Q-Align",
    }
    require(frame[FOLD].drop_duplicates().shape[0] == 15, "Incomplete grouped folds")
    quantiles, errors = [], []
    for keys, part in frame.groupby(FOLD):
        tags = dict(zip(FOLD, keys, strict=True))
        for model in titles:
            for dataset, rows in part.groupby("dataset"):
                values = rows[[model, "score"]].rename(columns={model: "prediction"})
                q = values.quantile([0, .05, .25, .5, .75, .95, 1]).T
                q.columns = ["min", "p05", "p25", "p50", "p75", "p95", "max"]
                quantiles.append(q.assign(finite=values.count(), rows=len(rows), dataset=dataset, model=model, **tags)
                                 .rename_axis("variable").reset_index())
            scored = part.loc[part.dataset.eq(BT) & part.score.notna()]
            values = scored[[model, "score", "correlation_group", "method"]].rename(columns={model: "prediction"})
            ranks = values.groupby("correlation_group")[["score", "prediction"]].rank(pct=True)
            delta = ranks.prediction - ranks.score
            table = values.assign(signed=delta, absolute=delta.abs()).groupby("method").agg(
                n=("score", "size"), mean_rank_error=("signed", "mean"),
                mean_absolute_rank_error=("absolute", "mean"),
            ).reset_index()
            errors.append(table.assign(model=model, **tags))
    args.output.mkdir()
    pd.concat(quantiles, ignore_index=True).to_csv(args.output / "output_quantiles.csv", index=False)
    pd.concat(errors, ignore_index=True).to_csv(args.output / "realsrq_rank_errors.csv", index=False)
    feature_plot(args.parent, args.output)
    prediction_plot(args.root, args.output, tuple(titles), titles)
    # The previous RF output diagnostics must remain identical for that backend.
    parent_quantiles = read(args.parent / "analysis/output_quantiles.csv")
    q = pd.concat(quantiles, ignore_index=True)
    reference = parent_quantiles.loc[parent_quantiles.model.eq("qualisr")]
    columns = FOLD + ["model", "dataset", "variable", "min", "p05", "p25", "p50", "p75", "p95", "max", "finite", "rows"]
    actual = q.loc[q.model.eq("qualisr_randomforest")].assign(model="qualisr")
    order = FOLD + ["dataset", "variable"]
    pd.testing.assert_frame_equal(actual[columns].sort_values(order, ignore_index=True),
                                  reference[columns].sort_values(order, ignore_index=True),
                                  atol=1e-12, rtol=0)
    require(np.isfinite(q[["min", "max"]]).all().all(), "Nonfinite prediction quantiles")
    write_json(args.output / "verification.json", {
        "status": "passed", "models": list(titles), "backends": BACKENDS, "folds": 15,
        "parent_rf_quantile_parity": True, "feature_distribution": "shared original train-fitted PCA inputs",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit tabular backends on the MUSIQ partitions")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--pilot", action="store_true")
    summary = commands.add_parser("summarize", help="Summarize transfer distributions for all backends")
    summary.add_argument("--root", type=Path, required=True)
    for cmd in (fit, summary):
        cmd.add_argument("--parent", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    plot = commands.add_parser("plot", help="Plot feature and prediction shifts for the MUSIQ run")
    plot.add_argument("root", type=Path)
    plot.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "fit":
        with threadpool_limits(limits=2):
            fit_backends(args)
    else:
        matplotlib.use("Agg")
        if args.command == "summarize":
            summarize_backends(args)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            feature_plot(args.root, args.output)
            prediction_plot(args.root, args.output)


if __name__ == "__main__":
    main()
