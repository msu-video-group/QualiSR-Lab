"""Verify the paired MUSIQ transfer and summarize predictions by dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import read_csv, require, sha256_file
from experiments.finetune_musiq import FOLD, MODELS, partition_table, partitions
from experiments.run_grouped import write_json
from experiments.transfer import DISRQAD, ISRGEN, REALSRQ, SET120, metric_rows


def verify(root):
    run = json.loads((root / "run.json").read_text())
    status = json.loads((root / "status.json").read_text())
    require(status["status"] == "complete", "MUSIQ run is incomplete")
    samples = read_csv(root, "samples.csv")
    assignment = read_csv(root, "partitions.csv.gz")
    scores = read_csv(root, "inner_scores.csv.gz")
    weights = read_csv(root, "weights.csv")
    predictions = read_csv(root, "predictions.csv.gz")
    require(samples.sample_id.is_unique, "Duplicate sample")
    require(not assignment.duplicated().any(), "Duplicate role assignment")
    require(not predictions.duplicated(FOLD + ["sample_id"]).any(), "Duplicate prediction")
    require(np.isfinite(predictions[MODELS[:3]]).all().all(), "Incomplete paired prediction")
    require(not np.isinf(predictions.qalign_direct).any(), "Infinite Q-Align score")
    for metric in ("musiq_direct", "qalign_direct"):
        require(predictions.groupby("sample_id")[metric].nunique(dropna=False).le(1).all(),
                f"Frozen {metric} differs across folds")
    require(weights.initial_state.nunique() == 1 and weights.initial_state.ne(weights.trained_state).all(),
            "MUSIQ starting or final weights differ from protocol")
    require(np.isfinite(weights.max_mlp_update).all() and weights.max_mlp_update.gt(0).all(),
            "MUSIQ MLP update was absent or nonfinite")
    cfg = run["config"]["evaluation"]
    expected = list(partitions(samples, cfg, run["pilot"]))
    keys = {(seed, fold) for seed, fold, *_ in expected}
    for frame in (assignment, scores, weights, predictions):
        require(set(frame[FOLD].itertuples(index=False, name=None)) == keys, "Unexpected fold coverage")
    require(len(expected) == len(weights) == status["folds"], "Unexpected number of weight files")
    require(status["image_rows"] == assignment.sample_id.nunique(), "Unexpected active sample count")
    train_ids = set(samples.loc[samples.dataset.eq(SET120), "sample_id"])
    require(set(assignment.loc[assignment.role.str.endswith("train"), "sample_id"]) <= train_ids,
            "Non-Set120 training row")
    for seed, fold, train, evaluate, inner in expected:
        expected_rows = partition_table(samples, seed, fold, train, evaluate, inner)
        actual = assignment.loc[assignment.split_seed.eq(seed) & assignment.fold.eq(fold)]
        order = FOLD + ["role", "sample_id"]
        pd.testing.assert_frame_equal(actual.sort_values(order).reset_index(drop=True),
                                      expected_rows.sort_values(order).reset_index(drop=True))
        valid = predictions.loc[predictions.split_seed.eq(seed) & predictions.fold.eq(fold)]
        require(set(valid.sample_id) == set(samples.loc[evaluate, "sample_id"]), "Prediction IDs differ")
        w = weights.loc[weights.split_seed.eq(seed) & weights.fold.eq(fold)].iloc[0]
        trace = scores.loc[scores.split_seed.eq(seed) & scores.fold.eq(fold)]
        means = trace.groupby("epoch").srcc.mean()
        require(np.isfinite(trace[["plcc", "srcc"]]).all().all(), "Nonfinite inner score")
        require(list(means.index) == list(range(1, len(means) + 1)) and len(means) <= run["epochs"],
                "Invalid epoch budget")
        require(trace.groupby("epoch").inner_fold.agg(set).eq(set(range(1, len(inner) + 1))).all()
                and not trace.duplicated(["epoch", "inner_fold"]).any(), "Incomplete inner fold scores")
        best = int(means.idxmax())
        require(best == w.selected_epoch, "Selected epoch differs from mean inner SRCC maximum")
        for epoch in means.index[:-1]:
            require(epoch - means.loc[:epoch].idxmax() < run["patience"], "Late early stopping")
        require(len(means) == run["epochs"] or len(means) - best == run["patience"], "Premature stopping")
        require(sha256_file(root / f"musiq_{seed}_{fold}.pt") == w.checkpoint_sha256, "Weight file changed")
    if not run["pilot"]:
        bt = samples.loc[samples.dataset.eq(REALSRQ)]
        require(len(bt) == 1620 and bt.score.notna().sum() == 1617 and bt.correlation_group.nunique() == 180
                and bt.source_group.nunique() == 60, "Unexpected full RealSRQ population")
    return run, predictions.merge(samples, on="sample_id", validate="many_to_one")


def summarize(root):
    run, joined = verify(root)
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    metrics, group_metrics, distribution, rank_errors = [], [], [], []
    for (seed, fold), pair in joined.groupby(FOLD):
        for scope, population, models in [
            ("primary", pair, MODELS[:3]),
            ("qalign_common", pair[pair.qalign_direct.notna()], MODELS),
        ]:
            for model in models:
                subset = population.rename(columns={model: "prediction"})
                summary, bt = metric_rows(subset)
                for name, table in [("external_mos_macro", summary[summary.dataset.isin([DISRQAD, ISRGEN])])]:
                    if len(table) == 2:
                        summary.loc[len(summary)] = dict(
                            dataset=name, score_type="mos", aggregation="mean_per_dataset",
                            n_samples=table.n_samples.sum(), n_groups=len(table), plcc=table.plcc.mean(),
                            srcc=table.srcc.mean(),
                        )
                info = dict(split_seed=seed, fold=fold, scope=scope, model=model)
                metrics.append(summary.assign(**info))
                group_metrics.append(bt.assign(**info))
                if scope != "primary" and model != "qalign_direct":
                    continue
                for dataset, rows in subset.groupby("dataset"):
                    quantiles = rows[["prediction", "score"]].quantile([0, .05, .25, .5, .75, .95, 1]).T
                    quantiles.columns = ["min", "p05", "p25", "p50", "p75", "p95", "max"]
                    counts = rows[["prediction", "score"]].count()
                    distribution.append(quantiles.assign(dataset=dataset, finite=counts, rows=len(rows), **info)
                                        .rename_axis("variable").reset_index())
                scored = subset[subset.dataset.eq(REALSRQ) & subset.score.notna()].copy()
                ranks = scored.groupby("correlation_group")[["score", "prediction"]].rank(pct=True)
                signed = ranks.prediction - ranks.score
                by_method = scored.assign(signed=signed, absolute=signed.abs()).groupby("method").agg(
                    n=("sample_id", "size"), mean_rank_error=("signed", "mean"), mean_absolute_rank_error=("absolute", "mean"),
                ).reset_index()
                rank_errors.append(by_method.assign(**info))
    metric = pd.concat(metrics, ignore_index=True)
    metric.to_csv(out / "metrics.csv", index=False)
    metric.groupby(["scope", "model", "dataset", "score_type"]).agg(
        plcc_mean=("plcc", "mean"), srcc_mean=("srcc", "mean"),
        srcc_min=("srcc", "min"), srcc_max=("srcc", "max"), folds=("srcc", "count"),
        min_scored=("n_samples", "min"), max_scored=("n_samples", "max"),
    ).to_csv(out / "summary.csv")
    pd.concat(group_metrics, ignore_index=True).to_csv(out / "realsrq_group_metrics.csv.gz", index=False)
    pd.concat(distribution, ignore_index=True).to_csv(out / "output_quantiles.csv", index=False)
    pd.concat(rank_errors, ignore_index=True).to_csv(out / "realsrq_rank_errors.csv", index=False)
    primary = metric[metric.scope.eq("primary") & metric.dataset.isin(
        [SET120, DISRQAD, ISRGEN, "external_mos_macro"]
    )]
    wide = primary.pivot(index=FOLD + ["dataset"], columns="model", values=["plcc", "srcc"])
    contrasts = []
    for name in ("plcc", "srcc"):
        internal = wide[name].xs(SET120, level="dataset")
        for dataset in (DISRQAD, ISRGEN, "external_mos_macro"):
            external = wide[name].xs(dataset, level="dataset")
            gap = internal - external
            contrasts.append(pd.DataFrame({
                "dataset": dataset, "metric": name,
                "musiq_finetuned_minus_qualisr_external": external.musiq_finetuned - external.qualisr,
                "musiq_finetuned_minus_direct_external": external.musiq_finetuned - external.musiq_direct,
                "musiq_finetuned_gap": gap.musiq_finetuned, "qualisr_gap": gap.qualisr,
                "musiq_direct_gap": gap.musiq_direct,
                "musiq_finetuned_minus_qualisr_gap": gap.musiq_finetuned - gap.qualisr,
            }).reset_index())
    pd.concat(contrasts, ignore_index=True).to_csv(out / "mos_transfer_differences.csv", index=False)
    write_json(out / "verification.json", {
        "status": "passed", "pilot": run["pilot"], "prediction_rows": len(joined),
        "checks": ["exact outer and inner IDs", "identical paired evaluation rows and targets",
                   "Set120-only fitting", "frozen scores invariant across folds", "early stopping and selected epoch",
                   "MLP weight update and saved checkpoint SHA256", "complete RealSRQ population in full run"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
