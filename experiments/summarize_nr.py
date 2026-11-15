"""Verify NR head-adaptation records and compare matched transfer correlations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import read_csv, require, sha256_file
from experiments.finetune_musiq import FOLD, partition_table, partitions
from experiments.finetune_nr import MAX_EPOCHS, METRICS, PATIENCE
from experiments.run_grouped import write_json
from experiments.transfer import DISRQAD, ISRGEN, REALSRQ, SET120, metric_rows

KEY = FOLD + ["sample_id"]


def paired_bt(groups, direct, tuned, coefficient):
    """Average differences only over BT groups defined for both predictors."""
    selected = groups[groups.scope.eq("primary") & groups.model.isin([direct, tuned])]
    paired = selected.pivot(index=FOLD + ["dataset", "correlation_group"], columns="model", values=coefficient)
    common = paired.dropna(subset=[direct, tuned])
    effects = (common[tuned] - common[direct]).rename("finetuned_minus_direct")
    return effects.groupby(FOLD + ["dataset"]).agg(
        finetuned_minus_direct="mean", n_common_groups="count").reset_index()


def verify(root):
    run = json.loads((root / "run.json").read_text())
    status = json.loads((root / "status.json").read_text())
    require(run["metric"] in METRICS and status["status"] == "complete", "NR run incomplete")
    samples = read_csv(root, "samples.csv")
    predictions = read_csv(root, "predictions.csv.gz")
    assignments = read_csv(root, "partitions.csv.gz")
    scores = read_csv(root, "inner_scores.csv.gz")
    weights = read_csv(root, "weights.csv")
    require(samples.sample_id.is_unique and not predictions.duplicated(KEY).any(), "Duplicate IDs")
    require(not assignments.duplicated().any() and not weights.duplicated(FOLD).any(), "Duplicate assignments/fits")
    require(np.isfinite(predictions[["direct", "finetuned"]]).all().all(), "Nonfinite NR prediction")
    require(predictions.groupby("sample_id").direct.nunique().eq(1).all(), "Frozen scores changed between fits")
    require(weights.initial_state.nunique() == 1 and weights.initial_state.ne(weights.trained_state).all(),
            "Invalid parameter initialization/update")
    require(np.isfinite(weights.max_parameter_update).all() and weights.max_parameter_update.gt(0).all(),
            "Head parameters did not change")
    folds = list(partitions(samples, run["config"]["evaluation"], run["pilot"]))
    keys = {(s, f) for s, f, *_ in folds}
    require(len(folds) == len(weights) == status["folds"], "Wrong fit count")
    for table in (predictions, assignments, scores, weights):
        require(set(table[FOLD].itertuples(index=False, name=None)) == keys, "Wrong fold coverage")
    require(assignments.sample_id.nunique() == status["image_rows"], "Wrong image count")
    for seed, fold, train, evaluate, inner in folds:
        def select(frame, seed=seed, fold=fold):
            return frame.loc[frame.split_seed.eq(seed) & frame.fold.eq(fold)]
        actual = select(assignments)
        expected = partition_table(samples, seed, fold, train, evaluate, inner)
        order = FOLD + ["role", "sample_id"]
        pd.testing.assert_frame_equal(actual.sort_values(order).reset_index(drop=True),
                                      expected.sort_values(order).reset_index(drop=True))
        require(set(select(predictions).sample_id) == set(samples.loc[evaluate, "sample_id"]),
                "Prediction population differs")
        trace, w = select(scores), select(weights).iloc[0]
        require(not trace.duplicated(["epoch", "inner_fold"]).any(), "Duplicate selection scores")
        require(np.isfinite(trace[["plcc", "srcc"]]).all().all(), "Nonfinite selection score")
        means = trace.groupby("epoch").srcc.mean()
        require(list(means.index) == list(range(1, len(means) + 1)) and len(means) <= MAX_EPOCHS,
                "Invalid epoch budget")
        require(trace.groupby("epoch").inner_fold.agg(set).eq(set(range(1, len(inner) + 1))).all(),
                "Missing inner fold")
        best = int(means.idxmax())
        require(best == w.selected_epoch, "Epoch selection differs")
        require(all(epoch - means.loc[:epoch].idxmax() < PATIENCE for epoch in means.index[:-1]),
                "Late stopping")
        require(len(means) == MAX_EPOCHS or len(means) - best == PATIENCE, "Premature stopping")
        require(sha256_file(root / f"{run['metric']}_{seed}_{fold}.pt") == w.checkpoint_sha256,
                "Saved parameters changed")
    if not run["pilot"]:
        bt = samples[samples.dataset.eq(REALSRQ)]
        require(len(bt) == 1620 and bt.score.notna().sum() == 1617 and bt.correlation_group.nunique() == 180,
                "RealSRQ population changed")
    return run, samples, predictions


def summarize(root, parent, tabular_parent, output=None):
    run, samples, predictions = verify(root)
    require(sha256_file(parent / "predictions.csv.gz") == run["parent_predictions_sha256"],
            "MUSIQ parent predictions changed")
    require(json.loads((tabular_parent / "verification.json").read_text())["status"] == "passed",
            "Tabular parent unverified")
    tabular_run = json.loads((tabular_parent / "run.json").read_text())
    require(tabular_run["config"] == run["config"] and not tabular_run["pilot"], "Tabular config differs")
    require(tabular_run["parent_predictions_sha256"] == run["parent_predictions_sha256"],
            "Tabular and neural parents differ")
    prior = read_csv(tabular_parent, "predictions.csv.gz").set_index(KEY)
    require(prior.index.is_unique, "Duplicate parent prediction")
    prior = prior.loc[pd.MultiIndex.from_frame(predictions[KEY])].reset_index()
    wide = predictions.rename(columns={n: f"{run['metric']}_{n}" for n in ("direct", "finetuned")})
    wide = wide.merge(prior, on=KEY, validate="one_to_one")
    joined = wide.merge(samples, on="sample_id", validate="many_to_one")
    models = [f"{run['metric']}_direct", f"{run['metric']}_finetuned", "musiq_direct", "musiq_finetuned",
              "qualisr_randomforest", "qualisr_xgb", "qualisr_catboost"]
    # Live Q-Align and the earlier cached Q-Align have distinct identities.
    # Preserve all primary rows; the cached baseline has a separate common scope.
    if run["metric"] == "qalign":
        joined = joined.rename(columns={"qalign_direct_y": "qalign_cached", "qalign_direct_x": "qalign_direct"})
    else:
        joined = joined.rename(columns={"qalign_direct": "qalign_cached"})
    require(np.isfinite(joined[models]).all().all(), "Missing matched model prediction")
    metrics, groups = [], []
    for (seed, fold), rows in joined.groupby(FOLD):
        for scope, population, selected in [("primary", rows, models),
                                             ("qalign_common", rows[rows.qalign_cached.notna()], models + ["qalign_cached"])]:
            for model in selected:
                table, bt = metric_rows(population.assign(prediction=population[model]))
                external = table[table.dataset.isin([DISRQAD, ISRGEN])]
                require(len(external) == 2, "Missing MOS transfer dataset")
                table.loc[len(table)] = dict(dataset="external_mos_macro", score_type="mos",
                    aggregation="mean_per_dataset", n_samples=external.n_samples.sum(), n_groups=2,
                    plcc=external.plcc.mean(), srcc=external.srcc.mean())
                info = dict(split_seed=seed, fold=fold, scope=scope, model=model)
                metrics.append(table.assign(**info))
                groups.append(bt.assign(**info))
    values = pd.concat(metrics, ignore_index=True)
    group_values = pd.concat(groups, ignore_index=True)
    primary = values[values.scope.eq("primary")]
    contrasts, gaps = [], []
    for coefficient in ("plcc", "srcc"):
        table = primary.pivot(index=FOLD + ["dataset"], columns="model", values=coefficient)
        direct, tuned = f"{run['metric']}_direct", f"{run['metric']}_finetuned"
        delta = (table[tuned] - table[direct]).rename("finetuned_minus_direct").reset_index()
        delta = delta[delta.dataset.ne(REALSRQ)]
        delta = pd.concat([delta, paired_bt(group_values, direct, tuned, coefficient)], ignore_index=True)
        contrasts.append(delta.assign(coefficient=coefficient))
        internal = table.xs(SET120, level="dataset")
        for dataset in (DISRQAD, ISRGEN, "external_mos_macro"):
            gap = internal - table.xs(dataset, level="dataset")
            gaps.append(gap.rename_axis(columns="model").stack().rename("internal_minus_external")
                        .reset_index().assign(dataset=dataset, coefficient=coefficient))
    out = root / "analysis" if output is None else output
    out.mkdir(exist_ok=False)
    values.to_csv(out / "metrics.csv", index=False)
    group_values.to_csv(out / "realsrq_groups.csv.gz", index=False)
    values.groupby(["scope", "model", "dataset", "score_type"]).agg(
        plcc_mean=("plcc", "mean"), plcc_sd=("plcc", "std"),
        srcc_mean=("srcc", "mean"), srcc_sd=("srcc", "std"), folds=("srcc", "count"),
        min_scored=("n_samples", "min"), max_scored=("n_samples", "max"),
    ).to_csv(out / "summary.csv")
    pd.concat(contrasts, ignore_index=True).to_csv(out / "paired_deltas.csv", index=False)
    pd.concat(gaps, ignore_index=True).to_csv(out / "mos_transfer_gaps.csv", index=False)
    write_json(out / "verification.json", {
        "status": "passed", "pilot": run["pilot"], "prediction_rows": len(predictions),
        "parent_tabular_predictions_sha256": sha256_file(tabular_parent / "predictions.csv.gz"),
        "checks": ["exact grouped outer and inner IDs", "Set120-only fitting and epoch selection",
                   "finite paired rows", "frozen scores invariant", "trained head and checkpoint hashes",
                   "matching three-backend and MUSIQ parent predictions", "separate MOS macro and BT groups",
                   "paired BT effects use jointly defined comparison groups"],
        "interpretation": "Exploratory comparisons of frozen and adapted NR models",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "parent", "tabular-parent"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.parent, args.tabular_parent, args.output)
