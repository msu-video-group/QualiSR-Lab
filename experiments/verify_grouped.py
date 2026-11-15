"""Verify a completed experiment run and write a compact, source-traceable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qualisr.regressors import safe_corr, validation_correlations


def read_results_csv(path: Path) -> pd.DataFrame:
    """Preserve serialized float ranks/ties when recomputing Spearman scores."""
    return pd.read_csv(path, float_precision="round_trip")


def verify_and_report(root: Path, report: Path) -> None:
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Only a completed run can be reported as verified")
    run = json.loads((root / "run.json").read_text())
    settings, options = run["config"]["evaluation"], run["options"]
    seeds = options["split_seeds"] or settings["split_seeds"]
    folds = min(options["fold_limit"] or settings["outer_folds"], settings["outer_folds"])
    expected_folds = {(seed, fold) for seed in seeds for fold in range(1, folds + 1)}
    assignments = read_results_csv(root / "fold_assignments.csv")
    actual_folds = set(map(tuple, assignments[["split_seed", "fold"]].drop_duplicates().to_numpy()))
    if actual_folds != expected_folds:
        raise ValueError("Completed fold assignments do not match the recorded run options")
    samples = read_results_csv(root / "samples.csv").set_index("sample_id")
    specs = json.loads((root / "experiment_specs.json").read_text())
    models = options["models"] or settings["models"]
    checked_models = 0
    method_rows = []
    method_omissions = {}
    for seed, fold in sorted(expected_folds):
        folder = root / f"seed_{seed}" / f"fold_{fold:02d}"
        assignment = assignments[(assignments.split_seed == seed) & (assignments.fold == fold)]
        train_ids = set(assignment.loc[assignment.partition == "train", "sample_id"])
        heldout_ids = set(assignment.loc[assignment.partition == "heldout", "sample_id"])
        if set(samples.loc[list(train_ids), "source_group"]) & set(samples.loc[list(heldout_ids), "source_group"]):
            raise ValueError("A source crosses a saved outer partition")
        fitted = set(pd.read_csv(folder / "outer_transform/fit_samples.csv")["sample_id"])
        if fitted != train_ids:
            raise ValueError("Outer transform fit IDs do not match the outer training partition")
        for inner in folder.glob("inner_*"):
            fit_ids = set(pd.read_csv(inner / "fit_samples.csv")["sample_id"])
            valid_ids = set(pd.read_csv(inner / "validation_samples.csv")["sample_id"])
            if fit_ids & valid_ids or fit_ids | valid_ids != train_ids:
                raise ValueError("Inner fit/validation IDs do not partition the outer training samples")
            if set(samples.loc[list(fit_ids), "source_group"]) & set(samples.loc[list(valid_ids), "source_group"]):
                raise ValueError("A source crosses a saved inner partition")
        targets = []
        for spec in specs:
            for model in models:
                targets.append(folder / spec["name"] / model / "fixed")
                if spec["tune"] and not options["skip_tuning"]:
                    targets.append(folder / spec["name"] / model / "tuned")
        if not options["skip_selection"]:
            targets.append(folder / "compact_selection/ridge")
        expected_ids = heldout_ids | set(samples.index[samples.role == "external"])
        for target in targets:
            if not (target / "model.joblib").is_file():
                raise FileNotFoundError(target / "model.joblib")
            prediction = read_results_csv(target / "predictions.csv")
            if prediction.sample_id.duplicated().any() or set(prediction.sample_id) != expected_ids:
                raise ValueError(f"Prediction coverage differs: {target}")
            np.testing.assert_allclose(prediction.score, samples.loc[prediction.sample_id, "score"], equal_nan=True)
            if not np.isfinite(prediction.prediction).all():
                raise ValueError(f"Nonfinite predictions: {target}")
            experiment = target.relative_to(folder).parts[0]
            model = target.relative_to(folder).parts[1]
            track = "nested_selection" if experiment == "compact_selection" else target.name
            for (scope, dataset, method), group in prediction.loc[prediction.score_type == "mos"].groupby(
                ["evaluation", "dataset", "method"]
            ):
                if group.source_group.nunique() < 2:
                    key = f"{dataset}: fewer than two sources per parser method ID"
                    method_omissions[key] = method_omissions.get(key, 0) + 1
                    continue
                plcc, srcc = safe_corr(group.score, group.prediction)
                method_rows.append({"split_seed": seed, "fold": fold, "experiment": experiment,
                                    "model": model, "track": track, "evaluation": scope, "dataset": dataset,
                                    "method": method, "n_samples": len(group),
                                    "n_sources": group.source_group.nunique(), "plcc": plcc, "srcc": srcc})
            for scope, group in prediction.groupby("evaluation"):
                _, _, recalculated, _ = validation_correlations(group.score, group.prediction, group)
                saved = read_results_csv(target / f"metrics_{scope}.csv")
                paired = saved.merge(recalculated, on="dataset", validate="one_to_one", suffixes=("_saved", "_recomputed"))
                if len(paired) != len(saved) or len(paired) != len(recalculated):
                    raise ValueError(f"Dataset coverage differs: {target}")
                for metric in ("plcc", "srcc", "n_samples", "n_groups"):
                    np.testing.assert_allclose(paired[f"{metric}_saved"], paired[f"{metric}_recomputed"],
                                               rtol=1e-9, atol=1e-10, equal_nan=True)
            checked_models += 1
    table = read_results_csv(root / "metrics_all_folds.csv")
    summary = read_results_csv(root / "metrics_summary.csv")
    pd.DataFrame(method_rows).to_csv(root / "method_metrics.csv", index=False)
    write_method_notes = {
        "scope": "Diagnostic correlations within repeated MOS parser-method groups; not a leave-method-out experiment",
        "omitted_method_groups": method_omissions,
        "bt_policy": "No per-method global BT correlation: disconnected comparison-group scales are not pooled",
        "isrgenqa_note": "The existing parser uses SR filenames as method IDs, so many groups have only one source; no algorithm identity is inferred",
    }
    (root / "method_metrics_notes.json").write_text(json.dumps(write_method_notes, indent=2) + "\n")
    pair_rows = []
    pairs = [("nr", name) for name in ("nr_ref", "nr_diff", "nr_fr", "nr_stats", "nr_all")]
    pairs += [("full_without_stats", "full_with_stats"), ("nr_fr_bicubic", "nr_fr"), ("nr_fr_span", "nr_fr")]
    pair_keys = ["model", "track", "split_seed", "fold", "evaluation", "dataset"]
    for baseline, added in pairs:
        if baseline not in set(table.experiment) or added not in set(table.experiment):
            continue
        comparison = table[table.experiment == baseline].merge(
            table[table.experiment == added], on=pair_keys, suffixes=("_baseline", "_added"), validate="one_to_one"
        )
        for row in comparison.to_dict("records"):
            pair_rows.append({**{key: row[key] for key in pair_keys}, "baseline": baseline, "comparison": added,
                              "delta_plcc": row["plcc_added"] - row["plcc_baseline"],
                              "delta_srcc": row["srcc_added"] - row["srcc_baseline"]})
    pd.DataFrame(pair_rows).to_csv(root / "paired_differences.csv", index=False)
    verification = {"status": "passed", "outer_folds": len(expected_folds), "model_prediction_sets": checked_models,
                    "checks": ["source separation", "exact transform fit IDs", "inner partition separation",
                               "paired prediction coverage", "unchanged targets", "finite predictions",
                               "recomputed per-dataset MOS/BT metrics and sample/group counts"]}
    (root / "result_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    lines = ["# Verified grouped experiment results", "", f"Run: `{root}`", "",
             f"Verification passed for **{len(expected_folds)} outer folds and {checked_models} model prediction sets**.",
             "All required models, split/fit IDs, paired evaluation rows, targets, and recomputed dataset metrics passed.",
             "", "The numbers below are means of backend correlation coefficients, not ensemble predictions. "
             "Fold/repeat SD describes split variability; no sampling confidence interval or significance test is implied.", ""]
    main = [s["name"] for s in specs if s["tune"]]
    for scope in ("internal", "external"):
        lines.extend([f"## {scope.capitalize()} evaluation", "",
                      "| Track | Configuration | Dataset | Mean PLCC | Mean SRCC |", "|---|---|---|---:|---:|"])
        grouped = table[(table.experiment.isin(main)) & (table.evaluation == scope)].groupby(
            ["track", "experiment", "dataset"]
        )[["plcc", "srcc"]].mean()
        for (track, experiment, dataset), row in grouped.iterrows():
            lines.append(f"| {track} | {experiment} | {dataset} | {row.plcc:.4f} | {row.srcc:.4f} |")
        lines.append("")
    lines.extend(["## Compact selection", ""])
    compact = summary[summary.experiment == "compact_selection"]
    for row in compact.itertuples():
        lines.append(f"- {row.evaluation}, {row.dataset}: PLCC {row.plcc_mean:.4f} ± {row.plcc_std:.4f}; "
                     f"SRCC {row.srcc_mean:.4f} ± {row.srcc_std:.4f}.")
    selected_rows = []
    for path in root.glob("seed_*/fold_*/compact_selection/ridge/inner_selection.csv"):
        history = read_results_csv(path)
        selected_rows.extend(history.loc[history.selected, "features"].tolist())
    if selected_rows:
        counts = pd.Series([name for subset in selected_rows for name in subset.split("|")]).value_counts()
        lines.extend(["", "Selected-feature frequencies (PCA component axes are not aligned across refitted folds):", ""])
        lines.extend(f"- `{name}`: {count}/{len(selected_rows)} outer fits." for name, count in counts.items())
    lines.extend(["", "## Interpretation limits", "",
                  "- External scores come from source-only outer-training fits; they are not full-source-refit transfer results.",
                  "- Preserve poor/negative external results and assess each dataset separately; no universal RR advantage is assumed.",
                  "- Three missing RealSRQ targets retain predictions but are unscored by the existing evaluator.",
                  "- The dataset parsers normalize the targets. data_provenance.json records the input files and source-content checks.",
                  "- Per-backend means/SDs, all fixed ablations, and tuning histories are retained in the accompanying CSVs.", ""])
    lines.extend(["Per-method MOS diagnostics are in `method_metrics.csv`; group omissions and BT limitations are explicit "
                  "in `method_metrics_notes.json`. Paired per-fold effects are in `paired_differences.csv`; no significance "
                  "test is inferred from their signs.", ""])
    report.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    verify_and_report(args.run_dir.resolve(), args.report.resolve())
