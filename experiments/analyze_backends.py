"""Extend the verified paper statistics to all three tabular regressors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_bt_pairs import joint_delta
from experiments.analyze_statistics import (
    BACKENDS,
    BT,
    EXTERNAL,
    FOLD,
    MACRO,
    META,
    SET,
    Analysis,
    combine,
    read,
    run_model_seeds,
    scene_draws,
)
from experiments.data import require, sha256_file
from experiments.run_grouped import write_json
from experiments.transfer import verify


def common_bt(left, right, model):
    models = BACKENDS if model == "backend_mean" else (model,)
    aligned = []
    names = [f"pred_{backend}" for backend in models]
    ids = FOLD + ["sample_id"]
    for side in (left, right):
        merged = None
        for backend, name in zip(models, names, strict=True):
            frame = side[backend]
            frame = frame.loc[frame.dataset.eq(BT)].dropna(subset=["score", "prediction"]).sort_values(ids)
            frame = frame.reset_index(drop=True)
            info = frame[ids + META[1:]]
            if merged is None:
                merged = info.copy()
            pd.testing.assert_frame_equal(merged[info.columns], info)
            merged[name] = frame.prediction
        aligned.append(merged)
    a, b = aligned
    pd.testing.assert_frame_equal(a[ids + META[1:]], b[ids + META[1:]])
    collapsed = []
    for side in aligned:
        metadata = side[META].drop_duplicates()
        require(metadata.sample_id.is_unique, "BT metadata changes across folds")
        collapsed.append(metadata.merge(side.groupby("sample_id")[names].mean(), on="sample_id",
                                        validate="one_to_one"))
    effects, keep = joint_delta(*collapsed, names)
    groups = collapsed[0].groupby("correlation_group").source_group.first()
    scenes = pd.Index(sorted(groups.unique()))
    draws = scene_draws(BT, len(scenes), 10000)[:, scenes.get_indexer(groups[keep])]
    totals = draws.sum(axis=1, keepdims=True)
    reps = np.divide(draws @ effects, totals, out=np.full((len(draws), 2), np.nan), where=totals > 0)
    require(np.isfinite(reps).all(), "Degenerate common BT bootstrap")
    rows = []
    for keys, part in a.groupby(FOLD):
        effect, support = joint_delta(part, b.loc[part.index], names)
        rows.append(dict(zip(FOLD, keys, strict=True), plcc=effect[:, 0].mean(),
                         srcc=effect[:, 1].mean(), n_groups=int(support.sum())))
    folds = pd.DataFrame(rows)
    ci = np.quantile(reps, [.025, .975], axis=0)
    return [dict(metric=metric, conditional_point=effects[:, i].mean(), ci_low=ci[0, i], ci_high=ci[1, i],
                 fold_mean=folds[metric].mean(), fold_sd=folds[metric].std(), n_scenes=len(scenes),
                 n_common_groups=int(keep.sum()), min_fold_groups=int(folds.n_groups.min()))
            for i, metric in enumerate(("plcc", "srcc"))]


def run(args):
    args.output.mkdir()
    analysis = Analysis(1000 if args.pilot else 10000)
    cells, frames = {}, {}
    summaries, paired, ordinary, influence, seeds, verified = [], [], [], [], [], []
    stage = "pilot" if args.pilot else "full"
    for backend in BACKENDS:
        for pool in ("historical", "qalign"):
            root = args.root / f"{stage}-{backend}-{pool}"
            manifest, fits, predictions = verify(root)
            require(manifest["backend"] == backend, "Wrong backend manifest")
            info = fits.set_index("fit_id")
            digest = sha256_file(root / "predictions.csv.gz")
            verified.append({"backend": backend, "pool": pool, "prediction_sets": len(fits),
                             "prediction_rows": len(predictions), "predictions_sha256": digest})
            summary = read(root / "analysis/summary.csv").assign(backend=backend, pool=pool)
            summaries.append(summary)
            paired.append(read(root / "analysis/paired_differences.csv").assign(backend=backend, pool=pool))
            if pool == "historical":
                ordinary.append(read(root / "analysis/set120_common_ordinary_metrics.csv").assign(backend=backend))
                influence.append(read(root / "analysis/set120_leave_one_source_out.csv").assign(backend=backend))
                if not args.pilot:
                    directory = args.output / backend
                    directory.mkdir()
                    run_model_seeds(root, directory)
                    seeds.append(read(directory / "model_seed_summary.csv").assign(backend=backend))
                continue
            selected = predictions.merge(info, on="fit_id", validate="many_to_one")
            for (case, spec), frame in selected.loc[selected.track.eq("impact")].groupby(["case", "spec"]):
                name = f"{case}/{spec}"
                cells[name, backend] = analysis.cell(frame, "13", name, backend)
                frames.setdefault(name, {})[backend] = frame
    for name in sorted(frames):
        cells[name, "backend_mean"] = analysis.mean_backend(
            [cells[name, backend] for backend in BACKENDS], "13", name)
    # The first three primary contrasts already cover all backends. Preserve
    # their exact original results and replace the RF-only transfer contrasts.
    if not args.pilot:
        old = read(args.previous_primary)
        keep = old[old.name.isin(["nr_to_nr_ref", "candidate_add_wasd", "all_rr_add_wasd"])]
        require(len(keep) == 6, "Missing unchanged primary contrasts")
        analysis.primary.extend(keep.to_dict("records"))
    bt = []
    for model in (*BACKENDS, "backend_mean"):
        effects = []
        for suffix, spec in (("no_wasd", "baseline_qalign"), ("wasd", "baseline_qalign_stats")):
            a, b = f"dsr_only/{spec}", f"dsr_plus_set120/{spec}"
            name = f"add_set120_{suffix}"
            effect = analysis.pair(cells[a, model], cells[b, model], "13", name, model,
                                   MACRO if model == "backend_mean" and not args.pilot else None)
            effects.append(effect)
            bt.extend(dict(name=name, model=model, **row) for row in common_bt(frames[a], frames[b], model))
        analysis.pair(*effects, "13", "set120_by_wasd_interaction", model)
        for case in ("dsr_only", "dsr_plus_set120"):
            a, b = f"{case}/baseline_qalign", f"{case}/baseline_qalign_stats"
            name = f"{case}_add_wasd"
            analysis.pair(cells[a, model], cells[b, model], "13", name, model)
            bt.extend(dict(name=name, model=model, **row) for row in common_bt(frames[a], frames[b], model))
    musiq = args.root / "musiq-full"
    require(json.loads((musiq / "verification.json").read_text())["status"] == "passed", "Unverified MUSIQ inputs")
    original = read(musiq / "predictions.csv.gz").merge(read(musiq / "samples.csv"),
                                                       on="sample_id", validate="many_to_one")
    direct = {}
    for name in ("musiq_direct", "musiq_finetuned", "qalign_direct"):
        direct[name] = analysis.cell(original[META + FOLD].assign(prediction=original[name]),
                                     "transfer", name, name)
    models = {}
    musiq_frames = {}
    for backend in BACKENDS:
        frame = original[META + FOLD].assign(prediction=original[f"qualisr_{backend}"])
        models[backend] = analysis.cell(frame, "transfer", "qualisr", backend)
        musiq_frames[backend] = frame
    models["backend_mean"] = analysis.mean_backend([models[b] for b in BACKENDS], "transfer", "qualisr")
    tuned_frames = {b: original[META + FOLD].assign(prediction=original.musiq_finetuned) for b in BACKENDS}
    for backend, values in models.items():
        # Repeat the fixed neural endpoint's signature when averaging backend correlations.
        tuned = direct["musiq_finetuned"]
        if backend == "backend_mean":
            tuned = {d: combine([(1 / 3, value)] * 3, average_backends=True) for d, value in tuned.items()}
        analysis.pair(values, tuned, "transfer", "finetuned_minus_qualisr", backend)
        rf_gap = {"mos_gap": combine([(1, values[SET]), (-1, values[EXTERNAL])])}
        tuned_gap = {"mos_gap": combine([(1, tuned[SET]), (-1, tuned[EXTERNAL])])}
        analysis.record(analysis.cells, rf_gap, table="transfer", name="qualisr_gap", model=backend, kind="score")
        analysis.pair(rf_gap, tuned_gap, "transfer", "finetuned_minus_qualisr_gap", backend,
                      "mos_gap" if backend == "backend_mean" and not args.pilot else None)
        bt.extend(dict(name="finetuned_minus_qualisr", model=backend, **row)
                  for row in common_bt(musiq_frames, tuned_frames, backend))
    analysis.write(args.output, expected_primary=0 if args.pilot else 12)
    correlations = pd.DataFrame(analysis.cells)
    for table, filename in (("13", "table_13.csv"), ("transfer", "table_musiq.csv")):
        selected = correlations.loc[correlations.table.eq(table)]
        wide = selected.pivot(index=["name", "model", "population"], columns="metric",
                              values=["fold_mean", "fold_sd", "conditional_point", "ci_low", "ci_high"])
        wide.columns = [f"{metric}_{statistic}" for statistic, metric in wide.columns]
        wide.to_csv(args.output / filename)
    pd.concat(summaries, ignore_index=True).to_csv(args.output / "transfer_summary.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(args.output / "transfer_paired_folds.csv", index=False)
    pd.concat(ordinary, ignore_index=True).to_csv(args.output / "ordinary_common.csv", index=False)
    pd.concat(influence, ignore_index=True).to_csv(args.output / "source_influence.csv", index=False)
    pd.DataFrame(bt).to_csv(args.output / "bt_common_deltas.csv", index=False)
    if seeds:
        pd.concat(seeds, ignore_index=True).to_csv(args.output / "model_seed_summary.csv", index=False)
    # Check that every backend evaluates the same targets, groups, feature inputs,
    # and tuning partitions before averaging correlations.
    for pool in ("historical", "qalign"):
        roots = [args.root / f"{stage}-{backend}-{pool}" for backend in BACKENDS]
        for file in ("samples.csv", "partitions.csv.gz"):
            baseline = read(roots[0] / file)
            for root in roots[1:]:
                pd.testing.assert_frame_equal(baseline, read(root / file), check_exact=True)
        columns = read(roots[0] / "fits.csv")[["fit_id", "columns"]]
        for root in roots[1:]:
            pd.testing.assert_frame_equal(columns, read(root / "fits.csv")[["fit_id", "columns"]], check_exact=True)
    write_json(args.output / "verification.json", {
        "status": "passed", "pilot": args.pilot, "backends": BACKENDS, "inputs": verified,
        "bootstrap_reps": analysis.n_reps, "primary_tests": len(analysis.primary),
        "scope": "identical target rows and scene partitions across backends",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "previous-primary"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())
