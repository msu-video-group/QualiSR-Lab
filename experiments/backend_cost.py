"""Prepare out-of-fold quality scores, verify CPU timings and summarize backend costs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_statistics import BACKENDS, FOLD, SET, read
from experiments.data import require, sha256_file
from experiments.profile_compact import FEATURES, pipeline_specs
from experiments.run_grouped import write_json
from experiments.summarize_cost import EXTENDED, LABELS, analyze, verify

BACKEND_LABELS = {"randomforest": "RF", "xgb": "XGBoost", "catboost": "CatBoost"}


def export_nr_scores(grouped, output):
    manifest = json.loads((grouped / "data_provenance.json").read_text())
    (record,) = [r for r in manifest["files"] if r["dataset"] == SET and Path(r["path"]).name == "nr.csv"]
    path = Path(record["path"])
    require(sha256_file(path) == record["sha256"], "The Set120 NR cache changed after fitting")
    direct = read(path, usecols=["sample_id", "qalign"]).set_index("sample_id", verify_integrity=True)
    folds = []
    for path in sorted(grouped.glob("seed_*/fold_*/nr/randomforest/fixed/predictions.csv")):
        seed, fold = path.relative_to(grouped).parts[:2]
        frame = read(path)
        fold_scores = frame.loc[frame.dataset.eq(SET), ["sample_id", "source_group", "score", "prediction"]]
        folds.append(fold_scores.rename(columns={"prediction": "nr5"}).assign(
            split_seed=int(seed.removeprefix("seed_")), fold=int(fold.removeprefix("fold_"))))
    require(len(folds) == 15, "Expected all 15 fixed five-NR Random Forest folds")
    values = pd.concat(folds, ignore_index=True)
    require(set(values.sample_id) == set(direct.index) and not values.duplicated(["split_seed", "sample_id"]).any()
            and values.groupby("sample_id").size().eq(3).all(), "Incomplete Set120 out-of-fold coverage")
    values.join(direct, on="sample_id", validate="many_to_one").to_csv(
        output, columns=FOLD + ["sample_id", "source_group", "score", "nr5", "qalign"], index=False, mode="x")


def verify_extended(root):
    protocol = json.loads((root / "protocol.json").read_text())
    measurements = read(root / "measurements.csv")
    scores = read(root / "scores.csv")
    inputs = read(root / "inputs.csv")
    keys = ["sample_id", "iteration"]
    pd.testing.assert_frame_equal(measurements[keys], scores[keys])
    require(protocol["status"] == "complete" and protocol["device"] == "cpu", "Incomplete CPU profile")
    require(inputs.sample_id.is_unique and set(inputs.sample_id) == set(scores.sample_id),
            "Extended timing inputs differ")
    require(len(scores) == len(inputs) * protocol["repetitions_per_image"], "Missing measurements")
    require(np.isfinite(scores[list(protocol["stages"])]).all().all(), "Nonfinite live score")
    for name, stages in protocol["stages"].items():
        require(np.isfinite(measurements[stages]).all().all() and measurements[stages].gt(0).all().all(),
                "Invalid CPU stage timing")
        np.testing.assert_allclose(measurements[stages].sum(axis=1), measurements[f"total_{name}"],
                                   atol=1e-8, rtol=0)
    write_json(root / "verification.json", {
        "status": "passed", "pipelines": len(protocol["stages"]), "measured_rows": len(scores),
        "checks": ["one live input per shared extractor", "finite live scores", "per-pipeline stage sums"],
    })
    return protocol, measurements, inputs


def summarize(args):
    features, stages, _ = pipeline_specs(BACKENDS)
    compact_labels = {"musiq": "MUSIQ"}
    for name in features:
        base, backend = name.split("__")
        compact_labels[name] = LABELS[base].replace("RF", BACKEND_LABELS[backend])
    measured, oof = verify(args.compact, args.core, args.pilot, compact_labels)
    write_json(args.compact / "verification.json", {
        "status": "passed", "pipelines": len(stages), "latency_rows": len(measured), "oof_rows": len(oof),
    })
    if args.pilot:
        return
    args.output.mkdir()
    protocols, batches, inputs = zip(*(verify_extended(root) for root in (args.extended_pilot, args.extended)),
                                     strict=True)
    fields = ("threads", "batch_size", "qalign_dtype", "features", "stages", "generators",
              "repetitions_per_image")
    require(all(protocols[0][f] == protocols[1][f] for f in fields), "Extended protocol changed")
    require(protocols[1]["tabular_bundle_sha256"] == sha256_file(args.extended_pilot / "models.joblib"),
            "Extended series used a different model bundle from the pilot")
    extra = pd.concat(batches, ignore_index=True)
    timing_inputs = pd.concat(inputs, ignore_index=True)
    expected = read(args.compact / "inputs.csv").iloc[[0, 4, 8]].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, timing_inputs)
    common = set(expected.sample_id)
    ids = ["split_seed", "fold", "sample_id"]
    meta = [*ids, "source_group", "score"]
    data = []
    for backend in BACKENDS:
        paths = sorted(args.prior.glob(f"seed_*/fold_*/nr/{backend}/fixed/predictions.csv"))
        require(len(paths) == 15, "Missing five-NR grouped models")
        for path in paths:
            seed, fold = path.relative_to(args.prior).parts[:2]
            frame = read(path)
            frame = frame[frame.dataset.eq("QualiSR-Set120")]
            frame = frame.assign(split_seed=int(seed[5:]), fold=int(fold[5:]))
            frame = frame[meta + ["prediction"]].rename(columns={"prediction": f"nr5__{backend}"})
            data.append((f"nr5__{backend}", frame))
    for name in sorted({n for n, _ in data}):
        frame = pd.concat([f for n, f in data if n == name], ignore_index=True)
        pd.testing.assert_frame_equal(oof[meta].sort_values(ids, ignore_index=True),
                                      frame[meta].sort_values(ids, ignore_index=True))
        oof = oof.merge(frame[ids + [name]], on=ids, validate="one_to_one")
    rr = []
    columns = meta + ["dataset", "experiment", "model", "track", "prediction"]
    for chunk in pd.read_csv(args.core / "predictions.csv.gz", usecols=columns, chunksize=150000,
                             float_precision="round_trip"):
        rr.append(chunk.loc[chunk.dataset.eq("QualiSR-Set120") & chunk.track.eq("fixed")
                            & chunk.model.isin(BACKENDS) & chunk.experiment.isin(["all_rr", "all_rr_no_wasd"])])
    for (name, backend), frame in pd.concat(rr).groupby(["experiment", "model"]):
        pd.testing.assert_frame_equal(oof[meta].sort_values(ids, ignore_index=True),
                                      frame[meta].sort_values(ids, ignore_index=True))
        key = f"{name}__{backend}"
        oof = oof.merge(frame[ids + ["prediction"]].rename(columns={"prediction": key}),
                        on=ids, validate="one_to_one")
    direct = read(args.prior_nr)
    pd.testing.assert_frame_equal(oof[meta].sort_values(ids, ignore_index=True),
                                  direct[meta].sort_values(ids, ignore_index=True))
    oof = oof.merge(direct[ids + ["qalign"]], on=ids, validate="one_to_one")
    require(np.isfinite(oof.drop(columns=["source_group", "sample_id", "method"])).all().all(),
            "Nonfinite OOF values")
    points = []
    for backend in BACKENDS:
        labels = {"musiq": "MUSIQ", "qalign": "Q-Align"}
        for base, label in (LABELS | EXTENDED).items():
            if base in ("musiq", "qalign"):
                continue
            labels[f"{base}__{backend}"] = label.replace("RF", BACKEND_LABELS[backend])
        totals = pd.concat([
            measured.loc[measured.sample_id.isin(common),
                         [f"total_{name}__{backend}" for name in FEATURES]],
            extra[[f"total_{key}" for key in labels if key in protocols[0]["stages"]]],
        ], ignore_index=True)
        pairs = [(f"{a}__{backend}", f"{b}__{backend}") for a, b in [
            ("nr4", "nr4_diff"), ("nr4", "nr4_wasd"), ("nr4_diff", "nr4_diff_wasd"),
            ("nr4", "nr5"), ("nr5", "all_rr_no_wasd"), ("all_rr_no_wasd", "all_rr"),
        ]]
        out = args.output / backend
        out.mkdir()
        analyze(totals.assign(sample_id="", iteration=0), oof, out, labels, pairs)
        table = read(out / "front.csv")
        points.append(table.assign(backend=backend))
    points = pd.concat(points, ignore_index=True).drop_duplicates("pipeline")
    points.loc[points.pipeline.isin(["musiq", "qalign"]), "backend"] = "direct"
    points["pareto_all_backends"] = [
        not ((points.p50_ms.le(row.p50_ms) & points.srcc.ge(row.srcc))
             & (points.p50_ms.lt(row.p50_ms) | points.srcc.gt(row.srcc))).any()
        for row in points.itertuples()
    ]
    points.to_csv(args.output / "front_all_backends.csv", index=False)
    extra.to_csv(args.output / "extended_measurements.csv", index=False)
    oof.to_csv(args.output / "oof.csv", index=False)
    write_json(args.output / "verification.json", {
        "status": "passed", "backends": BACKENDS, "pipelines": len(points), "oof_rows": len(oof),
        "timing_scenes": len(common), "bootstrap_reps": 10000,
        "quality_inputs": "archived features, shared folds and averaged OOF predictions",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Export five-NR out-of-fold scores")
    prepare.add_argument("--grouped", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Check extended stage timings")
    verify.add_argument("run", type=Path, help="Extended profiling run directory")
    summary = commands.add_parser("summarize", help="Combine quality and latency for each backend")
    for name in ("compact", "core", "output"):
        summary.add_argument(f"--{name}", type=Path, required=True)
    for name in ("extended", "extended-pilot", "prior", "prior-nr"):
        summary.add_argument(f"--{name}", type=Path)
    summary.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        export_nr_scores(args.grouped, args.output)
    elif args.command == "verify":
        verify_extended(args.run)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
