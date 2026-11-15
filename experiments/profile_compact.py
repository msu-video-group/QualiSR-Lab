"""Time compact CPU pipelines and evaluate the same feature sets on saved grouped folds."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS
from experiments.data import NR_COLUMNS, STAT_COLUMNS, _read_block, sha256_file
from experiments.fold_features import FoldTransformer
from experiments.profile_cost import init_detector, measure, prepare_model_input, read_rgb
from experiments.run_grouped import make_model, write_json
from experiments.transfer import DISRQAD, SET120
from qualisr.datasets import load_dataset
from qualisr.statistics import (
    DEFAULT_AREA_THRESHOLDS,
    DEFAULT_PERCENTILES,
    compute_statistics_row,
    load_heatmap,
)

NR = [name for name in NR_COLUMNS if name != "qalign"]
DIFF = [f"resnet_diff_{i:03d}" for i in range(5)]
FEATURES = {
    "nr4": NR,
    "nr4_diff": NR + DIFF,
    "nr4_wasd": NR + STAT_COLUMNS,
    "nr4_diff_wasd": NR + DIFF + STAT_COLUMNS,
}
SR = ["sr_decode", "nr_tensor"]
BICUBIC = ["lr_decode", "bicubic"]
RESNET = ["sr_resnet", "ref_resnet"]
WASD = ["wasd", "statistics"]
STAGES = {
    "musiq": [*SR, "musiq"],
    "nr4": [*SR, *NR],
    "nr4_diff": [*SR, *NR, *BICUBIC, *RESNET],
    "nr4_wasd": [*SR, *NR, *BICUBIC, *WASD],
    "nr4_diff_wasd": [*SR, *NR, *BICUBIC, *RESNET, *WASD],
}
for name in FEATURES:
    STAGES[name].extend([f"{name}_transform", f"{name}_predict"])


def load_cached(run, samples):
    provenance = json.loads((run / "data_provenance.json").read_text())["files"]
    combined = []
    for dataset in (SET120, DISRQAD):
        ids = samples.index[samples.dataset.eq(dataset)]
        sources = {Path(item["path"]).name: item for item in provenance if item["dataset"] == dataset}
        tables = []
        for filename, columns in {
            "nr.csv": NR, "stats.csv": STAT_COLUMNS, "resnet.csv": None, "ref_resnet.csv": None,
        }.items():
            source = sources[filename]
            path = Path(source["path"])
            prefix = path.stem + "_" if columns is None else None
            table, manifest = _read_block(path, ids, columns=columns, prefix=prefix)
            if manifest["sha256"] != source["sha256"]:
                raise ValueError(f"Cached input changed: {path}")
            tables.append(table)
        blocks = {"resnet": (tables[2].columns.tolist(), tables[3].columns.tolist())}
        combined.append(pd.concat(tables, axis=1))
    return pd.concat(combined).rename_axis("sample_id").reset_index().set_index("sample_id", drop=False), blocks


def pipeline_specs(backends):
    if tuple(backends) == ("randomforest",):
        return FEATURES, STAGES, dict.fromkeys(FEATURES, "randomforest")
    features, stages, models = {}, {"musiq": STAGES["musiq"]}, {}
    for name, columns in FEATURES.items():
        for backend in backends:
            key = f"{name}__{backend}"
            features[key] = columns
            stages[key] = [*STAGES[name][:-2], f"{key}_transform", f"{key}_predict"]
            models[key] = backend
    return features, stages, models


def fit_fold(raw, samples, blocks, fold_rows, backends=("randomforest",)):
    fit_ids = fold_rows.loc[fold_rows.partition.eq("train"), "sample_id"]
    heldout = fold_rows.loc[fold_rows.partition.eq("heldout") & fold_rows.dataset.eq(SET120), "sample_id"]
    if (fit_ids.duplicated().any() or heldout.duplicated().any()
            or set(samples.loc[fit_ids, "source_group"]) & set(samples.loc[heldout, "source_group"])):
        raise ValueError("Duplicate IDs or overlap between fitting and held-out scenes")
    frozen = FoldTransformer(NR + STAT_COLUMNS, blocks, {"resnet": 50}, seed=42)
    train = frozen.fit_transform(raw.loc[fit_ids])
    test = frozen.transform(raw.loc[heldout])
    features, stages, names = pipeline_specs(backends)
    models = {name: make_model(names[name], {}, 42, 2).fit(train[cols], samples.loc[fit_ids, "score"])
              for name, cols in features.items()}
    preds = samples.loc[heldout, ["source_group", "score", "method"]].copy()
    preds["musiq"] = raw.loc[heldout, "musiq"]
    for name, model in models.items():
        preds[name] = model.predict(test[features[name]])
    if not np.isfinite(preds[["score", *stages]]).all().all():
        raise ValueError("Nonfinite held-out prediction or target")
    return frozen, models, preds


def live(args, config, raw, frozen, models, heldout, features=FEATURES, stages=STAGES):
    import pyiqa
    import torch
    import torchvision

    from qualisr.features import extract_pretrained_features, image_to_tensor, init_nr_models, init_resnet

    device = torch.device("cpu")
    generate, crop, _ = init_detector(args.wasd_repo, args.checkpoint)
    nr = init_nr_models(device, NR)
    encoder, transform = init_resnet(device)
    entry = next(item for item in config["datasets"] if item["name"] == SET120)
    by_id = {item["sample_id"]: item for item in load_dataset(entry, args.config.resolve().parent)}
    ordinary = heldout.loc[~heldout.index.isin(args.controls)]
    groups = []
    for source, part in ordinary.groupby("source_group"):
        if len(part) == 3:
            area = max(np.prod(read_rgb(by_id[sid]["sr_path"]).size) for sid in part.index)
            groups.append((area, source, part.sort_values("method").index.tolist()))
    groups.sort()
    if len(groups) < 3:
        raise ValueError("Need three complete ordinary scenes in the profiling fold")
    chosen = groups[:1] if args.pilot else [groups[0], groups[len(groups) // 2], groups[-1]]
    warmup, repetitions = (1, 1) if args.pilot else (2, 5)
    measurements, scores, images, parity = [], [], [], []
    sr_cols, ref_cols = frozen.embedding_blocks["resnet"]
    with torch.inference_mode():
        for _, source, ids in chosen:
            for sid in ids[:1] if args.pilot else ids:
                sample = by_id[sid]
                print(f"Profiling {sid}", flush=True)
                sr_size = read_rgb(sample["sr_path"]).size
                lr_size = read_rgb(sample["lr_path"]).size
                images.append({"sample_id": sid, "source_group": source, "method": heldout.at[sid, "method"],
                               "sr_width": sr_size[0], "sr_height": sr_size[1],
                               "lr_width": lr_size[0], "lr_height": lr_size[1]})
                for repeat in range(-warmup, repetitions):
                    sr, t_sr = measure("sr_decode", read_rgb, sample["sr_path"])
                    x, t_tensor = measure("nr_tensor", image_to_tensor, sr, device)
                    times = dict([t_sr, t_tensor])
                    row = {}
                    for name, model in nr.items():
                        value, elapsed = measure(name, model, x)
                        row[name] = float(value.item())
                        times.update([elapsed])
                    lr, t_lr = measure("lr_decode", read_rgb, sample["lr_path"])
                    ref, t_ref = measure("bicubic", lr.resize, sr.size, Image.Resampling.BICUBIC)
                    se, t_se = measure("sr_resnet", extract_pretrained_features, sr, encoder, transform, device)
                    re, t_re = measure("ref_resnet", extract_pretrained_features, ref, encoder, transform, device)
                    mask, t_mask = measure("wasd", generate, sr, ref)
                    stats, t_stats = measure("statistics", compute_statistics_row, sid, mask,
                                             DEFAULT_PERCENTILES, DEFAULT_AREA_THRESHOLDS)
                    times.update([t_lr, t_ref, t_se, t_re, t_mask, t_stats])
                    row.update(zip(sr_cols + ref_cols, np.concatenate([se, re]), strict=True))
                    row.update({k: stats[k] for k in STAT_COLUMNS})
                    inputs = pd.Series(row)
                    prediction = {"musiq": row["musiq"]}
                    for name, columns in features.items():
                        x, elapsed = measure(f"{name}_transform", prepare_model_input, inputs, frozen, columns)
                        value, model_time = measure(f"{name}_predict", models[name].predict, x)
                        prediction[name] = float(value[0])
                        times.update([elapsed, model_time])
                    if not np.isfinite(list(prediction.values())).all():
                        raise ValueError(f"Nonfinite live score for {sid}")
                    if repeat >= 0:
                        measurements.append({"sample_id": sid, "iteration": repeat, **times,
                                             **{f"total_{k}": sum(times[s] for s in steps)
                                                for k, steps in stages.items()}})
                        scores.append({"sample_id": sid, "iteration": repeat, **prediction})
                    if repeat == 0:
                        archived = load_heatmap(Path(sample["heatmap_path"]))
                        if archived.shape != mask.shape:
                            raise ValueError(f"Archived mask has a different shape for {sid}")
                        archived_ref = read_rgb(sample["ref_paths"]["bicubic"])
                        parity.append({"sample_id": sid, "bicubic_exact": np.array_equal(ref, archived_ref),
                                       "mask_mae": np.abs(archived.astype(float) - mask).mean(),
                                       **{f"delta_{k}": row[k] - raw.at[sid, k] for k in NR + STAT_COLUMNS}})
    for name, rows in (("measurements", measurements), ("scores", scores), ("inputs", images), ("parity", parity)):
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
    write_json(args.output / "protocol.json", {
        "status": "complete",
        "hardware": next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")),
        "hostname": platform.node(), "device": "cpu", "threads": 2,
        "batch_size": 1, "warmup_per_image": warmup, "repetitions_per_image": repetitions,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "pyiqa": pyiqa.__version__,
        "precision": "uint8 RGB decode, FP32 neural inference, float64 PCA and scaling",
        "fit_split_seed": 42, "fit_fold": 1, "model_seed": 42, "features": features, "stages": stages,
        "latency_definition": "Sum of synchronous in-process stage times, one execution per dependency and image",
        "detector_checkpoint_sha256": sha256_file(args.checkpoint),
        "detector_source_sha256": sha256_file(args.wasd_repo / "calculate_subjective_masks.py"),
        "detector_config_sha256": sha256_file(args.wasd_repo / "configs/heatmaps-iqa-datasets.yaml"),
        "detector_patch_size": crop // 2,
        "weights": {p.name: sha256_file(p) for p in sorted(Path(torch.hub.get_dir()).rglob("*"))
                    if p.suffix.lower() in {".pt", ".pth"}},
    })


def run(args):
    args.output.mkdir()
    config = json.loads(args.config.read_text())
    args.controls = json.loads((args.run / "run.json").read_text())["controls"]
    samples = pd.read_csv(args.run / "samples.csv", float_precision="round_trip").set_index("sample_id")
    folds = pd.read_csv(args.run / "fold_assignments.csv", usecols=[
        "sample_id", "dataset", "split_seed", "fold", "partition"
    ])
    raw, blocks = load_cached(args.run, samples)
    features, stages, _ = pipeline_specs(args.backends)
    predictions = []
    for (seed, fold), rows in folds.groupby(["split_seed", "fold"]):
        if args.pilot and (seed, fold) != (42, 1):
            continue
        print(f"Fitting grouped fold {seed}/{fold}", flush=True)
        frozen, models, preds = fit_fold(raw, samples, blocks, rows, args.backends)
        if (seed, fold) == (42, 1):
            live(args, config, raw, frozen, models, preds, features, stages)
        predictions.append(preds.assign(split_seed=seed, fold=fold))
    pd.concat(predictions).to_csv(args.output / "oof.csv")
    write_json(args.output / "status.json", {"status": "complete", "folds": len(predictions)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "run", "wasd-repo", "checkpoint", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["randomforest"])
    with threadpool_limits(limits=2):
        run(parser.parse_args())
