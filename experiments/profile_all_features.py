"""Measure paper pipelines on the CPU using pinned live extractors."""

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS
from experiments.data import (
    FR_COLUMNS,
    FR_METRICS,
    NR_COLUMNS,
    SIGLIP_COLUMNS,
    STAT_COLUMNS,
    _read_block,
    sha256_file,
)
from experiments.fold_features import FoldTransformer
from experiments.profile_cost import init_detector, measure, read_rgb
from experiments.run_grouped import make_model, write_json
from experiments.transfer import DISRQAD, SET120
from qualisr.datasets import load_dataset
from qualisr.features import (
    align_fr_images,
    compute_siglip_scores,
    extract_pretrained_features,
    image_to_tensor,
    init_fr_models,
    init_nr_models,
    init_resnet,
    init_siglip,
    init_vgg,
)
from qualisr.statistics import (
    DEFAULT_AREA_THRESHOLDS,
    DEFAULT_PERCENTILES,
    compute_statistics_row,
    load_heatmap,
)

SCALARS = NR_COLUMNS + FR_COLUMNS + STAT_COLUMNS + SIGLIP_COLUMNS
DEVICE = torch.device("cpu")
SEED = 42
FOLD = 1
MODELS = ("nr5", "all_rr_no_wasd", "all_rr")
STAGES = {
    "musiq": ["sr_decode", "nr_tensor", "musiq"],
    "qalign": ["sr_decode", "nr_tensor", "qalign"],
    "nr5": ["sr_decode", "nr_tensor", *NR_COLUMNS, "nr5_transform", "nr5_predict"],
}
RR = [
    "sr_decode", "nr_tensor", *NR_COLUMNS, "lr_decode", "bicubic", "rlfn", "span",
    "fr_bicubic", "fr_rlfn", "fr_span", "sr_vgg", "ref_vgg", "sr_resnet", "ref_resnet", "siglip",
]
STAGES["all_rr_no_wasd"] = [*RR, "all_rr_no_wasd_transform", "all_rr_no_wasd_predict"]
STAGES["all_rr"] = [*RR, "wasd", "statistics", "all_rr_transform", "all_rr_predict"]


def fit_models(run, samples, train, heldout, config, prior_nr, backends=("randomforest",), prior=None):
    sources = json.loads((run / "data_provenance.json").read_text())["files"]
    blocks = {}
    tables = []
    for dataset in (SET120, DISRQAD):
        ids = samples.index[samples.dataset.eq(dataset)]
        selected_ids = train[train.isin(ids)].append(heldout[heldout.isin(ids)])
        records = {Path(item["path"]).name: item for item in sources if item["dataset"] == dataset}
        pieces = []
        for file, columns in {
            "nr": NR_COLUMNS, "fr": FR_COLUMNS, "stats": STAT_COLUMNS, "siglip": SIGLIP_COLUMNS,
            "vgg": None, "ref_vgg": None, "resnet": None, "ref_resnet": None,
        }.items():
            record = records[f"{file}.csv"]
            values, found = _read_block(Path(record["path"]), ids, columns=columns,
                                        prefix=file + "_" if columns is None else None)
            if found["sha256"] != record["sha256"]:
                raise ValueError(f"Changed input for {dataset}/{file}")
            if columns is None:
                blocks[file] = list(values.columns)
            pieces.append(values.loc[selected_ids])
        tables.append(pd.concat(pieces, axis=1))
    all_rows = pd.concat(tables)
    raw = all_rows.loc[train].reset_index()
    heldout_raw = all_rows.loc[heldout].reset_index()
    families = {name: (blocks[name], blocks[f"ref_{name}"]) for name in ("vgg", "resnet")}
    dimensions = config["evaluation"]["max_pca_dimensions"]
    transforms = {
        "nr5": FoldTransformer(NR_COLUMNS, {}, {}, seed=SEED),
        "all_rr_no_wasd": FoldTransformer(SCALARS, families, dimensions, seed=SEED),
    }
    transforms["all_rr"] = transforms["all_rr_no_wasd"]
    matrix = {name: transformer.fit_transform(raw) for name, transformer in transforms.items() if name != "all_rr"}
    matrix["all_rr"] = matrix["all_rr_no_wasd"]
    fits = pd.read_csv(run / "fits.csv")
    fits = fits.loc[fits.split_seed.eq(SEED) & fits.fold.eq(FOLD) & fits.model.isin(backends)
                    & fits.track.eq("fixed")].set_index(["experiment", "model"])
    columns = ["split_seed", "fold", "experiment", "model", "track", "sample_id", "prediction"]
    archived = pd.read_csv(run / "predictions.csv.gz", usecols=columns, nrows=250_000,
                           float_precision="round_trip")
    archived = archived.loc[archived.split_seed.eq(SEED) & archived.fold.eq(FOLD)
                            & archived.model.isin(backends) & archived.track.eq("fixed")]
    nr_ref = pd.read_csv(prior_nr, float_precision="round_trip")
    nr_ref = nr_ref.loc[nr_ref.split_seed.eq(SEED) & nr_ref.fold.eq(FOLD)].set_index("sample_id")
    models, frozen = {}, {}
    for name in MODELS:
        for backend in backends:
            key = name if tuple(backends) == ("randomforest",) else f"{name}__{backend}"
            columns = NR_COLUMNS if name == "nr5" else fits.at[(name, backend), "columns"].split("|")
            model = make_model(backend, {}, SEED, 2).fit(matrix[name][columns], samples.loc[train, "score"])
            if name == "nr5":
                if backend == "randomforest":
                    expected = nr_ref.loc[heldout, "nr5"]
                else:
                    expected = pd.read_csv(prior / "seed_42/fold_01/nr" / backend / "fixed/predictions.csv",
                                           float_precision="round_trip").set_index("sample_id").loc[heldout, "prediction"]
            else:
                expected = archived.loc[archived.experiment.eq(name) & archived.model.eq(backend)]
                expected = expected.set_index("sample_id").loc[heldout, "prediction"]
            actual = model.predict(transforms[name].transform(heldout_raw)[columns])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=key)
            models[key] = (model, columns)
            frozen[key] = transforms[name]
    return frozen, models, blocks


def init_generator(folder, name):
    source = folder / name / f"inference-{name}.py"
    weights = folder / name / f"{name.lower()}-tuned-4x.pth"
    spec = importlib.util.spec_from_file_location(name.lower(), source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.RLFN(upscale=4) if name == "RLFN" else module.SPAN(3, 3, upscale=4, feature_channels=48)
    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()

    def generate(lr, size):
        x = image_to_tensor(lr, DEVICE)
        y = model(x).clamp(0, 1)[0].permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
        image = Image.fromarray(y)
        return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)

    return generate, {path.name: sha256_file(path) for path in (source, weights)}


def fr_scores(name, reference, sr, tensor, sample, fr):
    a, b = align_fr_images(sr, reference, Path(sample["sr_path"]), Path(sample["lr_path"]), name, True)
    ar = tensor if a is sr else image_to_tensor(a, DEVICE)
    br = image_to_tensor(b, DEVICE)
    return {f"{metric}_{name}": float(net(ar, br).item()) for metric, net in fr.items()}


def prepare(frozen, name, sid, features, columns):
    row = {"sample_id": sid, **{key: features[key] for key in frozen.scalar_columns},
           **{key: features[key] for sr, ref in frozen.embedding_blocks.values() for key in sr + ref}}
    if name.split("__")[0] == "all_rr_no_wasd":
        row.update(dict.fromkeys(STAT_COLUMNS, np.nan))
    return frozen.transform(pd.DataFrame([row]))[columns]


def main(args):
    if any(backend != "randomforest" for backend in args.backends) and args.prior is None:
        raise ValueError("The three-backend profile requires the verified --prior run")
    args.output.mkdir()
    cfg = json.loads(args.config.read_text())
    samples = pd.read_csv(args.core / "samples.csv", float_precision="round_trip").set_index("sample_id")
    split = pd.read_csv(args.core / "fold_assignments.csv", usecols=["split_seed", "fold", "partition", "sample_id"])
    split = split.loc[split.split_seed.eq(SEED) & split.fold.eq(FOLD)]
    train = pd.Index(split.loc[split.partition.eq("train"), "sample_id"])
    heldout = pd.Index(split.loc[split.partition.eq("heldout") & split.sample_id.isin(
        samples.index[samples.dataset.eq(SET120)]), "sample_id"])
    inputs = pd.read_csv(args.inputs)
    if len(inputs) != 9:
        raise ValueError("Expected nine compact inputs across three scenes")
    inputs = inputs.iloc[[0] if args.pilot else [4, 8]]
    if set(samples.loc[train].source_group) & set(inputs.source_group):
        raise ValueError("Timing scenes overlap fitting scenes")
    print("Initializing offline neural extractors", flush=True)
    wasd, crop, _ = init_detector(args.wasd_repo, args.checkpoint)
    nr = init_nr_models(DEVICE, NR_COLUMNS)
    # Use FP32 on this CPU. Match Q-Align's input dtype and its five quality weights to the model.
    qalign = nr["qalign"].net
    qalign.model.float()
    qalign.model_dtype = torch.float32
    qalign.model.weight_tensor = torch.tensor([5., 4., 3., 2., 1.], dtype=torch.float32, device=DEVICE)
    fr = init_fr_models(DEVICE, FR_METRICS)
    encoders = {"vgg": init_vgg(DEVICE), "resnet": init_resnet(DEVICE)}
    siglip, processor = init_siglip(str(args.siglip), DEVICE)
    generators, gen_hashes = {}, {}
    for name in ("RLFN", "SPAN"):
        generators[name.lower()], hashes = init_generator(args.generators, name)
        gen_hashes.update(hashes)
    if args.model_bundle:
        frozen, models, blocks = joblib.load(args.model_bundle)
        if any(list(train) != transformer.fit_sample_ids_ for transformer in frozen.values()):
            raise ValueError("Serialized fit IDs differ from the verified training fold")
    else:
        print("Fitting tabular heads on saved fold 42/1", flush=True)
        frozen, models, blocks = fit_models(args.core, samples, train, heldout, cfg, args.prior_nr,
                                           args.backends, args.prior)
        print(f"All {len(models)} tabular heads reproduce saved held-out scores", flush=True)
        joblib.dump((frozen, models, blocks), args.output / "models.joblib", compress=3)
    stages = {name: STAGES[name] for name in ("musiq", "qalign")}
    for key in models:
        name = key.split("__")[0]
        stages[key] = [*STAGES[name][:-2], f"{key}_transform", f"{key}_predict"]
    entry = next(item for item in cfg["datasets"] if item["name"] == SET120)
    images = {row["sample_id"]: row for row in load_dataset(entry, args.config.resolve().parent)}
    warmup, repetitions = 1, 1
    measurements, scores, parity = [], [], []
    with torch.inference_mode():
        for index, sid in enumerate(inputs.sample_id):
            sample = images[sid]
            print(f"Profiling {sid}", flush=True)
            for repeat in range(-warmup if index == 0 else 0, repetitions):
                timings, features = {}, {}

                def timed(stage, function, *values, timings=timings):
                    result, (_, duration) = measure(stage, function, *values)
                    timings[stage] = duration
                    return result

                sr = timed("sr_decode", read_rgb, sample["sr_path"])
                tensor = timed("nr_tensor", image_to_tensor, sr, DEVICE)
                for name, net in nr.items():
                    features[name] = float(timed(name, net, tensor).item())
                print(f"  iteration {repeat}: Q-Align {timings['qalign'] / 1000:.1f} s", flush=True)
                lr = timed("lr_decode", read_rgb, sample["lr_path"])
                refs = {"bicubic": timed("bicubic", lr.resize, sr.size, Image.Resampling.BICUBIC)}
                for name, generator in generators.items():
                    refs[name] = timed(name, generator, lr, sr.size)

                for name, ref in refs.items():
                    stage = f"fr_{name}"
                    features.update(timed(stage, fr_scores, name, ref, sr, tensor, sample, fr))
                    print(f"  iteration {repeat}: {stage} {timings[stage] / 1000:.1f} s", flush=True)
                for family, (net, transform) in encoders.items():
                    for block, label, image in ((family, "sr", sr), (f"ref_{family}", "ref", refs["bicubic"])):
                        z = timed(f"{label}_{family}", extract_pretrained_features, image, net, transform, DEVICE)
                        features.update(zip(blocks[block], z, strict=True))
                features.update(timed("siglip", compute_siglip_scores, lr, sr, siglip, processor, DEVICE, .5))
                mask = timed("wasd", wasd, sr, refs["bicubic"])
                stats = timed("statistics", compute_statistics_row, sid, mask, DEFAULT_PERCENTILES, DEFAULT_AREA_THRESHOLDS)
                features.update({name: stats[name] for name in STAT_COLUMNS})
                prediction = {name: features[name] for name in ("musiq", "qalign")}
                for name, (model, columns) in models.items():
                    prepared = timed(f"{name}_transform", prepare, frozen[name], name, sid, features, columns)
                    prediction[name] = float(timed(f"{name}_predict", model.predict, prepared)[0])
                if not np.isfinite(list(prediction.values())).all() or not np.isfinite(list(features.values())).all():
                    raise ValueError(f"Nonfinite live feature or score on {sid}")
                if repeat >= 0:
                    timings.update({f"total_{k}": sum(timings[s] for s in steps) for k, steps in stages.items()})
                    measurements.append({"sample_id": sid, "iteration": repeat, **timings})
                    scores.append({"sample_id": sid, "iteration": repeat, **prediction})
                    print(f"  recorded {len(models)} tabular heads", flush=True)
                if repeat == 0:
                    archive = load_heatmap(Path(sample["heatmap_path"]))
                    parity.append({"sample_id": sid, "wasd_mask_mae": np.abs(mask.astype(float) - archive).mean(),
                                   **{f"{name}_pixel_mae": np.abs(np.asarray(ref).astype(float) -
                                         np.asarray(read_rgb(sample["ref_paths"][name]))).mean()
                                      for name, ref in refs.items()}})
    for name, rows in (("measurements", measurements), ("scores", scores), ("parity", parity)):
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
    inputs.to_csv(args.output / "inputs.csv", index=False)
    write_json(args.output / "protocol.json", {
        "status": "complete", "device": "cpu", "threads": 2, "batch_size": 1,
        "warmup_on_first_image": warmup, "repetitions_per_image": repetitions, "stages": stages,
        "torch": torch.__version__, "qalign_dtype": str(qalign.model_dtype),
        "features": {name: columns for name, (_, columns) in models.items()},
        "split_seed": SEED, "fold": FOLD, "detector_patch_size": crop // 2,
        "detector_checkpoint_sha256": sha256_file(args.checkpoint), "generators": gen_hashes,
        "tabular_bundle_sha256": sha256_file(args.model_bundle) if args.model_bundle else None,
        "latency_definition": "Sum of synchronous in-process stage times, one execution per dependency and image",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "core", "inputs", "generators", "siglip", "wasd-repo", "checkpoint", "prior-nr", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["randomforest"])
    parser.add_argument("--prior", type=Path)
    with threadpool_limits(limits=2):
        main(parser.parse_args())
