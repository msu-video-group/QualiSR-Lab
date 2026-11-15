"""Measure CPU pipeline stages and benchmark the WASD detector against the cached maps."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from experiments.data import NR_COLUMNS, _read_block, sha256_file
from experiments.presets import COMPACT_STAT_COLUMNS
from experiments.run_grouped import write_json
from experiments.transfer import SET120
from qualisr.datasets import load_dataset
from qualisr.profiling import estimate_regressor_predict_flops
from qualisr.statistics import (
    DEFAULT_AREA_THRESHOLDS,
    DEFAULT_PERCENTILES,
    compute_statistics_row,
    load_heatmap,
)

WARMUP = 3
REPETITIONS = 20
WASD_REPETITIONS = 10
PIPELINES = {
    "direct_musiq": ["sr_decode"],
    "nr": ["sr_decode", "nr_preprocess", "nr_predict"],
    "nr_diff": ["sr_decode", "lr_decode", "bicubic", "nr_diff_preprocess", "nr_diff_predict"],
    "nr_stats": ["sr_decode", "lr_decode", "bicubic", "archived_heatmap_decode",
                 "heatmap_statistics", "nr_stats_preprocess", "nr_stats_predict"],
}
OMITTED = {
    "direct_musiq": "MUSIQ extraction",
    "nr": "five NR metric extractions",
    "nr_diff": "five NR metric extractions; VGG/ResNet extraction on SR and bicubic",
    "nr_stats": "five NR metric extractions; WASD generation on SR and bicubic",
}


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def prepare_model_input(row, transformer, columns):
    """Evaluate only the fitted scalar columns and PCA coordinates the model uses."""
    offsets = [transformer.feature_names_out_.index(col) for col in columns]
    projected = {col: row[col] for col in columns if col in transformer.scalar_columns}
    for family, (sr_columns, ref_columns) in transformer.embedding_blocks.items():
        names = [col for col in columns if col.startswith(f"{family}_diff_")]
        if not names:
            continue
        pca = transformer.pcas_[family]
        components = [int(col.rsplit("_", 1)[1]) for col in names]
        # A common basis makes the centering term cancel in SR minus reference.
        difference = row[sr_columns].to_numpy(dtype=float) - row[ref_columns].to_numpy(dtype=float)
        projected.update(zip(names, pca.components_[components] @ difference, strict=True))
    values = np.array([projected[col] for col in columns], dtype=float)
    values = np.where(np.isfinite(values), values, transformer.imputer_.statistics_[offsets])
    scaled = values * transformer.scaler_.scale_[offsets] + transformer.scaler_.min_[offsets]
    return pd.DataFrame([scaled], columns=columns)


def measure(stage, call, *args):
    start = perf_counter_ns()
    result = call(*args)
    return result, (stage, (perf_counter_ns() - start) / 1e6)


def run_profile(config_path, run, profile_root, output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text())
    entry = next(item for item in config["datasets"] if item["name"] == SET120)
    dataset = load_dataset(entry, config_path.resolve().parent)
    by_id = {item["sample_id"]: item for item in dataset}
    samples = pd.read_csv(run / "samples.csv").set_index("sample_id")
    partition = "robustness/seed_42/fold_01/released"
    roles = pd.read_csv(run / "partitions.csv.gz")
    eligible = samples.loc[roles.loc[(roles.partition_id == partition) & (roles.role == "evaluate"), "sample_id"]]
    eligible = eligible[eligible.dataset == SET120]
    sources = eligible.groupby("source_group").filter(
        lambda df: len(df) == 3 and not df.screened_control.any()
    )
    chosen = sources[sources.source_group.eq(sorted(sources.source_group.unique())[0])].sort_values("method")
    fitted = {name: joblib.load(run / "profile_models" / f"{name}.joblib")
              for name in ("nr", "nr_diff", "nr_stats")}
    basis = fitted["nr_diff"][0]
    feature_files = {"nr.csv": NR_COLUMNS}
    for sr_cols, ref_cols in basis.embedding_blocks.values():
        feature_files[f"{sr_cols[0].rsplit('_', 1)[0]}.csv"] = sr_cols
        feature_files[f"{ref_cols[0].rsplit('_', 1)[0]}.csv"] = ref_cols
    cached = pd.concat([
        _read_block(Path(dataset[0]["features_root"]) / name, pd.Index(by_id), columns=columns)[0]
        for name, columns in feature_files.items()
    ], axis=1)

    results, sample_info, tree_ops = [], [], []
    for sample_id in chosen.index:
        sample = by_id[sample_id]
        sr_path, lr_path, mask_path = map(Path, (sample["sr_path"], sample["lr_path"], sample["heatmap_path"]))
        sr, lr, mask = read_rgb(sr_path), read_rgb(lr_path), load_heatmap(mask_path)
        sample_info.append({"sample_id": sample_id, "method": sample["method"], "sr_pixels": sr.size,
                            "lr_pixels": lr.size, "heatmap_shape": mask.shape, "heatmap_dtype": str(mask.dtype)})
        values = cached.loc[sample_id]
        stats = compute_statistics_row(sample_id, mask, DEFAULT_PERCENTILES, DEFAULT_AREA_THRESHOLDS)
        with_stats = pd.concat([values, pd.Series({key: stats[key] for key in COMPACT_STAT_COLUMNS})])
        for name, (transformer, model, columns) in fitted.items():
            row = with_stats if name == "nr_stats" else values
            model_x = prepare_model_input(row, transformer, columns)
            # Check specialized preparation against the fitted transformation on active columns.
            dummy = pd.Series(0.0, index=transformer.scalar_columns).combine_first(row)
            dummy.loc[row.index] = row
            dummy["sample_id"] = sample_id
            reference = transformer.transform(dummy.to_frame().T)[columns]
            np.testing.assert_allclose(model_x, reference, atol=1e-10, rtol=1e-9)
            ops = estimate_regressor_predict_flops("randomforest", model, model_x)
            tree_ops.append({"sample_id": sample_id, "pipeline": name, **ops})

        for iteration in range(-WARMUP, REPETITIONS):
            _, sr_decode = measure("sr_decode", read_rgb, sr_path)
            lr_rgb, lr_decode = measure("lr_decode", read_rgb, lr_path)
            _, bicubic = measure("bicubic", lr_rgb.resize, sr.size, Image.Resampling.BICUBIC)
            heatmap, heatmap_decode = measure("archived_heatmap_decode", load_heatmap, mask_path)
            stats, stats_cost = measure(
                "heatmap_statistics", compute_statistics_row, sample_id, heatmap,
                DEFAULT_PERCENTILES, DEFAULT_AREA_THRESHOLDS
            )
            times = [sr_decode, lr_decode, bicubic, heatmap_decode, stats_cost]
            for name, (transformer, model, columns) in fitted.items():
                row = values
                if name == "nr_stats":
                    row = pd.concat([values, pd.Series({key: stats[key] for key in COMPACT_STAT_COLUMNS})])
                x, prep = measure(f"{name}_preprocess", prepare_model_input, row, transformer, columns)
                _, predict = measure(f"{name}_predict", model.predict, x)
                times.extend([prep, predict])
            if iteration >= 0:
                for stage, ms in times:
                    results.append({"sample_id": sample_id, "iteration": iteration, "stage": stage, "ms": ms})

    timings = pd.DataFrame(results)
    timings.to_csv(output / "measurements.csv", index=False)
    timings.groupby("stage").ms.agg(p50_ms="median", p95_ms=lambda x: x.quantile(.95)).to_csv(
        output / "stage_latency.csv"
    )
    pd.DataFrame(tree_ops).to_csv(output / "estimated_tree_ops.csv", index=False)
    historical = {}
    for feature in ("nr", "vgg", "resnet", "stats"):
        path = profile_root / f"{feature}_profile.csv"
        historical[feature] = pd.read_csv(path).set_index("feature")
    estimates = {
        "direct_musiq": historical["nr"].loc[["musiq"]],
        "nr": historical["nr"].loc[NR_COLUMNS],
        "nr_diff": pd.concat([historical["nr"].loc[NR_COLUMNS], historical["vgg"], historical["resnet"],
                              historical["vgg"].rename(index={"vgg": "ref_vgg_estimate"}),
                              historical["resnet"].rename(index={"resnet": "ref_resnet_estimate"})]),
        "nr_stats": pd.concat([historical["nr"].loc[NR_COLUMNS], historical["stats"].loc[COMPACT_STAT_COLUMNS]]),
    }
    cost_rows, components = [], []
    for name, stages in PIPELINES.items():
        partial = timings[timings.stage.isin(stages)].groupby(["sample_id", "iteration"]).ms.sum()
        historical_rows = estimates[name]
        components.append(historical_rows.assign(pipeline=name, source_feature=historical_rows.index))
        cost_rows.append({
            "pipeline": name, "cpu_partial_p50_ms": partial.median(), "cpu_partial_p95_ms": partial.quantile(.95),
            "cpu_measured_stages": "|".join(stages), "omitted_cpu_latency": OMITTED[name],
            "historical_partial_profile_flops": historical_rows.mean_flops.sum(),
            "historical_extractor_and_statistic_ms": 1000 * historical_rows.mean_runtime_sec.sum(),
            "estimated_tree_ops": (np.mean([row["estimated_predict_flops"] for row in tree_ops
                                            if row["pipeline"] == name]) if name in fitted else np.nan),
        })
    pd.DataFrame(cost_rows).to_csv(output / "cost_summary.csv", index=False)
    pd.concat(components, ignore_index=True).to_csv(output / "historical_cost_components.csv", index=False)
    write_json(output / "protocol.json", {
        "status": "complete", "hardware": next(line.split(":", 1)[1].strip()
                                                for line in Path("/proc/cpuinfo").read_text().splitlines()
                                                if line.startswith("model name")),
        "hostname": platform.node(), "python": platform.python_version(), "threads": 2,
        "sample_batch_size": 1, "warmup_per_image": WARMUP, "repetitions_per_image": REPETITIONS,
        "precision": "uint8 RGB images; native saved mask dtype; float64 PCA/scaling; sklearn RF internal dtype",
        "sample_selection": "First lexical seed-42 fold-1 held-out Set120 source with all three ordinary outputs",
        "input_samples": sample_info, "measured_scope": "Synchronous CPU warm-cache partial latency",
        "historical_profiles": {p.name: sha256_file(p) for p in sorted(profile_root.glob("*_profile.csv"))},
        "historical_qualifications": "Reference encoder costs proxy SR forwards; original hardware/warm-up not "
        "recorded here. Historical costs and new CPU timings must remain separate.",
        "omitted_costs": "No detector inference, median FLOPs, bicubic FLOPs, preprocessing FLOPs, or end-to-end "
        "neural pipeline timing. Archived heatmap decoding substitutes for accessing detector output.",
    })




def init_detector(repo, checkpoint):
    import torch

    sys.path.insert(0, str(repo))
    from calculate_subjective_masks import method_from_model
    from utils.create_model import create_model_and_optimizer, load_config

    device = torch.device("cpu")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    detector_config = repo / "configs/heatmaps-iqa-datasets.yaml"
    config = load_config(detector_config)
    # Every tensor must come from the supplied checkpoint. Skip ImageNet initialization.
    config.model.pretrained = False
    model = create_model_and_optimizer(config)["model"]
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    # --subj_masks saves the coarse output but executes the complete forward.
    crop = int(config.model.crop_size)
    method = method_from_model(model, crop, crop // 2, device)

    @torch.inference_mode()
    def generate(sr, reference):
        _, coarse = method(sr, reference)
        return coarse.numpy()

    return generate, crop, torch.__version__


def compact_stats(sample_id, mask):
    row = compute_statistics_row(sample_id, mask, DEFAULT_PERCENTILES, DEFAULT_AREA_THRESHOLDS)
    return pd.Series({key: row[key] for key in COMPACT_STAT_COLUMNS})


def run_benchmark(config_path, run, cost, wasd_repo, checkpoint, output):
    output.mkdir(parents=True, exist_ok=False)
    generate, crop, torch_version = init_detector(wasd_repo, checkpoint)
    config = json.loads(config_path.read_text())
    entry = next(item for item in config["datasets"] if item["name"] == SET120)
    by_id = {item["sample_id"]: item for item in load_dataset(entry, config_path.resolve().parent)}
    nr, _ = _read_block(Path(next(iter(by_id.values()))["features_root"]) / "nr.csv", pd.Index(by_id),
                        columns=NR_COLUMNS)
    transformer, model, columns = joblib.load(run / "profile_models/nr_stats.joblib")
    prior_protocol = json.loads((cost / "protocol.json").read_text())
    sample_ids = [item["sample_id"] for item in prior_protocol["input_samples"]]
    observations, mask_checks = [], []
    for sample_id in sample_ids:
        sample = by_id[sample_id]
        sr_path, lr_path = Path(sample["sr_path"]), Path(sample["lr_path"])
        cached_ref_path = Path(sample["ref_paths"]["bicubic"])
        sr = read_rgb(sr_path)
        bicubic = read_rgb(lr_path).resize(sr.size, Image.Resampling.BICUBIC)
        cached_ref = read_rgb(cached_ref_path)
        old = load_heatmap(Path(sample["heatmap_path"]))
        new_cached_ref = generate(str(sr_path), str(cached_ref_path))
        new_in_memory = generate(sr, cached_ref)
        np.testing.assert_array_equal(new_cached_ref, new_in_memory)
        new_generated_ref = generate(sr, bicubic)
        if old.shape != new_cached_ref.shape or old.shape != new_generated_ref.shape:
            raise ValueError(f"Saved and regenerated mask shapes differ for {sample_id}")
        old_stats = compact_stats(sample_id, old)
        for label, mask in (("cached_bicubic", new_cached_ref), ("generated_bicubic", new_generated_ref)):
            error = np.abs(old.astype(np.float64) - mask)
            delta = compact_stats(sample_id, mask) - old_stats
            mask_checks.append({"sample_id": sample_id, "reference": label, "mask_mae": error.mean(),
                                "mask_max_abs_difference": error.max(),
                                "bicubic_pixels_match": np.array_equal(np.asarray(bicubic), np.asarray(cached_ref)),
                                **{f"delta_{key}": delta[key] for key in COMPACT_STAT_COLUMNS}})
        print(f"Timing WASD on {sample_id} at {sr.size}", flush=True)
        nr_row = nr.loc[sample_id]
        for repeat in range(-WARMUP, WASD_REPETITIONS):
            source, sr_time = measure("sr_decode", read_rgb, sr_path)
            low_res, lr_time = measure("lr_decode", read_rgb, lr_path)
            ref, ref_time = measure("bicubic", low_res.resize, source.size, Image.Resampling.BICUBIC)
            mask, wasd_time = measure("wasd_inference", generate, source, ref)
            stats, stats_time = measure("heatmap_statistics", compact_stats, sample_id, mask)
            prepared, prep_time = measure("nr_stats_preprocess", prepare_model_input,
                                          pd.concat([nr_row, stats]), transformer, columns)
            _, pred_time = measure("nr_stats_predict", model.predict, prepared)
            if repeat >= 0:
                times = dict([sr_time, lr_time, ref_time, wasd_time, stats_time, prep_time, pred_time])
                observations.append({"sample_id": sample_id, "iteration": repeat, **times,
                                     "sum_ms_without_nr_extraction": sum(times.values())})
    observations = pd.DataFrame(observations)
    observations.to_csv(output / "measurements.csv", index=False)
    stage_cols = list(observations.columns[2:])
    pd.DataFrame({"p50_ms": observations[stage_cols].median(),
                  "p95_ms": observations[stage_cols].quantile(.95)}).to_csv(output / "stage_latency.csv")
    pd.DataFrame(mask_checks).to_csv(output / "archived_mask_comparison.csv", index=False)
    write_json(output / "protocol.json", {
        "status": "complete", "hardware": prior_protocol["hardware"], "hostname": prior_protocol["hostname"],
        "device": "cpu", "torch": torch_version, "threads": 2, "precision": "FP32 detector",
        "sample_batch_size": 1, "warmup_per_image": WARMUP, "repetitions_per_image": WASD_REPETITIONS,
        "input_samples": prior_protocol["input_samples"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "detector_commit": subprocess.check_output(["git", "-C", str(wasd_repo), "rev-parse", "HEAD"], text=True).strip(),
        "inference_source_sha256": sha256_file(wasd_repo / "calculate_subjective_masks.py"),
        "config_sha256": sha256_file(wasd_repo / "configs/heatmaps-iqa-datasets.yaml"),
        "configured_crop_size": crop, "actual_detector_patch_size": crop // 2,
        "detector_scope": "Full WASD forward as used by --subj_masks, coarse mask selected, in-memory RGB inputs",
        "omitted_costs": "Live NR extraction; model setup and first-use costs; detector FLOPs",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cached = commands.add_parser("cached", help="Time cached-feature stages and account for extractor costs")
    detector = commands.add_parser("wasd", help="Time WASD mask generation with a supplied checkpoint")
    for cmd in (cached, detector):
        for name in ("config", "run", "output"):
            cmd.add_argument(f"--{name}", type=Path, required=True)
    cached.add_argument("--historical-profiles", type=Path, required=True)
    for name in ("cost", "wasd-repo", "checkpoint"):
        detector.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.command == "cached":
            run_profile(args.config, args.run, args.historical_profiles, args.output)
        else:
            run_benchmark(args.config, args.run, args.cost, args.wasd_repo.resolve(), args.checkpoint, args.output)


if __name__ == "__main__":
    main()
