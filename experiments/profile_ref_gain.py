"""Matched CPU latency of five NR and five NR + bicubic Ref, including source reuse."""

import argparse
import json
import platform
from functools import cache
from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS, MOS, read
from experiments.data import NR_COLUMNS, _read_block, require, sha256_file
from experiments.run_grouped import write_json


def project_selected(features, transformer, columns):
    """Apply the saved transform only to used NR/Ref coordinates; no SR embedding is needed."""
    projected = {name: features[name] for name in columns if name in transformer.scalar_columns}
    for family, (_, reference_columns) in transformer.embedding_blocks.items():
        names = [c for c in columns if c.startswith(f"{family}_ref_")]
        if names:
            components = [int(c.rsplit("_", 1)[1]) for c in names]
            pca = transformer.pcas_[family]
            reference = np.asarray([features[c] for c in reference_columns], dtype=float)
            projected.update(zip(names, pca.components_[components] @ (reference - pca.mean_), strict=True))
    positions = [transformer.feature_names_out_.index(c) for c in columns]
    values = np.asarray([projected[c] for c in columns], dtype=float)
    values = np.where(np.isfinite(values), values, transformer.imputer_.statistics_[positions])
    values = values * transformer.scaler_.scale_[positions] + transformer.scaler_.min_[positions]
    return pd.DataFrame([values], columns=columns)


def paired_costs(times, backend, reused_reference_ms):
    common = sum(times[s] for s in ["sr_decode", "nr_tensor", *NR_COLUMNS])
    reference = sum(times[s] for s in ["lr_decode", "bicubic", "ref_vgg", "ref_resnet"])
    nr = common + times["nr_transform"] + times[f"nr_{backend}_predict"]
    ref_head = times["nr_ref_transform"] + times[f"nr_ref_{backend}_predict"]
    standalone = common + reference + ref_head
    reused = common + reused_reference_ms + ref_head
    return dict(nr_ms=nr, nr_ref_standalone_ms=standalone, delta_standalone_ms=standalone - nr,
                nr_ref_reuse_ms=reused, delta_reuse_ms=reused - nr,
                standalone_reference_ms=reference, amortized_reference_ms=reused_reference_ms)


def select_inputs(samples, fit_ids, evaluated_ids):
    """Choose two methods at low/median/high area in each MOS dataset, without using scores."""
    fit_sources = set(samples.loc[fit_ids, "source_group"])

    @cache
    def size(path):
        with Image.open(path) as image:
            return image.size

    @cache
    def digest(path):
        return sha256_file(Path(path))

    chosen = []
    for dataset in MOS:
        eligible = samples.loc[samples.index.isin(evaluated_ids) & samples.dataset.eq(dataset)].copy()
        require(not set(eligible.source_group) & fit_sources, "Timing/evaluation source overlaps fitted sources")
        eligible["size"] = eligible.sr_path.map(size)
        eligible["lr_sha256"] = eligible.lr_path.map(digest)
        candidates = []
        for key, part in eligible.groupby(["source_group", "lr_sha256", "size"], sort=True):
            if part.method.nunique() < 2:
                continue
            ids = part.sort_values(["method", "sample_id"]).drop_duplicates("method").head(2).index.tolist()
            candidates.append((key[2][0] * key[2][1], key[0], key[1], key[2], ids))
        candidates.sort()
        require(len(candidates) >= 3, f"Fewer than three complete LR/SR groups in {dataset}")
        used = set()
        for label, target in zip(("low", "middle", "high"), (0, (len(candidates) - 1) / 2, len(candidates) - 1), strict=True):
            order = sorted(range(len(candidates)), key=lambda i, target=target: (abs(i - target), i))
            index = next(i for i in order if candidates[i][1] not in used)
            area, source, lr_hash, sr_size, ids = candidates[index]
            used.add(source)
            for sid in ids:
                row = samples.loc[sid].to_dict()
                row.update(sample_id=sid, resolution_bin=label, sr_pixels=area,
                           sr_width=sr_size[0], sr_height=sr_size[1], lr_sha256=lr_hash,
                           sr_sha256=digest(row["sr_path"]), lr_width=size(row["lr_path"])[0],
                           lr_height=size(row["lr_path"])[1], eligible_groups=len(candidates))
                chosen.append(row)
    return pd.DataFrame(chosen)


def load_heads(prior):
    root = prior / "seed_42/fold_01"
    transformer = joblib.load(root / "outer_transform/transformer.joblib")
    models, columns, hashes = {}, {}, {}
    for name in ("nr", "nr_ref"):
        for backend in BACKENDS:
            folder = root / name / backend / "fixed"
            models[(name, backend)] = joblib.load(folder / "model.joblib")
            found = json.loads((folder / "parameters.json").read_text())["columns"]
            require(name not in columns or columns[name] == found, "Backend feature columns differ")
            columns[name] = found
            hashes[f"{name}/{backend}"] = sha256_file(folder / "model.joblib")
    require(columns["nr"] == NR_COLUMNS, "The baseline must contain all five NR metrics")
    require(len(columns["nr_ref"]) == 15, "Expected five NR and ten reference coordinates")
    hashes["transformer"] = sha256_file(root / "outer_transform/transformer.joblib")
    return transformer, models, columns, hashes


def cached_parity(inputs, samples, prior, core, transformer, models, columns):
    records = json.loads((core / "data_provenance.json").read_text())["files"]
    frames, hashes = [], []
    for dataset in inputs.dataset.unique():
        ids = samples.index[samples.dataset.eq(dataset)]
        wanted = inputs.loc[inputs.dataset.eq(dataset), "sample_id"]
        tables = []
        for name in ("nr", "ref_vgg", "ref_resnet"):
            record, = [r for r in records if r["dataset"] == dataset and Path(r["path"]).stem == name]
            table, source = _read_block(Path(record["path"]), ids,
                                        columns=NR_COLUMNS if name == "nr" else None,
                                        prefix=None if name == "nr" else name + "_")
            require(source["sha256"] == record["sha256"], f"Changed feature cache: {record['path']}")
            hashes.append(record)
            tables.append(table.loc[wanted])
        frames.append(pd.concat(tables, axis=1))
    cached = pd.concat(frames)
    checks = []
    for (name, backend), model in models.items():
        expected = read(prior / f"seed_42/fold_01/{name}/{backend}/fixed/predictions.csv").set_index("sample_id")
        for sid, features in cached.iterrows():
            actual = float(model.predict(project_selected(features, transformer, columns[name]))[0])
            error = abs(actual - expected.at[sid, "prediction"])
            require(error <= 1e-12, f"Cached prediction parity failed: {sid}/{name}/{backend}: {error}")
            checks.append(dict(sample_id=sid, pipeline=name, model=backend, absolute_error=error))
    return pd.DataFrame(checks), hashes, cached


def summarize_latencies(frame):
    rows = []
    values = [c for c in frame.columns if c.endswith("_ms")]
    for key, group in frame.groupby(["dataset", "resolution_bin", "model"], sort=True):
        row = dict(zip(["dataset", "resolution_bin", "model"], key, strict=True),
                   n_measurements=len(group), n_images=group.sample_id.nunique(),
                   n_sources=group.source_group.nunique(),
                   sr_pixels_min=int(group.sr_pixels.min()), sr_pixels_max=int(group.sr_pixels.max()))
        for name in values:
            lo, median, hi = group[name].quantile([0.25, 0.5, 0.75])
            row.update({name + "_median": median, name + "_q25": lo, name + "_q75": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def run(args):
    import importlib.metadata

    import torch

    from qualisr.features import (
        extract_pretrained_features,
        image_to_tensor,
        init_nr_models,
        init_resnet,
        init_vgg,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    samples = read(args.core / "samples.csv").set_index("sample_id", drop=False)
    samples.index.name = None
    transformer, models, columns, hashes = load_heads(args.prior)
    expected = read(args.prior / "seed_42/fold_01/nr/randomforest/fixed/predictions.csv")
    inputs = select_inputs(samples, transformer.fit_sample_ids_, expected.sample_id)
    # The stored index and explicit sample ID are both useful to the selector; avoid pandas ambiguity.
    if args.pilot:
        inputs = inputs.iloc[:2].copy()
    inputs.to_csv(args.output / "inputs.csv", index=False)
    parity, feature_hashes, cached = cached_parity(inputs, samples, args.prior, args.core, transformer, models, columns)
    parity.to_csv(args.output / "cached_prediction_parity.csv", index=False)
    print(f"Verified {len(parity)} saved-head/cache predictions; initializing live CPU extractors", flush=True)
    device = torch.device("cpu")
    nr = init_nr_models(device, NR_COLUMNS)
    qalign = nr["qalign"].net
    qalign.model.float()
    qalign.model_dtype = torch.float32
    qalign.model.weight_tensor = torch.tensor([5., 4., 3., 2., 1.], dtype=torch.float32, device=device)
    encoders = {"vgg": init_vgg(device), "resnet": init_resnet(device)}
    protocol = dict(
        pilot=args.pilot, repetitions=args.repetitions, warmup="one complete first-image iteration, excluded",
        threads=torch.get_num_threads(), device="cpu", precision="FP32 neural models, float64 saved PCA/scaling",
        selection="two distinct methods sharing identical LR bytes and SR size at low/median/high area per MOS dataset",
        split_seed=42, fold=1, training="saved heads and saved training-only PCA/imputer/scaler; no fitting",
        timing="synchronous in-process stage sums; shared five-NR extractions across both pools and all three heads",
        reuse="one measured reference extraction per source/repeat, amortized equally over its two timed SR outputs",
        exclusions="model initialization, checkpoint loading, initial input audit and cache-parity verification",
        quality="separately recorded cached OOF/transfer quality; live extractor parity is measured, not assumed",
        versions={x: importlib.metadata.version(x) for x in ["torch", "torchvision", "pyiqa", "numpy", "pandas", "scikit-learn", "catboost"]},
        hostname=platform.node(), platform=platform.platform(), model_sha256=hashes,
        feature_inputs=feature_hashes, code_sha256=sha256_file(Path(__file__)))
    write_json(args.output / "protocol.json", protocol)
    measurements, costs, live, reuse_checks, live_parity = [], [], [], [], []
    warm = True
    with torch.inference_mode():
        for (dataset, _source), group in inputs.groupby(["dataset", "source_group"], sort=False):
            for repeat in range(args.repetitions):
                reference_cost, reference_features = None, None
                for row in group.to_dict("records"):
                    sid = row["sample_id"]
                    for is_warmup in ([True, False] if warm else [False]):
                        times, features = {}, {}

                        def timed(stage, function, *values, times=times):
                            start = perf_counter_ns()
                            result = function(*values)
                            times[stage] = (perf_counter_ns() - start) / 1e6
                            return result

                        def decode(path):
                            with Image.open(path) as image:
                                return image.convert("RGB")

                        sr = timed("sr_decode", decode, row["sr_path"])
                        tensor = timed("nr_tensor", image_to_tensor, sr, device)
                        for name, net in nr.items():
                            features[name] = float(timed(name, net, tensor).item())
                        lr = timed("lr_decode", decode, row["lr_path"])
                        reference = timed("bicubic", lr.resize, sr.size, Image.Resampling.BICUBIC)
                        for family, (net, transform) in encoders.items():
                            vector = timed("ref_" + family, extract_pretrained_features, reference, net, transform, device)
                            names = transformer.embedding_blocks[family][1]
                            require(len(vector) == len(names), "Live and cached embedding dimensions differ")
                            features.update(zip(names, vector, strict=True))
                        nr_x = timed("nr_transform", project_selected, features, transformer, columns["nr"])
                        ref_x = timed("nr_ref_transform", project_selected, features, transformer, columns["nr_ref"])
                        predictions = {}
                        for (name, backend), model in models.items():
                            x = nr_x if name == "nr" else ref_x
                            predictions[f"{name}_{backend}"] = float(timed(f"{name}_{backend}_predict", model.predict, x)[0])
                        require(np.isfinite(list(features.values())).all(), "Nonfinite live feature")
                        require(np.isfinite(list(predictions.values())).all(), "Nonfinite live prediction")
                        print(f"{dataset} {row['resolution_bin']} {sid} repeat={repeat} warmup={is_warmup} Q-Align={times['qalign']/1000:.2f}s", flush=True)
                        if is_warmup:
                            warm = False
                            continue
                        metadata = {k: row[k] for k in ["sample_id", "dataset", "source_group", "resolution_bin", "sr_pixels"]}
                        metadata["repeat"] = repeat
                        ref_keys = [c for c in features if c.startswith("ref_")]
                        if reference_cost is None:
                            reference_cost = sum(times[s] for s in ["lr_decode", "bicubic", "ref_vgg", "ref_resnet"])
                            reference_features = np.array([features[k] for k in ref_keys])
                        reuse_error = float(np.max(np.abs(reference_features - [features[k] for k in ref_keys])))
                        require(reuse_error == 0, "Reference embeddings differ between outputs of identical LR")
                        reuse_checks.append(dict(**metadata, maximum_embedding_error=reuse_error))
                        measurements.extend(dict(**metadata, stage=stage, milliseconds=value) for stage, value in times.items())
                        for backend in BACKENDS:
                            costs.append(dict(**metadata, model=backend,
                                              **paired_costs(times, backend, reference_cost / len(group))))
                        live.append(dict(**metadata, **{k: features[k] for k in NR_COLUMNS}, **predictions))
                        for name in NR_COLUMNS:
                            live_parity.append(dict(**metadata, feature=name, live=features[name],
                                                    cached=float(cached.at[sid, name]),
                                                    absolute_error=abs(features[name] - cached.at[sid, name])))
                        expected_x = project_selected(cached.loc[sid], transformer, columns["nr_ref"])
                        for name in columns["nr_ref"][5:]:
                            live_parity.append(dict(**metadata, feature=name, live=float(ref_x.at[0, name]),
                                                    cached=float(expected_x.at[0, name]),
                                                    absolute_error=abs(ref_x.at[0, name] - expected_x.at[0, name])))
                        # Persist progress in the new run only, so an interrupted run remains inspectable.
                        pd.DataFrame(measurements).to_csv(args.output / "measurements.csv", index=False)
                        pd.DataFrame(costs).to_csv(args.output / "paired_latency.csv", index=False)
                        pd.DataFrame(live).to_csv(args.output / "live_predictions.csv", index=False)
                        pd.DataFrame(live_parity).to_csv(args.output / "live_cache_feature_parity.csv", index=False)
                        pd.DataFrame(reuse_checks).to_csv(args.output / "reference_reuse_parity.csv", index=False)
    summarize_latencies(pd.DataFrame(costs)).to_csv(args.output / "latency_summary.csv", index=False)
    write_json(args.output / "status.json", dict(status="complete", n_inputs=len(inputs), n_cost_rows=len(costs),
                                                 maximum_cached_prediction_error=float(parity.absolute_error.max())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
