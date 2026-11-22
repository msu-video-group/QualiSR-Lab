"""Compare compact spatial summaries of archived WASD maps on the fixed rich pool."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from experiments.analyze_bt_pairs import paired_bt
from experiments.analyze_statistics import (
    BACKENDS,
    BT,
    MACRO,
    MOS,
    SEED,
    Analysis,
    estimate,
    holm,
    macro,
    read,
)
from experiments.data import STAT_COLUMNS, load_experiment_data, require, sha256_file
from experiments.main_feature_fusion import MAX_DIMS, checked_prior, fit_predict, reuse_transform
from experiments.run_grouped import METADATA, grouped_splits, write_json
from experiments.transfer import metric_rows
from experiments.transfer_features import predict_model, read_metadata
from experiments.wasd_ablation import REPLICATES, columns_for
from qualisr.datasets import load_datasets
from qualisr.statistics import compute_statistics_row, load_heatmap

POOL = "all_rr_no_fr"
CONTROL_ARMS = ("without_wasd", "with_wasd")
TILE = "wasd_max_tile_mean_4x4"
THRESHOLDS = {"region_010": .10, "region_020": .20, "region_050": .50, "region_075": .75}
SPATIAL_ARMS = {
    "spatial_050_075": [TILE, "region_050", "region_075"],
    "spatial_010_020": [TILE, "region_010", "region_020"],
}
ARMS = (*CONTROL_ARMS, *SPATIAL_ARMS)


def mask_path(sample):
    root = Path(sample["dataset_root"]) / "heatmaps"
    name = Path(sample["sr_path"]).stem + ".npy.gz"
    if sample["dataset"] == "dsr-dataset":
        return root / Path(sample["sample_id"]).parent.name / name
    if sample["dataset"] == "isrgenqa":
        return root / name
    return Path(sample["heatmap_path"])


def largest_region_fraction(heatmap, threshold):
    active = heatmap >= threshold
    if not active.any():
        return 0.
    regions, _ = label(active)  # SciPy's default in 2D uses four shared edges.
    return np.bincount(regions.ravel())[1:].max() / heatmap.size


def spatial_values(heatmap):
    require(heatmap.ndim == 2 and min(heatmap.shape) >= 4, "Expected a 2D map at least 4 by 4")
    require(np.isfinite(heatmap).all(), "Nonfinite WASD response")
    tiles = [tile for strip in np.array_split(heatmap, 4, axis=0) for tile in np.array_split(strip, 4, axis=1)]
    return {TILE: max(float(tile.mean(dtype=np.float64)) for tile in tiles),
            **{name: largest_region_fraction(heatmap, threshold) for name, threshold in THRESHOLDS.items()}}


def read_map(sample, cached, full_parity):
    path = mask_path(sample)
    heatmap = load_heatmap(path)
    with Image.open(sample["sr_path"]) as image:
        w, h = image.size
    require(heatmap.shape == (1, h, w), f"Archived map and SR geometry differ: {path}")
    heatmap = heatmap[0]
    np.testing.assert_allclose(float(heatmap.max()), cached["max"], atol=1e-7, rtol=0, err_msg=str(path))
    if full_parity:
        old = compute_statistics_row(sample["sample_id"], heatmap, (5., 95.), (0., .5, .75))
        np.testing.assert_allclose([old[k] for k in STAT_COLUMNS], [cached[k] for k in STAT_COLUMNS],
                                   atol=1e-7, rtol=0, err_msg=str(path))
    return dict(sample_id=sample["sample_id"], dataset=sample["dataset"], map_path=path, sr_path=sample["sr_path"],
                map_sha256=sha256_file(path), width=w, height=h, dtype=str(heatmap.dtype),
                **spatial_values(heatmap))


def extract(config, base, archive, output):
    samples = sorted(load_datasets(config["datasets"], base_dir=base), key=lambda sample: sample["sample_id"])
    previous = read_metadata(archive / "samples.csv")
    current = pd.DataFrame(samples)
    pd.testing.assert_frame_equal(current[["sample_id", "dataset"]],
                                  previous[["sample_id", "dataset"]], check_exact=True)
    paths = [mask_path(sample) for sample in samples]
    require(len(set(paths)) == len(paths), "A map path maps to multiple samples")
    missing = [path for path in paths if not path.is_file()]
    require(not missing, f"{len(missing)} missing maps: {missing[:3]}")
    caches = [read(Path(root) / "stats.csv") for root in current.features_root.unique()]
    cached = pd.concat(caches, ignore_index=True).set_index("sample_id", verify_integrity=True)
    require(set(cached.index) == set(current.sample_id), "Cached WASD IDs differ from archive map IDs")
    cached = cached[STAT_COLUMNS].to_dict(orient="index")
    pilot_indices = []
    for dataset, ids in current.groupby("dataset").groups.items():
        print(f"{dataset}: {len(ids)} archived maps located", flush=True)
        pilot_indices.extend((ids[0], ids[len(ids) // 2], ids[-1]))
    pilot = [read_map(samples[i], cached[samples[i]["sample_id"]], True) for i in pilot_indices]
    pd.DataFrame(pilot).to_csv(output / "map_pilot.csv", index=False)
    write_json(output / "map_pilot.json", dict(status="passed", sample_maps=len(pilot), matched_paths=len(paths)))
    print("Map pilot verified. Extracting all archived maps.", flush=True)
    done = {row["sample_id"]: row for row in pilot}

    def process(sample):
        sid = sample["sample_id"]
        return done[sid] if sid in done else read_map(sample, cached[sid], False)

    records = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for row in executor.map(process, samples):
            records.append(row)
            if len(records) % 1000 == 0:
                print(f"Verified and extracted {len(records)}/{len(samples)} maps", flush=True)
    spatial = pd.DataFrame(records).set_index("sample_id", verify_integrity=True)
    require(spatial[["region_050", "region_075"]].eq(0).all().all(),
            "Archive no longer matches the frozen high-threshold coverage audit")
    spatial.to_csv(output / "map_features.csv.gz")
    counts = spatial.groupby("dataset")[list(THRESHOLDS)].agg(lambda x: int(x.gt(0).sum()))
    counts.to_csv(output / "region_coverage.csv")
    return spatial[[TILE, *THRESHOLDS]]


def fit_fold(raw, spatial, settings, train, evaluate, archive, matched, output, seed, fold):
    require(pd.Index(train).is_unique and pd.Index(evaluate).is_unique, "Duplicate partition rows")
    (tx, vx), hashes = reuse_transform(raw, train, evaluate, archive / "outer", MAX_DIMS)
    meta = raw.loc[evaluate, METADATA].reset_index(drop=True)
    meta["evaluation"] = np.where(raw.loc[evaluate, "role"].eq("train"), "internal", "external")
    path = matched / "predictions.csv.gz"
    proof_path = matched / "verification.json"
    proof = json.loads(proof_path.read_text())
    require(proof["status"] == "passed" and proof["seed"] == seed and proof["fold"] == fold
            and proof["transform_sha256"] == hashes and proof["predictions_sha256"] == sha256_file(path),
            "Archived controls or transforms differ")
    controls = read_metadata(path).query("pool == @POOL")
    require(len(controls) == 6 * len(meta), "Incorrect archived control count")
    archived_params = {fit["model"]: fit["parameters"] for fit in proof["fits"] if fit["pool"] == POOL}
    target = raw.loc[train, "score"]
    scaler = MinMaxScaler().fit(spatial.loc[train])
    st = pd.DataFrame(scaler.transform(spatial.loc[train]), index=train, columns=spatial.columns)
    sv = pd.DataFrame(scaler.transform(spatial.loc[evaluate]), index=evaluate, columns=spatial.columns)
    columns = columns_for(POOL, "with_wasd")
    output.mkdir()
    joblib.dump(scaler, output / "spatial_scaler.joblib", compress=3)
    target.to_csv(output / "fit_targets.csv")
    forecasts, metrics, fits = [], [], []
    for backend in BACKENDS:
        values = {}
        for arm in CONTROL_ARMS:
            control = controls.loc[controls.model.eq(backend) & controls.arm.eq(arm)]
            require(len(control) == len(meta), "Control arm has wrong coverage")
            control = control.set_index("sample_id", verify_integrity=True).loc[meta.sample_id].reset_index()
            pd.testing.assert_frame_equal(control[meta.columns], meta, check_exact=True)
            require(control.split_seed.eq(seed).all() and control.fold.eq(fold).all(), "Control fold differs")
            values[arm] = control.prediction.to_numpy()
        for arm, extra in SPATIAL_ARMS.items():
            print(f"{output.name} {backend}/{arm}", flush=True)
            train_x = pd.concat([tx[columns], st[extra]], axis=1)
            valid_x = pd.concat([vx[columns], sv[extra]], axis=1)
            model, prediction = fit_predict(backend, {}, settings, train_x, valid_x, target, None)
            params = {key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
                      for key, value in model.get_params().items()}
            require(params == archived_params[backend], "New and archived model parameters differ")
            model_path = output / f"{backend}_{arm}.joblib"
            joblib.dump(model, model_path, compress=3)
            replay = predict_model(joblib.load(model_path), valid_x, backend, settings["threads"])
            np.testing.assert_allclose(replay, prediction, atol=1e-12, rtol=0)
            fits.append(dict(model=backend, arm=arm, columns=train_x.columns.tolist(), parameters=params,
                             model_sha256=sha256_file(model_path),
                             replay_error=float(np.max(np.abs(replay - prediction)))))
            values[arm] = prediction
        for arm, prediction in values.items():
            frame = meta.assign(prediction=prediction)
            keys = dict(model=backend, arm=arm, split_seed=seed, fold=fold)
            forecasts.append(frame.assign(**keys))
            scores, _ = metric_rows(frame)
            metrics.append(scores.assign(**keys))
    predictions = pd.concat(forecasts, ignore_index=True)
    saved = output / "predictions.csv.gz"
    predictions.to_csv(saved, index=False)
    pd.testing.assert_frame_equal(read_metadata(saved), predictions, check_exact=True)
    pd.concat(metrics, ignore_index=True).to_csv(output / "metrics.csv", index=False)
    write_json(output / "verification.json", dict(
        status="passed", seed=seed, fold=fold, new_fits=len(fits), controls=6, fits=fits,
        prediction_rows=len(predictions), predictions_sha256=sha256_file(saved),
        control_proof_sha256=sha256_file(proof_path), transform_sha256=hashes,
        spatial_scaler_sha256=sha256_file(output / "spatial_scaler.joblib"), n_fit=len(train),
    ))
    return len(predictions)


def analyze(root, output):
    panels = []
    for path in sorted(root.glob("seed_*_fold_*/predictions.csv.gz")):
        proof = json.loads((path.parent / "verification.json").read_text())
        require(proof["status"] == "passed" and sha256_file(path) == proof["predictions_sha256"],
                "Prediction file differs from proof")
        panels.append(read_metadata(path))
    require(len(panels) == 15, "Require all 15 verified folds")
    predictions = pd.concat(panels, ignore_index=True)
    output.mkdir()
    analysis = Analysis(REPLICATES)
    bt = dict(tuple(predictions.loc[predictions.dataset.eq(BT)].groupby(["arm", "model"])))
    scores = {}
    for arm, rows in predictions.loc[predictions.dataset.isin(MOS)].groupby("arm"):
        print(f"Scene bootstrap for {arm}", flush=True)
        for backend, group in rows.groupby("model"):
            result = estimate(group, REPLICATES)
            result[MACRO] = macro(result, MOS)
            analysis.record(analysis.cells, result, table=POOL, name=arm, model=backend, kind="score")
            scores[arm, backend] = result
        scores[arm, "backend_mean"] = analysis.mean_backend([scores[arm, b] for b in BACKENDS], POOL, arm)
    bt_rows, bt_folds = [], []
    for left, right in (CONTROL_ARMS, *((a, b) for a in CONTROL_ARMS for b in SPATIAL_ARMS)):
        contrast = f"{right}_minus_{left}"
        for backend in (*BACKENDS, "backend_mean"):
            result = analysis.pair(scores[left, backend], scores[right, backend], POOL, contrast, backend)
            if backend == "backend_mean" and left == "with_wasd":
                analysis.test(result[MACRO], contrast, MACRO, POOL, backend)
            summary, folds = paired_bt(bt, SimpleNamespace(model=backend), keys=((left,), (right,)),
                                       n_reps=REPLICATES)
            bt_rows.extend(dict(contrast=contrast, model=backend, **row) for row in summary)
            bt_folds.append(folds.assign(contrast=contrast, model=backend))
    analysis.primary = [row for row in analysis.primary if row["metric"] == "srcc"]
    analysis.write(output, expected_primary=2)
    primary = read(output / "primary_tests.csv")
    primary["p_holm"] = holm(primary.p_raw)
    primary.to_csv(output / "primary_tests.csv", index=False)
    pd.DataFrame(bt_rows).to_csv(output / "bt_common_groups_deltas.csv", index=False)
    pd.concat(bt_folds, ignore_index=True).to_csv(output / "bt_common_groups_folds.csv", index=False)
    write_json(output / "status.json", dict(status="complete", replicates=REPLICATES, seed=SEED, primary_tests=2))
    print(primary[["name", "conditional_point", "ci_low", "ci_high", "p_holm"]].to_string(index=False), flush=True)


def run(config_path, archive, controls, output):
    config = json.loads(config_path.read_text())
    control_status = json.loads((controls / "status.json").read_text())
    control_run = json.loads((controls / "run.json").read_text())
    require(control_status["status"] == "complete" and control_status["folds"] == 15
            and control_run["config"] == config and Path(control_run["verified_pca"]) == archive,
            "Require completed point-1 controls with the identical config and PCA archive")
    output.mkdir()
    write_json(output / "run.json", dict(
        config=config, archive=archive, controls=controls, spatial_arms=SPATIAL_ARMS, thresholds=THRESHOLDS,
        grid=[4, 4], connectivity=4, region_denominator="all map pixels", control_arms=CONTROL_ARMS,
        primary="Change in MOS macro SRCC over the ten-statistic control for each spatial arm, Holm corrected",
        threshold_rationale="The original thresholds never activate. Lower thresholds were fixed after inspecting "
                            "cached response maxima and before fitting or evaluating any spatial model.",
        source_sha256=sha256_file(Path(__file__)),
    ))
    write_json(output / "status.json", dict(status="extracting_archived_maps", folds=0))
    spatial = extract(config, config_path.resolve().parent, archive / "complete", output)
    print("Loading raw cached features and verifying all archive inputs", flush=True)
    raw, _, _, provenance = load_experiment_data(config, config_path.resolve().parent, include_all_rr=True)
    prior, _ = checked_prior(raw, provenance, archive)
    require(provenance == json.loads((controls / "data_provenance.json").read_text()), "Control inputs changed")
    write_json(output / "data_provenance.json", provenance)
    spatial = spatial.loc[raw.sample_id].set_axis(raw.index)
    settings = config["evaluation"]
    internal = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    n_folds = n_rows = 0
    for seed in settings["split_seeds"]:
        for fold, (train, valid) in enumerate(grouped_splits(raw, internal, settings["outer_folds"], seed), 1):
            key = f"seed_{seed}_fold_{fold:02d}"
            write_json(output / "status.json", dict(status="pilot" if not n_folds else "fitting",
                                                   folds=n_folds, active=[seed, fold]))
            rows = fit_fold(raw, spatial, settings, train, np.concatenate((valid, external)), prior / key,
                            controls / key, output / key, seed, fold)
            require(rows == 12 * (len(valid) + len(external)), "Missing prediction rows")
            n_rows += rows
            n_folds += 1
            if n_folds == 1:
                write_json(output / "pilot_verification.json", dict(
                    status="passed", fold=[seed, fold], new_fits=6, controls=6, prediction_rows=rows,
                    proof_sha256=sha256_file(output / key / "verification.json"),
                ))
                print("Model pilot passed. Continuing with the remaining 14 folds.", flush=True)
    require(n_folds == 15 and n_rows == control_status["prediction_rows"], "Mismatched 15-fold panel")
    counts = dict(folds=n_folds, new_fits=90, controls=90, map_rows=len(spatial), prediction_rows=n_rows)
    write_json(output / "status.json", dict(status="fits_verified", **counts))
    analyze(output, output / "analysis")
    write_json(output / "status.json", dict(status="complete", **counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "archive", "controls", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.config, args.archive, args.controls, args.output)
