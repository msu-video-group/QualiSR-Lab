"""Compare five paired PCA dimensions with fixed heads and nested dimension selection."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.data import (
    NR_COLUMNS,
    SIGLIP_COLUMNS,
    STAT_COLUMNS,
    load_experiment_data,
    read_csv,
    require,
    sha256_file,
)
from experiments.fold_features import FoldTransformer
from experiments.run_grouped import METADATA, feature_columns, grouped_splits, make_model, write_json
from experiments.transfer import check_partition
from experiments.transfer_features import predict_model
from qualisr.regressors import validation_correlations

PAIRS = ((5, 5), (5, 50), (25, 50), (50, 50), (96, 96))
MAXIMUM = {"vgg": 96, "resnet": 96}
LEGACY = {"vgg": 10, "resnet": 50}
KEY = ["split_seed", "fold", "pool", "model", "track"]


def pools():
    return {
        "nr_ref": {"scalars": list(NR_COLUMNS), "representations": ["ref"]},
        "all_rr_no_fr": {"scalars": NR_COLUMNS + STAT_COLUMNS + SIGLIP_COLUMNS,
                         "representations": ["sr", "ref", "diff"]},
    }


def pair_name(pair):
    return f"vgg{pair[0]:02d}_resnet{pair[1]:02d}"


def choose(history):
    require(np.isfinite(history.mean_srcc).all(), "Every candidate must have finite inner SRCC")
    return int(history.sort_values(["mean_srcc", "candidate"], ascending=[False, True]).candidate.iat[0])


def prepare(raw, train, evaluate, blocks, scalars, dimensions, seed, folder):
    check_partition(raw, train, evaluate)
    folder.mkdir()
    transform = FoldTransformer(scalars, blocks, dimensions, seed=seed)
    tx = transform.fit_transform(raw.loc[train])
    vx = transform.transform(raw.loc[evaluate])
    for family, pca in transform.pcas_.items():
        rank = int((pca.singular_values_ > pca.singular_values_[0] *
                    max(2 * len(train), len(blocks[family][0])) *
                    np.finfo(pca.singular_values_.dtype).eps).sum())
        require(rank >= dimensions[family], f"Insufficient numerical training rank for {family}")
    joblib.dump(transform, folder / "transformer.joblib", compress=3)
    write_json(folder / "partition.json", dict(train=train.tolist(), evaluate=evaluate.tolist(),
               fit_sample_ids=transform.fit_sample_ids_, dimensions=dimensions,
               explained_variance={k: float(p.explained_variance_ratio_.sum())
                                   for k, p in transform.pcas_.items()}))
    np.savez_compressed(folder / "evaluation.npz", values=vx.to_numpy(),
                        columns=vx.columns.to_numpy(dtype=str), indices=evaluate)
    return tx, vx


def fit_backend(raw, train, evaluate, settings, backend, prepared, inner, folder):
    seed, threads = settings["model_seed"], settings["threads"]
    folder.mkdir()
    metadata = raw.loc[evaluate, METADATA].copy()
    metadata["evaluation"] = np.where(raw.loc[evaluate, "role"].eq("train"), "internal", "external")
    records, forecasts, metric_records, search_records = [], [], [], []
    for pool, spec in pools().items():
        target = folder / pool
        target.mkdir()
        history = []
        for candidate, pair in enumerate(PAIRS):
            columns = feature_columns(spec, dict(zip(("vgg", "resnet"), pair, strict=True)))
            inner_scores, inner_plcc = [], []
            for i, (it, iv) in enumerate(inner, 1):
                tx, vx = prepared[f"inner_{i}"]
                model = make_model(backend, {}, seed, threads).fit(tx[columns], raw.loc[it, "score"])
                prediction = np.asarray(predict_model(model, vx[columns], backend, threads), dtype=float)
                metadata_inner = raw.loc[iv, METADATA]
                plcc, srcc, datasets, _ = validation_correlations(metadata_inner.score, prediction, metadata_inner)
                require(np.isfinite(datasets[["plcc", "srcc"]]).all().all(),
                        "Every inner dataset must have finite correlations")
                inner_plcc.append(plcc)
                inner_scores.append(srcc)
                np.savez_compressed(target / f"inner_c{candidate}_f{i}.npz", indices=iv, prediction=prediction)
            history.append(dict(candidate=candidate, vgg=pair[0], resnet=pair[1],
                                mean_plcc=float(np.mean(inner_plcc)), mean_srcc=float(np.mean(inner_scores)),
                                fold_srcc=json.dumps(inner_scores), fold_plcc=json.dumps(inner_plcc)))
        history = pd.DataFrame(history)
        winner = choose(history)
        history["selected"] = history.candidate.eq(winner)
        history.to_csv(target / "inner_search.csv", index=False)
        search_records.append(history.assign(pool=pool, model=backend))
        pool_forecasts = []
        tracks = [(pair_name(p), p, "outer", i) for i, p in enumerate(PAIRS)]
        tracks.append(("legacy_basis_5_5", PAIRS[0], "legacy", 0))
        for track, pair, basis, candidate in tracks:
            print(f"  {backend}/{pool}/{track}", flush=True)
            started = time.monotonic()
            columns = feature_columns(spec, dict(zip(("vgg", "resnet"), pair, strict=True)))
            tx, vx = prepared[basis]
            model = make_model(backend, {}, seed, threads).fit(tx[columns], raw.loc[train, "score"])
            prediction = np.asarray(predict_model(model, vx[columns], backend, threads), dtype=float)
            model_dir = target / track
            model_dir.mkdir()
            joblib.dump(model, model_dir / "model.joblib", compress=3)
            actual = {k: str(v) if isinstance(v, float) and not np.isfinite(v) else v
                      for k, v in model.get_params().items()}
            write_json(model_dir / "parameters.json", dict(columns=columns, actual=actual))
            labels = dict(pool=pool, model=backend, track=track)
            row = dict(**labels, vgg=pair[0], resnet=pair[1], basis=basis, candidate=candidate,
                       n_fit=len(train), n_predict=len(evaluate), n_features=len(columns),
                       fitted=True, seconds=time.monotonic() - started)
            records.append(row)
            pred = metadata.assign(prediction=prediction, **labels)
            pool_forecasts.append(pred)
            if basis == "outer" and candidate == winner:
                records.append({**row, "track": "inner_selected", "fitted": False, "seconds": 0.0})
                pool_forecasts.append(pred.assign(track="inner_selected"))
        forecasts.extend(pool_forecasts)
        for pred in pool_forecasts:
            for scope, group in pred.groupby("evaluation", sort=True):
                _, _, values, _ = validation_correlations(group.score, group.prediction, group)
                metric_records.append(values.assign(evaluation=scope, pool=pool, model=backend,
                                                    track=pred.track.iat[0]))
    return pd.DataFrame(records), pd.concat(forecasts), pd.concat(metric_records), pd.concat(search_records)


def run(config, base, prior, output, *, pilot=False, workers=3, start_fold=1, data=None):
    output.mkdir(parents=True, exist_ok=False)
    settings = config["evaluation"]
    require(settings["threads"] == 2 and 1 <= workers <= 3, "Use two model threads and at most three workers")
    require(1 <= start_fold <= settings["outer_folds"], "Starting fold is outside the outer grid")
    require(not pilot or start_fold == 1, "The pilot always uses the first outer fold")
    raw, blocks, scalars, provenance = load_experiment_data(config, base, include_all_rr=True) if data is None else data
    original = read_csv(prior, "samples.csv")
    raw[original.columns].to_csv(output / "samples.csv", index=False)
    pd.testing.assert_frame_equal(read_csv(output, "samples.csv"), original, check_exact=True)
    previous_files = json.loads((prior / "data_provenance.json").read_text())["files"]
    hashes = {(p["dataset"], p["path"]): p["sha256"] for p in provenance["files"]}
    require(all(hashes.get((p["dataset"], p["path"])) == p["sha256"] for p in previous_files),
            "Raw caches changed since the original grouped run")
    write_json(output / "data_provenance.json", provenance)
    write_json(output / "run.json", dict(config=config, pilot=pilot, workers=workers, start_fold=start_fold, pairs=PAIRS,
               maximum=MAXIMUM, legacy=LEGACY, pools=pools(), python=platform.python_version(),
               versions={p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib")},
               code_sha256=sha256_file(Path(__file__)),
               selection="Mean dataset-macro SRCC over grouped inner folds; fixed default heads; earliest candidate breaks ties",
               basis="Shared SR/reference training-only basis per encoder, fitted to maximum dimensions; prefixes reused",
               external="Retrospective evaluation only; external labels excluded from preprocessing and selection"))
    original_assignments = read_csv(prior, "fold_assignments.csv")
    pool = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    fits, metrics, searches, assignments = [], [], [], []
    seeds = settings["split_seeds"][:1] if pilot else settings["split_seeds"]
    for split_seed in seeds:
        for fold, (train, valid) in enumerate(grouped_splits(raw, pool, settings["outer_folds"], split_seed), 1):
            if fold < start_fold:
                continue
            if pilot and fold > 1:
                break
            print(f"seed={split_seed} fold={fold}: preparing shared paired PCA", flush=True)
            target = output / f"seed_{split_seed}_fold_{fold:02d}"
            target.mkdir()
            assignment = original_assignments.loc[(original_assignments.split_seed == split_seed) &
                                                   (original_assignments.fold == fold)]
            for partition, indices in (("train", train), ("heldout", valid)):
                require(set(assignment.loc[assignment.partition.eq(partition), "sample_id"]) ==
                        set(raw.loc[indices, "sample_id"]), "Original outer partitions changed")
            assignments.append(assignment)
            evaluate = np.concatenate([valid, external])
            inner = list(grouped_splits(raw, train, settings["inner_folds"], split_seed))
            prepared = {}
            for name, it, iv, dims in [("outer", train, evaluate, MAXIMUM), ("legacy", train, evaluate, LEGACY),
                                      *[(f"inner_{i}", it, iv, MAXIMUM) for i, (it, iv) in enumerate(inner, 1)]]:
                print(f"  preparing {name}", flush=True)
                prepared[name] = prepare(raw, it, iv, blocks, scalars, dims, settings["model_seed"], target / name)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = [executor.submit(fit_backend, raw, train, evaluate, settings, backend,
                                           prepared, inner, target / backend) for backend in settings["models"]]
                predictions = []
                for future in pending:
                    fit, pred, metric, search = future.result()
                    labels = dict(split_seed=split_seed, fold=fold)
                    fits.append(fit.assign(**labels))
                    predictions.append(pred.assign(**labels))
                    metrics.append(metric.assign(**labels))
                    searches.append(search.assign(**labels))
            pd.concat(predictions, ignore_index=True).to_csv(target / "predictions.csv.gz", index=False)
            for name, tables in (("fits", fits), ("metrics", metrics), ("inner_search", searches),
                                 ("fold_assignments", assignments)):
                pd.concat(tables, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
            write_json(output / "status.json", dict(status="running", last_completed=[split_seed, fold]))
    write_json(output / "status.json", dict(status="complete", n_records=len(pd.concat(fits)),
               n_fitted=int(pd.concat(fits).fitted.sum()), n_inner_fits=len(pd.concat(searches)) * settings["inner_folds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "prior", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--start-fold", type=int, default=1,
                        help="First outer fold to execute, retaining original fold numbers")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with threadpool_limits(limits=2):
        run(config, args.config.parent, args.prior, args.output, pilot=args.pilot,
            workers=args.workers, start_fold=args.start_fold)


if __name__ == "__main__":
    main()
