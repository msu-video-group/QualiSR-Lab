"""Reproduce the archived five-NR result for every backend on one frozen outer fold."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.analyze_statistics import BACKENDS
from experiments.data import NR_COLUMNS, _read_block, sha256_file
from experiments.fold_features import FoldTransformer
from experiments.run_grouped import METADATA, grouped_splits, make_model, save_evaluation, write_json
from qualisr.datasets import load_datasets

SPLIT_SEED = 42
FOLD = 1
TOLERANCE = 1e-12


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def reproduce(config_path, archive, output, backend):
    config = json.loads(config_path.read_text())
    if config != json.loads((archive / "run.json").read_text())["config"]:
        raise ValueError("Configuration differs from the archived experiment run")
    settings = config["evaluation"]
    seed, threads = settings["model_seed"], settings["threads"]
    expected = read_csv(archive / "samples.csv").set_index("sample_id", verify_integrity=True).sort_index()
    parsed = load_datasets(config["datasets"], base_dir=config_path.resolve().parent)
    parsed = pd.DataFrame(parsed).set_index("sample_id", verify_integrity=True).sort_index()
    parsed["role"] = ["train" if usage["train"] else "external" for usage in parsed.regressors]
    # Reuse the source identities from the archived byte-level audit.
    fields = ["dataset", "test_case", "correlation_group", "score_type", "score", "method", "role"]
    pd.testing.assert_frame_equal(parsed[fields], expected[fields], check_exact=True)
    raw = parsed[fields].join(expected.source_group).reset_index()
    files = json.loads((archive / "data_provenance.json").read_text())["files"]
    nr_digests = {
        item["dataset"]: item["sha256"] for item in files if Path(item["path"]).name == "nr.csv"
    }
    blocks, provenance = [], []
    for dataset, frame in parsed.groupby("dataset"):
        root, = frame.features_root.unique()
        block, info = _read_block(Path(root) / "nr.csv", frame.index, columns=NR_COLUMNS)
        if info["sha256"] != nr_digests[dataset]:
            raise ValueError(f"Changed NR cache for {dataset}")
        blocks.append(block)
        provenance.append({"dataset": dataset, **info})
    raw = raw.join(pd.concat(blocks), on="sample_id", validate="one_to_one")
    train_pool = raw.index[raw.role.eq("train")].to_numpy()
    external = raw.index[raw.role.eq("external")].to_numpy()
    train, heldout = next(grouped_splits(raw, train_pool, settings["outer_folds"], SPLIT_SEED))
    assignment = read_csv(archive / "fold_assignments.csv")
    assignment = assignment.loc[assignment.split_seed.eq(SPLIT_SEED) & assignment.fold.eq(FOLD)]
    for label, ids in (("train", train), ("heldout", heldout)):
        assert raw.loc[ids, "sample_id"].tolist() == assignment.loc[assignment.partition.eq(label), "sample_id"].tolist()
    fold_root = archive / "seed_42/fold_01"
    assert raw.loc[train, "sample_id"].tolist() == read_csv(fold_root / "outer_transform/fit_samples.csv").sample_id.tolist()
    evaluated = np.concatenate([heldout, external])
    metadata = raw.loc[evaluated, METADATA].assign(evaluation=np.where(np.isin(evaluated, heldout), "internal", "external"))
    output.mkdir()
    with threadpool_limits(limits=threads):
        transformer = FoldTransformer(NR_COLUMNS, {}, {}, seed=seed).fit(raw.loc[train])
        x_train = transformer.transform(raw.loc[train])
        x_eval = transformer.transform(raw.loc[evaluated])
        model = make_model(backend, {}, seed, threads).fit(x_train, raw.loc[train, "score"])
        params = {"experiment": "nr", "track": "fixed", "split_seed": SPLIT_SEED, "fold": FOLD, "model_seed": seed}
        save_evaluation(output / "result", model, NR_COLUMNS, x_eval, metadata, params)
    reference = fold_root / "nr" / backend / "fixed"
    predictions = read_csv(output / "result/predictions.csv")
    saved = read_csv(reference / "predictions.csv")
    pd.testing.assert_frame_equal(predictions.drop(columns="prediction"), saved.drop(columns="prediction"), check_exact=True)
    np.testing.assert_allclose(predictions.prediction, saved.prediction, atol=TOLERANCE, rtol=0)
    metrics = read_csv(output / "result/metrics.csv")
    pd.testing.assert_frame_equal(metrics, read_csv(reference / "metrics.csv"), atol=TOLERANCE, rtol=0)
    raw.loc[train, ["sample_id"]].to_csv(output / "fit_samples.csv", index=False)
    write_json(output / "verification.json", {
        "status": "passed",
        "backend": backend,
        "config_sha256": sha256_file(config_path),
        "archive": str(archive),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "features": NR_COLUMNS,
        "feature_files": provenance,
        "settings": params,
        "fit_rows": len(train),
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "prediction_max_abs_error": float((predictions.prediction - saved.prediction).abs().max()),
        "absolute_tolerance": TOLERANCE,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    for backend in BACKENDS:
        reproduce(args.config, args.archive, args.output / backend, backend)
