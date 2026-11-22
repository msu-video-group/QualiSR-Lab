"""Strict raw-feature loading and source-content audit for the paper experiments."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qualisr.datasets import load_datasets

NR_COLUMNS = ["musiq", "arniqa", "qalign", "unique", "paq2piq"]
FR_METRICS = ["psnr", "ssim", "lpips-vgg", "stlpips-vgg", "pieapp", "ahiq"]
FR_COLUMNS = [f"{metric}_{reference}" for reference in ("bicubic", "rlfn", "span") for metric in FR_METRICS]
GT_COLUMNS = [f"{metric}_gt" for metric in FR_METRICS]
SIGLIP_COLUMNS = ["content_fidelity", "perceptual_enhancement", "final_rr_score"]
STAT_COLUMNS = ["min", "max", "mean", "median", "std", "p05", "p95", "area00", "area05", "area075"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(root, name):
    return pd.read_csv(root / name, float_precision="round_trip")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_block(
    path: Path,
    sample_ids: pd.Index,
    columns: list[str] | None = None,
    prefix: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if prefix is not None:
        pattern = re.compile(re.escape(prefix) + r"(\d+)$")
        columns = sorted(
            (name for name in header if pattern.fullmatch(name)),
            key=lambda name: int(name[len(prefix):]),
        )
        if not columns:
            raise ValueError(f"No raw {prefix} embedding columns in {path}")
    if columns is None:
        raise ValueError("A block must specify columns or an embedding prefix")
    missing = sorted({"sample_id", *columns} - set(header))
    if missing:
        raise ValueError(f"Feature file {path} is missing columns: {missing}")
    numeric_type = np.float32 if prefix is not None else np.float64
    frame = pd.read_csv(
        path, usecols=["sample_id", *columns], dtype={"sample_id": str, **dict.fromkeys(columns, numeric_type)}
    )
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"Feature file has missing or duplicate sample IDs: {path}")
    actual = set(frame["sample_id"])
    expected = set(sample_ids)
    if actual != expected:
        raise ValueError(
            f"Feature sample coverage differs in {path}: "
            f"{len(expected - actual)} missing, {len(actual - expected)} unexpected IDs"
        )
    frame = frame.set_index("sample_id").loc[sample_ids, columns]
    provenance = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "n_rows": len(frame),
        "n_columns": len(header),
        "selected_columns": columns,
    }
    return frame, provenance


def _audit_content(frame: pd.DataFrame) -> dict[str, Any]:
    """Unify byte-identical source images and reject train/external overlap."""
    paths: dict[str, str] = {}
    hashes = []
    kinds = []
    for sample in frame[["gt_path", "lr_path"]].itertuples(index=False):
        kind = "hr" if sample.gt_path else "lr"
        path = sample.gt_path if sample.gt_path else sample.lr_path
        if path not in paths:
            paths[path] = sha256_file(Path(path))
        hashes.append(paths[path])
        kinds.append(kind)
    frame["content_sha256"] = hashes
    frame["content_kind"] = kinds
    identities = frame["dataset"].astype(str) + "/" + frame["test_case"].astype(str)
    parent = {identity: identity for identity in identities}

    def find(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    frame["source_group"] = identities
    shared = []
    overlaps = []
    for (kind, digest), group in frame.groupby(["content_kind", "content_sha256"], sort=True):
        source_ids = sorted(group["source_group"].unique())
        roots = sorted({find(identity) for identity in source_ids})
        for root in roots[1:]:
            parent[root] = roots[0]
        if len(group["dataset"].unique()) > 1:
            entry = {"kind": kind, "sha256": digest, "sources": source_ids}
            shared.append(entry)
            if set(group["role"]) == {"train", "external"}:
                overlaps.append(entry)
    if overlaps:
        raise ValueError(
            f"Source-content audit found {len(overlaps)} byte-identical source images shared "
            "between training and external datasets. Resolve the evaluation protocol before running."
        )
    frame["source_group"] = frame["source_group"].map(find)
    return {
        "method": "SHA256 of source file bytes; HR preferred, LR used only when HR is unavailable",
        "limitation": "Different encodings, crops, and near duplicates are not detected; HR and LR hashes are distinct",
        "n_unique_source_files": len(paths),
        "cross_dataset_shared_sources": shared,
    }


def load_experiment_data(
    config: dict[str, Any], base_dir: Path, *, include_all_rr: bool = False
) -> tuple[pd.DataFrame, dict[str, tuple[list[str], list[str]]], list[str], dict[str, Any]]:
    """Load existing parser labels and raw features without refitting or renormalizing labels."""
    samples = load_datasets(config["datasets"], base_dir=base_dir)
    if not samples:
        raise ValueError("Paper experiments require at least one sample")
    metadata = []
    for sample in samples:
        usage = sample["regressors"]
        if not usage["train"] and not usage["validate"]:
            raise ValueError(f"Inactive dataset cannot participate in paper experiments: {sample['dataset']}")
        metadata.append(
            {
                "sample_id": str(sample["sample_id"]),
                "dataset": str(sample["dataset"]),
                "test_case": str(sample["test_case"]),
                "correlation_group": str(sample["correlation_group"]),
                "score_type": str(sample["score_type"]),
                "score": float(sample["score"]),
                "method": str(sample["method"]),
                "role": "train" if usage["train"] else "external",
                "gt_path": str(sample.get("hr_path") or ""),
                "lr_path": str(sample["lr_path"]),
                "features_root": str(sample["features_root"]),
            }
        )
        if include_all_rr:
            metadata[-1].update(sr_path=str(sample["sr_path"]), declared_scale=sample.get("scale"))
    meta = pd.DataFrame(metadata).sort_values("sample_id").set_index("sample_id")
    if not meta.index.is_unique:
        raise ValueError("Dataset parser returned duplicate sample IDs")
    if not np.isfinite(meta.loc[meta["role"] == "train", "score"]).all():
        raise ValueError("Dataset parser returned nonfinite training target scores")
    if np.isinf(meta["score"]).any():
        raise ValueError("Dataset parser returned infinite target scores")
    frames = []
    files = []
    schema: dict[str, tuple[list[str], list[str]]] | None = None
    scalar_columns = NR_COLUMNS + FR_COLUMNS + STAT_COLUMNS
    scalar_files = [("nr.csv", NR_COLUMNS), ("fr.csv", FR_COLUMNS), ("stats.csv", STAT_COLUMNS)]
    if include_all_rr:
        scalar_columns += SIGLIP_COLUMNS
        scalar_files = [("nr.csv", NR_COLUMNS), ("fr.csv", FR_COLUMNS + GT_COLUMNS),
                        ("stats.csv", STAT_COLUMNS), ("siglip.csv", SIGLIP_COLUMNS)]
    for dataset_name, dataset_meta in meta.groupby("dataset", sort=True):
        roots = dataset_meta["features_root"].unique()
        if len(roots) != 1:
            raise ValueError(f"Dataset {dataset_name} has more than one feature root")
        root = Path(roots[0])
        pieces = [dataset_meta.drop(columns="features_root")]
        for filename, columns in scalar_files:
            values, provenance = _read_block(root / filename, dataset_meta.index, columns=columns)
            pieces.append(values)
            files.append({"dataset": dataset_name, **provenance})
        current_schema = {}
        for family in ("vgg", "resnet"):
            pair = []
            for prefix in (family + "_", "ref_" + family + "_"):
                values, provenance = _read_block(root / f"{prefix[:-1]}.csv", dataset_meta.index, prefix=prefix)
                pair.append(values.columns.tolist())
                pieces.append(values)
                files.append({"dataset": dataset_name, **provenance})
            sr_columns, reference_columns = pair
            if [name[len(family) + 1:] for name in sr_columns] != [
                name[len(family) + 5:] for name in reference_columns
            ]:
                raise ValueError(f"Dataset {dataset_name} has unpaired {family} embedding coordinates")
            current_schema[family] = (sr_columns, reference_columns)
        if schema is not None and current_schema != schema:
            raise ValueError(f"Dataset {dataset_name} has incompatible raw embedding columns")
        schema = current_schema
        frames.append(pd.concat(pieces, axis=1))
    frame = pd.concat(frames, axis=0).sort_index().reset_index()
    audit = _audit_content(frame)
    provenance = {
        "files": files,
        "datasets": frame.groupby(["dataset", "role"], sort=True).size().rename("n_rows").reset_index().to_dict("records"),
        "source_content_audit": audit,
        "label_handling": "Existing dataset parser scores preserved without additional normalization",
        "unscored_external_samples": frame.loc[
            (frame["role"] == "external") & frame["score"].isna(),
            ["sample_id", "dataset", "test_case", "correlation_group"],
        ].to_dict("records"),
        "missing_external_label_policy": "Predict every sample; existing validation_correlations excludes NaN targets and records scored sample counts",
    }
    assert schema is not None
    return frame, schema, scalar_columns, provenance
