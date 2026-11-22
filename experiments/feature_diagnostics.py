"""Audit raw features, compare FR metrics by SR method and check detector training overlap."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from experiments.data import FR_METRICS, _read_block, sha256_file
from experiments.run_grouped import write_json
from experiments.transfer import dataset_method
from experiments.verify_grouped import read_results_csv
from qualisr.datasets import load_datasets
from qualisr.references import generate_bicubic
from qualisr.regressors import safe_corr, validation_correlations

COMPACT = {
    "musiq": "NR", "arniqa": "NR", "unique": "NR", "paq2piq": "NR",
    "lpips-vgg_rlfn": "pseudo-FR", "ahiq_rlfn": "pseudo-FR",
    "vgg_diff_000": "VGG Diff", "resnet_diff_000": "ResNet Diff",
    "max": "WASD", "median": "WASD", "std": "WASD", "area00": "WASD",
}


SPECS = (
    "fr_bicubic", "fr_rlfn", "fr_span", "nr", "nr_fr_bicubic", "nr_fr", "nr_fr_span",
)


def compact_summary(prior, output):
    records = []
    for folder in sorted(prior.glob("seed_*/fold_*/compact_selection/ridge")):
        history = read_results_csv(folder / "inner_selection.csv")
        selected = history.loc[history.selected]
        assert len(selected) == 1
        assert len(history) == 24
        assert set(history.loc[history.n_features == 12, "features"].iloc[0].split("|")) == set(COMPACT)
        chosen = selected.iloc[0]
        columns = json.loads((folder / "parameters.json").read_text())["columns"]
        assert columns == chosen.features.split("|") and len(columns) == chosen.n_features
        assert set(columns) <= set(COMPACT)
        best = min(history.itertuples(), key=lambda row: (-row.mean_srcc, row.n_features,
                                                          row.direction != "forward", row.features))
        assert best.selected
        entry = {"split_seed": int(folder.parts[-4].split("_")[-1]), "fold": int(folder.parts[-3].split("_")[-1]),
                 "n_features": len(columns), "direction": chosen.direction, "columns": chosen.features,
                 "mean_inner_srcc": chosen.mean_srcc}
        entry.update({family: sum(COMPACT[col] == family for col in columns) for family in set(COMPACT.values())})
        records.append(entry)
    table = pd.DataFrame(records).sort_values(["split_seed", "fold"])
    assert len(table) == 15 and not table.duplicated(["split_seed", "fold"]).any()
    table.to_csv(output / "compact_subsets.csv", index=False)
    families = table[sorted(set(COMPACT.values()))]
    pd.DataFrame({"folds_included": families.gt(0).sum(), "mean_n_features": families.mean(),
                  "fraction_included": families.gt(0).mean()}).to_csv(output / "compact_families.csv")


def audit_inputs(config_path, prior, output):
    config = json.loads(config_path.read_text())
    samples = pd.DataFrame(load_datasets(config["datasets"], base_dir=config_path.parent)).set_index("sample_id")
    manifest = read_results_csv(prior / "samples.csv").set_index("sample_id")
    pd.testing.assert_series_equal(samples.loc[manifest.index, "score"], manifest.score, check_exact=True)
    sizes = {}

    def size(path):
        if path not in sizes:
            with Image.open(path) as img:
                sizes[path] = img.size
        return sizes[path]

    geometry = []
    for sample in samples.itertuples():
        lw, lh = size(sample.lr_path)
        sw, sh = size(sample.sr_path)
        nominal = round((sw / lw + sh / lh) / 2)
        # Allow up to scale-1 output pixels lost when LR is generated on an integer grid.
        scale = nominal if nominal > 0 and max(abs(sw - nominal * lw), abs(sh - nominal * lh)) < nominal else None
        declared = sample.scale
        if sample.dataset == "isrgenqa":
            declared = int(re.search(r"x(\d+)$", sample.test_case)[1])
        elif sample.dataset == "realsrq":
            declared = int(re.search(r"_LR(\d+)$", sample.correlation_group)[1])
        elif sample.dataset == "QualiSR-Set120":
            declared = 4
        geometry.append({"sample_id": sample.Index, "dataset": sample.dataset, "test_case": sample.test_case,
                         "lr_width": lw, "lr_height": lh, "sr_width": sw, "sr_height": sh,
                         "geometry_scale": scale, "declared_scale": declared})
    geom = pd.DataFrame(geometry).set_index("sample_id").join(manifest.source_group)
    geom.to_csv(output / "geometry.csv", index=True)
    geom.groupby(["dataset", "declared_scale", "geometry_scale"], dropna=False).agg(
        images=("source_group", "size"), sources=("source_group", "nunique")
    ).to_csv(output / "scale_coverage.csv")
    dsr = geom.loc[geom.dataset.eq("dsr-dataset")]
    common = set(dsr.loc[dsr.declared_scale.eq(2), "source_group"]) & set(
        dsr.loc[dsr.declared_scale.eq(4), "source_group"]
    )
    write_json(output / "scale_overlap.json", {"DISRQAD_scenes_at_both_declared_scales": len(common)})
    for scale, part in geom.dropna(subset=["geometry_scale"]).groupby("geometry_scale"):
        row = part.iloc[0]
        example = output / f"bicubic_x{int(scale)}.png"
        generate_bicubic(Path(samples.loc[row.name, "lr_path"]), (int(row.sr_width), int(row.sr_height)), example)
        assert size(example) == (row.sr_width, row.sr_height)

    target = manifest.loc[manifest.dataset.eq("QualiSR-Set120")]
    feature_root = Path(samples.loc[target.index[0], "features_root"])
    ref, provenance = _read_block(feature_root / "ref_vgg.csv", target.index, prefix="ref_vgg_")
    by_scene = ref.groupby(target.source_group).mean()
    assert len(by_scene) == 40
    pca = PCA(n_components=2)
    coords = pd.DataFrame(pca.fit_transform(by_scene), index=by_scene.index, columns=["PC1", "PC2"])
    coords.to_csv(output / "set120_source_pca.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(coords.PC1, coords.PC2, s=18, color="#27608c", alpha=0.8)
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
           title="QualiSR-Set120 source scenes\nVGG embeddings of bicubic references")
    fig.tight_layout()
    fig.savefig(output / "set120_source_pca.pdf")
    plt.close(fig)
    write_json(output / "set120_source_pca.json", {
        "source": provenance, "projection": "Mean per source of cached bicubic reference embeddings",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    })

    direct = []
    for dataset, part in manifest.groupby("dataset"):
        path = Path(samples.loc[part.index[0], "features_root"]) / "fr.csv"
        columns = [f"{metric}_{reference}" for metric in FR_METRICS for reference in ("gt", "bicubic", "rlfn", "span")]
        values, _ = _read_block(path, part.index, columns=columns)
        for metric in FR_METRICS:
            refs = [f"{metric}_{reference}" for reference in ("gt", "bicubic", "rlfn", "span")]
            rows = part.loc[np.isfinite(values[refs]).all(axis=1) & np.isfinite(part.score)].copy()
            for column in refs:
                for method, subset in [("ALL", rows), *(
                    list(rows.groupby(dataset_method(rows))) if len(rows) and rows.score_type.iat[0] == "mos" else []
                )]:
                    pred = values.loc[subset.index, column]
                    if len(subset) and method == "ALL":
                        p, s, _, _ = validation_correlations(subset.score, pred, subset)
                    else:
                        p, s = safe_corr(subset.score, pred)
                    direction = 1 if metric in ("psnr", "ssim", "ahiq") else -1
                    direct.append({"dataset": dataset, "method": method, "column": column, "n_samples": len(subset),
                                   "n_sources": subset.source_group.nunique(), "plcc_raw": p, "srcc_raw": s,
                                   "plcc_quality_oriented": direction * p, "srcc_quality_oriented": direction * s})
    pd.DataFrame(direct).to_csv(output / "direct_fr_common_hr.csv", index=False)


def fr_by_method(prior, output):
    rows = []
    for folder in sorted(prior.glob("seed_*/fold_*")):
        seed = int(folder.parent.name.split("_")[-1])
        fold = int(folder.name.split("_")[-1])
        for spec in SPECS:
            for model in ("randomforest", "xgb", "catboost"):
                pred = read_results_csv(folder / spec / model / "fixed/predictions.csv")
                for dataset, part in pred.loc[pred.score_type.eq("mos")].groupby("dataset"):
                    for method, group in part.groupby(dataset_method(part)):
                        n_sources = group.source_group.nunique()
                        if n_sources >= 3:
                            plcc, srcc = safe_corr(group.score, group.prediction)
                            rows.append({"split_seed": seed, "fold": fold, "spec": spec, "model": model,
                                         "dataset": dataset, "method": method, "n_sources": n_sources,
                                         "n_images": len(group), "plcc": plcc, "srcc": srcc})
    table = pd.DataFrame(rows)
    assert len(table[["split_seed", "fold"]].drop_duplicates()) == 15
    table.to_csv(output, index=False)


def wasd_overlap(args):
    iq = pd.read_csv(args.samples, keep_default_na=False)
    sets = {
        "openimages_train": set(json.loads(args.train.read_text())),
        "openimages_validation": set(json.loads(args.validation.read_text())),
        "clean_validation": set(pd.read_csv(args.clean, dtype={"folder": str}).folder),
    }
    hashes = {}
    metadata = []
    comparisons = []
    for name, ids in sets.items():
        assert all(re.fullmatch(r"[0-9a-f]{16}", ident) for ident in ids)
        available = set()
        for ident in sorted(ids):
            hr = args.openimages / ident / f"{ident}.png"
            if hr.is_file():
                if ident not in hashes:
                    hashes[ident] = sha256_file(hr)
                available.add(hashes[ident])
        metadata.append({"split": name, "manifest_ids": len(ids),
                         "hashed_hr_files": sum(ident in hashes for ident in ids)})
        for dataset, rows in iq.groupby("dataset"):
            records = rows[["test_case", "gt_path", "content_kind", "content_sha256", "source_group"]].copy()
            stems = records.gt_path.map(lambda path: Path(path).stem.lower())
            id_match = records.test_case.str.lower().isin(ids) | stems.isin(ids)
            hr_hash = records.content_kind.eq("hr") & records.content_sha256.isin(available)
            comparisons.append({
                "split": name, "dataset": dataset, "iqa_sources": records.source_group.nunique(),
                "source_id_matches": records.loc[id_match, "source_group"].nunique(),
                "exact_hr_sha256_matches": records.loc[hr_hash, "source_group"].nunique(),
            })
    args.output.mkdir()
    pd.DataFrame(comparisons).to_csv(args.output / "wasd_overlap.csv", index=False)
    write_json(args.output / "wasd_audit.json", {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split_manifest_sha256": {name: sha256_file(getattr(args, name))
                                  for name in ("train", "validation", "clean")},
        "config_sha256": sha256_file(args.config),
        "splits": metadata,
        "train_validation_overlap": len(sets["openimages_train"] & sets["openimages_validation"]),
        "scope": "Current files named by a matching config and training log; exact ID and file-byte matches only.",
        "limits": "Historical split-file hashes and pretrained encoder exposure are not known. Crops, reencodings and near duplicates can evade this comparison.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs", help="Audit feature caches and source geometry")
    methods = commands.add_parser("fr-methods", help="Summarize fixed FR fits by SR method")
    overlap = commands.add_parser("wasd-overlap", help="Check WASD split manifests against IQA sources")
    for cmd in (inputs, methods):
        cmd.add_argument("--prior", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    inputs.add_argument("--config", type=Path, required=True)
    for name in ("samples", "train", "validation", "clean", "openimages", "checkpoint", "config", "output"):
        overlap.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inputs":
        args.output.mkdir()
        compact_summary(args.prior, args.output)
        audit_inputs(args.config, args.prior, args.output)
        write_json(args.output / "status.json", {"status": "complete"})
    elif args.command == "fr-methods":
        fr_by_method(args.prior, args.output)
    else:
        wasd_overlap(args)


if __name__ == "__main__":
    main()
