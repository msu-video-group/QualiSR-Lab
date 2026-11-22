"""Complete secondary RealSRQ pairs on jointly defined BT groups."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_statistics import (
    BACKENDS,
    BT,
    FOLD,
    ID,
    META,
    METRICS,
    metric_by_group,
    read,
    scene_draws,
)
from experiments.data import require, sha256_file


def pair_keys(name):
    if name.startswith("nr_to_"):
        return ("prior", "nr", "fixed"), ("prior", name.removeprefix("nr_to_"), "fixed")
    if name == "all_rr_minus_nr_ref":
        return ("prior", "nr_ref", "fixed"), ("core", "all_rr", "fixed")
    if name == "nr_ref_resnet50_minus_5":
        return ("prior", "nr_ref", "fixed"), ("core", "nr_ref", "resnet50")
    encoder, _, count, _, baseline = name.split("_")
    require(encoder in ("vgg", "resnet") and baseline == "5", f"Unexpected contrast: {name}")
    return ("pca", f"{encoder}_sr", "pca_5"), ("pca", f"{encoder}_sr", f"pca_{count}")


def load(root, keys, grouped):
    columns = META + ["prediction"]
    data = {}
    for stage in ("core", "pca"):
        names = ["experiment", "model", "track"]
        frame = read(root / stage / "predictions.csv.gz", usecols=columns + FOLD + names)
        for (experiment, model, track), group in frame.loc[frame.dataset.eq(BT)].groupby(names):
            key = (stage, experiment, track, model)
            if key in keys:
                data[key] = group[columns + FOLD]
    for key in sorted(keys):
        stage, experiment, track, model = key
        if stage != "prior":
            continue
        paths = sorted(grouped.glob(f"seed_*/fold_*/{experiment}/{model}/{track}/predictions.csv"))
        require(len(paths) == 15, f"Missing archive folds for {key}")
        parts = []
        for path in paths:
            seed, fold = path.relative_to(grouped).parts[:2]
            part = read(path, usecols=columns)
            parts.append(part.loc[part.dataset.eq(BT)].assign(split_seed=int(seed[5:]), fold=int(fold[5:])))
        data[key] = pd.concat(parts, ignore_index=True)
    require(data.keys() == keys, "Incomplete paired BT inputs")
    return data


def joint_delta(left, right, prediction_columns):
    scores = [metric_by_group(rows.rename(columns={name: "prediction"}))
              for rows in (left, right) for name in prediction_columns]
    scores = np.array(scores)
    valid = np.isfinite(scores).all(axis=(0, 2))
    base, added = np.split(scores[:, valid, :], 2, axis=0)
    return (added - base).mean(axis=0), valid


def paired_bt(data, label, *, keys=None, n_folds=15, n_reps=10000):
    left_key, right_key = pair_keys(label.name) if keys is None else keys
    models = BACKENDS if label.model == "backend_mean" else (label.model,)
    aligned = []
    columns = [f"pred_{model}" for model in models]
    for key in (left_key, right_key):
        side = None
        for model, column in zip(models, columns, strict=True):
            rows = data[(*key, model)].dropna(subset=["score", "prediction"]).sort_values(ID).reset_index(drop=True)
            require(not rows.duplicated(ID).any(), "Duplicate saved BT prediction")
            info = rows[ID + META[1:]]
            if side is None:
                side = info.copy()
            pd.testing.assert_frame_equal(side[info.columns], info)
            side[column] = rows.prediction
        aligned.append(side)
    a, b = aligned
    pd.testing.assert_frame_equal(a[ID + META[1:]], b[ID + META[1:]])
    require(a.groupby("correlation_group").source_group.nunique().eq(1).all(), "Group straddles scenes")
    require(a.groupby("sample_id").size().eq(n_folds).all(), "Missing a repeated prediction")

    collapsed = []
    for arm in (a, b):
        meta = arm[META].drop_duplicates()
        require(meta.sample_id.is_unique, "Metadata changes across folds")
        collapsed.append(meta.merge(arm.groupby("sample_id")[columns].mean(), on="sample_id", validate="one_to_one"))
    effects, keep = joint_delta(*collapsed, columns)
    groups = collapsed[0].groupby("correlation_group").source_group.first()
    _, scenes = pd.factorize(groups, sort=True)
    index = scenes.get_indexer(groups[keep])
    require(keep.any(), "No jointly defined BT comparison group")
    draws = scene_draws(BT, len(scenes), n_reps)[:, index]
    count = draws.sum(axis=1, keepdims=True)
    reps = np.divide(draws @ effects, count, out=np.full((len(draws), 2), np.nan), where=count > 0)
    require(np.isfinite(reps).all(axis=1).sum() >= 0.99 * n_reps, "Insufficient valid BT replicates")
    point = effects.mean(axis=0)
    ci = np.nanquantile(reps, [0.025, 0.975], axis=0)

    folds = []
    for key, group in a.groupby(FOLD):
        delta, valid = joint_delta(group, b.loc[group.index], columns)
        plcc, srcc = delta.mean(axis=0)
        folds.append(dict(zip(FOLD, key, strict=True), plcc=plcc, srcc=srcc, n_common_groups=int(valid.sum())))
    folds = pd.DataFrame(folds)
    result = [dict(metric=m, fold_mean=folds[m].mean(), fold_sd=folds[m].std(), n_folds=folds[m].count(),
                   conditional_point=point[i], ci_low=ci[0, i], ci_high=ci[1, i],
                   valid_reps=int(np.isfinite(reps[:, i]).sum()), n_scenes=len(scenes),
                   n_common_scenes=len(np.unique(index)), n_common_groups=int(keep.sum()),
                   min_fold_common_groups=folds.n_common_groups.min(), max_fold_common_groups=folds.n_common_groups.max())
              for i, m in enumerate(METRICS)]
    return result, folds


def run(root, skipped, output, grouped=None):
    labels = read(skipped, dtype={"table": str}).drop(columns="reason")
    require(labels.population.eq(BT).all() and not labels.duplicated().any(), "Invalid list of BT pairs")
    keys = {(*key, model) for label in labels.itertuples(index=False) for key in pair_keys(label.name)
            for model in (BACKENDS if label.model == "backend_mean" else (label.model,))}
    if labels.empty:
        output.mkdir()
        (output / "status.json").write_text(json.dumps({"status": "complete", "pairs": 0}) + "\n")
        return
    data = load(root, keys, grouped if grouped is not None else root / "prior")
    rows, fold_rows = [], []
    for label in labels.itertuples(index=False):
        summary, folds = paired_bt(data, label)
        meta = label._asdict()
        rows.extend({**meta, **row} for row in summary)
        fold_rows.append(folds.assign(**meta))
    output.mkdir()
    pd.DataFrame(rows).to_csv(output / "bt_common_groups_deltas.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(output / "bt_common_groups_folds.csv", index=False)
    (output / "status.json").write_text(json.dumps(dict(
        status="complete", pairs=len(labels), replicates=10000, skipped_sha256=sha256_file(skipped)
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skipped", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--grouped", type=Path, help="Grouped results directory (default: ROOT/prior)")
    args = parser.parse_args()
    run(args.root, args.skipped, args.output, args.grouped)
