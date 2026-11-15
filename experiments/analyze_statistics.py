"""Scene-bootstrap intervals and paired tests for the QualiSR-Lab paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.data import GT_COLUMNS, NR_COLUMNS, require, sha256_file
from qualisr.regressors import safe_corr

SET, DSR, ISR, BT = "QualiSR-Set120", "dsr-dataset", "isrgenqa", "realsrq"
MOS = (SET, DSR, ISR)
DATASETS = (*MOS, BT)
EXTERNAL_MOS = (DSR, ISR)
MACRO = "macro_mos"
EXTERNAL = "external_mos_macro"
BACKENDS = ("randomforest", "xgb", "catboost")
FOLD = ["split_seed", "fold"]
ID = FOLD + ["sample_id"]
META = ["sample_id", "dataset", "source_group", "correlation_group", "score_type", "score"]
METRICS = ["plcc", "srcc"]
SEED = 20261106


def read(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def json_read(path):
    return json.loads(path.read_text())


def fingerprints(frame, columns):
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def metric_by_group(frame):
    if frame.score_type.iat[0] == "mos":
        return np.array([safe_corr(frame.score, frame.prediction)])
    require(frame.score_type.eq("bradley_terry").all(), "Unknown score type")
    groups = frame.groupby("correlation_group", sort=True)
    codes = groups.ngroup().to_numpy()
    positions = groups.cumcount().to_numpy()
    length = groups.size().to_numpy()
    vectors = []
    for column in ("score", "prediction"):
        values = np.full((len(length), positions.max() + 1), np.nan)
        values[codes, positions] = frame[column].to_numpy()
        vectors.append(values)
    y, pred = vectors
    valid = ((length > 1) &
             np.any(np.isfinite(y) & ~np.isclose(y, y[:, :1]), axis=1) &
             np.any(np.isfinite(pred) & ~np.isclose(pred, pred[:, :1]), axis=1))

    def row_corr(a, b):
        a = np.nan_to_num(a - np.nanmean(a, axis=1, keepdims=True))
        b = np.nan_to_num(b - np.nanmean(b, axis=1, keepdims=True))
        numerator = np.einsum("ij,ij->i", a, b)
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return np.divide(numerator, denominator, out=np.full(len(a), np.nan), where=valid)

    return np.column_stack((row_corr(y, pred), row_corr(
        rankdata(y, axis=1, nan_policy="omit"), rankdata(pred, axis=1, nan_policy="omit")
    )))


def mean_defined(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1], np.nan), where=count > 0)


@lru_cache(maxsize=6)
def scene_draws(dataset, n, n_reps):
    rng = np.random.default_rng(SEED + DATASETS.index(dataset))
    return rng.multinomial(n, np.full(n, 1 / n), size=n_reps).astype(np.int32)


def rank_groups(values):
    order = np.argsort(values, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    return order, starts, sizes


def twice_centered_ranks(weights, total, groups):
    order, starts, sizes = groups
    mass = np.add.reduceat(weights[:, order], starts, axis=1, dtype=np.int32)
    midpoints = 2 * np.cumsum(mass, axis=1, dtype=np.int32) - mass - total[:, None]
    ranks = np.empty_like(weights, dtype=np.int32)
    ranks[:, order] = np.repeat(midpoints, sizes, axis=1)
    return ranks


def moment_corr(x, y, xx, yy, xy, total):
    covariance = xy - x * y / total
    vx = np.maximum(0, xx - x * x / total)
    vy = np.maximum(0, yy - y * y / total)
    denominator = np.sqrt(vx * vy)
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)


def bootstrap(frame, draws, scene_codes):
    if frame.score_type.iat[0] == "bradley_terry":
        # Group correlations are invariant to replication of a whole scene.
        groups = frame.groupby("correlation_group", sort=True).source_group.first()
        source_codes = pd.Series(scene_codes, index=frame.source_group).groupby(level=0).first()
        counts = draws[:, source_codes.loc[groups].to_numpy()]
        values = metric_by_group(frame)
        valid = np.isfinite(values)
        numerator = counts @ np.nan_to_num(values)
        denominator = counts @ valid.astype(float)
        reps = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        return mean_defined(values), reps, valid.sum(axis=0)
    y, pred = (frame[c].to_numpy() for c in ("score", "prediction"))
    orders = [rank_groups(z) for z in (y, pred)]
    a, b = y - y.mean(), pred - pred.mean()
    moments = np.column_stack([np.bincount(scene_codes, weights=v, minlength=draws.shape[1])
                               for v in (a, b, a * a, b * b, a * b, np.ones(len(y)))])
    reps = []
    for start in range(0, len(draws), 200):
        batch = draws[start:start + 200]
        weights = batch[:, scene_codes]
        total = weights.sum(axis=1, dtype=np.int32)
        yr, pr = (twice_centered_ranks(weights, total, group) for group in orders)
        cross, v_y, v_p = [np.einsum("ij,ij,ij->i", weights, x, z, dtype=np.float64)
                            for x, z in ((yr, pr), (yr, yr), (pr, pr))]
        denominator = np.sqrt(v_y * v_p)
        srcc = np.divide(cross, denominator, out=np.full_like(cross, np.nan), where=denominator > 0)
        plcc = np.where(denominator > 0, moment_corr(*(batch @ moments).T), np.nan)
        reps.append(np.column_stack((plcc, srcc)))
    return np.array(safe_corr(y, pred)), np.concatenate(reps), np.full(2, 1)


@dataclass
class Estimate:
    point: np.ndarray
    reps: np.ndarray
    folds: pd.DataFrame
    # Coverage is stored separately for every constituent dataset.
    coverage: dict
    counts: dict
    bt_support: tuple = ()


def estimate(frame, n_reps):
    require(not frame.duplicated(ID).any(), "Duplicate evaluation row")
    require(frame[META[:-1]].notna().all().all(), "Missing evaluation metadata")
    require(not np.isinf(frame[["score", "prediction"]]).any().any(), "Infinite score or prediction")
    output = {}
    for dataset, group in frame.groupby("dataset"):
        group = group.dropna(subset=["score", "prediction"]).sort_values(ID)
        if group.empty:
            continue
        is_bt = group.score_type.iat[0] == "bradley_terry"
        require(group.score_type.nunique() == 1, f"Mixed score types in {dataset}")
        metadata = group[META].drop_duplicates()
        require(metadata.sample_id.is_unique, f"Changed metadata across folds in {dataset}")
        if is_bt:
            require(group.groupby("correlation_group").source_group.nunique().eq(1).all(),
                    "A BT group straddles scenes")
        repetitions = group.groupby("sample_id").size()
        # Missing direct scores have already been removed, consistently across folds.
        require(repetitions.nunique() == 1, f"Inconsistent prediction repetition in {dataset}")
        require(not group.duplicated(["split_seed", "fold", "sample_id"]).any(), "Duplicate sample")
        # Mean of fixed forecasts, never a change of targets or ranks averaged across folds.
        collapsed = metadata.merge(group.groupby("sample_id").prediction.mean(), on="sample_id",
                                   validate="one_to_one").sort_values("sample_id")
        scene_codes, scenes = pd.factorize(collapsed.source_group, sort=True)
        n = len(scenes)
        draws = scene_draws(dataset, n, n_reps)
        point, reps, defined = bootstrap(collapsed, draws, scene_codes)
        folds = {}
        group_coverage = []
        for key, rows in group.groupby(FOLD, sort=True):
            values = metric_by_group(rows)
            folds[key] = mean_defined(values)
            if is_bt:
                group_coverage.append(list(zip(sorted(rows.correlation_group.unique()),
                                               np.isfinite(values).tolist(), strict=True)))
        fold_frame = pd.DataFrame.from_dict(folds, orient="index", columns=METRICS)
        fold_frame.index = pd.MultiIndex.from_tuples(fold_frame.index, names=FOLD)
        n_groups = collapsed.correlation_group.nunique() if is_bt else 1
        bt_support = ()
        if is_bt:
            point_groups = metric_by_group(collapsed)
            support = [list(zip(sorted(collapsed.correlation_group.unique()),
                                np.isfinite(point_groups).tolist(), strict=True)), group_coverage]
            bt_support = (hashlib.sha256(json.dumps(support).encode()).hexdigest(),)
        output[dataset] = Estimate(
            point, reps, fold_frame,
            {dataset: fingerprints(group, ID + META[1:])},
            {dataset: dict(n_scenes=n, n_samples=len(collapsed), n_groups=n_groups,
                           defined_bt_groups=defined.tolist() if is_bt else [],
                           all_fold_bt_groups_defined=all(all(all(defined) for _, defined in fold) for fold in group_coverage),
                           max_scene_fraction=float(np.bincount(scene_codes).max() / len(collapsed)),
                           predictions_per_sample=int(repetitions.iat[0]))},
            bt_support,
        )
    return output


def combine(terms, *, average_backends=False):
    """Combine already aligned dataset/backend correlations or matched differences."""
    first = terms[0][1]
    coverage, counts = {}, {}
    for _, item in terms:
        require(item.folds.index.equals(first.folds.index), "Unequal fold keys")
        for dataset, key in item.coverage.items():
            require(dataset not in coverage or coverage[dataset] == key,
                    f"Comparison has different evaluation rows in {dataset}")
            coverage[dataset] = key
            counts[dataset] = item.counts[dataset]
    signatures = [x.bt_support for _, x in terms if x.bt_support]
    if average_backends:
        support = tuple(signature for sig in signatures for signature in sig)
    else:
        require(not signatures or all(sig == signatures[0] for sig in signatures),
                "Matched BT contrast has unequal defined comparison groups")
        support = signatures[0] if signatures else ()
    return Estimate(sum(w * x.point for w, x in terms), sum(w * x.reps for w, x in terms),
                    sum(w * x.folds for w, x in terms), coverage, counts, support)


def macro(tables, datasets):
    return combine([(1 / len(datasets), tables[d]) for d in datasets])


def add_macros(values):
    if set(MOS) <= values.keys():
        values[MACRO] = macro(values, MOS)
    if set(EXTERNAL_MOS) <= values.keys():
        values[EXTERNAL] = macro(values, EXTERNAL_MOS)
    return values


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def describe(item, tested=False):
    rows = []
    for i, metric in enumerate(METRICS):
        reps = item.reps[:, i]
        valid = reps[np.isfinite(reps)]
        point = item.point[i]
        enough = len(valid) >= 0.99 * len(reps) and np.isfinite(point)
        lo, hi = np.quantile(valid, [0.025, 0.975]) if enough else (np.nan, np.nan)
        if tested:
            require(enough and all(c["n_scenes"] > 1 for c in item.counts.values()),
                    "Primary test has insufficient valid resamples or fewer than two scenes")
        p = (1 + (np.abs(valid - point) >= abs(point)).sum()) / (1 + len(valid)) if tested else np.nan
        folds = item.folds[metric]
        rows.append(dict(metric=metric, fold_mean=folds.mean(), fold_sd=folds.std(), n_folds=folds.count(),
                         conditional_point=point, ci_low=lo, ci_high=hi, p_raw=p, valid_reps=len(valid),
                         n_scenes=sum(c["n_scenes"] for c in item.counts.values()),
                         n_samples=sum(c["n_samples"] for c in item.counts.values()),
                         max_scene_fraction=max(c["max_scene_fraction"] for c in item.counts.values()),
                         coverage=json.dumps(item.counts, sort_keys=True)))
    return rows


class Analysis:
    def __init__(self, n_reps):
        self.n_reps = n_reps
        self.cells = []
        self.contrasts = []
        self.folds = []
        self.primary = []
        self.skipped = []

    def record(self, dest, tables, **label):
        for dataset, item in tables.items():
            self.folds.append(item.folds.reset_index().assign(population=dataset, **label))
            dest.extend(dict(population=dataset, **label, **row) for row in describe(item))

    def cell(self, frame, table, name, model):
        values = add_macros(estimate(frame, self.n_reps))
        self.record(self.cells, values, table=table, name=name, model=model, kind="score")
        return values

    def mean_backend(self, panels, table, name):
        keys = set(panels[0])
        require(all(p.keys() == keys for p in panels), "Backend populations differ")
        values = {d: combine([(1 / len(panels), p[d]) for p in panels], average_backends=True) for d in sorted(keys)}
        self.record(self.cells, values, table=table, name=name, model="backend_mean", kind="score")
        return values

    def pair(self, left, right, table, name, model, primary=None):
        values = {}
        for d in left.keys() & right.keys():
            try:
                values[d] = combine([(-1, left[d]), (1, right[d])])
            except ValueError as error:
                if d != BT or not str(error).startswith("Matched BT contrast"):
                    raise
                self.skipped.append(dict(table=table, name=name, model=model, population=d, reason=str(error)))
        values = dict(sorted(values.items()))
        self.record(self.contrasts, values, table=table, name=name, model=model, kind="delta")
        if primary is not None:
            self.test(values[primary], name, primary, table, model)
        return values

    def test(self, item, name, population, table, model):
        self.primary.extend(dict(name=name, population=population, table=table, model=model, **row)
                            for row in describe(item, tested=True))

    def write(self, path, *, expected_primary=12):
        primary = pd.DataFrame(self.primary)
        require(len(primary) == expected_primary, "Incomplete primary contrasts")
        if expected_primary:
            require(primary.p_raw.notna().all(), "Undefined primary test")
            if expected_primary == 12:
                primary["p_holm"] = holm(primary.p_raw)
            primary.to_csv(path / "primary_tests.csv", index=False)
        pd.DataFrame(self.cells).to_csv(path / "correlations.csv", index=False)
        pd.DataFrame(self.contrasts).to_csv(path / "paired_deltas.csv", index=False)
        pd.concat(self.folds, ignore_index=True).to_csv(path / "fold_metrics.csv.gz", index=False)
        pd.DataFrame(self.skipped, columns=["table", "name", "model", "population", "reason"]).to_csv(
            path / "undefined_bt_pairs.csv", index=False
        )


def matrix(analysis, root, table, stage, names=None):
    groups = {}
    if stage == "prior":
        files = sorted(root.glob("seed_*/fold_*/*/*/fixed/predictions.csv"))
        selected = {}
        for path in files:
            seed, fold, experiment, model, track, _ = path.relative_to(root).parts
            if names is None or experiment in names:
                key = (experiment, model, track)
                selected.setdefault(key, []).append((int(seed[5:]), int(fold[5:]), path))
        for key, paths in selected.items():
            require(len(paths) == 15, f"Incomplete fold matrix for {key}")
            groups[key] = pd.concat([read(path, usecols=META + ["prediction"]).assign(split_seed=seed, fold=fold)
                                     for seed, fold, path in paths], ignore_index=True)
    else:
        keys = ["experiment", "model", "track"]
        frame = read(root / "predictions.csv.gz", usecols=META + FOLD + keys + ["prediction"])
        if names is not None:
            frame = frame[frame.experiment.isin(names)]
        groups = dict(tuple(frame.groupby(keys, sort=True)))
    require(groups, f"No prediction sets found in {root}")
    values = {}
    for index, ((experiment, model, track), group) in enumerate(groups.items(), start=1):
        name = f"{experiment}/{track}"
        values[name, model] = analysis.cell(group, table, name, model)
        if index % 12 == 0 or index == len(groups):
            print(f"  {stage}: {index}/{len(groups)} model configurations", flush=True)
    for name in sorted({key[0] for key in values}):
        if all((name, backend) in values for backend in BACKENDS):
            values[name, "backend_mean"] = analysis.mean_backend(
                [values[name, backend] for backend in BACKENDS], table, name
            )
    return values


def compare_models(analysis, values, pairs, table, primary=None):
    for base, added, name in pairs:
        for model in (*BACKENDS, "backend_mean", "ridge"):
            if (base, model) in values and (added, model) in values:
                analysis.pair(values[base, model], values[added, model], table, name, model,
                              MACRO if model == "backend_mean" and name == primary else None)


def run_pca(analysis, root):
    print("Table 5: PCA", flush=True)
    values = matrix(analysis, root, "5", "pca")
    pairs = [(f"{encoder}_sr/pca_5", f"{encoder}_sr/pca_{count}", f"{encoder}_pca_{count}_minus_5")
             for encoder in ("vgg", "resnet") for count in (2, 10, 25, 50, 75, 96)]
    compare_models(analysis, values, pairs, "5")


def run_prior(analysis, root):
    print("Grouped feature comparisons with fixed model settings", flush=True)
    values = matrix(analysis, root, "6,9,10,15", "prior")
    comparisons = [
        ("nr", name, f"nr_to_{name}")
        for name in ("nr_ref", "nr_fr", "nr_diff", "nr_stats", "nr_all", "nr_stats_all",
                     "nr_fr_bicubic", "nr_fr_span")
    ] + [
        ("fr_rlfn", "fr_bicubic", "fr_only_bicubic_minus_rlfn"),
        ("fr_rlfn", "fr_span", "fr_only_span_minus_rlfn"),
        ("nr_fr", "nr_fr_bicubic", "nr_fr_bicubic_minus_rlfn"),
        ("nr_fr", "nr_fr_span", "nr_fr_span_minus_rlfn"),
    ]
    compare_models(analysis, values, [(f"{a}/fixed", f"{b}/fixed", name) for a, b, name in comparisons],
                   "6", "nr_to_nr_ref")
    compare_models(analysis, values, [
        ("full_without_stats/fixed", "full_with_stats/fixed", "candidate_add_wasd"),
        ("full_with_stats/fixed", "full_all_statistics/fixed", "candidate_ten_minus_four_wasd"),
        ("nr_stats/fixed", "nr_stats_all/fixed", "nr_ten_minus_four_wasd"),
    ], "10,15", "candidate_add_wasd")
    compare_models(analysis, values, [
        ("full_without_qalign/fixed", "full_with_stats/fixed", "candidate_add_qalign"),
        ("full_with_stats/fixed", "nr_all/fixed", "ref_diff_minus_sr_diff"),
    ], "15")
    return values


def run_core(analysis, root, prior):
    print("Tables 6, 9, 10, 15: All RR matrix", flush=True)
    values = matrix(analysis, root, "6,9,10,15", "core")
    compare_models(analysis, values, [
        ("all_rr_no_wasd/fixed", "all_rr/fixed", "all_rr_add_wasd"),
    ], "6,9,10,15", "all_rr_add_wasd")
    compare_models(analysis, values, [
        ("all_rr_no_fr/fixed", "all_rr/fixed", "all_rr_add_fr"),
        ("all_rr_no_qalign/fixed", "all_rr/fixed", "all_rr_add_qalign"),
        ("all_rr_no_siglip/fixed", "all_rr/fixed", "all_rr_add_siglip"),
        ("all_rr_hr_common/fixed", "true_hr_oracle/fixed", "all_rr_add_hr"),
        ("all_rr/fixed", "all_rr/standard", "all_rr_standard_minus_minmax"),
        ("all_rr/fixed", "all_rr/tuned", "all_rr_tuned_minus_fixed"),
        ("all_rr/fixed", "all_rr/resnet50", "all_rr_resnet50_minus_5"),
    ], "6,9,15")
    compare_models(analysis, {**prior, **values}, [
        (f"{key}/fixed", f"{key}/resnet50", f"{key}_resnet50_minus_5")
        for key in ("nr_ref", "nr_all", "nr_diff")
    ] + [
        ("nr_ref/fixed", "all_rr/fixed", "all_rr_minus_nr_ref"),
        ("full_with_stats/fixed", "all_rr/fixed", "all_rr_minus_candidate"),
    ], "6,9,15")
    return values


def join_fits(root, select):
    fits = read(root / "fits.csv")
    chosen = fits.loc[select(fits)]
    raw = read(root / "predictions.csv.gz")
    samples = read(root / "samples.csv", usecols=META)
    pred = raw.merge(chosen, on="fit_id", validate="many_to_one")
    joined = pred.merge(samples, on="sample_id", validate="many_to_one")
    require(len(joined) == len(pred), "Unknown evaluation IDs")
    return joined


def run_factorial(analysis, root):
    print("Table 13: matched factorial", flush=True)
    rows = join_fits(root, lambda f: f.track.eq("impact"))
    cells = {}
    for (case, spec), group in rows.groupby(["case", "spec"]):
        require(group[FOLD].drop_duplicates().shape[0] == 15, "Incomplete matched factorial")
        cells[case, spec] = analysis.cell(group, "13", f"{case}/{spec}", "randomforest")
    contrasts = []
    for suffix, spec in (("no_wasd", "baseline_qalign"), ("wasd", "baseline_qalign_stats")):
        contrasts.append(analysis.pair(cells["dsr_only", spec], cells["dsr_plus_set120", spec],
                                       "13", f"add_set120_{suffix}", "randomforest", MACRO))
    analysis.pair(*contrasts, "13", "set120_by_wasd_interaction", "randomforest")
    for case in ("dsr_only", "dsr_plus_set120"):
        analysis.pair(cells[case, "baseline_qalign"], cells[case, "baseline_qalign_stats"],
                      "13", f"{case}_add_wasd", "randomforest")


def run_musiq(analysis, root):
    print("Transfer: paired fine-tuning", flush=True)
    samples = read(root / "samples.csv", usecols=META)
    raw = read(root / "predictions.csv.gz")
    joined = raw.merge(samples, on="sample_id", validate="many_to_one")
    require(len(joined) == len(raw) and joined[FOLD].drop_duplicates().shape[0] == 15,
            "Incomplete MUSIQ evaluation")
    names = ("qualisr", "musiq_finetuned", "musiq_direct")
    values = {name: analysis.cell(joined.rename(columns={name: "prediction"}), "transfer", name, name)
              for name in names}
    tune = values["musiq_finetuned"]
    rf = values["qualisr"]
    analysis.pair(rf, tune, "transfer", "finetuned_minus_qualisr", "paired")
    analysis.pair(values["musiq_direct"], tune, "transfer", "finetuned_minus_direct", "paired")
    gaps = {name: {"mos_gap": combine([(1, value[SET]), (-1, value[EXTERNAL])])}
            for name, value in values.items()}
    for name, gap in gaps.items():
        analysis.record(analysis.cells, gap, table="transfer", name=f"{name}_gap", model=name, kind="score")
    analysis.pair(gaps["qualisr"], gaps["musiq_finetuned"], "transfer",
                  "finetuned_minus_qualisr_gap", "paired", "mos_gap")
    common = joined.loc[joined.qalign_direct.notna()]
    qalign = analysis.cell(common.rename(columns={"qalign_direct": "prediction"}),
                           "transfer", "qalign_direct_common", "qalign_direct")
    for name in names:
        on_common = (values[name] if len(joined) == len(common) else
                     analysis.cell(common.rename(columns={name: "prediction"}), "transfer",
                                   f"{name}_on_qalign_common", name))
        analysis.pair(qalign, on_common, "transfer", f"{name}_minus_qalign_common", "paired")


def run_direct(analysis, root, core_values):
    print("Table 9: direct scores", flush=True)
    core = read(root / "predictions.csv.gz",
                usecols=META + FOLD + ["prediction", "experiment", "model", "track"])
    core = core.loc[core.experiment.eq("all_rr") & core.track.eq("fixed")]
    inputs = json_read(root / "data_provenance.json")["files"]
    selected = [(Path(item["path"]), item["sha256"]) for item in inputs
                if Path(item["path"]).name in ("nr.csv", "fr.csv")]
    require(len(selected) == 8, "Expected two direct-metric files per benchmark")
    frames = {"nr.csv": [], "fr.csv": []}
    for path, digest in selected:
        require(sha256_file(path) == digest, f"Changed source metric file: {path}")
        columns = NR_COLUMNS if path.name == "nr.csv" else GT_COLUMNS
        frame = read(path, usecols=["sample_id", *columns]).set_index("sample_id")
        require(frame.index.is_unique, f"Duplicate direct score IDs in {path}")
        frames[path.name].append(frame)
    raw = pd.concat([pd.concat(parts) for parts in frames.values()], axis=1, verify_integrity=True)
    require(raw.index.is_unique and set(core.sample_id) <= set(raw.index), "Direct score coverage differs")
    for metric in (*NR_COLUMNS, *GT_COLUMNS):
        direction = -1 if metric in ("lpips-vgg_gt", "stlpips-vgg_gt", "pieapp_gt") else 1
        # Drop missing scores explicitly and compare against All RR on these same rows.
        predicted = core.sample_id.map(raw[metric]) * direction
        mask = np.isfinite(predicted)
        subset = core.loc[mask]
        frame = subset.loc[subset.model.eq(BACKENDS[0]), META + FOLD].assign(prediction=predicted)
        direct = analysis.cell(frame, "9", metric, "direct")
        if mask.all():
            rr = core_values["all_rr/fixed", "backend_mean"]
        else:
            label = f"all_rr_on_{metric}_coverage"
            models = [analysis.cell(subset.loc[subset.model.eq(model)], "9", label, model) for model in BACKENDS]
            rr = analysis.mean_backend(models, "9", label)
        # Repeat the same direct endpoint for each paired backend.
        direct = {d: combine([(1 / len(BACKENDS), v)] * len(BACKENDS), average_backends=True)
                  for d, v in direct.items()}
        analysis.pair(direct, rr, "9", f"all_rr_minus_{metric}", "backend_mean")


def run_model_seeds(root, output):
    rows = join_fits(root, lambda f: f.track.eq("robustness") & f.case.eq("released") &
                     f.split_seed.eq(42) & f.spec.isin(["nr", "nr_all"]))
    results, keys = [], {}
    for (name, seed, fold, dataset), group in rows.groupby(["spec", "model_seed", "fold", "dataset"]):
        group = group.dropna(subset=["score", "prediction"]).sort_values("sample_id")
        ident = (name, fold, dataset)
        fingerprint = fingerprints(group, META)
        require(keys.setdefault(ident, fingerprint) == fingerprint, "Model seed changed evaluation rows")
        p, s = mean_defined(metric_by_group(group))
        results.append(dict(spec=name, model_seed=int(seed), split_seed=42, fold=fold,
                            dataset=dataset, plcc=p, srcc=s, n_scenes=group.source_group.nunique()))
    folds = pd.DataFrame(results)
    require(len(folds) == 2 * 3 * 5 * 4 and set(folds.model_seed) == {42, 43, 44}, "Incomplete model-seed runs")
    macro_values = folds[folds.dataset.isin(MOS)].groupby(["spec", "model_seed", "split_seed", "fold"], as_index=False)
    folds = pd.concat([folds, macro_values.agg(plcc=("plcc", "mean"), srcc=("srcc", "mean"),
                                             n_scenes=("n_scenes", "sum")).assign(dataset=MACRO)])
    folds = folds.melt(id_vars=["spec", "model_seed", "split_seed", "fold", "dataset", "n_scenes"],
                       value_vars=METRICS, var_name="metric", value_name="correlation")
    keys = ["spec", "split_seed", "fold", "dataset", "metric"]
    base = folds.loc[folds.model_seed.eq(42), keys + ["correlation"]].rename(columns={"correlation": "seed42"})
    folds = folds.merge(base, on=keys, validate="many_to_one")
    folds["delta_from_42"] = folds.correlation - folds.seed42
    folds.to_csv(output / "model_seed_folds.csv", index=False)
    folds.groupby(["spec", "model_seed", "dataset", "metric"]).agg(
        mean=("correlation", "mean"), sd=("correlation", "std"), n_folds=("correlation", "count"),
        mean_delta_from_42=("delta_from_42", "mean"), max_abs_delta_from_42=("delta_from_42", lambda x: x.abs().max())
    ).reset_index().to_csv(output / "model_seed_summary.csv", index=False)


def main(args):
    require(args.output.parent.is_dir(), "Output parent must exist")
    args.output.mkdir()
    analysis = Analysis(1_000 if args.pilot else 10_000)
    stages = [(args.core, args.core_verification)]
    if not args.pilot:
        stages += [(args.pca, args.pca_verification),
                   (args.base, args.base / "result_verification.json")]
    inputs = {}
    for root, verification in stages:
        check = json_read(verification)
        require(check["status"] == "passed", f"Input not verified: {root}")
        inputs[str(root)] = {"verification": str(verification), "verification_sha256": sha256_file(verification),
                             "run_sha256": sha256_file(root / "run.json")}
        predictions = root / "predictions.csv.gz"
        if predictions.exists():
            inputs[str(root)]["predictions_sha256"] = sha256_file(predictions)
    if args.pilot:
        values = matrix(analysis, args.core, "6,9,10,15", "core",
                        names=("all_rr", "all_rr_no_wasd"))
        compare_models(analysis, values, [
            ("all_rr_no_wasd/fixed", "all_rr/fixed", "all_rr_add_wasd"),
        ], "6,9,10,15")
        actual = pd.concat(analysis.folds, ignore_index=True)
        actual = actual.loc[actual.kind.eq("score") & actual.model.isin(BACKENDS) &
                            actual.population.isin(DATASETS)]
        archived = read(args.core / "metrics.csv").assign(
            name=lambda x: x.experiment + "/" + x.track
        ).rename(columns={"dataset": "population"})
        cols = FOLD + ["name", "model", "population"]
        match = actual.merge(archived, on=cols, validate="one_to_one", suffixes=("_computed", "_archived"))
        require(len(match) == len(actual) and len(match) > 0, "Missing pilot metric")
        for metric in METRICS:
            np.testing.assert_allclose(match[f"{metric}_computed"], match[f"{metric}_archived"], atol=1e-12)
    else:
        base = run_prior(analysis, args.base)
        core = run_core(analysis, args.core, base)
        run_direct(analysis, args.core, core)
        run_pca(analysis, args.pca)
    analysis.write(args.output, expected_primary=0 if args.pilot else 6)
    (args.output / "status.json").write_text(json.dumps(
        dict(status="complete", pilot=args.pilot, bootstrap_seed=SEED, replicates=analysis.n_reps,
             inputs=inputs, correlation_rows=len(analysis.cells), contrast_rows=len(analysis.contrasts),
             undefined_bt_pairs=len(analysis.skipped)), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "core", "pca", "core-verification", "pca-verification"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    main(parser.parse_args())
