"""Verify the CPU benchmark and plot latency versus grouped Set120 quality."""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from experiments.run_grouped import write_json
from experiments.summarize_feature_study import weighted_corr, weighted_ranks
from experiments.transfer import SET120

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

SEED = 20261106
REPS = 10_000
LABELS = {
    "musiq": "MUSIQ",
    "nr4": "4 NR + RF",
    "nr4_diff": "4 NR + ResNet Diff + RF",
    "nr4_wasd": "4 NR + WASD + RF",
    "nr4_diff_wasd": "4 NR + ResNet Diff + WASD + RF",
}
EXTENDED = {
    "qalign": "Q-Align",
    "nr5": "5 NR + RF",
    "all_rr_no_wasd": "All RR without WASD + RF",
    "all_rr": "All RR + RF",
}


def verify(root, core, pilot, labels=LABELS):
    protocol = json.loads((root / "protocol.json").read_text())
    status = json.loads((root / "status.json").read_text())
    repeats = protocol["repetitions_per_image"]
    measurements = pd.read_csv(root / "measurements.csv", float_precision="round_trip")
    samples = pd.read_csv(root / "inputs.csv").set_index("sample_id")
    scores = pd.read_csv(root / "scores.csv")
    oof = pd.read_csv(root / "oof.csv", float_precision="round_trip")
    fold_cols = ["split_seed", "fold", "sample_id", "source_group", "score", "method"]
    manifest = pd.read_csv(core / "fold_assignments.csv", float_precision="round_trip")
    expected = manifest.loc[manifest.dataset.eq(SET120) & manifest.partition.eq("heldout"), fold_cols]
    first = expected.loc[expected.split_seed.eq(42) & expected.fold.eq(1)]
    if pilot:
        expected = first
    pd.testing.assert_frame_equal(oof[fold_cols].sort_values(fold_cols[:3], ignore_index=True),
                                  expected.sort_values(fold_cols[:3], ignore_index=True))
    pd.testing.assert_frame_equal(measurements[["sample_id", "iteration"]], scores[["sample_id", "iteration"]])
    n_images = 1 if pilot else 9
    controls = json.loads((core / "run.json").read_text())["controls"]
    if (len(samples) != n_images or not samples.index.is_unique
            or samples.source_group.nunique() != (1 if pilot else 3)
            or set(samples.index) & set(controls)
            or not set(samples.index).issubset(first.sample_id)
            or set(samples.index) != set(scores.sample_id)
            or any(set(g.iteration) != set(range(repeats)) for _, g in scores.groupby("sample_id"))
            or not np.isfinite(scores[list(labels)]).all().all()
            or not np.isfinite(oof[list(labels)]).all().all()):
        raise ValueError("Invalid live or held-out score coverage")
    assert len(measurements) == n_images * repeats
    assert protocol["status"] == status["status"] == "complete"
    assert status["folds"] == (1 if pilot else 15)
    assert set(protocol["stages"]) == set(labels)
    assert set(pd.read_csv(root / "parity.csv").sample_id) == set(samples.index)
    for name, stages in protocol["stages"].items():
        assert np.isfinite(measurements[stages]).all().all() and measurements[stages].gt(0).all().all()
        np.testing.assert_allclose(measurements[stages].sum(axis=1), measurements[f"total_{name}"],
                                   rtol=0, atol=1e-8)
    return measurements, oof


def correlations(score, prediction):
    return {"plcc": pearsonr(score, prediction).statistic, "srcc": spearmanr(score, prediction).statistic}


def analyze(measurements, oof, output, labels, pairs):
    latency = measurements.drop(columns=["sample_id", "iteration"]).quantile([.25, .5, .75, .95]).T
    latency.columns = ["p25_ms", "p50_ms", "p75_ms", "p95_ms"]
    latency.rename_axis("stage_or_total").to_csv(output / "latency.csv")
    repeated = oof.groupby("sample_id")
    assert repeated.size().eq(3).all() and repeated.split_seed.nunique().eq(3).all()
    assert repeated[["score", "source_group", "musiq"]].nunique().eq(1).all().all()
    frame = repeated.agg({"score": "first", "source_group": "first", **dict.fromkeys(labels, "mean")})
    group_ids, sources = pd.factorize(frame.source_group, sort=True)
    n = len(sources)
    draws = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=REPS)
    weights = draws[:, group_ids]
    target = frame.score.to_numpy()
    target_ranks = weighted_ranks(target, weights)
    quality, intervals = [], {}
    for name, label in labels.items():
        pred = frame[name].to_numpy()
        fold_stats = pd.DataFrame([correlations(part.score, part[name])
                                   for _, part in oof.groupby(["split_seed", "fold"])])
        estimates = correlations(target, pred)
        reps = {"plcc": weighted_corr(target, pred, weights),
                "srcc": weighted_corr(target_ranks, weighted_ranks(pred, weights), weights)}
        row = {"pipeline": name, "label": label, **latency.loc[f"total_{name}"].to_dict(),
               "n_latency": int(measurements[f"total_{name}"].count()), "n_sources": n, "n_samples": len(frame)}
        for metric, values in reps.items():
            intervals[name, metric] = values
            low, high = np.nanquantile(values, [.025, .975])
            row.update({metric: estimates[metric], f"{metric}_ci_low": low, f"{metric}_ci_high": high,
                        f"{metric}_valid_replicates": np.isfinite(values).sum(),
                        f"{metric}_fold_mean": fold_stats[metric].mean(),
                        f"{metric}_fold_sd": fold_stats[metric].std()})
        quality.append(row)
    points = pd.DataFrame(quality).set_index("pipeline")
    points["pareto"] = [
        not ((points.p50_ms.le(row.p50_ms) & points.srcc.ge(row.srcc))
             & (points.p50_ms.lt(row.p50_ms) | points.srcc.gt(row.srcc))).any()
        for row in points.itertuples()
    ]
    points.to_csv(output / "front.csv")
    rows = []
    for base, added in pairs:
        for metric in ("plcc", "srcc"):
            delta = intervals[added, metric] - intervals[base, metric]
            low, high = np.nanquantile(delta, [.025, .975])
            rows.append({"base": base, "added": added, "metric": metric,
                         "delta": points.at[added, metric] - points.at[base, metric],
                         "ci_low": low, "ci_high": high, "valid_replicates": np.isfinite(delta).sum()})
    pd.DataFrame(rows).to_csv(output / "paired_deltas.csv", index=False)
    lines = [
        "# Качество и задержка на Set120",
        "",
        "| Путь | Замеров | Медиана CPU, с | IQR, с | PLCC | SRCC [95% ДИ] | На фронте |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in points.itertuples():
        lines.append(f"| {row.label} | {row.n_latency} | {row.p50_ms / 1000:.2f} | "
                     f"{row.p25_ms / 1000:.2f}–{row.p75_ms / 1000:.2f} | {row.plcc:.3f} | "
                     f"{row.srcc:.3f} [{row.srcc_ci_low:.3f}, {row.srcc_ci_high:.3f}] | "
                     f"{'да' if row.pareto else '—'} |")
    lines.extend(["", "Фронт построен по медианам времени и точечным оценкам SRCC.",
                  "Качество оценивается на 120 выходах по архивным признакам.",
                  "[График](quality_latency.pdf) · [Точки](front.csv) · [Разности](paired_deltas.csv)"])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    plot(points, output)


def plot(points, output):
    fig, ax = plt.subplots(figsize=(7.1, 5.6 if len(points) <= 5 else 6.6))
    front = points.loc[points.pareto].sort_values("p50_ms")
    ax.plot(front.p50_ms / 1000, front.srcc, color="#536173", linestyle="--", linewidth=1, zorder=1)
    colors = ("#4d4d4d", "#2878a5", "#04855d", "#ad5914", "#843e84", "#81542a", "#b92142", "#23585a", "#575ca4")
    markers = ("o", "s", "^", "D", "P", "v", "X", "p", "<")
    for i, (_, row) in enumerate(points.iterrows()):
        color, marker = colors[i], markers[i]
        ax.hlines(row.srcc, row.p25_ms / 1000, row.p75_ms / 1000, color=color, alpha=.65)
        ax.vlines(row.p50_ms / 1000, row.srcc_ci_low, row.srcc_ci_high, color=color, alpha=.65)
        ax.scatter(row.p50_ms / 1000, row.srcc, color=color, marker=marker, s=45,
                   label=textwrap.fill(row.label, width=29), zorder=3)
    ax.set(xscale="log", xlabel="CPU latency per image (s, median and IQR)",
           ylabel="Set120 SRCC (95% scene bootstrap CI)")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8.5, ncols=2,
               loc="lower center", bbox_to_anchor=(.5, .015))
    fig.subplots_adjust(left=.13, right=.98, top=.96, bottom=.32)
    fig.savefig(output / "quality_latency.pdf")
    fig.savefig(output / "quality_latency.png", dpi=220)
    plt.close(fig)


def add_extended(measurements, oof, compact, extended, extended_pilot, core, prior_nr):
    def read_batches(name):
        return pd.concat([pd.read_csv(root / f"{name}.csv", float_precision="round_trip")
                          for root in (extended_pilot, extended)], ignore_index=True)

    extra = read_batches("measurements")
    protocol = json.loads((extended / "protocol.json").read_text())
    pilot_protocol = json.loads((extended_pilot / "protocol.json").read_text())
    for field in ("device", "threads", "batch_size", "qalign_dtype", "features", "stages",
                  "tabular_bundle_sha256", "generators", "repetitions_per_image"):
        assert protocol[field] == pilot_protocol[field], field
    for root in (extended_pilot, extended):
        assert json.loads((root / "verification.json").read_text())["status"] == "passed"
    assert protocol["status"] == "complete" and protocol["device"] == "cpu"
    common = pd.read_csv(compact / "inputs.csv").iloc[[0, 4, 8]].reset_index(drop=True)
    pd.testing.assert_frame_equal(common, read_batches("inputs"))
    expected_ids = set(common.sample_id)
    scores = read_batches("scores")
    pd.testing.assert_frame_equal(extra[["sample_id", "iteration"]], scores[["sample_id", "iteration"]])
    assert set(scores.sample_id) == expected_ids and np.isfinite(scores[["musiq", *EXTENDED]]).all().all()
    repeats = protocol["repetitions_per_image"]
    assert len(extra) == len(expected_ids) * repeats
    assert all(set(group.iteration) == set(range(repeats)) for _, group in extra.groupby("sample_id"))
    for name, stages in protocol["stages"].items():
        assert np.isfinite(extra[stages]).all().all() and extra[stages].gt(0).all().all()
        np.testing.assert_allclose(extra[stages].sum(axis=1), extra[f"total_{name}"], rtol=0, atol=1e-8)
    assert set(read_batches("parity").sample_id) == expected_ids
    ids = ["split_seed", "fold", "sample_id"]
    compare = [*ids, "score", "source_group"]
    nr = pd.read_csv(prior_nr, float_precision="round_trip")
    for source in ("nr5", "qalign"):
        assert np.isfinite(nr[source]).all()
    pd.testing.assert_frame_equal(oof[compare].sort_values(ids, ignore_index=True),
                                  nr[compare].sort_values(ids, ignore_index=True))
    oof = oof.merge(nr[[*ids, "nr5", "qalign"]], on=ids, validate="one_to_one")
    all_rr = []
    columns = [*compare, "experiment", "model", "track", "dataset", "prediction"]
    for block in pd.read_csv(core / "predictions.csv.gz", chunksize=150_000, usecols=columns,
                             float_precision="round_trip"):
        all_rr.append(block.loc[block.dataset.eq(SET120) & block.model.eq("randomforest") & block.track.eq("fixed")
                                & block.experiment.isin(["all_rr", "all_rr_no_wasd"])])
    variants = pd.concat(all_rr, ignore_index=True)
    for name, frame in variants.groupby("experiment"):
        pd.testing.assert_frame_equal(oof[compare].sort_values(ids, ignore_index=True),
                                      frame[compare].sort_values(ids, ignore_index=True))
        oof = oof.merge(frame[[*ids, "prediction"]].rename(columns={"prediction": name}),
                        on=ids, validate="one_to_one")
    assert np.isfinite(oof[list(EXTENDED)]).all().all()
    # MUSIQ appears in both batches. Use the extended batch alongside the full pipelines.
    compact_totals = measurements.loc[measurements.sample_id.isin(expected_ids),
                                      [f"total_{name}" for name in LABELS if name != "musiq"]]
    extended_totals = extra[[f"total_{name}" for name in ("musiq", *EXTENDED)]]
    totals = pd.concat([compact_totals, extended_totals], ignore_index=True)
    return totals.assign(sample_id="", iteration=0), oof, extra


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "core", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--extended", type=Path)
    parser.add_argument("--extended-pilot", type=Path)
    parser.add_argument("--prior-nr", type=Path)
    args = parser.parse_args()
    measured, predictions = verify(args.run, args.core, args.pilot)
    labels = LABELS
    pairs = [("nr4", "nr4_diff"), ("nr4", "nr4_wasd"), ("nr4_diff", "nr4_diff_wasd")]
    if args.extended:
        if args.pilot or args.prior_nr is None or args.extended_pilot is None:
            parser.error("--extended requires full results, --extended-pilot and --prior-nr")
        measured, predictions, extended_stages = add_extended(
            measured, predictions, args.run, args.extended, args.extended_pilot, args.core, args.prior_nr
        )
        labels = LABELS | EXTENDED
        pairs += [("nr4", "nr5"), ("nr5", "all_rr_no_wasd"), ("all_rr_no_wasd", "all_rr")]
    args.output.mkdir()
    if not args.pilot:
        analyze(measured, predictions, args.output, labels, pairs)
    if args.extended:
        stages = extended_stages.drop(columns=["sample_id", "iteration"])
        latency = stages.quantile([.25, .5, .75, .95]).T
        latency.columns = ["p25_ms", "p50_ms", "p75_ms", "p95_ms"]
        latency.rename_axis("stage_or_total").to_csv(args.output / "extended_latency.csv")
    write_json(args.output / "verification.json", {
        "status": "passed", "latency_rows": len(measured), "oof_rows": len(predictions),
        "bootstrap_seed": SEED, "bootstrap_reps": 0 if args.pilot else REPS,
    })
