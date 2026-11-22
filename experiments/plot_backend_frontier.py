"""Plot the common CPU frontier and each backend on aligned Set120 inputs."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FAMILIES = {
    "musiq": ("MUSIQ", "#444444"),
    "qalign": ("Q-Align", "#111111"),
    "nr4": ("4 NR", "#0072b2"),
    "nr4_diff": ("4 NR + Diff", "#009e73"),
    "nr4_wasd": ("4 NR + WASD", "#d55e00"),
    "nr4_diff_wasd": ("4 NR + Diff + WASD", "#cc79a7"),
    "nr5": ("5 NR", "#7a5195"),
    "all_rr_no_wasd": ("All RR without WASD", "#8c564b"),
    "all_rr": ("All RR", "#bc8f00"),
}
BACKENDS = {"randomforest": ("Random Forest", "o"), "xgb": ("XGBoost", "s"), "catboost": ("CatBoost", "^")}


def plot(source, output):
    points = pd.read_csv(source, float_precision="round_trip")
    if len(points) != 23 or not points.pipeline.is_unique:
        raise ValueError("Expected 23 unique profiled paths")
    front = points[points.pareto_all_backends].sort_values("p50_ms")
    fig, grid = plt.subplots(2, 2, figsize=(8.1, 6.5), sharex=True, sharey=True)
    for ax, (backend, (title, marker)) in zip(grid.flat, BACKENDS.items(), strict=False):
        selected = points[points.backend.isin([backend, "direct"])]
        if len(selected) != 9:
            raise ValueError(f"Missing paths for {backend}")
        ax.plot(front.p50_ms / 1000, front.srcc, "--", color="#999999", linewidth=1.1, zorder=1)
        for row in selected.itertuples():
            family = row.pipeline.split("__")[0]
            _, color = FAMILIES[family]
            symbol = {"musiq": "D", "qalign": "*"}.get(family, marker)
            ax.hlines(row.srcc, row.p25_ms / 1000, row.p75_ms / 1000, color=color, alpha=.65)
            ax.vlines(row.p50_ms / 1000, row.srcc_ci_low, row.srcc_ci_high, color=color, alpha=.65)
            ax.scatter(row.p50_ms / 1000, row.srcc, color=color, marker=symbol,
                       s=48 if family == "qalign" else 32, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xscale("log")
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=.2)
        ax.tick_params(labelbottom=True)
    legend = grid[1, 1]
    legend.set_axis_off()
    handles = [Line2D([], [], color=color, marker={"musiq": "D", "qalign": "*"}.get(name, "o"),
                      linestyle="None", label=label) for name, (label, color) in FAMILIES.items()]
    handles.append(Line2D([], [], color="#999999", linestyle="--", label="Frontier across backends"))
    legend.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.6,
                  borderaxespad=0, handletextpad=.6, labelspacing=.45)
    legend.text(0, .05, "120 outputs, 40 scenes\nCPU, 2 threads, batch 1\nQuality uses cached OOF predictions",
                transform=legend.transAxes, va="bottom", fontsize=8.2, color="#444444")
    lower = points.p25_ms.min() / 1000
    upper = points.p75_ms.max() / 1000
    grid[0, 0].set_xlim(lower * .7, upper * 1.4)
    fig.supxlabel("CPU latency per image (s, median and IQR)", fontsize=10)
    fig.supylabel("Set120 SRCC (95% scene bootstrap CI)", fontsize=10)
    fig.subplots_adjust(left=.10, right=.98, bottom=.10, top=.95, wspace=.16, hspace=.30)
    for extension in ("pdf", "svg", "png"):
        target = output / f"quality_latency_all_backends.{extension}"
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.source, args.output)
