"""Compute SHAP values for the RandomForest regressor and save PNG + PDF + SVG plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from qualisr_lab.regressors import (
    build_dataset,
    split_dataset,
    feature_family,
    get_pretty_feature,
    importance_palette,
    importance_legend_labels,
    ensure_dir,
)

FONT_SIZE = 14
DPI = 300


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = DPI) -> None:
    """Save figure as PNG, vector PDF, and SVG."""
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    print(f"  Saved {stem}.png / .pdf / .svg")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
) -> None:
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=True)

    palette = importance_palette()
    colors = [palette[feature_family(n)] for n in mean_abs.index]
    pretty_labels = [get_pretty_feature(n) or n for n in mean_abs.index]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.45 * len(mean_abs))))
    bars = ax.barh(range(len(mean_abs)), mean_abs.values, color=colors, height=0.65)

    for bar, val in zip(bars, mean_abs.values):
        ax.text(
            val + mean_abs.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"+{val:.4f}",
            va="center",
            ha="left",
            fontsize=FONT_SIZE - 2,
            color="#d81b60",
        )

    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels(pretty_labels, fontsize=FONT_SIZE)
    ax.set_xlabel("mean(|SHAP value|)", fontsize=FONT_SIZE)
    ax.set_title("Feature Importance — Random Forest (SHAP)", fontsize=FONT_SIZE + 1)
    ax.set_xlim(0, mean_abs.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 2)

    labels_map = importance_legend_labels()
    present = {feature_family(n) for n in mean_abs.index}
    handles = [
        mpatches.Patch(color=palette[k], label=labels_map[k])
        for k in labels_map if k in present
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=FONT_SIZE - 3)

    plt.tight_layout()
    save_fig(fig, out_dir, "shap_bar_randomforest")
    plt.close(fig)


def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    sample_idx: int,
    out_dir: Path,
) -> None:
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_idx].values,
        feature_names=[get_pretty_feature(n) or n for n in X_test.columns],
    )

    plt.figure(figsize=(10, max(5, 0.45 * len(X_test.columns))))
    shap.plots.waterfall(explanation, show=False, max_display=20)
    plt.title(f"SHAP Waterfall — Random Forest (test sample #{sample_idx})", fontsize=FONT_SIZE)
    plt.tight_layout()

    fig = plt.gcf()
    save_fig(fig, out_dir, f"shap_waterfall_randomforest_sample{sample_idx}")
    plt.close(fig)


def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    out_dir: Path,
) -> None:
    X_pretty = X_test.rename(
        columns={n: (get_pretty_feature(n) or n) for n in X_test.columns}
    )
    plt.figure(figsize=(9, max(5, 0.45 * len(X_test.columns))))
    shap.summary_plot(shap_values, X_pretty, show=False, plot_size=None)
    plt.title("SHAP Summary (beeswarm) — Random Forest", fontsize=FONT_SIZE)
    plt.tight_layout()
    fig = plt.gcf()
    save_fig(fig, out_dir, "shap_summary_randomforest")
    plt.close(fig)


def main() -> None:
    config_path = Path("configs/default.json")
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["models"] = {"randomforest": {"enabled": True, "params": {}}}
    np.random.seed(cfg["seed"])

    dataset = build_dataset(cfg)
    X_train, X_test, y_train, y_test = split_dataset(dataset, cfg)
    print(f"Dataset: {X_train.shape[0]} train / {X_test.shape[0]} test, {X_train.shape[1]} features")

    model = RandomForestRegressor(random_state=cfg["seed"])
    model.fit(X_train, y_train)
    print("Random Forest trained.")

    out_dir = ensure_dir(Path(cfg["paths"]["plots_root"]) / "shap_randomforest")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_arr = np.array(shap_values)

    pd.DataFrame(shap_arr, columns=X_test.columns).to_csv(out_dir / "shap_values.csv", index=False)
    mean_abs = pd.Series(np.abs(shap_arr).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    mean_abs.to_csv(out_dir / "shap_mean_abs.csv", header=["mean_abs_shap"])
    print("Saved shap_values.csv + shap_mean_abs.csv")

    print("Generating plots...")
    plot_shap_bar(shap_arr, list(X_test.columns), out_dir)

    pred = model.predict(X_test)
    best_idx = int(np.argmax(pred))
    plot_shap_waterfall(explainer, shap_arr, X_test, best_idx, out_dir)
    median_idx = int(np.argmin(np.abs(pred - float(np.mean(pred)))))
    if median_idx != best_idx:
        plot_shap_waterfall(explainer, shap_arr, X_test, median_idx, out_dir)

    plot_shap_summary(shap_arr, X_test, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print("\nMean |SHAP| ranking:")
    print(mean_abs.to_string())


if __name__ == "__main__":
    main()
